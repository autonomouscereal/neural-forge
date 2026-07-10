"""Zero-gradient post-nonlinearity donor-path decomposition.

This module deliberately operates after the relevant nonlinearity.  SwiGLU
inputs are the realized ``silu(gate) * up`` activation columns.  Attention
inputs are realized post-softmax head-response vectors.  Support selection
therefore never averages Q/K before softmax or gate/up before SiLU.

The selected support contains actual donor paths.  A closed-form interpolation
then maps every donor path channel into the selected channels and folds every
donor output direction into aggregate output coefficients.  No optimizer,
backward pass, autograd graph, adapter, or model mutation is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT_SCHEMA_VERSION = 1

_SPARSE_MATRIX_ELEMENT_BUDGET = 1_048_576
_SPARSE_DEFAULT_CHUNK_SIZE = 256
_SPARSE_DIAGNOSTIC_ROW_CHUNK_SIZE = 128
_SPARSE_DIAGNOSTIC_OUTPUT_SKETCH_SIZE = 64

__all__ = ["compress_post_nonlinearity_paths"]


def _torch():
    import torch

    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer.")
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer.")
    return result


def _validate_tensor(torch, value: Any, name: str, ranks: set[int]):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim not in ranks:
        expected = ", ".join(str(item) for item in sorted(ranks))
        raise ValueError(f"{name} must have rank in {{{expected}}}.")
    if any(int(size) <= 0 for size in value.shape):
        raise ValueError(f"{name} must have positive dimensions.")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _tensor_sha256(torch, tensor, chunk_rows: int = 512) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "dtype": str(tensor.dtype),
                "shape": [int(item) for item in tensor.shape],
            }
        )
    )
    detached = tensor.detach()
    if detached.ndim == 0:
        pieces = [detached]
    else:
        pieces = [
            detached[start : start + int(chunk_rows)]
            for start in range(0, int(detached.shape[0]), int(chunk_rows))
        ]
    for piece in pieces:
        raw = piece.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _gradient_fingerprint(torch, tensor) -> dict[str, Any]:
    gradient = tensor.grad
    if gradient is None:
        return {"present": False, "sha256": None}
    return {"present": True, "sha256": _tensor_sha256(torch, gradient)}


def _safe_scalar(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(
        f"{name} values must be finite JSON scalar labels, got {value!r}."
    )


def _label_key(value: Any, name: str) -> str:
    scalar = _safe_scalar(value, name)
    return json.dumps(
        {"type": type(scalar).__name__, "value": scalar},
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _coerce_labels(
    values: Sequence[Any], count: int, name: str
) -> tuple[list[Any], list[str]]:
    if hasattr(values, "tolist") and not isinstance(values, (str, bytes)):
        values = values.tolist()
    if isinstance(values, (str, bytes)) or len(values) != count:
        raise ValueError(f"{name} length must be {count}.")
    raw = [_safe_scalar(item, name) for item in values]
    return raw, [_label_key(item, name) for item in raw]


def _ordered_unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _quota_map(
    specification: Mapping[Any, int] | int | None,
    ordered_keys: Sequence[str],
    original_by_key: Mapping[str, Any],
    name: str,
) -> dict[str, int]:
    quotas = {key: 0 for key in ordered_keys}
    if specification is None:
        return quotas
    if isinstance(specification, int) and not isinstance(specification, bool):
        quota = _nonnegative_int(specification, name)
        return {key: quota for key in ordered_keys}
    if not isinstance(specification, Mapping):
        raise TypeError(f"{name} must be an integer, mapping, or None.")
    for raw_key, raw_value in specification.items():
        key = _label_key(raw_key, name)
        if key not in quotas:
            known = [original_by_key[item] for item in ordered_keys]
            raise ValueError(f"{name} contains unknown label {raw_key!r}; known={known!r}.")
        quotas[key] = _nonnegative_int(raw_value, name)
    return quotas


def _prompt_split(
    prompt_values: Sequence[Any],
    prompt_keys: Sequence[str],
    domain_keys: Sequence[str],
    *,
    validation_fraction: float,
    min_train_prompts: int,
    min_validation_prompts: int,
    split_seed: int,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if not math.isfinite(float(validation_fraction)) or not 0.0 < float(
        validation_fraction
    ) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    minimum_train = _positive_int(min_train_prompts, "min_train_prompts")
    minimum_validation = _positive_int(
        min_validation_prompts, "min_validation_prompts"
    )

    prompt_domain: dict[str, str] = {}
    prompt_value_by_key: dict[str, Any] = {}
    prompt_order: list[str] = []
    for prompt_value, prompt_key, domain_key in zip(
        prompt_values, prompt_keys, domain_keys, strict=True
    ):
        previous = prompt_domain.setdefault(prompt_key, domain_key)
        if previous != domain_key:
            raise ValueError("Every row for a prompt must have the same domain label.")
        if prompt_key not in prompt_value_by_key:
            prompt_order.append(prompt_key)
            prompt_value_by_key[prompt_key] = prompt_value

    if len(prompt_order) < minimum_train + minimum_validation:
        raise ValueError(
            "Not enough distinct prompts for a prompt-level held-out split: "
            f"need {minimum_train + minimum_validation}, got {len(prompt_order)}."
        )

    prompts_by_domain: dict[str, list[str]] = {}
    for prompt_key in prompt_order:
        prompts_by_domain.setdefault(prompt_domain[prompt_key], []).append(prompt_key)

    def rank(prompt_key: str) -> bytes:
        return hashlib.sha256(
            f"{int(split_seed)}\x00{prompt_key}".encode("utf-8")
        ).digest()

    validation: set[str] = set()
    for domain_prompts in prompts_by_domain.values():
        if len(domain_prompts) < 2:
            continue
        count = max(1, int(round(len(domain_prompts) * float(validation_fraction))))
        count = min(count, len(domain_prompts) - 1)
        validation.update(sorted(domain_prompts, key=rank)[:count])

    while len(prompt_order) - len(validation) < minimum_train:
        if not validation:
            break
        validation.remove(max(validation, key=rank))

    while len(validation) < minimum_validation:
        train_counts: dict[str, int] = {}
        for prompt_key in prompt_order:
            if prompt_key not in validation:
                domain_key = prompt_domain[prompt_key]
                train_counts[domain_key] = train_counts.get(domain_key, 0) + 1
        candidates = [
            prompt_key
            for prompt_key in prompt_order
            if prompt_key not in validation
            and train_counts[prompt_domain[prompt_key]] > 1
        ]
        if not candidates:
            break
        validation.add(min(candidates, key=rank))

    train_prompts = [item for item in prompt_order if item not in validation]
    validation_prompts = [item for item in prompt_order if item in validation]
    if len(train_prompts) < minimum_train or len(validation_prompts) < minimum_validation:
        raise ValueError("Prompt-level split cannot satisfy the requested minima.")

    train_rows = [
        index for index, prompt_key in enumerate(prompt_keys) if prompt_key not in validation
    ]
    validation_rows = [
        index for index, prompt_key in enumerate(prompt_keys) if prompt_key in validation
    ]
    if sorted(train_rows + validation_rows) != list(range(len(prompt_keys))):
        raise RuntimeError("Prompt split produced overlapping or missing rows.")

    report = {
        "unit": "prompt",
        "stratified_by": "domain",
        "seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "prompt_count": len(prompt_order),
        "train_prompt_count": len(train_prompts),
        "validation_prompt_count": len(validation_prompts),
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "train_prompt_ids": [prompt_value_by_key[item] for item in train_prompts],
        "validation_prompt_ids": [
            prompt_value_by_key[item] for item in validation_prompts
        ],
        "train_row_indices": train_rows,
        "validation_row_indices": validation_rows,
    }
    report["assignment_sha256"] = _json_sha256(
        {
            "seed": int(split_seed),
            "train": train_prompts,
            "validation": validation_prompts,
        }
    )
    return train_rows, validation_rows, report


def _domain_balanced_row_weights(
    torch,
    train_domain_keys: Sequence[str],
    ordered_domain_keys: Sequence[str],
    domain_weight_specification: Mapping[Any, float] | None,
    domain_original_by_key: Mapping[str, Any],
):
    configured = {key: 1.0 for key in ordered_domain_keys}
    if domain_weight_specification is not None:
        if not isinstance(domain_weight_specification, Mapping):
            raise TypeError("domain_weights must be a mapping or None.")
        for raw_key, raw_weight in domain_weight_specification.items():
            key = _label_key(raw_key, "domain_weights")
            if key not in configured:
                raise ValueError(f"domain_weights contains unknown label {raw_key!r}.")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("domain_weights values must be finite and positive.")
            configured[key] = weight

    counts = {key: 0 for key in ordered_domain_keys}
    for key in train_domain_keys:
        counts[key] += 1
    if any(counts[key] == 0 for key in ordered_domain_keys):
        missing = [domain_original_by_key[key] for key in ordered_domain_keys if not counts[key]]
        raise ValueError(f"Every domain needs at least one training row; missing={missing!r}.")

    weights = torch.tensor(
        [configured[key] / counts[key] for key in train_domain_keys],
        dtype=torch.float64,
    )
    weights.mul_(len(weights) / float(weights.sum().item()))
    totals = {
        key: float(
            weights[
                torch.tensor(
                    [item == key for item in train_domain_keys], dtype=torch.bool
                )
            ].sum().item()
        )
        for key in ordered_domain_keys
    }
    report = {
        "method": "inverse-domain-row-count",
        "normalized_mean": float(weights.mean().item()),
        "domains": [
            {
                "domain": domain_original_by_key[key],
                "train_rows": counts[key],
                "configured_weight": configured[key],
                "aggregate_row_weight": totals[key],
            }
            for key in ordered_domain_keys
        ],
    }
    return weights, report


def _coerce_path_weights(torch, path_weights: Any, path_count: int):
    if path_weights is None:
        return torch.ones(path_count, dtype=torch.float64), "uniform"
    if torch.is_tensor(path_weights):
        value = path_weights.detach().cpu().to(torch.float64)
    else:
        value = torch.tensor(list(path_weights), dtype=torch.float64)
    if value.ndim != 1 or int(value.shape[0]) != path_count:
        raise ValueError("path_weights must have shape [path_count].")
    if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
        raise ValueError("path_weights must be finite and non-negative.")
    if not bool((value > 0).any().item()):
        raise ValueError("At least one path weight must be positive.")
    return value, "provided"


def _path_domain_mass(
    torch,
    features,
    directions,
    train_rows: Sequence[int],
    train_domain_keys: Sequence[str],
    ordered_domain_keys: Sequence[str],
    row_weights,
):
    path_count = int(features.shape[1])
    masses = torch.zeros(
        (path_count, len(ordered_domain_keys)), dtype=torch.float64
    )
    train_index = torch.tensor(train_rows, dtype=torch.long)
    train_features = features.index_select(0, train_index)
    for domain_index, domain_key in enumerate(ordered_domain_keys):
        local_rows = [
            index for index, key in enumerate(train_domain_keys) if key == domain_key
        ]
        row_index = torch.tensor(local_rows, dtype=torch.long)
        domain_features = train_features.index_select(0, row_index)
        domain_weights = row_weights.index_select(0, row_index)
        for start in range(0, path_count, 128):
            end = min(start + 128, path_count)
            contribution = torch.einsum(
                "npq,pqo->npo",
                domain_features[:, start:end],
                directions[start:end],
            )
            masses[start:end, domain_index] = (
                contribution.square()
                .mul(domain_weights[:, None, None])
                .sum(dim=(0, 2))
            )
    dominant = masses.argmax(dim=1).tolist()
    return masses, [ordered_domain_keys[int(index)] for index in dominant]


def _rademacher(torch, rows: int, columns: int, generator, denominator: int):
    values = torch.randint(
        0,
        2,
        (rows, columns),
        generator=generator,
        dtype=torch.int8,
    ).to(torch.float64)
    values.mul_(2.0).sub_(1.0)
    values.div_(math.sqrt(float(denominator)))
    return values


def _sketch_dimensions(rows: int, outputs: int, budget: int) -> tuple[int, int]:
    row_dimension = min(
        rows,
        max(1, int(round(math.sqrt(budget * rows / max(outputs, 1))))),
    )
    output_dimension = min(outputs, max(1, budget // row_dimension))
    while row_dimension * output_dimension > budget:
        if row_dimension >= output_dimension and row_dimension > 1:
            row_dimension -= 1
        elif output_dimension > 1:
            output_dimension -= 1
        else:
            break
    return row_dimension, output_dimension


def _response_signature(
    torch,
    train_features,
    directions,
    row_weights,
    path_weights,
    *,
    sketch_size: int,
    split_seed: int,
):
    rows, path_count, _channels = [int(item) for item in train_features.shape]
    output_count = int(directions.shape[2])
    weighted_features = train_features.mul(row_weights.sqrt()[:, None, None])
    exact_dimension = rows * output_count
    if exact_dimension <= sketch_size:
        signature = torch.empty(
            (path_count, exact_dimension), dtype=torch.float64
        )
        for start in range(0, path_count, 128):
            end = min(start + 128, path_count)
            contribution = torch.einsum(
                "npq,pqo->npo",
                weighted_features[:, start:end],
                directions[start:end],
            )
            signature[start:end] = contribution.permute(1, 0, 2).reshape(
                end - start, exact_dimension
            )
        sketch_report = {
            "mode": "exact-weighted-response",
            "exact": True,
            "row_projection_dimension": rows,
            "output_projection_dimension": output_count,
            "signature_dimension": exact_dimension,
        }
    else:
        row_dimension, output_dimension = _sketch_dimensions(
            rows, output_count, sketch_size
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(split_seed) % (2**63 - 1))
        row_projection = _rademacher(
            torch, rows, row_dimension, generator, row_dimension
        )
        output_projection = _rademacher(
            torch, output_count, output_dimension, generator, output_dimension
        )
        left = torch.einsum("nr,npq->prq", row_projection, weighted_features)
        right = torch.einsum("pqo,os->pqs", directions, output_projection)
        signature = torch.einsum("prq,pqs->prs", left, right).reshape(
            path_count, row_dimension * output_dimension
        )
        sketch_report = {
            "mode": "deterministic-two-sided-rademacher-response-sketch",
            "exact": False,
            "row_projection_dimension": row_dimension,
            "output_projection_dimension": output_dimension,
            "signature_dimension": row_dimension * output_dimension,
        }
    signature.mul_(path_weights.sqrt()[:, None])
    sketch_report.update(
        {
            "requested_max_dimension": sketch_size,
            "seed": int(split_seed),
            "domain_balanced": True,
            "path_weighted": True,
        }
    )
    return signature, sketch_report


def _select_support(
    torch,
    signature,
    layer_keys: Sequence[str],
    dominant_domain_keys: Sequence[str],
    layer_quotas: Mapping[str, int],
    domain_quotas: Mapping[str, int],
    target_count: int,
):
    path_count = int(signature.shape[0])
    residual = signature.clone()
    original_norms = signature.square().sum(dim=1)
    selected: list[int] = []
    selected_set: set[int] = set()
    layer_counts = {key: 0 for key in layer_quotas}
    domain_counts = {key: 0 for key in domain_quotas}
    pivots = []
    rank = 0

    for _position in range(target_count):
        slots_after = target_count - len(selected) - 1
        residual_norms = residual.square().sum(dim=1)
        best_index = None
        best_score = float("-inf")
        for path_index in range(path_count):
            if path_index in selected_set:
                continue
            layer_key = layer_keys[path_index]
            domain_key = dominant_domain_keys[path_index]
            layer_deficit = sum(
                max(
                    0,
                    quota
                    - layer_counts[key]
                    - int(key == layer_key and layer_counts[key] < quota),
                )
                for key, quota in layer_quotas.items()
            )
            domain_deficit = sum(
                max(
                    0,
                    quota
                    - domain_counts[key]
                    - int(key == domain_key and domain_counts[key] < quota),
                )
                for key, quota in domain_quotas.items()
            )
            if max(layer_deficit, domain_deficit) > slots_after:
                continue
            score = float(residual_norms[path_index].item())
            if score > best_score or (
                score == best_score
                and (best_index is None or path_index < best_index)
            ):
                best_index = path_index
                best_score = score
        if best_index is None:
            raise ValueError(
                "Layer/domain quota constraints became infeasible during support selection."
            )

        selected.append(best_index)
        selected_set.add(best_index)
        layer_key = layer_keys[best_index]
        domain_key = dominant_domain_keys[best_index]
        layer_counts[layer_key] += 1
        domain_counts[domain_key] += 1

        pivot_norm = math.sqrt(max(best_score, 0.0))
        threshold = (
            torch.finfo(torch.float64).eps
            * max(1.0, math.sqrt(float(original_norms.max().item())))
            * max(1, int(signature.shape[1]))
        )
        rank_increased = pivot_norm > threshold
        if rank_increased:
            basis = residual[best_index].div(pivot_norm)
            projection = residual.matmul(basis)
            residual.sub_(projection[:, None].mul(basis[None, :]))
            residual[best_index].zero_()
            rank += 1
        pivots.append(
            {
                "position": len(selected) - 1,
                "path_index": best_index,
                "residual_norm": pivot_norm,
                "original_norm": math.sqrt(
                    max(float(original_norms[best_index].item()), 0.0)
                ),
                "rank_increased": rank_increased,
            }
        )

    if any(layer_counts[key] < quota for key, quota in layer_quotas.items()):
        raise RuntimeError("Internal error: selected support violates a layer quota.")
    if any(domain_counts[key] < quota for key, quota in domain_quotas.items()):
        raise RuntimeError("Internal error: selected support violates a domain quota.")
    return selected, layer_counts, domain_counts, pivots, rank


def _stable_top_candidates(torch, scores, candidates, count: int) -> list[int]:
    if count <= 0 or not candidates:
        return []
    candidate_index = torch.tensor(sorted(candidates), dtype=torch.long)
    candidate_scores = scores.index_select(0, candidate_index)
    order = torch.argsort(candidate_scores, descending=True, stable=True)
    return candidate_index.index_select(0, order[:count]).tolist()


def _vectorized_candidate_pivots(
    torch,
    signature,
    candidates: Sequence[int],
    *,
    target_count: int,
    layer_keys: Sequence[str],
    dominant_domain_keys: Sequence[str],
    layer_quotas: Mapping[str, int],
    domain_quotas: Mapping[str, int],
    position_offset: int = 0,
):
    candidate_values = sorted(candidates)
    if len(candidate_values) < target_count:
        raise ValueError("Sparse candidate pool is smaller than its target count.")
    candidate_index = torch.tensor(candidate_values, dtype=torch.long)
    residual = signature.index_select(0, candidate_index).clone()
    original_norms = residual.square().sum(dim=1)
    available = torch.ones(len(candidate_values), dtype=torch.bool)
    ordered_layers = list(layer_quotas)
    ordered_domains = list(domain_quotas)
    layer_position = {key: index for index, key in enumerate(ordered_layers)}
    domain_position = {key: index for index, key in enumerate(ordered_domains)}
    candidate_layers = torch.tensor(
        [layer_position[layer_keys[index]] for index in candidate_values],
        dtype=torch.long,
    )
    candidate_domains = torch.tensor(
        [
            domain_position[dominant_domain_keys[index]]
            for index in candidate_values
        ],
        dtype=torch.long,
    )
    layer_quota_tensor = torch.tensor(
        [layer_quotas[key] for key in ordered_layers], dtype=torch.long
    )
    domain_quota_tensor = torch.tensor(
        [domain_quotas[key] for key in ordered_domains], dtype=torch.long
    )
    layer_counts = torch.zeros_like(layer_quota_tensor)
    domain_counts = torch.zeros_like(domain_quota_tensor)
    selected: list[int] = []
    pivots: list[dict[str, Any]] = []
    rank = 0
    rank_exhausted_at = None
    rank_exhausted_pivots = 0
    rank_exhausted_mode = False
    threshold = (
        torch.finfo(torch.float64).eps
        * max(1.0, math.sqrt(float(original_norms.max().item())))
        * max(1, int(signature.shape[1]))
    )

    for local_position in range(target_count):
        slots_after = target_count - local_position - 1
        layer_deficit = (layer_quota_tensor - layer_counts).clamp_min(0)
        domain_deficit = (domain_quota_tensor - domain_counts).clamp_min(0)
        layer_reduction = (
            layer_deficit.index_select(0, candidate_layers) > 0
        ).to(torch.long)
        domain_reduction = (
            domain_deficit.index_select(0, candidate_domains) > 0
        ).to(torch.long)
        layer_remaining = int(layer_deficit.sum().item()) - layer_reduction
        domain_remaining = int(domain_deficit.sum().item()) - domain_reduction
        eligible = (
            available
            & (layer_remaining <= slots_after)
            & (domain_remaining <= slots_after)
        )
        if not bool(eligible.any().item()):
            raise ValueError(
                "Sparse candidate pool cannot satisfy the layer/domain quotas."
            )
        residual_norms = (
            original_norms
            if rank_exhausted_mode
            else residual.square().sum(dim=1)
        )
        scores = residual_norms.masked_fill(~eligible, float("-inf"))
        candidate_position = int(scores.argmax().item())
        source_index = candidate_values[candidate_position]
        selected.append(source_index)
        available[candidate_position] = False
        layer_counts[candidate_layers[candidate_position]] += 1
        domain_counts[candidate_domains[candidate_position]] += 1

        pivot_norm = (
            float(residual[candidate_position].norm().item())
            if rank_exhausted_mode
            else math.sqrt(max(float(scores[candidate_position].item()), 0.0))
        )
        rank_increased = not rank_exhausted_mode and pivot_norm > threshold
        if rank_increased:
            basis = residual[candidate_position].div(pivot_norm)
            projection = residual.matmul(basis)
            residual.sub_(projection[:, None].mul(basis[None, :]))
            residual[candidate_position].zero_()
            rank += 1
        else:
            if rank_exhausted_at is None:
                rank_exhausted_at = position_offset + local_position
            rank_exhausted_mode = True
            rank_exhausted_pivots += 1
        pivots.append(
            {
                "position": position_offset + local_position,
                "path_index": source_index,
                "residual_norm": pivot_norm,
                "original_norm": math.sqrt(
                    max(float(original_norms[candidate_position].item()), 0.0)
                ),
                "rank_increased": rank_increased,
            }
        )

    layer_count_map = {
        key: int(layer_counts[index].item())
        for index, key in enumerate(ordered_layers)
    }
    domain_count_map = {
        key: int(domain_counts[index].item())
        for index, key in enumerate(ordered_domains)
    }
    if any(
        layer_count_map[key] < quota for key, quota in layer_quotas.items()
    ) or any(
        domain_count_map[key] < quota for key, quota in domain_quotas.items()
    ):
        raise RuntimeError("Vectorized sparse selector violated a quota.")
    return {
        "selected": selected,
        "layer_counts": layer_count_map,
        "domain_counts": domain_count_map,
        "pivots": pivots,
        "rank": rank,
        "rank_exhausted_at": rank_exhausted_at,
        "rank_exhausted_pivots": rank_exhausted_pivots,
        "score_elements_bound": target_count * len(candidate_values),
        "residual_projection_elements_bound": (
            target_count * len(candidate_values) * int(signature.shape[1])
        ),
        "residual_projection_elements_actual": (
            rank * len(candidate_values) * int(signature.shape[1])
        ),
    }


def _sparse_candidate_pool(
    torch,
    signature,
    layer_keys: Sequence[str],
    dominant_domain_keys: Sequence[str],
    layer_quotas: Mapping[str, int],
    domain_quotas: Mapping[str, int],
    target_count: int,
    *,
    multiplier: int = 2,
):
    source_count = int(signature.shape[0])
    scores = signature.square().sum(dim=1)
    desired = min(source_count, max(target_count, multiplier * target_count))
    selected: set[int] = set()
    group_pool_report = []
    for group_type, keys, quotas in (
        ("layer", layer_keys, layer_quotas),
        ("domain", dominant_domain_keys, domain_quotas),
    ):
        for key, quota in quotas.items():
            if quota <= 0:
                continue
            available = [index for index, item in enumerate(keys) if item == key]
            count = min(len(available), max(quota, multiplier * quota))
            chosen = _stable_top_candidates(torch, scores, available, count)
            selected.update(chosen)
            group_pool_report.append(
                {
                    "group_type": group_type,
                    "group_key": key,
                    "required_count": quota,
                    "candidate_count": len(chosen),
                    "available_source_count": len(available),
                }
            )
    if len(selected) < desired:
        selected.update(
            _stable_top_candidates(
                torch,
                scores,
                [index for index in range(source_count) if index not in selected],
                desired - len(selected),
            )
        )
    return sorted(selected), {
        "requested_multiplier": multiplier,
        "desired_global_candidate_count": desired,
        "group_candidate_pools": group_pool_report,
        "full_source_score_passes": 1,
        "preselection_method": "stable-response-norm-ranking",
    }


def _select_support_sparse(
    torch,
    signature,
    layer_keys: Sequence[str],
    dominant_domain_keys: Sequence[str],
    layer_quotas: Mapping[str, int],
    domain_quotas: Mapping[str, int],
    target_count: int,
):
    source_count, signature_dimension = [int(item) for item in signature.shape]
    layer_quotas_consume_target = sum(layer_quotas.values()) == target_count
    has_domain_quotas = any(quota > 0 for quota in domain_quotas.values())
    fallback_used = False
    fallback_reason = None

    if layer_quotas_consume_target and not has_domain_quotas:
        scores = signature.square().sum(dim=1)
        selected: list[int] = []
        pivots: list[dict[str, Any]] = []
        rank = 0
        candidate_union: set[int] = set()
        group_reports = []
        score_elements_bound = 0
        residual_projection_elements_bound = 0
        residual_projection_elements_actual = 0
        rank_exhausted_at = None
        rank_exhausted_pivots = 0
        for layer_key, quota in layer_quotas.items():
            if quota <= 0:
                continue
            available = [
                index for index, key in enumerate(layer_keys) if key == layer_key
            ]
            candidate_count = min(len(available), max(quota, 2 * quota))
            candidates = _stable_top_candidates(
                torch, scores, available, candidate_count
            )
            candidate_union.update(candidates)
            local = _vectorized_candidate_pivots(
                torch,
                signature,
                candidates,
                target_count=quota,
                layer_keys=layer_keys,
                dominant_domain_keys=dominant_domain_keys,
                layer_quotas={
                    key: (quota if key == layer_key else 0)
                    for key in layer_quotas
                },
                domain_quotas={key: 0 for key in domain_quotas},
                position_offset=len(selected),
            )
            selected.extend(local["selected"])
            pivots.extend(local["pivots"])
            rank += local["rank"]
            score_elements_bound += local["score_elements_bound"]
            residual_projection_elements_bound += local[
                "residual_projection_elements_bound"
            ]
            residual_projection_elements_actual += local[
                "residual_projection_elements_actual"
            ]
            if rank_exhausted_at is None and local["rank_exhausted_at"] is not None:
                rank_exhausted_at = local["rank_exhausted_at"]
            rank_exhausted_pivots += local["rank_exhausted_pivots"]
            group_reports.append(
                {
                    "layer_key": layer_key,
                    "source_count": len(available),
                    "quota": quota,
                    "candidate_pool_size": len(candidates),
                    "selected_count": len(local["selected"]),
                    "numerical_rank": local["rank"],
                }
            )
        layer_counts = {key: 0 for key in layer_quotas}
        domain_counts = {key: 0 for key in domain_quotas}
        for source_index in selected:
            layer_counts[layer_keys[source_index]] += 1
            domain_counts[dominant_domain_keys[source_index]] += 1
        candidate_pool_size = len(candidate_union)
        strategy = "independent-per-layer-vectorized-pivoted-qr"
        preselection_report = {
            "requested_multiplier": 2,
            "preselection_method": "stable-response-norm-ranking-per-layer",
            "group_candidate_pools": group_reports,
            "full_source_score_passes": 1,
        }
    else:
        candidates, preselection_report = _sparse_candidate_pool(
            torch,
            signature,
            layer_keys,
            dominant_domain_keys,
            layer_quotas,
            domain_quotas,
            target_count,
        )
        try:
            selection = _vectorized_candidate_pivots(
                torch,
                signature,
                candidates,
                target_count=target_count,
                layer_keys=layer_keys,
                dominant_domain_keys=dominant_domain_keys,
                layer_quotas=layer_quotas,
                domain_quotas=domain_quotas,
            )
        except ValueError:
            fallback_used = True
            fallback_reason = "bounded-candidate-pool-quota-infeasible"
            candidates = list(range(source_count))
            selection = _vectorized_candidate_pivots(
                torch,
                signature,
                candidates,
                target_count=target_count,
                layer_keys=layer_keys,
                dominant_domain_keys=dominant_domain_keys,
                layer_quotas=layer_quotas,
                domain_quotas=domain_quotas,
            )
        selected = selection["selected"]
        layer_counts = selection["layer_counts"]
        domain_counts = selection["domain_counts"]
        pivots = selection["pivots"]
        rank = selection["rank"]
        rank_exhausted_at = selection["rank_exhausted_at"]
        rank_exhausted_pivots = selection["rank_exhausted_pivots"]
        score_elements_bound = selection["score_elements_bound"]
        residual_projection_elements_bound = selection[
            "residual_projection_elements_bound"
        ]
        residual_projection_elements_actual = selection[
            "residual_projection_elements_actual"
        ]
        candidate_pool_size = len(candidates)
        strategy = "bounded-candidate-vectorized-pivoted-qr"

    if len(selected) != target_count or len(set(selected)) != target_count:
        raise RuntimeError("Sparse selector did not produce a unique full support.")
    if any(layer_counts[key] < quota for key, quota in layer_quotas.items()):
        raise RuntimeError("Sparse selector violated a layer quota.")
    if any(domain_counts[key] < quota for key, quota in domain_quotas.items()):
        raise RuntimeError("Sparse selector violated a domain quota.")
    report = {
        "strategy": strategy,
        "candidate_pool_size": candidate_pool_size,
        "source_path_count": source_count,
        "candidate_pool_fraction": candidate_pool_size / source_count,
        "preselection": preselection_report,
        "complexity_bounds": {
            "python_pivot_iterations": target_count,
            "python_source_iterations_per_pivot": 0,
            "pivot_score_elements": score_elements_bound,
            "residual_projection_elements": residual_projection_elements_bound,
            "residual_projection_elements_actual": (
                residual_projection_elements_actual
            ),
            "signature_dimension": signature_dimension,
            "dense_python_target_by_source_scan_avoided": True,
            "candidate_signature_matrix_shape_bound": [
                candidate_pool_size,
                signature_dimension,
            ],
            "candidate_signature_matrix_element_bound": (
                candidate_pool_size * signature_dimension
            ),
            "candidate_score_vector_shape_bound": [candidate_pool_size],
        },
        "fallback": {
            "used": fallback_used,
            "reason": fallback_reason,
            "expanded_to_all_sources": fallback_used,
        },
        "rank_exhaustion": {
            "occurred": rank_exhausted_at is not None,
            "first_pivot_position": rank_exhausted_at,
            "fallback_pivot_count": rank_exhausted_pivots,
            "selection_continued_deterministically": rank_exhausted_at is not None,
        },
        "layer_quotas_consume_target": layer_quotas_consume_target,
        "independent_layer_selection_used": (
            layer_quotas_consume_target and not has_domain_quotas
        ),
    }
    return selected, layer_counts, domain_counts, pivots, rank, report


def _fit_interpolation(torch, selected_design, all_design, row_weights, ridge: float):
    weighted_selected = selected_design.mul(row_weights.sqrt()[:, None])
    weighted_all = all_design.mul(row_weights.sqrt()[:, None])
    feature_gram = weighted_selected.t().matmul(weighted_selected)
    trace = float(torch.trace(feature_gram).item())
    trace_scale = trace / max(1, int(feature_gram.shape[0]))
    effective_ridge = float(ridge) * max(
        trace_scale, torch.finfo(torch.float64).tiny
    )
    if effective_ridge > 0.0:
        system = feature_gram + effective_ridge * torch.eye(
            int(feature_gram.shape[0]), dtype=torch.float64
        )
        rhs = weighted_selected.t().matmul(weighted_all)
        factor, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if int(info.max().item()) == 0:
            coefficients = torch.cholesky_solve(rhs, factor)
            solver = "cholesky"
        else:
            coefficients = torch.linalg.lstsq(system, rhs).solution
            solver = "lstsq-normal-equations"
    else:
        coefficients = torch.linalg.lstsq(weighted_selected, weighted_all).solution
        solver = "lstsq-design"
    if not bool(torch.isfinite(coefficients).all().item()):
        raise RuntimeError("Closed-form path interpolation produced non-finite values.")
    return coefficients, {
        "solver": solver,
        "dtype": "float64",
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled",
        "feature_gram_trace": trace,
        "trace_scale": trace_scale,
        "effective_ridge": effective_ridge,
        "objective": "||W^(1/2)(X_selected C - X_all)||_F^2 + lambda||C||_F^2",
    }


def _bounded_chunk_size(
    total: int,
    requested: int,
    *,
    elements_per_item: int,
    avoid_full: bool = False,
) -> int:
    element_limit = max(1, _SPARSE_MATRIX_ELEMENT_BUDGET // elements_per_item)
    full_limit = max(1, total - 1) if avoid_full and total > 1 else total
    return max(1, min(total, requested, element_limit, full_limit))


def _fit_sparse_signature_interpolation(
    torch,
    signature,
    selected: Sequence[int],
    *,
    ridge: float,
    requested_chunk_size: int,
):
    source_count, signature_dimension = [int(item) for item in signature.shape]
    target_count = len(selected)
    selected_index = torch.tensor(selected, dtype=torch.long)
    selected_signature = signature.index_select(0, selected_index)
    selected_squared_norm = selected_signature.square().sum(dim=1)
    selected_norm = selected_squared_norm.sqrt()
    trace = float(selected_squared_norm.sum().item())
    trace_scale = trace / max(1, target_count)
    effective_ridge = float(ridge) * max(
        trace_scale, torch.finfo(torch.float64).tiny
    )
    chunk_size = _bounded_chunk_size(
        source_count,
        requested_chunk_size,
        elements_per_item=max(1, target_count),
        avoid_full=True,
    )

    selected_positions = torch.empty(source_count, dtype=torch.long)
    coefficients = torch.empty(source_count, dtype=torch.float64)
    assigned_similarity = torch.empty(source_count, dtype=torch.float64)
    epsilon = torch.finfo(torch.float64).eps
    for start in range(0, source_count, chunk_size):
        end = min(start + chunk_size, source_count)
        source_chunk = signature[start:end]
        dot = source_chunk.matmul(selected_signature.t())
        source_norm = source_chunk.square().sum(dim=1).sqrt()
        cosine_denominator = source_norm[:, None].mul(selected_norm[None, :])
        absolute_cosine = torch.where(
            cosine_denominator > epsilon,
            dot.abs() / cosine_denominator.clamp_min(epsilon),
            torch.zeros_like(dot),
        )
        position = absolute_cosine.argmax(dim=1)
        selected_dot = dot.gather(1, position[:, None]).squeeze(1)
        selected_denominator = selected_squared_norm.index_select(0, position)
        selected_denominator = selected_denominator.add(effective_ridge)
        scalar = torch.where(
            selected_denominator > torch.finfo(torch.float64).tiny,
            selected_dot / selected_denominator,
            torch.zeros_like(selected_dot),
        )
        selected_positions[start:end] = position
        coefficients[start:end] = scalar
        assigned_similarity[start:end] = absolute_cosine.gather(
            1, position[:, None]
        ).squeeze(1)

    identity_threshold = (
        epsilon
        * max(trace_scale, torch.finfo(torch.float64).tiny)
        * max(1, signature_dimension)
    )
    identity_forced = 0
    identity_unsafe = 0
    for selected_position, source_index in enumerate(selected):
        if float(selected_squared_norm[selected_position].item()) > identity_threshold:
            selected_positions[source_index] = selected_position
            coefficients[source_index] = 1.0
            assigned_similarity[source_index] = 1.0
            identity_forced += 1
        else:
            identity_unsafe += 1

    if not bool(torch.isfinite(coefficients).all().item()):
        raise RuntimeError("Sparse signature interpolation produced non-finite values.")
    if bool((selected_positions < 0).any().item()) or bool(
        (selected_positions >= target_count).any().item()
    ):
        raise RuntimeError("Sparse signature interpolation left an invalid mapping.")

    coefficient_abs = coefficients.abs()
    coefficient_report = {
        "count": source_count,
        "finite_count": int(torch.isfinite(coefficients).sum().item()),
        "minimum": float(coefficients.min().item()),
        "maximum": float(coefficients.max().item()),
        "mean": float(coefficients.mean().item()),
        "standard_deviation": float(coefficients.std(unbiased=False).item()),
        "absolute_mean": float(coefficient_abs.mean().item()),
        "absolute_maximum": float(coefficient_abs.max().item()),
        "negative_count": int((coefficients < 0.0).sum().item()),
        "zero_count": int((coefficients == 0.0).sum().item()),
        "positive_count": int((coefficients > 0.0).sum().item()),
    }
    fit_report = {
        "solver": "one-sparse-closed-form-signature-ridge",
        "dtype": "float64",
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled",
        "feature_gram_trace": trace,
        "trace_scale": trace_scale,
        "effective_ridge": effective_ridge,
        "objective": (
            "per source: min_c ||signature_source - "
            "c*signature_selected||_2^2 + lambda*c^2"
        ),
        "assignment_method": "deterministic-chunked-absolute-cosine",
        "signed_scalar_method": "signature-dot-over-selected-norm-plus-ridge",
        "identity_coefficient": 1.0,
        "identity_numerical_safety_threshold": identity_threshold,
        "selected_anchor_identity_forced_count": identity_forced,
        "selected_anchor_identity_unsafe_count": identity_unsafe,
        "assigned_absolute_cosine_minimum": float(
            assigned_similarity.min().item()
        ),
        "assigned_absolute_cosine_mean": float(
            assigned_similarity.mean().item()
        ),
        "sparse_coefficient_statistics": coefficient_report,
    }
    materialization = {
        "dense_target_by_source_coefficients_materialized": False,
        "assignment_source_chunk_size": chunk_size,
        "assignment_similarity_matrix_shape_bound": [chunk_size, target_count],
        "assignment_similarity_matrix_element_bound": chunk_size * target_count,
    }
    return selected_positions, coefficients, fit_report, materialization


def _aggregate_sparse_output_directions(
    torch,
    directions,
    selected_positions,
    coefficients,
    *,
    target_count: int,
    requested_chunk_size: int,
):
    source_count, channel_count, output_count = [
        int(item) for item in directions.shape
    ]
    chunk_size = _bounded_chunk_size(
        source_count,
        requested_chunk_size,
        elements_per_item=max(1, channel_count * output_count),
        avoid_full=True,
    )
    aggregate = torch.zeros(
        (target_count, channel_count, output_count), dtype=torch.float64
    )
    for start in range(0, source_count, chunk_size):
        end = min(start + chunk_size, source_count)
        contribution = directions[start:end].mul(
            coefficients[start:end, None, None]
        )
        aggregate.index_add_(
            0,
            selected_positions[start:end],
            contribution,
        )
    source_output_norms = directions.flatten(1).norm(dim=1).mul(coefficients.abs())
    if not bool(torch.isfinite(aggregate).all().item()):
        raise RuntimeError("Sparse output aggregation produced non-finite values.")
    report = {
        "method": "chunked-index-add",
        "source_chunk_size": chunk_size,
        "contribution_tensor_shape_bound": [
            chunk_size,
            channel_count,
            output_count,
        ],
        "contribution_tensor_element_bound": (
            chunk_size * channel_count * output_count
        ),
    }
    return aggregate, source_output_norms, report


def _diagnostic_output_projection(
    torch,
    directions,
    *,
    sketch_size: int,
    split_seed: int,
    requested_chunk_size: int,
):
    source_count, channel_count, output_count = [
        int(item) for item in directions.shape
    ]
    diagnostic_outputs = min(output_count, sketch_size)
    if diagnostic_outputs == output_count:
        return directions, {
            "method": "exact-output-space",
            "approximate": False,
            "approximation_explicit": True,
            "original_output_features": output_count,
            "evaluated_output_features": output_count,
            "projection_seed": None,
            "projection_matrix_shape_bound": None,
            "projection_matrix_element_bound": 0,
            "projection_multiply_result_matrix_shape_bound": None,
            "projection_multiply_result_matrix_element_bound": 0,
            "projected_direction_tensor_shape": None,
        }

    projection_seed = (int(split_seed) + 0x5DEECE66D) % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_seed)
    output_chunk_size = _bounded_chunk_size(
        output_count,
        requested_chunk_size,
        elements_per_item=diagnostic_outputs,
        avoid_full=True,
    )
    flat_source_count = source_count * channel_count
    source_chunk_size = _bounded_chunk_size(
        flat_source_count,
        requested_chunk_size * channel_count,
        elements_per_item=diagnostic_outputs,
        avoid_full=True,
    )
    projected = torch.zeros(
        (source_count, channel_count, diagnostic_outputs), dtype=torch.float64
    )
    flat_directions = directions.reshape(flat_source_count, output_count)
    flat_projected = projected.reshape(
        flat_source_count, diagnostic_outputs
    )
    for start in range(0, output_count, output_chunk_size):
        end = min(start + output_chunk_size, output_count)
        projection = _rademacher(
            torch,
            end - start,
            diagnostic_outputs,
            generator,
            diagnostic_outputs,
        )
        for source_start in range(0, flat_source_count, source_chunk_size):
            source_end = min(source_start + source_chunk_size, flat_source_count)
            flat_projected[source_start:source_end].addmm_(
                flat_directions[source_start:source_end, start:end], projection
            )
    if not bool(torch.isfinite(projected).all().item()):
        raise RuntimeError("Diagnostic output sketch produced non-finite values.")
    return projected, {
        "method": "deterministic-rademacher-output-sketch",
        "approximate": True,
        "approximation_explicit": True,
        "original_output_features": output_count,
        "evaluated_output_features": diagnostic_outputs,
        "projection_seed": projection_seed,
        "projection_matrix_shape_bound": [
            output_chunk_size,
            diagnostic_outputs,
        ],
        "projection_matrix_element_bound": (
            output_chunk_size * diagnostic_outputs
        ),
        "projection_multiply_result_matrix_shape_bound": [
            source_chunk_size,
            diagnostic_outputs,
        ],
        "projection_multiply_result_matrix_element_bound": (
            source_chunk_size * diagnostic_outputs
        ),
        "projected_direction_tensor_shape": [
            source_count,
            channel_count,
            diagnostic_outputs,
        ],
    }


def _new_response_accumulator() -> dict[str, float | int]:
    return {
        "row_count": 0,
        "element_count": 0,
        "squared_error": 0.0,
        "absolute_error": 0.0,
        "target_squared": 0.0,
        "prediction_squared": 0.0,
        "dot": 0.0,
        "cosine_sum": 0.0,
        "cosine_min": float("inf"),
    }


def _update_response_accumulator(torch, accumulator, target, prediction) -> None:
    difference = prediction - target
    target_squared_by_row = target.square().sum(dim=1)
    prediction_squared_by_row = prediction.square().sum(dim=1)
    dot_by_row = target.mul(prediction).sum(dim=1)
    target_norm = target_squared_by_row.sqrt()
    prediction_norm = prediction_squared_by_row.sqrt()
    denominator = target_norm.mul(prediction_norm)
    epsilon = torch.finfo(torch.float64).eps
    cosine = torch.where(
        denominator > epsilon,
        dot_by_row / denominator.clamp_min(epsilon),
        torch.where(
            (target_norm <= epsilon) & (prediction_norm <= epsilon),
            torch.ones_like(denominator),
            torch.zeros_like(denominator),
        ),
    )
    accumulator["row_count"] += int(target.shape[0])
    accumulator["element_count"] += int(target.numel())
    accumulator["squared_error"] += float(difference.square().sum().item())
    accumulator["absolute_error"] += float(difference.abs().sum().item())
    accumulator["target_squared"] += float(target_squared_by_row.sum().item())
    accumulator["prediction_squared"] += float(
        prediction_squared_by_row.sum().item()
    )
    accumulator["dot"] += float(dot_by_row.sum().item())
    accumulator["cosine_sum"] += float(cosine.sum().item())
    accumulator["cosine_min"] = min(
        float(accumulator["cosine_min"]), float(cosine.min().item())
    )


def _finalize_response_accumulator(torch, accumulator) -> dict[str, float]:
    row_count = int(accumulator["row_count"])
    element_count = int(accumulator["element_count"])
    if row_count <= 0 or element_count <= 0:
        raise RuntimeError("Response diagnostics received no rows.")
    squared_error = float(accumulator["squared_error"])
    target_squared = float(accumulator["target_squared"])
    prediction_squared = float(accumulator["prediction_squared"])
    denominator = math.sqrt(target_squared * prediction_squared)
    epsilon = torch.finfo(torch.float64).eps
    if denominator > epsilon:
        global_cosine = float(accumulator["dot"]) / denominator
    else:
        global_cosine = float(target_squared <= epsilon and prediction_squared <= epsilon)
    return {
        "absolute_rmse": math.sqrt(squared_error / element_count),
        "mean_absolute_error": float(accumulator["absolute_error"]) / element_count,
        "cosine_similarity_mean": float(accumulator["cosine_sum"]) / row_count,
        "cosine_similarity_min": float(accumulator["cosine_min"]),
        "cosine_similarity_global": global_cosine,
        "target_rms": math.sqrt(target_squared / element_count),
        "prediction_rms": math.sqrt(prediction_squared / element_count),
        "explained_response_energy": (
            1.0 - squared_error / target_squared if target_squared > epsilon else 1.0
        ),
    }


def _new_metric_pair() -> dict[str, dict[str, float | int]]:
    return {
        "baseline": _new_response_accumulator(),
        "fitted": _new_response_accumulator(),
    }


def _update_metric_pair(torch, pair, target, baseline, fitted) -> None:
    _update_response_accumulator(torch, pair["baseline"], target, baseline)
    _update_response_accumulator(torch, pair["fitted"], target, fitted)


def _finalize_metric_pair(torch, pair) -> dict[str, dict[str, float]]:
    return {
        "baseline": _finalize_response_accumulator(torch, pair["baseline"]),
        "fitted": _finalize_response_accumulator(torch, pair["fitted"]),
    }


def _chunked_sparse_response_diagnostics(
    torch,
    features,
    diagnostic_directions,
    selected_index,
    aggregate_directions,
    selected_native_directions,
    *,
    train_rows: Sequence[int],
    validation_rows: Sequence[int],
    prompt_keys: Sequence[str],
    domain_keys: Sequence[str],
    prompt_original: Mapping[str, Any],
    domain_original: Mapping[str, Any],
    requested_chunk_size: int,
    requested_row_chunk_size: int,
):
    row_count, source_count, channel_count = [int(item) for item in features.shape]
    target_count = int(selected_index.shape[0])
    diagnostic_outputs = int(diagnostic_directions.shape[2])
    row_chunk_size = _bounded_chunk_size(
        row_count,
        requested_row_chunk_size,
        elements_per_item=max(1, diagnostic_outputs),
    )
    source_chunk_size = _bounded_chunk_size(
        source_count,
        requested_chunk_size,
        elements_per_item=max(1, row_chunk_size * channel_count),
    )
    selected_chunk_size = _bounded_chunk_size(
        target_count,
        requested_chunk_size,
        elements_per_item=max(1, row_chunk_size * channel_count),
    )

    train_pair = _new_metric_pair()
    validation_pair = _new_metric_pair()
    prompt_pairs: dict[str, dict[str, dict[str, float | int]]] = {}
    domain_pairs: dict[str, dict[str, dict[str, float | int]]] = {}

    def evaluate_rows(rows: Sequence[int], pair, *, grouped: bool) -> None:
        for row_start in range(0, len(rows), row_chunk_size):
            row_values = list(rows[row_start : row_start + row_chunk_size])
            row_index = torch.tensor(row_values, dtype=torch.long)
            local_count = len(row_values)
            target = torch.zeros(
                (local_count, diagnostic_outputs), dtype=torch.float64
            )
            for source_start in range(0, source_count, source_chunk_size):
                source_end = min(source_start + source_chunk_size, source_count)
                feature_chunk = features[:, source_start:source_end].index_select(
                    0, row_index
                )
                target.add_(
                    torch.einsum(
                        "npc,pco->no",
                        feature_chunk,
                        diagnostic_directions[source_start:source_end],
                    )
                )

            fitted = torch.zeros_like(target)
            baseline = torch.zeros_like(target)
            for selected_start in range(0, target_count, selected_chunk_size):
                selected_end = min(
                    selected_start + selected_chunk_size, target_count
                )
                path_index = selected_index[selected_start:selected_end]
                feature_chunk = features[
                    row_index[:, None], path_index[None, :], :
                ]
                fitted.add_(
                    torch.einsum(
                        "ntc,tco->no",
                        feature_chunk,
                        aggregate_directions[selected_start:selected_end],
                    )
                )
                baseline.add_(
                    torch.einsum(
                        "ntc,tco->no",
                        feature_chunk,
                        selected_native_directions[selected_start:selected_end],
                    )
                )
            _update_metric_pair(torch, pair, target, baseline, fitted)

            if not grouped:
                continue
            local_prompt_keys = [prompt_keys[index] for index in row_values]
            local_domain_keys = [domain_keys[index] for index in row_values]
            for key in _ordered_unique(local_prompt_keys):
                local_index = torch.tensor(
                    [
                        index
                        for index, row_key in enumerate(local_prompt_keys)
                        if row_key == key
                    ],
                    dtype=torch.long,
                )
                group_pair = prompt_pairs.setdefault(key, _new_metric_pair())
                _update_metric_pair(
                    torch,
                    group_pair,
                    target.index_select(0, local_index),
                    baseline.index_select(0, local_index),
                    fitted.index_select(0, local_index),
                )
            for key in _ordered_unique(local_domain_keys):
                local_index = torch.tensor(
                    [
                        index
                        for index, row_key in enumerate(local_domain_keys)
                        if row_key == key
                    ],
                    dtype=torch.long,
                )
                group_pair = domain_pairs.setdefault(key, _new_metric_pair())
                _update_metric_pair(
                    torch,
                    group_pair,
                    target.index_select(0, local_index),
                    baseline.index_select(0, local_index),
                    fitted.index_select(0, local_index),
                )

    evaluate_rows(train_rows, train_pair, grouped=False)
    evaluate_rows(validation_rows, validation_pair, grouped=True)
    validation_prompt_keys = [prompt_keys[index] for index in validation_rows]
    validation_domain_keys = [domain_keys[index] for index in validation_rows]
    prompt_reports = [
        {
            "prompt_id": prompt_original[key],
            "row_count": int(prompt_pairs[key]["baseline"]["row_count"]),
            **_finalize_metric_pair(torch, prompt_pairs[key]),
        }
        for key in _ordered_unique(validation_prompt_keys)
    ]
    domain_reports = [
        {
            "domain": domain_original[key],
            "row_count": int(domain_pairs[key]["baseline"]["row_count"]),
            **_finalize_metric_pair(torch, domain_pairs[key]),
        }
        for key in _ordered_unique(validation_domain_keys)
    ]
    report = {
        "method": "bounded-row-and-source-chunk-accumulation",
        "row_projection_used": False,
        "prompt_grouping_preserved": True,
        "domain_grouping_preserved": True,
        "row_chunk_size": row_chunk_size,
        "source_path_chunk_size": source_chunk_size,
        "selected_path_chunk_size": selected_chunk_size,
        "response_matrix_shape_bound": [row_chunk_size, diagnostic_outputs],
        "response_matrix_element_bound": row_chunk_size * diagnostic_outputs,
        "source_feature_tensor_shape_bound": [
            row_chunk_size,
            source_chunk_size,
            channel_count,
        ],
        "source_feature_tensor_element_bound": (
            row_chunk_size * source_chunk_size * channel_count
        ),
        "selected_feature_tensor_shape_bound": [
            row_chunk_size,
            selected_chunk_size,
            channel_count,
        ],
        "selected_feature_tensor_element_bound": (
            row_chunk_size * selected_chunk_size * channel_count
        ),
    }
    return {
        "train": _finalize_metric_pair(torch, train_pair),
        "held_out": _finalize_metric_pair(torch, validation_pair),
        "by_prompt": prompt_reports,
        "by_domain": domain_reports,
        "materialization": report,
    }


def _response_metrics(torch, target, prediction) -> dict[str, float]:
    difference = prediction - target
    absolute_rmse = float(difference.square().mean().sqrt().item())
    mean_absolute_error = float(difference.abs().mean().item())
    target_norm = target.norm(dim=1)
    prediction_norm = prediction.norm(dim=1)
    denominator = target_norm * prediction_norm
    epsilon = torch.finfo(torch.float64).eps
    row_dot = (target * prediction).sum(dim=1)
    cosine = torch.where(
        denominator > epsilon,
        row_dot / denominator.clamp_min(epsilon),
        torch.where(
            (target_norm <= epsilon) & (prediction_norm <= epsilon),
            torch.ones_like(denominator),
            torch.zeros_like(denominator),
        ),
    )
    flat_denominator = float(target.norm().item() * prediction.norm().item())
    flat_cosine = (
        float((target * prediction).sum().item()) / flat_denominator
        if flat_denominator > epsilon
        else float(bool(target.norm().item() <= epsilon and prediction.norm().item() <= epsilon))
    )
    target_energy = float(target.square().sum().item())
    squared_error = float(difference.square().sum().item())
    explained_energy = (
        1.0 - squared_error / target_energy if target_energy > epsilon else 1.0
    )
    return {
        "absolute_rmse": absolute_rmse,
        "mean_absolute_error": mean_absolute_error,
        "cosine_similarity_mean": float(cosine.mean().item()),
        "cosine_similarity_min": float(cosine.min().item()),
        "cosine_similarity_global": flat_cosine,
        "target_rms": float(target.square().mean().sqrt().item()),
        "prediction_rms": float(prediction.square().mean().sqrt().item()),
        "explained_response_energy": explained_energy,
    }


def _group_metrics(
    torch,
    target,
    baseline,
    fitted,
    row_keys: Sequence[str],
    ordered_keys: Sequence[str],
    original_by_key: Mapping[str, Any],
    label_name: str,
) -> list[dict[str, Any]]:
    reports = []
    for key in ordered_keys:
        indices = [index for index, row_key in enumerate(row_keys) if row_key == key]
        if not indices:
            continue
        index = torch.tensor(indices, dtype=torch.long)
        reports.append(
            {
                label_name: original_by_key[key],
                "row_count": len(indices),
                "baseline": _response_metrics(
                    torch,
                    target.index_select(0, index),
                    baseline.index_select(0, index),
                ),
                "fitted": _response_metrics(
                    torch,
                    target.index_select(0, index),
                    fitted.index_select(0, index),
                ),
            }
        )
    return reports


def _support_diversity(torch, signature, selected: Sequence[int]):
    selected_signature = signature.index_select(
        0, torch.tensor(selected, dtype=torch.long)
    )
    norms = selected_signature.norm(dim=1, keepdim=True)
    normalized = selected_signature / norms.clamp_min(torch.finfo(torch.float64).eps)
    cosine = normalized.matmul(normalized.t()).abs()
    if len(selected) > 1:
        mask = ~torch.eye(len(selected), dtype=torch.bool)
        off_diagonal = cosine[mask]
        mean_cosine = float(off_diagonal.mean().item())
        max_cosine = float(off_diagonal.max().item())
    else:
        mean_cosine = 0.0
        max_cosine = 0.0
    singular_values = torch.linalg.svdvals(selected_signature)
    threshold = (
        float(singular_values.max().item())
        * max(selected_signature.shape)
        * torch.finfo(torch.float64).eps
        if singular_values.numel()
        else 0.0
    )
    numerical_rank = int((singular_values > threshold).sum().item())
    return {
        "numerical_signature_rank": numerical_rank,
        "mean_absolute_pairwise_cosine": mean_cosine,
        "max_absolute_pairwise_cosine": max_cosine,
    }


def compress_post_nonlinearity_paths(
    path_features,
    projected_output_directions,
    *,
    prompt_ids: Sequence[Any],
    domain_labels: Sequence[Any],
    donor_layer_ids: Sequence[Any],
    target_path_count: int,
    minimum_paths_per_layer: Mapping[Any, int] | int | None = None,
    minimum_paths_per_domain: Mapping[Any, int] | int | None = None,
    domain_weights: Mapping[Any, float] | None = None,
    path_weights: Sequence[float] | Any | None = None,
    path_kind: str = "swiglu",
    ridge: float = 1e-8,
    validation_fraction: float = 0.25,
    min_train_prompts: int = 2,
    min_validation_prompts: int = 1,
    split_seed: int = 0,
    selection_sketch_size: int = 256,
    interpolation_mode: str = "dense",
    sparse_chunk_size: int = _SPARSE_DEFAULT_CHUNK_SIZE,
    diagnostic_sketch_size: int = _SPARSE_DIAGNOSTIC_OUTPUT_SKETCH_SIZE,
) -> dict[str, Any]:
    """Select and merge realized donor paths without gradient updates.

    ``path_features`` is ``[rows, paths]`` for scalar post-SiLU SwiGLU
    activations, paired with directions ``[paths, output]``.  It may instead be
    ``[rows, paths, channels]`` for post-softmax attention-head responses,
    paired with directions ``[paths, channels, output]``.

    The return value contains actual selected donor indices, a mapping from
    every donor path to a selected path, interpolation coefficients, aggregate
    selected-path output coefficients, and prompt-held-out reports.  Dense
    interpolation remains the default.  ``sparse-signature`` returns one
    selected-path assignment and one signed scalar for each source path.
    """

    torch = _torch()
    features_input = _validate_tensor(
        torch, path_features, "path_features", {2, 3}
    )
    directions_input = _validate_tensor(
        torch, projected_output_directions, "projected_output_directions", {2, 3}
    )
    if features_input.ndim != directions_input.ndim:
        raise ValueError(
            "path_features and projected_output_directions must both be rank 2 "
            "or both be rank 3."
        )
    row_count = int(features_input.shape[0])
    path_count = int(features_input.shape[1])
    if int(directions_input.shape[0]) != path_count:
        raise ValueError("Feature and output-direction path counts must match.")
    if features_input.ndim == 3 and int(features_input.shape[2]) != int(
        directions_input.shape[1]
    ):
        raise ValueError("Feature and output-direction channel counts must match.")
    if path_kind not in {"swiglu", "attention"}:
        raise ValueError("path_kind must be 'swiglu' or 'attention'.")
    if path_kind == "swiglu" and features_input.ndim != 2:
        raise ValueError("swiglu path_features must have shape [rows, paths].")
    if not math.isfinite(float(ridge)) or float(ridge) < 0.0:
        raise ValueError("ridge must be finite and non-negative.")
    if interpolation_mode not in {"dense", "sparse-signature"}:
        raise ValueError(
            "interpolation_mode must be 'dense' or 'sparse-signature'."
        )
    target_count = _positive_int(target_path_count, "target_path_count")
    if target_count > path_count:
        raise ValueError("target_path_count cannot exceed the donor path count.")
    sketch_size = _positive_int(selection_sketch_size, "selection_sketch_size")
    sparse_chunk = _positive_int(sparse_chunk_size, "sparse_chunk_size")
    diagnostic_size = _positive_int(
        diagnostic_sketch_size, "diagnostic_sketch_size"
    )

    prompt_values, prompt_keys = _coerce_labels(prompt_ids, row_count, "prompt_ids")
    domain_values, domain_keys = _coerce_labels(
        domain_labels, row_count, "domain_labels"
    )
    layer_values, layer_keys = _coerce_labels(
        donor_layer_ids, path_count, "donor_layer_ids"
    )
    ordered_domain_keys = _ordered_unique(domain_keys)
    ordered_layer_keys = _ordered_unique(layer_keys)
    domain_original = {
        key: value for key, value in zip(domain_keys, domain_values, strict=True)
    }
    layer_original = {
        key: value for key, value in zip(layer_keys, layer_values, strict=True)
    }

    layer_quotas = _quota_map(
        minimum_paths_per_layer,
        ordered_layer_keys,
        layer_original,
        "minimum_paths_per_layer",
    )
    if sum(layer_quotas.values()) > target_count:
        raise ValueError("Layer minimum quotas exceed target_path_count.")
    for key, quota in layer_quotas.items():
        if quota > layer_keys.count(key):
            raise ValueError(
                f"Layer quota for {layer_original[key]!r} exceeds available paths."
            )

    train_rows, validation_rows, split_report = _prompt_split(
        prompt_values,
        prompt_keys,
        domain_keys,
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
    )
    train_domain_keys = [domain_keys[index] for index in train_rows]
    row_weights, row_weight_report = _domain_balanced_row_weights(
        torch,
        train_domain_keys,
        ordered_domain_keys,
        domain_weights,
        domain_original,
    )

    named_inputs = {
        "path_features": features_input,
        "projected_output_directions": directions_input,
    }
    if torch.is_tensor(path_weights):
        named_inputs["path_weights"] = path_weights
    input_hashes_before = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradient_hashes_before = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    metadata_before = _json_sha256(
        {
            "prompt_ids": prompt_values,
            "domain_labels": domain_values,
            "donor_layer_ids": layer_values,
        }
    )

    with torch.no_grad():
        autograd_enabled_during_fit = torch.is_grad_enabled()
        features = features_input.detach().cpu().to(torch.float64)
        directions = directions_input.detach().cpu().to(torch.float64)
        if features.ndim == 2:
            features = features.unsqueeze(2)
            directions = directions.unsqueeze(1)
        channel_count = int(features.shape[2])
        output_count = int(directions.shape[2])
        path_weight_values, path_weight_source = _coerce_path_weights(
            torch, path_weights, path_count
        )

        domain_mass, dominant_domain_keys = _path_domain_mass(
            torch,
            features,
            directions,
            train_rows,
            train_domain_keys,
            ordered_domain_keys,
            row_weights,
        )
        domain_quotas = _quota_map(
            minimum_paths_per_domain,
            ordered_domain_keys,
            domain_original,
            "minimum_paths_per_domain",
        )
        if sum(domain_quotas.values()) > target_count:
            raise ValueError("Domain minimum quotas exceed target_path_count.")
        for key, quota in domain_quotas.items():
            if quota > dominant_domain_keys.count(key):
                raise ValueError(
                    f"Domain quota for {domain_original[key]!r} exceeds paths "
                    "whose largest response mass is in that domain."
                )

        train_index = torch.tensor(train_rows, dtype=torch.long)
        train_features = features.index_select(0, train_index)
        signature, sketch_report = _response_signature(
            torch,
            train_features,
            directions,
            row_weights,
            path_weight_values,
            sketch_size=sketch_size,
            split_seed=split_seed,
        )
        sparse_selection_report = None
        if interpolation_mode == "sparse-signature":
            (
                selected,
                layer_counts,
                domain_counts,
                pivots,
                pivot_rank,
                sparse_selection_report,
            ) = _select_support_sparse(
                torch,
                signature,
                layer_keys,
                dominant_domain_keys,
                layer_quotas,
                domain_quotas,
                target_count,
            )
        else:
            (
                selected,
                layer_counts,
                domain_counts,
                pivots,
                pivot_rank,
            ) = _select_support(
                torch,
                signature,
                layer_keys,
                dominant_domain_keys,
                layer_quotas,
                domain_quotas,
                target_count,
            )

        selected_index = torch.tensor(selected, dtype=torch.long)
        sparse_materialization = None
        diagnostic_evaluation = None
        if interpolation_mode == "dense":
            all_train_design = train_features.reshape(
                len(train_rows), path_count * channel_count
            )
            selected_train_design = train_features.index_select(
                1, selected_index
            ).reshape(len(train_rows), target_count * channel_count)
            interpolation, fit_report = _fit_interpolation(
                torch,
                selected_train_design,
                all_train_design,
                row_weights,
                float(ridge),
            )
            flattened_directions = directions.reshape(
                path_count * channel_count, output_count
            )
            aggregate_flat = interpolation.matmul(flattened_directions)
            aggregate_coefficients = aggregate_flat.reshape(
                target_count, channel_count, output_count
            )
            selected_native_directions = directions.index_select(
                0, selected_index
            )

            source_to_selected = interpolation.t().reshape(
                path_count, channel_count, target_count, channel_count
            )
            source_mapping = []
            source_transfer_norms = []
            source_output_contribution_norms = []
            selected_signature = signature.index_select(0, selected_index)
            selected_signature_norm = selected_signature.norm(dim=1)
            epsilon = torch.finfo(torch.float64).eps
            for path_index in range(path_count):
                coefficient_block = interpolation[
                    :, path_index * channel_count : (path_index + 1) * channel_count
                ].reshape(target_count, channel_count, channel_count)
                per_selected_norm = coefficient_block.square().sum(dim=(1, 2)).sqrt()
                transfer_norm = float(per_selected_norm.norm().item())
                source_transfer_norms.append(transfer_norm)
                output_contribution = interpolation[
                    :, path_index * channel_count : (path_index + 1) * channel_count
                ].matmul(directions[path_index])
                source_output_contribution_norms.append(
                    float(output_contribution.norm().item())
                )
                if float(per_selected_norm.max().item()) > epsilon:
                    selected_position = int(per_selected_norm.argmax().item())
                else:
                    source_signature = signature[path_index]
                    similarities = selected_signature.matmul(source_signature).abs()
                    similarities.div_(
                        selected_signature_norm.mul(source_signature.norm()).clamp_min(
                            epsilon
                        )
                    )
                    selected_position = int(similarities.argmax().item())
                source_mapping.append(selected[selected_position])

            all_design = features.reshape(row_count, path_count * channel_count)
            selected_design = features.index_select(1, selected_index).reshape(
                row_count, target_count * channel_count
            )
            target_response = all_design.matmul(flattened_directions)
            fitted_response = selected_design.matmul(aggregate_flat)
            baseline_response = selected_design.matmul(
                selected_native_directions.reshape(
                    target_count * channel_count, output_count
                )
            )

            train_target = target_response.index_select(0, train_index)
            train_fitted = fitted_response.index_select(0, train_index)
            train_baseline = baseline_response.index_select(0, train_index)
            validation_index = torch.tensor(validation_rows, dtype=torch.long)
            validation_target = target_response.index_select(0, validation_index)
            validation_fitted = fitted_response.index_select(0, validation_index)
            validation_baseline = baseline_response.index_select(
                0, validation_index
            )
            train_baseline_metrics = _response_metrics(
                torch, train_target, train_baseline
            )
            train_fitted_metrics = _response_metrics(
                torch, train_target, train_fitted
            )
            validation_baseline_metrics = _response_metrics(
                torch, validation_target, validation_baseline
            )
            validation_fitted_metrics = _response_metrics(
                torch, validation_target, validation_fitted
            )
        else:
            (
                selected_positions,
                source_to_selected,
                fit_report,
                sparse_materialization,
            ) = _fit_sparse_signature_interpolation(
                torch,
                signature,
                selected,
                ridge=float(ridge),
                requested_chunk_size=sparse_chunk,
            )
            (
                aggregate_coefficients,
                source_output_norm_tensor,
                aggregation_materialization,
            ) = _aggregate_sparse_output_directions(
                torch,
                directions,
                selected_positions,
                source_to_selected,
                target_count=target_count,
                requested_chunk_size=sparse_chunk,
            )
            selected_native_directions = directions.index_select(
                0, selected_index
            )
            source_mapping_tensor = selected_index.index_select(
                0, selected_positions
            )
            source_mapping = source_mapping_tensor.tolist()
            source_transfer_norms = source_to_selected.abs().tolist()
            source_output_contribution_norms = source_output_norm_tensor.tolist()
            sparse_materialization["output_aggregation"] = (
                aggregation_materialization
            )

            diagnostic_directions, diagnostic_projection = (
                _diagnostic_output_projection(
                    torch,
                    directions,
                    sketch_size=diagnostic_size,
                    split_seed=split_seed,
                    requested_chunk_size=sparse_chunk,
                )
            )
            (
                diagnostic_aggregate,
                _diagnostic_source_norms,
                diagnostic_aggregation,
            ) = _aggregate_sparse_output_directions(
                torch,
                diagnostic_directions,
                selected_positions,
                source_to_selected,
                target_count=target_count,
                requested_chunk_size=sparse_chunk,
            )
            diagnostic_native = diagnostic_directions.index_select(
                0, selected_index
            )
            diagnostics = _chunked_sparse_response_diagnostics(
                torch,
                features,
                diagnostic_directions,
                selected_index,
                diagnostic_aggregate,
                diagnostic_native,
                train_rows=train_rows,
                validation_rows=validation_rows,
                prompt_keys=prompt_keys,
                domain_keys=domain_keys,
                prompt_original={
                    key: value
                    for key, value in zip(prompt_keys, prompt_values, strict=True)
                },
                domain_original=domain_original,
                requested_chunk_size=sparse_chunk,
                requested_row_chunk_size=_SPARSE_DIAGNOSTIC_ROW_CHUNK_SIZE,
            )
            train_baseline_metrics = diagnostics["train"]["baseline"]
            train_fitted_metrics = diagnostics["train"]["fitted"]
            validation_baseline_metrics = diagnostics["held_out"]["baseline"]
            validation_fitted_metrics = diagnostics["held_out"]["fitted"]
            prompt_metrics = diagnostics["by_prompt"]
            domain_metrics = diagnostics["by_domain"]
            diagnostic_evaluation = {
                "approximate": diagnostic_projection["approximate"],
                "approximation_explicit": True,
                "output_space": diagnostic_projection,
                "response_accumulation": diagnostics["materialization"],
                "sketched_output_aggregation": diagnostic_aggregation,
            }

    input_hashes_after = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradient_hashes_after = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    metadata_after = _json_sha256(
        {
            "prompt_ids": prompt_values,
            "domain_labels": domain_values,
            "donor_layer_ids": layer_values,
        }
    )
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("An input tensor changed during path decomposition.")
    if gradient_hashes_before != gradient_hashes_after:
        raise RuntimeError("An input gradient changed during path decomposition.")
    if metadata_before != metadata_after:
        raise RuntimeError("Input metadata changed during path decomposition.")

    validation_prompt_keys = [prompt_keys[index] for index in validation_rows]
    validation_domain_keys = [domain_keys[index] for index in validation_rows]
    validation_prompt_original = {
        key: value for key, value in zip(prompt_keys, prompt_values, strict=True)
    }
    selected_mass = float(domain_mass.index_select(0, selected_index).sum().item())
    total_mass = float(domain_mass.sum().item())
    nonzero_threshold = torch.finfo(torch.float64).eps * max(
        1.0, max(source_output_contribution_norms, default=0.0)
    )
    held_out_rmse_reduction = (
        validation_baseline_metrics["absolute_rmse"]
        - validation_fitted_metrics["absolute_rmse"]
    )
    held_out_cosine_gain = (
        validation_fitted_metrics["cosine_similarity_mean"]
        - validation_baseline_metrics["cosine_similarity_mean"]
    )

    layer_coverage = []
    for key in ordered_layer_keys:
        source_indices = [index for index, item in enumerate(layer_keys) if item == key]
        source_mass = float(domain_mass[source_indices].sum().item())
        layer_coverage.append(
            {
                "layer": layer_original[key],
                "source_path_count": len(source_indices),
                "selected_path_count": layer_counts[key],
                "minimum_quota": layer_quotas[key],
                "quota_satisfied": layer_counts[key] >= layer_quotas[key],
                "source_path_mass": source_mass,
                "selected_native_mass": float(
                    domain_mass[
                        [index for index in selected if layer_keys[index] == key]
                    ].sum().item()
                ),
            }
        )
    domain_coverage = []
    for domain_index, key in enumerate(ordered_domain_keys):
        domain_coverage.append(
            {
                "domain": domain_original[key],
                "source_response_mass": float(domain_mass[:, domain_index].sum().item()),
                "selected_native_response_mass": float(
                    domain_mass.index_select(0, selected_index)[:, domain_index].sum().item()
                ),
                "selected_dominant_path_count": domain_counts[key],
                "minimum_quota": domain_quotas[key],
                "quota_satisfied": domain_counts[key] >= domain_quotas[key],
            }
        )

    if interpolation_mode == "dense":
        prompt_metrics = _group_metrics(
            torch,
            validation_target,
            validation_baseline,
            validation_fitted,
            validation_prompt_keys,
            _ordered_unique(validation_prompt_keys),
            validation_prompt_original,
            "prompt_id",
        )
        domain_metrics = _group_metrics(
            torch,
            validation_target,
            validation_baseline,
            validation_fitted,
            validation_domain_keys,
            _ordered_unique(validation_domain_keys),
            domain_original,
            "domain",
        )

    if interpolation_mode == "dense":
        coefficient_shape = [
            path_count,
            channel_count,
            target_count,
            channel_count,
        ]
        coefficient_hashes = {
            "interpolation_sha256": _tensor_sha256(torch, interpolation)
        }
    else:
        coefficient_shape = [path_count]
        coefficient_hashes = {
            "source_scalar_coefficients_sha256": _tensor_sha256(
                torch, source_to_selected
            )
        }
        matrix_shape_bounds = {
            "selector_candidate_signature": sparse_selection_report[
                "complexity_bounds"
            ]["candidate_signature_matrix_shape_bound"],
            "assignment_similarity": sparse_materialization[
                "assignment_similarity_matrix_shape_bound"
            ],
            "diagnostic_output_projection": diagnostic_evaluation["output_space"][
                "projection_matrix_shape_bound"
            ],
            "diagnostic_projection_multiply_result": diagnostic_evaluation[
                "output_space"
            ]["projection_multiply_result_matrix_shape_bound"],
            "diagnostic_response": diagnostic_evaluation[
                "response_accumulation"
            ]["response_matrix_shape_bound"],
        }
        matrix_element_bounds = {
            "selector_candidate_signature": sparse_selection_report[
                "complexity_bounds"
            ]["candidate_signature_matrix_element_bound"],
            "assignment_similarity": sparse_materialization[
                "assignment_similarity_matrix_element_bound"
            ],
            "diagnostic_output_projection": diagnostic_evaluation["output_space"][
                "projection_matrix_element_bound"
            ],
            "diagnostic_projection_multiply_result": diagnostic_evaluation[
                "output_space"
            ]["projection_multiply_result_matrix_element_bound"],
            "diagnostic_response": diagnostic_evaluation[
                "response_accumulation"
            ]["response_matrix_element_bound"],
        }
        peak_matrix_name = max(matrix_element_bounds, key=matrix_element_bounds.get)
        sparse_materialization.update(
            {
                "matrix_shape_bounds": matrix_shape_bounds,
                "matrix_element_bounds": matrix_element_bounds,
                "peak_materialized_matrix_name": peak_matrix_name,
                "peak_materialized_matrix_shape_bound": matrix_shape_bounds[
                    peak_matrix_name
                ],
                "peak_materialized_matrix_element_bound": matrix_element_bounds[
                    peak_matrix_name
                ],
                "linear_source_mapping_shape": [path_count],
                "linear_source_scalar_coefficient_shape": [path_count],
            }
        )

    selected_layer_values = [layer_values[index] for index in selected]
    selected_domain_values = [
        domain_original[dominant_domain_keys[index]] for index in selected
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": "post-nonlinearity-contribution-aware-path-decomposition",
        "interpolation_mode": interpolation_mode,
        "zero_gradient": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_enabled_during_fit": bool(autograd_enabled_during_fit),
        "inputs_mutated": False,
        "input_gradients_mutated": False,
        "provenance": {
            "path_kind": path_kind,
            "input_stage": (
                "post-silu-swiglu-activation"
                if path_kind == "swiglu"
                else "post-softmax-attention-head-response"
            ),
            "selected_paths_are_actual_donor_paths": True,
            "all_donor_paths_in_aggregate_fit_target": True,
            "qk_averaged_before_softmax": False,
            "gate_up_averaged_before_silu": False,
            "pre_nonlinearity_averaging": False,
            "tensors_detached": True,
        },
        "shape": {
            "rows": row_count,
            "source_paths": path_count,
            "selected_paths": target_count,
            "channels_per_path": channel_count,
            "output_features": output_count,
            "aggregate_output_coefficients": [
                target_count,
                channel_count,
                output_count,
            ],
            "source_to_selected_coefficients": [
                *coefficient_shape,
            ],
        },
        "split_provenance": split_report,
        "row_weighting": row_weight_report,
        "selection": {
            "method": "deterministic-domain-balanced-weighted-pivoted-qr",
            "support_type": "actual-donor-paths",
            "selected_path_indices": selected,
            "selected_layer_ids": selected_layer_values,
            "selected_dominant_domains": selected_domain_values,
            "target_path_count": target_count,
            "pivot_numerical_rank": pivot_rank,
            "pivots": pivots,
            "sketch": sketch_report,
            "path_weight_source": path_weight_source,
            **(
                {"sparse_scalability": sparse_selection_report}
                if sparse_selection_report is not None
                else {}
            ),
        },
        "coefficient_fit": {
            **fit_report,
            "interpolation_mode": interpolation_mode,
            "every_source_path_entered_response_target": True,
            "source_path_count": path_count,
            "all_path_coverage_fraction": len(source_mapping) / path_count,
            "all_source_paths_have_finite_coefficients": bool(
                torch.isfinite(source_to_selected).all().item()
            ),
            "source_paths_with_nonzero_fitted_output_contribution": sum(
                value > nonzero_threshold
                for value in source_output_contribution_norms
            ),
            "source_transfer_norms": source_transfer_norms,
            "source_output_contribution_norms": source_output_contribution_norms,
            "aggregate_output_coefficient_norm": float(
                aggregate_coefficients.norm().item()
            ),
        },
        "coverage": {
            "all_source_paths_mapped": len(source_mapping) == path_count,
            "mapped_source_path_count": len(source_mapping),
            "all_path_coverage_fraction": len(source_mapping) / path_count,
            "layers": layer_coverage,
            "domains": domain_coverage,
        },
        "mass": {
            "measure": "domain-balanced-sum-of-individual-path-response-squared-norms",
            "all_source_path_mass": total_mass,
            "selected_native_path_mass": selected_mass,
            "selected_native_mass_fraction": (
                selected_mass / total_mass if total_mass > 0.0 else 1.0
            ),
        },
        "diversity": {
            **_support_diversity(torch, signature, selected),
            "selected_dominant_domain_count": len(set(selected_domain_values)),
            "selected_dominant_domains": list(dict.fromkeys(selected_domain_values)),
        },
        "diagnostics": {
            "target": "sum of every donor post-nonlinearity path response",
            "baseline": "sum of selected paths with their native output directions",
            "train": {
                "baseline": train_baseline_metrics,
                "fitted": train_fitted_metrics,
            },
            "held_out": {
                "unit": "prompt",
                "approximate": bool(
                    diagnostic_evaluation["approximate"]
                    if diagnostic_evaluation is not None
                    else False
                ),
                "approximation_explicit": True,
                "baseline": validation_baseline_metrics,
                "fitted": validation_fitted_metrics,
                "absolute_rmse_reduction": held_out_rmse_reduction,
                "cosine_similarity_gain": held_out_cosine_gain,
                "improved_absolute_rmse": held_out_rmse_reduction > 0.0,
                "by_prompt": prompt_metrics,
                "by_domain": domain_metrics,
            },
            **(
                {"scalable_evaluation": diagnostic_evaluation}
                if diagnostic_evaluation is not None
                else {}
            ),
        },
        **(
            {"materialization": sparse_materialization}
            if sparse_materialization is not None
            else {}
        ),
        "hashes": {
            "algorithm": "sha256",
            "inputs_before": input_hashes_before,
            "inputs_after": input_hashes_after,
            "input_gradients_before": gradient_hashes_before,
            "input_gradients_after": gradient_hashes_after,
            "metadata_before": metadata_before,
            "metadata_after": metadata_after,
            "inputs_unchanged": True,
            "input_gradients_unchanged": True,
            "prompt_split_sha256": split_report["assignment_sha256"],
            "response_signature_sha256": _tensor_sha256(torch, signature),
            "selected_path_indices_sha256": _tensor_sha256(torch, selected_index),
            **coefficient_hashes,
            "aggregate_output_coefficients_sha256": _tensor_sha256(
                torch, aggregate_coefficients
            ),
            "source_mapping_sha256": _json_sha256(source_mapping),
            "held_out_diagnostics_sha256": _json_sha256(
                {
                    "baseline": validation_baseline_metrics,
                    "fitted": validation_fitted_metrics,
                    "by_prompt": prompt_metrics,
                    "by_domain": domain_metrics,
                }
            ),
        },
    }

    if interpolation_mode == "sparse-signature":
        aggregate_return = (
            aggregate_coefficients[:, 0].contiguous()
            if channel_count == 1
            else aggregate_coefficients.contiguous()
        )
        native_return = (
            selected_native_directions[:, 0].contiguous()
            if channel_count == 1
            else selected_native_directions.contiguous()
        )
        transfer_return = source_to_selected.contiguous()
    elif channel_count == 1:
        aggregate_return = aggregate_coefficients[:, 0].contiguous()
        native_return = selected_native_directions[:, 0].contiguous()
        transfer_return = source_to_selected[:, 0, :, 0].contiguous()
    else:
        aggregate_return = aggregate_coefficients.contiguous()
        native_return = selected_native_directions.contiguous()
        transfer_return = source_to_selected.contiguous()
    result = {
        "selected_path_indices": selected_index,
        "selected_layer_ids": selected_layer_values,
        "selected_dominant_domains": selected_domain_values,
        "source_path_to_selected_path": torch.tensor(
            source_mapping, dtype=torch.long
        ),
        "source_to_selected_coefficients": transfer_return,
        "aggregate_output_coefficients": aggregate_return,
        "selected_native_output_directions": native_return,
        "report": report,
    }
    if interpolation_mode == "sparse-signature":
        result["source_scalar_coefficients"] = transfer_return
    return result
