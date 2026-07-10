"""Zero-gradient layer-varying logit-observable charts.

This module fits a separate low-rank chart at each donor boundary.  A chart is
an oblique encoder/decoder pair that predicts a shared future observable rather
than preserving directions merely because they have high activation variance.
All fitting is closed form on detached FP64 CPU tensors.  No optimizer,
autograd graph, backward pass, adapter, or input mutation is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


REPORT_SCHEMA_VERSION = 1

__all__ = [
    "apply_chart_transition",
    "apply_observable_chart",
    "compose_chart_transitions",
    "decode_observable_chart",
    "encode_observable_chart",
    "fit_layerwise_logit_observable_charts",
    "fit_logit_observable_chart",
    "fit_logit_observable_charts",
    "materialize_chart_transition_maps",
    "predict_future_observable",
    "split_prompt_rows",
]


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


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _relative_tolerance(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("rank_tolerance must be finite and strictly between 0 and 1.")
    return result


def _validate_matrix(torch, value: Any, name: str):
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 tensor.")
    if int(value.shape[0]) <= 0 or int(value.shape[1]) <= 0:
        raise ValueError(f"{name} must have positive dimensions.")
    if not value.dtype.is_floating_point:
        raise ValueError(f"{name} must have a floating-point dtype.")
    if not bool(torch.isfinite(value.detach()).all().item()):
        raise ValueError(f"{name} must contain only finite values.")
    return value


def _tensor_sha256(torch, tensor) -> str:
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
    row_count = int(detached.shape[0]) if detached.ndim else 1
    for start in range(0, row_count, 4096):
        chunk = detached[start : start + 4096] if detached.ndim else detached
        digest.update(
            chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _gradient_fingerprint(torch, tensor) -> dict[str, Any]:
    gradient = tensor.grad if tensor.is_leaf else None
    if gradient is None:
        return {"present": False, "sha256": None}
    return {"present": True, "sha256": _tensor_sha256(torch, gradient)}


def _normalize_identifier(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    return str(value)


def split_prompt_rows(
    prompt_ids: Sequence[Any] | None,
    row_count: int,
    *,
    validation_fraction: float = 0.25,
    min_train_prompts: int = 2,
    min_validation_prompts: int = 1,
    split_seed: int = 0,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Create a deterministic whole-prompt train/validation row split."""

    rows = _positive_int(row_count, "row_count")
    fraction = float(validation_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    minimum_train = _positive_int(min_train_prompts, "min_train_prompts")
    minimum_validation = _positive_int(
        min_validation_prompts, "min_validation_prompts"
    )
    if prompt_ids is None:
        normalized = [f"implicit-row-{index}" for index in range(rows)]
        source = "implicit-row-ids"
    else:
        if len(prompt_ids) != rows:
            raise ValueError("prompt_ids length must match the boundary row count.")
        normalized = [_normalize_identifier(item) for item in prompt_ids]
        source = "provided"

    unique_prompts = list(dict.fromkeys(normalized))
    required = minimum_train + minimum_validation
    if len(unique_prompts) < required:
        raise ValueError(
            "Not enough distinct prompts for a prompt-level split: "
            f"need at least {required}, got {len(unique_prompts)}."
        )
    validation_count = max(
        minimum_validation,
        int(round(len(unique_prompts) * fraction)),
    )
    validation_count = min(
        validation_count, len(unique_prompts) - minimum_train
    )
    ranked = sorted(
        enumerate(unique_prompts),
        key=lambda item: hashlib.sha256(
            f"{int(split_seed)}\x00{item[0]}\x00{item[1]}".encode("utf-8")
        ).digest(),
    )
    validation_prompt_set = {item[1] for item in ranked[:validation_count]}
    train_prompts = [
        item for item in unique_prompts if item not in validation_prompt_set
    ]
    validation_prompts = [
        item for item in unique_prompts if item in validation_prompt_set
    ]
    train_rows = [
        index
        for index, prompt in enumerate(normalized)
        if prompt not in validation_prompt_set
    ]
    validation_rows = [
        index
        for index, prompt in enumerate(normalized)
        if prompt in validation_prompt_set
    ]
    if set(train_rows) & set(validation_rows):
        raise RuntimeError("Prompt split produced overlapping rows.")
    if sorted(train_rows + validation_rows) != list(range(rows)):
        raise RuntimeError("Prompt split omitted rows.")

    report = {
        "unit": "prompt",
        "source": source,
        "seed": int(split_seed),
        "validation_fraction": fraction,
        "prompt_count": len(unique_prompts),
        "train_prompt_count": len(train_prompts),
        "validation_prompt_count": len(validation_prompts),
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "train_prompt_ids": train_prompts,
        "validation_prompt_ids": validation_prompts,
        "train_row_indices": train_rows,
        "validation_row_indices": validation_rows,
        "row_prompt_ids_sha256": _json_sha256(normalized),
        "train_prompt_ids_sha256": _json_sha256(train_prompts),
        "validation_prompt_ids_sha256": _json_sha256(validation_prompts),
    }
    report["assignment_sha256"] = _json_sha256(
        {
            "seed": report["seed"],
            "train_prompt_ids": train_prompts,
            "validation_prompt_ids": validation_prompts,
        }
    )
    return train_rows, validation_rows, report


def _normalize_boundaries(
    torch,
    boundary_rows: Mapping[Any, Any] | Sequence[Any] | Any,
    boundary_names: Sequence[Any] | None,
) -> tuple[list[str], list[Any]]:
    if torch.is_tensor(boundary_rows):
        values = [boundary_rows]
        names = ["boundary_0"] if boundary_names is None else list(boundary_names)
    elif isinstance(boundary_rows, Mapping):
        if boundary_names is not None:
            raise ValueError("boundary_names cannot be used with a boundary mapping.")
        names = [_normalize_identifier(item) for item in boundary_rows.keys()]
        values = list(boundary_rows.values())
    elif isinstance(boundary_rows, Sequence) and not isinstance(
        boundary_rows, (str, bytes)
    ):
        values = list(boundary_rows)
        names = (
            [f"boundary_{index}" for index in range(len(values))]
            if boundary_names is None
            else [_normalize_identifier(item) for item in boundary_names]
        )
    else:
        raise TypeError("boundary_rows must be a tensor, sequence, or mapping.")
    if not values:
        raise ValueError("At least one boundary tensor is required.")
    if len(names) != len(values):
        raise ValueError("boundary_names length must match boundary_rows.")
    if len(set(names)) != len(names):
        raise ValueError("Boundary names must be unique.")
    matrices = [
        _validate_matrix(torch, value, f"boundary_rows[{name!r}]")
        for name, value in zip(names, values, strict=True)
    ]
    row_count = int(matrices[0].shape[0])
    if any(int(matrix.shape[0]) != row_count for matrix in matrices[1:]):
        raise ValueError("All boundary tensors must have the same row count.")
    return names, matrices


def _domain_row_weights(
    torch,
    domains: Sequence[Any] | None,
    domain_weights: Mapping[Any, float] | None,
    row_count: int,
    train_rows: Sequence[int],
) -> tuple[Any, list[str] | None, dict[str, Any]]:
    if domains is None:
        if domain_weights is not None:
            raise ValueError("domain_weights requires a domains value for every row.")
        labels = None
        raw_weights = torch.ones(row_count, dtype=torch.float64)
        normalized_mapping = {"__all__": 1.0}
    else:
        if len(domains) != row_count:
            raise ValueError("domains length must match the boundary row count.")
        labels = [_normalize_identifier(item) for item in domains]
        provided = {
            _normalize_identifier(key): _finite_nonnegative(
                value, f"domain_weights[{key!r}]"
            )
            for key, value in (domain_weights or {}).items()
        }
        normalized_mapping = {
            label: provided.get(label, 1.0) for label in sorted(set(labels))
        }
        raw_weights = torch.tensor(
            [normalized_mapping[label] for label in labels],
            dtype=torch.float64,
        )
    train_index = torch.tensor(train_rows, dtype=torch.long)
    train_mean = float(raw_weights.index_select(0, train_index).mean().item())
    if not math.isfinite(train_mean) or train_mean <= 0.0:
        raise ValueError("Training rows must have positive total domain weight.")
    normalized = raw_weights / train_mean
    report = {
        "mode": "domain" if labels is not None else "uniform",
        "domain_weights": normalized_mapping,
        "train_raw_weight_mean": train_mean,
        "normalization": "divide-by-training-row-mean",
        "normalized_row_weights_sha256": _tensor_sha256(torch, normalized),
        "ordered_domains_sha256": _json_sha256(labels) if labels is not None else None,
    }
    return normalized, labels, report


def _weighted_mean(torch, rows, weights):
    denominator = weights.sum()
    if not bool(torch.isfinite(denominator).item()) or float(denominator) <= 0.0:
        raise ValueError("Selected rows must have positive total weight.")
    return (rows * weights.unsqueeze(1)).sum(dim=0) / denominator


def _canonicalize_svd(torch, left, values, right):
    if int(values.numel()) == 0:
        return left, values, right
    pivots = right.abs().argmax(dim=1)
    pivot_values = right.gather(1, pivots.unsqueeze(1)).squeeze(1)
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return left * signs.unsqueeze(0), values, right * signs.unsqueeze(1)


def _compact_svd(torch, matrix):
    left, values, right = torch.linalg.svd(matrix, full_matrices=False)
    if not bool(torch.isfinite(values).all().item()):
        raise RuntimeError("FP64 compact SVD produced non-finite singular values.")
    return _canonicalize_svd(torch, left, values, right)


def _numerical_rank(torch, values, shape, relative_tolerance: float) -> int:
    if int(values.numel()) == 0 or float(values[0].item()) == 0.0:
        return 0
    threshold = max(
        float(relative_tolerance) * float(values[0].item()),
        max(int(shape[0]), int(shape[1]))
        * torch.finfo(torch.float64).eps
        * float(values[0].item()),
    )
    return int((values > threshold).sum().item())


def _effective_ridge(torch, singular_values, feature_count: int, ridge: float):
    trace = float(singular_values.square().sum().item())
    trace_scale = trace / max(1, int(feature_count))
    effective = float(ridge) * max(trace_scale, torch.finfo(torch.float64).tiny)
    return effective, trace, trace_scale


def _regularized_condition(
    singular_values,
    *,
    numerical_rank: int,
    feature_count: int,
    effective_ridge: float,
) -> float | None:
    if int(singular_values.numel()) == 0:
        return None
    largest = float(singular_values[0].item()) ** 2 + effective_ridge
    has_nullspace = numerical_rank < feature_count
    if has_nullspace:
        smallest = effective_ridge
    else:
        smallest = float(singular_values[numerical_rank - 1].item()) ** 2
        smallest += effective_ridge
    if smallest <= 0.0:
        return None
    value = largest / smallest
    return value if math.isfinite(value) else None


def _canonicalize_factors(torch, encoder, decoder):
    pivots = decoder.abs().argmax(dim=1)
    pivot_values = decoder.gather(1, pivots.unsqueeze(1)).squeeze(1)
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return encoder * signs.unsqueeze(0), decoder * signs.unsqueeze(1)


def _fit_rrr_chart(
    torch,
    x_train,
    z_train,
    weights,
    *,
    requested_rank: int,
    ridge: float,
    rank_tolerance: float,
):
    x_mean = _weighted_mean(torch, x_train, weights)
    z_mean = _weighted_mean(torch, z_train, weights)
    square_root_weight = weights.sqrt().unsqueeze(1)
    x_weighted = (x_train - x_mean) * square_root_weight
    z_weighted = (z_train - z_mean) * square_root_weight
    x_left, x_values, x_right = _compact_svd(torch, x_weighted)
    feature_rank = _numerical_rank(
        torch, x_values, x_weighted.shape, rank_tolerance
    )
    if feature_rank == 0:
        raise ValueError("A boundary has no weighted training variation.")
    effective_ridge, feature_trace, trace_scale = _effective_ridge(
        torch, x_values, int(x_train.shape[1]), ridge
    )
    denominators = x_values.square() + effective_ridge
    inverse_mask = x_values > rank_tolerance * x_values[0]
    ridge_inverse = torch.where(
        inverse_mask,
        x_values / denominators.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(x_values),
    )
    hat_shrinkage = x_values * ridge_inverse
    projected_observable = x_left.t().matmul(z_weighted)
    fitted_observable = x_left.matmul(
        hat_shrinkage.unsqueeze(1) * projected_observable
    )
    _, predictable_values, predictable_right = _compact_svd(
        torch, fitted_observable
    )
    predictable_rank = _numerical_rank(
        torch, predictable_values, fitted_observable.shape, rank_tolerance
    )
    retained_rank = min(requested_rank, feature_rank, predictable_rank)
    if retained_rank <= 0:
        raise ValueError("No future-observable rank survived truncation.")
    decoder = predictable_right[:retained_rank].contiguous()
    target_coordinates = z_weighted.matmul(decoder.t())
    encoder = x_right.t().matmul(
        ridge_inverse.unsqueeze(1)
        * x_left.t().matmul(target_coordinates)
    )
    encoder, decoder = _canonicalize_factors(torch, encoder, decoder)
    total_predictable_energy = float(predictable_values.square().sum().item())
    retained_energy = float(
        predictable_values[:retained_rank].square().sum().item()
    )
    report = {
        "method": "regularized-reduced-rank-regression",
        "solver": "compact-fp64-svd",
        "feature_covariance_materialized": False,
        "feature_gram_solve": False,
        "training_rows": int(x_train.shape[0]),
        "boundary_features": int(x_train.shape[1]),
        "observable_features": int(z_train.shape[1]),
        "requested_rank": int(requested_rank),
        "feature_numerical_rank": feature_rank,
        "predictable_observable_rank": predictable_rank,
        "retained_rank": retained_rank,
        "retained_predictable_energy_fraction": (
            retained_energy / total_predictable_energy
            if total_predictable_energy > 0.0
            else 0.0
        ),
        "retained_predictable_energy": retained_energy,
        "total_predictable_energy": total_predictable_energy,
        "feature_singular_values": [
            float(item) for item in x_values[:retained_rank].tolist()
        ],
        "predictable_singular_values": [
            float(item) for item in predictable_values[:retained_rank].tolist()
        ],
        "predictable_spectrum_sha256": _tensor_sha256(
            torch, predictable_values
        ),
        "ridge": float(ridge),
        "ridge_mode": "feature-trace-scaled",
        "feature_gram_trace": feature_trace,
        "feature_trace_scale": trace_scale,
        "effective_ridge": effective_ridge,
        "regularized_feature_condition_number": _regularized_condition(
            x_values,
            numerical_rank=feature_rank,
            feature_count=int(x_train.shape[1]),
            effective_ridge=effective_ridge,
        ),
    }
    return encoder, decoder, x_mean, z_mean, report


def _fit_cca_chart(
    torch,
    x_train,
    z_train,
    weights,
    *,
    requested_rank: int,
    ridge: float,
    observable_ridge: float,
    rank_tolerance: float,
):
    x_mean = _weighted_mean(torch, x_train, weights)
    z_mean = _weighted_mean(torch, z_train, weights)
    square_root_weight = weights.sqrt().unsqueeze(1)
    x_weighted = (x_train - x_mean) * square_root_weight
    z_weighted = (z_train - z_mean) * square_root_weight
    x_left, x_values, x_right = _compact_svd(torch, x_weighted)
    z_left, z_values, z_right = _compact_svd(torch, z_weighted)
    x_rank = _numerical_rank(torch, x_values, x_weighted.shape, rank_tolerance)
    z_rank = _numerical_rank(torch, z_values, z_weighted.shape, rank_tolerance)
    if x_rank == 0 or z_rank == 0:
        raise ValueError("CCA requires variation in both boundary and observable rows.")
    x_left = x_left[:, :x_rank]
    x_values = x_values[:x_rank]
    x_right = x_right[:x_rank]
    z_left = z_left[:, :z_rank]
    z_values = z_values[:z_rank]
    z_right = z_right[:z_rank]
    effective_x_ridge, x_trace, x_scale = _effective_ridge(
        torch, x_values, int(x_train.shape[1]), ridge
    )
    effective_z_ridge, z_trace, z_scale = _effective_ridge(
        torch, z_values, int(z_train.shape[1]), observable_ridge
    )
    x_whitening = x_values / (
        x_values.square() + effective_x_ridge
    ).sqrt()
    z_whitening = z_values / (
        z_values.square() + effective_z_ridge
    ).sqrt()
    cross = x_left.t().matmul(z_left)
    whitened_cross = (
        x_whitening.unsqueeze(1) * cross * z_whitening.unsqueeze(0)
    )
    canonical_left, correlations, _ = _compact_svd(torch, whitened_cross)
    canonical_rank = _numerical_rank(
        torch, correlations, whitened_cross.shape, rank_tolerance
    )
    retained_rank = min(requested_rank, x_rank, z_rank, canonical_rank)
    if retained_rank <= 0:
        raise ValueError("No canonical future-observable rank survived truncation.")
    inverse_x_scale = (
        x_values.square() + effective_x_ridge
    ).rsqrt()
    encoder = x_right.t().matmul(
        inverse_x_scale.unsqueeze(1) * canonical_left[:, :retained_rank]
    )
    coordinates = (x_train - x_mean).matmul(encoder)
    coordinate_weighted = coordinates * square_root_weight
    coordinate_left, coordinate_values, coordinate_right = _compact_svd(
        torch, coordinate_weighted
    )
    effective_decoder_ridge, coordinate_trace, coordinate_scale = (
        _effective_ridge(
            torch,
            coordinate_values,
            int(coordinates.shape[1]),
            ridge,
        )
    )
    decoder_inverse = coordinate_values / (
        coordinate_values.square() + effective_decoder_ridge
    ).clamp_min(torch.finfo(torch.float64).tiny)
    decoder = coordinate_right.t().matmul(
        decoder_inverse.unsqueeze(1)
        * coordinate_left.t().matmul(z_weighted)
    )
    encoder, decoder = _canonicalize_factors(torch, encoder, decoder)
    total_correlation_energy = float(correlations.square().sum().item())
    retained_correlation_energy = float(
        correlations[:retained_rank].square().sum().item()
    )
    report = {
        "method": "regularized-canonical-correlation-regression",
        "solver": "compact-fp64-svd",
        "feature_covariance_materialized": False,
        "observable_covariance_materialized": False,
        "training_rows": int(x_train.shape[0]),
        "boundary_features": int(x_train.shape[1]),
        "observable_features": int(z_train.shape[1]),
        "requested_rank": int(requested_rank),
        "feature_numerical_rank": x_rank,
        "observable_numerical_rank": z_rank,
        "canonical_numerical_rank": canonical_rank,
        "retained_rank": retained_rank,
        "canonical_correlations": [
            float(item) for item in correlations[:retained_rank].tolist()
        ],
        "retained_correlation_energy_fraction": (
            retained_correlation_energy / total_correlation_energy
            if total_correlation_energy > 0.0
            else 0.0
        ),
        "ridge": float(ridge),
        "observable_ridge": float(observable_ridge),
        "ridge_mode": "trace-scaled",
        "effective_feature_ridge": effective_x_ridge,
        "effective_observable_ridge": effective_z_ridge,
        "effective_decoder_ridge": effective_decoder_ridge,
        "feature_gram_trace": x_trace,
        "observable_gram_trace": z_trace,
        "coordinate_gram_trace": coordinate_trace,
        "feature_trace_scale": x_scale,
        "observable_trace_scale": z_scale,
        "coordinate_trace_scale": coordinate_scale,
        "regularized_feature_condition_number": _regularized_condition(
            x_values,
            numerical_rank=x_rank,
            feature_count=int(x_train.shape[1]),
            effective_ridge=effective_x_ridge,
        ),
        "regularized_observable_condition_number": _regularized_condition(
            z_values,
            numerical_rank=z_rank,
            feature_count=int(z_train.shape[1]),
            effective_ridge=effective_z_ridge,
        ),
    }
    return encoder, decoder, x_mean, z_mean, report


def _coerce_chart_tensor(torch, value, name: str):
    tensor = _validate_matrix(torch, value, name)
    return tensor.detach().cpu().to(torch.float64)


def encode_observable_chart(boundary_rows, chart: Mapping[str, Any]):
    """Encode boundary rows into a fitted chart's local coordinates."""

    torch = _torch()
    rows = _validate_matrix(torch, boundary_rows, "boundary_rows")
    encoder = _coerce_chart_tensor(torch, chart["encoder"], "chart.encoder")
    mean = torch.as_tensor(
        chart["boundary_mean"], dtype=torch.float64, device="cpu"
    ).reshape(-1)
    if int(rows.shape[1]) != int(encoder.shape[0]) or int(mean.numel()) != int(
        rows.shape[1]
    ):
        raise ValueError("Boundary features do not match the observable chart.")
    with torch.inference_mode():
        x = rows.detach().cpu().to(torch.float64)
        return (x - mean).matmul(encoder).contiguous()


def decode_observable_chart(coordinates, chart: Mapping[str, Any]):
    """Decode local chart coordinates into the shared future observable."""

    torch = _torch()
    values = _validate_matrix(torch, coordinates, "coordinates")
    decoder = _coerce_chart_tensor(torch, chart["decoder"], "chart.decoder")
    mean = torch.as_tensor(
        chart["observable_mean"], dtype=torch.float64, device="cpu"
    ).reshape(-1)
    if int(values.shape[1]) != int(decoder.shape[0]) or int(mean.numel()) != int(
        decoder.shape[1]
    ):
        raise ValueError("Coordinate or observable features do not match the chart.")
    with torch.inference_mode():
        local = values.detach().cpu().to(torch.float64)
        return (local.matmul(decoder) + mean).contiguous()


def predict_future_observable(boundary_rows, chart: Mapping[str, Any]):
    """Predict the future observable from boundary rows through one chart."""

    return decode_observable_chart(
        encode_observable_chart(boundary_rows, chart), chart
    )


apply_observable_chart = predict_future_observable


def _prediction_metrics(torch, actual, predicted, observable_mean, weights):
    errors = predicted - actual
    normalized_weights = weights / weights.sum()
    row_mse = errors.square().mean(dim=1)
    rmse = float((normalized_weights * row_mse).sum().sqrt().item())
    centered_actual = actual - observable_mean
    centered_predicted = predicted - observable_mean
    numerator = (centered_actual * centered_predicted).sum(dim=1)
    denominator = (
        centered_actual.square().sum(dim=1).sqrt()
        * centered_predicted.square().sum(dim=1).sqrt()
    )
    cosine_rows = torch.where(
        denominator > torch.finfo(torch.float64).tiny,
        numerator / denominator.clamp_min(torch.finfo(torch.float64).tiny),
        torch.zeros_like(numerator),
    ).clamp(-1.0, 1.0)
    cosine = float((normalized_weights * cosine_rows).sum().item())
    baseline_row_mse = centered_actual.square().mean(dim=1)
    baseline_rmse = float(
        (normalized_weights * baseline_row_mse).sum().sqrt().item()
    )
    relative_rmse = rmse / baseline_rmse if baseline_rmse > 0.0 else None
    explained = (
        1.0 - (rmse * rmse) / (baseline_rmse * baseline_rmse)
        if baseline_rmse > 0.0
        else None
    )
    return {
        "row_count": int(actual.shape[0]),
        "future_observable_rmse": rmse,
        "future_observable_relative_rmse": relative_rmse,
        "future_observable_cosine": cosine,
        "future_observable_variance_explained": explained,
    }


def _held_out_chart_metrics(
    torch,
    boundary,
    observable,
    chart,
    validation_rows,
    row_weights,
    domains,
):
    index = torch.tensor(validation_rows, dtype=torch.long)
    x_validation = boundary.index_select(0, index)
    z_validation = observable.index_select(0, index)
    weights = row_weights.index_select(0, index)
    predicted = predict_future_observable(x_validation, chart)
    overall = _prediction_metrics(
        torch,
        z_validation,
        predicted,
        chart["observable_mean"],
        weights,
    )
    per_domain = {}
    if domains is not None:
        validation_domains = [domains[item] for item in validation_rows]
        for domain in sorted(set(validation_domains)):
            positions = [
                offset
                for offset, label in enumerate(validation_domains)
                if label == domain
            ]
            domain_index = torch.tensor(positions, dtype=torch.long)
            per_domain[domain] = _prediction_metrics(
                torch,
                z_validation.index_select(0, domain_index),
                predicted.index_select(0, domain_index),
                chart["observable_mean"],
                weights.index_select(0, domain_index),
            )
    return {"overall": overall, "per_domain": per_domain}


def _normalize_indices(torch, values, row_count: int, name: str):
    if torch.is_tensor(values):
        index = values.detach().cpu().to(torch.long).reshape(-1)
    else:
        index = torch.tensor(list(values), dtype=torch.long)
    if int(index.numel()) == 0:
        raise ValueError(f"{name} must not be empty.")
    if bool((index < 0).any().item()) or bool((index >= row_count).any().item()):
        raise ValueError(f"{name} contains an out-of-range row index.")
    if int(torch.unique(index).numel()) != int(index.numel()):
        raise ValueError(f"{name} must not contain duplicate indices.")
    return index


def _normalize_coordinates(
    torch,
    coordinates: Mapping[Any, Any] | Sequence[Any],
    names: Sequence[Any] | None,
):
    return _normalize_boundaries(torch, coordinates, names)


def _fit_transition(
    torch,
    source,
    target,
    train_index,
    validation_index,
    row_weights,
    *,
    source_name: str,
    target_name: str,
    ridge: float,
    rank_tolerance: float,
):
    source_train = source.index_select(0, train_index)
    target_train = target.index_select(0, train_index)
    train_weights = row_weights.index_select(0, train_index)
    source_mean = _weighted_mean(torch, source_train, train_weights)
    target_mean = _weighted_mean(torch, target_train, train_weights)
    square_root_weight = train_weights.sqrt().unsqueeze(1)
    source_weighted = (source_train - source_mean) * square_root_weight
    target_weighted = (target_train - target_mean) * square_root_weight
    left, values, right = _compact_svd(torch, source_weighted)
    numerical_rank = _numerical_rank(
        torch, values, source_weighted.shape, rank_tolerance
    )
    if numerical_rank == 0:
        raise ValueError(
            f"Chart {source_name!r} has no weighted transition variation."
        )
    effective_ridge, trace, trace_scale = _effective_ridge(
        torch, values, int(source.shape[1]), ridge
    )
    inverse = torch.where(
        values > rank_tolerance * values[0],
        values
        / (values.square() + effective_ridge).clamp_min(
            torch.finfo(torch.float64).tiny
        ),
        torch.zeros_like(values),
    )
    matrix = right.t().matmul(
        inverse.unsqueeze(1) * left.t().matmul(target_weighted)
    )
    bias = target_mean - source_mean.matmul(matrix)
    source_validation = source.index_select(0, validation_index)
    target_validation = target.index_select(0, validation_index)
    validation_weights = row_weights.index_select(0, validation_index)
    predicted = source_validation.matmul(matrix) + bias
    metrics = _prediction_metrics(
        torch,
        target_validation,
        predicted,
        target_mean,
        validation_weights,
    )
    transition = {
        "source": source_name,
        "target": target_name,
        "matrix": matrix.contiguous(),
        "bias": bias.contiguous(),
    }
    report = {
        "source": source_name,
        "target": target_name,
        "source_rank": int(source.shape[1]),
        "target_rank": int(target.shape[1]),
        "source_numerical_rank": numerical_rank,
        "solver": "compact-fp64-svd-ridge",
        "feature_covariance_materialized": False,
        "ridge": float(ridge),
        "ridge_mode": "source-trace-scaled",
        "source_gram_trace": trace,
        "source_trace_scale": trace_scale,
        "effective_ridge": effective_ridge,
        "regularized_condition_number": _regularized_condition(
            values,
            numerical_rank=numerical_rank,
            feature_count=int(source.shape[1]),
            effective_ridge=effective_ridge,
        ),
        "held_out": metrics,
        "matrix_sha256": _tensor_sha256(torch, matrix),
        "bias_sha256": _tensor_sha256(torch, bias),
    }
    return transition, report


def materialize_chart_transition_maps(
    chart_coordinates: Mapping[Any, Any] | Sequence[Any],
    *,
    train_indices: Sequence[int] | Any,
    validation_indices: Sequence[int] | Any,
    names: Sequence[Any] | None = None,
    sample_weights: Sequence[float] | Any | None = None,
    ridge: float = 1e-6,
    rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit affine maps between each pair of adjacent chart coordinates."""

    torch = _torch()
    ridge_value = _finite_nonnegative(ridge, "ridge")
    tolerance = _relative_tolerance(rank_tolerance)
    coordinate_names, coordinate_inputs = _normalize_coordinates(
        torch, chart_coordinates, names
    )
    row_count = int(coordinate_inputs[0].shape[0])
    train_index = _normalize_indices(
        torch, train_indices, row_count, "train_indices"
    )
    validation_index = _normalize_indices(
        torch, validation_indices, row_count, "validation_indices"
    )
    if set(train_index.tolist()) & set(validation_index.tolist()):
        raise ValueError("train_indices and validation_indices must not overlap.")
    if sample_weights is None:
        weights = torch.ones(row_count, dtype=torch.float64)
    else:
        weights = torch.as_tensor(sample_weights, dtype=torch.float64).reshape(-1)
        if int(weights.numel()) != row_count:
            raise ValueError("sample_weights length must match coordinate rows.")
        if not bool(torch.isfinite(weights).all().item()) or bool(
            (weights < 0.0).any().item()
        ):
            raise ValueError("sample_weights must be finite and non-negative.")
    if float(weights.index_select(0, train_index).sum().item()) <= 0.0:
        raise ValueError("Training rows must have positive total sample weight.")
    if float(weights.index_select(0, validation_index).sum().item()) <= 0.0:
        raise ValueError("Validation rows must have positive total sample weight.")

    named_inputs = dict(zip(coordinate_names, coordinate_inputs, strict=True))
    hashes_before = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradients_before = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    fp64_coordinates = [
        value.detach().cpu().to(torch.float64) for value in coordinate_inputs
    ]
    transitions = []
    transition_reports = []
    with torch.inference_mode():
        for index in range(len(fp64_coordinates) - 1):
            transition, report = _fit_transition(
                torch,
                fp64_coordinates[index],
                fp64_coordinates[index + 1],
                train_index,
                validation_index,
                weights,
                source_name=coordinate_names[index],
                target_name=coordinate_names[index + 1],
                ridge=ridge_value,
                rank_tolerance=tolerance,
            )
            transitions.append(transition)
            transition_reports.append(report)
    hashes_after = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradients_after = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "neural_forge_observable_chart_transitions",
        "zero_gradient": True,
        "closed_form": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_enabled_during_fit": False,
        "inputs_mutated": hashes_before != hashes_after,
        "input_gradients_mutated": gradients_before != gradients_after,
        "chart_count": len(fp64_coordinates),
        "transition_count": len(transitions),
        "transitions": transition_reports,
        "hashes": {
            "inputs_before": hashes_before,
            "inputs_after": hashes_after,
            "input_gradients_before": gradients_before,
            "input_gradients_after": gradients_after,
            "ordered_names_sha256": _json_sha256(coordinate_names),
        },
    }
    return {"transitions": transitions, "report": report}


def apply_chart_transition(coordinates, transition: Mapping[str, Any]):
    """Apply one materialized affine chart transition to row coordinates."""

    torch = _torch()
    values = _validate_matrix(torch, coordinates, "coordinates")
    matrix = _coerce_chart_tensor(
        torch, transition["matrix"], "transition.matrix"
    )
    bias = torch.as_tensor(
        transition["bias"], dtype=torch.float64, device="cpu"
    ).reshape(-1)
    if int(values.shape[1]) != int(matrix.shape[0]) or int(bias.numel()) != int(
        matrix.shape[1]
    ):
        raise ValueError("Coordinate features do not match the chart transition.")
    with torch.inference_mode():
        source = values.detach().cpu().to(torch.float64)
        return (source.matmul(matrix) + bias).contiguous()


def compose_chart_transitions(
    transitions: Sequence[Mapping[str, Any]],
    *,
    start: int = 0,
    stop: int | None = None,
) -> dict[str, Any]:
    """Compose a contiguous sequence of row-vector affine transitions."""

    torch = _torch()
    items = list(transitions)
    if not items:
        raise ValueError("At least one chart transition is required.")
    end = len(items) if stop is None else int(stop)
    begin = int(start)
    if begin < 0 or end > len(items) or begin >= end:
        raise ValueError("start and stop must select a nonempty transition range.")
    selected = items[begin:end]
    first_matrix = _coerce_chart_tensor(
        torch, selected[0]["matrix"], "transition.matrix"
    )
    matrix = first_matrix.clone()
    bias = torch.as_tensor(
        selected[0]["bias"], dtype=torch.float64, device="cpu"
    ).reshape(-1).clone()
    source_name = _normalize_identifier(selected[0].get("source", begin))
    previous_target = selected[0].get("target")
    with torch.inference_mode():
        for offset, transition in enumerate(selected[1:], start=1):
            source = transition.get("source")
            if previous_target is not None and source is not None:
                if _normalize_identifier(previous_target) != _normalize_identifier(source):
                    raise ValueError("Chart transitions are not contiguous by name.")
            next_matrix = _coerce_chart_tensor(
                torch, transition["matrix"], "transition.matrix"
            )
            next_bias = torch.as_tensor(
                transition["bias"], dtype=torch.float64, device="cpu"
            ).reshape(-1)
            if int(matrix.shape[1]) != int(next_matrix.shape[0]):
                raise ValueError(
                    f"Transition {begin + offset} has incompatible chart rank."
                )
            if int(bias.numel()) != int(next_matrix.shape[0]):
                raise ValueError("Transition bias has an incompatible chart rank.")
            matrix = matrix.matmul(next_matrix)
            bias = bias.matmul(next_matrix) + next_bias
            previous_target = transition.get("target")
    target_name = _normalize_identifier(
        selected[-1].get("target", end)
    )
    return {
        "source": source_name,
        "target": target_name,
        "matrix": matrix.contiguous(),
        "bias": bias.contiguous(),
        "composed_transition_count": len(selected),
    }


def fit_logit_observable_charts(
    boundary_rows: Mapping[Any, Any] | Sequence[Any] | Any,
    final_observable,
    *,
    rank: int,
    boundary_names: Sequence[Any] | None = None,
    prompt_ids: Sequence[Any] | None = None,
    domains: Sequence[Any] | None = None,
    domain_weights: Mapping[Any, float] | None = None,
    method: str = "rrr",
    ridge: float = 1e-6,
    observable_ridge: float | None = None,
    transition_ridge: float = 1e-6,
    validation_fraction: float = 0.25,
    min_train_prompts: int = 2,
    min_validation_prompts: int = 1,
    split_seed: int = 0,
    rank_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Fit predictive charts and adjacent transitions for donor boundaries.

    ``rrr`` fits trace-regularized reduced-rank regression.  ``cca`` fits
    regularized canonical boundary directions, followed by an oblique ridge
    decoder into the complete future observable.  Both use only compact row
    SVDs and never materialize a boundary-width covariance matrix.
    """

    torch = _torch()
    requested_rank = _positive_int(rank, "rank")
    ridge_value = _finite_nonnegative(ridge, "ridge")
    transition_ridge_value = _finite_nonnegative(
        transition_ridge, "transition_ridge"
    )
    observable_ridge_value = (
        ridge_value
        if observable_ridge is None
        else _finite_nonnegative(observable_ridge, "observable_ridge")
    )
    tolerance = _relative_tolerance(rank_tolerance)
    normalized_method = str(method).strip().lower().replace("_", "-")
    if normalized_method in {"reduced-rank-regression", "reduced-rank", "rrr"}:
        normalized_method = "rrr"
    elif normalized_method in {"canonical-correlation", "rrr-cca", "cca"}:
        normalized_method = "cca"
    else:
        raise ValueError("method must be rrr or cca.")

    names, boundaries = _normalize_boundaries(
        torch, boundary_rows, boundary_names
    )
    observable_input = _validate_matrix(
        torch, final_observable, "final_observable"
    )
    row_count = int(boundaries[0].shape[0])
    if int(observable_input.shape[0]) != row_count:
        raise ValueError(
            "final_observable and all boundary tensors must have the same row count."
        )
    train_rows, validation_rows, split_report = split_prompt_rows(
        prompt_ids,
        row_count,
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
    )
    row_weights, normalized_domains, weight_report = _domain_row_weights(
        torch, domains, domain_weights, row_count, train_rows
    )
    train_index = torch.tensor(train_rows, dtype=torch.long)
    validation_index = torch.tensor(validation_rows, dtype=torch.long)
    if float(row_weights.index_select(0, validation_index).sum().item()) <= 0.0:
        raise ValueError("Validation rows must have positive total domain weight.")

    named_inputs = {
        **{
            f"boundary:{name}": value
            for name, value in zip(names, boundaries, strict=True)
        },
        "final_observable": observable_input,
    }
    input_hashes_before = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradient_hashes_before = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    fp64_boundaries = [
        value.detach().cpu().to(torch.float64) for value in boundaries
    ]
    observable = observable_input.detach().cpu().to(torch.float64)
    train_weights = row_weights.index_select(0, train_index)
    z_train = observable.index_select(0, train_index)
    charts = []
    chart_reports = []
    coordinates = []
    with torch.inference_mode():
        for name, boundary in zip(names, fp64_boundaries, strict=True):
            x_train = boundary.index_select(0, train_index)
            if normalized_method == "rrr":
                encoder, decoder, x_mean, z_mean, fit_report = _fit_rrr_chart(
                    torch,
                    x_train,
                    z_train,
                    train_weights,
                    requested_rank=requested_rank,
                    ridge=ridge_value,
                    rank_tolerance=tolerance,
                )
            else:
                encoder, decoder, x_mean, z_mean, fit_report = _fit_cca_chart(
                    torch,
                    x_train,
                    z_train,
                    train_weights,
                    requested_rank=requested_rank,
                    ridge=ridge_value,
                    observable_ridge=observable_ridge_value,
                    rank_tolerance=tolerance,
                )
            chart = {
                "name": name,
                "method": normalized_method,
                "encoder": encoder.contiguous(),
                "decoder": decoder.contiguous(),
                "boundary_mean": x_mean.contiguous(),
                "observable_mean": z_mean.contiguous(),
                "requested_rank": requested_rank,
                "retained_rank": int(encoder.shape[1]),
            }
            held_out = _held_out_chart_metrics(
                torch,
                boundary,
                observable,
                chart,
                validation_rows,
                row_weights,
                normalized_domains,
            )
            chart_hashes = {
                "encoder_sha256": _tensor_sha256(torch, encoder),
                "decoder_sha256": _tensor_sha256(torch, decoder),
                "boundary_mean_sha256": _tensor_sha256(torch, x_mean),
                "observable_mean_sha256": _tensor_sha256(torch, z_mean),
            }
            chart_report = {
                "name": name,
                **fit_report,
                "held_out": held_out,
                "hashes": chart_hashes,
            }
            chart["report"] = chart_report
            charts.append(chart)
            chart_reports.append(chart_report)
            coordinates.append((boundary - x_mean).matmul(encoder).contiguous())

    transition_result = materialize_chart_transition_maps(
        dict(zip(names, coordinates, strict=True)),
        train_indices=train_rows,
        validation_indices=validation_rows,
        sample_weights=row_weights,
        ridge=transition_ridge_value,
        rank_tolerance=tolerance,
    )
    input_hashes_after = {
        name: _tensor_sha256(torch, value) for name, value in named_inputs.items()
    }
    gradient_hashes_after = {
        name: _gradient_fingerprint(torch, value)
        for name, value in named_inputs.items()
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "neural_forge_layerwise_logit_observable_charts",
        "zero_gradient": True,
        "closed_form": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_enabled_during_fit": False,
        "inputs_mutated": input_hashes_before != input_hashes_after,
        "input_gradients_mutated": (
            gradient_hashes_before != gradient_hashes_after
        ),
        "method": normalized_method,
        "row_count": row_count,
        "boundary_count": len(boundaries),
        "observable_features": int(observable.shape[1]),
        "requested_rank": requested_rank,
        "rank_tolerance": tolerance,
        "split": split_report,
        "weighting": weight_report,
        "charts": chart_reports,
        "transitions": transition_result["report"]["transitions"],
        "computation": {
            "fit_dtype": "torch.float64",
            "fit_device": "cpu",
            "feature_covariance_materialized": False,
            "observable_covariance_materialized": False,
            "boundary_width_cubic_eigendecomposition_required": False,
            "factorization": "compact row-matrix SVD",
        },
        "hashes": {
            "inputs_before": input_hashes_before,
            "inputs_after": input_hashes_after,
            "input_gradients_before": gradient_hashes_before,
            "input_gradients_after": gradient_hashes_after,
            "ordered_boundary_names_sha256": _json_sha256(names),
            "split_assignment_sha256": split_report["assignment_sha256"],
            "domain_layout_sha256": weight_report["ordered_domains_sha256"],
        },
    }
    return {
        "charts": charts,
        "transitions": transition_result["transitions"],
        "split": {
            "train_indices": train_index,
            "validation_indices": validation_index,
            "report": split_report,
        },
        "row_weights": row_weights,
        "report": report,
    }


fit_layerwise_logit_observable_charts = fit_logit_observable_charts


def fit_logit_observable_chart(
    boundary_rows,
    final_observable,
    *,
    rank: int,
    **kwargs,
) -> dict[str, Any]:
    """Fit one future-observable chart with the layerwise audit contract."""

    if "boundary_names" in kwargs:
        raise ValueError("fit_logit_observable_chart assigns the name 'boundary'.")
    result = fit_logit_observable_charts(
        [boundary_rows],
        final_observable,
        rank=rank,
        boundary_names=["boundary"],
        **kwargs,
    )
    return {
        **result["charts"][0],
        "split": result["split"],
        "row_weights": result["row_weights"],
        "audit_report": result["report"],
    }
