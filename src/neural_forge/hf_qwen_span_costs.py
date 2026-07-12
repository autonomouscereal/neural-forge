"""Deterministic activation costs for progressive Qwen depth folding.

The estimator consumes aligned residual-stream rows at each donor layer.  It
uses detached FP64 CPU tensors and closed-form linear algebra only; it never
constructs an optimizer, records an autograd graph, or mutates the inputs.

Singleton spans are the identity baseline for *depth* folding and therefore
have zero cost.  Every adjacent pair receives an incremental merge cost from
three bounded terms: causal residual magnitude, consecutive-delta
incompatibility, and the energy lost by the best rank-``target_hidden_width``
approximation of the cumulative residual trajectory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any


REPORT_SCHEMA_VERSION = 1

__all__ = [
    "build_donor_activation_span_costs",
    "estimate_donor_activation_span_costs",
    "estimate_qwen_span_costs",
    "plan_progressive_qwen_spans",
    "plan_qwen_activation_spans",
    "plan_qwen_spans_from_activations",
]


def _torch():
    import torch

    return torch


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _critical_quantile(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("critical_quantile must be finite and in [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("critical_quantile must be finite and in [0, 1].") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("critical_quantile must be finite and in [0, 1].")
    return result


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} became non-finite during activation analysis.")
    return 0.0 if result == 0.0 else result


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
    for start in range(0, int(detached.shape[0]), 512):
        raw = (
            detached[start : start + 512]
            .cpu()
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(raw)
    return digest.hexdigest()


def _normalize_layer_mapping(value: Any, name: str) -> dict[int, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by donor-layer index.")
    normalized: dict[int, Any] = {}
    for raw_index, tensor in value.items():
        if isinstance(raw_index, bool):
            raise ValueError(f"{name} keys must be non-negative integers.")
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} keys must be non-negative integers.") from exc
        if index < 0 or index != raw_index:
            raise ValueError(f"{name} keys must be non-negative integers.")
        if index in normalized:
            raise ValueError(f"{name} contains duplicate layer index {index}.")
        normalized[index] = tensor
    if not normalized:
        raise ValueError(f"{name} must contain at least one donor layer.")
    return normalized


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


def _resolve_target_hidden_width(target_hidden_width: Any, target_hidden: Any) -> int:
    if target_hidden_width is None and target_hidden is None:
        raise ValueError("target_hidden_width is required.")
    if target_hidden_width is not None and target_hidden is not None:
        first = _positive_int(target_hidden_width, "target_hidden_width")
        second = _positive_int(target_hidden, "target_hidden")
        if first != second:
            raise ValueError(
                "target_hidden_width and target_hidden must match when both are set."
            )
        return first
    return _positive_int(
        target_hidden_width if target_hidden_width is not None else target_hidden,
        "target_hidden_width",
    )


def _row_sources(
    input_rows_by_layer: Any,
    output_rows_by_layer: Any,
    target_hidden_width: Any,
) -> tuple[Any, Any, Any, str]:
    is_profile = isinstance(input_rows_by_layer, Mapping) and (
        "hidden_basis_input_rows" in input_rows_by_layer
        or "hidden_basis_output_rows" in input_rows_by_layer
    )
    if not is_profile:
        if output_rows_by_layer is None:
            raise ValueError("output_rows_by_layer is required.")
        return (
            input_rows_by_layer,
            output_rows_by_layer,
            target_hidden_width,
            "separate-layer-mappings",
        )

    profile = input_rows_by_layer
    if "hidden_basis_input_rows" not in profile:
        raise ValueError("activation profile is missing hidden_basis_input_rows.")
    if "hidden_basis_output_rows" not in profile:
        raise ValueError("activation profile is missing hidden_basis_output_rows.")
    if output_rows_by_layer is not None:
        if target_hidden_width is not None or isinstance(output_rows_by_layer, Mapping):
            raise ValueError(
                "Do not provide output_rows_by_layer with an activation profile."
            )
        target_hidden_width = output_rows_by_layer
    return (
        profile["hidden_basis_input_rows"],
        profile["hidden_basis_output_rows"],
        target_hidden_width,
        "activation-profile",
    )


def _prepare_rows(torch, input_rows: Any, output_rows: Any):
    inputs = _normalize_layer_mapping(input_rows, "input_rows_by_layer")
    outputs = _normalize_layer_mapping(output_rows, "output_rows_by_layer")
    input_indices = sorted(inputs)
    output_indices = sorted(outputs)
    expected = list(range(len(input_indices)))
    if input_indices != expected:
        raise ValueError(
            "input_rows_by_layer keys must be contiguous donor indices starting at 0."
        )
    if output_indices != input_indices:
        raise ValueError(
            "input_rows_by_layer and output_rows_by_layer must have identical keys."
        )

    prepared_inputs = []
    prepared_outputs = []
    manifests = []
    expected_shape: tuple[int, int] | None = None
    any_requires_grad = False
    for layer in input_indices:
        input_tensor = _validate_matrix(
            torch, inputs[layer], f"input_rows_by_layer[{layer}]"
        )
        output_tensor = _validate_matrix(
            torch, outputs[layer], f"output_rows_by_layer[{layer}]"
        )
        input_shape = tuple(int(item) for item in input_tensor.shape)
        output_shape = tuple(int(item) for item in output_tensor.shape)
        if input_shape != output_shape:
            raise ValueError(
                f"Layer {layer} input/output rows must have exactly matching shapes."
            )
        if expected_shape is None:
            expected_shape = input_shape
        elif input_shape != expected_shape:
            raise ValueError(
                "All donor layers must use the same aligned row count and hidden width."
            )
        any_requires_grad = any_requires_grad or bool(input_tensor.requires_grad)
        any_requires_grad = any_requires_grad or bool(output_tensor.requires_grad)
        manifests.append(
            {
                "layer": int(layer),
                "shape": list(input_shape),
                "input_dtype": str(input_tensor.dtype),
                "output_dtype": str(output_tensor.dtype),
                "input_sha256": _tensor_sha256(torch, input_tensor),
                "output_sha256": _tensor_sha256(torch, output_tensor),
            }
        )
        prepared_inputs.append(
            input_tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
        prepared_outputs.append(
            output_tensor.detach().to(device="cpu", dtype=torch.float64).contiguous()
        )
    return prepared_inputs, prepared_outputs, manifests, any_requires_grad


def _vector_norm(torch, value, name: str) -> float:
    return _finite_float(torch.linalg.vector_norm(value).item(), name)


def _rms_from_norm(norm: float, element_count: int, name: str) -> float:
    return _finite_float(norm / math.sqrt(float(element_count)), name)


def _symmetric_relative_magnitude(
    delta_norm: float, input_norm: float, output_norm: float
) -> float:
    denominator = 0.5 * (input_norm + output_norm)
    if denominator <= 0.0:
        return 0.0
    return _finite_float(
        min(2.0, max(0.0, delta_norm / denominator)),
        "symmetric relative residual magnitude",
    )


def _rank_reconstruction_metrics(torch, matrix, target_hidden_width: int):
    singular_values = torch.linalg.svdvals(matrix)
    if not bool(torch.isfinite(singular_values).all().item()):
        raise ValueError("SVD produced non-finite singular values.")
    squared = singular_values.square()
    total_energy = _finite_float(squared.sum().item(), "singular-value energy")
    available_rank = int(singular_values.numel())
    retained_rank = min(int(target_hidden_width), available_rank)
    if total_energy <= 0.0:
        retained_fraction = 1.0
        loss_fraction = 0.0
    else:
        retained_energy = _finite_float(
            squared[:retained_rank].sum().item(), "retained singular-value energy"
        )
        retained_fraction = min(1.0, max(0.0, retained_energy / total_energy))
        loss_fraction = min(1.0, max(0.0, 1.0 - retained_fraction))
    relative_error = _finite_float(
        math.sqrt(loss_fraction), "relative rank reconstruction error"
    )
    if available_rank == 0:
        numerical_rank = 0
        largest = 0.0
    else:
        largest = _finite_float(singular_values[0].item(), "largest singular value")
        tolerance = (
            torch.finfo(torch.float64).eps
            * float(max(int(matrix.shape[0]), int(matrix.shape[1])))
            * largest
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
    return {
        "analysis_rows": int(matrix.shape[0]),
        "analysis_columns": int(matrix.shape[1]),
        "available_singular_value_count": available_rank,
        "target_rank": int(target_hidden_width),
        "retained_rank": retained_rank,
        "numerical_rank": numerical_rank,
        "largest_singular_value": largest,
        "total_singular_value_energy": total_energy,
        "retained_energy_fraction": float(retained_fraction),
        "reconstruction_loss_fraction": float(loss_fraction),
        "relative_reconstruction_error": relative_error,
        "singular_values_sha256": _tensor_sha256(torch, singular_values),
    }


def _delta_cosine(torch, first, second) -> tuple[float, bool]:
    first_norm = _vector_norm(torch, first, "first consecutive delta norm")
    second_norm = _vector_norm(torch, second, "second consecutive delta norm")
    if first_norm <= 0.0 or second_norm <= 0.0:
        # A no-op residual does not introduce a competing direction.
        return 1.0, False
    cosine = _finite_float(
        torch.sum(first * second).item() / (first_norm * second_norm),
        "consecutive delta cosine",
    )
    return min(1.0, max(-1.0, cosine)), True


def _select_critical_layers(
    criticality_scores: list[float], quantile: float
) -> tuple[list[int], int, float]:
    layer_count = len(criticality_scores)
    tail_count = max(
        1,
        min(
            layer_count,
            int(math.ceil(((1.0 - quantile) * layer_count) - 1e-12)),
        ),
    )
    ranked = sorted(
        range(layer_count), key=lambda index: (-criticality_scores[index], index)
    )
    selected = sorted(ranked[:tail_count])
    threshold = min(criticality_scores[index] for index in selected)
    return selected, tail_count, float(threshold)


def _assert_finite_diagnostics(value: Any, path: str = "diagnostics") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite diagnostic.")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_diagnostics(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_diagnostics(item, f"{path}[{index}]")
        return
    raise TypeError(f"{path} contains unsupported diagnostic type {type(value)!r}.")


def estimate_qwen_span_costs(
    input_rows_by_layer: Any,
    output_rows_by_layer: Any = None,
    target_hidden_width: Any = None,
    *,
    target_hidden: Any = None,
    max_span_length: int = 2,
    critical_quantile: float = 0.8,
    causal_magnitude_weight: float = 1.0,
    incompatibility_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
) -> dict[str, Any]:
    """Estimate deterministic singleton/pair costs from residual activations.

    ``input_rows_by_layer`` may be either a layer-indexed mapping or an
    activation profile containing ``hidden_basis_input_rows`` and
    ``hidden_basis_output_rows``.  With separate mappings, the output mapping
    is the second argument.  All matrices must have one common ``[rows,
    donor_hidden]`` shape so row identities remain aligned across depth.

    ``critical_quantile`` selects ``ceil((1 - q) * layer_count)`` highest
    criticality layers, with at least one layer and deterministic layer-index
    tie breaking.  The returned ``span_costs`` mapping uses half-open keys and
    can be passed directly to :func:`hf_span_planner.plan_contiguous_spans`.
    """

    torch = _torch()
    (
        raw_inputs,
        raw_outputs,
        target_hidden_width,
        input_layout,
    ) = _row_sources(input_rows_by_layer, output_rows_by_layer, target_hidden_width)
    target_width = _resolve_target_hidden_width(target_hidden_width, target_hidden)
    maximum = _positive_int(max_span_length, "max_span_length")
    if maximum > 2:
        raise ValueError("max_span_length cannot exceed 2 for Qwen span costs.")
    quantile = _critical_quantile(critical_quantile)
    weights = {
        "causal_residual_magnitude": _finite_nonnegative(
            causal_magnitude_weight, "causal_magnitude_weight"
        ),
        "consecutive_delta_incompatibility": _finite_nonnegative(
            incompatibility_weight, "incompatibility_weight"
        ),
        "rank_reconstruction_loss": _finite_nonnegative(
            reconstruction_weight, "reconstruction_weight"
        ),
    }
    weight_total = sum(weights.values())
    if weight_total <= 0.0:
        raise ValueError("At least one span-cost weight must be positive.")

    with torch.no_grad():
        inputs, outputs, input_manifest, any_requires_grad = _prepare_rows(
            torch, raw_inputs, raw_outputs
        )
        layer_count = len(inputs)
        row_count, source_hidden_width = (
            int(inputs[0].shape[0]),
            int(inputs[0].shape[1]),
        )
        deltas = [
            output - input_ for input_, output in zip(inputs, outputs, strict=True)
        ]

        base_layer_metrics = []
        maximum_delta_rms = 0.0
        maximum_delta_energy = 0.0
        for layer, (input_, output, delta) in enumerate(
            zip(inputs, outputs, deltas, strict=True)
        ):
            input_norm = _vector_norm(torch, input_, f"layer {layer} input norm")
            output_norm = _vector_norm(torch, output, f"layer {layer} output norm")
            delta_norm = _vector_norm(torch, delta, f"layer {layer} delta norm")
            delta_rms = _rms_from_norm(
                delta_norm, int(delta.numel()), f"layer {layer} delta RMS"
            )
            delta_energy = _finite_float(
                (delta_norm * delta_norm) / float(row_count),
                f"layer {layer} residual-delta energy",
            )
            relative_magnitude = _symmetric_relative_magnitude(
                delta_norm, input_norm, output_norm
            )
            maximum_delta_rms = max(maximum_delta_rms, delta_rms)
            maximum_delta_energy = max(maximum_delta_energy, delta_energy)
            base_layer_metrics.append(
                {
                    "layer": int(layer),
                    "row_count": row_count,
                    "hidden_width": source_hidden_width,
                    "input_frobenius_norm": input_norm,
                    "output_frobenius_norm": output_norm,
                    "residual_delta_frobenius_norm": delta_norm,
                    "residual_delta_rms": delta_rms,
                    "residual_delta_energy_per_row": delta_energy,
                    "relative_residual_magnitude": relative_magnitude,
                    "causal_residual_magnitude": 0.5 * relative_magnitude,
                }
            )

        layer_diagnostics = []
        criticality_scores = []
        total_delta_energy = sum(
            item["residual_delta_energy_per_row"] for item in base_layer_metrics
        )
        for layer, (base, delta) in enumerate(
            zip(base_layer_metrics, deltas, strict=True)
        ):
            energy_scale = (
                math.sqrt(base["residual_delta_energy_per_row"] / maximum_delta_energy)
                if maximum_delta_energy > 0.0
                else 0.0
            )
            relative_scale = min(
                1.0, max(0.0, 0.5 * base["relative_residual_magnitude"])
            )
            criticality = _finite_float(
                0.5 * energy_scale + 0.5 * relative_scale,
                f"layer {layer} criticality score",
            )
            criticality_scores.append(criticality)
            rank_metrics = _rank_reconstruction_metrics(
                torch, delta, target_hidden_width=target_width
            )
            layer_diagnostics.append(
                {
                    **base,
                    "residual_delta_energy_fraction": (
                        base["residual_delta_energy_per_row"] / total_delta_energy
                        if total_delta_energy > 0.0
                        else 0.0
                    ),
                    "normalized_delta_energy_magnitude": energy_scale,
                    "normalized_relative_residual_magnitude": relative_scale,
                    "criticality_score": criticality,
                    "rank_reconstruction": rank_metrics,
                    "delta_sha256": _tensor_sha256(torch, delta),
                }
            )

        critical_layers, critical_count, critical_threshold = _select_critical_layers(
            criticality_scores, quantile
        )
        critical_set = set(critical_layers)
        for item in layer_diagnostics:
            item["critical"] = int(item["layer"]) in critical_set

        span_costs: dict[tuple[int, int], float] = {}
        span_diagnostics = []
        for start in range(layer_count):
            last_end = min(layer_count, start + maximum)
            cumulative_delta = torch.zeros_like(deltas[start])
            cumulative_trajectory = []
            for end in range(start + 1, last_end + 1):
                cumulative_delta = cumulative_delta + deltas[end - 1]
                cumulative_trajectory.append(cumulative_delta.clone())
                trajectory_matrix = torch.cat(cumulative_trajectory, dim=0)
                span_length = end - start
                composed_input = inputs[start]
                composed_output = composed_input + cumulative_delta
                composed_delta_norm = _vector_norm(
                    torch,
                    cumulative_delta,
                    f"span [{start}, {end}) composed delta norm",
                )
                composed_delta_rms = _rms_from_norm(
                    composed_delta_norm,
                    int(cumulative_delta.numel()),
                    f"span [{start}, {end}) composed delta RMS",
                )
                input_norm = _vector_norm(
                    torch, composed_input, f"span [{start}, {end}) input norm"
                )
                output_norm = _vector_norm(
                    torch, composed_output, f"span [{start}, {end}) output norm"
                )
                relative_magnitude = _symmetric_relative_magnitude(
                    composed_delta_norm, input_norm, output_norm
                )
                normalized_endpoint_magnitude = (
                    min(
                        1.0,
                        max(
                            0.0,
                            composed_delta_rms
                            / (float(span_length) * maximum_delta_rms),
                        ),
                    )
                    if maximum_delta_rms > 0.0
                    else 0.0
                )
                constituent_criticality = max(
                    criticality_scores[start:end], default=0.0
                )
                causal_cost = _finite_float(
                    0.5 * (normalized_endpoint_magnitude + constituent_criticality),
                    f"span [{start}, {end}) causal magnitude cost",
                )
                if span_length == 2:
                    cosine, cosine_defined = _delta_cosine(
                        torch, deltas[start], deltas[start + 1]
                    )
                    incompatibility = 0.5 * (1.0 - cosine)
                else:
                    cosine, cosine_defined = 1.0, True
                    incompatibility = 0.0
                rank_metrics = _rank_reconstruction_metrics(
                    torch, trajectory_matrix, target_hidden_width=target_width
                )
                observed_endpoint_delta = outputs[end - 1] - inputs[start]
                closure_error = observed_endpoint_delta - cumulative_delta
                closure_error_norm = _vector_norm(
                    torch,
                    closure_error,
                    f"span [{start}, {end}) boundary closure error norm",
                )
                closure_reference = max(
                    _vector_norm(
                        torch,
                        observed_endpoint_delta,
                        f"span [{start}, {end}) observed endpoint norm",
                    ),
                    composed_delta_norm,
                )
                closure_relative = (
                    closure_error_norm / closure_reference
                    if closure_reference > 0.0
                    else 0.0
                )
                if span_length == 1:
                    cost = 0.0
                else:
                    cost = _finite_float(
                        (
                            weights["causal_residual_magnitude"] * causal_cost
                            + weights["consecutive_delta_incompatibility"]
                            * incompatibility
                            + weights["rank_reconstruction_loss"]
                            * rank_metrics["reconstruction_loss_fraction"]
                        )
                        / weight_total,
                        f"span [{start}, {end}) cost",
                    )
                key = (int(start), int(end))
                span_costs[key] = cost
                span_diagnostics.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "half_open_span": [int(start), int(end)],
                        "length": int(span_length),
                        "donor_layers": list(range(start, end)),
                        "cost": cost,
                        "cost_is_singleton_identity_baseline": span_length == 1,
                        "causal_residual_magnitude_cost": causal_cost,
                        "normalized_composed_delta_magnitude": normalized_endpoint_magnitude,
                        "composed_delta_frobenius_norm": composed_delta_norm,
                        "composed_delta_rms": composed_delta_rms,
                        "relative_composed_residual_magnitude": relative_magnitude,
                        "maximum_constituent_criticality": constituent_criticality,
                        "consecutive_delta_cosine": float(cosine),
                        "consecutive_delta_cosine_defined": bool(cosine_defined),
                        "consecutive_delta_incompatibility": float(incompatibility),
                        "rank_reconstruction": rank_metrics,
                        "boundary_closure_error_frobenius_norm": closure_error_norm,
                        "boundary_closure_relative_error": _finite_float(
                            closure_relative,
                            f"span [{start}, {end}) boundary closure relative error",
                        ),
                        "contains_critical_layers": [
                            layer
                            for layer in range(start, end)
                            if layer in critical_set
                        ],
                        "cumulative_trajectory_sha256": _tensor_sha256(
                            torch, trajectory_matrix
                        ),
                    }
                )

    span_cost_records = [
        [start, end, span_costs[(start, end)]] for start, end in sorted(span_costs)
    ]
    expected_span_count = layer_count + (layer_count - 1 if maximum == 2 else 0)
    if len(span_costs) != expected_span_count:
        raise RuntimeError("Internal error: incomplete singleton/pair span-cost map.")

    configuration = {
        "target_hidden_width": target_width,
        "maximum_span_length": maximum,
        "critical_quantile": quantile,
        "weights": weights,
        "singleton_cost_semantics": "zero-cost identity baseline for depth folding",
    }
    input_hash = _json_sha256(input_manifest)
    diagnostics_payload = {
        "layers": layer_diagnostics,
        "spans": span_diagnostics,
    }
    hashes = {
        "algorithm": "sha256",
        "input_activation_rows_sha256": input_hash,
        "configuration_sha256": _json_sha256(configuration),
        "layer_diagnostics_sha256": _json_sha256(layer_diagnostics),
        "span_diagnostics_sha256": _json_sha256(span_diagnostics),
        "span_costs_sha256": _json_sha256(span_cost_records),
        "critical_layers_sha256": _json_sha256(critical_layers),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": "deterministic-qwen-donor-activation-depth-span-costs",
        "inputs": {
            "layout": input_layout,
            "donor_layer_count": layer_count,
            "activation_row_count": row_count,
            "source_hidden_width": source_hidden_width,
            "target_hidden_width": target_width,
            "input_manifest": input_manifest,
        },
        "configuration": configuration,
        "cost_model": {
            "interval_convention": "half-open [start, end)",
            "available_span_lengths": list(range(1, maximum + 1)),
            "singleton_cost": 0.0,
            "pair_cost_formula": (
                "weighted_mean(causal_residual_magnitude_cost, "
                "consecutive_delta_incompatibility, "
                "rank_reconstruction_loss_fraction)"
            ),
            "composed_span_matrix": (
                "row-concatenated cumulative residual deltas in causal layer order"
            ),
            "complete_span_cost_count": len(span_costs),
            "span_cost_records": span_cost_records,
        },
        "critical_layers": {
            "indices": critical_layers,
            "selection": "deterministic-top-ceil-tail",
            "quantile": quantile,
            "selected_count": critical_count,
            "criticality_threshold": critical_threshold,
            "score_formula": (
                "mean(normalized_delta_energy_magnitude, "
                "normalized_relative_residual_magnitude)"
            ),
        },
        "diagnostics": {
            **diagnostics_payload,
            "all_finite": True,
            "input_tensors_required_grad": any_requires_grad,
            "autograd_graph_constructed": False,
            "optimizer_constructed": False,
            "analysis_device": "cpu",
            "analysis_dtype": "torch.float64",
        },
        "determinism": {
            "layer_order": "ascending donor-layer index",
            "critical_tie_break": "lower donor-layer index first",
            "linear_algebra": "detached CPU torch.float64 torch.linalg.svdvals",
        },
        "hashes": hashes,
    }
    _assert_finite_diagnostics(report)
    hashes["report_sha256"] = _json_sha256(report)

    return {
        "donor_layer_count": layer_count,
        "target_hidden_width": target_width,
        "max_span_length": maximum,
        "span_cost_layout": "half-open",
        "span_costs": span_costs,
        "costs": span_costs,
        "span_cost_records": span_cost_records,
        "critical_layers": critical_layers,
        "criticality_scores": criticality_scores,
        "layer_diagnostics": layer_diagnostics,
        "span_diagnostics": span_diagnostics,
        "hashes": dict(hashes),
        "report": report,
    }


def plan_qwen_activation_spans(
    input_rows_by_layer: Any,
    output_rows_by_layer: Any = None,
    target_hidden_width: Any = None,
    target_layer_count: Any = None,
    *,
    target_hidden: Any = None,
    max_span_length: int = 2,
    critical_quantile: float = 0.8,
    causal_magnitude_weight: float = 1.0,
    incompatibility_weight: float = 1.0,
    reconstruction_weight: float = 1.0,
    critical_layer_sharing_penalty: float | None = None,
    tie_break: str = "earliest-boundaries",
) -> dict[str, Any]:
    """Estimate activation costs and invoke the contiguous-span planner."""

    is_profile = isinstance(input_rows_by_layer, Mapping) and (
        "hidden_basis_input_rows" in input_rows_by_layer
        or "hidden_basis_output_rows" in input_rows_by_layer
    )
    if (
        is_profile
        and output_rows_by_layer is not None
        and not isinstance(output_rows_by_layer, Mapping)
        and target_hidden_width is not None
        and target_layer_count is None
    ):
        # Support ``(activation_profile, target_hidden_width, target_layer_count)``.
        target_layer_count = target_hidden_width
        target_hidden_width = output_rows_by_layer
        output_rows_by_layer = None
    if target_layer_count is None:
        raise ValueError("target_layer_count is required.")
    target_count = _positive_int(target_layer_count, "target_layer_count")

    estimate = estimate_qwen_span_costs(
        input_rows_by_layer,
        output_rows_by_layer,
        target_hidden_width,
        target_hidden=target_hidden,
        max_span_length=max_span_length,
        critical_quantile=critical_quantile,
        causal_magnitude_weight=causal_magnitude_weight,
        incompatibility_weight=incompatibility_weight,
        reconstruction_weight=reconstruction_weight,
    )
    from neural_forge.hf_span_planner import plan_contiguous_spans

    plan = plan_contiguous_spans(
        estimate["donor_layer_count"],
        target_count,
        estimate["span_costs"],
        min_span_length=1,
        max_span_length=estimate["max_span_length"],
        critical_layers=estimate["critical_layers"],
        critical_layer_sharing_penalty=critical_layer_sharing_penalty,
        span_cost_layout="half-open",
        tie_break=tie_break,
    )
    planning_provenance = {
        "operation": "estimate-qwen-span-costs-and-plan-contiguous-spans",
        "cost_report_sha256": estimate["hashes"]["report_sha256"],
        "span_costs_sha256": estimate["hashes"]["span_costs_sha256"],
        "planner_input_sha256": plan["report"]["hashes"]["input_sha256"],
        "planner_plan_sha256": plan["report"]["hashes"]["plan_sha256"],
        "critical_layers": list(estimate["critical_layers"]),
    }
    planning_provenance["report_sha256"] = _json_sha256(planning_provenance)
    return {
        **plan,
        "critical_layers": list(estimate["critical_layers"]),
        "span_cost_estimate": estimate,
        "cost_estimate": estimate,
        "planning_provenance": planning_provenance,
    }


# Descriptive compatibility names for callers that lead with the data source.
estimate_donor_activation_span_costs = estimate_qwen_span_costs
build_donor_activation_span_costs = estimate_qwen_span_costs
plan_progressive_qwen_spans = plan_qwen_activation_spans
plan_qwen_spans_from_activations = plan_qwen_activation_spans
