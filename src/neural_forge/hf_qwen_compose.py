"""Zero-gradient ordered pair composition for Qwen causal language models.

The initializer in this module compresses two consecutive donor decoder blocks
into one child block.  It is deliberately narrower than the generic folding
utilities: both models must expose Qwen-style GQA attention, SwiGLU MLPs, and a
shared tokenizer.  The narrow contract lets the implementation preserve head
and channel structure instead of treating every tensor as an anonymous array.

No optimizer is constructed and no backward pass is performed. Post-softmax
attention paths and post-SwiGLU neuron paths are merged through their consumer
directions, while layer-varying observable charts retain directions predictive
of the donor's final state. Linear producer/output maps are fitted with
prior-preserving, trace-scaled FP64 ridge solves and committed transactionally
only when held-out *prompts* improve. Token rows from one prompt are never split
between fitting and validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from neural_forge.hf_tools import (
    _decoder_layer_container,
    _dtype_from_name,
    _ensure_tokenizer_vocab_capacity,
    _torch_and_transformers,
)


DEFAULT_COMPOSITION_TEXTS = (
    "A reliable compressed language model keeps facts and syntax stable after structural initialization.",
    "Explain why a held-out prompt is stronger evidence than reusing calibration tokens.",
    "Question: What is 37 plus 58?\nAnswer: 95",
    "Write a Python function that returns the larger of two integers.",
    "Grouped-query attention shares key and value heads across several query heads.",
    "A SwiGLU unit couples its gate channel with the matching up-projection channel.",
    "The second decoder block receives the actual output of the first decoder block.",
    "A stable linear solve preserves its prior when calibration evidence is weak.",
    "List three reasons to validate a model transformation before deployment.",
    "Question: If a box contains six rows of seven parts, how many parts are present?\nAnswer: 42",
    "Complete this code: def square(x):\n    return x * x",
    "Residual coordinates must keep one consistent meaning across every compressed layer.",
)


def keyword_domain_balanced_texts(
    texts: Sequence[str], max_texts: int | None
) -> list[str]:
    """Select calibration prompts round-robin across detected domains."""

    values = [str(item) for item in texts]
    if max_texts is None or len(values) <= int(max_texts):
        return values
    if int(max_texts) <= 0:
        raise ValueError("max_texts must be positive when provided.")
    from neural_forge.hf_xfamily import _activation_calibration_domain

    buckets: dict[str, list[str]] = {}
    for value in values:
        domain = _activation_calibration_domain(value)
        buckets.setdefault(domain, []).append(value)
    offsets = {domain: 0 for domain in buckets}
    selected = []
    while len(selected) < int(max_texts):
        added = False
        for domain in sorted(buckets):
            offset = offsets[domain]
            if offset >= len(buckets[domain]):
                continue
            selected.append(buckets[domain][offset])
            offsets[domain] = offset + 1
            added = True
            if len(selected) == int(max_texts):
                break
        if not added:
            break
    return selected


class LayerPair(NamedTuple):
    """One ordered donor pair assigned to a child layer."""

    target_layer: int
    donor_first: int
    donor_second: int

    @property
    def donor_layers(self) -> tuple[int, int]:
        return self.donor_first, self.donor_second


def _compose_progress(event: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    print(
        f"neural_forge_progress event={event}{' ' + details if details else ''}",
        flush=True,
    )


def build_contiguous_pair_plan(
    donor_layer_count: int, target_layer_count: int
) -> list[LayerPair]:
    """Return the strict ``(2j, 2j + 1)`` donor plan.

    Pair composition is intentionally not proportional depth sampling.  Every
    donor layer appears once, in order, and the donor must have exactly twice
    as many layers as the target.
    """

    donor_count = int(donor_layer_count)
    target_count = int(target_layer_count)
    if donor_count <= 0 or target_count <= 0:
        raise ValueError("donor_layer_count and target_layer_count must be positive.")
    if donor_count != 2 * target_count:
        raise ValueError(
            "Ordered pair composition requires the donor to have exactly twice the target layer count "
            "(donor_layer_count == 2 * target_layer_count); "
            f"got {donor_count} and {target_count}."
        )
    return [LayerPair(j, 2 * j, 2 * j + 1) for j in range(target_count)]


def contiguous_donor_pairs(
    donor_layer_count: int, target_layer_count: int
) -> list[tuple[int, int]]:
    """Return only the donor-index portion of :func:`build_contiguous_pair_plan`."""

    return [
        item.donor_layers
        for item in build_contiguous_pair_plan(donor_layer_count, target_layer_count)
    ]


def prompt_level_split(
    prompts: Sequence[str],
    validation_fraction: float = 0.25,
    min_validation_prompts: int = 1,
    min_train_prompts: int = 2,
    seed: int = 0,
) -> dict[str, Any]:
    """Deterministically split whole prompts into fitting and validation sets."""

    prompt_values = [str(item) for item in prompts]
    count = len(prompt_values)
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1.")
    if min_validation_prompts <= 0 or min_train_prompts <= 0:
        raise ValueError("minimum prompt counts must be positive.")
    if count < int(min_validation_prompts) + int(min_train_prompts):
        raise ValueError(
            "Not enough calibration prompts for a prompt-level split: "
            f"need at least {int(min_validation_prompts) + int(min_train_prompts)}, got {count}."
        )

    validation_count = max(
        int(min_validation_prompts), int(round(count * float(validation_fraction)))
    )
    validation_count = min(validation_count, count - int(min_train_prompts))
    ranked = sorted(
        range(count),
        key=lambda idx: hashlib.sha256(
            f"{int(seed)}\x00{idx}\x00{prompt_values[idx]}".encode("utf-8")
        ).digest(),
    )
    validation = sorted(ranked[:validation_count])
    validation_set = set(validation)
    train = [idx for idx in range(count) if idx not in validation_set]
    if set(train) & validation_set or len(train) + len(validation) != count:
        raise RuntimeError(
            "Prompt-level split produced overlapping or missing indices."
        )
    return {
        "train_indices": train,
        "validation_indices": validation,
        "train_prompt_count": len(train),
        "validation_prompt_count": len(validation),
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
        "unit": "prompt",
    }


def build_gqa_group_packing(
    donor_query_heads: int,
    donor_key_value_heads: int,
    target_query_heads: int,
    target_key_value_heads: int,
    head_dim: int = 128,
    pair_width: int = 2,
    selected_kv_heads: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Select intact GQA groups from each donor layer without averaging heads."""

    donor_q = int(donor_query_heads)
    donor_kv = int(donor_key_value_heads)
    target_q = int(target_query_heads)
    target_kv = int(target_key_value_heads)
    width = int(pair_width)
    coordinate_count = int(head_dim)
    if min(donor_q, donor_kv, target_q, target_kv, width, coordinate_count) <= 0:
        raise ValueError("GQA head counts, pair_width, and head_dim must be positive.")
    if donor_q % donor_kv or target_q % target_kv:
        raise ValueError("Query head counts must be divisible by KV head counts.")
    donor_ratio = donor_q // donor_kv
    target_ratio = target_q // target_kv
    if donor_ratio != target_ratio:
        raise ValueError(
            "Exact GQA packing requires the donor and target query-per-KV ratios to match; "
            f"got {donor_ratio} and {target_ratio}."
        )
    available_source_groups = width * donor_kv
    if target_kv % width:
        raise ValueError(
            "Intact pair packing requires target_key_value_heads to be divisible by pair_width."
        )
    groups_per_layer = target_kv // width
    if groups_per_layer <= 0 or groups_per_layer > donor_kv:
        raise ValueError(
            "Target GQA capacity cannot select an intact group from every donor layer."
        )
    if selected_kv_heads is None:
        if groups_per_layer == 1:
            per_layer = [0]
        else:
            per_layer = [
                int(round(index * (donor_kv - 1) / (groups_per_layer - 1)))
                for index in range(groups_per_layer)
            ]
        selected = per_layer * width
    else:
        selected = [int(value) for value in selected_kv_heads]
    if len(selected) != target_kv:
        raise ValueError(
            "selected_kv_heads must contain one donor head per target KV head."
        )
    for pair_offset in range(width):
        layer_heads = selected[
            pair_offset * groups_per_layer : (pair_offset + 1) * groups_per_layer
        ]
        if any(value < 0 or value >= donor_kv for value in layer_heads):
            raise ValueError(
                "selected_kv_heads contains an out-of-range donor KV head."
            )
        if len(set(layer_heads)) != len(layer_heads):
            raise ValueError(
                "selected_kv_heads must be unique within each donor layer."
            )

    groups: list[dict[str, Any]] = []
    for target_kv_head, donor_kv_head in enumerate(selected):
        pair_offset = target_kv_head // groups_per_layer
        donor_query_start = donor_kv_head * donor_ratio
        target_query_start = target_kv_head * target_ratio
        q_heads = []
        for local_query in range(donor_ratio):
            donor_query_head = donor_query_start + local_query
            target_query_head = target_query_start + local_query
            q_heads.append(
                {
                    "local_query_head": local_query,
                    "donor_query_head": donor_query_head,
                    "target_query_head": target_query_head,
                    "donor_slice": [
                        donor_query_head * coordinate_count,
                        (donor_query_head + 1) * coordinate_count,
                    ],
                    "target_slice": [
                        target_query_head * coordinate_count,
                        (target_query_head + 1) * coordinate_count,
                    ],
                }
            )
        groups.append(
            {
                "target_kv_head": target_kv_head,
                "source": {
                    "pair_offset": pair_offset,
                    "donor_kv_head": donor_kv_head,
                    "donor_kv_slice": [
                        donor_kv_head * coordinate_count,
                        (donor_kv_head + 1) * coordinate_count,
                    ],
                    "target_kv_slice": [
                        target_kv_head * coordinate_count,
                        (target_kv_head + 1) * coordinate_count,
                    ],
                    "query_heads": q_heads,
                },
            }
        )
    covered_q_coordinates = target_q * coordinate_count
    return {
        "pair_width": width,
        "head_dim": coordinate_count,
        "donor_query_heads": donor_q,
        "donor_key_value_heads": donor_kv,
        "target_query_heads": target_q,
        "target_key_value_heads": target_kv,
        "query_heads_per_kv": donor_ratio,
        "available_source_group_count": available_source_groups,
        "selected_source_group_count": target_kv,
        "discarded_source_group_count": available_source_groups - target_kv,
        "groups_per_donor_layer": groups_per_layer,
        "selected_kv_heads": selected,
        "groups": groups,
        "selected_full_head_coordinates_preserved": True,
        "nonlinear_head_averaging": False,
        "covered_query_coordinates": covered_q_coordinates,
    }


def build_coupled_swiglu_packing(
    donor_intermediate_size: int,
    target_intermediate_size: int,
    pair_width: int = 2,
    channel_scores: Sequence[Sequence[float]] | None = None,
    selected_sources: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Select intact paired SwiGLU channels with shared gate/up/down indices."""

    donor_size = int(donor_intermediate_size)
    target_size = int(target_intermediate_size)
    width = int(pair_width)
    if min(donor_size, target_size, width) <= 0:
        raise ValueError("intermediate sizes and pair_width must be positive.")
    source_count = donor_size * width
    if target_size < width or source_count < target_size:
        raise ValueError(
            "Coupled SwiGLU selection needs capacity from every donor layer."
        )
    if channel_scores is not None:
        if len(channel_scores) != width or any(
            len(row) != donor_size for row in channel_scores
        ):
            raise ValueError(
                "channel_scores must have shape [pair_width, donor_intermediate_size]."
            )
        score_rows = [
            [float(value) for value in channel_scores[layer_offset]]
            for layer_offset in range(width)
        ]
    else:
        score_rows = [[0.0] * donor_size for _ in range(width)]

    if selected_sources is None:
        base_quota, remainder = divmod(target_size, width)
        quotas = [
            base_quota + (1 if layer_offset < remainder else 0)
            for layer_offset in range(width)
        ]
        if any(quota <= 0 or quota > donor_size for quota in quotas):
            raise ValueError(
                "Target SwiGLU quotas do not fit donor intermediate channels."
            )
        normalized_sources: list[tuple[int, int]] = []
        for layer_offset, quota in enumerate(quotas):
            ranked = sorted(
                range(donor_size),
                key=lambda channel: (-score_rows[layer_offset][channel], channel),
            )
            normalized_sources.extend(
                (layer_offset, channel) for channel in ranked[:quota]
            )
        selection_source = "score-ranked-intact-channels"
    else:
        if len(selected_sources) != target_size:
            raise ValueError(
                "selected_sources must contain one donor source per target channel."
            )
        normalized_sources = []
        for item in selected_sources:
            if len(item) != 2:
                raise ValueError(
                    "Each selected SwiGLU source must be [pair_offset, channel]."
                )
            pair_offset, channel = int(item[0]), int(item[1])
            if not 0 <= pair_offset < width or not 0 <= channel < donor_size:
                raise ValueError("selected_sources contains an out-of-range channel.")
            normalized_sources.append((pair_offset, channel))
        if len(set(normalized_sources)) != len(normalized_sources):
            raise ValueError("selected_sources must not contain duplicate channels.")
        quotas = [
            sum(pair_offset == layer_offset for pair_offset, _ in normalized_sources)
            for layer_offset in range(width)
        ]
        if any(quota <= 0 for quota in quotas):
            raise ValueError("selected_sources must preserve every donor layer.")
        selection_source = "explicit-post-nonlinearity-support"

    source_to_target = [-1] * source_count
    packed = []
    for target_channel, source in enumerate(normalized_sources):
        source_flat_index = int(source[0]) * donor_size + int(source[1])
        source_to_target[source_flat_index] = target_channel
        refs = [[int(source[0]), int(source[1])]]
        packed.append(
            {
                "target_channel": target_channel,
                "sources": refs,
                "gate_indices": refs,
                "up_indices": refs,
                "down_indices": refs,
            }
        )
    return {
        "pair_width": width,
        "donor_intermediate_size": donor_size,
        "target_intermediate_size": target_size,
        "source_channel_count": source_count,
        "selected_channel_count": target_size,
        "discarded_channel_count": source_count - target_size,
        "selected_per_donor_layer": quotas,
        "selection_source": selection_source,
        "target_channels": packed,
        "source_to_target": source_to_target,
        "gate_up_down_indices_coupled": True,
        "all_source_channels_packed": target_size == source_count,
        "nonlinear_channel_averaging": False,
    }


def _require_torch(torch_module=None):
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def _stable_fp64_solve(
    torch, system, rhs, trace_scale: float
) -> tuple[Any, dict[str, Any]]:
    """Solve a symmetric system with Cholesky+jitter and least-squares fallback."""

    if system.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise ValueError("_stable_fp64_solve requires FP64 tensors.")
    eye = torch.eye(int(system.shape[0]), dtype=torch.float64, device=system.device)
    base_jitter = max(float(trace_scale), 1.0) * torch.finfo(torch.float64).eps * 16.0
    last_info = None
    for attempt in range(6):
        jitter = 0.0 if attempt == 0 else base_jitter * (10.0 ** (attempt - 1))
        candidate = system if jitter == 0.0 else system + jitter * eye
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        last_info = int(info.max().detach().cpu().item())
        if last_info == 0:
            solved = torch.cholesky_solve(rhs, factor)
            if bool(torch.isfinite(solved).all().detach().cpu().item()):
                return solved, {
                    "solver": "cholesky",
                    "solver_attempts": attempt + 1,
                    "solver_jitter": float(jitter),
                }
    solved = torch.linalg.lstsq(system, rhs, driver="gelsd").solution
    if not bool(torch.isfinite(solved).all().detach().cpu().item()):
        raise RuntimeError("FP64 ridge solve produced non-finite coefficients.")
    return solved, {
        "solver": "lstsq-gelsd",
        "solver_attempts": 7,
        "solver_jitter": 0.0,
        "last_cholesky_info": last_info,
    }


def prior_preserving_trace_scaled_ridge(
    input_rows,
    target_rows,
    prior_weight,
    prior_bias=None,
    ridge: float = 1e-3,
    fit_bias: bool | None = None,
    torch_module=None,
) -> tuple[Any, Any | None, dict[str, Any]]:
    """Fit ``Y = X W^T + b`` around a prior with a trace-scaled FP64 solve.

    The objective is ``||XW-Y||^2 + lambda||W-W_prior||^2`` where
    ``lambda = ridge * trace(X^T X) / input_features``.  A bias, when present,
    is included as an augmented feature and is anchored to its prior too.
    """

    torch = _require_torch(torch_module)
    if not math.isfinite(float(ridge)) or float(ridge) < 0.0:
        raise ValueError("ridge must be finite and non-negative.")
    x = input_rows.detach().cpu().to(torch.float64)
    y = target_rows.detach().cpu().to(torch.float64)
    prior_w = prior_weight.detach().cpu().to(torch.float64)
    if x.dim() != 2 or y.dim() != 2 or prior_w.dim() != 2:
        raise ValueError(
            "input_rows, target_rows, and prior_weight must be rank-2 tensors."
        )
    if int(x.shape[0]) != int(y.shape[0]) or int(x.shape[0]) <= 0:
        raise ValueError(
            "input_rows and target_rows must have the same positive row count."
        )
    if tuple(prior_w.shape) != (int(y.shape[1]), int(x.shape[1])):
        raise ValueError(
            "prior_weight shape must be [target_features, input_features]."
        )
    use_bias = prior_bias is not None if fit_bias is None else bool(fit_bias)
    if use_bias:
        if prior_bias is None:
            prior_b = torch.zeros(int(y.shape[1]), dtype=torch.float64)
        else:
            prior_b = prior_bias.detach().cpu().to(torch.float64)
            if tuple(prior_b.shape) != (int(y.shape[1]),):
                raise ValueError("prior_bias shape must match target_features.")
        design = torch.cat(
            [x, torch.ones((int(x.shape[0]), 1), dtype=torch.float64)], dim=1
        )
        prior = torch.cat([prior_w, prior_b.unsqueeze(1)], dim=1).t().contiguous()
    else:
        design = x
        prior = prior_w.t().contiguous()
    if not (
        bool(torch.isfinite(design).all().item())
        and bool(torch.isfinite(y).all().item())
        and bool(torch.isfinite(prior).all().item())
    ):
        raise ValueError("Ridge inputs and priors must be finite.")

    feature_gram = x.t().matmul(x)
    trace = float(torch.trace(feature_gram).detach().cpu().item())
    trace_scale = trace / max(1, int(x.shape[1]))
    effective_ridge = float(ridge) * max(trace_scale, torch.finfo(torch.float64).tiny)
    gram = design.t().matmul(design)
    system = gram + effective_ridge * torch.eye(int(gram.shape[0]), dtype=torch.float64)
    rhs = design.t().matmul(y) + effective_ridge * prior
    solved, solve_report = _stable_fp64_solve(torch, system, rhs, trace_scale)
    weight = solved[: int(x.shape[1]), :].t().contiguous()
    bias = solved[-1, :].contiguous() if use_bias else None
    delta_sq = weight.sub(prior_w).pow(2).sum()
    if use_bias:
        delta_sq = delta_sq + bias.sub(prior[-1]).pow(2).sum()
    report = {
        "dtype": "float64",
        "rows": int(x.shape[0]),
        "input_features": int(x.shape[1]),
        "target_features": int(y.shape[1]),
        "fit_bias": use_bias,
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled",
        "feature_gram_trace": trace,
        "trace_scale": float(trace_scale),
        "effective_ridge": float(effective_ridge),
        "prior_delta_norm": float(delta_sq.sqrt().detach().cpu().item()),
        **solve_report,
    }
    return weight, bias, report


def _prediction_metrics(torch, predicted, target) -> dict[str, float]:
    predicted64 = predicted.detach().cpu().to(torch.float64)
    target64 = target.detach().cpu().to(torch.float64)
    error = predicted64 - target64
    mse = error.pow(2).mean()
    target_rms = target64.pow(2).mean().sqrt().clamp_min(1e-18)
    relative_rmse = mse.sqrt() / target_rms
    numerator = (predicted64 * target64).sum(dim=1)
    denominator = (
        predicted64.pow(2)
        .sum(dim=1)
        .sqrt()
        .mul(target64.pow(2).sum(dim=1).sqrt())
        .clamp_min(1e-18)
    )
    cosine = (numerator / denominator).mean()
    return {
        "mse": float(mse.item()),
        "relative_rmse": float(relative_rmse.item()),
        "mean_row_cosine": float(cosine.item()),
    }


def _validation_gate(
    prior_metrics: dict[str, float],
    fitted_metrics: dict[str, float],
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_fitted_relative_rmse: float | None = None,
) -> tuple[bool, str, dict[str, float]]:
    prior_mse = float(prior_metrics["mse"])
    fitted_mse = float(fitted_metrics["mse"])
    improvement = (prior_mse - fitted_mse) / max(abs(prior_mse), 1e-18)
    cosine_delta = float(fitted_metrics["mean_row_cosine"]) - float(
        prior_metrics["mean_row_cosine"]
    )
    diagnostics = {
        "validation_relative_improvement": float(improvement),
        "validation_cosine_delta": float(cosine_delta),
    }
    if not all(
        math.isfinite(value)
        for value in (*prior_metrics.values(), *fitted_metrics.values())
    ):
        return False, "nonfinite_validation_metrics", diagnostics
    if fitted_mse >= prior_mse:
        return False, "validation_mse_not_improved", diagnostics
    if improvement < float(min_relative_improvement):
        return False, "validation_improvement_below_threshold", diagnostics
    if cosine_delta < -float(max_cosine_drop):
        return False, "validation_cosine_drop", diagnostics
    if max_fitted_relative_rmse is not None and float(
        fitted_metrics["relative_rmse"]
    ) > float(max_fitted_relative_rmse):
        return False, "validation_relative_rmse_above_ceiling", diagnostics
    return True, "accepted", diagnostics


def _cat_prompt_rows(
    torch,
    rows: Sequence[Any],
    indices: Sequence[int],
    max_rows: int | None = None,
):
    selected = [rows[int(index)] for index in indices]
    if not selected or any(item is None for item in selected):
        raise ValueError("Prompt trace rows are missing for the requested split.")
    combined = torch.cat([item.detach().cpu() for item in selected], dim=0)
    if max_rows is not None and int(combined.shape[0]) > int(max_rows):
        keep = (
            torch.linspace(0, int(combined.shape[0]) - 1, steps=int(max_rows))
            .round()
            .to(torch.long)
        )
        combined = combined.index_select(0, keep)
    return combined


def _cat_prompt_rows_with_ids(
    torch, rows: Sequence[Any], max_rows: int | None = None
) -> tuple[Any, list[int]]:
    if not rows or any(item is None for item in rows):
        raise ValueError("Prompt trace rows are missing.")
    combined = torch.cat([item.detach().cpu() for item in rows], dim=0)
    prompt_ids = [
        int(prompt_index)
        for prompt_index, item in enumerate(rows)
        for _ in range(int(item.shape[0]))
    ]
    if max_rows is not None and int(combined.shape[0]) > int(max_rows):
        keep = (
            torch.linspace(0, int(combined.shape[0]) - 1, steps=int(max_rows))
            .round()
            .to(torch.long)
        )
        combined = combined.index_select(0, keep)
        prompt_ids = [prompt_ids[int(index)] for index in keep.tolist()]
    return combined, prompt_ids


def _fit_linear_prompt_split(
    torch,
    input_by_prompt: Sequence[Any],
    target_by_prompt: Sequence[Any],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    prior_weight,
    prior_bias,
    ridge: float,
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_rows: int | None = None,
    min_validation_rows: int = 1,
    max_fitted_relative_rmse: float | None = None,
) -> dict[str, Any]:
    if len(input_by_prompt) != len(target_by_prompt):
        raise ValueError("Input and target prompt traces must have the same length.")
    for prompt_idx, (inputs, targets) in enumerate(
        zip(input_by_prompt, target_by_prompt)
    ):
        if (
            inputs is None
            or targets is None
            or int(inputs.shape[0]) != int(targets.shape[0])
        ):
            raise ValueError(
                f"Prompt {prompt_idx} has misaligned input and target rows."
            )
    train_x = _cat_prompt_rows(torch, input_by_prompt, train_indices, max_rows)
    train_y = _cat_prompt_rows(torch, target_by_prompt, train_indices, max_rows)
    val_x = _cat_prompt_rows(torch, input_by_prompt, validation_indices, max_rows)
    val_y = _cat_prompt_rows(torch, target_by_prompt, validation_indices, max_rows)
    if int(val_x.shape[0]) < int(min_validation_rows):
        raise ValueError(
            f"Need at least {int(min_validation_rows)} held-out rows; got {int(val_x.shape[0])}."
        )
    fitted_weight, fitted_bias, solve_report = prior_preserving_trace_scaled_ridge(
        train_x,
        train_y,
        prior_weight,
        prior_bias=prior_bias,
        ridge=ridge,
        fit_bias=prior_bias is not None,
        torch_module=torch,
    )

    prior_prediction = val_x.to(torch.float64).matmul(
        prior_weight.detach().cpu().to(torch.float64).t()
    )
    fitted_prediction = val_x.to(torch.float64).matmul(fitted_weight.t())
    if prior_bias is not None:
        prior_prediction = prior_prediction + prior_bias.detach().cpu().to(
            torch.float64
        )
        fitted_prediction = fitted_prediction + fitted_bias
    prior_metrics = _prediction_metrics(torch, prior_prediction, val_y)
    fitted_metrics = _prediction_metrics(torch, fitted_prediction, val_y)
    accepted, reason, gate_report = _validation_gate(
        prior_metrics,
        fitted_metrics,
        min_relative_improvement=min_relative_improvement,
        max_cosine_drop=max_cosine_drop,
        max_fitted_relative_rmse=max_fitted_relative_rmse,
    )
    return {
        "weight": fitted_weight
        if accepted
        else prior_weight.detach().cpu().to(torch.float64),
        "bias": fitted_bias
        if accepted
        else (
            None if prior_bias is None else prior_bias.detach().cpu().to(torch.float64)
        ),
        "report": {
            **solve_report,
            "accepted": bool(accepted),
            "reverted": not bool(accepted),
            "reason": reason,
            "split_unit": "prompt",
            "train_prompt_indices": [int(item) for item in train_indices],
            "validation_prompt_indices": [int(item) for item in validation_indices],
            "train_rows": int(train_x.shape[0]),
            "validation_rows": int(val_x.shape[0]),
            "validation_prior": prior_metrics,
            "validation_fitted": fitted_metrics,
            **gate_report,
        },
    }


def composed_pair_endpoint(
    first_layer, second_layer, hidden_states, layer_kwargs: dict[str, Any] | None = None
):
    """Evaluate the ordered nonlinear endpoint ``second(first(hidden))``."""

    torch = _require_torch()
    kwargs = dict(layer_kwargs or {})
    with torch.inference_mode():
        first_output = first_layer(hidden_states, **kwargs)
        first_hidden = _first_tensor(first_output)
        second_output = second_layer(first_hidden, **kwargs)
        return _first_tensor(second_output)


def _first_tensor(value):
    if hasattr(value, "dim"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            if hasattr(item, "dim"):
                return item
    candidate = getattr(value, "last_hidden_state", None)
    if candidate is not None:
        return candidate
    raise TypeError("Expected a tensor-bearing module output.")


def build_global_residual_basis(
    donor_embeddings,
    target_hidden_size: int,
    max_rows: int = 4096,
    torch_module=None,
) -> dict[str, Any]:
    """Build one deterministic grouped-orthonormal donor-to-child basis.

    Every donor coordinate is assigned to exactly one child coordinate.  The
    disjoint groups make the columns orthonormal while embedding energy is
    balanced across child coordinates.  This is inexpensive enough for large
    vocabularies and, unlike a per-layer ridge map, keeps residual coordinates
    globally fixed.
    """

    torch = _require_torch(torch_module)
    weights = donor_embeddings.detach().cpu()
    if weights.dim() != 2:
        raise ValueError("donor_embeddings must be rank-2.")
    source_hidden = int(weights.shape[1])
    target_hidden = int(target_hidden_size)
    if target_hidden <= 0 or target_hidden > source_hidden:
        raise ValueError(
            "target_hidden_size must be positive and no larger than donor hidden size."
        )
    if max_rows <= 0:
        raise ValueError("max_rows must be positive.")
    sample_count = min(int(weights.shape[0]), int(max_rows))
    if sample_count <= 0:
        raise ValueError("donor_embeddings must contain at least one row.")
    if sample_count == int(weights.shape[0]):
        sampled = weights
    else:
        sample_indices = (
            torch.linspace(0, int(weights.shape[0]) - 1, steps=sample_count)
            .round()
            .to(torch.long)
        )
        sampled = weights.index_select(0, sample_indices)
    sampled64 = sampled.to(torch.float64)
    energy = sampled64.pow(2).mean(dim=0)
    signed_mean = sampled64.mean(dim=0)
    ordered = torch.argsort(energy, descending=True, stable=True).tolist()
    groups: list[list[int]] = [[] for _ in range(target_hidden)]
    loads = [0.0] * target_hidden
    for source_coordinate in ordered:
        target_coordinate = min(
            range(target_hidden), key=lambda idx: (loads[idx], len(groups[idx]), idx)
        )
        groups[target_coordinate].append(int(source_coordinate))
        loads[target_coordinate] += float(energy[int(source_coordinate)].item())
    if any(not group for group in groups):
        raise RuntimeError(
            "Residual basis construction left an empty child coordinate."
        )

    basis = torch.zeros((source_hidden, target_hidden), dtype=torch.float32)
    assignment = torch.empty(source_hidden, dtype=torch.long)
    assignment_signs = torch.empty(source_hidden, dtype=torch.float32)
    for target_coordinate, group in enumerate(groups):
        coefficient = 1.0 / math.sqrt(len(group))
        for source_coordinate in group:
            sign = 1.0 if float(signed_mean[source_coordinate].item()) >= 0.0 else -1.0
            basis[source_coordinate, target_coordinate] = sign * coefficient
            assignment[source_coordinate] = target_coordinate
            assignment_signs[source_coordinate] = sign
    identity = torch.eye(target_hidden, dtype=torch.float64)
    orthogonality_error = float(
        (basis.to(torch.float64).t().matmul(basis.to(torch.float64)) - identity)
        .abs()
        .max()
        .item()
    )
    if not math.isfinite(orthogonality_error) or orthogonality_error > 1e-6:
        raise RuntimeError("Constructed residual basis is not column-orthonormal.")
    basis_hash = hashlib.sha256(basis.numpy().tobytes()).hexdigest()
    return {
        "basis": basis.contiguous(),
        "output_basis": basis.contiguous(),
        "input_basis": basis.contiguous(),
        "norm_basis": basis.square().contiguous(),
        "assignment": assignment,
        "assignment_signs": assignment_signs,
        "kind": "grouped-orthonormal",
        "scope": "global",
        "source_hidden_size": source_hidden,
        "target_hidden_size": target_hidden,
        "sample_rows": sample_count,
        "group_size_min": min(len(group) for group in groups),
        "group_size_max": max(len(group) for group in groups),
        "orthogonality_max_abs_error": orthogonality_error,
        "sha256": basis_hash,
    }


def _validate_residual_basis(
    torch, basis, source_hidden: int, target_hidden: int
) -> dict[str, Any]:
    values = basis.detach().cpu().to(torch.float32).contiguous()
    if tuple(values.shape) != (int(source_hidden), int(target_hidden)):
        raise ValueError(
            "Residual basis shape mismatch: expected "
            f"{(int(source_hidden), int(target_hidden))}, got {tuple(values.shape)}."
        )
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("Residual basis must be finite.")
    identity = torch.eye(int(target_hidden), dtype=torch.float64)
    error = float(
        (values.to(torch.float64).t().matmul(values.to(torch.float64)) - identity)
        .abs()
        .max()
        .item()
    )
    if error > 1e-3:
        raise ValueError(
            f"Residual basis columns must be approximately orthonormal; max error is {error:.6g}."
        )
    return {
        "basis": values,
        "output_basis": values,
        "input_basis": values,
        "norm_basis": values.square(),
        "kind": "supplied-orthonormal",
        "scope": "global",
        "source_hidden_size": int(source_hidden),
        "target_hidden_size": int(target_hidden),
        "orthogonality_max_abs_error": error,
        "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
    }


class _ParameterWriteTracker:
    def __init__(self, model, torch) -> None:
        self.torch = torch
        try:
            named = list(model.named_parameters(remove_duplicate=False))
        except TypeError:
            named = list(model.named_parameters())
        self.names = [str(name) for name, _param in named]
        self._names_by_id: dict[int, list[str]] = {}
        self._parameters_by_name = {str(name): param for name, param in named}
        for name, parameter in named:
            self._names_by_id.setdefault(id(parameter), []).append(str(name))
        self.written: set[str] = set()
        self.reasons: dict[str, str] = {}

    def mark(self, parameter, reason: str) -> None:
        names = self._names_by_id.get(id(parameter), [])
        if not names:
            raise RuntimeError(
                "Attempted to mark a parameter that is not registered on the target model."
            )
        for name in names:
            self.written.add(name)
            self.reasons[name] = str(reason)

    def write(self, parameter, value, reason: str) -> None:
        if tuple(parameter.shape) != tuple(value.shape):
            raise ValueError(
                f"Parameter write shape mismatch: {tuple(parameter.shape)} != {tuple(value.shape)}."
            )
        with self.torch.no_grad():
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.mark(parameter, reason)

    def preserve_unwritten(self) -> list[str]:
        preserved = []
        for name in self.names:
            if name in self.written:
                continue
            parameter = self._parameters_by_name[name]
            with self.torch.no_grad():
                parameter.copy_(parameter.detach())
            self.written.add(name)
            self.reasons[name] = "explicit-preserve-unrecognized-qwen-parameter"
            preserved.append(name)
        return preserved

    def report(self) -> dict[str, Any]:
        unwritten = sorted(set(self.names) - self.written)
        return {
            "target_parameter_names": len(self.names),
            "written_target_parameter_names": len(self.written),
            "unwritten_target_parameters": len(unwritten),
            "unwritten_target_parameter_names": unwritten,
        }


def _qwen_layer_spec(layer) -> dict[str, Any]:
    from neural_forge.hf_xfamily import _attention_projection_modules, _mlp_spec

    attention_modules = _attention_projection_modules(layer)
    q_proj = attention_modules.get("q")
    k_proj = attention_modules.get("k")
    v_proj = attention_modules.get("v")
    o_proj = attention_modules.get("o")
    if any(module is None for module in (q_proj, k_proj, v_proj, o_proj)):
        raise ValueError(
            "Qwen pair composition requires separate q_proj, k_proj, v_proj, and o_proj modules."
        )
    mlp_spec = _mlp_spec(layer)
    if (
        mlp_spec is None
        or mlp_spec.get("kind") != "gated"
        or len(mlp_spec.get("inputs", [])) != 2
    ):
        raise ValueError("Qwen pair composition requires a gate/up/down SwiGLU MLP.")
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        raise ValueError("Qwen pair composition requires layer.self_attn.")
    head_dim = int(getattr(attention, "head_dim", 0) or 0)
    query_heads = int(getattr(attention, "num_heads", 0) or 0)
    key_value_heads = int(getattr(attention, "num_key_value_heads", 0) or 0)
    if query_heads <= 0 and head_dim > 0:
        query_heads = int(q_proj.weight.shape[0]) // head_dim
    if key_value_heads <= 0 and head_dim > 0:
        key_value_heads = int(k_proj.weight.shape[0]) // head_dim
    if head_dim <= 0 and query_heads > 0:
        head_dim = int(q_proj.weight.shape[0]) // query_heads
    if min(head_dim, query_heads, key_value_heads) <= 0:
        raise ValueError("Could not infer Qwen attention head geometry.")
    if int(q_proj.weight.shape[0]) != query_heads * head_dim:
        raise ValueError("q_proj output rows do not match num_heads * head_dim.")
    if int(k_proj.weight.shape[0]) != key_value_heads * head_dim:
        raise ValueError(
            "k_proj output rows do not match num_key_value_heads * head_dim."
        )
    if int(v_proj.weight.shape[0]) != key_value_heads * head_dim:
        raise ValueError(
            "v_proj output rows do not match num_key_value_heads * head_dim."
        )
    input_norm = getattr(layer, "input_layernorm", None)
    post_attention_norm = getattr(layer, "post_attention_layernorm", None)
    if input_norm is None or post_attention_norm is None:
        raise ValueError(
            "Qwen pair composition requires input_layernorm and post_attention_layernorm."
        )
    return {
        "layer": layer,
        "attention": attention,
        "q": q_proj,
        "k": k_proj,
        "v": v_proj,
        "o": o_proj,
        "query_heads": query_heads,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "hidden_size": int(q_proj.weight.shape[1]),
        "input_norm": input_norm,
        "post_attention_norm": post_attention_norm,
        "q_norm": getattr(attention, "q_norm", None),
        "k_norm": getattr(attention, "k_norm", None),
        "mlp": mlp_spec["mlp"],
        "gate": mlp_spec["inputs"][0],
        "up": mlp_spec["inputs"][1],
        "down": mlp_spec["output"],
        "intermediate_size": int(mlp_spec["inputs"][0].weight.shape[0]),
    }


def _group_average(
    torch,
    source_groups: Sequence[Any],
    source_to_target: Sequence[int],
    target_groups: int,
):
    if not source_groups:
        raise ValueError("Need at least one source group tensor.")
    concatenated = torch.cat(
        [item.detach().to(torch.float32) for item in source_groups], dim=0
    )
    assignment = torch.tensor(
        source_to_target, dtype=torch.long, device=concatenated.device
    )
    if int(assignment.numel()) != int(concatenated.shape[0]):
        raise ValueError("Packing assignment length does not match source group count.")
    output = torch.zeros(
        (int(target_groups), *tuple(concatenated.shape[1:])),
        dtype=torch.float32,
        device=concatenated.device,
    )
    output.index_add_(0, assignment, concatenated)
    counts = torch.bincount(assignment, minlength=int(target_groups)).to(torch.float32)
    if bool((counts <= 0).any().detach().cpu().item()):
        raise ValueError("Packing assignment left an empty target group.")
    view_shape = (int(target_groups),) + (1,) * (output.dim() - 1)
    return output / counts.reshape(view_shape)


def _pack_vector_bias(
    torch, modules: Sequence[Any], reshape: tuple[int, ...], packing, target_groups: int
):
    biases = [getattr(module, "bias", None) for module in modules]
    if not any(item is not None for item in biases):
        return None
    rows = []
    for module, bias in zip(modules, biases):
        if bias is None:
            rows.append(
                torch.zeros(reshape, dtype=torch.float32, device=module.weight.device)
            )
        else:
            rows.append(bias.detach().reshape(reshape))
    return _group_average(torch, rows, packing, target_groups)


def _pack_attention_pair(
    torch,
    donor_specs: Sequence[dict[str, Any]],
    target_spec: dict[str, Any],
    input_basis,
    output_basis,
    tracker: _ParameterWriteTracker,
    activation_scores: Sequence[Any] | None = None,
    path_fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(donor_specs) != 2:
        raise ValueError("Attention pair packing requires exactly two donor layers.")
    if activation_scores is not None and len(activation_scores) != 2:
        raise ValueError("Attention activation scores must cover both donor layers.")
    first = donor_specs[0]
    for item in donor_specs[1:]:
        for key in ("query_heads", "key_value_heads", "head_dim", "hidden_size"):
            if int(item[key]) != int(first[key]):
                raise ValueError(
                    f"Donor attention geometry differs across the pair at {key}."
                )
    if int(first["head_dim"]) != int(target_spec["head_dim"]):
        raise ValueError(
            "Donor and target head_dim must match for exact full-coordinate packing."
        )
    donor_kv = int(first["key_value_heads"])
    target_kv = int(target_spec["key_value_heads"])
    if target_kv % 2:
        raise ValueError("A two-layer pair requires an even target KV-head count.")
    groups_per_layer = target_kv // 2
    ratio = int(first["query_heads"]) // donor_kv
    head_dim = int(first["head_dim"])
    donor_hidden = int(first["hidden_size"])
    selected_kv_heads = []
    selection_scores = []
    for pair_offset, spec in enumerate(donor_specs):
        q_rows = (
            spec["q"].weight.detach().reshape(donor_kv, ratio, head_dim, donor_hidden)
        )
        k_rows = spec["k"].weight.detach().reshape(donor_kv, head_dim, donor_hidden)
        v_rows = spec["v"].weight.detach().reshape(donor_kv, head_dim, donor_hidden)
        o_columns = (
            spec["o"].weight.detach().reshape(donor_hidden, donor_kv, ratio, head_dim)
        )
        weight_scores = (
            q_rows.float().pow(2).mean(dim=(1, 2, 3)).sqrt()
            + k_rows.float().pow(2).mean(dim=(1, 2)).sqrt()
            + v_rows.float().pow(2).mean(dim=(1, 2)).sqrt()
            + o_columns.float().pow(2).mean(dim=(0, 2, 3)).sqrt()
        )
        supplied_scores = (
            None if activation_scores is None else activation_scores[pair_offset]
        )
        if supplied_scores is not None:
            supplied_scores = supplied_scores.detach().to(
                device=weight_scores.device, dtype=torch.float32
            )
            if tuple(supplied_scores.shape) != (donor_kv,):
                raise ValueError(
                    "Attention activation scores must have one value per donor KV head."
                )
            scores = supplied_scores
            criterion = "domain-balanced-kv-activation"
        else:
            scores = weight_scores
            criterion = "qkvo-weight-rms"
        chosen = sorted(
            range(donor_kv),
            key=lambda index: (-float(scores[index].detach().cpu()), index),
        )[:groups_per_layer]
        selected_kv_heads.extend(chosen)
        selection_scores.append(
            {
                "pair_offset": pair_offset,
                "scores": [float(value) for value in scores.detach().cpu()],
                "selected": [int(value) for value in chosen],
                "criterion": criterion,
            }
        )
    if path_fold is not None:
        selected_kv_heads = [int(item) for item in path_fold["selected_kv_heads"]]
        if len(selected_kv_heads) != target_kv:
            raise ValueError(
                "Attention path fold must select one donor KV group per target group."
            )
    plan = build_gqa_group_packing(
        first["query_heads"],
        first["key_value_heads"],
        target_spec["query_heads"],
        target_spec["key_value_heads"],
        head_dim=first["head_dim"],
        pair_width=2,
        selected_kv_heads=selected_kv_heads,
    )
    input_basis_device = input_basis.to(
        device=first["q"].weight.device, dtype=torch.float32
    )
    output_basis_device = output_basis.to(
        device=first["q"].weight.device, dtype=torch.float32
    )

    for name, heads_per_group in (("q", ratio), ("k", 1), ("v", 1)):
        projected = torch.zeros_like(target_spec[name].weight, dtype=torch.float32)
        target_bias = getattr(target_spec[name], "bias", None)
        projected_bias = (
            None
            if target_bias is None
            else torch.zeros_like(target_bias, dtype=torch.float32)
        )
        for group in plan["groups"]:
            target_group = int(group["target_kv_head"])
            source = group["source"]
            spec = donor_specs[int(source["pair_offset"])]
            donor_group = int(source["donor_kv_head"])
            source_start = donor_group * heads_per_group * head_dim
            source_end = source_start + heads_per_group * head_dim
            target_start = target_group * heads_per_group * head_dim
            target_end = target_start + heads_per_group * head_dim
            projected[target_start:target_end] = (
                spec[name]
                .weight[source_start:source_end]
                .detach()
                .to(torch.float32)
                .matmul(input_basis_device)
            )
            source_bias = getattr(spec[name], "bias", None)
            if projected_bias is not None and source_bias is not None:
                projected_bias[target_start:target_end] = (
                    source_bias[source_start:source_end].detach().to(torch.float32)
                )
        tracker.write(
            target_spec[name].weight, projected, f"gqa-intact-selected-{name}-pair-pack"
        )
        if target_bias is not None:
            tracker.write(
                target_bias,
                projected_bias,
                f"gqa-intact-selected-{name}-bias-pair-pack",
            )

    projected_output = torch.zeros_like(target_spec["o"].weight, dtype=torch.float32)
    if path_fold is None:
        for group in plan["groups"]:
            target_group = int(group["target_kv_head"])
            source = group["source"]
            spec = donor_specs[int(source["pair_offset"])]
            donor_group = int(source["donor_kv_head"])
            source_start = donor_group * ratio * head_dim
            source_end = source_start + ratio * head_dim
            target_start = target_group * ratio * head_dim
            target_end = target_start + ratio * head_dim
            projected_output[:, target_start:target_end] = (
                output_basis_device.t().matmul(
                    spec["o"]
                    .weight[:, source_start:source_end]
                    .detach()
                    .to(torch.float32)
                )
            )
        output_reason = "gqa-intact-selected-o-pair-pack"
    else:
        aggregate = path_fold["aggregate_output_coefficients"].to(
            device=projected_output.device, dtype=torch.float32
        )
        expected = (target_kv, ratio * head_dim, int(target_spec["hidden_size"]))
        if tuple(aggregate.shape) != expected:
            raise ValueError(
                f"Attention path aggregate shape must be {expected}; got {tuple(aggregate.shape)}."
            )
        for target_group in range(target_kv):
            start = target_group * ratio * head_dim
            end = start + ratio * head_dim
            projected_output[:, start:end] = aggregate[target_group].t()
        output_reason = (
            "post-softmax-all-path-merged-o-pair-pack"
            if path_fold["materialization_mode"] == "all-path-aggregate"
            else "post-softmax-selected-native-o-pair-pack"
        )
    tracker.write(target_spec["o"].weight, projected_output, output_reason)
    target_o_bias = getattr(target_spec["o"], "bias", None)
    if target_o_bias is not None:
        donor_biases = [getattr(spec["o"], "bias", None) for spec in donor_specs]
        if any(item is not None for item in donor_biases):
            values = [
                torch.zeros(
                    donor_hidden, dtype=torch.float32, device=output_basis_device.device
                )
                if item is None
                else item.detach().to(torch.float32)
                for item in donor_biases
            ]
            packed_bias = (
                torch.stack(values, dim=0).sum(dim=0).matmul(output_basis_device)
            )
        else:
            packed_bias = torch.zeros_like(target_o_bias, dtype=torch.float32)
        tracker.write(target_o_bias, packed_bias, "gqa-intact-o-bias-pair-pack")

    for norm_name in ("q_norm", "k_norm"):
        target_norm = target_spec.get(norm_name)
        if target_norm is None or getattr(target_norm, "weight", None) is None:
            continue
        source_norms = [spec.get(norm_name) for spec in donor_specs]
        source_weights = [
            item.weight.detach().to(torch.float32)
            for item in source_norms
            if item is not None
        ]
        if source_weights and all(
            tuple(item.shape) == tuple(target_norm.weight.shape)
            for item in source_weights
        ):
            tracker.write(
                target_norm.weight,
                torch.stack(source_weights, dim=0).mean(dim=0),
                f"{norm_name}-pair-prior",
            )
    plan["selection"] = selection_scores
    if path_fold is not None:
        plan["selection_source"] = (
            "post-softmax-all-path-decomposition"
            if path_fold["materialization_mode"] == "all-path-aggregate"
            else "post-softmax-selected-native-path-fallback"
        )
        plan["path_materialization_mode"] = path_fold["materialization_mode"]
        plan["post_nonlinearity_path_decomposition"] = path_fold["report"]
    return plan


def _pack_swiglu_pair(
    torch,
    donor_specs: Sequence[dict[str, Any]],
    target_spec: dict[str, Any],
    basis,
    tracker: _ParameterWriteTracker,
    activation_scores: Sequence[Any] | None = None,
    path_fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(donor_specs) != 2:
        raise ValueError("SwiGLU pair packing requires exactly two donor layers.")
    donor_intermediate = int(donor_specs[0]["intermediate_size"])
    donor_hidden = int(donor_specs[0]["hidden_size"])
    if any(
        int(item["intermediate_size"]) != donor_intermediate for item in donor_specs
    ):
        raise ValueError("Donor SwiGLU intermediate sizes differ across the pair.")
    target_intermediate = int(target_spec["intermediate_size"])
    if activation_scores is not None and len(activation_scores) != 2:
        raise ValueError("SwiGLU activation scores must cover both donor layers.")
    channel_scores = []
    score_criterion = "gate-up-down-weight-rms-product"
    for pair_offset, item in enumerate(donor_specs):
        supplied_scores = (
            None if activation_scores is None else activation_scores[pair_offset]
        )
        if supplied_scores is not None:
            supplied_scores = supplied_scores.detach().cpu().to(torch.float32)
            if tuple(supplied_scores.shape) != (donor_intermediate,):
                raise ValueError(
                    "SwiGLU activation scores must have one value per donor intermediate channel."
                )
            channel_scores.append(supplied_scores.tolist())
            score_criterion = "domain-balanced-swiglu-activation"
            continue
        gate_score = item["gate"].weight.detach().float().pow(2).mean(dim=1).sqrt()
        up_score = item["up"].weight.detach().float().pow(2).mean(dim=1).sqrt()
        down_score = item["down"].weight.detach().float().pow(2).mean(dim=0).sqrt()
        channel_scores.append(
            (gate_score * up_score * down_score).detach().cpu().tolist()
        )
    plan = build_coupled_swiglu_packing(
        donor_intermediate,
        target_intermediate,
        pair_width=2,
        channel_scores=channel_scores,
        selected_sources=(None if path_fold is None else path_fold["selected_sources"]),
    )
    basis_device = basis.to(
        device=donor_specs[0]["gate"].weight.device, dtype=torch.float32
    )
    for name in ("gate", "up"):
        projected_rows = torch.zeros_like(target_spec[name].weight, dtype=torch.float32)
        for item in plan["target_channels"]:
            target_channel = int(item["target_channel"])
            pair_offset, donor_channel = item["sources"][0]
            projected_rows[target_channel] = (
                donor_specs[int(pair_offset)][name]
                .weight[int(donor_channel)]
                .detach()
                .to(torch.float32)
                .matmul(basis_device)
            )
        tracker.write(
            target_spec[name].weight,
            projected_rows,
            f"coupled-swiglu-intact-selected-{name}-pair-pack",
        )
        target_bias = getattr(target_spec[name], "bias", None)
        if target_bias is not None:
            packed_bias = torch.zeros_like(target_bias, dtype=torch.float32)
            for item in plan["target_channels"]:
                target_channel = int(item["target_channel"])
                pair_offset, donor_channel = item["sources"][0]
                source_bias = getattr(donor_specs[int(pair_offset)][name], "bias", None)
                if source_bias is not None:
                    packed_bias[target_channel] = (
                        source_bias[int(donor_channel)].detach().to(torch.float32)
                    )
            tracker.write(
                target_bias,
                packed_bias,
                f"coupled-swiglu-intact-selected-{name}-bias-pair-pack",
            )

    projected_down = torch.zeros_like(target_spec["down"].weight, dtype=torch.float32)
    if path_fold is None:
        for item in plan["target_channels"]:
            target_channel = int(item["target_channel"])
            pair_offset, donor_channel = item["sources"][0]
            projected_down[:, target_channel] = basis_device.t().matmul(
                donor_specs[int(pair_offset)]["down"]
                .weight[:, int(donor_channel)]
                .detach()
                .to(torch.float32)
            )
        down_reason = "coupled-swiglu-intact-selected-down-pair-pack"
    else:
        aggregate = path_fold["aggregate_output_coefficients"].to(
            device=projected_down.device, dtype=torch.float32
        )
        expected = (target_intermediate, int(target_spec["hidden_size"]))
        if tuple(aggregate.shape) != expected:
            raise ValueError(
                f"SwiGLU path aggregate shape must be {expected}; got {tuple(aggregate.shape)}."
            )
        projected_down.copy_(aggregate.t())
        down_reason = (
            "post-swiglu-all-path-merged-down-pair-pack"
            if path_fold["materialization_mode"] == "all-path-aggregate"
            else "post-swiglu-selected-native-down-pair-pack"
        )
    tracker.write(
        target_spec["down"].weight,
        projected_down,
        down_reason,
    )
    target_down_bias = getattr(target_spec["down"], "bias", None)
    if target_down_bias is not None:
        source_biases = [getattr(item["down"], "bias", None) for item in donor_specs]
        if any(item is not None for item in source_biases):
            values = [
                torch.zeros(
                    donor_hidden, dtype=torch.float32, device=basis_device.device
                )
                if item is None
                else item.detach().to(torch.float32)
                for item in source_biases
            ]
            packed_bias = torch.stack(values, dim=0).sum(dim=0).matmul(basis_device)
        else:
            packed_bias = torch.zeros_like(target_down_bias, dtype=torch.float32)
        tracker.write(
            target_down_bias,
            packed_bias,
            "coupled-swiglu-intact-selected-down-bias-pair-pack",
        )
    plan["selection_criterion"] = score_criterion
    if path_fold is not None:
        plan["selection_criterion"] = (
            "post-swiglu-all-path-decomposition"
            if path_fold["materialization_mode"] == "all-path-aggregate"
            else "post-swiglu-selected-native-path-fallback"
        )
        plan["path_materialization_mode"] = path_fold["materialization_mode"]
        plan["post_nonlinearity_path_decomposition"] = path_fold["report"]
    return plan


def _project_norm_prior(
    torch, donor_norms: Sequence[Any], target_norm, basis, tracker, reason: str
) -> None:
    if getattr(target_norm, "weight", None) is None:
        return
    source_weights = [getattr(item, "weight", None) for item in donor_norms]
    source_weights = [
        item.detach().to(torch.float32) for item in source_weights if item is not None
    ]
    if not source_weights:
        return
    donor_hidden = int(basis.shape[0])
    if any(tuple(item.shape) != (donor_hidden,) for item in source_weights):
        raise ValueError(
            "Donor RMSNorm shape does not match the global residual basis."
        )
    mean_weight = torch.stack(source_weights, dim=0).mean(dim=0)
    norm_basis = basis.to(device=mean_weight.device, dtype=torch.float32).square()
    projected = mean_weight.matmul(norm_basis)
    tracker.write(target_norm.weight, projected, reason)
    target_bias = getattr(target_norm, "bias", None)
    if target_bias is not None:
        source_biases = [getattr(item, "bias", None) for item in donor_norms]
        source_biases = [
            item.detach().to(torch.float32)
            for item in source_biases
            if item is not None
        ]
        projected_bias = (
            torch.stack(source_biases, dim=0)
            .mean(dim=0)
            .matmul(basis.to(torch.float32))
            if source_biases
            else torch.zeros_like(target_bias, dtype=torch.float32)
        )
        tracker.write(target_bias, projected_bias, f"{reason}-bias")


def _token_id_map(
    source_tokenizer, target_tokenizer, source_rows: int, target_rows: int
) -> list[int]:
    mapping = [-1] * int(target_rows)
    source_vocab = (
        source_tokenizer.get_vocab() if hasattr(source_tokenizer, "get_vocab") else None
    )
    target_vocab = (
        target_tokenizer.get_vocab() if hasattr(target_tokenizer, "get_vocab") else None
    )
    if isinstance(source_vocab, dict) and isinstance(target_vocab, dict):
        for token, target_id in target_vocab.items():
            source_id = source_vocab.get(token)
            if source_id is None:
                continue
            target_index = int(target_id)
            source_index = int(source_id)
            if 0 <= target_index < target_rows and 0 <= source_index < source_rows:
                mapping[target_index] = source_index
        if source_vocab == target_vocab:
            # Qwen configs can reserve embedding rows that are intentionally
            # absent from tokenizer.get_vocab().  Identical token-id maps make
            # same-index transfer unambiguous for that configured capacity.
            for index in range(min(int(source_rows), int(target_rows))):
                if mapping[index] < 0:
                    mapping[index] = index
    else:
        for index in range(min(int(source_rows), int(target_rows))):
            mapping[index] = index
    return mapping


def _project_vocab_matrix(
    torch,
    source_parameter,
    target_parameter,
    token_mapping: Sequence[int],
    basis,
    tracker: _ParameterWriteTracker,
    reason: str,
    chunk_rows: int = 2048,
) -> dict[str, Any]:
    if source_parameter.dim() != 2 or target_parameter.dim() != 2:
        raise ValueError("Vocabulary matrices must be rank-2.")
    if int(source_parameter.shape[1]) != int(basis.shape[0]) or int(
        target_parameter.shape[1]
    ) != int(basis.shape[1]):
        raise ValueError(
            "Vocabulary matrix hidden dimensions do not match the residual basis."
        )
    if len(token_mapping) != int(target_parameter.shape[0]):
        raise ValueError("token_mapping length must match target vocabulary rows.")
    basis_device = basis.to(device=source_parameter.device, dtype=torch.float32)
    source_sum = torch.zeros(
        int(source_parameter.shape[1]),
        dtype=torch.float32,
        device=source_parameter.device,
    )
    for source_start in range(0, int(source_parameter.shape[0]), int(chunk_rows)):
        source_end = min(int(source_parameter.shape[0]), source_start + int(chunk_rows))
        source_sum.add_(
            source_parameter[source_start:source_end]
            .detach()
            .to(torch.float32)
            .sum(dim=0)
        )
    source_mean = source_sum.div(float(source_parameter.shape[0])).matmul(basis_device)
    mapped = 0
    with torch.no_grad():
        for start in range(0, int(target_parameter.shape[0]), int(chunk_rows)):
            end = min(int(target_parameter.shape[0]), start + int(chunk_rows))
            target_ids = list(range(start, end))
            source_ids = [int(token_mapping[index]) for index in target_ids]
            output = source_mean.unsqueeze(0).expand(end - start, -1).clone()
            valid_positions = [
                idx for idx, source_idx in enumerate(source_ids) if source_idx >= 0
            ]
            if valid_positions:
                gather = torch.tensor(
                    [source_ids[idx] for idx in valid_positions],
                    dtype=torch.long,
                    device=source_parameter.device,
                )
                projected = (
                    source_parameter.detach()
                    .index_select(0, gather)
                    .to(torch.float32)
                    .matmul(basis_device)
                )
                position_tensor = torch.tensor(
                    valid_positions, dtype=torch.long, device=output.device
                )
                output.index_copy_(0, position_tensor, projected)
                mapped += len(valid_positions)
            target_parameter[start:end].copy_(
                output.to(device=target_parameter.device, dtype=target_parameter.dtype)
            )
    tracker.mark(target_parameter, reason)
    return {
        "source_rows": int(source_parameter.shape[0]),
        "target_rows": int(target_parameter.shape[0]),
        "mapped_rows": int(mapped),
        "fallback_rows": int(target_parameter.shape[0]) - int(mapped),
        "fallback": "projected-donor-mean",
    }


def _find_final_norm(model):
    parent, _attr, _layers = _decoder_layer_container(model)
    for name in ("norm", "final_layer_norm", "final_norm", "ln_f", "norm_f"):
        module = getattr(parent, name, None)
        if module is not None:
            return module
    raise ValueError("Could not find the Qwen final norm module.")


def _model_backbone(model):
    parent, _attr, _layers = _decoder_layer_container(model)
    if not callable(parent):
        raise ValueError("Qwen decoder-layer parent is not callable.")
    return parent


def _tokenizer_prompt_ids(tokenizer, text: str, max_length: int) -> list[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return [
                int(item)
                for item in tokenizer.encode(text, add_special_tokens=True)[
                    : int(max_length)
                ]
            ]
        except TypeError:
            return [int(item) for item in tokenizer.encode(text)[: int(max_length)]]
    encoded = tokenizer(
        [str(text)],
        return_tensors=None,
        padding=False,
        truncation=True,
        max_length=int(max_length),
    )
    values = encoded["input_ids"]
    if values and isinstance(values[0], (list, tuple)):
        values = values[0]
    return [int(item) for item in values]


def _validate_prompt_tokenization(
    source_tokenizer, target_tokenizer, texts: Sequence[str], max_length: int
) -> None:
    for prompt_index, text in enumerate(texts):
        source_ids = _tokenizer_prompt_ids(source_tokenizer, str(text), max_length)
        target_ids = _tokenizer_prompt_ids(target_tokenizer, str(text), max_length)
        if source_ids != target_ids:
            raise ValueError(
                "Qwen pair composition requires identical prompt token IDs; "
                f"prompt {prompt_index} differs between donor and target tokenizers."
            )


def _make_trace_batches(
    torch,
    tokenizer,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
):
    batches = []
    for start in range(0, len(texts), int(batch_size)):
        chunk = [str(item) for item in texts[start : start + int(batch_size)]]
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(max_length),
        )
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
        model_inputs = {
            key: value.to(device)
            for key, value in dict(encoded).items()
            if hasattr(value, "to")
            and key in {"input_ids", "attention_mask", "position_ids"}
        }
        batches.append(
            {
                "start": int(start),
                "count": len(chunk),
                "attention_mask": model_inputs["attention_mask"],
                "model_inputs": model_inputs,
            }
        )
    return batches


def _run_backbone(backbone, model_inputs: dict[str, Any]):
    try:
        return backbone(**model_inputs, use_cache=False)
    except TypeError as exc:
        if "use_cache" not in str(exc):
            raise
        return backbone(**model_inputs)


def _new_prompt_slots(prompt_count: int) -> list[Any | None]:
    return [None for _ in range(int(prompt_count))]


def _capture_prompt_rows(
    torch,
    destination: list[Any | None],
    tensor,
    attention_mask,
    prompt_start: int,
    projection=None,
    max_rows_per_prompt: int | None = None,
    storage_dtype=None,
) -> None:
    values = _first_tensor(tensor).detach()
    if values.dim() == 2:
        values = values.unsqueeze(0)
    if values.dim() != 3:
        raise ValueError(
            f"Expected traced hidden states with rank 3, got rank {values.dim()}."
        )
    if projection is not None:
        values = values.to(torch.float32).matmul(
            projection.to(device=values.device, dtype=torch.float32)
        )
    else:
        values = values.to(torch.float32 if storage_dtype is None else storage_dtype)
    mask = attention_mask.to(device=values.device, dtype=torch.bool)
    if tuple(mask.shape[:2]) != tuple(values.shape[:2]):
        raise ValueError("Trace attention mask does not align with hidden states.")
    for local_index in range(int(values.shape[0])):
        prompt_index = int(prompt_start) + local_index
        if destination[prompt_index] is not None:
            raise RuntimeError(
                f"Prompt trace {prompt_index} was captured more than once."
            )
        prompt_rows = values[local_index][mask[local_index]]
        if max_rows_per_prompt is not None and int(prompt_rows.shape[0]) > int(
            max_rows_per_prompt
        ):
            keep = (
                torch.linspace(
                    0,
                    int(prompt_rows.shape[0]) - 1,
                    steps=int(max_rows_per_prompt),
                    device=prompt_rows.device,
                )
                .round()
                .to(torch.long)
            )
            prompt_rows = prompt_rows.index_select(0, keep)
        destination[prompt_index] = prompt_rows.detach().cpu().contiguous()


def _packed_donor_attention_output(
    torch,
    tensor,
    plan: dict[str, Any],
    name: str,
    pair_offset: int,
):
    values = _first_tensor(tensor)
    ratio = int(plan["query_heads_per_kv"])
    head_dim = int(plan["head_dim"])
    heads_per_group = ratio if name == "q" else 1
    target_width = (
        int(plan["target_query_heads"])
        if name == "q"
        else int(plan["target_key_value_heads"])
    ) * head_dim
    packed = torch.zeros(
        (*tuple(values.shape[:-1]), target_width),
        dtype=values.dtype,
        device=values.device,
    )
    for group in plan["groups"]:
        source = group["source"]
        if int(source["pair_offset"]) != int(pair_offset):
            continue
        donor_group = int(source["donor_kv_head"])
        target_group = int(group["target_kv_head"])
        source_start = donor_group * heads_per_group * head_dim
        source_end = source_start + heads_per_group * head_dim
        target_start = target_group * heads_per_group * head_dim
        target_end = target_start + heads_per_group * head_dim
        packed[..., target_start:target_end] = values[..., source_start:source_end]
    return packed


def _packed_donor_mlp_output(
    torch,
    tensor,
    plan: dict[str, Any],
    pair_offset: int,
):
    values = _first_tensor(tensor)
    packed = torch.zeros(
        (*tuple(values.shape[:-1]), int(plan["target_intermediate_size"])),
        dtype=values.dtype,
        device=values.device,
    )
    for item in plan["target_channels"]:
        target_channel = int(item["target_channel"])
        source_offset, source_channel = item["sources"][0]
        if int(source_offset) == int(pair_offset):
            packed[..., target_channel] = values[..., int(source_channel)]
    return packed


def _sum_prompt_partitions(parts: Sequence[Sequence[Any]], name: str) -> list[Any]:
    if not parts:
        raise ValueError(f"Trace {name!r} has no partitions.")
    output = []
    for prompt_index, values in enumerate(zip(*parts, strict=True)):
        if any(value is None for value in values):
            raise RuntimeError(
                f"Trace {name!r} is missing a partition for prompt {prompt_index}."
            )
        combined = values[0].clone()
        for value in values[1:]:
            if tuple(value.shape) != tuple(combined.shape):
                raise ValueError(f"Trace {name!r} partition shapes do not align.")
            combined.add_(value)
        output.append(combined)
    return output


def _add_prompt_rows(left: Sequence[Any], right: Sequence[Any]) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("Prompt traces must have the same length.")
    output = []
    for prompt_index, (left_rows, right_rows) in enumerate(zip(left, right)):
        if tuple(left_rows.shape) != tuple(right_rows.shape):
            raise ValueError(f"Prompt {prompt_index} trace shapes do not align.")
        output.append(left_rows + right_rows)
    return output


def _assert_complete_trace(rows: Sequence[Any | None], name: str) -> None:
    missing = [index for index, item in enumerate(rows) if item is None]
    if missing:
        raise RuntimeError(f"Trace {name!r} is missing prompts {missing}.")
    if any(int(item.shape[0]) <= 0 for item in rows if item is not None):
        raise RuntimeError(f"Trace {name!r} contains an empty prompt.")


def _trace_donor_post_nonlinearity_paths(
    torch,
    donor,
    pair_plan: Sequence[LayerPair],
    batches: Sequence[dict[str, Any]],
    prompt_count: int,
    max_rows: int,
    capture_paths: bool = True,
) -> dict[str, Any]:
    if int(max_rows) < int(prompt_count):
        raise ValueError(
            "post-nonlinearity path tracing needs at least one row per prompt."
        )
    rows_per_prompt = max(1, int(max_rows) // int(prompt_count))
    donor_layers = list(_decoder_layer_container(donor)[2])
    traces = {
        int(pair.target_layer): {
            "pair_input": _new_prompt_slots(prompt_count),
            "pair_output": _new_prompt_slots(prompt_count),
            "attention": (
                [
                    _new_prompt_slots(prompt_count),
                    _new_prompt_slots(prompt_count),
                ]
                if capture_paths
                else None
            ),
            "mlp": (
                [
                    _new_prompt_slots(prompt_count),
                    _new_prompt_slots(prompt_count),
                ]
                if capture_paths
                else None
            ),
        }
        for pair in pair_plan
    }
    final_hidden = _new_prompt_slots(prompt_count)
    context: dict[str, Any] = {}
    handles = []

    def capture_pre(destination):
        def hook(_module, args):
            _capture_prompt_rows(
                torch,
                destination,
                args[0],
                context["attention_mask"],
                context["prompt_start"],
                max_rows_per_prompt=rows_per_prompt,
                storage_dtype=torch.float16,
            )

        return hook

    for pair in pair_plan:
        pair_trace = traces[int(pair.target_layer)]
        first_layer = donor_layers[int(pair.donor_first)]
        second_layer = donor_layers[int(pair.donor_second)]
        handles.append(
            first_layer.register_forward_pre_hook(capture_pre(pair_trace["pair_input"]))
        )

        def capture_pair_output(destination):
            def hook(_module, _args, output):
                _capture_prompt_rows(
                    torch,
                    destination,
                    output,
                    context["attention_mask"],
                    context["prompt_start"],
                    max_rows_per_prompt=rows_per_prompt,
                    storage_dtype=torch.float16,
                )

            return hook

        handles.append(
            second_layer.register_forward_hook(
                capture_pair_output(pair_trace["pair_output"])
            )
        )
        for pair_offset, donor_layer_index in enumerate(pair.donor_layers):
            if not capture_paths:
                continue
            spec = _qwen_layer_spec(donor_layers[int(donor_layer_index)])
            handles.append(
                spec["o"].register_forward_pre_hook(
                    capture_pre(pair_trace["attention"][pair_offset])
                )
            )
            handles.append(
                spec["down"].register_forward_pre_hook(
                    capture_pre(pair_trace["mlp"][pair_offset])
                )
            )

    def capture_final(_module, _args, output):
        _capture_prompt_rows(
            torch,
            final_hidden,
            output,
            context["attention_mask"],
            context["prompt_start"],
            max_rows_per_prompt=rows_per_prompt,
            storage_dtype=torch.float16,
        )

    handles.append(_find_final_norm(donor).register_forward_hook(capture_final))

    backbone = _model_backbone(donor)
    was_training = donor.training
    donor.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                context["attention_mask"] = batch["attention_mask"]
                context["prompt_start"] = int(batch["start"])
                _run_backbone(backbone, batch["model_inputs"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            donor.train()
    for pair_index, pair_trace in traces.items():
        _assert_complete_trace(
            pair_trace["pair_input"], f"donor_pair_{pair_index}_input"
        )
        _assert_complete_trace(
            pair_trace["pair_output"], f"donor_pair_{pair_index}_output"
        )
        for name in (() if not capture_paths else ("attention", "mlp")):
            for pair_offset, rows in enumerate(pair_trace[name]):
                _assert_complete_trace(
                    rows,
                    f"donor_pair_{pair_index}_{name}_paths_{pair_offset}",
                )
    _assert_complete_trace(final_hidden, "donor_final_observable_hidden")
    return {
        "pairs": traces,
        "final_hidden": final_hidden,
        "max_rows": int(max_rows),
        "rows_per_prompt": int(rows_per_prompt),
        "storage_dtype": "float16",
        "paths_captured": bool(capture_paths),
    }


def _aligned_pair_path_rows(
    torch,
    partitions: Sequence[Sequence[Any]],
    paths_per_layer: int,
    channels_per_path: int | None,
) -> tuple[Any, list[int]]:
    if len(partitions) != 2:
        raise ValueError("Ordered path decomposition requires two donor partitions.")
    rows = []
    prompt_ids = []
    for prompt_index, (first, second) in enumerate(
        zip(partitions[0], partitions[1], strict=True)
    ):
        if int(first.shape[0]) != int(second.shape[0]):
            raise ValueError("Donor pair path traces do not align by token row.")
        if channels_per_path is None:
            expected = int(paths_per_layer)
            if int(first.shape[1]) != expected or int(second.shape[1]) != expected:
                raise ValueError("SwiGLU path trace width does not match the donor.")
            combined = torch.cat((first, second), dim=1)
        else:
            expected = int(paths_per_layer) * int(channels_per_path)
            if int(first.shape[1]) != expected or int(second.shape[1]) != expected:
                raise ValueError("Attention path trace width does not match the donor.")
            combined = torch.cat(
                (
                    first.reshape(
                        int(first.shape[0]), paths_per_layer, channels_per_path
                    ),
                    second.reshape(
                        int(second.shape[0]), paths_per_layer, channels_per_path
                    ),
                ),
                dim=1,
            )
        rows.append(combined)
        prompt_ids.extend([int(prompt_index)] * int(combined.shape[0]))
    return torch.cat(rows, dim=0), prompt_ids


def _aligned_observable_boundary_rows(
    torch,
    named_rows: Sequence[tuple[str, Sequence[Any]]],
    final_rows: Sequence[Any],
) -> tuple[dict[str, Any], Any, list[int]]:
    if not named_rows:
        raise ValueError("Need at least one donor boundary for observable charts.")
    flattened: dict[str, Any] = {}
    prompt_ids = []
    for prompt_index, final_prompt_rows in enumerate(final_rows):
        expected = int(final_prompt_rows.shape[0])
        for name, rows in named_rows:
            if int(rows[prompt_index].shape[0]) != expected:
                raise ValueError(
                    f"Observable boundary {name!r} does not align at prompt {prompt_index}."
                )
        prompt_ids.extend([int(prompt_index)] * expected)
    for name, rows in named_rows:
        flattened[name] = torch.cat(list(rows), dim=0)
    return flattened, torch.cat(list(final_rows), dim=0), prompt_ids


def _canonical_observable_basis(torch, encoder, fallback_basis, target_hidden: int):
    candidate = torch.cat(
        (
            encoder.detach().cpu().to(torch.float64),
            fallback_basis.detach().cpu().to(torch.float64),
        ),
        dim=1,
    )
    basis, _ = torch.linalg.qr(candidate, mode="reduced")
    if int(basis.shape[1]) < int(target_hidden):
        raise RuntimeError("Observable chart could not span the target hidden width.")
    basis = basis[:, : int(target_hidden)].contiguous()
    pivots = basis.abs().argmax(dim=0)
    pivot_values = basis.gather(0, pivots.unsqueeze(0)).squeeze(0)
    signs = torch.where(
        pivot_values < 0.0,
        -torch.ones_like(pivot_values),
        torch.ones_like(pivot_values),
    )
    return (basis * signs.unsqueeze(0)).to(torch.float32).contiguous()


def _build_logit_observable_pair_bases(
    torch,
    pair_plan: Sequence[LayerPair],
    path_traces: dict[str, Any],
    global_basis,
    texts: Sequence[str],
    ridge: float,
    validation_fraction: float,
    min_train_prompts: int,
    min_validation_prompts: int,
    split_seed: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    from neural_forge.hf_qwen_observable import fit_logit_observable_charts
    from neural_forge.hf_xfamily import _activation_calibration_domain

    ordered_pairs = sorted(pair_plan, key=lambda item: int(item.target_layer))
    named_rows: list[tuple[str, Sequence[Any]]] = [
        ("pair_input_0", path_traces["pairs"][0]["pair_input"])
    ]
    named_rows.extend(
        (
            f"pair_output_{int(pair.target_layer)}",
            path_traces["pairs"][int(pair.target_layer)]["pair_output"],
        )
        for pair in ordered_pairs
    )
    boundaries, final_observable, prompt_ids = _aligned_observable_boundary_rows(
        torch, named_rows, path_traces["final_hidden"]
    )
    domains_by_prompt = [_activation_calibration_domain(str(text)) for text in texts]
    domains = [domains_by_prompt[index] for index in prompt_ids]
    target_hidden = int(global_basis.shape[1])
    result = fit_logit_observable_charts(
        boundaries,
        final_observable,
        rank=target_hidden,
        prompt_ids=prompt_ids,
        domains=domains,
        method="rrr",
        ridge=ridge,
        transition_ridge=ridge,
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
    )
    chart_by_name = {str(chart["name"]): chart for chart in result["charts"]}
    materialized = {}
    materialization_report = []
    for name, chart in chart_by_name.items():
        basis = _canonical_observable_basis(
            torch, chart["encoder"], global_basis, target_hidden
        )
        descriptor = _validate_residual_basis(
            torch, basis, int(global_basis.shape[0]), target_hidden
        )
        descriptor.update(
            {
                "kind": "logit-observable-rrr-orthonormalized",
                "scope": "layer-boundary",
                "chart_name": name,
                "observable_retained_rank": int(chart["retained_rank"]),
                "affine_chart_mean_materialized": False,
            }
        )
        materialized[name] = descriptor["basis"]
        materialization_report.append(_public_basis_report(descriptor))
    pair_bases = {}
    for pair in ordered_pairs:
        target_index = int(pair.target_layer)
        input_name = (
            "pair_input_0" if target_index == 0 else f"pair_output_{target_index - 1}"
        )
        output_name = f"pair_output_{target_index}"
        pair_bases[target_index] = {
            "input": materialized[input_name],
            "output": materialized[output_name],
            "input_name": input_name,
            "output_name": output_name,
        }
    return pair_bases, {
        "enabled": True,
        "future_observable": "donor-final-rmsnorm-hidden",
        "fit": result["report"],
        "materialization": materialization_report,
        "coordinate_policy": "layer-varying-orthonormal-predictive-subspaces",
        "chart_means_materialized": False,
    }


def _path_materialization_gate(
    report: dict[str, Any],
    *,
    min_relative_improvement: float,
    max_cosine_drop: float,
) -> dict[str, Any]:
    held_out = report["diagnostics"]["held_out"]
    baseline = held_out["baseline"]
    fitted = held_out["fitted"]
    baseline_rmse = float(baseline["absolute_rmse"])
    fitted_rmse = float(fitted["absolute_rmse"])
    baseline_cosine = float(baseline["cosine_similarity_mean"])
    fitted_cosine = float(fitted["cosine_similarity_mean"])
    values = (baseline_rmse, fitted_rmse, baseline_cosine, fitted_cosine)
    finite = all(math.isfinite(value) for value in values)
    scale = max(abs(baseline_rmse), 1e-12)
    relative_improvement = (baseline_rmse - fitted_rmse) / scale
    cosine_drop = baseline_cosine - fitted_cosine
    rmse_improved = (
        finite
        and fitted_rmse < baseline_rmse
        and relative_improvement >= float(min_relative_improvement)
    )
    cosine_preserved = finite and cosine_drop <= float(max_cosine_drop)
    accepted = bool(rmse_improved and cosine_preserved)
    if not finite:
        reason = "nonfinite_held_out_metrics"
    elif fitted_rmse >= baseline_rmse:
        reason = "held_out_absolute_rmse_not_improved"
    elif relative_improvement < float(min_relative_improvement):
        reason = "held_out_relative_improvement_below_threshold"
    elif not cosine_preserved:
        reason = "held_out_cosine_drop_above_threshold"
    else:
        reason = "held_out_response_improved"
    return {
        "accepted": accepted,
        "reverted": not accepted,
        "reason": reason,
        "materialization_mode": (
            "all-path-aggregate" if accepted else "selected-native-fallback"
        ),
        "held_out_approximate": bool(held_out.get("approximate", False)),
        "baseline_absolute_rmse": baseline_rmse,
        "fitted_absolute_rmse": fitted_rmse,
        "relative_rmse_improvement": relative_improvement,
        "minimum_relative_improvement": float(min_relative_improvement),
        "baseline_cosine_similarity": baseline_cosine,
        "fitted_cosine_similarity": fitted_cosine,
        "cosine_drop": cosine_drop,
        "maximum_cosine_drop": float(max_cosine_drop),
    }


def _build_post_nonlinearity_path_folds(
    torch,
    donor,
    target,
    pair_plan: Sequence[LayerPair],
    path_traces: dict[str, Any],
    basis,
    pair_bases: dict[int, dict[str, Any]] | None,
    texts: Sequence[str],
    ridge: float,
    validation_fraction: float,
    min_train_prompts: int,
    min_validation_prompts: int,
    split_seed: int,
    selection_sketch_size: int,
    interpolation_mode: str,
    min_relative_improvement: float,
    max_cosine_drop: float,
) -> dict[int, dict[str, Any]]:
    from neural_forge.hf_qwen_path_decomposition import (
        compress_post_nonlinearity_paths,
    )
    from neural_forge.hf_xfamily import _activation_calibration_domain

    donor_specs = [
        _qwen_layer_spec(layer) for layer in _decoder_layer_container(donor)[2]
    ]
    target_specs = [
        _qwen_layer_spec(layer) for layer in _decoder_layer_container(target)[2]
    ]
    prompt_domains = [_activation_calibration_domain(str(text)) for text in texts]
    folds: dict[int, dict[str, Any]] = {}
    for pair in pair_plan:
        target_index = int(pair.target_layer)
        basis_cpu = (
            (basis if pair_bases is None else pair_bases[target_index]["output"])
            .detach()
            .cpu()
            .to(torch.float32)
        )
        source_specs = [donor_specs[index] for index in pair.donor_layers]
        target_spec = target_specs[target_index]
        donor_kv = int(source_specs[0]["key_value_heads"])
        ratio = int(source_specs[0]["query_heads"]) // donor_kv
        head_dim = int(source_specs[0]["head_dim"])
        attention_channels = ratio * head_dim
        target_kv = int(target_spec["key_value_heads"])
        attention_features, attention_prompt_ids = _aligned_pair_path_rows(
            torch,
            path_traces["pairs"][target_index]["attention"],
            donor_kv,
            attention_channels,
        )
        attention_directions = []
        for spec in source_specs:
            grouped = (
                spec["o"]
                .weight.detach()
                .cpu()
                .to(torch.float32)
                .reshape(int(spec["hidden_size"]), donor_kv, attention_channels)
                .permute(1, 2, 0)
            )
            attention_directions.append(grouped.matmul(basis_cpu))
        attention_directions = torch.cat(attention_directions, dim=0)
        attention_domains = [prompt_domains[index] for index in attention_prompt_ids]
        attention_quota = target_kv // 2
        attention_result = compress_post_nonlinearity_paths(
            attention_features,
            attention_directions,
            prompt_ids=attention_prompt_ids,
            domain_labels=attention_domains,
            donor_layer_ids=[int(pair.donor_first)] * donor_kv
            + [int(pair.donor_second)] * donor_kv,
            target_path_count=target_kv,
            minimum_paths_per_layer={
                int(pair.donor_first): attention_quota,
                int(pair.donor_second): target_kv - attention_quota,
            },
            path_kind="attention",
            ridge=ridge,
            validation_fraction=validation_fraction,
            min_train_prompts=min_train_prompts,
            min_validation_prompts=min_validation_prompts,
            split_seed=int(split_seed) + target_index,
            selection_sketch_size=selection_sketch_size,
            interpolation_mode=interpolation_mode,
        )
        attention_selected = [
            int(item) for item in attention_result["selected_path_indices"].tolist()
        ]
        attention_order = sorted(
            range(len(attention_selected)),
            key=lambda index: (attention_selected[index] // donor_kv, index),
        )
        attention_order_tensor = torch.tensor(attention_order, dtype=torch.long)
        attention_gate = _path_materialization_gate(
            attention_result["report"],
            min_relative_improvement=min_relative_improvement,
            max_cosine_drop=max_cosine_drop,
        )
        attention_result["report"]["materialization_gate"] = attention_gate
        attention_coefficients = (
            attention_result["aggregate_output_coefficients"]
            if attention_gate["accepted"]
            else attention_result["selected_native_output_directions"]
        )

        donor_intermediate = int(source_specs[0]["intermediate_size"])
        target_intermediate = int(target_spec["intermediate_size"])
        mlp_features, mlp_prompt_ids = _aligned_pair_path_rows(
            torch,
            path_traces["pairs"][target_index]["mlp"],
            donor_intermediate,
            None,
        )
        mlp_directions = torch.cat(
            [
                spec["down"]
                .weight.detach()
                .cpu()
                .to(torch.float32)
                .t()
                .matmul(basis_cpu)
                for spec in source_specs
            ],
            dim=0,
        )
        mlp_domains = [prompt_domains[index] for index in mlp_prompt_ids]
        first_quota = target_intermediate // 2
        mlp_result = compress_post_nonlinearity_paths(
            mlp_features,
            mlp_directions,
            prompt_ids=mlp_prompt_ids,
            domain_labels=mlp_domains,
            donor_layer_ids=[int(pair.donor_first)] * donor_intermediate
            + [int(pair.donor_second)] * donor_intermediate,
            target_path_count=target_intermediate,
            minimum_paths_per_layer={
                int(pair.donor_first): first_quota,
                int(pair.donor_second): target_intermediate - first_quota,
            },
            path_kind="swiglu",
            ridge=ridge,
            validation_fraction=validation_fraction,
            min_train_prompts=min_train_prompts,
            min_validation_prompts=min_validation_prompts,
            split_seed=int(split_seed) + 1000 + target_index,
            selection_sketch_size=selection_sketch_size,
            interpolation_mode=interpolation_mode,
        )
        mlp_selected = [
            int(item) for item in mlp_result["selected_path_indices"].tolist()
        ]
        mlp_order = sorted(
            range(len(mlp_selected)),
            key=lambda index: (mlp_selected[index] // donor_intermediate, index),
        )
        mlp_order_tensor = torch.tensor(mlp_order, dtype=torch.long)
        mlp_gate = _path_materialization_gate(
            mlp_result["report"],
            min_relative_improvement=min_relative_improvement,
            max_cosine_drop=max_cosine_drop,
        )
        mlp_result["report"]["materialization_gate"] = mlp_gate
        mlp_coefficients = (
            mlp_result["aggregate_output_coefficients"]
            if mlp_gate["accepted"]
            else mlp_result["selected_native_output_directions"]
        )
        folds[target_index] = {
            "attention": {
                "selected_kv_heads": [
                    attention_selected[index] % donor_kv for index in attention_order
                ],
                "aggregate_output_coefficients": attention_coefficients.index_select(
                    0, attention_order_tensor
                ),
                "materialization_mode": attention_gate["materialization_mode"],
                "report": attention_result["report"],
            },
            "mlp": {
                "selected_sources": [
                    [
                        mlp_selected[index] // donor_intermediate,
                        mlp_selected[index] % donor_intermediate,
                    ]
                    for index in mlp_order
                ],
                "aggregate_output_coefficients": mlp_coefficients.index_select(
                    0, mlp_order_tensor
                ),
                "materialization_mode": mlp_gate["materialization_mode"],
                "report": mlp_result["report"],
            },
        }
        _compose_progress(
            "qwen_pair_path_decomposition_done",
            current=target_index + 1,
            total=len(pair_plan),
            attention_paths=target_kv,
            mlp_paths=target_intermediate,
        )
    return folds


def _trace_donor_pair_targets(
    torch,
    donor,
    pair_plan: Sequence[LayerPair],
    structural_pair_plans: dict[int, dict[str, Any]],
    batches: Sequence[dict[str, Any]],
    basis,
    prompt_count: int,
    pair_bases: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    donor_layers = list(_decoder_layer_container(donor)[2])
    final_norm = _find_final_norm(donor)
    pair_traces = {
        int(item.target_layer): {
            "pair_input": _new_prompt_slots(prompt_count),
            "attention_endpoint": _new_prompt_slots(prompt_count),
            "pair_endpoint": _new_prompt_slots(prompt_count),
            "producer_partitions": {
                name: [_new_prompt_slots(prompt_count), _new_prompt_slots(prompt_count)]
                for name in ("q", "k", "v", "gate", "up")
            },
            "residual_partitions": {
                name: [_new_prompt_slots(prompt_count), _new_prompt_slots(prompt_count)]
                for name in ("attention", "mlp")
            },
        }
        for item in pair_plan
    }
    final_projected = _new_prompt_slots(prompt_count)
    final_hidden = _new_prompt_slots(prompt_count)
    context: dict[str, Any] = {}
    handles = []

    def capture_pre(destination, projection):
        def hook(_module, args):
            _capture_prompt_rows(
                torch,
                destination,
                args[0],
                context["attention_mask"],
                context["prompt_start"],
                projection=projection,
            )

        return hook

    def capture_post(destination, projection):
        def hook(_module, _args, output):
            _capture_prompt_rows(
                torch,
                destination,
                output,
                context["attention_mask"],
                context["prompt_start"],
                projection=projection,
            )

        return hook

    def capture_packed(destination, packer):
        def hook(_module, _args, output):
            _capture_prompt_rows(
                torch,
                destination,
                packer(output),
                context["attention_mask"],
                context["prompt_start"],
            )

        return hook

    for pair in pair_plan:
        target_index = int(pair.target_layer)
        output_basis = (
            basis
            if pair_bases is None
            else pair_bases[target_index]["output"].to(
                device=basis.device, dtype=basis.dtype
            )
        )
        pair_plans = structural_pair_plans[target_index]
        first_layer = donor_layers[int(pair.donor_first)]
        second_layer = donor_layers[int(pair.donor_second)]
        post_attention_norm = getattr(second_layer, "post_attention_layernorm", None)
        if post_attention_norm is None:
            raise ValueError("Donor Qwen layer is missing post_attention_layernorm.")
        handles.append(
            first_layer.register_forward_pre_hook(
                capture_pre(pair_traces[target_index]["pair_input"], output_basis)
            )
        )
        handles.append(
            post_attention_norm.register_forward_pre_hook(
                capture_pre(
                    pair_traces[int(pair.target_layer)]["attention_endpoint"],
                    output_basis,
                )
            )
        )
        handles.append(
            second_layer.register_forward_hook(
                capture_post(
                    pair_traces[int(pair.target_layer)]["pair_endpoint"], output_basis
                )
            )
        )
        for pair_offset, donor_layer_index in enumerate(pair.donor_layers):
            spec = _qwen_layer_spec(donor_layers[int(donor_layer_index)])
            handles.append(
                spec["attention"].register_forward_hook(
                    capture_post(
                        pair_traces[target_index]["residual_partitions"]["attention"][
                            pair_offset
                        ],
                        output_basis,
                    )
                )
            )
            handles.append(
                spec["mlp"].register_forward_hook(
                    capture_post(
                        pair_traces[target_index]["residual_partitions"]["mlp"][
                            pair_offset
                        ],
                        output_basis,
                    )
                )
            )
            for name in ("q", "k", "v"):
                handles.append(
                    spec[name].register_forward_hook(
                        capture_packed(
                            pair_traces[target_index]["producer_partitions"][name][
                                pair_offset
                            ],
                            lambda output, name=name, pair_offset=pair_offset, plan=pair_plans["attention"]: (
                                _packed_donor_attention_output(
                                    torch, output, plan, name, pair_offset
                                )
                            ),
                        )
                    )
                )
            for name in ("gate", "up"):
                handles.append(
                    spec[name].register_forward_hook(
                        capture_packed(
                            pair_traces[target_index]["producer_partitions"][name][
                                pair_offset
                            ],
                            lambda output, pair_offset=pair_offset, plan=pair_plans["mlp"]: (
                                _packed_donor_mlp_output(
                                    torch, output, plan, pair_offset
                                )
                            ),
                        )
                    )
                )
    final_basis = (
        basis
        if pair_bases is None
        else pair_bases[max(pair_bases)]["output"].to(
            device=basis.device, dtype=basis.dtype
        )
    )
    handles.append(
        final_norm.register_forward_hook(capture_post(final_projected, final_basis))
    )
    handles.append(final_norm.register_forward_hook(capture_post(final_hidden, None)))
    backbone = _model_backbone(donor)
    was_training = donor.training
    donor.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                context["attention_mask"] = batch["attention_mask"]
                context["prompt_start"] = int(batch["start"])
                _run_backbone(backbone, batch["model_inputs"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            donor.train()
    for pair_index, values in pair_traces.items():
        _assert_complete_trace(values["pair_input"], f"donor_pair_{pair_index}_input")
        _assert_complete_trace(
            values["attention_endpoint"], f"donor_pair_{pair_index}_attention_endpoint"
        )
        _assert_complete_trace(
            values["pair_endpoint"], f"donor_pair_{pair_index}_endpoint"
        )
        partitions = values.pop("producer_partitions")
        for name, name_parts in partitions.items():
            values[f"{name}_output"] = _sum_prompt_partitions(
                name_parts, f"donor_pair_{pair_index}_{name}_output"
            )
        residual_partitions = values.pop("residual_partitions")
        for name, name_parts in residual_partitions.items():
            values[f"{name}_delta"] = _sum_prompt_partitions(
                name_parts, f"donor_pair_{pair_index}_{name}_delta"
            )
    _assert_complete_trace(final_projected, "donor_final_projected")
    _assert_complete_trace(final_hidden, "donor_final_hidden")
    return {
        "pairs": pair_traces,
        "final_projected": final_projected,
        "final_hidden": final_hidden,
    }


def _trace_child_layer(
    torch,
    target,
    target_layer_index: int,
    batches: Sequence[dict[str, Any]],
    prompt_count: int,
) -> dict[str, Any]:
    layers = list(_decoder_layer_container(target)[2])
    layer = layers[int(target_layer_index)]
    spec = _qwen_layer_spec(layer)
    traces = {
        "layer_input": _new_prompt_slots(prompt_count),
        "attention_input": _new_prompt_slots(prompt_count),
        "q_output": _new_prompt_slots(prompt_count),
        "k_output": _new_prompt_slots(prompt_count),
        "v_output": _new_prompt_slots(prompt_count),
        "attention_projection_input": _new_prompt_slots(prompt_count),
        "attention_endpoint": _new_prompt_slots(prompt_count),
        "mlp_input": _new_prompt_slots(prompt_count),
        "gate_output": _new_prompt_slots(prompt_count),
        "up_output": _new_prompt_slots(prompt_count),
        "mlp_projection_input": _new_prompt_slots(prompt_count),
        "block_output": _new_prompt_slots(prompt_count),
    }
    context: dict[str, Any] = {}

    def capture_pre(name: str):
        def hook(_module, args):
            _capture_prompt_rows(
                torch,
                traces[name],
                args[0],
                context["attention_mask"],
                context["prompt_start"],
            )

        return hook

    def capture_post(name: str):
        def hook(_module, _args, output):
            _capture_prompt_rows(
                torch,
                traces[name],
                output,
                context["attention_mask"],
                context["prompt_start"],
            )

        return hook

    handles = [
        layer.register_forward_pre_hook(capture_pre("layer_input")),
        spec["q"].register_forward_pre_hook(capture_pre("attention_input")),
        spec["q"].register_forward_hook(capture_post("q_output")),
        spec["k"].register_forward_hook(capture_post("k_output")),
        spec["v"].register_forward_hook(capture_post("v_output")),
        spec["o"].register_forward_pre_hook(capture_pre("attention_projection_input")),
        spec["post_attention_norm"].register_forward_pre_hook(
            capture_pre("attention_endpoint")
        ),
        spec["gate"].register_forward_pre_hook(capture_pre("mlp_input")),
        spec["gate"].register_forward_hook(capture_post("gate_output")),
        spec["up"].register_forward_hook(capture_post("up_output")),
        spec["down"].register_forward_pre_hook(capture_pre("mlp_projection_input")),
        layer.register_forward_hook(capture_post("block_output")),
    ]
    backbone = _model_backbone(target)
    was_training = target.training
    target.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                context["attention_mask"] = batch["attention_mask"]
                context["prompt_start"] = int(batch["start"])
                _run_backbone(backbone, batch["model_inputs"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            target.train()
    for name, rows in traces.items():
        _assert_complete_trace(rows, f"child_layer_{target_layer_index}_{name}")
    return traces


def _trace_child_final_interface(
    torch,
    target,
    batches: Sequence[dict[str, Any]],
    prompt_count: int,
) -> dict[str, Any]:
    final_norm = _find_final_norm(target)
    traces = {
        "norm_input": _new_prompt_slots(prompt_count),
        "norm_output": _new_prompt_slots(prompt_count),
    }
    context: dict[str, Any] = {}

    def pre_hook(_module, args):
        _capture_prompt_rows(
            torch,
            traces["norm_input"],
            args[0],
            context["attention_mask"],
            context["prompt_start"],
        )

    def post_hook(_module, _args, output):
        _capture_prompt_rows(
            torch,
            traces["norm_output"],
            output,
            context["attention_mask"],
            context["prompt_start"],
        )

    handles = [
        final_norm.register_forward_pre_hook(pre_hook),
        final_norm.register_forward_hook(post_hook),
    ]
    backbone = _model_backbone(target)
    was_training = target.training
    target.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                context["attention_mask"] = batch["attention_mask"]
                context["prompt_start"] = int(batch["start"])
                _run_backbone(backbone, batch["model_inputs"])
    finally:
        for handle in handles:
            handle.remove()
        if was_training:
            target.train()
    for name, rows in traces.items():
        _assert_complete_trace(rows, f"child_final_{name}")
    return traces


def _subtract_prompt_rows(left: Sequence[Any], right: Sequence[Any]) -> list[Any]:
    if len(left) != len(right):
        raise ValueError("Prompt traces must have the same length.")
    output = []
    for prompt_index, (left_rows, right_rows) in enumerate(zip(left, right)):
        if tuple(left_rows.shape) != tuple(right_rows.shape):
            raise ValueError(f"Prompt {prompt_index} trace shapes do not align.")
        output.append(left_rows - right_rows)
    return output


def _snapshot_linear(module) -> dict[str, Any]:
    bias = getattr(module, "bias", None)
    return {
        "weight": module.weight.detach().cpu().clone(),
        "bias": None if bias is None else bias.detach().cpu().clone(),
    }


def _restore_linear(
    tracker: _ParameterWriteTracker, module, snapshot: dict[str, Any], reason: str
) -> None:
    tracker.write(module.weight, snapshot["weight"], reason)
    bias = getattr(module, "bias", None)
    if bias is not None:
        tracker.write(bias, snapshot["bias"], f"{reason}-bias")


def _commit_linear_fit(
    tracker: _ParameterWriteTracker,
    module,
    fit: dict[str, Any],
    reason: str,
) -> None:
    if not bool(fit["report"]["accepted"]):
        return
    tracker.write(module.weight, fit["weight"], reason)
    bias = getattr(module, "bias", None)
    if bias is not None:
        tracker.write(bias, fit["bias"], f"{reason}-bias")


def _prompt_trace_metrics(
    torch,
    rows: Sequence[Any],
    targets: Sequence[Any],
    indices: Sequence[int],
    max_rows: int | None = None,
):
    return _prediction_metrics(
        torch,
        _cat_prompt_rows(torch, rows, indices, max_rows),
        _cat_prompt_rows(torch, targets, indices, max_rows),
    )


def _fit_ordered_child_pair(
    torch,
    target,
    pair: LayerPair,
    donor_targets: dict[str, Any],
    batches: Sequence[dict[str, Any]],
    prompt_count: int,
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    ridge: float,
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_fitted_relative_rmse: float | None,
    max_fit_rows: int,
    min_validation_rows: int,
    tracker: _ParameterWriteTracker,
) -> dict[str, Any]:
    layer = list(_decoder_layer_container(target)[2])[int(pair.target_layer)]
    spec = _qwen_layer_spec(layer)
    attention_snapshots = {
        name: _snapshot_linear(spec[name]) for name in ("q", "k", "v", "o")
    }
    mlp_snapshots = {
        name: _snapshot_linear(spec[name]) for name in ("gate", "up", "down")
    }
    before = _trace_child_layer(torch, target, pair.target_layer, batches, prompt_count)

    attention_producer_fits = {}
    for name in ("q", "k", "v"):
        attention_producer_fits[name] = _fit_linear_prompt_split(
            torch,
            before["attention_input"],
            donor_targets[f"{name}_output"],
            train_indices,
            validation_indices,
            spec[name].weight,
            getattr(spec[name], "bias", None),
            ridge,
            min_relative_improvement,
            max_cosine_drop,
            max_rows=max_fit_rows,
            min_validation_rows=min_validation_rows,
            max_fitted_relative_rmse=max_fitted_relative_rmse,
        )
    attention_producers_accepted = all(
        bool(item["report"]["accepted"]) for item in attention_producer_fits.values()
    )
    if attention_producers_accepted:
        for name, fit in attention_producer_fits.items():
            _commit_linear_fit(
                tracker,
                spec[name],
                fit,
                f"validated-composed-attention-{name}-producer-fit",
            )

    after_attention_producers = _trace_child_layer(
        torch, target, pair.target_layer, batches, prompt_count
    )

    attention_endpoint_target = _add_prompt_rows(
        donor_targets["pair_input"], donor_targets["attention_delta"]
    )
    attention_delta_target = _subtract_prompt_rows(
        attention_endpoint_target, after_attention_producers["layer_input"]
    )
    attention_fit = _fit_linear_prompt_split(
        torch,
        after_attention_producers["attention_projection_input"],
        attention_delta_target,
        train_indices,
        validation_indices,
        spec["o"].weight,
        getattr(spec["o"], "bias", None),
        ridge,
        min_relative_improvement,
        max_cosine_drop,
        max_rows=max_fit_rows,
        min_validation_rows=min_validation_rows,
        max_fitted_relative_rmse=max_fitted_relative_rmse,
    )
    _commit_linear_fit(
        tracker, spec["o"], attention_fit, "validated-composed-attention-endpoint-fit"
    )

    after_attention = _trace_child_layer(
        torch, target, pair.target_layer, batches, prompt_count
    )
    attention_endpoint_prior = _prompt_trace_metrics(
        torch,
        before["attention_endpoint"],
        attention_endpoint_target,
        validation_indices,
        max_fit_rows,
    )
    attention_endpoint_fitted = _prompt_trace_metrics(
        torch,
        after_attention["attention_endpoint"],
        attention_endpoint_target,
        validation_indices,
        max_fit_rows,
    )
    (
        attention_endpoint_accepted,
        attention_endpoint_reason,
        attention_endpoint_gate,
    ) = _validation_gate(
        attention_endpoint_prior,
        attention_endpoint_fitted,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
    )
    attention_transaction_accepted = (
        attention_producers_accepted
        and bool(attention_fit["report"]["accepted"])
        and bool(attention_endpoint_accepted)
    )
    if not attention_transaction_accepted:
        for name, snapshot in attention_snapshots.items():
            _restore_linear(
                tracker,
                spec[name],
                snapshot,
                f"attention-module-transaction-revert-{name}",
            )
        after_attention = _trace_child_layer(
            torch, target, pair.target_layer, batches, prompt_count
        )

    mlp_producer_fits = {}
    for name in ("gate", "up"):
        mlp_producer_fits[name] = _fit_linear_prompt_split(
            torch,
            after_attention["mlp_input"],
            donor_targets[f"{name}_output"],
            train_indices,
            validation_indices,
            spec[name].weight,
            getattr(spec[name], "bias", None),
            ridge,
            min_relative_improvement,
            max_cosine_drop,
            max_rows=max_fit_rows,
            min_validation_rows=min_validation_rows,
            max_fitted_relative_rmse=max_fitted_relative_rmse,
        )
    mlp_producers_accepted = all(
        bool(item["report"]["accepted"]) for item in mlp_producer_fits.values()
    )
    if mlp_producers_accepted:
        for name, fit in mlp_producer_fits.items():
            _commit_linear_fit(
                tracker,
                spec[name],
                fit,
                f"validated-composed-mlp-{name}-producer-fit",
            )

    after_mlp_producers = _trace_child_layer(
        torch, target, pair.target_layer, batches, prompt_count
    )
    mlp_endpoint_target = _add_prompt_rows(
        attention_endpoint_target, donor_targets["mlp_delta"]
    )
    mlp_delta_target = _subtract_prompt_rows(
        mlp_endpoint_target, after_mlp_producers["attention_endpoint"]
    )
    mlp_fit = _fit_linear_prompt_split(
        torch,
        after_mlp_producers["mlp_projection_input"],
        mlp_delta_target,
        train_indices,
        validation_indices,
        spec["down"].weight,
        getattr(spec["down"], "bias", None),
        ridge,
        min_relative_improvement,
        max_cosine_drop,
        max_rows=max_fit_rows,
        min_validation_rows=min_validation_rows,
        max_fitted_relative_rmse=max_fitted_relative_rmse,
    )
    _commit_linear_fit(
        tracker, spec["down"], mlp_fit, "validated-composed-pair-endpoint-fit"
    )

    after_mlp = _trace_child_layer(
        torch, target, pair.target_layer, batches, prompt_count
    )
    mlp_endpoint_prior = _prompt_trace_metrics(
        torch,
        after_attention["block_output"],
        mlp_endpoint_target,
        validation_indices,
        max_fit_rows,
    )
    mlp_endpoint_fitted = _prompt_trace_metrics(
        torch,
        after_mlp["block_output"],
        mlp_endpoint_target,
        validation_indices,
        max_fit_rows,
    )
    mlp_endpoint_accepted, mlp_endpoint_reason, mlp_endpoint_gate = _validation_gate(
        mlp_endpoint_prior,
        mlp_endpoint_fitted,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
    )
    mlp_transaction_accepted = (
        mlp_producers_accepted
        and bool(mlp_fit["report"]["accepted"])
        and bool(mlp_endpoint_accepted)
    )
    if not mlp_transaction_accepted:
        for name, snapshot in mlp_snapshots.items():
            _restore_linear(
                tracker,
                spec[name],
                snapshot,
                f"mlp-module-transaction-revert-{name}",
            )
        after_mlp = _trace_child_layer(
            torch, target, pair.target_layer, batches, prompt_count
        )

    after = after_mlp
    prior_metrics = _prompt_trace_metrics(
        torch,
        before["block_output"],
        donor_targets["pair_endpoint"],
        validation_indices,
        max_fit_rows,
    )
    fitted_metrics = _prompt_trace_metrics(
        torch,
        after["block_output"],
        donor_targets["pair_endpoint"],
        validation_indices,
        max_fit_rows,
    )
    accepted, reason, gate_report = _validation_gate(
        prior_metrics,
        fitted_metrics,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
    )
    if not accepted:
        for name, snapshot in attention_snapshots.items():
            _restore_linear(
                tracker,
                spec[name],
                snapshot,
                f"pair-validation-revert-attention-{name}",
            )
        for name, snapshot in mlp_snapshots.items():
            _restore_linear(
                tracker,
                spec[name],
                snapshot,
                f"pair-validation-revert-mlp-{name}",
            )
    return {
        "target_layer": int(pair.target_layer),
        "donor_layers": [int(pair.donor_first), int(pair.donor_second)],
        "pair_order": "first-then-second",
        "sequential_prefix_trace": True,
        "child_prefix_policy": "current-initialized-child-prefix",
        "attention_target": "sum-of-both-donor-attention-residual-deltas",
        "mlp_target": "sum-of-both-donor-mlp-residual-deltas",
        "block_target": "projected-donor-composed-pair-endpoint",
        "parameter_average_used_as_target": False,
        "attention_producer_fits": {
            name: fit["report"] for name, fit in attention_producer_fits.items()
        },
        "attention_producer_transaction": {
            "accepted": bool(attention_transaction_accepted),
            "reverted": not bool(attention_transaction_accepted),
            "producer_applied": bool(attention_producers_accepted),
            "output_applied": bool(attention_fit["report"]["accepted"]),
            "endpoint_validation": {
                "accepted": bool(attention_endpoint_accepted),
                "reason": attention_endpoint_reason,
                "prior": attention_endpoint_prior,
                "fitted": attention_endpoint_fitted,
                **attention_endpoint_gate,
            },
        },
        "attention_output_fit": attention_fit["report"],
        "mlp_producer_fits": {
            name: fit["report"] for name, fit in mlp_producer_fits.items()
        },
        "mlp_producer_transaction": {
            "accepted": bool(mlp_transaction_accepted),
            "reverted": not bool(mlp_transaction_accepted),
            "producer_applied": bool(mlp_producers_accepted),
            "output_applied": bool(mlp_fit["report"]["accepted"]),
            "endpoint_validation": {
                "accepted": bool(mlp_endpoint_accepted),
                "reason": mlp_endpoint_reason,
                "prior": mlp_endpoint_prior,
                "fitted": mlp_endpoint_fitted,
                **mlp_endpoint_gate,
            },
        },
        "mlp_output_fit": mlp_fit["report"],
        "pair_validation": {
            "accepted": bool(accepted),
            "reverted": not bool(accepted),
            "reason": reason,
            "prior": prior_metrics,
            "fitted": fitted_metrics,
            **gate_report,
        },
    }


def _rms_normalized_prompt_rows(
    torch, input_by_prompt: Sequence[Any], norm_module
) -> list[Any]:
    epsilon = float(
        getattr(norm_module, "variance_epsilon", None)
        or getattr(norm_module, "eps", None)
        or 1e-6
    )
    output = []
    for rows in input_by_prompt:
        values = rows.detach().cpu().to(torch.float64)
        variance = values.pow(2).mean(dim=-1, keepdim=True)
        output.append(values * torch.rsqrt(variance + epsilon))
    return output


def _fit_final_rms_norm(
    torch,
    target,
    child_trace: dict[str, Any],
    donor_projected_final: Sequence[Any],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    ridge: float,
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_fitted_relative_rmse: float | None,
    max_fit_rows: int,
    min_validation_rows: int,
    tracker: _ParameterWriteTracker,
) -> dict[str, Any]:
    final_norm = _find_final_norm(target)
    weight = getattr(final_norm, "weight", None)
    if weight is None or weight.dim() != 1:
        raise ValueError("Qwen final norm must expose a rank-1 weight.")
    if getattr(final_norm, "bias", None) is not None:
        raise ValueError(
            "Qwen pair composition currently requires a bias-free RMS final norm."
        )
    unit_rows = _rms_normalized_prompt_rows(
        torch, child_trace["norm_input"], final_norm
    )
    train_unit = _cat_prompt_rows(torch, unit_rows, train_indices, max_fit_rows).to(
        torch.float64
    )
    train_target = _cat_prompt_rows(
        torch, donor_projected_final, train_indices, max_fit_rows
    ).to(torch.float64)
    val_unit = _cat_prompt_rows(torch, unit_rows, validation_indices, max_fit_rows).to(
        torch.float64
    )
    val_target = _cat_prompt_rows(
        torch, donor_projected_final, validation_indices, max_fit_rows
    ).to(torch.float64)
    if int(val_unit.shape[0]) < int(min_validation_rows):
        raise ValueError(
            f"Need at least {int(min_validation_rows)} held-out final-norm rows; "
            f"got {int(val_unit.shape[0])}."
        )
    if tuple(train_unit.shape) != tuple(train_target.shape):
        raise ValueError("Final norm child and donor projected traces do not align.")
    prior = weight.detach().cpu().to(torch.float64)
    gram = train_unit.pow(2).sum(dim=0)
    trace = float(gram.sum().item())
    trace_scale = trace / max(1, int(gram.numel()))
    effective_ridge = float(ridge) * max(trace_scale, torch.finfo(torch.float64).tiny)
    rhs = (train_unit * train_target).sum(dim=0) + effective_ridge * prior
    fitted = rhs / (gram + effective_ridge).clamp_min(torch.finfo(torch.float64).tiny)
    prior_prediction = val_unit * prior
    fitted_prediction = val_unit * fitted
    prior_metrics = _prediction_metrics(torch, prior_prediction, val_target)
    fitted_metrics = _prediction_metrics(torch, fitted_prediction, val_target)
    accepted, reason, gate_report = _validation_gate(
        prior_metrics,
        fitted_metrics,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
    )
    if accepted:
        tracker.write(weight, fitted, "validated-final-rms-norm-fit")
    return {
        "accepted": bool(accepted),
        "reverted": not bool(accepted),
        "reason": reason,
        "dtype": "float64",
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled",
        "feature_gram_trace": trace,
        "trace_scale": trace_scale,
        "effective_ridge": effective_ridge,
        "fit_kind": "diagonal-rmsnorm",
        "split_unit": "prompt",
        "train_prompt_indices": [int(item) for item in train_indices],
        "validation_prompt_indices": [int(item) for item in validation_indices],
        "train_rows": int(train_unit.shape[0]),
        "validation_rows": int(val_unit.shape[0]),
        "validation_prior": prior_metrics,
        "validation_fitted": fitted_metrics,
        "prior_delta_norm": float((fitted - prior).norm().item()),
        **gate_report,
    }


def _evenly_spaced_items(
    torch, values: Sequence[tuple[int, int]], limit: int
) -> list[tuple[int, int]]:
    if len(values) <= int(limit):
        return list(values)
    indices = (
        torch.linspace(0, len(values) - 1, steps=int(limit))
        .round()
        .to(torch.long)
        .tolist()
    )
    return [values[int(index)] for index in indices]


def _fit_final_head(
    torch,
    target,
    child_final_rows: Sequence[Any],
    donor_projected_final: Sequence[Any],
    token_mapping: Sequence[int],
    train_indices: Sequence[int],
    validation_indices: Sequence[int],
    ridge: float,
    min_relative_improvement: float,
    max_cosine_drop: float,
    max_fitted_relative_rmse: float | None,
    max_fit_rows: int,
    min_validation_rows: int,
    max_validation_vocab_rows: int,
    tracker: _ParameterWriteTracker,
) -> dict[str, Any]:
    output_head = target.get_output_embeddings()
    if output_head is None or getattr(output_head, "weight", None) is None:
        raise ValueError("Target model does not expose an output head.")
    hidden_size = int(output_head.weight.shape[1])
    identity = torch.eye(hidden_size, dtype=torch.float64)
    transport_fit = _fit_linear_prompt_split(
        torch,
        child_final_rows,
        donor_projected_final,
        train_indices,
        validation_indices,
        identity,
        None,
        ridge,
        min_relative_improvement,
        max_cosine_drop,
        max_rows=max_fit_rows,
        min_validation_rows=min_validation_rows,
        max_fitted_relative_rmse=max_fitted_relative_rmse,
    )
    transport_report = transport_fit["report"]
    transport = transport_fit["weight"].to(torch.float64)
    valid_pairs = [
        (target_id, int(source_id))
        for target_id, source_id in enumerate(token_mapping)
        if int(source_id) >= 0 and target_id < int(output_head.weight.shape[0])
    ]
    if not valid_pairs:
        raise ValueError(
            "No shared output-head vocabulary rows are available for validation."
        )
    sampled_pairs = _evenly_spaced_items(
        torch, valid_pairs, max(1, int(max_validation_vocab_rows))
    )
    sampled_target_ids = torch.tensor(
        [item[0] for item in sampled_pairs], dtype=torch.long
    )
    prior_rows = (
        output_head.weight.detach()
        .cpu()
        .index_select(0, sampled_target_ids)
        .to(torch.float64)
    )
    val_child = _cat_prompt_rows(
        torch, child_final_rows, validation_indices, max_fit_rows
    ).to(torch.float64)
    val_donor = _cat_prompt_rows(
        torch, donor_projected_final, validation_indices, max_fit_rows
    ).to(torch.float64)
    desired_logits = val_donor.matmul(prior_rows.t())
    prior_logits = val_child.matmul(prior_rows.t())
    fitted_head_rows = prior_rows.matmul(transport)
    fitted_logits = val_child.matmul(fitted_head_rows.t())
    prior_metrics = _prediction_metrics(torch, prior_logits, desired_logits)
    fitted_metrics = _prediction_metrics(torch, fitted_logits, desired_logits)
    logits_accepted, logits_reason, logits_gate = _validation_gate(
        prior_metrics,
        fitted_metrics,
        min_relative_improvement,
        max_cosine_drop,
        max_fitted_relative_rmse,
    )
    accepted = bool(transport_report["accepted"]) and bool(logits_accepted)
    if not bool(transport_report["accepted"]):
        reason = f"hidden_transport_{transport_report['reason']}"
    elif not logits_accepted:
        reason = f"head_logits_{logits_reason}"
    else:
        reason = "accepted"

    if accepted:
        transport_device = transport.to(
            device=output_head.weight.device, dtype=torch.float32
        )
        with torch.no_grad():
            for start in range(0, int(output_head.weight.shape[0]), 2048):
                end = min(int(output_head.weight.shape[0]), start + 2048)
                transformed = (
                    output_head.weight[start:end]
                    .detach()
                    .to(torch.float32)
                    .matmul(transport_device)
                )
                output_head.weight[start:end].copy_(
                    transformed.to(output_head.weight.dtype)
                )
        tracker.mark(output_head.weight, "validated-final-head-hidden-transport")
        head_bias = getattr(output_head, "bias", None)
        if head_bias is not None:
            tracker.write(
                head_bias, head_bias.detach(), "validated-final-head-bias-preserve"
            )
    return {
        "accepted": bool(accepted),
        "reverted": not bool(accepted),
        "reason": reason,
        "fit_kind": "projected-donor-hidden-transport-composed-into-head",
        "parameter_average_used_as_target": False,
        "transport": transport_report,
        "validation_vocab_rows": len(sampled_pairs),
        "validation_prior": prior_metrics,
        "validation_fitted": fitted_metrics,
        **logits_gate,
    }


def _public_basis_report(basis_descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in basis_descriptor.items()
        if not hasattr(value, "detach")
    }


def _packing_summary(plan: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "attention":
        return {
            "head_dim": int(plan["head_dim"]),
            "donor_query_heads": int(plan["donor_query_heads"]),
            "donor_key_value_heads": int(plan["donor_key_value_heads"]),
            "target_query_heads": int(plan["target_query_heads"]),
            "target_key_value_heads": int(plan["target_key_value_heads"]),
            "query_heads_per_kv": int(plan["query_heads_per_kv"]),
            "available_source_group_count": int(plan["available_source_group_count"]),
            "selected_source_group_count": int(plan["selected_source_group_count"]),
            "discarded_source_group_count": int(plan["discarded_source_group_count"]),
            "selected_kv_heads": [int(value) for value in plan["selected_kv_heads"]],
            "selected_full_head_coordinates_preserved": bool(
                plan["selected_full_head_coordinates_preserved"]
            ),
            "nonlinear_head_averaging": bool(plan["nonlinear_head_averaging"]),
            "covered_query_coordinates": int(plan["covered_query_coordinates"]),
            "selection": plan.get("selection", []),
            "selection_source": plan.get("selection_source"),
            "post_nonlinearity_path_decomposition": plan.get(
                "post_nonlinearity_path_decomposition"
            ),
        }
    assignment_bytes = json.dumps(
        plan["source_to_target"], separators=(",", ":")
    ).encode("ascii")
    return {
        "donor_intermediate_size": int(plan["donor_intermediate_size"]),
        "target_intermediate_size": int(plan["target_intermediate_size"]),
        "source_channel_count": int(plan["source_channel_count"]),
        "selected_channel_count": int(plan["selected_channel_count"]),
        "discarded_channel_count": int(plan["discarded_channel_count"]),
        "selected_per_donor_layer": [
            int(value) for value in plan["selected_per_donor_layer"]
        ],
        "gate_up_down_indices_coupled": bool(plan["gate_up_down_indices_coupled"]),
        "all_source_channels_packed": bool(plan["all_source_channels_packed"]),
        "nonlinear_channel_averaging": bool(plan["nonlinear_channel_averaging"]),
        "selection_criterion": plan.get("selection_criterion"),
        "selection_source": plan.get("selection_source"),
        "selected_sources": [
            [int(item["sources"][0][0]), int(item["sources"][0][1])]
            for item in plan["target_channels"]
        ],
        "post_nonlinearity_path_decomposition": plan.get(
            "post_nonlinearity_path_decomposition"
        ),
        "assignment_sha256": hashlib.sha256(assignment_bytes).hexdigest(),
    }


def _initialize_structured_priors(
    torch,
    donor,
    target,
    source_tokenizer,
    target_tokenizer,
    pair_plan: Sequence[LayerPair],
    basis,
    tracker: _ParameterWriteTracker,
    activation_profile: dict[str, Any] | None = None,
    path_folds: dict[int, dict[str, Any]] | None = None,
    pair_bases: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    donor_layers = list(_decoder_layer_container(donor)[2])
    target_layers = list(_decoder_layer_container(target)[2])
    donor_specs = [_qwen_layer_spec(layer) for layer in donor_layers]
    target_specs = [_qwen_layer_spec(layer) for layer in target_layers]
    pair_reports = []
    pair_plans = {}
    for pair in pair_plan:
        source_specs = [
            donor_specs[int(pair.donor_first)],
            donor_specs[int(pair.donor_second)],
        ]
        attention_scores = None
        mlp_scores = None
        if activation_profile is not None:
            attention_scores = []
            mlp_scores = []
            for source_layer in pair.donor_layers:
                layer_attention = activation_profile.get(
                    "attention_head_profiles", {}
                ).get(int(source_layer), {})
                attention_scores.append((layer_attention.get("kv") or {}).get("scores"))
                mlp_scores.append(
                    activation_profile.get("mlp_scores", {}).get(int(source_layer))
                )
        child_spec = target_specs[int(pair.target_layer)]
        pair_basis = (pair_bases or {}).get(int(pair.target_layer), {})
        input_basis = pair_basis.get("input", basis)
        output_basis = pair_basis.get("output", basis)
        pair_path_fold = (path_folds or {}).get(int(pair.target_layer), {})
        attention_plan = _pack_attention_pair(
            torch,
            source_specs,
            child_spec,
            input_basis,
            output_basis,
            tracker,
            activation_scores=attention_scores,
            path_fold=pair_path_fold.get("attention"),
        )
        mlp_plan = _pack_swiglu_pair(
            torch,
            source_specs,
            child_spec,
            output_basis,
            tracker,
            activation_scores=mlp_scores,
            path_fold=pair_path_fold.get("mlp"),
        )
        _project_norm_prior(
            torch,
            [item["input_norm"] for item in source_specs],
            child_spec["input_norm"],
            input_basis,
            tracker,
            "pair-input-rmsnorm-prior",
        )
        _project_norm_prior(
            torch,
            [item["post_attention_norm"] for item in source_specs],
            child_spec["post_attention_norm"],
            output_basis,
            tracker,
            "pair-post-attention-rmsnorm-prior",
        )
        pair_reports.append(
            {
                "target_layer": int(pair.target_layer),
                "donor_layers": [int(pair.donor_first), int(pair.donor_second)],
                "input_basis": pair_basis.get("input_name", "global"),
                "output_basis": pair_basis.get("output_name", "global"),
                "attention": _packing_summary(attention_plan, "attention"),
                "mlp": _packing_summary(mlp_plan, "mlp"),
            }
        )
        pair_plans[int(pair.target_layer)] = {
            "attention": attention_plan,
            "mlp": mlp_plan,
        }

    source_embeddings = donor.get_input_embeddings()
    target_embeddings = target.get_input_embeddings()
    source_head = donor.get_output_embeddings()
    target_head = target.get_output_embeddings()
    if any(
        item is None
        for item in (source_embeddings, target_embeddings, source_head, target_head)
    ):
        raise ValueError(
            "Donor and target must expose input embeddings and output heads."
        )
    embedding_mapping = _token_id_map(
        source_tokenizer,
        target_tokenizer,
        int(source_embeddings.weight.shape[0]),
        int(target_embeddings.weight.shape[0]),
    )
    head_mapping = _token_id_map(
        source_tokenizer,
        target_tokenizer,
        int(source_head.weight.shape[0]),
        int(target_head.weight.shape[0]),
    )
    first_basis = basis if pair_bases is None else pair_bases[min(pair_bases)]["input"]
    final_basis = basis if pair_bases is None else pair_bases[max(pair_bases)]["output"]
    embedding_report = _project_vocab_matrix(
        torch,
        source_embeddings.weight,
        target_embeddings.weight,
        embedding_mapping,
        first_basis,
        tracker,
        "global-basis-input-embedding-projection",
    )
    head_report = _project_vocab_matrix(
        torch,
        source_head.weight,
        target_head.weight,
        head_mapping,
        final_basis,
        tracker,
        "global-basis-output-head-projection",
    )
    target_head_bias = getattr(target_head, "bias", None)
    if target_head_bias is not None:
        source_head_bias = getattr(source_head, "bias", None)
        projected_bias = torch.zeros_like(target_head_bias, dtype=torch.float32)
        if source_head_bias is not None:
            for target_id, source_id in enumerate(head_mapping):
                if 0 <= int(source_id) < int(source_head_bias.shape[0]):
                    projected_bias[target_id] = (
                        source_head_bias[int(source_id)].detach().to(torch.float32)
                    )
        tracker.write(target_head_bias, projected_bias, "output-head-bias-token-map")
    _project_norm_prior(
        torch,
        [_find_final_norm(donor)],
        _find_final_norm(target),
        final_basis,
        tracker,
        "global-basis-final-rmsnorm-prior",
    )
    return {
        "pairs": pair_reports,
        "embedding": embedding_report,
        "head": head_report,
        "pair_plans": pair_plans,
        "embedding_token_mapping": embedding_mapping,
        "head_token_mapping": head_mapping,
    }


def _validate_model_contract(
    donor,
    target,
    expected_donor_layers: int | None,
    expected_target_layers: int | None,
    required_head_dim: int | None,
) -> tuple[list[LayerPair], list[dict[str, Any]], list[dict[str, Any]]]:
    donor_layers = list(_decoder_layer_container(donor)[2])
    target_layers = list(_decoder_layer_container(target)[2])
    if expected_donor_layers is not None and len(donor_layers) != int(
        expected_donor_layers
    ):
        raise ValueError(
            f"Expected {int(expected_donor_layers)} donor layers, found {len(donor_layers)}."
        )
    if expected_target_layers is not None and len(target_layers) != int(
        expected_target_layers
    ):
        raise ValueError(
            f"Expected {int(expected_target_layers)} target layers, found {len(target_layers)}."
        )
    pair_plan = build_contiguous_pair_plan(len(donor_layers), len(target_layers))
    donor_specs = [_qwen_layer_spec(layer) for layer in donor_layers]
    target_specs = [_qwen_layer_spec(layer) for layer in target_layers]
    donor_reference = donor_specs[0]
    target_reference = target_specs[0]
    for label, specs, reference in (
        ("donor", donor_specs, donor_reference),
        ("target", target_specs, target_reference),
    ):
        for layer_index, spec in enumerate(specs):
            for key in (
                "query_heads",
                "key_value_heads",
                "head_dim",
                "hidden_size",
                "intermediate_size",
            ):
                if int(spec[key]) != int(reference[key]):
                    raise ValueError(f"{label} layer {layer_index} differs in {key}.")
    if required_head_dim is not None:
        if int(donor_reference["head_dim"]) != int(required_head_dim):
            raise ValueError(
                f"Donor head_dim must be {int(required_head_dim)}; got {int(donor_reference['head_dim'])}."
            )
        if int(target_reference["head_dim"]) != int(required_head_dim):
            raise ValueError(
                f"Target head_dim must be {int(required_head_dim)}; got {int(target_reference['head_dim'])}."
            )
    if int(donor_reference["head_dim"]) != int(target_reference["head_dim"]):
        raise ValueError("Donor and target must use the same head_dim.")
    if int(donor_reference["query_heads"]) // int(
        donor_reference["key_value_heads"]
    ) != (
        int(target_reference["query_heads"]) // int(target_reference["key_value_heads"])
    ):
        raise ValueError("Donor and target must use the same query-per-KV GQA ratio.")
    return pair_plan, donor_specs, target_specs


def initialize_qwen_pair_composition_models(
    donor,
    target,
    source_tokenizer,
    target_tokenizer=None,
    *,
    calibration_texts: Sequence[str] | None = None,
    residual_basis=None,
    ridge: float = 1e-3,
    validation_fraction: float = 0.25,
    min_validation_prompts: int = 1,
    min_train_prompts: int = 2,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.01,
    max_fitted_relative_rmse: float | None = 0.75,
    max_length: int = 160,
    batch_size: int = 2,
    max_fit_rows: int = 4096,
    min_validation_rows: int = 128,
    basis_max_rows: int = 4096,
    basis_mode: str = "activation-pca",
    basis_preprocess: str = "uncentered-rowrms",
    activation_domain_balance: str = "keyword",
    observable_coordinates: bool = True,
    observable_coordinate_ridge: float = 1e-6,
    path_decomposition: bool = True,
    path_decomposition_max_rows: int = 1024,
    path_decomposition_ridge: float = 1e-6,
    path_decomposition_sketch_size: int = 64,
    path_decomposition_interpolation_mode: str = "sparse-signature",
    head_validation_vocab_rows: int = 4096,
    expected_donor_layers: int | None = None,
    expected_target_layers: int | None = None,
    required_head_dim: int | None = None,
    split_seed: int = 0,
    device: str = "cuda:0",
    output_dir: str | Path | None = None,
    donor_name: str | None = None,
    target_name: str | None = None,
) -> dict[str, Any]:
    """Initialize already-loaded donor and blank target Qwen model objects."""

    torch = _require_torch()
    texts = [str(item) for item in (calibration_texts or DEFAULT_COMPOSITION_TEXTS)]
    if not texts:
        raise ValueError("Need at least one calibration prompt.")
    if not math.isfinite(float(ridge)) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative.")
    if (
        max_length <= 1
        or batch_size <= 0
        or max_fit_rows <= 0
        or min_validation_rows <= 0
        or basis_max_rows <= 0
        or (path_decomposition and path_decomposition_max_rows <= 0)
        or (path_decomposition and path_decomposition_sketch_size <= 0)
        or head_validation_vocab_rows <= 0
    ):
        raise ValueError(
            "max_length, batch_size, max_fit_rows, min_validation_rows, "
            "basis_max_rows, path decomposition limits, and head_validation_vocab_rows "
            "must be positive."
        )
    if min_relative_improvement < 0 or max_cosine_drop < 0:
        raise ValueError("validation gate thresholds must be non-negative.")
    if (
        not math.isfinite(float(path_decomposition_ridge))
        or float(path_decomposition_ridge) < 0.0
    ):
        raise ValueError("path_decomposition_ridge must be finite and non-negative.")
    if path_decomposition_interpolation_mode not in {"dense", "sparse-signature"}:
        raise ValueError(
            "path_decomposition_interpolation_mode must be dense or sparse-signature."
        )
    if (
        not math.isfinite(float(observable_coordinate_ridge))
        or float(observable_coordinate_ridge) < 0.0
    ):
        raise ValueError("observable_coordinate_ridge must be finite and non-negative.")
    if max_fitted_relative_rmse is not None and (
        not math.isfinite(float(max_fitted_relative_rmse))
        or float(max_fitted_relative_rmse) <= 0.0
    ):
        raise ValueError("max_fitted_relative_rmse must be finite and positive.")
    if basis_mode not in {"activation-pca", "embedding-grouped"}:
        raise ValueError("basis_mode must be activation-pca or embedding-grouped.")
    if basis_preprocess not in {"centered-rowrms", "uncentered-rowrms", "uncentered"}:
        raise ValueError(
            "basis_preprocess must be centered-rowrms, uncentered-rowrms, or uncentered."
        )
    if activation_domain_balance not in {"none", "keyword"}:
        raise ValueError("activation_domain_balance must be none or keyword.")
    target_tokenizer = target_tokenizer or source_tokenizer
    if (
        getattr(target_tokenizer, "pad_token_id", None) is None
        and getattr(target_tokenizer, "eos_token_id", None) is not None
    ):
        target_tokenizer.pad_token = target_tokenizer.eos_token
    if (
        getattr(source_tokenizer, "pad_token_id", None) is None
        and getattr(source_tokenizer, "eos_token_id", None) is not None
    ):
        source_tokenizer.pad_token = source_tokenizer.eos_token
    _validate_prompt_tokenization(source_tokenizer, target_tokenizer, texts, max_length)
    vocab_capacity_report = _ensure_tokenizer_vocab_capacity(target, target_tokenizer)
    from neural_forge.hf_xfamily import _untie_output_head

    output_head_untied = _untie_output_head(torch, target)
    pair_plan, donor_specs, target_specs = _validate_model_contract(
        donor,
        target,
        expected_donor_layers,
        expected_target_layers,
        required_head_dim,
    )
    donor_hidden = int(donor_specs[0]["hidden_size"])
    target_hidden = int(target_specs[0]["hidden_size"])
    if int(donor.get_input_embeddings().weight.shape[1]) != donor_hidden:
        raise ValueError(
            "Donor embedding hidden size does not match decoder hidden size."
        )
    if int(target.get_input_embeddings().weight.shape[1]) != target_hidden:
        raise ValueError(
            "Target embedding hidden size does not match decoder hidden size."
        )
    activation_profile = None
    if residual_basis is None and basis_mode == "activation-pca":
        from neural_forge.hf_xfamily import (
            _build_activation_hidden_basis,
            _collect_donor_axis_activation_scores,
        )

        _compose_progress(
            "qwen_pair_activation_profile_start",
            layers=len(pair_plan) * 2,
            prompts=len(texts),
            max_rows=basis_max_rows,
        )
        activation_profile = _collect_donor_axis_activation_scores(
            torch,
            donor,
            source_tokenizer,
            list(_decoder_layer_container(donor)[2]),
            texts,
            max_length,
            device,
            hidden_basis_max_rows=basis_max_rows,
            linear_fit_max_rows=0,
            activation_feature_mode="stats",
            activation_subspace_rank=0,
            activation_domain_balance=activation_domain_balance,
        )
        activation_rows = activation_profile.get("hidden_basis_rows")
        if activation_rows is None:
            raise RuntimeError(
                "Activation profiling did not produce residual-basis rows."
            )
        basis, activation_basis_report = _build_activation_hidden_basis(
            torch,
            activation_rows,
            donor_hidden,
            target_hidden,
            preprocess=basis_preprocess,
        )
        basis_descriptor = _validate_residual_basis(
            torch, basis, donor_hidden, target_hidden
        )
        basis_descriptor.update(
            {
                "kind": "activation-pca",
                "scope": "global",
                "activation_basis": activation_basis_report,
                "activation_sampling": activation_profile.get("hidden_basis_sampling"),
                "activation_domain_balance": activation_domain_balance,
                "activation_domain_counts": activation_profile.get(
                    "activation_domain_counts", {}
                ),
            }
        )
        _compose_progress(
            "qwen_pair_activation_profile_done",
            retained_energy=activation_basis_report.get("retained_energy"),
            rows=activation_basis_report.get("rows"),
        )
    elif residual_basis is None:
        basis_descriptor = build_global_residual_basis(
            donor.get_input_embeddings().weight,
            target_hidden,
            max_rows=basis_max_rows,
            torch_module=torch,
        )
    else:
        basis_descriptor = _validate_residual_basis(
            torch,
            residual_basis,
            donor_hidden,
            target_hidden,
        )
    basis = basis_descriptor["basis"]
    split = prompt_level_split(
        texts,
        validation_fraction=validation_fraction,
        min_validation_prompts=min_validation_prompts,
        min_train_prompts=min_train_prompts,
        seed=split_seed,
    )
    train_indices = split["train_indices"]
    validation_indices = split["validation_indices"]

    for model in (donor, target):
        model.to(device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        for parameter in model.parameters():
            parameter.grad = None
    tracker = _ParameterWriteTracker(target, torch)
    basis_device = basis.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        batches = _make_trace_batches(
            torch, target_tokenizer, texts, batch_size, max_length, device
        )
        path_folds: dict[int, dict[str, Any]] = {}
        pair_bases = None
        observable_coordinate_report = {"enabled": False}
        path_trace_report = None
        if path_decomposition or observable_coordinates:
            _compose_progress(
                "qwen_pair_path_trace_start",
                pairs=len(pair_plan),
                max_rows=path_decomposition_max_rows,
            )
            path_traces = _trace_donor_post_nonlinearity_paths(
                torch,
                donor,
                pair_plan,
                batches,
                len(texts),
                path_decomposition_max_rows,
                capture_paths=path_decomposition,
            )
            path_trace_report = {
                key: value
                for key, value in path_traces.items()
                if key not in {"pairs", "final_hidden"}
            }
            if observable_coordinates:
                pair_bases, observable_coordinate_report = (
                    _build_logit_observable_pair_bases(
                        torch,
                        pair_plan,
                        path_traces,
                        basis,
                        texts,
                        observable_coordinate_ridge,
                        validation_fraction,
                        min_train_prompts,
                        min_validation_prompts,
                        split_seed,
                    )
                )
            if path_decomposition:
                path_folds = _build_post_nonlinearity_path_folds(
                    torch,
                    donor,
                    target,
                    pair_plan,
                    path_traces,
                    basis,
                    pair_bases,
                    texts,
                    path_decomposition_ridge,
                    validation_fraction,
                    min_train_prompts,
                    min_validation_prompts,
                    split_seed,
                    path_decomposition_sketch_size,
                    path_decomposition_interpolation_mode,
                    min_relative_improvement,
                    max_cosine_drop,
                )
            del path_traces
            _compose_progress(
                "qwen_pair_path_trace_done",
                pairs=len(pair_plan),
            )
        _compose_progress("qwen_pair_structural_init_start", pairs=len(pair_plan))
        structural = _initialize_structured_priors(
            torch,
            donor,
            target,
            source_tokenizer,
            target_tokenizer,
            pair_plan,
            basis_device,
            tracker,
            activation_profile=activation_profile,
            path_folds=path_folds,
            pair_bases=pair_bases,
        )
        _compose_progress("qwen_pair_structural_init_done", pairs=len(pair_plan))
        _compose_progress("qwen_pair_donor_trace_start", batches=len(batches))
        donor_traces = _trace_donor_pair_targets(
            torch,
            donor,
            pair_plan,
            structural["pair_plans"],
            batches,
            basis_device,
            len(texts),
            pair_bases=pair_bases,
        )
        _compose_progress("qwen_pair_donor_trace_done", pairs=len(pair_plan))
        donor.to("cpu")
        if str(device).startswith("cuda") and hasattr(torch, "cuda"):
            torch.cuda.empty_cache()
        layer_reports = []
        for pair in pair_plan:
            _compose_progress(
                "qwen_pair_fit_start",
                current=int(pair.target_layer) + 1,
                total=len(pair_plan),
                donor_first=int(pair.donor_first),
                donor_second=int(pair.donor_second),
            )
            layer_report = _fit_ordered_child_pair(
                torch,
                target,
                pair,
                donor_traces["pairs"][int(pair.target_layer)],
                batches,
                len(texts),
                train_indices,
                validation_indices,
                ridge,
                min_relative_improvement,
                max_cosine_drop,
                max_fitted_relative_rmse,
                max_fit_rows,
                min_validation_rows,
                tracker,
            )
            layer_reports.append(layer_report)
            _compose_progress(
                "qwen_pair_fit_done",
                current=int(pair.target_layer) + 1,
                total=len(pair_plan),
                accepted=bool(layer_report["pair_validation"]["accepted"]),
            )
        child_final_before = _trace_child_final_interface(
            torch, target, batches, len(texts)
        )
        final_norm_report = _fit_final_rms_norm(
            torch,
            target,
            child_final_before,
            donor_traces["final_projected"],
            train_indices,
            validation_indices,
            ridge,
            min_relative_improvement,
            max_cosine_drop,
            max_fitted_relative_rmse,
            max_fit_rows,
            min_validation_rows,
            tracker,
        )
        child_final_after_norm = _trace_child_final_interface(
            torch, target, batches, len(texts)
        )
        projected_final_head_report = _fit_final_head(
            torch,
            target,
            child_final_after_norm["norm_output"],
            donor_traces["final_projected"],
            structural["head_token_mapping"],
            train_indices,
            validation_indices,
            ridge,
            min_relative_improvement,
            max_cosine_drop,
            max_fitted_relative_rmse,
            max_fit_rows,
            min_validation_rows,
            head_validation_vocab_rows,
            tracker,
        )
        child_head_rows, child_head_prompt_ids = _cat_prompt_rows_with_ids(
            torch, child_final_after_norm["norm_output"], max_fit_rows
        )
        donor_head_rows, donor_head_prompt_ids = _cat_prompt_rows_with_ids(
            torch, donor_traces["final_hidden"], max_fit_rows
        )
        if child_head_prompt_ids != donor_head_prompt_ids:
            raise RuntimeError(
                "Child and donor final-state rows do not align for the full logit head."
            )
        from neural_forge.hf_qwen_logit_head import fit_qwen_logit_head

        full_head_result = fit_qwen_logit_head(
            child_head_rows,
            donor_head_rows,
            donor.get_output_embeddings().weight,
            target.get_output_embeddings().weight,
            prompt_ids=child_head_prompt_ids,
            ridge=ridge,
            validation_fraction=validation_fraction,
            min_train_prompts=min_train_prompts,
            min_validation_prompts=min_validation_prompts,
            split_seed=split_seed,
            vocab_chunk_size=head_validation_vocab_rows,
        )
        tracker.write(
            target.get_output_embeddings().weight,
            full_head_result["weight"],
            "full-vocabulary-parent-logit-head-fit",
        )
        final_head_report = full_head_result["report"]
        del full_head_result
        _compose_progress(
            "qwen_pair_final_interface_done",
            norm_accepted=bool(final_norm_report["accepted"]),
            projected_head_accepted=bool(projected_final_head_report["accepted"]),
            full_head_alpha=float(final_head_report["chosen_alpha"]),
        )

    for parameter in target.parameters():
        parameter.grad = None
    gradient_parameter_count = sum(
        1 for parameter in target.parameters() if parameter.grad is not None
    )
    write_report = tracker.report()
    if write_report["unwritten_target_parameters"] != 0:
        raise RuntimeError(
            "Target parameter write accounting is incomplete: "
            f"{write_report['unwritten_target_parameter_names']}"
        )
    if gradient_parameter_count != 0:
        raise RuntimeError(
            "Zero-gradient initializer unexpectedly populated parameter gradients."
        )
    target.to("cpu")
    donor.to("cpu")
    if str(device).startswith("cuda") and hasattr(torch, "cuda"):
        torch.cuda.empty_cache()

    artifact_report = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        from neural_forge.hf_xfamily import _save_hidden_basis_artifact

        artifact_report = _save_hidden_basis_artifact(
            torch, output_path, basis_descriptor
        )
    report = {
        "schema": "neural-forge-qwen-ordered-pair-composition-v2",
        "operation": "initialize_qwen_pair_composition",
        "donor_model": donor_name,
        "target_model": target_name,
        "zero_gradient": True,
        "optimizer_steps": 0,
        "optimizer_constructed": False,
        "backward_calls": 0,
        "autograd_enabled_during_parameter_writes": False,
        "parameters_with_grad": int(gradient_parameter_count),
        "unwritten_target_parameters": int(write_report["unwritten_target_parameters"]),
        "donor_layer_count": len(donor_specs),
        "target_layer_count": len(target_specs),
        "pair_plan": [
            {
                "target_layer": int(item.target_layer),
                "donor_layers": [int(item.donor_first), int(item.donor_second)],
                "contiguous": int(item.donor_second) == int(item.donor_first) + 1,
            }
            for item in pair_plan
        ],
        "pair_plan_formula": "(2j, 2j+1)",
        "residual_basis": _public_basis_report(basis_descriptor),
        "residual_basis_artifact": artifact_report,
        "residual_basis_scope": (
            "layer-varying-logit-observable"
            if observable_coordinates
            else "global-fixed"
        ),
        "basis_mode": basis_mode,
        "basis_preprocess": basis_preprocess,
        "activation_domain_balance": activation_domain_balance,
        "activation_selection_used": activation_profile is not None,
        "observable_coordinates": bool(observable_coordinates),
        "observable_coordinate_ridge": float(observable_coordinate_ridge),
        "observable_coordinate_report": observable_coordinate_report,
        "post_nonlinearity_path_decomposition": bool(path_decomposition),
        "path_decomposition_max_rows": int(path_decomposition_max_rows),
        "path_decomposition_ridge": float(path_decomposition_ridge),
        "path_decomposition_sketch_size": int(path_decomposition_sketch_size),
        "path_decomposition_interpolation_mode": (
            path_decomposition_interpolation_mode
        ),
        "path_decomposition_materialization_gate": {
            "minimum_relative_improvement": float(min_relative_improvement),
            "maximum_cosine_drop": float(max_cosine_drop),
            "fallback": "selected-native-output-directions",
        },
        "path_trace": path_trace_report,
        "head_dim": int(donor_specs[0]["head_dim"]),
        "gqa_selected_full_head_coordinate_packing": True,
        "gqa_all_donor_groups_preserved": False,
        "nonlinear_producer_parameter_averaging": False,
        "swiglu_gate_up_channel_coupling": True,
        "swiglu_all_donor_channels_preserved": False,
        "sequential_on_policy_child_prefix_tracing": True,
        "composed_endpoint_targets": True,
        "nonlinear_producer_path_fitting": True,
        "attention_producer_fit_transaction": "all-qkv-or-none-per-pair",
        "mlp_producer_fit_transaction": "gate-and-up-or-none-per-pair",
        "parameter_average_used_as_training_target": False,
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled",
        "ridge_solve_dtype": "float64",
        "prompt_split": split,
        "calibration_prompt_count": len(texts),
        "max_length": int(max_length),
        "max_fit_rows": int(max_fit_rows),
        "min_validation_rows": int(min_validation_rows),
        "max_fitted_relative_rmse": (
            None
            if max_fitted_relative_rmse is None
            else float(max_fitted_relative_rmse)
        ),
        "structural_initialization": {
            "pairs": structural["pairs"],
            "embedding": structural["embedding"],
            "head": structural["head"],
        },
        "layers": layer_reports,
        "accepted_pair_fits": sum(
            1 for item in layer_reports if bool(item["pair_validation"]["accepted"])
        ),
        "reverted_pair_fits": sum(
            1 for item in layer_reports if bool(item["pair_validation"]["reverted"])
        ),
        "accepted_attention_producer_transactions": sum(
            1
            for item in layer_reports
            if bool(item["attention_producer_transaction"]["accepted"])
        ),
        "accepted_mlp_producer_transactions": sum(
            1
            for item in layer_reports
            if bool(item["mlp_producer_transaction"]["accepted"])
        ),
        "final_norm_fit": final_norm_report,
        "projected_final_head_fit": projected_final_head_report,
        "final_head_fit": final_head_report,
        "vocab_capacity": vocab_capacity_report,
        "output_head_untied": bool(output_head_untied),
        "parameter_writes": write_report,
        "explicitly_preserved_parameter_count": 0,
        "explicitly_preserved_parameters": [],
        "target_parameter_count": sum(
            int(parameter.numel()) for parameter in target.parameters()
        ),
    }
    if output_dir is not None:
        _write_json(
            Path(output_dir) / "neural_forge_qwen_pair_compose_report.json", report
        )
    return report


def initialize_qwen_pair_composition(
    donor_model: str,
    target_model: str,
    output_dir: str | Path,
    *,
    calibration_texts: Sequence[str] | None = None,
    ridge: float = 1e-3,
    validation_fraction: float = 0.25,
    min_validation_prompts: int = 1,
    min_train_prompts: int = 2,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.01,
    max_fitted_relative_rmse: float | None = 0.75,
    max_length: int = 160,
    batch_size: int = 2,
    max_fit_rows: int = 4096,
    min_validation_rows: int = 128,
    basis_max_rows: int = 4096,
    basis_mode: str = "activation-pca",
    basis_preprocess: str = "uncentered-rowrms",
    activation_domain_balance: str = "keyword",
    observable_coordinates: bool = True,
    observable_coordinate_ridge: float = 1e-6,
    path_decomposition: bool = True,
    path_decomposition_max_rows: int = 1024,
    path_decomposition_ridge: float = 1e-6,
    path_decomposition_sketch_size: int = 64,
    path_decomposition_interpolation_mode: str = "sparse-signature",
    head_validation_vocab_rows: int = 4096,
    torch_dtype: str = "float16",
    target_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
    expected_donor_layers: int | None = None,
    expected_target_layers: int | None = None,
    required_head_dim: int | None = None,
) -> dict[str, Any]:
    """Load, initialize, validate, and save a 36-layer Qwen into an 18-layer blank Qwen."""

    torch, transformers = _torch_and_transformers()
    donor_dtype = _dtype_from_name(torch, torch_dtype)
    child_dtype = _dtype_from_name(torch, target_torch_dtype)
    donor = transformers.AutoModelForCausalLM.from_pretrained(
        donor_model,
        torch_dtype=donor_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    target = transformers.AutoModelForCausalLM.from_pretrained(
        target_model,
        torch_dtype=child_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model,
        trust_remote_code=trust_remote_code,
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        target_model,
        trust_remote_code=trust_remote_code,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = initialize_qwen_pair_composition_models(
        donor,
        target,
        source_tokenizer,
        target_tokenizer,
        calibration_texts=calibration_texts,
        ridge=ridge,
        validation_fraction=validation_fraction,
        min_validation_prompts=min_validation_prompts,
        min_train_prompts=min_train_prompts,
        min_relative_improvement=min_relative_improvement,
        max_cosine_drop=max_cosine_drop,
        max_fitted_relative_rmse=max_fitted_relative_rmse,
        max_length=max_length,
        batch_size=batch_size,
        max_fit_rows=max_fit_rows,
        min_validation_rows=min_validation_rows,
        basis_max_rows=basis_max_rows,
        basis_mode=basis_mode,
        basis_preprocess=basis_preprocess,
        activation_domain_balance=activation_domain_balance,
        observable_coordinates=observable_coordinates,
        observable_coordinate_ridge=observable_coordinate_ridge,
        path_decomposition=path_decomposition,
        path_decomposition_max_rows=path_decomposition_max_rows,
        path_decomposition_ridge=path_decomposition_ridge,
        path_decomposition_sketch_size=path_decomposition_sketch_size,
        path_decomposition_interpolation_mode=(
            path_decomposition_interpolation_mode
        ),
        head_validation_vocab_rows=head_validation_vocab_rows,
        expected_donor_layers=expected_donor_layers,
        expected_target_layers=expected_target_layers,
        required_head_dim=required_head_dim,
        device=device,
        output_dir=output_path,
        donor_name=donor_model,
        target_name=target_model,
    )
    from neural_forge.hf_xfamily import _sanitize_generation_config_for_save

    generation_config_report = _sanitize_generation_config_for_save(
        target, target_tokenizer
    )
    target.save_pretrained(output_path, safe_serialization=True)
    target_tokenizer.save_pretrained(output_path)
    report["output_dir"] = str(output_path)
    report["generation_config_sanitized"] = generation_config_report
    _write_json(output_path / "neural_forge_qwen_pair_compose_report.json", report)
    return report


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# Explicit aliases keep experiment code readable without creating a second implementation.
compose_qwen_pairs = initialize_qwen_pair_composition
qwen_pair_composition_init = initialize_qwen_pair_composition


__all__ = [
    "DEFAULT_COMPOSITION_TEXTS",
    "LayerPair",
    "build_contiguous_pair_plan",
    "build_coupled_swiglu_packing",
    "build_global_residual_basis",
    "build_gqa_group_packing",
    "compose_qwen_pairs",
    "composed_pair_endpoint",
    "contiguous_donor_pairs",
    "initialize_qwen_pair_composition",
    "initialize_qwen_pair_composition_models",
    "keyword_domain_balanced_texts",
    "prior_preserving_trace_scaled_ridge",
    "prompt_level_split",
    "qwen_pair_composition_init",
]
