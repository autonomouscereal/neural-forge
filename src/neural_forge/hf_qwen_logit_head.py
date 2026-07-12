"""Zero-gradient full-vocabulary Qwen logit-head transplantation.

The routines here operate on detached hidden-state and head tensors.  They fit
an oblique child-to-donor decoder with a closed-form ridge solve, materialize
the resulting head in bounded vocabulary chunks, and choose a blend using
held-out *actual* donor and child logits.  No optimizer, autograd graph,
backward pass, adapter, or model mutation is involved.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neural_forge.hf_logit_metrics import teacher_margin_sign_retention


REPORT_SCHEMA_VERSION = 1

__all__ = [
    "fit_full_vocabulary_logit_head",
    "fit_qwen_logit_head",
    "materialize_full_vocabulary_head",
    "transplant_qwen_logit_head_checkpoint",
    "transplant_full_vocabulary_logit_head",
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


def _validate_tensor(torch, value: Any, name: str):
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


def _tensor_sha256(torch, tensor, row_chunk_size: int) -> str:
    """Hash logical tensor values without requiring a full contiguous copy."""

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
    rows = int(detached.shape[0]) if detached.ndim else 1
    for start in range(0, rows, int(row_chunk_size)):
        chunk = detached[start : start + int(row_chunk_size)] if detached.ndim else detached
        raw = chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _gradient_fingerprint(torch, tensor, row_chunk_size: int) -> dict[str, Any]:
    gradient = tensor.grad
    if gradient is None:
        return {"present": False, "sha256": None}
    return {
        "present": True,
        "sha256": _tensor_sha256(torch, gradient, row_chunk_size),
    }


def _normalize_prompt_id(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return json.dumps(value, ensure_ascii=True, allow_nan=False)
    return str(value)


def _prompt_row_split(
    prompt_ids: Sequence[Any] | None,
    row_count: int,
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
    min_train = _positive_int(min_train_prompts, "min_train_prompts")
    min_validation = _positive_int(
        min_validation_prompts, "min_validation_prompts"
    )
    if prompt_ids is None:
        normalized = [f"implicit-row-{index}" for index in range(row_count)]
        source = "implicit-row-ids"
    else:
        if len(prompt_ids) != row_count:
            raise ValueError("prompt_ids length must match the hidden-state row count.")
        normalized = [_normalize_prompt_id(item) for item in prompt_ids]
        source = "provided"

    unique_prompts = list(dict.fromkeys(normalized))
    required = min_train + min_validation
    if len(unique_prompts) < required:
        raise ValueError(
            "Not enough distinct prompts for a prompt-level split: "
            f"need at least {required}, got {len(unique_prompts)}."
        )
    validation_count = max(
        min_validation,
        int(round(len(unique_prompts) * float(validation_fraction))),
    )
    validation_count = min(validation_count, len(unique_prompts) - min_train)
    ranked = sorted(
        enumerate(unique_prompts),
        key=lambda item: hashlib.sha256(
            f"{int(split_seed)}\x00{item[0]}\x00{item[1]}".encode("utf-8")
        ).digest(),
    )
    validation_prompts = {item[1] for item in ranked[:validation_count]}
    train_prompts = [item for item in unique_prompts if item not in validation_prompts]
    validation_prompt_list = [
        item for item in unique_prompts if item in validation_prompts
    ]
    train_rows = [
        index for index, prompt in enumerate(normalized) if prompt not in validation_prompts
    ]
    validation_rows = [
        index for index, prompt in enumerate(normalized) if prompt in validation_prompts
    ]
    if set(train_rows) & set(validation_rows) or sorted(train_rows + validation_rows) != list(
        range(row_count)
    ):
        raise RuntimeError("Prompt split produced overlapping or missing rows.")
    report = {
        "unit": "prompt",
        "source": source,
        "seed": int(split_seed),
        "validation_fraction": float(validation_fraction),
        "prompt_count": len(unique_prompts),
        "train_prompt_count": len(train_prompts),
        "validation_prompt_count": len(validation_prompt_list),
        "train_row_count": len(train_rows),
        "validation_row_count": len(validation_rows),
        "train_prompt_ids": train_prompts,
        "validation_prompt_ids": validation_prompt_list,
        "train_row_indices": train_rows,
        "validation_row_indices": validation_rows,
        "row_prompt_ids_sha256": _json_sha256(normalized),
        "train_prompt_ids_sha256": _json_sha256(train_prompts),
        "validation_prompt_ids_sha256": _json_sha256(validation_prompt_list),
    }
    report["assignment_sha256"] = _json_sha256(
        {
            "seed": report["seed"],
            "train_prompt_ids": train_prompts,
            "validation_prompt_ids": validation_prompt_list,
        }
    )
    return train_rows, validation_rows, report


def _stable_fp64_solve(torch, system, rhs, trace_scale: float):
    if system.dtype != torch.float64 or rhs.dtype != torch.float64:
        raise ValueError("Closed-form decoder solve requires FP64 tensors.")
    eye = torch.eye(int(system.shape[0]), dtype=torch.float64, device=system.device)
    base_jitter = (
        max(float(trace_scale), 1.0)
        * torch.finfo(torch.float64).eps
        * 16.0
    )
    last_info = None
    for attempt in range(6):
        jitter = 0.0 if attempt == 0 else base_jitter * (10.0 ** (attempt - 1))
        candidate = system if jitter == 0.0 else system + jitter * eye
        factor, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        last_info = int(info.max().detach().cpu().item())
        if last_info == 0:
            solved = torch.cholesky_solve(rhs, factor)
            if bool(torch.isfinite(solved).all().item()):
                return solved, {
                    "solver": "cholesky",
                    "solver_attempts": attempt + 1,
                    "solver_jitter": float(jitter),
                }
    solved = torch.linalg.lstsq(system, rhs, driver="gelsd").solution
    if not bool(torch.isfinite(solved).all().item()):
        raise RuntimeError("FP64 decoder solve produced non-finite coefficients.")
    return solved, {
        "solver": "lstsq-gelsd",
        "solver_attempts": 7,
        "solver_jitter": 0.0,
        "last_cholesky_info": last_info,
    }


def _fit_oblique_decoder(
    torch,
    child_rows,
    donor_rows,
    prior,
    ridge: float,
):
    x = child_rows.detach().cpu().to(torch.float64)
    z = donor_rows.detach().cpu().to(torch.float64)
    prior_fp64 = prior.detach().cpu().to(torch.float64)
    feature_gram = x.t().matmul(x)
    trace = float(torch.trace(feature_gram).item())
    trace_scale = trace / max(1, int(x.shape[1]))
    effective_ridge = float(ridge) * max(
        trace_scale, torch.finfo(torch.float64).tiny
    )
    system = feature_gram + effective_ridge * torch.eye(
        int(x.shape[1]), dtype=torch.float64
    )
    rhs = x.t().matmul(z) + effective_ridge * prior_fp64.t()
    solved, solve_report = _stable_fp64_solve(torch, system, rhs, trace_scale)
    decoder = solved.t().contiguous()
    return decoder, {
        "dtype": "float64",
        "rows": int(x.shape[0]),
        "child_features": int(x.shape[1]),
        "donor_features": int(z.shape[1]),
        "fit_bias": False,
        "ridge": float(ridge),
        "ridge_mode": "trace-scaled-prior-preserving",
        "feature_gram_trace": trace,
        "trace_scale": float(trace_scale),
        "effective_ridge": float(effective_ridge),
        "prior_delta_norm": float(decoder.sub(prior_fp64).norm().item()),
        **solve_report,
    }


def _matmul_dtype(torch, tensors: Sequence[Any]):
    if all(tensor.dtype == torch.float64 for tensor in tensors):
        return torch.float64
    return torch.float32


def materialize_full_vocabulary_head(
    donor_head,
    decoder,
    *,
    vocab_chunk_size: int = 4096,
    output_dtype=None,
    output_device=None,
):
    """Materialize ``H @ D`` while bounding each vocabulary-side product."""

    torch = _torch()
    donor = _validate_tensor(torch, donor_head, "donor_head")
    decoder_value = _validate_tensor(torch, decoder, "decoder")
    chunk_size = _positive_int(vocab_chunk_size, "vocab_chunk_size")
    if int(donor.shape[1]) != int(decoder_value.shape[0]):
        raise ValueError(
            "donor_head columns must match decoder donor features."
        )
    device = torch.device(output_device) if output_device is not None else donor.device
    dtype = output_dtype if output_dtype is not None else donor.dtype
    if not dtype.is_floating_point:
        raise ValueError("output_dtype must be floating point.")
    compute_dtype = _matmul_dtype(torch, [donor, decoder_value])
    decoder_compute = decoder_value.detach().to(device=device, dtype=compute_dtype)
    output = torch.empty(
        (int(donor.shape[0]), int(decoder_value.shape[1])),
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for start in range(0, int(donor.shape[0]), chunk_size):
            end = min(start + chunk_size, int(donor.shape[0]))
            donor_chunk = donor[start:end].detach().to(
                device=device, dtype=compute_dtype
            )
            output[start:end].copy_(
                donor_chunk.matmul(decoder_compute).to(dtype=dtype)
            )
    return output


def _merge_topk(torch, values, indices, new_values, new_indices, k: int):
    if values is None:
        candidate_values = new_values
        candidate_indices = new_indices
    else:
        candidate_values = torch.cat([values, new_values], dim=1)
        candidate_indices = torch.cat([indices, new_indices], dim=1)
    keep = min(int(k), int(candidate_values.shape[1]))
    kept_values, positions = torch.topk(candidate_values, k=keep, dim=1)
    kept_indices = candidate_indices.gather(1, positions)
    return kept_values, kept_indices


def _evaluate_real_logits(
    torch,
    child_rows,
    donor_rows,
    donor_head,
    current_child_head,
    fitted_child_head,
    alphas: Sequence[float],
    *,
    vocab_chunk_size: int,
    top_k: int,
    temperature: float,
):
    evaluation_device = child_rows.device
    compute_dtype = _matmul_dtype(
        torch,
        [
            child_rows,
            donor_rows,
            donor_head,
            current_child_head,
            fitted_child_head,
        ],
    )
    x = child_rows.detach().to(device=evaluation_device, dtype=compute_dtype)
    z = donor_rows.detach().to(device=evaluation_device, dtype=compute_dtype)
    row_count = int(x.shape[0])
    vocab_size = int(donor_head.shape[0])
    effective_top_k = min(int(top_k), vocab_size)
    negative_infinity = float("-inf")

    teacher_logsumexp = torch.full(
        (row_count,), negative_infinity, device=evaluation_device, dtype=compute_dtype
    )
    teacher_top_values = None
    teacher_top_indices = None
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        donor_chunk = donor_head[start:end].detach().to(
            device=evaluation_device, dtype=compute_dtype
        )
        teacher_logits = z.matmul(donor_chunk.t())
        chunk_logsumexp = torch.logsumexp(
            teacher_logits / float(temperature), dim=1
        )
        teacher_logsumexp = torch.logaddexp(
            teacher_logsumexp, chunk_logsumexp
        )
        chunk_k = min(2, end - start)
        chunk_values, chunk_indices = torch.topk(
            teacher_logits, k=chunk_k, dim=1
        )
        teacher_top_values, teacher_top_indices = _merge_topk(
            torch,
            teacher_top_values,
            teacher_top_indices,
            chunk_values,
            chunk_indices + start,
            2,
        )

    teacher_expected_teacher = torch.zeros_like(teacher_logsumexp)
    teacher_expected_current = torch.zeros_like(teacher_logsumexp)
    teacher_expected_fitted = torch.zeros_like(teacher_logsumexp)
    child_logsumexp = [torch.full_like(teacher_logsumexp, negative_infinity) for _ in alphas]
    child_top_values = [None for _ in alphas]
    child_top_indices = [None for _ in alphas]

    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        donor_chunk = donor_head[start:end].detach().to(
            device=evaluation_device, dtype=compute_dtype
        )
        current_chunk = current_child_head[start:end].detach().to(
            device=evaluation_device, dtype=compute_dtype
        )
        fitted_chunk = fitted_child_head[start:end].detach().to(
            device=evaluation_device, dtype=compute_dtype
        )
        teacher_logits = z.matmul(donor_chunk.t())
        current_logits = x.matmul(current_chunk.t())
        fitted_logits = x.matmul(fitted_chunk.t())
        teacher_scaled = teacher_logits / float(temperature)
        current_scaled = current_logits / float(temperature)
        fitted_scaled = fitted_logits / float(temperature)
        teacher_probability = torch.exp(
            teacher_scaled - teacher_logsumexp.unsqueeze(1)
        )
        teacher_expected_teacher.add_(
            (teacher_probability * teacher_scaled).sum(dim=1)
        )
        teacher_expected_current.add_(
            (teacher_probability * current_scaled).sum(dim=1)
        )
        teacher_expected_fitted.add_(
            (teacher_probability * fitted_scaled).sum(dim=1)
        )
        delta_logits = fitted_logits - current_logits
        for index, alpha in enumerate(alphas):
            candidate_logits = current_logits + float(alpha) * delta_logits
            child_logsumexp[index] = torch.logaddexp(
                child_logsumexp[index],
                torch.logsumexp(candidate_logits / float(temperature), dim=1),
            )
            chunk_k = min(effective_top_k, end - start)
            chunk_values, chunk_indices = torch.topk(
                candidate_logits, k=chunk_k, dim=1
            )
            child_top_values[index], child_top_indices[index] = _merge_topk(
                torch,
                child_top_values[index],
                child_top_indices[index],
                chunk_values,
                chunk_indices + start,
                effective_top_k,
            )

    flat_teacher_pair = teacher_top_indices.reshape(-1)
    current_pair_rows = current_child_head.detach().index_select(
        0, flat_teacher_pair.to(device=current_child_head.device)
    )
    fitted_pair_rows = fitted_child_head.detach().index_select(
        0, flat_teacher_pair.to(device=fitted_child_head.device)
    )
    current_pair_rows = current_pair_rows.to(
        device=evaluation_device, dtype=compute_dtype
    ).reshape(row_count, 2, int(x.shape[1]))
    fitted_pair_rows = fitted_pair_rows.to(
        device=evaluation_device, dtype=compute_dtype
    ).reshape(row_count, 2, int(x.shape[1]))
    current_pair_logits = (current_pair_rows * x.unsqueeze(1)).sum(dim=2)
    fitted_pair_logits = (fitted_pair_rows * x.unsqueeze(1)).sum(dim=2)

    metrics = []
    teacher_top1 = teacher_top_indices[:, 0]
    for index, alpha in enumerate(alphas):
        expected_child = (
            (1.0 - float(alpha)) * teacher_expected_current
            + float(alpha) * teacher_expected_fitted
        )
        kl_rows = (
            teacher_expected_teacher
            - expected_child
            + child_logsumexp[index]
            - teacher_logsumexp
        ).clamp_min(0.0)
        top1 = child_top_indices[index][:, 0].eq(teacher_top1).to(torch.float32)
        topk = (
            child_top_indices[index]
            .eq(teacher_top1.unsqueeze(1))
            .any(dim=1)
            .to(torch.float32)
        )
        child_pair_logits = current_pair_logits + float(alpha) * (
            fitted_pair_logits - current_pair_logits
        )
        margin_retention = teacher_margin_sign_retention(
            teacher_top_values,
            child_pair_logits,
        )
        metrics.append(
            {
                "alpha": float(alpha),
                "row_count": row_count,
                "top1_agreement": float(top1.mean().item()),
                "teacher_top1_in_child_topk": float(topk.mean().item()),
                "top_k": effective_top_k,
                "kl_teacher_child_mean": float(kl_rows.mean().item()),
                "kl_teacher_child_max": float(kl_rows.max().item()),
                "teacher_margin_sign_retention": float(
                    margin_retention.mean().item()
                ),
            }
        )
    return metrics, {
        "compute_dtype": str(compute_dtype),
        "device": str(evaluation_device),
        "passes_over_full_vocabulary": 2,
        "vocabulary_rows_evaluated": vocab_size,
        "held_out_rows": row_count,
        "actual_teacher_logits": "Z @ H.T",
        "actual_child_logits": "X @ W.T",
        "surrogate_logits": False,
    }


def _normalize_alphas(candidate_alphas: Sequence[float] | None) -> list[float]:
    values = (
        [index / 10.0 for index in range(11)]
        if candidate_alphas is None
        else [float(item) for item in candidate_alphas]
    )
    if not values:
        raise ValueError("candidate_alphas must not be empty.")
    normalized = []
    seen = set()
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("candidate_alphas must be finite values in [0, 1].")
        key = float(value)
        if key not in seen:
            normalized.append(key)
            seen.add(key)
    return sorted(normalized)


def _choose_candidate(candidate_metrics: Sequence[dict[str, Any]]):
    def key(item: dict[str, Any]):
        return (
            float(item["top1_agreement"]),
            float(item["teacher_top1_in_child_topk"]),
            float(item["teacher_margin_sign_retention"]),
            -float(item["kl_teacher_child_mean"]),
            -float(item["alpha"]),
        )

    return max(candidate_metrics, key=key)


def _blend_heads(
    torch,
    current_head,
    fitted_head,
    alpha: float,
    vocab_chunk_size: int,
):
    if alpha == 1.0:
        return fitted_head
    if alpha == 0.0:
        return current_head.detach().clone()
    compute_dtype = _matmul_dtype(torch, [current_head, fitted_head])
    output = torch.empty_like(current_head, requires_grad=False)
    with torch.no_grad():
        for start in range(0, int(current_head.shape[0]), vocab_chunk_size):
            end = min(start + vocab_chunk_size, int(current_head.shape[0]))
            current_chunk = current_head[start:end].detach().to(dtype=compute_dtype)
            fitted_chunk = fitted_head[start:end].detach().to(dtype=compute_dtype)
            output[start:end].copy_(
                (current_chunk + float(alpha) * (fitted_chunk - current_chunk)).to(
                    dtype=current_head.dtype
                )
            )
    return output


def fit_qwen_logit_head(
    child_final_hidden_rows,
    donor_final_hidden_rows,
    donor_lm_head,
    current_child_head,
    *,
    prompt_ids: Sequence[Any] | None = None,
    decoder_prior=None,
    ridge: float = 1e-3,
    validation_fraction: float = 0.25,
    min_train_prompts: int = 2,
    min_validation_prompts: int = 1,
    split_seed: int = 0,
    candidate_alphas: Sequence[float] | None = None,
    top_k: int = 5,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
) -> dict[str, Any]:
    """Fit and select a full-vocabulary child head without gradient updates.

    ``D`` is fit on whole-prompt training rows to minimize ``||X D.T - Z||``
    with a trace-scaled ridge penalty around ``decoder_prior``.  The default
    prior is the neutral zero map.  Held-out selection evaluates the real
    equations ``Z @ H.T`` and ``X @ W.T`` over every vocabulary row.
    """

    torch = _torch()
    x = _validate_tensor(torch, child_final_hidden_rows, "child_final_hidden_rows")
    z = _validate_tensor(torch, donor_final_hidden_rows, "donor_final_hidden_rows")
    h = _validate_tensor(torch, donor_lm_head, "donor_lm_head")
    w0 = _validate_tensor(torch, current_child_head, "current_child_head")
    if int(x.shape[0]) != int(z.shape[0]):
        raise ValueError("Child and donor hidden rows must have the same row count.")
    if int(h.shape[1]) != int(z.shape[1]):
        raise ValueError("donor_lm_head columns must match donor hidden features.")
    if int(w0.shape[1]) != int(x.shape[1]):
        raise ValueError("current_child_head columns must match child hidden features.")
    if int(h.shape[0]) != int(w0.shape[0]):
        raise ValueError("Donor and child heads must have the same vocabulary size.")
    if int(h.shape[0]) < 2:
        raise ValueError("At least two vocabulary rows are required for margin metrics.")
    if not math.isfinite(float(ridge)) or float(ridge) < 0.0:
        raise ValueError("ridge must be finite and non-negative.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    chunk_size = _positive_int(vocab_chunk_size, "vocab_chunk_size")
    requested_top_k = _positive_int(top_k, "top_k")
    alphas = _normalize_alphas(candidate_alphas)

    if decoder_prior is None:
        prior = torch.zeros(
            (int(z.shape[1]), int(x.shape[1])), dtype=torch.float64
        )
        prior_source = "zero"
    else:
        prior = _validate_tensor(torch, decoder_prior, "decoder_prior")
        if tuple(prior.shape) != (int(z.shape[1]), int(x.shape[1])):
            raise ValueError(
                "decoder_prior shape must be [donor_features, child_features]."
            )
        prior_source = "provided"

    train_rows, validation_rows, split_report = _prompt_row_split(
        prompt_ids,
        int(x.shape[0]),
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
    )
    train_index_x = torch.tensor(train_rows, dtype=torch.long, device=x.device)
    train_index_z = torch.tensor(train_rows, dtype=torch.long, device=z.device)
    validation_index_x = torch.tensor(
        validation_rows, dtype=torch.long, device=x.device
    )
    validation_index_z = torch.tensor(
        validation_rows, dtype=torch.long, device=z.device
    )

    hash_rows = min(chunk_size, 4096)
    named_inputs = {
        "child_final_hidden_rows": x,
        "donor_final_hidden_rows": z,
        "donor_lm_head": h,
        "current_child_head": w0,
        "decoder_prior": prior,
    }
    input_hashes_before = {
        name: _tensor_sha256(torch, value, hash_rows)
        for name, value in named_inputs.items()
    }
    gradient_hashes_before = {
        name: _gradient_fingerprint(torch, value, hash_rows)
        for name, value in named_inputs.items()
    }

    with torch.no_grad():
        decoder, ridge_report = _fit_oblique_decoder(
            torch,
            x.index_select(0, train_index_x),
            z.index_select(0, train_index_z),
            prior,
            float(ridge),
        )
        fitted_head = materialize_full_vocabulary_head(
            h,
            decoder,
            vocab_chunk_size=chunk_size,
            output_dtype=w0.dtype,
            output_device=w0.device,
        )
        candidate_metrics, evaluation_report = _evaluate_real_logits(
            torch,
            x.index_select(0, validation_index_x),
            z.index_select(0, validation_index_z),
            h,
            w0,
            fitted_head,
            alphas,
            vocab_chunk_size=chunk_size,
            top_k=requested_top_k,
            temperature=float(temperature),
        )
        chosen = _choose_candidate(candidate_metrics)
        chosen_alpha = float(chosen["alpha"])
        chosen_head = _blend_heads(
            torch, w0, fitted_head, chosen_alpha, chunk_size
        )

    input_hashes_after = {
        name: _tensor_sha256(torch, value, hash_rows)
        for name, value in named_inputs.items()
    }
    gradient_hashes_after = {
        name: _gradient_fingerprint(torch, value, hash_rows)
        for name, value in named_inputs.items()
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("An input tensor changed during the closed-form transplant.")
    if gradient_hashes_before != gradient_hashes_after:
        raise RuntimeError("An input gradient changed during the closed-form transplant.")

    chunk_count = math.ceil(int(h.shape[0]) / chunk_size)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation": "qwen-full-vocabulary-oblique-logit-head-transplant",
        "zero_gradient": True,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "autograd_enabled_during_fit": False,
        "inputs_mutated": False,
        "input_gradients_mutated": False,
        "shape": {
            "rows": int(x.shape[0]),
            "child_hidden_features": int(x.shape[1]),
            "donor_hidden_features": int(z.shape[1]),
            "vocabulary_size": int(h.shape[0]),
            "decoder": [int(decoder.shape[0]), int(decoder.shape[1])],
            "head": [int(chosen_head.shape[0]), int(chosen_head.shape[1])],
        },
        "split_provenance": split_report,
        "decoder_fit": {
            **ridge_report,
            "prior_source": prior_source,
            "objective": "||X @ D.T - Z||^2 + lambda * ||D - D_prior||^2",
        },
        "chunking": {
            "bounded": True,
            "axis": "vocabulary",
            "vocabulary_chunk_size": chunk_size,
            "vocabulary_chunk_count": chunk_count,
            "materialized_vocabulary_rows": int(h.shape[0]),
            "full_vocabulary_materialized": True,
            "full_vocabulary_evaluated": True,
            "hash_row_chunk_size": hash_rows,
        },
        "evaluation": {
            **evaluation_report,
            "temperature": float(temperature),
            "candidate_count": len(candidate_metrics),
        },
        "candidate_metrics": candidate_metrics,
        "chosen_alpha": chosen_alpha,
        "chosen_metrics": dict(chosen),
        "selection": {
            "held_out_only": True,
            "method": "lexicographic",
            "maximize_in_order": [
                "top1_agreement",
                "teacher_top1_in_child_topk",
                "teacher_margin_sign_retention",
                "negative_kl_teacher_child_mean",
            ],
            "tie_breaker": "smallest_alpha",
        },
        "hashes": {
            "algorithm": "sha256",
            "inputs_before": input_hashes_before,
            "inputs_after": input_hashes_after,
            "input_gradients_before": gradient_hashes_before,
            "input_gradients_after": gradient_hashes_after,
            "inputs_unchanged": True,
            "input_gradients_unchanged": True,
            "prompt_split_sha256": split_report["assignment_sha256"],
            "decoder_sha256": _tensor_sha256(torch, decoder, hash_rows),
            "fitted_head_sha256": _tensor_sha256(
                torch, fitted_head, hash_rows
            ),
            "chosen_head_sha256": _tensor_sha256(
                torch, chosen_head, hash_rows
            ),
            "candidate_metrics_sha256": _json_sha256(candidate_metrics),
        },
    }
    return {
        "weight": chosen_head,
        "fitted_weight": fitted_head,
        "decoder": decoder,
        "report": report,
    }


def _validate_aligned_tokenizers(source_tokenizer, target_tokenizer, texts) -> None:
    source_special = (
        getattr(source_tokenizer, "bos_token_id", None),
        getattr(source_tokenizer, "eos_token_id", None),
        getattr(source_tokenizer, "pad_token_id", None),
    )
    target_special = (
        getattr(target_tokenizer, "bos_token_id", None),
        getattr(target_tokenizer, "eos_token_id", None),
        getattr(target_tokenizer, "pad_token_id", None),
    )
    if source_special != target_special:
        raise ValueError("Donor and child special token ids must match.")
    for index, text in enumerate(texts):
        source_ids = source_tokenizer.encode(str(text), add_special_tokens=True)
        target_ids = target_tokenizer.encode(str(text), add_special_tokens=True)
        if source_ids != target_ids:
            raise ValueError(
                f"Donor and child tokenization differs for calibration prompt {index}."
            )


def _trace_final_norm_prompt_rows(torch, model, batches, prompt_count: int):
    from neural_forge.hf_qwen_compose import (
        _assert_complete_trace,
        _capture_prompt_rows,
        _find_final_norm,
        _model_backbone,
        _run_backbone,
    )

    rows = [None for _ in range(int(prompt_count))]
    context: dict[str, Any] = {}

    def hook(_module, _args, output):
        _capture_prompt_rows(
            torch,
            rows,
            output,
            context["attention_mask"],
            context["prompt_start"],
        )

    handle = _find_final_norm(model).register_forward_hook(hook)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batches:
                context["attention_mask"] = batch["attention_mask"]
                context["prompt_start"] = int(batch["start"])
                _run_backbone(_model_backbone(model), batch["model_inputs"])
    finally:
        handle.remove()
        model.train(was_training)
    _assert_complete_trace(rows, "logit_head_final_norm")
    return rows


def _bounded_aligned_prompt_rows(
    torch,
    child_by_prompt,
    donor_by_prompt,
    prompt_labels: Sequence[str],
    max_rows: int,
):
    child_pieces = []
    donor_pieces = []
    row_prompt_ids = []
    for index, (child_rows, donor_rows) in enumerate(
        zip(child_by_prompt, donor_by_prompt, strict=True)
    ):
        if int(child_rows.shape[0]) != int(donor_rows.shape[0]):
            raise ValueError(f"Prompt {index} donor and child row counts differ.")
        child_pieces.append(child_rows.detach().cpu().to(torch.float32))
        donor_pieces.append(donor_rows.detach().cpu().to(torch.float32))
        row_prompt_ids.extend([str(prompt_labels[index])] * int(child_rows.shape[0]))
    child = torch.cat(child_pieces, dim=0)
    donor = torch.cat(donor_pieces, dim=0)
    original_rows = int(child.shape[0])
    if original_rows > int(max_rows):
        keep = (
            torch.linspace(0, original_rows - 1, steps=int(max_rows))
            .round()
            .to(torch.long)
        )
        child = child.index_select(0, keep)
        donor = donor.index_select(0, keep)
        row_prompt_ids = [row_prompt_ids[int(index)] for index in keep.tolist()]
    return child, donor, row_prompt_ids, {
        "original_rows": original_rows,
        "selected_rows": int(child.shape[0]),
        "selection": "global-evenly-spaced",
        "max_rows": int(max_rows),
    }


def transplant_qwen_logit_head_checkpoint(
    donor_model: str,
    child_model: str,
    output_dir: str | Path,
    *,
    calibration_texts: Sequence[str],
    ridge: float = 1e-3,
    validation_fraction: float = 0.25,
    min_train_prompts: int = 2,
    min_validation_prompts: int = 1,
    split_seed: int = 0,
    candidate_alphas: Sequence[float] | None = None,
    top_k: int = 5,
    temperature: float = 1.0,
    vocab_chunk_size: int = 4096,
    max_length: int = 160,
    batch_size: int = 2,
    max_rows: int = 4096,
    torch_dtype: str = "float16",
    child_torch_dtype: str = "float32",
    device: str = "cuda:0",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Fit, materialize, and save an untied child head from real donor logits."""

    if not calibration_texts:
        raise ValueError("Need at least one calibration prompt.")
    if min(int(max_length), int(batch_size), int(max_rows)) <= 0:
        raise ValueError("max_length, batch_size, and max_rows must be positive.")
    torch = _torch()
    from neural_forge.hf_qwen_compose import _make_trace_batches
    from neural_forge.hf_tools import _dtype_from_name, _torch_and_transformers

    _torch_value, transformers = _torch_and_transformers()
    donor_dtype = _dtype_from_name(torch, torch_dtype)
    child_dtype = _dtype_from_name(torch, child_torch_dtype)
    donor = transformers.AutoModelForCausalLM.from_pretrained(
        donor_model,
        torch_dtype=donor_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    child = transformers.AutoModelForCausalLM.from_pretrained(
        child_model,
        torch_dtype=child_dtype,
        device_map=None,
        trust_remote_code=trust_remote_code,
    )
    source_tokenizer = transformers.AutoTokenizer.from_pretrained(
        donor_model, trust_remote_code=trust_remote_code
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        child_model, trust_remote_code=trust_remote_code
    )
    if source_tokenizer.pad_token_id is None and source_tokenizer.eos_token_id is not None:
        source_tokenizer.pad_token = source_tokenizer.eos_token
    if target_tokenizer.pad_token_id is None and target_tokenizer.eos_token_id is not None:
        target_tokenizer.pad_token = target_tokenizer.eos_token
    texts = [str(item) for item in calibration_texts]
    _validate_aligned_tokenizers(source_tokenizer, target_tokenizer, texts)

    input_embeddings = child.get_input_embeddings()
    output_embeddings = child.get_output_embeddings()
    donor_output_embeddings = donor.get_output_embeddings()
    if any(
        item is None
        for item in (input_embeddings, output_embeddings, donor_output_embeddings)
    ):
        raise ValueError("Donor and child must expose input/output embeddings.")
    if output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr():
        raise ValueError("The child output head must be untied before transplantation.")
    if int(donor_output_embeddings.weight.shape[0]) != int(
        output_embeddings.weight.shape[0]
    ):
        raise ValueError("Donor and child vocabulary capacities must match.")

    batches = _make_trace_batches(
        torch, source_tokenizer, texts, int(batch_size), int(max_length), device
    )
    donor.to(device)
    donor_rows = _trace_final_norm_prompt_rows(torch, donor, batches, len(texts))
    donor_head = donor.get_output_embeddings().weight.detach().clone()
    donor.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    child.to(device)
    child_rows = _trace_final_norm_prompt_rows(torch, child, batches, len(texts))
    current_head = child.get_output_embeddings().weight.detach().clone()
    child.to("cpu")
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    prompt_labels = [
        f"{index}:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
        for index, text in enumerate(texts)
    ]
    child_fit_rows, donor_fit_rows, row_prompt_ids, row_report = (
        _bounded_aligned_prompt_rows(
            torch,
            child_rows,
            donor_rows,
            prompt_labels,
            int(max_rows),
        )
    )
    fit = fit_qwen_logit_head(
        child_fit_rows.to(device),
        donor_fit_rows.to(device),
        donor_head,
        current_head,
        prompt_ids=row_prompt_ids,
        ridge=ridge,
        validation_fraction=validation_fraction,
        min_train_prompts=min_train_prompts,
        min_validation_prompts=min_validation_prompts,
        split_seed=split_seed,
        candidate_alphas=candidate_alphas,
        top_k=top_k,
        temperature=temperature,
        vocab_chunk_size=vocab_chunk_size,
    )
    with torch.no_grad():
        child.get_output_embeddings().weight.copy_(
            fit["weight"].to(
                device=child.get_output_embeddings().weight.device,
                dtype=child.get_output_embeddings().weight.dtype,
            )
        )
    for parameter in child.parameters():
        parameter.grad = None
    if any(parameter.grad is not None for parameter in child.parameters()):
        raise RuntimeError("Head transplantation unexpectedly populated gradients.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    from neural_forge.hf_xfamily import _sanitize_generation_config_for_save

    generation_config_report = _sanitize_generation_config_for_save(
        child, target_tokenizer
    )
    child.save_pretrained(output_path, safe_serialization=True)
    target_tokenizer.save_pretrained(output_path)
    report = {
        **fit["report"],
        "schema": "neural-forge-qwen-full-vocabulary-logit-head-v1",
        "donor_model": str(donor_model),
        "child_model": str(child_model),
        "output_dir": str(output_path),
        "calibration_prompt_count": len(texts),
        "trace_rows": row_report,
        "torch_dtype": str(torch_dtype),
        "child_torch_dtype": str(child_torch_dtype),
        "device": str(device),
        "target_head_mutated": True,
        "target_body_mutated": False,
        "target_parameter_count": sum(
            int(parameter.numel()) for parameter in child.parameters()
        ),
        "generation_config_sanitized": generation_config_report,
    }
    report_path = output_path / "neural_forge_qwen_logit_head_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


fit_full_vocabulary_logit_head = fit_qwen_logit_head
transplant_full_vocabulary_logit_head = fit_qwen_logit_head
