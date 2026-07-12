"""Low-rank clone helpers for materialized linear-weight deltas.

This module intentionally stays small and orchestration-free. It fits a dense
weight update whose delta is constrained through low-rank factors during the
fit, then materializes that delta into a normal ``torch.nn.Linear`` weight.
"""

from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch installed.
        raise RuntimeError("neural_forge.lrc helpers require torch.") from exc
    return torch


def _float_matrix(torch, value, name: str):
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32)
    if tensor.dim() != 2:
        raise ValueError(f"{name} must be a rank-2 tensor.")
    if not torch.isfinite(tensor).all():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor.contiguous()


def _float_bias(torch, value, out_features: int):
    if value is None:
        return None
    tensor = torch.as_tensor(value).detach().cpu().to(torch.float32)
    if tensor.dim() != 1 or int(tensor.shape[0]) != int(out_features):
        raise ValueError("bias must be rank-1 with length matching weight output features.")
    if not torch.isfinite(tensor).all():
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor.contiguous()


def _linear_output(torch, rows, weight, bias):
    output = rows.matmul(weight.t())
    if bias is not None:
        output = output + bias
    return output


def relative_reconstruction_error(prediction, target, *, eps: float = 1e-12) -> float:
    """Return ``||prediction-target|| / max(||target||, eps)`` as a Python float."""

    torch = _require_torch()
    pred = torch.as_tensor(prediction).detach().cpu().to(torch.float32)
    tgt = torch.as_tensor(target).detach().cpu().to(torch.float32)
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have the same shape.")
    error = (pred - tgt).norm()
    denom = tgt.norm().clamp_min(float(eps))
    return float((error / denom).detach().cpu())


def mean_row_cosine(prediction, target, *, eps: float = 1e-12) -> float:
    """Return the mean row-wise cosine similarity as a Python float."""

    torch = _require_torch()
    pred = _float_matrix(torch, prediction, "prediction")
    tgt = _float_matrix(torch, target, "target")
    if pred.shape != tgt.shape:
        raise ValueError("prediction and target must have the same shape.")
    pred_norm = pred.norm(dim=1).clamp_min(float(eps))
    tgt_norm = tgt.norm(dim=1).clamp_min(float(eps))
    cosine = (pred * tgt).sum(dim=1) / (pred_norm * tgt_norm)
    return float(cosine.mean().detach().cpu())


def _validation_gate(
    torch,
    before,
    after,
    target,
    *,
    mode: str,
    min_relative_improvement: float,
    max_cosine_drop: float,
) -> dict[str, Any]:
    before_error = relative_reconstruction_error(before, target)
    after_error = relative_reconstruction_error(after, target)
    before_cosine = mean_row_cosine(before, target)
    after_cosine = mean_row_cosine(after, target)
    relative_improvement = (before_error - after_error) / max(before_error, 1e-12)
    error_improved = after_error < before_error
    accepted = (
        error_improved
        and relative_improvement >= float(min_relative_improvement)
        and after_cosine >= (before_cosine - float(max_cosine_drop))
    )
    reason = "accepted"
    if not error_improved:
        reason = "relative_error_not_improved"
    elif relative_improvement < float(min_relative_improvement):
        reason = "insufficient_relative_error_improvement"
    elif after_cosine < before_cosine - float(max_cosine_drop):
        reason = "cosine_regression"
    return {
        "mode": mode,
        "accepted": bool(accepted),
        "reason": reason,
        "relative_error_before": before_error,
        "relative_error_after": after_error,
        "relative_error_improvement": before_error - after_error,
        "relative_error_improvement_ratio": float(relative_improvement),
        "relative_error_improved": bool(error_improved),
        "min_relative_improvement": float(min_relative_improvement),
        "mean_cosine_before": before_cosine,
        "mean_cosine_after": after_cosine,
        "cosine_drop": before_cosine - after_cosine,
        "max_cosine_drop": float(max_cosine_drop),
    }


def _split_fit_rows(torch, inputs, targets, validation_fraction: float, min_validation_rows: int):
    rows = int(inputs.shape[0])
    if rows <= 1 or validation_fraction <= 0.0:
        return None
    validation_rows = max(int(min_validation_rows), int(round(rows * float(validation_fraction))))
    if validation_rows >= rows:
        return None
    validation_rows = min(validation_rows, rows - 1)
    if validation_rows < int(min_validation_rows):
        return None
    train_rows = rows - validation_rows
    if train_rows <= 0:
        return None
    return {
        "rows": rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "train_inputs": inputs[:train_rows],
        "train_targets": targets[:train_rows],
        "validation_inputs": inputs[train_rows:],
        "validation_targets": targets[train_rows:],
    }


def _subsample_rows(torch, inputs, targets, max_rows: int | None):
    if max_rows is None or int(max_rows) <= 0 or int(inputs.shape[0]) <= int(max_rows):
        return inputs, targets, False
    indices = torch.linspace(0, int(inputs.shape[0]) - 1, int(max_rows)).round().to(torch.long)
    return inputs.index_select(0, indices), targets.index_select(0, indices), True


def _ridge_delta_solution(torch, inputs, residual, ridge: float):
    in_features = int(inputs.shape[1])
    gram = inputs.t().matmul(inputs)
    if float(ridge) > 0.0:
        gram = gram + float(ridge) * torch.eye(in_features, dtype=inputs.dtype)
    rhs = inputs.t().matmul(residual)
    try:
        mapping = torch.linalg.solve(gram, rhs)
        solver = "ridge_solve"
    except RuntimeError:
        mapping = torch.linalg.lstsq(gram, rhs).solution
        solver = "ridge_lstsq_fallback"
    return mapping.t().contiguous(), solver


def _svd_low_rank_factors(torch, delta, rank: int):
    full_rank = min(int(delta.shape[0]), int(delta.shape[1]))
    used_rank = max(1, min(int(rank), full_rank))
    try:
        u, singular_values, vh = torch.linalg.svd(delta, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError("Unable to compute low-rank delta factors with SVD.") from exc
    factors_sqrt = singular_values[:used_rank].clamp_min(0.0).sqrt()
    left = u[:, :used_rank] * factors_sqrt.unsqueeze(0)
    right = factors_sqrt.unsqueeze(1) * vh[:used_rank, :]
    truncated = left.matmul(right).contiguous()
    total_energy = singular_values.pow(2).sum().clamp_min(1e-12)
    kept_energy = singular_values[:used_rank].pow(2).sum()
    return left.contiguous(), right.contiguous(), truncated, {
        "requested_rank": int(rank),
        "effective_rank": int(used_rank),
        "full_rank": int(full_rank),
        "rank_energy_ratio": float((kept_energy / total_energy).detach().cpu()),
        "singular_value_count": int(singular_values.numel()),
    }


def _refine_low_rank_delta(
    torch,
    base_weight,
    bias,
    train_inputs,
    train_targets,
    initial_left,
    initial_right,
    *,
    alpha: float,
    steps: int,
    lr: float,
    l2: float,
):
    if int(steps) <= 0:
        delta = initial_left.matmul(initial_right).detach().contiguous()
        initial_loss = torch.mean(
            (_linear_output(torch, train_inputs, base_weight + float(alpha) * delta, bias) - train_targets) ** 2
        )
        return delta, {
            "refined": False,
            "steps": 0,
            "initial_loss": float(initial_loss.detach().cpu()),
            "final_loss": None,
        }

    left = initial_left.clone().detach().requires_grad_(True)
    right = initial_right.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([left, right], lr=float(lr))
    best_delta = initial_left.matmul(initial_right).detach().contiguous()
    best_loss = float(
        torch.mean(
            (_linear_output(torch, train_inputs, base_weight + float(alpha) * best_delta, bias) - train_targets) ** 2
        )
        .detach()
        .cpu()
    )
    initial_loss = best_loss
    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        delta = left.matmul(right)
        prediction = _linear_output(torch, train_inputs, base_weight + float(alpha) * delta, bias)
        loss = torch.mean((prediction - train_targets) ** 2)
        if float(l2) > 0.0:
            loss = loss + float(l2) * (left.pow(2).mean() + right.pow(2).mean())
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_delta = left.detach().matmul(right.detach()).contiguous()
    return best_delta, {
        "refined": True,
        "steps": int(steps),
        "lr": float(lr),
        "l2": float(l2),
        "initial_loss": float(initial_loss),
        "final_loss": float(best_loss),
    }


def fit_materialized_linear_lrc_delta(
    weight,
    child_input_rows,
    donor_projected_output_rows,
    *,
    bias=None,
    rank: int = 8,
    alpha: float = 1.0,
    ridge: float = 1e-4,
    factor_steps: int = 80,
    factor_lr: float = 0.03,
    factor_l2: float = 0.0,
    validation_fraction: float = 0.2,
    min_validation_rows: int = 8,
    min_relative_improvement: float = 0.0,
    max_cosine_drop: float = 0.02,
    require_validation: bool = True,
    max_rows: int | None = 4096,
    max_delta_rms_ratio: float = 0.0,
):
    """Fit a gated materialized low-rank delta for a Linear weight.

    ``child_input_rows`` are rows entering the target child ``Linear``.
    ``donor_projected_output_rows`` are donor outputs already projected into
    the child Linear's output dimension. The returned weight is a normal dense
    tensor; no adapter or factor module is left behind.
    """

    torch = _require_torch()
    original_weight = torch.as_tensor(weight)
    original_device = original_weight.device
    original_dtype = original_weight.dtype
    base_weight = _float_matrix(torch, original_weight, "weight")
    inputs = _float_matrix(torch, child_input_rows, "child_input_rows")
    targets = _float_matrix(torch, donor_projected_output_rows, "donor_projected_output_rows")
    out_features, in_features = int(base_weight.shape[0]), int(base_weight.shape[1])
    bias_cpu = _float_bias(torch, bias, out_features)

    if int(rank) <= 0:
        raise ValueError("rank must be positive.")
    if float(alpha) <= 0.0:
        raise ValueError("alpha must be positive.")
    if float(ridge) < 0.0:
        raise ValueError("ridge must be non-negative.")
    if int(factor_steps) < 0:
        raise ValueError("factor_steps must be non-negative.")
    if int(factor_steps) > 0 and float(factor_lr) <= 0.0:
        raise ValueError("factor_lr must be positive when factor_steps is positive.")
    if float(max_delta_rms_ratio) < 0.0:
        raise ValueError("max_delta_rms_ratio must be non-negative.")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("child_input_rows and donor_projected_output_rows must have the same row count.")
    if int(inputs.shape[1]) != in_features:
        raise ValueError("child_input_rows feature size must match weight input features.")
    if int(targets.shape[1]) != out_features:
        raise ValueError("donor_projected_output_rows feature size must match weight output features.")

    inputs, targets, subsampled = _subsample_rows(torch, inputs, targets, max_rows)
    split = _split_fit_rows(torch, inputs, targets, validation_fraction, min_validation_rows)
    report: dict[str, Any] = {
        "operation": "fit_materialized_linear_lrc_delta",
        "accepted": False,
        "reason": "not_evaluated",
        "weight_shape": [out_features, in_features],
        "input_rows": int(inputs.shape[0]),
        "subsampled": bool(subsampled),
        "max_rows": None if max_rows is None else int(max_rows),
        "alpha": float(alpha),
        "ridge": float(ridge),
        "has_bias": bias_cpu is not None,
    }

    if split is None:
        if require_validation:
            report.update(
                {
                    "reason": "insufficient_validation_rows",
                    "fit_gate": {
                        "mode": "validation",
                        "accepted": False,
                        "reason": "insufficient_validation_rows",
                        "rows": int(inputs.shape[0]),
                        "min_validation_rows": int(min_validation_rows),
                    },
                    "mutation_skipped": True,
                }
            )
            return original_weight.detach().clone(), report
        train_inputs = inputs
        train_targets = targets
        validation_inputs = inputs
        validation_targets = targets
        gate_mode = "train"
        report.update({"train_rows": int(inputs.shape[0]), "validation_rows": 0})
    else:
        train_inputs = split["train_inputs"]
        train_targets = split["train_targets"]
        validation_inputs = split["validation_inputs"]
        validation_targets = split["validation_targets"]
        gate_mode = "validation"
        report.update(
            {
                "train_rows": int(split["train_rows"]),
                "validation_rows": int(split["validation_rows"]),
                "validation_fraction": float(validation_fraction),
                "min_validation_rows": int(min_validation_rows),
            }
        )

    before_train = _linear_output(torch, train_inputs, base_weight, bias_cpu)
    residual = train_targets - before_train
    full_delta, solver = _ridge_delta_solution(torch, train_inputs, residual, ridge)
    left, right, truncated_delta, rank_report = _svd_low_rank_factors(torch, full_delta, int(rank))
    refined_delta, refine_report = _refine_low_rank_delta(
        torch,
        base_weight,
        bias_cpu,
        train_inputs,
        train_targets,
        left,
        right,
        alpha=float(alpha),
        steps=int(factor_steps),
        lr=float(factor_lr),
        l2=float(factor_l2),
    )
    candidate_delta = refined_delta if bool(refine_report.get("refined")) else truncated_delta
    materialized_delta = float(alpha) * candidate_delta
    base_rms = base_weight.pow(2).mean().sqrt().clamp_min(1e-12)
    delta_rms_before_clip = materialized_delta.pow(2).mean().sqrt()
    delta_rms_ratio_before_clip = delta_rms_before_clip / base_rms
    delta_clip_scale = 1.0
    if float(max_delta_rms_ratio) > 0.0 and float(delta_rms_ratio_before_clip.detach().cpu()) > float(max_delta_rms_ratio):
        delta_clip_scale = float(max_delta_rms_ratio) / max(float(delta_rms_ratio_before_clip.detach().cpu()), 1e-12)
        candidate_delta = candidate_delta * float(delta_clip_scale)
        materialized_delta = float(alpha) * candidate_delta
    candidate_weight = (base_weight + float(alpha) * candidate_delta).contiguous()
    delta_rms_after_clip = materialized_delta.pow(2).mean().sqrt()
    delta_rms_ratio_after_clip = delta_rms_after_clip / base_rms

    train_after = _linear_output(torch, train_inputs, candidate_weight, bias_cpu)
    validation_before = _linear_output(torch, validation_inputs, base_weight, bias_cpu)
    validation_after = _linear_output(torch, validation_inputs, candidate_weight, bias_cpu)
    gate = _validation_gate(
        torch,
        validation_before,
        validation_after,
        validation_targets,
        mode=gate_mode,
        min_relative_improvement=float(min_relative_improvement),
        max_cosine_drop=float(max_cosine_drop),
    )
    accepted = bool(gate["accepted"])
    selected_weight = candidate_weight if accepted else base_weight
    report.update(
        {
            "accepted": accepted,
            "reason": gate["reason"],
            "mutation_skipped": not accepted,
            "solver": solver,
            "rank": rank_report,
            "factor_refinement": refine_report,
            "fit_gate": gate,
            "relative_error_train_before": relative_reconstruction_error(before_train, train_targets),
            "relative_error_train_after": relative_reconstruction_error(train_after, train_targets),
            "mean_cosine_train_before": mean_row_cosine(before_train, train_targets),
            "mean_cosine_train_after": mean_row_cosine(train_after, train_targets),
            "full_delta_norm": float(full_delta.norm().detach().cpu()),
            "truncated_delta_norm": float(truncated_delta.norm().detach().cpu()),
            "materialized_delta_norm": float(materialized_delta.norm().detach().cpu()),
            "base_weight_rms": float(base_rms.detach().cpu()),
            "materialized_delta_rms_before_clip": float(delta_rms_before_clip.detach().cpu()),
            "materialized_delta_rms_ratio_before_clip": float(delta_rms_ratio_before_clip.detach().cpu()),
            "materialized_delta_rms_after_clip": float(delta_rms_after_clip.detach().cpu()),
            "materialized_delta_rms_ratio_after_clip": float(delta_rms_ratio_after_clip.detach().cpu()),
            "max_delta_rms_ratio": float(max_delta_rms_ratio),
            "delta_rms_clipped": bool(float(delta_clip_scale) < 1.0),
            "delta_rms_clip_scale": float(delta_clip_scale),
        }
    )
    return selected_weight.to(device=original_device, dtype=original_dtype), report


def fit_linear_lrc_delta(
    linear,
    child_input_rows,
    donor_projected_output_rows,
    **kwargs,
) -> dict[str, Any]:
    """Fit and conditionally apply a materialized LRC delta to ``torch.nn.Linear``."""

    torch = _require_torch()
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError("linear must be an instance of torch.nn.Linear.")
    updated_weight, report = fit_materialized_linear_lrc_delta(
        linear.weight.detach(),
        child_input_rows,
        donor_projected_output_rows,
        bias=None if linear.bias is None else linear.bias.detach(),
        **kwargs,
    )
    if report.get("accepted"):
        with torch.no_grad():
            linear.weight.copy_(updated_weight.to(device=linear.weight.device, dtype=linear.weight.dtype))
    return report


__all__ = [
    "fit_linear_lrc_delta",
    "fit_materialized_linear_lrc_delta",
    "mean_row_cosine",
    "relative_reconstruction_error",
]
