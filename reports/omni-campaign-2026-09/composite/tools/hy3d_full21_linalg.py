"""Bounded, FIT-only closed form. No feature-union Gram and no numeric veto.

Selection differs explicitly from the historical full-width pivoted Gram.
Use only after policy/source/grouping review; unit tests establish mechanics.
"""
from __future__ import annotations

import math
import torch
from hy3d_full21_contract import require


def finite(value, name):
    for chunk in value.reshape(-1).split(1 << 20):
        require(bool(torch.isfinite(chunk).all()), "nonfinite " + name)


def stable_energy_selection(features, keep, row_chunk=256):
    require(features.ndim == 2 and features.dtype == torch.float16 and features.device.type == "cpu",
            "sealed native F16 CPU features required")
    require(0 < keep <= features.shape[1] and row_chunk > 0, "bounded channel selection geometry required")
    energy = torch.zeros(features.shape[1], dtype=torch.float64)
    for row in features.split(row_chunk):
        values = row.double()
        finite(values, "FIT features")
        energy.add_(values.square().sum(0))
    finite(energy, "FIT feature energy")
    # Stable ties preserve original channel order. Zero-energy columns remain
    # legitimate source channels; low rank is a diagnostic, not a discard gate.
    selected = torch.argsort(energy, descending=True, stable=True)[:keep].sort().values
    return selected, {"selection": "fit_only_uncentered_native_feature_energy_stable_index_ties",
                      "nonzero_feature_columns": int((energy > 0).sum()),
                      "selected_zero_energy_columns": int((energy[selected] == 0).sum()),
                      "rank_veto": False, "union_gram_allocated": False}


def sufficient_statistics(features, target, selected, row_chunk, max_bytes):
    require(features.ndim == target.ndim == 2 and features.shape[0] == target.shape[0], "FIT feature/target shape mismatch")
    width, output = selected.numel() + 1, target.shape[1]
    # Account live Gram/cross/anchor/solution/factor/shifted copies conservatively.
    planned = 8 * (4 * width * width + 4 * width * output + output)
    require(planned <= max_bytes, f"selected-statistics bound insufficient: need {planned}, cap {max_bytes}")
    gram = torch.zeros(width, width, dtype=torch.float64)
    cross = torch.zeros(width, output, dtype=torch.float64)
    target_square = torch.zeros(output, dtype=torch.float64)
    for start in range(0, len(features), row_chunk):
        x = features[start:start + row_chunk, selected].double()
        x = torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), dim=1)
        y = target[start:start + row_chunk].double()
        finite(x, "FIT design")
        finite(y, "FIT target")
        gram.add_(x.T @ x)
        cross.add_(x.T @ y)
        target_square.add_(y.square().sum(0))
    for name, value in (("FIT Gram", gram), ("FIT cross", cross), ("FIT target energy", target_square)):
        finite(value, name)
    return gram, cross, target_square, {"selected_statistics_plan_bytes": planned,
            "selected_gram_shape": list(gram.shape), "union_gram_allocated": False,
            "row_chunk": row_chunk, "fit_rows": len(features)}


def closed_form_fit(features, target, selected, prior, policy):
    gram, cross, target_square, notes = sufficient_statistics(features, target, selected,
                              policy["row_chunk"], policy["max_statistics_bytes"])
    require(prior.shape == (target.shape[1], selected.numel() + 1), "native affine prior mismatch")
    anchor = prior.double()
    finite(anchor, "native affine prior")
    diagonal_scale = float(gram.diagonal().mean())
    # A wholly zero input still includes the affine ones column, so scale > 0.
    require(math.isfinite(diagonal_scale) and diagonal_scale > 0, "invalid FIT statistics scale")
    best, best_error, trail = None, math.inf, []
    for rung in policy["ridge_ladder"]:
        lam = float(rung) * diagonal_scale
        shifted = gram.clone()
        shifted.diagonal().add_(lam)
        anchored = cross + lam * anchor.T
        solution = torch.linalg.solve(shifted, anchored).T.contiguous()
        finite(solution, "FP64 affine ridge solution")
        # Select on direct streamed FIT residuals, avoiding cancellation from
        # subtracting quadratic sufficient statistics on almost-exact fits.
        residual_square = 0.0
        for start in range(0, len(features), policy["row_chunk"]):
            x = features[start:start + policy["row_chunk"], selected].double()
            x = torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), dim=1)
            residual = x @ solution.T - target[start:start + len(x)].double()
            finite(residual, "FIT residual")
            residual_square += float(residual.square().sum())
        error = math.sqrt(residual_square / max(float(target_square.sum()), 1e-30))
        update = float((solution - anchor).norm() / anchor.norm().clamp_min(1e-30))
        require(math.isfinite(error) and math.isfinite(update), "nonfinite FIT solve diagnostic")
        trail.append({"ridge_rel": float(rung), "ridge_lambda": lam,
                      "fit_relative_l2": error, "update_scale": update, "vetoed": False})
        if error < best_error:
            best, best_error = solution, error
            notes.update(ridge_rel=float(rung), ridge_lambda=lam, update_scale=update)
        del shifted, anchored
    require(best is not None, "no finite solve candidate")
    # Nonfinite F16 storage is a functional safeguard; finite large updates are
    # kept with their measurements and never replaced with the prior by policy.
    stored = best.to(torch.float16)
    finite(stored, "stored F16 affine readout")
    notes.update(ladder=trail, fit_relative_l2=best_error,
                 stored_update_scale=float((stored.double() - anchor).norm() / anchor.norm().clamp_min(1e-30)),
                 numeric_veto=False, held_used_for_selection_or_refit=False,
                 low_rank_diagnostic="nonzero energy count; no full-width eigendecomposition",
                 solver="selected-width streamed FP64 prior-anchored ridge; FIT-only direct residual ranking")
    return stored, notes


def scores(actual, reference, row_chunk=256):
    require(actual.shape == reference.shape and actual.ndim == 2 and len(actual) > 0, "score shape mismatch")
    delta_square = ref_square = max_abs = cosine_sum = 0.0
    cosines = []
    for start in range(0, len(actual), row_chunk):
        a, b = actual[start:start + row_chunk].double(), reference[start:start + row_chunk].double()
        finite(a, "functional candidate output")
        finite(b, "functional source output")
        delta = a - b
        delta_square += float(delta.square().sum())
        ref_square += float(b.square().sum())
        max_abs = max(max_abs, float(delta.abs().max()))
        cosine = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)
        finite(cosine, "finite cosine diagnostics")
        cosine_sum += float(cosine.sum())
        cosines.append(cosine)
    order = torch.cat(cosines).sort().values
    return {"relative_l2": math.sqrt(delta_square / max(ref_square, 1e-30)),
            "rmse": math.sqrt(delta_square / actual.numel()), "max_abs": max_abs,
            "mean_cosine": cosine_sum / len(actual), "cosine_p05": float(order[int(.05 * (len(order) - 1))]),
            "rows": len(actual), "width": actual.shape[1], "numeric_veto": False}
