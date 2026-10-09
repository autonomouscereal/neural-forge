"""Bounded FIT-only native source-unit selection and full-output affine ridge.

No full expert-union feature tensor or Gram is formed. Chunked first-layer
features may differ in F16 GEMM grouping from source expert dispatch; the child
is explicitly approximate, and real complete-context diagnostics measure it.
"""
from __future__ import annotations
import math
import torch
from hy3d_full21_contract import require
from hy3d_full21_linalg import finite
from hy3d_dense_child_contract import memory_plan


def check_fit(inputs, targets, branches, keep, policy, route_masses=None):
    require(isinstance(inputs, list) and inputs and len(inputs) == len(targets), "FIT-only paired contexts required")
    width = inputs[0].shape[-1]
    require(all(x.ndim == y.ndim == 2 and x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1] == width and
                x.dtype == y.dtype == torch.float16 and x.device.type == y.device.type == "cpu" for x, y in zip(inputs, targets)),
            "bounded sealed F16 CPU FIT inputs/whole native outputs required")
    require(branches and all(set(b) == {"id", "first.weight", "first.bias", "second.weight", "second.bias"} for b in branches),
            "complete explicitly identified source branches required")
    widths = []
    for b in branches:
        first, bias, second, outbias = (b[n] for n in ("first.weight", "first.bias", "second.weight", "second.bias"))
        inner = first.shape[0]
        require(first.shape == (inner, width) and bias.shape == (inner,) and second.shape == (width, inner) and
                outbias.shape == (width,) and all(t.dtype == torch.float16 and t.device.type == "cpu"
                for t in (first, bias, second, outbias)), "native branch shape/storage mismatch")
        for name, t in b.items():
            if name != "id": finite(t, "source " + name)
        widths.append(inner)
    require(len({b["id"] for b in branches}) == len(branches), "duplicate source branch identity")
    require(type(keep) is int and 0 < keep <= sum(widths), "selected source-unit count out of range")
    plan = memory_plan(width, keep, policy["row_chunk"], max(widths))
    require(plan["working_bytes"] <= policy["max_working_bytes"], "selected-width memory admission exceeded")
    if route_masses is None:
        require(len(branches) == 1, "MoE source requires actual FIT route masses and shared mass")
        route_masses = [torch.ones(len(x), 1, dtype=torch.float16) for x in inputs]
    require(len(route_masses) == len(inputs) and all(m.shape == (len(x), len(branches)) and m.dtype == torch.float16 and
            m.device.type == "cpu" and bool(((m >= 0) & (m <= 1)).all()) for m, x in zip(route_masses, inputs)),
            "native FIT branch masses required")
    for x, y, m in zip(inputs, targets, route_masses):
        finite(x, "FIT input"); finite(y, "full native FIT target"); finite(m, "FIT native mass")
    return width, widths, route_masses, plan


@torch.inference_mode()
def fit_dense(inputs, targets, branches, keep, policy, route_masses=None, splits=None):
    require(splits == ["fit"] * len(inputs), "only explicitly FIT-labelled contexts may enter selection or solve")
    width, widths, masses, plan = check_fit(inputs, targets, branches, keep, policy, route_masses)
    row_chunk = policy["row_chunk"]
    energies, mean_masses, coverage = [], [], []
    # Source-unit contribution energy is scored one branch/chunk at a time.
    # Gate masses are captured/replayed native FIT values, never HELD values.
    for branch_index, branch in enumerate(branches):
        energy = torch.zeros(widths[branch_index], dtype=torch.float64)
        mass_sum = 0.0; routed_rows = 0; rows = 0
        readout_norm = torch.zeros(widths[branch_index], dtype=torch.float64)
        for first in range(0, widths[branch_index], row_chunk):
            value = branch["second.weight"][:, first:first + row_chunk].double()
            readout_norm[first:first + len(value.T)] = value.square().sum(0)
            del value
        for x, m in zip(inputs, masses):
            for start in range(0, len(x), row_chunk):
                chunk, mass = x[start:start + row_chunk], m[start:start + row_chunk, branch_index].double()
                hidden = torch.nn.functional.gelu(torch.nn.functional.linear(chunk, branch["first.weight"], branch["first.bias"]), approximate="none")
                finite(hidden, "FIT source-unit features")
                energy.add_((hidden.double().square() * mass.square()[:, None]).sum(0) * readout_norm)
                mass_sum += float(mass.sum()); routed_rows += int((mass > 0).sum()); rows += len(chunk)
        finite(energy, "FIT source-unit contribution energy")
        energies.append(energy); mean_masses.append(mass_sum / rows)
        coverage.append({"branch_id": branch["id"], "fit_rows": rows, "positive_mass_rows": routed_rows,
                         "fit_mean_native_mass": mass_sum / rows, "unvisited": routed_rows == 0})
    all_energy = torch.cat(energies)
    selected = torch.argsort(all_energy, descending=True, stable=True)[:keep].sort().values
    require(selected.unique().numel() == keep, "duplicate selected source units")
    first_weights = []; first_biases = []; prior_columns = []
    offset = 0
    for i, branch in enumerate(branches):
        local = selected[(selected >= offset) & (selected < offset + widths[i])] - offset
        if local.numel():
            first_weights.append(branch["first.weight"][local]); first_biases.append(branch["first.bias"][local])
            prior_columns.append(branch["second.weight"][:, local].double() * mean_masses[i])
        offset += widths[i]
    first_weight, first_bias = torch.cat(first_weights), torch.cat(first_biases)
    bias = sum((b["second.bias"].double() * mean_masses[i] for i, b in enumerate(branches)), torch.zeros(width, dtype=torch.float64))
    prior = torch.cat((torch.cat(prior_columns, dim=1), bias[:, None]), dim=1)
    k = keep + 1
    gram = torch.zeros(k, k, dtype=torch.float64); cross = torch.zeros(k, width, dtype=torch.float64)
    target_energy = 0.0
    def design_chunks():
        for x, y in zip(inputs, targets):
            for start in range(0, len(x), row_chunk):
                hidden = torch.nn.functional.gelu(torch.nn.functional.linear(x[start:start + row_chunk], first_weight, first_bias), approximate="none")
                finite(hidden, "selected FIT dense features")
                h = torch.cat((hidden.double(), torch.ones(len(hidden), 1, dtype=torch.float64)), dim=1)
                yield h, y[start:start + len(h)].double()
    for h, y in design_chunks():
        gram.add_(h.T @ h); cross.add_(h.T @ y); target_energy += float(y.square().sum())
    finite(gram, "selected Gram"); finite(cross, "full-output cross")
    scale = float(gram.diagonal().mean())
    require(math.isfinite(scale) and scale > 0, "invalid FIT statistics scale")
    best = None; best_error = math.inf; ladder = []
    for rung in policy["ridge_ladder"]:
        lam = scale * rung
        shifted = gram.clone(); shifted.diagonal().add_(lam)
        solution = torch.linalg.solve(shifted, cross + lam * prior.T).T.contiguous()
        finite(solution, "dense child affine solution")
        residual = sum(float((h @ solution.T - y).square().sum()) for h, y in design_chunks())
        error = math.sqrt(residual / max(target_energy, 1e-30))
        require(math.isfinite(error), "nonfinite FIT residual")
        update = float((solution - prior).norm() / prior.norm().clamp_min(1e-30))
        require(math.isfinite(update), "nonfinite update diagnostic")
        ladder.append({"ridge_rel": rung, "fit_relative_l2": error, "update_scale": update, "numeric_veto": False})
        if error < best_error: best = solution; best_error = error
        del shifted, solution
    require(best is not None, "missing finite ridge solution")
    stored = best.to(torch.float16); finite(stored, "stored F16 dense child")
    state = {"fc1.weight": first_weight.contiguous(), "fc1.bias": first_bias.contiguous(),
             "fc2.weight": stored[:, :-1].contiguous(), "fc2.bias": stored[:, -1].contiguous()}
    return state, selected, {"memory_plan": plan, "coverage": coverage, "fit_rows": sum(len(x) for x in inputs),
            "scratch_cap_scope": "temporary selected solve/chunks only; not container RAM",
            "persistent_source_basis_bytes": sum(t.numel() * t.element_size() for b in branches for key, t in b.items() if key != "id"),
            "selection": policy["selection"], "selected_zero_energy_units": int((all_energy[selected] == 0).sum()),
            "ridge_ladder": ladder, "fit_relative_l2": best_error, "held_used_for_selection_or_refit": False,
            "teacher_target": "full native FFN output including native routing masses, output biases and shared branch",
            "source_unit_feature_grouping": "bounded CPU F16 chunks; explicit approximate child grouping",
            "union_feature_matrix_allocated": False, "union_gram_allocated": False, "numeric_veto": False}
