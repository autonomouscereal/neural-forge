"""C2 -- the MLP channel coreset.  62% of a Qwen3.8-27B lives here.

The transaction per site: keep a subset of the SwiGLU's channels VERBATIM (both
``gate_proj`` and ``up_proj`` rows, unchanged, not a rotation of them) and
re-solve ``down_proj`` on the kept channels against the block's own local
target.  Verbatim selection beat low-rank on SwiGLU 40 times out of 42 on LFM2,
which is why there is no SVD arm here at all.

Two campaign laws are ORDERING here and are implemented as such:

* install ``group_size`` sites at a time and re-capture between groups, so each
  group is priced on the real, already-folded base rather than on the parent.
  ONE capture per group, shared by every site in it (``mlp_axis`` captures the
  whole group in a single corpus pass and hands each site its own block); the
  next group is captured against a model that already contains the previous
  one.  ``group_size=1`` degenerates to re-capturing between every site, which
  is the most conditioned and the most expensive schedule, and the schedule
  actually used is written into the axis report;
* ``intermediate_size`` is a single global field in ``Qwen3_5TextConfig``
  (configuration_qwen3_5.py:78), so a config-STANDARD child must end with every
  site at one width.  The three-at-a-time schedule is therefore the order the
  sites are solved in, not a set of different widths -- a parent-shaped
  diagnostic child with mixed widths is possible but is not exportable and this
  module refuses to call one standard.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

import qf_arch
from qf_corpus import fit_check_rows
from qf_harness import pristine_copy
from qf_solve import (
    law_abiding_solve,
    law_abiding_solve_split,
    metric_column_scales,
    pivoted_columns,
    swiglu,
    write_scores,
)

#: The precondition on the identity: the captured design matrix times the
#: parent's own writer must reproduce the parent's own output.  A proposed
#: decision line for Cyril at 5e-2 (bf16 capture), never a gate this file
#: invents -- but a site that fails it is being solved against something that is
#: not its output, so the build says so and stops.
IDENTITY_TOLERANCE = 5e-2


def install_narrow_mlp(
    layer: nn.Module, keep: int, sel: torch.Tensor, writer: torch.Tensor, dtype: torch.dtype
) -> None:
    """Install a narrower SwiGLU: verbatim gate/up rows, solved down."""

    mlp = layer.mlp
    gate = mlp.gate_proj.weight.data
    up = mlp.up_proj.weight.data
    device = gate.device
    index = sel.to(device)
    hidden = gate.shape[1]

    new_gate = nn.Linear(hidden, keep, bias=False, dtype=dtype, device=device)
    new_up = nn.Linear(hidden, keep, bias=False, dtype=dtype, device=device)
    new_down = nn.Linear(keep, hidden, bias=False, dtype=dtype, device=device)
    new_gate.weight.data.copy_(gate[index])
    new_up.weight.data.copy_(up[index])
    new_down.weight.data.copy_(writer.to(device=device, dtype=dtype))
    mlp.gate_proj, mlp.up_proj, mlp.down_proj = new_gate, new_up, new_down
    mlp.intermediate_size = keep


def capture_mlp_group(harness, sites: list[int]) -> dict[str, Any]:
    """One corpus pass that captures every site in a group, with its pristine target.

    This is the group schedule: the sites in ``sites`` are all priced against
    the SAME state of the model, and whatever is installed afterwards is only
    seen by the next group's capture.
    """

    layers = harness.layers()
    pristines = {f"L{i:03d}": pristine_copy(layers[i].mlp) for i in sites}
    captured = harness.capture(
        {f"L{i:03d}": layers[i].mlp for i in sites}, pristine=pristines
    )
    return {
        f"L{i:03d}": {"rows": captured[f"L{i:03d}"], "pristine": pristines[f"L{i:03d}"]}
        for i in sites
    }


def narrow_mlp_site(
    harness,
    layer_index: int,
    keep: int,
    *,
    metric: torch.Tensor | None = None,
    lambda_split: bool = True,
    dtype: torch.dtype | None = None,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fold ONE MLP site, conditioned on the live (partly folded) model.

    The target is the PRISTINE parent block evaluated at the child's own input,
    which is what ``Harness.capture(pristine=...)`` produces and what nothing in
    a recording of the parent's run contains.

    ``prepared`` is one entry of :func:`capture_mlp_group`'s output -- the rows
    and the pristine copy from a capture this site SHARES with the rest of its
    group.  Without it the site takes its own corpus pass, which is the
    ``group_size=1`` schedule.
    """

    dtype = dtype or harness.dtype
    layer = harness.layers()[layer_index]
    name = f"L{layer_index:03d}"
    if prepared is None:
        parent_mlp = pristine_copy(layer.mlp)
        captured = harness.capture({name: layer.mlp}, pristine={name: parent_mlp})[name]
        shared_capture = False
    else:
        parent_mlp = prepared["pristine"]
        captured = prepared["rows"]
        shared_capture = True
    gate = parent_mlp.gate_proj.weight.detach().to("cpu", torch.float64)
    up = parent_mlp.up_proj.weight.detach().to("cpu", torch.float64)
    down = parent_mlp.down_proj.weight.detach().to("cpu", torch.float64)

    phi_fit = swiglu(captured["fit"], gate, up)
    phi_hld = swiglu(captured["hld"], gate, up) if "hld" in captured else None
    Y_fit = captured["local_fit"]
    Y_hld = captured.get("local_hld")

    identity = float((phi_fit @ down.T - Y_fit).norm() / Y_fit.norm())
    if identity > IDENTITY_TOLERANCE:
        raise SystemExit(
            f"{name}: the parent's own writer reproduces its own output to only "
            f"relL2 {identity:.3e} on the captured design matrix (line "
            f"{IDENTITY_TOLERANCE}).  The capture, not the fold, is wrong."
        )

    if metric is None:
        selection_source = phi_fit
        selection_kind = "evidence"
    else:
        scales = metric_column_scales(down, metric)
        selection_source = phi_fit * scales.unsqueeze(0)
        selection_kind = "reader-metric"
    selection_report: dict[str, Any] = {}
    sel = pivoted_columns(selection_source, keep, report=selection_report)
    if int(sel.numel()) != keep:
        raise SystemExit(
            f"{name}: pivoted selection returned {int(sel.numel())} of {keep} channels "
            f"-- the design matrix is rank deficient at this width"
        )

    A = phi_fit[:, sel]
    prior = down[:, sel]
    if lambda_split:
        check = fit_check_rows(harness.manifest)
        if check.numel() != A.shape[0]:
            raise SystemExit(
                f"{name}: internal check mask is {check.numel()} rows, design matrix "
                f"is {A.shape[0]}"
            )
        writer, guard = law_abiding_solve_split(A, Y_fit, check, prior)
    else:
        writer, guard = law_abiding_solve(A, Y_fit, prior)

    scores = {"fit": write_scores(A @ writer.T, Y_fit)}
    if phi_hld is not None and Y_hld is not None:
        scores["held"] = write_scores(phi_hld[:, sel] @ writer.T, Y_hld)
    drop = write_scores(phi_fit[:, sel] @ prior.T, Y_fit)

    install_narrow_mlp(layer, keep, sel, writer, dtype)

    hidden = int(gate.shape[1])
    return {
        "site": name,
        "layer": layer_index,
        "axis": "intermediate_size",
        "parent_intermediate": int(gate.shape[0]),
        "child_intermediate": keep,
        "component_ratio": keep / float(gate.shape[0]),
        "selection": selection_kind,
        "selection_greedy": selection_report,
        "shared_group_capture": bool(shared_capture),
        "selected_channels": [int(i) for i in sel],
        "identity_precondition_relL2": identity,
        "solve": guard,
        "scores": scores,
        "no_repair_scores": drop,
        "parameters_removed": 3 * hidden * (int(gate.shape[0]) - keep),
    }


def mlp_axis(
    harness,
    keep: int,
    *,
    sites: list[int] | None = None,
    group_size: int = 3,
    metric: torch.Tensor | None = None,
    lambda_split: bool = True,
    dtype: torch.dtype | None = None,
    log=print,
) -> dict[str, Any]:
    """Fold every named site to one width, ``group_size`` at a time, re-capturing between.

    Sites default to every layer, because that is what a config-standard child
    requires.  The groups exist so the fourth site is solved against a model
    that already contains the first three -- the conditioned-local arm.  One
    corpus pass per group, and the count of passes is in the report so a
    reviewer can tell which conditioning actually produced the numbers.
    """

    layers = harness.layers()
    sites = list(range(len(layers))) if sites is None else [int(s) for s in sites]
    parent_width = int(layers[0].mlp.gate_proj.weight.shape[0])
    if keep >= parent_width:
        raise SystemExit(f"keep {keep} is not a fold of {parent_width}")

    group_size = max(1, int(group_size))
    reports: list[dict[str, Any]] = []
    groups: list[list[int]] = [
        sites[start : start + group_size] for start in range(0, len(sites), group_size)
    ]
    for group_index, group in enumerate(groups):
        prepared = capture_mlp_group(harness, group)
        for layer_index in group:
            report = narrow_mlp_site(
                harness, layer_index, keep, metric=metric,
                lambda_split=lambda_split, dtype=dtype,
                prepared=prepared[f"L{layer_index:03d}"],
            )
            report["group_index"] = group_index
            report["captured_with"] = list(group)
            reports.append(report)
            log(
                f"[qf] C2 {report['site']}: {report['parent_intermediate']} -> {keep} "
                f"({1.0 / report['component_ratio']:.2f}x)  fit relL2 "
                f"{report['scores']['fit']['relL2']:.4f}  lambda "
                f"{report['solve']['lambda_relative']:g}"
            )

    uniform = len({r["child_intermediate"] for r in reports}) == 1
    covered = sorted(r["layer"] for r in reports)
    complete = covered == list(range(len(layers)))
    if uniform and complete:
        qf_arch.config_field(harness.text_config, "intermediate_size", keep)
    return {
        "axis": "intermediate_size",
        "kind": "channel coreset (verbatim rows) + prior-anchored writer refit",
        "sites": len(reports),
        "keep": keep,
        "parent_intermediate": parent_width,
        "component_ratio": keep / float(parent_width),
        "component_fold": parent_width / float(keep),
        "schedule": {
            "group_size": group_size,
            "groups": [list(g) for g in groups],
            "corpus_passes": len(groups),
            "law": (
                "one capture per group, shared by every site in it; the next group is "
                "captured against a model that already contains the previous one"
            ),
        },
        "config_standard": bool(uniform and complete),
        "config_note": (
            "intermediate_size is one global field; a child that folds only some "
            "sites is a parent-shaped diagnostic and is NOT exportable as standard"
        ),
        "parameters_removed": sum(r["parameters_removed"] for r in reports),
        "per_site": reports,
    }


@torch.no_grad()
def reader_metric(harness, positions: int, topk: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """``M = S W^T (diag(p) - p p^T) W S`` at the final norm's input.

    The first-order cross-entropy cost of a residual perturbation at fixed RMS,
    built forward-only from the model's own head and its own probabilities --
    no Jacobian.  Consumed only as a SELECTION weighting (see
    :func:`qf_solve.metric_column_scales` for why it cancels out of the solve).
    """

    head = harness.lm_head().weight.data.to(torch.float32).to(harness.device)
    norm = harness.final_norm()
    gain = qf_arch.effective_gain(norm).to(torch.float32).to(harness.device)
    eps = harness.eps
    hidden = int(head.shape[1])
    grabbed: dict[str, torch.Tensor] = {}

    def hook(module, inputs):
        grabbed["x"] = inputs[0].detach()[0].to(torch.float32)
        return None

    handle = norm.register_forward_pre_hook(hook)
    acc = torch.zeros(hidden, hidden, dtype=torch.float32, device=harness.device)
    used, mass = 0, 0.0
    try:
        for index, ids in harness.batches:
            if harness.split[index] != "fit":
                continue
            out = harness.model(input_ids=ids.to(harness.device), use_cache=False)
            x = grabbed["x"]
            rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True) + eps)
            probs = torch.softmax(out.logits[0].to(torch.float32), dim=-1)
            for row in range(x.shape[0]):
                if used >= positions:
                    break
                top = torch.topk(probs[row], topk)
                scaled = head[top.indices] * (gain / rms[row]).unsqueeze(0)
                weighted = scaled * top.values.unsqueeze(1)
                mean = weighted.sum(dim=0, keepdim=True)
                acc += scaled.T @ weighted - mean.T @ mean
                mass += float(top.values.sum())
                used += 1
            if used >= positions:
                break
    finally:
        handle.remove()

    M = acc.double().cpu()
    M = 0.5 * (M + M.T)
    M = M * (hidden / float(torch.diagonal(M).sum()))
    eigs = torch.linalg.eigvalsh(M)
    if float(eigs.min()) < -1e-9:
        raise SystemExit(
            f"the reader metric has eigenvalue {float(eigs.min()):.3e}; the softmax "
            "Hessian is PSD and this one is not"
        )
    M = M + max(0.0, -float(eigs.min())) * torch.eye(hidden, dtype=torch.float64)
    return M, {
        "positions_used": used,
        "topk": topk,
        "mean_probability_mass_on_topk": round(mass / max(used, 1), 6),
        "eigenvalue_min": round(float(eigs.min()), 10),
        "eigenvalue_max": round(float(eigs.max()), 6),
    }
