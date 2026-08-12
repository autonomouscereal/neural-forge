#!/usr/bin/env python3
"""Wave-14 battery, third pass: the matched-storage control, and the L05 anomaly.

Two things pass two got wrong, both worth fixing before a cell is spent.

**1. The ledger comparison was not like-for-like.** `post_operator_heldout_relL2`
normalizes by the *endpoint* (residual carry included); pass two normalized by the
*write*.  Rather than argue about normalizations, this pass builds the control
**inside the same harness**: at every site and level, the verbatim channel coreset
is run against an evidence-weighted low-rank approximation of the same map at
**identical stored-parameter count**, predicting the same target from the same
features on the same held-out rows.  One variable: verbatim channel subset vs
low-rank basis.  That is the question the campaign has been asking since wave 5,
and this answers it per site for the cost of one SVD.

**2. `OFF_EPSILON` is not scale-robust.** L05 reported 1,915 certified-OFF units
while every other site reported zero, and L05 is also where the S+L screen found
kurtosis 2.5e6 with 98% of the entry energy in the top 1%.  A share-of-total
threshold cannot tell "this unit is dead" from "one other unit is enormous".  The
diagnostic here separates the two, and the OFF certificate is **withdrawn as a
free-drop licence** unless it survives the separation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mx14_measure import ATTN_LAYERS, CONV_LAYERS, MLP_LAYERS, RIDGE_REL, have, rows, topk_indices, weight
from mx14_measure2 import BAND, arm_sets, prior_anchored, write_scores

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

SCHEMA = "neural-forge-matrix-wave14-matched-control-v1"


def weighted_lra(F: torch.Tensor, W: torch.Tensor, rank: int) -> torch.Tensor:
    """Exact one-sided evidence-weighted rank-r approximation of ``W`` under ``F``.

    ``argmin_rank(W') ||F W'^T - F W^T||_F`` -- one eigh of the Gram and one SVD.
    """
    G = F.T @ F / F.shape[0]
    ev, evec = torch.linalg.eigh(G)
    Rt = evec @ torch.diag(ev.clamp_min(0).sqrt()) @ evec.T
    Rt_inv = evec @ torch.diag(1.0 / ev.clamp_min(1e-14).sqrt()) @ evec.T
    U, S, Vh = torch.linalg.svd(W @ Rt, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vh[:rank] @ Rt_inv.T


def control_site(
    F: torch.Tensor,
    F_h: torch.Tensor,
    W: torch.Tensor,
    Y: torch.Tensor,
    Y_h: torch.Tensor,
    keepn: int,
) -> dict:
    """Coreset vs matched-storage LRA on one map."""
    out_dim, in_dim = W.shape
    stored = out_dim * keepn  # what the coreset's re-solved writer stores
    rank = stored // (out_dim + in_dim)
    sets = arm_sets(F, F_h, W, keepn)
    sets.pop("_off_count", None)
    best = None
    for arm, keep in sets.items():
        Wp = prior_anchored(F[:, keep], Y, W[:, keep])
        sc = write_scores(F_h[:, keep] @ Wp.T, Y_h)
        sc["arm"] = arm
        if best is None or sc["relL2"] < best["relL2"]:
            best = sc
    L = weighted_lra(F, W, rank)
    lra = write_scores(F_h @ L.T, Y_h)
    return {
        "keep": keepn,
        "coreset_stored": stored,
        "lra_rank": rank,
        "lra_stored": rank * (out_dim + in_dim),
        "coreset": best,
        "lra": lra,
        "coreset_wins_relL2": bool(best["relL2"] <= lra["relL2"]),
        "coreset_wins_cos": bool(best["cos"] >= lra["cos"]),
        "coreset_in_band": bool(best["relL2"] <= BAND),
        "lra_in_band": bool(lra["relL2"] <= BAND),
    }


def task_control_mlp() -> dict:
    out = []
    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        if not have(name):
            continue
        F = rows(name, "mlp.post_swiglu", "fit")
        F_h = rows(name, "mlp.post_swiglu", "hld")
        W = weight(f"model.layers.{layer}.feed_forward.w2.weight")
        Y = rows(name, "mlp.down", "fit")
        Y_h = rows(name, "mlp.down", "hld")
        entry = {"layer": layer, "levels": {}}
        for level, keepn in (("S4", 640), ("S8", 320), ("S16", 160)):
            entry["levels"][level] = control_site(F, F_h, W, Y, Y_h, keepn)
        out.append(entry)
        line = f"  L{layer:02d}"
        for level in ("S4", "S8", "S16"):
            e = entry["levels"][level]
            line += (
                f" | {level} core {e['coreset']['cos']:.4f}/{e['coreset']['relL2']:.4f}"
                f" lra {e['lra']['cos']:.4f}/{e['lra']['relL2']:.4f}"
                f" {'CORE' if e['coreset_wins_cos'] else 'lra '}"
            )
        print(line, flush=True)
    return {"task": "control_mlp", "component": "C13", "sites": out}


def task_control_conv() -> dict:
    out = []
    for layer in CONV_LAYERS:
        name = f"L{layer:02d}-C05"
        if not have(name):
            continue
        F = rows(name, "short_conv.context", "fit")
        F_h = rows(name, "short_conv.context", "hld")
        W = weight(f"model.layers.{layer}.conv.out_proj.weight")
        Y = F @ W.T
        Y_h = F_h @ W.T
        entry = {"layer": layer, "levels": {}}
        for level, keepn in (("S4", 256), ("S8", 128), ("S16", 64)):
            entry["levels"][level] = control_site(F, F_h, W, Y, Y_h, keepn)
        out.append(entry)
        line = f"  L{layer:02d}"
        for level in ("S4", "S8", "S16"):
            e = entry["levels"][level]
            line += (
                f" | {level} core {e['coreset']['cos']:.4f} lra {e['lra']['cos']:.4f}"
                f" {'CORE' if e['coreset_wins_cos'] else 'lra '}"
            )
        print(line, flush=True)
    return {"task": "control_conv", "component": "C05", "sites": out}


def task_l05_anomaly() -> dict:
    """Is L05's SwiGLU sparse, or is one channel enormous?"""
    out = []
    for layer in (4, 5, 6, 13):
        name = f"L{layer:02d}-C13"
        phi = rows(name, "mlp.post_swiglu", "fit")
        w2 = weight(f"model.layers.{layer}.feed_forward.w2.weight")
        w1 = weight(f"model.layers.{layer}.feed_forward.w1.weight")
        w3 = weight(f"model.layers.{layer}.feed_forward.w3.weight")
        energy = phi.pow(2).mean(dim=0) * w2.norm(dim=0).pow(2)
        srt = torch.sort(energy, descending=True).values
        act = phi.abs().mean(dim=0)
        entry = {
            "layer": layer,
            "energy_top1_share": float(srt[0] / srt.sum()),
            "energy_top4_share": float(srt[:4].sum() / srt.sum()),
            "energy_ratio_max_to_median": float(srt[0] / srt[1280]),
            # scale-free deadness: a unit is dead if its own activation is ~0
            "units_activation_below_1e-6_of_median": int(
                (act < 1e-6 * act.median()).sum()
            ),
            "units_activation_below_1e-3_of_median": int(
                (act < 1e-3 * act.median()).sum()
            ),
            "w1_rows_below_1e-3_of_median": int(
                (w1.norm(dim=1) < 1e-3 * w1.norm(dim=1).median()).sum()
            ),
            "w3_rows_below_1e-3_of_median": int(
                (w3.norm(dim=1) < 1e-3 * w3.norm(dim=1).median()).sum()
            ),
            "phi_max_abs": float(phi.abs().max()),
            "phi_median_abs": float(phi.abs().median()),
        }
        out.append(entry)
        print(
            f"  L{layer:02d} top1 energy {entry['energy_top1_share']:.6f} "
            f"max/median {entry['energy_ratio_max_to_median']:.3e} "
            f"dead(1e-3) {entry['units_activation_below_1e-3_of_median']} "
            f"phi max {entry['phi_max_abs']:.3e} median {entry['phi_median_abs']:.3e}",
            flush=True,
        )
    return {
        "task": "l05_anomaly",
        "sites": out,
        "note": "OFF_EPSILON is a share-of-total test and is not scale-robust; "
        "a single enormous unit pushes every other unit below any share threshold",
    }


TASKS = {
    "control_mlp": task_control_mlp,
    "control_conv": task_control_conv,
    "l05_anomaly": task_l05_anomaly,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS) + ["all"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in sorted(TASKS) if args.task == "all" else [args.task]:
        print(f"== {name} ==", flush=True)
        t0 = time.time()
        try:
            result = TASKS[name]()
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            result = {"task": name, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
        result["schema"] = SCHEMA
        result["seconds"] = round(time.time() - t0, 1)
        (outdir / f"census-{name}.json").write_text(json.dumps(result, indent=1))
        print(f"  -> census-{name}.json ({result['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
