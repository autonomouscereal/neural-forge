#!/usr/bin/env python3
"""Wave-14 battery, second pass: measure what the class actually installs.

The first pass (`mx14_measure.py`) scored the kept channel set against the
parent's **verbatim** writer.  Every one of these classes re-solves its writer,
so that measurement understates them by construction and its 0.81 floors were
answering a question no cell asks.  This pass measures the installed object: the
kept features with the writer **re-solved by the same prior-anchored ridge the
production kernel uses**, priced on the 1,584 held-out rows.

The bar is not invented here either.  It is read off the sealed ledger, at the
same component, level and site, from lanes that were actually built:

| component | site | level | best sealed | relL2 | write cos |
|---|---|---|---|---|---|
| C13 `mlp_down` | L05 | S4 | `T8` | 0.1834 | 0.8587 |
| C13 `mlp_down` | L05 | S8 | `T8` | 0.2296 | 0.7699 |
| C05 `conv_out` | L05 | S4 | `T4` | 0.1489 | 0.8818 |
| C09 `attn_o`   | L02 | S4 | `T4` | 0.1527 | 0.9605 |

plus the campaign's own pre-registered endpoint band, ``relL2 <= 0.4``
(`endpoint_band_le_0p4` on all 447 rows).  A wave-14 class earns a build by
beating the best sealed lane at identical storage, or by reaching a level the
sealed lane cannot express.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from mx14_measure import (  # noqa: F401  (shared, already committed and run)
    ATTN_LAYERS,
    CONV_LAYERS,
    MLP_LAYERS,
    OFF_EPSILON,
    RIDGE_REL,
    have,
    lorenz,
    ridge_fit,
    rows,
    topk_indices,
    weight,
)

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

SCHEMA = "neural-forge-matrix-wave14-refit-battery-v1"
BAND = 0.4

SEALED = {
    "C13.S4": {"relL2": 0.1834, "cos": 0.8587, "tech": "T8", "site": "L05"},
    "C13.S8": {"relL2": 0.2296, "cos": 0.7699, "tech": "T8", "site": "L05"},
    "C05.S4": {"relL2": 0.1489, "cos": 0.8818, "tech": "T4", "site": "L05"},
    "C09.S4": {"relL2": 0.1527, "cos": 0.9605, "tech": "T4", "site": "L02"},
}


def write_scores(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """The two numbers the ledger seals: held-out relative L2 and mean row cosine."""
    diff = pred - target
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    return {
        "relL2": float(diff.norm() / target.norm()),
        "cos": float(cos.mean()),
        "cos_p05": float(torch.quantile(cos, torch.tensor(0.05))),
    }


def prior_anchored(A: torch.Tensor, Y: torch.Tensor, prior: torch.Tensor, rel: float = RIDGE_REL) -> torch.Tensor:
    """``argmin ||A W^T - Y||^2 + lam ||W - prior||^2`` -- the production objective."""
    G = A.T @ A
    lam = rel * float(torch.diagonal(G).mean())
    G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
    B = A.T @ Y + lam * prior.T
    L = torch.linalg.cholesky(G)
    return torch.cholesky_solve(B, L).T.contiguous()


def pivoted_columns(F: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic column-pivoted selection (greedy pivoted Cholesky on the Gram).

    Selects the same columns a column-pivoted QR would, without needing LAPACK's
    pivoting driver, and with the tie rule pinned to the lowest column index.
    """
    G = (F.T @ F).clone()
    n = G.shape[0]
    diag = torch.diagonal(G).clone()
    chosen: list[int] = []
    L = torch.zeros(n, k, dtype=G.dtype)
    for step in range(k):
        best = -1.0
        pick = -1
        for i in range(n):
            if diag[i] > best + 1e-15:
                best = float(diag[i])
                pick = i
        if pick < 0 or best <= 1e-30:
            break
        chosen.append(pick)
        col = G[:, pick] - L[:, :step] @ L[pick, :step]
        col = col / max(best, 1e-300) ** 0.5
        L[:, step] = col
        diag = diag - col * col
        diag[chosen] = -1.0
    return torch.tensor(sorted(chosen), dtype=torch.long)


def arm_sets(phi: torch.Tensor, phi_h: torch.Tensor, w: torch.Tensor, keepn: int) -> dict:
    """The four registered VC-FOLD selector arms, as index sets."""
    colnorm = w.norm(dim=0)
    mass = phi.abs().mean(dim=0) * colnorm
    energy = phi.pow(2).mean(dim=0) * colnorm.pow(2)
    energy_h = phi_h.pow(2).mean(dim=0) * colnorm.pow(2)

    sets = {"A_mass": topk_indices(mass, keepn)}

    G = phi.T @ phi
    lam = RIDGE_REL * float(torch.diagonal(G).mean())
    lev = torch.diagonal(G @ torch.linalg.inv(G + lam * torch.eye(G.shape[0], dtype=G.dtype)))
    sets["B_leverage"] = topk_indices(lev, keepn)

    sets["C_pivoted"] = pivoted_columns(phi, keepn)

    off = (energy / energy.sum() < OFF_EPSILON) & (energy_h / energy_h.sum() < OFF_EPSILON)
    live = (~off).nonzero().flatten()
    if live.numel() >= keepn:
        sub = topk_indices(mass[live], keepn)
        sets["E_off_then_mass"] = torch.tensor(sorted(int(live[i]) for i in sub), dtype=torch.long)
    else:  # every live unit survives and the remainder is filled from the OFF bin
        extra = topk_indices(mass * off.to(mass.dtype), keepn - live.numel())
        sets["E_off_then_mass"] = torch.tensor(
            sorted(set(live.tolist()) | set(extra.tolist())), dtype=torch.long
        )
    sets["_off_count"] = int(off.sum())
    return sets


def task_vcfold_refit() -> dict:
    """VC-FOLD, measured as installed: verbatim w1/w3 columns + re-solved w2."""
    out = []
    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        if not have(name):
            continue
        t0 = time.time()
        phi = rows(name, "mlp.post_swiglu", "fit")
        phi_h = rows(name, "mlp.post_swiglu", "hld")
        w2 = weight(f"model.layers.{layer}.feed_forward.w2.weight")
        Y = rows(name, "mlp.down", "fit")
        Y_h = rows(name, "mlp.down", "hld")
        entry = {"layer": layer, "levels": {}}
        for level, keepn in (("S4", 640), ("S8", 320), ("S16", 160)):
            sets = arm_sets(phi, phi_h, w2, keepn)
            entry["off_certified"] = sets.pop("_off_count")
            per_arm = {}
            for arm, keep in sets.items():
                W = prior_anchored(phi[:, keep], Y, w2[:, keep])
                sc = write_scores(phi_h[:, keep] @ W.T, Y_h)
                sc["update_scale"] = float((W - w2[:, keep]).norm() / w2[:, keep].norm())
                sc["in_band"] = bool(sc["relL2"] <= BAND)
                per_arm[arm] = sc
            best = min(per_arm, key=lambda a: per_arm[a]["relL2"])
            entry["levels"][level] = {"arms": per_arm, "best_arm": best, **per_arm[best]}
            ref = SEALED.get(f"C13.{level}")
            if ref:
                entry["levels"][level]["beats_sealed_relL2"] = bool(
                    per_arm[best]["relL2"] <= ref["relL2"]
                )
                entry["levels"][level]["beats_sealed_cos"] = bool(per_arm[best]["cos"] >= ref["cos"])
        entry["seconds"] = round(time.time() - t0, 1)
        out.append(entry)
        line = f"  L{layer:02d} OFF={entry['off_certified']:4d}"
        for level in ("S4", "S8", "S16"):
            e = entry["levels"][level]
            line += f" | {level} {e['best_arm'][0]} rel {e['relL2']:.4f} cos {e['cos']:.4f}"
        print(line, flush=True)
    return {"task": "vcfold_refit", "sealed_reference": SEALED, "band": BAND, "sites": out}


def task_convc_refit() -> dict:
    """F-C, measured as installed: verbatim channel slices + re-solved out_proj."""
    out = []
    for layer in CONV_LAYERS:
        name = f"L{layer:02d}-C05"
        if not have(name):
            continue
        ctx = rows(name, "short_conv.context", "fit")
        ctx_h = rows(name, "short_conv.context", "hld")
        op = weight(f"model.layers.{layer}.conv.out_proj.weight")
        Y = ctx @ op.T
        Y_h = ctx_h @ op.T
        entry = {"layer": layer, "levels": {}}
        for level, keepn in (("S4", 256), ("S8", 128), ("S16", 64)):
            sets = arm_sets(ctx, ctx_h, op, keepn)
            entry["off_certified"] = sets.pop("_off_count")
            per_arm = {}
            for arm, keep in sets.items():
                W = prior_anchored(ctx[:, keep], Y, op[:, keep])
                sc = write_scores(ctx_h[:, keep] @ W.T, Y_h)
                sc["update_scale"] = float((W - op[:, keep]).norm() / op[:, keep].norm())
                sc["in_band"] = bool(sc["relL2"] <= BAND)
                per_arm[arm] = sc
            best = min(per_arm, key=lambda a: per_arm[a]["relL2"])
            entry["levels"][level] = {"arms": per_arm, "best_arm": best, **per_arm[best]}
        out.append(entry)
        line = f"  L{layer:02d} OFF={entry['off_certified']:4d}"
        for level in ("S4", "S8", "S16"):
            e = entry["levels"][level]
            line += f" | {level} {e['best_arm'][0]} rel {e['relL2']:.4f} cos {e['cos']:.4f}"
        print(line, flush=True)
    return {"task": "convc_refit", "sealed_reference": SEALED["C05.S4"], "band": BAND, "sites": out}


def task_vo_lra() -> dict:
    """Attention value/output contraction: per-KV-group rank-r bottleneck, o re-solved.

    ``context_h = A_h (X Wv_g)``, so replacing ``Wv_g`` by ``P_g Wv_g`` gives
    ``context'_h = context_h P_g^T`` **exactly** -- no attention replay, and the
    attention pattern is bit-identical.  ``P_g`` is chosen by the evidence-weighted
    LRA of the group's stacked out-slices; every ``Wo'_h`` is then re-solved
    exactly against its own head's data, so the number reported is the installed
    error and not the selection's own estimate.

    At r = 16 this is ``head_dim`` 64 -> 16 on v and o: 4x exact on the pair.
    """
    out = []
    for layer in ATTN_LAYERS:
        name = f"L{layer:02d}-C09"
        if not have(name):
            continue
        ctx = rows(name, "attention.context", "fit")
        ctx_h = rows(name, "attention.context", "hld")
        wo = weight(f"model.layers.{layer}.self_attn.out_proj.weight")
        Y = rows(name, "attention.o", "fit")
        Y_h = rows(name, "attention.o", "hld")
        entry = {"layer": layer, "levels": {}}
        for level, r in (("S4", 16), ("S8", 8)):
            pred = torch.zeros_like(Y_h)
            fit_pred = torch.zeros_like(Y)
            per_group = []
            for group in range(8):
                heads = (2 * group, 2 * group + 1)
                Cg = torch.cat([ctx[:, h * 64 : (h + 1) * 64] for h in heads], dim=0)
                Gg = Cg.T @ Cg / Cg.shape[0]
                ev, evec = torch.linalg.eigh(Gg)
                Rt = evec @ torch.diag(ev.clamp_min(0).sqrt()) @ evec.T
                stack = torch.cat([wo[:, h * 64 : (h + 1) * 64] for h in heads], dim=0)
                U, S, Vh = torch.linalg.svd(stack @ Rt, full_matrices=False)
                # P maps 64 -> r in the *value channel* basis
                Rt_inv = evec @ torch.diag(1.0 / ev.clamp_min(1e-12).sqrt()) @ evec.T
                P = (Vh[:r] @ Rt_inv.T)  # r x 64  (rows span the retained directions)
                P = torch.linalg.qr(P.T)[0].T  # orthonormalize, pinned by QR sign
                per_group.append({"group": group, "retained_energy": float((S[:r] ** 2).sum() / (S**2).sum())})
                for h in heads:
                    Ch = ctx[:, h * 64 : (h + 1) * 64]
                    Ch_h = ctx_h[:, h * 64 : (h + 1) * 64]
                    prior = wo[:, h * 64 : (h + 1) * 64] @ P.T
                    W = prior_anchored(Ch @ P.T, Ch @ wo[:, h * 64 : (h + 1) * 64].T, prior)
                    fit_pred += (Ch @ P.T) @ W.T
                    pred += (Ch_h @ P.T) @ W.T
            sc = write_scores(pred, Y_h)
            sc["in_band"] = bool(sc["relL2"] <= BAND)
            sc["retained_energy_min"] = min(g["retained_energy"] for g in per_group)
            sc["retained_energy_median"] = sorted(g["retained_energy"] for g in per_group)[4]
            entry["levels"][level] = sc
        out.append(entry)
        line = f"  L{layer:02d}"
        for level in ("S4", "S8"):
            e = entry["levels"][level]
            line += (
                f" | {level} rel {e['relL2']:.4f} cos {e['cos']:.4f} "
                f"energy {e['retained_energy_median']:.4f}"
            )
        print(line, flush=True)
    return {
        "task": "vo_lra",
        "sealed_reference": SEALED["C09.S4"],
        "band": BAND,
        "export": "head_dim is a live config field in transformers 5.12.1 Lfm2 "
        "(modeling_lfm2.py:233); v+o at r=16 is config-consistent on the v/o half only",
        "sites": out,
    }


TASKS = {
    "vcfold_refit": task_vcfold_refit,
    "convc_refit": task_convc_refit,
    "vo_lra": task_vo_lra,
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
        path = outdir / f"census-{name}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"  -> {path} ({result['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
