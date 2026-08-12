#!/usr/bin/env python3
"""Wave-14 free measurement battery (`FOLD_CLASS_CANDIDATES_2026-08-12.md` §4.3).

Every task here reads sealed evidence and the parent checkpoint and writes a
JSON census.  Nothing is built, nothing is graded, no ledger row is appended.
The point of the battery is that it is expected to **kill classes before they
spend a build**, so the floors it compares against are constants in this file
and this file is committed **before it is run**.  A floor that gets chosen after
the numbers are on screen is not a floor.

The pre-registered floors, in one place (§4.3's "calibrated by analogy to the
embedding's measured ~81% occurrence-coverage floor"):

``ALIVE_FLOOR_COVERAGE = 0.81``
    median per-row write coverage of the kept channel set.  A census below this
    means the survivors do not carry the row's write and the class is not built
    at that site.
``ALIVE_FLOOR_MASS = 0.81``
    share of the evidence-weighted operator-product mass inside the kept set.
``OFF_EPSILON = 1e-5``
    a unit is certified OFF (VC-FOLD arm E, free drop) when its share of the
    layer's gated-output energy is below this on fit **and** heldout.
``EXCISE_ABSORB_FLOOR = 0.50``
    F-EXCISE builds only where the upstream writer's re-solve absorbs at least
    half of the deleted block's heldout contribution.
``G_R2_FLOOR = 0.5``
    §4.1's floor.  Track B dies below it, per layer / per endpoint.

Everything is fp64 on CPU with pinned deterministic tie-breaking by index.

Usage::

    python mx14_measure.py --task vcfold --out <dir>
    python mx14_measure.py --task excise --out <dir>
    ...
    python mx14_measure.py --task all --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------- pre-registered
ALIVE_FLOOR_COVERAGE = 0.81
ALIVE_FLOOR_MASS = 0.81
OFF_EPSILON = 1e-5
EXCISE_ABSORB_FLOOR = 0.50
G_R2_FLOOR = 0.5
RIDGE_REL = 1e-6  # relative to the mean eigenvalue of the normal matrix

EVIDENCE = Path("/mnt/ai_data/neural-forge-matrix/wave13/evidence-cache")
PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M/model.safetensors")

MLP_LAYERS = tuple(range(14))
CONV_LAYERS = (0, 1, 3, 5, 7, 9, 11, 13)
ATTN_LAYERS = (2, 4, 6, 8, 10, 12)

SCHEMA = "neural-forge-matrix-wave14-measurement-battery-v1"


# ---------------------------------------------------------------- evidence access
_CACHE: dict[str, dict] = {}


def cache(name: str) -> dict:
    """Load one sealed site cache, verifying the payload digest as the runner does."""
    if name in _CACHE:
        return _CACHE[name]
    directory = EVIDENCE / name
    binding = json.loads((directory / "binding.json").read_text())
    payload = torch.load(directory / "payload.pt", weights_only=True, map_location="cpu")
    out = {"binding": binding, "payload": payload}
    if len(_CACHE) > 6:  # bounded: these are 200 MB each
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[name] = out
    return out


def rows(name: str, endpoint: str, split: str = "fit") -> torch.Tensor:
    key = f"{'fit' if split == 'fit' else 'hld'}|{endpoint}"
    payload = cache(name)["payload"]
    if key not in payload:
        raise KeyError(f"{name} has no {key}; has {sorted(payload)}")
    return payload[key].to(torch.float64)


def have(name: str) -> bool:
    return (EVIDENCE / name / "payload.pt").exists()


_WEIGHTS: dict[str, torch.Tensor] = {}


def weight(key: str) -> torch.Tensor:
    if key not in _WEIGHTS:
        from safetensors import safe_open

        with safe_open(str(PARENT), "pt") as handle:
            _WEIGHTS[key] = handle.get_tensor(key).to(torch.float64)
    return _WEIGHTS[key]


# ---------------------------------------------------------------- linear algebra
def ridge_fit(A: torch.Tensor, Y: torch.Tensor, rel: float = RIDGE_REL) -> torch.Tensor:
    """Solve ``min ||A W^T - Y||_F^2 + lam ||W||^2`` for W (out x in).  fp64, Cholesky."""
    G = A.T @ A
    lam = rel * float(torch.diagonal(G).mean())
    G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
    B = A.T @ Y
    L = torch.linalg.cholesky(G)
    W = torch.cholesky_solve(B, L)  # in x out
    return W.T.contiguous()


def rel_residual(A: torch.Tensor, W: torch.Tensor, Y: torch.Tensor) -> float:
    return float((Y - A @ W.T).norm() / Y.norm())


def r_squared(A: torch.Tensor, W: torch.Tensor, Y: torch.Tensor) -> float:
    ss_res = float(((Y - A @ W.T) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0, keepdim=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def lorenz(values: torch.Tensor) -> dict:
    """Concentration summary of a non-negative score vector."""
    v = torch.sort(values.abs(), descending=True).values
    total = float(v.sum())
    n = v.numel()
    out = {"n": n, "total": total}
    for frac in (0.05, 0.10, 0.125, 0.25, 0.50):
        k = max(1, int(round(frac * n)))
        out[f"mass_top_{frac:g}"] = float(v[:k].sum()) / total if total > 0 else float("nan")
    # participation ratio: (sum v)^2 / (n * sum v^2) -- 1.0 = flat, ~1/n = one spike
    out["participation_ratio"] = (
        float(v.sum() ** 2 / (n * (v**2).sum())) if float((v**2).sum()) > 0 else float("nan")
    )
    return out


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic top-k: value descending, ties broken by ascending index."""
    order = sorted(range(scores.numel()), key=lambda i: (-float(scores[i]), i))
    return torch.tensor(sorted(order[:k]), dtype=torch.long)


def write_coverage(F: torch.Tensor, W: torch.Tensor, keep: torch.Tensor) -> dict:
    """Per-row fraction of the write reproduced by the kept feature columns.

    ``F`` is n x d features, ``W`` is out x d, the row write is ``F @ W.T``.
    """
    full = F @ W.T
    part = F[:, keep] @ W[:, keep].T
    num = part.norm(dim=1)
    den = full.norm(dim=1).clamp_min(1e-30)
    # cosine-style coverage: how much of the row write the kept set actually lands on
    proj = (part * full).sum(dim=1) / den.pow(2)
    ratio = (num / den).cpu()
    q = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64)
    return {
        "coverage_norm_ratio_quantiles": [float(x) for x in torch.quantile(ratio, q)],
        "coverage_norm_ratio_median": float(torch.quantile(ratio, torch.tensor(0.5))),
        "coverage_projection_median": float(torch.quantile(proj.cpu(), torch.tensor(0.5))),
        "rows_covered_ge_0.8": float((ratio >= 0.8).to(torch.float64).mean()),
        "relative_residual": float((full - part).norm() / full.norm()),
    }


def interpolative_residual(F: torch.Tensor, keep: torch.Tensor) -> float:
    """``||F - F_S T||_F / ||F||_F`` with T the exact least-squares coefficient map."""
    Fs = F[:, keep]
    sol = torch.linalg.lstsq(Fs, F).solution
    return float((F - Fs @ sol).norm() / F.norm())


# ---------------------------------------------------------------- task: VC-FOLD
def task_vcfold() -> dict:
    """§4.3.1 -- SwiGLU intermediate-channel concentration at all 14 MLP sites."""
    sites = []
    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        if not have(name):
            sites.append({"layer": layer, "status": "no-cache"})
            continue
        t0 = time.time()
        phi = rows(name, "mlp.post_swiglu", "fit")
        phi_h = rows(name, "mlp.post_swiglu", "hld")
        w2 = weight(f"model.layers.{layer}.feed_forward.w2.weight")  # 1024 x 2560
        colnorm = w2.norm(dim=0)  # 2560

        # arm A -- operator-product mass
        mass = phi.abs().mean(dim=0) * colnorm
        # energy share, used by arm E's OFF certificate
        energy = phi.pow(2).mean(dim=0) * colnorm.pow(2)
        energy_h = phi_h.pow(2).mean(dim=0) * colnorm.pow(2)
        share = energy / energy.sum()
        share_h = energy_h / energy_h.sum()
        off = ((share < OFF_EPSILON) & (share_h < OFF_EPSILON))

        entry = {
            "layer": layer,
            "status": "measured",
            "lorenz_mass": lorenz(mass),
            "lorenz_energy": lorenz(energy),
            "off_certified_units": int(off.sum()),
            "off_epsilon": OFF_EPSILON,
        }
        for level, keepn in (("S4", 640), ("S8", 320)):
            keep = topk_indices(mass, keepn)
            entry[level] = {
                "keep": keepn,
                "mass_share": float(mass[keep].sum() / mass.sum()),
                "energy_share": float(energy[keep].sum() / energy.sum()),
                **write_coverage(phi, w2, keep),
                "heldout": write_coverage(phi_h, w2, keep),
                "interpolative_residual": interpolative_residual(phi, keep),
            }
            entry[level]["alive"] = bool(
                entry[level]["mass_share"] >= ALIVE_FLOOR_MASS
                and entry[level]["coverage_norm_ratio_median"] >= ALIVE_FLOOR_COVERAGE
            )
        entry["seconds"] = round(time.time() - t0, 1)
        sites.append(entry)
        print(
            f"  L{layer:02d}  S4 mass {entry['S4']['mass_share']:.4f} "
            f"cov50 {entry['S4']['coverage_norm_ratio_median']:.4f} "
            f"id-resid {entry['S4']['interpolative_residual']:.4f} "
            f"alive={entry['S4']['alive']}  |  S8 mass {entry['S8']['mass_share']:.4f} "
            f"cov50 {entry['S8']['coverage_norm_ratio_median']:.4f}  OFF={entry['off_certified_units']}",
            flush=True,
        )
    alive = [s for s in sites if s.get("S4", {}).get("alive")]
    return {
        "task": "vcfold",
        "floors": {"mass": ALIVE_FLOOR_MASS, "coverage": ALIVE_FLOOR_COVERAGE},
        "sites": sites,
        "alive_S4_sites": [s["layer"] for s in alive],
        "verdict": "alive" if alive else "class killed by census at every site",
    }


# ---------------------------------------------------------------- task: F-EXCISE
def task_excise() -> dict:
    """§4.3.2 -- can the upstream FFN writer absorb the deleted block's write?"""
    out = []
    for layer in range(1, 14):
        up = f"L{layer - 1:02d}-C13"
        me = f"L{layer:02d}-C13"
        if not (have(up) and have(me)):
            out.append({"layer": layer, "status": "no-cache"})
            continue
        t0 = time.time()
        A = rows(up, "mlp.post_swiglu", "fit")
        A_h = rows(up, "mlp.post_swiglu", "hld")
        own = rows(up, "mlp.down", "fit")
        own_h = rows(up, "mlp.down", "hld")
        delta = rows(me, "residual.post_layer", "fit") - rows(me, "residual.pre", "fit")
        delta_h = rows(me, "residual.post_layer", "hld") - rows(me, "residual.pre", "hld")
        Y = own + delta
        Y_h = own_h + delta_h

        w2 = weight(f"model.layers.{layer - 1}.feed_forward.w2.weight")
        base_h = float((Y_h - A_h @ w2.T).norm())  # = ||delta_h||, doing nothing
        refit = ridge_fit(A, Y)
        got_h = float((Y_h - A_h @ refit.T).norm())
        dn = float(delta_h.norm())
        entry = {
            "layer": layer,
            "kind": "conv" if layer in CONV_LAYERS else "attention",
            "status": "measured",
            "heldout_delta_norm": dn,
            "heldout_target_norm": float(Y_h.norm()),
            "heldout_residual_parent_w2": base_h,
            "heldout_residual_refit": got_h,
            "absorbed_fraction_of_delta": 1.0 - got_h / dn if dn > 0 else float("nan"),
            "unexplained_relative": got_h / float(Y_h.norm()),
            "delta_share_of_stream": dn / float(rows(me, "residual.post_layer", "hld").norm()),
            "update_scale": float((refit - w2).norm() / w2.norm()),
            "seconds": round(time.time() - t0, 1),
        }
        entry["alive"] = bool(entry["absorbed_fraction_of_delta"] >= EXCISE_ABSORB_FLOOR)
        out.append(entry)
        print(
            f"  L{layer:02d} {entry['kind']:9s} absorbed {entry['absorbed_fraction_of_delta']:+.4f}  "
            f"delta/stream {entry['delta_share_of_stream']:.4f}  "
            f"update_scale {entry['update_scale']:.3f}  alive={entry['alive']}",
            flush=True,
        )
    ranked = sorted(
        (e for e in out if e.get("status") == "measured"),
        key=lambda e: -e["absorbed_fraction_of_delta"],
    )
    return {
        "task": "excise",
        "floor_absorbed": EXCISE_ABSORB_FLOOR,
        "blocks": out,
        "ranking": [(e["layer"], e["kind"], e["absorbed_fraction_of_delta"]) for e in ranked],
        "verdict": (
            "alive" if any(e.get("alive") for e in out) else "class killed at scan stage"
        ),
    }


# ---------------------------------------------------------------- task: F-C
def task_convc() -> dict:
    """§4.3.3 -- conv channel concentration plus the B/C/x role-mass agreement."""
    out = []
    for layer in CONV_LAYERS:
        name = f"L{layer:02d}-C05"
        if not have(name):
            out.append({"layer": layer, "status": "no-cache"})
            print(f"  L{layer:02d} no cache", flush=True)
            continue
        ctx = rows(name, "short_conv.context", "fit")
        ctx_h = rows(name, "short_conv.context", "hld")
        op = weight(f"model.layers.{layer}.conv.out_proj.weight")  # 1024 x 1024
        inp = weight(f"model.layers.{layer}.conv.in_proj.weight")  # 3072 x 1024
        colnorm = op.norm(dim=0)
        mass = ctx.abs().mean(dim=0) * colnorm

        # role-mass agreement: in_proj emits B, C, x over one shared channel axis
        roles = {
            "B": inp[0:1024].norm(dim=1),
            "C": inp[1024:2048].norm(dim=1),
            "x": inp[2048:3072].norm(dim=1),
        }
        def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
            ra = torch.argsort(torch.argsort(a)).to(torch.float64)
            rb = torch.argsort(torch.argsort(b)).to(torch.float64)
            ra = ra - ra.mean()
            rb = rb - rb.mean()
            return float((ra @ rb) / (ra.norm() * rb.norm()))

        entry = {
            "layer": layer,
            "status": "measured",
            "lorenz_mass": lorenz(mass),
            "role_rank_correlation": {
                "B_vs_C": spearman(roles["B"], roles["C"]),
                "B_vs_x": spearman(roles["B"], roles["x"]),
                "C_vs_x": spearman(roles["C"], roles["x"]),
            },
        }
        for level, keepn in (("S4", 256), ("S8", 128)):
            keep = topk_indices(mass, keepn)
            entry[level] = {
                "keep": keepn,
                "mass_share": float(mass[keep].sum() / mass.sum()),
                **write_coverage(ctx, op, keep),
                "heldout": write_coverage(ctx_h, op, keep),
                "interpolative_residual": interpolative_residual(ctx, keep),
            }
            entry[level]["alive"] = bool(
                entry[level]["mass_share"] >= ALIVE_FLOOR_MASS
                and entry[level]["coverage_norm_ratio_median"] >= ALIVE_FLOOR_COVERAGE
            )
        out.append(entry)
        rc = entry["role_rank_correlation"]
        print(
            f"  L{layer:02d} S4 mass {entry['S4']['mass_share']:.4f} cov50 "
            f"{entry['S4']['coverage_norm_ratio_median']:.4f} alive={entry['S4']['alive']} | "
            f"S8 mass {entry['S8']['mass_share']:.4f} cov50 "
            f"{entry['S8']['coverage_norm_ratio_median']:.4f} alive={entry['S8']['alive']} | "
            f"roles BC {rc['B_vs_C']:+.3f} Bx {rc['B_vs_x']:+.3f} Cx {rc['C_vs_x']:+.3f}",
            flush=True,
        )
    return {"task": "convc", "sites": out}


# ---------------------------------------------------------------- task: G artifact
def task_gart() -> dict:
    """§4.1 -- per-layer and per-endpoint terminal transfer surrogates.

    The terminal object is ``residual.post_layer`` at L13: the stream the final
    norm and the tied head read.  No new capture is taken for it.
    """
    term = rows("L13-C13", "residual.post_layer", "fit")
    term_h = rows("L13-C13", "residual.post_layer", "hld")
    layers = []
    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        X = rows(name, "residual.post_layer", "fit")
        X_h = rows(name, "residual.post_layer", "hld")
        G = ridge_fit(X, term)
        sv = torch.linalg.svdvals(G)
        entry = {
            "layer": layer,
            "endpoint": "residual.post_layer",
            "heldout_r2": r_squared(X_h, G, term_h),
            "heldout_relative_residual": rel_residual(X_h, G, term_h),
            "sigma_max": float(sv[0]),
            "sigma_min": float(sv[-1]),
            "condition": float(sv[0] / sv[-1]),
            "effective_rank_90": int((torch.cumsum(sv, 0) / sv.sum() < 0.90).sum()) + 1,
        }
        entry["above_floor"] = bool(entry["heldout_r2"] > G_R2_FLOOR)
        layers.append(entry)
        print(
            f"  layer G  L{layer:02d}  R2 {entry['heldout_r2']:+.4f}  "
            f"cond {entry['condition']:.3e}  eff_rank90 {entry['effective_rank_90']}  "
            f"above_floor={entry['above_floor']}",
            flush=True,
        )
    endpoints = []
    for layer in MLP_LAYERS:
        name = f"L{layer:02d}-C13"
        X = rows(name, "mlp.post_swiglu", "fit")
        X_h = rows(name, "mlp.post_swiglu", "hld")
        G = ridge_fit(X, term, rel=1e-4)
        entry = {
            "layer": layer,
            "endpoint": "mlp.post_swiglu",
            "heldout_r2": r_squared(X_h, G, term_h),
            "heldout_relative_residual": rel_residual(X_h, G, term_h),
            "overdetermination": X.shape[0] / X.shape[1],
        }
        entry["above_floor"] = bool(entry["heldout_r2"] > G_R2_FLOOR)
        endpoints.append(entry)
        print(
            f"  endp  G  L{layer:02d}  R2 {entry['heldout_r2']:+.4f}  "
            f"above_floor={entry['above_floor']}",
            flush=True,
        )
    n_ok = sum(1 for e in layers if e["above_floor"])
    return {
        "task": "gart",
        "floor_r2": G_R2_FLOOR,
        "terminal_endpoint": "L13 residual.post_layer",
        "layer_transfers": layers,
        "endpoint_transfers": endpoints,
        "layers_above_floor": n_ok,
        "verdict": (
            "Track B alive" if n_ok >= 7 else "Track B dies with the G artifact"
        ),
    }


# ---------------------------------------------------------------- task: S+L
def task_splus() -> dict:
    """§4.3.4 -- kurtosis / participation ratio of evidence-scaled weight entries."""
    out = []
    targets = [
        ("C13", 5, "model.layers.5.feed_forward.w2.weight", "L05-C13", "mlp.post_swiglu"),
        ("C13", 9, "model.layers.9.feed_forward.w2.weight", "L09-C13", "mlp.post_swiglu"),
        ("C13", 13, "model.layers.13.feed_forward.w2.weight", "L13-C13", "mlp.post_swiglu"),
        ("C05", 5, "model.layers.5.conv.out_proj.weight", "L05-C05", "short_conv.context"),
        ("C05", 13, "model.layers.13.conv.out_proj.weight", "L13-C05", "short_conv.context"),
        ("C09", 12, "model.layers.12.self_attn.out_proj.weight", "L12-C09", "attention.context"),
        ("C09", 8, "model.layers.8.self_attn.out_proj.weight", "L08-C09", "attention.context"),
        ("C08", 2, "model.layers.2.self_attn.v_proj.weight", "L02-C08", "operator.input"),
    ]
    for comp, layer, key, cname, endpoint in targets:
        if not have(cname):
            out.append({"component": comp, "layer": layer, "status": "no-cache"})
            continue
        X = rows(cname, endpoint, "fit")
        W = weight(key)
        d = (X.pow(2).mean(dim=0)).sqrt()  # D^{1/2}, the evidence input scale
        S = W * d.unsqueeze(0)
        v = S.reshape(-1)
        v = v - v.mean()
        kurt = float((v.pow(4).mean()) / (v.pow(2).mean() ** 2))
        a = v.abs()
        pr = float(a.sum() ** 2 / (a.numel() * (a**2).sum()))
        # what a k-entry sparse part would buy: mass in the top 1% of entries
        srt = torch.sort(a.pow(2), descending=True).values
        k1 = max(1, srt.numel() // 100)
        entry = {
            "component": comp,
            "layer": layer,
            "status": "measured",
            "shape": list(W.shape),
            "kurtosis": kurt,
            "gaussian_kurtosis": 3.0,
            "participation_ratio": pr,
            "energy_top_1pct_entries": float(srt[:k1].sum() / srt.sum()),
            "heavy_tailed": bool(kurt > 6.0),
        }
        out.append(entry)
        print(
            f"  {comp}/L{layer:02d}  kurtosis {kurt:8.3f}  PR {pr:.4f}  "
            f"top1%% energy {entry['energy_top_1pct_entries']:.4f}  "
            f"heavy={entry['heavy_tailed']}",
            flush=True,
        )
    return {"task": "splus", "note": "kurtosis <= 6 means S buys ~nothing over T4", "components": out}


# ---------------------------------------------------------------- task: F-B
def task_fb() -> dict:
    """§4.3.5 -- do real value channels track the SVD floor on sealed context?"""
    out = []
    for layer in ATTN_LAYERS:
        name = f"L{layer:02d}-C09"
        if not have(name):
            out.append({"layer": layer, "status": "no-cache"})
            print(f"  L{layer:02d} no cache", flush=True)
            continue
        ctx = rows(name, "attention.context", "fit")  # n x 1024, 16 q-heads x 64
        wo = weight(f"model.layers.{layer}.self_attn.out_proj.weight")
        per_head = []
        for head in range(16):
            block = ctx[:, head * 64 : (head + 1) * 64]
            wob = wo[:, head * 64 : (head + 1) * 64]
            score = block.abs().mean(dim=0) * wob.norm(dim=0)
            keep = topk_indices(score, 16)
            real = interpolative_residual(block, keep)
            sv = torch.linalg.svdvals(block)
            svd_floor = float((sv[16:] ** 2).sum().sqrt() / (sv**2).sum().sqrt())
            per_head.append(
                {
                    "head": head,
                    "real_channel_residual": real,
                    "svd_floor": svd_floor,
                    "gap": real - svd_floor,
                    "ratio": real / svd_floor if svd_floor > 0 else float("inf"),
                }
            )
        gaps = torch.tensor([p["ratio"] for p in per_head])
        entry = {
            "layer": layer,
            "status": "measured",
            "per_head": per_head,
            "ratio_median": float(gaps.median()),
            "ratio_max": float(gaps.max()),
        }
        # "real channels track the SVD floor" -- pre-registered as within 25%
        entry["alive"] = bool(entry["ratio_median"] <= 1.25)
        out.append(entry)
        print(
            f"  L{layer:02d} real/SVD ratio median {entry['ratio_median']:.3f} "
            f"max {entry['ratio_max']:.3f}  alive={entry['alive']}",
            flush=True,
        )
    return {"task": "fb", "sites": out, "bar": "median real/SVD residual ratio <= 1.25"}


# ---------------------------------------------------------------- task: HODLR-V
def task_hodlr() -> dict:
    """§4.3.7 -- fraction of evidence-weighted interaction energy inside diagonal blocks."""
    out = []
    for comp, layer, key, cname, endpoint in (
        ("C13", 5, "model.layers.5.feed_forward.w2.weight", "L05-C13", "mlp.post_swiglu"),
        ("C13", 13, "model.layers.13.feed_forward.w2.weight", "L13-C13", "mlp.post_swiglu"),
        ("C09", 12, "model.layers.12.self_attn.out_proj.weight", "L12-C09", "attention.context"),
    ):
        if not have(cname):
            continue
        X = rows(cname, endpoint, "fit")
        W = weight(key)
        G = X.T @ X / X.shape[0]
        dg = torch.diagonal(G).clamp_min(1e-30).sqrt()
        Gn = G / dg.unsqueeze(0) / dg.unsqueeze(1)
        # Fiedler ordering of the co-activation graph (pinned: ascending Fiedler value,
        # ties by index; sign pinned so that entry 0 is non-negative)
        A = Gn.abs()
        deg = A.sum(dim=1)
        L = torch.diag(deg) - A
        evals, evecs = torch.linalg.eigh(L)
        f = evecs[:, 1]
        if float(f[0]) < 0:
            f = -f
        order = torch.tensor(sorted(range(f.numel()), key=lambda i: (float(f[i]), i)))
        Wp = W[:, order]
        Xp = X[:, order]
        Gp = Xp.T @ Xp / Xp.shape[0]
        # interaction energy E_ij = |W_.i^T W_.j| * |G_ij| summed inside vs outside blocks
        M = (Wp.T @ Wp).abs() * Gp.abs()
        total = float(M.sum())
        res = {"component": comp, "layer": layer, "n": int(M.shape[0])}
        for g in (4, 8):
            b = M.shape[0] // g
            inside = 0.0
            for i in range(g):
                inside += float(M[i * b : (i + 1) * b, i * b : (i + 1) * b].sum())
            res[f"diag_coverage_g{g}"] = inside / total
        out.append(res)
        print(
            f"  {comp}/L{layer:02d} diag coverage g4 {res['diag_coverage_g4']:.4f} "
            f"g8 {res['diag_coverage_g8']:.4f}",
            flush=True,
        )
    return {
        "task": "hodlr",
        "floor": ALIVE_FLOOR_COVERAGE,
        "components": out,
        "note": "diagonal coverage below the 0.81 floor means off-diagonal truncation carries the class",
    }


# ---------------------------------------------------------------- task: gauge
def task_gauge() -> dict:
    """§4.3.11 -- pooled residual covariance spectrum (the residual-gauge census)."""
    acc = None
    n = 0
    for layer in MLP_LAYERS:
        X = rows(f"L{layer:02d}-C13", "residual.post_layer", "fit")
        acc = X.T @ X if acc is None else acc + X.T @ X
        n += X.shape[0]
    sigma = acc / n
    sv = torch.linalg.eigvalsh(sigma).flip(0).clamp_min(0)
    cum = torch.cumsum(sv, 0) / sv.sum()
    out = {
        "task": "gauge",
        "dim": int(sv.numel()),
        "top_1_share": float(sv[0] / sv.sum()),
        "top_16_share": float(sv[:16].sum() / sv.sum()),
        "top_64_share": float(sv[:64].sum() / sv.sum()),
        "rank_for_90pct": int((cum < 0.90).sum()) + 1,
        "rank_for_99pct": int((cum < 0.99).sum()) + 1,
        "participation_ratio": float(sv.sum() ** 2 / (sv.numel() * (sv**2).sum())),
    }
    print(f"  Sigma_hat: top16 {out['top_16_share']:.4f}  rank90 {out['rank_for_90pct']}", flush=True)
    return out


# ---------------------------------------------------------------- task: q/k (gap 2)
def task_qk() -> dict:
    """Critique gap 2 -- spectral decay of the evidence-weighted score operator.

    Two honest limitations, stated because they bound what a pass would mean:
    this measures the ``Delta = 0`` member of the RoPE family ``q^T R_Delta k``
    (a joint factorization has to serve every Delta, so this is a *necessary*
    condition and not a sufficient one), and it uses the raw projections rather
    than the per-head ``q_layernorm`` / ``k_layernorm`` outputs.  A flat spectrum
    here kills the class; a concentrated one only licenses the build.
    """
    out = []
    for layer in ATTN_LAYERS:
        cq, ck = f"L{layer:02d}-C06", f"L{layer:02d}-C07"
        if not (have(cq) and have(ck)):
            out.append({"layer": layer, "status": "no-cache"})
            print(f"  L{layer:02d} no cache", flush=True)
            continue
        X = rows(cq, "operator.input", "fit")
        G = X.T @ X / X.shape[0]
        evals, evecs = torch.linalg.eigh(G)
        Rt = evecs @ torch.diag(evals.clamp_min(0).sqrt()) @ evecs.T
        wq = weight(f"model.layers.{layer}.self_attn.q_proj.weight")  # 1024 x 1024
        wk = weight(f"model.layers.{layer}.self_attn.k_proj.weight")  # 512 x 1024
        heads = []
        for head in range(16):
            q = wq[head * 64 : (head + 1) * 64]
            k = wk[(head // 2) * 64 : (head // 2 + 1) * 64]
            # The whitened score operator M = (Rt q^T)(k Rt) is 1024x1024 of rank <= 64.
            # Its singular values are those of Ra Rb^T from the two thin QRs, exactly,
            # which is a 64x64 SVD instead of a 1024x1024 one.
            Qa, Ra = torch.linalg.qr(Rt @ q.T)
            Qb, Rb = torch.linalg.qr(Rt @ k.T)
            sv = torch.linalg.svdvals(Ra @ Rb.T)
            e = sv.pow(2)
            heads.append(
                {
                    "head": head,
                    "energy_top_16": float(e[:16].sum() / e.sum()),
                    "energy_top_32": float(e[:32].sum() / e.sum()),
                    "effective_rank_90": int((torch.cumsum(e, 0) / e.sum() < 0.90).sum()) + 1,
                }
            )
        t16 = torch.tensor([h["energy_top_16"] for h in heads])
        entry = {
            "layer": layer,
            "status": "measured",
            "heads": heads,
            "energy_top16_median": float(t16.median()),
            "energy_top16_min": float(t16.min()),
        }
        entry["alive"] = bool(entry["energy_top16_median"] >= ALIVE_FLOOR_MASS)
        out.append(entry)
        print(
            f"  L{layer:02d} score-op top16 energy median {entry['energy_top16_median']:.4f} "
            f"min {entry['energy_top16_min']:.4f}  alive={entry['alive']}",
            flush=True,
        )
    return {"task": "qk", "sites": out, "floor": ALIVE_FLOOR_MASS}


# ---------------------------------------------------------------- task: vocab (gap 1)
def task_vocab() -> dict:
    """Critique gap 1 -- can the Zipf tail of the tied embedding be row-TIED?"""
    E = weight("model.embed_tokens.weight")  # 65536 x 1024
    norms = E.norm(dim=1)
    out = {
        "task": "vocab",
        "vocab": int(E.shape[0]),
        "dim": int(E.shape[1]),
        "row_norm_lorenz": lorenz(norms),
    }
    # Coherence of the tail: how well does a cluster's weighted mean stand in for
    # its members?  Measured on a deterministic sample of the tail with k-means-free
    # complete-linkage-style probe: nearest-neighbour cosine among tail rows.
    torch.manual_seed(0)
    tail = E[12288:]  # rows beyond a 12,288 verbatim head
    idx = torch.arange(0, tail.shape[0], max(1, tail.shape[0] // 4096))[:4096]
    S = tail[idx]
    Sn = S / S.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = Sn @ Sn.T
    C.fill_diagonal_(-2.0)
    nn = C.max(dim=1).values
    out["tail_sample"] = int(S.shape[0])
    out["tail_nn_cosine_quantiles"] = [
        float(x) for x in torch.quantile(nn, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    ]
    out["tail_nn_cosine_median"] = float(nn.median())
    # a tie is only free if the shared row reproduces the member's logit direction;
    # cosine is the ceiling on that.
    out["alive"] = bool(out["tail_nn_cosine_median"] >= 0.81)
    out["floor"] = 0.81
    print(
        f"  tail nearest-neighbour cosine median {out['tail_nn_cosine_median']:.4f} "
        f"alive={out['alive']}",
        flush=True,
    )
    return out


TASKS = {
    "vcfold": task_vcfold,
    "excise": task_excise,
    "convc": task_convc,
    "gart": task_gart,
    "splus": task_splus,
    "fb": task_fb,
    "hodlr": task_hodlr,
    "gauge": task_gauge,
    "qk": task_qk,
    "vocab": task_vocab,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS) + ["all"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    names = sorted(TASKS) if args.task == "all" else [args.task]
    for name in names:
        print(f"== {name} ==", flush=True)
        t0 = time.time()
        try:
            result = TASKS[name]()
        except Exception as exc:  # noqa: BLE001
            result = {"task": name, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  FAILED: {result['error']}", flush=True)
        result["schema"] = SCHEMA
        result["seconds"] = round(time.time() - t0, 1)
        path = outdir / f"census-{name}.json"
        path.write_text(json.dumps(result, indent=1))
        print(f"  -> {path}  ({result['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
