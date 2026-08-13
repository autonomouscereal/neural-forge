#!/usr/bin/env python3
"""Wave 25, pre-read P2: the SEGBASIS cut census, over all thirteen cuts.

The slate's S2 asks for a depth-segmented residual basis -- one basis per depth
segment instead of one global basis serving all twenty-nine readers -- and names
a free pre-read as its first step: ``J(c)`` over all 13 cuts, per-endpoint
retained read energy printed rather than pooled-only, from captures rather than
from new builds.  Waves 23 and 24 both scheduled it and neither ran it.  This
runs it, and it is a census: **it orders, it never gates** (campaign rule 3).

### Why it is one capture pass and not thirteen

FIX 1 in the slate pins segment A's pool as the endpoint-equalized pool over ALL
twenty-nine endpoints, where each POST-cut endpoint enters as its reader moment
*restricted to the pre-cut residual*.  Written out, a post-cut endpoint ``i``
reads ``diag(g_i) n``, so restricting it to the residual as it exists at the cut
means substituting the cut's normalized residual ``n_c`` for its own::

    M_i|c  =  diag(g_i) [ sum_rows n_c n_c^T ] diag(g_i)  =  diag(g_i) N_c diag(g_i)

and ``N_c`` -- the GAIN-FREE second moment of the normalized residual at the cut
-- does not depend on ``i``.  So one pass that accumulates ``N_e`` beside the
usual reader moment ``M_e`` at every endpoint gives every cut's restricted pool
in closed form.  Thirteen cuts cost thirteen eigendecompositions, not thirteen
capture passes.  ``N_e`` is exactly ``M_e`` at unit gain, which is also why the
degeneracy the slate demands is visible here: at ``c`` past the last block there
are no post-cut endpoints and the pool is the sealed global one, entry for entry.

### The seam, and what the census can and cannot say about it

Segment B's basis is free up to an orthogonal gauge inside its own span, and the
seam passes the accumulator through UNCHANGED (that is what "zero added tensors"
means -- there is no transition tensor to store).  So the gauge is chosen to make
the pass-through as close to the identity as the two spans allow::

    min over orthogonal Q of  sum_rows || E_B Q E_A^T r  -  r ||^2_(seam)

whose first term is constant because ``Q`` is orthogonal, leaving an orthogonal
Procrustes problem: ``Q = U V^T`` from ``SVD(E_B^T M_seam E_A)``.  The residual
cost is reported as ``seam_identity_relative_cost`` beside the retention gain and
is NEVER netted against it -- one is an energy a reader keeps and the other is an
error a stream carries, and adding them would be inventing a currency.

What this census CANNOT say is whether any of it buys a review case.  Wave 22's
MM896 cleared every under-served reader and lost two cases.  L1 stands: only the
build answers.

Usage::

    mx25_segcensus.py --width 896 --device cuda:0 --out <path>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import mx25_build as B


def log(message: str) -> None:
    print(f"[mx25seg] {message}", flush=True)


@torch.no_grad()
def capture(harness: Any) -> dict[str, Any]:
    """Per endpoint and split: the reader moment AND the gain-free normal moment.

    ``mx25_build.residual_moments`` accumulates ``raw`` and ``reader``; this needs
    a third, ``normal`` -- the normalized residual's own second moment, which is
    the reader moment at unit gain -- because every restricted moment in the
    census is that matrix conjugated by some other endpoint's gain.  Rather than
    widen the shared function and change what every other wave's census means,
    the third accumulator lives here.
    """

    ends = B.residual_endpoints(harness.model)
    hidden = int(harness.model.config.hidden_size)
    eps = float(harness.model.config.norm_eps)
    gains = {
        name: module.weight.data.to(torch.float64).cpu() for name, module in ends.items()
    }
    acc = {
        name: {
            split: {
                "reader": torch.zeros(hidden, hidden, dtype=torch.float64),
                "normal": torch.zeros(hidden, hidden, dtype=torch.float64),
                "rows": 0,
            }
            for split in ("fit", "hld")
        }
        for name in ends
    }
    state: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, kwargs):
            state[name] = (args[0] if args else kwargs["hidden_states"]).detach()
            return None

        return hook

    for name, module in ends.items():
        handles.append(
            module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
        )
    try:
        for index, ids in harness.batches:
            split = "fit" if harness.split[index] == "fit" else "hld"
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
            harness.forward_calls += 1
            for name in ends:
                r = state[name].reshape(-1, hidden).to("cpu", torch.float64)
                normed = r * r.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
                bucket = acc[name][split]
                bucket["normal"] += normed.T @ normed
                read = normed * gains[name]
                bucket["reader"] += read.T @ read
                bucket["rows"] += int(r.shape[0])
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {"endpoints": tuple(ends), "hidden": hidden, "gains": gains, "moments": acc}


def equalized(blocks: list[torch.Tensor]) -> torch.Tensor:
    """Endpoint-equalized pool: every contributor enters at trace one."""

    hidden = int(blocks[0].shape[0])
    pooled = torch.zeros(hidden, hidden, dtype=torch.float64)
    for block in blocks:
        trace = float(torch.diagonal(block).sum())
        if trace <= 0.0:
            raise SystemExit("zero-trace moment in the pool")
        pooled += block / trace
    return pooled / len(blocks)


def top_basis(pooled: torch.Tensor, width: int) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(pooled)
    order = torch.argsort(values, descending=True)[:width]
    basis = vectors[:, order].contiguous()
    for column in range(basis.shape[1]):  # deterministic signs, as every basis has
        pivot = int(basis[:, column].abs().argmax())
        if float(basis[pivot, column]) < 0.0:
            basis[:, column] = -basis[:, column]
    return basis


def retained(moment: torch.Tensor, basis: torch.Tensor) -> float:
    total = float(torch.diagonal(moment).sum())
    return float((basis.T @ moment @ basis).diagonal().sum() / total)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=896)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    started = time.time()
    torch.set_grad_enabled(False)
    harness = B.Harness(B.PARENT, args.device, torch.bfloat16)
    log("capturing the residual stream at 29 endpoints (reader and normal moments)")
    cap = capture(harness)
    ends = list(cap["endpoints"])
    gains = cap["gains"]
    width = int(args.width)
    layers = (len(ends) - 1) // 2
    log(f"{layers} blocks, {len(ends)} endpoints, width {width}")

    def reader(name: str, split: str) -> torch.Tensor:
        return cap["moments"][name][split]["reader"]

    def normal(name: str, split: str) -> torch.Tensor:
        return cap["moments"][name][split]["normal"]

    # The sealed global object, rebuilt here so every number below is measured
    # against a basis this script derived from these same rows.
    global_pool = equalized([reader(n, "fit") for n in ends])
    global_basis = top_basis(global_pool, width)
    global_per_endpoint = {
        split: {n: retained(reader(n, split), global_basis) for n in ends}
        for split in ("fit", "hld")
    }

    rows: list[dict[str, Any]] = []
    for cut in range(1, layers):
        pre = [n for n in ends if n.startswith("B") and int(n[1:3]) < cut]
        post = [n for n in ends if n not in pre]
        seam_endpoint = f"B{cut:02d}.op"
        results: dict[str, Any] = {"cut": cut, "seam_endpoint": seam_endpoint}
        for split in ("fit", "hld"):
            seam_normal = normal(seam_endpoint, split)
            # FIX 1: through-traffic pinned.  A post-cut reader enters segment A's
            # pool as ITS OWN gain applied to the residual AT THE CUT.
            restricted = [
                gains[n][:, None] * seam_normal * gains[n][None, :] for n in post
            ]
            basis_a = top_basis(
                equalized([reader(n, split) for n in pre] + restricted), width
            )
            basis_b = top_basis(equalized([reader(n, split) for n in post]), width)
            # The seam metric is the same object segment A is asked to preserve.
            seam_metric = equalized(restricted)
            u, _, vt = torch.linalg.svd(basis_b.T @ seam_metric @ basis_a)
            gauge = u @ vt
            basis_b_gauged = basis_b @ gauge
            transfer = basis_b_gauged @ basis_a.T
            delta = transfer - torch.eye(int(basis_a.shape[0]), dtype=torch.float64)
            seam_cost = float(
                ((delta @ seam_metric) * delta).sum()
                / float(torch.diagonal(seam_metric).sum())
            )
            angles = torch.linalg.svdvals(basis_b.T @ basis_a).clamp(-1.0, 1.0)
            segmented = {n: retained(reader(n, split), basis_a) for n in pre}
            segmented.update(
                {n: retained(reader(n, split), basis_b_gauged) for n in post}
            )
            base = global_per_endpoint[split]
            worst_seg = min(segmented, key=segmented.get)
            worst_glb = min(base, key=base.get)
            results[split] = {
                "segmented_min": segmented[worst_seg],
                "segmented_argmin": worst_seg,
                "segmented_mean": sum(segmented.values()) / len(segmented),
                "global_min": base[worst_glb],
                "global_argmin": worst_glb,
                "global_mean": sum(base.values()) / len(base),
                "delta_min": segmented[worst_seg] - base[worst_glb],
                "delta_mean": (
                    sum(segmented.values()) - sum(base.values())
                ) / len(base),
                "endpoints_below_0_90_segmented": sorted(
                    k for k, v in segmented.items() if v < 0.90
                ),
                "endpoints_below_0_90_global": sorted(
                    k for k, v in base.items() if v < 0.90
                ),
                "seam_identity_relative_cost": seam_cost,
                "seam_principal_angle_cos_min": float(angles.min()),
                "seam_principal_angle_cos_mean": float(angles.mean()),
                "per_endpoint_segmented": segmented,
            }
        rows.append(results)
        log(
            f"cut {cut:2d} @ {seam_endpoint}: fit min {results['fit']['segmented_min']:.4f} "
            f"(global {results['fit']['global_min']:.4f}, delta "
            f"{results['fit']['delta_min']:+.4f})  seam cost "
            f"{results['fit']['seam_identity_relative_cost']:.4f}"
        )

    best = max(rows, key=lambda r: r["fit"]["delta_min"])
    report = {
        "schema": "neural-forge-matrix-wave25-segbasis-cut-census-v1",
        "pre_read": "P2",
        "purpose": (
            "SEGBASIS J(c) over all admissible cuts, per-endpoint retained READ "
            "energy printed rather than pooled-only, from one capture pass.  "
            "Orders the cut; gates nothing (campaign rule 3)."
        ),
        "width": width,
        "blocks": layers,
        "endpoints": ends,
        "parent_forward_calls": harness.forward_calls,
        "global_basis": {
            "mode": "reader-pca over the endpoint-equalized pool of all 29 readers",
            "per_endpoint": global_per_endpoint,
        },
        "cuts": rows,
        "argmax_cut_by_fit_delta_min": best["cut"],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    log(f"census -> {args.out}; argmax cut by fit delta_min is {best['cut']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
