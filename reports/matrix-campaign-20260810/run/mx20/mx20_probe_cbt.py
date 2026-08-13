#!/usr/bin/env python3
"""CBT -- block-Toeplitz tied ridge, the never-built slate class §A6.

**What it is, and why it is the one class the truncation graves cannot answer.**
Every fold this campaign has built at a linear site is a RANK statement: keep k
directions, throw the rest away.  Wave 14 located a cliff in per-site write
cosine, wave 19 found the cliff is really in retained energy, and both are
statements about a subspace.  CBT does not truncate at all.  It constrains the
child's weight to a WEIGHT-TYING linear subspace -- the blocks of a block
partition are tied by a pinned rule -- and the result is generically **full
rank**.  It is the first family offered to the sites whose graves are full of
rank truncations.

**The pinned rule, stated before anything is solved.**  Partition the 1024x1024
`attn_o` at L12 into an 8x8 grid of 128x128 blocks and tie by

    class(I, J) = I - J          (block-Toeplitz, 15 base classes)

with IDENTITY permutations (the level-1 arm; adaptive co-clustering permutations
are an ablation, not this cell).  Fifteen classes of 128x128 is 245,760 numbers
plus a 64-entry class map = 245,824 against the parent's 1,048,576, which is
**4.266x** with every map counted and no permutation storage to count.

**The solve is exact and it decouples.**  With ``y = x W'^T`` and
``W'[I, J] = B_{I-J}``, the loss separates over the output index *within* a
block: for a fixed row ``p`` of every ``B_k``, the prediction at output block
``I`` is ``sum_k X_{I-k} B_k[p, :]^T``, so the unknowns for different ``p`` never
meet.  That gives 128 independent (15*128)-dimensional normal systems which
share ONE Gram, so it is one Cholesky and 128 right-hand sides.  Better: both
the Gram and the right-hand side are assembled from the sealed input Gram alone,

    G[(k,q),(k',q')] = sum_I  Gram[(I-k)*128+q, (I-k')*128+q']
    R[(k,q), p]      = sum_I  (Gram @ W^T)[(I-k)*128+q, I*128+p]

so no target matrix is ever materialised and the whole class costs one capture.

**The prior is the parent's own tied projection** -- each class's blocks
averaged -- so THE LAW's trust region is measured against the best tying of the
parent that involves no evidence at all, and the ridge is the campaign's
production prior-anchored form.

**Status: PROBE, research-only, and the reason is arithmetic rather than
taste.**  The installed tensor is DENSE.  A stock `Lfm2ForCausalLM` loads this
child and counts 229,693,184 parameters -- **1.000000x** -- because the loader
counts what it reads and the 4.266x lives in the fact that 245,824 numbers
determine the tensor.  R3 counts the loader.  So this cell is a behavioural
measurement of the tying constraint and never a frontier claim, and both numbers
are reported side by side rather than the flattering one alone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx20_build as mx  # noqa: E402

SCHEMA = "neural-forge-matrix-wave20-cbt-probe-v1"
#: The block edge.  8x8 grid over 1024 with 15 Toeplitz classes is §A6's own
#: `attn_o` C09 arithmetic, and it is pinned here rather than swept.
BLOCK = 128


def toeplitz_classes(grid: int) -> list[int]:
    """The base classes of the pinned rule, in a fixed order: ``I - J``."""

    return list(range(-(grid - 1), grid))


def assemble(
    gram: torch.Tensor, weight: torch.Tensor, grid: int, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tied normal equations and the parent's tied projection, from the Gram.

    Returns ``(G, R, prior)`` with ``G`` the shared (c*b, c*b) Gram of the tied
    design, ``R`` its (c*b, b) right-hand side, and ``prior`` the parent's own
    blocks averaged within each class -- the tying of the parent that uses no
    evidence, which is what THE LAW's update scale is measured against.
    """

    classes = toeplitz_classes(grid)
    span = len(classes) * block
    cross = gram @ weight.T  # (in, out): the x^T y of the local target
    G = torch.zeros(span, span, dtype=torch.float64)
    R = torch.zeros(span, block, dtype=torch.float64)
    prior = torch.zeros(span, block, dtype=torch.float64)
    for a, k in enumerate(classes):
        rows_a = [I for I in range(grid) if 0 <= I - k < grid]
        for b_index, k2 in enumerate(classes):
            rows_b = [I for I in range(grid) if 0 <= I - k2 < grid]
            shared = sorted(set(rows_a) & set(rows_b))
            if not shared:
                continue
            acc = torch.zeros(block, block, dtype=torch.float64)
            for I in shared:
                u, v = (I - k) * block, (I - k2) * block
                acc += gram[u : u + block, v : v + block]
            G[a * block : (a + 1) * block, b_index * block : (b_index + 1) * block] = acc
        acc_r = torch.zeros(block, block, dtype=torch.float64)
        acc_p = torch.zeros(block, block, dtype=torch.float64)
        for I in rows_a:
            u = (I - k) * block
            acc_r += cross[u : u + block, I * block : (I + 1) * block]
            # `weight` is (out, in): output block I, input block I-k.
            acc_p += weight[I * block : (I + 1) * block, u : u + block].T
        R[a * block : (a + 1) * block] = acc_r
        prior[a * block : (a + 1) * block] = acc_p / len(rows_a)
    return G, R, prior


def materialise(solution: torch.Tensor, grid: int, block: int) -> torch.Tensor:
    """Expand the tied classes back into the dense (out, in) weight."""

    classes = toeplitz_classes(grid)
    out = torch.zeros(grid * block, grid * block, dtype=torch.float64)
    for a, k in enumerate(classes):
        piece = solution[a * block : (a + 1) * block]  # (in-block, out-block)
        for I in range(grid):
            J = I - k
            if 0 <= J < grid:
                out[I * block : (I + 1) * block, J * block : (J + 1) * block] = piece.T
    return out


def tied_solve(
    G: torch.Tensor, R: torch.Tensor, prior: torch.Tensor, grid: int, block: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The lambda ladder on the tied system, with THE LAW as the veto it is."""

    reference = float(materialise(prior, grid, block).norm())
    scale_of = lambda sol: float(  # noqa: E731
        (materialise(sol, grid, block) - materialise(prior, grid, block)).norm()
        / reference
    )
    mean_diagonal = float(torch.diagonal(G).mean())
    free = None
    for rel in mx.LAMBDA_LADDER:
        lam = rel * mean_diagonal
        system = G + lam * torch.eye(G.shape[0], dtype=torch.float64)
        candidate = torch.cholesky_solve(
            R + lam * prior, torch.linalg.cholesky(system)
        )
        if free is None:
            free = scale_of(candidate)
        scale = scale_of(candidate)
        if scale <= 1.0:
            return candidate, {
                "update_scale": scale,
                "lambda_relative": rel,
                "free_solve_update_scale": free,
            }
    raise SystemExit(
        f"no lambda on the ladder brings update_scale to 1.0 (free {free:.3f}); "
        "THE LAW refuses the install"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--child", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--receipt", type=Path, required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    started = time.time()
    parent = mx.Harness(mx.PARENT, args.device, torch.bfloat16)
    config = json.loads((mx.PARENT / "config.json").read_text())
    if config["layer_types"][args.layer] != "full_attention":
        raise SystemExit(f"L{args.layer:02d} is not an attention block")
    site = parent.layers()[args.layer].self_attn.out_proj
    if getattr(site, "bias", None) is not None:
        raise SystemExit("this out_proj carries a bias; the tied solve does not")

    mx.log(f"capturing attn_o input at L{args.layer:02d}")
    captured = parent.capture({"o": site})
    X_fit, X_hld = captured["o"]["fit"], captured["o"]["hld"]
    weight = site.weight.data.to(torch.float64).cpu()
    grid = weight.shape[0] // BLOCK
    if grid * BLOCK != weight.shape[0] or weight.shape[0] != weight.shape[1]:
        raise SystemExit(f"attn_o is {tuple(weight.shape)}; the pinned grid wants square")

    gram = X_fit.T @ X_fit
    G, R, prior = assemble(gram, weight, grid, BLOCK)
    solution, guard = tied_solve(G, R, prior, grid, BLOCK)
    tied = materialise(solution, grid, BLOCK)
    projection = materialise(prior, grid, BLOCK)

    target_fit = X_fit @ weight.T
    target_hld = X_hld @ weight.T
    scores = {
        "solved": mx.write_scores(X_hld @ tied.T, target_hld),
        "tied_projection_only": mx.write_scores(X_hld @ projection.T, target_hld),
        "fit_solved": mx.write_scores(X_fit @ tied.T, target_fit),
    }
    stored = len(toeplitz_classes(grid)) * BLOCK * BLOCK + grid * grid
    mx.log(
        f"tied ridge L{args.layer:02d}: held-out cos {scores['solved']['cos']:.4f} "
        f"(tied projection alone {scores['tied_projection_only']['cos']:.4f}), "
        f"update_scale {guard['update_scale']:.3f}, rank "
        f"{int(torch.linalg.matrix_rank(tied))} of {tied.shape[0]}"
    )

    site.weight.data.copy_(tied.to(device=site.weight.device, dtype=site.weight.dtype))
    live = mx.export(
        parent,
        Path(args.child),
        keep_ff=None,
        keep_head_dim=None,
        keep_groups=None,
        keep_hidden=None,
        excised=[],
        rope_theta=float(config["rope_parameters"]["rope_theta"]),
    )
    receipt = {
        "schema": SCHEMA,
        "class": "CBT",
        "spec": "docs/FOLD_CLASS_CANDIDATES_2026-08-12.md section A6",
        "status": "probe",
        "export_class": "standard-export",
        "research_only": True,
        "research_only_reason": (
            "the installed tensor is dense, so the loader counts the parent's own "
            "parameter count and R3 counts the loader; the 4.266x is a STORAGE "
            "ratio (245,824 numbers determine the tensor) and is reported as one"
        ),
        "site": {
            "component": "C09 attn_o",
            "layer": args.layer,
            "shape": list(weight.shape),
            "grid": grid,
            "block": BLOCK,
            "permutations": "identity (level-1 arm)",
            "tying_rule": "class(I,J) = I - J",
            "base_classes": len(toeplitz_classes(grid)),
            "merged_classes": len(toeplitz_classes(grid)),
        },
        "storage": {
            "parent_numbers": int(weight.numel()),
            "child_numbers_stored": int(stored),
            "storage_ratio": float(weight.numel()) / float(stored),
            "loader_counted_parameters": int(live),
            "loader_counted_ratio": float(live) / float(mx.PARENT_PARAMS),
        },
        "rank": {
            "installed": int(torch.linalg.matrix_rank(tied)),
            "parent": int(torch.linalg.matrix_rank(weight)),
            "full": int(tied.shape[0]),
        },
        "solve": guard,
        "heldout": scores,
        "rows": {"fit": int(X_fit.shape[0]), "held_out": int(X_hld.shape[0])},
        "parent_forward_calls": parent.forward_calls,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": round(time.time() - started, 1),
        "child_tensors_sha256": mx.file_sha256(Path(args.child) / "model.safetensors"),
    }
    Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
    Path(args.receipt).write_text(json.dumps(receipt, indent=1))
    mx.log(
        f"CBT probe -> {args.child} ({live:,} loader-counted parameters, "
        f"storage {receipt['storage']['storage_ratio']:.3f}x at the site)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
