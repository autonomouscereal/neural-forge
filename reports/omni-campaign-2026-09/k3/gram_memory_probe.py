#!/usr/bin/env python3
"""Measure the C14 ridge's peak RSS, old code path vs new, at a real width.

Blocker 4 was an arithmetic claim ("~205 GB peak at K = 32,768, not the ~103 GB
the gap list quoted").  Arithmetic is how the number got wrong in the first
place, so this runs both implementations in SEPARATE processes on the same
synthetic Gram problem and reports each one's `VmHWM`.

  --variant old   `gram + lam * torch.eye(n)`, three accumulated Grams, all
                  nine ladder solutions held at once -- i.e. exactly what
                  `law_abiding_solve_gram` did before this pass.
  --variant new   the shipped `c14_union_coreset.law_abiding_solve_gram`.

Both produce the same W (the runner checks that too), so the difference is
memory and nothing else.  Width is a knob because the effect is quadratic:
K = 8,192 already separates them by ~1.5 GB on a box where K = 65,536 would
separate them by ~120 GB.

    python gram_memory_probe.py --width 8192 --out-dim 512
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

from c14_union_coreset import LAMBDA_LADDER, _check_sse, law_abiding_solve_gram
from k3_common import peak_rss_bytes


def problem(width: int, out_dim: int, tokens: int, seed: int):
    """A well-conditioned Gram pair, built in chunks so the design never exists."""
    generator = torch.Generator().manual_seed(seed)
    gram_inner = torch.zeros(width, width, dtype=torch.float64)
    gram_check = torch.zeros_like(gram_inner)
    cross_inner = torch.zeros(width, out_dim, dtype=torch.float64)
    cross_check = torch.zeros_like(cross_inner)
    check_y_sq = 0.0
    chunk = 256
    for start in range(0, tokens, chunk):
        rows = min(chunk, tokens - start)
        design = torch.randn(rows, width, generator=generator, dtype=torch.float64)
        target = torch.randn(rows, out_dim, generator=generator, dtype=torch.float64)
        if (start // chunk) % 3 == 2:
            gram_check += design.T @ design
            cross_check += design.T @ target
            check_y_sq += float(target.pow(2).sum())
        else:
            gram_inner += design.T @ design
            cross_inner += design.T @ target
        del design, target
    gram_inner.diagonal().add_(float(tokens))  # keep the Cholesky honest
    gram_check.diagonal().add_(float(tokens))
    prior = (torch.randn(out_dim, width, generator=generator, dtype=torch.float64) * 0.05).float()
    return gram_inner, cross_inner, gram_check, cross_check, check_y_sq, prior


def old_prior_anchored(gram, cross, prior, rel):
    """The pre-fix ridge, verbatim: a dense identity plus a summed copy."""
    lam = rel * float(torch.diagonal(gram).mean())
    regularised = gram + lam * torch.eye(gram.shape[0], dtype=gram.dtype)
    rhs = cross + lam * prior.T.to(gram.dtype)
    factor = torch.linalg.cholesky(regularised)
    return torch.cholesky_solve(rhs, factor).T.contiguous()


def old_solver(gram_inner, cross_inner, gram_check, cross_check, check_y_sq, prior):
    """The pre-fix ladder: a third accumulated Gram and nine stored solutions."""
    gram_all = gram_inner + gram_check
    cross_all = cross_inner + cross_check
    reference = float(prior.norm())
    free = old_prior_anchored(gram_all, cross_all, prior, LAMBDA_LADDER[0])
    free_scale = float((free - prior).norm() / reference)
    rungs = []
    for rel in LAMBDA_LADDER:
        inner = old_prior_anchored(gram_inner, cross_inner, prior, rel)
        full = old_prior_anchored(gram_all, cross_all, prior, rel)
        rungs.append({
            "lambda_relative": rel,
            "check_sse": _check_sse(inner, gram_check, cross_check, check_y_sq),
            "update_scale": float((full - prior).norm() / reference),
            "solved": full,
        })
    admitted = [row for row in rungs if row["update_scale"] <= 1.0]
    if not admitted:
        raise SystemExit(f"no admitted lambda (free {free_scale:.3f})")
    chosen = min(admitted, key=lambda row: row["check_sse"])
    return chosen["solved"], {"lambda_relative": chosen["lambda_relative"],
                              "update_scale": chosen["update_scale"]}


def run_variant(args) -> int:
    parts = problem(args.width, args.out_dim, args.tokens, args.seed)
    baseline = peak_rss_bytes()
    # NEITHER variant gets a defensive copy.  Cloning the inputs for the new
    # path only (an earlier version of this probe did) hands it a whole extra
    # Gram pair and turns a 2x result into a 1.07x one -- the measurement has to
    # compare the solvers, not one solver plus a copy.  `law_abiding_solve_gram`
    # consumes its inputs by contract and nothing here needs them afterwards.
    if args.variant == "old":
        solved, info = old_solver(*parts)
    else:
        solved, info = law_abiding_solve_gram(*parts)
    peak = peak_rss_bytes()
    payload = {
        "variant": args.variant,
        "width": args.width,
        "out_dim": args.out_dim,
        "rss_after_build_gigabytes": round(baseline / 1e9, 3) if baseline else None,
        "peak_rss_gigabytes": round(peak / 1e9, 3) if peak else None,
        "lambda_relative": info["lambda_relative"],
        "update_scale": round(float(info["update_scale"]), 8),
        "solution_sha_proxy": round(float(solved.double().norm()), 8),
    }
    print(json.dumps(payload))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--out-dim", type=int, default=3584,
                    help="K3's real routed_expert_hidden_size; the stored-solution term "
                         "scales with it and is half of what the old path wasted")
    ap.add_argument("--tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--variant", default=None, choices=("old", "new"),
                    help="internal: run one variant in this process")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.variant:
        return run_variant(args)

    results = {}
    for variant in ("old", "new"):
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--width", str(args.width),
             "--out-dim", str(args.out_dim), "--tokens", str(args.tokens),
             "--seed", str(args.seed), "--variant", variant],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(f"{variant} variant failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        results[variant] = json.loads(proc.stdout.strip().splitlines()[-1])

    same = (results["old"]["lambda_relative"] == results["new"]["lambda_relative"]
            and abs(results["old"]["solution_sha_proxy"] - results["new"]["solution_sha_proxy"]) < 1e-6)
    summary = {
        "same_answer": bool(same),
        "old_peak_rss_gigabytes": results["old"]["peak_rss_gigabytes"],
        "new_peak_rss_gigabytes": results["new"]["peak_rss_gigabytes"],
        "saved_gigabytes": round(results["old"]["peak_rss_gigabytes"] - results["new"]["peak_rss_gigabytes"], 3),
        "ratio": round(results["old"]["peak_rss_gigabytes"] / max(results["new"]["peak_rss_gigabytes"], 1e-9), 3),
        "detail": results,
        "note": ("measured VmHWM of two processes solving the same problem; the ratio is what "
                 "the K = 65,536 rung's feasibility turns on and it is quadratic in width"),
    }
    print(json.dumps(summary, indent=1))
    if not same:
        raise SystemExit("the two variants disagreed on the answer; the comparison is meaningless")
    if args.out:
        from k3_common import write_receipt

        write_receipt(args.out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
