#!/usr/bin/env python3
"""Wave 25's free pre-reads: P3 (NKP rearrangement spectrum) and P6 (ATTR stage A).

Both are pure arithmetic -- one over the checkpoint's own bytes, one over sealed
per-case readings the campaign already owns -- and both **order attempts and
gate nothing** (L1, the standing clause; see docs/PHASE1 section 2.10 for the four
times a census would have picked the wrong child, in both directions).

**P3, the rearrangement spectrum.**  Van Loan-Pitsianis: for a grid
``(m1 x m2) (x) (n1 x n2)`` the best sum of ``r`` Kronecker terms is the rank-``r``
truncated SVD of the *rearranged* matrix ``R(W)``, which is ``(m1 n1) x (m2 n2)``.
The point of measuring it is that **rearrangement rank and matrix rank are
independent**: this parent's channel-space spectra are flat, which is what killed
every low-rank lane, and that says nothing about the never-measured
rearrangement spectrum.  Storage is ``r (m1 n1 + m2 n2)``, which is what the 4x
bound is solved against.

The **Monarch control** runs through the identical census: partition ``W`` into a
grid of blocks and truncate each block's own SVD, at the SAME parameter budget.
It is the closest structural relative of what is known to fold here, and a class
that cannot beat blockwise SVD at matched storage is not a class.

What P3 as run here does NOT do: the evidence-weighted selection
``tr[(W - What) M_x (W - What)^T]``.  That needs the sealed ffn-input moment,
which is a capture, and the capture is not free.  The unweighted spectrum is the
load-bearing half -- a flat rearrangement spectrum is a grave whatever the
weighting -- and the omission is named rather than papered over.

**P6, ATTR stage A.**  ``D(block, c)`` is the case-by-block excision-survival
matrix: for each single-block excision child the campaign sealed, which of the
fifteen cases it held.  It is read off `per-case.json` and nothing else.  The
width half of the original ATTR proposal is **unidentified** -- no sealed row
varies one width writer -- and is not computed here, which is why the standalone
wave was merged rather than funded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
MATRIX = Path("/mnt/ai_data/neural-forge-matrix")
#: The fifteen graded lanes, in the campaign's own difficulty order (docs/PHASE1
#: section 2.12).  The sixteenth (safety) is charged but not scored.
CASES = (
    "summarization-priorities",
    "uncertainty-insufficient-data",
    "writing-concise-email",
    "instruction-three-actions",
    "context-retrieval",
    "planning-dependencies",
    "conversation-natural-greeting",
    "creative-continuity",
    "comparison-ram-rom",
    "debugging-off-by-one",
    "multi-turn-correction",
    "arithmetic-multi-step",
    "coding-deduplicate",
    "logic-ordering",
    "explanation-seasons",
)
#: A3's default priority vector, in force until Cyril rules otherwise
#: (ADDENDUM 15 section 15.2).  It is his stated want and it is labelled as a
#: default everywhere it is consumed.
A3_DEFAULT_PRIORITY = {
    "conversation-natural-greeting": 3.0,
    "multi-turn-correction": 3.0,
    "instruction-three-actions": 3.0,
    "explanation-seasons": 0.25,
    "logic-ordering": 0.25,
}


def log(message: str) -> None:
    print(f"[mx25pre] {message}", flush=True)


# --------------------------------------------------------------------- P3
def rearrange(weight: Any, m1: int, m2: int, n1: int, n2: int) -> Any:
    """``R(W)[(i1,j1), (i2,j2)] = W[i1 m2 + i2, j1 n2 + j2]`` -- Van Loan's map."""

    import torch

    m, n = weight.shape
    if m1 * m2 != m or n1 * n2 != n:
        raise SystemExit(f"grid ({m1}x{m2})(x)({n1}x{n2}) does not tile {m}x{n}")
    view = weight.reshape(m1, m2, n1, n2)
    return view.permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)


def monarch_blockwise(weight: Any, rows: int, cols: int, rank: int) -> tuple[float, int]:
    """Exact blockwise-SVD projection: relative Frobenius error and its storage."""

    import torch

    m, n = weight.shape
    br, bc = m // rows, n // cols
    total_error = 0.0
    stored = 0
    for i in range(rows):
        for j in range(cols):
            block = weight[i * br : (i + 1) * br, j * bc : (j + 1) * bc].to(torch.float64)
            u, s, vh = torch.linalg.svd(block, full_matrices=False)
            k = min(rank, int(s.shape[0]))
            kept = s[:k]
            total_error += float((s[k:] ** 2).sum())
            stored += k * (br + bc)
    denominator = float((weight.to(torch.float64) ** 2).sum())
    return math.sqrt(total_error / denominator) if denominator else 0.0, stored


def p3(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(PARENT / "model.safetensors"))
    config = json.loads((PARENT / "config.json").read_text())
    layer_types = config["layer_types"]
    hidden = int(config["hidden_size"])
    inner = int(config["intermediate_size"])

    # Admissible grids for w1/w3 (2560 x 1024) and w2 (1024 x 2560).  A grid is
    # admissible when both factor pairs tile the shape; the ones below are the
    # square-ish factorizations, which are the ones with a storage advantage.
    grids_tall = [(40, 64, 32, 32), (64, 40, 32, 32), (80, 32, 32, 32), (32, 80, 32, 32)]
    grids_wide = [(32, 32, 40, 64), (32, 32, 64, 40), (32, 32, 80, 32), (32, 32, 32, 80)]

    rows: list[dict[str, Any]] = []
    for layer, kind in enumerate(layer_types):
        for leaf, grids in (
            ("feed_forward.w1.weight", grids_tall),
            ("feed_forward.w3.weight", grids_tall),
            ("feed_forward.w2.weight", grids_wide),
        ):
            name = f"model.layers.{layer}.{leaf}"
            if name not in tensors:
                continue
            weight = tensors[name].to(torch.float64)
            budget = int(weight.numel()) // 4
            best: dict[str, Any] | None = None
            for m1, m2, n1, n2 in grids:
                rearranged = rearrange(weight, m1, m2, n1, n2)
                spectrum = torch.linalg.svdvals(rearranged)
                energy = float((spectrum**2).sum())
                per_term = m1 * n1 + m2 * n2
                r = max(1, budget // per_term)
                r = min(r, int(spectrum.shape[0]))
                retained = float((spectrum[:r] ** 2).sum()) / energy
                # the whole point of the census: is the spectrum concentrated?
                top1 = float(spectrum[0] ** 2) / energy
                half = int(spectrum.shape[0]) // 2
                row = {
                    "grid": f"({m1}x{m2})(x)({n1}x{n2})",
                    "rearranged_shape": list(rearranged.shape),
                    "params_per_term": per_term,
                    "r_at_4x": r,
                    "terms_available": int(spectrum.shape[0]),
                    "retained_energy_at_4x": retained,
                    "top_term_energy_fraction": top1,
                    "energy_in_top_half": float((spectrum[:half] ** 2).sum()) / energy,
                }
                if best is None or row["retained_energy_at_4x"] > best["retained_energy_at_4x"]:
                    best = row
            assert best is not None
            # matched-storage controls: plain rank truncation and Monarch
            m, n = weight.shape
            rank_at_4x = budget // (m + n)
            spectrum = torch.linalg.svdvals(weight)
            plain = float((spectrum[:rank_at_4x] ** 2).sum()) / float((spectrum**2).sum())
            monarch_error, monarch_stored = monarch_blockwise(
                weight, args.monarch_rows, args.monarch_cols, args.monarch_rank
            )
            rows.append(
                {
                    "tensor": name,
                    "layer": layer,
                    "layer_type": kind,
                    "shape": [int(m), int(n)],
                    "budget_at_4x": budget,
                    "kronecker_best": best,
                    "rank_truncation_at_4x": {
                        "rank": int(rank_at_4x),
                        "retained_energy": plain,
                    },
                    "monarch_blockwise": {
                        "grid": f"{args.monarch_rows}x{args.monarch_cols}",
                        "rank_per_block": args.monarch_rank,
                        "stored": monarch_stored,
                        "storage_ratio_vs_dense": monarch_stored / float(weight.numel()),
                        "relative_frobenius_error": monarch_error,
                        "retained_energy": 1.0 - monarch_error**2,
                    },
                }
            )
            log(
                f"{name}: kron {best['grid']} r={best['r_at_4x']} retains "
                f"{best['retained_energy_at_4x']:.4f}; rank-{rank_at_4x} retains "
                f"{plain:.4f}; monarch retains {1.0 - monarch_error**2:.4f} at "
                f"{monarch_stored / float(weight.numel()):.4f}x"
            )
    wins = sum(
        1
        for r in rows
        if r["kronecker_best"]["retained_energy_at_4x"] > r["rank_truncation_at_4x"]["retained_energy"]
    )
    return {
        "pre_read": "P3",
        "schema": "neural-forge-matrix-wave25-nkp-rearrangement-census-v1",
        "purpose": "the never-measured rearrangement spectrum of the SwiGLU "
        "stack at the 4x storage bound, with a matched-storage rank-truncation "
        "twin and a Monarch blockwise-SVD control.  Orders; gates nothing.",
        "not_run": "the evidence-weighted selection tr[(W-What) M_x (W-What)^T] "
        "needs the sealed ffn-input moment, which is a capture and is not free; "
        "the unweighted spectrum is reported instead and the omission is named",
        "hidden_size": hidden,
        "intermediate_size": inner,
        "sites": len(rows),
        "kronecker_beats_rank_truncation_at_4x": wins,
        "rows": rows,
    }


# --------------------------------------------------------------------- P6
def p6(args: argparse.Namespace) -> dict[str, Any]:
    """The case x block excision-survival matrix, read off sealed per-case files."""

    survival: dict[int, dict[str, str]] = {}
    sources: dict[int, str] = {}
    for path in sorted(MATRIX.glob("wave*/cells/ARCH.X*/per-case.json")):
        tag = path.parent.name.replace("ARCH.", "")
        # `X<nn>` is the excision with the sealed local re-solve and `X<nn>A` is
        # the same child under wave 15's naming.  Nothing else qualifies:
        # `X05T` is the TRAJECTORY arm, a different mechanism that wave 15
        # priced at 0/15, and a suffix-stripping rule that let it in would
        # silently overwrite block 5's real reading with it -- which the first
        # run of this census did, printing block 5 at 0/15 against `X05A`'s
        # sealed 6/15.  `X13KV6`, `X07KV6JQC` and the rest are compositions.
        if not (
            len(tag) in (3, 4)
            and tag[0] == "X"
            and tag[1:3].isdigit()
            and (len(tag) == 3 or tag[3] == "A")
        ):
            continue
        block = int(tag[1:3])
        body = json.loads(path.read_text())
        rows = body.get("cases", body) if isinstance(body, dict) else body
        if isinstance(rows, dict):
            verdicts = {str(k): str(v) for k, v in rows.items()}
        else:
            verdicts = {str(r["case"]): str(r["verdict"]) for r in rows}
        if block in survival:
            raise SystemExit(
                f"block {block} is claimed twice: {sources[block]} and {path}.  "
                "Two sealed readings for one excision is a choice this census "
                "must not make silently."
            )
        survival[block] = verdicts
        sources[block] = str(path)

    held = {
        block: {c: v for c, v in verdicts.items() if v.lower() in ("useful", "partial")}
        for block, verdicts in survival.items()
    }
    blocks = sorted(survival)
    matrix: list[dict[str, Any]] = []
    for case in CASES:
        row = {
            "case": case,
            "held_by_block": {str(b): (case in held[b]) for b in blocks},
            "blocks_survived": sum(1 for b in blocks if case in held[b]),
            "blocks_measured": len(blocks),
        }
        row["survival_rate"] = (
            row["blocks_survived"] / row["blocks_measured"] if blocks else 0.0
        )
        matrix.append(row)

    # D(block, c) = 1 when excising `block` COSTS case c -- the damage the
    # allocation would be aimed at.  s_w = sum_c p_c D(block, c) with p the A3
    # default priority vector, labelled `default-pending-ruling` at its use site.
    damage = []
    for block in blocks:
        per_case = {c: (0 if c in held[block] else 1) for c in CASES}
        score = sum(A3_DEFAULT_PRIORITY.get(c, 1.0) * per_case[c] for c in CASES)
        damage.append(
            {
                "block": block,
                "cases_held": len(held[block]),
                "damage_unweighted": sum(per_case.values()),
                "damage_A3_weighted": score,
                "source": sources[block],
            }
        )
    damage.sort(key=lambda r: r["damage_A3_weighted"])
    return {
        "pre_read": "P6",
        "schema": "neural-forge-matrix-wave25-attr-stage-a-v1",
        "purpose": "the complete case x block excision-survival matrix, which is "
        "the ONLY identified half of the ATTR proposal.  Orders; gates nothing.",
        "unidentified_and_not_computed": "the width-writer half: no sealed row "
        "varies one width writer, so D(writer, c) is not estimable from anything "
        "on disk and inventing it would be a guess wearing a number's clothes",
        "priority_vector": {
            "status": "A3 default-pending-ruling",
            "weights": A3_DEFAULT_PRIORITY,
            "note": "cases not named carry weight 1.0",
        },
        "blocks_measured": blocks,
        "case_survival": matrix,
        "block_damage": damage,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-read", required=True, choices=["P3", "P6"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--monarch-rows", type=int, default=8)
    ap.add_argument("--monarch-cols", type=int, default=8)
    ap.add_argument("--monarch-rank", type=int, default=32)
    args = ap.parse_args()
    report = p3(args) if args.pre_read == "P3" else p6(args)
    text = json.dumps(report, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        log(f"{args.pre_read} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
