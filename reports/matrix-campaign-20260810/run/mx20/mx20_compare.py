#!/usr/bin/env python3
"""The wave's central table: one row per child, with the variable that moved.

Wave 20's children are unusually comparable to each other and to wave 19's.  At
768 the parameter count is IDENTICAL across five of them -- 165,978,624,
0.722610x -- because the only thing that changed is which 768-dimensional
subspace of the residual the child keeps, or how the writers chose their
regularisation, or whether they were re-solved at all.  That makes the review
score a controlled measurement, and it makes a table that puts the retained
energies beside the score the right way to read the wave.

Reads sealed receipts and sealed verdicts where they exist and build receipts
where they do not, so it can be run mid-wave.  Nothing here scores anything.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

HELD = ("Useful", "Partial")


def verdict_of(cell: Path) -> tuple[str, int, int]:
    """The ADDENDUM 6 reading, if this child has been graded."""

    sealed = cell / "cell-receipt-behavior-reviewed.json"
    per_case = cell / "per-case.json"
    if not sealed.is_file():
        return ("ungraded", -1, -1)
    body = json.loads(sealed.read_text())
    held = -1
    if per_case.is_file():
        cases = json.loads(per_case.read_text())
        held = sum(1 for v in cases.values() if v in HELD)
    behaviour = body.get("addendum6") or body.get("behavior_review") or {}
    verdict = str(behaviour.get("verdict", body.get("status", "?")))
    return (verdict, held, int(behaviour.get("behavior_useful_or_partial", held)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/ai_data/neural-forge-matrix/wave20")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    root = Path(args.root)

    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(f"{root}/children/*/mx20-build-receipt.json")):
        receipt = json.load(open(path, encoding="utf-8"))
        tag = os.path.basename(os.path.dirname(path)).replace("MX20-", "")
        width = next(
            (a for a in receipt["axes"] if a["axis"] == "hidden_size"), None
        )
        if width is None:
            continue
        basis = width["basis"]
        selector = basis.get("selector_retained") or {}
        verdict, held, sealed_held = verdict_of(root / "cells" / f"ARCH.{tag}")
        rows.append(
            {
                "tag": tag,
                "arm": receipt["arm"],
                "width": basis["width"],
                "basis_mode": basis["mode"],
                "beta": (basis.get("objective") or {}).get("beta"),
                "basis_sha256": (basis.get("sha256") or "")[:12],
                "lambda_split": bool(receipt["axes"][0].get("width_lambda_split")),
                "net_ratio": receipt["net_ratio"],
                "parameters": receipt["child_parameters"],
                "retained_reader_hld": (basis.get("retained") or {}).get("reader.hld"),
                "retained_raw_hld": (basis.get("retained") or {}).get("raw.hld"),
                "retrieval_reader_hld": selector.get("retrieval.reader.hld"),
                "procedural_reader_hld": selector.get("procedural.reader.hld"),
                # `--arm stale` installs the closed form and re-solves nothing,
                # so it has no per-site record and these are legitimately absent.
                "median_cos": width["heldout"].get("median_cos"),
                "min_cos": width["heldout"].get("min_cos"),
                "max_drift": width["heldout"].get("max_input_drift"),
                "verdict": verdict,
                "held_cases": held,
            }
        )

    header = (
        f"{'child':<11}{'H':>5}{'basis':>22}{'beta':>6}{'arm':>7}{'lam':>5}"
        f"{'net':>10}{'read.hld':>10}{'retr':>8}{'proc':>8}{'medcos':>8}"
        f"{'drift':>7}{'review':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (-r["width"], r["tag"])):
        print(
            f"{row['tag']:<11}{row['width']:>5}{row['basis_mode']:>22}"
            f"{('-' if row['beta'] is None else f'{row["beta"]:.3g}'):>6}"
            f"{row['arm'].replace('conditioned-', 'c-'):>7}"
            f"{('yes' if row['lambda_split'] else '-'):>5}"
            f"{row['net_ratio']:>10.6f}"
            f"{(row['retained_reader_hld'] or 0):>10.4f}"
            f"{(row['retrieval_reader_hld'] or 0):>8.4f}"
            f"{(row['procedural_reader_hld'] or 0):>8.4f}"
            f"{(row['median_cos'] if row['median_cos'] is not None else float('nan')):>8.4f}"
            f"{(row['max_drift'] if row['max_drift'] is not None else float('nan')):>7.3f}"
            f"{(f'{row["held_cases"]}/15' if row['held_cases'] >= 0 else '-'):>8}"
        )
    if args.out is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {
                    "schema": "neural-forge-matrix-wave20-child-comparison-v1",
                    "children": rows,
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
