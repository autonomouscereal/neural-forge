#!/usr/bin/env python3
"""Wave-5 status table: what a cell stores, what the chain said, what exists.

Reads receipts only, because the ledger shards are not merged until a batch is
complete and a half-swept wave still has to be readable.  Prints the ADDENDUM 6
columns first -- the storage bound and whether a review dossier exists yet --
and the pre-addendum chain metrics beside them as the diagnostics they now are.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/ai_data/neural-forge-matrix/cells"))
    parser.add_argument("--wave", type=int, default=5)
    arguments = parser.parse_args()

    rows = []
    for receipt_path in sorted(arguments.root.rglob("cell-receipt.json")):
        body = json.loads(receipt_path.read_text(encoding="utf-8"))
        if int(body.get("campaign_wave", 0)) != arguments.wave:
            continue
        solve = body["solve"]
        directory = receipt_path.parent
        dossier = directory / "frozen-review-dossier.json"
        verdict = "-"
        score = "-"
        reviewed = directory / "cell-receipt-behavior-reviewed.json"
        if reviewed.is_file():
            block = json.loads(reviewed.read_text(encoding="utf-8")).get("addendum6", {})
            verdict = str(block.get("verdict"))
            score = f"{block.get('useful_or_partial')}/15"
        rows.append(
            {
                "cell": body["cell_id"],
                "level": body["level"],
                "site": body["site"],
                "stored": solve.get("stored_parameter_count"),
                "ratio": solve.get("storage_realized_ratio"),
                "bound": solve.get("storage_bound_satisfied"),
                "width": solve.get("storage_level_width"),
                "token": body["behavior"]["free_run"]["token_match_vs_control"],
                "drift": body["composed"].get("vs_copy_exact_control_delta"),
                "floor": body["composed"].get("emit_zero_floor_passed"),
                "writecos": body["boundary"].get("write_heldout_cos"),
                "link": body["outcome"]["first_divergent_metric"],
                "status": body["outcome"]["status"],
                "dossier": "yes" if dossier.is_file() else "no",
                "verdict": verdict,
                "score": score,
                "seconds": round(float(body["wall_time_seconds"]), 1),
                "path": str(directory),
            }
        )
    header = (
        f"{'cell':<18} {'stored/parent':>13} {'bnd':>4} {'w':>5} "
        f"{'token':>7} {'drift':>7} {'flr':>4} {'wcos':>7} {'link':<16} "
        f"{'doss':<5} {'verdict':<19} {'score':<6} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        ratio = "--" if row["ratio"] is None else f"{row['ratio']:.6f}"
        bound = {True: "ok", False: "MISS", None: "--"}[row["bound"]]
        wcos = "--" if row["writecos"] is None else f"{row['writecos']:.4f}"
        drift = "--" if row["drift"] is None else f"{row['drift']:.4f}"
        print(
            f"{row['cell']:<18} {ratio:>13} {bound:>4} {row['width'] or '--':>5} "
            f"{row['token']:>7.4f} {drift:>7} {str(row['floor'])[:4]:>4} {wcos:>7} "
            f"{str(row['link']):<16} {row['dossier']:<5} {row['verdict']:<19} "
            f"{row['score']:<6} {row['seconds']:>6}"
        )
    print(f"\n{len(rows)} wave-{arguments.wave} receipts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
