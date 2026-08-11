#!/usr/bin/env python3
"""Per-SITE solve conditioning and install cost across every wave-7 child.

ADDENDUM 7 opened sites the campaign had never solved at.  The question this
table answers is the mechanical one the wave was asked to record: **is the
solve at an unexplored depth ill-conditioned before the install, and does that
predict the behaviour?**  Every number is read from the sealed receipts; nothing
is recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/wave7/cells")

def _num(value, places: int, width: int) -> str:
    if value is None:
        return "n/a".rjust(width) if width else "n/a"
    return f"{float(value):>{width}.{places}f}" if width else f"{float(value):.{places}f}"


rows = []
for directory in sorted(ROOT.iterdir()):
    receipt = directory / "cell-receipt.json"
    reviewed = directory / "cell-receipt-behavior-reviewed.json"
    if not receipt.is_file():
        continue
    body = json.loads(receipt.read_text(encoding="utf-8"))
    verdict = score = None
    if reviewed.is_file():
        block = json.loads(reviewed.read_text(encoding="utf-8")).get("addendum6") or {}
        verdict = block.get("verdict")
        score = block.get("useful_or_partial")
    composition = body["composition"]
    for index, entry in enumerate(composition["components"]):
        solve = entry["solve"]
        storage = entry["storage"]
        after = entry["after_install"]
        rows.append(
            {
                "child": body["cell_id"].replace("COMP.MA.", ""),
                "order": index,
                "cell": entry["isolation_cell_id"],
                "site": entry["site"],
                "level": entry["level"],
                "width": entry.get("level_width"),
                "stored": storage["stored_parameter_count"],
                "owned": storage["owned_parameter_count"],
                "ratio": storage["stored_parameter_count"] / storage["owned_parameter_count"],
                "condition": solve.get("condition_number"),
                "saturated": solve.get("condition_cap_saturated"),
                "ridge": solve.get("ridge_relative"),
                "gram": solve.get("gram_relative"),
                "cosine": solve.get("cosine"),
                "rank": solve.get("rank"),
                "drift": after.get("terminal_heldout_relL2")
                if after.get("terminal_heldout_relL2") is not None
                else after.get("global_terminal_relL2"),
                "free_run": after.get("token_match_vs_control")
                if after.get("token_match_vs_control") is not None
                else after.get("free_run_token_match"),
                "measured_before": entry.get("isolation", {}).get("measured_at_this_site"),
                "verdict": verdict,
                "score": score,
            }
        )

print(
    f"{'child':<8}{'#':<3}{'cell':<16}{'lvl':<4}{'width':>6}{'stored/owned':>14}"
    f"{'cond':>12}{'gram':>9}{'cos':>9}{'drift':>10}{'free':>8}  {'verdict':<18}{'score'}"
)
for row in rows:
    cond = row["condition"]
    cond_text = "n/a" if cond is None else f"{float(cond):.4g}"
    gram = "n/a" if row["gram"] is None else f"{float(row['gram']):.5f}"
    cosine = "n/a" if row["cosine"] is None else f"{float(row['cosine']):.5f}"
    print(
        f"{row['child']:<8}{row['order']:<3}{row['cell']:<16}{row['level']:<4}"
        f"{str(row['width']):>6}{row['ratio']:>14.6f}{cond_text:>12}{gram:>9}{cosine:>9}"
        f"{_num(row['drift'], 6, 10)}{_num(row['free_run'], 4, 8)}  "
        f"{str(row['verdict']):<18}{row['score']}"
    )

print()
print("=== per-site FIRST-INSTALL conditioning, one line per (cell id, level) ===")
seen: dict[tuple[str, str], dict] = {}
for row in rows:
    key = (row["cell"], row["level"])
    if key not in seen or row["order"] < seen[key]["order"]:
        seen[key] = row
for (cell, level), row in sorted(seen.items()):
    cond = row["condition"]
    cond_text = "n/a" if cond is None else f"{float(cond):.6g}"
    print(
        f"{cell:<16}{level:<4}  cond={cond_text:<14}"
        f"ridge={row['ridge']}  stored/owned={row['ratio']:.6f}  "
        f"first-install drift={_num(row['drift'], 6, 0).strip()}  "
        f"had prior isolation cell: {row['measured_before']}"
    )
