#!/usr/bin/env python3
"""Compute the ADDENDUM 8 success flag and frontier from sealed receipts only.

ADDENDUM 8 §8.2 defines success as "every folded component at 4x+ AND retained
coherence", which in the ADDENDUM 6 verdict vocabulary is exactly

    success  <=>  verdict in {alive-parent-band, alive-degraded, impaired}
    failure  <=>  verdict in {dead, cannot-express-4x}

so the flag needs no new receipt field: it is a total function of
`addendum6.verdict` and the per-component storage bound, both already sealed.
This script computes it and REFUSES to accept one, for the same reason
`behavior_verdict_for` refuses a hand-supplied verdict -- a success criterion a
reviewer can assert is a success criterion a reviewer can move after seeing the
answer.

The storage half is re-derived here rather than trusted: every component's
sealed stored/owned counts are re-checked against the bound its declared level
implies, and a child with even one component over the bound is NOT a success at
any coherence, because the 4x floor is Cyril's hard rule and not a preference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-63e95cd8")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_cell_receipt import STORAGE_LEVEL_RATIOS  # noqa: E402

ROOT = Path("/mnt/ai_data/neural-forge-matrix")
COHERENT_VERDICTS = {"alive-parent-band", "alive-degraded", "impaired"}


def evaluate(wave: str, cell: str) -> dict | None:
    path = ROOT / wave / "cells" / cell / "cell-receipt-behavior-reviewed.json"
    if not path.is_file():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    block = receipt["addendum6"]
    composition = receipt["composition"]
    verdict = str(block["verdict"])

    violations = []
    worst = 0.0
    for entry in composition["components"]:
        level = str(entry["level"])
        bound = STORAGE_LEVEL_RATIOS.get(level)
        stored = int(entry["storage"]["stored_parameter_count"])
        owned = int(entry["storage"]["owned_parameter_count"])
        realized = stored / owned
        worst = max(worst, realized)
        if bound is None:
            violations.append(f"{entry['isolation_cell_id']}: level {level} has no S bound")
        elif stored > int(bound * owned):
            violations.append(
                f"{entry['isolation_cell_id']}: {realized:.6f} > {bound}"
            )

    # ADDENDUM 9 9.1 says the PER-LANE reading governs the frontier.  NOTES-MX-O
    # 7a refused to make this calculator prefer that reading while the reading
    # was only a sentence in a note, and it was right to: the fix it specified is
    # a SEALED per-lane verdict computed from per-lane readings the reviewer
    # supplies case by case, which is what `addendum9.per_lane_verdict` now is.
    # So the flag is computed from it WHEN IT EXISTS, and from the per-child
    # verdict when the review named no lanes -- which is every wave-1..11 row,
    # so no earlier row changes meaning.
    lane_block = receipt.get("addendum9") or {}
    per_lane = lane_block.get("per_lane_verdict")
    governing = str(per_lane) if per_lane else verdict
    coherent = governing in COHERENT_VERDICTS
    return {
        "per_child_verdict": verdict,
        "per_lane_verdict": per_lane,
        "degeneracy_reading": lane_block.get("degeneracy_reading", "unstated"),
        "governing_verdict": governing,
        "readings_disagree": bool(lane_block.get("readings_disagree", False)),
        "wave": wave,
        "cell": cell,
        "verdict": verdict,
        "useful_or_partial": int(block["useful_or_partial"]),
        "parent": int(block["parent_useful_or_partial"]),
        "folded": len(composition["components"]),
        "net_ratio": float(composition["storage"]["net_ratio"]),
        # ADDENDUM 10: a repaired rung is compared at FULL price.  The number is
        # read straight out of the sealed storage block, which already carries
        # the corrective term, and reported separately so a reader can see how
        # much of the net ratio is additive tensors the parent does not have.
        "corrective_stored": int(
            composition["storage"].get("corrective_stored_parameter_count", 0) or 0
        ),
        "corrective_routes": int(
            (composition.get("corrections") or {}).get("route_count", 0) or 0
        ),
        "removed": int(composition["storage"]["parent_parameter_count"])
        - int(composition["storage"]["child_parameter_count"]),
        "worst_component_stored_over_owned": worst,
        "storage_bound_violations": violations,
        "coherent": coherent,
        "addendum8_success": bool(coherent and not violations),
        "install_order": composition["install_order"],
        "global_drift": composition["global_drift"]["terminal_relL2_vs_control"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", action="append", required=True, metavar="WAVE:CELL")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()

    rows = []
    for spec in arguments.cell:
        wave, _, cell = spec.partition(":")
        row = evaluate(wave, cell)
        if row is None:
            print(f"!! no sealed review receipt for {spec}", file=sys.stderr)
            continue
        rows.append(row)

    if arguments.json:
        print(json.dumps(rows, indent=1, sort_keys=True))
        return 0

    # ADDENDUM 8 §8.4: rank coherent children by net stored ratio, deepest first;
    # accuracy is the tie-break and is printed beside every row, never instead.
    rows.sort(key=lambda r: (not r["addendum8_success"], r["net_ratio"]))
    print(
        f"{'child':<22}{'n':>3}{'net ratio':>12}{'removed':>12}{'corr':>10}"
        f"{'worst s/o':>11}{'review':>8}{'per-child':>18}{'per-lane':>18}{'A8':>6}"
    )
    for row in rows:
        print(
            f"{row['cell']:<22}{row['folded']:>3}{row['net_ratio']:>12.6f}"
            f"{row['removed']:>12,}{row['corrective_stored']:>10,}"
            f"{row['worst_component_stored_over_owned']:>11.6f}"
            f"{str(row['useful_or_partial']) + '/15':>8}"
            f"{row['per_child_verdict']:>18}"
            f"{str(row['per_lane_verdict'] or 'unstated'):>18}"
            f"{'PASS' if row['addendum8_success'] else 'fail':>6}"
        )
        for violation in row["storage_bound_violations"]:
            print(f"    STORAGE BOUND VIOLATED: {violation}")
    successes = [r for r in rows if r["addendum8_success"]]
    print()
    print(
        "the A8 flag reads the PER-LANE verdict wherever the review sealed one "
        "(ADDENDUM 9 9.1) and the per-child verdict otherwise; both are printed"
    )
    for row in rows:
        if row["readings_disagree"]:
            print(
                f"    {row['cell']}: the readings DISAGREE -- "
                f"{row['degeneracy_reading']} degeneracy, "
                f"{row['per_child_verdict']} per child, "
                f"{row['per_lane_verdict']} per lane"
            )
    print(f"ADDENDUM 8 successes: {len(successes)} of {len(rows)}")
    if successes:
        deepest = min(successes, key=lambda r: r["net_ratio"])
        best = max(successes, key=lambda r: r["useful_or_partial"])
        print(
            f"deepest coherent  : {deepest['cell']} at {deepest['net_ratio']:.6f}x "
            f"({deepest['folded']} components, {deepest['useful_or_partial']}/15, "
            f"{deepest['verdict']})"
        )
        print(
            f"best accuracy     : {best['cell']} at {best['net_ratio']:.6f}x "
            f"({best['useful_or_partial']}/15, {best['verdict']})"
        )
    print(
        "every component of every row above stores at or below its declared S bound"
        if not any(r["storage_bound_violations"] for r in rows)
        else "AT LEAST ONE ROW VIOLATES THE 4x FLOOR -- see above"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
