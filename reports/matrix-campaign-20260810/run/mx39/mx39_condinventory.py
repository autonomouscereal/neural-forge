#!/usr/bin/env python3
"""A2: which sealed children were built STALE, read out of the receipts.

Wave 32 answered B4 with a controlled pair: `ARCH.CL1792S` is `ARCH.CL1792`
rebuilt with `--arm stale` at byte-identical storage, the identical solver, the
identical evidence budget and the identical K, and it reads **4/15 against
7/15**.  Sequentially conditioned capture is worth three cases, and that is the
largest single-mechanism gain the campaign has measured.

So the question this file answers is the obvious next one and nobody has asked
it: **is anything on or near the frontier standing on a stale build?**  If a
solve-bearing child was captured off the unfolded parent, rebuilding it
conditioned is three cases of headroom for one build.

**It is read, not remembered.**  Each cell receipt carries three independent
witnesses and this file records all three rather than trusting one:

* `evidence.conditioned` -- the boolean the builder wrote;
* `architecture.axes[].evidence` -- prose, either "sequentially conditioned
  capture through the partially built child" or "the parent's own stream
  (stale)";
* `architecture.axes[].heldout.max_input_drift` -- **the physical fingerprint**,
  and the one that cannot be mistyped.  A stale capture reads the parent's
  stream at every site, so the input drift versus the parent is EXACTLY 0.0; a
  conditioned capture on this family reads 0.17-0.27.  Wave 32's build log is
  its own receipt for that and this is the same quantity, per axis, on every
  sealed row in the campaign.

A disagreement between the three is reported as a disagreement.  A child whose
`solve.solver_path` is null solved nothing at all -- a vocabulary gather, an
excision -- so conditioning is not applicable to its own build; it inherits its
source body's axes verbatim, which is why those rows still carry a drift number
and why they are counted separately rather than as stale.

Usage::

    mx39_condinventory.py --matrix /mnt/ai_data/neural-forge-matrix --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HELD = ("Useful", "Partial")
STALE_MARKERS = ("stale", "parent's own stream", "unfolded parent")
#: `conditioned capture` on its own is WAVES 16 AND 17's WORDING and the first
#: run of this file did not carry it, so forty-eight solve-bearing cells came
#: back unclassified -- including ARCH.X13A at 13/15, the campaign's best
#: solve-bearing row.  The record was complete; the marker list was not.  The
#: repair is this line and the third witness below, and the miss is recorded
#: here because an inventory that silently under-reports its own coverage is
#: worse than one that refuses.
CONDITIONED_MARKERS = (
    "sequentially conditioned",
    "partially built child",
    "conditioned capture",
)


def classify_axis(axis: Any, conditioned_flag: Any = None) -> dict[str, Any]:
    # Receipts from the campaign's earlier waves carry `axes` entries whose
    # `heldout` is a STRING rather than the per-site mapping the later waves
    # write, and one wave writes the axis itself as a bare string.  An inventory
    # that walks 160-odd receipts across 28 waves meets every schema the campaign
    # has ever had, so the shape is checked rather than assumed and an
    # unreadable axis is reported `unstated` instead of crashing the run.
    if not isinstance(axis, dict):
        return {"axis": str(axis)[:60], "evidence_prose": "", "max_input_drift": None,
                "arm_by_prose": "unstated", "arm_by_drift": "unstated",
                "arm_by_flag": "unstated", "witnesses_agree": True,
                "schema_note": "axis is not a mapping"}
    prose = str(axis.get("evidence") or "")
    low = prose.lower()
    heldout = axis.get("heldout")
    drift = heldout.get("max_input_drift") if isinstance(heldout, dict) else None
    by_prose = ("stale" if any(m in low for m in STALE_MARKERS)
                else "conditioned" if any(m in low for m in CONDITIONED_MARKERS)
                else "unstated")
    by_drift = ("stale" if drift == 0.0
                else "conditioned" if isinstance(drift, (int, float)) and drift > 0.0
                else "unstated")
    # THE THIRD WITNESS.  `evidence.conditioned` is a per-CHILD boolean rather
    # than a per-axis one, so it is the weakest of the three and is used only
    # where the other two are silent -- but it is present on every receipt the
    # campaign has ever written, which the axis prose is not.
    by_flag = ("conditioned" if conditioned_flag is True
               else "stale" if conditioned_flag is False else "unstated")
    spoken = [w for w in (by_prose, by_drift, by_flag) if w != "unstated"]
    return {
        "axis": axis.get("axis"),
        "evidence_prose": prose[:90],
        "max_input_drift": drift,
        "arm_by_prose": by_prose,
        "arm_by_drift": by_drift,
        "arm_by_flag": by_flag,
        # The flag is EXCLUDED from the agreement test on purpose: a zero-solve
        # composition inherits its body's axis prose and carries
        # `conditioned: false` for its own build, so flag-versus-prose disagrees
        # on every vocabulary gather in the campaign and would drown the two
        # disagreements that would actually mean something.
        "witnesses_agree": by_prose == by_drift or "unstated" in (by_prose, by_drift),
        "classified": bool(spoken),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", type=Path, default=Path("/mnt/ai_data/neural-forge-matrix"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    for receipt_path in sorted(args.matrix.glob("*/cells/ARCH.*/cell-receipt.json")) + \
            sorted(args.matrix.glob("cells/ARCH.*/cell-receipt.json")):
        doc = json.loads(receipt_path.read_text(encoding="utf-8"))
        arch = doc.get("architecture") or {}
        evidence = doc.get("evidence") or {}
        solve = doc.get("solve") or {}
        cases_path = receipt_path.parent / "per-case.json"
        sealed = None
        if cases_path.is_file():
            cases = json.loads(cases_path.read_text(encoding="utf-8"))
            sealed = sum(1 for v in cases.values() if v in HELD)
        axes = [classify_axis(a, evidence.get("conditioned"))
                for a in (arch.get("axes") or [])]
        arms = {a["arm_by_drift"] for a in axes} | {a["arm_by_prose"] for a in axes}
        rows.append({
            "cell_id": doc.get("cell_id"),
            "wave": doc.get("campaign_wave"),
            "tag": arch.get("tag"),
            "sealed_15": sealed,
            "net_ratio": (arch.get("storage") or {}).get("net_ratio"),
            "solver_path": solve.get("solver_path"),
            "solve_bearing": solve.get("solver_path") is not None,
            "conditioned_flag": evidence.get("conditioned"),
            "parent_forward_calls": evidence.get("parent_model_forward_calls"),
            "axes": axes,
            "any_axis_stale": any(a["arm_by_drift"] == "stale"
                                  or a["arm_by_prose"] == "stale" for a in axes),
            "classified": bool(axes) and all(a.get("classified") for a in axes),
            "arms_seen": sorted(arms),
        })

    solve_bearing = [r for r in rows if r["solve_bearing"]]
    zero_solve = [r for r in rows if not r["solve_bearing"]]
    stale_solve_bearing = [r for r in solve_bearing if r["any_axis_stale"]]
    disagreements = [
        {"cell_id": r["cell_id"], "axes": [a for a in r["axes"] if not a["witnesses_agree"]]}
        for r in rows if any(not a["witnesses_agree"] for a in r["axes"])
    ]
    solving_children_with_cases = [r for r in solve_bearing if r["sealed_15"] is not None]

    body = {
        "schema": "neural-forge-matrix-wave39-conditioning-inventory-v1",
        "purpose": (
            "A2.  Every sealed cell's build arm, read from three independent witnesses "
            "in its own receipt: the builder's boolean, the axis prose, and the "
            "measured max_input_drift, which is exactly 0.0 on a stale capture."
        ),
        "cells_read": len(rows),
        "solve_bearing": len(solve_bearing),
        "zero_solve": len(zero_solve),
        "solve_bearing_with_a_stale_axis": len(stale_solve_bearing),
        "solve_bearing_classified": sum(1 for r in solve_bearing if r["classified"]),
        "solve_bearing_unclassifiable": sorted(
            r["cell_id"] for r in solve_bearing if not r["classified"]),
        "stale_rows": sorted(
            [{k: r[k] for k in ("cell_id", "wave", "tag", "sealed_15", "net_ratio",
                                "conditioned_flag", "arms_seen")}
             for r in stale_solve_bearing],
            key=lambda r: -(r["sealed_15"] or -1)),
        "witness_disagreements": disagreements,
        "solve_bearing_ranked": sorted(
            [{"cell_id": r["cell_id"], "wave": r["wave"], "sealed_15": r["sealed_15"],
              "net_ratio": r["net_ratio"], "any_axis_stale": r["any_axis_stale"],
              "conditioned_flag": r["conditioned_flag"],
              "drift": [a["max_input_drift"] for a in r["axes"]]}
             for r in solving_children_with_cases],
            key=lambda r: (-(r["sealed_15"] or -1), r["net_ratio"] or 1.0)),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(body, indent=1), encoding="utf-8", newline="")

    print(f"cells read {len(rows)}  solve-bearing {len(solve_bearing)}  "
          f"zero-solve {len(zero_solve)}")
    print(f"solve-bearing rows CLASSIFIED by at least one witness: "
          f"{body['solve_bearing_classified']} of {len(solve_bearing)}")
    if body["solve_bearing_unclassifiable"]:
        print("  unclassifiable:", ", ".join(body["solve_bearing_unclassifiable"][:12]))
    print(f"solve-bearing rows with a STALE axis: {len(stale_solve_bearing)}")
    for r in body["stale_rows"]:
        print(f"  {r['cell_id']:<26} wave {r['wave']}  sealed {r['sealed_15']}  "
              f"ratio {r['net_ratio']}  arms {r['arms_seen']}")
    print(f"witness disagreements: {len(disagreements)}")
    for d in disagreements[:10]:
        print("  ", d["cell_id"], d["axes"])
    print("\nsolve-bearing children by sealed count:")
    for r in body["solve_bearing_ranked"][:25]:
        print(f"  {r['cell_id']:<26} sealed {str(r['sealed_15']):>4}  "
              f"ratio {str(r['net_ratio'])[:8]:>8}  stale={r['any_axis_stale']}  "
              f"drift {[round(d, 4) if isinstance(d, float) else d for d in r['drift']]}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
