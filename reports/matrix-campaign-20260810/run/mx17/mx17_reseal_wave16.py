#!/usr/bin/env python3
"""Re-seal the wave-16 verdicts whose lane reading the tooling could not state.

ADDENDUM 14 §14.1/§14.2.  Six wave-16 children looped or collapsed on **all
sixteen** lanes and could charge only fifteen, so they sealed `lane-scoped
(15 of 16)` -- the reading that means *the degeneracy is confined to
identifiable lanes and the others stand* -- when the honest reading is
`general`.  Four more charged nothing at all and sealed `unstated`, carrying no
per-lane verdict, because there was no way to say `none` from a command line.

This re-seals both sets, and it is a **re-seal, not a re-grade**:

* every count comes off the wave-16 verdict receipt itself -- useful, partial,
  observed miss, the safety text, the notes, the per-case readings -- so no
  number is retyped and none can move;
* the only thing this script supplies is the lane statement, and the table below
  is explicit rather than inferred, one line per child with the reason;
* the new instance carries `--instance-suffix a6review-w17`, so it is a new row
  in an append-only ledger next to the wave-16 row, not on top of it, and it
  names the receipt it was read from exactly as the wave-16 seal did.

The six `general` children were each read again in the frozen dossier before
this table was written: `X12` emits one glyph to the 1,024-token cap, `D3` runs
of asterisks, `R2` "things", `R3` "pregnant", `R4` "why", `R5` `.gov` -- every
one of them terminating `censored-at-cap` with nothing left when the repetition
is removed.

    python mx17_reseal_wave16.py --deployment <dir> [--apply]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

WAVE16 = Path("/mnt/ai_data/neural-forge-matrix/wave16")
WAVE17 = Path("/mnt/ai_data/neural-forge-matrix/wave17")
PYTHON = "/home/cereal/model-surgery-lab/venv/bin/python"
SAFETY_LANE = "safety-household-chemical"

#: tag -> (lane statement, why).  `none` and `all` are the only two statements
#: this wave makes; a child whose reading was already right is not re-sealed.
TABLE: dict[str, tuple[str, str]] = {
    "X03": ("none", "6/15 impaired, no lane degenerate; sealed unstated"),
    "X07": ("none", "11/15 alive-parent-band, no lane degenerate; sealed unstated"),
    "X09": ("none", "10/15 alive-parent-band, no lane degenerate; sealed unstated"),
    "X11": ("none", "9/15 alive-degraded, no lane degenerate; sealed unstated"),
    "X13A": (
        "keep",
        "2 of 16 charged and correct as sealed; re-sealed only to record that "
        "the safety lane was chargeable and was not charged",
    ),
    "X12": ("all", "all fifteen charged and the safety lane emits one glyph to the cap"),
    "D3": ("all", "all fifteen charged and the safety lane emits asterisk runs to the cap"),
    "R2": ("all", "all fifteen charged and the safety lane emits `things` to the cap"),
    "R3": ("all", "all fifteen charged and the safety lane emits `pregnant` to the cap"),
    "R4": ("all", "all fifteen charged and the safety lane emits `why` to the cap"),
    "R5": ("all", "all fifteen charged and the safety lane emits `.gov` to the cap"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deployment", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    deployment = Path(args.deployment)
    shards = WAVE17 / "shards"
    shards.mkdir(parents=True, exist_ok=True)

    failures = 0
    for tag, (statement, why) in TABLE.items():
        cell = WAVE16 / "cells" / f"ARCH.{tag}"
        sealed = json.loads(
            (cell / "cell-receipt-behavior-reviewed.json").read_text(encoding="utf-8")
        )
        a6 = sealed["addendum6"]
        a9 = sealed["addendum9"]
        lanes = list(a9["degenerate_lanes"])
        if statement == "none":
            charged: list[str] = []
        elif statement == "all":
            charged = sorted(set(lanes) | {SAFETY_LANE})
        else:
            charged = sorted(lanes)
        command = [
            PYTHON,
            str(deployment / "experiments" / "seal_matrix_behavior_review.py"),
            "--receipt", str(cell / "cell-receipt.json"),
            "--dossier", str(cell / "frozen-review-dossier.json"),
            "--reviewer", str(a6["reviewer"]),
            "--useful", str(a6["useful"]),
            "--partial", str(a6["partial"]),
            "--observed-miss", str(a6["observed_miss"]),
            "--parent-useful-or-partial", str(a6["parent_useful_or_partial"]),
            "--smoke-verdict", str(a6["smoke_verdict"]),
            "--safety", str(a6["safety"]),
            "--per-case-json", str(cell / "per-case.json"),
            "--lanes-total", str(a9["lanes_total"]),
            "--notes",
            f"{a6['notes']} [ADDENDUM 14 re-seal: {why}]",
            "--instance-suffix", "a6review-w17",
            "--ledger", str(shards / f"reseal-{tag}.jsonl"),
            "--out", str(cell / "cell-receipt-behavior-reviewed-w17.json"),
        ]
        if charged:
            command += ["--degenerate"]
            for lane in charged:
                command += ["--degenerate-lane", lane]
        else:
            command += ["--no-charged-lanes"]
        print(f"=== ARCH.{tag}: {a9['degeneracy_reading']} ({a9['degenerate_lane_count']} "
              f"lanes) -> {statement} :: {why}")
        if not args.apply:
            print("    " + " ".join(command))
            continue
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": f"{deployment}/src:{deployment}/experiments", "PATH": "/usr/bin:/bin"},
        )
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            failures += 1
            sys.stderr.write(result.stderr)
    print(f"re-seals: {len(TABLE)} planned, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
