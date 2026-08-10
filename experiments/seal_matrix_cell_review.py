#!/usr/bin/env python3
"""Seal the frozen decode-smoke review of one matrix cell.

`docs/MATRIX_CAMPAIGN_DESIGN_2026-08-10.md` §5 rule 1 makes three things
jointly necessary for a cell to PASS: free-run token match >= 0.90, all three
decode-smoke outputs judged coherent-and-on-task under the frozen review rubric,
and the emit-zero wreck detector.  The first and third are measured by the cell
runner.  The second is a human judgement of the verbatim text in the sealed
receipt, and it cannot exist at the moment the cell is sealed.

So it is sealed here instead, as a **separate instance** of the same cell:

* the runner can never certify its own behaviour -- it has no path to write
  anything but ``pending`` into ``outcome.decode_smoke_review``;
* the reviewed receipt records the digest of the receipt it was read from, so a
  verdict is always traceable to the exact text it graded;
* the ledger stays append-only -- the unreviewed instance remains, and the
  reviewed one is a new row, exactly as §3 requires of any re-run.

Usage::

    python experiments/seal_matrix_cell_review.py \
        --receipt reports/.../CTRL.copy_exact/cell-receipt.json \
        --verdict coherent --reviewer "cyril" \
        --notes "3/3 on task; the dinner plan hit the cap and is censored" \
        --ledger reports/matrix-campaign-20260810/ledger.jsonl

Cap-hit outputs are censored, not failed (the GOAL censoring rule): judge the
observed prefix, and if the prefix is on task the output is coherent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    append_ledger,
    seal_review,
    write_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--verdict", choices=("coherent", "incoherent"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print the verbatim decode-smoke text and exit without sealing.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    if args.show:
        for index, entry in enumerate(receipt["behavior"]["decode_smoke"], start=1):
            print(f"--- smoke {index} [{entry['terminated']}] {entry['prompt_sha'][:12]}")
            print(entry["output_verbatim"])
            print()
        return 0
    sealed = seal_review(
        receipt,
        verdict=args.verdict,
        reviewer=args.reviewer,
        notes=args.notes,
    )
    destination = (
        Path(args.out)
        if args.out is not None
        else Path(args.receipt).with_name("cell-receipt-reviewed.json")
    )
    write_receipt(destination, sealed)
    ledger = (
        Path(args.ledger)
        if args.ledger is not None
        else Path(args.receipt).parents[1] / "ledger.jsonl"
    )
    append_ledger(ledger, sealed)
    print(
        f"sealed review of {sealed['cell_id']} verdict={args.verdict} "
        f"status={sealed['outcome']['status']} receipt={sealed['receipt_sha'][:12]}"
    )
    print(f"receipt -> {destination}")
    print(f"ledger  -> {ledger}")
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
