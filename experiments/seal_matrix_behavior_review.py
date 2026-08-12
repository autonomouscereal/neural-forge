#!/usr/bin/env python3
"""Seal ADDENDUM 6's behaviour verdict for one matrix cell.

ADDENDUM 6 §6.1 makes the frozen 15-prompt review of the **whole model, with
one folded component installed** the primary grade of a cell, and retires token
match to a diagnostic column.  That grade is a judgement of verbatim text, so
the same split §5 rule 1 forced on the decode-smoke verdict applies one level
up and for the same reason: a runner may not certify its own behaviour, and the
ledger is append-only, so the verdict is sealed here as a **new instance** of
the cell carrying the digest of the receipt it was read from.

What this script will not let a reviewer do:

* **Move the band after seeing the score.**  ``--verdict`` is optional; when it
  is omitted the verdict is computed by `behavior_verdict_for` from the counts,
  the smoke verdict, the wreck detector and the degeneracy flag.  When it *is*
  supplied it must equal the computed one, or the seal is refused.  The only
  verdict a reviewer states unilaterally is ``cannot-express-4x``, which is a
  statement about a technique's storage algebra and involves no text at all.
* **Grade text nobody can find.**  A behavioural verdict must name the frozen
  review dossier it was taken on; the dossier in turn pins the protocol digest,
  the prompt pack's byte-form and the reconstituted child's tensor digests.
  ``--dossier`` reads all three out of the file so they cannot be mistyped.
* **Report 15 cases out of a review that graded fewer.**  useful + partial +
  observed-miss must be exactly the surface's 15 intelligence cases; the safety
  case is reported separately and never contributes an intelligence failure.

Usage::

    python experiments/seal_matrix_behavior_review.py \
        --receipt   /mnt/.../cells/C13.T5.S4.L05/cell-receipt.json \
        --dossier   /mnt/.../cells/C13.T5.S4.L05/frozen-review-dossier.json \
        --reviewer  claude-opus5-mxi \
        --useful 5 --partial 6 --observed-miss 4 \
        --parent-useful-or-partial 12 \
        --smoke-verdict coherent --safety concerning \
        --per-case-json cases.json \
        --notes "..." --ledger /mnt/.../ledger.jsonl
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
    BEHAVIOR_VERDICTS,
    MatrixCellReceiptError,
    append_ledger,
    behavior_verdict_for,
    seal_behavior_review,
    write_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--dossier",
        type=Path,
        help=(
            "The frozen review dossier the grade was read from.  Required for "
            "every verdict but cannot-express-4x."
        ),
    )
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--useful", type=int)
    parser.add_argument("--partial", type=int)
    parser.add_argument("--observed-miss", type=int)
    parser.add_argument("--parent-useful-or-partial", type=int)
    parser.add_argument(
        "--smoke-verdict",
        choices=("pending", "coherent", "incoherent"),
        default="coherent",
    )
    parser.add_argument("--safety")
    parser.add_argument(
        "--degenerate",
        action="store_true",
        help=(
            "Any lane showed looping, collapse, tokenizer debris or unrelated "
            "script drift.  Forces the verdict to dead at any score."
        ),
    )
    parser.add_argument(
        "--verdict",
        choices=BEHAVIOR_VERDICTS,
        help=(
            "Optional.  Checked against the verdict the counts imply; supply "
            "it only to state cannot-express-4x, or to assert agreement."
        ),
    )
    parser.add_argument(
        "--per-case-json",
        type=Path,
        help="JSON object mapping case id -> Useful/Partial/Observed miss.",
    )
    parser.add_argument(
        "--degenerate-lane",
        action="append",
        default=None,
        metavar="CASE_ID",
        help=(
            "ADDENDUM 9 9.1, and NOTES-MX-O 7a's open item: the case id of one "
            "lane the degeneracy clause charges -- a lane that leaves nothing "
            "when its repetitions are removed.  Repeatable.  The per-lane "
            "verdict is COMPUTED from these readings; it is never supplied.  A "
            "review that names none seals `unstated` and claims no per-lane "
            "verdict, which is what every wave-1..11 review is."
        ),
    )
    parser.add_argument(
        "--lanes-total",
        type=int,
        default=16,
        help=(
            "How many lanes the review covered; the frozen surface is sixteen "
            "(fifteen graded plus safety).  Charging all of them is a GENERAL "
            "degeneracy and is death under either reading."
        ),
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))

    dossier_sha = protocol_sha = pack_sha = None
    if args.dossier is not None:
        dossier = json.loads(Path(args.dossier).read_text(encoding="utf-8"))
        dossier_sha = str(dossier["dossier_sha256"])
        protocol_sha = str(dossier["protocol"]["protocol_sha256"])
        pack_sha = str(dossier["protocol"]["prompt_pack_sha256"])
        if str(dossier["receipt_sha256"]) != str(receipt["receipt_sha"]):
            raise MatrixCellReceiptError(
                f"The dossier graded receipt {dossier['receipt_sha256'][:16]} "
                f"and this is {str(receipt['receipt_sha'])[:16]}; a verdict "
                "must be traceable to the exact text it graded."
            )

    if args.verdict == "cannot-express-4x":
        verdict = args.verdict
    else:
        if args.useful is None or args.partial is None or args.observed_miss is None:
            raise MatrixCellReceiptError(
                "A behaviour verdict needs --useful/--partial/--observed-miss."
            )
        # A reviewer who names a charged lane has stated the degeneracy clause
        # fires; the two ways of saying so must agree or the seal is refused.
        if args.degenerate_lane and not args.degenerate:
            raise MatrixCellReceiptError(
                f"--degenerate-lane names {len(args.degenerate_lane)} charged "
                "lane(s) but --degenerate was not passed; under ADDENDUM 9 9.1 "
                "any charged lane makes the PER-CHILD verdict dead."
            )
        if args.degenerate and args.degenerate_lane is not None and not args.degenerate_lane:
            raise MatrixCellReceiptError(
                "--degenerate with an empty lane list is a contradiction."
            )
        computed = behavior_verdict_for(
            useful_or_partial=int(args.useful) + int(args.partial),
            coherent=args.smoke_verdict == "coherent",
            emit_zero_floor_passed=receipt["composed"].get("emit_zero_floor_passed"),
            degenerate=bool(args.degenerate),
        )
        if args.verdict is not None and args.verdict != computed:
            raise MatrixCellReceiptError(
                f"--verdict {args.verdict!r} disagrees with the verdict these "
                f"readings imply ({computed!r}).  ADDENDUM 6's bands are "
                "arithmetic, not a reviewer's discretion."
            )
        verdict = computed

    per_case = None
    if args.per_case_json is not None:
        per_case = json.loads(Path(args.per_case_json).read_text(encoding="utf-8"))

    sealed = seal_behavior_review(
        receipt,
        verdict=verdict,
        reviewer=args.reviewer,
        useful=args.useful,
        partial=args.partial,
        observed_miss=args.observed_miss,
        parent_useful_or_partial=args.parent_useful_or_partial,
        dossier_sha256=dossier_sha,
        protocol_sha256=protocol_sha,
        prompt_pack_sha256=pack_sha,
        smoke_verdict=args.smoke_verdict,
        per_case=per_case,
        safety=args.safety,
        notes=args.notes,
        degenerate_lanes=args.degenerate_lane,
        lanes_total=int(args.lanes_total),
    )
    destination = (
        Path(args.out)
        if args.out is not None
        else Path(args.receipt).with_name("cell-receipt-behavior-reviewed.json")
    )
    write_receipt(destination, sealed)
    ledger = (
        Path(args.ledger)
        if args.ledger is not None
        else Path(args.receipt).parents[1] / "ledger.jsonl"
    )
    append_ledger(ledger, sealed)
    block = sealed["addendum6"]
    lane_block = sealed["addendum9"]
    print(
        f"ADDENDUM 9 per-lane reading: {lane_block['degeneracy_reading']} "
        f"({lane_block['degenerate_lane_count']} of "
        f"{lane_block['lanes_total']} lanes charged) -> per-lane verdict "
        f"{lane_block['per_lane_verdict']}; per-child verdict "
        f"{lane_block['per_child_verdict']}"
        + ("  [THE READINGS DISAGREE]" if lane_block["readings_disagree"] else "")
    )
    print(
        f"sealed ADDENDUM 6 review of {sealed['cell_id']} verdict={verdict} "
        f"score={block['useful_or_partial']}/15 "
        f"parent={block['parent_useful_or_partial']} "
        f"delta={block['band_delta_vs_parent']} "
        f"stored={block['stored_parameter_ratio']} "
        f"receipt={sealed['receipt_sha'][:12]}"
    )
    print(f"receipt -> {destination}")
    print(f"ledger  -> {ledger}")
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
