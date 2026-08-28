#!/usr/bin/env python3
"""Wave 31 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 31 does the same and copies neither: the tags below are written after this
wave's reviews are sealed, from the verdicts this wave actually recorded, and the
run is a DRY RUN first, always, with the delete list printed and read before
anything is unlinked.

Wave 31 also puts this file in `cut_wave31.py`'s HAND_EDIT set, so the tree cut
copied it VERBATIM -- last wave's five tags arrived here still spelled for last
wave, which is exactly what makes the emptied mapping below a decision and not
an artifact of a substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx31_retain.py --plan            # print what would happen; never deletes
    mx31_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave31")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THE POLICY IS WRITTEN AFTER THE VERDICTS, so it is not written here yet, and
#: wave 30's policy paragraph was DELETED rather than renamed: it argued for a
#: specific set of last wave's rows, and inherited prose justifying a keep set is how
#: a retention decision gets made by the previous wave.  The standing rule the
#: next wave may argue with: keep the rows carrying a CLAIM no other artifact
#: re-derives, drop what one CPU pass rebuilds from a retained source, and state
#: the deviation whenever the count is not best-five.
KEEP: dict[str, str] = {
    "MX31-CL1664": (
        "5/15 at 0.832232x -- the rung that ends the MLP axis's headroom.  A "
        "capture plus fourteen site solves, 585 seconds; nothing rebuilds it "
        "from a retained source, and it is the upper end of the 128-channel "
        "bracket the allocator now reports for the mlp transition"
    ),
    "MX31-CL1536": (
        "4/15 at 0.808265x, the middle rung, and the only child in the wave to "
        "answer context-retrieval in full.  546 seconds of GPU"
    ),
    "MX31-CL1408": (
        "3/15 at 0.784298x -- the rung that shows the mlp axis is NOT monotone, "
        "reading one BELOW ARCH.CL1280 at a shallower ratio.  479 seconds"
    ),
    "MX31-CL1792X13": (
        "4/15 at 0.813949x, the triple base.  It is the measurement that moved "
        "the fitted depth x mlp interference from 0.81 to 2.90 cases, and the "
        "only two-axis mlp-with-depth child the campaign has ever built"
    ),
    "MX31-CL1792ME": (
        "2/15 at 0.856199x.  The whole finding is a DIFFERENCE against "
        "ARCH.CL1792M's 8/15 at byte-identical storage, and a difference whose "
        "two halves are not both on disk is a claim rather than a measurement.  "
        "1,121 seconds, and the only child ever solved against the retokenized "
        "evidence stream"
    ),
    "MX31-CL1792MV": (
        "7/15 at 0.856199x, the emittable-row reader metric.  Its selection "
        "depends on a metric accumulated from 768 positions of the parent's own "
        "logits masked to 19,454 rows; 1,038 seconds, and re-deriving it needs "
        "the GPU pass, not a citation"
    ),
    "MX31-V16CL1792": (
        "8/15 at 0.637073x -- the deepest child this campaign has sealed at or "
        "above 8/15, by 0.092 of ratio.  It IS one CPU pass off retained "
        "MX30-CL1792 and it is kept anyway, for the reason wave 30 kept "
        "MX30-V19CL1792: a frontier row that has to be regenerated before it can "
        "be re-examined is a frontier row nobody re-examines, and its eighth "
        "case is a one-case margin somebody should be able to argue with"
    ),
    "MX31-V12CL1792": (
        "7/15 at 0.618813x -- the deepest child at or above 7/15.  Same argument, "
        "same deviation from the drop rule, stated rather than hidden"
    ),
}

DROP: dict[str, str] = {
    "MX31-V19CL1664": (
        "3/15 at 0.626802x.  One CPU vocabulary gather off retained MX31-CL1664; "
        "its finding is the minus-two composition tax, which the packaged dossier "
        "and per-case table carry in full"
    ),
    "MX31-V19CL1536": (
        "2/15 at 0.602835x.  One CPU gather off retained MX31-CL1536"
    ),
    "MX31-V19CL1408": (
        "1/15 at 0.578868x.  One CPU gather off retained MX31-CL1408"
    ),
    "MX31-V19CL1792X13": (
        "2/15 at 0.608519x, the triple.  One CPU gather off retained "
        "MX31-CL1792X13; the verdict is a count and the count is packaged"
    ),
    "MX31-V19CL1792ME": (
        "3/15 at 0.650769x.  One CPU gather off retained MX31-CL1792ME"
    ),
    "MX31-V19CL1792MV": (
        "5/15 at 0.650769x.  One CPU gather off retained MX31-CL1792MV"
    ),
    "MX31-V26CL1792": (
        "7/15 at 0.682724x.  One CPU gather off wave 30's retained MX30-CL1792 "
        "at K=26,624; ten of its sixteen lanes are already comparable to "
        "MX31-V24CL1792's, which is the finding"
    ),
    "MX31-V24CL1792": (
        "7/15 at 0.673594x.  One CPU gather off MX30-CL1792 at K=24,576"
    ),
    "MX31-V18CL1792": (
        "7/15 at 0.646204x.  One CPU gather off MX30-CL1792 at K=18,432"
    ),
    "MX31-V9CL1792": (
        "6/15 at 0.605117x, ON the coverage crossing.  One CPU gather off "
        "MX30-CL1792 at K=9,216"
    ),
    "MX31-V8CL1792": (
        "5/15 at 0.600552x, below the crossing.  One CPU gather off MX30-CL1792 "
        "at K=8,192, and the deepest >=4/15 and >=5/15 row in the campaign -- "
        "kept as a NUMBER in the ledger and the packaged dossier, and rebuilt in "
        "twenty-nine seconds by anyone who wants the bytes"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave31.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX31-*") if p.is_dir())
    if not KEEP and not DROP:
        print(
            "RETENTION REFUSED: this wave's KEEP/DROP mapping is empty.  It is "
            "written by hand from this wave's sealed verdicts and is never "
            "inherited.  Children on disk:\n  " + "\n  ".join(built),
            file=sys.stderr,
        )
        return 3

    unclassified = [c for c in built if c not in KEEP and c not in DROP]
    both = sorted(set(KEEP) & set(DROP))
    phantom = sorted(t for t in list(KEEP) + list(DROP) if t not in built)
    if unclassified or both or phantom:
        print(
            "RETENTION REFUSED:\n"
            f"  unclassified on disk : {unclassified}\n"
            f"  in KEEP and DROP     : {both}\n"
            f"  named but not on disk: {phantom}",
            file=sys.stderr,
        )
        return 4

    sizes = {}
    for tag in built:
        total = 0
        for base, _dirs, files in os.walk(W / "children" / tag):
            for name in files:
                total += os.path.getsize(os.path.join(base, name))
        sizes[tag] = total

    manifest = {
        "schema": "neural-forge-matrix-wave31-children-retention-v1",
        "root": str(W / "children"),
        "built": built,
        "kept": {t: {"reason": KEEP[t], "bytes": sizes[t]} for t in sorted(KEEP)},
        "deleted": {t: {"reason": DROP[t], "bytes": sizes[t]} for t in sorted(DROP)},
        "bytes_kept": sum(sizes[t] for t in KEEP),
        "bytes_deleted": sum(sizes[t] for t in DROP),
        "applied": bool(args.apply),
    }
    args.out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: v for k, v in manifest.items() if k != "kept"}, indent=1))
    for tag in sorted(DROP):
        print(f"  DELETE {tag}  ({sizes[tag] / 2**20:.0f} MiB)  {DROP[tag]}")
    if not args.apply:
        print("\nDRY RUN: nothing was deleted.  Re-run with --apply.")
        return 0
    for tag in sorted(DROP):
        shutil.rmtree(W / "children" / tag)
        print(f"deleted {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
