#!/usr/bin/env python3
"""Wave 26 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the second wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Wave 25 replaced the
inherited body with a refusal and wrote its KEEP set by hand.  Wave 26 does the
same thing and does not copy wave 25's KEEP set either: the tags below are
written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

Usage::

    mx26_retain.py --plan            # print what would happen; never deletes
    mx26_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave26")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THE POLICY THIS WAVE APPLIED, stated so the next one can argue with it: keep
#: every child carrying a SOLVE (a GPU pass at one lambda that nothing else
#: reproduces), keep the zero-solve control each solve is differenced against at
#: an identical parameter count, keep the rungs that carry a curve claim, and
#: keep any child in the top five of the campaign scorecard's depth ranking.
#: Drop what one CPU pass rebuilds from a retained source and that carries no
#: row of its own.  Seven kept, three dropped -- best-five plus the two extra
#: the first clause forces, and the deviation is named rather than assumed.
KEEP: dict[str, str] = {
    "MX26-V20X13G03": (
        "11/15 at 0.746614x -- the DEEPEST ELEVEN in the campaign, 1.6 points "
        "below the previous one, and a solve: 1,024 diagonal entries at relative "
        "lambda 0.3 that no other artifact reproduces"
    ),
    "MX26-V20X13": (
        "10/15 at 0.746614x -- the DEEPEST TEN in the campaign by 5.5 points, "
        "the third child ever to hold coding-deduplicate, and the byte-identical "
        "zero-solve control the rung above is differenced against.  If any "
        "checkpoint in this wave is re-opened by somebody who does not believe "
        "the number, it is this one"
    ),
    "MX26-V32X13G03": (
        "10/15, the lambda 0.3 rung that fills the only hole in the only FLAT "
        "ladder the campaign has, and the measurement that stops lambda 0.3 "
        "being called a universal peak"
    ),
    "MX26-V18": (
        "13/15 at 0.790005x -- the lower edge of the plateau, and nine "
        "thousandths of ratio deeper than the thirteen wave 25 sealed"
    ),
    "MX26-V22": (
        "13/15 at 0.808265x -- the upper edge of the plateau.  With V18 it is "
        "the whole evidence that wave 25's thirteen is a 4,096-row shelf and "
        "not one lucky table, which is the claim wave 26 was sent to test"
    ),
    "MX26-V9": (
        "9/15 at 0.748919x -- the checkpoint that locates the death of the "
        "vocabulary axis to 1,024 rows.  8,192 reads four and this reads nine; "
        "nothing else in the campaign narrows that window"
    ),
    "MX26-V20X13W896": (
        "2/15 at 0.639307x -- rank three on the campaign scorecard's depth "
        "ranking and the deepest child ever to answer a lane in FULL"
    ),
}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child must
#: appear in exactly one of the two mappings or the script refuses.
DROP: dict[str, str] = {
    "MX26-X1307": (
        "not a reviewed child at all: the block 13 + block 7 body, rebuilt in 35 "
        "seconds as the SOURCE for V20X13X07 and found BIT-IDENTICAL to wave "
        "17's sealed ARCH.P1307 (child_tensors_sha256 329167100d024d33, "
        "205,565,696 parameters).  A body that reproduces to the byte from a "
        "one-line invocation, and that already has a sealed cell somewhere else, "
        "is the clearest possible case for not storing it twice"
    ),
    "MX26-V20W896": (
        "2/15 at 0.683266x.  One CPU pass off the retained MX19-W896, and its "
        "finding -- the third sighting of prompt-echo, all three on width "
        "children -- lives in the sealed verdict and the packaged dossier text, "
        "which is where the evidence for a text signature actually is.  The "
        "checkpoint adds nothing the READ- file does not already say"
    ),
    "MX26-V20X13X07": (
        "5/15 at 0.694093x.  One CPU pass off a body that itself rebuilds in 35 "
        "seconds, and the row it carries -- the third axis is fatal -- is an "
        "arithmetic statement about three sealed numbers (13, 6, 5) that needs "
        "no bytes to re-check"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave26.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX26-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave26-children-retention-v1",
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
