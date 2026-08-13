#!/usr/bin/env python3
"""Wave 25 retention -- REFUSES TO RUN until this wave's own manifest is written.

**This file was cut from wave 24's by the same blanket rename that cuts the rest
of the tree, and a retention manifest is the one artifact in the tree that a
blanket rename makes ACTIVELY DANGEROUS.**  Wave 24's KEEP set named twelve
children by tag -- ``MX24-V32``, ``MX24-V8``, ``MX24-V8X13`` and so on -- and the
rename turns those into ``MX25-*`` tags that either do not exist or, worse, name
different children.  Run as inherited, it would have deleted every checkpoint
this wave built, because a wave-25 child whose tag is not in a wave-24 KEEP set
is, to this script, a duplicate.

So the inherited body is gone and this is what is left in its place.  The wave's
real manifest is written HERE, by hand, after the reviews are sealed, from the
verdicts this wave actually recorded -- and the run is a DRY RUN first, always,
with the delete list printed and read before anything is unlinked.

Usage::

    mx25_retain.py --plan            # print what would happen; never deletes
    mx25_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave25")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THIS WAVE KEEPS ALL TWELVE AND THE DEVIATION FROM BEST-FIVE IS STATED HERE.
#: Seven of the twelve are consecutive points of ONE dose curve on ONE axis and
#: five are consecutive rungs of THREE lambda ladders.  A curve point without the
#: checkpoint it was read from is a number in a table that nobody can re-open, and
#: this wave's central claim is that the curve is non-monotone -- which is exactly
#: the kind of claim a later wave will want to re-measure rather than believe.
#: Nothing here is a duplicate: every child has a distinct sealed verdict, and the
#: four that share a parameter count (0.801395x) differ in 1,024 overwritten
#: values and read 10, 10, 10 and 9 with three different held sets.  The children
#: live on the 4.5 TB share with 1.8 TB free, not on the root filesystem the disk
#: policy is about, and the whole wave is under 4 GiB.
KEEP: dict[str, str] = {
    "MX25-V20": (
        "13/15 at 0.799135x, zero cap hits -- the highest review in 25 waves and "
        "one more than the parent scores on the same frozen surface.  If any "
        "checkpoint in this campaign is going to be re-opened by somebody who "
        "does not believe the number, it is this one"
    ),
    "MX25-V12": (
        "11/15 at 0.762614x -- ties the campaign's best non-excision review nine "
        "points of ratio deeper, and its position agreement of 0.0992 is the "
        "measurement that falsifies wave 24's segmentation law"
    ),
    "MX25-V14": (
        "9/15 at 0.771744x -- the point between the two elevens that reads nine, "
        "which is the whole evidence that the axis is not monotone in K"
    ),
    "MX25-V10": (
        "10/15 at 0.753484x -- the bisection of the 8,192-to-12,288 cliff.  Four "
        "cases below it and ten here, 2,048 rows apart; the collapse of this axis "
        "is located by this checkpoint and by nothing else"
    ),
    "MX25-V32X13": (
        "10/15 at 0.801395x -- the quality base under the campaign's best "
        "excision, and the zero-solve control every rung of its ladder is "
        "measured against at an identical parameter count"
    ),
    "MX25-V32X13G": "10/15, the split-selected rung of that ladder (lambda 0.01)",
    "MX25-V32X13GL01": "9/15, the lambda 0.1 rung -- the only rung of four to lose a case",
    "MX25-V32X13GL1": "10/15, the lambda 1.0 rung, and the only one holding summarisation in full",
    "MX25-V16X13GL03": (
        "7/15 at 0.728353x -- the V16 recovery, and the first child built at a "
        "lambda the fixed selection ladder does not contain.  The ladder on this "
        "base is unimodal with its peak here"
    ),
    "MX25-V8X13GL03": (
        "7/15 at 0.691832x with THREE Useful lanes -- matches the child Cyril is "
        "scoring at an identical parameter count and carries one more full answer"
    ),
    "MX25-V12X13": (
        "7/15 at 0.710093x -- the composed rung that turns wave 24's sign-change "
        "story into a four-point curve"
    ),
    "MX25-V32W896": (
        "3/15 at 0.731199x -- kept for the SIGNATURE, as wave 24 kept V8CL896 for "
        "the same reason: three of sixteen lanes return the prompt itself, which "
        "is now the second sighting of prompt-echo and the first alongside cap hits"
    ),
}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child must
#: appear in exactly one of the two mappings or the script refuses.
DROP: dict[str, str] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave25.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX25-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave25-children-retention-v1",
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
