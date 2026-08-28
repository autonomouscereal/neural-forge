#!/usr/bin/env python3
"""Wave 33 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the fourth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31 and
32 each replaced the inherited body with a refusal and wrote their KEEP set by
hand.  Wave 33 does the same and copies neither: the tags below are written after
this wave's reviews are sealed, from the verdicts this wave actually recorded,
and the run is a DRY RUN first, always, with the delete list printed and read
before anything is unlinked.

`cut_wave33.py`'s HAND_EDIT set copied this file VERBATIM, so wave 32's eighteen
tags arrived here still spelled for wave 32 -- which is exactly what makes the
emptied mapping below a decision and not an artifact of a substitution that
happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx33_retain.py --plan            # print what would happen; never deletes
    mx33_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave33")

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
    "MX33-GV16L01": (
        "8/15 at 0.637073x, the only child in the wave to reach eight and the only "
        "one that ties its source.  It is kept because the tie is the wave's result "
        "and the result is a claim about these bytes: the eighth case is "
        "creative-continuity, which wave 32's paraphrase triplets already retired on "
        "the same source, so somebody who wants to argue that the terminal gain "
        "recovered a real case should be able to do it against a checkpoint rather "
        "than against a dossier.  It is also the only wave-33 child sitting on a "
        "campaign frontier row"
    ),
    "MX33-GV12L01": (
        "6/15 at 0.618813x, and the ONLY rung of six that moved at all.  It moved "
        "DOWN, and the mechanism is the interesting part: instruction-three-actions "
        "is its source's only Useful lane and here it hits the generation cap after "
        "eighteen repetitions, so the lambda ladder's most aggressive rung buys its "
        "loss with degeneracy rather than with content.  Three cap hits against zero "
        "on the other two rungs on the same source.  A future wave arguing about the "
        "atlas needs the checkpoint where the ladder breaks, not the five where it "
        "does nothing"
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX33-GV12L10": (
        "7/15 at 0.618813x, flat at its source.  One gainq.sh invocation from the "
        "retained MX31-V12CL1792 at --lambda-relative 1.0; dossier, receipt, "
        "per-case verdicts and build receipt are all packaged"
    ),
    "MX33-GV12L03": (
        "7/15 at 0.618813x, flat at its source.  One gainq.sh invocation from the "
        "retained MX31-V12CL1792 at --lambda-relative 0.3"
    ),
    "MX33-GV16L10": (
        "7/15 at 0.637073x, one below its source.  One gainq.sh invocation from the "
        "retained MX31-V16CL1792 at --lambda-relative 1.0"
    ),
    "MX33-GV16L03": (
        "7/15 at 0.637073x, holding the identical seven lanes as GV16L10 a factor of "
        "three away in lambda.  One gainq.sh invocation from the retained "
        "MX31-V16CL1792 at --lambda-relative 0.3.  The pair's agreement is the "
        "finding and it is readable off two packaged dossiers"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave33.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX33-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave33-children-retention-v1",
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
