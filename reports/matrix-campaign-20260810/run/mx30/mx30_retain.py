#!/usr/bin/env python3
"""Wave 29 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 29 does the same and copies neither: the tags below are written after this
wave's reviews are sealed, from the verdicts this wave actually recorded, and the
run is a DRY RUN first, always, with the delete list printed and read before
anything is unlinked.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx30_retain.py --plan            # print what would happen; never deletes
    mx30_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave30")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THE POLICY IS WRITTEN AFTER THE VERDICTS, so it is not written here yet, and
#: wave 28's policy paragraph was DELETED rather than renamed: it argued for a
#: specific set of wave-28 rows, and inherited prose justifying a keep set is how
#: a retention decision gets made by the previous wave.  The standing rule the
#: next wave may argue with: keep the rows carrying a CLAIM no other artifact
#: re-derives, drop what one CPU pass rebuilds from a retained source, and state
#: the deviation whenever the count is not best-five.
KEEP: dict[str, str] = {
    "MX30-B75": (
        "11/15 at 0.742049x BY BLEND -- the only child in the campaign to reach "
        "eleven with no solve of its own.  1,024 numbers convexly mixed from two "
        "sealed twins, zero capture, zero GPU, and it ties the deepest eleven"
    ),
    "MX30-BP64": (
        "10/15, and the row the over-determination law rests on: 64 of 1,024 "
        "coordinates reproduce the solved twin's whole behaviour, gains and "
        "LIST-to-SET loss together, while MASK03 holding the top 100 at the prior "
        "does not prevent the loss"
    ),
    "MX30-MASK03": (
        "9/15 -- BELOW both its endpoints, which is the pre-registered #6 "
        "falsifier firing.  A terminal-gain solve is a GPU pass that nothing "
        "re-derives from a citation, and this one is differenced against a mask "
        "written before it ran"
    ),
    "MX30-KKTD": (
        "0/15 at 0.483001x.  Twenty installed writers solved jointly, 20,480 "
        "unknowns with real cross-writer blocks, objective improved ten percent "
        "on held-out and the child lost both cases it had.  A capture plus an "
        "operator assembly plus a 20,480-square solve; no citation rebuilds it"
    ),
    "MX30-KKTS": (
        "0/15 by a DIFFERENT failure -- echo on ten of sixteen lanes where the "
        "diagonal arm produces filler.  Twenty numbers, update norm 0.336, same "
        "zero: the row that shows the damage is not a function of degrees of "
        "freedom"
    ),
    "MX30-V19MXO3": (
        "3/15.  The CHASM BRIDGE row, and its build receipt is the only artifact "
        "carrying the void-arithmetic finding with its arithmetic: every "
        "parent-shaped base the plan named already folds the embedding, so the "
        "vocabulary gather displaces that fold rather than stacking with it"
    ),
}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child must
#: appear in exactly one of the two mappings or the script refuses.  Written by
#: hand from THIS wave's sealed verdicts, after package.sh.
DROP: dict[str, str] = {
    "MX30-B25": (
        "10/15.  One CPU pass off two retained wave-28 children, and its finding "
        "-- the swap is already gone at the nearest interior point -- is a delta "
        "between sealed reviews whose text is in the packaged dossier"
    ),
    "MX30-B50": (
        "10/15.  The midpoint, same CPU pass, and its contribution is the SHAPE "
        "of the segment, which is arithmetic over five sealed reviews"
    ),
    "MX30-V19CL896": (
        "2/15 at 0.483001x.  Rebuilt this wave in one CPU pass off retained "
        "MX15-CL896 and reproducing wave 28's sealed ratio exactly; it is a "
        "citation, not bytes, and the two KKT children built from it are kept"
    ),
    "MX30-KKTINF": (
        "not a reviewed child: the lambda-to-infinity identity rung.  Its whole "
        "claim IS a digest -- 0fa5175dda6d58ec, byte-identical to its source -- "
        "and that digest is in the receipt, which is packaged"
    ),
    "MX30-V19MXP2": (
        "2/15.  One CPU pass off a materialization that is itself re-derivable "
        "from a sealed wave-12 cell; the MXO3 arm is retained and carries the "
        "same void-arithmetic finding at a deeper base ratio"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave30.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX30-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave30-children-retention-v1",
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
