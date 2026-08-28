#!/usr/bin/env python3
"""Wave 30 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 30 does the same and copies neither: the tags below are written after this
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
    "MX30-V19CL1792": (
        "7/15 at 0.650769x -- the deepest child this campaign has sealed at >=4, "
        ">=5, >=6 AND >=7 of fifteen, all four rungs at once, and the frontier "
        "row every claim in NOTES-MX-AF rests on.  A capture plus fourteen site "
        "solves plus a vocabulary gather; no citation rebuilds it"
    ),
    "MX30-CL1792": (
        "7/15 at 0.856199x -- the body the frontier child is composed onto, and "
        "the first rung ever built inside the MLP axis's 2,560-to-896 hole.  It "
        "is also the control for the zero-cost vocabulary composition and for "
        "both reader-metric arms, so every A/B in section 5 is differenced "
        "against these bytes"
    ),
    "MX30-CL1792M": (
        "8/15 at 0.856199x, the reader-metric coreset at the solo rung.  Keeping "
        "it keeps the selection A/B reproducible without re-running a 1,040-second "
        "build whose selection depends on a metric accumulated from 768 positions "
        "of the parent's own logits"
    ),
    "MX30-V19CL1792M": (
        "6/15 at 0.650769x, the composed half of the same A/B.  The two rungs "
        "disagree in sign and the extension bank refuses both margins, so the "
        "pair is the evidence and half of it is not evidence"
    ),
    "MX30-CL1280": (
        "4/15 at 0.760331x -- the second rung in the hole, and the one that "
        "locates where the MLP axis falls.  Two rungs make a slope; one makes a "
        "point, and the allocator's next fit reads this row"
    ),
}

DROP: dict[str, str] = {
    "MX30-CL1792LG": (
        "7/15 at 0.856199x, and it is NOT A SECOND MEASUREMENT: all sixteen "
        "response digests are byte-identical to MX30-CL1792's.  Its entire "
        "finding is that identity plus four numbers -- 0.200287 against 0.200310, "
        "325.5163 against 325.5198 -- and all of them are in the packaged build "
        "receipt.  It is one CPU-bounded post-process off a retained child"
    ),
    "MX30-V19CL1280": (
        "1/15 at 0.554901x, the allocator's 0.55x recommendation and its own "
        "EXTRAPOLATES-INTO-THE-GAP flag firing.  One CPU pass -- a vocabulary "
        "gather -- off retained MX30-CL1280, and its finding is a count that the "
        "packaged dossier and per-case table carry in full"
    ),
    "MX30-SR24": (
        "10/15 at 0.742049x.  One CPU pass over 1,024 numbers off two RETAINED "
        "wave-28 children, from a coordinate set that is written out in full in "
        "the packaged attribution artifact.  Reproducible exactly from the "
        "artifact plus mx30_blend.py --restore --restore-k 24"
    ),
    "MX30-SR64": (
        "10/15 at 0.742049x, same one CPU pass at k=64 from the same packaged "
        "coordinate set.  The pair's finding is that both land on the control's "
        "count with dedup still a set, which is two sealed reviews and not bytes"
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
