#!/usr/bin/env python3
"""Wave 32 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 32 does the same and copies neither: the tags below are written after this
wave's reviews are sealed, from the verdicts this wave actually recorded, and the
run is a DRY RUN first, always, with the delete list printed and read before
anything is unlinked.

Wave 32 also puts this file in `cut_wave32.py`'s HAND_EDIT set, so the tree cut
copied it VERBATIM -- last wave's nineteen tags arrived here still spelled for
last wave, which is exactly what makes the emptied mapping below a decision and
not an artifact of a substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx32_retain.py --plan            # print what would happen; never deletes
    mx32_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave32")

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
    "MX32-CL1792S": (
        "4/15 at 0.856199x, and 661 seconds of GPU.  It is one half of B4's "
        "controlled evidence pair -- the whole finding is a DIFFERENCE against "
        "ARCH.CL1792's 7/15 at byte-identical storage, identical solver and "
        "identical evidence budget -- and a difference whose two halves are not "
        "both on disk is a claim rather than a measurement.  Its twin is retained "
        "in wave 30.  It is also the only body in this wave that no CPU pass "
        "rebuilds from a retained source"
    ),
    "MX32-BMV3": (
        "7/15 at 0.856199x, the three-way simplex barycentre and the best blend "
        "in the wave.  IT IS ONE CPU PASS off three retained sources and it is "
        "kept anyway, for a reason that is not its score: `mx32_blend_body.py` is "
        "a builder this wave WROTE, and no wave has yet reproduced its output "
        "across a wave boundary.  Keeping exactly one of its seven outputs gives "
        "the next wave the same free reproduction test that wave 27 got on the "
        "vocabulary path and this wave got three times over on CL1408, V9 and V8.  "
        "It is also the source ARCH.V16BMV3 is composed from"
    ),
    "MX32-V6CL1792": (
        "3/15 at 0.591422x, the FIRST child this campaign has ever built below "
        "its own vocabulary census floor.  Wave 32 released the coverage half of "
        "wave 25's gate; the argument for that release is in `mx32_vocab.py` and "
        "the measured cost is in this child's receipt -- 1 donor-mapped and 782 "
        "canonical occurrences dropped, merge closure intact, lossless on text.  "
        "Somebody who wants to argue with the release should be able to do it "
        "against bytes rather than against a receipt.  The deeper rung "
        "MX32-V4CL1792 is dropped because one of the two carries the argument"
    ),
}

DROP: dict[str, str] = {
    "MX32-BM25": (
        "6/15 at 0.856199x.  One CPU pass of `mx32_blend_body.py` over retained "
        "MX30-CL1792 and MX30-CL1792M at alpha 0.25"
    ),
    "MX32-BM50": (
        "4/15 at 0.856199x.  Same two retained sources at alpha 0.50"
    ),
    "MX32-BM75": (
        "2/15 at 0.856199x, the deepest point on the segment.  Same two retained "
        "sources at alpha 0.75; the finding is a COUNT and the count is packaged "
        "with its dossier and its per-case table"
    ),
    "MX32-BMS75": (
        "5/15 at 0.856199x, the source-swap control.  Same two retained sources, "
        "the roles reversed; its finding is a ONE-CASE difference against "
        "MX32-BM25 and both halves are packaged as dossiers"
    ),
    "MX32-BV50": (
        "4/15 at 0.856199x.  One CPU pass over retained MX30-CL1792 and retained "
        "MX31-CL1792MV at alpha 0.50"
    ),
    "MX32-BM50N": (
        "0/15 at 0.856199x, sixteen of sixteen lanes charged.  The naive arm, "
        "rebuilt by adding `--align none` to the command that built MX32-BM50.  "
        "Its whole content is that it is unreadable, which the packaged READ- "
        "file carries better than the checkpoint does"
    ),
    "MX32-V16CL1792M": (
        "3/15 at 0.637073x.  One CPU vocabulary gather off retained MX30-CL1792M "
        "at K=16,384"
    ),
    "MX32-V12CL1792M": (
        "4/15 at 0.618813x.  One CPU gather off MX30-CL1792M at K=12,288"
    ),
    "MX32-V9CL1792M": (
        "2/15 at 0.605117x.  One CPU gather off MX30-CL1792M at K=9,216"
    ),
    "MX32-V8CL1792M": (
        "3/15 at 0.600552x.  One CPU gather off MX30-CL1792M at K=8,192"
    ),
    "MX32-V16BMV3": (
        "3/15 at 0.637073x.  One CPU gather at K=16,384 off MX32-BMV3, which this "
        "manifest KEEPS -- so this child is one command away for as long as its "
        "source is on disk, which is the exact condition the drop rule asks for"
    ),
    "MX32-V4CL1792": (
        "1/15 at 0.582292x, 3,117 rows below the census floor.  One CPU gather "
        "off retained MX30-CL1792 with --below-floor; MX32-V6CL1792 is kept as "
        "the below-floor argument's checkable artifact and one of the two is "
        "enough"
    ),
    "MX32-CL1408R": (
        "NOT REVIEWED and not a cell: the reproduction check wave 31 owed.  Its "
        "bytes are BIT-IDENTICAL to retained MX31-CL1408 -- sha256 "
        "ca0b64c88073cf01... on both -- so this is a literal duplicate of a "
        "checkpoint already on disk, which is the strongest drop case this "
        "manifest can make.  The measurement is the digest comparison and the "
        "digest comparison is in the log and in this wave's note"
    ),
    "MX32-V9CL1792R": (
        "NOT REVIEWED and not a cell: a rebuild of wave 31's dropped ARCH.V9CL1792 "
        "so the paraphrase surface could be run on it.  BIT-IDENTICAL to wave 31's "
        "sealed digest 902d686fa77006e5..., and the paraphrase dossier it exists "
        "to produce is packaged"
    ),
    "MX32-V8CL1792R": (
        "NOT REVIEWED and not a cell: the same rebuild for ARCH.V8CL1792, "
        "BIT-IDENTICAL to sealed digest 29433f6ce9b7fe58..., dossier packaged"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave32.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX32-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave32-children-retention-v1",
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
