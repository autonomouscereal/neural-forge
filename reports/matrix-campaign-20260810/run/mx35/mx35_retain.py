#!/usr/bin/env python3
"""Wave 35 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the sixth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33 and 34 each replaced the inherited body with a refusal and wrote their KEEP
set by hand.  Wave 35 does the same and copies neither: the tags below are
written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-35 cutter's HAND_EDIT set copied this file VERBATIM, so wave 34's five
keeps and six drops arrived here still spelled for wave 34 -- which is exactly
what makes the emptied mapping below a decision and not an artifact of a
substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx35_retain.py --plan            # print what would happen; never deletes
    mx35_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave35")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: FILLED IN AFTER THIS WAVE'S VERDICTS WERE SEALED.  It was EMPTY AT THE
#: TREE CUT, BY DECISION.  Wave 34's five keeps and six drops
#: arrived here VERBATIM and were deleted rather than renamed.  The mapping is
#: filled in after this wave's verdicts are sealed, from those verdicts, and the
#: standing rule the next wave may argue with is unchanged: keep the rows
#: carrying a CLAIM no other artifact re-derives, drop what one CPU pass rebuilds
#: from a retained source, and state the deviation whenever the count is not
#: best-five.  Wave 34 stated one deviation -- that the two BODIES cost 451 s and
#: 743 s of GPU each and are kept on rebuild COST rather than on claim -- and
#: that deviation applies again here, because this wave builds on those same
#: bodies.
#:
#: WAVE 35 KEEPS FOUR RATHER THAN FIVE AND STATES THE DEVIATION.  Three of the
#: four hold a frontier rung on at least one ladder and the fourth is the only
#: conditioned capture in the wave.  Nothing else here carries a claim that a
#: retained source plus one or two cheap passes does not re-derive, and one of
#: the dropped rows -- MX35-V10CL896G10, which TIES the 3/15 arm -- is dropped
#: on the strength of a digest comparison showing ten of its sixteen lanes are
#: byte-identical to the arm that is kept.
KEEP: dict[str, str] = {
    "MX35-CL896X13": (
        "THE NEW BODY, and the only artifact in this wave that cost a conditioned "
        "capture: 296 s of GPU over 24 batches.  It is the source of "
        "ARCH.V12CL896X13, which is one 40-second CPU vocabulary pass on top of "
        "it, and it is the only body in the estate carrying the --ff 896 plus "
        "excision composition.  Deleting it converts every future child on that "
        "composition -- including the V9 and V10 corners of it that this wave did "
        "not build -- from a CPU pass into a GPU build."
    ),
    "MX35-V12CL896X13": (
        "0.420779x, THE DEEPEST CHILD THIS CAMPAIGN HAS EVER BUILT, below the "
        "ARCH.V8CL896 content death, and it holds the 1/15 rung on BOTH ladders.  "
        "Its dossier is committed, but the checkpoint is what any future decode, "
        "robustness or penalty probe has to re-run, and the floor of the frontier "
        "is the row least worth re-deriving from scratch."
    ),
    "MX35-V9CL896": (
        "0.437349x and it holds the 2/15 rung on BOTH ladders.  It is also one of "
        "the two rows that DEFINE the campaign's 0/15 bracket -- 0.004565 wide, "
        "between it and ARCH.V8CL896 -- and the next wave was told to close that "
        "bracket, which means re-reading this child beside whatever it builds "
        "inside it."
    ),
    "MX35-V10CL896G03": (
        "0.441915x at 3/15 sealed and 4/15 canonical, the deepest three the "
        "campaign has sealed and the deepest four it has read.  It holds the 3/15 "
        "sealed rung and the 3/15 and 4/15 canonical rungs.  It is also the arm "
        "the lambda ladder's whole conclusion is a delta against."
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX35-V10CL896G01": (
        "terminal gain lambda 0.1 on wave 34's retained MX34-V10CL896, ~100 s of "
        "GPU.  It reads its control's two and takes no rung; its one distinguishing "
        "fact -- the largest update and the best held-out objective on the ladder, "
        "buying nothing -- is a number in its seal, not a property of its bytes"
    ),
    "MX35-V10CL896G10": (
        "terminal gain lambda 1.0 on the same retained source, ~100 s of GPU.  It "
        "TIES ARCH.V10CL896G03 at three and is dropped anyway, because TEN of its "
        "sixteen lanes are byte-identical to that arm: most of its dossier IS the "
        "retained arm's dossier, and the tie is recorded as a digest comparison "
        "rather than as a second checkpoint"
    ),
    "MX35-V14CL896": (
        "one 40-second CPU vocabulary fold at K=14,336 on wave 34's retained "
        "MX34-CL896.  At 0.460175x it sits above every rung it could take"
    ),
    "MX35-V14CL896G03": (
        "terminal gain lambda 0.3 on MX35-V14CL896, which is itself one CPU pass "
        "from the retained MX34-CL896.  Two cheap passes; it reads three, and the "
        "three rung is held 0.018 deeper by a retained child"
    ),
    "MX35-V16CL896R": (
        "one 40-second CPU vocabulary fold at K=16,384 on the retained MX34-CL896, "
        "built only as the base of the D4 re-materialisation below.  It reproduced "
        "wave 23's and wave 34's sealed digest bit-identically"
    ),
    "MX35-V16CL896G03R": (
        "the D4 diagnostic's re-materialisation of wave 34's ARCH.V16CL896G03.  It "
        "rebuilt BIT-IDENTICALLY -- terminal-gain solve included -- and its "
        "identity-control decode reproduced wave 34's sealed dossier on 16 of 16 "
        "lanes byte for byte, so the next wave that needs it can re-make it from "
        "MX34-CL896 with two cheap passes and KNOW it has the same bytes"
    ),
    "MX35-V8CL1792X13R": (
        "the D4 diagnostic's re-materialisation of wave 34's ARCH.V8CL1792X13, one "
        "CPU vocabulary fold on the retained MX34-CL1792X13.  Bit-identical, and "
        "its identity-control decode reproduced the sealed dossier byte for byte"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave35.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX35-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave35-children-retention-v1",
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
