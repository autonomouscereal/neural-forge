#!/usr/bin/env python3
"""Wave 27 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 27 does the same and copies neither: the tags below are written after this
wave's reviews are sealed, from the verdicts this wave actually recorded, and the
run is a DRY RUN first, always, with the delete list printed and read before
anything is unlinked.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx27_retain.py --plan            # print what would happen; never deletes
    mx27_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave27")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THE POLICY THIS WAVE APPLIED, stated so the next one can argue with it.  Keep
#: the rows that carry a CLAIM no other artifact can re-derive: the campaign's
#: new best review and the shoulder that proves it is a peak; the two children
#: that tie the deepest ten; the block-7 arm, which is the single row the whole
#: identity-matrix finding rests on; one solve and the zero-solve control it is
#: differenced against at an identical parameter count; the deepest child that
#: still answers anything; the first zero; and the one BODY in this wave that
#: does not reproduce a sealed cell and therefore cannot be recovered from a
#: citation.  Drop what one CPU pass rebuilds from a retained source and that
#: carries no row of its own.  Nine kept and sixteen dropped, which is more than
#: best-five, and the deviation is named rather than assumed: this wave built
#: twenty-three children instead of nine and the extra four keeps are the
#: identity matrix's own axes, which is the result the wave was sent to get.
KEEP: dict[str, str] = {
    "MX27-V19": (
        "14/15 at 0.794570x -- the HIGHEST REVIEW IN 27 WAVES, two cases above "
        "the parent, and the only child that has ever beaten the parent by more "
        "than one.  If any checkpoint in this wave is re-opened by somebody who "
        "does not believe the number, it is this one"
    ),
    "MX27-V17": (
        "11/15 at 0.785440x -- the lower shoulder.  On its own it is an "
        "unremarkable row; with V19 it is the entire evidence that the 13/15 "
        "band is a PEAK with 1,024-row shoulders and not the flat plateau wave "
        "26 sealed, which is this wave's correction to the campaign record"
    ),
    "MX27-V18X13": (
        "10/15 at 0.737484x -- the DEEPEST TEN in the campaign, nine "
        "thousandths below wave 26's V20X13, and the byte-identical zero-solve "
        "control that V18X13G03 is differenced against"
    ),
    "MX27-V18X13G03": (
        "the lambda 0.3 solve at an identical 169,394,944 parameters -- a GPU "
        "pass at one lambda that nothing else reproduces, and the fifth ladder "
        "in the atlas.  It reads the same ten as its control, which is the "
        "measurement that stops lambda 0.3 being called the rung that never hurts"
    ),
    "MX27-V18X09": (
        "10/15 at 0.737484x -- ties the deepest ten, and is the block-9 arm of "
        "the identity matrix at the K where block 9 costs three"
    ),
    "MX27-V18X07": (
        "9/15 at 0.737484x -- THE ROW THE WAVE'S CENTRAL FINDING RESTS ON.  Same "
        "K, same ratio, same parameter count and same base as V18X13 and V18X09, "
        "and four cases cheaper instead of three, which is what makes the "
        "excision-composition cost a property of the BLOCK"
    ),
    "MX27-V20CL896": (
        "2/15 at 0.487566x -- the DEEPEST CHILD IN THE CAMPAIGN TO ANSWER "
        "ANYTHING AT ALL, fifteen points of ratio below the previous holder, and "
        "the floor of the coherence spectrum"
    ),
    "MX27-V18X13X01KV6JQ": (
        "0/15 at 0.664419x -- the first zero of fifteen in 27 waves, the child "
        "that shows summarisation is the last lane to go, and a four-axis stack "
        "no other row in the campaign has"
    ),
    "MX27-F1301KV6Q": (
        "not a reviewed child: the blocks 1+13 excision plus the 16-to-12 joint "
        "head merge, rebuilt as the source for V18X13X01KV6JQ and found NOT to "
        "reproduce wave 17's sealed ARCH.F1301KV6Q (rebuilt "
        "ad406f480a181901, sealed 90a9e5187feef678, identical parameter counts).  "
        "It is the one body in this wave that cannot be recovered from a sealed "
        "citation, so the bytes are the only record of what was actually composed"
    ),
}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child must
#: appear in exactly one of the two mappings or the script refuses.
DROP: dict[str, str] = {
    "MX27-F1301KV6Q-R2": (
        "the second rebuild, kept only long enough to answer whether the "
        "non-reproduction was nondeterminism or code drift.  It is BYTE-IDENTICAL "
        "to MX27-F1301KV6Q (both ad406f480a181901), which settles the question as "
        "drift between wave 17's builder and wave 27's, and storing the same "
        "bytes twice records nothing the digest equality does not"
    ),
    "MX27-V20X07": "9/15 at 0.746614x.  One CPU pass off the retained MX16-X07; its row -- block 7 costs four at BOTH shelf K -- is preserved by V18X07 plus this sealed verdict",
    "MX27-V20X09": "9/15 at 0.746614x.  One CPU pass off the retained MX16-X09; the K-sensitivity it shows is an arithmetic statement about two sealed numbers",
    "MX27-V20X09G03": "8/15.  A solve that COST a case; the finding is the sign of the delta and that is in the verdict, not the weights",
    "MX27-V18X09G03": "9/15.  Same: a lambda arm that lost a case, differenced against a retained control",
    "MX27-V18X07G03": "8/15.  Same, and its one unique contribution -- the third SEAL THE ROOM safety defect -- is in the sealed safety field and the packaged READ- file",
    "MX27-V22X13KV6JQ": "2/15.  One CPU pass off the retained MX17-X13KV6JQ; seven of its lanes are byte-identical to V20X13KV6JQ's, which is the point and needs no second copy",
    "MX27-V20X13KV6JQ": "2/15.  One CPU pass off MX17-X13KV6JQ; the head-merge-costs-eleven row is arithmetic over three sealed numbers",
    "MX27-V20X13KV6JQG03": "2/15.  A solve on a collapsed base that changed nothing but the cap count",
    "MX27-V18X13KV6JQ": "2/15.  The third of three identical head-merge collapses",
    "MX27-V9X13": "6/15 at 0.696398x.  One CPU pass off the retained MX16-X13A; its finding -- block 13 costs three even off the shelf -- is a statement about two sealed reviews",
    "MX27-V20X13X01": "3/15.  One CPU pass off the retained MX17-P1301; the prompt-echo evidence is TEXT and lives in the packaged READ- file, which is where evidence for a text signature belongs",
    "MX27-V20X13X01G03": "1/15.  The solve that cost two cases while improving its own objective by 23 percent out of sample; both numbers are in the verdict",
    "MX27-V18X13X01": "3/15.  One CPU pass off MX17-P1301, and the width-signature falsification it carries is the READ- text, not the weights",
    "MX27-V18W896": "2/15.  One CPU pass off the retained MX19-W896; the empty-string failure mode is visible in the packaged dossier",
    "MX27-V32CL896": "2/15 at 0.542347x.  One CPU pass off the retained MX15-CL896, and V20CL896 is both deeper and retained",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave27.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX27-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave27-children-retention-v1",
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
