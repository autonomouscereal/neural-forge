#!/usr/bin/env python3
"""Wave 28 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the third wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25 and 26 each
replaced the inherited body with a refusal and wrote their KEEP set by hand.
Wave 28 does the same and copies neither: the tags below are written after this
wave's reviews are sealed, from the verdicts this wave actually recorded, and the
run is a DRY RUN first, always, with the delete list printed and read before
anything is unlinked.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx28_retain.py --plan            # print what would happen; never deletes
    mx28_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave28")

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
    "MX28-V19X13G03": (
        "11/15 at 0.742049x -- THE DEEPEST ELEVEN IN THE CAMPAIGN, eleven points "
        "of ratio below the previous holder, and the first lambda arm to GAIN a "
        "case since wave 27 demoted lambda 0.3 on six arms that all lost or tied"
    ),
    "MX28-V19X13G01": (
        "11/15 at 0.742049x -- ties the deepest eleven at a different lambda and "
        "completes the first MONOTONE ladder any base has produced (10/10/11/11)"
    ),
    "MX28-V19X13": (
        "10/15 at 0.742049x -- the zero-solve control both elevens are "
        "differenced against at an identical 170,443,520 parameters, and the row "
        "that shows the summit lands at ten where the shoulders land at ten"
    ),
    "MX28-V19X09": (
        "10/15 at the identical parameter count and ratio as V19X13 and by a "
        "COMPLETELY DIFFERENT lane profile -- four Useful against three, winning "
        "and losing opposite lanes.  This is the row that makes the ten a "
        "property of the composition and not of the block"
    ),
    "MX28-V19X13X09": (
        "5/15 at 0.689528x -- the pair-distance-law INVERSION.  Four apart reads "
        "five where twelve apart reads three, at one parameter count on one "
        "vocabulary base, against wave 17's monotone farther-is-better"
    ),
    "MX28-V18CL896": (
        "2/15 at 0.478436x -- THE DEEPEST CHILD IN THE CAMPAIGN TO HOLD ANY "
        "CASE, nine points of ratio below wave 27's holder"
    ),
    "MX28-V19X13CL896": (
        "0/15 at 0.452734x -- the deepest child ever built, the second zero in "
        "28 waves, and the discriminating case that KILLS wave 27's "
        "summarisation-is-the-last-lane claim: it paraphrases instead of copying "
        "the prompt tail, and the paraphrase drops the decision"
    ),
    "MX28-V18X13X01G03": (
        "4/15 at 0.684962x -- one lambda arm off the deepest ladder in the "
        "atlas, kept because a terminal-gain solve is a GPU pass that nothing "
        "re-derives from a citation, and because the 3/4/4/3 ladder it sits in "
        "is single-peaked in the middle while its own held-out objective is not"
    ),
    "MX28-CL896X13": (
        "not a reviewed child: the block-13 excision composed with wave 15's "
        "CL896 MLP fold, 0.658165x, built in this wave and sealed by no wave.  "
        "There is no citation to recover it from, so the bytes are the record"
    ),
    "MX28-P1309": (
        "not a reviewed child: blocks 9 and 13 excised, 0.894958x, the body the "
        "distance-law inversion is measured on and a cell no wave has sealed"
    ),
}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child
#: must appear in exactly one of the two mappings or the script refuses.
#: Written by hand from THIS wave's sealed verdicts, after package.sh.
DROP: dict[str, str] = {
    "MX28-V18X13X01": (
        "3/15.  Its whole claim IS a digest: the tensors are bit-identical to "
        "wave 27's sealed V18X13X01 (fe649aae5fa48fa2...), which is recorded in "
        "the build receipt and reproduced by one CPU pass off retained MX17-P1301"
    ),
    "MX28-V18X13X01G10": "4/15.  Same +1 and the same four lanes as the retained lambda 0.3 arm; storing a second solve that buys the same two lanes records nothing the two verdicts do not",
    "MX28-V18X13X01G01": "3/15.  The no-gain rung.  Its contribution is the SHAPE of the ladder, which is arithmetic over three sealed reviews",
    "MX28-V19X13G10": "10/15.  The lambda arm with no delta; its finding is that six lanes trade at a constant count, and the lane trades are in the packaged READ- text",
    "MX28-V19X13X01": "3/15.  One CPU pass off retained MX17-P1301; the prompt-echo evidence is TEXT and lives in the packaged dossier",
    "MX28-V19X13X01G03": "4/15.  The +1 transfers across K from the retained V18X13X01G03 arm, which is a statement about two sealed reviews at one lambda",
    "MX28-V19CL896": "2/15 at 0.483001x.  One CPU pass off retained MX15-CL896, and V18CL896 is both deeper and retained",
    "MX28-V19CL896G03": "3/15.  The gate-pass is recorded in the build receipt (identity_gate max_abs_error 8.0e-14), which is packaged; V20CL896G03 is the same measurement on a second base",
    "MX28-V20CL896G03": "3/15.  Same: the +1 at the bottom of the spectrum is a delta between two sealed reviews, and the gate evidence is the packaged build receipt",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave28.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX28-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave28-children-retention-v1",
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
