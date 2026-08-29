#!/usr/bin/env python3
"""Wave 41 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the eighth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33, 34, 35, 36, 37, 38, 39 and 40 each replaced the inherited body with a
refusal and wrote their KEEP set by hand.  Wave 41 does the same and copies
neither: the tags below
are written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-41 cutter's HAND_EDIT set copied this file VERBATIM, so wave 40's
three keeps and eight drops arrived here still spelled for wave 40 -- which is
exactly what makes the emptied mapping below a decision and not an artifact of
a substitution that happened to succeed.

**WAVE 38 ADDED ONE LINE THAT IS NOT ABOUT RETENTION, AND IT IS KEPT.**  `NOTES-MX-AO` section 9
is wave 37's account of reproducing wave 35's corruption bug and its instruction
to the next wave: *every artifact a wave writes gets parsed by the command that
writes it, in the same invocation.*  This script wrote its manifest and never
read it back, so a truncated or unserialisable manifest would have been noticed a
wave later, by whoever tried to read the retention record.  The `json.loads`
below is that rule, applied to the one artifact this file produces.  It changes
no keep, no drop and no deletion.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx41_retain.py --plan            # print what would happen; never deletes
    mx41_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave41")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: EMPTY AT THE TREE CUT, BY DECISION.  Wave 40's three keeps and eight drops
#: arrived here VERBATIM and were deleted rather than renamed; the inherited
#: justification paragraphs went with them, because prose arguing for last
#: wave's rows is how a retention decision gets made by the previous wave.  The
#: mapping is filled in after this wave's verdicts are sealed, from those
#: verdicts, and the standing rule the next wave may argue with is unchanged:
#: keep the rows carrying a CLAIM no other artifact re-derives, drop what one
#: CPU pass rebuilds from a retained source, and state the deviation whenever
#: the count is not best-five.
KEEP: dict[str, str] = {
    "MX41-V8CL896X13": (
        "THE FRONTIER HOLDER.  1/15 at 0.4025181696x, the campaign's sealed and "
        "canonical 1/15 rung, moved this wave from 0.4070832855x.  A frontier "
        "holder is retained on principle -- wave 40's rule, kept: the next "
        "wave's floor cell is graded against it, and a floor whose child has "
        "been deleted is a floor that can only be cited and not checked.  It is "
        "also the child whose held lane is byte-identical to its control's, "
        "which is a claim a future wave may want to re-run rather than re-read"
    ),
    "MX41-V7CL768": (
        "THE DEEPEST TWO-AXIS CHILD IN THE CAMPAIGN and the ONLY NEW HELD-LANE "
        "WITNESS IN THIS WAVE.  1/15 at 0.4042523787x on mlp x vocabulary with "
        "NO EXCISION, held-lane sha256 b5c9c35edfad354b, shared with no other "
        "dossier in the estate.  Its claim -- that the 0.404 to 0.407 death "
        "band is NOT a ratio wall, because a two-axis child is alive inside it "
        "while three-axis children at the same depth are not -- is re-derivable "
        "from no other retained artifact.  Its body MX40-CL768 is retained, so "
        "this is the one rung on that ladder worth its own bytes"
    ),
}


#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX41-V8CL768": (
        "1/15 at 0.4088174946x, and eleven of its sixteen lanes are byte-"
        "identical to ARCH.V9CL768's, so it re-reads a decode the estate "
        "already holds.  REBUILT BY one CPU pass: vocabq.sh "
        "V8CL768:8192:<MX40-CL768>, off wave 40's RETAINED MX40-CL768"
    ),
    "MX41-V95CL896X09": (
        "0/15 at 0.4093658434x.  Its finding -- the X09 cliff is in the upper "
        "half of the bracket -- is fully carried by the sealed cell receipt and "
        "the ledger row; the checkpoint adds nothing a reader can use.  REBUILT "
        "BY one CPU pass: vocabq.sh V95CL896X09:9728:<MX37-CL896X09>, off wave "
        "37's RETAINED MX37-CL896X09"
    ),
    "MX41-RP9CL896X13": (
        "1/15 at 0.4070832855x, the arm P rowset child.  REBUILT BY one CPU "
        "pass: rowsetq.sh RP9CL896X13:P:9216:93504256:<MX35-CL896X13>, off wave "
        "35's RETAINED MX35-CL896X13, and the arm is deterministic so the "
        "rebuild is the same bytes"
    ),
    "MX41-RR9CL896X13": (
        "1/15 at 0.4070832855x, the arm R rowset child.  Its permutation is "
        "SEEDED and the seed is in its receipt, so the rebuild is reproducible "
        "from the receipt alone.  REBUILT BY one CPU pass: rowsetq.sh "
        "RR9CL896X13:R:9216:93504256:<MX35-CL896X13>, off wave 35's RETAINED "
        "MX35-CL896X13"
    ),
    "MX41-RC9CL896X13": (
        "1/15 at 0.4070832855x, the arm C rowset child.  REBUILT BY one CPU "
        "pass: rowsetq.sh RC9CL896X13:C:9216:93504256:<MX35-CL896X13>, off wave "
        "35's RETAINED MX35-CL896X13.  All three arms are dropped together "
        "BECAUSE their result is that they are indistinguishable from their "
        "control on the review, and the three sealed receipts carry the row-"
        "movement numbers that ARE the result"
    ),
    "MX41-V9CL896X13G10": (
        "1/15 at 0.4070832855x, ZERO storage delta.  REBUILT BY one GPU pass of "
        "about two minutes: gainq.sh <gpu> V9CL896X13G10:<MX40-V9CL896X13>:1.0, "
        "off wave 40's RETAINED MX40-V9CL896X13"
    ),
    "MX41-V9CL896X13G29": (
        "0/15 at 0.4070832855x, ZERO storage delta, and the arm that DESTROYED "
        "its source's one case while improving its own objective.  That finding "
        "is carried entirely by the two sealed receipts and the update_scale and "
        "relSSE numbers in them.  REBUILT BY one GPU pass: gainq.sh <gpu> "
        "V9CL896X13G29:<MX40-V9CL896X13>:1.0:29"
    ),
    "MX41-V9CL896X13G03": (
        "1/15 at 0.4070832855x, ZERO storage delta.  REBUILT BY one GPU pass: "
        "gainq.sh <gpu> V9CL896X13G03:<MX40-V9CL896X13>:0.3"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave41.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX41-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave41-children-retention-v1",
        "root": str(W / "children"),
        "built": built,
        "kept": {t: {"reason": KEEP[t], "bytes": sizes[t]} for t in sorted(KEEP)},
        "deleted": {t: {"reason": DROP[t], "bytes": sizes[t]} for t in sorted(DROP)},
        "bytes_kept": sum(sizes[t] for t in KEEP),
        "bytes_deleted": sum(sizes[t] for t in DROP),
        "applied": bool(args.apply),
    }
    args.out.write_text(json.dumps(manifest, indent=1))
    # NOTES-MX-AO section 9, applied to the one artifact this file writes: parse
    # it back in the invocation that wrote it, and refuse if the bytes on disk do
    # not say what this run decided.
    reread = json.loads(args.out.read_text())
    if (reread["schema"] != manifest["schema"]
            or sorted(reread["kept"]) != sorted(manifest["kept"])
            or sorted(reread["deleted"]) != sorted(manifest["deleted"])):
        print(f"RETENTION REFUSED: {args.out} does not read back as written",
              file=sys.stderr)
        return 5
    print(json.dumps({k: v for k, v in manifest.items() if k != "kept"}, indent=1))
    print(f"manifest re-parsed OK: {len(reread['kept'])} kept, "
          f"{len(reread['deleted'])} deleted, {args.out}")
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
