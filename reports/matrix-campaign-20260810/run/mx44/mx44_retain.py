#!/usr/bin/env python3
"""Wave 44 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the ninth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33, 34, 35, 36, 37, 38, 39, 40, 41, 42 and 43 each replaced the inherited body
with a refusal and wrote their KEEP set by hand.  Wave 44 does the same and
copies neither: the tags below
are written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-44 cutter's HAND_EDIT set copied this file VERBATIM, so wave 43's
TWO keeps and ONE drop arrived here still spelled for wave 43, with every
justification paragraph intact -- which is exactly what makes the emptied
mapping below a decision and not an artifact of a substitution that happened
to succeed.

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

    mx44_retain.py --plan            # print what would happen; never deletes
    mx44_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave44")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: EMPTY AT THE TREE CUT, BY DECISION.  Wave 43's two keeps and one drop
#: arrived here VERBATIM and were deleted rather than renamed; the inherited
#: justification paragraphs went with them, because prose arguing for last
#: wave's rows is how a retention decision gets made by the previous wave.  The
#: mapping is filled in after this wave's verdicts are sealed, from those
#: verdicts, and the standing rule the next wave may argue with is unchanged:
#: keep the rows carrying a CLAIM no other artifact re-derives, drop what one
#: CPU pass rebuilds from a retained source, and state the deviation whenever
#: the count is not best-five.
KEEP: dict[str, str] = {
    "MX44-V3CL768": (
        "THE FRONTIER HOLDER, AND THE DEEPEST CHILD THIS CAMPAIGN HAS BUILT "
        "THAT HOLDS ANYTHING.  1/15 at 0.3859919152x, which is the sealed rung "
        "this wave moved from 0.3951221469x -- the FIFTH consecutive wave the "
        "sub-0.40x floor moves.  A frontier holder is retained on principle -- "
        "wave 40's rule, kept for the fifth wave: the next wave's floor cell is "
        "graded against it, and a floor whose child has been deleted is a floor "
        "that can only be cited and not checked.  It is ALSO the wave's only new "
        "deep-band held-lane witness, sha256 d10952d96d99eeea, shared with "
        "nothing in the estate.  AND IT IS THE SUBJECT OF THIS WAVE'S ONE "
        "ADJUDICATION: NOTES-MX-AV section 4.2 holds a lane that returns the "
        "WHOLE PROMPT, instruction line included, on the ground that it "
        "preserves both asked-for items more cleanly than the lane wave 43 held "
        "one rung above it.  A reader who wants to overturn that call needs "
        "these bytes and not a rebuild, and the call is worth 0.0091302317 of "
        "column-A ratio."
    ),
}


#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
#:
#: EMPTY AT THE TREE CUT, BY DECISION, exactly as KEEP is.  Wave 43's one drop
#: arrived here VERBATIM with its whole justification paragraph -- an argument
#: about a wave-43 checkpoint, in a wave-44 file, which is how a retention
#: decision gets made by the previous wave.
DROP: dict[str, str] = {
    "MX44-V4CL768": (
        "0/15 at 0.3905570311x.  Its whole finding -- that the CL768 ladder's "
        "eight-rung flat run ENDS at K=4,096, and that it dies with the copy "
        "path HALF FORMED, opening on the living rung's own substitution "
        "signature and then looping to the cap with the decision and the risk "
        "both gone -- is carried entirely by the sealed cell receipt, the "
        "per-case file, wave44/lane-byte-comparison-wave44.json and its "
        "benchmark row, and nothing a reader can do with the checkpoint adds to "
        "it.  REBUILT BY one CPU pass: BELOW_FLOOR=1 vocabq.sh "
        "V4CL768:4096:<MX40-CL768>:<its cell receipt>, off wave 40's RETAINED "
        "MX40-CL768.  The vocabulary census reads post-closure coverage of "
        "0.971463 at K=4,096, so the coverage guard is released on the rebuild "
        "exactly as it was released on the build."
    ),
    "MX44-V5CL896X13": (
        "0/15 at 0.3888228220x.  Its whole finding is the MATCHED-K CONTROL, "
        "and the control is checkable from artifacts rather than from bytes: "
        "its build receipt and MX43-V5CL768's agree on every vocabulary field "
        "-- k 5,120, kept_rows 5,120, mandatory_rows 765, admitted 3,027, "
        "skipped-for-closure 0, dead rows dropped 1,134, and the same ranking "
        "rule -- and their tokenizer.json files are BYTE-IDENTICAL at sha256 "
        "041435dd8361647e while their tensors differ.  Both receipts are "
        "committed under wave44/human-review/ and wave43/human-review/, so the "
        "claim that only the BODY differs survives the deletion.  REBUILT BY "
        "one CPU pass: BELOW_FLOOR=1 vocabq.sh "
        "V5CL896X13:5120:<MX35-CL896X13>:<its cell receipt>, off wave 35's "
        "RETAINED MX35-CL896X13."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave44.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX44-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave44-children-retention-v1",
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
