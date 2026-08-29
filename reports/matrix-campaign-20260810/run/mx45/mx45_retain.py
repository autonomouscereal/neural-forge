#!/usr/bin/env python3
"""Wave 45 retention -- REFUSES TO RUN until this wave's own manifest is written.

WAVE 45.  Same edit, same reason, sixteenth consecutive wave: the KEEP and
DROP mappings arrived carrying wave 44's one keep and two drops with every
justification paragraph intact, and both were EMPTIED by hand at the cut.
The wave root, the child glob, the schema string and the default manifest
name were retargeted by hand at pinned counts of one each.

--- wave 44's header, kept ---

Wave 44 retention -- REFUSES TO RUN until this wave's own manifest is written.

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

    mx45_retain.py --plan            # print what would happen; never deletes
    mx45_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave45")

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
    "MX45-V10CL896X09": (
        "THE COLUMN-B FLOOR HOLDER, AND THIS KEEP IS A DEVIATION FROM THE DROP "
        "RULE THAT IS ARGUED RATHER THAN HIDDEN.  By the letter of the rule it "
        "should go: it is 25 seconds of CPU off wave 37's RETAINED "
        "MX37-CL896X09, and wave 45 PROVED that -- the rebuild reproduces wave "
        "38's committed child_tensors_sha256 32ed4000ba710780 and "
        "child_directory_sha256 24b8a5b106924e9f exactly, which is the first "
        "time in sixteen waves that any DROP promise has been cashed.  Keeping "
        "bytes to save 25 seconds is bad economics and this reason says so "
        "first.  IT IS KEPT ON WAVE 40'S FRONTIER-HOLDER RULE, APPLIED TO "
        "COLUMN B'S HOLDER FOR THE FIRST TIME.  That rule -- a floor whose "
        "child has been deleted is a floor that can only be cited and not "
        "checked -- has retained column A's holder for five consecutive waves. "
        "Column B's holder has had no checkpoint since wave 38 dropped it, and "
        "wave 38 was RIGHT to at the time, because the dual column did not "
        "exist until wave 42 and this child was then only a joint holder of a "
        "rung another retained checkpoint also held.  It is now the sole cell "
        "behind 0.4116484014x, the number the pending copy-back ruling may make "
        "THE floor of the whole campaign, and wave 45 measured the concrete "
        "cost of its absence: the extension surface this wave was ordered to "
        "run could not touch it until it was rebuilt.  A ruling that lands on "
        "column B should not have to wait on a build.  189,733,004 bytes "
        "against 1.7T free."
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, and the exact command that rebuilds it.
#:
#: EMPTY AT THE TREE CUT, BY DECISION, for the SIXTEENTH consecutive wave.
#: Wave 44's one keep and two drops arrived here VERBATIM -- the cutter copies
#: this file byte for byte -- and were deleted rather than renamed, because a
#: substitution that happened to succeed on `MX44-V3CL768` would have produced
#: a wave-45 tag that either does not exist or, worse, names a different child.
#: The inherited justification paragraphs went with them.
DROP: dict[str, str] = {
    "MX45-V2CL768": (
        "0/15 at 0.3814267993x.  Its whole finding is a READING of one lane's "
        "text -- that the echo terminates COMPLETE at 168 tokens against a "
        "1,024-token cap, preserves the decision verbatim, and destroys the "
        "unresolved risk by SELECTIVE CORRUPTION with the sentence frame "
        "intact -- and every character a reader needs to overturn that reading "
        "is in the committed dossier at wave45/human-review/dossier-V2CL768.json "
        "under response sha256 959f866ca8f0e732, beside the per-case file, the "
        "cell receipt and the seal.  The checkpoint adds nothing to it.  "
        "REBUILT BY one CPU pass: BELOW_FLOOR=1 vocabq.sh "
        "V2CL768:2048:<MX40-CL768>:<its cell receipt>, off wave 40's RETAINED "
        "MX40-CL768.  The census reads post-closure occurrence coverage 0.895980 "
        "at K=2,048, so the coverage guard is released on the rebuild exactly as "
        "it was released on the build, and the merge closure skips ONE token "
        "for want of room, which the build receipt records and the rebuild must "
        "reproduce."
    ),
    "MX45-V1CL768": (
        "0/15 at 0.3768616835x, the deepest child this campaign has built, and "
        "depth with no case is a number.  Its findings -- ten cap rides on "
        "sub-word loop units, one empty lane, and the only grammatical clause "
        "on sixteen lanes being a copy of the prompt's own assistant turn -- "
        "are carried entirely by the committed dossier at "
        "wave45/human-review/dossier-V1CL768.json, the per-case file, the "
        "receipt and the seal.  There is nothing a reader can do with these "
        "bytes that the text does not already show.  REBUILT BY one CPU pass: "
        "BELOW_FLOOR=1 vocabq.sh V1CL768:1024:<MX40-CL768>:<its cell receipt>, "
        "off wave 40's RETAINED MX40-CL768.  Note for whoever rebuilds it: the "
        "765-row mandatory set leaves 259 discretionary rows at this K, so the "
        "structural refusal is REACHABLE and the rebuild is the cheapest way to "
        "re-exercise it."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave45.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX45-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave45-children-retention-v1",
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
