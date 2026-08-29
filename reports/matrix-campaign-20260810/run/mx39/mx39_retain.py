#!/usr/bin/env python3
"""Wave 38 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the eighth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33, 34, 35, 36 and 37 each replaced the inherited body with a refusal and wrote
their KEEP set by hand.  Wave 38 does the same and copies neither: the tags below
are written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-38 cutter's HAND_EDIT set copied this file VERBATIM, so wave 37's two
keeps and seven drops arrived here still spelled for wave 37 -- which is exactly
what makes the emptied mapping below a decision and not an artifact of a
substitution that happened to succeed.

**WAVE 38 ADDS ONE LINE THAT IS NOT ABOUT RETENTION.**  `NOTES-MX-AO` section 9
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

    mx39_retain.py --plan            # print what would happen; never deletes
    mx39_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave39")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: EMPTY AT THE TREE CUT, BY DECISION.  Wave 37's two keeps and seven drops
#: arrived here VERBATIM and were deleted rather than renamed; the inherited
#: justification paragraphs went with them, because prose arguing for last
#: wave's rows is how a retention decision gets made by the previous wave.  The
#: mapping is filled in after this wave's verdicts are sealed, from those
#: verdicts, and the standing rule the next wave may argue with is unchanged:
#: keep the rows carrying a CLAIM no other artifact re-derives, drop what one
#: CPU pass rebuilds from a retained source, and state the deviation whenever
#: the count is not best-five.
KEEP: dict[str, str] = {}
#: FILLED IN AFTER THIS WAVE'S VERDICTS WERE SEALED.  WAVE 39 KEEPS ZERO OF TWO,
#: WHICH IS A DEVIATION FROM THE BEST-FIVE HABIT AND IS THE CLAIM RULE APPLIED
#: RATHER THAN AN OVERSIGHT.  The standing rule is: keep the rows carrying a
#: CLAIM no other artifact re-derives, drop what one CPU pass rebuilds from a
#: retained source, and state the deviation whenever the count is not best-five.
#:
#: BOTH of this wave's children are ONE CPU VOCABULARY PASS off wave 35's
#: RETAINED MX35-CL896X13, and neither takes a frontier rung -- one ties the
#: sealed 1/15 rung that MX38-V10CL896X09G10 already holds at the identical
#: ratio, and the other is 0.018 of stored ratio shallower than the incumbent
#: 2/15 row.  Every claim either makes is a claim about TEXT, and the text is in
#: a committed dossier, a committed per-case file and a sealed verdict.  Nothing
#: about either child is a claim about BYTES that only a checkpoint can answer,
#: which is the test wave 38 applied when it kept its multi-index excision body.
#:
#: THE ONE CHECKPOINT THIS WAVE'S RESULT DEPENDS ON IS NOT THIS WAVE'S.
#: MX35-CL896X13 is what both children were built from, what the next open cell
#: (K=9,216 on that body) will be built from, and what makes wave 39's answer
#: re-derivable at all.  It is wave 35's keep and this script does not touch it.


#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX39-V10CL896X13": (
        "K=10,240 on ARCH.CL896X13, ONE CPU VOCABULARY PASS off wave 35's "
        "RETAINED MX35-CL896X13, which wave 35's own retention keeps and which "
        "this wave used twice.  It reads 1/15 at 0.4116484014x and TAKES NO "
        "RUNG: the sealed one-of-fifteen rung already sits at that exact ratio "
        "and is held by MX38-V10CL896X09G10, which wave 38 retains, so every "
        "frontier position these bytes occupy is occupied by a retained "
        "checkpoint.  Its result -- that it TIES its X09 twin at an identical "
        "94,552,832 parameters while sharing zero of sixteen lanes with it -- is "
        "a statement about two dossiers, and both are committed.  Its one lane, "
        "the best summarisation anywhere below 0.44x, is quoted verbatim in its "
        "seal, in DEEP_RECORD section 3 and in NOTES-MX-AQ section 4.  It "
        "rebuilds in one CPU pass"
    ),
    "MX39-V14CL896X13": (
        "K=14,336 on ARCH.CL896X13, ONE CPU VOCABULARY PASS off the same "
        "RETAINED MX35-CL896X13.  It reads 1/15 at 0.4299088649x and TAKES NO "
        "RUNG: both rungs it could hold are held 0.018 of stored ratio deeper "
        "by children at 0.4116484014x.  It is the row that makes the "
        "X09-over-X13 advantage rest on two K values rather than one, and that "
        "claim is a pair of counts read off two committed dossiers.  Its "
        "canonical-versus-sealed disagreement -- the one call in this wave that "
        "is a single lane wide -- is recorded with the sentence quoted in "
        "wave39/canonical-reread-wave39.json.  It rebuilds in one CPU pass"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave39.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX39-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave39-children-retention-v1",
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
