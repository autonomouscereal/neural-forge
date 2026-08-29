#!/usr/bin/env python3
"""Wave 40 retention -- REFUSES TO RUN until this wave's own manifest is written.

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

    mx40_retain.py --plan            # print what would happen; never deletes
    mx40_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave40")

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
KEEP: dict[str, str] = {
    "MX40-CL1728": (
        "618 s of conditioned GPU solve, and the base a next wave needs.  It is "
        "the body under ARCH.V19CL1728, which reads 4/15 at 0.6387852066x with "
        "ZERO of sixteen lanes riding the cap -- the best-behaved child below "
        "0.65x in three waves -- and it is one end of the campaign's largest "
        "single mlp transition, 1728 to 1792, worth three cases over 64 "
        "channels.  A CL1760 rung splits that transition and needs this body "
        "beside it to be a comparison rather than a reading"
    ),
    "MX40-CL768": (
        "371 s of conditioned GPU solve, and the body under the deepest FLAT "
        "vocabulary ladder in the campaign: ARCH.CL768, ARCH.V12CL768, "
        "ARCH.V10CL768 and ARCH.V9CL768 all read 1/15 across 0.2511 of stored "
        "ratio.  Any further rung on that ladder -- K = 8,192 is the obvious one "
        "-- is a 40-second CPU pass off THIS directory and a 371-second GPU "
        "solve without it"
    ),
    "MX40-V9CL896X13": (
        "THE CAMPAIGN'S SEALED AND CANONICAL 1/15 FLOOR CHILD at 0.4070832855x. "
        "A frontier holder is retained on principle: the next wave's V8CL896X13 "
        "is graded against it, and a floor whose child has been deleted is a "
        "floor that can only be cited, not checked.  It is 178M"
    ),
}


#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX40-CL768X09": (
        "0/15 at 0.6359097186x, and both of its own rungs read 0/15 too, so the "
        "whole arm is dead.  REBUILT BY: bodyq.sh <gpu> CL768X09:NEW -- --ff 768 "
        "--excise 9, 318 s of GPU from the parent, no retained source needed"
    ),
    "MX40-V12CL768": (
        "1/15 at 0.4270779580x.  REBUILT BY one CPU pass: vocabq.sh "
        "V12CL768:12288:<MX40-CL768>, off THIS WAVE'S RETAINED MX40-CL768"
    ),
    "MX40-V10CL768": (
        "1/15 at 0.4179477263x.  REBUILT BY one CPU pass: vocabq.sh "
        "V10CL768:10240:<MX40-CL768>, off THIS WAVE'S RETAINED MX40-CL768"
    ),
    "MX40-V9CL768": (
        "1/15 at 0.4133826104x, and it is ABOVE the floor so it holds no rung.  "
        "REBUILT BY one CPU pass: vocabq.sh V9CL768:9216:<MX40-CL768>, off THIS "
        "WAVE'S RETAINED MX40-CL768"
    ),
    "MX40-V12CL768X09": (
        "0/15 at 0.3985236932x.  REBUILT BY a 318 s GPU rebuild of CL768X09 "
        "(which is dropped above) plus one CPU pass.  That is stated rather than "
        "hidden: this child's rebuild is NOT free, and it is dropped anyway "
        "because a zero at a depth already witnessed by a deeper zero buys "
        "nothing a receipt does not"
    ),
    "MX40-V10CL768X09": (
        "0/15 at 0.3893934615x -- THE DEEPEST CHILD THIS CAMPAIGN HAS BUILT, and "
        "it is dropped, which is the retention policy working rather than "
        "sentiment losing.  It holds nothing; its dossier, cell receipt, "
        "per-case reading and build receipt all ship in the human-review "
        "package, so every claim made about it stays checkable.  REBUILT BY a "
        "318 s GPU rebuild of CL768X09 plus one CPU pass"
    ),
    "MX40-T9CL896X09": (
        "0/15 at 0.4070832855x, the C1 row-tying arm.  REBUILT BY one CPU pass: "
        "tieq.sh T9CL896X09:9216:512:93504256:<wave37 MX37-CL896X09>, off WAVE "
        "37'S RETAINED MX37-CL896X09, which wave 37 kept and which is still on "
        "the share.  Its whole result -- fifteen of sixteen lanes byte-identical "
        "to its deletion twin -- is a comparison between two DOSSIERS, and both "
        "dossiers are retained"
    ),
    "MX40-V19CL1728": (
        "4/15 at 0.6387852066x, the wave's best deep child.  REBUILT BY one CPU "
        "pass: vocabq.sh V19CL1728:19456:<MX40-CL1728>, off THIS WAVE'S RETAINED "
        "MX40-CL1728.  It is dropped rather than kept precisely BECAUSE its body "
        "is kept: a 40-second rebuild off a retained source is the cheapest "
        "checkpoint in the estate"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave40.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX40-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave40-children-retention-v1",
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
