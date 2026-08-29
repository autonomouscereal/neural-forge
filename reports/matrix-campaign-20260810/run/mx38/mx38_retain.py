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

    mx38_retain.py --plan            # print what would happen; never deletes
    mx38_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave38")

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
#: FILLED IN AFTER THIS WAVE'S VERDICTS WERE SEALED.  WAVE 38 KEEPS TWO OF SIX
#: AND STATES THE DEVIATION, because six checkpoints is not six claims.  One keep
#: is ON the claim rule and one is AGAINST it, and both are argued below rather
#: than counted.  Four of the six are ONE 40-SECOND CPU VOCABULARY PASS off a
#: retained body -- wave 37's MX37-CL896X09 for three of them and this wave's own
#: kept body for the fourth -- and this wave demonstrated that exact path FOUR
#: TIMES in its own hour, every one landing on the parameter count the closed
#: form predicted before the build.
KEEP: dict[str, str] = {
    "MX38-CL896X09X13": (
        "THE NEW BODY, and the only artifact in this wave that cost a conditioned "
        "capture: 289 s of GPU on --ff 896 --excise 9 --excise 13.  THIS IS THE "
        "KEEP THAT IS AGAINST THE CLAIM RULE AND THE EXCEPTION IS ARGUED RATHER "
        "THAN HIDDEN: its claim -- that composing both excisions leaves nothing on "
        "a body where either alone leaves one -- is entirely in its dossier and "
        "its seal, and a claim that lives in a dossier is normally the reason to "
        "delete the checkpoint.  It is kept for two reasons that are about the "
        "BYTES and not about the claim.  FIRST, it is the ONLY multi-index "
        "excision body in this estate and the only one ever composed with --ff: "
        "twelve layer types, a DOUBLE full_attention at positions 8 and 9 the "
        "parent never has, and a body that ENDS ON full_attention, which no other "
        "CL896 body does.  SECOND, wave 38 is the first wave in ten to exercise "
        "the descending deletion order and the module re-index that a multi-index "
        "excision depends on, and it had to assert both by name because nothing "
        "else had.  A future wave that wants to run mx18_mask_probe.py against "
        "that path -- the probe that measured 28.6 to 48.0 per cent of an "
        "attention module's mass landing on tokens that had not happened yet, on a "
        "body without wave 18's repair -- needs this checkpoint and not this "
        "dossier.  Rebuilding it is 289 s of GPU rather than 40 s of CPU, which is "
        "the other half of the argument"
    ),
    "MX38-V10CL896X09G10": (
        "0.4116484014x at 2/15 sealed and 2/15 canonical -- IT IS THE SOLE HOLDER "
        "OF THE SEALED TWO-OF-FIFTEEN RUNG, which it takes from ARCH.V12CL896X09 "
        "at 0.4207786331x by 0.0091302317 of stored ratio, and it is the deepest "
        "TWO this campaign has ever sealed.  This keep is ON the claim rule and is "
        "not an exception: wave 37 dropped both of its gain arms because each "
        "shared every rung it held with a RETAINED control at the identical ratio, "
        "and that is not true here -- its own control ARCH.V10CL896X09 holds ONE, "
        "so no other checkpoint in the estate holds this rung.  It also carries "
        "the strongest form of the uncertainty lane anywhere in the X09 family "
        "(all four withheld quantities AND the correct direction, where its "
        "control and every other X09 rung invert it), and it is the measurement "
        "that settled lambda 1.0 as a property of the vocabulary rung: +1 at "
        "K=10,240 on this excised body against +1 at K=10,240 on the unexcised "
        "one, on decodes that share ZERO of sixteen lanes"
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX38-V10CL896X09": (
        "K=10,240 on ARCH.CL896X09, ONE 40-SECOND CPU VOCABULARY PASS off wave "
        "37's RETAINED MX37-CL896X09, which wave 37's own retention keeps and "
        "which this wave used four times.  It holds the sealed ONE-of-fifteen rung "
        "at 0.4116484014x -- but it holds it JOINTLY with MX38-V10CL896X09G10, "
        "which this list KEEPS at the identical ratio, so every frontier position "
        "these bytes take is also taken by a retained checkpoint.  It is the upper "
        "half of the campaign's second 1,024-row bracket and that is the one "
        "argument for keeping it; the bracket is re-derivable in 80 seconds of CPU "
        "from a body wave 37 retains, and both dossiers are committed"
    ),
    "MX38-V9CL896X09": (
        "K=9,216 on ARCH.CL896X09, ONE 40-SECOND CPU VOCABULARY PASS off wave 37's "
        "RETAINED MX37-CL896X09.  It is the campaign's SECOND VOCABULARY DEATH and "
        "the most content-weighted zero in the deep record -- eight "
        "readable-and-wrong misses against seven artifacts -- and both of those "
        "are statements about a dossier, which is committed, and about a per-case "
        "file, which is committed.  A death holds no rung by construction.  It is "
        "the lower half of the second 1,024-row bracket and rebuilds in the same "
        "40 seconds as its neighbour"
    ),
    "MX38-V14CL896X09": (
        "K=14,336 on ARCH.CL896X09, ONE 40-SECOND CPU VOCABULARY PASS off wave "
        "37's RETAINED MX37-CL896X09.  It reads 2/15 at 0.4299088649x and TAKES NO "
        "RUNG: both rungs it could hold are held 0.018 of stored ratio deeper by "
        "children at 0.4116484014x.  One of its two lanes is not even its own -- "
        "uncertainty-insufficient-data is BYTE-IDENTICAL to ARCH.V12CL896X09's, so "
        "that Partial is wave 37's verdict on wave 37's bytes"
    ),
    "MX38-V12CL896X09X13": (
        "K=12,288 on ARCH.CL896X09X13, ONE 40-SECOND CPU VOCABULARY PASS off "
        "MX38-CL896X09X13, which this list KEEPS.  It is the deepest child this "
        "campaign has ever built at 0.3905124499x and it reads ZERO on both "
        "standards, with fourteen of sixteen lanes riding the cap.  Depth with no "
        "case is a number, and the number is in the ledger, in the dossier and in "
        "the seal.  Its one recorded oddity -- the note's 7:30 a.m. carried with "
        "the CORRECT meridiem, which no other child in the X09 family manages -- "
        "is quoted verbatim in its seal and in the canonical re-read"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave38.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX38-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave38-children-retention-v1",
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
