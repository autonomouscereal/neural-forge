#!/usr/bin/env python3
"""Wave 37 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the eighth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33, 34, 35 and 36 each replaced the inherited body with a refusal and wrote their
KEEP set by hand.  Wave 37 does the same and copies neither: the tags below are
written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-37 cutter's HAND_EDIT set copied this file VERBATIM, so wave 36's three
keeps and five drops arrived here still spelled for wave 36 -- which is exactly
what makes the emptied mapping below a decision and not an artifact of a
substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx37_retain.py --plan            # print what would happen; never deletes
    mx37_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave37")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: EMPTY AT THE TREE CUT, BY DECISION.  Wave 36's three keeps and five drops
#: arrived here VERBATIM and were deleted rather than renamed; the inherited
#: justification paragraphs went with them, because prose arguing for last
#: wave's rows is how a retention decision gets made by the previous wave.  The
#: mapping is filled in after this wave's verdicts are sealed, from those
#: verdicts, and the standing rule the next wave may argue with is unchanged:
#: keep the rows carrying a CLAIM no other artifact re-derives, drop what one
#: CPU pass rebuilds from a retained source, and state the deviation whenever
#: the count is not best-five.
#: FILLED IN AFTER THIS WAVE'S VERDICTS WERE SEALED.  WAVE 37 KEEPS TWO OF NINE
#: AND STATES THE DEVIATION, because nine checkpoints is not nine claims.  Seven
#: of them are ONE 40-SECOND CPU VOCABULARY PASS or a ~100-SECOND GPU
#: TERMINAL-GAIN SOLVE off a checkpoint this list or wave 34 retains, and this
#: wave PROVED that path three times: V14CL896, V16CL896 and V14CL896G03 were all
#: rebuilt from wave 34's retained CL896 body and all three came back
#: BIT-IDENTICAL to their sealed digests, the last of them including its GPU
#: solve.  A rebuild path that has been demonstrated three times in the same wave
#: is a rebuild path, not a hope.
KEEP: dict[str, str] = {
    "MX37-CL896X09": (
        "THE NEW BODY, and the only artifact in this wave that cost a conditioned "
        "capture: 316 s of GPU on --ff 896 --excise 9.  It is the ONLY body in "
        "this estate whose excision is NOT the final block -- blocks 10 through 13 "
        "shift down one slot, an attention block lands in a conv's index, and its "
        "exported config carries a DOUBLE full_attention at positions 8 and 9 that "
        "the parent never has.  It is the source of all four of this wave's "
        "0.420779x children, and deleting it converts each of them from a "
        "40-second CPU pass into a 316-second GPU build.  It also carries the "
        "per-site solve record the block-9-costs-three result rests on, and it is "
        "the control the two-token +1 in section 4.2 of NOTES-MX-AO is differenced "
        "against: its uncertainty lane names TWO of the four withheld quantities "
        "where its vocabulary child names FOUR, and that single divergence is the "
        "whole of the frontier move this wave made"
    ),
    "MX37-V12CL896X09": (
        "0.420779x at 2/15 sealed and 3/15 canonical -- it MOVES the sealed "
        "two-of-fifteen rung from ARCH.V9CL896 at 0.437349x and the canonical "
        "three-of-fifteen rung from ARCH.V9CL896G03 at the same ratio, 0.016570 of "
        "stored ratio each, and it does it at the campaign floor.  THIS IS A "
        "STATED DEVIATION FROM THE CLAIM RULE: its claim IS re-derivable, in 40 s "
        "of CPU, from the body this list also keeps.  It is kept anyway because it "
        "is the OTHER HALF of a two-token measurement.  The +1 that moved both "
        "rungs is one word: its body writes 'informed the time range, units, and "
        "information' and this child writes 'told the time range, units, sample "
        "size, or any information about the price'.  A future wave that wants to "
        "examine that divergence at the token or logit level needs both "
        "checkpoints on disk, not both dossiers -- a dossier records what was "
        "decoded and cannot answer a question about what was nearly decoded"
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX37-V12CL896X09G03": (
        "terminal gain lambda 0.3 on MX37-V12CL896X09, which this list KEEPS, "
        "~100 s of GPU at zero storage delta.  Its result is a LOSS rather than a "
        "count -- it goes from its control's two to ONE, the first terminal-gain "
        "arm in this campaign to lose a case outright on the deep family -- and a "
        "loss is a statement about two dossiers, both of which are committed.  It "
        "takes no rung: every rung at its ratio is held by the control this list "
        "keeps and by its own sibling"
    ),
    "MX37-V12CL896X09G10": (
        "terminal gain lambda 1.0 on MX37-V12CL896X09, which this list KEEPS, "
        "~100 s of GPU at zero storage delta.  It TIES its control at two and "
        "holds the same two rungs, so every frontier position it takes is also "
        "taken by a retained checkpoint at the identical ratio, and its whole "
        "result -- that lambda 1.0 buys nothing at K=12,288 on a body DEEPER in "
        "stored ratio than the rung where it bought two -- is a comparison of two "
        "counts, not a property of these bytes.  Its distinguishing lane is "
        "byte-identical to its sibling's on retrieval and its uncertainty lane "
        "is quoted verbatim in its own seal"
    ),
    "MX37-V14CL896": (
        "K=14,336 on ARCH.CL896, one 40-second CPU vocabulary pass on wave 34's "
        "retained MX34-CL896.  It is NOT a wave-37 cell -- wave 35 sealed it -- and "
        "it was rebuilt here only so a gain arm had a source.  Its entire content "
        "as an artifact is the reproduction control it already delivered: it "
        "digests f93af77c25340120, bit-identical to wave 35's seal two waves and "
        "two builders later, and that sha256 is in a build receipt"
    ),
    "MX37-V16CL896": (
        "K=16,384 on ARCH.CL896, one 40-second CPU vocabulary pass on wave 34's "
        "retained MX34-CL896.  Also not a wave-37 cell -- wave 23 sealed it.  Its "
        "content as an artifact is the campaign's LONGEST reproduction control and "
        "that control is already delivered: 55061febb9795b42, bit-identical to "
        "wave 23's seal FOURTEEN WAVES and four builders later"
    ),
    "MX37-V14CL896G03R": (
        "wave 35's ARCH.V14CL896G03 rebuilt so the F3 paraphrase surface had a "
        "checkpoint to decode.  It is BYTE-IDENTICAL to wave 35's sealed child "
        "(f26b4ee6c95b42a1) INCLUDING ITS GPU TERMINAL-GAIN SOLVE, so its whole "
        "content is a sha256 that is already in a receipt, and it rebuilds in one "
        "CPU pass plus ~100 s of GPU from wave 34's retained MX34-CL896.  Its one "
        "result -- 0 of 39 paraphrase restatements held -- is in "
        "wave37/paraphrase-score-wave37.json"
    ),
    "MX37-V14CL896G10": (
        "terminal gain lambda 1.0 at K=14,336, ~100 s of GPU on a source that is "
        "itself one CPU pass off wave 34's retained MX34-CL896.  It reads TWO, "
        "which is its control's count, and takes no rung -- the sealed two rung is "
        "held 0.040 of stored ratio deeper by a child this list keeps.  Its one "
        "result is a comparison: the ladder at this K is 2 / 3 / 2, the only "
        "ladder in the campaign where the solver objective and the review agree"
    ),
    "MX37-V16CL896G10": (
        "terminal gain lambda 1.0 at K=16,384, ~100 s of GPU on a source that is "
        "one CPU pass off wave 34's retained MX34-CL896.  It reads TWO against a "
        "control of one and takes no rung; its result is the +1 that BREAKS the "
        "monotone-in-K account, and that is a row in a table rather than a "
        "checkpoint.  Two of its three load-bearing lanes are byte-identical to "
        "children of waves 23 and 34 and the third is byte-identical to "
        "MX37-V14CL896G10's, so its dossier is the least independent in this wave"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave37.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX37-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave37-children-retention-v1",
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
