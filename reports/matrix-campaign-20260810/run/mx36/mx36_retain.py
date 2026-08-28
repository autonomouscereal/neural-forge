#!/usr/bin/env python3
"""Wave 36 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the seventh wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32,
33, 34 and 35 each replaced the inherited body with a refusal and wrote their
KEEP set by hand.  Wave 36 does the same and copies neither: the tags below are
written after this wave's reviews are sealed, from the verdicts this wave
actually recorded, and the run is a DRY RUN first, always, with the delete list
printed and read before anything is unlinked.

The wave-36 cutter's HAND_EDIT set copied this file VERBATIM, so wave 35's four
keeps and seven drops arrived here still spelled for wave 35 -- which is exactly
what makes the emptied mapping below a decision and not an artifact of a
substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx36_retain.py --plan            # print what would happen; never deletes
    mx36_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave36")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: EMPTY AT THE TREE CUT, BY DECISION.  Wave 35's four keeps and seven drops
#: arrived here VERBATIM and were deleted rather than renamed; the inherited
#: justification paragraphs went with them, because prose arguing for last
#: wave's rows is how a retention decision gets made by the previous wave.  The
#: mapping is filled in after this wave's verdicts are sealed, from those
#: verdicts, and the standing rule the next wave may argue with is unchanged:
#: keep the rows carrying a CLAIM no other artifact re-derives, drop what one
#: CPU pass rebuilds from a retained source, and state the deviation whenever
#: the count is not best-five.
#:
#: FILLED IN AFTER THIS WAVE'S VERDICTS WERE SEALED.  WAVE 36 KEEPS THREE OF
#: EIGHT AND STATES THE DEVIATION, because eight checkpoints is not eight
#: claims: one of them is BYTE-IDENTICAL TO THE PARENT by construction, two are
#: single CPU passes off a body this list keeps, and two are ~100-second
#: terminal-gain arms off wave-35 sources that wave 35 retained.  Three is what
#: is left carrying something no other artifact re-derives.
KEEP: dict[str, str] = {
    "MX36-W896CL1792": (
        "THE NEW BODY, and the only artifact in this wave that cost a conditioned "
        "capture: 793 s of GPU over 24 batches on the composed width x MLP path "
        "that no builder in this estate could produce until this wave lifted the "
        "guard.  It is the ONLY --hidden-size composed with --ff checkpoint in "
        "existence, it is the source of both of this wave's composed vocabulary "
        "children, and deleting it converts every future child on that "
        "composition -- the K = 9,216 and K = 8,192 corners nobody has built, the "
        "excised three-axis body this wave registered and did not build -- from a "
        "40-second CPU pass into a 793-second GPU build.  It also carries the "
        "fourteen per-site COMPOSED_READER_EXACTNESS measurements in its receipt, "
        "which are the numbers the lift's whole argument rests on."
    ),
    "MX36-W896R": (
        "THE ONLY EXISTING CHECKPOINT OF ARCH.W896, and it is PROVABLY wave 19's "
        "bytes: rebuilt by this wave's builder against wave 19's own sealed basis "
        "and digesting bfa58964549778222141e01f071e8e7fd63935499ad3f26daf203da1adbb5b82, "
        "identical to the sealed body across SEVENTEEN WAVES.  It is the control "
        "the entire E1 result is differenced against -- ARCH.W896 reads 4/15 and "
        "the composed body reads 0/15, and that subtraction is only attributable "
        "because the width path is proven not to have moved.  THIS IS A STATED "
        "DEVIATION FROM THE CLAIM RULE: its claim IS re-derivable, in 294 s, from "
        "the retained wave-19 basis.  It is kept anyway because it is a bridge to "
        "a wave whose children were all deleted, and that bridge cannot be "
        "rebuilt at all if the basis artifact is ever lost."
    ),
    "MX36-V9CL896G10": (
        "0.437349x at 4/15 sealed and 5/15 canonical -- the deepest four the "
        "campaign has sealed and the deepest five the canonical lens has read.  "
        "It holds the sealed 3/15 and 4/15 rungs and the canonical 3/15, 4/15 and "
        "5/15 rungs, five frontier positions on one checkpoint, and it is the "
        "first arm in twelve ladders where lambda 1.0 beat lambda 0.3.  Its "
        "distinguishing lane -- context-retrieval -- is shared with neither its "
        "control nor the 0.3 arm, so unlike wave 35's tie it is NOT re-derivable "
        "from another child's bytes."
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX36-GATE1024": (
        "THE IDENTITY GATE, and it is the cleanest deletion this estate has ever "
        "made: its model.safetensors digests f630da86651136c9aee893b04b7542007e"
        "90fdd718355358e57e7ecc31517cfd, which is THE PARENT'S OWN FILE DIGEST, so "
        "the checkpoint is a second copy of a file that is already on this "
        "server.  What it proves -- that the composed width x MLP path is a strict "
        "generalisation of the parent, 132 of 132 tensors bit-identical, loader "
        "count 229,693,184 -- is a sha256 in a build receipt and does not need "
        "197 MB behind it.  It rebuilds in 196 s from the parent"
    ),
    "MX36-V12W896CL1792": (
        "K=12,288 on the composed body, one 40-second CPU vocabulary pass on "
        "MX36-W896CL1792, which this list KEEPS.  It reads zero on both standards "
        "and takes no rung; its one result -- that 120,700,032 is the loader's "
        "count and not merely the arithmetic's -- is in its build receipt and its "
        "seal"
    ),
    "MX36-V16W896CL1792": (
        "K=16,384 on the composed body, one 40-second CPU vocabulary pass on the "
        "same kept body.  Zero on both standards; it was built as the CONTROL for "
        "the rung above it and a control that reads zero beside a zero has "
        "reported everything it can"
    ),
    "MX36-V9CL896G03": (
        "terminal gain lambda 0.3 on wave 35's retained MX35-V9CL896, ~100 s of "
        "GPU.  It seals THREE and holds the sealed 3/15 rung -- but the arm this "
        "list KEEPS reads FOUR at the identical ratio on a strict superset of its "
        "lanes, so every rung this child takes is also taken by a retained "
        "checkpoint.  Both endpoints of the ladder it is the middle of are on "
        "disk: its source is retained by wave 35 and its sibling is retained here"
    ),
    "MX36-V12CL896X13G03": (
        "terminal gain lambda 0.3 on wave 35's retained MX35-V12CL896X13, ~100 s "
        "of GPU.  Its result is a LANE SWAP rather than a count -- it gains the "
        "summarisation lane and loses the email lane for a net of zero against "
        "its control -- and a lane swap is a statement about two dossiers, both "
        "of which are committed, rather than about a checkpoint.  It takes no "
        "rung: its ratio's sealed 1/15 rung is held by its own control"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave36.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX36-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave36-children-retention-v1",
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
