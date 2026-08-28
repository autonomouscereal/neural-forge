#!/usr/bin/env python3
"""Wave 34 retention -- REFUSES TO RUN until this wave's own manifest is written.

**A retention manifest is the one artifact in the tree that a blanket rename
makes ACTIVELY DANGEROUS, and this is the fifth wave to say so in this file.**
Wave 24's KEEP set named twelve children by tag -- ``MX24-V32``, ``MX24-V8`` and
so on -- and wave 25's rename turned those into ``MX25-*`` tags that either did
not exist or, worse, named different children; run as inherited it would have
deleted every checkpoint that wave built, because a wave-25 child whose tag is
not in a wave-24 KEEP set is, to this script, a duplicate.  Waves 25, 26, 31, 32
and 33 each replaced the inherited body with a refusal and wrote their KEEP set
by hand.  Wave 34 does the same and copies neither: the tags below are written
after this wave's reviews are sealed, from the verdicts this wave actually
recorded, and the run is a DRY RUN first, always, with the delete list printed
and read before anything is unlinked.

The wave-34 cutter's HAND_EDIT set copied this file VERBATIM, so wave 33's two
keeps and four drops arrived here still spelled for wave 33 -- which is exactly
what makes the emptied mapping below a decision and not an artifact of a
substitution that happened to succeed.

**And it runs AFTER the package, not before.**  Wave 26 ran retention first and
deleted three children, taking their build receipts with them, because a build
receipt lives inside the child directory.  All three rebuilt byte-for-byte, but
the ordering is now a rule rather than a preference: ``package.sh`` first, then
``--plan``, then ``--apply``.

Usage::

    mx34_retain.py --plan            # print what would happen; never deletes
    mx34_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave34")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
#:
#: THE POLICY IS WRITTEN AFTER THE VERDICTS, and wave 34's verdicts are sealed,
#: so it is written here now.  Wave 33's policy paragraphs were DELETED rather
#: than renamed: they argued for a specific set of last wave's rows, and
#: inherited prose justifying a keep set is how a retention decision gets made by
#: the previous wave.  The standing rule the next wave may argue with: keep the
#: rows carrying a CLAIM no other artifact re-derives, drop what one CPU pass
#: rebuilds from a retained source, and state the deviation whenever the count is
#: not best-five.
#:
#: WAVE 34 STATES ONE DEVIATION FROM THAT RULE UP FRONT, because the rule's
#: second clause is not quite true of this wave.  The vocabulary folds ARE one
#: CPU pass (~40 s each).  The terminal-gain arms are NOT: they are GPU
#: post-processes, cheap (~100 s) but not CPU.  And the two BODIES are neither --
#: they are 451 s and 743 s of GPU with a 24-batch conditioned capture each.  So
#: the keep set is chosen on rebuild COST as well as on claim, and the two bodies
#: are kept even though neither carries a frontier claim, because everything this
#: wave and the next one will build sits one cheap pass on top of them.
KEEP: dict[str, str] = {
    "MX34-CL896": (
        "THE SOURCE BODY OF THE ENTIRE DEEP FAMILY.  451 s of GPU with a 24-batch "
        "conditioned capture; bit-identical to wave 15's sealed body 19 waves later "
        "and to its own second build in this wave.  Every C1 child, and the "
        "ARCH.V9CL896 the wave note tells the next wave to build, is one 40-second "
        "CPU pass on top of it.  Deleting it converts that pass into a GPU build."
    ),
    "MX34-CL1792X13": (
        "The other body: 743 s of GPU, bit-identical to wave 31's sealed cell.  It "
        "carries no frontier claim -- C2 is closed -- but it is the only artifact "
        "from which the two C2 children can be re-made without a capture, and the "
        "next wave may want to re-read them under a changed standard rather than "
        "rebuild them under a changed builder."
    ),
    "MX34-V10CL896": (
        "0.441915x, THE DEEPEST CHILD THIS CAMPAIGN HAS EVER BUILT, and it holds the "
        "1/15 and 2/15 rungs on BOTH ladders.  Its dossier is committed, but the "
        "checkpoint is what a future robustness or penalty probe has to re-decode, "
        "and the frontier's floor is the row least worth re-deriving from scratch."
    ),
    "MX34-V12CL896": (
        "The lambda ladder's ZERO-SOLVE CONTROL.  The 4/15 claim is a DELTA against "
        "this child's two, and gainq.sh needs the source checkpoint on disk to "
        "rebuild any arm at all.  A control whose bytes are gone is a control."
    ),
    "MX34-V12CL896G03": (
        "0.451045x at 4/15, the deepest four in the campaign on BOTH ladders and the "
        "holder of the 3/15 and 4/15 rungs.  It is also the child the C3 penalty "
        "diagnostic and the C4 calibration both ran on, so three of this wave's four "
        "results are read off these bytes."
    ),
}

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: NOT worth its bytes, naming what rebuilds it and from which retained source.
DROP: dict[str, str] = {
    "MX34-V16CL896": (
        "one 40-second CPU vocabulary fold at K=16,384 on the retained MX34-CL896, "
        "and it reproduced BIT-IDENTICALLY to wave 23's sealed child, so the rebuild "
        "is not merely equivalent, it is the same bytes"
    ),
    "MX34-V16CL896G03": (
        "terminal gain lambda 0.3 on MX34-V16CL896, which is itself one CPU pass "
        "from the retained MX34-CL896.  Two cheap passes; it reads three, which is "
        "the count its two larger-K siblings already carry"
    ),
    "MX34-V12CL896G01": (
        "terminal gain lambda 0.1 on the retained MX34-V12CL896, ~100 s of GPU.  It "
        "reads its control's two and its one distinguishing lane -- 4:15 MINUTES -- "
        "is quoted verbatim in the seal and the wave note"
    ),
    "MX34-V12CL896G10": (
        "terminal gain lambda 1.0 on the retained MX34-V12CL896, ~100 s of GPU.  It "
        "reads its control's two, and four of its sixteen lanes are byte-identical to "
        "that control, so most of its dossier IS the control's dossier"
    ),
    "MX34-V8CL1792X13": (
        "one CPU vocabulary fold at K=8,192 on the retained MX34-CL1792X13"
    ),
    "MX34-V12CL1792X13": (
        "one CPU vocabulary fold at K=12,288 on the retained MX34-CL1792X13.  C2 is "
        "closed by its own result and nothing will be built on it"
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave34.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX34-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave34-children-retention-v1",
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
