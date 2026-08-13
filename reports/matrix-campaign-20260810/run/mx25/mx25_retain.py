#!/usr/bin/env python3
"""Wave 25 retention -- REFUSES TO RUN until this wave's own manifest is written.

**This file was cut from wave 24's by the same blanket rename that cuts the rest
of the tree, and a retention manifest is the one artifact in the tree that a
blanket rename makes ACTIVELY DANGEROUS.**  Wave 24's KEEP set named twelve
children by tag -- ``MX24-V32``, ``MX24-V8``, ``MX24-V8X13`` and so on -- and the
rename turns those into ``MX25-*`` tags that either do not exist or, worse, name
different children.  Run as inherited, it would have deleted every checkpoint
this wave built, because a wave-25 child whose tag is not in a wave-24 KEEP set
is, to this script, a duplicate.

So the inherited body is gone and this is what is left in its place.  The wave's
real manifest is written HERE, by hand, after the reviews are sealed, from the
verdicts this wave actually recorded -- and the run is a DRY RUN first, always,
with the delete list printed and read before anything is unlinked.

Usage::

    mx25_retain.py --plan            # print what would happen; never deletes
    mx25_retain.py --apply           # delete, after --plan has been read
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

W = Path("/mnt/ai_data/neural-forge-matrix/wave25")

#: Written after the reviews, never inherited.  Tag -> why this checkpoint is
#: worth its bytes.  An empty mapping is not a policy and the script says so.
KEEP: dict[str, str] = {}

#: Tag -> why this checkpoint is NOT worth its bytes.  Every built child must
#: appear in exactly one of the two mappings or the script refuses.
DROP: dict[str, str] = {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", type=Path, default=W / "children-retention-manifest-wave25.json")
    args = ap.parse_args()

    built = sorted(p.name for p in (W / "children").glob("MX25-*") if p.is_dir())
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
        "schema": "neural-forge-matrix-wave25-children-retention-v1",
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
