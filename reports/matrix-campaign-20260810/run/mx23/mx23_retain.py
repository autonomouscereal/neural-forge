#!/usr/bin/env python3
"""Wave 23 retention: eleven children built, five kept, six deleted and named.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive -- and this wave is the first one where
the second clause does real work in the *other* direction.

Every vocabulary child is a **row gather on a sealed child's own bytes**.
Re-deriving one costs about forty seconds of CPU and no GPU, no capture and no
solve, and the source it is derived from is itself retained under an earlier
wave's manifest.  So a wave-23 child that was never graded is not a measurement
and not an artifact: it is a cache.  The five that carry a sealed ADDENDUM 6
verdict are kept because a verdict without the checkpoint it was read from is a
claim; the six that were built and never reviewed are deleted, by name, with
the exact command that rebuilds them.

* ``MX23-V16`` (9/15, 0.780874x) -- the pure axis, and the deepest child in the
  campaign at eight or more cases by seventeen points of ratio.  Everything else
  in the wave is attributed against it.
* ``MX23-V16X13`` (5/15, 0.728353x) -- the vocabulary fold on the campaign's
  highest-scoring body.  The mildest second composition step ever measured here.
* ``MX23-V16W896`` (2/15, 0.667288x) -- the multiplying pair, where both folds
  land on the token table from opposite sides.
* ``MX23-V16W896X13`` (1/15, 0.623329x) -- the three-axis child, and L3's rule
  holding on a third axis it had never been tried against.
* ``MX23-V16CL896`` (1/15, 0.469305x) -- under half the parent, standard shape,
  and the first child below 0.5x in this campaign that holds anything.

The children live on the 4.5 TB share, not on the root filesystem the disk
policy is about, and the freed figure is reported for the share.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave23"
KEEP = {
    "MX23-V16",
    "MX23-V16X13",
    "MX23-V16W896",
    "MX23-V16W896X13",
    "MX23-V16CL896",
}
REASON = {
    "MX23-V16": (
        "9/15 at 0.780874x, standard shape -- the vocabulary row coreset alone, "
        "zero solve, one tensor touched.  Deepest child in the campaign at >= 8 "
        "of 15 by seventeen points of ratio (the previous holder was ARCH.X13A "
        "at 0.947479x), lambada perplexity 72.99 against the parent's 24.39, and "
        "zero of sixteen lanes at the token cap"
    ),
    "MX23-V16X13": (
        "5/15 at 0.728353x -- the vocabulary fold composed with the excision of "
        "block 13, i.e. the campaign's 13/15 child carrying it.  13/15 and 9/15 "
        "compose to 5/15, which is the mildest second step in 662 rows and is "
        "the whole measurement"
    ),
    "MX23-V16W896": (
        "2/15 at 0.667288x -- the multiplying pair.  The width fold cuts the "
        "token table's columns and the vocabulary fold its rows, 82% of the "
        "interface for two folds neither of which was aimed at it, and the "
        "summarisation lane survives intact"
    ),
    "MX23-V16W896X13": (
        "1/15 at 0.623329x -- three axes.  L3's one-step rule tried against an "
        "axis it had never met and holding"
    ),
    "MX23-V16CL896": (
        "1/15 at 0.469305x -- UNDER HALF THE PARENT, standard shape, and the "
        "first child below 0.5x in this campaign to hold a case at all "
        "(ARCH.W512 at 0.463481x reads 0/15; MXO2 and MXQ6 reached 0.5x and read "
        "0/15 twice)"
    ),
}
REBUILD = (
    "bash queues/vocabq.sh 16384 <TAG> -- a row gather on a sealed child's own "
    "bytes, about forty seconds of CPU, no GPU, no capture, no solve"
)


def main() -> int:
    apply = "--apply" in sys.argv
    rows = []
    freed = 0
    root = os.path.join(W, "children")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        size = sum(
            os.path.getsize(os.path.join(base, f))
            for base, _, files in os.walk(path)
            for f in files
        )
        keep = name in KEEP
        rows.append(
            {
                "child": name,
                "kept": keep,
                "bytes": size,
                "reason": REASON.get(
                    name,
                    "built and never graded; a vocabulary child is a row gather "
                    "on a sealed child's bytes, so this is a cache and not a "
                    "measurement",
                ),
                "rebuild": None if keep else REBUILD,
            }
        )
        if not keep:
            freed += size
            if apply:
                shutil.rmtree(path)
    report = {
        "schema": "neural-forge-matrix-wave23-children-retention-v1",
        "policy": "the best five by sealed score, plus any child a later wave "
        "cannot re-derive.  Every wave-23 child IS re-derivable in seconds, so "
        "the second clause keeps nothing here and the graded five are kept "
        "because a verdict without its checkpoint is a claim",
        "built": len(rows),
        "kept": sum(1 for r in rows if r["kept"]),
        "deleted": sum(1 for r in rows if not r["kept"]),
        "freed_bytes": freed,
        "freed_gib": round(freed / (1024**3), 3),
        "applied": apply,
        "children": rows,
    }
    out = f"{W}/children-retention-manifest-wave23.json"
    with open(out, "w") as handle:
        json.dump(report, handle, indent=1)
    print(
        f"kept {report['kept']}, deleted {report['deleted']}, "
        f"freed {report['freed_gib']} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
