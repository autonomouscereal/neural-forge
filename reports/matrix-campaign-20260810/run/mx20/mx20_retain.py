#!/usr/bin/env python3
"""Wave 20 retention: five children kept, three deleted and named by digest.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive -- with this wave's readings written
down rather than assumed:

* the five kept are the wave's ANSWERS, not its best scores.  Three of them
  (`W768LS`, `W768ST`, `W768AW25`) are the same 165,978,624 parameters against
  the same corpus and differ only in the basis or the solve, which is what makes
  the comparison between 3/15, 2/15 and 0/15 a controlled measurement rather
  than three unrelated children.  Deleting any one of them deletes one arm of
  the only controlled experiment this wave ran.
* ``MX20-W768AW25`` is kept although it scored **0/15**.  It is the wave's
  central negative result and the one child whose benchmark row disagrees with
  its review in the awkward direction -- better than the living reader-PCA child
  on four of five accuracy tasks, and zero cases held.  A disagreement you have
  deleted is a sentence in a note.
* ``MX20-W704`` is kept as the bisection rung: it is what locates the cliff to
  a 64-wide interval, and its retained read energy (0.8953) is the tightest
  upper bound the campaign has on where the floor sits.

``MX20-W768AW100``, ``MX20-W768RG`` and ``MX20-W704AW25`` are deleted as
re-derivable: all three are 0/15, each is one command against a committed
builder and a sealed basis whose digest is in its receipt, and their tensor
digests are named below so a rebuild can be checked rather than trusted.

The two CBT probe checkpoints and the F-LANDMARK state are NOT children and are
not covered by the best-five rule; `probeq.sh` writes them and the wave keeps
the reviewed one.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave20"
KEEP = {
    "MX20-W768LS",
    "MX20-W768ST",
    "MX20-W768AW25",
    "MX20-W768EV",
    "MX20-W704",
}
REASON = {
    "MX20-W768LS": (
        "3/15 at 0.722610x -- wave 19's sealed reader-PCA basis with the "
        "writers' lambda chosen on an internal fit split; the same score as the "
        "child it controls for, on a different set of cases, and the only child "
        "in the wave that recovers context-retrieval"
    ),
    "MX20-W768ST": (
        "2/15 at 0.722610x -- the stale arm, which prices the conditioned solve "
        "on this axis for the first time at one review case"
    ),
    "MX20-W768AW25": (
        "0/15 at 0.722610x -- the attribution-weighted basis at beta 0.25: same "
        "parameters, same retained read energy to 0.0015, seventeen of 768 "
        "directions exchanged, three review cases gone and four of five "
        "accuracy benchmarks slightly better.  The wave's central measurement"
    ),
    "MX20-W768EV": (
        "1/15 at 0.722610x -- wave 19's sealed evidence-pca basis, finally built "
        "against: more raw energy retained and less read energy, and it is the "
        "measurement that says which of the two the law is about"
    ),
    "MX20-W704": (
        "0/15 at 0.656116x -- the bisection rung inside wave 19's cliff, at "
        "0.8953 retained read energy: the tightest bound the campaign has on "
        "where the behavioural floor sits"
    ),
}
DELETE_REASON = {
    "MX20-W768AW100": (
        "0/15 at 0.722610x -- the attribution-weighted basis taken to beta 1.0, "
        "0.8967 retained read energy, below the floor; re-derivable from the "
        "committed builder and the sealed basis E-attribution-weighted-b10-768"
    ),
    "MX20-W768RG": (
        "0/15 at 0.722610x -- the reader-generalized basis at 0.4780 retained "
        "read energy; re-derivable from the committed builder and the sealed "
        "basis E-reader-generalized-768"
    ),
    "MX20-W704AW25": (
        "0/15 at 0.656116x -- attribution-weighted at 704, fifteen empty lanes; "
        "re-derivable from the committed builder and its sealed basis"
    ),
}


def digest_of(name: str) -> str | None:
    receipt = f"{W}/children/{name}/mx20-build-receipt.json"
    if not os.path.exists(receipt):
        return None
    return json.load(open(receipt, encoding="utf-8")).get("child_tensors_sha256")


def main() -> int:
    apply = "--apply" in sys.argv
    manifest = {
        "schema": "neural-forge-matrix-wave20-children-retention-v1",
        "policy": (
            "the best five by sealed score plus any child a later wave cannot "
            "re-derive; every deleted child is named with the sha256 of the "
            "model.safetensors it was sealed from"
        ),
        "kept": [],
        "deleted": [],
    }
    for name in sorted(os.listdir(f"{W}/children")):
        path = f"{W}/children/{name}"
        if not os.path.isdir(path):
            continue
        size = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _, files in os.walk(path)
            for f in files
        )
        row = {
            "child": name,
            "bytes": size,
            "child_tensors_sha256": digest_of(name),
        }
        if name in KEEP:
            row["reason"] = REASON.get(name, "")
            manifest["kept"].append(row)
        else:
            row["reason"] = DELETE_REASON.get(name, "not in the retained set")
            manifest["deleted"].append(row)
            if apply:
                shutil.rmtree(path)
    manifest["bytes_freed"] = sum(r["bytes"] for r in manifest["deleted"])
    manifest["applied"] = apply
    out = f"{W}/children-retention-manifest-wave20.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(
        f"kept {len(manifest['kept'])}, deleted {len(manifest['deleted'])}, "
        f"freed {manifest['bytes_freed'] / 2**30:.2f} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
