#!/usr/bin/env python3
"""Wave 21 retention: all five children kept, and the reason is not sentiment.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive.  This wave built exactly five, and each
of the five is one arm of a controlled comparison whose other arm is already
sealed in an earlier wave.  Deleting any of them deletes a measurement, not a
checkpoint:

* ``MX21-W736`` (1/15, 0.689220x) is the rung that closes the cliff.  It is the
  only child ever built inside the 0.90-0.92 band of retained read energy, and
  its 0.9075 is the tightest LOWER bound the campaign has on where the floor is
  not.
* ``MX21-W736LS`` (0/15) is its matched control at the IDENTICAL 158,309,216
  parameters and the identical sealed basis.  The pair is the finding -- the
  writers' lambda split costs the last case at the cliff, where at 768 it cost
  nothing -- and one arm of a controlled pair is not a result.
* ``MX21-MM768`` (0/15) is the wave's central negative and the most expensive
  thing in it: the basis that maximises the MINIMUM per-endpoint retained read
  energy, which does exactly what it was built to do and is dead.  A refutation
  you have deleted is a sentence in a note.
* ``MX21-W896LS`` (3/15) is the wave's best child and the control for wave 19's
  sealed 4/15 at the same parameter count.
* ``MX21-W896LSX13`` (2/15) is the only composition in the wave and the child
  that shows the lambda split's recovered capability surviving a composition
  step, against wave 19's sealed 1/15 twin whose summarisation lane it
  reproduces byte for byte.

Nothing is deleted, so nothing is freed, and that is reported as zero rather
than dressed up.  The children live on the 4.5 TB share, not on the root
filesystem the disk policy is about.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave21"
KEEP = {
    "MX21-W736",
    "MX21-W736LS",
    "MX21-MM768",
    "MX21-W896LS",
    "MX21-W896LSX13",
}
REASON = {
    "MX21-W736": (
        "1/15 at 0.689220x -- the only child ever built inside the 0.90-0.92 "
        "band of retained read energy (0.9075 held-out), and the rung that "
        "closes the cliff to 32 dimensions"
    ),
    "MX21-W736LS": (
        "0/15 at the IDENTICAL 158,309,216 parameters and the identical sealed "
        "basis -- the matched control that prices the writers' lambda split at "
        "the cliff, where it costs the last case"
    ),
    "MX21-MM768": (
        "0/15 at 0.722610x -- the reader-maxmin basis, which raises the worst "
        "endpoint's retained read energy from 0.8801 to 0.8997, cuts the "
        "endpoints under the 0.90 floor from four to one, reconstructs the "
        "parent's logits better than the living child, and is dead in all "
        "sixteen lanes.  The wave's central negative result"
    ),
    "MX21-W896LS": (
        "3/15 at 0.859023x -- the wave's best child and the matched control for "
        "wave 19's sealed 4/15 at the same parameter count; it recovers "
        "planning-dependencies and gives up two other cases"
    ),
    "MX21-W896LSX13": (
        "2/15 at 0.815064x -- the wave's only composition, against wave 19's "
        "sealed 1/15 twin whose summarisation lane it reproduces byte for byte; "
        "the capability the lambda split recovers survives the composition step"
    ),
}
DELETE_REASON: dict[str, str] = {}


def digest_of(name: str) -> str | None:
    receipt = f"{W}/children/{name}/mx21-build-receipt.json"
    if not os.path.exists(receipt):
        return None
    return json.load(open(receipt, encoding="utf-8")).get("child_tensors_sha256")


def main() -> int:
    apply = "--apply" in sys.argv
    manifest = {
        "schema": "neural-forge-matrix-wave21-children-retention-v1",
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
    out = f"{W}/children-retention-manifest-wave21.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(
        f"kept {len(manifest['kept'])}, deleted {len(manifest['deleted'])}, "
        f"freed {manifest['bytes_freed'] / 2**30:.2f} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
