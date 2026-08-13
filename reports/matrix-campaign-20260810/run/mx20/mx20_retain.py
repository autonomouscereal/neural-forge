#!/usr/bin/env python3
"""Wave 19 retention: five children kept, three deleted and named by digest.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive -- with two wave-specific readings
written down rather than assumed:

* ``MX20-W512`` is kept although it scored **0/15**.  It is the only child in
  this campaign under half the parent (0.463481x), and what a 0.463481x LFM2
  built by an exact fold actually emits -- eight empty strings and eight lanes
  of quotation marks -- is the answer to a question the deck has asked since
  wave 14.  A landmark you have deleted is a number in a note.
* ``MX20-IDENT`` is **deleted**, and precisely because its gate passed: all 132
  of its tensors are bit-identical to the parent's, so keeping it is keeping the
  parent twice.  The gate is in its build receipt, which is committed.

``MX20-W640`` and ``MX20-Q01050913`` are deleted as re-derivable: both are
0/15, both are one command against a committed builder and a sealed basis, and
their tensor digests are named below so a rebuild can be checked rather than
trusted.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave20"
KEEP = {
    "MX20-W896",
    "MX20-W768",
    "MX20-W896KV6JS",
    "MX20-W896X13",
    "MX20-W512",
}
REASON = {
    "MX20-W896": (
        "4/15 at 0.859023x -- the best rung of the hidden_size ladder and the "
        "child the axis's whole claim rests on"
    ),
    "MX20-W768": (
        "3/15 at 0.722610x -- the deepest child in the campaign with a non-zero "
        "score, 4.7 points of net ratio below wave 16's frontier"
    ),
    "MX20-W896KV6JS": (
        "2/15 at 0.841047x -- width composed with the head merge, and the best "
        "of the two width compositions"
    ),
    "MX20-W896X13": (
        "1/15 at 0.815064x -- width composed with the census's best single "
        "excision, and the wave's cleanest demonstration that the second axis "
        "is still fatal"
    ),
    "MX20-W512": (
        "0/15 at 0.463481x -- the campaign's only child under half the parent, "
        "kept as the landmark it is: a 0/15 nobody has to take on trust"
    ),
}
DELETE_REASON = {
    "MX20-IDENT": (
        "the identity gate: 132 of 132 tensors bit-identical to the parent, so "
        "keeping it keeps the parent twice; the gate rides in its build receipt"
    ),
    "MX20-W640": "0/15 at 0.590763x, re-derivable from the committed builder and the sealed basis",
    "MX20-Q01050913": "0/15 at 0.789916x, re-derivable from the committed builder",
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
