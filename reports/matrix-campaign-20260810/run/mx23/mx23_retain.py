#!/usr/bin/env python3
"""Wave 23 retention: three children built, three kept, and nothing freed.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive.  This wave built exactly three, and
each of the three is one arm of a controlled comparison whose other arm is
already sealed in an earlier wave at the IDENTICAL parameter count or on the
same ladder.  Deleting any of them deletes a measurement, not a checkpoint:

* ``MX23-MM896`` (2/15, 0.859023x) is the max-min basis at the living width and
  the wave's central negative.  Its two controls -- wave 19's ``W896`` at 4/15
  and wave 21's ``W896LS`` at 3/15 -- are the same 197,311,616 parameters, so
  the only variable in the comparison is which 896-dimensional subspace the
  child keeps.  It is also the only child in the campaign whose basis equalises
  the fit-side per-endpoint read retention to five decimal places.
* ``MX23-W720`` (0/15, 0.672632x) is the rung that halves the last interval of
  the width slope.  It is the deepest 0/15 width child anyone will need to
  build between 736 and 704, and its 0.9015 is the tightest UPPER bound the
  campaign has on where the floor is.
* ``MX23-W896LSKV6JQ`` (1/15, 0.841047x) is the campaign's two case-recovery
  mechanisms in one child, against wave 19's ``W896KV6JS`` at 2/15 and the same
  193,182,848 parameters.  The pair is the finding -- the two mechanisms do not
  add, and their benchmark rows are indistinguishable while their reviews are
  not -- and one arm of a controlled pair is not a result.

Nothing is deleted, so nothing is freed, and that is reported as zero rather
than dressed up.  The children live on the 4.5 TB share, not on the root
filesystem the disk policy is about.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave23"
KEEP = {
    "MX23-MM896",
    "MX23-W720",
    "MX23-W896LSKV6JQ",
}
REASON = {
    "MX23-MM896": (
        "2/15 at 0.859023x -- the reader-maxmin basis at the LIVING width, "
        "which lifts the worst endpoint's held-out retained read energy from "
        "0.9440 to 0.9530, flattens the fit-side spread across 29 readers from "
        "0.0231 to 0.00055, exchanges 24.69 of 896 directions, and holds two "
        "cases against its controls' four and three at the same parameters"
    ),
    "MX23-W720": (
        "0/15 at 0.672632x, 0.9015 pooled retained read energy -- the rung "
        "that halves the last interval of the width slope from 32 dimensions "
        "to 16, and the child that puts the floor between 736 and 720"
    ),
    "MX23-W896LSKV6JQ": (
        "1/15 at 0.841047x -- the width lambda split composed with the head "
        "merge, against wave 19's W896KV6JS at 2/15 and the identical "
        "193,182,848 parameters; the two case-recovery mechanisms do not add, "
        "and the seven benchmark columns cannot tell the pair apart"
    ),
}
DELETE_REASON: dict[str, str] = {}


def digest_of(name: str) -> str | None:
    receipt = f"{W}/children/{name}/mx23-build-receipt.json"
    if not os.path.exists(receipt):
        return None
    return json.load(open(receipt, encoding="utf-8")).get("child_tensors_sha256")


def main() -> int:
    apply = "--apply" in sys.argv
    if not KEEP:
        raise SystemExit(
            "REFUSED: KEEP is empty, which is the builder commit's state.  The "
            "retained set is a function of the sealed scores; fill it in the "
            "sealing commit, naming every child kept and every child deleted."
        )
    manifest = {
        "schema": "neural-forge-matrix-wave23-children-retention-v1",
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
    out = f"{W}/children-retention-manifest-wave23.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(
        f"kept {len(manifest['kept'])}, deleted {len(manifest['deleted'])}, "
        f"freed {manifest['bytes_freed'] / 2**30:.2f} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
