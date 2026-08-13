#!/usr/bin/env python3
"""Wave 22 retention, written in the sealing commit and not before.

The policy is the campaign's standing one -- the best five by sealed score, plus
any child a later wave cannot re-derive -- and the reason ``KEEP`` is empty in
the builder commit is that the policy cannot be applied to scores that do not
exist yet.  A retention script shipped with last wave's ``KEEP`` set is a script
that deletes this wave's children by inheritance.

Running it before the wave is graded is a refusal, not a no-op that quietly
reports zero kept.
"""

import json
import os
import shutil
import sys

W = "/mnt/ai_data/neural-forge-matrix/wave22"
KEEP: set[str] = set()
REASON: dict[str, str] = {}
DELETE_REASON: dict[str, str] = {}


def digest_of(name: str) -> str | None:
    receipt = f"{W}/children/{name}/mx22-build-receipt.json"
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
        "schema": "neural-forge-matrix-wave22-children-retention-v1",
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
    out = f"{W}/children-retention-manifest-wave22.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    print(
        f"kept {len(manifest['kept'])}, deleted {len(manifest['deleted'])}, "
        f"freed {manifest['bytes_freed'] / 2**30:.2f} GiB, applied={apply} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
