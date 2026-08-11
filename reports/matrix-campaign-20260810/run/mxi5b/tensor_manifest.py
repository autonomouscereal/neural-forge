#!/usr/bin/env python3
"""Digest every wave-5 installed-tensors.pt, and record that none was pruned.

Step 0 of wave 5B authorised freeing disk by deleting bit-exact re-derivable
`installed-tensors.pt` files after writing a sha256 manifest of them.  The
manifest is written; the deletions are not, and the reason is arithmetic rather
than caution:

* every one of these files lives under ``/mnt/ai_data``, a CIFS mount with
  1.8 TB free, which the same instruction puts explicitly off limits;
* the filesystem that is actually full is ``/`` at 100%, and not one byte of
  these files is on it.

So the manifest records what a future prune would delete, with the digests that
would let it prove the re-derivation, and records that this wave deleted
nothing.  A prune that frees no space on the full disk is not hygiene.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/cells")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    entries = []
    for path in sorted(ROOT.rglob("installed-tensors.pt")):
        receipt = path.with_name("cell-receipt.json")
        wave = None
        cell = None
        if receipt.is_file():
            body = json.loads(receipt.read_text(encoding="utf-8"))
            wave = body.get("campaign_wave")
            cell = body.get("cell_id")
        if wave != 5:
            continue
        entries.append(
            {
                "cell_id": cell,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "receipt_sha": (
                    json.loads(receipt.read_text(encoding="utf-8"))["receipt_sha"]
                    if receipt.is_file()
                    else None
                ),
                "pruned": False,
            }
        )
    free = subprocess.run(
        ["df", "-B1", "--output=source,size,used,avail,pcent,target", "/", "/mnt/ai_data"],
        capture_output=True, text=True, check=False,
    ).stdout
    manifest = {
        "schema": "neural-forge-lfm2-matrix-wave5-pruned-tensors-manifest-v1",
        "wave": 5,
        "pruned_count": 0,
        "candidate_count": len(entries),
        "candidate_bytes": sum(entry["bytes"] for entry in entries),
        "decision": (
            "NOTHING PRUNED. Every wave-5 installed-tensors.pt is on /mnt/ai_data, "
            "which has 1.8 TB free and is explicitly off limits; the filesystem under "
            "pressure is / at 100%, and none of these files is on it. Deleting them "
            "would free zero bytes where the pressure is."
        ),
        "deployment_prune": (
            "NOTHING PRUNED. deployments/ holds exactly two neural-forge-matrix-* trees, "
            "dee1c952 (57 MB) and eb32153e (71 MB, current); keeping the current one and "
            "one previous leaves nothing superseded to delete."
        ),
        "df_at_manifest_time": free,
        "entries": entries,
    }
    out = Path("/home/cereal/nf-mxi/pruned-tensors-manifest.json")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(entries)} wave-5 tensor files digested, 0 pruned -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
