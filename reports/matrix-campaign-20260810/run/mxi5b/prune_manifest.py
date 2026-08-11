#!/usr/bin/env python3
"""Digest every file wave 5B is authorised to delete, before it deletes any.

Cyril's mid-wave directive relaxes the preserve-everything rule for HEAVY
artifacts only: the paper trail stays complete (ledger, receipts, diagnostics,
review dossiers, logs, notes), the primary sealed evidence cache
``2dec87ac...`` stays, the current deployment stays, and the tensor artifacts of
the best folds stay.  What goes is bulk that is re-derivable and whose digests
are already sealed elsewhere.

This script writes the manifest.  It does not delete anything -- deletion is a
separate, explicit step that reads this manifest back, so a prune can never run
ahead of its own record.

Two groups:

``conditioned-evidence``
    ``/mnt/ai_data/neural-forge-matrix/evidence-conditioned`` -- wave 4's M-B
    conditioned captures, re-derivable by replaying the partially folded child
    against the sealed split manifest.  ADDENDUM 4's M-B mode is defined by that
    replay, so the capture is a cache of a defined function, not a primary.

``installed-tensors``
    ``installed-tensors.pt`` beside a cell receipt, for cells that are NOT in
    the keep list.  Wave 5A's reproduction verifier established that these
    solves reproduce bit-exact across commits, deployments and code paths, and
    every one of these files already has its per-key digests sealed under
    ``solve.installed_tensor_digests`` in its own receipt.

Usage::

    python prune_manifest.py --out manifest.json --keep C13.T8.S4.L05 ...
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CELLS = Path("/mnt/ai_data/neural-forge-matrix/cells")
CONDITIONED = Path("/mnt/ai_data/neural-forge-matrix/evidence-conditioned")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep", nargs="*", default=[])
    arguments = parser.parse_args()
    keep = set(arguments.keep)

    entries = []
    for path in sorted(CONDITIONED.rglob("*")):
        if not path.is_file():
            continue
        entries.append(
            {
                "group": "conditioned-evidence",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "rederivable_by": (
                    "ADDENDUM 4 M-B replay of the partially folded child against "
                    "the sealed evidence split manifest"
                ),
            }
        )

    for path in sorted(CELLS.rglob("installed-tensors.pt")):
        receipt_path = path.with_name("cell-receipt.json")
        cell = None
        receipt_sha = None
        digests = None
        if receipt_path.is_file():
            body = json.loads(receipt_path.read_text(encoding="utf-8"))
            cell = body.get("cell_id")
            receipt_sha = body.get("receipt_sha")
            digests = body.get("solve", {}).get("installed_tensor_digests")
        if cell in keep:
            continue
        entries.append(
            {
                "group": "installed-tensors",
                "cell_id": cell,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "receipt_sha": receipt_sha,
                "sealed_per_key_digests": digests,
                "rederivable_by": (
                    "re-running the cell; wave 5A's mxi_verify_reproduction.py "
                    "proved these solves reproduce bit-exact across commits, "
                    "deployments and code paths"
                ),
            }
        )

    free = subprocess.run(
        ["df", "-B1", "--output=source,size,used,avail,pcent,target", "/", "/mnt/ai_data"],
        capture_output=True, text=True, check=False,
    ).stdout

    manifest = {
        "schema": "neural-forge-lfm2-matrix-wave5-pruned-tensors-manifest-v2",
        "wave": 5,
        "authority": (
            "Cyril, mid-wave directive relayed 2026-08-11: keep the complete paper "
            "trail, the primary sealed evidence cache, the current deployment and the "
            "tensor artifacts of the best folds; delete the wave-4 M-B conditioned "
            "captures, installed tensors beyond the keep list, superseded deployments "
            "beyond current+1, and scratch."
        ),
        "kept_cells": sorted(keep),
        "not_touched": [
            "/mnt/ai_data/neural-forge-stage2 (stage-2 estate, preserved)",
            "/mnt/ai_data/neural-forge-matrix/backup (ledger backups and stale-manifest backups)",
            "artifacts/neural_forge_matrix_campaign_20260810/evidence/2dec87ac... (primary sealed evidence)",
            "deployments/neural-forge-matrix-eb32153e (current) and -dee1c952 (one previous)",
            "every cell-receipt.json, cell-diagnostics.json, frozen-review-dossier.json, "
            "cell-receipt-behavior-reviewed.json, ledger and log (the paper trail is complete)",
            "everything belonging to the parallel Codex attention-geometry lane",
        ],
        "entry_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "df_at_manifest_time": free,
        "deleted": False,
        "entries": entries,
    }
    arguments.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "%d files, %.2f GB manifested -> %s"
        % (len(entries), manifest["total_bytes"] / 1e9, arguments.out)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
