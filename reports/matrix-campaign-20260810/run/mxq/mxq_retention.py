#!/usr/bin/env python3
"""Wave-12 retention manifest: what is kept, what was deleted, and why.

ADDENDUM 8's retention rule is "best-5, manifest + delete".  Wave 12 sealed two
reviewed children and one `infra` row, so there is nothing above five and nothing
to prune; the manifest says so with digests rather than leaving the absence of a
file to be inferred.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/wave13/cells")
OUT = Path("/mnt/ai_data/neural-forge-matrix/wave13/pruned-tensors-manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    retained = []
    for path in sorted(ROOT.rglob("composed-child-*.pt")):
        if path.is_symlink():
            continue
        retained.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "neural-forge-matrix-wave13-retention-manifest-v1",
        "policy": (
            "Keep the composed tensor set of every child that was REVIEWED, "
            "because the frozen review is taken on those exact bytes and a "
            "dossier whose child cannot be reconstituted is not evidence. Wave "
            "12 sealed two reviewed children and one infra row; nothing was "
            "deleted, because the infra row died before its first tensor save "
            "and has no tensor set to delete."
        ),
        "retained": retained,
        "retained_count": len(retained),
        "deleted": [],
        "deleted_count": 0,
        "re_derivation": (
            "Re-run experiments/run_lfm2_matrix_composition.py from the "
            "deployment each cell receipt names in code_identity "
            "[neural-forge-matrix-63e95cd8 for COMP.MA.MXP2, -f753b9e7 for "
            "COMP.MA.MXP1; both tarballs retained under "
            "/mnt/ai_data/neural-forge-matrix/wave13/] with that cell's sealed "
            "recipe from reports/matrix-campaign-20260810/run/mxp/recipes/ on a "
            "PCI-pinned sm86 RTX 3090. Every installed tensor's sha256 is under "
            "composition.components[].installed_tensor_digests, the folded file "
            "digest under composition.composed_child_tensors_sha256, and -- for "
            "a child carrying ADDENDUM 10 routes -- every route tensor's digest "
            "under composition.corrections.routes[].route_tensor_sha256 plus the "
            "file digest under composition.corrections.corrections_tensors_"
            "sha256. A re-derivation is checkable against the seal rather than "
            "trusted."
        ),
        "materialized_hf_children_deleted_after_benchmark": [
            "/mnt/ai_data/neural-forge-matrix/wave13/hf-children/"
            "COMP.MA.MXP2/model.safetensors"
        ],
        "conditioned_captures": (
            "Five conditioned captures (one per corrective route plus the base) "
            "ran in /dev/shm, were deleted as their rows were read, and the "
            "tmpfs was cleared at wave end. Each capture's cache key, split "
            "manifest digest, collector implementation digest and child "
            "state_dict digest are sealed in "
            "composition.corrections.captures[]."
        ),
    }
    OUT.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    for row in retained:
        print(f"{row['bytes']:>12,}  {row['sha256'][:16]}  {row['path']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
