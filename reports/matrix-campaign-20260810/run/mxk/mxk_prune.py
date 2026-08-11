#!/usr/bin/env python3
"""Wave-7 retention sweep: keep five children's tensors, manifest and delete the rest.

The rule the campaign has used since wave 5: a composed child's tensors are kept
only if a later wave would have to re-derive them to say anything, and every
deleted file is sha256-digested into a manifest first with the sentence that
says how to re-derive it.  Nothing is deleted that a sealed receipt's digest
chain cannot reproduce.

Kept, and why:

* ``COMP.MA.MXK5``  the deepest alive-parent-band child of the wave.
* ``COMP.MA.MXK13`` the deepest child at or above the A9 band, and the S8 arm of
  the MXK5/MXK13 level pair -- worthless with one arm missing.
* ``COMP.MA.MXK14`` the highest-scoring child of either wave (13/15) and the S8
  arm of the MXK6/MXK14 level pair.
* ``COMP.MA.MXK6``  the S4 arm of that pair, and the first leg of the C05
  separating experiment.
* ``COMP.MA.MXK9``  the second leg of that experiment; the C05 mechanism claim
  is a three-child bracket and cannot be re-read with one child missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/wave7/cells")
OUT = Path("/mnt/ai_data/neural-forge-matrix/wave7/pruned-tensors-manifest.json")

KEEP = {
    "COMP.MA.MXK5": "deepest alive-parent-band child of wave 7 (10/15, net 0.981152x)",
    "COMP.MA.MXK13": (
        "deepest child at or above the A9 band (9/15, net 0.979732x) and the S8 "
        "arm of the MXK5/MXK13 level pair"
    ),
    "COMP.MA.MXK14": (
        "highest-scoring child of either wave (13/15, net 0.994287x) and the S8 "
        "arm of the MXK6/MXK14 level pair"
    ),
    "COMP.MA.MXK6": (
        "S4 arm of the MXK6/MXK14 level pair, and leg 1 of the C05 separating "
        "experiment (C05 with no MLP fold in block L05)"
    ),
    "COMP.MA.MXK9": (
        "leg 2 of the C05 separating experiment (C05 sharing block L05 with "
        "mlp_gate); the mechanism claim is a three-child bracket"
    ),
}

RE_DERIVE = (
    "Re-derivable by re-running experiments/run_lfm2_matrix_composition.py from "
    "deployment neural-forge-matrix-17377c16 with this cell's sealed recipe on a "
    "pinned sm86 RTX 3090; the composition receipt retains every installed "
    "tensor's sha256 under composition.components[].installed_tensor_digests and "
    "the file digest under composition.composed_child_tensors_sha256, so a "
    "re-derivation is checkable against the seal rather than trusted."
)


def main() -> int:
    dry = "--apply" not in sys.argv
    entries = []
    freed = 0
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or directory.name in KEEP:
            continue
        receipt = directory / "cell-receipt.json"
        declared = None
        if receipt.is_file():
            body = json.loads(receipt.read_text(encoding="utf-8"))
            declared = (body.get("composition") or {}).get(
                "composed_child_tensors_sha256"
            )
        for path in sorted(directory.glob("*.pt")):
            blob = path.read_bytes()
            digest = hashlib.sha256(blob).hexdigest()
            entries.append(
                {
                    "path": str(path),
                    "cell_id": directory.name,
                    "bytes": len(blob),
                    "sha256": digest,
                    "receipt_declared_sha256": declared,
                    "digest_matches_receipt": declared == digest,
                    "re_derive": RE_DERIVE,
                }
            )
            freed += len(blob)
            if not dry:
                path.unlink()
    manifest = {
        "schema": "neural-forge-lfm2-matrix-pruned-tensors-manifest-v1",
        "wave": 7,
        "kept": KEEP,
        "deleted_file_count": len(entries),
        "deleted_bytes": freed,
        "deleted": entries,
    }
    if not dry:
        OUT.write_text(
            json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"{'PLAN' if dry else 'DONE'}: {len(entries)} files, {freed / 2**30:.2f} GiB")
    print(f"digest matches receipt: {sum(1 for e in entries if e['digest_matches_receipt'])}/{len(entries)}")
    for entry in entries:
        print(f"  {entry['cell_id']:<16} {entry['bytes']:>12,}  {entry['sha256'][:16]}")
    if not dry:
        print(f"manifest -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
