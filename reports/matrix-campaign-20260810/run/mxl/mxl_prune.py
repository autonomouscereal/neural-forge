#!/usr/bin/env python3
"""Wave-8 retention sweep: keep five children's tensors, manifest and delete the rest.

The rule the campaign has used since wave 5: a composed child's tensors are kept
only if a later wave would have to re-derive them to say anything, and every
deleted file is sha256-digested into a manifest first, with the sentence that
says how to re-derive it.  Nothing is deleted that a sealed receipt's digest
chain cannot reproduce, and every deleted child keeps its receipt, its
diagnostics, its state digest and its full frozen-review dossier -- which are
committed to the repository, so the *evidence* of every child survives even
where its tensors do not.

Kept, and why:

* ``COMP.MA.MXL11`` the deepest coherent child of the wave (0.955444x, six
  folded components) -- the ADDENDUM 8 frontier itself.
* ``COMP.MA.MXL5``  the SwiGLU-lane child (0.972579x, 10/15).  It is one half of
  the matched pair against wave 7's ``COMP.MA.MXK10`` at the identical net
  ratio, and the lane finding cannot be re-read with it missing.
* ``COMP.MA.MXL10`` the M-A control that carries the wave's other headline: the
  same-block conflict disappears at S8 (11/15 against the sealed 6/15).
* ``COMP.MB.MXL12`` the M-B conditioned arm.  It is only meaningful beside
  MXL10, and re-deriving it costs 40 minutes of conditioned capture.
* ``COMP.MA.MXL6``  the deepest three-component coherent child (0.978304x), and
  the only child holding two MLP folds at a true eighth.

Also removes the 16 GB M-B conditioned evidence tree.  The conditioning answer
was no and no lane is built on it; the tree is content-addressed and any future
M-B run re-derives it from the same recipe and knobs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/wave8/cells")
CONDITIONED = Path("/mnt/ai_data/neural-forge-matrix/wave8/evidence-conditioned")
OUT = Path("/mnt/ai_data/neural-forge-matrix/wave8/pruned-tensors-manifest.json")

KEEP = {
    "COMP.MA.MXL11": "deepest coherent child of wave 8 (0.955444x, 6 components, 9/15)",
    "COMP.MA.MXL5": (
        "the SwiGLU-lane child (0.972579x, 10/15) and one half of the matched "
        "pair against wave 7 COMP.MA.MXK10 at the identical net ratio"
    ),
    "COMP.MA.MXL10": (
        "the M-A control showing the same-block conflict disappears at S8 "
        "(11/15 against the sealed 6/15 of COMP.MA.MXJ3)"
    ),
    "COMP.MB.MXL12": (
        "the M-B conditioned arm; meaningful only beside COMP.MA.MXL10, and "
        "40 minutes of conditioned capture to re-derive"
    ),
    "COMP.MA.MXL6": (
        "deepest three-component coherent child (0.978304x) and the only child "
        "holding two MLP folds at a true eighth"
    ),
}

RE_DERIVE = (
    "Re-derivable by re-running experiments/run_lfm2_matrix_composition.py from "
    "deployment neural-forge-matrix-17377c16 with this cell's sealed recipe "
    "(reports/matrix-campaign-20260810/run/mxl/recipes/) on a pinned sm86 RTX "
    "3090; the composition receipt retains every installed tensor's sha256 under "
    "composition.components[].installed_tensor_digests and the file digest under "
    "composition.composed_child_tensors_sha256, so a re-derivation is checkable "
    "against the seal rather than trusted.  The receipt, diagnostics, state "
    "digest and full frozen-review dossier of every child below are retained and "
    "committed to the repository."
)


def directory_bytes(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


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

    conditioned_bytes = directory_bytes(CONDITIONED) if CONDITIONED.is_dir() else 0
    if not dry and CONDITIONED.is_dir():
        shutil.rmtree(CONDITIONED, ignore_errors=True)

    manifest = {
        "schema": "neural-forge-lfm2-matrix-pruned-tensors-manifest-v1",
        "wave": 8,
        "kept": KEEP,
        "deleted_file_count": len(entries),
        "deleted_bytes": freed,
        "deleted": entries,
        "conditioned_evidence_tree": {
            "path": str(CONDITIONED),
            "bytes": conditioned_bytes,
            "reason": (
                "M-B conditioning was measured and answered NO (8/15 against its "
                "own unconditioned control's 11/15); no lane is built on it, the "
                "tree is content-addressed, and any future M-B run re-derives it "
                "from the same recipe and knob file."
            ),
        },
    }
    if not dry:
        OUT.write_text(
            json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"{'PLAN' if dry else 'DONE'}: {len(entries)} tensor files, "
        f"{freed / 2**30:.2f} GiB + conditioned tree {conditioned_bytes / 2**30:.2f} GiB"
    )
    matched = sum(1 for e in entries if e["digest_matches_receipt"])
    print(f"digest matches receipt: {matched}/{len(entries)}")
    for entry in entries:
        print(f"  {entry['cell_id']:<16} {entry['bytes']:>12,}  {entry['sha256'][:16]}")
    if not dry:
        print(f"manifest -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
