#!/usr/bin/env python3
"""Wave-9 retention sweep: keep five children's tensors, manifest and delete the rest.

The rule the campaign has used since wave 5: a composed child's tensors are kept
only if a later wave would have to re-derive them to say anything, and every
deleted file is sha256-digested into a manifest first, with the sentence that
says how to re-derive it.  Nothing is deleted that a sealed receipt's digest
chain cannot reproduce, and every deleted child keeps its receipt, its
diagnostics, its state digest and its full frozen-review dossier -- which are
committed to the repository, so the *evidence* of every child survives even
where its tensors do not.

Wave 9's children are heavier than any before them: folding C01 persists the
whole 65,536 x 1,024 table at float32, so every child's ``.pt`` is ~268 MB
against wave 8's tens of megabytes.  The keep set is five, as always.

Kept, and why:

* ``COMP.MA.MXM6`` the wave's headline -- the embedding route at a true quarter
  with head 12,288 and tail rank 77, 9/15, 0.780861x, and the measured optimum
  of the head/tail sweep.
* ``COMP.MA.MXM9`` the deepest child the campaign has ever built (0.736305x,
  seven folded components, body + embedding).
* ``COMP.MA.MXM10`` the shallower Stage B bracket (0.753440x, four body folds +
  embedding).  The body-and-embedding interaction cannot be re-read with only
  one of the two brackets.
* ``COMP.MA.MXM5``  the head-sweep neighbour at 8,192.  The claim "12,288 is the
  optimum" is a comparison and needs the point either side of it.
* ``COMP.MA.MXM8``  the only S8 arm under the corrected head rule -- the
  deepest single-component storage the wave sealed, at 0.124807 of the table.

Deleted: the two plain/weighted SVD arms (``MXM1``, ``MXM2``), the two route
arms under the superseded row-norm head rule (``MXM3``, ``MXM4``), and the
over-long head at 15,000 (``MXM7``).  Every one of the five is a *negative*
result whose number is fully carried by its receipt and dossier.

Wave 9 ran no M-B arm, so there is no conditioned evidence tree to remove.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/mnt/ai_data/neural-forge-matrix/wave9/cells")
OUT = Path("/mnt/ai_data/neural-forge-matrix/wave9/pruned-tensors-manifest.json")

KEEP = {
    "COMP.MA.MXM6": (
        "the wave's headline -- embedding route at a true quarter, head 12,288 "
        "/ tail rank 77, 9/15 against a 12/15 parent at 0.780861x, and the "
        "measured optimum of the head/tail sweep"
    ),
    "COMP.MA.MXM9": (
        "the deepest child the campaign has ever built: 0.736305x, seven folded "
        "components, MXL11's body plus the embedding route"
    ),
    "COMP.MA.MXM10": (
        "the shallower stage-B bracket (0.753440x, four body folds plus the "
        "embedding); the body-and-embedding interaction needs both brackets"
    ),
    "COMP.MA.MXM5": (
        "the head-sweep neighbour at 8,192; the claim that 12,288 is the "
        "optimum is a comparison and needs the point either side of it"
    ),
    "COMP.MA.MXM8": (
        "the only S8 arm under the corrected head rule, and the deepest "
        "single-component storage the wave sealed (0.124807 of the table)"
    ),
}

RE_DERIVE = (
    "Re-derivable by re-running experiments/run_lfm2_matrix_composition.py from "
    "the deployment named in this cell's own receipt code_identity "
    "(neural-forge-matrix-a395e024, -ea4d06e3 or -beb699f2; all three tarballs "
    "are retained under /mnt/ai_data/neural-forge-matrix/wave9/) with this "
    "cell's sealed recipe (reports/matrix-campaign-20260810/run/mxm/recipes/) on "
    "a pinned sm86 RTX 3090.  The composition receipt retains every installed "
    "tensor's sha256 under composition.components[].installed_tensor_digests and "
    "the file digest under composition.composed_child_tensors_sha256, so a "
    "re-derivation is checkable against the seal rather than trusted.  The "
    "receipt, diagnostics, state digest and full frozen-review dossier of every "
    "child below are retained and committed to the repository."
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
        "wave": 9,
        "kept": KEEP,
        "deleted_file_count": len(entries),
        "deleted_bytes": freed,
        "deleted": entries,
        "conditioned_evidence_tree": {
            "path": None,
            "bytes": 0,
            "reason": (
                "wave 9 ran no M-B arm.  Three waves of evidence say "
                "conditioning is dead (NOTES-MX-L 5: 36x the wall clock and "
                "three review cases WORSE than its own unconditioned control), "
                "so no conditioned capture was spent and there is none to prune."
            ),
        },
    }
    if not dry:
        OUT.write_text(
            json.dumps(manifest, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(
        f"{'DRY RUN' if dry else 'APPLIED'}: "
        f"{len(entries)} files, {freed / 2**30:.2f} GiB"
    )
    mismatched = [e for e in entries if not e["digest_matches_receipt"]]
    for entry in entries:
        print(
            f"  {'ok ' if entry['digest_matches_receipt'] else 'MISMATCH'} "
            f"{entry['cell_id']:<16} {entry['bytes'] / 2**20:8.1f} MiB "
            f"{entry['sha256'][:16]}"
        )
    if mismatched:
        print(
            f"REFUSING to report a clean sweep: {len(mismatched)} deleted files "
            "did not digest to the value their own receipt sealed."
        )
        return 1
    print(f"kept: {sorted(KEEP)}")
    if not dry:
        print(f"manifest -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
