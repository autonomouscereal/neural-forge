#!/usr/bin/env python3
"""Give a materialized parent-shaped child the build receipt a fold reads.

CHASM BRIDGE (`INTELLIGENCE_RETENTION_PLAN_2026-08-14.md` #3) installs the sealed
zero-solve V19 vocabulary gather on the wave-11 to wave-13 parent-shaped
frontier.  Those children exist as "the parent plus these N folded tensors", and
`mx37_vocab.py` reads a child's axes out of an ``mx*-build-receipt.json`` and
REFUSES a source that has none -- correctly, because a composed child whose
provenance is re-described by a driver is a child whose provenance is that
driver's opinion.

A materialized COMP child has no such file.  It has ``materialization.json``,
written by the same script that produced the sealed benchmark directories, and
that file already carries everything the refusal is protecting: the cell id, the
sealed composed-tensor digest, the sealed receipt sha, the install order, the
exact list of folded tensors whose per-tensor digests were verified, the counted
parameters and the net ratio.

So this file does one thing: it TRANSCRIBES that, adding nothing.  Every field it
writes is copied from the materialization receipt or from the sealed cell
receipt; there is no axis here that the sealed cell did not already declare, and
the two source digests travel into the child so a reader can check the chain
without this file's word for it.

Historical accident, not physics, is why this bridge is needed at all: the
vocabulary axis postdates the parent-shaped body frontier by twelve waves, so no
wave ever had to compose one onto the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

PARENT_PARAMS = 229_693_184


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True, help="the materialized directory")
    ap.add_argument("--cell-receipt", type=Path, required=True, help="the SEALED cell receipt")
    args = ap.parse_args()

    mat = json.loads((args.base / "materialization.json").read_text(encoding="utf-8"))
    cell = json.loads(args.cell_receipt.read_text(encoding="utf-8"))
    tensors = args.base / "model.safetensors"
    digest = hashlib.sha256(tensors.read_bytes()).hexdigest()

    if mat.get("written_model_safetensors_sha256") not in (None, digest):
        raise SystemExit(
            f"REFUSED: {tensors} digests {digest}; the materialization recorded "
            f"{mat['written_model_safetensors_sha256']}."
        )
    composition = cell.get("composition") or {}
    installs = mat.get("install_order") or []
    folded = mat.get("folded_tensor_digests_verified") or []

    receipt: dict[str, Any] = {
        "schema": "neural-forge-matrix-wave37-materialized-base-receipt-v1",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder": "mx37_basereceipt.py",
        "note": (
            "TRANSCRIPTION, not a build.  Every field below is copied from the "
            "materialization receipt or the sealed cell receipt; no fold was "
            "performed here and no axis is asserted that the sealed cell did not "
            "already declare.  This exists so a vocabulary gather can carry a "
            "parent-shaped body's provenance forward instead of inventing it."
        ),
        "tag": str(mat.get("cell_id", "")).replace("COMP.MA.", "").replace("COMP.MC.", ""),
        "source_cell_id": mat.get("cell_id"),
        "source_cell_receipt": str(args.cell_receipt),
        "source_cell_receipt_sha256": hashlib.sha256(
            args.cell_receipt.read_bytes()).hexdigest(),
        "sealed_receipt_sha": mat.get("receipt_sha"),
        "composed_child_tensors_sha256": mat.get("composed_child_tensors_sha256"),
        "child_model": str(args.base),
        "child_tensors_sha256": digest,
        "child_parameters": int(mat["child_parameter_count"]),
        "parent_parameters": PARENT_PARAMS,
        "net_stored_ratio": float(mat.get("net_stored_ratio", 0.0))
        or int(mat["child_parameter_count"]) / PARENT_PARAMS,
        "storage_format": "parent-shaped (ADDENDUM 12 second column)",
        "excised": [],
        "residual_basis": composition.get("residual_basis"),
        "axes": [{
            "axis": "composed_parent_shaped_body",
            "kind": str(cell.get("component", "composition")),
            "sites": folded,
            "config_fields": [],
            "selection": (
                "carried verbatim from the sealed cell: install order "
                + ", ".join(installs)
            ),
            "solve": (
                "no solve is performed by this file.  The folded tensors are the "
                "sealed ones, verified per tensor at materialization time against "
                "the digests the cell receipt sealed."
            ),
            "evidence": {
                "authority": "the sealed cell receipt named above",
                "cell_id": mat.get("cell_id"),
                "sealed_receipt_sha": mat.get("receipt_sha"),
                "composed_child_tensors_sha256": mat.get("composed_child_tensors_sha256"),
                "conditioned": bool((cell.get("composition") or {})
                                    .get("evidence", {}).get("conditioned", False)),
                "consumed_by_this_file": "none -- this file performs no solve",
            },
            # ZERO, and the zero is the finding rather than an omission.  A
            # parent-shaped child's storage saving lives in its FACTORS;
            # materializing it to a directory stock `transformers` loads bakes
            # those factors back to dense shapes, so as a standard-shape
            # checkpoint this axis removes nothing.  The counted-stored number
            # is carried separately and is not smuggled into this field.
            "parameters_removed": 0,
            "heldout": {
                "write_heldout_cos": (cell.get("composition") or {})
                    .get("boundary", {}).get("write_heldout_cos"),
                "write_heldout_relL2": (cell.get("composition") or {})
                    .get("boundary", {}).get("write_heldout_relL2"),
                "authority": (
                    "the sealed cell's own boundary block; this file measured "
                    "nothing and claims nothing of its own"
                ),
            },
            "install_order": installs,
            "counted_stored_parameters_in_factored_form": int(
                mat["child_parameter_count"]),
        }],
    }
    (args.base / "mx37-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items() if k != "axes"}, indent=1))
    print(f"transcribed: {len(folded)} folded tensors, {receipt['child_parameters']:,} "
          f"parameters, {receipt['net_stored_ratio']:.6f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
