#!/usr/bin/env python3
"""Re-verify every wave-7 child this wave intends to build on, before it builds.

Three separate checks, because they can fail independently:

* the receipt's content address recomputes to the digest it was sealed under;
* the receipt still satisfies today's mandatory field set;
* every tensor in `composed-child-tensors.pt` re-digests to the value
  `composed-child-state-digest.json` sealed for it -- the check that actually
  protects this wave, since the composed tensors are what a wave-8 child
  inherits and a receipt that verifies says nothing about the .pt beside it.

The digest is `run_lfm2_matrix_cell._tensor_sha256` reproduced exactly (dtype
string, then shape repr, then the raw uint8 view).  Reproduced rather than
imported because the runner module pulls a transformers stack this check does
not need; the three lines are pinned by the receipts they have to match.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(
    0, "/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-17377c16/src"
)

import torch  # noqa: E402

from neural_forge.matrix_cell_receipt import (  # noqa: E402
    validate_receipt,
    verify_seal,
)

ROOT = Path("/mnt/ai_data/neural-forge-matrix")


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(repr(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> int:
    failures = 0
    for spec in sys.argv[1:]:
        wave, _, cell = spec.partition(":")
        root = ROOT / wave / "cells" / cell
        receipt = json.loads((root / "cell-receipt.json").read_text(encoding="utf-8"))
        recomputed = verify_seal(receipt)
        sealed = receipt["receipt_sha"]
        match = recomputed == sealed
        failures += 0 if match else 1
        try:
            validate_receipt(receipt)
            mandatory = "OK"
        except Exception as exc:  # noqa: BLE001
            mandatory = f"FAIL {type(exc).__name__}: {exc}"
            failures += 1
        print(f"{cell}  ({wave})")
        print(f"  sealed receipt_sha    {sealed}")
        print(f"  recomputed            {recomputed}  match={match}")
        print(f"  mandatory field set   {mandatory}")

        state = json.loads(
            (root / "composed-child-state-digest.json").read_text(encoding="utf-8")
        )
        sealed_digests: dict[str, str] = {}
        for cell_id, entries in (state.get("installed_tensor_digests") or {}).items():
            for entry in entries:
                key, _, digest = str(entry).partition("=")
                sealed_digests[key] = digest
        print(f"  install order         {state.get('install_order')}")

        tensors = torch.load(root / "composed-child-tensors.pt", weights_only=True)
        print(f"  composed tensors      {len(tensors)}")
        for key in sorted(tensors):
            live = tensor_sha256(tensors[key])
            claim = sealed_digests.get(key)
            if claim is None:
                verdict = "NO SEALED DIGEST"
                failures += 1
            elif claim == live:
                verdict = "matches the seal"
            else:
                verdict = f"MISMATCH sealed={claim[:16]}"
                failures += 1
            shape = str(tuple(tensors[key].shape))
            print(f"    {key:52} {shape:16} {live[:16]}  {verdict}")
        print()
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
