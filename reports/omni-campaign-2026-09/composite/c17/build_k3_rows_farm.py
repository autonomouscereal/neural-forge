#!/usr/bin/env python3
"""Lane E3: one --k3-resid directory for c17 over K3 MoE layers 1..60 (symlinks only, read-only sources).

c17_chart_fit.py takes a single --k3-resid DIR/L{l:02d}/held_rows.safetensors, while the K3
prefix L01-60 is spread over three roots (Lexar recovery 048cff57 for L01-36, NAS window
37-48, NAS window 49-60).  For every layer this links the candidate whose sha256 equals the
held_rows sha256 recorded in the sealed window-49-60 pass receipt (which carries layers
0..60 forward), so the farm is pinned to the receipt c17 is given as --k3-pass-receipt.
No file is copied; a layer with no matching candidate is a refusal (exit 2, manifest still
written with the mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOTS = {
    "lexar-048cff57": "/mnt/lexar/omni-campaign/k3-diagnostic-recovery-oct07-048cff57-935c-4e2b-a31b-96a3c7fa6318/fold/resid",
    "nas-w37-48": "/mnt/ai_data/omni-campaign/k3-nas-window37-48-oct07-764524b1-96a9-4636-be74-942f9c54754c/model/fold/resid",
    "nas-w49-60": "/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139/model/fold/resid",
    "lexar-oct06": "/mnt/lexar/omni-campaign/k3-continuation-oct06-bde8b657-90d4-41e8-83b3-7908616aeaf1/fold/resid",
}
RECEIPT = "/mnt/ai_data/omni-campaign/k3-nas-window49-60-oct07-03603908-570d-4c90-96cb-ee3a63a9a139/model/receipts/k3_dense_pass_receipt.json"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def preferred_order(layer: int) -> list[str]:
    if layer <= 36:
        return ["lexar-048cff57", "lexar-oct06"]
    if layer <= 48:
        return ["nas-w37-48"]
    return ["nas-w49-60"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--receipt", default=RECEIPT)
    ap.add_argument("--layers", default="1-60")
    args = ap.parse_args()
    a, b = (int(x) for x in args.layers.split("-"))
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    bad = []
    for layer in range(a, b + 1):
        want = receipt["layers"][str(layer)]["held_rows"]
        chosen = None
        tried = []
        for name in preferred_order(layer):
            path = os.path.join(ROOTS[name], f"L{layer:02d}", "held_rows.safetensors")
            if not os.path.isfile(path):
                tried.append({"root": name, "path": path, "exists": False})
                continue
            t0 = time.time()
            digest = sha256(path)
            entry = {"root": name, "path": path, "exists": True, "sha256": digest, "bytes": os.path.getsize(path),
                     "matches_receipt": digest == want["sha256"], "hash_seconds": round(time.time() - t0, 1)}
            tried.append(entry)
            if entry["matches_receipt"]:
                chosen = entry
                break
        row = {"layer": layer, "receipt_held_rows": want, "chosen": chosen, "tried": tried}
        rows.append(row)
        if chosen is None:
            bad.append(layer)
            print(f"L{layer:02d}: NO candidate matches receipt sha {want['sha256'][:16]}", flush=True)
            continue
        d = out / f"L{layer:02d}"
        d.mkdir(exist_ok=True)
        link = d / "held_rows.safetensors"
        if link.is_symlink() or link.exists():
            if os.path.realpath(link) != os.path.realpath(chosen["path"]):
                bad.append(layer)
                print(f"L{layer:02d}: existing link points elsewhere: {os.path.realpath(link)}", flush=True)
                continue
        else:
            os.symlink(chosen["path"], link)
        print(f"L{layer:02d}: {chosen['root']} sha {chosen['sha256'][:16]} == receipt ({chosen['hash_seconds']}s)", flush=True)
    receipt_sha = sha256(args.receipt)
    manifest = {
        "schema": "omni-campaign.e3.k3-rows-farm.v1",
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_receipt": {"path": args.receipt, "sha256": receipt_sha},
        "rule": "link the candidate whose sha256 equals receipt.layers[l].held_rows.sha256; symlinks only",
        "layers": rows,
        "refused_layers": bad,
    }
    (out / "farm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"farm {out}: {len(rows) - len(bad)} linked, refused {bad}; receipt sha {receipt_sha}", flush=True)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
