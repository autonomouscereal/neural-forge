#!/usr/bin/env python3
"""Lane E3 (2026-10-10): extend the c17 K3 rows farm from L01-60 to L01-92 under the full K3 pass receipt.

The K3 dense fold run k3-run-oct09-claude finished all 92 MoE layers; its sealed receipt (layers
0..92, the window-49-60 prefix carried forward) is the receipt c17 is now given as
--k3-pass-receipt.  build_k3_rows_farm.py pinned L01-60 to the window-49-60 receipt; this script
re-pins the farm to the new receipt without relinking the prefix.  Symlinks only, two phases:

1. prefix check (L01-60, never relinked): every existing link must still resolve to the path the
   previous farm_manifest.json chose, and the target's sha256 (hashed NOW) must equal the NEW
   receipt's layers[l].held_rows.sha256 and the previous manifest's recorded sha256; the two
   receipts' corpus_sha256 and evidence input_ids_sha256 must be equal.  Any mismatch is a refusal:
   exit 2 before a single new link is made, farm_manifest.json untouched (the refusal is written
   to farm_manifest_extend_refused.json).
2. link (L61-92): DIR/L{l:02d}/held_rows.safetensors -> <run>/fold/resid/L{l:02d}/held_rows.safetensors
   when its sha256 equals the new receipt's held_rows sha256.

Every file (both phases) must also carry input_ids_sha256 metadata equal to the new receipt's
corpus_sha256 (the check c17 runs at fit time, done here from the safetensors header).  The
previous manifest is kept as farm_manifest_L01-60.json; the new farm_manifest.json is pinned to
the new receipt and keeps the v1 row shape (layer / receipt_held_rows / chosen / tried).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

RUN_ROOT = "/home/cereal/omni-campaign/k3-run-oct09-claude"
RECEIPT = RUN_ROOT + "/receipts/k3_dense_pass_receipt.json"
NEW_ROOT_NAME = "k3-run-oct09-claude"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def header_metadata(path: str) -> dict:
    """safetensors ``__metadata__`` from the header (8-byte little-endian length + JSON)."""

    with open(path, "rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length))
    return header.get("__metadata__") or {}


def hash_entry(root: str, path: str, want: dict, corpus_sha256: str) -> dict:
    if not os.path.isfile(path):
        return {"root": root, "path": path, "exists": False}
    t0 = time.time()
    digest = sha256(path)
    stamped = header_metadata(path).get("input_ids_sha256")
    return {"root": root, "path": path, "exists": True, "sha256": digest, "bytes": os.path.getsize(path),
            "matches_receipt": digest == want["sha256"], "input_ids_sha256": stamped,
            "input_ids_matches_corpus": stamped == corpus_sha256, "hash_seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--farm", required=True, help="the c17 --k3-resid farm dir holding farm_manifest.json")
    ap.add_argument("--receipt", default=RECEIPT)
    ap.add_argument("--new-root", default=RUN_ROOT + "/fold/resid")
    ap.add_argument("--prefix", default="1-60")
    ap.add_argument("--link", default="61-92")
    args = ap.parse_args()
    farm = Path(args.farm)
    old_manifest_path = farm / "farm_manifest.json"
    old_manifest_raw = old_manifest_path.read_bytes()
    old_manifest = json.loads(old_manifest_raw)
    old_rows = {int(r["layer"]): r for r in old_manifest["layers"]}
    old_receipt_path = old_manifest["pinned_receipt"]["path"]
    receipt = json.loads(Path(args.receipt).read_text(encoding="utf-8"))
    old_receipt = json.loads(Path(old_receipt_path).read_text(encoding="utf-8"))
    corpus_sha256 = receipt["corpus"]["corpus_sha256"]
    bad: list[str] = []

    corpus = {
        "new_corpus_sha256": corpus_sha256, "old_corpus_sha256": old_receipt["corpus"]["corpus_sha256"],
        "new_evidence_input_ids_sha256": receipt["corpus"]["segments"][0]["input_ids_sha256"],
        "old_evidence_input_ids_sha256": old_receipt["corpus"]["segments"][0]["input_ids_sha256"],
    }
    corpus["match"] = (corpus["new_corpus_sha256"] == corpus["old_corpus_sha256"]
                       and corpus["new_evidence_input_ids_sha256"] == corpus["old_evidence_input_ids_sha256"])
    print(f"corpus: new {corpus_sha256[:16]} old {corpus['old_corpus_sha256'][:16]} evidence ids "
          f"{corpus['new_evidence_input_ids_sha256'][:16]} / {corpus['old_evidence_input_ids_sha256'][:16]} "
          f"match {corpus['match']}", flush=True)
    if not corpus["match"]:
        bad.append("corpus")

    # phase 1: the prefix, verified in place
    a, b = (int(x) for x in args.prefix.split("-"))
    prefix_rows = []
    for layer in range(a, b + 1):
        want = receipt["layers"][str(layer)]["held_rows"]
        old_want = old_receipt["layers"][str(layer)]["held_rows"]
        old_row = old_rows.get(layer) or {}
        old_chosen = old_row.get("chosen") or {}
        link = farm / f"L{layer:02d}" / "held_rows.safetensors"
        real = os.path.realpath(link)
        entry = hash_entry(old_chosen.get("root", "?"), real, want, corpus_sha256)
        checks = {
            "is_symlink": link.is_symlink(),
            "realpath_is_previous_choice": real == os.path.realpath(old_chosen.get("path", "")),
            "sha256_matches_new_receipt": entry.get("matches_receipt", False),
            "sha256_matches_previous_manifest": entry.get("sha256") == old_chosen.get("sha256"),
            "new_receipt_entry_equals_old_receipt_entry": want == old_want,
            "input_ids_matches_corpus": entry.get("input_ids_matches_corpus", False),
        }
        ok = all(checks.values())
        prefix_rows.append({"layer": layer, "receipt_held_rows": want, "chosen": entry if ok else None,
                            "tried": [entry], "prefix_check": checks, "carried_from_previous_manifest": True})
        if not ok:
            bad.append(f"L{layer:02d}")
        print(f"L{layer:02d}: prefix {'OK' if ok else 'MISMATCH'} sha {str(entry.get('sha256'))[:16]} "
              f"receipt {want['sha256'][:16]} {'' if ok else checks} ({entry.get('hash_seconds')}s)", flush=True)

    receipt_sha = sha256(args.receipt)
    manifest = {
        "schema": "omni-campaign.e3.k3-rows-farm.v1",
        "made_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pinned_receipt": {"path": args.receipt, "sha256": receipt_sha},
        "rule": "link the candidate whose sha256 equals receipt.layers[l].held_rows.sha256; symlinks only",
        "extension": {
            "made_by": "extend_k3_rows_farm.py (lane E3, 2026-10-10)",
            "previous_manifest": {"path": str(farm / "farm_manifest_L01-60.json"),
                                  "sha256": hashlib.sha256(old_manifest_raw).hexdigest(),
                                  "pinned_receipt": old_manifest["pinned_receipt"]},
            "prefix_layers": args.prefix, "linked_layers": args.link, "new_root": args.new_root,
            "corpus_identity": corpus,
            "prefix_rule": "existing links kept; target hashed now == new receipt held_rows sha256 == previous manifest "
                           "sha256, realpath unchanged, input_ids_sha256 metadata == new receipt corpus_sha256",
        },
        "layers": prefix_rows,
        "refused_layers": [],
    }
    if bad:
        manifest["refused_layers"] = bad
        (farm / "farm_manifest_extend_refused.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"REFUSE: prefix / corpus mismatch {bad}; no link made, farm_manifest.json untouched "
              f"(details: {farm / 'farm_manifest_extend_refused.json'})", flush=True)
        return 2

    # phase 2: link the new layers
    a, b = (int(x) for x in args.link.split("-"))
    for layer in range(a, b + 1):
        want = receipt["layers"][str(layer)]["held_rows"]
        path = os.path.join(args.new_root, f"L{layer:02d}", "held_rows.safetensors")
        entry = hash_entry(NEW_ROOT_NAME, path, want, corpus_sha256)
        ok = entry.get("matches_receipt", False) and entry.get("input_ids_matches_corpus", False)
        row = {"layer": layer, "receipt_held_rows": want, "chosen": entry if ok else None, "tried": [entry]}
        manifest["layers"].append(row)
        if not ok:
            bad.append(f"L{layer:02d}")
            print(f"L{layer:02d}: NO match: sha {str(entry.get('sha256'))[:16]} receipt {want['sha256'][:16]} "
                  f"input_ids {str(entry.get('input_ids_sha256'))[:16]}", flush=True)
            continue
        d = farm / f"L{layer:02d}"
        d.mkdir(exist_ok=True)
        link = d / "held_rows.safetensors"
        if link.is_symlink() or link.exists():
            if os.path.realpath(link) != os.path.realpath(path):
                bad.append(f"L{layer:02d}")
                row["chosen"] = None
                print(f"L{layer:02d}: existing link points elsewhere: {os.path.realpath(link)}", flush=True)
                continue
        else:
            os.symlink(path, link)
        print(f"L{layer:02d}: {NEW_ROOT_NAME} sha {entry['sha256'][:16]} == receipt ({entry['hash_seconds']}s)", flush=True)

    manifest["refused_layers"] = bad
    backup = farm / "farm_manifest_L01-60.json"
    if backup.exists():
        if backup.read_bytes() != old_manifest_raw:
            print(f"REFUSE: {backup} exists with other content; farm_manifest.json not rewritten", flush=True)
            return 3
    else:
        shutil.copy2(old_manifest_path, backup)
    (farm / "farm_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    linked = sum(1 for r in manifest["layers"] if r["chosen"] is not None)
    print(f"farm {farm}: {linked} layers verified/linked, refused {bad}; previous manifest kept as {backup}; "
          f"receipt sha {receipt_sha}", flush=True)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
