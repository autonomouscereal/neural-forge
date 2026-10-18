#!/usr/bin/env python3
"""omni1_lineage.py -- the --lineage-sources file for the OMNI1 install (c16_install_media.py >= 1.4.0).

OMNI1 assembly lane, Claude, 2026-10-13.  OMNI1 = the S5 union child (OC02-F-Q38-27BU-C2K4352K3U) plus the
five S5-bound media units.  The S5 child already carries two sources; this script states them with the
header-derived numbers of their own receipts (nothing typed from memory) and names the five media sources
by the labels c16 uses for them, so c16 can write ONE all-source table into composite_manifest.json:

  1. Qwen3.8-27B-Uncensored  -- the text trunk (S3 C2K4352 fold, 64 qwen.mlp components at 0.25), read from
                                S5 register_counts.json (s5_register_counts.py, from the parent / child headers);
  2. Kimi K3                 -- the C10 union rows (64 x 3 x 5120 x 12288 dense K3 L1-92 lifted through the c17
                                charts), passed through as added rows, from the same receipt + union_receipt.json;
  3-7. media_aliases         -- ACE-Step, MiniMax H3, Qwen-Image, Hunyuan3D 2.1, Qwen3-VL-32B (c16 derives their
                                accounting from the written headers).

    python omni1_lineage.py --oc /home/cereal/omni-campaign --out <file.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt(path: Path, what: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing receipt {path}")
    return {"path": str(path), "sha256": sha256_file(path), "what": what}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oc", default="/home/cereal/omni-campaign")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    oc = Path(args.oc)
    s5 = oc / "composite/c10/S5-C2K4352K3U"
    counts_path = s5 / "register_counts.json"
    c = json.loads(counts_path.read_text(encoding="utf-8"))
    layers = c["layers"]
    if len(layers) != 64 or c["accounting_identity_residual"] != 0:
        raise SystemExit("S5 register_counts.json is not the 64-layer, identity-0 receipt")
    q_parent = sum(int(l["parent_mlp_header_params"]) for l in layers)
    q_child = sum(int(l["qwen_rows_params"]) for l in layers)
    k3_rows = sum(int(l["k3_rows_params"]) for l in layers)
    if q_parent - q_child != int(c["qwen_mlp_saving"]) or k3_rows != int(c["k3_rows_added"]):
        raise SystemExit("per-layer sums disagree with the receipt totals")
    registry = oc / "registry"
    qwen = {
        "source": "Qwen3.8-27B-Uncensored",
        "role": "text trunk and accounting parent (PARENT-Q38-27BU); folded at S3 (OC02-F-Q38-27BU-C2K4352): 4,352 Qwen "
                "FFN channels kept per layer, attention / GDN mixers, embeddings and head passed through; carried "
                "unchanged by S5 and S7",
        "accounting": {
            "parent_header_parameters": int(c["parent"]["header_parameters"]),
            "components": {"count": len(layers), "form": "qwen.mlp.layerNN:PARENT:CHILD (asserted, derived from both headers)",
                           "parent_params_each": sorted({int(l["parent_mlp_header_params"]) for l in layers}),
                           "child_params_each": sorted({int(l["qwen_rows_params"]) for l in layers}),
                           "parent_params_total": q_parent, "child_params_total": q_child,
                           "ratio": q_child / q_parent, "meets_4x": all(l["qwen_ratio"] <= 0.25 for l in layers)},
            "qwen_mlp_saving": int(c["qwen_mlp_saving"]),
            "mtp_dropped_not_folded": int(c["parent"]["mtp_header_parameters"]),
            "non_ffn_parameters_in_child": int(c["child_non_ffn_parameters"]),
            "parent_non_mlp_non_mtp_parameters": int(c["parent_non_mlp_non_mtp_parameters"]),
            "native_export_extra_parameters": int(c["native_export_extra_parameters"]),
            "passed_through": ["gdn_mixer", "attention_mixer", "embed / lm_head (ledger bands)", "native interface"],
            "counts_source": f"S5 register_counts.json (s5_register_counts.py, oc_seal header reader) sha {sha256_file(counts_path)}",
        },
        "receipts": [
            receipt(counts_path, "S5 header-derived counts (per-layer Qwen rows, K3 rows, native-export difference, identity 0)"),
            receipt(registry / "PARENT-Q38-27BU" / "PARENT-Q38-27BU-0001-register.json", "parent record"),
            receipt(registry / "OC02-F-Q38-27BU-C2K4352" / "OC02-F-Q38-27BU-C2K4352-0001-register.json", "S3 trunk register record"),
            receipt(registry / "OC02-F-Q38-27BU-C2K4352K3U" / "OC02-F-Q38-27BU-C2K4352K3U-0001-register.json", "S5 register record"),
        ],
    }
    k3 = {
        "source": "Kimi K3",
        "role": "C10 union (S5, OC02-F-Q38-27BU-C2K4352K3U): dense K3 L1-92 rows lifted through the c17 ridge charts into "
                "blocks.k.ffn, design a003 = trunk/additive:0.03; ADDED rows, carried as --passed-through (not a component)",
        "accounting": {
            "added_parameters": k3_rows,
            "layers": len(layers),
            "rows_per_layer": int(c["child"]["k3_channels"]),
            "form": "64 x 3 x 5120 x 12288 rows inside the fused union FFN (blocks.k.ffn.up / down), split at "
                    "c10_trunk_intermediate_size 4352",
            "union_width": int(c["child"]["union_width"]),
            "counts_source": f"S5 register_counts.json sha {sha256_file(counts_path)}",
        },
        "receipts": [
            receipt(s5 / "child" / "union_receipt.json", "c10_union_install 1.5.1 union receipt (S5 child)"),
            receipt(oc / "k3-run-oct09-claude" / "receipts" / "k3_dense_pass_receipt.json", "K3 dense pass L1-92"),
            receipt(oc / "composite/c17/charts/receipt.json", "c17 ridge charts L00-L63, merged receipt"),
            receipt(oc / "composite/c17/charts/alignment.json", "92 MoE layers -> 64 trunk layers alignment"),
        ],
    }
    body = {
        "schema": "omni-campaign.composite_lineage_sources.v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": {"name": "omni1_lineage", "sha256": sha256_file(Path(__file__).resolve())},
        "sources": [qwen, k3],
        "media_aliases": {
            "ACE-Step": ["ace", "ACE", "ACE-LM"],
            "MiniMax H3": ["h3", "H3"],
            "Qwen-Image": ["qimg", "QIMG"],
            "Hunyuan3D 2.1": ["hy3d", "HY3D"],
            "Qwen3-VL-32B": ["qvl32", "QVL32"],
        },
    }
    out = Path(args.out)
    out.write_text(json.dumps(body, indent=1) + "\n", encoding="utf-8")
    print(f"qwen components 64: parent {q_parent:,} child {q_child:,}; k3 rows {k3_rows:,}")
    print(f"lineage -> {out} sha256 {sha256_file(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
