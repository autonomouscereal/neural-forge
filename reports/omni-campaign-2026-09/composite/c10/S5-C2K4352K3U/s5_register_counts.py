#!/usr/bin/env python3
"""S5 registration counts, read from safetensors headers (2026-10-11, Claude, S5 build lane).

The ledger can derive a --component only from ONE tensor-name prefix that both directories share.  The
accounting parent (HF Qwen3.8-27B-Uncensored, ``model.language_model.layers.k.mlp.*``) and the S5 child
(Chimera native, ``blocks.k.ffn.*``) do not share names, and inside the child one fused tensor carries the
Qwen rows AND the K3 rows, so a prefix can only name the whole union FFN (ratio ~0.956, not a fold).  This
script therefore derives every number the register line types from the headers itself, splits the union
at the trunk width the child's own config records (``c10_trunk_intermediate_size``), checks the split
against the tensor shapes, and writes a receipt the record's notes cite.  Nothing here is a fit number.

    python s5_register_counts.py --parent <HF parent dir> --child <S5 child dir> --trunk <S3 native dir>
                                 --out <receipt.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

LEDGER = "/home/cereal/omni-campaign/ledger"
sys.path.insert(0, LEDGER)
import oc_seal  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def headers(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for shard in oc_seal.checkpoint_shards(root):
        for name, spec in oc_seal.safetensors_header(shard).items():
            if name != "__metadata__":
                out[name] = spec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--child", required=True)
    ap.add_argument("--trunk", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parent, child, trunk = Path(args.parent), Path(args.child), Path(args.trunk)
    pc = oc_seal.safetensors_tensor_counts(parent)
    cc = oc_seal.safetensors_tensor_counts(child)
    tc = oc_seal.safetensors_tensor_counts(trunk)
    ch = headers(child)
    cfg = json.loads((child / "config.json").read_text(encoding="utf-8"))
    conf = cfg.get("config", cfg)
    meta = conf.get("metadata", {})
    q = int(meta["c10_trunk_intermediate_size"])
    k3 = int(meta["c10_k3_channels"])
    hidden = int(conf["hidden_size"])
    n_layers = int(conf["num_layers"])
    width = q + k3
    layers = []
    for k in range(n_layers):
        p_mlp = sum(v for n, v in pc.items() if n.startswith(f"model.language_model.layers.{k}.mlp."))
        c_ffn = sum(v for n, v in cc.items() if n.startswith(f"blocks.{k}.ffn."))
        up = [int(x) for x in ch[f"blocks.{k}.ffn.up.weight"]["shape"]]
        down = [int(x) for x in ch[f"blocks.{k}.ffn.down.weight"]["shape"]]
        if up != [2 * width, hidden] or down != [hidden, width]:
            raise SystemExit(f"layer {k}: child FFN shapes {up}/{down} != [{2 * width},{hidden}]/[{hidden},{width}]")
        if c_ffn != 3 * hidden * width:
            raise SystemExit(f"layer {k}: child FFN headers {c_ffn} != 3 x {hidden} x {width}")
        qwen = 3 * hidden * q
        k3_rows = 3 * hidden * k3
        if p_mlp <= 0:
            raise SystemExit(f"layer {k}: no parent MLP tensors")
        layers.append({"layer": k, "parent_mlp_header_params": p_mlp, "child_ffn_header_params": c_ffn,
                       "qwen_rows_params": qwen, "k3_rows_params": k3_rows, "qwen_ratio": qwen / p_mlp,
                       "union_ratio_vs_parent_mlp": c_ffn / p_mlp})
    parent_total, child_total, trunk_total = sum(pc.values()), sum(cc.values()), sum(tc.values())
    mtp = sum(v for n, v in pc.items() if n.startswith("mtp."))
    qwen_saving = sum(l["parent_mlp_header_params"] - l["qwen_rows_params"] for l in layers)
    k3_added = sum(l["k3_rows_params"] for l in layers)
    # the S3 native trunk vs the parent outside the MLPs: what the native export added (ledger band rows etc.)
    trunk_non_ffn = trunk_total - sum(v for n, v in tc.items() if ".ffn." in n)
    child_non_ffn = child_total - sum(l["child_ffn_header_params"] for l in layers)
    parent_non_mlp_non_mtp = parent_total - mtp - sum(l["parent_mlp_header_params"] for l in layers)
    native_extra = child_non_ffn - parent_non_mlp_non_mtp
    measured_drop = parent_total - child_total
    identity = measured_drop - (qwen_saving + mtp - k3_added - native_extra)
    slack = k3_added + native_extra
    groups: dict[str, int] = {}
    for n, v in pc.items():
        key = ".".join(n.split(".")[:2]) if n.startswith("model.") else n.split(".")[0]
        groups[key] = groups.get(key, 0) + v
    child_groups: dict[str, int] = {}
    for n, v in cc.items():
        key = n.split(".")[0]
        child_groups[key] = child_groups.get(key, 0) + v
    receipt = {
        "parent_header_parameters_by_group": groups, "child_header_parameters_by_group": child_groups,
        "schema": "neural-forge-s5-register-counts-v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_sha256": sha256_file(Path(__file__)), "oc_seal_sha256": sha256_file(Path(oc_seal.__file__)),
        "parent": {"path": str(parent), "header_parameters": parent_total, "mtp_header_parameters": mtp},
        "child": {"path": str(child), "header_parameters": child_total, "config_sha256": sha256_file(child / "config.json"),
                  "trunk_intermediate_size": q, "k3_channels": k3, "union_width": width, "hidden": hidden},
        "trunk": {"path": str(trunk), "header_parameters": trunk_total, "non_ffn_parameters": trunk_non_ffn},
        "child_non_ffn_parameters": child_non_ffn, "child_non_ffn_equals_trunk_non_ffn": child_non_ffn == trunk_non_ffn,
        "parent_non_mlp_non_mtp_parameters": parent_non_mlp_non_mtp,
        "native_export_extra_parameters": native_extra,
        "measured_drop": measured_drop, "qwen_mlp_saving": qwen_saving, "k3_rows_added": k3_added,
        "accounting_identity_residual": identity,
        "accounting_slack_argued": slack,
        "rule": "measured_drop = qwen_mlp_saving + mtp - k3_rows_added - native_export_extra (identity residual must be 0)",
        "layers": layers,
        "component_lines": [f"qwen.mlp.layer{l['layer']:02d}:{l['parent_mlp_header_params']}:{l['qwen_rows_params']}"
                            for l in layers],
    }
    if identity != 0:
        raise SystemExit(f"accounting identity residual {identity} != 0: {json.dumps(receipt)[:2000]}")
    if any(l["qwen_ratio"] > 0.25 for l in layers):
        raise SystemExit("a qwen.mlp component exceeds 0.25 of its parent MLP")
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"parent {parent_total:,}  child {child_total:,}  trunk {trunk_total:,}  mtp {mtp:,}")
    print(f"measured_drop {measured_drop:,}  qwen_saving {qwen_saving:,}  k3_added {k3_added:,}  native_extra {native_extra:,}")
    print(f"slack {slack:,}  identity residual {identity}  child_non_ffn == trunk_non_ffn {child_non_ffn == trunk_non_ffn}")
    print(f"receipt -> {args.out} sha256 {sha256_file(Path(args.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
