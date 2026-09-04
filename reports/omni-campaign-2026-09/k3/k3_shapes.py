#!/usr/bin/env python3
"""Header-only shape census for a K3 layer.  Reads no tensor data at all.

A safetensors header is a JSON blob at the front of the file, so this answers
"what shape is `A_log` really" for a few megabytes of sequential read instead
of the 16.99 GB (53 minutes at the measured 5.3 MB/s) a layer load costs.  It
is the cheapest thing in this lane and it settles the one shape R6 left
[UNRESOLVED]: the modeling code declares `A_log` as `[num_heads]` and fla's
gate does `A_log.view(H, 1)`, while a first-pass header read was reported as
f32[128] (= head_dim).  One of those is wrong and the header decides.

  python k3_shapes.py --model-dir /mnt/ai_data/models/moonshotai/Kimi-K3 --layer 1
"""
from __future__ import annotations

import argparse
import json
import os

from k3_common import K3Index, log, read_header, write_receipt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--pattern", default=None, help="only report keys containing this substring")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    index = K3Index(args.model_dir)
    cfg = index.text_config
    keys = index.layer_keys(args.layer)
    shards = index.shards_for(keys)
    shapes: dict[str, dict] = {}
    for shard in shards:
        header, _ = read_header(os.path.join(args.model_dir, shard))
        for key in keys:
            if index.weight_map[key] != shard or key not in header:
                continue
            if args.pattern and args.pattern not in key:
                continue
            meta = header[key]
            shapes[key] = {"shape": meta["shape"], "dtype": meta["dtype"], "shard": shard,
                           "bytes": meta["data_offsets"][1] - meta["data_offsets"][0]}
        log(f"{shard}: header parsed, {sum(1 for k in keys if index.weight_map[k] == shard)} layer tensors")

    linear = cfg.get("linear_attn_config") or {}
    prefix = index.layer_prefix_for(args.layer)
    verdict = {}
    for short, expected, why in (
        ("self_attn.A_log", linear.get("num_heads"),
         "fla's gate does A_log.view(num_heads, 1); Chimera maps it to decay_log_scale[heads]"),
        ("self_attn.dt_bias", (linear.get("num_heads") or 0) * (linear.get("head_dim") or 0),
         "dt_bias.view(num_heads, head_dim) -- one bias per (head, key dim)"),
        ("self_attn.o_norm.weight", linear.get("head_dim"),
         "FusedRMSNormGated(head_dim): ONE vector shared by every head"),
    ):
        key = prefix + short
        if key not in shapes:
            continue
        numel = 1
        for dim in shapes[key]["shape"]:
            numel *= dim
        verdict[short] = {
            "shape": shapes[key]["shape"], "dtype": shapes[key]["dtype"], "numel": numel,
            "expected_numel": expected, "matches": (expected is not None and numel == expected),
            "why_it_matters": why,
        }

    receipt = {
        "model_dir": args.model_dir, "layer": args.layer, "shards": shards,
        "is_kda_layer": index.is_kda_layer(args.layer),
        "is_moe_layer": index.is_moe_layer(args.layer),
        "tensor_count": len(shapes),
        "kda_shape_verdict": verdict,
        "shapes": shapes if args.pattern or len(shapes) <= 200 else
                  {k: v for k, v in shapes.items() if ".experts." not in k},
        "note": "header-only read: no tensor data was transferred off the share",
    }
    out = args.out or f"k3_layer{args.layer}_shapes.json"
    receipt["receipt_sha256"] = write_receipt(out, receipt)
    print(json.dumps(verdict, indent=1))
    log(f"receipt -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
