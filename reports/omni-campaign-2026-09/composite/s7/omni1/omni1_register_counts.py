#!/usr/bin/env python3
"""OMNI1 registration counts, read from safetensors headers (2026-10-13, Claude, OMNI1 assembly lane).

Same method as s5_register_counts.py (S5 build lane): the ledger derives a --component only from ONE tensor-name
prefix that the accounting parent's and the child's headers share.  Neither the Qwen trunk FFNs (HF names vs one
fused Chimera union tensor) nor the media block MLPs (their parents live in the five SOURCE model files, not in the
accounting parent's directory) have such a prefix, so this script derives every number the register line types from
the headers itself and writes the receipt the record's notes cite.  Nothing here is a fit number.

  trunk  : per trunk block of OMNI1 (block kinds from its config), the parent MLP (HF headers) and the child's
           union FFN split at c10_trunk_intermediate_size into the Qwen rows (the component, 0.25) and the K3 rows
           (passed through, added); the 64 component lines must equal the S5 receipt's (the trunk was copied).
  media  : per charted block, header sums under blocks.j.charted. (total), .mlp. (the component's child), .chart_
           (charts) and the rest (attention / time / norm, passed through at ~1x); the component's PARENT is the MLP
           prefix summed over the source model file's own headers (--media-source SRC=FILE:PREFIX), and must equal
           the c16 manifest's container accounting for it.
  totals : identity  omni1 = parent - qwen_saving - mtp + k3_rows + native_extra + media_installed  (residual 0),
           with media_installed = sum(parent_mlp) - media_saving + charts + media_non_mlp; the argued slack is the
           mass present in the child but not in the accounting parent: k3_rows + native_extra + media parent MLPs
           + charts + media non-MLP (the S5 rule, extended by the media).

    python omni1_register_counts.py --parent <HF dir> --omni1 <OMNI1 dir> --s5-counts <S5 register_counts.json>
        --media-source ace=<file>:<prefix> ... --out <receipt.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
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


def file_prefix_count(path: Path, prefix: str) -> tuple[int, dict[str, list[int]]]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    found = {k: v["shape"] for k, v in header.items() if k != "__metadata__" and k.startswith(prefix)}
    return sum(math.prod(s) for s in found.values()), found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--omni1", required=True)
    ap.add_argument("--s5-counts", required=True)
    ap.add_argument("--media-source", action="append", default=[], help="SRC=FILE:PREFIX (SRC = the charted spec's source label)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parent, omni = Path(args.parent), Path(args.omni1)
    s5c_path = Path(args.s5_counts)
    s5c = json.loads(s5c_path.read_text(encoding="utf-8"))
    manifest_path = omni / "composite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pc = oc_seal.safetensors_tensor_counts(parent)
    oc = oc_seal.safetensors_tensor_counts(omni)
    oh = headers(omni)
    cfg_doc = json.loads((omni / "config.json").read_text(encoding="utf-8"))
    conf = cfg_doc.get("config", cfg_doc)
    meta = conf.get("metadata", {})
    q = int(meta["c10_trunk_intermediate_size"])
    k3 = int(meta["c10_k3_channels"])
    hidden = int(conf["hidden_size"])
    kinds = list(conf["block_pattern"])
    if len(kinds) != int(conf["num_layers"]):
        raise SystemExit(f"block_pattern has {len(kinds)} entries for {conf['num_layers']} layers")
    charted_specs = {int(s["layer_index"]): s for s in conf.get("charted_blocks", [])}
    trunk_idx = [i for i, k in enumerate(kinds) if k != "charted"]
    charted_idx = [i for i, k in enumerate(kinds) if k == "charted"]
    if sorted(charted_specs) != charted_idx:
        raise SystemExit(f"charted specs {sorted(charted_specs)} != charted kinds {charted_idx}")
    width = q + k3
    # -- trunk ------------------------------------------------------------------------------------------
    layers = []
    for k, j in enumerate(trunk_idx):
        p_mlp = sum(v for n, v in pc.items() if n.startswith(f"model.language_model.layers.{k}.mlp."))
        c_ffn = sum(v for n, v in oc.items() if n.startswith(f"blocks.{j}.ffn."))
        up = [int(x) for x in oh[f"blocks.{j}.ffn.up.weight"]["shape"]]
        down = [int(x) for x in oh[f"blocks.{j}.ffn.down.weight"]["shape"]]
        if up != [2 * width, hidden] or down != [hidden, width] or c_ffn != 3 * hidden * width:
            raise SystemExit(f"trunk block {j} (layer {k}): FFN {up}/{down}/{c_ffn} is not the union width {width}")
        if p_mlp <= 0:
            raise SystemExit(f"layer {k}: no parent MLP tensors")
        layers.append({"layer": k, "omni1_block": j, "parent_mlp_header_params": p_mlp, "child_ffn_header_params": c_ffn,
                       "qwen_rows_params": 3 * hidden * q, "k3_rows_params": 3 * hidden * k3, "qwen_ratio": 3 * hidden * q / p_mlp})
    qwen_lines = [f"qwen.mlp.layer{l['layer']:02d}:{l['parent_mlp_header_params']}:{l['qwen_rows_params']}" for l in layers]
    if qwen_lines != list(s5c["component_lines"]):
        raise SystemExit("the trunk component lines differ from the S5 receipt's (the trunk was supposed to be copied)")
    # -- media --------------------------------------------------------------------------------------------
    sources = {}
    for item in args.media_source:
        label, rest = item.split("=", 1)
        file, prefix = rest.rsplit(":", 1)
        sources[label] = (Path(file), prefix)
    comp_by = {c["component"]: c for c in manifest["components"]}
    media = []
    for j in charted_idx:
        spec = charted_specs[j]
        label = spec["source"]
        pre = f"blocks.{j}.charted."
        total = sum(v for n, v in oc.items() if n.startswith(pre))
        mlp = sum(v for n, v in oc.items() if n.startswith(pre + "mlp."))
        charts = sum(v for n, v in oc.items() if n.startswith(pre + "chart_"))
        if label not in sources:
            raise SystemExit(f"no --media-source for {label!r}")
        sfile, sprefix = sources[label]
        p_mlp, found = file_prefix_count(sfile, sprefix)
        c16c = comp_by[spec["component"]]
        if int(c16c["parent_params"]) != p_mlp or int(c16c["child_params"]) != mlp:
            raise SystemExit(f"{spec['component']}: source headers {p_mlp} / child headers {mlp} != c16 manifest "
                             f"{c16c['parent_params']} / {c16c['child_params']}")
        media.append({"omni1_block": j, "source": label, "component": spec["component"], "type_ids": spec["type_ids"],
                      "header_parameters": total, "mlp_child_params": mlp, "chart_params": charts,
                      "non_mlp_block_params": total - mlp - charts, "parent_mlp_params": p_mlp, "ratio": mlp / p_mlp,
                      "parent_source_file": str(sfile), "parent_prefix": sprefix, "parent_tensors": found,
                      "parent_source_file_header_only": True})
    if any(m["ratio"] > 0.25 for m in media) or any(l["qwen_ratio"] > 0.25 for l in layers):
        raise SystemExit("a component exceeds 0.25 of its parent")
    media_lines = [f"{m['component']}:{m['parent_mlp_params']}:{m['mlp_child_params']}" for m in media]
    # -- totals / identity ----------------------------------------------------------------------------------
    parent_total, omni_total = sum(pc.values()), sum(oc.values())
    mtp = sum(v for n, v in pc.items() if n.startswith("mtp."))
    qwen_saving = sum(l["parent_mlp_header_params"] - l["qwen_rows_params"] for l in layers)
    k3_added = sum(l["k3_rows_params"] for l in layers)
    media_installed = sum(m["header_parameters"] for m in media)
    media_parent = sum(m["parent_mlp_params"] for m in media)
    media_saving = sum(m["parent_mlp_params"] - m["mlp_child_params"] for m in media)
    charts = sum(m["chart_params"] for m in media)
    media_non_mlp = sum(m["non_mlp_block_params"] for m in media)
    child_non_ffn = omni_total - media_installed - sum(l["child_ffn_header_params"] for l in layers)
    parent_non_mlp_non_mtp = parent_total - mtp - sum(l["parent_mlp_header_params"] for l in layers)
    native_extra = child_non_ffn - parent_non_mlp_non_mtp
    measured_drop = parent_total - omni_total
    identity = omni_total - (parent_total - qwen_saving - mtp + k3_added + native_extra + media_installed)
    identity_media = media_installed - (media_parent - media_saving + charts + media_non_mlp)
    slack = k3_added + native_extra + media_parent + charts + media_non_mlp
    claimed = qwen_saving + media_saving
    claimed_parent = sum(l["parent_mlp_header_params"] for l in layers) + media_parent
    interface_written = sum(sum(int(w["parameters"]) for w in e["writes"].values()) for e in (manifest.get("interface") or {}).get("entries", []))
    band_written = sum(int(w["parameters"]) for w in ((manifest.get("band") or {}).get("writes") or {}).values())
    receipt = {
        "schema": "neural-forge-omni1-register-counts-v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_sha256": sha256_file(Path(__file__)), "oc_seal_sha256": sha256_file(Path(oc_seal.__file__)),
        "parent": {"path": str(parent), "header_parameters": parent_total, "mtp_header_parameters": mtp},
        "omni1": {"path": str(omni), "header_parameters": omni_total, "config_sha256": sha256_file(omni / "config.json"),
                  "manifest_sha256": sha256_file(manifest_path), "layers": len(kinds), "trunk_blocks": len(trunk_idx),
                  "charted_blocks": charted_idx, "trunk_intermediate_size": q, "k3_channels": k3, "hidden": hidden},
        "s5_counts": {"path": str(s5c_path), "sha256": sha256_file(s5c_path), "s5_header_parameters": s5c["child"]["header_parameters"],
                      "component_lines_equal": True, "native_export_extra_equal": native_extra == int(s5c["native_export_extra_parameters"])},
        "omni1_minus_s5": omni_total - int(s5c["child"]["header_parameters"]),
        "measured_drop": measured_drop, "qwen_mlp_saving": qwen_saving, "k3_rows_added": k3_added,
        "native_export_extra_parameters": native_extra, "child_non_ffn_non_charted_parameters": child_non_ffn,
        "parent_non_mlp_non_mtp_parameters": parent_non_mlp_non_mtp,
        "media": media, "media_installed_parameters": media_installed, "media_parent_mlp_parameters": media_parent,
        "media_mlp_saving": media_saving, "media_chart_parameters": charts, "media_non_mlp_block_parameters": media_non_mlp,
        "interface_parameters_written_in_place": interface_written, "band_parameters_written_in_place": band_written,
        "claimed_saving": claimed, "claimed_parent_parameters": claimed_parent,
        "accounting_identity_residual": identity, "media_identity_residual": identity_media,
        "accounting_slack_argued": slack,
        "slack_covers_claim": claimed <= measured_drop + slack,
        "rule": "omni1 = parent - qwen_saving - mtp + k3_rows + native_extra + media_installed (residual 0); "
                "slack = k3_rows + native_extra + media parent MLPs + charts + media non-MLP",
        "layers": layers,
        "component_lines": qwen_lines + media_lines,
        "qwen_component_lines": qwen_lines, "media_component_lines": media_lines,
    }
    if identity != 0 or identity_media != 0:
        raise SystemExit(f"accounting identity residual {identity} / media {identity_media} != 0")
    if not receipt["slack_covers_claim"] or claimed_parent > parent_total or measured_drop <= 0:
        raise SystemExit("the ledger's fold accounting would refuse these numbers")
    if not receipt["s5_counts"]["native_export_extra_equal"]:
        raise SystemExit("native-export difference changed between S5 and OMNI1")
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"parent {parent_total:,}  omni1 {omni_total:,}  (S5 + {receipt['omni1_minus_s5']:,})  mtp {mtp:,}")
    print(f"measured_drop {measured_drop:,}  qwen_saving {qwen_saving:,}  media_saving {media_saving:,}  k3_added {k3_added:,}  native_extra {native_extra:,}")
    print(f"media installed {media_installed:,} = parent MLP {media_parent:,} - saving {media_saving:,} + charts {charts:,} + non-MLP {media_non_mlp:,}")
    for m in media:
        print(f"  {m['component']}: parent {m['parent_mlp_params']:,} ({Path(m['parent_source_file']).name}:{m['parent_prefix']}) -> child {m['mlp_child_params']:,} "
              f"ratio {m['ratio']:.5f}; charts {m['chart_params']:,}; non-MLP {m['non_mlp_block_params']:,}")
    print(f"slack {slack:,}  claimed {claimed:,} <= drop + slack {measured_drop + slack:,}  identity residual {identity} / media {identity_media}")
    print(f"receipt -> {args.out} sha256 {sha256_file(Path(args.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
