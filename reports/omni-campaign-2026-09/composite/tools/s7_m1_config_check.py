#!/usr/bin/env python3
"""s7_m1_config_check.py -- load the five S7 units (with their 5120-wide charts) against the REAL S3 trunk config.

Lane M1, Claude, 2026-10-10. No trunk tensor is read: the S3 native `config.json` alone gives the
ChimeraConfig (hidden 5120, 64 layers, kda x3 + gated_gqa). Through c16_install_media's own functions
it then

  * load_container(unit, cfg)       -- tensor sets / shapes at hidden 5120, declared tensor_sha256 (charts included)
  * verify_time_reference(...)      -- adaLN units (ACE, QIMG) against their captured modulation (tolerance per unit)
  * build_source_block(...)         -- ChartedSourceBlock strict load of block + charts at hidden 5120
  * pair_containers(depth-fraction) -- where each block would be inserted in the real trunk
  * write strength (diagnostic)     -- |chart_out(block(chart_in x) - chart_in x)| / |x| on HELD host rows of one
                                       forward per unit, through the installed ChartedSourceBlock (H3 needs
                                       h3_thw_v1 source positions; not built here, recorded as not measured)

Prints one JSON document; `--out` also writes it. Diagnostics only; a load/shape/sha failure is a refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
HERE = Path(__file__).resolve().parent
S3_CONFIG = Path("/mnt/lexar/omni-campaign/s3-native-v2-3a25a459-4347-416d-a58a-2cb3c8e738fd/actual/native-s3-text-stage/native/config.json")
UNITS = ["ace/ace_dit_L12", "h3/h3_dit_L25_curve6", "qimg/qimg_mmdit_L30_img", "hy3d/hy3d_dit_L10", "qvl32/qvl32_vit_L13"]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chimera-src", default=str(OC / "import/rq27b/tree_ef1ffb9"))
    p.add_argument("--units-root", default=str(OC / "composite/s7/units"))
    p.add_argument("--tap-rows", default=str(OC / "composite/s7/tap_rows_s3"))
    p.add_argument("--qimg-time-tol", type=float, default=2.1e-3)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    sys.path.insert(0, str(Path(args.chimera_src) / "src"))
    spec = importlib.util.spec_from_file_location("c16_m1", HERE / "c16_install_media.py")
    c16 = importlib.util.module_from_spec(spec)
    sys.modules["c16_m1"] = c16
    spec.loader.exec_module(c16)
    import torch
    from safetensors import safe_open
    from chimera_omni.config import ChimeraConfig

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    document = json.loads(S3_CONFIG.read_text())
    cfg = ChimeraConfig.from_manifest(document["config"])
    hidden = int(cfg.hidden_size)
    report = {"config": str(S3_CONFIG), "config_sha256": sha256_file(S3_CONFIG), "hidden": hidden, "num_layers": int(cfg.num_layers),
              "chimera_tree": args.chimera_src, "c16_sha256": sha256_file(HERE / "c16_install_media.py"), "units": {}}
    containers = []
    for rel in UNITS:
        d = Path(args.units_root) / rel
        c = c16.load_container(d, cfg)
        c["hidden"] = hidden
        entry = {"dir": str(d), "chart_source": c["chart_source"], "tensor_sha256_check": c["tensor_sha256_check"],
                 "dropped_chart_keys": c["dropped_chart_keys"], "type_ids": list(c["spec"].type_ids),
                 "chart_shapes": {k: list(v.shape) for k, v in (c["charts"] or {}).items()},
                 "files": c["files"]}
        if c["charts"] is None:
            raise SystemExit(f"{d}: no charts.safetensors")
        if c["spec"].modulation == "adaln6":
            tol = args.qimg_time_tol if c["spec"].source == "qimg" else 1.0e-5
            entry["time_reference"] = {k: v for k, v in c16.verify_time_reference(c, hidden, rel_tol=tol).items() if k in ("rel_l2", "tolerance_rel_l2", "verified", "sigma")}
        block = c16.build_source_block(c["spec"], hidden, c["tensors"], c["charts"])
        entry["installed_parameters"] = int(sum(int(t.numel()) for t in block.parameters()))
        entry["chart_parameters"] = int(sum(int(v.numel()) for v in c["charts"].values()))
        entry["spec_mlp_parameters"] = int(c["spec"].mlp_parameter_count())
        # write strength on held host rows of one forward (diagnostic)
        name = d.name
        rows_path = Path(args.tap_rows) / name / "paired_rows.safetensors"
        if c["spec"].source == "h3":
            entry["write_strength"] = {"measured": False, "why": "H3 needs explicit h3_thw_v1 source positions per row; not constructed in this check"}
        elif rows_path.is_file():
            with safe_open(str(rows_path), "pt") as h:
                held = h.get_tensor("held")
                fwd = h.get_tensor("forward")
                pick = int(fwd[held][0])
                m = fwd == pick
                x = h.get_tensor("host_rows")[m].unsqueeze(0)
                types = h.get_tensor("type_ids")[m].unsqueeze(0)
                sigma = h.get_tensor("sigma")[m].unsqueeze(0)
            levels = None if bool((sigma < 0).all()) else sigma
            valid = torch.ones(x.shape[:2], dtype=torch.bool)
            with torch.no_grad():
                write = block(x, types, levels, valid)
            if not bool(torch.isfinite(write).all()):
                raise SystemExit(f"{name}: non-finite charted write on real host rows")
            ratio = write.norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-30)
            entry["write_strength"] = {"measured": True, "forward": pick, "rows": int(x.shape[1]),
                                       "write_over_host_norm_median": float(ratio.median()), "p95": float(ratio.quantile(0.95)),
                                       "host_norm_median": float(x.norm(dim=-1).median())}
        del block
        containers.append(c)
        report["units"][name] = entry
    c16.pair_containers(containers, cfg, pair_by="depth-fraction")
    for c in containers:
        report["units"][c["name"]]["pairing"] = c["pairing"]
    report["installed_parameters_total"] = int(sum(u["installed_parameters"] for u in report["units"].values()))
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
