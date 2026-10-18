#!/usr/bin/env python3
"""s5m_plan_check.py -- header/meta-mode check of the S7 install plan against the WRITTEN S5 child (no write).

S7-on-S5 media lane, Claude, 2026-10-13. s7_dryrun.py has no header-only mode, so this check runs
c16_install_media's OWN loaders and pairing (imported, not edited) against the S5 child's config and
lazily-read tensors, and stops before anything is written:

  1. config: read_checkpoint_config(S5) -> ChimeraConfig; header count of the S5 child (c16.header_counts).
  2. interface: c16.load_interface_set(C15) + (C11) against the S5 config (hidden, native_feature_dim, type
     range, band fit, declared tensor sha256), c16.check_bands, the per-entry row shapes c16 install_interface
     requires, and its zero preconditions read from the S5 child's own tensors (type_feature_scale band, both
     type_affine scale halves).
  3. band: the C18 table against the S5 ledger music_codes band (shape, untied head, declared sha256) and the
     S5 band rows' current non-zero count in embed.weight and lm_head.weight.
  4. units: c16.load_container on every unit (canonical tensor set, shapes at the S5 hidden, declared sha256),
     c16.verify_time_reference (diagnostic policy), c16.pair_containers(depth-fraction) on the S5 config -> the
     trunk layer each unit inserts after (must equal the host-row capture site, 31).
  5. finite forward: c16.build_source_block for every unit; the block's forward on a sample of the unit's own
     S5 host rows (tap_rows_s5, x_31 of the S5 child) through its charts -> finite outputs; written rows are
     compared with the input rows (delta norm ratio recorded, a diagnostic).
  6. accounting: S5 header + the units' installed parameters (c16's container count at hidden 5120) vs the
     accounting parent's typed count; the C15/C11/C18 writes add no parameters.

Structural failures exit 3 with the reason; finite numbers are recorded (owner policy 2026-10-06).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

TOOL = {"name": "s5m_plan_check", "version": "1.0.0"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--child", required=True)
    p.add_argument("--tools", required=True)
    p.add_argument("--c15", required=True)
    p.add_argument("--c11", required=True)
    p.add_argument("--c18", required=True)
    p.add_argument("--units", nargs="+", required=True)
    p.add_argument("--tap-rows", required=True)
    p.add_argument("--expected-insert-after", type=int, default=31)
    p.add_argument("--parent-parameter-count", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    started = time.time()
    c16 = load("c16_install_media_plan", Path(args.tools) / "c16_install_media.py")
    import torch
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import read_checkpoint_config

    refusals = []
    rep = {"tool": TOOL, "tool_sha256": sha256_file(__file__), "c16_sha256": sha256_file(Path(args.tools) / "c16_install_media.py"),
           "c16_version": getattr(c16, "TOOL_VERSION", None), "argv": vars(args)}
    child = Path(args.child)
    doc = read_checkpoint_config(child)
    cfg = ChimeraConfig.from_manifest(doc["config"])
    counts = c16.header_counts(child)
    s5_header = int(sum(counts.values())) if isinstance(counts, dict) else int(counts)
    rep["child"] = {"directory": str(child), "config_sha256": sha256_file(child / "config.json"),
                    "index_sha256": sha256_file(child / "model.safetensors.index.json"), "header_parameters": s5_header,
                    "hidden": int(cfg.hidden_size), "layers": int(cfg.num_layers), "native_feature_dim": int(cfg.native_feature_dim),
                    "native_type_count": int(cfg.native_type_count), "vocab": int(cfg.vocab_size), "tied": bool(cfg.tie_word_embeddings)}
    views, index = c16.open_lazy_checkpoint(child)
    hidden = int(cfg.hidden_size)

    # 2. interface
    entries = []
    try:
        entries = c16.load_interface_set(Path(args.c15), "c15", cfg) + c16.load_interface_set(Path(args.c11), "c11", cfg)
        rep["bands"] = c16.check_bands(entries, allow_overlap=False)
    except c16.Refusal as e:
        refusals.append(f"interface: {e}")
    tfs = views["native_media_bridge.type_feature_scale.weight"].torch(dtype=torch.float32)
    baff = views["native_media_bridge.type_affine.weight"].torch(dtype=torch.float32)
    haff = views["native_media_head.type_affine.weight"].torch(dtype=torch.float32)
    fproj_shape = tuple(views["native_media_bridge.feature_proj.weight"].shape)
    hout_shape = tuple(views["native_media_head.out.weight"].shape)
    iface = []
    for e in entries:
        v = c16.read_single(e["rows_path"])
        b, en = e["feature_band"]
        eb, ee = e["emission_band"]
        need = {"bridge.feature_proj.weight": (hidden, en - b), "bridge.type_embed_delta": (hidden,),
                "head.out.weight": (ee - eb, hidden), "head.type_affine_shift": (ee - eb,)}
        shapes_ok = all(k in v and tuple(v[k].shape) == s for k, s in need.items())
        finite = all(bool(torch.isfinite(v[k].torch(dtype=torch.float32)).all()) for k in need if k in v)
        t = e["type_id"]
        pre = {"type_feature_scale[type, band]": int(tfs[t, b:en].ne(0).sum()),
               "bridge.type_affine[type, scale half]": int(baff[t, :hidden].ne(0).sum()),
               "head.type_affine[type, scale half]": int(haff[t, :hidden].ne(0).sum())}
        iface.append({"set": e["set"], "name": e["name"], "type_id": t, "feature_band": [b, en], "emission_band": [eb, ee],
                      "rows_sha256": e["rows_sha256"], "tensor_sha256_check": e["tensor_sha256_check"], "row_shapes_ok": shapes_ok,
                      "finite": finite, "preconditions_nonzero": pre})
        if not shapes_ok:
            refusals.append(f"interface {e['name']}: row shapes do not fit the S5 bridge/head")
        if not finite:
            refusals.append(f"interface {e['name']}: non-finite rows")
        if any(pre.values()):
            refusals.append(f"interface {e['name']}: non-zero preconditions {pre}")
    rep["interface"] = {"entries": iface, "bridge_feature_proj_shape": list(fproj_shape), "head_out_shape": list(hout_shape),
                        "parameters_added": 0, "note": "rows written into the counted bridge/head tensors"}

    # 3. band
    band = cfg.ledger().range("music_codes")
    table = c16.read_single(Path(args.c18))
    side = Path(args.c18 + ".json")
    side_body = json.loads(side.read_text()) if side.is_file() else {}
    sha_check = c16.verify_declared_sha256(table, side_body.get("tensor_sha256"), where=args.c18)
    emb_rows = views["embed.weight"].torch(dtype=torch.float32)[band.begin:band.end]
    head_rows = views["lm_head.weight"].torch(dtype=torch.float32)[band.begin:band.end] if "lm_head.weight" in views else None
    tshape = tuple(table["embed.weight"].shape)
    rep["band"] = {"band": [band.begin, band.end, band.width], "table": args.c18, "table_sha256": sha256_file(args.c18),
                   "table_shape": list(tshape), "lm_head_in_table": "lm_head.weight" in table, "sha_check": sha_check,
                   "s5_embed_band_nonzero": int(emb_rows.ne(0).sum()),
                   "s5_lm_head_band_nonzero": None if head_rows is None else int(head_rows.ne(0).sum()),
                   "table_finite": all(bool(torch.isfinite(table[k].torch(dtype=torch.float32)).all()) for k in table),
                   "parameters_added": 0}
    if tshape != (band.width, hidden):
        refusals.append(f"band: table {tshape} vs band [{band.width}, {hidden}]")
    if not cfg.tie_word_embeddings and "lm_head.weight" not in table:
        refusals.append("band: untied S5 head but the table carries no lm_head.weight")
    if rep["band"]["s5_embed_band_nonzero"] or (rep["band"]["s5_lm_head_band_nonzero"] or 0):
        refusals.append("band: S5 band rows are not zero")
    if not rep["band"]["table_finite"]:
        refusals.append("band: non-finite table")
    del emb_rows, head_rows

    # 4. units
    containers = []
    for d in args.units:
        try:
            c = c16.load_container(Path(d), cfg)
            c["hidden"] = hidden
            if c["charts"] is None:
                refusals.append(f"{d}: no charts.safetensors")
                continue
            c["time_report"] = c16.verify_time_reference(c, hidden, rel_tol=c16.DEFAULT_TIME_REFERENCE_REL_TOL, strict=False)
            containers.append(c)
        except c16.Refusal as e:
            refusals.append(f"unit {d}: {e}")
    try:
        c16.pair_containers(containers, cfg, pair_by="depth-fraction")
    except c16.Refusal as e:
        refusals.append(f"pairing: {e}")
    units = []
    installed_total = 0
    for c in containers:
        spec = c["spec"]
        block = c16.build_source_block(spec, hidden, c["tensors"], c["charts"])
        n_block = sum(int(t.numel()) for t in block.state_dict().values())
        installed_total += n_block
        rows_path = Path(args.tap_rows) / c["name"] / "paired_rows.safetensors"
        fwd = None
        if rows_path.is_file():
            r = c16.read_single(rows_path)
            host = r["host_rows"].torch(dtype=torch.float32)
            held = r["held"].torch().to(torch.bool)
            forward = r["forward"].torch()
            f0 = int(forward[held][0])
            idx = torch.nonzero(forward == f0).flatten()
            fwd = forward_probe(block, host[idx], r["type_ids"].torch()[idx], r["sigma"].torch(dtype=torch.float32)[idx], torch)
            fwd.update({"forward_index": f0, "rows": int(idx.numel()), "held_forward": True,
                        "host_rows_file": str(rows_path), "host_rows_sha256": sha256_file(rows_path)})
            if not fwd["finite"]:
                refusals.append(f"{c['name']}: non-finite forward")
        else:
            refusals.append(f"{c['name']}: no S5 host rows at {rows_path}")
        ia = int(c["insert_after"])
        if ia != int(args.expected_insert_after):
            refusals.append(f"{c['name']}: pairs after layer {ia}, host rows were captured at {args.expected_insert_after}")
        units.append({"name": c["name"], "directory": str(c["directory"]), "files": c["files"], "insert_after": ia,
                      "pairing": c["pairing"], "type_ids": list(spec.type_ids), "width": int(spec.width),
                      "block_parameters_at_hidden": n_block, "tensor_sha256_check": c["tensor_sha256_check"],
                      "time_reference": None if c.get("time_report") is None else {k: c["time_report"][k] for k in ("rel_l2", "verified", "tolerance_rel_l2")},
                      "forward_on_s5_host_rows": fwd})
    rep["units"] = units
    total = s5_header + installed_total
    rep["accounting"] = {"s5_header_parameters": s5_header, "units_installed_parameters": installed_total,
                         "interface_and_band_parameters": 0, "composite_header_parameters": total,
                         "accounting_parent_parameters": int(args.parent_parameter_count),
                         "headroom": int(args.parent_parameter_count) - total,
                         "below_parent": total < int(args.parent_parameter_count)}
    if total >= int(args.parent_parameter_count):
        refusals.append("accounting: composite would not be below the accounting parent")
    rep["refusals"] = refusals
    rep["verdict"] = ("PLAN HOLDS on the S5 child: interface + C11 + C18 + 5 units load, pair, account and run finite"
                      if not refusals else "REFUSED")
    rep["seconds"] = time.time() - started
    out = Path(args.out)
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    Path(str(out) + ".sha256").write_text(sha256_file(out) + "\n")
    print(rep["verdict"])
    for r in refusals:
        print("REFUSAL", r)
    print("accounting", json.dumps(rep["accounting"]))
    for u in units:
        f = u["forward_on_s5_host_rows"] or {}
        print(u["name"], "insert_after", u["insert_after"], "params", u["block_parameters_at_hidden"], "finite", f.get("finite"),
              "rows", f.get("rows"), "written", f.get("rows_written"), "write/x", f.get("write_norm_over_x_norm_median"), "time", u["time_reference"])
    print("out", out, sha256_file(out))
    return 0 if not refusals else 3


def forward_probe(block, x, types, sigma, torch):
    """The charted block's residual write on ONE held forward of its own S5 host rows (x_31), as one segment.

    row types and per-row sigma are the pair set's; source positions (H3 only) are zeros, because no tool emits
    h3_thw_v1 positions yet (a recorded limitation, S7 readiness table)."""

    n = int(x.shape[0])
    hs = x.unsqueeze(0)
    kwargs = {}
    if getattr(block, "h3_time", None) is not None:
        kwargs["source_positions"] = torch.zeros(1, n, 3)
    noise = None if bool((sigma < 0).all()) else sigma.unsqueeze(0)
    with torch.no_grad():
        write = block(hs, types.unsqueeze(0), noise, torch.ones(1, n, dtype=torch.bool), **kwargs)[0].float()
    owned = write.abs().sum(dim=1) > 0
    ratio = (write.norm(dim=1) / x.norm(dim=1).clamp_min(1e-30))
    return {"finite": bool(torch.isfinite(write).all()), "output_shape": list(write.shape), "rows_written": int(owned.sum()),
            "source_positions": "zeros (H3)" if kwargs else None, "noise_levels": None if noise is None else "pair-set sigma",
            "write_norm_over_x_norm_median": float(ratio[owned].median()) if bool(owned.any()) else 0.0,
            "x_norm_median": float(x.norm(dim=1).median())}


if __name__ == "__main__":
    sys.exit(main())
