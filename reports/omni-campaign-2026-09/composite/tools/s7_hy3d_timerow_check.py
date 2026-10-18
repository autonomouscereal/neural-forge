#!/usr/bin/env python3
"""s7_hy3d_timerow_check.py -- prove the HY3D time-token row reaches the installed charted block.

PREP-B, Claude, 2026-10-10. Runs on a c16 dry-run output (`s7_dryrun.py --hidden 5120 --build-child`:
a tiny synthetic Chimera child at hidden 5,120 with the hy3d_dit_L10 unit installed by
c16_install_media) and, optionally, on the unit's real paired rows.

Part A, end to end through the synthetic trunk (ChimeraOmniModel.forward, native media rows, velocity
layout so the per-row sigma reaches the bridge noise_proj):

    sequence = [prompt | TIME ROW | N latent rows]    (runtime_requirements.hy3d_time_token_row layout:
                                                       type 7, zero features, position [-1,-1,-1,-1],
                                                       the segment sigma)

  A forward hook on the HY3D ChartedSourceBlock records what the block receives. STRUCTURAL checks
  (refuse, exit 3): the time row's sequence position is routed to the block (row type 7), the block
  owns 1 + N rows, it writes a non-zero delta onto the time row, and re-running the SAME block input
  with the time row un-owned changes the latent-row write (an exactly-zero change would mean the row
  never reaches the block). DIAGNOSTICS (recorded, never gates): that change's relL2; the change in the
  final hidden state of the latent rows with vs without the time row in the sequence; the latent-row
  write when only the time row's trunk state is swapped between sigma 0.2 and 0.8 (the block sees the
  timestep through that row).

Part B (with --paired-rows), the real unit at hidden 5,120 on real S3-trunk host rows of one HELD
object: the latent-row write with the time row owned vs not; the chart-mapped time row vs the parent's
block-10 row 0; and (with --parent-capture) the source-space reference: the unit's own source block on
the parent's block-10 input rows with vs without row 0.

    CHIMERA_OMNI_SRC=<tree> PYTHONPATH=<tree>/src python s7_hy3d_timerow_check.py --out OUT --unit UNIT_DIR \
        [--paired-rows P] [--parent-capture C] --report R.json
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
from typing import Any

TOOL_NAME = "s7_hy3d_timerow_check"
TOOL_VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
TIME_ROW_POSITION = (-1.0, -1.0, -1.0, -1.0)
VOLUME_TYPE = 7


class Refusal(Exception):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_trc", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"{name}_trc"] = module
    spec.loader.exec_module(module)
    return module


def rel(a: Any, b: Any) -> float:
    """||a - b|| / ||a||."""

    return float((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-30))


def media_inputs(torch: Any, cfg: Any, *, prompt: Any, latent_features: Any, latent_positions: Any, sigma: float,
                 with_time: bool) -> dict[str, Any]:
    """Model kwargs for [prompt | (time row) | latent rows] under the hy3d_time_token_row layout."""

    plen = int(prompt.shape[0])
    n_lat = int(latent_features.shape[0])
    n = n_lat + (1 if with_time else 0)
    placeholder = int(cfg.ledger().range("media").begin)
    ids = torch.cat((prompt, torch.full((n,), placeholder, dtype=torch.long))).view(1, -1)
    feats = torch.zeros(1, n, int(cfg.native_feature_dim))
    pos = torch.zeros(1, n, 4)
    off = 1 if with_time else 0
    feats[0, off:] = latent_features
    pos[0, off:] = latent_positions
    if with_time:
        pos[0, 0] = torch.tensor(TIME_ROW_POSITION)
    valid = torch.ones(1, n, dtype=torch.bool)
    return {
        "input_ids": ids,
        "native_media_features": feats,
        "native_media_type_ids": torch.full((1, n), VOLUME_TYPE, dtype=torch.long),
        "native_media_positions": pos,
        "native_media_indices": torch.arange(plen, plen + n).view(1, -1),
        "native_media_valid_mask": valid,
        "native_media_update_mask": valid.clone(),
        "native_media_noise_levels": torch.full((1, n), float(sigma)),
        "native_media_target_kind": "velocity",
        "native_media_bidirectional_refinement": False,
    }


def part_a(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from chimera_omni.importing.safetensors_io import load_chimera_checkpoint

    model, cfg, load_receipt = load_chimera_checkpoint(Path(args.out))
    model = model.eval().to(torch.float32)
    found = [(i, b) for i, b in enumerate(model.blocks) if getattr(b, "charted", None) is not None and b.charted.spec.source == "hy3d"]
    if len(found) != 1:
        raise Refusal(f"{args.out}: {len(found)} hy3d charted blocks (want 1)")
    layer, block = found[0]
    if VOLUME_TYPE not in [int(t) for t in block.charted.spec.type_ids]:
        raise Refusal(f"hy3d block does not route type {VOLUME_TYPE}: {block.charted.spec.type_ids}")
    g = torch.Generator().manual_seed(int(args.seed))
    plen, n_lat = int(args.prompt_tokens), int(args.latent_rows)
    prompt = torch.randint(10, 200, (plen,), generator=g)
    latent = torch.randn(n_lat, int(cfg.native_feature_dim), generator=g)
    lpos = torch.zeros(n_lat, 4)
    lpos[:, :3] = torch.rand(n_lat, 3, generator=g)
    seen: list[dict[str, Any]] = []

    def hook(module, hook_args, hook_kwargs, output):
        x, row_types, noise, token_valid = hook_args[:4]
        seen.append({"x": x.detach().clone(), "row_types": None if row_types is None else row_types.detach().clone(),
                     "noise": None if noise is None else noise.detach().clone(), "token_valid": token_valid.detach().clone(),
                     "write": output.detach().clone()})

    handle = block.charted.register_forward_hook(hook, with_kwargs=True)

    def run(with_time: bool, sigma: float) -> dict[str, Any]:
        seen.clear()
        kw = media_inputs(torch, cfg, prompt=prompt, latent_features=latent, latent_positions=lpos, sigma=sigma, with_time=with_time)
        with torch.no_grad():
            out = model(**kw)
        if len(seen) != 1:
            raise Refusal(f"the hy3d charted block ran {len(seen)} times in one forward (want 1)")
        return {"out": out, "cap": seen[0]}

    started = time.time()
    with_t = run(True, args.sigma)
    without_t = run(False, args.sigma)
    lo_t = run(True, 0.2)
    hi_t = run(True, 0.8)
    handle.remove()
    cap = with_t["cap"]
    t_pos = plen  # the time row's sequence position
    lat = slice(plen + 1, plen + 1 + n_lat)
    row_types = cap["row_types"]
    checks = {
        "time_row_type_at_its_position": int(row_types[0, t_pos]),
        "rows_owned_by_block": int(row_types[0].eq(VOLUME_TYPE).sum()),
        "rows_owned_expected": n_lat + 1,
        "time_row_write_norm": float(cap["write"][0, t_pos].norm()),
        "latent_write_norm": float(cap["write"][0, lat].norm()),
        "time_row_sigma_at_block": float(cap["noise"][0, t_pos]) if cap["noise"] is not None else None,
    }
    if checks["time_row_type_at_its_position"] != VOLUME_TYPE:
        raise Refusal(f"the time row is not routed to the hy3d block: {checks}")
    if checks["rows_owned_by_block"] != checks["rows_owned_expected"]:
        raise Refusal(f"owned rows {checks['rows_owned_by_block']} != {checks['rows_owned_expected']}")
    if not checks["time_row_write_norm"] > 0.0:
        raise Refusal("the block writes nothing onto its time row")
    # same block input, time row un-owned (row type 0): isolates the block's self-attention over the time row
    unowned = row_types.clone()
    unowned[0, t_pos] = 0
    with torch.no_grad():
        write_unowned = block.charted(cap["x"], unowned, cap["noise"], cap["token_valid"])
    if float(write_unowned[0, t_pos].abs().max()) != 0.0:
        raise Refusal("an un-owned row received a write")
    change = rel(cap["write"][0, lat], write_unowned[0, lat])
    if not change > 0.0:
        raise Refusal("the latent-row write is bit-identical with and without the time row: it never reaches the block")
    # timestep through the time row: hold every latent row's block input, swap only the time row's trunk state
    x_lo = cap["x"].clone()
    x_hi = cap["x"].clone()
    x_lo[0, t_pos] = lo_t["cap"]["x"][0, t_pos]
    x_hi[0, t_pos] = hi_t["cap"]["x"][0, t_pos]
    with torch.no_grad():
        w_lo = block.charted(x_lo, row_types, cap["noise"], cap["token_valid"])
        w_hi = block.charted(x_hi, row_types, cap["noise"], cap["token_valid"])
    hidden_with = with_t["out"]["hidden"][0, lat]
    hidden_without = without_t["out"]["hidden"][0, plen : plen + n_lat]
    return {
        "child": {"out": str(args.out), "charted_layer": layer, "spec_type_ids": [int(t) for t in block.charted.spec.type_ids],
                  "hidden": int(cfg.hidden_size), "layers": int(cfg.num_layers), "loaded_tensors": load_receipt.get("tensors"),
                  "manifest_sha256": sha256_file(Path(args.out) / "composite_manifest.json")},
        "layout": {"prompt_tokens": plen, "latent_rows": n_lat, "time_row_sequence_position": t_pos, "sigma": float(args.sigma),
                   "time_row_position": list(TIME_ROW_POSITION), "target_kind": "velocity (per-row sigma -> bridge noise_proj)"},
        "structural_checks": {**checks, "verdict": "PASS: the time row is routed to, attended by and written by the hy3d block"},
        "diagnostics": {
            "latent_write_rel_change_time_row_owned_vs_unowned": change,
            "latent_write_rel_change_time_row_state_sigma_0.2_vs_0.8": rel(w_lo[0, lat], w_hi[0, lat]),
            "time_row_trunk_state_rel_change_sigma_0.2_vs_0.8": rel(lo_t["cap"]["x"][0, t_pos], hi_t["cap"]["x"][0, t_pos]),
            "final_hidden_latent_rows_rel_change_with_vs_without_time_row_in_sequence": rel(hidden_with, hidden_without),
            "note": "synthetic trunk: magnitudes say nothing about the real child; the end-to-end number mixes the trunk's own "
                    "causal attention to the extra row with the block's",
        },
        "seconds": time.time() - started,
    }


def part_b(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from safetensors import safe_open
    from chimera_omni.charted import ChartedBlockSpec

    c16 = _load("c16_install_media")
    unit = Path(args.unit)
    body = json.loads((unit / "charted_block.json").read_text())
    fields = dict(body["spec"])
    fields["type_ids"] = tuple(int(v) for v in fields["type_ids"])
    spec = ChartedBlockSpec(layer_index=0, **fields)
    spec.validate()
    tensors = {k: v.torch(dtype=torch.float32) for k, v in c16.read_single(unit / "charted_block.safetensors").items()}
    charts = {k: v.torch(dtype=torch.float32) for k, v in c16.read_single(unit / "charts.safetensors").items()}
    hidden = int(charts["chart_in.weight"].shape[1])
    block = c16.build_source_block(spec, hidden, tensors, charts).float()
    with safe_open(str(args.paired_rows), "pt") as h:
        keys = set(h.keys())
        if "time_row" not in keys:
            raise Refusal(f"{args.paired_rows}: no time_row array")
        time_row = h.get_tensor("time_row").to(torch.bool)
        held = h.get_tensor("held").to(torch.bool)
        forward = h.get_tensor("forward")
        token = h.get_tensor("token")
        candidates = sorted(set(int(f) for f in forward[held & time_row].tolist()))
        if not candidates:
            raise Refusal("no held forward with a time row")
        f = candidates[0]
        m = forward == f
        order = torch.argsort(token[m])
        host = h.get_tensor("host_rows")[m][order].float()
        parent = h.get_tensor("parent_rows")[m][order].float()
        tflag = time_row[m][order]
    if not bool(tflag[0]) or int(tflag.sum()) != 1:
        raise Refusal(f"forward {f}: the time row is not the single first row")
    n = int(host.shape[0])
    types = torch.full((1, n), VOLUME_TYPE, dtype=torch.long)
    types_wo = types.clone()
    types_wo[0, 0] = 0
    valid = torch.ones(1, n, dtype=torch.bool)
    started = time.time()
    with torch.no_grad():
        w_with = block(host[None], types, None, valid)
        w_wo = block(host[None], types_wo, None, valid)
        mapped_time = block.chart_in(host[0:1])[0]
    out: dict[str, Any] = {
        "unit": str(unit), "unit_json_sha256": sha256_file(unit / "charted_block.json"),
        "charts_sha256": sha256_file(unit / "charts.safetensors"),
        "paired_rows": str(args.paired_rows), "paired_rows_sha256": sha256_file(args.paired_rows),
        "held_forward": f, "rows": n,
        "latent_write_rel_change_time_row_owned_vs_unowned": rel(w_with[0, 1:], w_wo[0, 1:]),
        "time_row_write_norm": float(w_with[0, 0].norm()),
        "latent_write_norm": float(w_with[0, 1:].norm()),
        "chart_in_time_row_vs_parent_row0": {"cos": float(torch.nn.functional.cosine_similarity(mapped_time, parent[0], dim=0)),
                                             "rel_l2": rel(parent[0], mapped_time), "parent_row0_norm": float(parent[0].norm()),
                                             "mapped_norm": float(mapped_time.norm())},
    }
    if args.parent_capture:
        # source-space reference: the unit's source block on the parent's own block-10 inputs, with vs without row 0
        with safe_open(str(args.parent_capture), "pt") as h:
            bi = h.get_tensor("block_input").float()
        sample = None
        for s in range(int(bi.shape[0])):
            if tuple(bi[s].shape) == tuple(parent.shape) and torch.equal(bi[s], parent):
                sample = s
                break
        if sample is None:
            out["source_space_reference"] = {"checked": False, "reason": "the held forward's parent rows match no capture sample"}
        else:
            x = bi[sample][None]
            kv = torch.ones(1, int(x.shape[1]), dtype=torch.bool)
            kv_wo = kv.clone()
            kv_wo[0, 0] = False
            with torch.no_grad():
                y_with = block.block(x, torch.zeros(1, int(x.shape[1])), kv)
                y_wo = block.block(x[:, 1:], torch.zeros(1, int(x.shape[1]) - 1), kv[:, 1:])
            d_with = y_with[0, 1:] - x[0, 1:]
            d_wo = y_wo[0] - x[0, 1:]
            out["source_space_reference"] = {
                "checked": True, "capture": str(args.parent_capture), "sample": sample,
                "latent_delta_rel_change_with_vs_without_row0": rel(d_with, d_wo),
                "note": "the unit's own source block (4x MLP, no cross-attention) on the parent's block-10 input rows: how much "
                        "the parent's time token moves the latent rows' block delta in source space",
            }
    out["seconds"] = time.time() - started
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="c16 dry-run output (hidden 5120 synthetic child + hy3d unit)")
    p.add_argument("--unit", required=True, help="the hy3d_dit_L10 unit directory (real tensors + charts)")
    p.add_argument("--paired-rows", default=None)
    p.add_argument("--parent-capture", default=None)
    p.add_argument("--prompt-tokens", type=int, default=8)
    p.add_argument("--latent-rows", type=int, default=256)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", required=True)
    args = p.parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    report: dict[str, Any] = {
        "schema": "omni-campaign.s7_hy3d_timerow_check.v1",
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__)},
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": vars(args),
        "chimera_omni_src": os.environ.get("CHIMERA_OMNI_SRC"),
        "policy": "structural checks refuse; every magnitude is a diagnostic (owner 2026-10-06)",
    }
    code = 0
    try:
        report["part_a_synthetic_5120"] = part_a(args)
        if args.paired_rows:
            report["part_b_real_rows"] = part_b(args)
    except Refusal as error:
        report["refused"] = str(error)
        code = 3
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str), encoding="utf-8")
    Path(str(path) + ".sha256").write_text(sha256_file(path) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "exit": code, "refused": report.get("refused"),
                      "part_a": (report.get("part_a_synthetic_5120") or {}).get("diagnostics"),
                      "part_b": {k: v for k, v in (report.get("part_b_real_rows") or {}).items() if "rel" in k or "reference" in k or "chart_in" in k}},
                     default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
