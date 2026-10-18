#!/usr/bin/env python3
"""s7_chart_fit.py -- R4 of the media-chart pairing: fit the 5120-wide C_in / C_out charts of one S7 unit.

Lane M1, Claude, 2026-10-10. Reads `<tap-root>/<unit>/paired_rows.safetensors` (`s7_host_rows.py`:
host_rows = S3 trunk x_31 at the media rows, parent_rows = the source block input for the same token,
held = the group-split holdout) and writes, INTO the unit container directory, exactly what
`c16_install_media.load_container` consumes:

    <unit dir>/charts.safetensors      chart_in.weight [W, 5120], chart_in.bias [W], chart_out.weight [5120, W]  (fp32)
    <unit dir>/charted_block.json      tensor_sha256["charts.safetensors"] added (safetensors_io TensorView.sha256
                                       formula, verified by c16 on load) + a `charts` provenance block;
                                       the pre-M1 json is kept once as charted_block.json.pre-m1
    <unit dir>/chart_receipt.json      (+ .sha256) every arm's numbers -- DIAGNOSTICS, never gates

Arms (c17_chart_fit.fit_layer_charts, imported): ridge (lambda ladder on an inner split of the FIT rows,
refit on every fit row), and the svd / procrustes controls. All three are fitted on the fit groups and
scored on the HELD groups (prompt / clip / image split, never a row split). The chart of RECORD is the
ridge chart REFITTED ON EVERY ROW (fit + held) at the lambdas the fit-split ladder chose, because the
pair sets are small (HY3D has 4 objects); its numbers are in-sample and labelled so.

c16 drops chart_out.bias (a zero delta must write nothing), so none is written. `--type-ids` rewrites the
unit's spec type_ids (R0(b): QVL32 [1, 6] -> [6], recorded with the old value).

1.1.0 (PREP-B, Claude, 2026-10-10): TIME ROWS. paired_rows from s7_host_rows >= 1.2.0 carry `time_row`.
A unit whose charted_block.json declares runtime_requirements.hy3d_time_token_row is REFUSED rows without
time rows (structural: the installed block must see the timestep the parent's block saw). The time rows
are ordinary rows of both charts; because there is ONE per forward (4 of 16,388 for HY3D) each FIT time
row is replicated `--time-row-weight` times in the fit design (= a row weight in the normal equations;
held rows are never replicated, so held numbers stay one row per token). The weight is a DESIGN
parameter: DEFAULT_TIME_ROW_WEIGHT was chosen from the PREP-B sweep (receipt in the unit's chart
receipt `time_rows.weight_choice`). `--arms` restricts the arms (sweeps run ridge only). Per-kind held
diagnostics (time rows vs content rows) are recorded; every number stays a diagnostic.

1.2.0 (PREP-B close-out, Claude, 2026-10-11): DEFAULT_TIME_ROW_WEIGHT = 4096, the weight the PREP-B
sweep chose (`composite/s7/prep_b/time_row_weight_sweep.json`, rule fixed before the sweep ran: lowest
held time-row forward relL2 among weights whose held content-row forward relL2 stays within +1% of the
weight-1 value; swept 1 / 64 / 512 / 4096). When the rows carry time rows the receipt and the unit json
record `time_rows.weight_choice` (sweep file, its sha256, the chosen weight, whether the weight used
equals it) -- a provenance record, never a refusal. `--lane` labels the producer. charts.pairs now
binds `parent_pairs_sha256` (from the host-row receipt) and `paired_rows_json_sha256` (S7 readiness 3.3).

1.3.0 (S7-on-S5 media lane, Claude, 2026-10-13): `--trunk-label` (default "S3"). The provenance text
(`charts.valid_only_with`, `s7_unit.charts_status`) named the S3 trunk unconditionally; with a label other
than S3 it names the labelled trunk and the host-row receipt's trunk directory, and `charts.trunk_label`
is recorded. The default output is unchanged key for key; no fit arithmetic changed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "s7_chart_fit"
TOOL_VERSION = "1.3.0"
TIME_ROW_REQUIREMENT = "hy3d_time_token_row"
DEFAULT_TIME_ROW_WEIGHT = 4096  # chosen by the PREP-B sweep; see time_rows.weight_choice in the HY3D chart receipt
OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
DEFAULT_WEIGHT_CHOICE = OC / "composite/s7/prep_b/time_row_weight_sweep.json"
HERE = Path(__file__).resolve().parent
UNIT_DIRS = {
    "ace_dit_L12": "ace/ace_dit_L12",
    "h3_dit_L25_curve6": "h3/h3_dit_L25_curve6",
    "qimg_mmdit_L30_img": "qimg/qimg_mmdit_L30_img",
    "hy3d_dit_L10": "hy3d/hy3d_dit_L10",
    "qvl32_vit_L13": "qvl32/qvl32_vit_L13",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_c17():
    spec = importlib.util.spec_from_file_location("c17_chart_fit_m1", HERE / "c17_chart_fit.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["c17_chart_fit_m1"] = module
    spec.loader.exec_module(module)
    return module


def charts_from(c_in: Any, b_in: Any, c_out: Any) -> dict[str, Any]:
    """c17 convention (y = x C_in + b_in; x = y C_out + b_out) -> c16 chart tensors (Linear weights [out, in])."""

    import torch

    return {
        "chart_in.weight": c_in.transpose(0, 1).to(torch.float32).contiguous(),
        "chart_in.bias": b_in.to(torch.float32).contiguous(),
        "chart_out.weight": c_out.transpose(0, 1).to(torch.float32).contiguous(),
    }


def scores_table(entry: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = entry.get(key)
    if not block:
        return None
    return {d: {m: block[d][m] for m in ("cos", "rel_l2", "cos_p05", "r2", "rows") if m in block[d]} for d in ("forward", "backward", "roundtrip")}


def replicate_rows(base: Any, extra: Any, weight: int) -> Any:
    """Row index list: every `base` row once, plus (weight - 1) more copies of each `extra` row (bool masks)."""

    import torch

    index = torch.nonzero(base).flatten()
    if int(weight) <= 1 or not bool(extra.any()):
        return index
    copies = torch.nonzero(extra).flatten().repeat(int(weight) - 1)
    return torch.cat((index, copies))


def kind_scores(linalg: Any, pred: Any, target: Any, mask: Any) -> dict[str, Any] | None:
    import torch

    if int(mask.sum()) == 0:
        return None
    s = linalg.write_scores(pred[mask], target[mask])
    s["rows"] = int(mask.sum())
    s["row_rel_l2"] = [float(v) for v in ((pred[mask] - target[mask]).norm(dim=1) / target[mask].norm(dim=1).clamp_min(1e-30))][:8]
    s["row_cos"] = [float(v) for v in torch.nn.functional.cosine_similarity(pred[mask], target[mask], dim=1)][:8]
    return s


def weight_choice_record(path: str | None, weight_used: int) -> dict[str, Any]:
    """Provenance of the time-row weight: the sweep receipt that chose it. A record, never a refusal."""

    if not path:
        return {"file": None, "weight_used": int(weight_used), "note": "no sweep receipt given (--weight-choice '')"}
    p = Path(path)
    if not p.is_file():
        return {"file": str(p), "present": False, "weight_used": int(weight_used), "note": "sweep receipt not found"}
    sweep = json.loads(p.read_text(encoding="utf-8"))
    chosen = sweep.get("chosen_weight")
    return {"file": str(p), "present": True, "sha256": sha256_file(p), "schema": sweep.get("schema"),
            "rule": sweep.get("rule"), "weights_swept": [r.get("weight") for r in sweep.get("sweep", [])],
            "chosen_weight": chosen, "weight_used": int(weight_used),
            "weight_used_equals_choice": chosen is not None and int(chosen) == int(weight_used),
            "policy": "design parameter chosen from measurements (owner 2026-10-06); recorded, never a gate"}


def tensor_sha256_table(path: Path, chimera_src: str) -> dict[str, str]:
    src = Path(chimera_src) / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from chimera_omni.importing.safetensors_io import read_safetensors

    return {name: str(view.sha256()) for name, view in sorted(read_safetensors(path, verify=False).items())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--unit", required=True)
    p.add_argument("--unit-dir", default=None, help="override the container directory (tests)")
    p.add_argument("--hidden", type=int, default=5120)
    p.add_argument("--tap-rows", default=str(OC / "composite/s7/tap_rows_s3"))
    p.add_argument("--units-root", default=str(OC / "composite/s7/units"))
    p.add_argument("--chimera-src", default=str(OC / "import/rq27b/tree_ef1ffb9"))
    p.add_argument("--type-ids", default=None, help="comma list; rewrite spec.type_ids (R0(b))")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ladder-rows", type=int, default=0)
    p.add_argument("--bridge-diagnostic", action="store_true", default=True)
    p.add_argument("--time-row-weight", type=int, default=None,
                   help=f"fit-design replication of each time row (default {DEFAULT_TIME_ROW_WEIGHT} when the rows carry time rows)")
    p.add_argument("--arms", default="ridge,svd,procrustes", help="comma list; ridge is required (the chart of record)")
    p.add_argument("--weight-choice", default=str(DEFAULT_WEIGHT_CHOICE),
                   help="sweep receipt that chose the time-row weight (recorded when the rows carry time rows; '' = none)")
    p.add_argument("--lane", default="M1 2026-10-10", help="producer label recorded in the receipt")
    p.add_argument("--trunk-label", default="S3", help="the trunk the host rows were captured on (provenance text; 1.3.0)")
    args = p.parse_args(argv)

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    c17 = load_c17()
    linalg = c17.linalg
    unit_dir = Path(args.unit_dir) if args.unit_dir else Path(args.units_root) / UNIT_DIRS[args.unit]
    body_path = unit_dir / "charted_block.json"
    body = json.loads(body_path.read_text())
    width = int(body["spec"]["width"])
    rows_path = Path(args.tap_rows) / args.unit / "paired_rows.safetensors"
    rows_json_path = rows_path.parent / "paired_rows.json"
    started = time.time()
    with safe_open(str(rows_path), "pt") as h:
        host = h.get_tensor("host_rows").to(torch.float64)
        parent = h.get_tensor("parent_rows").to(torch.float64)
        held = h.get_tensor("held").to(torch.bool)
        group = h.get_tensor("group")
        types = h.get_tensor("type_ids")
        bridge = h.get_tensor("bridge_rows_sample").to(torch.float64) if "bridge_rows_sample" in h.keys() else None
        bridge_index = h.get_tensor("bridge_rows_index") if "bridge_rows_index" in h.keys() else None
        time_row = h.get_tensor("time_row").to(torch.bool) if "time_row" in h.keys() else torch.zeros(host.shape[0], dtype=torch.bool)
    requirement = (body.get("runtime_requirements") or {}).get(TIME_ROW_REQUIREMENT)
    if requirement is not None and not bool(time_row.any()):
        raise SystemExit(f"{body_path}: the unit declares runtime_requirements.{TIME_ROW_REQUIREMENT} but {rows_path} carries no time "
                         "row; the installed block would never see the timestep (rebuild pairs with s7_pairs_parent >= 1.1.0 and host "
                         "rows with s7_host_rows >= 1.2.0)")
    if bool(time_row.any()) and requirement is None:
        raise SystemExit(f"{rows_path} carries time rows but {body_path} declares no {TIME_ROW_REQUIREMENT}")
    arm_names = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    if "ridge" not in arm_names:
        raise SystemExit("--arms must include ridge (the chart of record)")
    weight = int(args.time_row_weight if args.time_row_weight is not None else DEFAULT_TIME_ROW_WEIGHT)
    if weight < 1:
        raise SystemExit("--time-row-weight must be >= 1")
    if host.shape[1] != int(args.hidden) or parent.shape[1] != width:
        raise SystemExit(f"{rows_path}: host [{host.shape[1]}] / parent [{parent.shape[1]}] vs {args.hidden} / {width}")
    for name, t in (("host", host), ("parent", parent)):
        if not bool(torch.isfinite(t).all()):
            raise SystemExit(f"{rows_path}: non-finite {name} rows")
    if bool(torch.isin(torch.unique(group[held]), torch.unique(group[~held])).any()):
        raise SystemExit("held and fit rows share a group")
    fit_mask = ~held
    n_rows = int(host.shape[0])
    log(f"{args.unit}: rows {n_rows} (fit {int(fit_mask.sum())}, held {int(held.sum())}, time {int(time_row.sum())} x weight {weight}), parent width {width}")
    # split fit: FIT time rows replicated (weight), held rows once
    split_index = torch.cat((torch.arange(n_rows), replicate_rows(torch.zeros(n_rows, dtype=torch.bool), fit_mask & time_row, weight)))
    extra = int(split_index.numel()) - n_rows
    split_fit = torch.cat((fit_mask, torch.ones(extra, dtype=torch.bool)))
    split_held = torch.cat((held, torch.zeros(extra, dtype=torch.bool)))
    arms = c17.fit_layer_charts(host[split_index], parent[split_index], split_fit, split_held, arms=arm_names, seed=args.seed,
                                ladder=True, fit_bias=True, ladder_rows=int(args.ladder_rows))
    log(f"{args.unit}: split arms {arm_names} fitted ({time.time() - started:.0f}s)")
    ridge = arms["ridge"]
    rung_in = float(ridge["notes"]["in"]["ridge_rel_selected"])
    rung_out = float(ridge["notes"]["out"]["ridge_rel_selected"])
    # chart of record: ridge refit on EVERY row at the selected rungs (every time row replicated by the same weight)
    record_index = replicate_rows(torch.ones(n_rows, dtype=torch.bool), time_row, weight)
    fwd = linalg.fit_ridge_linear(host[record_index], parent[record_index], ridge_rel=rung_in, fit_bias=True, holdout_fraction=0.0, seed=args.seed)
    bwd = linalg.fit_ridge_linear(parent[record_index], host[record_index], ridge_rel=rung_out, fit_bias=True, holdout_fraction=0.0, seed=args.seed)
    record = charts_from(fwd.weight.transpose(0, 1), fwd.bias, bwd.weight.transpose(0, 1))
    in_sample = {
        "forward": linalg.write_scores(host @ fwd.weight.T + fwd.bias, parent),
        "backward": linalg.write_scores(parent @ bwd.weight.T + bwd.bias, host),
        "roundtrip": linalg.write_scores((host @ fwd.weight.T + fwd.bias) @ bwd.weight.T + bwd.bias, host),
        "label": "IN-SAMPLE (every row fitted); the held numbers of the split ridge arm are the honest ones",
    }
    split_record = charts_from(ridge["C_in"], ridge["C_in_bias"], ridge["C_out"])
    per_type = {}
    if len(torch.unique(types)) > 1:
        for t in torch.unique(types).tolist():
            m = held & (types == t)
            if int(m.sum()) == 0:
                continue
            pred = host[m] @ ridge["C_in"] + ridge["C_in_bias"]
            per_type[str(int(t))] = linalg.write_scores(pred, parent[m])
    time_diag = None
    if bool(time_row.any()):
        split_fwd = host @ ridge["C_in"] + ridge["C_in_bias"]
        split_bwd = parent @ ridge["C_out"] + ridge["C_out_bias"]
        rec_fwd = host @ fwd.weight.T + fwd.bias
        time_diag = {
            "rows": int(time_row.sum()), "rows_held": int((time_row & held).sum()), "weight": weight,
            "split_ridge_held_forward": {"time": kind_scores(linalg, split_fwd, parent, held & time_row),
                                         "content": kind_scores(linalg, split_fwd, parent, held & ~time_row)},
            "split_ridge_held_backward": {"time": kind_scores(linalg, split_bwd, host, held & time_row),
                                          "content": kind_scores(linalg, split_bwd, host, held & ~time_row)},
            "split_ridge_fit_forward_time": kind_scores(linalg, split_fwd, parent, fit_mask & time_row),
            "record_in_sample_forward": {"time": kind_scores(linalg, rec_fwd, parent, time_row),
                                         "content": kind_scores(linalg, rec_fwd, parent, ~time_row)},
            "norms": {"host_time": [float(v) for v in host[time_row].norm(dim=1)], "parent_time": [float(v) for v in parent[time_row].norm(dim=1)],
                      "host_content_median": float(host[~time_row].norm(dim=1).median()),
                      "parent_content_median": float(parent[~time_row].norm(dim=1).median())},
            "label": "diagnostics only; 'time' = the parent's evolved time token rows, 'content' = latent rows",
            "weight_choice": weight_choice_record(args.weight_choice, weight),
        }
    bridge_diag = None
    if bridge is not None and bridge.shape[0] >= 64:
        target = host[bridge_index]
        lm = linalg.fit_ridge_lambda_ladder(bridge, target, fit_bias=True, holdout_fraction=0.2, seed=args.seed)
        bridge_diag = {"rows": int(bridge.shape[0]), "design_width": int(bridge.shape[1]) + 1,
                       "regime": "underdetermined" if int(0.64 * bridge.shape[0]) < bridge.shape[1] + 1 else "overdetermined",
                       "held_scores_bridge_row_to_x31": lm.holdout_scores, "ridge_rel": lm.notes.get("ridge_rel"),
                       "cos_bridge_vs_x31_median": float(torch.nn.functional.cosine_similarity(bridge, target, dim=1).median()),
                       "reading": "how much of the trunk's x_31 at a media row is a linear function of the bridge row alone (row-split, small sample)"}
    # write charts into the unit dir
    charts_path = unit_dir / "charts.safetensors"
    if charts_path.is_symlink():
        raise SystemExit(f"{charts_path} is a symlink; refusing to write through it")
    backup = unit_dir / "charted_block.json.pre-m1"
    if not backup.exists():
        shutil.copy2(body_path, backup)
    tmp = charts_path.with_suffix(".tmp")
    save_file(record, str(tmp))
    os.replace(tmp, charts_path)
    digests = tensor_sha256_table(charts_path, args.chimera_src)
    split_path = unit_dir / "charts_split_ridge.diagnostic.safetensors"
    save_file(split_record, str(split_path))
    old_types = list(body["spec"]["type_ids"])
    if args.type_ids:
        body["spec"]["type_ids"] = [int(v) for v in args.type_ids.split(",")]
    rows_receipt = json.loads(rows_json_path.read_text()) if rows_json_path.is_file() else {}
    diag = {
        "ridge_split_held": scores_table(ridge, "held"),
        "svd_control_held": scores_table(arms.get("svd", {}), "held"),
        "procrustes_control_held": scores_table(arms.get("procrustes", {}), "held"),
        "ridge_split_fit": scores_table(ridge, "fit"),
        "record_in_sample": in_sample,
        "per_type_held_forward": per_type or None,
        "time_rows": time_diag,
        "bridge_row_predicts_x31": bridge_diag,
    }
    charts_block = {
        "file": "charts.safetensors",
        "sha256": sha256_file(charts_path),
        "hidden": int(args.hidden),
        "shapes": {k: list(v.shape) for k, v in record.items()},
        "method": "ridge (mediafold.linalg.fit_ridge_linear, fp64) refitted on every paired row at the lambdas chosen by the c17 fit-split ladder",
        "ridge_rel": {"chart_in": rung_in, "chart_out": rung_out},
        "chart_out_bias": "not written (c16 never installs it)",
        "pairs": {"paired_rows": str(rows_path), "paired_rows_sha256": sha256_file(rows_path),
                  "rows": int(host.shape[0]), "rows_held": int(held.sum()), "holdout": "group split (see the pair receipt)",
                  "host_site": (rows_receipt.get("trunk") or {}).get("host_site"),
                  "trunk": (rows_receipt.get("trunk") or {}).get("directory"),
                  "interface": (rows_receipt.get("interface") or {}).get("companions"),
                  "parent_pairs": (rows_receipt.get("pairs") or {}).get("file"),
                  "parent_pairs_sha256": (rows_receipt.get("pairs") or {}).get("sha256"),
                  "paired_rows_json_sha256": sha256_file(rows_json_path) if rows_json_path.is_file() else None},
        "diagnostics_held": {"forward": (diag["ridge_split_held"] or {}).get("forward"), "roundtrip": (diag["ridge_split_held"] or {}).get("roundtrip")},
        "arms": list(arm_names),
        "time_rows": None if time_diag is None else {
            "rows": time_diag["rows"], "rows_held": time_diag["rows_held"], "weight": weight,
            "requirement": TIME_ROW_REQUIREMENT,
            "how": "the time rows are ordinary rows of both charts; each is replicated `weight` times in the fit design "
                   "(split fit: fit time rows only; record: every time row)",
            "held_forward_time": time_diag["split_ridge_held_forward"]["time"],
            "held_forward_content": time_diag["split_ridge_held_forward"]["content"],
            "weight_choice": time_diag["weight_choice"],
        },
        "valid_only_with": ("the latent interface rows these host rows were captured under (s7/interface_latent) and the S3 trunk prefix 0..31; S5 (C10 rewrites FFNs 0-31) needs a recapture"
                            if args.trunk_label == "S3" else
                            f"the latent interface rows these host rows were captured under (s7/interface_latent) and the {args.trunk_label} trunk prefix 0..31 "
                            f"({(rows_receipt.get('trunk') or {}).get('directory')}); any other trunk prefix needs a recapture"),
        "producer": {"tool": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__), "lane": args.lane},
        "policy": "owner 2026-10-06: these numbers are diagnostics, never gates",
    }
    if args.trunk_label != "S3":
        charts_block["trunk_label"] = args.trunk_label
    if args.type_ids:
        charts_block["type_ids_rewrite"] = {"from": old_types, "to": body["spec"]["type_ids"],
                                            "why": "R0(b): QVL32 [1, 6] collides with QIMG image type 1; the charts were fitted on type-6 (ocr) rows only"}
    body.setdefault("tensor_sha256", {})["charts.safetensors"] = digests
    body["charts"] = charts_block
    s7 = body.setdefault("s7_unit", {})
    s7["charts_status"] = (f"PRESENT at hidden {int(args.hidden)} (lane {args.lane}, {time.strftime('%Y-%m-%d')}); fitted on {args.trunk_label}-trunk host rows under the latent interface set"
                           + ("" if time_diag is None else f"; time rows {time_diag['rows']} x weight {weight} (runtime_requirements.{TIME_ROW_REQUIREMENT})"))
    tmp_json = body_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(body, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp_json, body_path)
    receipt = {
        "schema": "omni-campaign.s7_chart_receipt.v1",
        "unit": args.unit, "unit_dir": str(unit_dir),
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": sha256_file(__file__),
                 "c17_chart_fit_sha256": sha256_file(HERE / "c17_chart_fit.py")},
        "charts": charts_block, "tensor_sha256": digests,
        "split_ridge_file": {"file": str(split_path), "sha256": sha256_file(split_path), "note": "fit-split ridge chart, diagnostic only"},
        "charted_block_json": {"file": str(body_path), "sha256": sha256_file(body_path), "pre_m1_backup": str(backup), "pre_m1_sha256": sha256_file(backup)},
        "arms_notes": {k: {kk: vv for kk, vv in v["notes"].items() if kk not in ("in", "out")} for k, v in arms.items()},
        "ridge_notes": {"in": {k: v for k, v in ridge["notes"]["in"].items()}, "out": {k: v for k, v in ridge["notes"]["out"].items()}},
        "diagnostics": diag,
        "seconds": time.time() - started,
    }
    rpath = unit_dir / "chart_receipt.json"
    rpath.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (unit_dir / "chart_receipt.json.sha256").write_text(sha256_file(rpath) + "\n", encoding="utf-8")
    h = diag["ridge_split_held"] or {}
    log(f"{args.unit}: charts written {charts_path} sha {charts_block['sha256'][:16]}; held forward cos "
        f"{(h.get('forward') or {}).get('cos')} relL2 {(h.get('forward') or {}).get('rel_l2')}; roundtrip cos {(h.get('roundtrip') or {}).get('cos')} "
        f"({time.time() - started:.0f}s)")
    if time_diag is not None:
        ht, hc = time_diag["split_ridge_held_forward"]["time"] or {}, time_diag["split_ridge_held_forward"]["content"] or {}
        log(f"{args.unit}: time rows {time_diag['rows']} (held {time_diag['rows_held']}) weight {weight}: held forward time "
            f"cos {ht.get('cos')} relL2 {ht.get('rel_l2')}; content cos {hc.get('cos')} relL2 {hc.get('rel_l2')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
