#!/usr/bin/env python3
"""PREP-B close-out: update S7_UNITS_GAP_TABLE.json (HY3D entry + a prepb_update block). Server-side, stdlib only."""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

OC = Path("/home/cereal/omni-campaign")
S7 = OC / "composite/s7"
PB = S7 / "prep_b"
TABLE = S7 / "units/S7_UNITS_GAP_TABLE.json"
UNIT = S7 / "units/hy3d/hy3d_dit_L10"
OLD = S7 / "units/hy3d/hy3d_dit_L10.pre-timerow"
TOOLS = OC / "composite/tools"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def main():
    prev_sha = sha(TABLE)
    backup = TABLE.with_name(TABLE.name + ".bak-prepb-20261011")
    if not backup.exists():
        shutil.copy2(TABLE, backup)
    t = json.loads(TABLE.read_text())
    body = json.loads((UNIT / "charted_block.json").read_text())
    rec = json.loads((UNIT / "chart_receipt.json").read_text())
    sweep = json.loads((PB / "time_row_weight_sweep.json").read_text())
    chk = json.loads((PB / "timerow_check.json").read_text())
    dry = json.loads((PB / "scratch_dryrun5120/check_PBT.json").read_text())
    pairs = json.loads((S7 / "pairs/hy3d_dit_L10/parent_capture.json").read_text())
    rows = json.loads((S7 / "tap_rows_s3/hy3d_dit_L10/paired_rows.json").read_text())
    td = rec["diagnostics"]["time_rows"]
    held = rec["diagnostics"]["ridge_split_held"]
    a = chk["part_a_synthetic_5120"]
    b = chk.get("part_b_real_rows") or {}
    entry = next(u for u in t["units"] if u.get("unit") == "hy3d/hy3d_dit_L10")
    entry["charts_5120"] = body["s7_unit"]["charts_status"]
    entry["status"] = ("installable; charts refitted WITH the time-token row (PREP-B 2026-10-11): pairs carry the parent block-10 "
                       "input row 0, host rows emit the type-7 time row first in each volume segment, the charts fit it; "
                       "cross-attention dropped (D3 design, S7_DECISIONS_2026-10-10.md)")
    entry["owner_next"] = "Claude (S7 step 5: recapture host rows on the S5 prefix WITH the time row, then refit at weight 4096)"
    entry["smallest_next_step"] = ("S7 step 5 on the S5 child: s7_host_rows.py >= 1.2.0 (emits the time row) + s7_chart_fit.py >= 1.2.0 "
                                   "(default time-row weight 4096); the S7 driver must emit the time row per volume segment")
    entry["blocker_prepb"] = ("FIXED 2026-10-11: s7_pairs_parent 1.1.0 keeps block_input[:, 0:] (row 0 = time token, time_row=True); "
                              "s7_host_rows 1.2.0 emits the time row (zero features, position [-1,-1,-1,-1], segment sigma, "
                              "first in each forward) and refuses a unit that requires it when it is absent; s7_chart_fit "
                              "1.2.0 refuses time-row-less rows for a unit that declares hy3d_time_token_row")
    entry["unit_dir_pre_timerow"] = str(OLD)
    entry["prepb"] = {
        "unit_json_sha256": sha(UNIT / "charted_block.json"),
        "charted_block_safetensors_sha256": sha(UNIT / "charted_block.safetensors"),
        "charted_block_safetensors_identical_to_pre_timerow": sha(UNIT / "charted_block.safetensors") == sha(OLD / "charted_block.safetensors"),
        "charts_sha256": sha(UNIT / "charts.safetensors"),
        "chart_receipt_sha256": sha(UNIT / "chart_receipt.json"),
        "pre_timerow_charts_sha256": sha(OLD / "charts.safetensors"),
        "pairs": {"file": str(S7 / "pairs/hy3d_dit_L10/parent_capture.safetensors"), "sha256": sha(S7 / "pairs/hy3d_dit_L10/parent_capture.safetensors"),
                  "rows": pairs["rows"], "rows_time": pairs["rows_time"], "tool": pairs["tool"]},
        "host_rows": {"file": str(S7 / "tap_rows_s3/hy3d_dit_L10/paired_rows.safetensors"),
                      "sha256": sha(S7 / "tap_rows_s3/hy3d_dit_L10/paired_rows.safetensors"), "rows": rows["rows"],
                      "rows_time": rows["rows_time"], "tool": rows["tool"], "trunk": "S3 native prefix 0..31 (GPU4, 937 s)"},
        "time_row_weight": {"used": td["weight"], "sweep": str(PB / "time_row_weight_sweep.json"),
                            "sweep_sha256": sha(PB / "time_row_weight_sweep.json"), "chosen": sweep["chosen_weight"],
                            "rule": sweep["rule"]},
        "diagnostics_not_gates": {
            "held_forward_all": held["forward"], "held_roundtrip_all": held["roundtrip"],
            "held_forward_time_rows": {k: td["split_ridge_held_forward"]["time"][k] for k in ("cos", "rel_l2", "rows")},
            "held_forward_content_rows": {k: td["split_ridge_held_forward"]["content"][k] for k in ("cos", "rel_l2", "rows")},
        },
    }
    t["prepb_update"] = {
        "author": "Claude PREP-B, 2026-10-10/11",
        "previous_table_sha256": prev_sha,
        "scope": "HY3D time-token row (S7 readiness 1/D3): pairs, host rows, chart refit, unit rebuild, dry run",
        "tools": {n: {"sha256": sha(TOOLS / n)} for n in ("s7_pairs_parent.py", "s7_host_rows.py", "s7_chart_fit.py",
                                                          "s7_unit_build.py", "s7_hy3d_timerow_check.py", "test_s7_media_charts.py",
                                                          "test_s7_unit_build.py")},
        "dry_run": {
            "run": "PBT: s7_dryrun.py --hidden 5120 --rows 1024 --build-child --charted-dtype match (tiny synthetic child at hidden 5120, NOT the S3 trunk), unit hy3d_dit_L10 only, real charts",
            "check": {k: dry.get(k) for k in ("loaded_tensors", "parameters", "child_parameters", "charted_layers",
                                              "text_hidden_bit_identical", "text_logits_finite")},
            "manifest_sha256": sha(PB / "scratch_dryrun5120/out_PBT/composite_manifest.json"),
            "timerow_check": {"report": str(PB / "timerow_check.json"), "sha256": sha(PB / "timerow_check.json"),
                              "structural_verdict": a["structural_checks"]["verdict"],
                              "part_a_diagnostics": a["diagnostics"],
                              "part_b_real_rows": {k: v for k, v in b.items() if k not in ("unit", "paired_rows")}},
        },
        "policy": "owner 2026-10-06: every fit / change magnitude above is a diagnostic; structural checks refuse",
        "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = TABLE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(t, indent=2) + "\n")
    os.replace(tmp, TABLE)
    print(json.dumps({"table": str(TABLE), "sha256": sha(TABLE), "previous": prev_sha, "backup": str(backup)}))


if __name__ == "__main__":
    sys.exit(main())
