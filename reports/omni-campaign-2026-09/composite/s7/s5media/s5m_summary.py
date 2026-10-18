#!/usr/bin/env python3
"""s5m_summary.py -- one receipt for the S7-on-S5 media lane: per unit host rows, chart diagnostics (S5 vs the S3
charts, both diagnostics), container files, sizes and sha256; plus the dry run and the S5 plan check.
usage: s5m_summary.py <s7 dir> <dryrun scratch> <run name> <out json>
"""
import hashlib
import json
import sys
import time
from pathlib import Path

UNITS = {"ace_dit_L12": "ace/ace_dit_L12", "h3_dit_L25_curve6": "h3/h3_dit_L25_curve6", "qimg_mmdit_L30_img": "qimg/qimg_mmdit_L30_img",
         "hy3d_dit_L10": "hy3d/hy3d_dit_L10", "qvl32_vit_L13": "qvl32/qvl32_vit_L13"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


def held(block):
    d = (block or {}).get("diagnostics_held") or {}
    return {k: {m: (d.get(k) or {}).get(m) for m in ("cos", "rel_l2", "rows")} for k in ("forward", "roundtrip")}


def main():
    s7, sc, run, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    rep = {"schema": "omni-campaign.s7_s5_media_lane.v1", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "policy": "owner 2026-10-06: chart numbers are diagnostics, never gates", "units": {}}
    for u, rel in UNITS.items():
        hr = json.loads((s7 / "tap_rows_s5" / u / "paired_rows.json").read_text())
        hr3 = json.loads((s7 / "tap_rows_s3" / u / "paired_rows.json").read_text())
        ud = s7 / "units_s5" / rel
        body = json.loads((ud / "charted_block.json").read_text())
        body3 = json.loads((s7 / "units" / rel / "charted_block.json").read_text())
        cr = json.loads((ud / "chart_receipt.json").read_text())
        files = {}
        for f in sorted(ud.iterdir()):
            files[f.name] = {"bytes": f.resolve().stat().st_size, "sha256": sha(f.resolve()), "symlink_to": str(f.resolve()) if f.is_symlink() else None}
        tr = (body["charts"].get("time_rows") or {})
        rep["units"][u] = {
            "host_rows": {"file": hr["file"], "sha256": hr["file_sha256"], "json_sha256": sha(s7 / "tap_rows_s5" / u / "paired_rows.json"),
                          "tool": hr["tool"]["version"], "rows": hr["rows"], "rows_held": hr["rows_held"], "rows_time": hr.get("rows_time"),
                          "forwards": hr["forwards"], "batches": hr["batches"], "seconds": hr["seconds"], "cuda_peak_gib": hr["cuda_peak_gib"],
                          "trunk": hr["trunk"]["directory"], "block_residency": hr["trunk"].get("block_residency"),
                          "pairs_sha256": hr["pairs"]["sha256"], "same_pairs_as_s3": hr["pairs"]["sha256"] == hr3["pairs"]["sha256"],
                          "same_rows_as_s3": hr["rows"] == hr3["rows"] and hr["rows_held"] == hr3["rows_held"],
                          "interface_companions_sha256": hr["interface"]["companions_sha256"],
                          "same_interface_as_s3": hr["interface"]["companions_sha256"] == hr3["interface"]["companions_sha256"],
                          "causal_prompt_check": hr["diagnostics"]["causal_prompt_check"],
                          "host_row_norm_median": hr["diagnostics"]["host_row_norm_median"],
                          "s3_host_row_norm_median": hr3["diagnostics"]["host_row_norm_median"]},
            "charts_s5": {"sha256": body["charts"]["sha256"], "ridge_rel": body["charts"]["ridge_rel"], "held": held(body["charts"]),
                          "time_rows": None if not tr else {"rows": tr.get("rows"), "rows_held": tr.get("rows_held"), "weight": tr.get("weight"),
                                                              "held_forward_time": {k: (tr.get("held_forward_time") or {}).get(k) for k in ("cos", "rel_l2")},
                                                              "held_forward_content": {k: (tr.get("held_forward_content") or {}).get(k) for k in ("cos", "rel_l2")}},
                          "producer": body["charts"]["producer"], "trunk_label": body["charts"].get("trunk_label"),
                          "valid_only_with": body["charts"]["valid_only_with"], "type_ids": body["spec"]["type_ids"],
                          "chart_receipt_sha256": sha(ud / "chart_receipt.json"), "fit_seconds": cr.get("seconds")},
            "charts_s3_for_comparison": {"sha256": body3["charts"]["sha256"], "ridge_rel": body3["charts"]["ridge_rel"], "held": held(body3["charts"])},
            "unit_dir": str(ud), "files": files, "unit_bytes": sum(v["bytes"] for v in files.values()),
            "installed_parameters": (body["s7_unit"].get("canonical") or {}).get("installed_parameters_at_hidden_5120"),
            "component_4x": body["s7_unit"].get("component_4x"),
        }
    chk = json.loads((sc / f"check_{run}.json").read_text())
    man = json.loads((sc / f"out_{run}" / "composite_manifest.json").read_text())
    plan = json.loads((sc / f"plan_check_{run}.json").read_text())
    rep["dryrun_tiny_5120"] = {"manifest": str(sc / f"out_{run}" / "composite_manifest.json"), "manifest_sha256": sha(sc / f"out_{run}" / "composite_manifest.json"),
                               "check": {k: chk.get(k) for k in ("loaded_tensors", "parameters", "child_parameters", "layers", "charted_layers",
                                                                  "text_hidden_bit_identical", "text_logits_finite")},
                               "added_parameters": chk["parameters"] - chk["child_parameters"],
                               "components": [{k: c.get(k) for k in ("component", "parent_params", "child_params", "ratio", "meets_4x")} for c in man.get("components", [])],
                               "interface_entries": [(e["name"], e["type_id"], e["feature_band"], e["emission_band"]) for e in (man.get("interface") or {}).get("entries", [])],
                               "band": (man.get("band") or {}).get("band"), "warnings": man.get("warnings"),
                               "charted_dtype": (man.get("blocks") or {}).get("charted_dtype")}
    rep["plan_check_s5_child"] = {"file": str(sc / f"plan_check_{run}.json"), "sha256": sha(sc / f"plan_check_{run}.json"), "verdict": plan["verdict"],
                                  "refusals": plan["refusals"], "accounting": plan["accounting"],
                                  "units": {x["name"]: {"insert_after": x["insert_after"], "params": x["block_parameters_at_hidden"],
                                                        "forward": x["forward_on_s5_host_rows"], "time_reference": x["time_reference"]} for x in plan["units"]},
                                  "band": {k: plan["band"][k] for k in ("band", "s5_embed_band_nonzero", "s5_lm_head_band_nonzero", "table_finite")},
                                  "interface_preconditions": {e["name"]: e["preconditions_nonzero"] for e in plan["interface"]["entries"]}}
    out.write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    Path(str(out) + ".sha256").write_text(sha(out) + "\n")
    for u, r in rep["units"].items():
        h, c, c3 = r["host_rows"], r["charts_s5"], r["charts_s3_for_comparison"]
        print(f"{u}: rows {h['rows']} held {h['rows_held']} time {h['rows_time']} | S5 held fwd cos {c['held']['forward']['cos']:.4f} relL2 {c['held']['forward']['rel_l2']:.4f}"
              f" rt cos {c['held']['roundtrip']['cos']:.4f} relL2 {c['held']['roundtrip']['rel_l2']:.4f} | S3 fwd cos {c3['held']['forward']['cos']:.4f} relL2 {c3['held']['forward']['rel_l2']:.4f}"
              f" | charts {c['sha256'][:16]} | unit bytes {r['unit_bytes']} | params {r['installed_parameters']}")
    print("dryrun", json.dumps(rep["dryrun_tiny_5120"]["check"]), "added", rep["dryrun_tiny_5120"]["added_parameters"])
    print("plan", rep["plan_check_s5_child"]["verdict"], json.dumps(rep["plan_check_s5_child"]["accounting"]))
    print("summary", out, sha(out))


if __name__ == "__main__":
    main()
