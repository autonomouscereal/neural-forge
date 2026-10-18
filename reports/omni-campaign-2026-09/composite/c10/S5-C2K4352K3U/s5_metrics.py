#!/usr/bin/env python3
"""S5 metrics in the D4 sweep's row format (design_sweep/build_results.py formulas, verbatim), 2026-10-11 Claude.

Reads one c10_virtual_child_check.py --variant receipt (the built child as a virtual child beside the S3
trunk and a reference) and the per-layer install records (<child>.partial/L*.json or the union receipt),
and prints / writes one row per model: R-Q top-1 vs trunk / reference, held NLL, KL, residual norms, plus
the install diagnostics (K3 target energy, relL2, update_scale, veto counts, fallback rungs).
Every number is a DIAGNOSTIC (owner policy 2026-10-06).  Stdlib only.

    python3 s5_metrics.py --vcheck <receipt.json> [--layers-dir <partial dir> | --union-receipt <json>] --out <json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

RULE = {"rq_top1_vs_trunk_min": 0.80, "held_nll_over_trunk_max": 0.10, "residual_norm_ratio_max": 1.5}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_row(m: dict, trunk: dict) -> dict:
    row = {
        "rq_top1_vs_trunk": m["prompts"]["top1_agreement_vs_trunk"],
        "rq_top1_vs_trunk_count": f'{m["prompts"]["top1_agree_vs_trunk"]}/{m["prompts"]["positions"]}',
        "rq_top1_vs_reference": m["prompts"].get("top1_agreement_vs_reference"),
        "rq_kl_trunk": m["prompts"].get("kl_trunk_child_mean"),
        "held_nll": m["held"]["nll_mean"] if m["held"]["positions"] else None,
        "held_top1_vs_trunk": m["held"].get("top1_agreement_vs_trunk"),
        "held_top1_vs_reference": m["held"].get("top1_agreement_vs_reference"),
        "held_kl_trunk": m["held"].get("kl_trunk_child_mean"),
        "held_kl_reference": m["held"].get("kl_reference_model_mean"),
        "logits_finite": bool(m["prompts"]["logits_finite"] and (m["held"]["logits_finite"] if m["held"]["positions"] else True)),
    }
    if row["held_nll"] is not None:
        row["held_nll_minus_trunk"] = row["held_nll"] - trunk["held"]["nll_mean"]
    per = m.get("per_layer", [])
    if per:
        ratios = [(p["held_residual_norm_ratio_vs_trunk"], p["layer"]) for p in per]
        rels = [(p["held_residual_relL2_vs_trunk"], p["layer"]) for p in per]
        row.update({"residual_norm_ratio_max": max(ratios)[0], "residual_norm_ratio_max_layer": max(ratios)[1],
                    "residual_norm_ratio_final": ratios[-1][0], "residual_relL2_max": max(rels)[0],
                    "residual_relL2_final": rels[-1][0],
                    "residual_finite_all_layers": all(p.get("residual_finite", True) for p in per)})
        row["coherent_by_the_d4_rule"] = bool(row["rq_top1_vs_trunk"] >= RULE["rq_top1_vs_trunk_min"]
                                              and row.get("held_nll_minus_trunk", 0.0) <= RULE["held_nll_over_trunk_max"]
                                              and row["residual_norm_ratio_max"] <= RULE["residual_norm_ratio_max"])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vcheck", required=True)
    ap.add_argument("--layers-dir", default=None)
    ap.add_argument("--union-receipt", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vpath = Path(args.vcheck)
    v = json.loads(vpath.read_text(encoding="utf-8"))
    models = v["eval"]["models"]
    trunk = models["trunk"]
    doc = {"schema": "neural-forge-s5-metrics-v1", "vcheck": str(vpath), "vcheck_sha256": sha(vpath),
           "vcheck_verdict": v.get("verdict"), "rule_d4": RULE, "models": {}}
    for name, m in models.items():
        doc["models"][name] = model_row(m, trunk)
    for name, var in (v.get("variants") or {}).items():
        doc.setdefault("structure", {})[name] = {
            "accounting": var.get("accounting"), "structural_load": var.get("structural_load"),
            "veto_binding_layers": var.get("veto_binding_layers"), "c10_tool_version": var.get("c10_tool_version"),
            "union_target": var.get("union_target"), "open_loop": var.get("open_loop")}
    recs = []
    if args.layers_dir:
        d = Path(args.layers_dir)
        recs = [json.loads((d / f"L{k:02d}.json").read_text(encoding="utf-8")) for k in range(64)
                if (d / f"L{k:02d}.json").is_file()]
    elif args.union_receipt:
        recs = json.loads(Path(args.union_receipt).read_text(encoding="utf-8"))["layers"]
    if recs:
        fb = [(r["layer"], (r.get("ridge") or {}).get("solver_fallback", {}).get("rungs_not_factorisable"))
              for r in recs if (r.get("ridge") or {}).get("solver_fallback")]
        doc["install"] = {
            "layers": len(recs),
            "k3_target_energy_held_mean": mean(r["k3_term_energy_fraction"]["held"] for r in recs),
            "k3_target_energy_held_min_max": [min(r["k3_term_energy_fraction"]["held"] for r in recs),
                                              max(r["k3_term_energy_fraction"]["held"] for r in recs)],
            "k3_columns_write_fraction_held_mean": mean(r.get("k3_columns_write_fraction_held") for r in recs),
            "fit_held_relL2_mean": mean(r.get("heldout_relL2") for r in recs),
            "fit_held_relL2_min_max": [min(r["heldout_relL2"] for r in recs), max(r["heldout_relL2"] for r in recs)],
            "update_scale_mean": mean(r.get("update_scale") for r in recs),
            "ridge_veto_layers": sum(bool((r.get("ridge") or {}).get("veto_binding")) for r in recs),
            "veto_binding_layers": sum(bool(r.get("veto_binding")) for r in recs),
            "solver_fallback_layers": fb,
            "seconds_total": sum(float(r.get("seconds") or 0) for r in recs),
            "per_layer": [{"layer": r["layer"], "k3_layers": r.get("k3_layers"), "heldout_relL2": r.get("heldout_relL2"),
                           "k3_energy_held": r["k3_term_energy_fraction"]["held"], "update_scale": r.get("update_scale"),
                           "veto_binding": r.get("veto_binding"), "lambda": (r.get("ridge") or {}).get("lambda_relative"),
                           "w": (r.get("union_target") or {}).get("w"), "seconds": r.get("seconds")} for r in recs],
        }
    out = Path(args.out)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for name, row in doc["models"].items():
        keys = ("rq_top1_vs_trunk_count", "rq_top1_vs_reference", "rq_kl_trunk", "held_nll", "held_nll_minus_trunk",
                "held_top1_vs_trunk", "held_top1_vs_reference", "held_kl_trunk", "held_kl_reference",
                "residual_norm_ratio_max", "residual_norm_ratio_max_layer", "residual_norm_ratio_final", "logits_finite",
                "coherent_by_the_d4_rule")
        print(name, {k: (round(row[k], 4) if isinstance(row.get(k), float) else row.get(k)) for k in keys if k in row})
    if "install" in doc:
        i = doc["install"]
        print("install", {k: i[k] for k in ("layers", "k3_target_energy_held_mean", "k3_target_energy_held_min_max",
                                            "fit_held_relL2_mean", "update_scale_mean", "ridge_veto_layers",
                                            "veto_binding_layers", "solver_fallback_layers", "seconds_total")})
    print(f"metrics -> {out} sha256 {sha(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
