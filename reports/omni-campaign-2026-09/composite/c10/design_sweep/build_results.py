#!/usr/bin/env python3
"""Build RESULTS.json + RESULTS.md for the C10 D4 design sweep (2026-10-11, Claude).

Reads the sweep records (sweep/shared/L*.json, sweep/variants/<name>/L*.json) and the one-pass
virtual-child evaluation (eval/virtual_check_variants.json), applies the selection rule that
DESIGN.md registered before any result existed, and writes one row per variant.  Stdlib only.

    python3 build_results.py [--dir $OC/composite/c10/design_sweep]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

ORDER = ["legacy", "a0", "a003", "a01", "a03", "b01", "b025", "b05", "g01", "g03"]
RULE = {"rq_top1_vs_trunk_min": 0.80, "held_nll_over_trunk_max": 0.10, "residual_norm_ratio_max": 1.5,
        "tie_energy": 0.01}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    d = Path(args.dir)
    ev_paths = sorted((d / "eval").glob("virtual_check_*.json"))
    if not ev_paths:
        raise SystemExit("no eval/virtual_check_*.json")
    evs = [json.loads(p.read_text(encoding="utf-8")) for p in ev_paths]
    ev, ev_path = evs[0], ev_paths[0]
    models = {}
    part_trunk = {}     # each child is compared with the trunk run in its OWN part (same card, same kernels)
    trunk_parent_parts = {}
    for path, part in zip(ev_paths, evs):
        if part.get("verdict", "").startswith("REFUSED"):
            raise SystemExit(f"{path}: {part['verdict']}")
        for name, m in part["eval"]["models"].items():
            if name in ("trunk", "parent"):
                trunk_parent_parts.setdefault(name, []).append((path.name, m))
                models.setdefault(name, m)
            else:
                models[name] = m
                part_trunk[name] = part["eval"]["models"]["trunk"]
    # the trunk / parent are re-run in every part: same inputs, same ops -> their numbers must agree
    consistency = {name: {pn: {"held_nll": m["held"]["nll_mean"], "rq_top1_vs_trunk": m["prompts"]["top1_agreement_vs_trunk"],
                               "rq_top1_vs_reference": m["prompts"].get("top1_agreement_vs_reference")}
                          for pn, m in parts} for name, parts in trunk_parent_parts.items()}
    trunk, parent = models["trunk"], models.get("parent")
    rows = []
    for name in ORDER:
        vdir = d / "sweep" / "variants" / name
        recs = [json.loads((vdir / f"L{k:02d}.json").read_text(encoding="utf-8")) for k in range(64)
                if (vdir / f"L{k:02d}.json").is_file()]
        refused = sorted(int(p.name[1:3]) for p in vdir.glob("L*.refused.json"))
        m = models.get(name)
        row = {"name": name, "spec": recs[0]["union_target"]["spec"] if recs else None, "layers": len(recs),
               "layers_refused_by_the_law": refused,
               "w_mean": mean(r["union_target"]["w"] for r in recs),
               "a": recs[0]["union_target"]["a"] if recs else None,
               "k3_target_energy_held_mean": mean(r["k3_term_energy_fraction"]["held"] for r in recs),
               "k3_columns_write_fraction_held_mean": mean(r.get("k3_columns_write_fraction_held") for r in recs),
               "fit_held_relL2_mean": mean(r.get("heldout_relL2") for r in recs),
               "update_scale_mean": mean(r.get("update_scale") for r in recs),
               "ridge_veto_layers": sum(bool((r.get("ridge") or {}).get("veto_binding")) for r in recs),
               "veto_binding_layers": sum(bool(r.get("veto_binding")) for r in recs)}
        if m is not None:
            per = m.get("per_layer", [])
            ratios = [(p["held_residual_norm_ratio_vs_trunk"], p["layer"]) for p in per]
            rels = [(p["held_residual_relL2_vs_trunk"], p["layer"]) for p in per]
            row.update({
                "rq_top1_vs_trunk": m["prompts"]["top1_agreement_vs_trunk"],
                "rq_top1_vs_trunk_count": f'{m["prompts"]["top1_agree_vs_trunk"]}/{m["prompts"]["positions"]}',
                "rq_top1_vs_parent": m["prompts"].get("top1_agreement_vs_reference"),
                "rq_kl_trunk": m["prompts"]["kl_trunk_child_mean"],
                "held_nll": m["held"]["nll_mean"],
                "held_nll_minus_trunk": m["held"]["nll_mean"] - part_trunk.get(name, trunk)["held"]["nll_mean"],
                "held_nll_trunk_same_part": part_trunk.get(name, trunk)["held"]["nll_mean"],
                "held_top1_vs_trunk": m["held"]["top1_agreement_vs_trunk"],
                "held_top1_vs_parent": m["held"].get("top1_agreement_vs_reference"),
                "held_kl_trunk": m["held"]["kl_trunk_child_mean"],
                "held_kl_parent": m["held"].get("kl_reference_model_mean"),
                "residual_norm_ratio_max": max(ratios)[0] if ratios else None,
                "residual_norm_ratio_max_layer": max(ratios)[1] if ratios else None,
                "residual_norm_ratio_final": ratios[-1][0] if ratios else None,
                "residual_relL2_max": max(rels)[0] if rels else None,
                "residual_relL2_final": rels[-1][0] if rels else None,
                "logits_finite": bool(m["prompts"]["logits_finite"] and m["held"]["logits_finite"]),
            })
            row["coherent"] = bool(row["rq_top1_vs_trunk"] >= RULE["rq_top1_vs_trunk_min"]
                                   and row["held_nll_minus_trunk"] <= RULE["held_nll_over_trunk_max"]
                                   and row["residual_norm_ratio_max"] <= RULE["residual_norm_ratio_max"])
        rows.append(row)
    coherent = [r for r in rows if r.get("coherent")]
    pick = None
    if coherent:
        best = max(r["k3_target_energy_held_mean"] for r in coherent)
        near = [r for r in coherent if best - r["k3_target_energy_held_mean"] <= RULE["tie_energy"]]
        pick = min(near, key=lambda r: r["held_nll"])["name"]
    refs = {"trunk": {"held_nll": trunk["held"]["nll_mean"], "held_top1_vs_parent": trunk["held"].get("top1_agreement_vs_reference"),
                      "rq_top1_vs_parent": trunk["prompts"].get("top1_agreement_vs_reference"),
                      "held_kl_parent": trunk["held"].get("kl_reference_model_mean")}}
    if parent is not None:
        refs["parent"] = {"held_nll": parent["held"]["nll_mean"],
                          "held_top1_vs_trunk": parent["held"]["top1_agreement_vs_trunk"],
                          "rq_top1_vs_trunk": parent["prompts"]["top1_agreement_vs_trunk"]}
    doc = {"schema": "neural-forge-c10-design-sweep-results-v1", "design": str(d / "DESIGN.md"),
           "design_sha256": sha(d / "DESIGN.md"), "eval_receipts": {p.name: sha(p) for p in ev_paths},
           "trunk_parent_consistency_across_parts": consistency,
           "eval_sequences": ev["eval"]["sequences"], "selection_rule": RULE, "references": refs, "rows": rows,
           "coherent": [r["name"] for r in coherent], "pick": pick,
           "note": "every number is a DIAGNOSTIC (owner policy 2026-10-06); the pick is a design recommendation, "
                   "the C10 default target is unchanged"}
    (d / "RESULTS.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")

    def f(x, nd=3):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))
    lines = ["# C10 D4 design sweep: results", "",
             "Eval receipts: " + ", ".join(f"`{k}` `{v[:16]}...`" for k, v in doc["eval_receipts"].items()) + "; DESIGN.md sha "
             f"`{doc['design_sha256'][:16]}...` (selection rule registered before the run).", "",
             f"References: S3 trunk held NLL {f(refs['trunk']['held_nll'])}, trunk vs parent top-1 held "
             f"{f(refs['trunk']['held_top1_vs_parent'])} / R-Q {f(refs['trunk']['rq_top1_vs_parent'])}"
             + (f"; parent held NLL {f(refs['parent']['held_nll'])}" if 'parent' in refs else "") + ".", "",
             "| variant | spec | w (mean) | K3 target energy | K3 cols write | R-Q top1 vs trunk | R-Q top1 vs parent | "
             "held NLL | dNLL vs trunk | held top1 vs trunk | held top1 vs parent | KL trunk (held) | max norm ratio (layer) | "
             "veto layers | coherent |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['spec']} | {f(r['w_mean'])} | {f(r['k3_target_energy_held_mean'])} | "
                     f"{f(r['k3_columns_write_fraction_held_mean'])} | {f(r.get('rq_top1_vs_trunk'))} ({r.get('rq_top1_vs_trunk_count')}) | "
                     f"{f(r.get('rq_top1_vs_parent'))} | {f(r.get('held_nll'))} | {f(r.get('held_nll_minus_trunk'))} | "
                     f"{f(r.get('held_top1_vs_trunk'))} | {f(r.get('held_top1_vs_parent'))} | {f(r.get('held_kl_trunk'))} | "
                     f"{f(r.get('residual_norm_ratio_max'))} (L{r.get('residual_norm_ratio_max_layer')}) | "
                     f"{r['veto_binding_layers']}/{r['layers']} | {r.get('coherent')} |")
    lines += ["", f"Coherent under the registered rule: {', '.join(doc['coherent']) or 'none'}. "
              f"Pick: **{pick or 'none (D4 stays open)'}**.", "",
              "dNLL is against the trunk run in the same eval part (same card): the trunk reads 3.756 on the RTX 8000 "
              "parts and 3.760 on the 3090 parts.",
              "CAVEAT: the c17 charts were fitted on trunk taps of THESE six held rows (c17 trunk_taps s3native-c2k4352-held-all, "
              "24,576 rows/layer), so the K3 term is in-sample on the held rows and the held-NLL gains of K3-carrying variants "
              "are not out-of-sample evidence; the R-Q prompts are the clean measure. The parent (S0 import, bf16) reads a "
              "HIGHER held NLL than the trunk (4.29 vs 3.76) while its R-Q NLL is lower (2.21 vs 3.55): unexplained.", "",
              "All numbers are diagnostics. The sweep is open loop (each layer fitted on the S3 trunk's rows); the "
              "evaluation runs each variant as a real virtual child (closed forward)."]
    (d / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
