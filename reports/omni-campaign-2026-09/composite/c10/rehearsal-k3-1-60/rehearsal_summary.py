"""M4 rehearsal summary: child.partial/L{k:02d}.json (c10 per-layer records) -> rehearsal_layers.json + a
printed table.  Every number is a DIAGNOSTIC (owner policy 2026-10-06).  Read-only on the partial."""
import hashlib
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/cereal/omni-campaign/composite/c10/rehearsal-k3-1-60")
P = R / "child.partial"
rows = []
for k in range(64):
    f = P / f"L{k:02d}.json"
    if not f.is_file():
        break
    r = json.loads(f.read_text(encoding="utf-8"))
    held = (r.get("scores") or {}).get("held") or {}
    fit = (r.get("scores") or {}).get("fit") or {}
    sel = r.get("selection") or {}
    rows.append({
        "layer": k, "k3_layers": r.get("k3_layers"),
        "pool": (r.get("pool") or {}).get("k3_candidates"), "k3_rows": r["installed"]["k3_rows"],
        "zero_pad_rows": r["installed"]["zero_pad_rows"],
        "heldout_relL2_union": r.get("heldout_relL2"),
        "heldout_relL2_trunk_only": (held.get("trunk_only") or {}).get("relL2"),
        "heldout_relL2_no_repair_prior": (held.get("no_repair_prior") or {}).get("relL2"),
        "fit_relL2_union": (fit.get("union") or {}).get("relL2"),
        "k3_energy_fit": (r.get("k3_term_energy_fraction") or {}).get("fit"),
        "k3_energy_held": (r.get("k3_term_energy_fraction") or {}).get("held"),
        "veto_binding": r.get("veto_binding"), "k3_rows_function_absent": r.get("k3_rows_function_absent"),
        "veto_reason": r.get("veto_reason"),
        "update_scale": r.get("update_scale"), "ridge_lambda": (r.get("ridge") or {}).get("chosen_lambda_relative",
                                                                                          (r.get("ridge") or {}).get("lambda_relative")),
        "ridge_veto_binding": (r.get("ridge") or {}).get("veto_binding"),
        "ridge_free_solve_update_scale": (r.get("ridge") or {}).get("free_solve_update_scale"),
        "ridge_chosen_is_largest_lambda": (r.get("ridge") or {}).get("chosen_is_largest_lambda"),
        "chart_bias_worst_input_shift_relL2": ((r.get("chart_bias") or {}).get("input") or {}).get("worst_shift_relL2"),
        "chart_bias_output_energy_fraction": ((r.get("chart_bias") or {}).get("output") or {}).get("energy_fraction_fit"),
        "situ_to_silu_relL2": {s["source"]: s.get("relL2") for s in (r.get("situ_to_silu") or [])},
        "pivot": sel.get("pivot"), "pivot_seconds": sel.get("pivot_seconds"), "rank_starved": sel.get("rank_starved"),
        "phase_seconds": r.get("phase_seconds"), "seconds": r.get("seconds"),
        "peak_rss_gigabytes": r.get("peak_rss_gigabytes"),
        "rows_fit": (r.get("rows") or {}).get("fit"), "rows_held": (r.get("rows") or {}).get("held"),
        "record_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
    })
k3_layers = [x for x in rows if x["k3_layers"]]
summary = {
    "layers_persisted": len(rows),
    "veto_binding_layers": sum(bool(x["veto_binding"]) for x in rows),
    "veto_binding_layers_with_k3": sum(bool(x["veto_binding"]) for x in k3_layers),
    "ridge_veto_binding_layers": sum(bool(x["ridge_veto_binding"]) for x in rows),
    "function_absent_layers": sum(bool(x["k3_rows_function_absent"]) for x in rows),
    "k3_energy_held_mean_over_k3_layers": (sum(x["k3_energy_held"] or 0 for x in k3_layers) / len(k3_layers)) if k3_layers else None,
    "union_beats_trunk_only_held": sum(1 for x in rows if x["heldout_relL2_union"] is not None and x["heldout_relL2_trunk_only"] is not None
                                       and x["heldout_relL2_union"] < x["heldout_relL2_trunk_only"]),
    "seconds_total": round(sum(x["seconds"] or 0 for x in rows), 1),
    "peak_rss_gigabytes_max": max((x["peak_rss_gigabytes"] or 0) for x in rows) if rows else None,
    "note": "diagnostics only; veto_binding = ridge veto OR K3 held energy < 1e-6 (layers 42-63 carry no K3 by construction)",
}
doc = {"schema": "m4-c10-rehearsal-layers-v1", "partial": str(P), "summary": summary, "layers": rows}
out = R / "rehearsal_layers.json"
out.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
print(json.dumps(summary))
print("k  k3      pool  rows   held_union held_trunk  K3E_held  veto upd   piv_s  sec    rss")
for x in rows:
    print(f"{x['layer']:02d} {str(x['k3_layers']):7s} {x['pool'] or 0:5d} {x['k3_rows']:5d}  {x['heldout_relL2_union'] or 0:.4f}     "
          f"{x['heldout_relL2_trunk_only'] or 0:.4f}     {x['k3_energy_held'] or 0:.4f}    {str(x['veto_binding'])[0]}    "
          f"{x['update_scale'] or 0:.3f} {x['pivot_seconds'] or 0:6.1f} {x['seconds'] or 0:6.0f} {x['peak_rss_gigabytes']}")
print("sha256", hashlib.sha256(out.read_bytes()).hexdigest(), out)
