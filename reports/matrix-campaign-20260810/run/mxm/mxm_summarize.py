#!/usr/bin/env python3
"""One line of the wave-8 greedy log per composed child, from its own receipt.

Everything here is READ from the sealed receipt.  Nothing is recomputed except
the ADDENDUM 6 §6.2 storage-bound check, which the composition runner does not
seal (the isolation runner does, per component, and a composed child has no
single component to seal it against) -- so it is evaluated here, per component,
against the level each entry declares, and a violation is printed rather than
inferred from a width that "should" be right.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_DEPLOY = Path("/home/cereal/model-surgery-lab/deployments/neural-forge-matrix-beb699f2")
if str(_DEPLOY / "src") not in sys.path:
    sys.path.insert(0, str(_DEPLOY / "src"))

from neural_forge.matrix_cell_receipt import STORAGE_LEVEL_RATIOS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    body = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    composition = body["composition"]
    storage = composition["storage"]
    behavior = body.get("behavior") or {}
    free_run = behavior.get("free_run") or {}
    composed = body.get("composed") or {}
    drift = composition.get("global_drift") or {}

    rows = []
    violations = []
    for entry in composition["components"]:
        level = str(entry["level"])
        bound = STORAGE_LEVEL_RATIOS.get(level)
        stored = int(entry["storage"]["stored_parameter_count"])
        owned = int(entry["storage"]["owned_parameter_count"])
        realized = stored / owned if owned else None
        satisfied = None if bound is None else bool(stored <= int(bound * owned))
        if satisfied is False:
            violations.append(f"{entry['isolation_cell_id']} {realized:.6f} > {bound}")
        rows.append(
            {
                "cell_id": entry["isolation_cell_id"],
                "component": entry["component"],
                "level": level,
                "site": entry["site"],
                "level_width": entry["level_width"],
                "stored": stored,
                "owned": owned,
                "realized_ratio": realized,
                "storage_bound_satisfied": satisfied,
                "local_drift": entry["after_install"]["terminal_relL2_vs_control"],
                "free_run_after_install": entry["after_install"][
                    "free_run_token_match_vs_control"
                ],
                "write_heldout_cos": (entry.get("boundary") or {}).get(
                    "write_heldout_cos"
                ),
                "post_operator_heldout_relL2": (entry.get("boundary") or {}).get(
                    "post_operator_heldout_relL2"
                ),
                "wall_time_seconds": entry["wall_time_seconds"],
                "conditioned": bool((entry.get("evidence") or {}).get("conditioned")),
                "evidence_cache_reused": bool(
                    (entry.get("evidence") or {}).get("evidence_cache_reused")
                ),
            }
        )

    summary = {
        "cell_id": body["cell_id"],
        "cell_instance_id": body.get("cell_instance_id"),
        "mode": composition.get("mode"),
        "folded": len(rows),
        "install_order": composition["install_order"],
        "net_stored_ratio": storage["net_ratio"],
        "child_parameter_count": storage["child_parameter_count"],
        "parent_parameter_count": storage["parent_parameter_count"],
        "parameters_removed": storage["parent_parameter_count"]
        - storage["child_parameter_count"],
        "contracts": storage["contracts"],
        "global_drift": drift.get("terminal_relL2_vs_control"),
        "worst_local_drift": drift.get("per_component_worst_local_drift"),
        "free_run_token_match_diagnostic": free_run.get(
            "token_match_vs_control"
        ),
        "free_run_mean_lcp": free_run.get("mean_lcp"),
        "emit_zero_floor_passed": composed.get("emit_zero_floor_passed"),
        "norm_ratio_per_layer_minmax": composed.get(
            "norm_ratio_per_layer_minmax"
        ),
        "decode_smoke_review": (body.get("outcome") or {}).get(
            "decode_smoke_review"
        ),
        "smoke_terminations": [
            row.get("terminated") for row in (behavior.get("decode_smoke") or [])
        ],
        "status": (body.get("outcome") or {}).get("status"),
        "first_divergent_metric": (body.get("outcome") or {}).get(
            "first_divergent_metric"
        ),
        "wall_time_seconds": body.get("wall_time_seconds"),
        "storage_bound_violations": violations,
        "components": rows,
    }

    if arguments.json:
        print(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False))
        return 0

    print(f"child        {summary['cell_id']}  mode={summary['mode']}  "
          f"folded={summary['folded']}")
    print(f"install      {' -> '.join(summary['install_order'])}")
    print(f"net storage  {summary['net_stored_ratio']:.6f}x  "
          f"({summary['child_parameter_count']} of "
          f"{summary['parent_parameter_count']}, "
          f"-{summary['parameters_removed']})  "
          f"contracts={summary['contracts']}")
    print(f"global drift {summary['global_drift']}  "
          f"worst local {summary['worst_local_drift']}")
    print(f"token match  {summary['free_run_token_match_diagnostic']}  "
          f"[DIAGNOSTIC ONLY]   lcp {summary['free_run_mean_lcp']}")
    print(f"emit-zero    {summary['emit_zero_floor_passed']}    smoke "
          f"{summary['decode_smoke_review']} {summary['smoke_terminations']}"
          f"   norm-ratio {summary['norm_ratio_per_layer_minmax']}")
    print(f"chain        status={summary['status']}  "
          f"first_divergent={summary['first_divergent_metric']}  "
          f"wall {summary['wall_time_seconds']:.1f}s")
    if violations:
        print(f"STORAGE BOUND VIOLATED: {'; '.join(violations)}")
    else:
        print("storage bound satisfied at every component")
    print()
    header = (f"{'cell':<18}{'width':>6}{'stored':>9}{'/owned':>9}"
              f"{'ratio':>10}{'drift':>10}{'free-run':>10}{'cond':>6}{'s':>7}")
    print(header)
    for row in rows:
        print(f"{row['cell_id']:<18}{row['level_width']:>6}{row['stored']:>9}"
              f"{row['owned']:>9}{row['realized_ratio']:>10.6f}"
              f"{row['local_drift']:>10.6f}"
              f"{(row['free_run_after_install'] if row['free_run_after_install'] is not None else float('nan')):>10.4f}"
              f"{('yes' if row['conditioned'] else 'no'):>6}"
              f"{row['wall_time_seconds']:>7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
