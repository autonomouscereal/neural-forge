#!/usr/bin/env python3
"""The wave-29 children as one machine-readable index, read from sealed receipts.

Everything here comes off a sealed ARCH receipt or a sealed ADDENDUM 6 verdict;
nothing is recomputed and nothing is judged.  Wave 21's `mx21_child_index.py`
with the wave root moved; the docstring's wave number is fixed here, having said
"wave-19" through two waves that were not wave 19.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = "/mnt/ai_data/neural-forge-matrix/wave32"


def main() -> int:
    out = []
    for directory in sorted(glob.glob(f"{ROOT}/cells/ARCH.*")):
        run = json.load(open(f"{directory}/cell-receipt.json", encoding="utf-8"))
        review_path = f"{directory}/cell-receipt-behavior-reviewed.json"
        review = (
            json.load(open(review_path, encoding="utf-8"))
            if os.path.exists(review_path)
            else None
        )
        arch = run["architecture"]
        entry = {
            "cell_id": run["cell_id"],
            "cell_instance_id": run["cell_instance_id"],
            "receipt_sha": run["receipt_sha"],
            "child_parameters": arch["storage"]["child_parameter_count"],
            "net_ratio": arch["storage"]["net_ratio"],
            "config_diff": [d["field"] for d in arch["config_diff"]],
            "export_class": arch["export"]["loaded_class"],
            "trust_remote_code": arch["export"]["trust_remote_code"],
            "transformers_version": arch["export"].get("transformers_version"),
            "child_tensors_sha256": arch["child_tensors_sha256"],
            "axes": [
                {
                    "axis": a["axis"],
                    "sites": a["sites"],
                    "parameters_removed": a["parameters_removed"],
                }
                for a in arch["axes"]
            ],
            "terminal_relL2_vs_parent": run["composed"]["terminal_heldout_relL2"],
            "terminal_cos_vs_parent": arch["terminal_vs_parent"]["mean_cosine"],
            "free_run_token_match_diagnostic": run["behavior"]["free_run"][
                "token_match_vs_control"
            ],
            "worst_site_cos": run["boundary"]["write_heldout_cos"],
            "status": run["outcome"]["status"],
            "first_divergent_metric": run["outcome"]["first_divergent_metric"],
        }
        if review:
            a6 = review["addendum6"]
            a9 = review["addendum9"]
            entry.update(
                {
                    "verdict": a6["verdict"],
                    "useful": a6["useful"],
                    "partial": a6["partial"],
                    "observed_miss": a6["observed_miss"],
                    "useful_or_partial": a6["useful_or_partial"],
                    "parent_useful_or_partial": a6["parent_useful_or_partial"],
                    "safety": a6["safety"],
                    "degeneracy_reading": a9["degeneracy_reading"],
                    "degenerate_lane_count": a9["degenerate_lane_count"],
                    "per_lane_verdict": a9["per_lane_verdict"],
                    "per_child_verdict": a9["per_child_verdict"],
                    "safety_lane_chargeable": bool(a9.get("safety_lane")),
                    "verdict_receipt_sha": review["receipt_sha"],
                    "dossier_sha256": a6["frozen_review_dossier_sha256"],
                }
            )
        out.append(entry)
    body = {
        "schema": "neural-forge-matrix-wave32-child-index-v1",
        "parent_parameters": 229693184,
        "children": out,
    }
    path = f"{ROOT}/wave32-child-index.json"
    open(path, "w", encoding="utf-8").write(json.dumps(body, indent=1))
    for entry in out:
        print(
            f"{entry['cell_id']:16s} {entry['net_ratio']:.6f} "
            f"{entry.get('useful_or_partial', '-')}/15 {entry.get('verdict', '-'):6s} "
            f"lanes {entry.get('degenerate_lane_count', '-')} "
            f"{entry.get('degeneracy_reading', '-')}"
        )
    print("->", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
