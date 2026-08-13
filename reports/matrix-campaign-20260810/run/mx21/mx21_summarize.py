#!/usr/bin/env python3
"""Every wave-19 child's build receipt as one readable block, per axis.

Wave 16's `mx16_summarize.py` with the joint merge solve's own numbers added.  It reads build
receipts, not sealed ones, so it can be run on a rung the moment it is built and
before anything has graded it; nothing here scores anything.
"""

from __future__ import annotations

import glob
import json
import os

ROOT = "/mnt/ai_data/neural-forge-matrix/wave21/children"


def main() -> int:
    for path in sorted(glob.glob(f"{ROOT}/*/mx1[567]-build-receipt.json")):
        receipt = json.load(open(path, encoding="utf-8"))
        tag = os.path.basename(os.path.dirname(path))
        axes = {axis["axis"]: axis for axis in receipt["axes"]}
        print(
            f"{tag:12s} arm={receipt['arm']:18s} net={receipt['net_ratio']:.6f} "
            f"params={receipt['child_parameters']:,} ff={receipt['keep_ff']} "
            f"kv={receipt.get('keep_kv_groups')} exc={receipt['excised']}"
        )
        ff = axes.get("intermediate_size")
        if ff:
            held = ff["heldout"]
            print(
                f"             ff : median cos {held['median_cos']:.4f} "
                f"min {held['min_cos']:.4f} max drift {held['max_input_drift']:.4f} "
                f"removed {ff['parameters_removed']:,}"
            )
            print("             ff cos  :", " ".join("%.3f" % s["cos"] for s in held["per_site"]))
            print("             ff drift:", " ".join("%.2f" % s["input_drift_vs_parent"] for s in held["per_site"]))
        heads = axes.get("num_attention_heads")
        if heads:
            held = heads["heldout"]
            print(
                f"             kv : median cos {held['median_cos']:.4f} "
                f"min {held['min_cos']:.4f} removed {heads['parameters_removed']:,}"
            )
            for site in held["per_site"]:
                census = site["census"]
                merged = ", ".join(
                    f"{m['cluster']}(key {m['key_cos_sign_gauge']:.3f}, "
                    f"{m['signs_flipped']} flips, v {m['v_retained_energy']:.3f})"
                    for m in census["merged"]
                    if m["merged"]
                )
                print(
                    f"               A{site['parent_layer']:02d} cos {site['cos']:.4f} "
                    f"(prior-only {site['prior_only_cos']:.4f}) solve "
                    f"{site.get('solve_mode')}  {merged}"
                )
                for row in census["merged"]:
                    if not row.get("merged"):
                        continue
                    print(
                        f"                 pair {row['cluster']} relation relL2 "
                        f"average {row['relation_relL2_average']:.4f}"
                        + (
                            ""
                            if "relation_relL2_joint" not in row
                            else f" joint {row['relation_relL2_joint']:.4f}"
                        )
                    )
                value = site.get("value_solve") or {}
                if value:
                    print(
                        f"                 value/out gate "
                        f"{value.get('attention_recompute_relL2')} sweeps "
                        f"{value.get('sweeps_run')} installed "
                        f"{value.get('installed')} fit "
                        f"{value.get('fit_sse_improvement')}"
                    )
        if "num_hidden_layers" in axes:
            print(f"             exc: blocks {axes['num_hidden_layers']['sites']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
