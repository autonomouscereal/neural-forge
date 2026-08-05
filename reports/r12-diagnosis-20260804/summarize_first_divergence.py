#!/usr/bin/env python3
"""Summarise a sealed first-divergence ledger into a per-layer profile."""

import json
import sys

ORDER = {"residual.pre": 0, "residual.post_operator": 1, "residual.post_layer": 2}
BRANCH_ORDER = {"operator.output": 0, "mlp.down": 1}


def main(directory: str) -> int:
    ledger = json.load(open(f"{directory}/first-divergence-ledger.json"))
    print("ledger_sha256", ledger["ledger_sha256"])
    print("request_sha256", ledger["request"]["request_sha256"])
    print("batch_count", ledger["batch_count"])
    print("focus_layer_indices", ledger["request"]["focus_layer_indices"])
    print("layer_count", ledger["request"]["layer_count"])
    print(
        "first_discrete_c_out_parent_winner_mismatch:",
        json.dumps(ledger["first_discrete_c_out_parent_winner_mismatch"]),
    )
    print("operation_audit:", json.dumps(ledger["operation_audit"]))
    for split in ("fit", "heldout"):
        summary = ledger["split_summaries"][split]
        print("=" * 92)
        print("SPLIT", split, "batches", summary["batch_count"])
        print(
            "c_out states",
            summary["c_out_state_count"],
            "parent-winner mismatches",
            summary["c_out_parent_winner_mismatch_count"],
            "match fraction",
            round(summary["c_out_parent_winner_match_fraction"], 8),
        )
        print(
            "descriptive largest mean state-error jump:",
            json.dumps(summary["descriptive_largest_mean_state_error_jump"]),
        )
        print("parent_logit_domain:", json.dumps(summary["parent_logit_domain"]))
        rows = sorted(
            summary["state_components"],
            key=lambda row: (row["layer_index"], ORDER.get(row["endpoint"], 9)),
        )
        header = "%-36s %7s %12s %12s %11s %13s" % (
            "state component",
            "rows",
            "mean_relL2",
            "max_relL2",
            "mean_cos",
            "incr_delta",
        )
        print(header)
        for row in rows:
            incremental = row["mean_incremental_relative_l2_delta"]
            print(
                "%-36s %7d %12.6f %12.6f %11.6f %13s"
                % (
                    row["component_id"],
                    row["row_count"],
                    row["mean_relative_l2"],
                    row["maximum_relative_l2"],
                    row["mean_cosine"],
                    "n/a" if incremental is None else "%.6f" % incremental,
                )
            )
        branches = sorted(
            summary["branch_components"],
            key=lambda row: (row["layer_index"], BRANCH_ORDER.get(row["endpoint"], 9)),
        )
        if branches:
            print("-" * 92)
            print(
                "%-36s %7s %12s %12s %11s"
                % ("branch component", "rows", "mean_relL2", "max_relL2", "mean_cos")
            )
            for row in branches:
                print(
                    "%-36s %7d %12.6f %12.6f %11.6f"
                    % (
                        row["component_id"],
                        row["row_count"],
                        row["mean_relative_l2"],
                        row["maximum_relative_l2"],
                        row["mean_cosine"],
                    )
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
