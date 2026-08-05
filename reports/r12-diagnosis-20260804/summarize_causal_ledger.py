#!/usr/bin/env python3
"""Summarise a sealed exact-4x causal-diagnostic ledger.

Read-only.  Prints the eight-coalition raw metrics, the exact Shapley split,
the Harsanyi interaction dividends, and the heldout record-clustered bootstrap
intervals.  Numeric output is diagnostic only.
"""

import json
import sys

RAW = ("terminal_relative_l2_error", "defined_vocab_kl", "parent_winner_margin")
BENEFITS = (
    "terminal_error_reduction",
    "defined_vocab_kl_reduction",
    "parent_winner_margin_gain",
)


def members_label(members):
    return "{" + ",".join(members) + "}" if members else "{} unmodified child"


def main(path: str) -> int:
    ledger = json.load(open(path))
    request = ledger["request"]
    specs = request["operator_specs"]
    print("ledger_sha256          ", ledger["ledger_sha256"])
    print("request_sha256         ", request["request_sha256"])
    print("group_ids              ", request["group_ids"])
    print(
        "record_count           ",
        request["record_count"],
        "(fit %d / heldout %d)"
        % (request["fit_record_count"], request["heldout_record_count"]),
    )
    print("batch_count            ", ledger["batch_count"])
    print("checkpoint_chain_authenticated", ledger["checkpoint_chain_authenticated"])
    print("operation_audit        ", json.dumps(ledger["operation_audit"]))
    print("numeric_acceptance_decision", ledger["numeric_acceptance_decision"])
    print("metric_contract        ", json.dumps(request["metric_contract"])[:400])
    print("intervention sites (restore = install the projected parent activation):")
    for spec in specs:
        print(
            "  %-2s stage %d layer %2d %-18s parent_endpoint=%-18s target=%s"
            % (
                spec["group_id"],
                spec["stage_order"],
                spec["layer_index"],
                spec["component_kind"],
                spec["parent_endpoint"],
                spec["target_module_path"],
            )
        )

    for split in ("fit", "heldout"):
        summary = ledger["split_summaries"][split]
        print("=" * 100)
        print(
            "SPLIT %s   batches %d   records %d"
            % (split, summary["batch_count"], summary["record_count"])
        )
        print("-- mean metrics per intervention coalition")
        print("%-26s %26s %20s %20s" % ("coalition", RAW[0], RAW[1], RAW[2]))
        for row in summary["mean_subset_metrics"]:
            metrics = row["metrics"]
            print(
                "%-26s %26.6f %20.6f %20.6f"
                % (
                    members_label(row["members"]),
                    metrics[RAW[0]],
                    metrics[RAW[1]],
                    metrics[RAW[2]],
                )
            )
        for benefit in BENEFITS:
            decomposition = summary["decompositions"][benefit]
            print("-- %s" % benefit)
            print(
                "   grand-coalition gain over unmodified child: %.6f"
                % decomposition["grand_coalition_gain_over_empty"]
            )
            print(
                "   exact shapley: "
                + "   ".join(
                    "%s=%.6f" % (row["group_id"], row["value"])
                    for row in decomposition["shapley_values"]
                )
            )
            print("   harsanyi dividends (order>=2 are pure interaction):")
            for row in decomposition["harsanyi_dividends"]:
                if not row["members"]:
                    continue
                print(
                    "     order %d %-12s %14.6f"
                    % (
                        len(row["members"]),
                        members_label(row["members"]),
                        row["dividend"],
                    )
                )
            print(
                "   shapley_efficiency_remainder %.3e   harsanyi_reconstruction_remainder %.3e"
                % (
                    decomposition["shapley_efficiency_remainder"],
                    decomposition["harsanyi_reconstruction_remainder"],
                )
            )

    print("=" * 100)
    print("heldout record-clustered Shapley bootstraps (4096 resamples, 95%)")
    print(
        "%-30s %6s %14s %14s %14s"
        % ("benefit", "group", "point", "ci_low", "ci_high")
    )
    for row in ledger["heldout_record_clustered_shapley_bootstraps"]:
        bootstrap = row["bootstrap"]
        print(
            "%-30s %6s %14.6f %14.6f %14.6f"
            % (
                row["benefit_metric"],
                row["group_id"],
                bootstrap["point_estimate"],
                bootstrap["lower_percentile"],
                bootstrap["upper_percentile"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
