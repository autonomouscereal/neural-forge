#!/usr/bin/env python3
"""Derive a decision-boundary profile from sealed first-divergence batches.

Read-only over the sealed batch checkpoints written by
``audit_lfm2_exact4x_first_divergence.py``.  Writes one derived JSON summary
next to them; touches nothing else.  Numeric output is diagnostic only.
"""

import json
import math
import statistics
import sys
from pathlib import Path

STATE_ORDER = {"residual.pre": 0, "residual.post_operator": 1, "residual.post_layer": 2}


def percentile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def main(directory: str, output: str) -> int:
    root = Path(directory)
    batch_paths = sorted((root / "batches").glob("batch-*.json"))
    if not batch_paths:
        raise SystemExit(f"No batch checkpoints under {root / 'batches'}")

    winner_ranks = []
    parent_margins = []
    child_margins = []
    per_phase = {}
    per_phase_state = {}
    depth_by_phase = {}
    first_mismatch_by_batch = []

    for path in batch_paths:
        checkpoint = json.loads(path.read_text())
        result = checkpoint["result"]
        split = result["split"]
        batch_index = result["batch_index"]
        earliest = None
        for phase in result["phases"]:
            name = phase["phase"]
            index = phase["phase_index"]
            c_out = phase["c_out"]
            child = c_out["folded_child"]
            parent = c_out["cached_parent"]
            rank = int(child["parent_winner_rank"])
            winner_ranks.append(rank)
            parent_margins.append(float(parent["parent_winner_margin"]))
            child_margins.append(float(child["parent_winner_margin"]))
            bucket = per_phase.setdefault(
                (split, index, name),
                {"count": 0, "match": 0, "ranks": []},
            )
            bucket["count"] += 1
            bucket["match"] += int(bool(child["top1_matches_parent"]))
            bucket["ranks"].append(rank)
            if not child["top1_matches_parent"] and earliest is None:
                earliest = index
            for row in phase["state_path"]:
                key = (split, index, row["component_id"])
                per_phase_state.setdefault(key, []).append(
                    float(row["metrics"]["relative_l2"])
                )
            terminal = [
                row
                for row in phase["state_path"]
                if row["component_id"] == "layer.013.residual.post_layer"
            ]
            if terminal:
                depth_by_phase.setdefault((split, index), []).append(
                    float(terminal[0]["metrics"]["relative_l2"])
                )
        first_mismatch_by_batch.append((batch_index, split, earliest))

    summary = {
        "schema": "neural-forge-r12-first-divergence-derived-summary-v1",
        "source_directory": str(root),
        "batch_checkpoint_count": len(batch_paths),
        "numeric_metrics_are_diagnostic_only": True,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "parent_winner_rank_in_child_scores": {
            "observation_count": len(winner_ranks),
            "defined_semantic_row_count": 64402,
            "rank_1_count": sum(1 for value in winner_ranks if value == 1),
            "rank_le_10_count": sum(1 for value in winner_ranks if value <= 10),
            "rank_le_100_count": sum(1 for value in winner_ranks if value <= 100),
            "rank_le_1000_count": sum(1 for value in winner_ranks if value <= 1000),
            "median_rank": percentile(winner_ranks, 0.5),
            "p05_rank": percentile(winner_ranks, 0.05),
            "p95_rank": percentile(winner_ranks, 0.95),
            "mean_rank": statistics.fmean(winner_ranks),
            "max_rank": max(winner_ranks),
        },
        "parent_winner_margin": {
            "cached_parent_mean": statistics.fmean(parent_margins),
            "cached_parent_median": percentile(parent_margins, 0.5),
            "folded_child_mean": statistics.fmean(child_margins),
            "folded_child_median": percentile(child_margins, 0.5),
            "folded_child_negative_count": sum(
                1 for value in child_margins if value < 0.0
            ),
        },
        "phases": [
            {
                "split": split,
                "phase_index": index,
                "phase": name,
                "state_count": bucket["count"],
                "top1_match_count": bucket["match"],
                "top1_match_fraction": bucket["match"] / bucket["count"],
                "median_parent_winner_rank": percentile(bucket["ranks"], 0.5),
                "terminal_residual_relative_l2_mean": (
                    statistics.fmean(depth_by_phase[(split, index)])
                    if (split, index) in depth_by_phase
                    else None
                ),
            }
            for (split, index, name), bucket in sorted(per_phase.items())
        ],
        "batches_with_no_mismatch_in_any_phase": sorted(
            batch_index
            for batch_index, _split, earliest in first_mismatch_by_batch
            if earliest is None
        ),
        "batches_whose_first_mismatch_is_prefill": sum(
            1
            for _batch_index, _split, earliest in first_mismatch_by_batch
            if earliest == 0
        ),
    }

    depth_rows = {}
    for (split, index, component), values in per_phase_state.items():
        if split != "heldout":
            continue
        depth_rows.setdefault(component, {})[index] = statistics.fmean(values)
    summary["heldout_component_relative_l2_by_phase_index"] = {
        component: [values[index] for index in sorted(values)]
        for component, values in sorted(
            depth_rows.items(),
            key=lambda item: (
                0 if item[0] == "c_in" else 1,
                item[0],
            ),
        )
    }

    Path(output).write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")

    ranks = summary["parent_winner_rank_in_child_scores"]
    print("parent-winner rank in child scores (of 64402 defined rows)")
    print(
        "  observations %d  rank==1 %d  rank<=10 %d  rank<=100 %d  rank<=1000 %d"
        % (
            ranks["observation_count"],
            ranks["rank_1_count"],
            ranks["rank_le_10_count"],
            ranks["rank_le_100_count"],
            ranks["rank_le_1000_count"],
        )
    )
    print(
        "  median %s  p05 %s  p95 %s  mean %.1f  max %d"
        % (
            ranks["median_rank"],
            ranks["p05_rank"],
            ranks["p95_rank"],
            ranks["mean_rank"],
            ranks["max_rank"],
        )
    )
    print("margins:", json.dumps(summary["parent_winner_margin"]))
    print("batches with zero mismatches:", summary["batches_with_no_mismatch_in_any_phase"])
    print(
        "batches whose first mismatch is the prefill state:",
        summary["batches_whose_first_mismatch_is_prefill"],
    )
    print()
    print(
        "%-8s %6s %7s %7s %14s %12s %22s"
        % (
            "split",
            "phase",
            "states",
            "match",
            "match_fraction",
            "median_rank",
            "terminal_resid_relL2",
        )
    )
    for row in summary["phases"]:
        print(
            "%-8s %6d %7d %7d %14.6f %12.1f %22.6f"
            % (
                row["split"],
                row["phase_index"],
                row["state_count"],
                row["top1_match_count"],
                row["top1_match_fraction"],
                row["median_parent_winner_rank"],
                row["terminal_residual_relative_l2_mean"],
            )
        )
    print()
    print("heldout mean relative_l2 by decode phase index (0=prefill .. 8):")
    for component, values in summary[
        "heldout_component_relative_l2_by_phase_index"
    ].items():
        print("  %-34s %s" % (component, " ".join("%8.4f" % value for value in values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
