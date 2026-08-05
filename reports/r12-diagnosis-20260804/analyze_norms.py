#!/usr/bin/env python3
"""Separate scale error from direction error in the sealed first-divergence batches.

For each residual state boundary this prints
  * the mean child / projected-parent norm ratio, and
  * ``sqrt(1 - cos^2)``, the relative L2 error that would remain after an
    oracle per-vector rescale.  That is the part of the divergence no gauge or
    norm correction can remove.
Read-only, diagnostic only.
"""

import json
import math
import statistics
import sys
from pathlib import Path

ORDER = {"residual.pre": 0, "residual.post_operator": 1, "residual.post_layer": 2}


def main(directory: str, output: str) -> int:
    root = Path(directory)
    accumulator = {}
    branch_accumulator = {}
    for path in sorted((root / "batches").glob("batch-*.json")):
        result = json.loads(path.read_text())["result"]
        if result["split"] != "heldout":
            continue
        for phase in result["phases"]:
            for row in phase["state_path"]:
                _collect(accumulator, row)
            for row in phase["branch_components"]:
                _collect(branch_accumulator, row)

    payload = {
        "schema": "neural-forge-r12-first-divergence-scale-direction-split-v1",
        "source_directory": str(root),
        "split": "heldout",
        "numeric_metrics_are_diagnostic_only": True,
        "state_boundaries": _rows(accumulator),
        "branch_components": _rows(branch_accumulator),
    }
    Path(output).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    for title, rows in (
        ("STATE BOUNDARIES", payload["state_boundaries"]),
        ("BRANCH COMPONENTS", payload["branch_components"]),
    ):
        print("=" * 104)
        print(title, "(heldout)")
        print(
            "%-34s %11s %13s %13s %11s %15s"
            % (
                "component",
                "mean_relL2",
                "child_norm",
                "parent_norm",
                "norm_ratio",
                "irreducible*",
            )
        )
        for row in rows:
            print(
                "%-34s %11.6f %13.4f %13.4f %11.4f %15.6f"
                % (
                    row["component_id"],
                    row["mean_relative_l2"],
                    row["mean_candidate_norm"],
                    row["mean_projected_parent_norm"],
                    row["mean_norm_ratio"],
                    row["mean_irreducible_relative_l2_after_oracle_rescale"],
                )
            )
        print(
            "* irreducible = mean sqrt(1 - cos^2): the relative L2 error that survives "
            "an oracle per-vector rescale."
        )
    return 0


def _collect(accumulator, row):
    bucket = accumulator.setdefault(
        row["component_id"],
        {
            "component_id": row["component_id"],
            "layer_index": row["layer_index"],
            "endpoint": row["endpoint"],
            "relative_l2": [],
            "candidate_norm": [],
            "projected_parent_norm": [],
            "norm_ratio": [],
            "irreducible": [],
        },
    )
    metrics = row["metrics"]
    candidate = float(metrics["candidate_norm"])
    reference = float(metrics["projected_parent_norm"])
    cosine = float(metrics["cosine"])
    bucket["relative_l2"].append(float(metrics["relative_l2"]))
    bucket["candidate_norm"].append(candidate)
    bucket["projected_parent_norm"].append(reference)
    bucket["norm_ratio"].append(candidate / reference if reference else float("nan"))
    bucket["irreducible"].append(math.sqrt(max(0.0, 1.0 - cosine * cosine)))


def _rows(accumulator):
    return [
        {
            "component_id": bucket["component_id"],
            "layer_index": bucket["layer_index"],
            "endpoint": bucket["endpoint"],
            "row_count": len(bucket["relative_l2"]),
            "mean_relative_l2": statistics.fmean(bucket["relative_l2"]),
            "mean_candidate_norm": statistics.fmean(bucket["candidate_norm"]),
            "mean_projected_parent_norm": statistics.fmean(
                bucket["projected_parent_norm"]
            ),
            "mean_norm_ratio": statistics.fmean(bucket["norm_ratio"]),
            "mean_irreducible_relative_l2_after_oracle_rescale": statistics.fmean(
                bucket["irreducible"]
            ),
        }
        for bucket in sorted(
            accumulator.values(),
            key=lambda value: (
                value["layer_index"],
                ORDER.get(value["endpoint"], 9),
                value["component_id"],
            ),
        )
    ]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
