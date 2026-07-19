"""Benchmark the CPCF-v5 FIR-basis direct seed-distance table.

The default run keeps the real LFM2 4x channel geometry (1,024 -> 776) while
using a 4,096-column descriptor so it is practical on a workstation.  Add
``--full-production-width`` to execute the estimated broad-256 FIT descriptor
width as well:

    python experiments/benchmark_lfm2_cpcf_v5_seed_distance_table.py
    python experiments/benchmark_lfm2_cpcf_v5_seed_distance_table.py \
      --full-production-width --threads 16

The emitted production complexity receipt always uses 154 FIT trajectories,
the 10 activation columns per trajectory, and the two 776-wide endpoint
descriptor blocks used by the live 4x fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_forge import hf_lfm2_cpcf_short_conv_v5 as v5  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-channels", type=int, default=1024)
    parser.add_argument("--target-channels", type=int, default=776)
    parser.add_argument("--fit-cases", type=int, default=154)
    parser.add_argument("--descriptor-width", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--full-production-width",
        action="store_true",
        help="Execute FIR + fit_cases * (10 + 2 * target_channels) columns.",
    )
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    arguments = _arguments()
    if (
        arguments.source_channels <= 0
        or arguments.target_channels <= 0
        or arguments.target_channels > arguments.source_channels
        or arguments.fit_cases <= 0
        or arguments.descriptor_width <= 0
        or arguments.threads <= 0
    ):
        raise ValueError("benchmark geometry and thread counts are invalid")
    torch.set_num_threads(arguments.threads)
    source_channels = arguments.source_channels
    target_channels = arguments.target_channels
    production_descriptor_width = 4 + arguments.fit_cases * (10 + 2 * target_channels)
    executed_descriptor_width = (
        production_descriptor_width
        if arguments.full_production_width
        else arguments.descriptor_width
    )
    generator = torch.Generator().manual_seed(arguments.seed)
    descriptors = torch.randn(
        (source_channels, executed_descriptor_width),
        generator=generator,
        dtype=torch.float64,
    )

    started = time.perf_counter()
    seeds, groups, assignment, capacity_lookups, seed_distance_table = (
        v5._balanced_farthest_first_with_seed_distance_table(
            descriptors,
            target_channels=target_channels,
        )
    )
    table_and_partition_seconds = time.perf_counter() - started

    validation_locations = tuple(
        (
            (index * 37) % source_channels,
            (index * 101 + 17) % target_channels,
        )
        for index in range(32)
    )
    direct_errors = []
    direct_relative_errors = []
    direct_bitwise_matches = []
    for source, group in validation_locations:
        seed = seeds[group]
        direct_tensor = (descriptors[source] - descriptors[seed]).square().sum()
        direct = float(direct_tensor.item())
        direct_bitwise_matches.append(
            bool(seed_distance_table[source, group] == direct_tensor)
        )
        direct_errors.append(
            abs(float(seed_distance_table[source, group].item()) - direct)
        )
        direct_relative_errors.append(
            abs(float(seed_distance_table[source, group].item()) - direct)
            / max(abs(direct), torch.finfo(torch.float64).tiny)
        )
    executed_complexity = v5._seed_distance_table_complexity_receipt(
        source_channels=source_channels,
        target_channels=target_channels,
        descriptor_width=executed_descriptor_width,
        descriptor_block_count=1,
        capacity_assignment_distance_lookups=capacity_lookups,
    )
    maximum_group_capacity = (source_channels + target_channels - 1) // target_channels
    production_capacity_count_is_exact = (
        arguments.full_production_width or maximum_group_capacity <= 2
    )
    production_complexity = (
        v5._seed_distance_table_complexity_receipt(
            source_channels=source_channels,
            target_channels=target_channels,
            descriptor_width=production_descriptor_width,
            descriptor_block_count=1 + 2 * arguments.fit_cases,
            capacity_assignment_distance_lookups=capacity_lookups,
        )
        if production_capacity_count_is_exact
        else None
    )
    partition = {
        "seed_source_indices": seeds,
        "groups": groups,
        "source_to_target_group": assignment,
    }
    group_capacity_histogram = {
        str(capacity): sum(len(group) == capacity for group in groups)
        for capacity in sorted({len(group) for group in groups})
    }
    output = {
        "schema": ("neural-forge-lfm2-cpcf-v5-direct-seed-distance-table-benchmark-v1"),
        "executed_geometry": {
            "source_channels": source_channels,
            "target_channels": target_channels,
            "descriptor_width": executed_descriptor_width,
            "threads": arguments.threads,
            "full_production_width_executed": (arguments.full_production_width),
        },
        "timing": {
            "direct_seed_table_and_partition_seconds": (table_and_partition_seconds),
            "total_seconds": table_and_partition_seconds,
        },
        "validation": {
            "sampled_direct_seed_distance_count": len(validation_locations),
            "sampled_direct_distance_maximum_absolute_error": max(
                direct_errors,
                default=0.0,
            ),
            "sampled_direct_distance_maximum_relative_error": max(
                direct_relative_errors,
                default=0.0,
            ),
            "sampled_direct_distances_bitwise_equal_to_scalar_legacy": all(
                direct_bitwise_matches
            ),
            "seed_distance_table_shape": list(seed_distance_table.shape),
            "selected_seed_self_distances_exact_zero": bool(
                all(
                    seed_distance_table[seed, group].item() == 0.0
                    for group, seed in enumerate(seeds)
                )
            ),
            "seed_distance_table_finite": bool(
                torch.isfinite(seed_distance_table).all().item()
            ),
            "seed_distance_table_nonnegative": bool(
                torch.all(seed_distance_table >= 0.0).item()
            ),
            "balanced_group_capacity_histogram": (group_capacity_histogram),
            "partition_sha256": _canonical_sha256(partition),
        },
        "executed_complexity": executed_complexity,
        "production_complexity": production_complexity,
        "production_capacity_assignment_count_exact": (
            production_capacity_count_is_exact
        ),
        "execution": {
            "cached_direct_seed_distance_columns": True,
            "direct_distance_row_chunk_size": (
                v5.V5_DIRECT_SEED_DISTANCE_ROW_CHUNK_SIZE
            ),
            "direct_seed_temporary_reused_across_rows": True,
            "rowwise_direct_squared_euclidean_objective_preserved": True,
            "scalar_legacy_reduction_semantics_preserved": False,
            "finite_precision_reduction_association": (
                "fixed_128_row_batched_reduction"
            ),
            "full_source_by_descriptor_temporary_materialized": False,
            "pairwise_gram_identity_used": False,
            "negative_distance_clamping_used": False,
            "folding_objective_changed": False,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    output["report_sha256"] = _canonical_sha256(output)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
