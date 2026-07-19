"""Benchmark exact structural-attention runtime fixes.

The reference estimate records the reviewed LFM2 attempt-0009 geometry without
claiming that hard-coded dimensions authenticate a live run.  Production layer
results now derive and seal their receipt from actual phases, topology, and
cluster artifacts.  The executable benchmark uses smaller tensors, calls the
production preparation/verification helpers, and verifies exact legacy output
before reporting timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_forge.hf_activation_cluster import (  # noqa: E402
    _combine_modalities,
    _deterministic_medoid_cluster,
    _identity_cluster_result,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (  # noqa: E402
    _attention_output_transport_complexity_receipt,
    _prepare_attention_output_transport_from_tensors,
    _verify_attention_output_transport,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--benchmark-phases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260719)
    return parser.parse_args()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def reference_geometry_estimate() -> dict[str, Any]:
    fit_trajectories = 154
    phases_per_trajectory = 9
    fit_phases = fit_trajectories * phases_per_trajectory
    fit_rows = 40_656
    source_width = 1_024
    target_width = 776
    query_heads = 16
    key_value_heads = 8
    head_dim = 64
    source_offsets = query_heads // key_value_heads
    target_query_heads = 10
    target_key_value_heads = 5
    target_offsets = target_query_heads // target_key_value_heads
    fp64_bytes = 8

    composition_terms = source_width * query_heads * head_dim * target_width
    query_signature_elements = query_heads * fit_rows * head_dim
    contribution_signature_elements = query_heads * fit_rows * target_width
    offset_raw_elements = (
        2 * query_signature_elements + contribution_signature_elements
    )
    offset_feature_bytes = offset_raw_elements * fp64_bytes
    identity_result_bytes = 3 * source_offsets * 8
    return {
        "corpus": {
            "fit_trajectories": fit_trajectories,
            "fit_phases": fit_phases,
            "fit_rows": fit_rows,
        },
        "attention_topology": {
            "source_width": source_width,
            "target_width": target_width,
            "query_heads": query_heads,
            "key_value_heads": key_value_heads,
            "head_dim": head_dim,
            "source_query_offsets_per_kv": source_offsets,
            "target_query_offsets_per_kv": target_offsets,
            "offset_mapping_is_exact_identity": source_offsets == target_offsets,
        },
        "sealed_o_chart_transport": {
            "composition_scalar_terms": composition_terms,
            "flops_per_composition": 2 * composition_terms,
            "legacy_composition_count": fit_phases + 1,
            "prepared_composition_count": 1,
            "avoided_composition_count": fit_phases,
            "avoided_composition_flops": (
                2 * composition_terms * fit_phases
            ),
            "avoided_composition_teraflops": (
                2 * composition_terms * fit_phases / 1e12
            ),
            "avoided_cumulative_fp64_o_weight_materialization_bytes": (
                fit_phases * source_width * query_heads * head_dim * fp64_bytes
            ),
        },
        "equal_offset_identity_short_circuit": {
            "query_signature_elements": query_signature_elements,
            "contribution_signature_elements": contribution_signature_elements,
            "avoided_offset_feature_elements": offset_raw_elements,
            "avoided_offset_feature_bytes": offset_feature_bytes,
            "avoided_offset_feature_gib": offset_feature_bytes / 2**30,
            # At torch.cat, the legacy path simultaneously holds the copied
            # permute/reshape inputs, normalized parts, and concatenated result.
            "avoided_minimum_live_bytes_at_legacy_cat": 3
            * offset_feature_bytes,
            "avoided_minimum_live_gib_at_legacy_cat": (
                3 * offset_feature_bytes / 2**30
            ),
            "identity_result_bytes": identity_result_bytes,
        },
    }


def _median_seconds(
    operation: Callable[[], Any],
    *,
    iterations: int,
) -> float:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def _benchmark_o_transport(
    *,
    generator: torch.Generator,
    iterations: int,
    phases: int,
) -> dict[str, Any]:
    source_width = 64
    query_heads = 4
    head_dim = 16
    target_width = 48
    source_blocks = torch.randn(
        (source_width, query_heads, head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    encoder = torch.randn(
        (source_width, target_width),
        generator=generator,
        dtype=torch.float64,
    )

    def legacy() -> torch.Tensor:
        result = None
        for _ in range(phases + 1):
            result = torch.einsum("ohd,ot->hdt", source_blocks, encoder)
        assert result is not None
        return result

    prepared_transport = _prepare_attention_output_transport_from_tensors(
        source_output_weight=source_blocks.reshape(
            source_width,
            query_heads * head_dim,
        ),
        target_encoder=encoder,
        query_heads=query_heads,
        head_dim=head_dim,
        name="scaled benchmark O transport",
    )
    prepared_blocks = prepared_transport.target_output_blocks

    def prepared() -> torch.Tensor:
        result = prepared_blocks
        for _ in range(phases):
            result = _verify_attention_output_transport(
                prepared_transport,
                name="scaled benchmark O transport",
            ).target_output_blocks
        return result

    torch.testing.assert_close(legacy(), prepared(), atol=0.0, rtol=0.0)
    legacy_seconds = _median_seconds(legacy, iterations=iterations)
    prepared_seconds = _median_seconds(prepared, iterations=iterations)
    return {
        "geometry": {
            "source_width": source_width,
            "query_heads": query_heads,
            "head_dim": head_dim,
            "target_width": target_width,
            "phase_count": phases,
        },
        "legacy_median_seconds": legacy_seconds,
        "prepared_median_seconds": prepared_seconds,
        "observed_speedup": legacy_seconds / max(prepared_seconds, 1e-15),
        "exact_output_equivalence": True,
        "production_helper_receipt": dict(
            _attention_output_transport_complexity_receipt(
                transport=prepared_transport,
                phase_count=phases,
            )
        ),
    }


def _benchmark_identity_offsets(
    *,
    generator: torch.Generator,
    iterations: int,
) -> dict[str, Any]:
    source_groups = 8
    source_offsets = 2
    rows = 256
    head_dim = 16
    target_width = 48
    grouped_query = torch.randn(
        (source_groups, source_offsets, rows * head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    grouped_context = torch.randn(
        (source_groups, source_offsets, rows * head_dim),
        generator=generator,
        dtype=torch.float64,
    )
    grouped_contribution = torch.randn(
        (source_groups, source_offsets, rows * target_width),
        generator=generator,
        dtype=torch.float64,
    )

    def legacy():
        features = _combine_modalities(
            torch,
            (
                grouped_query.permute(1, 0, 2).reshape(source_offsets, -1),
                grouped_context.permute(1, 0, 2).reshape(source_offsets, -1),
                grouped_contribution.permute(1, 0, 2).reshape(
                    source_offsets, -1
                ),
            ),
        )
        return _deterministic_medoid_cluster(
            torch,
            features,
            source_offsets,
            max_iterations=16,
            chunk_size=512,
        )

    def optimized():
        return _identity_cluster_result(torch, source_offsets)

    legacy_result = legacy()
    optimized_result = optimized()
    torch.testing.assert_close(
        legacy_result.medoids,
        optimized_result.medoids,
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        legacy_result.assignment,
        optimized_result.assignment,
        atol=0,
        rtol=0,
    )
    assert legacy_result.strategy == optimized_result.strategy
    assert legacy_result.squared_error == optimized_result.squared_error
    legacy_seconds = _median_seconds(legacy, iterations=iterations)
    optimized_seconds = _median_seconds(optimized, iterations=iterations)
    return {
        "geometry": {
            "source_groups": source_groups,
            "source_offsets": source_offsets,
            "rows": rows,
            "head_dim": head_dim,
            "target_width": target_width,
        },
        "legacy_median_seconds": legacy_seconds,
        "identity_median_seconds": optimized_seconds,
        "observed_speedup": legacy_seconds / max(optimized_seconds, 1e-15),
        "exact_cluster_equivalence": True,
    }


def main() -> int:
    arguments = _arguments()
    if arguments.iterations <= 0 or arguments.benchmark_phases <= 0:
        raise ValueError("iterations and benchmark-phases must be positive")
    generator = torch.Generator().manual_seed(arguments.seed)
    report = {
        "schema": "neural-forge-lfm2-attention-runtime-fixes-benchmark-v1",
        "reference_attempt_0009_complexity_and_memory_estimate": (
            reference_geometry_estimate()
        ),
        "scaled_exact_benchmarks": {
            "sealed_o_chart_transport": _benchmark_o_transport(
                generator=generator,
                iterations=arguments.iterations,
                phases=arguments.benchmark_phases,
            ),
            "equal_offset_identity_short_circuit": (
                _benchmark_identity_offsets(
                    generator=generator,
                    iterations=arguments.iterations,
                )
            ),
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
