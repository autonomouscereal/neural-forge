"""Benchmark the isolated CPCF v5 global matrix-free B/C/x solver.

The default geometry is intentionally laptop-safe.  For the production-shaped
GPU benchmark use:

    python experiments/benchmark_lfm2_cpcf_v5_matrix_free_bcx.py \
      --backend cuda --hidden 776 --cases 154 --sequence 264 --iterations 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (  # noqa: E402
    Lfm2CpcfV4ProjectionCase,
    replay_sampled_cp_volterra,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_matrix_free_bcx import (  # noqa: E402
    Lfm2CpcfV5MatrixFreeBcxPolicy,
    solve_lfm2_cpcf_v5_matrix_free_bcx,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--cases", type=int, default=16)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--fir-length", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=73191)
    return parser.parse_args()


def main() -> int:
    arguments = _arguments()
    if (
        min(
            arguments.hidden,
            arguments.cases,
            arguments.sequence,
            arguments.fir_length,
            arguments.iterations,
        )
        <= 0
    ):
        raise ValueError("all geometry and iteration arguments must be positive")
    torch.manual_seed(arguments.seed)
    hidden = arguments.hidden
    evidence_dtype = torch.float32
    in_proj_weight = (
        torch.randn(
            (3 * hidden, hidden),
            dtype=evidence_dtype,
        )
        / hidden**0.5
    )
    in_proj_bias = torch.randn((3 * hidden,), dtype=evidence_dtype) * 0.01
    fir_kernel = (
        torch.randn(
            (hidden, arguments.fir_length),
            dtype=evidence_dtype,
        )
        / arguments.fir_length**0.5
    )
    fir_bias = torch.randn((hidden,), dtype=evidence_dtype) * 0.01
    writer_weight = (
        torch.randn(
            (hidden, hidden),
            dtype=evidence_dtype,
        )
        / hidden**0.5
    )
    cases = []
    replays = []
    for case_index in range(arguments.cases):
        child = torch.randn(
            (1, arguments.sequence, hidden),
            dtype=evidence_dtype,
        )
        valid = torch.ones((1, arguments.sequence), dtype=torch.bool)
        replay = replay_sampled_cp_volterra(
            child,
            in_proj_weight=in_proj_weight,
            in_proj_bias=in_proj_bias,
            fir_kernel=fir_kernel,
            fir_bias=fir_bias,
            writer_weight=writer_weight,
            writer_bias=None,
            valid_prefix_mask=valid,
        )
        desired_context = (
            replay.context_history + torch.randn_like(replay.context_history) * 0.01
        )
        desired_write = (
            replay.write_history + torch.randn_like(replay.write_history) * 0.01
        )
        cases.append(
            Lfm2CpcfV4ProjectionCase(
                case_id=f"benchmark-{case_index:04d}",
                split="fit",
                child_input_history=child,
                desired_context_history=desired_context,
                desired_write_suffix=desired_write,
                suffix_start=0,
                valid_prefix_mask=valid,
                valid_suffix_mask=valid,
            )
        )
        replays.append(replay)
    policy = Lfm2CpcfV5MatrixFreeBcxPolicy(
        backend=arguments.backend,
        pcg_iterations=arguments.iterations,
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = solve_lfm2_cpcf_v5_matrix_free_bcx(
        tuple(cases),
        tuple(replays),
        in_proj_weight=in_proj_weight,
        in_proj_bias=in_proj_bias,
        fir_kernel=fir_kernel,
        fir_bias=fir_bias,
        writer_weight=writer_weight,
        policy=policy,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = result.diagnostics
    output = {
        "schema": "neural-forge-lfm2-cpcf-v5-matrix-free-bcx-benchmark-v1",
        "geometry": {
            "hidden": hidden,
            "cases": arguments.cases,
            "sequence": arguments.sequence,
            "fir_length": arguments.fir_length,
            "physical_coordinates": report["physical_correction_coordinate_count"],
        },
        "backend": report["solve"]["backend"],
        "compute_dtype": report["solve"]["compute_dtype"],
        "pcg_iterations": arguments.iterations,
        "elapsed_seconds": elapsed,
        "seconds_per_iteration_including_pack_and_conditioning": (
            elapsed / arguments.iterations
        ),
        "relative_normal_residual": report["solve"]["relative_residual_norms"][-1],
        "objective_before": report["weighted_fit_objective_before"],
        "objective_after": report["weighted_fit_objective_after"],
        "condition_upper_bound": report["conditioning"][
            "regularized_condition_upper_bound"
        ],
        "dense_design_materialized": report["solve"]["normal_design_materialized"],
        "dense_gram_materialized": report["solve"]["gram_materialized"],
        "peak_cuda_bytes": (
            None
            if not torch.cuda.is_available()
            else int(torch.cuda.max_memory_allocated())
        ),
        "result_report_sha256": report["report_sha256"],
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
