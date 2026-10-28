"""Read-only scaling probe for the uncommitted sequence-effective attention arm.

This file is intentionally outside the tracked source/test trees.  It measures
representative dense kernels only; it does not compile or mutate a model.
"""

from __future__ import annotations

import gc
import json
import math
import os
import statistics
import time

import torch


N_REAL = 40_000
HIDDEN = 776
QUERY_HEADS = 10
KV_HEADS = 5
HEADS_PER_GROUP = 2
HEAD_DIM = 64
OUTPUT = 776
FP64_BYTES = 8


def gib(elements: int) -> float:
    return elements * FP64_BYTES / (1024**3)


def tflop(count: float) -> float:
    return count / 1e12


def timed(name: str, fn, *, warmups: int = 1, repeats: int = 3) -> dict:
    for _ in range(warmups):
        value = fn()
        del value
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        elapsed.append(time.perf_counter() - start)
        del value
    gc.collect()
    return {
        "name": name,
        "seconds_median": statistics.median(elapsed),
        "seconds_minimum": min(elapsed),
        "seconds_all": elapsed,
    }


def main() -> None:
    torch.manual_seed(20260719)
    torch.set_num_threads(min(6, os.cpu_count() or 1))
    dtype = torch.float64
    report: dict[str, object] = {
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "real_geometry": {
            "rows": N_REAL,
            "hidden": HIDDEN,
            "query_heads": QUERY_HEADS,
            "kv_heads": KV_HEADS,
            "heads_per_group": HEADS_PER_GROUP,
            "head_dim": HEAD_DIM,
            "output": OUTPUT,
        },
    }

    flat_features = QUERY_HEADS * HIDDEN
    value_features = HIDDEN * HEAD_DIM
    coefficient_elements = N_REAL * OUTPUT * value_features
    report["static_geometry"] = {
        "attended_inputs_gib": gib(N_REAL * QUERY_HEADS * HIDDEN),
        "endpoint_gib": gib(N_REAL * OUTPUT),
        "effective_primal_gram_gib": gib(flat_features * flat_features),
        "effective_solution_gib": gib(flat_features * OUTPUT),
        "value_coefficient_shape": [N_REAL * OUTPUT, value_features],
        "value_coefficient_elements": coefficient_elements,
        "value_coefficient_gib": gib(coefficient_elements),
        "value_normal_gram_gib": gib(value_features * value_features),
        "triangular_cross_head_stats_gib": gib(
            (QUERY_HEADS * (QUERY_HEADS + 1) // 2) * HIDDEN * HIDDEN
        ),
        "target_cross_stats_gib": gib(QUERY_HEADS * HIDDEN * OUTPUT),
    }

    benchmarks: list[dict] = []

    # Exact-size output ridge design Gram (640 features); extrapolate rows only.
    n_out = 4096
    z = torch.randn(n_out, QUERY_HEADS * HEAD_DIM, dtype=dtype)
    output_gram = timed("output_design_gram_n4096_f640", lambda: z.T @ z)
    output_gram["flops"] = 2 * n_out * (QUERY_HEADS * HEAD_DIM) ** 2
    output_gram["predicted_seconds_n40000_linear"] = (
        output_gram["seconds_median"] * N_REAL / n_out
    )
    benchmarks.append(output_gram)
    del z

    # Exact-size output solve, including all 776 right-hand sides.
    f_out = QUERY_HEADS * HEAD_DIM
    base = torch.randn(f_out, f_out, dtype=dtype)
    gram = base.T @ base + 1e-2 * torch.eye(f_out, dtype=dtype)
    rhs = torch.randn(f_out, OUTPUT, dtype=dtype)
    benchmarks.append(
        timed(
            "output_solve_f640_o776",
            lambda: torch.linalg.solve(gram, rhs),
            warmups=0,
            repeats=2,
        )
    )
    del base, gram, rhs

    # One exact cross-head sufficient-stat block, at a row count large enough
    # for stable BLAS throughput.  There are 55 triangular K blocks and 10 B.
    n_stats = 2048
    a = torch.randn(n_stats, HIDDEN, dtype=dtype)
    b = torch.randn(n_stats, HIDDEN, dtype=dtype)
    y = torch.randn(n_stats, OUTPUT, dtype=dtype)
    cross = timed("cross_head_stat_n2048_h776", lambda: a.T @ b)
    cross["flops"] = 2 * n_stats * HIDDEN * HIDDEN
    cross["predicted_seconds_55_blocks_n40000"] = (
        cross["seconds_median"] * 55 * N_REAL / n_stats
    )
    benchmarks.append(cross)
    target_cross = timed("target_cross_stat_n2048_h776_o776", lambda: a.T @ y)
    target_cross["flops"] = 2 * n_stats * HIDDEN * OUTPUT
    target_cross["predicted_seconds_10_blocks_n40000"] = (
        target_cross["seconds_median"] * 10 * N_REAL / n_stats
    )
    benchmarks.append(target_cross)
    del a, b, y

    # Exact structured value normal application after sufficient stats exist.
    k00 = torch.randn(HIDDEN, HIDDEN, dtype=dtype)
    k00 = k00.T @ k00 / HIDDEN
    k11 = torch.randn(HIDDEN, HIDDEN, dtype=dtype)
    k11 = k11.T @ k11 / HIDDEN
    k01 = torch.randn(HIDDEN, HIDDEN, dtype=dtype)
    kernels = ((k00, k01), (k01.T, k11))
    outputs = torch.randn(HEADS_PER_GROUP, HEAD_DIM, OUTPUT, dtype=dtype)
    value = torch.randn(HIDDEN, HEAD_DIM, dtype=dtype)
    output_grams = tuple(
        tuple(outputs[k] @ outputs[h].T for k in range(2)) for h in range(2)
    )

    def structured_value_normal() -> torch.Tensor:
        result = torch.zeros_like(value)
        for h in range(2):
            for k in range(2):
                result.add_(kernels[h][k] @ value @ output_grams[h][k])
        return result

    structured = timed(
        "structured_value_normal_h776_d64_two_heads",
        structured_value_normal,
    )
    structured["flops_approx"] = 4 * (
        2 * HIDDEN * HIDDEN * HEAD_DIM + 2 * HIDDEN * HEAD_DIM * HEAD_DIM
    )
    structured["predicted_seconds_16_pcg_5_groups_12_sweeps"] = (
        structured["seconds_median"] * 16 * KV_HEADS * 12
    )
    benchmarks.append(structured)
    del k00, k11, k01, kernels, outputs, value, output_grams

    # Raw row-space value normal: bounded memory, but repeats N work per PCG.
    n_raw = 128
    attended = torch.randn(
        n_raw, HEADS_PER_GROUP, HIDDEN, dtype=dtype
    )
    outputs = torch.randn(
        HEADS_PER_GROUP, HEAD_DIM, OUTPUT, dtype=dtype
    )
    value = torch.randn(HIDDEN, HEAD_DIM, dtype=dtype)

    def raw_value_normal() -> torch.Tensor:
        contexts = torch.einsum("nhi,id->nhd", attended, value)
        prediction = torch.einsum("nhd,hdo->no", contexts, outputs)
        context_adjoint = torch.einsum("no,hdo->nhd", prediction, outputs)
        return torch.einsum("nhi,nhd->id", attended, context_adjoint)

    raw = timed("raw_value_normal_n128", raw_value_normal, repeats=2)
    raw["flops_approx"] = (
        8 * n_raw * HEAD_DIM * (HIDDEN + OUTPUT) * HEADS_PER_GROUP / 2
    )
    raw["predicted_seconds_n40000_one_pass"] = (
        raw["seconds_median"] * N_REAL / n_raw
    )
    raw["predicted_seconds_16_pcg_5_groups_12_sweeps"] = (
        raw["predicted_seconds_n40000_one_pass"] * 16 * KV_HEADS * 12
    )
    benchmarks.append(raw)
    del attended, outputs, value

    # Exact group-local effective seed solve (2H x 2H, 776 RHS).
    local_f = HEADS_PER_GROUP * HIDDEN
    seed_base = torch.randn(local_f, local_f, dtype=dtype)
    seed_gram = seed_base.T @ seed_base + 1e-2 * torch.eye(
        local_f, dtype=dtype
    )
    seed_rhs = torch.randn(local_f, OUTPUT, dtype=dtype)
    local_seed = timed(
        "group_local_seed_solve_f1552_o776",
        lambda: torch.linalg.solve(seed_gram, seed_rhs),
        warmups=0,
        repeats=1,
    )
    local_seed["predicted_seconds_five_groups"] = (
        local_seed["seconds_median"] * KV_HEADS
    )
    benchmarks.append(local_seed)
    del seed_base, seed_gram, seed_rhs

    # Exact group SVD used after the effective operator solve.
    svd_input = torch.randn(HIDDEN, HEADS_PER_GROUP * OUTPUT, dtype=dtype)
    svd = timed(
        "group_svd_776x1552",
        lambda: torch.linalg.svd(svd_input, full_matrices=False),
        warmups=0,
        repeats=1,
    )
    svd["predicted_seconds_five_groups"] = svd["seconds_median"] * KV_HEADS
    benchmarks.append(svd)
    del svd_input

    # A reduced solve scaling probe; full 7760 solve is intentionally not
    # attempted because the audit exists to avoid disrupting live work.
    solve_scaling = []
    for dimension in (512, 1024, 1536):
        base = torch.randn(dimension, dimension, dtype=dtype)
        matrix = base.T @ base + 1e-2 * torch.eye(dimension, dtype=dtype)
        target = torch.randn(dimension, min(OUTPUT, 128), dtype=dtype)
        probe = timed(
            f"primal_solve_f{dimension}_o{target.shape[1]}",
            lambda matrix=matrix, target=target: torch.linalg.solve(
                matrix, target
            ),
            warmups=0,
            repeats=1,
        )
        solve_scaling.append(probe)
        del base, matrix, target
    # Fit t = a*f^3 as a deliberately simple decomposition-dominated estimate.
    rates = [
        item["seconds_median"] / int(item["name"].split("_f")[1].split("_")[0]) ** 3
        for item in solve_scaling
    ]
    report["primal_solve_scaling"] = solve_scaling
    report["predicted_full_f7760_seconds_cubic_median"] = (
        statistics.median(rates) * flat_features**3
    )

    report["benchmarks"] = benchmarks
    report["analytic_flops"] = {
        "effective_gram": tflop(2 * N_REAL * flat_features**2),
        "effective_lu_factorization_approx": tflop(
            2 / 3 * flat_features**3
        ),
        "effective_multi_rhs_triangular_approx": tflop(
            2 * flat_features**2 * OUTPUT
        ),
        "cross_head_sufficient_stats": tflop(
            2
            * N_REAL
            * HIDDEN
            * HIDDEN
            * (QUERY_HEADS * (QUERY_HEADS + 1) // 2)
        ),
        "target_cross_sufficient_stats": tflop(
            2 * N_REAL * HIDDEN * OUTPUT * QUERY_HEADS
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
