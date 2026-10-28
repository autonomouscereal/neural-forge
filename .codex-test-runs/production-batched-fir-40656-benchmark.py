#!/usr/bin/env python3
"""Production-shaped benchmark for the authenticated batched FIR solver."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any

import torch

try:
    import resource
except ImportError:  # pragma: no cover - benchmark production host is Linux.
    resource = None  # type: ignore[assignment]

from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    Lfm2CpcfV5AllChannelFirRidgeCase,
    Lfm2CpcfV5RidgeCase,
    Lfm2CpcfV5RidgePolicy,
    solve_condition_capped_batched_all_channel_fir_ridge_candidate,
    solve_condition_capped_equilibrated_prior_residual_ridge_candidate,
)


def _current_rss_bytes() -> int:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    if resource is not None:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return 0


def _resource_peak_rss_bytes() -> int:
    if resource is None:
        return 0
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _system_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", maxsplit=1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                result[f"{key}_bytes"] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return result


class _RssSampler:
    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_bytes = _current_rss_bytes()
        self.stage_peak_bytes: dict[str, int] = {}
        self.stage = "startup"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.sample()

    def sample(self) -> None:
        rss = _current_rss_bytes()
        self.peak_bytes = max(self.peak_bytes, rss)
        self.stage_peak_bytes[self.stage] = max(
            self.stage_peak_bytes.get(self.stage, 0),
            rss,
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.sample()

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        self._thread.join(timeout=2.0)


def _checkpoint(sampler: _RssSampler, stage: str, started: float) -> None:
    sampler.sample()
    payload = {
        "stage": stage,
        "elapsed_seconds": time.perf_counter() - started,
        "current_rss_gib": _current_rss_bytes() / 2**30,
        "sampled_peak_rss_gib": sampler.peak_bytes / 2**30,
        "resource_peak_rss_gib": _resource_peak_rss_bytes() / 2**30,
    }
    print(json.dumps(payload, sort_keys=True), flush=True)


def _tensor_payload_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _zero_mismatch_upper_bound(
    population: int,
    sample: int,
    *,
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Exact hypergeometric one-sided bound after zero sampled mismatches."""

    def log_probability_zero(bad_channels: int) -> float:
        if population - bad_channels < sample:
            return -math.inf
        total = 0.0
        for offset in range(sample):
            total += math.log(population - bad_channels - offset)
            total -= math.log(population - offset)
        return total

    largest_not_rejected = 0
    threshold = math.log(alpha)
    for bad_channels in range(population + 1):
        if log_probability_zero(bad_channels) >= threshold:
            largest_not_rejected = bad_channels
        else:
            break
    return {
        "confidence": 1.0 - alpha,
        "maximum_bad_channels_not_rejected": largest_not_rejected,
        "maximum_bad_fraction_not_rejected": largest_not_rejected / population,
    }


def _build_cases(
    *,
    channel_count: int,
    case_count: int,
    rows_per_case: int,
    feature_width: int,
    seed: int,
    sampler: _RssSampler,
    started: float,
) -> tuple[
    tuple[Lfm2CpcfV5AllChannelFirRidgeCase, ...],
    torch.Tensor,
]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    prior = (
        torch.randn(
            (channel_count, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.2
    )
    desired = (
        prior
        + torch.randn(
            (channel_count, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        * 0.35
    )
    cases = []
    for case_index in range(case_count):
        sampler.set_stage(f"setup_case_{case_index:02d}")
        design = torch.randn(
            (channel_count, rows_per_case, feature_width),
            generator=generator,
            dtype=torch.float64,
        )
        # A deterministic case-dependent scale gives the case balancer
        # non-identical systems while preserving ordinary FIR conditioning.
        design.mul_(0.75 + 0.05 * case_index)
        target = torch.bmm(design, desired.unsqueeze(-1)).squeeze(-1)
        noise = torch.randn(
            target.shape,
            generator=generator,
            dtype=torch.float64,
        )
        target.add_(noise, alpha=0.01)
        del noise
        case = Lfm2CpcfV5AllChannelFirRidgeCase(
            case_id=f"fit-{case_index:02d}",
            split="fit",
            design=design,
            target=target,
            valid_rows=None,
        )
        cases.append(case)
        del design, target
        gc.collect()
        _checkpoint(sampler, f"setup_case_{case_index:02d}_complete", started)
    del desired
    gc.collect()
    return tuple(cases), prior


def _scalar_reference_subset(
    *,
    cases: tuple[Lfm2CpcfV5AllChannelFirRidgeCase, ...],
    prior: torch.Tensor,
    result_parameters: torch.Tensor,
    condition_cap: float,
    policy: Lfm2CpcfV5RidgePolicy,
    sample_count: int,
    seed: int,
    sampler: _RssSampler,
    started: float,
) -> dict[str, Any]:
    channel_count = int(prior.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = (
        torch.randperm(channel_count, generator=generator)[:sample_count]
        .sort()
        .values
        .tolist()
    )
    scalar_parameters = []
    scalar_receipts = []
    sampler.set_stage("scalar_reference_subset")
    scalar_started = time.perf_counter()
    for ordinal, channel in enumerate(indices):
        scalar_cases = tuple(
            Lfm2CpcfV5RidgeCase(
                case_id=case.case_id,
                split="fit",
                design=case.design[channel],
                target=case.target[channel].unsqueeze(-1),
                valid_rows=None,
            )
            for case in cases
        )
        scalar = solve_condition_capped_equilibrated_prior_residual_ridge_candidate(
            scalar_cases,
            condition_cap=condition_cap,
            prior_weight=prior[channel].unsqueeze(-1),
            prior_bias=None,
            fit_intercept=False,
            policy=policy,
        )
        if not scalar.verify_authentication():
            raise RuntimeError(f"scalar reference channel {channel} failed authentication")
        scalar_parameters.append(scalar.weight[:, 0].clone())
        scalar_receipts.append(str(scalar.diagnostics["report_sha256"]))
        del scalar, scalar_cases
        if ordinal % 8 == 7:
            gc.collect()
            _checkpoint(
                sampler,
                f"scalar_reference_{ordinal + 1:03d}_of_{sample_count:03d}",
                started,
            )
    scalar_matrix = torch.stack(scalar_parameters)
    batched_matrix = result_parameters[indices].contiguous()
    difference = (batched_matrix - scalar_matrix).abs()
    tolerance = 5e-11
    fp64_mismatch_channels = (
        difference.max(dim=1).values > tolerance
    )
    batched_fp32 = batched_matrix.to(dtype=torch.float32)
    scalar_fp32 = scalar_matrix.to(dtype=torch.float32)
    batched_bf16 = batched_matrix.to(dtype=torch.bfloat16)
    scalar_bf16 = scalar_matrix.to(dtype=torch.bfloat16)
    fp32_channel_mismatches = (
        batched_fp32.view(sample_count, -1)
        != scalar_fp32.view(sample_count, -1)
    ).any(dim=1)
    bf16_channel_mismatches = (
        batched_bf16.view(sample_count, -1)
        != scalar_bf16.view(sample_count, -1)
    ).any(dim=1)
    elapsed = time.perf_counter() - scalar_started
    return {
        "sampling_method": (
            "fixed-seed simple random sample without replacement; "
            "chosen independently of solver outputs"
        ),
        "sampling_seed": seed,
        "population_channels": channel_count,
        "sample_count": sample_count,
        "sample_fraction": sample_count / channel_count,
        "sample_indices": indices,
        "scalar_wall_seconds": elapsed,
        "scalar_seconds_per_channel": elapsed / sample_count,
        "scalar_full_population_wall_seconds_extrapolated": (
            elapsed * channel_count / sample_count
        ),
        "all_scalar_results_authenticated": True,
        "scalar_report_sha256": scalar_receipts,
        "fp64_maximum_absolute_difference": float(difference.max().item()),
        "fp64_mean_absolute_difference": float(difference.mean().item()),
        "fp64_tolerance": tolerance,
        "fp64_mismatch_channel_count": int(fp64_mismatch_channels.sum().item()),
        "fp32_exact_equal": bool(torch.equal(batched_fp32, scalar_fp32)),
        "fp32_mismatch_channel_count": int(fp32_channel_mismatches.sum().item()),
        "fp32_batched_payload_sha256": _tensor_payload_sha256(batched_fp32),
        "fp32_scalar_payload_sha256": _tensor_payload_sha256(scalar_fp32),
        "bfloat16_exact_equal": bool(torch.equal(batched_bf16, scalar_bf16)),
        "bfloat16_mismatch_channel_count": int(bf16_channel_mismatches.sum().item()),
        "bfloat16_batched_payload_sha256": _tensor_payload_sha256(batched_bf16),
        "bfloat16_scalar_payload_sha256": _tensor_payload_sha256(scalar_bf16),
        "zero_mismatch_exact_hypergeometric_upper_bound": (
            _zero_mismatch_upper_bound(channel_count, sample_count)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=776)
    parser.add_argument("--cases", type=int, default=8)
    parser.add_argument("--rows-per-case", type=int, default=40_656)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--condition-cap", type=float, default=1_000.0)
    parser.add_argument("--scalar-sample", type=int, default=64)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20_260_719)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not (1 <= args.scalar_sample <= args.channels):
        raise ValueError("scalar-sample must be in [1, channels]")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    sampler = _RssSampler()
    sampler.start()
    memory_before = _system_memory()
    baseline_rss = _current_rss_bytes()
    _checkpoint(sampler, "startup", started)

    cases, prior = _build_cases(
        channel_count=args.channels,
        case_count=args.cases,
        rows_per_case=args.rows_per_case,
        feature_width=args.features,
        seed=args.seed,
        sampler=sampler,
        started=started,
    )
    setup_complete = time.perf_counter()
    input_bytes = sum(
        case.design.numel() * case.design.element_size()
        + case.target.numel() * case.target.element_size()
        for case in cases
    ) + prior.numel() * prior.element_size()

    policy = Lfm2CpcfV5RidgePolicy(
        parameter_trust_radius=0.8,
        effect_trust_radius=0.6,
    )
    sampler.set_stage("batched_solver")
    solve_started = time.perf_counter()
    result = solve_condition_capped_batched_all_channel_fir_ridge_candidate(
        cases,
        condition_cap=args.condition_cap,
        prior_parameters=prior,
        physical_bias_coordinate=True,
        policy=policy,
    )
    solve_complete = time.perf_counter()
    _checkpoint(sampler, "batched_solver_complete", started)

    sampler.set_stage("authentication")
    result_authenticated = bool(result.verify_authentication())
    result_finite = bool(torch.isfinite(result.parameters).all().item())
    cases_authenticated = all(case.verify_authentication() for case in cases)
    _checkpoint(sampler, "authentication_complete", started)

    scalar = _scalar_reference_subset(
        cases=cases,
        prior=prior,
        result_parameters=result.parameters,
        condition_cap=args.condition_cap,
        policy=policy,
        sample_count=args.scalar_sample,
        seed=args.seed + 91_337,
        sampler=sampler,
        started=started,
    )
    _checkpoint(sampler, "scalar_reference_complete", started)
    sampler.stop()

    resource_peak = _resource_peak_rss_bytes()
    measured_peak = max(resource_peak, sampler.peak_bytes)
    cap_bytes = 64 * 2**30
    report = {
        "schema": "neural-forge-production-batched-fir-benchmark-v1",
        "timestamp_unix": time.time(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "geometry": {
            "channel_count": args.channels,
            "case_count": args.cases,
            "rows_per_case": args.rows_per_case,
            "total_valid_rows_per_channel": args.cases * args.rows_per_case,
            "feature_width": args.features,
            "fir_length": args.features - 1,
            "physical_bias_coordinate": True,
            "authenticated_input_bytes": input_bytes,
            "authenticated_input_gib": input_bytes / 2**30,
        },
        "timing": {
            "setup_wall_seconds": setup_complete - started,
            "batched_solver_wall_seconds": solve_complete - solve_started,
            "authentication_wall_seconds": (
                time.perf_counter()
                - solve_complete
                - float(scalar["scalar_wall_seconds"])
            ),
            "total_wall_seconds": time.perf_counter() - started,
        },
        "memory": {
            "baseline_rss_bytes": baseline_rss,
            "baseline_rss_gib": baseline_rss / 2**30,
            "sampled_peak_rss_bytes": sampler.peak_bytes,
            "sampled_peak_rss_gib": sampler.peak_bytes / 2**30,
            "resource_peak_rss_bytes": resource_peak,
            "resource_peak_rss_gib": resource_peak / 2**30,
            "measured_peak_rss_bytes": measured_peak,
            "measured_peak_rss_gib": measured_peak / 2**30,
            "peak_increment_over_baseline_bytes": measured_peak - baseline_rss,
            "peak_increment_over_baseline_gib": (
                measured_peak - baseline_rss
            )
            / 2**30,
            "stage_peak_rss_bytes": sampler.stage_peak_bytes,
            "system_before": memory_before,
            "system_after": _system_memory(),
            "cap_bytes": cap_bytes,
            "cap_gib": 64.0,
            "margin_under_cap_bytes": cap_bytes - measured_peak,
            "margin_under_cap_gib": (cap_bytes - measured_peak) / 2**30,
            "within_64_gib_process_cap": measured_peak <= cap_bytes,
            "within_64_gib_with_10_percent_headroom": (
                measured_peak <= 0.9 * cap_bytes
            ),
        },
        "result": {
            "finite": result_finite,
            "authenticated": result_authenticated,
            "all_input_cases_authenticated": cases_authenticated,
            "shape": list(result.parameters.shape),
            "report_sha256": str(result.diagnostics["report_sha256"]),
            "solution_parameters_sha256": str(
                result.diagnostics["solution_parameters_sha256"]
            ),
            "condition_cap_passed": bool(
                result.diagnostics["condition_cap_passed"]
            ),
            "parameter_space_trust_passed": bool(
                result.diagnostics["parameter_space_trust"]["passed"]
            ),
            "function_space_trust_passed": bool(
                result.diagnostics["function_space_trust"]["passed"]
            ),
            "nullspace_preserves_prior": bool(
                result.diagnostics["nullspace_preserves_prior"]
            ),
            "scalar_oracle_channel_count_internal": int(
                result.diagnostics["scalar_oracle_policy"][
                    "oracle_channel_count"
                ]
            ),
            "scalar_oracle_maximum_absolute_difference_internal": float(
                result.diagnostics["scalar_oracle_policy"][
                    "maximum_absolute_batch_difference"
                ]
            ),
        },
        "scalar_reference_subset": scalar,
        "execution_contract": {
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "execution_device": "cpu",
            "execution_dtype": "float64",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
