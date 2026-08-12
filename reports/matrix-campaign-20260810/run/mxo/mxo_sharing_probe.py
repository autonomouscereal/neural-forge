#!/usr/bin/env python3
"""M3, bounded: what a cross-layer shared basis can buy at EQUAL storage.

ADDENDUM 9 §9.4 registers cross-layer shared-basis folds as a wave-11 mechanism.
The July estate already closed the naive form of it -- whole-FFN sharing, coupled
neuron banking, and a single shared residual output basis were all rejected at
the 15-20% targets (`docs/DYNAMIC_LFM2_RESULTS_2026-07-16.md`) -- so what is
worth measuring before spending a child on it is the thing those runs never
stated: **the ceiling**.  Sharing does not make a fold better, it makes a fold
*cheaper per layer*, and the whole benefit is the extra rank that buys.

For a component of shape (m, n) replicated across L layers:

    independent rank r_i   stores  r_i * (m + n)              per layer
    shared n-basis rank r_s stores  r_s * m  +  r_s * n / L    per layer

so at equal storage

    r_s / r_i  =  (m + n) / (m + n/L)   ->   1 + n/m   as L -> infinity

This probe (a) prints that arithmetic for every component class in the parent,
and (b) runs the **falsification half** of the experiment in weight space: at
equal storage, does one shared basis across all fourteen `w1` matrices
reconstruct them better than fourteen independent truncations?

It is a falsification probe and says so.  The campaign's own finding 3.5 is that
reconstruction does not predict behavior, so a *win* here would license nothing
-- but a *loss* here, in the metric most favourable to sharing (no activation
weighting, no composition, no depth), closes the mechanism without spending a
GPU-day on a child.  The solve is closed form throughout: one truncated SVD per
layer for the independent arm, and for the shared arm an alternating
least-squares over (basis, per-layer coefficients) which is the same alternating
closed form ADDENDUM 9 registers, applied to weights instead of activations.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch


def _load_safetensors(path: Path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    with open(path, "rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
        offset = 8 + length
        out: dict[str, torch.Tensor] = {}
        for key, meta in header.items():
            if key == "__metadata__" or not key.endswith(prefixes):
                continue
            start, end = meta["data_offsets"]
            handle.seek(offset + start)
            raw = handle.read(end - start)
            dtype = {"F32": torch.float32, "BF16": torch.bfloat16}[meta["dtype"]]
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(
                meta["shape"]
            )
            out[key] = tensor.to(torch.float64)
    return out


def _independent(weight: torch.Tensor, rank: int) -> torch.Tensor:
    u, s, vh = torch.linalg.svd(weight, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank]


def _shared(weights: list[torch.Tensor], rank: int, sweeps: int) -> list[torch.Tensor]:
    """One shared right-basis, per-layer coefficients, alternating closed form."""

    stacked = torch.cat(weights, dim=0)
    _u, _s, vh = torch.linalg.svd(stacked, full_matrices=False)
    basis = vh[:rank]                      # (rank, n)
    for _ in range(max(0, sweeps)):
        # coefficients: least squares of each weight onto the current basis
        coefficients = [w @ basis.T @ torch.linalg.inv(basis @ basis.T) for w in weights]
        # basis: least squares over every layer at once
        left = sum(c.T @ c for c in coefficients)
        right = sum(c.T @ w for c, w in zip(coefficients, weights))
        basis = torch.linalg.solve(left, right)
    coefficients = [w @ basis.T @ torch.linalg.inv(basis @ basis.T) for w in weights]
    return [c @ basis for c in coefficients]


def _relative(approximation: torch.Tensor, reference: torch.Tensor) -> float:
    return float((approximation - reference).norm() / reference.norm())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--component", default="feed_forward.w1.weight")
    parser.add_argument("--level-ratio", type=float, default=0.25)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    weights = _load_safetensors(
        arguments.model / "model.safetensors", (arguments.component,)
    )
    ordered = [weights[key] for key in sorted(weights, key=lambda k: int(k.split(".")[2]))]
    layers = len(ordered)
    rows, columns = ordered[0].shape
    budget = int(arguments.level_ratio * rows * columns)

    independent_rank = budget // (rows + columns)
    # r_s * rows + r_s * columns / layers <= budget
    shared_rank = int(budget / (rows + columns / layers))

    report = {
        "schema": "neural-forge-matrix-wave11-sharing-probe-v1",
        "component": arguments.component,
        "layers": layers,
        "shape": [rows, columns],
        "level_ratio": arguments.level_ratio,
        "storage_budget_per_layer": budget,
        "independent_rank": independent_rank,
        "shared_rank": shared_rank,
        "rank_uplift": shared_rank / independent_rank,
        "asymptotic_uplift_infinite_layers": 1 + columns / rows,
        "alternations": arguments.sweeps,
        "metric": "weight-space relative Frobenius (NO activation weighting)",
        "caveat": (
            "A win here licenses nothing -- reconstruction does not predict "
            "behavior (FOLDING_SCALING_THEORY 3.5).  A loss here closes the "
            "mechanism, because this is the metric most favourable to sharing."
        ),
    }

    shared = _shared(ordered, shared_rank, arguments.sweeps)
    per_layer = []
    for index, weight in enumerate(ordered):
        independent = _independent(weight, independent_rank)
        per_layer.append(
            {
                "layer": index,
                "independent_relative_frobenius": _relative(independent, weight),
                "shared_relative_frobenius": _relative(shared[index], weight),
            }
        )
        print(
            "layer %2d  independent r=%d %.6f   shared r=%d %.6f  %s"
            % (
                index,
                independent_rank,
                per_layer[-1]["independent_relative_frobenius"],
                shared_rank,
                per_layer[-1]["shared_relative_frobenius"],
                "SHARED WINS"
                if per_layer[-1]["shared_relative_frobenius"]
                < per_layer[-1]["independent_relative_frobenius"]
                else "independent wins",
            )
        )
    report["per_layer"] = per_layer
    report["mean_independent"] = sum(
        r["independent_relative_frobenius"] for r in per_layer
    ) / layers
    report["mean_shared"] = sum(r["shared_relative_frobenius"] for r in per_layer) / layers
    report["shared_wins_layers"] = sum(
        1
        for r in per_layer
        if r["shared_relative_frobenius"] < r["independent_relative_frobenius"]
    )
    print(
        "\nmean independent %.6f | mean shared %.6f | shared wins %d/%d layers"
        % (
            report["mean_independent"],
            report["mean_shared"],
            report["shared_wins_layers"],
            layers,
        )
    )
    print(
        "rank uplift at equal storage: %d -> %d (%.3fx), asymptote %.3fx"
        % (
            independent_rank,
            shared_rank,
            report["rank_uplift"],
            report["asymptotic_uplift_infinite_layers"],
        )
    )
    arguments.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
