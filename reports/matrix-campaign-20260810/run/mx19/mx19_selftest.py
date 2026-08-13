#!/usr/bin/env python3
"""Wave 19: prove the width fold's algebra before it is pointed at the parent.

Nine checks, none of which loads a model.  They exist because every one of them
is a claim the builder's docstring makes, and a claim in a docstring is worth
what its test is worth:

1. the transported gain is the parent's own gain at ``E = I``, entry for entry;
2. the reader fold at ``E = I`` returns the parent's weight BIT for bit, which
   is the whole reason the identity gate can be a byte comparison;
3. the writer fold at ``E = I`` likewise;
4. the reader fold is EXACT against the local target -- the child's post-norm
   rows through the folded reader equal the parent's post-norm rows at the
   lifted input through the parent's reader -- which is the load-bearing claim
   of the whole axis, and the one that makes the lift's norm preservation
   matter;
5. ``E`` from the top eigenvectors is orthonormal and its retained fraction is
   the cumulative spectrum, so the free census means what it says;
6. the writer fold is the L2-optimal write into the retained subspace (no
   other H x n matrix beats ``E^T W`` against ``E^T y``);
7. the gram-side coreset is the row-side one, index for index;
8. the final-norm solve's Hadamard normal equations reproduce a brute-force
   least squares on a small instance;
9. the axis's parameter arithmetic returns the parent at H = 1024.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx19_build as mx  # noqa: E402


def check(name: str, condition: bool, detail: str) -> bool:
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}", flush=True)
    return condition


def main() -> int:
    torch.manual_seed(1234)
    torch.set_grad_enabled(False)
    hidden, width, out_features = 64, 40, 24
    results: list[bool] = []

    # ---- a parent-like gain and weight, stored the way the parent stores them.
    gain = (torch.randn(hidden, dtype=torch.float64).abs() + 0.25).to(
        torch.bfloat16
    ).to(torch.float64)
    weight = torch.randn(out_features, hidden, dtype=torch.float64).to(
        torch.bfloat16
    ).to(torch.float64)
    identity = torch.eye(hidden, dtype=torch.float64)

    transported = mx.transported_gain(gain, identity)
    results.append(
        check(
            "transported gain at E=I",
            bool(torch.equal(transported, gain)),
            f"max |delta| {float((transported - gain).abs().max()):.1e}",
        )
    )

    folded = mx.fold_reader(weight, gain, identity, transported, 1.0)
    results.append(
        check(
            "reader fold at E=I is the parent's weight",
            bool(torch.equal(folded, weight))
            and bool(
                torch.equal(
                    folded.to(torch.bfloat16).view(torch.int16),
                    weight.to(torch.bfloat16).view(torch.int16),
                )
            ),
            f"max |delta| {float((folded - weight).abs().max()):.1e}, bf16 bits equal",
        )
    )

    written = mx.fold_writer(weight.T.contiguous(), identity)
    results.append(
        check(
            "writer fold at E=I is the parent's weight",
            bool(torch.equal(written, weight.T)),
            f"max |delta| {float((written - weight.T).abs().max()):.1e}",
        )
    )

    # ---- the exactness claim, on a real basis and real rows.
    sample = torch.randn(512, hidden, dtype=torch.float64) * 3.0
    moment = sample.T @ sample
    values, vectors = torch.linalg.eigh(moment)
    order = torch.argsort(values, descending=True)[:width]
    basis = vectors[:, order].contiguous()
    child_gain = mx.transported_gain(gain, basis)
    scale = (hidden / width) ** 0.5
    child_weight = mx.fold_reader(weight, gain, basis, child_gain, scale)
    eps = 1e-5

    rows = torch.randn(256, width, dtype=torch.float64) * 2.0
    lifted = rows @ basis.T
    parent_post = gain * (
        lifted * lifted.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
    )
    child_post = child_gain * (
        rows * rows.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
    )
    parent_out = parent_post @ weight.T
    child_out = child_post @ child_weight.T
    relative = float((child_out - parent_out).norm() / parent_out.norm())
    results.append(
        check(
            "reader fold is exact against the LOCAL target",
            relative < 5e-6,
            f"relL2 {relative:.3e} (the only gap is RMSNorm's eps, which the "
            f"two normalisations divide by different means)",
        )
    )
    lift_norm = float(
        (lifted.norm(dim=1) - rows.norm(dim=1)).abs().max() / rows.norm(dim=1).max()
    )
    results.append(
        check(
            "the lift preserves the row norm",
            lift_norm < 1e-14,
            f"max relative |delta| {lift_norm:.3e}",
        )
    )

    orthonormality = float(
        (basis.T @ basis - torch.eye(width, dtype=torch.float64)).abs().max()
    )
    spectrum = mx.spectrum(moment)
    retained = mx.retained_fraction(moment, basis)
    cumulative = float(torch.cumsum(spectrum, dim=0)[width - 1])
    results.append(
        check(
            "the census is the cumulative spectrum",
            orthonormality < 1e-12 and abs(retained - cumulative) < 1e-12,
            f"orthonormality {orthonormality:.1e}, retained {retained:.6f} vs "
            f"cumulative {cumulative:.6f}",
        )
    )

    # ---- the writer fold is optimal, not merely available.
    context = torch.randn(300, out_features, dtype=torch.float64)
    writer = torch.randn(hidden, out_features, dtype=torch.float64)
    target = (context @ writer.T) @ basis
    projected = mx.fold_writer(writer, basis)
    best = torch.linalg.lstsq(context, target).solution.T
    gap = float(
        ((context @ best.T - target).pow(2).sum() - (context @ projected.T - target).pow(2).sum())
        .abs()
    )
    results.append(
        check(
            "E^T W is the L2-optimal write into the retained subspace",
            gap < 1e-14 * float(target.pow(2).sum()),
            f"least-squares improvement over the projection {gap:.3e} on a target "
            f"of energy {float(target.pow(2).sum()):.3e}",
        )
    )

    features = torch.randn(400, hidden, dtype=torch.float64)
    row_side = mx.pivoted_columns(features, width)
    gram_side = mx.pivoted_columns_gram(features.T @ features, width)
    results.append(
        check(
            "the gram-side coreset is the row-side coreset",
            bool(torch.equal(row_side, gram_side)),
            f"{int((row_side != gram_side).sum())} of {width} indices differ",
        )
    )

    # ---- the final-norm solve's normal equations against brute force.
    vocab = 96
    embed = torch.randn(vocab, hidden, dtype=torch.float64)
    gram = embed.T @ embed
    final_gain = torch.randn(hidden, dtype=torch.float64).abs() + 0.5
    Z = torch.randn(64, width, dtype=torch.float64)
    N = torch.randn(64, hidden, dtype=torch.float64)
    AtA = basis.T @ gram @ basis
    Q = (basis.T @ gram) * final_gain[None, :]
    b = (Z * (N @ Q.T)).sum(dim=0)
    G = AtA * (Z.T @ Z)
    design = torch.stack(
        [(embed @ basis) @ torch.diag(z) for z in Z], dim=0
    ).reshape(-1, width)
    observed = ((N * final_gain) @ embed.T).reshape(-1)
    brute = torch.linalg.lstsq(design, observed).solution
    direct = torch.linalg.solve(G, b)
    results.append(
        check(
            "the Hadamard normal equations are the least squares",
            float((brute - direct).norm() / brute.norm()) < 1e-9,
            f"relative difference {float((brute - direct).norm() / brute.norm()):.3e}",
        )
    )

    predicted = mx.predicted_parameters(1024, excised=[], keep_groups=None)
    results.append(
        check(
            "the axis's parameter arithmetic returns the parent at H=1024",
            predicted == mx.PARENT_PARAMS,
            f"{predicted:,} vs {mx.PARENT_PARAMS:,}",
        )
    )
    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
