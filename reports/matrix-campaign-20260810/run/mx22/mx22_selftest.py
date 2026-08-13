#!/usr/bin/env python3
"""Wave 22: prove the width fold's algebra before it is pointed at the parent.

Twenty-three checks, none of which loads a model.  They exist because every one
of them is a claim the builder's docstring makes, and a claim in a docstring is
worth what its test is worth.  **Wave 22 adds none of its own and that is the
point**: this wave adds no mechanism, so a new check here would be a check of
nothing, and what these twenty-three are for in wave 22 is to prove that the
copy of the builder this wave runs is the one waves 19-21 measured with.  Nine
are wave 19's, unchanged; seven are wave 20's, unchanged, and cover the two
objectives it added, the two row selectors and the writers' internal split;
seven are wave 21's, unchanged, and cover the max-min objective:

17. the endpoint moments are the blocks ``pooled_moment`` averages, each at
    trace one, so a retained fraction and a retained trace are one number;
18. the max-min iteration at step zero IS the plain reader-PCA basis, bit for
    bit, because uniform weights over trace-one blocks are the pooled moment --
    the same control wave 20's ``beta = 0`` provided, and the reason every
    max-min child is comparable to the sealed one;
19. the basis it selects is orthonormal, which the fold's norm-preserving lift
    requires;
20. it never loses on its own objective: the minimum per-endpoint retained read
    energy of the selected basis is at least the plain basis's, by construction
    of the best-iterate rule, and the check is that the rule is actually
    implemented;
21. it beats the plain basis on that minimum on an instance built to have a
    disagreeing endpoint, which is the case the objective exists for;
22. it PAYS for that in the mean, which is the trade the wave has to report
    honestly: the pooled retention of a max-min basis cannot exceed the pooled
    retention of the pooled basis;
23. the weights stay on the simplex over the whole iteration.

The wave-20 list, kept because those checks still run:

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
9. the axis's parameter arithmetic returns the parent at H = 1024;
10. the retrieval selector marks recurring bigrams and only those;
11. the procedural selector opens on a list item and spans exactly one window;
12. the attribution-weighted objective at ``beta = 0`` IS the plain reader-PCA
    basis, so every wave-19 comparison is against the same construction;
13. raising ``beta`` never lowers the retained case energy, which is the one
    thing the mixture exists to move;
14. the reader-generalised basis is Euclidean-orthonormal -- the property the
    norm-preserving lift needs and generalised eigenvectors do not have;
15. and it wins on its own reader-per-raw ratio against reader-PCA;
16. the width writers' internal split is every third fit batch, by position.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mx22_build as mx  # noqa: E402


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

    # ------------------------------------------------- wave 20's own additions
    # 10. the retrieval selector marks exactly the repeated bigrams, and nothing
    #     else.  Hand-built sequence: the pair (7, 8) occurs at positions 0 and
    #     4, so position 4 is the only marked row.
    ids = torch.tensor([[7, 8, 3, 9, 7, 8, 2]])
    mask = mx.retrieval_mask(ids)
    results.append(
        check(
            "the retrieval selector marks recurring bigrams and only those",
            [int(i) for i in mask.nonzero().reshape(-1)] == [4],
            f"marked {[int(i) for i in mask.nonzero().reshape(-1)]} of 7 positions",
        )
    )

    # 11. the procedural selector opens on a list item and closes a window later.
    pieces = ["intro", " text", "\n1.", " verify", " the", " code"] + [" pad"] * 60
    proc = mx.procedural_mask(pieces)
    opened = int(proc.nonzero().reshape(-1)[0])
    results.append(
        check(
            "the procedural selector opens at the list item and spans one window",
            opened == 2 and int(proc.sum()) == mx.PROCEDURE_WINDOW,
            f"opens at {opened}, covers {int(proc.sum())} of {len(pieces)} rows",
        )
    )

    # 12. beta = 0 IS the plain reader-PCA basis.  If the mixture is not exactly
    #     the identity of the objective at its own zero, every comparison this
    #     wave makes against wave 19 is against a different construction.
    ends = ("E0", "E1")
    fake = {
        "hidden": hidden,
        "endpoints": ends,
        "moments": {
            name: {
                split: {
                    kind: (lambda M: M @ M.T)(
                        torch.randn(hidden, hidden, dtype=torch.float64)
                    )
                    for kind in ("raw", "reader")
                }
                for split in ("fit", "hld")
            }
            for name in ends
        },
        "selector_moments": {
            which: {
                name: {
                    split: {
                        kind: (lambda M: M @ M.T)(
                            torch.randn(hidden, hidden, dtype=torch.float64)
                        )
                        for kind in ("raw", "reader")
                    }
                    for split in ("fit", "hld")
                }
                for name in ends
            }
            for which in mx.SELECTOR_CASES
        },
        "selector_report": None,
    }
    plain, _ = mx.build_basis(fake, width, "reader-pca")
    at_zero, _ = mx.build_basis(fake, width, "attribution-weighted", beta=0.0)
    results.append(
        check(
            "the attribution-weighted objective at beta=0 is the plain basis",
            float((plain - at_zero).abs().max()) < 1e-12,
            f"max |E_plain - E_beta0| = {float((plain - at_zero).abs().max()):.3e}",
        )
    )

    # 13. the mixture is monotone in the thing it is built to move: raising beta
    #     never lowers the retained case energy of the basis it produces.
    case = torch.zeros(hidden, hidden, dtype=torch.float64)
    weights = {"retrieval": 0.5, "procedural": 0.5}
    for which, weight in weights.items():
        case += weight * mx.pooled_selector_moment(fake, which, "reader", "fit")
    retained = [
        mx.retained_fraction(
            case, mx.build_basis(fake, width, "attribution-weighted", beta=b)[0]
        )
        for b in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    results.append(
        check(
            "raising beta never lowers the retained case energy",
            all(b >= a - 1e-12 for a, b in zip(retained, retained[1:])),
            " -> ".join(f"{value:.4f}" for value in retained),
        )
    )

    # 14. the generalised basis is Euclidean-orthonormal -- the property the
    #     norm-preserving lift needs and the property generalised eigenvectors
    #     do NOT have before `orthonormalise_span` puts it back.
    gen, gen_census = mx.build_basis(fake, width, "reader-generalized")
    offdiag = float(
        (gen.T @ gen - torch.eye(width, dtype=torch.float64)).abs().max()
    )
    results.append(
        check(
            "the reader-generalized basis is Euclidean-orthonormal",
            offdiag < 1e-10,
            f"max |E^T E - I| = {offdiag:.3e}",
        )
    )

    # 15. and it beats the plain basis at the ratio it was chosen for.  The
    #     generalised objective maximises reader energy PER UNIT of raw energy,
    #     so on its own criterion it must not lose to reader-PCA.
    read_pooled = mx.pooled_moment(fake, "reader", "fit")
    raw_pooled = mx.pooled_moment(fake, "raw", "fit")

    def ratio(E: torch.Tensor) -> float:
        return mx.retained_fraction(read_pooled, E) / mx.retained_fraction(raw_pooled, E)

    results.append(
        check(
            "the generalized basis wins on its own reader-per-raw ratio",
            ratio(gen) >= ratio(plain) - 1e-12,
            f"generalized {ratio(gen):.6f} vs reader-pca {ratio(plain):.6f}",
        )
    )

    # 16. the internal fit-side split is a POSITION rule: six of eighteen fit
    #     batches, the same rows for every site, and disjoint from held-out by
    #     construction because it only ever indexes fit rows.
    batches = [(i, torch.zeros(1, 264, dtype=torch.long)) for i in range(24)]
    split = {i: ("fit" if i % 4 else "hld") for i in range(24)}
    rows = mx.fit_check_rows(batches, split)
    fit_batches = sum(1 for i in range(24) if split[i] == "fit")
    results.append(
        check(
            "the width writers' internal split is every third fit batch",
            int(rows.shape[0]) == fit_batches * 264
            and int(rows.sum()) == (fit_batches + 1) // 3 * 264,
            f"{int(rows.sum())} check rows of {int(rows.shape[0])} fit rows "
            f"({fit_batches} fit batches)",
        )
    )
    # ---- wave 21: the max-min objective, on an instance built so the two
    #      endpoints genuinely disagree about which directions matter.  A random
    #      pair would be nearly interchangeable at this size and the objective
    #      would have nothing to do; these two are deliberately misaligned, with
    #      one of them carrying a subspace the pooled moment under-weights.
    # Two ORTHOGONAL frames with the same spectrum.  A random pair of Wishart
    # matrices is not an instance of this problem -- both inherit the same
    # Marchenko-Pastur tilt and end up agreeing, which is how the first draft of
    # this check passed vacuously and then failed honestly.  Two independent
    # orthonormal frames disagree about WHICH directions carry the energy while
    # agreeing about how much there is, which is exactly the disagreement the
    # max-min objective exists to arbitrate.
    ramp = torch.diag(torch.linspace(1.0, 0.01, hidden, dtype=torch.float64))
    frame_a, _ = torch.linalg.qr(torch.randn(hidden, hidden, dtype=torch.float64))
    frame_b, _ = torch.linalg.qr(torch.randn(hidden, hidden, dtype=torch.float64))
    strong = frame_a @ ramp @ frame_a.T
    weak = frame_b @ ramp @ frame_b.T
    skewed = {
        "hidden": hidden,
        "endpoints": ("E0", "E1"),
        "moments": {
            "E0": {
                split: {"raw": strong.clone(), "reader": strong.clone()}
                for split in ("fit", "hld")
            },
            "E1": {
                split: {"raw": weak.clone(), "reader": weak.clone()}
                for split in ("fit", "hld")
            },
        },
        "selector_moments": None,
        "selector_report": None,
    }

    # 17. the endpoint blocks are what the pool averages, each at trace one.
    blocks = mx.endpoint_moments(skewed, "reader", "fit")
    traces = [float(torch.diagonal(block).sum()) for block in blocks.values()]
    mean_block = sum(blocks.values()) / len(blocks)
    results.append(
        check(
            "the endpoint moments are the pool's own blocks at trace one",
            max(abs(t - 1.0) for t in traces) < 1e-12
            and float(
                (mean_block - mx.pooled_moment(skewed, "reader", "fit")).abs().max()
            )
            < 1e-15,
            f"traces {['%.6f' % t for t in traces]}, mean equals pooled_moment",
        )
    )

    # 18. step zero is the plain basis, bit for bit.
    plain_skew, _ = mx.build_basis(skewed, width, "reader-pca")
    step_zero, _ = mx.build_basis(skewed, width, "reader-maxmin", maxmin_iters=0)
    results.append(
        check(
            "the max-min iteration at step 0 is the plain reader-PCA basis",
            bool(torch.equal(plain_skew, step_zero)),
            f"max |E_plain - E_step0| = "
            f"{float((plain_skew - step_zero).abs().max()):.3e}",
        )
    )

    maxmin, mm_census = mx.build_basis(
        skewed, width, "reader-maxmin", maxmin_iters=120, maxmin_eta=64.0
    )
    objective = mm_census["objective"]

    # 19. orthonormal, because the lift's norm preservation needs it.
    offdiag = float(
        (maxmin.T @ maxmin - torch.eye(width, dtype=torch.float64)).abs().max()
    )
    results.append(
        check(
            "the max-min basis is Euclidean-orthonormal",
            offdiag < 1e-10,
            f"max |E^T E - I| = {offdiag:.3e}",
        )
    )

    def per_endpoint(E: torch.Tensor) -> list[float]:
        return [float((E.T @ block @ E).diagonal().sum()) for block in blocks.values()]

    plain_min = min(per_endpoint(plain_skew))
    maxmin_min = min(per_endpoint(maxmin))

    # 20. the best-iterate rule is actually implemented: the selected basis is
    #     never worse than the uniform one on the objective it selects for.
    results.append(
        check(
            "the max-min basis never loses to the plain basis on the minimum",
            maxmin_min >= plain_min - 1e-12
            and abs(objective["min_retained_fit_selected"] - maxmin_min) < 1e-12,
            f"plain min {plain_min:.6f} -> max-min {maxmin_min:.6f} "
            f"(census says {objective['min_retained_fit_selected']:.6f})",
        )
    )

    # 21. and on an instance whose endpoints disagree it strictly wins, which is
    #     the whole reason to build it.
    results.append(
        check(
            "the max-min basis strictly lifts the worst endpoint when they differ",
            maxmin_min > plain_min + 1e-6,
            f"worst endpoint {plain_min:.6f} -> {maxmin_min:.6f} "
            f"(+{maxmin_min - plain_min:.6f}), argmin at uniform "
            f"{objective['worst_endpoint_uniform']}; per-endpoint plain "
            f"{['%.6f' % v for v in per_endpoint(plain_skew)]} -> max-min "
            f"{['%.6f' % v for v in per_endpoint(maxmin)]}",
        )
    )

    # 22. and it PAYS for it in the mean.  The pooled basis maximises the mean
    #     by construction, so a max-min basis that also beat it pooled would mean
    #     one of the two eigensolves is wrong.
    pooled_skew = mx.pooled_moment(skewed, "reader", "fit")
    plain_pooled = mx.retained_fraction(pooled_skew, plain_skew)
    maxmin_pooled = mx.retained_fraction(pooled_skew, maxmin)
    results.append(
        check(
            "the max-min basis cannot beat the pooled basis pooled",
            maxmin_pooled <= plain_pooled + 1e-12,
            f"pooled retention {plain_pooled:.6f} -> {maxmin_pooled:.6f} "
            f"({maxmin_pooled - plain_pooled:+.6f}), the price of the minimum",
        )
    )

    # 23. the endpoint weights are a probability vector at every step.
    chosen = list(objective["weights_selected"].values())
    results.append(
        check(
            "the endpoint weights stay on the simplex",
            abs(sum(chosen) - 1.0) < 1e-12 and min(chosen) >= 0.0,
            f"sum {sum(chosen):.12f}, min {min(chosen):.3e}, "
            f"{len(chosen)} endpoints",
        )
    )

    print(f"\n{sum(results)}/{len(results)} checks passed", flush=True)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
