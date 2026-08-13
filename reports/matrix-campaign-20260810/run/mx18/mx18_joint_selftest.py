#!/usr/bin/env python3
"""Gate the wave-18 joint head-merge solve on its own algebra, before any child.

ADDENDUM 14 §14.3 claims three things about `mx18_build.joint_relation_solve`
and the two-sided value solve beside it, and each one is either true of the code
or the whole merge is a guess:

1. **The rank-one relation solve is a solve.**  Fed two identical heads it must
   return that head's relation exactly (residual at the fp64 floor), and fed two
   different ones it must not be beaten by the average-in-gauge merge on the
   objective it optimises.  The first is an identity gate; the second is the
   entire reason the wave rebuilds the merge at all.
2. **The conjugate-gradient two-sided solve is T7's primitive.**  With one
   Kronecker term the normal equations ARE ``A S B + λS = E``, which
   `stein_two_sided_eigh` solves in closed form with two ``eigh``.  The two must
   agree to solver tolerance, or the CG path is solving something else.
3. **The direction-anchored recovery hits its target.**  Given rows that are
   exactly achievable, the recovered weight must reproduce them through the
   RMS-norm-and-gain the score actually applies.

Wave 18 adds two more, both about the lambda fix `NOTES-MX-U` §7 named:

4. **The check objective is the sum of squares it stands for.**  The split
   scores a candidate on rows it never saw from accumulated factors alone; if
   that closed form is not the direct residual, the selection selects on noise
   of its own making.
5. **The split generalises better than THE LAW's own pick**, on a problem built
   to punish fitting the noise -- on both sides of the alternation, the
   two-sided Stein solve and the ordinary ridge.  A fix that cannot beat the
   thing it replaces on a case constructed for it will not beat it on a model.

Nothing here touches the model, the corpus or a GPU: it is pure algebra on
seeded synthetic tensors, it runs in seconds, and it fails loudly.

    python mx18_joint_selftest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mx18_build import (  # noqa: E402
    LAMBDA_LADDER,
    as_planes,
    direction_anchored_solve,
    joint_relation_solve,
    law_abiding_solve,
    law_abiding_solve_split,
    law_abiding_stein,
    law_abiding_stein_split,
    relation_error_of,
    sign_gauge,
    stein_cg_solve,
    stein_factors,
    stein_objective,
    stein_two_sided_eigh,
)

HEAD_DIM = 64
ROWS = 512
HIDDEN = 96


def post_norm(rows: torch.Tensor, gain: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    scale = rows.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
    return rows * scale * gain


def case_identical_heads() -> dict[str, float]:
    """Two copies of one head: the joint solve must return that head."""

    torch.manual_seed(17)
    q = torch.randn(ROWS, HEAD_DIM, dtype=torch.float64)
    k = torch.randn(ROWS, HEAD_DIM, dtype=torch.float64)
    rows, key, diagnostics = joint_relation_solve([[q, q.clone()]], [k, k.clone()], [1.0, 1.0], HEAD_DIM)
    # The relation is what must be reproduced; the q/k scale split is a gauge.
    left = as_planes(rows[0], HEAD_DIM)
    right = as_planes(key, HEAD_DIM)
    ref_left = as_planes(q, HEAD_DIM)
    ref_right = as_planes(k, HEAD_DIM)
    error = 0.0
    reference = 0.0
    for plane in range(HEAD_DIM // 2):
        got = torch.outer(left[:, plane], right[:, plane].conj())
        want = torch.outer(ref_left[:, plane], ref_right[:, plane].conj())
        error += float((got - want).abs().pow(2).sum())
        reference += float(want.abs().pow(2).sum())
    return {
        "relation_relL2_vs_the_one_parent": (error / reference) ** 0.5,
        "solver_reported_relL2": diagnostics["relation_relL2"],
    }


def case_two_heads_beats_the_average() -> dict[str, float]:
    """Two different heads: the solve must beat averaging on the objective."""

    torch.manual_seed(1234)
    X = torch.randn(ROWS, HIDDEN, dtype=torch.float64)
    gq = torch.rand(HEAD_DIM, dtype=torch.float64) + 0.5
    gk = torch.rand(HEAD_DIM, dtype=torch.float64) + 0.5
    shared = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    out = {}
    for label, mix in (("near", 0.15), ("far", 1.0)):
        wq_a = shared + mix * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
        wq_b = shared + mix * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
        wk_a = shared + mix * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
        wk_b = shared + mix * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
        q_a, q_b = post_norm(X @ wq_a.T, gq), post_norm(X @ wq_b.T, gq)
        k_a, k_b = post_norm(X @ wk_a.T, gk), post_norm(X @ wk_b.T, gk)
        s = sign_gauge(k_a, k_b, HEAD_DIM)
        parents_q = [[q_a, q_b * s]]
        parents_k = [k_a, k_b * s]
        energy = [float(k_a.pow(2).sum()), float(k_b.pow(2).sum())]
        average_q = post_norm(
            X @ ((energy[0] * wq_a + energy[1] * s.unsqueeze(1) * wq_b) / sum(energy)).T, gq
        )
        average_k = post_norm(
            X @ ((energy[0] * wk_a + energy[1] * s.unsqueeze(1) * wk_b) / sum(energy)).T, gk
        )
        target_q, target_k, _ = joint_relation_solve(parents_q, parents_k, energy, HEAD_DIM)
        out[f"{label}_average"] = relation_error_of(
            [average_q], average_k, parents_q, parents_k, energy, HEAD_DIM
        )
        out[f"{label}_joint"] = relation_error_of(
            target_q, target_k, parents_q, parents_k, energy, HEAD_DIM
        )
    return out


def case_cg_is_the_stein_primitive() -> dict[str, float]:
    """One Kronecker term: CG must reproduce the two-``eigh`` closed form."""

    torch.manual_seed(7)
    A = torch.randn(HEAD_DIM, HEAD_DIM, dtype=torch.float64)
    A = A @ A.T + 0.1 * torch.eye(HEAD_DIM, dtype=torch.float64)
    B = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
    B = B @ B.T + 0.1 * torch.eye(HIDDEN, dtype=torch.float64)
    E = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    prior = torch.zeros(HEAD_DIM, HIDDEN, dtype=torch.float64)
    lam = 1e-3
    closed = stein_two_sided_eigh(A, B, E, lam)
    # Two iterations, not four hundred: with one term the preconditioner is the
    # operator, so a CG that needs more than a step is not solving this system.
    cg, report = stein_cg_solve({(0, 0): A}, {(0, 0): B}, E, prior, lam, iterations=2, tolerance=1e-14)
    return {
        "condition_number_product": float(torch.linalg.cond(A) * torch.linalg.cond(B)),
        "cg_vs_eigh_relL2": float((cg - closed).norm() / closed.norm()),
        "cg_iterations": report["cg_iterations"],
        "cg_relative_residual": report["cg_relative_residual"],
    }


def case_two_term_normal_equations() -> dict[str, float]:
    """Two query heads in a group: CG must satisfy its own normal equations."""

    torch.manual_seed(31)
    heads = [0, 1]
    Wo = {h: torch.randn(HIDDEN, HEAD_DIM, dtype=torch.float64) for h in heads}
    Z = {h: torch.randn(ROWS, HIDDEN, dtype=torch.float64) for h in heads}
    truth = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    Y = sum(Z[h] @ truth.T @ Wo[h].T for h in heads)
    left = {(h, hp): Wo[h].T @ Wo[hp] for h in heads for hp in heads}
    right = {(h, hp): Z[hp].T @ Z[h] for h in heads for hp in heads}
    rhs = sum(Wo[h].T @ (Y.T @ Z[h]) for h in heads)
    # A real prior, because THE LAW's trust region is measured against one and a
    # zero prior would exercise a path the builder never takes.
    prior = truth + 0.2 * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    solved, report = law_abiding_stein(left, right, rhs, prior)
    predicted = sum(Z[h] @ solved.T @ Wo[h].T for h in heads)
    prior_predicted = sum(Z[h] @ prior.T @ Wo[h].T for h in heads)
    return {
        "recovered_relL2_vs_truth": float((solved - truth).norm() / truth.norm()),
        "output_relL2": float((predicted - Y).norm() / Y.norm()),
        "prior_output_relL2": float((prior_predicted - Y).norm() / Y.norm()),
        "cg_iterations": report["cg_iterations"],
        "cg_relative_residual": report["cg_relative_residual"],
        "update_scale": report["update_scale"],
        "lambda_relative": report["lambda_relative"],
    }


def case_direction_recovery() -> dict[str, float]:
    """An achievable target: the recovered weight must reach it post-norm."""

    torch.manual_seed(99)
    X = torch.randn(ROWS, HIDDEN, dtype=torch.float64)
    gain = torch.rand(HEAD_DIM, dtype=torch.float64) + 0.5
    truth = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    prior = truth + 0.5 * torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    target = post_norm(X @ truth.T, gain)
    solved, guard = direction_anchored_solve(X, target, gain, prior, 1e-5)
    achieved = post_norm(X @ solved.T, gain)
    return {
        "post_norm_cos": float(
            torch.nn.functional.cosine_similarity(achieved, target, dim=1).mean()
        ),
        "reported_post_norm_cos": guard["post_norm_cos_vs_target"],
        "prior_post_norm_cos": float(
            torch.nn.functional.cosine_similarity(
                post_norm(X @ prior.T, gain), target, dim=1
            ).mean()
        ),
        "update_scale": guard["update_scale"],
    }


def two_sided_problem(
    rows: int, noise: float, seed: int
) -> tuple[dict, dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One group's two-sided problem: two query heads, a truth, and noise on Y."""

    torch.manual_seed(seed)
    heads = [0, 1]
    Wo = {h: torch.randn(HIDDEN, HEAD_DIM, dtype=torch.float64) for h in heads}
    Z = {h: torch.randn(rows, HIDDEN, dtype=torch.float64) for h in heads}
    truth = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    clean = sum(Z[h] @ truth.T @ Wo[h].T for h in heads)
    Y = clean + noise * clean.std() * torch.randn_like(clean)
    return Wo, Z, truth, Y, clean


def factors_of(
    Wo: dict, Z: dict, Y: torch.Tensor
) -> tuple[dict, dict, torch.Tensor, float]:
    heads = sorted(Z)
    left = {(h, hp): Wo[h].T @ Wo[hp] for h in heads for hp in heads}
    right = {(h, hp): Z[hp].T @ Z[h] for h in heads for hp in heads}
    rhs = sum(Wo[h].T @ (Y.T @ Z[h]) for h in heads)
    return left, right, rhs, float(Y.pow(2).sum())


def case_objective_is_the_sum_of_squares() -> dict[str, float]:
    """The closed-form check objective must equal the sum of squares it stands for.

    This is what makes the internal split cheap: a candidate is scored on rows it
    never saw without one attention probability being recomputed.  If the
    identity is wrong the selection is selecting on a number that means nothing.
    """

    Wo, Z, truth, Y, _ = two_sided_problem(ROWS, 0.3, seed=4242)
    left, right, rhs, reference = factors_of(Wo, Z, Y)
    candidate = truth + 0.4 * torch.randn_like(truth)
    direct = float((sum(Z[h] @ candidate.T @ Wo[h].T for h in Z) - Y).pow(2).sum())
    closed = stein_objective(left, right, rhs, reference, candidate)
    return {
        "direct_sse": direct,
        "closed_form_sse": closed,
        "relative_gap": abs(direct - closed) / direct,
    }


def case_hoisted_factors_are_the_same_solve() -> dict[str, float]:
    """The hoisted ``eigh`` must reproduce the in-solve one bit for bit."""

    Wo, Z, _, Y, _ = two_sided_problem(ROWS, 0.1, seed=808)
    left, right, rhs, _ = factors_of(Wo, Z, Y)
    prior = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    lam = 1e-3 * float(torch.diagonal(right[(0, 0)]).mean())
    plain, _ = stein_cg_solve(left, right, rhs, prior, lam)
    hoisted, _ = stein_cg_solve(
        left, right, rhs, prior, lam, factors=stein_factors(left, right)
    )
    return {"max_abs_delta": float((plain - hoisted).abs().max())}


def case_split_selects_out_of_sample() -> dict[str, Any]:
    """The named fix, on a problem built to punish selecting on the fit alone.

    Sixty inner rows against a 64x96 unknown and a prior that is already close:
    the smallest lambda on the ladder fits the noise, and THE LAW admits it
    because the update is small in norm.  A lambda chosen on thirty rows the
    solve never saw should generalise better on five hundred it never saw
    either -- and if it does not, the fix is not a fix.
    """

    Wo, Z, truth, Y, _ = two_sided_problem(90, 0.55, seed=1301)
    inner = {h: Z[h][:60] for h in Z}
    check = {h: Z[h][60:] for h in Z}
    left, right, rhs, _ = factors_of(Wo, Z, Y)
    left_in, right_in, rhs_in, _ = factors_of(Wo, inner, Y[:60])
    left_ck, right_ck, rhs_ck, ref_ck = factors_of(Wo, check, Y[60:])
    prior = truth + 0.35 * torch.randn_like(truth)

    torch.manual_seed(99)
    Z_true = {h: torch.randn(512, HIDDEN, dtype=torch.float64) for h in Z}
    Y_true = sum(Z_true[h] @ truth.T @ Wo[h].T for h in Z)

    def truth_sse(V: torch.Tensor) -> float:
        return float((sum(Z_true[h] @ V.T @ Wo[h].T for h in Z) - Y_true).pow(2).sum())

    law_solved, law_guard = law_abiding_stein(left, right, rhs, prior)
    split_solved, split_guard = law_abiding_stein_split(
        left, right, rhs, (right_in, rhs_in), (right_ck, rhs_ck, ref_ck), prior
    )
    admitted = [
        row for row in split_guard["ladder"] if (row["update_scale"] or 2.0) <= 1.0
    ]
    return {
        "law_lambda": law_guard["lambda_relative"],
        "split_lambda": split_guard["lambda_relative"],
        "law_would_have_picked": split_guard["lambda_law_would_have_picked"],
        "law_heldout_sse": truth_sse(law_solved),
        "split_heldout_sse": truth_sse(split_solved),
        "prior_heldout_sse": truth_sse(prior),
        "chosen_is_the_check_minimum": bool(
            admitted
            and split_guard["check_sse_chosen"]
            == min(row["check_sse"] for row in admitted)
        ),
        "ladder": split_guard["ladder"],
    }


def case_split_ridge_selects_out_of_sample() -> dict[str, Any]:
    """The same fix on the ``out_proj`` side, which is an ordinary ridge."""

    torch.manual_seed(1307)
    A = torch.randn(150, HIDDEN, dtype=torch.float64)
    truth = torch.randn(HIDDEN, HIDDEN, dtype=torch.float64)
    clean = A @ truth.T
    Y = clean + 0.6 * clean.std() * torch.randn_like(clean)
    prior = truth + 0.3 * torch.randn_like(truth)
    check = torch.zeros(A.shape[0], dtype=torch.bool)
    check[2::3] = True

    A_true = torch.randn(512, HIDDEN, dtype=torch.float64)
    Y_true = A_true @ truth.T

    def truth_sse(W: torch.Tensor) -> float:
        return float((A_true @ W.T - Y_true).pow(2).sum())

    law_solved, law_guard = law_abiding_solve(A, Y, prior)
    split_solved, split_guard = law_abiding_solve_split(A, Y, check, prior)
    return {
        "law_lambda": law_guard["lambda_relative"],
        "split_lambda": split_guard["lambda_relative"],
        "law_heldout_sse": truth_sse(law_solved),
        "split_heldout_sse": truth_sse(split_solved),
        "prior_heldout_sse": truth_sse(prior),
        "ladder_length": len(split_guard["ladder"]),
        "ladder_is_the_whole_ladder": len(split_guard["ladder"]) == len(LAMBDA_LADDER),
    }


def main() -> int:
    torch.set_grad_enabled(False)
    report = {
        "identical_heads": case_identical_heads(),
        "beats_the_average": case_two_heads_beats_the_average(),
        "cg_is_stein": case_cg_is_the_stein_primitive(),
        "two_term": case_two_term_normal_equations(),
        "direction_recovery": case_direction_recovery(),
        "objective_identity": case_objective_is_the_sum_of_squares(),
        "hoisted_factors": case_hoisted_factors_are_the_same_solve(),
        "split_two_sided": case_split_selects_out_of_sample(),
        "split_ridge": case_split_ridge_selects_out_of_sample(),
    }
    print(json.dumps(report, indent=1))

    failures = []
    if report["identical_heads"]["relation_relL2_vs_the_one_parent"] > 1e-10:
        failures.append("the identity gate: two copies of a head do not return the head")
    for label in ("near", "far"):
        if report["beats_the_average"][f"{label}_joint"] > report["beats_the_average"][f"{label}_average"]:
            failures.append(f"{label}: the joint solve loses to averaging on its own objective")
    if report["cg_is_stein"]["cg_vs_eigh_relL2"] > 1e-8:
        failures.append("the CG solve disagrees with T7's two-eigh closed form")
    if report["two_term"]["output_relL2"] > 1e-4:
        failures.append("the two-term solve does not reproduce its own construction")
    if report["two_term"]["output_relL2"] >= report["two_term"]["prior_output_relL2"]:
        failures.append("the two-term solve is no better than the prior it anchors on")
    if report["cg_is_stein"]["cg_iterations"] > 2:
        failures.append("preconditioned CG needed more than one step on a single term")
    if report["direction_recovery"]["post_norm_cos"] < 0.99:
        failures.append("the direction-anchored recovery misses an achievable target")
    if report["direction_recovery"]["post_norm_cos"] <= report["direction_recovery"]["prior_post_norm_cos"]:
        failures.append("the recovery is no better than the prior it started from")
    if report["objective_identity"]["relative_gap"] > 1e-10:
        failures.append(
            "the closed-form check objective is not the sum of squares it stands for"
        )
    if report["hoisted_factors"]["max_abs_delta"] != 0.0:
        failures.append("the hoisted eigh does not reproduce the in-solve one exactly")
    if not report["split_two_sided"]["chosen_is_the_check_minimum"]:
        failures.append(
            "the two-sided split did not choose the admitted rung with the lowest "
            "check objective"
        )
    if report["split_two_sided"]["split_heldout_sse"] >= report["split_two_sided"]["law_heldout_sse"]:
        failures.append(
            "the two-sided split does not generalise better than THE LAW's own pick "
            "on a problem built to punish fitting the noise"
        )
    if not report["split_ridge"]["ladder_is_the_whole_ladder"]:
        failures.append("the split ridge did not score every rung of the ladder")
    if report["split_ridge"]["split_heldout_sse"] >= report["split_ridge"]["law_heldout_sse"]:
        failures.append("the split ridge does not generalise better than THE LAW's own pick")
    for line in failures:
        print(f"FAIL: {line}")
    print("SELFTEST", "FAILED" if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
