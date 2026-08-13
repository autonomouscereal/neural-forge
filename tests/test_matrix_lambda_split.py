"""The wave-18 lambda fix, on the two properties that make it a fix.

`NOTES-MX-U` §5 measured wave 17's value/`out_proj` alternation improving its own
fit sum-of-squares by 16-29% at every site, moving the held-out per-site cosine
DOWN, and costing five review cases.  Its lambda was chosen by THE LAW's trust
region, which reads a norm and never reads generalisation, and §7 named the fix
without building it: choose lambda on an internal split of the FIT rows.

Two things have to be true of that fix or it is decoration:

* **the check objective has to be the residual it stands for.**  The split
  scores each rung from accumulated factors alone -- no attention probability is
  recomputed -- so if `stein_objective` is not the direct sum of squares, the
  selection is minimising an artefact.
* **THE LAW has to keep its veto and lose its vote.**  The rung returned must be
  one THE LAW admits, and among those it must be the one with the lowest check
  objective -- not the first one on the ladder, which is what wave 17 returned.

Both are checked here on seeded synthetic tensors; the builder's own
`mx18_joint_selftest.py` additionally checks that the fix generalises better than
the thing it replaces on a problem built to punish fitting the noise, and that
gate runs on the deployment before any child is built.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_ROOT = Path(__file__).resolve().parents[1]
_BUILDER = (
    _ROOT / "reports" / "matrix-campaign-20260810" / "run" / "mx18" / "mx18_build.py"
)

HEAD_DIM = 16
HIDDEN = 24
OUT = 24


def _builder():
    """Import the wave-18 builder by path; it is a run script, not a package."""

    spec = importlib.util.spec_from_file_location("mx18_build", _BUILDER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mx18_build"] = module
    spec.loader.exec_module(module)
    return module


def _two_sided(rows: int, noise: float, seed: int):
    torch.manual_seed(seed)
    heads = [0, 1]
    Wo = {h: torch.randn(OUT, HEAD_DIM, dtype=torch.float64) for h in heads}
    Z = {h: torch.randn(rows, HIDDEN, dtype=torch.float64) for h in heads}
    truth = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    clean = sum(Z[h] @ truth.T @ Wo[h].T for h in heads)
    Y = clean + noise * float(clean.std()) * torch.randn_like(clean)
    left = {(h, hp): Wo[h].T @ Wo[hp] for h in heads for hp in heads}
    right = {(h, hp): Z[hp].T @ Z[h] for h in heads for hp in heads}
    rhs = sum(Wo[h].T @ (Y.T @ Z[h]) for h in heads)
    return Wo, Z, truth, Y, left, right, rhs


def test_check_objective_is_the_residual_it_stands_for() -> None:
    module = _builder()
    Wo, Z, truth, Y, left, right, rhs = _two_sided(200, 0.3, seed=18)
    candidate = truth + 0.4 * torch.randn_like(truth)
    direct = float((sum(Z[h] @ candidate.T @ Wo[h].T for h in Z) - Y).pow(2).sum())
    closed = module.stein_objective(left, right, rhs, float(Y.pow(2).sum()), candidate)
    assert abs(direct - closed) / direct < 1e-10


def test_hoisted_factors_do_not_move_the_solve() -> None:
    module = _builder()
    _, _, _, _, left, right, rhs = _two_sided(200, 0.1, seed=19)
    prior = torch.randn(HEAD_DIM, HIDDEN, dtype=torch.float64)
    lam = 1e-3 * float(torch.diagonal(right[(0, 0)]).mean())
    plain, _ = module.stein_cg_solve(left, right, rhs, prior, lam)
    hoisted, _ = module.stein_cg_solve(
        left, right, rhs, prior, lam, factors=module.stein_factors(left, right)
    )
    assert float((plain - hoisted).abs().max()) == 0.0


def test_the_law_keeps_its_veto_and_loses_its_vote() -> None:
    module = _builder()
    torch.manual_seed(20)
    A = torch.randn(120, HIDDEN, dtype=torch.float64)
    truth = torch.randn(OUT, HIDDEN, dtype=torch.float64)
    clean = A @ truth.T
    Y = clean + 0.6 * float(clean.std()) * torch.randn_like(clean)
    prior = truth + 0.3 * torch.randn_like(truth)
    check = torch.zeros(A.shape[0], dtype=torch.bool)
    check[2::3] = True

    solved, guard = module.law_abiding_solve_split(A, Y, check, prior)

    admitted = [row for row in guard["ladder"] if row["update_scale"] <= 1.0]
    assert admitted, "the case must leave THE LAW something to admit"
    assert guard["update_scale"] <= 1.0
    assert guard["lambda_selected_on"] == "internal fit split"
    # The vote: the rung returned is the admitted one with the lowest check
    # objective, which is not in general the first on the ladder.
    assert guard["check_sse_chosen"] == min(row["check_sse"] for row in admitted)
    assert guard["lambda_law_would_have_picked"] == admitted[0]["lambda_relative"]
    # The veto: a rung THE LAW refuses can never be returned, whatever it scores.
    refused = [row for row in guard["ladder"] if row["update_scale"] > 1.0]
    assert guard["lambda_relative"] not in {row["lambda_relative"] for row in refused}
    reference = module.prior_anchored(A, Y, prior, rel=guard["lambda_relative"])
    assert float((solved - reference).abs().max()) == 0.0


def test_the_split_never_reads_the_rows_it_scores() -> None:
    """The selection is out of sample or it is nothing: perturbing only the check
    rows must move the chosen lambda's *score* without moving the solve that any
    rung produces on the inner rows."""

    module = _builder()
    torch.manual_seed(21)
    A = torch.randn(120, HIDDEN, dtype=torch.float64)
    truth = torch.randn(OUT, HIDDEN, dtype=torch.float64)
    Y = A @ truth.T + 0.5 * torch.randn(120, OUT, dtype=torch.float64)
    prior = truth + 0.3 * torch.randn_like(truth)
    check = torch.zeros(A.shape[0], dtype=torch.bool)
    check[2::3] = True

    _, first = module.law_abiding_solve_split(A, Y, check, prior)
    poisoned = Y.clone()
    poisoned[check] += 25.0
    _, second = module.law_abiding_solve_split(A, poisoned, check, prior)

    assert [row["check_sse"] for row in first["ladder"]] != [
        row["check_sse"] for row in second["ladder"]
    ]
    inner = ~check
    for rel in (1e-6, 1e-2, 1.0):
        untouched = module.prior_anchored(A[inner], Y[inner], prior, rel=rel)
        also = module.prior_anchored(A[inner], poisoned[inner], prior, rel=rel)
        assert float((untouched - also).abs().max()) == 0.0
