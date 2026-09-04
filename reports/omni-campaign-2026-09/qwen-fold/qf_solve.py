"""Closed-form algebra.  No gradients anywhere in this file, by rule.

Everything here is a port of the mx45 units named in the port spec, with three
substantive changes and no cosmetic ones:

1. Gram entry points beside every row-form solve, because a 27B site's design
   matrix is 17,408 columns wide and the rows do not fit but ``X^T X`` does.
2. :func:`plane_relations` / :func:`joint_relation_solve` work on Qwen's PARTIAL
   rotary geometry: ``rotary_dim/2`` complex planes plus ``head_dim -
   rotary_dim`` real NoPE coordinates, where LFM2 had all-rotary heads.
3. :func:`whitened_lra` and :func:`gated_out_proj_solve` are the two operators
   the Qwen-only fold classes need (value-space contraction under a use-weighted
   metric; a writer solved on a SIGMOID-GATED input).

THE LAW, unchanged: a solved weight may not move more than its own norm away
from the prior it was anchored to.  It is a guard whose cost is recorded, and it
orders attempts -- it is not a grade.
"""

from __future__ import annotations

import math
from typing import Any

import torch

RIDGE_REL = 1e-6
LAMBDA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
UPDATE_SCALE_CAP = 1.0

#: The eigenvalue floor the :func:`eigen_anchored_gram` fallback is allowed to
#: apply, RELATIVE to the largest eigenvalue of the ridged Gram.  It is the only
#: regularisation that routine adds beyond ``lam``, it is returned in the
#: fallback's own report, and it is written into the receipt as
#: ``solver_fallback`` -- a fallback whose regularisation is not in the record
#: is a fabricated number, not a rescued one.
EIGEN_FLOOR_REL = 1e-12

#: How many offending channel indices a refusal names before it says "and N
#: more".  A refusal that prints 4,096 integers is a refusal nobody reads.
DEFECT_NAMES = 16


# --------------------------------------------------------- Gram refusal census
class GramNotPositiveDefinite(RuntimeError):
    """``G + lam I`` will not factorise, WITH the channels that made it so.

    ``torch.linalg.cholesky`` says "the leading minor of order 1 is not
    positive-definite" and nothing else: not which channel, not whether the Gram
    is merely singular or carries a NaN -- and those two want opposite repairs.
    A singular-but-finite Gram is a conditioning problem and
    :func:`eigen_anchored_gram` answers it; a Gram with a non-finite entry is a
    ROW problem (one non-finite row of ``A`` poisons every entry of ``A^T A``)
    and no amount of ridge touches it, so it has to be refused here and repaired
    at the capture.

    :attr:`report` is the census a receipt records.
    """

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def gram_defect_report(
    G: torch.Tensor, lam: float, rel: float, limit: int = DEFECT_NAMES
) -> dict[str, Any]:
    """Which channels make ``G + lam I`` unfactorisable, and in which way.

    Three separable defects, counted independently because they do not imply one
    another: a non-finite DIAGONAL entry, a non-positive diagonal entry (the
    literal "leading minor" complaint when it is channel 0), and a channel whose
    ROW carries a non-finite anywhere -- which is how a single NaN token row in
    the design matrix presents, since ``A^T A`` spreads it across the whole Gram.
    """

    diag = torch.diagonal(G)
    finite_entries = torch.isfinite(G)
    finite_diag = torch.isfinite(diag)

    def census(mask: torch.Tensor) -> tuple[int, list[int]]:
        index = torch.nonzero(mask).flatten()
        return int(index.numel()), index[:limit].tolist()

    nonfinite_diag_n, nonfinite_diag_first = census(~finite_diag)
    nonpositive_n, nonpositive_first = census(finite_diag & ((diag + lam) <= 0))
    poisoned_n, poisoned_first = census(~finite_entries.all(dim=1))
    finite_values = diag[finite_diag]
    return {
        "columns": int(G.shape[0]),
        "ridge_relative": float(rel),
        "ridge_absolute": float(lam),
        "ridge_is_finite": bool(math.isfinite(lam)),
        "gram_nonfinite_entries": int((~finite_entries).sum()),
        "channels_with_nonfinite_diagonal": nonfinite_diag_n,
        "channels_with_nonfinite_diagonal_first": nonfinite_diag_first,
        "channels_with_nonpositive_diagonal": nonpositive_n,
        "channels_with_nonpositive_diagonal_first": nonpositive_first,
        "channels_with_any_nonfinite_entry": poisoned_n,
        "channels_with_any_nonfinite_entry_first": poisoned_first,
        "finite_diagonal_min": float(finite_values.min()) if finite_values.numel() else None,
        "finite_diagonal_max": float(finite_values.max()) if finite_values.numel() else None,
        "finite_diagonal_mean": float(finite_values.mean()) if finite_values.numel() else None,
    }


def _defect_message(report: dict[str, Any], where: str) -> str:
    parts = [
        f"{where}: the {report['columns']}-channel Gram will not factorise at "
        f"ridge rel={report['ridge_relative']:g} (absolute {report['ridge_absolute']!r})"
    ]
    if not report["ridge_is_finite"]:
        parts.append(
            "the ridge itself is not finite, so the Gram's diagonal mean is not "
            "finite either -- every entry of G + lam I is NaN and the 'leading "
            "minor of order 1' complaint is about channel 0 only by accident"
        )
    if report["gram_nonfinite_entries"]:
        parts.append(
            f"{report['gram_nonfinite_entries']} non-finite Gram entries; "
            f"{report['channels_with_any_nonfinite_entry']} channels carry one, "
            f"first {report['channels_with_any_nonfinite_entry_first']}"
        )
        parts.append(
            "a non-finite Gram is a ROW defect in the design matrix (A^T A "
            "spreads one bad row over every channel); no ridge repairs it -- fix "
            "the capture"
        )
    if report["channels_with_nonfinite_diagonal"]:
        parts.append(
            f"{report['channels_with_nonfinite_diagonal']} channels have a "
            f"non-finite diagonal, first "
            f"{report['channels_with_nonfinite_diagonal_first']}"
        )
    if report["channels_with_nonpositive_diagonal"]:
        parts.append(
            f"{report['channels_with_nonpositive_diagonal']} channels have a "
            f"non-positive diagonal after the ridge, first "
            f"{report['channels_with_nonpositive_diagonal_first']} -- an "
            f"identically dead channel in the captured stream"
        )
    parts.append(
        f"finite diagonal min {report['finite_diagonal_min']!r} max "
        f"{report['finite_diagonal_max']!r} mean {report['finite_diagonal_mean']!r}"
    )
    if "cholesky_error" in report:
        parts.append(f"cholesky said: {report['cholesky_error']}")
    return "; ".join(parts)


# ----------------------------------------------------------------- ridge core
def prior_anchored(
    A: torch.Tensor, Y: torch.Tensor, prior: torch.Tensor, rel: float = RIDGE_REL
) -> torch.Tensor:
    """``argmin ||A W^T - Y||^2 + lam ||W - prior||^2`` -- the production objective."""

    G = A.T @ A
    lam = rel * float(torch.diagonal(G).mean())
    G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
    B = A.T @ Y + lam * prior.T
    L = torch.linalg.cholesky(G)
    return torch.cholesky_solve(B, L).T.contiguous()


def prior_anchored_gram(
    G: torch.Tensor, B: torch.Tensor, prior: torch.Tensor, rel: float = RIDGE_REL
) -> torch.Tensor:
    """:func:`prior_anchored` from ``G = A^T A`` and ``B = A^T Y`` alone.

    Identical answer, no rows.  The 27B path never materialises ``A``.

    A failed factorisation is raised as :class:`GramNotPositiveDefinite` with the
    offending channels named, not as the bare LAPACK sentence.  The census costs
    nothing on the happy path -- it is only computed once Cholesky has already
    refused.
    """

    lam = rel * float(torch.diagonal(G).mean())
    Gl = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
    try:
        L = torch.linalg.cholesky(Gl)
    except Exception as error:  # torch._C._LinAlgError, and anything LAPACK adds
        report = gram_defect_report(G, lam, rel)
        report["cholesky_error"] = f"{type(error).__name__}: {error}"
        raise GramNotPositiveDefinite(
            _defect_message(report, "prior_anchored_gram"), report
        ) from error
    return torch.cholesky_solve(B + lam * prior.T, L).T.contiguous()


def eigen_anchored_gram(
    G: torch.Tensor,
    B: torch.Tensor,
    prior: torch.Tensor,
    rel: float = RIDGE_REL,
    floor_rel: float = EIGEN_FLOOR_REL,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """:func:`prior_anchored_gram` through a symmetric eigendecomposition.

    Same objective, same anchor, no Cholesky.  ``G`` is symmetrised, ridged by
    the same ``lam``, decomposed, and its eigenvalues are FLOORED at
    ``floor_rel * max(eigenvalue)`` before the inverse is applied.  That floor is
    an EXTRA ridge and it is returned -- how many eigenvalues it lifted, from
    what, to what -- so the receipt says exactly what was added to get a number.

    It refuses, rather than answering, in the two cases where an answer would be
    invented:

    * a Gram carrying a non-finite entry.  ``eigh`` of a NaN is a NaN; the defect
      is a row of the design matrix and the repair is at the capture.
    * a Gram with no positive eigenvalue at all (an identically dead stream).
      There is no direction in which the evidence says anything, and returning
      ``prior`` from inside a "solve" would put an un-fitted weight into a
      receipt that claims a fit.
    """

    lam = rel * float(torch.diagonal(G).mean())
    if not bool(torch.isfinite(G).all()):
        report = gram_defect_report(G, lam, rel)
        raise GramNotPositiveDefinite(
            _defect_message(report, "eigen_anchored_gram (fallback)"), report
        )
    n = int(G.shape[0])
    Gs = 0.5 * (G + G.T) + lam * torch.eye(n, dtype=G.dtype)
    values, vectors = torch.linalg.eigh(Gs)
    top = float(values.max())
    if not (top > 0.0):
        report = gram_defect_report(G, lam, rel)
        report["eigenvalue_max"] = top
        raise GramNotPositiveDefinite(
            _defect_message(
                report,
                "eigen_anchored_gram (fallback): the ridged Gram has no positive "
                "eigenvalue, so the stream carries no energy in any direction",
            ),
            report,
        )
    floor = floor_rel * top
    lifted = int((values < floor).sum())
    clamped = values.clamp_min(floor)
    solution = vectors @ ((vectors.T @ (B + lam * prior.T)) / clamped.unsqueeze(1))
    return solution.T.contiguous(), {
        "solver_fallback": "eigen-decomposition with an explicit eigenvalue floor",
        "reason": "torch.linalg.cholesky refused the ridged Gram",
        "ridge_relative": float(rel),
        "ridge_absolute": float(lam),
        "eigen_floor_relative": float(floor_rel),
        "eigen_floor_absolute": float(floor),
        "eigenvalues_lifted_to_the_floor": lifted,
        "eigenvalue_min_before_floor": float(values.min()),
        "eigenvalue_max": top,
        "condition_after_floor": float(top / floor),
        "is_an_extra_regularisation": (
            "yes -- the floor is a ridge this rung's lambda did not ask for; the "
            "installed weight is the floored solve and this block is what it cost"
        ),
    }


def _ladder_rung(
    G: torch.Tensor,
    B: torch.Tensor,
    prior: torch.Tensor,
    rel: float,
    fallbacks: list[dict[str, Any]],
    where: str,
) -> torch.Tensor:
    """One rung of the lambda ladder: Cholesky, or the RECORDED eigen fallback.

    A single degenerate site used to end the whole build in a LAPACK sentence.
    It now produces a number and a paragraph saying how that number differs from
    the one the ladder would otherwise have produced.  Nothing is skipped and
    nothing is silent.

    When the fallback refuses TOO, the refusal that propagates is the ORIGINAL
    one -- carrying the ridge Cholesky actually saw and the LAPACK sentence it
    actually printed -- with the fallback's reason appended.  Letting the
    fallback's own refusal replace it loses exactly the fact that matters most
    in the NaN case: that the ridge itself was not finite, because the Gram's
    diagonal mean was not.  (The K2KV layer-27 report is that fact.)
    """

    try:
        return prior_anchored_gram(G, B, prior, rel=rel)
    except GramNotPositiveDefinite as refusal:
        try:
            candidate, record = eigen_anchored_gram(G, B, prior, rel=rel)
        except GramNotPositiveDefinite as fallback_refusal:
            report = dict(refusal.report)
            report["gram"] = where
            report["eigen_fallback_refusal"] = str(fallback_refusal)
            raise GramNotPositiveDefinite(
                f"{refusal}; the eigen fallback refused as well -- "
                f"{fallback_refusal}",
                report,
            ) from refusal
        fallbacks.append({"lambda_relative": rel, "gram": where, **record,
                          "refusal": str(refusal)})
        return candidate


def law_abiding_solve(
    A: torch.Tensor, Y: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The lambda-ladder trust region.  THE LAW is a guard and its cost is recorded.

    The Gram and the right-hand side depend on ``A`` and ``Y`` alone -- not on
    lambda -- so they are formed ONCE and the ladder is walked in the Gram form.
    Re-forming ``A^T A`` per rung is the same answer at nine times the GEMM, and
    at a 27B site (``A`` is ``[N, 4352]``) that is hours of nothing.
    """

    G, B = A.T @ A, A.T @ Y
    reference = prior.norm()
    free_scale = None
    fallbacks: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        candidate = _ladder_rung(G, B, prior, rel, fallbacks, "A^T A")
        scale = float((candidate - prior).norm() / reference)
        # The free solve IS the first rung -- solving it a second time to name
        # it was the small version of the same waste.
        if free_scale is None:
            free_scale = scale
        if scale <= UPDATE_SCALE_CAP:
            return candidate, {
                "update_scale": scale,
                "lambda_relative": rel,
                "free_solve_update_scale": free_scale,
                "form": "gram (A^T A formed once, one factorisation per rung)",
                "solver": "cholesky" if not fallbacks else "cholesky + eigen fallback",
                "solver_fallback": fallbacks,
            }
    raise SystemExit(
        f"no lambda on the ladder brings update_scale to {UPDATE_SCALE_CAP} "
        f"(free {free_scale:.3f}); THE LAW refuses the install"
    )


def law_abiding_solve_gram(
    G: torch.Tensor, B: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    reference = prior.norm()
    free_scale = None
    fallbacks: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        candidate = _ladder_rung(G, B, prior, rel, fallbacks, "supplied gram")
        scale = float((candidate - prior).norm() / reference)
        if free_scale is None:
            free_scale = scale
        if scale <= UPDATE_SCALE_CAP:
            return candidate, {
                "update_scale": scale,
                "lambda_relative": rel,
                "free_solve_update_scale": free_scale,
                "form": "gram",
                "solver": "cholesky" if not fallbacks else "cholesky + eigen fallback",
                "solver_fallback": fallbacks,
            }
    raise SystemExit(
        f"no lambda on the ladder brings update_scale to {UPDATE_SCALE_CAP} "
        f"(free {free_scale:.3f}); THE LAW refuses the install"
    )


def law_abiding_solve_split(
    A: torch.Tensor, Y: torch.Tensor, check: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """:func:`law_abiding_solve` with lambda chosen on an internal split of its own rows.

    ``check`` is a boolean mask over ``A``'s rows: solved on the complement,
    scored on the mask, and the admitted rung with the lowest CHECK
    sum-of-squares is re-solved on all rows.  THE LAW is unchanged -- it is still
    a veto on the installed weight's update scale -- it simply no longer picks
    among the rungs it admits.  What the ladder-order rule WOULD have picked is
    recorded beside the choice so the difference is legible.

    Both Grams -- the full one and the inner-split one -- depend on the rows and
    not on lambda, so each is formed ONCE and the nine rungs are two triangular
    solves apiece.  The row form re-formed ``A^T A`` nineteen times per site; at
    the 27B C2 width that is ~4.6e11 fp64 flops of duplicate GEMM per rung.  The
    CHECK score stays on rows, because ``A_ck``/``Y_ck`` are small and the score
    is a residual, not a normal equation.
    """

    inner = ~check
    A_in, Y_in = A[inner], Y[inner]
    A_ck, Y_ck = A[check], Y[check]
    reference = float(prior.norm())
    G_all, B_all = A.T @ A, A.T @ Y
    G_in, B_in = A_in.T @ A_in, A_in.T @ Y_in
    rungs: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        candidate_in = _ladder_rung(G_in, B_in, prior, rel, fallbacks, "inner-split gram")
        candidate = _ladder_rung(G_all, B_all, prior, rel, fallbacks, "all-rows gram")
        rungs.append(
            {
                "lambda_relative": rel,
                "check_sse": float((A_ck @ candidate_in.T - Y_ck).pow(2).sum()),
                "update_scale": float((candidate - prior).norm() / reference),
                "solved": candidate,
            }
        )
    # The free solve is rung 0 of the ladder that was just walked.
    free_scale = rungs[0]["update_scale"]
    admitted = [row for row in rungs if row["update_scale"] <= UPDATE_SCALE_CAP]
    if not admitted:
        raise SystemExit(
            f"no lambda on the ladder brings update_scale to {UPDATE_SCALE_CAP} "
            f"(free {free_scale:.3f}); THE LAW refuses the install"
        )
    chosen = min(admitted, key=lambda row: row["check_sse"])
    return chosen["solved"], {
        "update_scale": chosen["update_scale"],
        "lambda_relative": chosen["lambda_relative"],
        "free_solve_update_scale": free_scale,
        "lambda_selected_on": "internal fit split",
        "lambda_law_would_have_picked": admitted[0]["lambda_relative"],
        "check_sse_chosen": chosen["check_sse"],
        "check_sse_law_pick": admitted[0]["check_sse"],
        "ladder": [{k: v for k, v in row.items() if k != "solved"} for row in rungs],
        "solver": "cholesky" if not fallbacks else "cholesky + eigen fallback",
        "solver_fallback": fallbacks,
    }


def write_scores(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    return {
        "relL2": float(diff.norm() / target.norm()),
        "cos": float(cos.mean()),
        "cos_p05": float(torch.quantile(cos, torch.tensor(0.05, dtype=cos.dtype))),
    }


# ------------------------------------------------------------------- coresets
def pivoted_columns(
    F: torch.Tensor, k: int, report: dict[str, Any] | None = None
) -> torch.Tensor:
    """Wave 14's deterministic column-pivoted selection, unchanged."""

    return pivoted_columns_gram(F.T @ F, k, report=report)


def pivoted_columns_gram(
    G: torch.Tensor, k: int, report: dict[str, Any] | None = None, tol: float = 1e-24
) -> torch.Tensor:
    """The same selection from the Gram alone -- the only form a 27B site can afford.

    Two things this routine must not do, both of which it used to do:

    * **Pick the pivot in Python.**  The greedy maximum was a
      ``for i in range(n)`` over 0-dim tensor comparisons inside the ``for step
      in range(k)`` loop, i.e. ``n*k`` interpreter-level tensor ops.  At the 27B
      C2 rung (17,408 -> 4,352) that is 75.7 MILLION of them for ONE site, and
      there are 62 sites.  ``torch.argmax`` is one kernel and picks the same
      column: ties go to the lowest index in both forms.
    * **Compare on an ABSOLUTE tolerance.**  ``diag[i] > best + 1e-15`` and
      ``best <= 1e-30`` are thresholds on a quantity with no fixed scale.  A
      site whose second moments arrive small (a down-scaled or bf16-captured
      design matrix, diag ~1e-16) never clears ``best + 1e-15`` after the first
      pivot, and the routine degenerates SILENTLY to ``0..k-1`` in index order
      -- a selection that is not a selection.  The stopping test is now relative
      to the largest diagonal entry seen at entry.

    ``report``, if given, is filled with the residual diagonal mass so a
    degenerate or early-stopped selection is visible in the axis report instead
    of only in the tensors.
    """

    n = int(G.shape[0])
    if k >= n:
        # Keeping every channel is not a selection, and running the greedy loop
        # to discover that costs n*k Python iterations.
        if report is not None:
            report.update(
                {
                    "selection": "identity (k >= n)",
                    "kept": n,
                    "residual_diagonal_fraction": 0.0,
                }
            )
        return torch.arange(n, dtype=torch.long)
    G = G.clone()
    diag = torch.diagonal(G).clone()
    entry_mass = float(diag.clamp_min(0.0).sum())
    entry_peak = float(diag.max())
    chosen: list[int] = []
    L = torch.zeros(n, k, dtype=G.dtype)
    stopped = "completed"
    for step in range(k):
        pick = int(torch.argmax(diag))
        best = float(diag[pick])
        if best <= tol * entry_peak:
            stopped = f"exhausted after {step} of {k} (residual pivot {best:.3e})"
            break
        chosen.append(pick)
        col = G[:, pick] - L[:, :step] @ L[pick, :step]
        col = col / best ** 0.5
        L[:, step] = col
        diag = diag - col * col
        diag[chosen] = -1.0
    if report is not None:
        residual = float(diag.clamp_min(0.0).sum())
        report.update(
            {
                "selection": "gram column-pivoted greedy",
                "kept": len(chosen),
                "requested": k,
                "entry_diagonal_mass": entry_mass,
                "entry_peak": entry_peak,
                "residual_diagonal_mass": residual,
                "residual_diagonal_fraction": residual / max(entry_mass, 1e-300),
                "stopped": stopped,
                "tolerance_is": "relative to the largest entry diagonal, not absolute",
            }
        )
    return torch.tensor(sorted(chosen), dtype=torch.long)


def swiglu(X: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """``silu(X gate^T) * (X up^T)`` -- ``Qwen3_5MLP.forward`` without the writer."""

    return torch.nn.functional.silu(X @ w_gate.T) * (X @ w_up.T)


def metric_column_scales(writer: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """``s_j = sqrt(w[:, j]^T M w[:, j])`` -- what dropping channel j costs the reader.

    The metric cancels out of an unconstrained ridge (for full column rank and a
    vanishing anchor the solution is ``(X^T X)^-1 X^T Y`` for every PD ``M``),
    so it can only matter in the rank-constrained step -- which on this axis is
    the SELECTION.  Dropping channel ``j`` unrepaired puts ``phi[:, j] w[:, j]^T``
    into the block output, whose squared cost in the reader metric factorises
    exactly into ``||phi[:, j]||^2`` times ``w[:, j]^T M w[:, j]``.
    """

    return torch.sqrt(((writer.T @ M) * writer.T).sum(dim=1).clamp_min(0.0))


def matrix_sqrt(G: torch.Tensor, floor: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """``(G^{1/2}, G^{-1/2})`` for a symmetric PSD ``G``, eigenvalues floored."""

    G = 0.5 * (G + G.T)
    values, vectors = torch.linalg.eigh(G)
    scale = float(values.max().clamp_min(floor))
    values = values.clamp_min(floor * scale)
    root = vectors @ torch.diag(values.sqrt()) @ vectors.T
    inverse = vectors @ torch.diag(values.rsqrt()) @ vectors.T
    return 0.5 * (root + root.T), 0.5 * (inverse + inverse.T)


def whitened_lra(
    stack: torch.Tensor, gram: torch.Tensor, keep: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The evidence-whitened low-rank contraction of a VALUE space.

    ``stack`` is the writer that consumes the space (``[out, n]``: an
    ``out_proj``'s columns for the heads being merged), ``gram`` is the space's
    own second moment on the evidence (``[n, n]``).  We want ``P [keep, n]`` so
    that ``stack v`` is reproducible from ``P v`` -- i.e. so that ``stack``
    restricted to the directions the evidence actually populates survives.

    The whitening transform is ``ṽ = G^{-1/2} v`` (that is the one under which
    ``E[ṽ ṽ^T] = I``), so ``stack v = (stack G^{1/2}) ṽ`` and

        ``E_v || stack v - R P v ||^2 = || (stack - R P) G^{1/2} ||_F^2``

    for any refit writer ``R``.  Minimising over ``R`` and over row spaces of
    rank ``keep`` is therefore the plain truncated SVD of ``A = stack G^{1/2}``:
    the optimal row space is ``V_keep^T`` in the WHITENED coordinates, and
    pulling it back gives ``P = V_keep^T G^{-1/2}``.

    The direction matters and is not a convention: with an anisotropic gram the
    two choices keep OPPOSITE ends of the spectrum.  ``gram = diag(1, eps)`` with
    ``stack = [1, 1]`` -- the writer reads both coordinates, the evidence only
    populates the first -- gives ``A = [1, sqrt(eps)]`` and ``keep=1`` retains
    coordinate 0 (error 0).  Feeding ``stack G^{-1/2} = [1, eps^{-1/2}]`` to the
    same SVD retains coordinate 1 instead: the one the evidence never populates,
    at reconstruction error ``Var(v_0)``.  This module used to do the latter;
    :func:`qf_selftest.test_solve_laws` now pins it with that fixture.

    Rows are orthonormalised afterwards (QR) so the child's value coordinates are
    a frame and not a skewed one -- the gated RMSNorm downstream normalises
    across those coordinates and a skewed frame would silently re-weight it.  QR
    preserves the ROW SPACE, which is the whole content of the optimum above and
    the only thing ``retained_energy`` describes; the writer is refit against the
    installed ``P`` by :func:`project_writer`, so the change of frame is free.
    """

    n = int(stack.shape[1])
    if keep > n:
        raise SystemExit(f"whitened_lra cannot keep {keep} of {n} coordinates")
    root, inverse_root = matrix_sqrt(gram)
    whitened = stack @ root
    # `full_matrices=True` so ``Vh`` is a COMPLETE n x n basis even when the
    # writer has fewer rows than the space is wide.  With `full_matrices=False`
    # a rank-deficient writer silently caps `keep` at its own rank -- which
    # would hand the installer a projection of the wrong shape and build a child
    # whose config says one width and whose tensors say another.
    U, S, Vh = torch.linalg.svd(whitened, full_matrices=True)
    P = Vh[:keep] @ inverse_root
    Q, _ = torch.linalg.qr(P.T)
    P = Q[:, :keep].T.contiguous()
    energy = float((S[:keep] ** 2).sum() / S.pow(2).sum().clamp_min(1e-300))
    return P, {
        "keep": keep,
        "writer_rank": int(min(stack.shape)),
        "parent_dim": n,
        "retained_energy": energy,
        "retained_energy_is": (
            "fraction of E||stack v||^2 on the evidence that survives the kept "
            "row space; singular values are those of stack @ gram^{1/2}"
        ),
        "singular_values": [float(v) for v in S[: min(len(S), 16)]],
    }


def project_writer(stack: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """The writer's prior in the contracted frame: ``stack P^+``."""

    return stack @ torch.linalg.pinv(P)


# ---------------------------------------------------------- rotary/NoPE planes
def as_planes(rows: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    """Head rows ``[head_dim, H]`` -> complex plane rows ``[n_planes, H]``.

    A rotary plane ``p`` is the pair ``(p, p + rotary_dim//2)`` -- that is what
    ``rotate_half`` pairs (modeling 562-566), NOT ``(2p, 2p+1)``.  A NoPE
    coordinate is its own plane with zero imaginary part, so one formula covers
    both: the relation error, the sign gauge and the joint solve never branch.
    """

    lo, hi, nope = layout["lo"], layout["hi"], layout["nope"]
    rotary = torch.complex(rows[lo], rows[hi])
    flat = torch.complex(rows[nope], torch.zeros_like(rows[nope]))
    return torch.cat([rotary, flat], dim=0)


def from_planes(planes: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    """Inverse of :func:`as_planes`."""

    lo, hi, nope = layout["lo"], layout["hi"], layout["nope"]
    rows = torch.zeros(
        int(layout["head_dim"]), planes.shape[1], dtype=planes.real.dtype
    )
    rows[lo] = planes[: len(lo)].real
    rows[hi] = planes[: len(lo)].imag
    rows[nope] = planes[len(lo) :].real
    return rows


def sign_gauge(A: torch.Tensor, B: torch.Tensor, layout: dict[str, Any]) -> torch.Tensor:
    """Per-plane +-1 that best aligns ``B``'s planes to ``A``'s.

    A rotary plane admits a free phase, but the child must WRITE real rows, and
    a NoPE coordinate admits nothing but a sign, because ``k_norm``'s per-dim
    weight breaks any cross-dimension rotation.  So the install gauge is +-1 on
    both kinds; the richer rotation gauge is measured as a census
    (:func:`rotation_gauge_cos`) and installs nothing.
    """

    inner = (A.conj() * B).sum(dim=1).real
    return torch.where(inner < 0, -torch.ones_like(inner), torch.ones_like(inner))


def rotation_gauge_cos(A: torch.Tensor, B: torch.Tensor, layout: dict[str, Any]) -> float:
    """The cosine a free per-plane PHASE would buy, on the rotary planes only."""

    n_rotary = len(layout["lo"])
    if n_rotary == 0:
        return float("nan")
    a, b = A[:n_rotary], B[:n_rotary]
    aligned = (a.conj() * b).sum(dim=1).abs()
    norms = a.norm(dim=1) * b.norm(dim=1)
    return float((aligned / norms.clamp_min(1e-30)).mean())


def rank_one_relation_error(
    q: torch.Tensor, k: torch.Tensor, q_hat: torch.Tensor, k_hat: torch.Tensor
) -> float:
    """Relative error of ``q_hat k_hat^H`` against ``q k^H``, per plane, pooled.

    The score a head contributes at plane ``p`` is determined by the rank-one
    operator ``q_p k_p^H`` and by nothing else, so this is the exact quantity a
    head merge trades away.
    """

    num = 0.0
    den = 0.0
    for p in range(q.shape[0]):
        exact = torch.outer(q[p], k[p].conj())
        approx = torch.outer(q_hat[p], k_hat[p].conj())
        num += float((approx - exact).abs().pow(2).sum())
        den += float(exact.abs().pow(2).sum())
    return (num / max(den, 1e-300)) ** 0.5


def joint_relation_solve(
    q_parents: list[torch.Tensor],
    k_parents: list[torch.Tensor],
    group_of: list[int],
    weights: list[float],
    layout: dict[str, Any],
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, Any]]:
    """Merge several key heads into ONE, re-fitting every query head that read them.

    Per plane the score head ``h`` contributes is the rank-one complex operator
    ``phi_h kappa_{g(h)}^H``.  Constrain the merged key to the span of its
    parents, ``kappa_hat = sum_a c_a kappa_a``.  For a FIXED ``kappa_hat`` the
    best query row is ``phi_hat_h = phi_h (kappa_g^H kappa_hat)/||kappa_hat||^2``
    and the residual is ``w_h ||phi_h||^2 (||kappa_g||^2 -
    |kappa_g^H kappa_hat|^2 / ||kappa_hat||^2)``.  Summing over heads, the whole
    problem collapses to maximising a Rayleigh quotient of

        ``C = sum_h w_h ||phi_h||^2 kappa_{g(h)} kappa_{g(h)}^H``

    over the span -- an ``m x m`` Hermitian generalised eigenproblem per plane
    with ``m`` = number of merged key heads.  Closed form, no sweeps, and the
    only free choices left are scale and phase, which are pinned below.

    ``group_of[h]`` is the index into ``k_parents`` that query head ``h`` reads.
    Passing one parent is legal and is the identity (a "merge" of one head).
    """

    q_planes = [as_planes(rows, layout) for rows in q_parents]
    k_planes = [as_planes(rows, layout) for rows in k_parents]
    n_planes = q_planes[0].shape[0]
    m = len(k_planes)

    k_hat = torch.zeros_like(k_planes[0])
    q_hats = [torch.zeros_like(p) for p in q_planes]
    per_plane: list[float] = []

    for p in range(n_planes):
        basis = torch.stack([kp[p] for kp in k_planes], dim=1)          # [H, m]
        mass = torch.zeros(m, dtype=torch.float64)
        for h, phi in enumerate(q_planes):
            mass[group_of[h]] += weights[h] * float(phi[p].abs().pow(2).sum())
        # C = sum_a mass_a kappa_a kappa_a^H, expressed in the basis itself.
        overlap = basis.conj().T @ basis                                # [m, m]
        C_basis = overlap @ torch.diag(mass.to(overlap.dtype)) @ overlap
        overlap = 0.5 * (overlap + overlap.conj().T)
        C_basis = 0.5 * (C_basis + C_basis.conj().T)
        jitter = 1e-18 * float(torch.diagonal(overlap).real.max().clamp_min(1e-30))
        overlap = overlap + jitter * torch.eye(m, dtype=overlap.dtype)
        chol = torch.linalg.cholesky(overlap)
        reduced = torch.linalg.solve_triangular(
            chol, torch.linalg.solve_triangular(chol, C_basis, upper=False).conj().T,
            upper=False,
        ).conj().T
        reduced = 0.5 * (reduced + reduced.conj().T)
        _, vectors = torch.linalg.eigh(reduced)
        c = torch.linalg.solve_triangular(
            chol.conj().T, vectors[:, -1].unsqueeze(1), upper=True
        ).squeeze(1)
        vector = basis @ c
        norm = vector.norm().clamp_min(1e-30)

        # Scale: the merged key keeps the energy-weighted average magnitude of
        # its parents, so the delta rule's state does not silently rescale.
        # Phase: anchored so the merged key is in phase with that average, which
        # is what makes the recovered query rows real-signed and reproducible.
        target = sum(
            float(mass[a]) * k_planes[a][p] for a in range(m)
        ) / float(mass.sum().clamp_min(1e-30))
        phase = (vector.conj() @ target)
        phase = phase / phase.abs().clamp_min(1e-30)
        magnitude = float(target.norm())
        vector = vector * (phase * magnitude / norm)
        k_hat[p] = vector

        # The query row is re-fit to the merged key by a single complex scalar
        # `gamma`, and `gamma` is pinned, not chosen: with `q_hat = gamma phi`
        # the plane's rank-one error is
        #     ||phi||^2 * || conj(gamma) khat - kappa ||^2,
        # whose minimiser is `conj(gamma) = khat^H kappa / ||khat||^2`.  The
        # coefficient computed here is `kappa^H khat / ||khat||^2`, which is the
        # conjugate of that -- i.e. it IS `gamma`, and installing
        # `coefficient.conj()` (as this line used to) installs the conjugate of
        # the optimum.  On the NoPE coordinates everything is real and the two
        # agree; on the rotary planes the wrong one costs a factor of ~2.5 in
        # `rank_one_relation_error` on a generic phase.  Pinned by
        # `qf_selftest.test_solve_laws`, which now compares against the
        # conjugated choice and against gamma = 1.
        denominator = float(vector.norm().pow(2).clamp_min(1e-30))
        for h, phi in enumerate(q_planes):
            coefficient = (k_planes[group_of[h]][p].conj() @ vector) / denominator
            q_hats[h][p] = phi[p] * coefficient
        per_plane.append(float(mass.sum()))

    errors = [
        rank_one_relation_error(
            q_planes[h], k_planes[group_of[h]].clone(), q_hats[h], k_hat
        )
        for h in range(len(q_planes))
    ]
    report = {
        "merged_key_heads": m,
        "query_slots": len(q_planes),
        "planes": n_planes,
        "rotary_planes": int(len(layout["lo"])),
        "nope_coordinates": int(len(layout["nope"])),
        "relation_error_per_slot": errors,
        "relation_error_mean": float(sum(errors) / max(len(errors), 1)),
        "relation_error_max": float(max(errors)) if errors else 0.0,
    }
    return (
        from_planes(k_hat, layout),
        [from_planes(q, layout) for q in q_hats],
        report,
    )


def direction_anchored_solve(
    X: torch.Tensor,
    target_rows: torch.Tensor,
    gain: torch.Tensor,
    prior: torch.Tensor,
    eps: float,
    sweeps: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit rows whose consumer sees them only after an RMS normalisation.

    ``k_norm``/``l2norm`` throw away the row block's per-token scale, so the
    ordinary least-squares target is the wrong target: matching the normalised
    direction is what the score depends on.  Alternate (i) solve the rows against
    the current per-token scales, (ii) recompute the scales -- two sweeps is what
    LFM2 used and what is carried, because the map is a contraction here and the
    third sweep moved nothing measurable.
    """

    scale = torch.ones(X.shape[0], 1, dtype=X.dtype)
    weight = prior.clone()
    history: list[float] = []
    for _ in range(max(1, sweeps)):
        Y = target_rows * scale
        weight, _ = law_abiding_solve(X, Y, prior)
        produced = X @ weight.T
        rms = torch.sqrt((produced * gain).pow(2).mean(dim=1, keepdim=True) + eps)
        scale = rms.clamp_min(1e-30)
        history.append(float((produced / scale - target_rows).norm() / target_rows.norm()))
    return weight, {"direction_relL2_by_sweep": history}


# ------------------------------------------------ gated writer / value operators
def gated_out_proj_solve(
    gated_input: torch.Tensor,
    target: torch.Tensor,
    prior: torch.Tensor,
    check: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Solve a writer on its OWN captured input, gate included.

    Both mixers multiply by a nonlinear gate before the writer -- attention by
    ``sigmoid(gate)`` (modeling 714) and the GDN by ``silu(z)`` inside the gated
    norm (200).  The gate is therefore part of the design matrix, not something
    to model: capture the writer's input on the installed child and the solve is
    the ordinary ridge again, with no gate approximation anywhere in it.
    """

    if check is None:
        return law_abiding_solve(gated_input, target, prior)
    return law_abiding_solve_split(gated_input, target, check, prior)


def gated_value_cg_refit(
    X: torch.Tensor,
    writer_blocks: list[torch.Tensor],
    gates: list[torch.Tensor],
    readers: list[torch.Tensor],
    target: torch.Tensor,
    prior: torch.Tensor,
    *,
    iterations: int = 40,
    rel: float = RIDGE_REL,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Re-fit a SHARED value projection under fixed per-head writers and gates.

    ``out(i) = sum_h Wo_h diag(s_h(i)) Wv (A_h X_i)``.  ``Wv`` appears inside a
    per-token diagonal, so there is no normal equation to Cholesky: the operator

        ``T(S) = sum_h sum_i Wo_h^T [ Wo_h ( s_h(i) o (S z_h(i)) ) ] z_h(i)^T``

    is symmetric positive semi-definite in ``S`` and is applied in closed form,
    so conjugate gradients on ``T + lam I`` is still a closed-form solve -- no
    gradients, no training.  Preconditioned by the MEAN gate, which is the
    ``stein``-style two-sided factorisation with the diagonal folded in.

    Returns the solved ``Wv`` and the objective at entry and exit.  Reported, not
    gated: a rung whose objective barely moves is a fold that did not need it.
    """

    projections = [X @ A.T for A in readers]   # z_h, [N, n] each

    def apply(S: torch.Tensor) -> torch.Tensor:
        """``T(S)``, the normal operator, applied without ever forming it."""

        out = torch.zeros_like(S)
        for Wo, gate, z in zip(writer_blocks, gates, projections):
            u = gate * (z @ S.T)              # [N, keep]  gated child value
            back = ((u @ Wo.T) @ Wo) * gate   # [N, keep]  Wo^T Wo, gate again
            out = out + back.T @ z
        return out

    def right_hand_side() -> torch.Tensor:
        rhs = torch.zeros_like(prior)
        for Wo, gate, z in zip(writer_blocks, gates, projections):
            rhs = rhs + ((target @ Wo) * gate).T @ z
        return rhs

    def objective(S: torch.Tensor) -> float:
        prediction = torch.zeros_like(target)
        for Wo, gate, z in zip(writer_blocks, gates, projections):
            prediction = prediction + (gate * (z @ S.T)) @ Wo.T
        return float((prediction - target).pow(2).sum())

    scale = 0.0
    for Wo, gate, z in zip(writer_blocks, gates, projections):
        scale += float(z.pow(2).mean()) * float(gate.pow(2).mean()) * float(Wo.pow(2).sum())
    lam = rel * max(scale, 1e-30)

    S = prior.clone()
    b = right_hand_side() + lam * prior
    r = b - (apply(S) + lam * S)
    p = r.clone()
    rs = float((r * r).sum())
    entry = objective(prior)
    for _ in range(iterations):
        if rs <= 1e-24 * max(entry, 1.0):
            break
        Ap = apply(p) + lam * p
        alpha = rs / max(float((p * Ap).sum()), 1e-300)
        S = S + alpha * p
        r = r - alpha * Ap
        rs_new = float((r * r).sum())
        p = r + (rs_new / max(rs, 1e-300)) * p
        rs = rs_new
    update = float((S - prior).norm() / prior.norm().clamp_min(1e-30))
    if update > UPDATE_SCALE_CAP:
        # THE LAW is a guard on the installed weight; back off to the prior and
        # say so rather than installing an update the law does not admit.
        return prior, {
            "cg_refit": "refused",
            "update_scale": update,
            "objective_entry": entry,
        }
    return S, {
        "cg_refit": "installed",
        "update_scale": update,
        "objective_entry": entry,
        "objective_exit": objective(S),
        "lambda_absolute": lam,
    }


def orthogonal_procrustes(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, float]:
    """``argmin_R ||A R - B||_F`` over orthogonal ``R``, plus the cosine it buys.

    Census only in this lane.  Rotating a GDN key head's frame is an exact gauge
    of the delta rule (no RoPE, and the kernel L2-normalises q and k) -- but only
    AFTER the depthwise conv and its SiLU, which a rotation of the projection
    rows cannot commute with.  So the number this returns says how much
    alignment a gauge would buy and the install does not use it.
    """

    U, S, Vh = torch.linalg.svd(A.T @ B, full_matrices=False)
    R = U @ Vh
    cos = float(
        torch.nn.functional.cosine_similarity(A @ R, B, dim=1).mean()
    )
    return R, cos


def inverse_softplus(y: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    """``log(exp(y) - 1)`` computed without overflowing for large ``y``.

    ``log(expm1(y))`` overflows at ``y > 709``; ``y + log(1 - exp(-y))`` does
    not.  The parentheses around the negation are load-bearing: ``expm1(-y)`` is
    NEGATIVE, so ``-expm1(-y).clamp_min(...)`` clamps before negating and hands
    ``log`` a negative number -- silently, as NaN, all the way into a decay row.
    """

    y = y.clamp_min(floor)
    return y + torch.log((-torch.expm1(-y)).clamp_min(1e-300))
