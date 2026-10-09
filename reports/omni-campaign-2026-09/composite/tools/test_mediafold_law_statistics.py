"""CPU parity with the pre-hoist ladder and direct matrix-operation counts."""
from pathlib import Path
import math
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parents[1] / "media"))
from mediafold import linalg as tool

LEGACY_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def legacy_prior(design, targets, prior, ridge_rel=1e-6):
    """Retained pre-hoist calculation, independent of the new statistics helper."""
    matrix = design.detach().to(torch.float64).cpu()
    answer = targets.detach().to(torch.float64).cpu()
    anchor = prior.detach().to(torch.float64).cpu()
    if anchor.shape[1] != matrix.shape[1] or anchor.shape[0] != answer.shape[1]:
        raise ValueError("prior shape does not match the design / target pair")
    gram = matrix.T @ matrix
    cross = matrix.T @ answer
    ridge_lambda = float(ridge_rel) * float(torch.diagonal(gram).mean().item())
    shifted = gram + ridge_lambda * torch.eye(gram.shape[0], dtype=gram.dtype)
    anchored = cross + ridge_lambda * anchor.T
    try:
        solution = torch.cholesky_solve(anchored, torch.linalg.cholesky(shifted))
    except Exception:
        solution = torch.linalg.lstsq(shifted, anchored).solution
    weight = solution.T.contiguous()
    update = float(((weight - anchor).norm() / anchor.norm().clamp_min(1e-30)).item())
    return weight, {"ridge_lambda": ridge_lambda, "update_scale": update}


def legacy_scores(prediction, target):
    """Retained original scoring expressions, independent of the cached scorer."""
    predicted = prediction.detach().to(torch.float64)
    wanted = target.detach().to(torch.float64)
    if predicted.shape != wanted.shape:
        raise ValueError(f"score shape mismatch: {tuple(predicted.shape)} vs {tuple(wanted.shape)}")
    residual = predicted - wanted
    denominator = wanted.norm().clamp_min(1.0e-30)
    rel_l2 = float((residual.norm() / denominator).item())
    numerator = (predicted * wanted).sum(dim=-1)
    scale = predicted.norm(dim=-1).clamp_min(1.0e-30) * wanted.norm(dim=-1).clamp_min(1.0e-30)
    cosine = numerator / scale
    order = torch.sort(cosine).values
    index = max(0, min(order.numel() - 1, int(math.floor(0.05 * order.numel()))))
    total_variance = (wanted - wanted.mean(dim=0, keepdim=True)).pow(2).sum()
    r_squared = float((1.0 - residual.pow(2).sum() / total_variance.clamp_min(1.0e-30)).item())
    return {"rel_l2": rel_l2, "cos": float(cosine.mean().item()),
            "cos_p05": float(order[index].item()), "r2": r_squared,
            "rows": int(wanted.shape[0]), "width": int(wanted.shape[1])}


def legacy_law(design, targets, prior, ladder=LEGACY_LADDER):
    best_weight, best_notes, best_error, trail = None, {}, float("inf"), []
    for rung in ladder:
        weight, notes = legacy_prior(design, targets, prior, float(rung))
        scores = legacy_scores(design.detach().to(torch.float64).cpu() @ weight.T,
                                   targets.detach().to(torch.float64).cpu())
        vetoed = notes["update_scale"] > 1.0
        trail.append({"ridge_rel": float(rung), "rel_l2": scores["rel_l2"],
                      "update_scale": notes["update_scale"], "vetoed": bool(vetoed)})
        if not vetoed and scores["rel_l2"] < best_error:
            best_weight, best_notes, best_error = weight, {**notes, "ridge_rel": float(rung), "scores": scores}, scores["rel_l2"]
    if best_weight is None:
        best_weight = prior.detach().to(torch.float64).cpu().clone()
        best_notes = {"ridge_rel": None, "update_scale": 0.0, "fell_back_to_prior": True}
    best_notes["ladder"] = trail
    return best_weight, best_notes


def problem(kind):
    generator = torch.Generator().manual_seed(8411378)
    design = torch.randn(48, 6, dtype=torch.float64, generator=generator)
    prior = torch.randn(4, 6, dtype=torch.float64, generator=generator) * .2
    answer = design @ (prior + torch.randn(4, 6, dtype=torch.float64, generator=generator) * .04).T
    ladder = LEGACY_LADDER
    if kind == "ill_conditioned":
        design[:, 1] = design[:, 0] * (1 + 1e-12)
        answer = design @ (prior * 1.2).T
    elif kind == "affine":
        design[:, -1] = 1
        prior[:, -1] = torch.tensor([2., -3., .5, 1.])
        truth = prior.clone(); truth[:, -1] += .25
        answer = design @ truth.T
    elif kind == "veto_some":
        answer = design @ (prior * 3).T
    elif kind == "veto_all":
        answer = design @ (prior * 1000).T
    elif kind == "singular_lstsq":
        design.zero_()
    elif kind == "empty_ladder":
        ladder = ()
    elif kind == "bf16_inputs":
        design, answer, prior = design.bfloat16(), answer.bfloat16(), prior.bfloat16()
    return design, answer, prior, ladder


@pytest.mark.parametrize("kind", ["well_conditioned", "ill_conditioned", "affine", "veto_some", "veto_all",
                                   "singular_lstsq", "empty_ladder", "bf16_inputs"])
def test_hoisted_ladder_is_bitwise_equal_to_pre_hoist_weights_and_all_diagnostics(kind):
    design, answer, prior, ladder = problem(kind)
    expected, expected_notes = legacy_law(design, answer, prior, ladder)
    actual, actual_notes = tool.law_abiding_solve(design, answer, prior, ladder=ladder)
    assert torch.equal(actual, expected)
    assert actual_notes == expected_notes
    if kind == "veto_some":
        assert any(r["vetoed"] for r in actual_notes["ladder"])
        assert any(not r["vetoed"] for r in actual_notes["ladder"])
    if kind in {"veto_all", "empty_ladder"}:
        assert actual_notes["fell_back_to_prior"]
        assert torch.equal(actual, prior.double())


def test_public_prior_anchored_api_retains_exact_coefficients_and_notes():
    design, answer, prior, _ = problem("affine")
    for rung in LEGACY_LADDER:
        expected, notes = legacy_prior(design, answer, prior, rung)
        actual, actual_notes = tool.prior_anchored(design, answer, prior, ridge_rel=rung)
        assert torch.equal(actual, expected)
        assert actual_notes == notes
    assert tool.LAMBDA_LADDER == LEGACY_LADDER


class MatrixProducts(TorchDispatchMode):
    def __init__(self, design, targets):
        super().__init__()
        self.design_pointer = design.data_ptr()
        self.target_pointer = targets.data_ptr()
        self.transposed_shape = (design.shape[1], design.shape[0])
        self.gram = self.cross = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.mm.default and tuple(args[0].shape) == self.transposed_shape:
            if args[1].data_ptr() == self.design_pointer:
                self.gram += 1
            elif args[1].data_ptr() == self.target_pointer:
                self.cross += 1
        return func(*args, **(kwargs or {}))


def test_nine_rungs_compute_gram_and_target_cross_once_instead_of_nine_times():
    design, answer, prior, _ = problem("well_conditioned")
    legacy = MatrixProducts(design, answer)
    with legacy:
        expected, expected_notes = legacy_law(design, answer, prior)
    assert (legacy.gram, legacy.cross) == (9, 9)
    optimized = MatrixProducts(design, answer)
    with optimized:
        actual, actual_notes = tool.law_abiding_solve(design, answer, prior)
    assert (optimized.gram, optimized.cross) == (1, 1)
    assert torch.equal(actual, expected)
    assert actual_notes == expected_notes
