"""Closed-form solvers for the media fold harness.

Naming and constants follow the LFM2 surgery tooling
(``reports/matrix-campaign-20260810/run/mx45/mx45_build.py`` and
``src/neural_forge/hf_closed_form_fit.py``) so results are comparable across lanes:
``RIDGE_REL`` 1e-6, the same lambda ladder, ``relL2 / cos / cos_p05`` score triple,
prior-anchored ridge with an ``update_scale <= 1`` veto, and greedy pivoted-Cholesky
column selection (verbatim channel selection, never SVD, on gated FFN channels).

Every solve runs in float64. No gradients anywhere in this module -- the fold stage is
closed form by rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

RIDGE_REL = 1.0e-6
LAMBDA_LADDER = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
IDENTITY_TOLERANCE = 5.0e-2


# --------------------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------------------


def _score_target_statistics(
    wanted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact fixed-target reductions; retain only two scalars and row norms."""

    denominator = wanted.norm().clamp_min(1.0e-30)
    row_norm = wanted.norm(dim=-1).clamp_min(1.0e-30)
    total_variance = (wanted - wanted.mean(dim=0, keepdim=True)).pow(2).sum()
    return denominator, row_norm, total_variance


def _write_scores_from_statistics(
    predicted: torch.Tensor,
    wanted: torch.Tensor,
    statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Original prediction-dependent scoring with normalized inputs and target stats."""

    denominator, target_row_norm, total_variance = statistics
    residual = predicted - wanted
    rel_l2 = float((residual.norm() / denominator).item())
    numerator = (predicted * wanted).sum(dim=-1)
    scale = predicted.norm(dim=-1).clamp_min(1.0e-30) * target_row_norm
    cosine = numerator / scale
    order = torch.sort(cosine).values
    index = max(0, min(order.numel() - 1, int(math.floor(0.05 * order.numel()))))
    r_squared = float((1.0 - residual.pow(2).sum() / total_variance.clamp_min(1.0e-30)).item())
    return {
        "rel_l2": rel_l2,
        "cos": float(cosine.mean().item()),
        "cos_p05": float(order[index].item()),
        "r2": r_squared,
        "rows": int(wanted.shape[0]),
        "width": int(wanted.shape[1]),
    }


def write_scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """relL2 / mean cosine / 5th-percentile cosine / R^2 over rows.

    ``prediction`` and ``target`` are [rows, width]. These are DIAGNOSTIC columns:
    behaviour is the only gate in this campaign.
    """

    predicted = prediction.detach().to(torch.float64)
    wanted = target.detach().to(torch.float64)
    if predicted.shape != wanted.shape:
        raise ValueError(f"score shape mismatch: {tuple(predicted.shape)} vs {tuple(wanted.shape)}")
    return _write_scores_from_statistics(predicted, wanted, _score_target_statistics(wanted))


def psnr(prediction: torch.Tensor, target: torch.Tensor, *, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB over the whole tensor."""

    predicted = prediction.detach().to(torch.float64)
    wanted = target.detach().to(torch.float64)
    span = float(data_range) if data_range is not None else float((wanted.max() - wanted.min()).item())
    if not math.isfinite(span) or span <= 0.0:
        span = 1.0
    mse = float((predicted - wanted).pow(2).mean().item())
    if mse <= 0.0:
        return float("inf")
    return float(10.0 * math.log10(span * span / mse))


def si_sdr(prediction: torch.Tensor, target: torch.Tensor, *, silence_energy: float = 1.0e-12) -> float:
    """Scale-invariant SDR in dB over waveform rows, ignoring silent target rows.

    A silent target row has no scale to be invariant to: ``alpha`` collapses to 0, the
    projection energy is 0, and the row contributes ``10*log10(1e-30) = -300 dB`` to the
    mean. The procedural corpus zero-pads up to half of every clip before its onset, so
    roughly a third of the rows are silent and the mean read -72 dB for a PARENT
    round-trip that the whole-clip probe measured at +10.3 dB. Silent rows are therefore
    excluded and their count is available through :func:`si_sdr_detail`.
    """

    return si_sdr_detail(prediction, target, silence_energy=silence_energy)["si_sdr_db"]


def si_sdr_detail(
    prediction: torch.Tensor, target: torch.Tensor, *, silence_energy: float = 1.0e-12
) -> dict[str, float]:
    """SI-SDR plus the bookkeeping: how many rows were scored and how many were silent."""

    predicted = prediction.detach().to(torch.float64).reshape(prediction.shape[0], -1)
    wanted = target.detach().to(torch.float64).reshape(target.shape[0], -1)
    wanted = wanted - wanted.mean(dim=-1, keepdim=True)
    predicted = predicted - predicted.mean(dim=-1, keepdim=True)
    energy = wanted.pow(2).mean(dim=-1)
    voiced = energy > float(silence_energy)
    if not bool(voiced.any()):
        return {
            "si_sdr_db": float("nan"),
            "scored_rows": 0.0,
            "silent_rows": float(wanted.shape[0]),
            "silence_energy": float(silence_energy),
        }
    predicted = predicted[voiced]
    wanted = wanted[voiced]
    alpha = (predicted * wanted).sum(dim=-1, keepdim=True) / wanted.pow(2).sum(dim=-1, keepdim=True)
    projection = alpha * wanted
    noise = predicted - projection
    ratio = projection.pow(2).sum(dim=-1) / noise.pow(2).sum(dim=-1).clamp_min(1.0e-30)
    return {
        "si_sdr_db": float((10.0 * torch.log10(ratio.clamp_min(1.0e-30))).mean().item()),
        "scored_rows": float(int(voiced.sum().item())),
        "silent_rows": float(int((~voiced).sum().item())),
        "silence_energy": float(silence_energy),
    }


def best_linear_rank_psnr(
    fit_rows: torch.Tensor,
    holdout_rows: torch.Tensor,
    rank: int,
    *,
    data_range: float = 2.0,
) -> dict[str, Any]:
    """PSNR of the BEST rank-``rank`` linear round-trip: the PCA bound.

    ``E`` and ``D`` are linear and factor through ``d_latent`` numbers, so no linear C15
    map can beat the rank-``d_latent`` PCA of the same rows. Reporting this bound beside
    the fitted map separates two very different findings: "the fit is bad" (fitted well
    below the bound) from "a linear map of this width cannot reach the parent" (fitted at
    the bound, bound below the parent's ceiling).
    """

    fit = fit_rows.detach().to(torch.float64)
    held = holdout_rows.detach().to(torch.float64)
    mean = fit.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(fit - mean, full_matrices=False)
    keep = min(int(rank), int(vh.shape[0]))
    basis = vh[:keep]
    reconstruction = (held - mean) @ basis.transpose(0, 1) @ basis + mean
    return {
        "rank": keep,
        "psnr_db": psnr(reconstruction, held, data_range=data_range),
        "scores": write_scores(reconstruction, held),
        "retained_energy": float(
            (singular[:keep].pow(2).sum() / singular.pow(2).sum().clamp_min(1.0e-30)).item()
        ),
    }


# --------------------------------------------------------------------------------------
# ridge
# --------------------------------------------------------------------------------------


@dataclass
class LinearMap:
    """``y = x @ weight.T + bias`` with ``weight`` [out, in]."""

    weight: torch.Tensor
    bias: torch.Tensor
    ridge_lambda: float = 0.0
    fit_scores: dict[str, float] = field(default_factory=dict)
    holdout_scores: dict[str, float] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(rows.dtype).to(rows.device)
        bias = self.bias.to(rows.dtype).to(rows.device)
        return rows @ weight.transpose(0, 1) + bias

    def state(self, prefix: str) -> dict[str, torch.Tensor]:
        return {f"{prefix}.weight": self.weight, f"{prefix}.bias": self.bias}


def _split_rows(count: int, holdout_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be in [0, 1)")
    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(count, generator=generator)
    held = int(round(holdout_fraction * count))
    if holdout_fraction > 0.0 and held == 0:
        held = 1
    if held >= count:
        raise ValueError("holdout split consumed every row")
    return order[held:], order[:held]


def _gram_solve(
    design: torch.Tensor,
    target: torch.Tensor,
    ridge_lambda: float,
) -> torch.Tensor:
    """Solve ``(A^T A + lambda I) W = A^T Y`` in float64, returning W [in, out]."""

    gram = design.transpose(0, 1) @ design
    cross = design.transpose(0, 1) @ target
    width = gram.shape[0]
    eye = torch.eye(width, dtype=gram.dtype, device=gram.device)
    for attempt in range(8):
        shifted = gram + (ridge_lambda * (10.0**attempt)) * eye
        try:
            factor = torch.linalg.cholesky(shifted)
            return torch.cholesky_solve(cross, factor)
        except Exception:
            continue
    return torch.linalg.lstsq(gram + ridge_lambda * eye, cross).solution


def fit_ridge_linear(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_rel: float = RIDGE_REL,
    fit_bias: bool = True,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> LinearMap:
    """Ridge-regress ``targets`` on ``inputs`` in float64.

    ``inputs`` [rows, d_in], ``targets`` [rows, d_out]. The bias column is never
    regularized. ``ridge_rel`` scales lambda by ``mean(diag(A^T A))`` so the same
    constant transfers across feature geometries.
    """

    rows_in = inputs.detach().to(torch.float64).cpu()
    rows_out = targets.detach().to(torch.float64).cpu()
    if rows_in.ndim != 2 or rows_out.ndim != 2 or rows_in.shape[0] != rows_out.shape[0]:
        raise ValueError("ridge inputs and targets must be [rows, width] with equal row counts")
    fit_index, hold_index = _split_rows(rows_in.shape[0], holdout_fraction, seed)

    def design_of(block: torch.Tensor) -> torch.Tensor:
        if not fit_bias:
            return block
        ones = torch.ones(block.shape[0], 1, dtype=block.dtype)
        return torch.cat((block, ones), dim=1)

    design = design_of(rows_in[fit_index])
    answer = rows_out[fit_index]
    gram_diagonal = design.pow(2).sum(dim=0)
    ridge_lambda = float(ridge_rel) * float(gram_diagonal.mean().item())
    solution = _gram_solve(design, answer, ridge_lambda)
    if fit_bias:
        weight = solution[:-1].transpose(0, 1).contiguous()
        bias = solution[-1].contiguous()
    else:
        weight = solution.transpose(0, 1).contiguous()
        bias = torch.zeros(rows_out.shape[1], dtype=torch.float64)

    result = LinearMap(weight=weight, bias=bias, ridge_lambda=ridge_lambda)
    result.fit_scores = write_scores(rows_in[fit_index] @ weight.transpose(0, 1) + bias, answer)
    if hold_index.numel():
        held_prediction = rows_in[hold_index] @ weight.transpose(0, 1) + bias
        result.holdout_scores = write_scores(held_prediction, rows_out[hold_index])
    result.notes = {
        "ridge_rel": float(ridge_rel),
        "fit_bias": bool(fit_bias),
        "fit_rows": int(fit_index.numel()),
        "holdout_rows": int(hold_index.numel()),
        "seed": int(seed),
    }
    return result


def fit_ridge_lambda_ladder(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    ladder: Sequence[float] = LAMBDA_LADDER,
    fit_bias: bool = True,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> LinearMap:
    """Walk the lambda ladder and keep the rung with the best held-out relL2."""

    best: LinearMap | None = None
    trail: list[dict[str, Any]] = []
    for rung in ladder:
        candidate = fit_ridge_linear(
            inputs,
            targets,
            ridge_rel=float(rung),
            fit_bias=fit_bias,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        score = candidate.holdout_scores.get("rel_l2", candidate.fit_scores["rel_l2"])
        trail.append({"ridge_rel": float(rung), "rel_l2": float(score)})
        if best is None:
            best = candidate
        else:
            incumbent = best.holdout_scores.get("rel_l2", best.fit_scores["rel_l2"])
            if score < incumbent:
                best = candidate
    assert best is not None
    best.notes["lambda_ladder"] = trail
    return best


def _prior_anchored_from_statistics(
    gram: torch.Tensor,
    cross: torch.Tensor,
    anchor: torch.Tensor,
    *,
    ridge_rel: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One unchanged ridge rung from fixed FP64 CPU sufficient statistics."""

    ridge_lambda = float(ridge_rel) * float(torch.diagonal(gram).mean().item())
    eye = torch.eye(gram.shape[0], dtype=gram.dtype)
    shifted = gram + ridge_lambda * eye
    anchored = cross + ridge_lambda * anchor.transpose(0, 1)
    try:
        factor = torch.linalg.cholesky(shifted)
        solution = torch.cholesky_solve(anchored, factor)
    except Exception:
        solution = torch.linalg.lstsq(shifted, anchored).solution
    weight = solution.transpose(0, 1).contiguous()
    update_scale = float(
        ((weight - anchor).norm() / anchor.norm().clamp_min(1.0e-30)).item()
    )
    return weight, {"ridge_lambda": ridge_lambda, "update_scale": update_scale}


def prior_anchored(
    design: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    *,
    ridge_rel: float = RIDGE_REL,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``argmin ||A W^T - Y||^2 + lambda ||W - prior||^2`` in float64.

    ``prior`` is [out, in] -- the parent's own weight, so a fold that learns nothing
    still reproduces the parent. Returns ``(weight, notes)`` with the ``update_scale``
    diagnostic used by ``law_abiding_solve``'s veto.
    """

    matrix = design.detach().to(torch.float64).cpu()
    answer = targets.detach().to(torch.float64).cpu()
    anchor = prior.detach().to(torch.float64).cpu()
    if anchor.shape[1] != matrix.shape[1] or anchor.shape[0] != answer.shape[1]:
        raise ValueError("prior shape does not match the design / target pair")
    gram = matrix.transpose(0, 1) @ matrix
    cross = matrix.transpose(0, 1) @ answer
    return _prior_anchored_from_statistics(gram, cross, anchor, ridge_rel=ridge_rel)


def law_abiding_solve(
    design: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    *,
    ladder: Sequence[float] = LAMBDA_LADDER,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Prior-anchored ridge over the lambda ladder with the update_scale <= 1 veto.

    Mirrors ``mx45_build.law_abiding_solve``: rungs whose solution moves further than
    the prior's own norm are vetoed, and among the survivors the best fit wins.
    Fixed Gram/cross and target score statistics are computed once; every rung
    retains its original solve, score, ranking and veto arithmetic. An empty
    ladder still keeps prior.
    """

    best_weight: torch.Tensor | None = None
    best_notes: dict[str, Any] = {}
    best_error = float("inf")
    trail: list[dict[str, Any]] = []
    statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    score_statistics: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for rung in ladder:
        if statistics is None:
            # Inputs and their products are identical for every lambda. Keep
            # the original FP64 CPU BLAS shapes/order, once per ladder call.
            matrix = design.detach().to(torch.float64).cpu()
            answer = targets.detach().to(torch.float64).cpu()
            anchor = prior.detach().to(torch.float64).cpu()
            if anchor.shape[1] != matrix.shape[1] or anchor.shape[0] != answer.shape[1]:
                raise ValueError("prior shape does not match the design / target pair")
            gram = matrix.transpose(0, 1) @ matrix
            cross = matrix.transpose(0, 1) @ answer
            statistics = (gram, cross, anchor)
            # Release full target-variance temporaries before prediction/residual
            # allocation, then reuse the exact reductions for every lambda.
            score_statistics = _score_target_statistics(answer)
        weight, notes = _prior_anchored_from_statistics(*statistics, ridge_rel=float(rung))
        prediction = matrix @ weight.transpose(0, 1)
        scores = _write_scores_from_statistics(prediction, answer, score_statistics)
        vetoed = notes["update_scale"] > 1.0
        trail.append(
            {
                "ridge_rel": float(rung),
                "rel_l2": scores["rel_l2"],
                "update_scale": notes["update_scale"],
                "vetoed": bool(vetoed),
            }
        )
        if vetoed:
            continue
        if scores["rel_l2"] < best_error:
            best_error = scores["rel_l2"]
            best_weight = weight
            best_notes = {**notes, "ridge_rel": float(rung), "scores": scores}
    if best_weight is None:
        # every rung moved further than the prior's norm: keep the prior itself.
        best_weight = prior.detach().to(torch.float64).cpu().clone()
        best_notes = {"ridge_rel": None, "update_scale": 0.0, "fell_back_to_prior": True}
    best_notes["ladder"] = trail
    return best_weight, best_notes


# --------------------------------------------------------------------------------------
# one closed-form nonlinearity: random-feature lift
# --------------------------------------------------------------------------------------


def situ_activation(gate: torch.Tensor, up: torch.Tensor, *, beta: float = 4.0, linear_beta: float = 25.0) -> torch.Tensor:
    """K3 / Chimera SiTU: ``b*tanh(g/b)*sigmoid(g) * bl*tanh(u/bl)`` in float32+."""

    g = gate.to(torch.float64)
    u = up.to(torch.float64)
    if math.isinf(beta):
        left = torch.nn.functional.silu(g)
    else:
        left = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if math.isinf(linear_beta):
        right = u
    else:
        right = linear_beta * torch.tanh(u / linear_beta)
    return left * right


@dataclass
class RandomFeatureLift:
    """Fixed random projection + one nonlinearity, with the raw columns kept alongside.

    Two properties make this usable as a fold rung rather than a gamble:

    * ``include_linear`` concatenates the untouched input columns to the lifted ones, so
      the nonlinear rung is a CORRECTION on top of the linear map and can never be worse
      than it (a ladder rung must not be able to lose ground);
    * ``input_scale`` sets the kernel bandwidth from the data instead of assuming unit
      variance -- an unscaled RFF on 32-dim audio frames saturates ``cos`` and behaves
      like noise.

    The projection is fixed at construction and never trained, but it IS stored, so its
    parameters are counted in full (the estate's storage rule: if it is part of the
    model it counts).
    """

    projection: torch.Tensor  # [features, d_in]
    offset: torch.Tensor  # [features]
    kind: str
    include_linear: bool = True
    beta: float = 4.0
    linear_beta: float = 25.0

    def apply(self, rows: torch.Tensor) -> torch.Tensor:
        matrix = self.projection.to(torch.float64)
        shift = self.offset.to(torch.float64)
        source = rows.to(torch.float64)
        pre = source @ matrix.transpose(0, 1) + shift
        if self.kind == "rff":
            lifted = math.sqrt(2.0 / matrix.shape[0]) * torch.cos(pre)
        elif self.kind == "situ":
            half = pre.shape[-1] // 2
            lifted = situ_activation(pre[..., :half], pre[..., half:], beta=self.beta, linear_beta=self.linear_beta)
        else:
            raise ValueError(f"unknown lift kind {self.kind!r}")
        if self.include_linear:
            return torch.cat((source, lifted), dim=-1)
        return lifted

    @property
    def parameters(self) -> int:
        return int(self.projection.numel() + self.offset.numel())


def make_random_feature_lift(
    d_in: int,
    features: int,
    *,
    kind: str = "rff",
    gamma: float = 1.0,
    input_scale: float = 1.0,
    include_linear: bool = True,
    seed: int = 0,
) -> RandomFeatureLift:
    generator = torch.Generator().manual_seed(int(seed))
    width = int(features) if kind == "rff" else 2 * int(features)
    spread = math.sqrt(2.0 * float(gamma)) * float(input_scale)
    projection = torch.randn(width, int(d_in), generator=generator, dtype=torch.float64) * spread
    if kind == "rff":
        offset = torch.rand(width, generator=generator, dtype=torch.float64) * (2.0 * math.pi)
    else:
        offset = torch.zeros(width, dtype=torch.float64)
    return RandomFeatureLift(projection=projection, offset=offset, kind=kind, include_linear=include_linear)


def data_bandwidth(rows: torch.Tensor) -> float:
    """Median-heuristic bandwidth: ``1 / (std * sqrt(width))``.

    Keeps the pre-activation of the lift at order 1 whatever the raw feature scale is,
    which is the difference between a working kernel and a noise generator.
    """

    matrix = rows.detach().to(torch.float64)
    spread = float(matrix.std().item())
    if not math.isfinite(spread) or spread <= 0.0:
        return 1.0
    return 1.0 / (spread * math.sqrt(float(matrix.shape[1])))


def fit_kernel_ridge_lift(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    features: int = 512,
    kind: str = "rff",
    gamma: float = 1.0,
    include_linear: bool = True,
    holdout_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[RandomFeatureLift, LinearMap]:
    """Closed-form kernel ridge via an explicit random-feature lift.

    Returns the (stored) lift and the linear readout fitted on top of it. Parameter cost
    = ``lift.parameters + readout weight + bias`` -- always report both when quoting a
    compression ratio; a lift that buys 1 dB for 10x the parameters is not a fold.
    """

    scale = data_bandwidth(inputs)
    lift = make_random_feature_lift(
        inputs.shape[1], features, kind=kind, gamma=gamma, input_scale=scale,
        include_linear=include_linear, seed=seed,
    )
    lifted = lift.apply(inputs)
    readout = fit_ridge_lambda_ladder(
        lifted,
        targets,
        fit_bias=True,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    readout.notes["lift"] = {
        "kind": kind,
        "features": int(features),
        "gamma": float(gamma),
        "input_scale": scale,
        "include_linear": bool(include_linear),
        "parameters": lift.parameters,
        "lifted_width": int(lifted.shape[1]),
    }
    return lift, readout


# --------------------------------------------------------------------------------------
# charts
# --------------------------------------------------------------------------------------


def orthogonal_procrustes(source: torch.Tensor, destination: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """``argmin_R ||X R - Y||`` over orthogonal R (equal widths only).

    Returns R as an [out, in] weight so it can be used exactly like a ridge map.
    """

    x = source.detach().to(torch.float64).cpu()
    y = destination.detach().to(torch.float64).cpu()
    if x.shape[1] != y.shape[1]:
        raise ValueError("orthogonal Procrustes requires equal widths; use fit_ridge_linear instead")
    cross = x.transpose(0, 1) @ y
    u, singular, vh = torch.linalg.svd(cross, full_matrices=False)
    rotation = u @ vh
    weight = rotation.transpose(0, 1).contiguous()
    prediction = x @ rotation
    return weight, {
        "scores": write_scores(prediction, y),
        # singular values are non-negative, so sum/abs-sum was identically 1.0 -- a dead
        # column in every Procrustes receipt. The informative number is how much of the
        # cross-covariance the rotation actually aligns: the top singular value's share.
        "singular_top_share": float((singular.max() / singular.sum().clamp_min(1e-30)).item()),
        "singular_effective_rank": float(
            torch.exp(
                -(
                    (singular / singular.sum().clamp_min(1e-30)).clamp_min(1e-30)
                    * (singular / singular.sum().clamp_min(1e-30)).clamp_min(1e-30).log()
                ).sum()
            ).item()
        ),
        "orthogonality_error": float((rotation.transpose(0, 1) @ rotation - torch.eye(rotation.shape[0], dtype=torch.float64)).abs().max().item()),
    }


def svd_chart(rows: torch.Tensor, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """One global orthonormal chart from pooled residual rows.

    Mirrors ``experiments/run_chimera_omni_toy_fold.build_residual_chart``: SVD of the
    pooled rows, top-``rank`` right singular vectors. Returned as [rank, width], i.e.
    ``coords = rows @ chart.T`` and ``rows_hat = coords @ chart``.
    """

    matrix = rows.detach().to(torch.float64).cpu()
    centred = matrix - matrix.mean(dim=0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centred, full_matrices=False)
    keep = min(int(rank), vh.shape[0])
    chart = vh[:keep].contiguous()
    energy = float((singular[:keep].pow(2).sum() / singular.pow(2).sum().clamp_min(1e-30)).item())
    return chart, {"rank": keep, "retained_energy": energy}


def random_chart(d_out: int, d_in: int, *, seed: int = 0) -> torch.Tensor:
    """Random gaussian chart, scaled to unit expected gain. The control arm."""

    generator = torch.Generator().manual_seed(int(seed))
    return torch.randn(int(d_out), int(d_in), generator=generator, dtype=torch.float64) / math.sqrt(float(d_in))


# --------------------------------------------------------------------------------------
# verbatim channel selection
# --------------------------------------------------------------------------------------


def pivoted_columns_gram(gram: torch.Tensor, keep: int) -> torch.Tensor:
    """Greedy pivoted Cholesky column selection on a Gram matrix.

    Vectorised (``diag.argmax`` per step + rank-1 update), O(n*k^2). Returns the chosen
    column indices in selection order. This is the verbatim-channel-selection primitive
    -- the LFM2 record's law 40/42 says never replace it with an SVD on gated FFNs.
    """

    matrix = gram.detach().to(torch.float64).cpu().clone()
    width = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pivoted_columns_gram expects a square Gram matrix")
    target = min(int(keep), width)
    diagonal = torch.diagonal(matrix).clone()
    chosen: list[int] = []
    factor = torch.zeros(width, target, dtype=torch.float64)
    for step in range(target):
        pivot = int(torch.argmax(diagonal).item())
        if diagonal[pivot] <= 1.0e-24:
            break
        chosen.append(pivot)
        column = matrix[:, pivot].clone()
        if step:
            column = column - factor[:, :step] @ factor[pivot, :step]
        column = column / torch.sqrt(diagonal[pivot].clamp_min(1.0e-30))
        factor[:, step] = column
        diagonal = (diagonal - column.pow(2)).clamp_min(0.0)
        diagonal[pivot] = 0.0
    return torch.tensor(chosen, dtype=torch.long)


def pivoted_columns(features: torch.Tensor, keep: int, *, chunk: int = 8192) -> torch.Tensor:
    """Pivoted column selection from a feature matrix [rows, channels]."""

    matrix = features.detach().to(torch.float64).cpu()
    gram = torch.zeros(matrix.shape[1], matrix.shape[1], dtype=torch.float64)
    for start in range(0, matrix.shape[0], int(chunk)):
        block = matrix[start : start + int(chunk)]
        gram += block.transpose(0, 1) @ block
    return pivoted_columns_gram(gram, keep)


def compression_receipt(parent_parameters: int, child_parameters: int, component: str) -> dict[str, Any]:
    """Per-COMPONENT compression, receipt-verified from tensor shapes.

    ``>= 4x`` is the estate's floor for calling something a fold. This function reports
    the ratio and whether the floor holds; it never silently accepts a lower rung.
    """

    ratio = float(parent_parameters) / max(1.0, float(child_parameters))
    return {
        "component": component,
        "parent_parameters": int(parent_parameters),
        "child_parameters": int(child_parameters),
        "compression_x": ratio,
        "meets_4x_component_floor": bool(ratio >= 4.0),
    }
