"""C5 -- the Gated-DeltaNet value-head fold.  20% of a Qwen3.8-27B, and the
only mixer class the LFM2 campaign never touched at all.

Two arms, both config-standard because ``Qwen3_5TextConfig`` carries the four
geometry fields independently of ``hidden_size``
(``linear_num_key_heads``, ``linear_num_value_heads``, ``linear_key_head_dim``,
``linear_value_head_dim`` -- configuration_qwen3_5.py:93-96):

* **Arm A, head COUNT.**  48/16 -> 24/8 -> 12/4 at the 27B; 32/16 -> 16/8 -> 8/4
  at the 4B.  Value heads that share a key head are merged in place.
* **Arm B, head DIM.**  128 -> 32 on the value side is 4x on ``in_proj_z``, the
  value third of ``in_proj_qkv``, ``out_proj`` and the recurrent state's value
  axis; the same on the key side is 4x on the q/k two-thirds and the state's key
  axis.  The reason ``head_dim`` folds died on LFM2 -- RoPE plane arithmetic --
  is simply absent here: the GDN has no RoPE and the kernel L2-normalises q and
  k (``use_qk_l2norm_in_kernel=True``, modeling 532/543), so a head's key space
  is defined only up to an orthogonal transform.

**What is exact and what is not**, stated once here rather than implied:

* the projections ``in_proj_qkv``/``in_proj_z``/``in_proj_b``/``in_proj_a`` are
  plain readers of the normed residual, so a linear map on their rows is exact
  *as a map on the rows*;
* the depthwise ``conv1d`` + SiLU sits between the projection and the delta rule
  and commutes with NOTHING -- not a rotation, not a sign flip, not a
  contraction.  Every arm here therefore combines conv taps by an explicit,
  named approximation and records ``conv_frame_mismatch``;
* the gated RMSNorm's elementwise weight and its ``silu(z)`` restrict the exact
  value-side gauge to a per-dimension sign, so Procrustes is run as a CENSUS and
  installs nothing;
* the query scale ``1/sqrt(head_k_dim)`` (modeling 275, 340) is applied after
  the L2 normalisation and its result is then RMS-normalised by the gated norm
  (199-200), which removes any global scale on ``o``.  **A change of
  ``linear_key_head_dim`` therefore needs no scale correction at all** -- that is
  an algebraic fact about this kernel, and it is recorded in the receipt so a
  reader can check the claim rather than trust it.

Everything the approximations cost is repaired where it can be: ``out_proj`` is
always re-solved against the parent block's own output, on the CHILD's actual
gated stream, which is why every arm installs first and solves second.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

import qf_arch
from qf_arch import LINEAR
from qf_corpus import fit_check_rows
from qf_harness import pristine_copy
from qf_solve import (
    gated_out_proj_solve,
    inverse_softplus,
    law_abiding_solve,
    matrix_sqrt,
    orthogonal_procrustes,
    prior_anchored_gram,
    project_writer,
    whitened_lra,
    write_scores,
)


# --------------------------------------------------------------- index helpers
def head_slices(mixer: nn.Module) -> dict[str, Any]:
    """Every row/column block of a GDN, by name.

    ``in_proj_qkv`` is one ``[2*key_dim + value_dim, H]`` tensor holding q, then
    k, then v; ``conv1d.weight`` is ``[same, 1, kernel]`` over the same channel
    order; ``out_proj`` is ``[H, value_dim]``.  Getting one of these offsets
    wrong produces a model that runs.
    """

    nk, nv = int(mixer.num_k_heads), int(mixer.num_v_heads)
    dk, dv = int(mixer.head_k_dim), int(mixer.head_v_dim)
    Kd = nk * dk
    return {
        "num_k_heads": nk,
        "num_v_heads": nv,
        "head_k_dim": dk,
        "head_v_dim": dv,
        "key_dim": Kd,
        "value_dim": nv * dv,
        "group": nv // nk,
        "q_rows": lambda m: torch.arange(m * dk, (m + 1) * dk),
        "k_rows": lambda m: torch.arange(Kd + m * dk, Kd + (m + 1) * dk),
        "v_rows": lambda j: torch.arange(2 * Kd + j * dv, 2 * Kd + (j + 1) * dv),
        "z_rows": lambda j: torch.arange(j * dv, (j + 1) * dv),
        "o_cols": lambda j: torch.arange(j * dv, (j + 1) * dv),
    }


def _new_gated_norm(old: nn.Module, size: int, eps: float, activation: str, device, dtype):
    """A gated RMSNorm of a new width, of whatever class the live model uses."""

    try:
        module = type(old)(size, eps=eps, activation=activation, device=device, dtype=dtype)
    except TypeError:
        module = type(old)(size, eps=eps)
    # `Qwen3_5RMSNormGated.__init__` swallows **kwargs, so the device/dtype above
    # can be silently ignored; the explicit move is what makes both classes land
    # on the same device as the rest of the mixer.
    return module.to(device=device, dtype=dtype)


def _linear(rows: torch.Tensor, hidden: int, device, dtype) -> nn.Linear:
    layer = nn.Linear(hidden, rows.shape[0], bias=False, dtype=dtype, device=device)
    layer.weight.data.copy_(rows.to(device=device, dtype=dtype))
    return layer


# ------------------------------------------------------------------ evidence
def _projection_moments(X: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """Second moment of ``rows @ x`` on the evidence -- the metric a fold is fitted in.

    Computed BEFORE the depthwise conv and its SiLU, because the flat row block
    the harness captures has no sequence axis to convolve along.  That is an
    approximation of the metric and it is labelled as one in every receipt this
    module writes (``value_metric``); it weights a selection, it is not a target.
    """

    Z = X @ rows.T
    return (Z.T @ Z) / max(X.shape[0], 1)


def _combine_conv(taps: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """Combine depthwise conv taps under a channel map ``P``.

    A depthwise conv is per-channel, so there is no correct way to push a mixing
    map through it.  The taps of the child channel are the ``|P|``-weighted mean
    of the parent taps it mixes -- which is exact when ``P`` selects one channel
    and is a documented approximation otherwise.
    """

    weight = P.abs()
    weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return weight @ taps


# ------------------------------------------------------------------- census
@torch.no_grad()
def value_head_census(harness, layer_index: int) -> dict[str, Any]:
    """Pairwise similarity of the GDN's heads, as an ORDERING for the merge.

    Key heads are compared on their L2-normalised q/k rows both raw and in the
    best orthogonal gauge (:func:`qf_solve.orthogonal_procrustes`); value heads
    on their ``out_proj`` blocks and on their decay/beta series.  Nothing here
    decides anything -- it picks which merge to build first.
    """

    layer = harness.layers()[layer_index]
    mixer = layer.linear_attn
    slices = head_slices(mixer)
    qkv = mixer.in_proj_qkv.weight.detach().to("cpu", torch.float64)
    out = mixer.out_proj.weight.detach().to("cpu", torch.float64)

    key_rows = [
        torch.cat([qkv[slices["q_rows"](m)], qkv[slices["k_rows"](m)]], dim=0)
        for m in range(slices["num_k_heads"])
    ]
    pairs = []
    for a in range(len(key_rows)):
        for b in range(a + 1, len(key_rows)):
            A = F.normalize(key_rows[a], dim=1)
            B = F.normalize(key_rows[b], dim=1)
            raw = float(F.cosine_similarity(A, B, dim=1).mean())
            _, gauged = orthogonal_procrustes(key_rows[a].T, key_rows[b].T)
            pairs.append(
                {"key_heads": [a, b], "cos_raw": raw, "cos_in_procrustes_gauge": gauged}
            )
    value_energy = [
        float(out[:, slices["o_cols"](j)].pow(2).sum()) for j in range(slices["num_v_heads"])
    ]
    decay = (-mixer.A_log.detach().float().exp()).to("cpu", torch.float64)
    return {
        "layer": layer_index,
        "key_head_pairs": pairs,
        "value_head_out_proj_energy": value_energy,
        "decay_rate_per_value_head": [float(v) for v in decay],
        "note": "ordering only; the merge is closed by a built child on the review",
    }


# -------------------------------------------------------------- arm A: counts
def _merge_gdn_counts(
    mixer: nn.Module,
    X: torch.Tensor,
    *,
    value_heads: int,
    key_heads: int,
    order: list[int],
    drop_arm: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build every child tensor for the head-count arm.  No install, no solve yet."""

    slices = head_slices(mixer)
    nk, nv = slices["num_k_heads"], slices["num_v_heads"]
    dk, dv, g = slices["head_k_dim"], slices["head_v_dim"], slices["group"]
    if nk % key_heads or nv % value_heads:
        raise SystemExit(f"{nk}/{nv} does not divide into {key_heads}/{value_heads}")
    r = nk // key_heads
    if nv // value_heads != r:
        raise SystemExit(
            f"key and value reduction differ ({nk // key_heads} vs {nv // value_heads}); "
            f"the child would not satisfy linear_num_value_heads % linear_num_key_heads == 0 "
            f"with the parent's group size"
        )
    if value_heads % key_heads:
        raise SystemExit(f"{value_heads} value heads do not divide into {key_heads} key heads")

    qkv = mixer.in_proj_qkv.weight.detach().to("cpu", torch.float64)
    taps = mixer.conv1d.weight.detach().to("cpu", torch.float64).squeeze(1)
    z = mixer.in_proj_z.weight.detach().to("cpu", torch.float64)
    b = mixer.in_proj_b.weight.detach().to("cpu", torch.float64)
    a = mixer.in_proj_a.weight.detach().to("cpu", torch.float64)
    dt = mixer.dt_bias.detach().to("cpu", torch.float64)
    A_log = mixer.A_log.detach().to("cpu", torch.float64)
    out = mixer.out_proj.weight.detach().to("cpu", torch.float64)
    hidden = int(qkv.shape[1])
    # ONE set of Grams for the whole layer.  Every child value head's beta and
    # decay refit below is a prior-anchored ridge over the SAME evidence rows,
    # and each used to form its own `A^T A` inside the loop -- see _refit_grams.
    shared_grams = _refit_grams(X)

    key_energy = [
        float(
            _projection_moments(X, qkv[slices["k_rows"](m)]).diagonal().sum()
        )
        for m in range(nk)
    ]
    value_energy = [float(out[:, slices["o_cols"](j)].pow(2).sum()) for j in range(nv)]

    groups = [order[s * r : (s + 1) * r] for s in range(key_heads)]
    child_q, child_k, child_v, child_z = [], [], [], []
    child_qk_taps, child_v_taps = [], []
    child_b, child_a, child_dt, child_A = [], [], [], []
    child_out = torch.zeros(hidden, value_heads * dv, dtype=torch.float64)
    detail: list[dict[str, Any]] = []

    for m_child, parents in enumerate(groups):
        weights = torch.tensor([max(key_energy[m], 1e-30) for m in parents], dtype=torch.float64)
        weights = weights / weights.sum()
        # Key side: an energy-weighted average of the parents' own rows.  The
        # Procrustes gauge that would align them first cannot be pushed through
        # the depthwise conv, so it stays a census (see value_head_census).
        child_q.append(sum(w * qkv[slices["q_rows"](m)] for w, m in zip(weights, parents)))
        child_k.append(sum(w * qkv[slices["k_rows"](m)] for w, m in zip(weights, parents)))
        child_qk_taps.append(
            (
                sum(w * taps[slices["q_rows"](m)] for w, m in zip(weights, parents)),
                sum(w * taps[slices["k_rows"](m)] for w, m in zip(weights, parents)),
            )
        )
        for t in range(g):
            parent_heads = [m * g + t for m in parents]
            child_index = m_child * g + t
            stack = torch.cat([out[:, slices["o_cols"](j)] for j in parent_heads], dim=1)
            v_rows = torch.cat([qkv[slices["v_rows"](j)] for j in parent_heads], dim=0)
            z_rows = torch.cat([z[slices["z_rows"](j)] for j in parent_heads], dim=0)

            if drop_arm:
                pick = max(parent_heads, key=lambda j: value_energy[j])
                P = torch.zeros(dv, len(parent_heads) * dv, dtype=torch.float64)
                offset = parent_heads.index(pick) * dv
                P[:, offset : offset + dv] = torch.eye(dv, dtype=torch.float64)
                lra = {"keep": dv, "parent_dim": int(P.shape[1]), "retained_energy": None,
                       "arm": "drop-lowest-energy"}
            else:
                gram = _projection_moments(X, v_rows)
                P, lra = whitened_lra(stack, gram, dv)

            child_v.append(P @ v_rows)
            child_z.append(P @ z_rows)
            child_v_taps.append(
                _combine_conv(torch.cat([taps[slices["v_rows"](j)] for j in parent_heads], dim=0), P)
            )
            child_out[:, child_index * dv : (child_index + 1) * dv] = project_writer(stack, P)

            vw = torch.tensor([max(value_energy[j], 1e-30) for j in parent_heads], dtype=torch.float64)
            vw = vw / vw.sum()
            anchor = parent_heads[int(vw.argmax())]
            beta_row, beta_refit = _refit_beta(shared_grams, b, parent_heads, vw)
            child_b.append(beta_row)
            child_A.append(A_log[anchor])
            row, bias, refit = _refit_decay(
                shared_grams, a, dt, A_log, parent_heads, vw, anchor
            )
            child_a.append(row)
            child_dt.append(bias)
            detail.append(
                {
                    "child_value_head": child_index,
                    "parent_value_heads": parent_heads,
                    "anchor": anchor,
                    "value_lra": lra,
                    "decay_refit": refit,
                    "beta_refit": beta_refit,
                }
            )

    tensors = {
        "in_proj_qkv": torch.cat(
            [torch.cat(child_q, dim=0), torch.cat(child_k, dim=0), torch.cat(child_v, dim=0)],
            dim=0,
        ),
        "conv1d": torch.cat(
            [
                torch.cat([pair[0] for pair in child_qk_taps], dim=0),
                torch.cat([pair[1] for pair in child_qk_taps], dim=0),
                torch.cat(child_v_taps, dim=0),
            ],
            dim=0,
        ),
        "in_proj_z": torch.cat(child_z, dim=0),
        "in_proj_b": torch.stack(child_b, dim=0),
        "in_proj_a": torch.stack(child_a, dim=0),
        "dt_bias": torch.stack(child_dt, dim=0),
        "A_log": torch.stack(child_A, dim=0),
        "out_proj": child_out,
    }
    report = {
        "arm": "drop" if drop_arm else "merge",
        "reduction": r,
        "key_head_groups": groups,
        "key_head_energy": key_energy,
        "value_head_energy": value_energy,
        "per_child_head": detail,
        "conv_frame_mismatch": (
            "a channel map does not commute with a depthwise conv or its SiLU, and "
            "this arm combines taps by TWO different rules: the q/k taps by the "
            "key-energy weights that merged the q/k rows themselves (no P exists on "
            "that side -- the merge is an average, not a projection), the value taps "
            "by the |P|-weighted mean of the parents' taps, which is exact only where "
            "P selects a single channel"
        ),
        "z_frame_mismatch": (
            "in_proj_z rows contracted by the value frame P; silu(z) is elementwise "
            "in the parent frame and cannot be pushed through P"
        ),
        "beta_frame_mismatch": (
            "the delta rule consumes sigmoid(b.x) (modeling_qwen3_5.py:466,516), not "
            "b.x, so a merged head's write strength cannot be a linear combination of "
            "its parents' LOGIT rows: sigmoid is concave above 0 and convex below, and "
            "the energy-weighted logit row understates a mixture of a saturated and an "
            "unsaturated parent (betas 0.0025 and 0.881 at weights 1/2 average to 0.44, "
            "the averaged logit gives 0.119).  _refit_beta therefore fits the row "
            "against sum_j w_j sigmoid(b_j.x) through the logit, and installs the fit "
            "ONLY where it beats the energy-weighted logit row on the evidence; where "
            "it does not, the weighted row is installed and the residual is the "
            "recorded beta_relL2"
        ),
        "value_metric": "pre-conv second moment of the value rows on the fit split",
        "value_refit_gram": (
            "one X^T X and one [X|1]^T [X|1] per LAYER, formed before the head loop "
            "and reused by every child head's beta and decay refit through "
            "prior_anchored_gram; the per-head right-hand sides are the only O(N h) "
            "work left inside the loop"
        ),
    }
    return tensors, report


def _refit_grams(X: torch.Tensor) -> dict[str, torch.Tensor]:
    """The design matrices and Grams a layer's value-head refits all share.

    Both repairs below are prior-anchored ridges over the SAME evidence rows:
    :func:`_refit_beta` on ``X``, :func:`_refit_decay` on ``[X | 1]``.  Each used
    to form its own ``A^T A`` from the rows on every call, so a layer paid
    ``2 * num_value_heads`` Gram formations to build two matrices: ``2 n_v N h^2``
    multiply-adds where ``2 N h^2`` suffice.  The saving is the ratio of those
    two counts, which is exactly ``n_v``: at the 27B's 48 value heads and
    4096-wide residual a layer formed 96 Grams where 2 suffice, i.e. 48x the
    multiply-adds on this term.  They are formed ONCE here; what is
    left inside the head loop is the two ``O(N h)`` right-hand sides, which
    genuinely differ per head.

    Two Grams, not one: ``X^T X`` is the top-left block of ``[X|1]^T [X|1]`` in
    exact arithmetic but NOT bit-for-bit, because the wider GEMM blocks
    differently -- measured max abs difference 4.547473508864641e-13 at
    ``N=768, h=128`` on the lab's BLAS (torch 2.11.0+cu128, 44 threads), 0.0 at
    ``(512,16)``, ``(1536,256)`` and ``(4096,2560)``.  Serving the beta solve
    from a slice of the design Gram would therefore move a shipped number for a
    factor-of-two saving on a term that is already off the per-head path, so the
    layer forms each one once and every refit result stays bit-identical to the
    row form it replaced (asserted in ``test_c5_refit_gram_hoist``).
    """

    design = torch.cat([X, torch.ones(X.shape[0], 1, dtype=torch.float64)], dim=1)
    return {
        "rows": X,
        "gram": X.T @ X,
        "design": design,
        "design_gram": design.T @ design,
    }


def _refit_decay(
    grams: dict[str, torch.Tensor],
    a: torch.Tensor,
    dt: torch.Tensor,
    A_log: torch.Tensor,
    parent_heads: list[int],
    weights: torch.Tensor,
    anchor: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Closed-form re-fit of one value head's decay row and bias.

    The decay is ``g = -exp(A_log) * softplus(a·x + dt)`` (modeling 518) with
    ``A_log`` kept at the anchor's value, so the merged head must satisfy

        ``softplus(a'·x + dt') ~ sum_j w_j exp(A_log_j) softplus(a_j·x + dt_j)
                                 / exp(A_log_anchor)``

    which is a linear least squares in ``[a'; dt']`` once the target is put
    through the inverse softplus.  Prior-anchored on the anchor's own row, so
    the fit degenerates to the anchor when the evidence says nothing.

    ``grams`` is the layer's shared :func:`_refit_grams`; only the right-hand
    side ``[X|1]^T y`` depends on this head.
    """

    X, design = grams["rows"], grams["design"]
    target = torch.zeros(X.shape[0], dtype=torch.float64)
    for w, j in zip(weights, parent_heads):
        target += float(w) * float(A_log[j].exp()) * F.softplus(X @ a[j] + dt[j])
    target = target / float(A_log[anchor].exp())
    prior = torch.cat([a[anchor], dt[anchor].reshape(1)]).unsqueeze(0)
    right_hand_side = design.T @ inverse_softplus(target).unsqueeze(1)
    solved = prior_anchored_gram(grams["design_gram"], right_hand_side, prior)
    produced = F.softplus(design @ solved.squeeze(0))
    scale = target.norm().clamp_min(1e-30)
    error = float((produced - target).norm() / scale)
    anchor_only = F.softplus(X @ a[anchor] + dt[anchor])
    anchor_error = float((anchor_only - target).norm() / scale)

    # The refit is a repair and is admitted only when it beats doing nothing on
    # the evidence.  The inverse-softplus target is exact but the least squares
    # is fitted in that transformed space, so a rung that is worse in the space
    # the kernel actually uses is possible -- and installing one would put a
    # decay row in the child that the parent's own evidence votes against.
    if not (error == error) or error > anchor_error:
        return (
            a[anchor].clone(),
            dt[anchor].clone(),
            {
                "softplus_relL2": anchor_error,
                "anchor_only_relL2": anchor_error,
                "refit": "refused (did not beat the anchor)",
                "refit_relL2": error,
            },
        )
    return (
        solved.squeeze(0)[:-1].contiguous(),
        solved.squeeze(0)[-1],
        {
            "softplus_relL2": error,
            "anchor_only_relL2": anchor_error,
            "refit": "installed",
        },
    )


def _refit_beta(
    grams: dict[str, torch.Tensor],
    b: torch.Tensor,
    parent_heads: list[int],
    weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Closed-form re-fit of one merged value head's BETA row.

    ``beta = sigmoid(in_proj_b(x))`` (modeling_qwen3_5.py:466 and 516) is the
    delta rule's write strength.  The pre-activation is linear in ``x`` but the
    quantity the kernel uses is not, and

        ``sigmoid(sum_j w_j l_j) != sum_j w_j sigmoid(l_j)``

    by an amount that is worst exactly where a merge is interesting -- one
    saturated parent and one unsaturated one.  Logits ``-6`` and ``+2`` at equal
    weight are betas ``0.0025`` and ``0.881`` (mean ``0.44``); the averaged
    logit ``-2`` is ``0.119``, a 3.7x understatement of the state write at every
    token of every GDN layer.  So the target is built in BETA space,

        ``t = sum_j w_j sigmoid(X b_j^T)``,

    pushed through the inverse of the sigmoid, and solved as an ordinary
    prior-anchored ridge in the logit space -- the same shape of construction
    as :func:`_refit_decay`, and with the same guard: the fit is installed only
    if it beats the energy-weighted logit row IN BETA SPACE on the evidence.
    ``t`` is a convex combination of sigmoids so it is strictly inside ``(0, 1)``
    and the logit is finite; the clamp is a floating-point guard, not a model.

    ``grams`` is the layer's shared :func:`_refit_grams`; only the right-hand
    side ``X^T logit(t)`` depends on this head.
    """

    X = grams["rows"]
    anchor_row = sum(float(w) * b[j] for w, j in zip(weights, parent_heads))
    target = torch.zeros(X.shape[0], dtype=torch.float64)
    for w, j in zip(weights, parent_heads):
        target += float(w) * torch.sigmoid(X @ b[j])
    scale = target.norm().clamp_min(1e-30)
    anchor_error = float((torch.sigmoid(X @ anchor_row) - target).norm() / scale)

    clamped = target.clamp(1e-12, 1.0 - 1e-12)
    logit_target = torch.log(clamped) - torch.log1p(-clamped)
    solved = prior_anchored_gram(
        grams["gram"], X.T @ logit_target.unsqueeze(1), anchor_row.unsqueeze(0)
    )
    row = solved.squeeze(0)
    error = float((torch.sigmoid(X @ row) - target).norm() / scale)

    spread = {
        "parent_beta_mean": [float(torch.sigmoid(X @ b[j]).mean()) for j in parent_heads],
        "weights": [float(w) for w in weights],
    }
    if not (error == error) or error >= anchor_error:
        return anchor_row.contiguous(), {
            "beta_relL2": anchor_error,
            "weighted_logit_relL2": anchor_error,
            "refit": "refused (did not beat the energy-weighted logit row)",
            "refit_relL2": error,
            **spread,
        }
    return row.contiguous(), {
        "beta_relL2": error,
        "weighted_logit_relL2": anchor_error,
        "refit": "installed",
        **spread,
    }


# ---------------------------------------------------------------- arm B: dims
def _merge_gdn_dims(
    mixer: nn.Module, X: torch.Tensor, *, key_head_dim: int, value_head_dim: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build every child tensor for the head-dimension arm."""

    slices = head_slices(mixer)
    nk, nv = slices["num_k_heads"], slices["num_v_heads"]
    dk, dv = slices["head_k_dim"], slices["head_v_dim"]
    if key_head_dim > dk or value_head_dim > dv:
        raise SystemExit("the dim arm contracts; it does not widen")

    qkv = mixer.in_proj_qkv.weight.detach().to("cpu", torch.float64)
    taps = mixer.conv1d.weight.detach().to("cpu", torch.float64).squeeze(1)
    z = mixer.in_proj_z.weight.detach().to("cpu", torch.float64)
    out = mixer.out_proj.weight.detach().to("cpu", torch.float64)
    gain = qf_arch.effective_gain(mixer.norm).detach().to("cpu", torch.float64)
    hidden = int(qkv.shape[1])

    child_q, child_k, child_qtap, child_ktap = [], [], [], []
    key_detail = []
    for m in range(nk):
        q_rows, k_rows = qkv[slices["q_rows"](m)], qkv[slices["k_rows"](m)]
        # One orthonormal frame for q and k together: the delta rule sees only
        # q.k after an L2 normalisation, so an orthonormal projection preserves
        # both the normalisation and the inner product on the kept subspace.
        joint = _projection_moments(X, q_rows) + _projection_moments(X, k_rows)
        values, vectors = torch.linalg.eigh(0.5 * (joint + joint.T))
        P = vectors[:, -key_head_dim:].T.flip(0).contiguous()
        child_q.append(P @ q_rows)
        child_k.append(P @ k_rows)
        child_qtap.append(_combine_conv(taps[slices["q_rows"](m)], P))
        child_ktap.append(_combine_conv(taps[slices["k_rows"](m)], P))
        key_detail.append(
            {
                "key_head": m,
                "retained_energy": float(
                    values[-key_head_dim:].sum() / values.sum().clamp_min(1e-30)
                ),
            }
        )

    child_v, child_z, child_vtap = [], [], []
    child_out = torch.zeros(hidden, nv * value_head_dim, dtype=torch.float64)
    child_gain = torch.zeros(value_head_dim, dtype=torch.float64)
    value_detail = []
    for j in range(nv):
        v_rows = qkv[slices["v_rows"](j)]
        stack = out[:, slices["o_cols"](j)]
        P, lra = whitened_lra(stack, _projection_moments(X, v_rows), value_head_dim)
        child_v.append(P @ v_rows)
        child_z.append(P @ z[slices["z_rows"](j)])
        child_vtap.append(_combine_conv(taps[slices["v_rows"](j)], P))
        child_out[:, j * value_head_dim : (j + 1) * value_head_dim] = project_writer(stack, P)
        child_gain += qf_arch.transported_gain(gain, P.T)
        value_detail.append({"value_head": j, "value_lra": lra})
    child_gain = child_gain / nv

    tensors = {
        "in_proj_qkv": torch.cat(
            [torch.cat(child_q, dim=0), torch.cat(child_k, dim=0), torch.cat(child_v, dim=0)],
            dim=0,
        ),
        "conv1d": torch.cat(
            [
                torch.cat(child_qtap, dim=0),
                torch.cat(child_ktap, dim=0),
                torch.cat(child_vtap, dim=0),
            ],
            dim=0,
        ),
        "in_proj_z": torch.cat(child_z, dim=0),
        "in_proj_b": mixer.in_proj_b.weight.detach().to("cpu", torch.float64),
        "in_proj_a": mixer.in_proj_a.weight.detach().to("cpu", torch.float64),
        "dt_bias": mixer.dt_bias.detach().to("cpu", torch.float64),
        "A_log": mixer.A_log.detach().to("cpu", torch.float64),
        "out_proj": child_out,
        "norm_gain": child_gain,
    }
    report = {
        "arm": "head-dim",
        "key_heads": key_detail,
        "value_heads": value_detail,
        "query_scale_correction": (
            "none required: the kernel's 1/sqrt(head_k_dim) is applied after the "
            "in-kernel L2 norm (modeling_qwen3_5.py:275,340) and the readout is then "
            "RMS-normalised by the gated norm (199-200), which removes any global "
            "scale on o"
        ),
        "conv_frame_mismatch": "depthwise taps combined by |P|-weighted mean",
        "z_frame_mismatch": "silu(z) is elementwise in the parent value frame",
        "norm_gain": "transported as diag(P diag(g) P^T), averaged over value heads",
    }
    return tensors, report


# -------------------------------------------------------------------- install
def install_gdn(
    mixer: nn.Module,
    tensors: dict[str, torch.Tensor],
    *,
    num_v_heads: int,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    dtype: torch.dtype,
) -> None:
    """Replace every tensor and every shape attribute of a GDN, together.

    ``key_dim``, ``value_dim`` and ``conv_dim`` are cached on the module at
    construction (modeling 379-389) and the forward pass splits ``mixed_qkv`` by
    them (502-510).  Writing new weights without writing these three would give a
    module whose split boundaries belong to the parent.
    """

    device = mixer.in_proj_qkv.weight.device
    hidden = int(mixer.in_proj_qkv.weight.shape[1])
    key_dim, value_dim = num_k_heads * head_k_dim, num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim

    mixer.in_proj_qkv = _linear(tensors["in_proj_qkv"], hidden, device, dtype)
    mixer.in_proj_z = _linear(tensors["in_proj_z"], hidden, device, dtype)
    mixer.in_proj_b = _linear(tensors["in_proj_b"], hidden, device, dtype)
    mixer.in_proj_a = _linear(tensors["in_proj_a"], hidden, device, dtype)
    mixer.out_proj = nn.Linear(value_dim, hidden, bias=False, dtype=dtype, device=device)
    mixer.out_proj.weight.data.copy_(tensors["out_proj"].to(device=device, dtype=dtype))

    kernel = int(mixer.conv1d.kernel_size[0])
    conv = nn.Conv1d(
        in_channels=conv_dim, out_channels=conv_dim, bias=False, kernel_size=kernel,
        groups=conv_dim, padding=kernel - 1, dtype=dtype, device=device,
    )
    conv.weight.data.copy_(tensors["conv1d"].unsqueeze(1).to(device=device, dtype=dtype))
    mixer.conv1d = conv

    # dt_bias and A_log keep the PARENT's dtype: a checkpoint with one tensor in
    # a different precision than its neighbours reloads at a different dtype
    # than it was written at, and the identity gate would never close.
    mixer.dt_bias = nn.Parameter(
        tensors["dt_bias"].to(device=device, dtype=mixer.dt_bias.dtype)
    )
    mixer.A_log = nn.Parameter(tensors["A_log"].to(device=device, dtype=mixer.A_log.dtype))

    if head_v_dim != int(mixer.head_v_dim):
        mixer.norm = _new_gated_norm(
            mixer.norm, head_v_dim, float(mixer.layer_norm_epsilon), mixer.activation, device, dtype
        )
    if "norm_gain" in tensors:
        qf_arch.store_gain(mixer.norm, tensors["norm_gain"])

    mixer.num_v_heads, mixer.num_k_heads = num_v_heads, num_k_heads
    mixer.head_k_dim, mixer.head_v_dim = head_k_dim, head_v_dim
    mixer.key_dim, mixer.value_dim, mixer.conv_dim = key_dim, value_dim, conv_dim


# ----------------------------------------------------------------- the axis
def gdn_axis(
    harness,
    *,
    value_heads: int | None = None,
    key_heads: int | None = None,
    key_head_dim: int | None = None,
    value_head_dim: int | None = None,
    pairing: str = "index",
    drop_arm: bool = False,
    lambda_split: bool = True,
    dtype: torch.dtype | None = None,
    log=print,
) -> dict[str, Any]:
    """Fold every GDN layer to one geometry, then re-solve each ``out_proj``.

    The four geometry fields are GLOBAL in the config, so a config-standard child
    has no choice but to fold every linear-attention layer -- there is no
    three-sites-at-a-time schedule available on this axis, and pretending
    otherwise would produce a checkpoint that cannot be loaded.
    """

    dtype = dtype or harness.dtype
    tc = harness.text_config
    layers = harness.layers()
    linear_positions = [i for i, layer in enumerate(layers) if qf_arch.layer_kind(layer) == LINEAR]
    if not linear_positions:
        raise SystemExit("this model has no linear-attention layers to fold")

    parent = head_slices(layers[linear_positions[0]].linear_attn)
    value_heads = int(value_heads or parent["num_v_heads"])
    key_heads = int(key_heads or parent["num_k_heads"])
    key_head_dim = int(key_head_dim or parent["head_k_dim"])
    value_head_dim = int(value_head_dim or parent["head_v_dim"])
    count_arm = (value_heads, key_heads) != (parent["num_v_heads"], parent["num_k_heads"])
    dim_arm = (key_head_dim, value_head_dim) != (parent["head_k_dim"], parent["head_v_dim"])
    if count_arm and dim_arm:
        raise SystemExit(
            "one axis per child: run the head-count arm and the head-dim arm as "
            "separate transactions so the review can attribute what it sees"
        )
    if not count_arm and not dim_arm:
        raise SystemExit("nothing to fold: the requested geometry is the parent's")

    per_layer: list[dict[str, Any]] = []
    for position in linear_positions:
        layer = layers[position]
        mixer = layer.linear_attn
        name = f"L{position:03d}"
        pristine = pristine_copy(mixer)

        captured = harness.capture({name: mixer}, pristine={name: pristine})[name]
        X = captured["fit"]
        Y = captured["local_fit"]
        Y_hld = captured.get("local_hld")

        if count_arm:
            order = list(range(parent["num_k_heads"]))
            if pairing == "census":
                order = _census_order(mixer, X, key_heads)
            tensors, report = _merge_gdn_counts(
                mixer, X, value_heads=value_heads, key_heads=key_heads,
                order=order, drop_arm=drop_arm,
            )
            geometry = dict(
                num_v_heads=value_heads, num_k_heads=key_heads,
                head_k_dim=parent["head_k_dim"], head_v_dim=parent["head_v_dim"],
            )
        else:
            tensors, report = _merge_gdn_dims(
                mixer, X, key_head_dim=key_head_dim, value_head_dim=value_head_dim
            )
            geometry = dict(
                num_v_heads=parent["num_v_heads"], num_k_heads=parent["num_k_heads"],
                head_k_dim=key_head_dim, head_v_dim=value_head_dim,
            )

        install_gdn(mixer, tensors, dtype=dtype, **geometry)

        # The writer is solved on the CHILD's own gated stream, which only
        # exists after the install -- hence the second capture.
        stream = harness.capture({name: mixer.out_proj})[name]
        prior = mixer.out_proj.weight.detach().to("cpu", torch.float64)
        check = fit_check_rows(harness.manifest) if lambda_split else None
        writer, guard = gated_out_proj_solve(stream["fit"], Y, prior, check)
        mixer.out_proj.weight.data.copy_(writer.to(device=harness.device, dtype=dtype))

        scores = {"fit": write_scores(stream["fit"] @ writer.T, Y)}
        if "hld" in stream and Y_hld is not None:
            scores["held"] = write_scores(stream["hld"] @ writer.T, Y_hld)
        report.update({"layer": position, "solve": guard, "scores": scores})
        per_layer.append(report)
        log(
            f"[qf] C5 {name}: {parent['num_v_heads']}/{parent['num_k_heads']}x"
            f"{parent['head_v_dim']} -> {geometry['num_v_heads']}/{geometry['num_k_heads']}x"
            f"{geometry['head_v_dim']}  fit relL2 {scores['fit']['relL2']:.4f}"
        )

    qf_arch.config_field(tc, "linear_num_value_heads", geometry["num_v_heads"])
    qf_arch.config_field(tc, "linear_num_key_heads", geometry["num_k_heads"])
    qf_arch.config_field(tc, "linear_key_head_dim", geometry["head_k_dim"])
    qf_arch.config_field(tc, "linear_value_head_dim", geometry["head_v_dim"])

    removed = _mixer_parameters(parent, int(tc.hidden_size), int(tc.linear_conv_kernel_dim)) - \
        _mixer_parameters(geometry, int(tc.hidden_size), int(tc.linear_conv_kernel_dim))
    axis = "linear_num_value_heads" if count_arm else "linear_value_head_dim"
    return {
        "axis": axis,
        "kind": "GDN value-head fold",
        "arm": "count" if count_arm else "dim",
        "parent_geometry": {k: parent[k] for k in
                            ("num_v_heads", "num_k_heads", "head_k_dim", "head_v_dim")},
        "child_geometry": geometry,
        "sites": len(per_layer),
        "component_ratio": _mixer_parameters(geometry, int(tc.hidden_size), int(tc.linear_conv_kernel_dim))
        / float(_mixer_parameters(parent, int(tc.hidden_size), int(tc.linear_conv_kernel_dim))),
        "parameters_removed": removed * len(per_layer),
        "per_layer": per_layer,
    }


def _mixer_parameters(geometry: dict[str, Any], hidden: int, kernel: int) -> int:
    nk, nv = int(geometry["num_k_heads"]), int(geometry["num_v_heads"])
    dk, dv = int(geometry["head_k_dim"]), int(geometry["head_v_dim"])
    Kd, Vd = nk * dk, nv * dv
    conv_dim = 2 * Kd + Vd
    return conv_dim * hidden + Vd * hidden + 2 * nv * hidden + conv_dim * kernel + 2 * nv + dv + hidden * Vd


def _census_order(mixer: nn.Module, X: torch.Tensor, key_heads: int) -> list[int]:
    """Greedy pairing of key heads by evidence-weighted similarity.

    An ordering.  The index-order arm is the control and both are built.
    """

    slices = head_slices(mixer)
    qkv = mixer.in_proj_qkv.weight.detach().to("cpu", torch.float64)
    rows = [
        torch.cat([qkv[slices["q_rows"](m)], qkv[slices["k_rows"](m)]], dim=0) @ X.T
        for m in range(slices["num_k_heads"])
    ]
    rows = [F.normalize(r.reshape(-1), dim=0) for r in rows]
    remaining = list(range(len(rows)))
    order: list[int] = []
    reduction = len(rows) // key_heads
    while remaining:
        head = remaining.pop(0)
        group = [head]
        for _ in range(reduction - 1):
            if not remaining:
                break
            best = max(remaining, key=lambda other: float(rows[head] @ rows[other]))
            remaining.remove(best)
            group.append(best)
        order.extend(group)
    return order
