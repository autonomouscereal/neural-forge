#!/usr/bin/env python3
"""PLANE -- the RoPE frequency-band fold, `docs/ROUND3_SLATE_2026-08-28.md` §3.4.

Waves 41 and 42 both deferred this module for the same stated reason -- authoring
time for a NEW SOLVER, not a driver change -- and `NOTES-MX-AT` §11.7 said the
choice was to budget a wave whose deliverable is the solver or take the class off
the slate.  This is the solver.

WHAT IS NEW AND WHAT IS NOT, stated first because the slate itself insisted on
the distinction and wave 15 already owns half of this machinery.

* NOT new: the ``(d, m, theta')`` mechanics.  `mx15_build.py:408`
  (`rope_plane_sets`) and `:431` (`plane_energy`) are wave 15's, they are carried
  forward in every builder since, and the census function in THIS file is wave
  15's measure unchanged -- deliberately, so that this wave's retention numbers
  are comparable to the receipt `ARCH.HD16` was built against.
* NOT new: the ``ParentRope`` shim.  ``head_dim`` and the rope parameters are one
  config field each, so the child's ladder is installed GLOBALLY before any block
  is folded and every not-yet-folded block would otherwise be handed cos/sin of
  the wrong width.  Each unfolded block keeps the parent's rotary until its own
  fold removes the shim.  No shim survives into the exported child, which is why
  the child loads with no custom module at all.
* NEW: the BAND SLIDE.  Wave 15 could only take planes ``{0, m, 2m, ...}``.  The
  ``rope_type: "linear"`` factor ``theta^(s/32)`` divides every child frequency
  by a constant, which is exactly a shift of the parent's plane index, so the
  child can take ``{s, s+m, s+2m, ...}`` and stay RoPE-EXACT on every kept plane.
  That is the knob the slate calls genuinely new, and it is why this class is a
  capability lever (§4 lever 3) rather than only a deeper rung.
* NEW: the RUNGS.  d=48 and d=32 were never visited; wave 15 went straight to
  d=16 and sealed it dead at 0/15.
* NEW: THE SOLVE, and this is why it is a separate module and not a flag on
  `mx43_build.py`.  Wave 15's `head_dim_axis` contracts ``v`` by a per-KV-group
  evidence-weighted low-rank projection and re-solves ``out_proj`` against a
  REPLAYED context.  This one keeps ``v`` VERBATIM on a scored channel subset and
  never replays attention.  Two different meanings behind one ``--head-dim`` flag
  in a 7,000-line dormant file is how an axis stops being legible.

THE FOLD, exactly.

1. **q and k rows are VERBATIM.**  Child coordinate ``j`` is parent coordinate
   ``s + m*j``; child coordinate ``j + d/2`` is parent coordinate
   ``s + m*j + 32``.  `rotate_half` splits at ``head_dim//2``, so those two are
   the two coordinates of one RoPE plane, and taking both is the only selection
   that is a fold of the parent's operator rather than a new operator at
   frequencies the parent never used.  Asserted with ``torch.equal``.
2. **The value channels are a VERBATIM SUBSET too**, scored
   ``s_c = E[|ctx_c|] * ||W_o[:, c]||_2`` on the evidence and chosen per KV
   GROUP, because ``v_proj`` has one row block per KV head and the two query
   heads sharing a group must agree about which channels survive.  Ties break on
   the channel index.  Asserted with ``torch.equal``.
3. **The q/k layernorms are the only inexactness on the score path**, and it is
   named: the parent's RMS denominator runs over 64 coordinates and the child's
   over ``d``, which is a POSITION-DEPENDENT scalar that no fixed vector can
   cancel.  The PRIOR is wave 15's answer -- the sliced gain times the mean RMS
   ratio, times ``sqrt(d/64)`` on the q side to absorb the softmax rescale -- so
   the prior-only column of every receipt this writes IS wave 15's install and
   the solved column is this wave's improvement on it, measured on HELD rows.
   The solve is a plain normal-equation prior-anchored ridge in ``d`` unknowns
   per layer against the CACHED PRE-SOFTMAX LOGITS, run ONCE on the q side and
   then ONCE on the k side.  The logit is bilinear in the two gain vectors, so a
   single joint solve would not be a plain normal equation; the stop is one pass
   per side and NOT an alternation loop, and the pre-registration says so.
   The target is the parent's own logit RESTRICTED TO THE KEPT PLANES -- the
   quantity a band fold can in principle reproduce exactly -- and never the
   parent's full logit, which no band fold can reach and which would therefore
   spend the whole ridge budget on the part of the operator this axis deletes on
   purpose.
4. **``out_proj`` is a prior-anchored ridge on the exact column subset of the
   SEALED context, with NO ATTENTION REPLAY**, per the slate.  The prior is the
   parent's own ``W_o`` columns at the kept channels.
5. **THE LAW is a veto**: the lambda ladder climbs until ``update_scale <= 1.0``
   and the install is refused if no rung admits.

WHAT (4) COSTS, SAID OUT LOUD.  Because attention is not replayed, ``out_proj``
is fitted against the context the PARENT's probabilities produce, and the child's
probabilities are not the parent's -- the dropped planes moved them.  That is a
real approximation and this module publishes the instrument that bounds it
rather than hiding it: ``held_logit_relL2_prior`` and ``held_logit_relL2_solved``
are the relative error of the INSTALLED pre-softmax logits against the kept-plane
target on held rows, per site.  A site whose logits move by a per cent is a site
whose probabilities moved by about a per cent; a site whose logits move by a
third is a site where the sealed-context ridge is fitting the wrong context, and
the receipt says which one every site is.  ``held_kept_plane_logit_share`` is the
third number and it is the honest ceiling for ANY fold at this rung: the share of
the parent's full logit energy the kept band carries at all.

THE CENSUS RUNS WITHOUT THE BUILD AND GATES NOTHING.  ``--mode census`` pools
wave 15's plane-energy measure over all six attention sites and all sixteen heads
on the fit rows and prints the retained share for EVERY (d, m, s) band the ladder
can express.  The slate requires those numbers PRE-REGISTERED before the build so
that a death attributes to the mechanism and not to the rung; a census computed
inside a build that has already chosen its rung cannot do that, which is why this
is a mode and not a by-product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

import mx43_build

PARENT = mx43_build.PARENT
PARENT_PARAMS = mx43_build.PARENT_PARAMS
ATTENTION_LAYERS = mx43_build.ATTENTION_LAYERS
SIDECARS = mx43_build.SIDECARS
LAMBDA_LADDER = mx43_build.LAMBDA_LADDER
#: ITS OWN SCHEMA, for wave 40's stated reason and wave 41's: a receipt that
#: carries fields no other receipt has gets its own string in the sealer's
#: append-only accept list, so a reader can tell from the string what the
#: artifact is.  This one carries the band, the ladder, the per-plane retention,
#: the per-site kept-plane logit share and the value-channel subsets, and no
#: gather receipt or blend receipt has any of those.
SCHEMA = "neural-forge-matrix-wave43-rope-band-child-v1"
CENSUS_SCHEMA = "neural-forge-matrix-wave43-plane-energy-census-v1"
PARENT_HEAD_DIM = 64
#: The child's ``rope_theta`` is ``theta^(m*d/64)`` and its linear factor is
#: ``theta^(s/32)``.  Both are re-derived from the parent's own theta and checked
#: against the value that actually landed in the written config, because if the
#: ladder is wrong then plane ``j`` of the child is not plane ``s + m*j`` of the
#: parent and the verbatim-rows claim is a claim about the rows of a DIFFERENT
#: operator.  The comparison is relative and the tolerance is a float round trip.
THETA_TOLERANCE = 1e-9
#: The widths the census reports.  64 is the parent and is included so the census
#: carries an identity row that must read exactly 1.0.
CENSUS_HEAD_DIMS = (64, 48, 32, 16)


def log(message: str) -> None:
    print(f"[mx43plane] {message}", flush=True)


# ---------------------------------------------------------------- the ladder
def plane_sets(head_dim: int) -> list[dict[str, Any]]:
    """Every ``(stride, slide)`` band a child of width ``head_dim`` can express.

    RoPE's rotation is block diagonal over frequency planes, so ``q^T R k`` is
    exactly a sum of per-plane terms and dropping planes drops their terms and
    nothing else.  The child's own ladder is forced by its config:
    ``freq_j = (theta')^(-2j/d) / F``.  Matching the parent's ``theta^(-2i/64)``
    at ``i = s + m*j`` therefore requires ``theta' = theta^(m*d/64)`` AND
    ``F = theta^(s/32)``; the second factor is what `rope_type: "linear"` is, and
    it is the only reason a SLID band is expressible at all.
    """

    planes = PARENT_HEAD_DIM // 2
    want = head_dim // 2
    out: list[dict[str, Any]] = []
    for stride in (1, 2, 4, 8):
        for slide in range(planes):
            picked = [slide + stride * j for j in range(want)]
            if picked[-1] >= planes:
                continue
            out.append({"stride": stride, "slide": slide, "planes": picked})
    return out


def plane_energy(q_rows: torch.Tensor, k_rows: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Per-plane share of the score operator's energy, on the evidence.

    WAVE 15'S MEASURE, UNCHANGED (`mx15_build.py:431`), and it is unchanged on
    purpose: `ARCH.HD16`'s receipt read about 0.24 retained score energy on this
    exact function, and a wave that re-derived the measure could not compare
    against that number.  Plane ``p`` contributes ``q_p . R k_p`` to every score
    and nothing else does; the product of the two planes' second moments is the
    scale of that contribution.  It is a CENSUS -- it ranks the bands and gates
    nothing.
    """

    planes = head_dim // 2
    energy = torch.zeros(planes, dtype=torch.float64)
    for p in range(planes):
        qp = q_rows[:, [p, p + planes]]
        kp = k_rows[:, [p, p + planes]]
        energy[p] = float(qp.pow(2).sum(dim=1).mean() * kp.pow(2).sum(dim=1).mean())
    return energy


class ParentRope:
    """Serve the parent's 64-wide RoPE to attention blocks that are not folded yet.

    WAVE 15'S SCAFFOLDING, carried in shape and in reason.  It is build-time only:
    no shim survives the export.
    """

    def __init__(self, rotary: Any) -> None:
        self.rotary = rotary
        self.handles: dict[int, Any] = {}

    def attach(self, module: Any) -> None:
        def hook(mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(
                    hidden.shape[1], device=hidden.device
                ).unsqueeze(0)
            kwargs["position_embeddings"] = self.rotary(hidden, position_ids)
            return args, kwargs

        self.handles[id(module)] = module.register_forward_pre_hook(hook, with_kwargs=True)

    def detach(self, module: Any) -> None:
        handle = self.handles.pop(id(module), None)
        if handle is not None:
            handle.remove()


# ------------------------------------------------------------------- capture
@torch.no_grad()
def capture_by_batch(harness: Any, targets: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Run the corpus once; keep each module's input GROUPED BY SEQUENCE.

    The flat capture every other solve in this campaign uses has forgotten where
    each sequence started, and a pre-softmax logit is a quantity over PAIRS of
    positions inside ONE sequence.  Same hooks, same corpus, same order, same
    fit/held rule -- the concatenation of these blocks in order IS the flat
    capture, and nothing below ever pairs two positions from different batches.
    """

    state: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            state[name] = hidden.detach().reshape(-1, hidden.shape[-1]).to(
                "cpu", torch.float64
            )
            return None

        return hook

    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    try:
        for index, ids in harness.batches:
            split = "fit" if harness.split[index] == "fit" else "hld"
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
            harness.forward_calls += 1
            for name in targets:
                out[name].append({"batch": int(index), "split": split, "rows": state[name]})
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return out


def flat(blocks: list[dict[str, Any]], split: str) -> torch.Tensor:
    return torch.cat([b["rows"] for b in blocks if b["split"] == split], dim=0)


# ------------------------------------------------------------------ geometry
def rope_angles(planes: list[int], length: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin of ``omega_p * i`` for every kept plane, at the PARENT's frequency.

    The child's plane ``j`` and the parent's plane ``planes[j]`` rotate at the
    same angular frequency by construction -- that is the entire point of the
    ladder -- so one table serves both sides of the comparison, and a discrepancy
    would show up as a target the child cannot hit at any gain.
    """

    omega = torch.tensor(
        [theta ** (-2.0 * p / PARENT_HEAD_DIM) for p in planes], dtype=torch.float64
    )
    position = torch.arange(length, dtype=torch.float64)
    angle = position[:, None] * omega[None, :]
    return torch.cos(angle), torch.sin(angle)


def rope_apply(vec: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """`apply_rotary_pos_emb` for one head, in fp64, on the kept planes only."""

    half = vec.shape[1] // 2
    first, second = vec[:, :half], vec[:, half:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin], dim=1)


def logit_terms(
    u_raw: torch.Tensor, other_rot: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor, scale: float,
) -> torch.Tensor:
    """The per-coordinate coefficients of one side's gain vector in the logit.

    `Lfm2Attention` applies q_layernorm and THEN apply_rotary_pos_emb, so the
    gains multiply the coordinates BEFORE the rotation and the score is LINEAR in
    the gain vector.  This returns the coefficient matrix of that linear map:
    ``z[a, b] = sum_c gain[c] * out[a, b, c]``, where ``a`` indexes the side being
    solved and ``b`` the side held fixed.  ``RoPE(e_p, a)`` has ``cos`` at ``p``
    and ``sin`` at ``p + half``; ``RoPE(e_{p+half}, a)`` has ``-sin`` at ``p`` and
    ``cos`` at ``p + half``.
    """

    half = u_raw.shape[1] // 2
    left, right = other_rot[:, :half], other_rot[:, half:]
    first = cos[:, None, :] * left[None, :, :] + sin[:, None, :] * right[None, :, :]
    second = cos[:, None, :] * right[None, :, :] - sin[:, None, :] * left[None, :, :]
    return scale * torch.cat(
        [first * u_raw[:, None, :half], second * u_raw[:, None, half:]], dim=2
    )


def normalised(
    rows: torch.Tensor, gain: torch.Tensor, dims: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """One head's PARENT-normalised (64 coords) and CHILD-normalised (d) band.

    The two differ by exactly one position-dependent scalar, ``rms_d / rms_64``,
    and that scalar is the whole of what the layernorm ridge is being asked to
    absorb.  Returning both from one place makes that visible instead of leaving
    it implied by two divisions written three hundred lines apart.
    """

    kept = rows[:, dims]
    r_full = rows.pow(2).mean(dim=1).add(eps).sqrt()[:, None]
    r_kept = kept.pow(2).mean(dim=1).add(eps).sqrt()[:, None]
    return kept / r_full * gain[dims][None, :], kept / r_kept


def head_bands(
    X: torch.Tensor, w: torch.Tensor, gain: torch.Tensor, dims: torch.Tensor,
    count: int, eps: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every head's band for one projection, from ONE matmul over the block."""

    rows = X @ w.T
    return [
        normalised(rows[:, i * PARENT_HEAD_DIM : (i + 1) * PARENT_HEAD_DIM], gain, dims, eps)
        for i in range(count)
    ]


# --------------------------------------------------------------------- solve
def law_abiding_diagonal(
    A: torch.Tensor, b: torch.Tensor, prior: torch.Tensor, energy: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    """One gain vector, prior-anchored, THE LAW as a veto on the update scale.

    ``A`` and ``b`` are the pooled normal equations of the logit residual over
    every head and every causal pair of every FIT batch.  The ridge pulls toward
    the PRIOR rather than toward zero, so the lambda that admits is the one that
    keeps the install near wave 15's answer instead of near nothing, and the
    ladder is `mx43_build.LAMBDA_LADDER` so the rungs are the campaign's.
    """

    reference = float(prior.norm())
    scale_of: dict[float, float] = {}
    chosen: torch.Tensor | None = None
    chosen_lambda: float | None = None
    trace = float(torch.diagonal(A).mean())
    for rel in LAMBDA_LADDER:
        lam = rel * trace
        solved = torch.linalg.solve(
            A + lam * torch.eye(A.shape[0], dtype=A.dtype), b + lam * prior
        )
        scale = (solved - prior).norm() / reference
        scale_of[rel] = float(scale)
        if chosen is None:
            if float(scale) <= 1.0:
                chosen, chosen_lambda = solved, rel
    if chosen is None:
        raise SystemExit(
            "no lambda on the ladder brings the gain update_scale to 1.0 "
            f"(free {scale_of[LAMBDA_LADDER[0]]:.3f}); THE LAW refuses the install"
        )

    def residual(vector: torch.Tensor) -> float:
        return float((vector @ A @ vector - 2.0 * (vector @ b) + energy) / max(energy, 1e-300))

    return chosen, {
        "update_scale": scale_of[chosen_lambda],
        "lambda_relative": chosen_lambda,
        "free_solve_update_scale": scale_of[LAMBDA_LADDER[0]],
        "fit_relative_sse_prior": residual(prior),
        "fit_relative_sse_solved": residual(chosen),
    }


def logit_systems(
    x_blocks: list[dict[str, Any]], geometry: dict[str, Any],
    a_gain: torch.Tensor, b_gain: torch.Tensor, *, side: str,
) -> dict[str, Any]:
    """Pooled normal equations for one side's gain vector, over the FIT batches.

    The target is the parent's own pre-softmax logit RESTRICTED TO THE KEPT
    PLANES.  Pooling is over heads because ``q_layernorm`` is ONE vector shared by
    every head -- a per-head diagonal is not expressible in a stock checkpoint --
    and over causal pairs only, because a non-causal pair is masked out before the
    softmax ever sees it and fitting one would be fitting noise.
    """

    dims, planes, theta = geometry["dims"], geometry["planes"], geometry["theta"]
    head_dim, eps = geometry["head_dim"], geometry["eps"]
    heads, kv_heads = geometry["heads"], geometry["kv_heads"]
    group_size = heads // kv_heads
    A = torch.zeros(head_dim, head_dim, dtype=torch.float64)
    b = torch.zeros(head_dim, dtype=torch.float64)
    energy = 0.0
    parent_scale = PARENT_HEAD_DIM**-0.5
    child_scale = head_dim**-0.5
    for block in x_blocks:
        if block["split"] != "fit":
            continue
        X = block["rows"]
        length = X.shape[0]
        cos, sin = rope_angles(planes, length, theta)
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
        q_bands = head_bands(X, geometry["wq"], geometry["gq"], dims, heads, eps)
        k_bands = head_bands(X, geometry["wk"], geometry["gk"], dims, kv_heads, eps)
        for h in range(heads):
            qp, qc = q_bands[h]
            kp, kc = k_bands[h // group_size]
            target = parent_scale * (
                rope_apply(qp, cos, sin) @ rope_apply(kp, cos, sin).T
            )
            if side == "q":
                other = rope_apply(kc * b_gain[None, :], cos, sin)
                terms = logit_terms(qc, other, cos, sin, child_scale)
            else:
                other = rope_apply(qc * a_gain[None, :], cos, sin)
                terms = logit_terms(kc, other, cos, sin, child_scale).transpose(0, 1)
            flat_terms = terms[mask]
            flat_target = target[mask]
            A += flat_terms.T @ flat_terms
            b += flat_terms.T @ flat_target
            energy += float(flat_target.pow(2).sum())
    return {"A": A, "b": b, "energy": energy}


def logit_scores(
    x_blocks: list[dict[str, Any]], geometry: dict[str, Any],
    prior: tuple[torch.Tensor, torch.Tensor], solved: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """The instrument that bounds what "no attention replay" costs, on HELD rows."""

    dims, planes, theta = geometry["dims"], geometry["planes"], geometry["theta"]
    head_dim, eps = geometry["head_dim"], geometry["eps"]
    heads, kv_heads = geometry["heads"], geometry["kv_heads"]
    group_size = heads // kv_heads
    parent_scale = PARENT_HEAD_DIM**-0.5
    child_scale = head_dim**-0.5
    every = torch.arange(PARENT_HEAD_DIM, dtype=torch.long)
    acc = {"prior": 0.0, "solved": 0.0, "target": 0.0, "full": 0.0}
    for block in x_blocks:
        if block["split"] != "hld":
            continue
        X = block["rows"]
        length = X.shape[0]
        cos, sin = rope_angles(planes, length, theta)
        cos_all, sin_all = rope_angles(list(range(PARENT_HEAD_DIM // 2)), length, theta)
        mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
        q_bands = head_bands(X, geometry["wq"], geometry["gq"], dims, heads, eps)
        k_bands = head_bands(X, geometry["wk"], geometry["gk"], dims, kv_heads, eps)
        q_full = head_bands(X, geometry["wq"], geometry["gq"], every, heads, eps)
        k_full = head_bands(X, geometry["wk"], geometry["gk"], every, kv_heads, eps)
        for h in range(heads):
            g = h // group_size
            qp, qc = q_bands[h]
            kp, kc = k_bands[g]
            target = parent_scale * (
                rope_apply(qp, cos, sin) @ rope_apply(kp, cos, sin).T
            )[mask]
            full = parent_scale * (
                rope_apply(q_full[h][0], cos_all, sin_all)
                @ rope_apply(k_full[g][0], cos_all, sin_all).T
            )[mask]
            for name, (a, bb) in (("prior", prior), ("solved", solved)):
                got = child_scale * (
                    rope_apply(qc * a[None, :], cos, sin)
                    @ rope_apply(kc * bb[None, :], cos, sin).T
                )[mask]
                acc[name] += float((got - target).pow(2).sum())
            acc["target"] += float(target.pow(2).sum())
            acc["full"] += float(full.pow(2).sum())
    return {
        "prior": math.sqrt(acc["prior"] / acc["target"]),
        "solved": math.sqrt(acc["solved"] / acc["target"]),
        "kept_share": acc["target"] / acc["full"],
    }


# ------------------------------------------------------------------ the fold
def install_band(
    harness: Any, position: int, head_dim: int, *,
    planes: list[int], theta: float,
    x_blocks: list[dict[str, Any]], ctx_blocks: list[dict[str, Any]],
    shim: ParentRope | None,
) -> dict[str, Any]:
    """Fold one attention block onto a band of whole RoPE planes."""

    attn = harness.layers()[position].self_attn
    config = harness.model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    group_size = heads // kv_heads
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    eps = float(config.norm_eps)

    dims = torch.tensor(
        planes + [p + PARENT_HEAD_DIM // 2 for p in planes], dtype=torch.long
    )
    q_rows = torch.cat([dims + h * PARENT_HEAD_DIM for h in range(heads)]).to(device)
    k_rows = torch.cat([dims + g * PARENT_HEAD_DIM for g in range(kv_heads)]).to(device)

    wq = attn.q_proj.weight.data.to(torch.float64).cpu()
    wk = attn.k_proj.weight.data.to(torch.float64).cpu()
    wo = attn.out_proj.weight.data.to(torch.float64).cpu()
    gq = attn.q_layernorm.weight.data.to(torch.float64).cpu()
    gk = attn.k_layernorm.weight.data.to(torch.float64).cpu()

    X_fit = flat(x_blocks, "fit")
    ctx_fit = flat(ctx_blocks, "fit")
    ctx_hld = flat(ctx_blocks, "hld")

    # ---- the value channels: a VERBATIM subset, scored per KV group.
    column_score = ctx_fit.abs().mean(dim=0) * wo.norm(dim=0)
    vsel: list[torch.Tensor] = []
    for g in range(kv_heads):
        pooled = torch.zeros(PARENT_HEAD_DIM, dtype=torch.float64)
        for h in range(g * group_size, (g + 1) * group_size):
            pooled = pooled + column_score[h * PARENT_HEAD_DIM : (h + 1) * PARENT_HEAD_DIM]
        order = sorted(range(PARENT_HEAD_DIM), key=lambda c: (-float(pooled[c]), c))
        vsel.append(torch.tensor(sorted(order[:head_dim]), dtype=torch.long))
    v_rows = torch.cat([vsel[g] + g * PARENT_HEAD_DIM for g in range(kv_heads)]).to(device)
    o_cols = torch.cat([vsel[h // group_size] + h * PARENT_HEAD_DIM for h in range(heads)])

    # ---- the PRIOR is wave 15's install: the sliced gain times the mean RMS
    # ratio, with sqrt(d/64) on the q side for the softmax rescale.
    def rms_ratio(rows: torch.Tensor) -> float:
        full = rows.reshape(-1, PARENT_HEAD_DIM)
        kept = full[:, dims]
        r_full = full.pow(2).mean(dim=1).add(eps).sqrt()
        r_kept = kept.pow(2).mean(dim=1).add(eps).sqrt()
        return float((r_kept / r_full).mean())

    q_ratio = rms_ratio(X_fit @ wq.T)
    k_ratio = rms_ratio(X_fit @ wk.T)
    softmax_correction = math.sqrt(head_dim / PARENT_HEAD_DIM)
    a_prior = gq[dims] * q_ratio * softmax_correction
    b_prior = gk[dims] * k_ratio

    geometry = {
        "dims": dims, "planes": planes, "theta": theta, "head_dim": head_dim,
        "eps": eps, "heads": heads, "kv_heads": kv_heads,
        "wq": wq, "wk": wk, "gq": gq, "gk": gk,
    }
    q_system = logit_systems(x_blocks, geometry, a_prior, b_prior, side="q")
    a_solved, a_guard = law_abiding_diagonal(
        q_system["A"], q_system["b"], a_prior, q_system["energy"]
    )
    k_system = logit_systems(x_blocks, geometry, a_solved, b_prior, side="k")
    b_solved, b_guard = law_abiding_diagonal(
        k_system["A"], k_system["b"], b_prior, k_system["energy"]
    )
    held = logit_scores(x_blocks, geometry, (a_prior, b_prior), (a_solved, b_solved))

    # ---- install q, k and v.  All three are the parent's own rows.
    wq_src = attn.q_proj.weight.data
    wk_src = attn.k_proj.weight.data
    wv_src = attn.v_proj.weight.data
    new_q = nn.Linear(wq.shape[1], heads * head_dim, bias=False, dtype=dtype, device=device)
    new_k = nn.Linear(wk.shape[1], kv_heads * head_dim, bias=False, dtype=dtype, device=device)
    new_v = nn.Linear(wq.shape[1], kv_heads * head_dim, bias=False, dtype=dtype, device=device)
    new_q.weight.data.copy_(wq_src[q_rows])
    new_k.weight.data.copy_(wk_src[k_rows])
    new_v.weight.data.copy_(wv_src[v_rows])
    if not torch.equal(new_q.weight.data, wq_src[q_rows]):
        raise SystemExit(f"L{position:02d}: the installed q rows are not the parent's own")
    if not torch.equal(new_k.weight.data, wk_src[k_rows]):
        raise SystemExit(f"L{position:02d}: the installed k rows are not the parent's own")
    if not torch.equal(new_v.weight.data, wv_src[v_rows]):
        raise SystemExit(f"L{position:02d}: the installed v channels are not the parent's own")
    attn.q_proj, attn.k_proj, attn.v_proj = new_q, new_k, new_v
    attn.q_layernorm.weight = nn.Parameter(a_solved.to(device=device, dtype=dtype))
    attn.k_layernorm.weight = nn.Parameter(b_solved.to(device=device, dtype=dtype))
    attn.head_dim = head_dim
    attn.scaling = head_dim**-0.5

    # ---- out_proj: prior-anchored ridge on the exact column subset of the
    # SEALED context.  No attention replay -- the slate's instruction -- and
    # `held` above is the instrument that says what that cost at this site.
    prior_o = wo[:, o_cols]
    prior_scores = mx43_build.write_scores(ctx_hld[:, o_cols] @ prior_o.T, ctx_hld @ wo.T)
    solved_o, o_guard = mx43_build.law_abiding_solve(
        ctx_fit[:, o_cols], ctx_fit @ wo.T, prior_o
    )
    solved_scores = mx43_build.write_scores(ctx_hld[:, o_cols] @ solved_o.T, ctx_hld @ wo.T)
    new_o = nn.Linear(heads * head_dim, wo.shape[0], bias=False, dtype=dtype, device=device)
    new_o.weight.data.copy_(solved_o.to(device=device, dtype=dtype))
    attn.out_proj = new_o
    if shim is not None:
        shim.detach(attn)
    return {
        "position": position,
        "head_dim": head_dim,
        "value_channels_per_group": [[int(c) for c in s] for s in vsel],
        "q_gain": {"prior_rms_ratio": q_ratio,
                   "softmax_correction": softmax_correction, **a_guard},
        "k_gain": {"prior_rms_ratio": k_ratio, **b_guard},
        "held_logit_relL2_prior": held["prior"],
        "held_logit_relL2_solved": held["solved"],
        "held_kept_plane_logit_share": held["kept_share"],
        "out_proj_prior_only_cos": prior_scores["cos"],
        **o_guard,
        **solved_scores,
    }


# -------------------------------------------------------------------- export
def export(
    harness: Any, outdir: Path, *, head_dim: int, theta: float, factor: float | None
) -> int:
    """Write a stock checkpoint: the live parameters plus a consistent config."""

    from safetensors.torch import save_file

    config = json.loads((PARENT / "config.json").read_text())
    state = {
        name: value.detach().to("cpu", torch.bfloat16).contiguous()
        for name, value in harness.model.named_parameters()
    }
    state.pop("lm_head.weight", None)
    config["head_dim"] = head_dim
    config["rope_parameters"] = dict(config["rope_parameters"])
    config["rope_parameters"]["rope_theta"] = theta
    if factor is not None:
        # The band SLIDE, and it is the only genuinely new mechanic on this axis.
        # `rope_type: "linear"` divides every child frequency by `factor`, which
        # is exactly a constant shift of the parent's plane index -- so a slid
        # band is still RoPE-exact on every plane it keeps.
        config["rope_parameters"]["rope_type"] = "linear"
        config["rope_parameters"]["factor"] = factor
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(outdir / "model.safetensors"), metadata={"format": "pt"})
    for side in SIDECARS:
        if (PARENT / side).is_file():
            (outdir / side).unlink(missing_ok=True)
            shutil.copy2(PARENT / side, outdir / side)
    return sum(int(v.numel()) for v in state.values())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_sha256(directory: Path) -> str:
    """Every file in the child directory, name and bytes, in sorted order.

    `mx43_vocab.py:202`'s helper, reproduced rather than imported because that
    module opens a tokenizer at import time and this one has no business doing
    so.  Reproduced EXACTLY, so the two waves' digests are comparable.
    """

    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def attention_parameters(head_dim: int, config: dict[str, Any]) -> int:
    """One attention block's parameter count at a head width, from the config alone."""

    hidden = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    return (
        hidden * heads * head_dim
        + hidden * kv_heads * head_dim
        + hidden * kv_heads * head_dim
        + heads * head_dim * hidden
        + 2 * head_dim
    )


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("census", "build"), required=True)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--slide", type=int, default=0)
    ap.add_argument("--tag")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--expect-parameters", type=int)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    started = time.time()
    parent_config = json.loads((PARENT / "config.json").read_text())
    theta = float(parent_config["rope_parameters"]["rope_theta"])
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    capability = (
        "%d.%d" % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "none"
    )
    harness = mx43_build.Harness(PARENT, args.device, torch.bfloat16)
    heads = int(parent_config["num_attention_heads"])
    kv_heads = int(parent_config["num_key_value_heads"])
    group_size = heads // kv_heads
    targets = {f"A{layer:02d}": harness.layers()[layer].self_attn for layer in ATTENTION_LAYERS}
    targets["LM"] = harness.model.lm_head
    captured = capture_by_batch(harness, targets)
    parent_endpoint = flat(captured["LM"], "hld")

    # ---- the census.  It runs in BOTH modes and it gates in NEITHER.
    pooled = torch.zeros(PARENT_HEAD_DIM // 2, dtype=torch.float64)
    per_site: dict[str, list[float]] = {}
    for layer in ATTENTION_LAYERS:
        attn = harness.layers()[layer].self_attn
        X = flat(captured[f"A{layer:02d}"], "fit")
        q_all = X @ attn.q_proj.weight.data.to(torch.float64).cpu().T
        k_all = X @ attn.k_proj.weight.data.to(torch.float64).cpu().T
        site = torch.zeros(PARENT_HEAD_DIM // 2, dtype=torch.float64)
        for h in range(heads):
            g = h // group_size
            site = site + plane_energy(
                q_all[:, h * PARENT_HEAD_DIM : (h + 1) * PARENT_HEAD_DIM],
                k_all[:, g * PARENT_HEAD_DIM : (g + 1) * PARENT_HEAD_DIM],
                PARENT_HEAD_DIM,
            )
        per_site[f"A{layer:02d}"] = [float(x / site.sum()) for x in site]
        pooled = pooled + site
    total = float(pooled.sum())
    bands: list[dict[str, Any]] = []
    for width in CENSUS_HEAD_DIMS:
        for band in plane_sets(width):
            picked = band["planes"]
            bands.append({
                "head_dim": width,
                "stride": band["stride"],
                "slide": band["slide"],
                "retained_plane_energy": float(pooled[picked].sum()) / total,
                "rope_theta": theta ** (band["stride"] * width / PARENT_HEAD_DIM),
                "linear_factor": (theta ** (band["slide"] / 32.0)) if band["slide"] else None,
                "storage_per_attention_block": attention_parameters(width, parent_config),
                "solo_parameters": PARENT_PARAMS - len(ATTENTION_LAYERS) * (
                    attention_parameters(PARENT_HEAD_DIM, parent_config)
                    - attention_parameters(width, parent_config)
                ),
            })
    census = {
        "schema": CENSUS_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gates": "NOTHING.  This census publishes and never decides a build.",
        "measure": (
            "mx15_build.py:431 plane_energy, unchanged, so these numbers are "
            "comparable to the receipt ARCH.HD16 was built against"
        ),
        "evidence": mx43_build.evidence_receipt(),
        "rows": "fit split only",
        "parent_rope_theta": theta,
        "pooled_plane_energy_share": [float(x / total) for x in pooled],
        "per_site_plane_energy_share": per_site,
        "bands": bands,
        "device": device_name,
        "compute_capability": capability,
        "forward_calls": harness.forward_calls,
    }
    census_only = args.mode == "census"
    if census_only:
        if args.out is None:
            raise SystemExit("--mode census needs --out")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(census, indent=1))
        reread = json.loads(args.out.read_text())
        assert reread["schema"] == CENSUS_SCHEMA, "the census did not survive the write"
        assert len(reread["bands"]) == len(bands), "the band table did not survive the write"
        for row in sorted(bands, key=lambda r: (-r["head_dim"], r["stride"], r["slide"])):
            if row["head_dim"] == 64 or (row["head_dim"] in (48, 32) and row["stride"] <= 2):
                log(f"  d={row['head_dim']:2d} m={row['stride']} s={row['slide']:2d} "
                    f"retained {row['retained_plane_energy']:.6f}  "
                    f"theta' {row['rope_theta']:.4f}  "
                    f"solo {row['solo_parameters']} "
                    f"({row['solo_parameters'] / PARENT_PARAMS:.10f}x)")
        log(f"-> {args.out} ({len(bands)} bands, re-parsed OK)")
        return 0

    # ---- the build.
    if args.head_dim is None or args.tag is None or args.child is None:
        raise SystemExit("--mode build needs --head-dim, --tag and --child")
    head_dim = int(args.head_dim)
    band = next(
        (b for b in plane_sets(head_dim)
         if b["stride"] == args.stride and b["slide"] == args.slide),
        None,
    )
    if band is None:
        raise SystemExit(
            f"head_dim {head_dim} with stride {args.stride} and slide {args.slide} "
            "is not a RoPE-exact band of the parent's ladder"
        )
    planes = band["planes"]
    ladder_theta = theta ** (args.stride * head_dim / PARENT_HEAD_DIM)
    factor = (theta ** (args.slide / 32.0)) if args.slide else None
    expected_parameters = PARENT_PARAMS - len(ATTENTION_LAYERS) * (
        attention_parameters(PARENT_HEAD_DIM, parent_config)
        - attention_parameters(head_dim, parent_config)
    )
    if args.expect_parameters is not None and int(args.expect_parameters) != expected_parameters:
        raise SystemExit(
            f"the pre-registered parameter count {args.expect_parameters} is not this "
            f"build's own arithmetic {expected_parameters}; one of the two is wrong and "
            "the build does not get to choose which"
        )
    log(f"band d={head_dim} m={args.stride} s={args.slide}: planes {planes[0]}..{planes[-1]}, "
        f"retained plane energy {float(pooled[planes].sum()) / total:.6f}")
    log(f"ladder: rope_theta {ladder_theta:.6f}"
        + (f", linear factor {factor:.6f}" if factor is not None else ", no slide"))

    from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

    shim = ParentRope(harness.model.model.rotary_emb)
    for layer in ATTENTION_LAYERS:
        shim.attach(harness.layers()[layer].self_attn)
    harness.model.config.head_dim = head_dim
    harness.model.config.rope_parameters = dict(harness.model.config.rope_parameters)
    harness.model.config.rope_parameters["rope_theta"] = ladder_theta
    if factor is not None:
        harness.model.config.rope_parameters["rope_type"] = "linear"
        harness.model.config.rope_parameters["factor"] = factor
    harness.model.model.rotary_emb = Lfm2RotaryEmbedding(
        harness.model.config, device=torch.device(args.device)
    )

    records: list[dict[str, Any]] = []
    for layer in ATTENTION_LAYERS:
        attn = harness.layers()[layer].self_attn
        # SEQUENTIALLY CONDITIONED ON THE PREFIX: this capture runs through the
        # blocks already folded, so each site is solved against the stream the
        # child actually presents it with.  Within the site the target is the
        # parent's own local behaviour, which is what "no attention replay" means
        # -- it is a statement about the ATTENTION probabilities, not about the
        # residual stream.
        blocks = capture_by_batch(harness, {"x": attn, "ctx": attn.out_proj})
        record = install_band(
            harness, layer, head_dim, planes=planes, theta=theta,
            x_blocks=blocks["x"], ctx_blocks=blocks["ctx"], shim=shim,
        )
        records.append(record)
        log(
            f"  A{layer:02d} logit relL2 {record['held_logit_relL2_prior']:.4f} -> "
            f"{record['held_logit_relL2_solved']:.4f} (kept-plane logit share "
            f"{record['held_kept_plane_logit_share']:.4f}); out_proj cos "
            f"{record['out_proj_prior_only_cos']:.4f} -> {record['cos']:.4f}, "
            f"update_scale {record['update_scale']:.3f}"
        )

    child_endpoint = flat(capture_by_batch(harness, {"LM": harness.model.lm_head})["LM"], "hld")
    endpoint = mx43_build.write_scores(child_endpoint, parent_endpoint)
    log(f"END TO END on held rows: relL2 {endpoint['relL2']:.4f} cos {endpoint['cos']:.4f}")

    live = export(harness, args.child, head_dim=head_dim, theta=ladder_theta, factor=factor)
    if int(live) != int(expected_parameters):
        raise SystemExit(
            f"the exported child carries {live} parameters, not the "
            f"{expected_parameters} this band's arithmetic predicts"
        )
    written = json.loads((args.child / "config.json").read_text())
    expected_theta = theta ** (args.stride * head_dim / PARENT_HEAD_DIM)
    for source, theta_child in (
        ("this build's own ladder", ladder_theta),
        ("the config that landed on disk", float(written["rope_parameters"]["rope_theta"])),
    ):
        if abs(theta_child / expected_theta - 1.0) > THETA_TOLERANCE:
            raise SystemExit(
                f"{source} carries rope_theta {theta_child}, not {expected_theta}: the "
                "child's plane j is not the parent's plane stride*j + slide, and the "
                "verbatim-rows claim is a claim about a different operator"
            )
    if int(written["head_dim"]) != head_dim:
        raise SystemExit("the written config does not carry this band's head_dim")
    if factor is not None and written["rope_parameters"].get("rope_type") != "linear":
        raise SystemExit("a slid band was built and the written config does not slide")

    # THE RECEIPT IS A CONTRACT WITH `experiments/seal_matrix_architecture_child.py`
    # AND THE CONTRACT IS READ FROM THAT FILE, NOT REMEMBERED.  It requires
    # `child_parameters` (checked against the loader's own count),
    # `parent_parameters` (checked against the parent's), `axes` (non-empty, and
    # every entry carrying all nine of `matrix_cell_receipt._ARCHITECTURE_AXIS_FIELDS`),
    # and it checks `child_tensors_sha256` when it is present.  Wave 43's first
    # attempt wrote `parameters` instead of `child_parameters` and had no `axes`
    # at all; the sealer refused, which is the refusal working.
    solve_note = (
        "q/k rows and v channels VERBATIM; q/k layernorm diagonals by one plain "
        "normal-equation prior-anchored ridge per side in head_dim unknowns "
        "against the cached pre-softmax logits, target = the parent's own logit "
        "restricted to the kept planes; out_proj by a prior-anchored ridge on "
        "the exact column subset of the sealed context, with NO ATTENTION REPLAY"
    )
    receipt = {
        "schema": SCHEMA,
        "tag": args.tag,
        "builder": "mx43_plane.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm": "rope-frequency-band (verbatim rows, two gain ridges, one writer ridge)",
        "axis": "rope-frequency-band",
        "module": "mx43_plane.py (PLANE, ROUND3 slate 3.4)",
        "solve": solve_note,
        "parent": str(PARENT),
        "parent_model": str(PARENT),
        "parent_parameters": PARENT_PARAMS,
        "child_model": str(args.child),
        "child_parameters": int(live),
        "child_tensors_sha256": file_sha256(args.child / "model.safetensors"),
        "child_directory_sha256": directory_sha256(args.child),
        "net_stored_ratio": int(live) / PARENT_PARAMS,
        "excised": [],
        "residual_basis": None,
        "passed_through": [
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
        ],
        "axes": [{
            "axis": "head_dim",
            "kind": "attention-width",
            "sites": list(ATTENTION_LAYERS),
            "config_fields": ["head_dim", "rope_parameters"],
            "selection": (
                "whole RoPE planes on the (stride, slide) band %d..%d -- the only "
                "RoPE-exact selection, because the child's own ladder is forced by "
                "its config -- plus a per-KV-group verbatim value-channel subset "
                "scored E[|ctx_c|] * ||W_o[:, c]||_2, ties broken on the channel "
                "index" % (planes[0], planes[-1])),
            "solve": solve_note,
            "evidence": (
                "sequentially conditioned capture through the prefix already "
                "folded; the target within each site is the parent's own local "
                "behaviour"),
            "parameters_removed": int(PARENT_PARAMS - live),
            "heldout": {
                "median_logit_relL2_prior": float(
                    sorted(r["held_logit_relL2_prior"] for r in records)[len(records) // 2]),
                "median_logit_relL2_solved": float(
                    sorted(r["held_logit_relL2_solved"] for r in records)[len(records) // 2]),
                "min_kept_plane_logit_share": min(
                    r["held_kept_plane_logit_share"] for r in records),
                "median_cos": float(sorted(r["cos"] for r in records)[len(records) // 2]),
                "min_cos": min(r["cos"] for r in records),
                "max_update_scale": max(r["update_scale"] for r in records),
                "per_site": records,
            },
        }],
        "parameters": int(live),
        "expected_parameters": int(expected_parameters),
        "ratio": int(live) / PARENT_PARAMS,
        "head_dim": head_dim,
        "stride": args.stride,
        "slide": args.slide,
        "planes": planes,
        "rope_theta": ladder_theta,
        "linear_factor": factor,
        "retained_plane_energy": float(pooled[planes].sum()) / total,
        "sites": records,
        "endpoint_held": endpoint,
        "endpoint_rows": int(parent_endpoint.shape[0]),
        "evidence": mx43_build.evidence_receipt(),
        "device": device_name,
        "compute_capability": capability,
        "forward_calls": harness.forward_calls,
        "seconds": time.time() - started,
        "elapsed_seconds": time.time() - started,
        "plane_energy_census": census,
    }
    (args.child / "mx43-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    reread = json.loads((args.child / "mx43-build-receipt.json").read_text())
    assert reread["parameters"] == int(live), "the build receipt did not survive the write"
    log(f"{args.tag}: {live} parameters, {live / PARENT_PARAMS:.10f}x, "
        f"{len(records)} sites folded, receipt re-parsed OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
