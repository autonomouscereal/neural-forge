#!/usr/bin/env python3
"""Wave 19: narrow the residual stream itself -- the last unbuilt standard axis.

Every fold class the campaign has built so far narrows something *inside* a
block: the gated intermediate (`--ff`), the head count (`--kv-groups`), the head
width (`--head-dim`), or the block list (`--excise`).  None of them touches
``hidden_size``, and ``hidden_size`` is where the money is: the tied embedding
is 65536 x 1024 = 67,108,864 parameters -- **29.2% of the model** -- and it is
the one tensor no other standard axis can reach.  That is the wall `NOTES-MX-V`
§5 names: 0.5x needs the 162.58M body to fall 3.4x while the embedding does not
move at all.  One config field moves both.

**The legal config, stated before anything is built.**  ``head_dim`` is PINNED
at the parent's 64 and written explicitly.  `Lfm2Attention` reads
``getattr(config, "head_dim", hidden_size // num_attention_heads)``, so leaving
it out at ``hidden_size`` 768 would silently build every attention block at
head_dim 48 -- a different RoPE frequency ladder, which is wave 15's
``--head-dim`` axis, which is sealed **dead** (`ARCH.HD16`, 0/15).  Pinning it
costs 1.5 points of net ratio at 768 (0.722610x against 0.707202x) and buys the
parent's own frequency ladder, the parent's own ``q_layernorm``/``k_layernorm``,
and an axis that moves ONE thing.  The child is therefore::

    hidden_size = block_dim = conv_dim = H     head_dim = 64 (explicit)
    num_attention_heads = num_heads = 16       num_key_value_heads = 8
    intermediate_size = block_ff_dim = 2560    layer_types, rope unchanged

and its parameter count is ``32*H^2 + 191541*H + 768``, which is 229,693,184 at
H=1024 -- the parent, exactly.

**The construction.**  One orthonormal ``E`` (1024 x H), shared by every
boundary, whose columns are the top eigenvectors of an evidence second moment of
the residual stream pooled over all 29 residual-write endpoints (14 blocks x
{operator_norm, ffn_norm} plus the final ``embedding_norm``).  A per-boundary
chart -- which is what wave 2's `hf_causal_cross_width_charts` compiles, and
which is excavated here rather than re-derived -- is NOT expressible in a stock
child: the residual is a running sum and there is nowhere to put a chart
transition, so the shared basis is forced by R3 and not chosen for convenience.

With ``r_c = E^T r`` the whole child is closed form, and every piece of it is
exact against the LOCAL target (the parent block evaluated at the child's own
stream, lifted back through ``E``):

* ``lift preserves the norm`` -- ``||E r_c|| = ||r_c||`` -- so the parent's
  RMSNorm at the lifted input and the child's RMSNorm at ``r_c`` differ by the
  single constant ``sqrt(1024/H)``, and no row-dependent scale is smuggled in.
* **readers** (q/k/v, conv ``in_proj``, w1/w3) fold EXACTLY:
  ``W_c = sqrt(1024/H) * W diag(g) E diag(1/g_c)``.
* **the norm gauge**: ``g_c = diag(E^T diag(g) E)`` -- the transported gain's
  own diagonal, a weighted average of the parent's gains, which is what makes
  the identity gate bit-exact (at ``E = I`` the reader fold is
  ``W diag(g) diag(1/g)``, and a bf16 product divided by its own bf16 factor is
  the other factor exactly in fp64).
* **writers** (attention ``out_proj``, conv ``out_proj``, w2, and the embedding
  rows) fold as ``E^T W`` -- the L2-optimal write into the retained subspace --
  and are then re-solved against the child's OWN installed activations by the
  campaign's prior-anchored ridge under THE LAW, with the closed-form fold as
  the prior.
* **the tied embedding** is ``W_emb E``, which is the same matrix in both roles
  -- reader ``W_emb diag(g_f) E`` wants the final gain folded in and writer
  ``E^T e_t`` does not -- so the final norm's diagonal is the only place the
  final gain can live, and it is SOLVED there in closed form against the tied
  head's own Gram (``(E^T C E) o (Z^T Z)``, one H x H system, C = W_emb^T W_emb).
* **the conv** is the one genuinely lossy piece.  `Lfm2ShortConv` gates
  channelwise (``Bx = B * x``, ``y = C * conv_out``), so an arbitrary rotation of
  its internal channel space is not representable and a basis fold is not
  available there.  Its H internal channels are a pivoted-Cholesky **coreset of
  the parent's own 1024**, verbatim conv taps and all, which is wave 14's
  selection applied to the one axis that forces it.

**The ladder** is 896, then 768, then 512 only if 768 shows life, and one
diagnostic rung at 960 if 896 dies outright.  The go/no-go census -- how much
residual energy lives in the top 896/768/512 directions -- is free and is
published by ``--mode basis``, which writes the basis as a sealed artifact so
every child at one width, and every composition on top of it, is built against
the same bytes.

Everything else -- the conditioned capture, the parity mode, the pivoted
Cholesky coreset, the prior-anchored ridge, the lambda-ladder trust region, the
excision path with wave 18's causal-mask repair, the joint relation solve and
the export -- is wave 18's, because a width child is only comparable to the
census, the pairs and the merge if it is built by the same code.

--- wave 18's header, kept because the machinery below is still its machinery ---

Wave 18: exploit the distance law, and fix the lambda the value path overfits.

Wave 17's builder with **two things added and one cosmetic defect repaired**:

* ``--kv-lambda-split`` selects the regularisation of the value/``out_proj``
  alternation on an **internal split of the FIT rows**, which is the fix
  `NOTES-MX-U` §7 named and did not build.  Wave 17 measured the alternation
  improving its own fit sum-of-squares by 16-29% at every site, moving the
  held-out per-site cosine DOWN, and costing five review cases (8/15 -> 3/15).
  Its lambda was chosen by THE LAW's trust region alone, which sees a norm and
  never sees generalisation.  With the flag on, every rung of the ladder is
  solved on twelve of the eighteen fit batches and scored on the other six, and
  the rung with the lowest **check** objective -- among those THE LAW still
  admits -- is the one re-solved on all eighteen and installed.  THE LAW is
  unchanged and still a veto; it just no longer does the choosing.  The
  campaign's own held-out block is never read by the selection.
* ``--kv-gate-probe`` writes the falsifying measurement `NOTES-MX-U` §7 named
  for the unattributed refusal: with blocks 13 AND 1 excised the attention
  recompute gate read 0.32-0.77 at all six sites against a 5e-2 tolerance and
  refused the value path, where block 13 alone reads 9.3e-3.  The probe compares
  the fp64 recomputation against the attention weights the **module itself**
  returns (this architecture's eager path returns them), measures the
  peakedness and the top-two margin of both, and repeats the whole gate in
  bf16.  It decides nothing and installs nothing.
* the log prefix reads ``[mx18]``.  Wave 17's read ``[mx16]``, which is a
  cosmetic defect the note shipped and named.

Everything else -- the conditioned capture, the parity mode, the pivoted
Cholesky coreset, the prior-anchored ridge, the lambda-ladder trust region, the
identity precondition, the excision path, the joint relation solve and the
export -- is byte-for-byte wave 17's, because wave 18's triples are only
comparable to wave 17's pairs and singles if they are built by the same code.

--- wave 17's header, kept because the machinery below is still its machinery ---

Wave 17: finish the excision census, map the pair space, solve the merge.

**The joint solve (ADDENDUM 14 section 14.3).**  ``--kv-solve joint`` replaces the
average-in-gauge merge with T7's joint two-sided evidence-weighted solve:

* q and k are solved **together**, per RoPE plane, against the RELATION both
  parents compute -- one rank-one matrix that has to serve both, with the key
  shared across the GQA group.  Closed form: the merged key lies in the span of
  the two parents' key rows, so it is a 2x2 Hermitian generalised eigenproblem
  per plane, and at two identical parents it returns that parent exactly.
* the weights are recovered by a direction-anchored ridge, because the shared
  per-head RMSNorm makes only the DIRECTION of a post-norm row achievable, with
  wave 16's average-in-gauge merge as the prior so THE LAW's trust region is
  measured against the merge this one has to beat.
* v and ``out_proj`` are then solved as the linear pair they are, with the
  attention probabilities held fixed -- T7's registered two-sided Stein form,
  one Kronecker term per query head in the group, solved by CG on the same
  operator and checked against the two-``eigh`` closed form where there is one
  term.  The probabilities are recomputed, and the recomputation is GATED
  against the context the installed module actually produced.

``--kv-solve average`` is wave 16's merge, kept because ``ARCH.KV6``'s 5/15 is
the number the joint solve has to beat and a baseline that cannot be rebuilt is
not a baseline.

Nothing here grades anything.  The child is sealed by
`experiments/seal_matrix_architecture_child.py` (ADDENDUM 12) and graded by the
frozen 15-prompt review.

--- wave 16's header, kept because the machinery below is still its machinery ---

Wave 16: assemble the LIVING standard-shape pieces in succession.

This is wave 15's builder (`run/mx15/mx15_build.py`) with **one new axis** and
nothing else moved.  The three arms, the conditioned capture, the parity mode,
the pivoted-Cholesky coreset, the prior-anchored ridge, the λ-ladder trust
region, the identity precondition and the export are byte-for-byte wave 15's,
because wave 16's question is what the pieces do *together* and a rebuilt piece
that is not the same piece answers nothing.

**Only local objectives are used this wave.**  `NOTES-MX-S` §2.1/§5 priced the
trajectory objective twice -- 0/15 at 4x width and 0/15 at *unchanged* width with
full capacity and zero parameter cost, against the same excision's 6/15 with
local targets -- so ``--arm conditioned-trajectory`` survives here only as the
control it already lost.  Every rung this wave is ``conditioned-local``.

**The new axis: head COUNT.**  ``--kv-groups`` merges whole attention heads,
which is a different axis from wave 15's dead ``--head-dim``: every surviving
head keeps the parent's full ``head_dim`` 64, so the RoPE frequency ladder is the
parent's own and the plane-stride problem that cost `ARCH.HD16` three quarters of
the score operator (`NOTES-MX-S` §3) does not arise at all.  What it costs
instead is stated in `merge_gauge` and measured per site.

The conv-channel axis is NOT available and this script cannot express it:
`Lfm2ShortConv` hardwires ``config.hidden_size`` (`NOTES-MX-R` §1), so it is a
research probe under R3 and never a frontier claim.

Nothing here grades anything.  The child is sealed by
`experiments/seal_matrix_architecture_child.py` (ADDENDUM 12) and graded by the
frozen 15-prompt review.

--- wave 15's header, kept because the machinery below is still its machinery ---

Wave 15: build standard-shape LFM2 children, sequentially conditioned.

`NOTES-MX-R` §3/§4 is the problem this script exists to attack.  Wave 14 folded
every one of the fourteen SwiGLU sites to 640 channels -- **verbatim channels**,
selected by pivoted Cholesky, with the writer re-solved, which is the coreset arm
and not a synthesis arm -- got a held-out write cosine of 0.81-0.94 at every
site, the same band the sealed lanes that compose into coherent children sit in,
and produced a child that answers nothing (`MX14-ff640`, 0.6405x, 0/15).  The
cliff it located sits between a per-site write cosine of 0.93 and 0.86 **when the
fold is installed at all fourteen sites at once**.

The one variable wave 14 never moved is the evidence.  Every one of its fourteen
solves was computed against the **parent's** activations and then installed into
a model whose layer-``l`` input is no longer the parent's, so site ``l``'s 0.86
cosine is a promise about a distribution the child never sees again once site 0
is installed.  This script makes that variable movable and measures it:

``--arm stale``
    Wave 14's objective, recomputed through this harness.  It is the **control**:
    if it does not reproduce wave 14's per-site numbers, nothing downstream of it
    means anything.
``--arm conditioned-local``
    Sequential.  Layer ``l``'s selection and solve are computed against the
    activations of the child that already has layers ``0..l-1`` folded, and the
    target is the **parent block evaluated at the child's own input** -- a local
    teacher.  Stale error stops compounding; drift is accepted, not corrected.
``--arm conditioned-trajectory``
    Sequential, same features, but the target is the parent block's output **on
    the parent's own stream**, row-aligned.  The writer is asked not merely to
    approximate its own map but to put the residual stream back where the parent
    left it.  Strictly harder to fit and strictly more corrective; which one wins
    is a measurement, not a prediction.

Everything else is wave 14's, deliberately: the same pivoted-Cholesky coreset of
real channels, the same prior-anchored ridge, the same λ-ladder trust region
enforcing THE LAW (``update_scale <= 1``), the same identity precondition on the
gated intermediate, the same fit/held-out split, the same corpus.

**The capture.**  The wave-13 trace cache is a frozen artifact and cannot be
re-run through a child, so conditioned captures are taken here, by hooks, on the
same 24 batches: each batch's ``input.input_ids`` (256 prompt tokens) and
``input.forced_decode_ids`` (8 forced tokens) are concatenated into one 264-token
teacher-forced sequence.  That reproduces the sealed row order exactly -- the
collector flattens prefill's 256 rows and then decode.000..007 one row each, in
ascending batch order, and a single causal forward over the concatenation
produces the same 264 positions in the same order.  ``--mode parity`` proves it
against the sealed rows before any child is built, and its number rides in every
build receipt.

**The axes.**  Three, all config-standard against `transformers` 5.12.1:

* ``--ff`` writes ``intermediate_size``/``block_ff_dim`` -- one global field, so
  the fold is at all fourteen sites or it is not config-consistent;
* ``--head-dim`` writes ``head_dim`` (a field this checkpoint does not have and
  `Lfm2Attention` reads through ``getattr``) plus ``rope_theta``, selecting whole
  RoPE planes so the child's frequency ladder is an exact **subset of the
  parent's own**, with the two scale corrections that change forces;
* ``--excise`` drops whole blocks, writing ``num_hidden_layers`` and
  ``layer_types``.

The conv-channel axis is NOT available and this script cannot express it:
`Lfm2ShortConv` hardwires ``config.hidden_size`` (`NOTES-MX-R` §1), so it is a
research probe under R3 and never a frontier claim.

Nothing here grades anything.  The child is sealed by
`experiments/seal_matrix_architecture_child.py` (ADDENDUM 12) and graded by the
frozen 15-prompt review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PARENT = Path("/home/cereal/model-surgery-lab/models/LFM2.5-230M")
TRACE = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/2dec87acc15822082be4da0bed0ba939a06d293eb3b87321ddaf383ae61a56ff"
)
SPLIT_MANIFEST = Path(
    "/home/cereal/model-surgery-lab/artifacts/neural_forge_matrix_campaign_20260810"
    "/evidence/evidence-manifest.json"
)
SEALED_EVIDENCE = Path("/mnt/ai_data/neural-forge-matrix/wave13/evidence-cache")

SIDECARS = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)

PARENT_PARAMS = 229_693_184
SCHEMA = "neural-forge-matrix-wave19-standard-shape-child-v1"
#: The parent's residual width, and the head width this wave PINS.  Both are
#: read back out of the parent's config and checked against these, because the
#: whole legality argument of the ``hidden_size`` axis rests on the second one
#: being written explicitly into the child's config.
PARENT_HIDDEN = 1024
PINNED_HEAD_DIM = 64
#: The basis artifact's schema.  A width child, and every composition built on
#: top of it, must be built against the same bytes or the rungs are not a ladder.
BASIS_SCHEMA = "neural-forge-matrix-wave19-residual-basis-v1"
#: The floor under a transported gain ``g_c = diag(E^T diag(g) E)`` before the
#: reader fold's division by it is refused.  ``g_c[j]`` is a convex combination
#: of the parent's gains (the weights ``E[:,j]**2`` sum to one), so it can only
#: approach zero if the parent's own gains straddle zero; the guard is here to
#: make that a refusal rather than a silent amplification, and every child's
#: receipt carries the measured minimum.
GAIN_FLOOR = 1e-3
RIDGE_REL = 1e-6
LAMBDA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
#: The internal FIT-side split the value path selects its lambda on when
#: ``--kv-lambda-split`` is set -- `NOTES-MX-U` §7's named fix, built here.
#: Eighteen of the campaign's twenty-four captured batches are fit; every third
#: one of those, counted in fit order from the third, is withheld from the
#: ladder's own solve and used only to score it.  Six of eighteen is the same
#: one-in-three the capture manifest holds out at the corpus level, and it is a
#: POSITION rule rather than a content rule, so it is identical for every child,
#: every site and every sweep.  The campaign's held-out block is not touched by
#: this: it is a split of the fit rows, and it is why the flag exists.
INTERNAL_CHECK_STRIDE = 3
INTERNAL_CHECK_OFFSET = 2
ATTENTION_LAYERS = (2, 4, 6, 8, 10, 12)
#: Wave 14's identity precondition, re-floored for this harness and measured
#: before it was moved.  The guard catches ONE thing -- selecting on an endpoint
#: that is not the gated intermediate ``w2`` reads -- and that error is O(1).
#: Wave 14 compared an fp64 recomputation against the **sealed fp64 evidence**
#: and could use 5e-3; here the reference is the live module's own **bf16**
#: output, so the residual carries a bf16 floor.  Measured on the parent's own
#: stream, that floor is 3.2e-3 to 3.8e-3 at thirteen sites and 1.0e-3 at L05
#: (where one unit carries 97% of the energy, so the norm is dominated by a
#: quantity bf16 represents well).  It grows with input drift: the
#: conditioned-trajectory arm reached 6.5e-3 at L05 with the stream 0.88 away
#: from the parent's.  The tolerance is set an order of magnitude above the
#: measured floor and two orders below an axis error, and every site's value is
#: sealed in the build receipt so the guard's headroom is visible.
IDENTITY_TOLERANCE = 5e-2
#: The gate on the recomputed attention probabilities (ADDENDUM 14 section
#: 14.3).  The harness runs the parent in bf16 and the recomputation is fp64, so
#: the two cannot agree better than bf16 rounding through a softmax and two
#: matmuls; the tolerance is the same order as `IDENTITY_TOLERANCE`, set well
#: above the floor and well below an indexing error, and the measured value
#: rides in every receipt so the headroom is visible rather than assumed.
ATTENTION_RECOMPUTE_TOLERANCE = 5e-2


def log(message: str) -> None:
    print(f"[mx19] {message}", flush=True)


# --------------------------------------------------------------------- corpus
def corpus() -> list[tuple[int, torch.Tensor]]:
    """The 24 captured batches, each as one 264-token teacher-forced sequence."""

    from safetensors import safe_open

    out: list[tuple[int, torch.Tensor]] = []
    for directory in sorted((TRACE / "batches").iterdir()):
        index = int(directory.name.split("-")[1])
        with safe_open(str(directory / "trace.safetensors"), "pt") as handle:
            prompt = handle.get_tensor("input.input_ids")
            forced = handle.get_tensor("input.forced_decode_ids")
        out.append((index, torch.cat([prompt, forced], dim=1).to(torch.long)))
    if len(out) != 24:
        raise SystemExit(f"expected 24 captured batches, found {len(out)}")
    return out


def split_by_batch() -> dict[int, str]:
    """The campaign's own sealed FIT/held-out split, re-sealed before it is used."""

    body = json.loads(SPLIT_MANIFEST.read_text())
    declared = body.get("manifest_sha256")
    recomputed = hashlib.sha256(
        json.dumps(
            {k: v for k, v in body.items() if k != "manifest_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    if declared != recomputed:
        raise SystemExit(f"split manifest seal is broken: {declared} != {recomputed}")
    return {int(row["batch_index"]): str(row["split"]) for row in body["split"]["rows"]}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------- algebra
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


def law_abiding_solve(
    A: torch.Tensor, Y: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The λ-ladder trust region.  THE LAW is a guard and its cost is recorded."""

    free = prior_anchored(A, Y, prior, rel=LAMBDA_LADDER[0])
    free_scale = float((free - prior).norm() / prior.norm())
    for rel in LAMBDA_LADDER:
        candidate = prior_anchored(A, Y, prior, rel=rel)
        scale = float((candidate - prior).norm() / prior.norm())
        if scale <= 1.0:
            return candidate, {
                "update_scale": scale,
                "lambda_relative": rel,
                "free_solve_update_scale": free_scale,
            }
    raise SystemExit(
        f"no lambda on the ladder brings update_scale to 1.0 (free {free_scale:.3f}); "
        "THE LAW refuses the install"
    )


def law_abiding_solve_split(
    A: torch.Tensor, Y: torch.Tensor, check: torch.Tensor, prior: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """`law_abiding_solve` with lambda chosen on an internal split of its own rows.

    ``check`` is a boolean mask over ``A``'s rows: the rows the ladder is scored
    on and never solved on.  Every rung is solved on the complement, scored on
    the mask, and the rung with the lowest **check** sum-of-squares that THE LAW
    still admits is re-solved on all the rows and returned.  THE LAW is
    unchanged -- it is still a veto on the update scale of the installed weight
    -- it simply no longer picks among the rungs it admits, which is the whole
    of `NOTES-MX-U` §7's named fix.  Wave 17's choice (ladder order, first
    admissible) is recorded beside the new one so the difference is legible.
    """

    inner = ~check
    A_in, Y_in = A[inner], Y[inner]
    A_ck, Y_ck = A[check], Y[check]
    reference = float(prior.norm())
    free_scale = float(
        (prior_anchored(A, Y, prior, rel=LAMBDA_LADDER[0]) - prior).norm() / reference
    )
    rungs: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        candidate_in = prior_anchored(A_in, Y_in, prior, rel=rel)
        candidate = prior_anchored(A, Y, prior, rel=rel)
        rungs.append(
            {
                "lambda_relative": rel,
                "check_sse": float((A_ck @ candidate_in.T - Y_ck).pow(2).sum()),
                "update_scale": float((candidate - prior).norm() / reference),
                "solved": candidate,
            }
        )
    admitted = [row for row in rungs if row["update_scale"] <= 1.0]
    if not admitted:
        raise SystemExit(
            f"no lambda on the ladder brings update_scale to 1.0 (free {free_scale:.3f}); "
            "THE LAW refuses the install"
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
        "ladder": [
            {k: v for k, v in row.items() if k != "solved"} for row in rungs
        ],
    }


def write_scores(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = pred - target
    cos = torch.nn.functional.cosine_similarity(pred, target, dim=1)
    return {
        "relL2": float(diff.norm() / target.norm()),
        "cos": float(cos.mean()),
        "cos_p05": float(torch.quantile(cos, torch.tensor(0.05, dtype=cos.dtype))),
    }


def pivoted_columns(F: torch.Tensor, k: int) -> torch.Tensor:
    """Wave 14's deterministic column-pivoted selection, unchanged (mx14_measure2)."""

    n = int(F.shape[1])
    if k >= n:
        # Keeping every channel is not a selection, and running the greedy loop
        # to discover that costs n*k Python iterations.  The identity is the
        # answer and the width-unchanged control arm needs it.
        return torch.arange(n, dtype=torch.long)
    G = (F.T @ F).clone()
    diag = torch.diagonal(G).clone()
    chosen: list[int] = []
    L = torch.zeros(n, k, dtype=G.dtype)
    for step in range(k):
        best = -1.0
        pick = -1
        for i in range(n):
            if diag[i] > best + 1e-15:
                best = float(diag[i])
                pick = i
        if pick < 0 or best <= 1e-30:
            break
        chosen.append(pick)
        col = G[:, pick] - L[:, :step] @ L[pick, :step]
        col = col / max(best, 1e-300) ** 0.5
        L[:, step] = col
        diag = diag - col * col
        diag[chosen] = -1.0
    return torch.tensor(sorted(chosen), dtype=torch.long)


def swiglu(X: torch.Tensor, w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(X @ w1.T) * (X @ w3.T)


# ------------------------------------------------------------------- harness
class Harness:
    """One model on one GPU plus hooks that capture a module's input and the
    **pristine** module's output on that same input.

    The second one is what makes a *local* target possible at all: the parent
    block evaluated at the child's input appears in no recording of the parent's
    own run, and it is exactly what a sequentially conditioned solve wants to hit.
    """

    def __init__(self, model_dir: Path, device: str, dtype: torch.dtype) -> None:
        from transformers import Lfm2ForCausalLM

        self.model = (
            Lfm2ForCausalLM.from_pretrained(
                str(model_dir), dtype=dtype, attn_implementation="eager"
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.dtype = dtype
        self.batches = corpus()
        self.split = split_by_batch()
        self.forward_calls = 0

    @classmethod
    def wrap(cls, model: Any, device: str, dtype: torch.dtype) -> "Harness":
        """The same harness around a model this script BUILT rather than loaded.

        The ``hidden_size`` axis cannot narrow the parent in place -- a residual
        of two widths is not a model -- so its child is instantiated from a
        config and folded into.  Everything downstream of the harness (the
        capture, the conditioned solve, the head-count merge) is width-agnostic
        and needs the child to arrive the same way the parent does.
        """

        harness = cls.__new__(cls)
        harness.model = model.eval()
        harness.device = device
        harness.dtype = dtype
        harness.batches = corpus()
        harness.split = split_by_batch()
        harness.forward_calls = 0
        return harness

    def layers(self):
        return self.model.model.layers

    @torch.no_grad()
    def capture(
        self, targets: dict[str, Any], *, pristine: dict[str, Any] | None = None
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Run the corpus once; return per-name fp64 ``fit``/``hld`` row blocks
        and, where a pristine module was supplied, ``local_fit``/``local_hld``."""

        buckets: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {"fit": [], "hld": [], "local_fit": [], "local_hld": []}
            for name in targets
        }
        state: dict[str, Any] = {}
        handles = []

        def make_hook(name: str):
            def hook(module, args, kwargs):
                state[name] = (args, kwargs)
                return None

            return hook

        for name, module in targets.items():
            handles.append(
                module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
            )
        try:
            for index, ids in self.batches:
                bucket = "fit" if self.split[index] == "fit" else "hld"
                self.model(input_ids=ids.to(self.device), use_cache=False)
                self.forward_calls += 1
                for name in targets:
                    args, kwargs = state[name]
                    hidden = args[0] if args else kwargs["hidden_states"]
                    buckets[name][bucket].append(
                        hidden.detach()
                        .reshape(-1, hidden.shape[-1])
                        .to("cpu", torch.float64)
                    )
                    if pristine is not None and name in pristine:
                        out = pristine[name](*args, **kwargs)
                        if isinstance(out, tuple):
                            out = out[0]
                        buckets[name][f"local_{bucket}"].append(
                            out.detach()
                            .reshape(-1, out.shape[-1])
                            .to("cpu", torch.float64)
                        )
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return {
            name: {k: torch.cat(v, dim=0).contiguous() for k, v in rows.items() if v}
            for name, rows in buckets.items()
        }


    @torch.no_grad()
    def capture_batches(
        self, targets: dict[str, Any]
    ) -> dict[str, list[tuple[str, torch.Tensor]]]:
        """`capture`, with the rows left grouped by batch and tagged with their split.

        The flat capture is the right shape for every solve in this file except
        one: attention probabilities are per SEQUENCE, and a flat row block has
        forgotten where each sequence started.  Same hooks, same corpus, same
        order -- the concatenation of the fit batches here is the flat capture's
        fit block, and `joint_value_sweeps` refuses to run if it is not.
        """

        buckets: dict[str, list[tuple[str, torch.Tensor]]] = {name: [] for name in targets}
        state: dict[str, Any] = {}
        handles = []

        def make_hook(name: str):
            def hook(module, args, kwargs):
                state[name] = (args, kwargs)
                return None

            return hook

        for name, module in targets.items():
            handles.append(
                module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
            )
        try:
            for index, ids in self.batches:
                split = "fit" if self.split[index] == "fit" else "hld"
                self.model(input_ids=ids.to(self.device), use_cache=False)
                self.forward_calls += 1
                for name in targets:
                    args, kwargs = state[name]
                    hidden = args[0] if args else kwargs["hidden_states"]
                    buckets[name].append(
                        (
                            split,
                            hidden.detach()
                            .reshape(-1, hidden.shape[-1])
                            .to("cpu", torch.float64),
                        )
                    )
                state.clear()
        finally:
            for handle in handles:
                handle.remove()
        return buckets


def sealed_rows(name: str, endpoint: str, split: str) -> torch.Tensor:
    payload = torch.load(
        SEALED_EVIDENCE / name / "payload.pt", weights_only=True, map_location="cpu"
    )
    return payload[f"{'fit' if split == 'fit' else 'hld'}|{endpoint}"].to(torch.float64)


# --------------------------------------------------------------------- parity
def parity(device: str, layers: tuple[int, ...] = (0, 5, 13)) -> dict[str, Any]:
    """Prove this harness's fresh capture reproduces the sealed collector's rows.

    Not a formality.  If the row order, the phase concatenation or the dtype were
    wrong, every conditioned number below would be measured against a different
    corpus than the campaign's and the comparison to wave 14 would be void.
    """

    harness = Harness(PARENT, device, torch.bfloat16)
    targets = {f"L{l:02d}": harness.layers()[l].feed_forward for l in layers}
    captured = harness.capture(targets)
    report: dict[str, Any] = {}
    for l in layers:
        name = f"L{l:02d}"
        w1 = harness.layers()[l].feed_forward.w1.weight.to(torch.float64).cpu()
        w3 = harness.layers()[l].feed_forward.w3.weight.to(torch.float64).cpu()
        for split in ("fit", "hld"):
            X = captured[name][split]
            phi = swiglu(X, w1, w3)
            reference = sealed_rows(f"{name}-C13", "mlp.post_swiglu", split)
            if reference.shape != phi.shape:
                raise SystemExit(
                    f"{name} {split}: fresh capture is {tuple(phi.shape)}, sealed is "
                    f"{tuple(reference.shape)}"
                )
            rel = float((phi - reference).norm() / reference.norm())
            cos = float(
                torch.nn.functional.cosine_similarity(phi, reference, dim=1).mean()
            )
            report[f"{name}.{split}"] = {
                "rows": int(phi.shape[0]),
                "relL2": rel,
                "cos": cos,
            }
            log(f"parity {name} {split}: rows {phi.shape[0]} relL2 {rel:.3e} cos {cos:.6f}")
    return report


# ------------------------------------------------------------- the MLP axis
def install_narrow_mlp(layer, keep: int, sel: torch.Tensor, w2p: torch.Tensor, dtype) -> None:
    """Install a narrower SwiGLU: verbatim gate/up rows, solved down."""

    w1 = layer.feed_forward.w1.weight.data
    w3 = layer.feed_forward.w3.weight.data
    device = w1.device
    index = sel.to(device)
    new_w1 = nn.Linear(w1.shape[1], keep, bias=False, dtype=dtype, device=device)
    new_w3 = nn.Linear(w3.shape[1], keep, bias=False, dtype=dtype, device=device)
    new_w2 = nn.Linear(keep, w1.shape[1], bias=False, dtype=dtype, device=device)
    new_w1.weight.data.copy_(w1[index])
    new_w3.weight.data.copy_(w3[index])
    new_w2.weight.data.copy_(w2p.to(device=device, dtype=dtype))
    layer.feed_forward.w1 = new_w1
    layer.feed_forward.w3 = new_w3
    layer.feed_forward.w2 = new_w2


# ------------------------------------------------------- the head_dim axis
def rope_plane_sets(parent_head_dim: int, keep_dim: int) -> dict[int, list[int]]:
    """The only RoPE-exact plane sets a smaller ``head_dim`` can express.

    RoPE's rotation is block diagonal over frequency planes, so ``q^T R_d k`` is
    **exactly** a sum of per-plane terms, and dropping planes drops their terms
    and nothing else.  The child's own ladder is forced by its config:
    ``freq_j = theta'^(-2j/keep_dim)``.  Matching a subset of the parent's
    ``theta^(-2i/parent)`` therefore requires ``i = s*j`` for an integer stride
    ``s``, with ``theta' = theta^(s*keep_dim/parent)``.  Any other selection
    would place the child's planes at frequencies the parent never used, which is
    not a fold of the parent's operator at all.
    """

    planes = parent_head_dim // 2
    want = keep_dim // 2
    out: dict[int, list[int]] = {}
    for stride in (1, 2, 4, 8):
        picked = [stride * j for j in range(want)]
        if picked[-1] < planes:
            out[stride] = picked
    return out


def plane_energy(q_rows: torch.Tensor, k_rows: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Per-plane share of the score operator's energy, on the evidence.

    Plane ``p`` contributes ``q_p . R_d k_p`` to every score and nothing else
    does; the product of the two planes' second moments is the scale of that
    contribution.  It is a census -- it ranks the strides and gates nothing.
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

    ``position_embeddings`` are computed once per forward by `Lfm2Model` from
    ``config.head_dim``, so the moment the child's ladder is installed globally --
    and it must be installed globally, because ``rope_theta`` and ``head_dim`` are
    one config field each -- every **unfolded** block would be handed cos/sin of
    the wrong width.  A sequentially conditioned build needs the mixed state to be
    valid for exactly as long as it takes to walk the blocks, so each unfolded
    block keeps the parent's rotary until its own fold removes the shim.  This is
    build-time scaffolding: no shim survives into the exported child, which is why
    the child loads with no custom module at all.
    """

    def __init__(self, rotary) -> None:
        self.rotary = rotary
        self.handles: dict[int, Any] = {}

    def attach(self, module) -> None:
        def hook(mod, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                position_ids = torch.arange(
                    hidden.shape[1], device=hidden.device
                ).unsqueeze(0)
            kwargs["position_embeddings"] = self.rotary(hidden, position_ids)
            return args, kwargs

        self.handles[id(module)] = module.register_forward_pre_hook(
            hook, with_kwargs=True
        )

    def detach(self, module) -> None:
        handle = self.handles.pop(id(module), None)
        if handle is not None:
            handle.remove()


def head_dim_axis(
    harness: Harness,
    position: int,
    parent_layer: int,
    keep_dim: int,
    *,
    stride: int,
    dims: torch.Tensor,
    stride_census: dict[str, Any],
    shim: "ParentRope | None",
    trajectory_target: tuple[torch.Tensor, torch.Tensor] | None,
) -> dict[str, Any]:
    """Fold one attention block to ``head_dim = keep_dim``, config-standard.

    q/k keep whole RoPE planes -- the only RoPE-exact choice -- and v is
    contracted by the per-KV-group evidence-weighted LRA wave 14 measured at
    cosine 0.84-0.94.  ``out_proj`` is re-solved against the **installed**
    context, so the number reported is the error of the thing that runs and not
    of the selection's own estimate.  Two scale corrections are folded into gains
    that already exist, because `Lfm2Attention` recomputes
    ``scaling = head_dim ** -0.5`` and the per-head RMSNorm now divides by an RMS
    over ``keep_dim`` channels instead of 64.  Both are measured, not assumed.
    """

    layer = harness.layers()[position]
    attn = layer.self_attn
    config = harness.model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    parent_head_dim = int(attn.head_dim)
    group_size = heads // kv_heads
    pristine = copy.deepcopy(attn)

    captured = harness.capture({"attn": attn}, pristine={"attn": pristine})
    X = captured["attn"]["fit"]
    X_h = captured["attn"]["hld"]
    if trajectory_target is not None:
        target, target_h = trajectory_target
    else:
        target, target_h = captured["attn"]["local_fit"], captured["attn"]["local_hld"]

    wq = attn.q_proj.weight.data.to(torch.float64).cpu()
    wk = attn.k_proj.weight.data.to(torch.float64).cpu()
    wv = attn.v_proj.weight.data.to(torch.float64).cpu()
    wo = attn.out_proj.weight.data.to(torch.float64).cpu()

    # ---- q/k: whole RoPE planes on the globally chosen stride (rope_theta is
    # one config field, so the ladder is a model-level decision, not a per-site one).
    q_all = X @ wq.T
    k_all = X @ wk.T
    census: dict[str, Any] = dict(stride_census)

    # ---- v: per-KV-group evidence-weighted low-rank contraction.
    v_all = X @ wv.T
    projections: list[torch.Tensor] = []
    retained = []
    for g in range(kv_heads):
        group_heads = [g * group_size + j for j in range(group_size)]
        Cg = v_all[:, g * parent_head_dim : (g + 1) * parent_head_dim]
        Gg = Cg.T @ Cg / Cg.shape[0]
        ev, evec = torch.linalg.eigh(Gg)
        Rt = evec @ torch.diag(ev.clamp_min(0).sqrt()) @ evec.T
        stack = torch.cat(
            [wo[:, h * parent_head_dim : (h + 1) * parent_head_dim] for h in group_heads],
            dim=0,
        )
        _, S, Vh = torch.linalg.svd(stack @ Rt, full_matrices=False)
        Rt_inv = evec @ torch.diag(1.0 / ev.clamp_min(1e-12).sqrt()) @ evec.T
        P = Vh[:keep_dim] @ Rt_inv.T
        P = torch.linalg.qr(P.T)[0].T
        projections.append(P)
        retained.append(float((S[:keep_dim] ** 2).sum() / (S**2).sum()))
    census["v_retained_energy_min"] = min(retained)
    census["v_retained_energy_median"] = float(torch.tensor(retained).median())

    # ---- install q/k/v at the new width, with both scale corrections.
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    q_rows = torch.cat([dims + h * parent_head_dim for h in range(heads)])
    k_rows = torch.cat([dims + g * parent_head_dim for g in range(kv_heads)])
    new_q = nn.Linear(wq.shape[1], heads * keep_dim, bias=False, dtype=dtype, device=device)
    new_k = nn.Linear(wk.shape[1], kv_heads * keep_dim, bias=False, dtype=dtype, device=device)
    new_v = nn.Linear(wv.shape[1], kv_heads * keep_dim, bias=False, dtype=dtype, device=device)
    new_q.weight.data.copy_(attn.q_proj.weight.data[q_rows.to(device)])
    new_k.weight.data.copy_(attn.k_proj.weight.data[k_rows.to(device)])
    new_v.weight.data.copy_(
        torch.cat(
            [
                projections[g] @ wv[g * parent_head_dim : (g + 1) * parent_head_dim]
                for g in range(kv_heads)
            ],
            dim=0,
        ).to(device=device, dtype=dtype)
    )

    def rms_gain(rows: torch.Tensor) -> float:
        full = rows.reshape(-1, parent_head_dim)
        kept = full[:, dims]
        eps = float(config.norm_eps)
        r_full = full.pow(2).mean(dim=1).add(eps).sqrt()
        r_kept = kept.pow(2).mean(dim=1).add(eps).sqrt()
        return float((r_kept / r_full).mean())

    q_gain = rms_gain(q_all)
    k_gain = rms_gain(k_all)
    softmax_correction = math.sqrt(keep_dim / parent_head_dim)
    attn.q_layernorm.weight = nn.Parameter(
        (
            attn.q_layernorm.weight.data[dims.to(device)].to(torch.float64)
            * q_gain
            * softmax_correction
        ).to(dtype)
    )
    attn.k_layernorm.weight = nn.Parameter(
        (attn.k_layernorm.weight.data[dims.to(device)].to(torch.float64) * k_gain).to(dtype)
    )
    attn.q_proj, attn.k_proj, attn.v_proj = new_q, new_k, new_v
    attn.head_dim = keep_dim
    attn.scaling = keep_dim**-0.5
    census["rms_gain_correction"] = {"q": q_gain, "k": k_gain}
    census["softmax_scale_correction"] = softmax_correction

    # ---- out_proj: re-solved against the context the installed attention makes.
    prior = torch.cat(
        [
            wo[:, h * parent_head_dim : (h + 1) * parent_head_dim] @ projections[h // group_size].T
            for h in range(heads)
        ],
        dim=1,
    )
    new_o = nn.Linear(heads * keep_dim, wo.shape[0], bias=False, dtype=dtype, device=device)
    new_o.weight.data.copy_(prior.to(device=device, dtype=dtype))
    attn.out_proj = new_o
    # The block is now the child's width, so it takes the child's ladder.
    if shim is not None:
        shim.detach(attn)
    context = harness.capture({"o": attn.out_proj})
    C = context["o"]["fit"]
    C_h = context["o"]["hld"]
    prior_scores = write_scores(C_h @ prior.T, target_h)
    solved, guard = law_abiding_solve(C, target, prior)
    new_o.weight.data.copy_(solved.to(device=device, dtype=dtype))
    scores = write_scores(C_h @ solved.T, target_h)
    return {
        "parent_layer": parent_layer,
        "position": position,
        "keep_dim": keep_dim,
        "census": census,
        "prior_only_cos": prior_scores["cos"],
        **guard,
        **scores,
    }


# ----------------------------------------------- the exact merge gauge
def _planes(head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The index pairs RoPE rotates together, in HF's ``rotate_half`` layout."""

    half = head_dim // 2
    return torch.arange(half), torch.arange(half) + half


def sign_gauge(A: torch.Tensor, B: torch.Tensor, head_dim: int) -> torch.Tensor:
    """The **exact** head-alignment gauge this architecture leaves open.

    Two heads can only be merged in a common frame, and the frame change has to
    be invisible to everything downstream of it.  Two things constrain it here
    and they intersect in one place:

    * **RoPE.**  ``q^T R_d k`` is a sum of per-plane terms and every plane's
      angle is distinct, so the only linear maps commuting with ``R_d`` at every
      offset are per-plane rotation-scalings.  A gauge that mixes planes changes
      the score at every offset and is not a gauge.
    * **The shared per-head RMSNorm.**  `Lfm2Attention` normalizes each head and
      then multiplies by ONE weight vector shared by all heads, so a gauge ``G``
      must satisfy ``diag(w) G = G diag(w)`` -- and ``w`` is measurably *not*
      plane-symmetric on this parent (``|w_p - w_{p+32}|`` is 0.11-0.40 of
      ``||w||`` at the twelve q/k norms), so a continuous plane rotation does
      **not** commute with it.  The normalization itself removes any uniform
      scale and re-weights the planes under any non-uniform one.

    What survives both is the per-plane scalar of unit modulus: ``s_p = ±1``,
    applied to both coordinates of plane ``p``.  It commutes with ``R_d`` (a
    scalar), commutes with ``diag(w)`` (a scalar), and preserves every plane's
    norm and therefore the head's RMS exactly.  Applied to the key rows of one
    head and to the query rows of the heads that read it, the score is
    **unchanged** -- ``(s q)^T R_d (s k) = q^T R_d k``.

    The optimum is per plane and closed form: flip when the evidence-weighted
    inner product is negative.
    """

    lo, hi = _planes(head_dim)
    inner = (A[:, lo] * B[:, lo]).sum(dim=0) + (A[:, hi] * B[:, hi]).sum(dim=0)
    sign = torch.where(inner < 0.0, -1.0, 1.0).to(A.dtype)
    return torch.cat([sign, sign])


def rotation_gauge_cos(A: torch.Tensor, B: torch.Tensor, head_dim: int) -> float:
    """What a CONTINUOUS per-plane Procrustes rotation would have bought.

    Census only, and it is the price of the constraint above rather than a
    candidate: this rotation is optimal in the plane and inadmissible through
    the shared norm weight.  Per plane it is one complex least-squares scalar
    restricted to unit modulus, ``c = <b, a> / |<b, a>|``.
    """

    lo, hi = _planes(head_dim)
    ar, ai = A[:, lo], A[:, hi]
    br, bi = B[:, lo], B[:, hi]
    # sum_n conj(b_n) a_n, per plane
    real = (br * ar + bi * ai).sum(dim=0)
    imag = (br * ai - bi * ar).sum(dim=0)
    magnitude = (real**2 + imag**2).sqrt().clamp_min(1e-30)
    cr, ci = real / magnitude, imag / magnitude
    rotated_r = br * cr - bi * ci
    rotated_i = br * ci + bi * cr
    rotated = torch.cat([rotated_r, rotated_i], dim=1)
    return float(
        torch.nn.functional.cosine_similarity(rotated, A, dim=1).mean()
    )



# --------------------------------------------- the T7 joint two-sided solve
def as_planes(rows: torch.Tensor, head_dim: int) -> torch.Tensor:
    """One head's rows as ``head_dim//2`` complex planes, in HF's own layout.

    `rotate_half` pairs coordinate ``p`` with ``p + head_dim//2``, so plane ``p``
    is the complex number ``rows[:, p] + i rows[:, p + head_dim//2]`` and RoPE at
    offset ``d`` multiplies it by ``e^{i θ_p d}``.  Every statement this module
    makes about relations is made in this basis.
    """

    lo, hi = _planes(head_dim)
    return torch.complex(rows[:, lo], rows[:, hi])


def from_planes(planes: torch.Tensor) -> torch.Tensor:
    """The inverse of `as_planes`: back to a real ``head_dim`` row block."""

    return torch.cat([planes.real, planes.imag], dim=1).contiguous()


def rank_one_relation_error(
    u: torch.Tensor, w: torch.Tensor, u_p: torch.Tensor, w_p: torch.Tensor
) -> tuple[float, float]:
    """``(‖u wᴴ − u_p w_pᴴ‖²_F , ‖u_p w_pᴴ‖²_F)`` without forming either matrix.

    The relation is an N×N object and N is six thousand rows; every quantity
    this solve needs from it is an inner product, so none of them is ever built.
    """

    cross = (u.conj() * u_p).sum() * (w_p.conj() * w).sum()
    left = float((u.abs() ** 2).sum() * (w.abs() ** 2).sum())
    right = float((u_p.abs() ** 2).sum() * (w_p.abs() ** 2).sum())
    return left - 2.0 * float(cross.real) + right, right


def joint_relation_solve(
    q_parents: list[list[torch.Tensor]],
    k_parents: list[torch.Tensor],
    weights: list[float],
    head_dim: int,
    anchor: tuple[list[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, Any]]:
    """T7's joint two-sided evidence-weighted solve for one merged head pair.

    ``q_parents[j][P]`` is parent ``P``'s post-norm query rows for query slot
    ``j`` of the group, ``k_parents[P]`` is parent ``P``'s post-norm key rows,
    and ``weights[P]`` is the evidence weight ADDENDUM 13's census already uses
    (the parent's key energy).  Everything is (N × head_dim) real, fp64.

    **The objective.**  What the model computes from a head pair is not a weight
    vector; it is a relation.  Per plane ``p`` the score contribution is
    ``q_p(i) conj(k_p(j)) e^{iθ_p(i−j)}`` -- an outer product over evidence rows
    whose frequency is shared by every head -- so the merged head's plane-``p``
    job is **one rank-one matrix that has to serve both parents at once**:

        min  Σ_j Σ_P ω_P ‖ û^j ŵᴴ − u^{P,j} (w^P)ᴴ ‖²_F

    over the merged query rows ``û^j`` (one per query slot) and the **shared**
    merged key rows ``ŵ`` -- shared because a GQA group has one KV head and
    ``group_size`` query heads read it, which is exactly what couples the slots.

    **Why it is closed form.**  For fixed ``ŵ`` the optimal ``û^j`` is one step
    (``û^j = Σ_P ω_P u^{P,j} ⟨w^P, ŵ⟩ / Σ_P ω_P``, at ‖ŵ‖ = 1).  Substituting it
    back leaves a Rayleigh quotient whose operator has the two parents' key rows
    as its entire range, so **ŵ lies in span{w^a, w^b}** and the problem is a 2×2
    Hermitian generalised eigenproblem per plane.  No iteration, no initialiser,
    no local optimum -- and at ``P = 1`` (or two identical parents) it returns
    that parent's own rank-one relation exactly, which is the gate below.

    **The scale split.**  Only the product ‖û‖‖ŵ‖ is determined; the split is
    the per-plane scale freedom RoPE's own gauge already contains.  It is fixed
    to preserve the parents' query/key magnitude ratio in each plane, because
    the shared RMSNorm divides out one length per head and re-weights the planes
    against each other, so the ratio is the part of the split that survives.

    **And the phase must be fixed too, which the first build of this solve found
    out by losing.**  A rank-one factorisation has a per-plane unit-modulus
    gauge -- multiply ``û_p`` and ``ŵ_p`` by the same ``c``, ``|c| = 1``, and the
    relation is unchanged -- and left free it lands wherever the eigenvector
    routine puts it, which was the REFLECTION of the average-in-gauge merge at
    every site (post-norm cosine −0.9998).  The ridge below then spends its whole
    trust region walking a weight to the far side of its own prior, and the
    recovered head misses.  So ``anchor`` (the average-in-gauge rows) fixes the
    phase per plane, in closed form, to whatever aligns the solved rows with the
    merge this solve has to beat.  It changes no relation and no objective value;
    it decides only which of a circle of identical answers is handed to a solver
    that has a prior.
    """

    slots = len(q_parents)
    parents = len(k_parents)
    rows = k_parents[0].shape[0]
    half = head_dim // 2
    omega = torch.tensor(weights, dtype=torch.float64)
    omega = omega / omega.sum()

    k_planes = [as_planes(k_parents[p], head_dim) for p in range(parents)]
    q_planes = [
        [as_planes(q_parents[j][p], head_dim) for p in range(parents)]
        for j in range(slots)
    ]

    q_out = [torch.zeros(rows, half, dtype=torch.complex128) for _ in range(slots)]
    k_out = torch.zeros(rows, half, dtype=torch.complex128)
    residual = 0.0
    reference = 0.0

    for plane in range(half):
        # W2: the parents' key rows in this plane, as the basis ŵ must live in.
        W2 = torch.stack([k_planes[p][:, plane] for p in range(parents)], dim=1)
        S = W2.conj().T @ W2
        S = 0.5 * (S + S.conj().T)
        evals, evecs = torch.linalg.eigh(S)
        floor = float(evals.max()) * 1e-12
        root = evecs @ torch.diag(evals.clamp_min(floor).sqrt().to(evecs.dtype)) @ evecs.conj().T
        root_inv = (
            evecs
            @ torch.diag((1.0 / evals.clamp_min(floor).sqrt()).to(evecs.dtype))
            @ evecs.conj().T
        )
        T = torch.zeros(parents, parents, dtype=torch.complex128)
        for j in range(slots):
            U2 = torch.stack(
                [omega[p] * q_planes[j][p][:, plane] for p in range(parents)], dim=1
            )
            T = T + U2.conj().T @ U2
        M = root.conj().T @ T @ root
        M = 0.5 * (M + M.conj().T)
        _, vectors = torch.linalg.eigh(M)
        c = root_inv @ vectors[:, -1]
        w_hat = W2 @ c
        norm = float(w_hat.abs().pow(2).sum().sqrt())
        if norm <= 0.0:
            # A plane with no key energy at all carries no relation; the average
            # in gauge is as good as anything and the ridge below will say so.
            w_hat = W2[:, 0].clone()
            norm = float(w_hat.abs().pow(2).sum().sqrt()) or 1.0
        w_hat = w_hat / norm
        alpha = [
            (k_planes[p][:, plane].conj() * w_hat).sum() for p in range(parents)
        ]
        u_hats = []
        for j in range(slots):
            u = torch.zeros(rows, dtype=torch.complex128)
            for p in range(parents):
                u = u + omega[p] * q_planes[j][p][:, plane] * alpha[p]
            u_hats.append(u)
        # The scale split: keep each plane's parent q/k magnitude ratio.
        q_scale = sum(
            float(omega[p]) * float(q_planes[j][p][:, plane].abs().pow(2).sum().sqrt())
            for j in range(slots)
            for p in range(parents)
        ) / max(slots, 1)
        k_scale = sum(
            float(omega[p]) * float(k_planes[p][:, plane].abs().pow(2).sum().sqrt())
            for p in range(parents)
        )
        product = sum(float(u.abs().pow(2).sum().sqrt()) for u in u_hats) / max(slots, 1)
        ratio = (q_scale / k_scale) if k_scale > 0 else 1.0
        beta = (product / ratio) ** 0.5 if ratio > 0 and product > 0 else 1.0
        beta = max(beta, 1e-30)
        # the per-plane phase: aligned to the anchor, or left as solved.
        phase = torch.tensor(1.0 + 0.0j, dtype=torch.complex128)
        if anchor is not None:
            anchor_q, anchor_k = anchor
            overlap = (as_planes(anchor_k, head_dim)[:, plane].conj() * w_hat).sum()
            for j in range(slots):
                overlap = overlap + (
                    as_planes(anchor_q[j], head_dim)[:, plane].conj() * u_hats[j]
                ).sum()
            magnitude = float(overlap.abs())
            if magnitude > 0.0:
                phase = (overlap / magnitude).conj()
        k_out[:, plane] = w_hat * beta * phase
        for j in range(slots):
            q_out[j][:, plane] = u_hats[j] * phase / beta
        for j in range(slots):
            for p in range(parents):
                err, ref = rank_one_relation_error(
                    q_out[j][:, plane],
                    k_out[:, plane],
                    q_planes[j][p][:, plane],
                    k_planes[p][:, plane],
                )
                residual += float(omega[p]) * err
                reference += float(omega[p]) * ref

    diagnostics = {
        "relation_relL2": (residual / reference) ** 0.5 if reference > 0 else None,
        "planes": half,
        "parents": parents,
        "slots": slots,
    }
    return [from_planes(q) for q in q_out], from_planes(k_out), diagnostics


def relation_error_of(
    q_rows: list[torch.Tensor],
    k_row: torch.Tensor,
    q_parents: list[list[torch.Tensor]],
    k_parents: list[torch.Tensor],
    weights: list[float],
    head_dim: int,
) -> float:
    """The joint objective, evaluated at ANY candidate merged head.

    The point of the number is that the average-in-gauge merge and the joint
    solve can be scored on the same objective, on the same rows, so the claim
    that the solve is a solve is checkable rather than asserted.
    """

    omega = torch.tensor(weights, dtype=torch.float64)
    omega = omega / omega.sum()
    k_hat = as_planes(k_row, head_dim)
    k_ref = [as_planes(k, head_dim) for k in k_parents]
    residual = 0.0
    reference = 0.0
    for j, q_row in enumerate(q_rows):
        q_hat = as_planes(q_row, head_dim)
        for p in range(len(k_parents)):
            q_ref = as_planes(q_parents[j][p], head_dim)
            for plane in range(head_dim // 2):
                err, ref = rank_one_relation_error(
                    q_hat[:, plane], k_hat[:, plane], q_ref[:, plane], k_ref[p][:, plane]
                )
                residual += float(omega[p]) * err
                reference += float(omega[p]) * ref
    return (residual / reference) ** 0.5 if reference > 0 else float("nan")


def direction_anchored_solve(
    X: torch.Tensor,
    target_rows: torch.Tensor,
    gain: torch.Tensor,
    prior: torch.Tensor,
    eps: float,
    sweeps: int = 2,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Recover the weights whose POST-NORM rows are the solved target rows.

    The solve above produces rows as the score sees them -- after the per-head
    RMS normalisation and after the shared gain.  Weights act before both, and
    the normalisation destroys one degree of freedom per row: only the direction
    of a post-norm row is achievable, its length is not.  So the target handed
    to the ridge is ``ρ_i · d_i``, direction from the solve and radius from the
    weights themselves, and ``ρ`` is a two-step fixed point (the optimal radius
    for a given W is the projection of W's own row onto the direction).

    ``prior`` is wave 16's average-in-gauge weight, so THE LAW's trust region is
    measured against the merge this one has to beat, and a solve that cannot
    beat it is clamped back toward it rather than allowed to run away.
    """

    if float(gain.abs().min()) <= 0.0:
        raise SystemExit(
            "a zero entry in the shared per-head norm gain makes that "
            "coordinate's target direction undefined; the solve is refused"
        )
    normed = target_rows / gain
    scale = normed.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-300).rsqrt()
    direction = normed * scale
    width = float(direction.shape[1])
    # The radius is a length and lengths are not negative.  Left signed, the
    # fixed point is free to flip INDIVIDUAL rows onto the reflection of their
    # own target -- which is not a gauge: flipping one query row's sign flips
    # that row's scores against every key.  A row whose prior points the wrong
    # way is aimed at zero and re-aimed by the next sweep, and the count of them
    # rides in the guard.
    radius = ((X @ prior.T) * direction).sum(dim=1, keepdim=True) / width
    flipped = int((radius < 0).sum())
    radius = radius.clamp_min(0.0)
    history = []
    solved = prior
    guard: dict[str, Any] = {}
    for _ in range(max(1, sweeps)):
        solved, guard = law_abiding_solve(X, radius * direction, prior)
        achieved = X @ solved.T
        history.append(
            float(
                torch.nn.functional.cosine_similarity(achieved, direction, dim=1).mean()
            )
        )
        radius = ((achieved * direction).sum(dim=1, keepdim=True) / width).clamp_min(0.0)
    post = achieved * achieved.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt() * gain
    guard = dict(guard)
    guard["direction_cos_sweeps"] = history
    guard["rows_with_negative_prior_radius"] = flipped
    guard["rows"] = int(direction.shape[0])
    guard["post_norm_cos_vs_target"] = float(
        torch.nn.functional.cosine_similarity(post, target_rows, dim=1).mean()
    )
    return solved, guard


def stein_two_sided_eigh(
    A: torch.Tensor, B: torch.Tensor, E: torch.Tensor, lam: float
) -> torch.Tensor:
    """T7's registered primitive: ``A S B + λ S = E`` by two ``eigh``.

    Used as the *reference* for the conjugate-gradient solve below, which is the
    same normal equations with one Kronecker term per query head in the group.
    At group size 1 there is exactly one term and the two must agree.
    """

    a_vals, a_vecs = torch.linalg.eigh(A)
    b_vals, b_vecs = torch.linalg.eigh(B)
    inner = a_vecs.T @ E @ b_vecs
    denominator = a_vals.unsqueeze(1) * b_vals.unsqueeze(0) + lam
    return a_vecs @ (inner / denominator) @ b_vecs.T


def stein_factors(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The preconditioner's two ``eigh``, which do not depend on lambda.

    Wave 17 recomputed them inside every solve, which cost nothing when the
    ladder exited on its first rung.  ``--kv-lambda-split`` walks the whole
    ladder, so they are hoisted; the arithmetic is `stein_cg_solve`'s own, and
    passing the result back in reproduces it exactly.
    """

    diagonal = [key for key in left if key[0] == key[1]] or list(left)
    a_vals, a_vecs = torch.linalg.eigh(sum(left[key] for key in diagonal))
    b_vals, b_vecs = torch.linalg.eigh(sum(right[key] for key in diagonal))
    return a_vals, a_vecs, b_vals, b_vecs


def stein_objective(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
    rhs: torch.Tensor,
    reference_norm: float,
    S: torch.Tensor,
) -> float:
    """``||R - Ŷ(S)||²`` on the rows ``left``/``right``/``rhs`` were accumulated on.

    The same identity the operator is built from: the quadratic term is exactly
    the operator's own form at lambda zero, so a candidate can be scored on rows
    it was never solved on without recomputing one attention probability.
    """

    quadratic = 0.0
    for (h, hp), L in left.items():
        quadratic += float((S * (L @ S @ right[(h, hp)])).sum())
    return reference_norm - 2.0 * float((rhs * S).sum()) + quadratic


def stein_cg_solve(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
    rhs: torch.Tensor,
    prior: torch.Tensor,
    lam: float,
    iterations: int = 120,
    tolerance: float = 1e-8,
    factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """``Σ_{h,h'} L[h,h'] S R[h,h'] + λS = rhs + λ·prior``, preconditioned by T7.

    The operator is self-adjoint under the trace inner product -- the ``(h,h')``
    and ``(h',h)`` terms are each other's transpose -- and positive definite for
    ``λ > 0``, so CG applies and needs only matrix products, never the
    ``(64·hidden)²`` normal matrix the Kronecker form would ask for.

    **It also needs a preconditioner, and the measurement that says so is in the
    self-test.**  These factors are a second moment of hidden states and a Gram
    of ``out_proj`` columns; on a synthetic problem two orders of magnitude
    better conditioned than the real one, unpreconditioned CG had made no
    progress at all after 400 iterations and needed 20,000 to reach the closed
    form.  The preconditioner is `stein_two_sided_eigh` itself, applied to the
    group-summed factors: one two-``eigh`` solve of the single-term Stein
    equation, which is exactly T7's registered primitive.  At one term the
    preconditioner IS the operator and CG converges in a single step -- which is
    the self-test's check that the CG path and the closed form are the same
    solve.
    """

    def apply(S: torch.Tensor) -> torch.Tensor:
        out = lam * S
        for (h, hp), L in left.items():
            out = out + L @ S @ right[(h, hp)]
        return out

    a_vals, a_vecs, b_vals, b_vecs = (
        stein_factors(left, right) if factors is None else factors
    )
    floor = 1e-12 * max(float(a_vals.max()) * float(b_vals.max()), 1e-300)
    denominator = (
        a_vals.unsqueeze(1) * b_vals.unsqueeze(0) + lam
    ).clamp_min(floor)

    def precondition(R: torch.Tensor) -> torch.Tensor:
        return a_vecs @ ((a_vecs.T @ R @ b_vecs) / denominator) @ b_vecs.T

    B = rhs + lam * prior
    S = prior.clone()
    r = B - apply(S)
    z = precondition(r)
    p = z.clone()
    rz = float((r * z).sum())
    base = float((B * B).sum()) or 1.0
    used = 0
    residual = (float((r * r).sum()) / base) ** 0.5
    for used in range(1, iterations + 1):
        Ap = apply(p)
        curvature = float((p * Ap).sum())
        if curvature <= 0.0 or rz <= 0.0:
            break
        alpha = rz / curvature
        S = S + alpha * p
        r = r - alpha * Ap
        residual = (float((r * r).sum()) / base) ** 0.5
        if residual <= tolerance:
            break
        z = precondition(r)
        rz_new = float((r * z).sum())
        p = z + (rz_new / rz) * p
        rz = rz_new
    return S, {"cg_iterations": used, "cg_relative_residual": residual}


def law_abiding_stein(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
    rhs: torch.Tensor,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The λ-ladder trust region, on the two-sided solve."""

    scale_reference = max(
        float(torch.diagonal(R).mean().abs()) * float(torch.diagonal(L).mean().abs())
        for (L, R) in zip(left.values(), right.values())
    )
    reference = float(prior.norm())
    solved = None
    report: dict[str, Any] = {}
    for rel in LAMBDA_LADDER:
        lam = rel * scale_reference
        candidate, report = stein_cg_solve(left, right, rhs, prior, lam)
        # A zero prior anchors nothing, so THE LAW's ratio is undefined rather
        # than infinite; the smallest lambda on the ladder is the answer.
        scale = 0.0 if reference == 0.0 else float((candidate - prior).norm() / reference)
        solved = candidate
        if scale <= 1.0:
            report = dict(report)
            report.update({"update_scale": scale, "lambda_relative": rel})
            return candidate, report
    report = dict(report)
    report.update(
        {
            "update_scale": (
                None if reference == 0.0 else float((solved - prior).norm() / reference)
            ),
            "lambda_relative": LAMBDA_LADDER[-1],
            "law_refused": True,
        }
    )
    return prior, report


def stein_scale_reference(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
) -> float:
    """`law_abiding_stein`'s own lambda scale, lifted so the split can reuse it."""

    return max(
        float(torch.diagonal(R).mean().abs()) * float(torch.diagonal(L).mean().abs())
        for (L, R) in zip(left.values(), right.values())
    )


def law_abiding_stein_split(
    left: dict[tuple[int, int], torch.Tensor],
    right: dict[tuple[int, int], torch.Tensor],
    rhs: torch.Tensor,
    inner: tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor],
    check: tuple[dict[tuple[int, int], torch.Tensor], torch.Tensor, float],
    prior: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """`law_abiding_stein` with lambda chosen on the internal fit-side split.

    ``left`` is the ``out_proj`` Gram, which does not depend on which rows were
    accumulated, so only the second factor and the right-hand side are split.
    Each rung is solved on the inner rows and scored, in closed form from the
    check rows' own factors, on rows it never saw.  The rungs are then walked in
    check order and the first one THE LAW admits -- measured, as always, on the
    weight that would actually be installed, solved on ALL the fit rows -- is the
    one returned.  Wave 17's pick (ladder order, first admissible) is solved for
    as well, and its check objective rides in the report beside the chosen one's.
    """

    right_in, rhs_in = inner
    right_ck, rhs_ck, rnorm_ck = check
    reference = float(prior.norm())
    scale_full = stein_scale_reference(left, right)
    scale_inner = stein_scale_reference(left, right_in)
    factors_full = stein_factors(left, right)
    factors_inner = stein_factors(left, right_in)

    rungs: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        candidate_in, _ = stein_cg_solve(
            left, right_in, rhs_in, prior, rel * scale_inner, factors=factors_inner
        )
        rungs.append(
            {
                "lambda_relative": rel,
                "check_sse": stein_objective(left, right_ck, rhs_ck, rnorm_ck, candidate_in),
            }
        )

    solved_cache: dict[float, tuple[torch.Tensor, dict[str, Any], float]] = {}

    def full(rel: float) -> tuple[torch.Tensor, dict[str, Any], float]:
        if rel not in solved_cache:
            candidate, report = stein_cg_solve(
                left, right, rhs, prior, rel * scale_full, factors=factors_full
            )
            scale = (
                0.0 if reference == 0.0 else float((candidate - prior).norm() / reference)
            )
            solved_cache[rel] = (candidate, report, scale)
        return solved_cache[rel]

    law_pick = None
    for rel in LAMBDA_LADDER:
        if full(rel)[2] <= 1.0:
            law_pick = rel
            break
    chosen = None
    for row in sorted(rungs, key=lambda row: row["check_sse"]):
        candidate, report, scale = full(row["lambda_relative"])
        row["update_scale"] = scale
        if scale <= 1.0:
            chosen = (row, candidate, report, scale)
            break
    # Every rung whose installable weight was actually solved for reports its
    # update scale, whichever walk solved it: a ladder row that says `null` is a
    # rung this call never had to price, not one THE LAW was silent about.
    for row in rungs:
        cached = solved_cache.get(row["lambda_relative"])
        row["update_scale"] = None if cached is None else cached[2]
    if chosen is None:
        candidate, report, scale = full(LAMBDA_LADDER[-1])
        report = dict(report)
        report.update(
            {
                "update_scale": None if reference == 0.0 else scale,
                "lambda_relative": LAMBDA_LADDER[-1],
                "law_refused": True,
                "lambda_selected_on": "internal fit split",
                "ladder": rungs,
            }
        )
        return prior, report
    row, candidate, report, scale = chosen
    by_lambda = {r["lambda_relative"]: r for r in rungs}
    report = dict(report)
    report.update(
        {
            "update_scale": scale,
            "lambda_relative": row["lambda_relative"],
            "lambda_selected_on": "internal fit split",
            "lambda_law_would_have_picked": law_pick,
            "check_sse_chosen": row["check_sse"],
            "check_sse_law_pick": (
                None if law_pick is None else by_lambda[law_pick]["check_sse"]
            ),
            "ladder": rungs,
        }
    )
    return candidate, report


def attention_probabilities(
    X: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    inv_freq: torch.Tensor,
    scaling: float,
    eps: float,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """One batch's attention probabilities, recomputed in fp64.

    Recomputed rather than hooked because the probabilities are not a module
    output on this architecture, and **gated** rather than trusted: the context
    they predict is compared against the context the installed module actually
    produced before anything is solved against them.
    """

    positions = torch.arange(X.shape[0], dtype=torch.float64)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    emb = torch.cat([freqs, freqs], dim=1)
    cos, sin = emb.cos(), emb.sin()
    half = head_dim // 2

    def rotate(rows: torch.Tensor) -> torch.Tensor:
        flipped = torch.cat([-rows[:, half:], rows[:, :half]], dim=1)
        return rows * cos + flipped * sin

    def head_rows(weight: torch.Tensor, index: int, gain: torch.Tensor) -> torch.Tensor:
        rows = X @ weight[index * head_dim : (index + 1) * head_dim].T
        rows = rows * rows.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt() * gain
        return rotate(rows)

    group = heads // kv_heads
    keys = [head_rows(wk, g, gk) for g in range(kv_heads)]
    mask = torch.full((X.shape[0], X.shape[0]), float("-inf"), dtype=torch.float64)
    mask = torch.triu(mask, diagonal=1)
    out = []
    for h in range(heads):
        q = head_rows(wq, h, gq)
        scores = (q @ keys[h // group].T) * scaling + mask
        out.append(torch.softmax(scores, dim=-1))
    return torch.stack(out, dim=0)


# ----------------------------------------------------- the head-COUNT axis
def head_count_axis(
    harness: Harness,
    position: int,
    parent_layer: int,
    keep_groups: int,
    *,
    trajectory_target: tuple[torch.Tensor, torch.Tensor] | None,
    solve_mode: str = "joint",
    als_sweeps: int = 1,
    identity_gate: bool = False,
    lambda_split: bool = False,
    gate_probe: Path | None = None,
) -> dict[str, Any]:
    """Merge whole attention heads down to ``keep_groups`` KV groups.

    The granularity is the **KV group**, not the head, because that is the only
    granularity a stock config can express: `Lfm2Attention` computes
    ``num_key_value_groups = num_attention_heads // num_key_value_heads`` and
    `repeat_kv` then repeats each KV head exactly that many times, so
    ``num_attention_heads`` must be a multiple of ``num_key_value_heads`` or the
    score matmul is handed two different head counts.  Merging two parent groups
    into one therefore merges one KV head pair *and* the two query-head pairs
    that read it, and the surviving group keeps the parent's ``head_dim`` 64 --
    which is the whole reason this axis exists beside `head_dim_axis`: every
    surviving head keeps the parent's own RoPE ladder and none of `NOTES-MX-S`
    §3's plane arithmetic applies.

    Which pairs merge is a **per-layer** decision (only the count is a config
    field) and is ordered by a redundancy census on the post-norm rows; the
    census orders, it gates nothing.

    ``solve_mode`` picks what happens once the pairs are chosen:

    ``average``
        Wave 16's merge, unchanged and reproducible: q/k averaged in the exact
        per-plane sign gauge with evidence-energy weights, v contracted by an
        evidence-whitened low-rank projection, ``out_proj`` re-solved.  It is
        the baseline the joint solve has to beat.
    ``joint``
        ADDENDUM 14 §14.3.  q and k are solved **together** against the relation
        both parents compute (per plane, closed form, shared key across the
        group), the weights are recovered by a direction-anchored ridge with the
        average-in-gauge merge as prior, and v and ``out_proj`` are then solved
        as the linear pair they are with the attention probabilities fixed.
    """

    layer = harness.layers()[position]
    attn = layer.self_attn
    config = harness.model.config
    heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(attn.head_dim)
    group_size = heads // kv_heads
    merges = kv_heads - keep_groups
    if merges < 0 or (merges == 0 and not identity_gate):
        raise SystemExit(f"--kv-groups {keep_groups} removes nothing from {kv_heads}")
    pristine = copy.deepcopy(attn)

    captured = harness.capture({"attn": attn}, pristine={"attn": pristine})
    X, X_h = captured["attn"]["fit"], captured["attn"]["hld"]
    if trajectory_target is not None:
        target, target_h = trajectory_target
    else:
        target, target_h = captured["attn"]["local_fit"], captured["attn"]["local_hld"]

    wq = attn.q_proj.weight.data.to(torch.float64).cpu()
    wk = attn.k_proj.weight.data.to(torch.float64).cpu()
    wv = attn.v_proj.weight.data.to(torch.float64).cpu()
    wo = attn.out_proj.weight.data.to(torch.float64).cpu()
    gq = attn.q_layernorm.weight.data.to(torch.float64).cpu()
    gk = attn.k_layernorm.weight.data.to(torch.float64).cpu()
    eps = float(config.norm_eps)

    def post_norm(rows: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        """One head's rows exactly as the score sees them: RMS-normed, gained."""

        scale = rows.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
        return rows * scale * gain

    q_rows = [
        post_norm(X @ wq[h * head_dim : (h + 1) * head_dim].T, gq) for h in range(heads)
    ]
    k_rows = [
        post_norm(X @ wk[g * head_dim : (g + 1) * head_dim].T, gk) for g in range(kv_heads)
    ]
    v_rows = [X @ wv[g * head_dim : (g + 1) * head_dim].T for g in range(kv_heads)]
    energy = [float(k.pow(2).sum()) for k in k_rows]

    # ---- the redundancy census: every pair, both gauges, ordered.
    pairs: list[dict[str, Any]] = []
    for a in range(kv_heads):
        for b in range(a + 1, kv_heads):
            s = sign_gauge(k_rows[a], k_rows[b], head_dim)
            key_cos = float(
                torch.nn.functional.cosine_similarity(k_rows[a], k_rows[b] * s, dim=1).mean()
            )
            query_cos = float(
                torch.tensor(
                    [
                        torch.nn.functional.cosine_similarity(
                            q_rows[a * group_size + j], q_rows[b * group_size + j] * s, dim=1
                        ).mean()
                        for j in range(group_size)
                    ]
                ).mean()
            )
            pairs.append(
                {
                    "groups": [a, b],
                    "key_cos_sign_gauge": key_cos,
                    "query_cos_sign_gauge": query_cos,
                    "key_cos_rotation_gauge": rotation_gauge_cos(
                        k_rows[a], k_rows[b], head_dim
                    ),
                    "redundancy": 0.5 * (key_cos + query_cos),
                }
            )
    ordered = sorted(pairs, key=lambda row: -row["redundancy"])
    chosen: list[tuple[int, int]] = []
    used: set[int] = set()
    for row in ordered:
        a, b = row["groups"]
        if a in used or b in used or len(chosen) == merges:
            continue
        chosen.append((a, b))
        used.update((a, b))
    if len(chosen) != merges:
        raise SystemExit(
            f"L{parent_layer:02d}: {merges} disjoint merges requested, {len(chosen)} "
            f"available from {kv_heads} groups"
        )

    # ---- the new group list, deterministic: merged pairs keyed by their anchor.
    clusters: list[list[int]] = []
    paired = {a: b for a, b in chosen}
    paired.update({b: a for a, b in chosen})
    seen: set[int] = set()
    for g in range(kv_heads):
        if g in seen:
            continue
        if g in paired:
            partner = paired[g]
            seen.update((g, partner))
            # The anchor is the higher-energy group; the gauge carries the other
            # onto it, so the merged frame is the one more of the evidence is in.
            clusters.append([g, partner] if energy[g] >= energy[partner] else [partner, g])
        else:
            seen.add(g)
            clusters.append([g])

    new_k: list[torch.Tensor] = []
    new_v: list[torch.Tensor] = []
    new_q: list[torch.Tensor] = []
    prior_blocks: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    merged_slots: list[dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            g = cluster[0]
            new_k.append(wk[g * head_dim : (g + 1) * head_dim])
            new_v.append(wv[g * head_dim : (g + 1) * head_dim])
            for j in range(group_size):
                h = g * group_size + j
                new_q.append(wq[h * head_dim : (h + 1) * head_dim])
                prior_blocks.append(wo[:, h * head_dim : (h + 1) * head_dim])
            records.append({"cluster": cluster, "merged": False})
            continue
        a, b = cluster
        s = sign_gauge(k_rows[a], k_rows[b], head_dim)
        wa, wb = energy[a], energy[b]
        total = wa + wb
        gauge = s.unsqueeze(1)
        # Wave 16's merge, always computed: it is the joint solve's prior, and
        # when `solve_mode` is `average` it is the merge.
        average_k = (
            wa * wk[a * head_dim : (a + 1) * head_dim]
            + wb * gauge * wk[b * head_dim : (b + 1) * head_dim]
        ) / total
        average_q = []
        for j in range(group_size):
            ha, hb = a * group_size + j, b * group_size + j
            average_q.append(
                (
                    wa * wq[ha * head_dim : (ha + 1) * head_dim]
                    + wb * gauge * wq[hb * head_dim : (hb + 1) * head_dim]
                )
                / total
            )
        merged_k, merged_q = average_k, average_q
        record: dict[str, Any] = {
            "cluster": cluster,
            "merged": True,
            "solve_mode": solve_mode,
            "key_cos_sign_gauge": float(
                torch.nn.functional.cosine_similarity(k_rows[a], k_rows[b] * s, dim=1).mean()
            ),
            "signs_flipped": int((s[: head_dim // 2] < 0).sum()),
            "energy_weights": [wa / total, wb / total],
        }
        # The objective, evaluated at the average-in-gauge merge, always: the
        # joint solve's number means nothing without the baseline's beside it.
        parents_k = [k_rows[a], k_rows[b] * s]
        parents_q = [
            [q_rows[a * group_size + j], q_rows[b * group_size + j] * s]
            for j in range(group_size)
        ]
        record["relation_relL2_average"] = relation_error_of(
            [post_norm(X @ w.T, gq) for w in average_q],
            post_norm(X @ average_k.T, gk),
            parents_q,
            parents_k,
            [wa, wb],
            head_dim,
        )
        if solve_mode == "joint":
            target_q, target_k, diagnostics = joint_relation_solve(
                parents_q,
                parents_k,
                [wa, wb],
                head_dim,
                anchor=(
                    [post_norm(X @ w.T, gq) for w in average_q],
                    post_norm(X @ average_k.T, gk),
                ),
            )
            record["relation_relL2_joint_target"] = diagnostics["relation_relL2"]
            solved_k, guard_k = direction_anchored_solve(
                X, target_k, gk, average_k, eps
            )
            solved_q = []
            guards_q = []
            for j in range(group_size):
                weight, guard = direction_anchored_solve(
                    X, target_q[j], gq, average_q[j], eps
                )
                solved_q.append(weight)
                guards_q.append(guard)
            record["relation_relL2_joint"] = relation_error_of(
                [post_norm(X @ w.T, gq) for w in solved_q],
                post_norm(X @ solved_k.T, gk),
                parents_q,
                parents_k,
                [wa, wb],
                head_dim,
            )
            record["key_recovery"] = guard_k
            record["query_recovery"] = guards_q
            # The target beats the average by construction -- it is the optimum
            # of the objective the average is one point of.  The RECOVERED head
            # need not: the target rows are a rank-one combination of two
            # post-normed row blocks and no linear map followed by an RMS norm
            # has to be able to produce them.  So the solve keeps its own answer
            # only when the answer that is actually installable beats the merge
            # it replaces, on the same objective and the same fit rows.  Both
            # numbers are in the receipt either way.
            record["joint_installed"] = bool(
                record["relation_relL2_joint"] < record["relation_relL2_average"]
            )
            if record["joint_installed"]:
                merged_k, merged_q = solved_k, solved_q
        new_k.append(merged_k)
        for weight in merged_q:
            new_q.append(weight)
        # v carries no gauge (no RoPE, no norm), so it is merged by the same
        # evidence-weighted low-rank contraction `head_dim_axis` uses -- the best
        # 64-dimensional subspace of the two heads' 128-dimensional joint value
        # space, whitened by the evidence and weighted by out_proj's use of it.
        C = torch.cat([v_rows[a], v_rows[b]], dim=1)
        G = C.T @ C / C.shape[0]
        ev, evec = torch.linalg.eigh(G)
        Rt = evec @ torch.diag(ev.clamp_min(0).sqrt()) @ evec.T
        Rt_inv = evec @ torch.diag(1.0 / ev.clamp_min(1e-12).sqrt()) @ evec.T
        blocks = []
        for j in range(group_size):
            ha, hb = a * group_size + j, b * group_size + j
            left = wo[:, ha * head_dim : (ha + 1) * head_dim]
            right = wo[:, hb * head_dim : (hb + 1) * head_dim]
            blocks.append(torch.cat([left, right], dim=1))
        stack = torch.cat(blocks, dim=0)
        _, S, Vh = torch.linalg.svd(stack @ Rt, full_matrices=False)
        P = Vh[:head_dim] @ Rt_inv.T
        P = torch.linalg.qr(P.T)[0].T
        new_v.append(
            P @ torch.cat(
                [wv[a * head_dim : (a + 1) * head_dim], wv[b * head_dim : (b + 1) * head_dim]],
                dim=0,
            )
        )
        for block in blocks:
            prior_blocks.append(block @ P.T)
        record["v_retained_energy"] = float((S[:head_dim] ** 2).sum() / (S**2).sum())
        records.append(record)
        merged_slots.append(
            {
                "cluster": cluster,
                "heads": [
                    len(new_q) - group_size + j for j in range(group_size)
                ],
                "group_index": len(new_k) - 1,
            }
        )

    keep_heads = keep_groups * group_size
    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    for name, rows, count in (
        ("q_proj", new_q, keep_heads),
        ("k_proj", new_k, keep_groups),
        ("v_proj", new_v, keep_groups),
    ):
        stacked = torch.cat(rows, dim=0)
        linear = nn.Linear(wq.shape[1], count * head_dim, bias=False, dtype=dtype, device=device)
        linear.weight.data.copy_(stacked.to(device=device, dtype=dtype))
        setattr(attn, name, linear)
    prior = torch.cat(prior_blocks, dim=1)
    new_o = nn.Linear(keep_heads * head_dim, wo.shape[0], bias=False, dtype=dtype, device=device)
    new_o.weight.data.copy_(prior.to(device=device, dtype=dtype))
    attn.out_proj = new_o
    attn.num_key_value_groups = group_size

    gate: dict[str, Any] = {}
    if identity_gate:
        # ADDENDUM 14 §14.3's trivial-grouping gate: nothing merged, so nothing
        # may have moved.  Checked bit-for-bit against the parent's own tensors.
        gate["q_max_abs_delta"] = float(
            (attn.q_proj.weight.data.to(torch.float64).cpu() - wq).abs().max()
        )
        gate["k_max_abs_delta"] = float(
            (attn.k_proj.weight.data.to(torch.float64).cpu() - wk).abs().max()
        )
        gate["v_max_abs_delta"] = float(
            (attn.v_proj.weight.data.to(torch.float64).cpu() - wv).abs().max()
        )
        gate["out_prior_max_abs_delta"] = float((prior - wo).abs().max())
        if max(gate.values()) != 0.0:
            raise SystemExit(
                f"L{parent_layer:02d}: trivial grouping moved a tensor "
                f"({gate}); the merge path is not an identity at zero merges"
            )

    context = harness.capture({"o": attn.out_proj})
    C_fit, C_hld = context["o"]["fit"], context["o"]["hld"]
    prior_scores = write_scores(C_hld @ prior.T, target_h)
    solved, guard = law_abiding_solve(C_fit, target, prior)
    new_o.weight.data.copy_(solved.to(device=device, dtype=dtype))
    scores = write_scores(C_hld @ solved.T, target_h)
    baseline_fit = float((C_fit @ solved.T - target).pow(2).sum())

    # ---- the value/out_proj pair, solved as the linear pair it is.
    value_report: dict[str, Any] = {"sweeps_requested": int(als_sweeps)}
    if als_sweeps > 0 and merged_slots:
        try:
            value_report.update(
                joint_value_sweeps(
                    harness,
                    attn,
                    parent_layer,
                    merged_slots=merged_slots,
                    keep_heads=keep_heads,
                    keep_groups=keep_groups,
                    head_dim=head_dim,
                    gq=gq,
                    gk=gk,
                    eps=eps,
                    target=target,
                    target_h=target_h,
                    prior_out=solved,
                    baseline_fit=baseline_fit,
                    sweeps=int(als_sweeps),
                    lambda_split=lambda_split,
                    gate_probe=gate_probe,
                )
            )
        except RuntimeError as error:  # pragma: no cover - measured, not assumed
            value_report["failed"] = str(error)
        if value_report.get("installed"):
            context = harness.capture({"o": attn.out_proj})
            C_hld = context["o"]["hld"]
            scores = write_scores(
                C_hld @ attn.out_proj.weight.data.to(torch.float64).cpu().T, target_h
            )
            guard = value_report.pop("out_guard", guard)

    return {
        "parent_layer": parent_layer,
        "position": position,
        "keep_groups": keep_groups,
        "keep_heads": keep_heads,
        "head_dim": head_dim,
        "solve_mode": solve_mode,
        "identity_gate": gate or None,
        "value_solve": value_report,
        "census": {
            "pairs": ordered,
            "chosen_merges": [list(pair) for pair in chosen],
            "clusters": clusters,
            "best_pair_redundancy": ordered[0]["redundancy"] if ordered else None,
            "worst_pair_redundancy": ordered[-1]["redundancy"] if ordered else None,
            "rotation_gauge_gain_median": float(
                torch.tensor(
                    [r["key_cos_rotation_gauge"] - r["key_cos_sign_gauge"] for r in ordered]
                ).median()
            )
            if ordered
            else None,
            "merged": records,
        },
        "prior_only_cos": prior_scores["cos"],
        **guard,
        **scores,
    }


def attention_gate_probe(
    harness: Harness,
    attn,
    parent_layer: int,
    *,
    batched: dict[str, list[tuple[str, torch.Tensor]]],
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    eps: float,
    scaling: float,
    keep_heads: int,
    keep_groups: int,
    head_dim: int,
    inv_freq: torch.Tensor,
    fp64_gate: float,
    outdir: Path,
) -> dict[str, Any]:
    """`NOTES-MX-U` §7's falsifying probe for the refusal nothing could attribute.

    Wave 17 measured the attention-recompute gate at 0.32-0.77 against a 5e-2
    tolerance at all six sites of the 13+1 composition, where block 13 alone
    reads 9.3e-3, and named one hypothesis -- that the composed child's attention
    is near-degenerate and bf16 rounding in the module decides an argmax the fp64
    recomputation decides differently -- without running anything that could
    falsify it.  This runs it, three ways, and installs nothing:

    1. **The module's own probabilities.**  This architecture's eager attention
       returns its ``attn_weights``, so the recomputation can be compared
       against the very tensor it is a model of, rather than against a context
       that has been through ``v`` and a concatenation.  If the disagreement
       lives in the probabilities, ``gate_module_probabilities`` closes and
       ``probabilities_relL2`` is large.  If it lives anywhere else -- the
       capture alignment, ``v``, the head-to-group map -- the reverse.
    2. **Peakedness and the top-two margin.**  The hypothesis is a statement
       about near-ties: a row whose two best keys are within bf16's resolution
       of each other is a row where the two arithmetics can disagree completely.
       Both are measured, on the recomputation and on the module's own rows, as
       distributions rather than as means alone.
    3. **The logit regression.**  If the probabilities disagree, they disagree
       either by a SCALE -- the module's logits systematically shrunk, which is a
       normalisation or a temperature and not a rounding -- or by NOISE at
       unchanged scale, which is arithmetic.  Softmax fixes logits only up to an
       additive constant per row, and that is exactly what a per-row centred
       regression of ``log p_module`` on ``log p_fp64`` is blind to, so the slope
       and the residual separate the two stories cleanly.  Slope one with a large
       residual is rounding; slope well below one is not.
    4. **The same gate with the recomputation's inputs rounded to bf16.**  This
       leg is kept and is REPORTED AS THE NULL IT IS: the captured input and the
       stored weights are already bf16, so rounding them changes nothing and the
       number reproduces leg zero exactly.  It is retained because it is the leg
       `NOTES-MX-U` §7 asked for, and the honest reading of it is that it cannot
       test what it was asked to test -- the module's inexactness is in its
       arithmetic, not in its inputs, and the arithmetic is not reachable from
       here.  The stream and row magnitudes beside it are what a rounding story
       has to explain.
    """

    weights: list[torch.Tensor] = []
    inputs: list[torch.Tensor] = []

    def pre_hook(module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        inputs.append(hidden.detach().reshape(-1, hidden.shape[-1]).to("cpu", torch.float64))
        return None

    def post_hook(module, args, kwargs, output):
        if not isinstance(output, tuple) or output[1] is None:
            raise SystemExit(
                f"L{parent_layer:02d}: the attention module returned no weights; "
                "the probe compares the recomputation against the module's own "
                "probabilities and there is nothing to compare against"
            )
        weights.append(output[1].detach()[0].to("cpu", torch.float64))
        return None

    handles = [
        attn.register_forward_pre_hook(pre_hook, with_kwargs=True),
        attn.register_forward_hook(post_hook, with_kwargs=True),
    ]
    try:
        for _, ids in harness.batches:
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
            harness.forward_calls += 1
    finally:
        for handle in handles:
            handle.remove()
    if len(weights) != len(batched["attn"]):
        raise SystemExit(
            f"L{parent_layer:02d}: the probe pass saw {len(weights)} calls and the "
            f"capture {len(batched['attn'])}; they are not the same pass"
        )

    group_size = keep_heads // keep_groups
    drift = max(
        float((X_probe - X_b).abs().max())
        for X_probe, (_, X_b) in zip(inputs, batched["attn"])
    )
    if drift != 0.0:
        raise SystemExit(
            f"L{parent_layer:02d}: the probe pass re-derived a different input "
            f"(max abs delta {drift:.3e}); the two passes are not comparable"
        )

    def peak(A: torch.Tensor) -> dict[str, float]:
        top = A.topk(2, dim=-1).values
        margin = (top[..., 0] - top[..., 1]).flatten()
        entropy = -(A.clamp_min(1e-300) * A.clamp_min(1e-300).log()).sum(-1).flatten()
        quantile = lambda t, q: float(  # noqa: E731 - one line, one meaning
            torch.quantile(t, torch.tensor(q, dtype=t.dtype))
        )
        return {
            "max_probability_median": quantile(top[..., 0].flatten(), 0.5),
            "top2_margin_median": quantile(margin, 0.5),
            "top2_margin_p05": quantile(margin, 0.05),
            "top2_margin_below_1e-2": float((margin < 1e-2).to(torch.float64).mean()),
            "entropy_median": quantile(entropy, 0.5),
        }

    def centred_logits(P: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """``log p`` with each query row's mean removed -- the part softmax keeps."""

        rows = P.clamp_min(1e-12).log()
        counts = keep.sum(-1, keepdim=True).clamp_min(1)
        means = (rows * keep).sum(-1, keepdim=True) / counts
        return (rows - means) * keep

    residual64 = reference = residual_module = residual_bf16 = 0.0
    probability_residual = probability_reference = 0.0
    argmax_agree = argmax_rows = 0.0
    slope_numerator = slope_denominator = 0.0
    logit_residual = logit_rows = 0.0
    stream_norm = query_norm = key_norm = 0.0
    stream_rows = 0.0
    fp64_rows: list[torch.Tensor] = []
    module_rows: list[torch.Tensor] = []
    for (split, X_b), (_, C_b), A_module in zip(batched["attn"], batched["o"], weights):
        A = attention_probabilities(
            X_b, wq, wk, gq, gk, inv_freq, scaling, eps, keep_heads, keep_groups, head_dim
        )
        A_bf16 = attention_probabilities(
            X_b.to(torch.bfloat16).to(torch.float64),
            wq.to(torch.bfloat16).to(torch.float64),
            wk.to(torch.bfloat16).to(torch.float64),
            gq.to(torch.bfloat16).to(torch.float64),
            gk.to(torch.bfloat16).to(torch.float64),
            inv_freq,
            scaling,
            eps,
            keep_heads,
            keep_groups,
            head_dim,
        )

        def context(P: torch.Tensor) -> torch.Tensor:
            return torch.cat(
                [
                    (P[h] @ X_b)
                    @ wv[(h // group_size) * head_dim : (h // group_size + 1) * head_dim].T
                    for h in range(keep_heads)
                ],
                dim=1,
            )

        residual64 += float((context(A) - C_b).pow(2).sum())
        residual_module += float((context(A_module) - C_b).pow(2).sum())
        residual_bf16 += float((context(A_bf16) - C_b).pow(2).sum())
        reference += float(C_b.pow(2).sum())
        probability_residual += float((A - A_module).pow(2).sum())
        probability_reference += float(A_module.pow(2).sum())
        agree = A.argmax(-1) == A_module.argmax(-1)
        argmax_agree += float(agree.to(torch.float64).sum())
        argmax_rows += float(agree.numel())

        # Where BOTH sides put representable mass, regress the module's centred
        # logits on the recomputation's.  A scale answers a different question
        # from a spread and only this separates them.
        keep = ((A > 1e-6) & (A_module > 1e-6)).to(torch.float64)
        b = centred_logits(A, keep)
        a = centred_logits(A_module, keep)
        slope_numerator += float((a * b).sum())
        slope_denominator += float((b * b).sum())
        logit_residual += float((a - b).pow(2).sum())
        logit_rows += float(keep.sum())

        stream_norm += float(X_b.pow(2).sum())
        stream_rows += float(X_b.shape[0])
        for h in range(keep_heads):
            query_norm += float(
                (X_b @ wq[h * head_dim : (h + 1) * head_dim].T).pow(2).sum()
            )
        for g in range(keep_groups):
            key_norm += float(
                (X_b @ wk[g * head_dim : (g + 1) * head_dim].T).pow(2).sum()
            )
        if split == "fit" and len(fp64_rows) < 2:
            fp64_rows.append(A)
            module_rows.append(A_module)

    row = {
        "parent_layer": parent_layer,
        "gate_fp64": fp64_gate,
        "gate_recomputed_fp64": (residual64 / reference) ** 0.5,
        "gate_module_probabilities": (residual_module / reference) ** 0.5,
        "gate_bf16_inputs": (residual_bf16 / reference) ** 0.5,
        "probabilities_relL2": (probability_residual / probability_reference) ** 0.5,
        "argmax_agreement": argmax_agree / argmax_rows,
        "logit_slope_module_on_fp64": slope_numerator / slope_denominator,
        "logit_residual_rms": (logit_residual / logit_rows) ** 0.5,
        "logit_scale_fp64_rms": (slope_denominator / logit_rows) ** 0.5,
        "peakedness_fp64": peak(torch.cat(fp64_rows, dim=1)),
        "peakedness_module": peak(torch.cat(module_rows, dim=1)),
        "context_norm": reference**0.5,
        "stream_row_rms": (stream_norm / stream_rows) ** 0.5,
        "prenorm_query_row_rms": (query_norm / (stream_rows * keep_heads)) ** 0.5,
        "prenorm_key_row_rms": (key_norm / (stream_rows * keep_groups)) ** 0.5,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"gate-probe-L{parent_layer:02d}.json").write_text(
        json.dumps(row, indent=1)
    )
    log(
        f"    gate probe L{parent_layer:02d}: fp64 {row['gate_recomputed_fp64']:.3e} "
        f"module-A {row['gate_module_probabilities']:.3e} bf16-in "
        f"{row['gate_bf16_inputs']:.3e} | A relL2 {row['probabilities_relL2']:.3e} "
        f"argmax agree {row['argmax_agreement']:.4f} | logit slope "
        f"{row['logit_slope_module_on_fp64']:.4f} residual rms "
        f"{row['logit_residual_rms']:.4f} of {row['logit_scale_fp64_rms']:.4f} | "
        f"stream rms {row['stream_row_rms']:.3f} q rms "
        f"{row['prenorm_query_row_rms']:.3f}"
    )
    return row


def joint_value_sweeps(
    harness: Harness,
    attn,
    parent_layer: int,
    *,
    merged_slots: list[dict[str, Any]],
    keep_heads: int,
    keep_groups: int,
    head_dim: int,
    gq: torch.Tensor,
    gk: torch.Tensor,
    eps: float,
    target: torch.Tensor,
    target_h: torch.Tensor,
    prior_out: torch.Tensor,
    baseline_fit: float,
    sweeps: int,
    lambda_split: bool = False,
    gate_probe: Path | None = None,
) -> dict[str, Any]:
    """Alternate ``v | out_proj`` with the attention probabilities held fixed.

    Once q and k are installed the probabilities are decided, and the block's
    output is **linear in v and in out_proj both**:

        out(i) = Σ_h  Wo_h · Wv_{g(h)} · z_h(i),   z_h(i) = (A_h X)(i)

    which is T7's two-sided Stein form with one Kronecker term per query head in
    the group.  The alternation is therefore two exact least-squares steps, both
    prior-anchored and both under THE LAW's trust region -- neither is a gradient
    step and neither is a heuristic.

    ``A_h`` is recomputed rather than hooked, because attention probabilities are
    not a module output on this architecture.  So it is **gated**: the context
    the recomputation predicts is compared against the context the installed
    module actually produced, and the whole value path is refused -- leaving wave
    16's contraction standing, and saying so in the receipt -- if the two
    disagree by more than the bf16 floor this harness runs at.

    Wave 18 adds two things and moves nothing else.  ``lambda_split`` selects
    both sides' regularisation on an internal split of the FIT batches rather
    than on THE LAW's norm alone (`NOTES-MX-U` §7's named fix); ``gate_probe``
    writes the falsifying measurement for the refusal that wave 17 could not
    attribute, and installs nothing.
    """

    device = attn.q_proj.weight.device
    dtype = attn.q_proj.weight.dtype
    hidden = int(harness.model.config.hidden_size)
    scaling = head_dim**-0.5
    rotary = harness.model.model.rotary_emb
    inv_freq = rotary.inv_freq.detach().to("cpu", torch.float64)
    group_size = keep_heads // keep_groups

    batched = harness.capture_batches({"attn": attn, "o": attn.out_proj})
    wq = attn.q_proj.weight.data.to(torch.float64).cpu()
    wk = attn.k_proj.weight.data.to(torch.float64).cpu()
    wv = attn.v_proj.weight.data.to(torch.float64).cpu()
    wv_initial = wv.clone()
    report: dict[str, Any] = {"sweeps_run": 0, "installed": False}

    fit_rows = sum(X.shape[0] for split, X in batched["attn"] if split == "fit")
    if fit_rows != target.shape[0]:
        raise SystemExit(
            f"L{parent_layer:02d}: the batched capture has {fit_rows} fit rows and "
            f"the flat one has {target.shape[0]}; the two are not the same evidence"
        )

    # The internal FIT-side split: every third fit batch, in fit order, is
    # scored on and never solved on.  It is a position rule, so it is the same
    # six batches at every site of every child.
    fit_positions = [i for i, (split, _) in enumerate(batched["attn"]) if split == "fit"]
    check_batches = {
        index
        for order, index in enumerate(fit_positions)
        if order % INTERNAL_CHECK_STRIDE == INTERNAL_CHECK_OFFSET
    }
    check_rows = torch.zeros(fit_rows, dtype=torch.bool)
    offset = 0
    for index in fit_positions:
        width = batched["attn"][index][1].shape[0]
        if index in check_batches:
            check_rows[offset : offset + width] = True
        offset += width
    if lambda_split:
        report["lambda_split"] = {
            "rule": (
                f"every {INTERNAL_CHECK_STRIDE}rd fit batch from the "
                f"{INTERNAL_CHECK_OFFSET + 1}rd, in fit order"
            ),
            "fit_batches": len(fit_positions),
            "check_batches": len(check_batches),
            "check_rows": int(check_rows.sum()),
            "inner_rows": int((~check_rows).sum()),
        }

    def probabilities(X_b: torch.Tensor) -> torch.Tensor:
        return attention_probabilities(
            X_b, wq, wk, gq, gk, inv_freq, scaling, eps,
            keep_heads, keep_groups, head_dim,
        )

    # ---- the gate, on every row the block will be solved against.
    residual_norm = 0.0
    reference_norm = 0.0
    for (split, X_b), (_, C_b) in zip(batched["attn"], batched["o"]):
        A = probabilities(X_b)
        rows = [
            (A[h] @ X_b)
            @ wv[(h // group_size) * head_dim : (h // group_size + 1) * head_dim].T
            for h in range(keep_heads)
        ]
        predicted = torch.cat(rows, dim=1)
        residual_norm += float((predicted - C_b).pow(2).sum())
        reference_norm += float(C_b.pow(2).sum())
    gate = (residual_norm / reference_norm) ** 0.5
    report["attention_recompute_relL2"] = gate
    if gate_probe is not None:
        report["gate_probe"] = attention_gate_probe(
            harness,
            attn,
            parent_layer,
            batched=batched,
            wq=wq,
            wk=wk,
            wv=wv,
            gq=gq,
            gk=gk,
            eps=eps,
            scaling=scaling,
            keep_heads=keep_heads,
            keep_groups=keep_groups,
            head_dim=head_dim,
            inv_freq=inv_freq,
            fp64_gate=gate,
            outdir=gate_probe,
        )
    if gate > ATTENTION_RECOMPUTE_TOLERANCE:
        report["refused"] = (
            f"the recomputed context is {gate:.3e} from the installed module's "
            "own; the joint value path is refused and wave 16's contraction stands"
        )
        return report

    out_weight = prior_out
    fit_objective = baseline_fit
    best_state: tuple[torch.Tensor, torch.Tensor] | None = None
    for sweep in range(sweeps):
        groups = [slot["group_index"] for slot in merged_slots]
        Zcov = {
            (g, h, hp): torch.zeros(hidden, hidden, dtype=torch.float64)
            for g in groups
            for h in range(g * group_size, (g + 1) * group_size)
            for hp in range(g * group_size, (g + 1) * group_size)
        }
        Rcross = {
            (g, h): torch.zeros(out_weight.shape[0], hidden, dtype=torch.float64)
            for g in groups
            for h in range(g * group_size, (g + 1) * group_size)
        }
        # The same factors, accumulated on the internal check batches ALONE.  A
        # candidate solved without them is scored against them in closed form,
        # which is what makes the lambda selection an out-of-sample one.
        Zcheck = {key: torch.zeros_like(value) for key, value in Zcov.items()}
        Rcheck = {key: torch.zeros_like(value) for key, value in Rcross.items()}
        Ncheck = {g: 0.0 for g in groups}
        columns = {
            g: [
                index
                for h in range(keep_heads)
                if h // group_size != g
                for index in range(h * head_dim, (h + 1) * head_dim)
            ]
            for g in groups
        }
        offset = 0
        for index, ((split, X_b), (_, C_b)) in enumerate(
            zip(batched["attn"], batched["o"])
        ):
            if split != "fit":
                continue
            width = X_b.shape[0]
            Y_b = target[offset : offset + width]
            offset += width
            A = probabilities(X_b)
            for g in groups:
                heads = list(range(g * group_size, (g + 1) * group_size))
                z = {h: A[h] @ X_b for h in heads}
                # Every head outside this group is held fixed, so its share of
                # the block output leaves the target rather than the unknown.
                residual = Y_b - C_b[:, columns[g]] @ out_weight[:, columns[g]].T
                for h in heads:
                    for hp in heads:
                        Zcov[(g, h, hp)] += z[h].T @ z[hp]
                    Rcross[(g, h)] += residual.T @ z[h]
                if index in check_batches:
                    Ncheck[g] += float(residual.pow(2).sum())
                    for h in heads:
                        for hp in heads:
                            Zcheck[(g, h, hp)] += z[h].T @ z[hp]
                        Rcheck[(g, h)] += residual.T @ z[h]
        for g in groups:
            heads = list(range(g * group_size, (g + 1) * group_size))
            left = {
                (h, hp): out_weight[:, h * head_dim : (h + 1) * head_dim].T
                @ out_weight[:, hp * head_dim : (hp + 1) * head_dim]
                for h in heads
                for hp in heads
            }
            right = {(h, hp): Zcov[(g, hp, h)] for h in heads for hp in heads}
            rhs = sum(
                out_weight[:, h * head_dim : (h + 1) * head_dim].T @ Rcross[(g, h)]
                for h in heads
            )
            prior_v = wv[g * head_dim : (g + 1) * head_dim]
            if lambda_split:
                inner_right = {
                    (h, hp): Zcov[(g, hp, h)] - Zcheck[(g, hp, h)]
                    for h in heads
                    for hp in heads
                }
                inner_rhs = sum(
                    out_weight[:, h * head_dim : (h + 1) * head_dim].T
                    @ (Rcross[(g, h)] - Rcheck[(g, h)])
                    for h in heads
                )
                check_right = {(h, hp): Zcheck[(g, hp, h)] for h in heads for hp in heads}
                check_rhs = sum(
                    out_weight[:, h * head_dim : (h + 1) * head_dim].T @ Rcheck[(g, h)]
                    for h in heads
                )
                solved_v, guard_v = law_abiding_stein_split(
                    left,
                    right,
                    rhs,
                    (inner_right, inner_rhs),
                    (check_right, check_rhs, Ncheck[g]),
                    prior_v,
                )
            else:
                solved_v, guard_v = law_abiding_stein(left, right, rhs, prior_v)
            wv = wv.clone()
            wv[g * head_dim : (g + 1) * head_dim] = solved_v
            report.setdefault("value_guards", []).append(
                {"sweep": sweep, "group": g, **guard_v}
            )
        attn.v_proj.weight.data.copy_(wv.to(device=device, dtype=dtype))
        context = harness.capture({"o": attn.out_proj})
        C_fit, C_hld = context["o"]["fit"], context["o"]["hld"]
        if lambda_split:
            candidate, out_guard = law_abiding_solve_split(
                C_fit, target, check_rows, prior_out
            )
        else:
            candidate, out_guard = law_abiding_solve(C_fit, target, prior_out)
        objective = float((C_fit @ candidate.T - target).pow(2).sum())
        report["sweeps_run"] = sweep + 1
        report.setdefault("fit_objective", []).append(
            {
                "sweep": sweep,
                "fit_sse": objective,
                "baseline_fit_sse": baseline_fit,
                # The internal check block rides here for every child, split or
                # not: it is the number that says whether a sweep that improved
                # its own objective improved anything else.  It decides nothing.
                "internal_check_sse": float(
                    (C_fit[check_rows] @ candidate.T - target[check_rows]).pow(2).sum()
                ),
                "heldout_cos": write_scores(C_hld @ candidate.T, target_h)["cos"],
            }
        )
        # The DECISION is on the fit objective the sweep optimises; the held-out
        # number rides beside it and decides nothing, which is the same posture
        # every other solve in this file takes.
        if objective < fit_objective:
            fit_objective = objective
            out_weight = candidate
            best_state = (wv.clone(), candidate.clone())
            report["out_guard"] = out_guard
        else:
            break
    if best_state is None:
        # Nothing improved: restore what the wave-16 path installed.
        attn.v_proj.weight.data.copy_(wv_initial.to(device=device, dtype=dtype))
        return report
    wv_final, out_final = best_state
    attn.v_proj.weight.data.copy_(wv_final.to(device=device, dtype=dtype))
    attn.out_proj.weight.data.copy_(out_final.to(device=device, dtype=dtype))
    report["installed"] = True
    report["fit_sse_improvement"] = (baseline_fit - fit_objective) / baseline_fit
    return report


# ------------------------------------------------------- the hidden_size axis
#: The two bases this wave will build against, and the third that exists only to
#: gate the machinery.  ``reader-pca`` is the DEFAULT and the reason is stated in
#: one sentence: RMSNorm makes the residual's scale irrelevant to every consumer
#: and its per-channel gain part of every read, so the rows a reader actually
#: sees are the rows the projection error is incurred in.  ``evidence-pca`` is
#: the raw residual's own second moment, endpoint-equalised, and is built as the
#: declared alternative if a width lands marginal.  Both spectra are published
#: from one pass whatever is chosen, so the choice is legible.
RESIDUAL_BASIS_MODES = ("reader-pca", "evidence-pca", "identity")
#: Widths the free census prices, whether or not they are on the ladder.
CENSUS_WIDTHS = (1024, 960, 896, 832, 768, 704, 640, 576, 512, 448, 384, 256)


def residual_endpoints(model: Any) -> dict[str, Any]:
    """Every module whose INPUT is the raw residual stream, in stream order.

    Twenty-nine of them: each block's ``operator_norm`` (the residual as the
    attention or conv branch finds it) and ``ffn_norm`` (the residual after that
    branch's write), plus the final ``embedding_norm`` (the residual as the tied
    head finds it).  This is the complete set of places a stock LFM2 reads the
    stream, which is what makes a basis pooled over them the whole object rather
    than a sample of it.
    """

    ends: dict[str, Any] = {}
    for index, layer in enumerate(model.model.layers):
        ends[f"B{index:02d}.op"] = layer.operator_norm
        ends[f"B{index:02d}.ffn"] = layer.ffn_norm
    ends["OUT.final"] = model.model.embedding_norm
    return ends


@torch.no_grad()
def residual_moments(harness: Harness, *, keep_rows: bool = False) -> dict[str, Any]:
    """Per-endpoint fp64 second moments of the residual, fit and held-out apart.

    Two per endpoint per split.  ``raw`` is the residual as it is carried;
    ``reader`` is the residual as it is READ -- RMS-normed and multiplied by that
    endpoint's own gain, the only form any consumer in this architecture ever
    sees.  ``keep_rows`` also retains the raw rows in bf16 so a child's stream
    can be measured against the parent's at every site; that is a diagnostic and
    bf16 is two orders below the drift it reports.
    """

    ends = residual_endpoints(harness.model)
    hidden = int(harness.model.config.hidden_size)
    eps = float(harness.model.config.norm_eps)
    gains = {
        name: module.weight.data.to(torch.float64).cpu() for name, module in ends.items()
    }
    acc: dict[str, Any] = {
        name: {
            split: {
                "raw": torch.zeros(hidden, hidden, dtype=torch.float64),
                "reader": torch.zeros(hidden, hidden, dtype=torch.float64),
                "rows": 0,
                "sum_square": 0.0,
            }
            for split in ("fit", "hld")
        }
        for name in ends
    }
    rows: dict[str, dict[str, list[torch.Tensor]]] = {
        name: {"fit": [], "hld": []} for name in ends
    }
    state: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs["hidden_states"]
            state[name] = hidden_states.detach()
            return None

        return hook

    for name, module in ends.items():
        handles.append(module.register_forward_pre_hook(make_hook(name), with_kwargs=True))
    try:
        for index, ids in harness.batches:
            split = "fit" if harness.split[index] == "fit" else "hld"
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
            harness.forward_calls += 1
            for name in ends:
                block = state[name].reshape(-1, hidden)
                r = block.to("cpu", torch.float64)
                bucket = acc[name][split]
                bucket["raw"] += r.T @ r
                normed = r * r.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
                read = normed * gains[name]
                bucket["reader"] += read.T @ read
                bucket["rows"] += int(r.shape[0])
                bucket["sum_square"] += float(r.pow(2).sum())
                if keep_rows:
                    rows[name][split].append(block.to("cpu", torch.bfloat16))
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {
        "endpoints": tuple(ends),
        "hidden": hidden,
        "moments": acc,
        "gains": gains,
        "rows": (
            {
                name: {
                    split: torch.cat(blocks, dim=0)
                    for split, blocks in per_split.items()
                    if blocks
                }
                for name, per_split in rows.items()
            }
            if keep_rows
            else None
        ),
    }


def pooled_moment(moments: dict[str, Any], kind: str, split: str) -> torch.Tensor:
    """Endpoint-equalised pool: each endpoint contributes exactly trace one.

    Unequalised, the deepest endpoints would decide the basis on their own --
    the residual's norm grows monotonically down this stack -- and every
    endpoint's read is RMS-normed before anything sees it, so an endpoint's
    absolute scale is precisely the quantity that does not matter.
    """

    hidden = int(moments["hidden"])
    pooled = torch.zeros(hidden, hidden, dtype=torch.float64)
    count = 0
    for name in moments["endpoints"]:
        block = moments["moments"][name][split][kind]
        trace = float(torch.diagonal(block).sum())
        if trace <= 0.0:
            raise SystemExit(f"{name} {split} {kind}: zero-trace second moment")
        pooled += block / trace
        count += 1
    return pooled / count


def spectrum(pooled: torch.Tensor) -> torch.Tensor:
    """Descending eigenvalues of a symmetric second moment, normalised to sum 1."""

    values = torch.linalg.eigvalsh(pooled).flip(0)
    return values / float(values.sum())


def retained_fraction(pooled: torch.Tensor, basis: torch.Tensor) -> float:
    """``tr(E^T M E) / tr(M)`` -- the energy the child's subspace keeps."""

    return float((basis.T @ pooled @ basis).diagonal().sum() / torch.diagonal(pooled).sum())


def build_basis(moments: dict[str, Any], width: int, mode: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """The shared orthonormal ``E`` (1024 x width) and the census that prices it."""

    hidden = int(moments["hidden"])
    if mode == "identity":
        if width != hidden:
            raise SystemExit(
                f"--basis identity is the gate and only exists at width {hidden}"
            )
        basis = torch.eye(hidden, dtype=torch.float64)
    else:
        kind = "reader" if mode == "reader-pca" else "raw"
        pooled = pooled_moment(moments, kind, "fit")
        values, vectors = torch.linalg.eigh(pooled)
        order = torch.argsort(values, descending=True)[:width]
        basis = vectors[:, order].contiguous()
        # Deterministic column signs: the basis is written to an artifact that
        # several children are built against, and an eigenvector's sign is free.
        for column in range(basis.shape[1]):
            pivot = int(basis[:, column].abs().argmax())
            if float(basis[pivot, column]) < 0.0:
                basis[:, column] = -basis[:, column]
    orthonormality = float(
        (basis.T @ basis - torch.eye(basis.shape[1], dtype=torch.float64)).abs().max()
    )
    if orthonormality > 1e-10:
        raise SystemExit(f"basis is not orthonormal to 1e-10 (max {orthonormality:.3e})")

    census: dict[str, Any] = {
        "schema": BASIS_SCHEMA,
        "mode": mode,
        "width": int(width),
        "parent_hidden": hidden,
        "orthonormality_max_abs_offdiag": orthonormality,
        "spectrum": {},
        "retained": {},
        "per_endpoint_retained": {},
    }
    for kind in ("raw", "reader"):
        for split in ("fit", "hld"):
            pooled = pooled_moment(moments, kind, split)
            values = spectrum(pooled)
            cumulative = torch.cumsum(values, dim=0)
            census["spectrum"][f"{kind}.{split}"] = {
                f"top_{w}": float(cumulative[w - 1]) for w in CENSUS_WIDTHS
            }
            if mode != "identity":
                census["retained"][f"{kind}.{split}"] = retained_fraction(pooled, basis)
    for name in moments["endpoints"]:
        for split in ("fit", "hld"):
            block = moments["moments"][name][split]["raw"]
            census["per_endpoint_retained"][f"{name}.{split}.raw"] = retained_fraction(
                block, basis
            )
    return basis, census


def write_basis(path: Path, basis: torch.Tensor, census: dict[str, Any]) -> str:
    """Seal the basis as one artifact so every rung is built against one byte string."""

    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"basis": basis.to(torch.float64).contiguous()},
        str(path),
        metadata={
            "schema": BASIS_SCHEMA,
            "mode": str(census["mode"]),
            "width": str(census["width"]),
        },
    )
    digest = file_sha256(path)
    census["basis_sha256"] = digest
    census["basis_path"] = str(path)
    path.with_suffix(".census.json").write_text(json.dumps(census, indent=1))
    return digest


def read_basis(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    from safetensors import safe_open

    with safe_open(str(path), "pt") as handle:
        basis = handle.get_tensor("basis").to(torch.float64)
        meta = dict(handle.metadata() or {})
    if meta.get("schema") != BASIS_SCHEMA:
        raise SystemExit(f"{path} declares schema {meta.get('schema')!r}")
    census_path = path.with_suffix(".census.json")
    census = json.loads(census_path.read_text()) if census_path.is_file() else {}
    declared = census.get("basis_sha256")
    digest = file_sha256(path)
    if declared is not None and declared != digest:
        raise SystemExit(f"{path} digests {digest}; its census claims {declared}")
    census["basis_sha256"] = digest
    return basis, census


# ---------------------------------------------------------------- fold algebra
def transported_gain(gain: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``diag(E^T diag(g) E)`` -- the gain each child coordinate sees.

    Every column of ``E`` is a unit vector, so ``g_c[j]`` is a convex combination
    of the parent's own gains and cannot leave their range.  It is what makes the
    identity gate bit-exact: at ``E = I`` it returns ``g`` entry for entry, and
    the reader fold below then divides a product by its own factor.
    """

    return (basis.pow(2) * gain[:, None]).sum(dim=0)


def fold_reader(
    weight: torch.Tensor, gain: torch.Tensor, basis: torch.Tensor,
    child_gain: torch.Tensor, scale: float,
) -> torch.Tensor:
    """``sqrt(1024/H) * W diag(g) E diag(1/g_c)`` -- the exact reader fold.

    Exact against the LOCAL target and not approximately so: the child's residual
    lifted through ``E`` has the same norm it had, so the parent's RMSNorm at the
    lifted row and the child's at the row itself differ by exactly
    ``sqrt(1024/H)``.  Everything this axis loses is lost in the stream, not here.
    """

    if float(child_gain.abs().min()) < GAIN_FLOOR:
        raise SystemExit(
            f"transported gain reaches {float(child_gain.abs().min()):.3e}, under the "
            f"{GAIN_FLOOR:g} floor; the reader fold's division is refused"
        )
    return (((weight * gain[None, :]) @ basis) / child_gain[None, :]) * scale


def fold_writer(weight: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """``E^T W`` -- the L2-optimal write into the retained subspace."""

    return basis.T @ weight


def pivoted_columns_gram(G: torch.Tensor, k: int) -> torch.Tensor:
    """`pivoted_columns` given the gram directly, for an accumulated one.

    Same arithmetic, same tie-breaks; the only difference is that the caller
    formed ``F^T F`` itself, which is what an incremental capture produces.
    """

    n = int(G.shape[1])
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    G = G.clone()
    diag = torch.diagonal(G).clone()
    chosen: list[int] = []
    L = torch.zeros(n, k, dtype=G.dtype)
    for step in range(k):
        best = -1.0
        pick = -1
        for i in range(n):
            if diag[i] > best + 1e-15:
                best = float(diag[i])
                pick = i
        if pick < 0 or best <= 1e-30:
            break
        chosen.append(pick)
        col = G[:, pick] - L[:, :step] @ L[pick, :step]
        col = col / max(best, 1e-300) ** 0.5
        L[:, step] = col
        diag = diag - col * col
        diag[chosen] = -1.0
    return torch.tensor(sorted(chosen), dtype=torch.long)


def _linear(weight: torch.Tensor, dtype: torch.dtype, device: Any) -> nn.Module:
    out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
    linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)
    linear.weight.data.copy_(weight.to(device=device, dtype=dtype))
    return linear


# ------------------------------------------------------------- the child model
def child_config(
    parent_config: dict[str, Any], *, hidden: int, excised: list[int]
) -> dict[str, Any]:
    """The child's config, written before the model is built and loaded FROM it.

    Building the model from the file the child ships is not a formality: it is
    the R3 claim executed at build time, so an illegal field is a crash here
    rather than a surprise at the seal.
    """

    config = json.loads(json.dumps(parent_config))
    config["hidden_size"] = hidden
    config["block_dim"] = hidden
    config["conv_dim"] = hidden
    # MANDATORY, and it is the whole legality argument of this axis.  Left out,
    # `Lfm2Attention` falls back to hidden_size // num_attention_heads -- 48 at
    # H=768 -- which changes the RoPE frequency ladder AND the head width, i.e.
    # it silently turns this axis into wave 15's sealed-dead `--head-dim` axis.
    config["head_dim"] = PINNED_HEAD_DIM
    if excised:
        types = [t for i, t in enumerate(parent_config["layer_types"]) if i not in excised]
        config["layer_types"] = types
        config["num_hidden_layers"] = len(types)
    return config


def instantiate_child(config_dict: dict[str, Any], outdir: Path, device: str, dtype: torch.dtype):
    from transformers import AutoConfig, Lfm2ForCausalLM

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config_dict, indent=2))
    config = AutoConfig.from_pretrained(str(outdir))
    config._attn_implementation = "eager"
    model = Lfm2ForCausalLM(config).to(device=device, dtype=dtype).eval()
    live = [
        int(layer.self_attn.head_dim)
        for layer in model.model.layers
        if getattr(layer, "is_attention_layer", False)
    ]
    if live and set(live) != {PINNED_HEAD_DIM}:
        raise SystemExit(
            f"the child built its attention at head_dim {sorted(set(live))}, not "
            f"{PINNED_HEAD_DIM}; the config's head_dim did not take"
        )
    return model


@torch.no_grad()
def fold_layer_closed_form(
    child_layer,
    parent_layer_module,
    basis: torch.Tensor,
    *,
    scale: float,
    hidden: int,
    conv_selection: torch.Tensor | None,
    dtype: torch.dtype,
    device: Any,
) -> dict[str, Any]:
    """Install one block's CLOSED-FORM fold: readers exact, writers projected.

    No evidence is read here beyond the conv coreset the caller chose.  This is
    the whole child at ``--arm stale`` and the prior for every solve in the
    conditioned arm.
    """

    record: dict[str, Any] = {}
    for norm_name in ("operator_norm", "ffn_norm"):
        gain = getattr(parent_layer_module, norm_name).weight.data.to(torch.float64).cpu()
        child_gain = transported_gain(gain, basis)
        getattr(child_layer, norm_name).weight.data.copy_(
            child_gain.to(device=device, dtype=dtype)
        )
        record[f"{norm_name}_gain_min"] = float(child_gain.min())
        record[f"{norm_name}_gain_max"] = float(child_gain.max())

    op_gain = parent_layer_module.operator_norm.weight.data.to(torch.float64).cpu()
    op_child_gain = transported_gain(op_gain, basis)
    ffn_gain = parent_layer_module.ffn_norm.weight.data.to(torch.float64).cpu()
    ffn_child_gain = transported_gain(ffn_gain, basis)

    if getattr(parent_layer_module, "is_attention_layer", False):
        parent_attn = parent_layer_module.self_attn
        child_attn = child_layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            weight = getattr(parent_attn, name).weight.data.to(torch.float64).cpu()
            setattr(
                child_attn,
                name,
                _linear(fold_reader(weight, op_gain, basis, op_child_gain, scale), dtype, device),
            )
        out_weight = parent_attn.out_proj.weight.data.to(torch.float64).cpu()
        child_attn.out_proj = _linear(fold_writer(out_weight, basis), dtype, device)
        for name in ("q_layernorm", "k_layernorm"):
            getattr(child_attn, name).weight.data.copy_(
                getattr(parent_attn, name).weight.data.to(device=device, dtype=dtype)
            )
        record["branch"] = "full_attention"
    else:
        parent_conv = parent_layer_module.conv
        child_conv = child_layer.conv
        selection = (
            torch.arange(hidden, dtype=torch.long)
            if conv_selection is None
            else conv_selection
        )
        if int(selection.numel()) != int(child_conv.conv.weight.shape[0]):
            raise SystemExit(
                f"conv coreset has {int(selection.numel())} channels and the child's "
                f"conv has {int(child_conv.conv.weight.shape[0])}"
            )
        in_weight = parent_conv.in_proj.weight.data.to(torch.float64).cpu()
        parent_hidden = int(in_weight.shape[1])
        rows = torch.cat(
            [selection + part * parent_hidden for part in range(3)], dim=0
        )
        child_conv.in_proj = _linear(
            fold_reader(in_weight[rows], op_gain, basis, op_child_gain, scale), dtype, device
        )
        out_weight = parent_conv.out_proj.weight.data.to(torch.float64).cpu()
        child_conv.out_proj = _linear(
            fold_writer(out_weight[:, selection], basis), dtype, device
        )
        child_conv.conv.weight.data.copy_(
            parent_conv.conv.weight.data[selection].to(device=device, dtype=dtype)
        )
        if parent_conv.conv.bias is not None:
            child_conv.conv.bias.data.copy_(
                parent_conv.conv.bias.data[selection].to(device=device, dtype=dtype)
            )
        record["branch"] = "conv"
        record["conv_channels_first8"] = [int(x) for x in selection[:8]]

    parent_mlp = parent_layer_module.feed_forward
    child_mlp = child_layer.feed_forward
    for name in ("w1", "w3"):
        weight = getattr(parent_mlp, name).weight.data.to(torch.float64).cpu()
        setattr(
            child_mlp,
            name,
            _linear(fold_reader(weight, ffn_gain, basis, ffn_child_gain, scale), dtype, device),
        )
    child_mlp.w2 = _linear(
        fold_writer(parent_mlp.w2.weight.data.to(torch.float64).cpu(), basis), dtype, device
    )
    return record


@torch.no_grad()
def site_evidence(
    child: Harness,
    position: int,
    parent_block,
    basis: torch.Tensor,
    *,
    branch: str,
) -> dict[str, torch.Tensor]:
    """The child's own residual at one endpoint, and the parent's answer to it.

    One pass over the corpus.  For each batch the child's RAW residual at the
    endpoint is captured, lifted through ``E`` (which preserves its norm), passed
    through the PARENT's own norm and the pristine parent branch with the
    kwargs the child's own call supplied -- position embeddings and masks are
    functions of ``head_dim`` and sequence length and this wave pins both -- and
    the parent's write is projected back.  That projected write is the local
    target; nothing here is measured against the parent's own stream.
    """

    layer = child.layers()[position]
    if branch == "ffn":
        norm_module, child_module = layer.ffn_norm, layer.feed_forward
        parent_norm, parent_module = parent_block.ffn_norm, parent_block.feed_forward
    elif branch == "operator":
        norm_module = layer.operator_norm
        parent_norm = parent_block.operator_norm
        if getattr(layer, "is_attention_layer", False):
            child_module, parent_module = layer.self_attn, parent_block.self_attn
        else:
            child_module, parent_module = layer.conv, parent_block.conv
    else:
        raise SystemExit(f"unknown branch {branch!r}")

    hidden = int(norm_module.weight.shape[0])
    lifter = basis.T.to(device=child.device, dtype=child.dtype).contiguous()
    buckets: dict[str, list[torch.Tensor]] = {
        "resid_fit": [], "resid_hld": [], "post_fit": [], "post_hld": [],
        "parent_post_fit": [], "parent_post_hld": [],
        "target_fit": [], "target_hld": [],
    }
    state: dict[str, Any] = {}
    handles = [
        norm_module.register_forward_pre_hook(
            lambda module, args, kwargs: state.__setitem__(
                "resid", (args[0] if args else kwargs["hidden_states"]).detach()
            ),
            with_kwargs=True,
        ),
        child_module.register_forward_pre_hook(
            lambda module, args, kwargs: state.__setitem__("call", (args, kwargs)),
            with_kwargs=True,
        ),
        norm_module.register_forward_hook(
            lambda module, args, output: state.__setitem__("post", output.detach())
        ),
    ]
    try:
        for index, ids in child.batches:
            split = "fit" if child.split[index] == "fit" else "hld"
            child.model(input_ids=ids.to(child.device), use_cache=False)
            child.forward_calls += 1
            resid = state["resid"]
            args, kwargs = state["call"]
            lifted = resid @ lifter
            parent_input = parent_norm(lifted)
            call_kwargs = dict(kwargs)
            call_args = list(args)
            if call_args:
                call_args[0] = parent_input
            else:
                call_kwargs["hidden_states"] = parent_input
            written = parent_module(*call_args, **call_kwargs)
            if isinstance(written, tuple):
                written = written[0]
            target = written.detach() @ lifter.T
            buckets[f"resid_{split}"].append(
                resid.reshape(-1, hidden).to("cpu", torch.float64)
            )
            buckets[f"post_{split}"].append(
                state["post"].reshape(-1, hidden).to("cpu", torch.float64)
            )
            buckets[f"parent_post_{split}"].append(
                parent_input.reshape(-1, int(lifter.shape[1])).to("cpu", torch.float64)
            )
            buckets[f"target_{split}"].append(
                target.reshape(-1, hidden).to("cpu", torch.float64)
            )
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.cat(blocks, dim=0).contiguous() for name, blocks in buckets.items()}


def stream_drift(
    child_rows: torch.Tensor, parent_rows: dict[str, torch.Tensor] | None, basis: torch.Tensor
) -> dict[str, float]:
    """How far the child's stream is from the parent's OWN, in the child's basis.

    ``E^T r_parent`` is what the child's residual would be if nothing had been
    lost; the gap between it and the child's actual residual is the compounding
    this axis pays, and it is the number wave 15 called ``input_drift_vs_parent``.
    ``retained`` is the other half of the same story: how much of the parent's
    residual the subspace can hold at all.
    """

    if parent_rows is None or "fit" not in parent_rows:
        return {}
    reference = parent_rows["fit"].to(torch.float64)
    projected = reference @ basis
    delta = child_rows - projected
    return {
        "input_drift_vs_parent": float(delta.norm() / projected.norm().clamp_min(1e-30)),
        "input_cos_vs_parent": float(
            torch.nn.functional.cosine_similarity(child_rows, projected, dim=1).mean()
        ),
        "parent_energy_retained": float(
            projected.pow(2).sum() / reference.pow(2).sum().clamp_min(1e-30)
        ),
    }


@torch.no_grad()
def conv_selection_grams(parent: Harness, conv_layers: list[int]) -> dict[int, torch.Tensor]:
    """Accumulate each conv block's gated-intermediate gram on the PARENT's stream.

    ``y = C * conv(B * x)`` is the tensor ``out_proj`` reads, so it is the axis
    the coreset must be selected on -- wave 14's rule, applied to the one axis
    that FORCES a coreset: `Lfm2ShortConv` gates channelwise, so a rotation of
    its internal channel space is not a representable object and the basis fold
    every other tensor gets is not available there.  The gram is accumulated
    rather than the rows kept because only the gram is used, and
    `pivoted_columns_gram` is `pivoted_columns` with its first line already done.
    """

    width = int(parent.model.config.hidden_size)
    grams = {layer: torch.zeros(width, width, dtype=torch.float64) for layer in conv_layers}
    state: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(module, args, kwargs):
            state[layer] = (args[0] if args else kwargs["input"]).detach()
            return None

        return hook

    for layer in conv_layers:
        handles.append(
            parent.layers()[layer].conv.out_proj.register_forward_pre_hook(
                make_hook(layer), with_kwargs=True
            )
        )
    try:
        for index, ids in parent.batches:
            if parent.split[index] != "fit":
                continue
            parent.model(input_ids=ids.to(parent.device), use_cache=False)
            parent.forward_calls += 1
            for layer in conv_layers:
                rows = state[layer].reshape(-1, width).to("cpu", torch.float64)
                grams[layer] += rows.T @ rows
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return grams


@torch.no_grad()
def parent_gated_rows(
    parent: Harness, child: Harness, position: int, parent_layer: int, basis: torch.Tensor
) -> torch.Tensor:
    """The parent conv's gated intermediate at the CHILD's lifted input, fit rows.

    The conditioned counterpart of `conv_selection_grams`: which channels the
    child should keep is a question about the stream the child actually has, and
    the answer moves once the blocks above it are folded.
    """

    parent_block = parent.layers()[parent_layer]
    parent_conv, parent_norm = parent_block.conv, parent_block.operator_norm
    layer = child.layers()[position]
    parent_hidden = int(parent.model.config.hidden_size)
    lifter = basis.T.to(device=child.device, dtype=child.dtype).contiguous()
    collected: list[torch.Tensor] = []
    state: dict[str, Any] = {}
    handles = [
        layer.operator_norm.register_forward_pre_hook(
            lambda module, args, kwargs: state.__setitem__(
                "resid", (args[0] if args else kwargs["hidden_states"]).detach()
            ),
            with_kwargs=True,
        ),
        layer.conv.register_forward_pre_hook(
            lambda module, args, kwargs: state.__setitem__("call", (args, kwargs)),
            with_kwargs=True,
        ),
        parent_conv.out_proj.register_forward_pre_hook(
            lambda module, args, kwargs: state.__setitem__(
                "gated", (args[0] if args else kwargs["input"]).detach()
            ),
            with_kwargs=True,
        ),
    ]
    try:
        for index, ids in child.batches:
            if child.split[index] != "fit":
                continue
            child.model(input_ids=ids.to(child.device), use_cache=False)
            child.forward_calls += 1
            args, kwargs = state["call"]
            lifted = parent_norm(state["resid"] @ lifter)
            call_kwargs = dict(kwargs)
            call_args = list(args)
            if call_args:
                call_args[0] = lifted
            else:
                call_kwargs["hidden_states"] = lifted
            parent_conv(*call_args, **call_kwargs)
            collected.append(
                state["gated"].reshape(-1, parent_hidden).to("cpu", torch.float64)
            )
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return torch.cat(collected, dim=0).contiguous()


@torch.no_grad()
def solve_head_gain(
    child: Harness, parent: Harness, basis: torch.Tensor, prior_gain: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """The final norm's diagonal, solved in closed form against the tied head.

    The tied matrix is the one tensor with two roles: as a WRITER it wants
    ``E^T e_t`` and as a READER it wants ``W_emb diag(g_f) E``, and those differ
    by exactly the final gain.  The writer's form wins -- the child must embed
    tokens -- so the final norm's diagonal is the only place the final gain can
    live, and it is solved there rather than transported:

        min_g  sum_i || (W_emb E) diag(g) z_i - W_emb diag(g_f) n_i ||^2

    whose normal equations are ``[(E^T C E) o (Z^T Z)] g = sum_i z_i o (Q n_i)``
    with ``C = W_emb^T W_emb`` and ``Q = E^T C diag(g_f)``: one H x H system, and
    no 65536-column target is ever materialised.  THE LAW guards the answer
    against the transported gain, which is also its prior.
    """

    width = int(basis.shape[1])
    embed = parent.model.model.embed_tokens.weight.data.to(torch.float64).cpu()
    gram = embed.T @ embed
    final_gain = parent.model.model.embedding_norm.weight.data.to(torch.float64).cpu()
    eps = float(parent.model.config.norm_eps)
    lifter = basis.T.to(device=child.device, dtype=child.dtype).contiguous()

    child_rows: list[torch.Tensor] = []
    lifted_rows: list[torch.Tensor] = []
    state: dict[str, torch.Tensor] = {}
    handle = child.model.model.embedding_norm.register_forward_pre_hook(
        lambda module, args, kwargs: state.__setitem__(
            "resid", (args[0] if args else kwargs["hidden_states"]).detach()
        ),
        with_kwargs=True,
    )
    try:
        for index, ids in child.batches:
            if child.split[index] != "fit":
                continue
            child.model(input_ids=ids.to(child.device), use_cache=False)
            child.forward_calls += 1
            resid = state["resid"].reshape(-1, width).to("cpu", torch.float64)
            child_rows.append(
                resid * resid.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt()
            )
            lift = resid @ basis.T
            lifted_rows.append(lift * lift.pow(2).mean(dim=1, keepdim=True).add(eps).rsqrt())
            state.clear()
    finally:
        handle.remove()
    Z = torch.cat(child_rows, dim=0)
    N = torch.cat(lifted_rows, dim=0)

    AtA = basis.T @ gram @ basis
    Q = (basis.T @ gram) * final_gain[None, :]
    cross = N @ Q.T
    b = (Z * cross).sum(dim=0)
    G = AtA * (Z.T @ Z)
    target_energy = float((((N * final_gain) @ gram) * (N * final_gain)).sum())

    def objective(gain: torch.Tensor) -> float:
        scaled = Z * gain[None, :]
        return float(
            ((scaled @ AtA) * scaled).sum() - 2.0 * (scaled * cross).sum() + target_energy
        )

    ones = torch.ones(width, dtype=torch.float64)
    alpha = float(b.sum() / float(ones @ G @ ones))
    reference = float(prior_gain.norm())
    scale = float(torch.diagonal(G).mean())
    chosen: dict[str, Any] | None = None
    ladder: list[dict[str, Any]] = []
    for rel in LAMBDA_LADDER:
        lam = rel * scale
        solved = torch.linalg.solve(
            G + lam * torch.eye(width, dtype=torch.float64), b + lam * prior_gain
        )
        update = float((solved - prior_gain).norm() / reference)
        ladder.append({"lambda_relative": rel, "update_scale": update})
        if chosen is None and update <= 1.0:
            chosen = {"gain": solved, "lambda_relative": rel, "update_scale": update}
    if chosen is None:
        raise SystemExit(
            "no lambda on the ladder brings the final gain's update_scale to 1.0; "
            "THE LAW refuses the install"
        )
    return chosen["gain"], {
        "lambda_relative": chosen["lambda_relative"],
        "update_scale": chosen["update_scale"],
        "scalar_anchor_alpha": alpha,
        "scalar_anchor_distance": float((alpha * ones - prior_gain).norm() / reference),
        "logit_relative_sse_transported": objective(prior_gain) / target_energy,
        "logit_relative_sse_scalar": objective(alpha * ones) / target_energy,
        "logit_relative_sse_solved": objective(chosen["gain"]) / target_energy,
        "ladder": ladder,
        "rows": int(Z.shape[0]),
    }


@torch.no_grad()
def width_axis(
    parent: Harness,
    basis: torch.Tensor,
    *,
    hidden: int,
    excised: list[int],
    arm: str,
    outdir: Path,
    device: str,
    dtype: torch.dtype,
    keep_groups: int | None,
    kv_solve: str,
    kv_als: int,
    kv_lambda_split: bool,
    kv_identity_gate: bool,
    kv_gate_probe: Path | None,
    fold_solve: bool,
    head_gain: bool,
    parent_rows: dict[str, Any] | None,
) -> tuple[Harness, dict[str, Any]]:
    """Build the narrowed child: closed form everywhere, then conditioned solves.

    The child is instantiated at the PARENT's head counts even when a merge is
    composed on top, because `head_count_axis` merges 8 KV groups down and would
    have nothing to merge if the config already said 6.  The merged counts are a
    config field and `export` writes them, which is where wave 16 put them.
    """

    parent_config = json.loads((PARENT / "config.json").read_text())
    parent_hidden = int(parent_config["hidden_size"])
    scale = math.sqrt(parent_hidden / hidden)
    layer_count = len(parent.layers())
    surviving = [l for l in range(layer_count) if l not in excised]
    config_dict = child_config(parent_config, hidden=hidden, excised=excised)
    child_model = instantiate_child(config_dict, outdir, device, dtype)
    child = Harness.wrap(child_model, device, dtype)
    log(
        f"child instantiated from its own config: hidden_size {hidden}, head_dim "
        f"{PINNED_HEAD_DIM}, {len(child.layers())} blocks, reader scale {scale:.6f}"
    )

    # ---- the tied embedding and the final norm's transported prior.
    embed = parent.model.model.embed_tokens.weight.data.to(torch.float64).cpu()
    child.model.model.embed_tokens.weight.data.copy_(
        (embed @ basis).to(device=device, dtype=dtype)
    )
    final_gain = parent.model.model.embedding_norm.weight.data.to(torch.float64).cpu()
    final_prior = transported_gain(final_gain, basis)
    child.model.model.embedding_norm.weight.data.copy_(
        final_prior.to(device=device, dtype=dtype)
    )

    # ---- the closed form at every site, so the child is a model from its first
    # forward and the conditioned pass has a stream to condition on.
    conv_layers = [
        l for l in surviving if parent_config["layer_types"][l] != "full_attention"
    ]
    grams = conv_selection_grams(parent, conv_layers) if conv_layers else {}
    stale_selection = {
        layer: pivoted_columns_gram(grams[layer], hidden) for layer in conv_layers
    }
    selection = dict(stale_selection)
    closed: list[dict[str, Any]] = []
    for position, parent_layer in enumerate(surviving):
        record = fold_layer_closed_form(
            child.layers()[position],
            parent.layers()[parent_layer],
            basis,
            scale=scale,
            hidden=hidden,
            conv_selection=selection.get(parent_layer),
            dtype=dtype,
            device=device,
        )
        record.update({"layer": parent_layer, "position": position})
        closed.append(record)
    log(f"closed-form fold installed at {len(closed)} blocks")

    per_site: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []
    reader_gate: list[dict[str, Any]] = []
    if arm != "stale":
        for position, parent_layer in enumerate(surviving):
            layer = child.layers()[position]
            parent_block = parent.layers()[parent_layer]
            is_attention = bool(getattr(layer, "is_attention_layer", False))

            # ------------------------------------------- the operator branch
            if not is_attention:
                gated = parent_gated_rows(parent, child, position, parent_layer, basis)
                picked = pivoted_columns(gated, hidden)
                fold_layer_closed_form(
                    layer,
                    parent_block,
                    basis,
                    scale=scale,
                    hidden=hidden,
                    conv_selection=picked,
                    dtype=dtype,
                    device=device,
                )
                selection[parent_layer] = picked
            evidence = site_evidence(
                child, position, parent_block, basis, branch="operator"
            )
            reader_gate.append(
                reader_fold_gate(
                    layer, parent_block, evidence, branch="operator", parent_layer=parent_layer
                )
            )
            record = _writer_site(
                child,
                layer.self_attn.out_proj if is_attention else layer.conv.out_proj,
                evidence,
                solve=fold_solve,
                dtype=dtype,
                device=device,
                fields={
                    "layer": parent_layer,
                    "position": position,
                    "branch": "full_attention" if is_attention else "conv",
                    "endpoint": "attention.o" if is_attention else "conv.out",
                    "conv_channels_first8": (
                        None if is_attention else [int(x) for x in selection[parent_layer][:8]]
                    ),
                    **stream_drift(
                        evidence["resid_fit"],
                        None if parent_rows is None else parent_rows[f"B{parent_layer:02d}.op"],
                        basis,
                    ),
                },
            )
            per_site.append(record)
            _log_site(record)

            if is_attention and keep_groups is not None:
                merge = head_count_axis(
                    child,
                    position,
                    parent_layer,
                    keep_groups,
                    trajectory_target=None,
                    solve_mode=kv_solve,
                    als_sweeps=int(kv_als),
                    identity_gate=bool(kv_identity_gate),
                    lambda_split=bool(kv_lambda_split),
                    gate_probe=kv_gate_probe,
                )
                merge_records.append(merge)
                log(
                    f"  A{parent_layer:02d} kv-groups {keep_groups} merged ON the "
                    f"narrowed block -> relL2 {merge['relL2']:.4f} cos "
                    f"{merge['cos']:.4f} (prior-only {merge['prior_only_cos']:.4f})"
                )

            # ------------------------------------------------ the MLP branch
            evidence = site_evidence(child, position, parent_block, basis, branch="ffn")
            reader_gate.append(
                reader_fold_gate(
                    layer, parent_block, evidence, branch="ffn", parent_layer=parent_layer
                )
            )
            record = _writer_site(
                child,
                layer.feed_forward.w2,
                evidence,
                solve=fold_solve,
                dtype=dtype,
                device=device,
                fields={
                    "layer": parent_layer,
                    "position": position,
                    "branch": "mlp",
                    "endpoint": "mlp.down",
                    **stream_drift(
                        evidence["resid_fit"],
                        None if parent_rows is None else parent_rows[f"B{parent_layer:02d}.ffn"],
                        basis,
                    ),
                },
            )
            per_site.append(record)
            _log_site(record)

    head_report: dict[str, Any] = {"solved": False}
    if head_gain:
        solved_gain, head_report = solve_head_gain(child, parent, basis, final_prior)
        child.model.model.embedding_norm.weight.data.copy_(
            solved_gain.to(device=device, dtype=dtype)
        )
        head_report["solved"] = True
        log(
            f"  final gain solved: update_scale {head_report['update_scale']:.4f} "
            f"(lam {head_report['lambda_relative']:g}); relative logit SSE "
            f"{head_report['logit_relative_sse_transported']:.4f} (transported) -> "
            f"{head_report['logit_relative_sse_solved']:.4f} (solved)"
        )

    return child, {
        "closed_form": closed,
        "per_site": per_site,
        "merge_records": merge_records,
        "reader_fold_gate": reader_gate,
        "head_gain": head_report,
        "conv_channels_moved_vs_stale": {
            str(layer): int((selection[layer] != stale_selection[layer]).sum())
            for layer in conv_layers
        },
        "reader_scale": scale,
        "surviving": surviving,
    }


def reader_fold_gate(
    layer, parent_block, evidence: dict[str, torch.Tensor], *, branch: str, parent_layer: int
) -> dict[str, Any]:
    """Measure the claim the whole axis rests on: the reader fold is EXACT.

    The child's first reader, fed the child's own post-norm rows, must reproduce
    the parent's first reader fed the parent's post-norm rows at the LIFTED
    input -- not approximately, but to the floor of the bf16 the weights are
    stored in.  If it does not, ``sqrt(1024/H) W diag(g) E diag(1/g_c)`` is not
    the map this file says it is and every number below it is measured against
    the wrong target.
    """

    if branch == "ffn":
        child_weight = layer.feed_forward.w1.weight
        parent_weight = parent_block.feed_forward.w1.weight
        name = "mlp.w1"
    elif getattr(layer, "is_attention_layer", False):
        child_weight = layer.self_attn.q_proj.weight
        parent_weight = parent_block.self_attn.q_proj.weight
        name = "attention.q"
    else:
        child_weight = layer.conv.in_proj.weight
        parent_weight = parent_block.conv.in_proj.weight
        name = "conv.in"
    child_out = evidence["post_hld"] @ child_weight.data.to(torch.float64).cpu().T
    parent_out = evidence["parent_post_hld"] @ parent_weight.data.to(torch.float64).cpu().T
    if name == "conv.in":
        # The child keeps a coreset of the parent's channels in each of the three
        # gated parts, so the comparable columns are the selected ones.
        keep = child_out.shape[1] // 3
        parent_width = parent_out.shape[1] // 3
        _ = (keep, parent_width)
        return {
            "layer": parent_layer,
            "reader": name,
            "note": (
                "a conv coreset keeps a SUBSET of the parent's channels, so the "
                "reader fold is gated on the attention and MLP readers, which "
                "keep all of them"
            ),
        }
    return {
        "layer": parent_layer,
        "reader": name,
        "relL2": float((child_out - parent_out).norm() / parent_out.norm()),
        "max_abs": float((child_out - parent_out).abs().max()),
    }


def _writer_site(
    child: Harness,
    writer,
    evidence: dict[str, torch.Tensor],
    *,
    solve: bool,
    dtype: torch.dtype,
    device: Any,
    fields: dict[str, Any],
) -> dict[str, Any]:
    """Score, and optionally re-solve, one writer against the projected target."""

    prior = writer.weight.data.to(torch.float64).cpu()
    context = child.capture({"w": writer})
    prior_scores = write_scores(context["w"]["hld"] @ prior.T, evidence["target_hld"])
    record = dict(fields)
    record["prior_only_cos"] = prior_scores["cos"]
    record["prior_only_relL2"] = prior_scores["relL2"]
    if solve:
        solved, guard = law_abiding_solve(
            context["w"]["fit"], evidence["target_fit"], prior
        )
        writer.weight.data.copy_(solved.to(device=device, dtype=dtype))
        record.update(guard)
        record.update(write_scores(context["w"]["hld"] @ solved.T, evidence["target_hld"]))
    else:
        record.update(
            {
                "update_scale": 0.0,
                "lambda_relative": RIDGE_REL,
                "free_solve_update_scale": 0.0,
                "writer_solve": "off",
            }
        )
        record.update(prior_scores)
    return record


def _log_site(record: dict[str, Any]) -> None:
    log(
        f"  L{record['layer']:02d} {record['branch'][:4]:4s} drift "
        f"{record.get('input_drift_vs_parent', float('nan')):.4f} relL2 "
        f"{record['relL2']:.4f} cos {record['cos']:.4f} (prior-only cos "
        f"{record['prior_only_cos']:.4f}) update_scale {record['update_scale']:.3f}"
    )


# --------------------------------------------------------------------- export
def export(
    harness: Harness,
    outdir: Path,
    *,
    keep_ff: int | None,
    keep_head_dim: int | None,
    keep_groups: int | None,
    keep_hidden: int | None,
    excised: list[int],
    rope_theta: float,
) -> int:
    """Write a stock checkpoint: the live parameters plus a consistent config."""

    from safetensors.torch import save_file

    parent_config = json.loads((PARENT / "config.json").read_text())
    config = json.loads((PARENT / "config.json").read_text())
    state = {
        name: value.detach().to("cpu", torch.bfloat16).contiguous()
        for name, value in harness.model.named_parameters()
    }
    state.pop("lm_head.weight", None)
    if keep_hidden is not None:
        config["hidden_size"] = keep_hidden
        # `block_dim` and `conv_dim` are this checkpoint's own restatements of
        # the residual width.  `Lfm2Config` declares neither and the modeling
        # code reads neither -- `Lfm2ShortConv` builds itself from
        # ``config.hidden_size`` -- but a config that says 1024 in two fields
        # and 768 in the one that is read is not config-consistent under R3.
        config["block_dim"] = keep_hidden
        config["conv_dim"] = keep_hidden
        # MANDATORY.  See `child_config`: without it every attention block
        # rebuilds itself at ``hidden_size // num_attention_heads``.
        config["head_dim"] = PINNED_HEAD_DIM
    if keep_ff is not None:
        config["intermediate_size"] = keep_ff
        config["block_ff_dim"] = keep_ff
    if keep_groups is not None:
        group_size = int(parent_config["num_attention_heads"]) // int(
            parent_config["num_key_value_heads"]
        )
        config["num_attention_heads"] = keep_groups * group_size
        config["num_heads"] = keep_groups * group_size
        config["num_key_value_heads"] = keep_groups
        # MANDATORY, and it is the whole trap on this axis: this parent's config
        # carries no `head_dim`, and `Lfm2Attention` falls back to
        # `hidden_size // num_attention_heads`.  Leave it out at 12 heads and
        # every attention block builds itself at head_dim 85 against tensors
        # stored at 64.  Written explicitly, the surviving heads keep the
        # parent's own width and its own RoPE ladder.
        config["head_dim"] = PINNED_HEAD_DIM
    if keep_head_dim is not None:
        config["head_dim"] = keep_head_dim
        config["rope_parameters"] = dict(config["rope_parameters"])
        config["rope_parameters"]["rope_theta"] = rope_theta
    if excised:
        types = [t for i, t in enumerate(parent_config["layer_types"]) if i not in excised]
        config["layer_types"] = types
        config["num_hidden_layers"] = len(types)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(config, indent=2))
    save_file(state, str(outdir / "model.safetensors"), metadata={"format": "pt"})
    for side in SIDECARS:
        if (PARENT / side).is_file():
            # The parent's sidecars are read-only and `copy2` preserves the mode,
            # so a rebuild into an existing child directory fails on the second
            # copy with EPERM -- after the tensors have already been written.
            # Remove first: the child directory is this script's to own.
            (outdir / side).unlink(missing_ok=True)
            shutil.copy2(PARENT / side, outdir / side)
    return sum(int(v.numel()) for v in state.values())


# ---------------------------------------------------------------- width build
def identity_gate_report(outdir: Path) -> dict[str, Any]:
    """At ``E = I`` and width 1024 the whole fold must be the identity, in BYTES.

    Not a metaphor for one: every tensor the child exports is compared with the
    parent's own, and the gate fails on a single differing byte.  It is provable
    rather than lucky.  The reader fold at identity is ``W diag(g) diag(1/g_c)``
    with ``g_c = g``; both factors are bf16 values, so their product is exact in
    fp64 (eight mantissa bits times eight fit in fifty-three) and dividing that
    exact product by its own exact factor returns the other factor exactly.  The
    writer fold is ``I^T W``, the embedding is ``W_emb I``, the coreset at
    ``k >= n`` is ``arange``, and the transported gain is ``g``.
    """

    from safetensors import safe_open

    report: dict[str, Any] = {"tensors": 0, "identical": 0, "differing": []}
    with safe_open(str(PARENT / "model.safetensors"), "pt") as parent_handle:
        with safe_open(str(outdir / "model.safetensors"), "pt") as child_handle:
            parent_names = set(parent_handle.keys())
            child_names = set(child_handle.keys())
            if parent_names != child_names:
                report["name_difference"] = sorted(parent_names ^ child_names)
            for name in sorted(parent_names & child_names):
                left = parent_handle.get_tensor(name)
                right = child_handle.get_tensor(name)
                report["tensors"] += 1
                if left.shape != right.shape:
                    report["differing"].append({"tensor": name, "reason": "shape"})
                    continue
                delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
                if float(delta.max()) == 0.0:
                    report["identical"] += 1
                else:
                    report["differing"].append(
                        {
                            "tensor": name,
                            "max_abs_delta": float(delta.max()),
                            "differing_elements": int((delta > 0).sum()),
                        }
                    )
    report["passed"] = (
        report["tensors"] > 0
        and not report["differing"]
        and "name_difference" not in report
    )
    return report


def build_width(args: argparse.Namespace, harness: Harness, started: float) -> int:
    """The ``hidden_size`` axis: narrow the residual stream and re-solve into it."""

    hidden = int(args.hidden_size)
    excised = sorted(int(x) for x in (args.excise or []))
    keep_groups = args.kv_groups
    outdir = Path(args.child)
    parent_config = json.loads((PARENT / "config.json").read_text())
    parent_hidden = int(parent_config["hidden_size"])
    if parent_hidden != PARENT_HIDDEN:
        raise SystemExit(f"the parent is {parent_hidden} wide, not {PARENT_HIDDEN}")
    if hidden > parent_hidden or hidden < 1:
        raise SystemExit(f"--hidden-size {hidden} is not a narrowing of {parent_hidden}")
    rope_theta = float(parent_config["rope_parameters"]["rope_theta"])

    # ---- the basis.  Either sealed by a previous `--mode basis` run and loaded
    # by digest, or computed here and written; every rung of one width and every
    # composition on top of it must be built against the same bytes.
    if args.basis_file is not None and Path(args.basis_file).is_file():
        basis, census = read_basis(Path(args.basis_file))
        if int(basis.shape[1]) != hidden or int(basis.shape[0]) != parent_hidden:
            raise SystemExit(
                f"{args.basis_file} is {tuple(basis.shape)}; this build wants "
                f"({parent_hidden}, {hidden})"
            )
        log(f"basis loaded: {args.basis_file} sha {census['basis_sha256'][:16]}")
        moments = None
    else:
        log("capturing the parent's residual stream at 29 endpoints")
        moments = residual_moments(harness, keep_rows=True)
        basis, census = build_basis(moments, hidden, args.basis)
        if args.basis_file is not None:
            write_basis(Path(args.basis_file), basis, census)
            log(f"basis written: {args.basis_file} sha {census['basis_sha256'][:16]}")
    if moments is None and args.drift:
        log("capturing the parent's residual stream for the drift diagnostic")
        moments = residual_moments(harness, keep_rows=True)
    parent_rows = None if moments is None else moments["rows"]

    child, report = width_axis(
        harness,
        basis,
        hidden=hidden,
        excised=excised,
        arm=args.arm,
        outdir=outdir,
        device=args.device,
        dtype=torch.bfloat16,
        keep_groups=keep_groups,
        kv_solve=args.kv_solve,
        kv_als=int(args.kv_als),
        kv_lambda_split=bool(args.kv_lambda_split),
        kv_identity_gate=bool(args.kv_identity_gate),
        kv_gate_probe=None if args.kv_gate_probe is None else Path(args.kv_gate_probe),
        fold_solve=bool(args.fold_solve),
        head_gain=bool(args.head_gain),
        parent_rows=parent_rows,
    )

    per_site = report["per_site"]
    merge_records = report["merge_records"]
    scored = per_site + merge_records
    heads = int(parent_config["num_attention_heads"])
    kv = int(parent_config["num_key_value_heads"])
    surviving = report["surviving"]
    attention_sites = [
        l for l in surviving if parent_config["layer_types"][l] == "full_attention"
    ]
    conv_sites = [l for l in surviving if parent_config["layer_types"][l] != "full_attention"]
    axes: list[dict[str, Any]] = [
        {
            "axis": "hidden_size",
            "kind": "residual",
            "sites": surviving,
            "config_fields": ["hidden_size", "block_dim", "conv_dim", "head_dim"],
            "selection": (
                f"one shared orthonormal basis E ({parent_hidden} x {hidden}), the top "
                f"eigenvectors of the {args.basis} second moment of the residual stream "
                "pooled over all 29 residual-write endpoints with each endpoint "
                "equalised to trace one; a per-boundary chart is not expressible in a "
                "stock child because the residual is a running sum with nowhere to put "
                "a chart transition.  The conv blocks' internal channels are a "
                "pivoted-Cholesky coreset of the parent's own, because Lfm2ShortConv "
                "gates channelwise and no rotation of that space is representable"
            ),
            "solve": (
                "readers folded EXACTLY as sqrt(1024/H) W diag(g) E diag(1/g_c) with "
                "g_c = diag(E^T diag(g) E) (the lift preserves the norm, so the parent's "
                "RMSNorm at the lifted row and the child's at the row differ by that one "
                "constant); writers folded as E^T W and re-solved against the child's own "
                "installed activations by the prior-anchored ridge under THE LAW; the "
                "final norm's diagonal solved in closed form against the tied head's Gram"
                if args.fold_solve
                else "closed form only: readers exact, writers projected, no re-solve"
            ),
            "evidence": (
                "the parent's own stream (stale)"
                if args.arm == "stale"
                else "sequentially conditioned capture through the partially built child"
            ),
            "parameters_removed": None,
            "basis": {
                "mode": census.get("mode"),
                "width": census.get("width"),
                "sha256": census.get("basis_sha256"),
                "path": census.get("basis_path"),
                "retained": census.get("retained"),
                "spectrum": census.get("spectrum"),
            },
            "reader_fold_gate": report["reader_fold_gate"],
            "final_norm_gain": report["head_gain"],
            "conv_coreset": {
                "sites": conv_sites,
                "channels_kept": hidden,
                "channels_dropped": parent_hidden - hidden,
                "channels_moved_vs_stale": report["conv_channels_moved_vs_stale"],
            },
            "heldout": {
                "median_cos": float(torch.tensor([r["cos"] for r in per_site]).median())
                if per_site
                else None,
                "min_cos": min((r["cos"] for r in per_site), default=None),
                "median_relL2": float(
                    torch.tensor([r["relL2"] for r in per_site]).median()
                )
                if per_site
                else None,
                "max_input_drift": max(
                    (r.get("input_drift_vs_parent", 0.0) for r in per_site), default=None
                ),
                "per_site": per_site,
            },
        }
    ]
    if keep_groups is not None:
        axes.append(
            {
                "axis": "num_attention_heads",
                "kind": "full_attention",
                "sites": [r["parent_layer"] for r in merge_records],
                "config_fields": [
                    "num_attention_heads",
                    "num_heads",
                    "num_key_value_heads",
                    "head_dim",
                ],
                "selection": (
                    f"whole KV groups merged in pairs, {kv} -> {keep_groups}, on the "
                    "ALREADY NARROWED block: the census, the gauge and the solve are "
                    "width-agnostic and read the child's own rows"
                ),
                "solve": (
                    "ADDENDUM 14 14.3 joint two-sided evidence-weighted solve"
                    if args.kv_solve == "joint"
                    else "wave 16: average in the sign gauge"
                )
                + (
                    "; the alternation's lambda selected on an internal split of the "
                    "fit rows" if args.kv_lambda_split else ""
                ),
                "solver_variant": args.kv_solve,
                "evidence": "conditioned capture",
                "parameters_removed": (
                    (heads - keep_groups * (heads // kv)) * PINNED_HEAD_DIM * hidden * 2
                    + 2 * (kv - keep_groups) * PINNED_HEAD_DIM * hidden
                )
                * len(merge_records),
                "heldout": {
                    "median_cos": float(
                        torch.tensor([r["cos"] for r in merge_records]).median()
                    ),
                    "min_cos": min(r["cos"] for r in merge_records),
                    "per_site": merge_records,
                },
            }
        )
    if excised:
        axes.append(
            {
                "axis": "num_hidden_layers",
                "kind": "block",
                "sites": excised,
                "config_fields": ["num_hidden_layers", "layer_types"],
                "selection": "F-EXCISE: whole blocks deleted",
                "solve": (
                    "absorbed by every downstream conditioned solve; the child is "
                    "INSTANTIATED from a config whose layer_types already omits the "
                    "excised blocks, so wave 18's mask defect cannot arise here -- "
                    "there is no live layer list to shift out from under a config one"
                ),
                "evidence": "conditioned capture",
                "parameters_removed": None,
                "heldout": {"note": "a deleted block has no local write to score"},
            }
        )

    child_parameters = export(
        child,
        outdir,
        keep_ff=None,
        keep_head_dim=None,
        keep_groups=keep_groups,
        keep_hidden=hidden,
        excised=excised,
        rope_theta=rope_theta,
    )
    predicted = predicted_parameters(hidden, excised=excised, keep_groups=keep_groups)
    if child_parameters != predicted:
        raise SystemExit(
            f"the exported child has {child_parameters} parameters; the axis's own "
            f"arithmetic says {predicted}"
        )
    worst = min(scored, key=lambda r: r["cos"]) if scored else None
    identity = identity_gate_report(outdir) if args.identity_gate else None
    if identity is not None:
        log(
            f"identity gate: {identity['identical']}/{identity['tensors']} tensors "
            f"bit-identical to the parent, passed={identity['passed']}"
        )
        if not identity["passed"] and args.identity_gate == "strict":
            raise SystemExit(f"identity gate FAILED: {identity['differing'][:4]}")
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "arm": args.arm,
        "keep_ff": None,
        "keep_head_dim": None,
        "keep_kv_groups": keep_groups,
        "keep_hidden_size": hidden,
        "head_dim_pinned": PINNED_HEAD_DIM,
        "kv_solve": None if keep_groups is None else args.kv_solve,
        "kv_als_sweeps": None if keep_groups is None else int(args.kv_als),
        "kv_lambda_split": None if keep_groups is None else bool(args.kv_lambda_split),
        "excised": excised,
        "rope_theta": rope_theta,
        "parent_parameters": PARENT_PARAMS,
        "child_parameters": child_parameters,
        "net_ratio": child_parameters / PARENT_PARAMS,
        "axes": axes,
        "identity_gate": identity,
        "passed_through": [
            axis
            for axis, used in (
                ("intermediate_size", False),
                ("head_dim", False),
                ("num_attention_heads", keep_groups is not None),
                ("num_hidden_layers", bool(excised)),
                ("hidden_size", True),
                ("vocab_size", False),
            )
            if not used
        ],
        "solver_path": (
            "mx19_build.fold_reader/fold_writer (closed-form basis fold, fp64 CPU)"
            + (
                " + mx15_build.law_abiding_solve (prior-anchored ridge on every writer)"
                if args.fold_solve
                else ""
            )
            + (" + mx19_build.solve_head_gain (final norm diagonal)" if args.head_gain else "")
            + (
                ""
                if keep_groups is None or args.kv_solve != "joint"
                else " + mx18_build.joint_relation_solve/law_abiding_stein"
            )
        ),
        "ridge_relative": RIDGE_REL,
        "condition_number": None,
        "prior_norm": None,
        "correction_norm": None,
        "clamp_engaged": any(r.get("lambda_relative", 0) > RIDGE_REL for r in scored),
        "requested_clamp_ratio": max(
            [r.get("free_solve_update_scale", 0.0) for r in scored] or [0.0]
        ),
        "allowed_clamp_ratio": 1.0,
        "condition_cap_saturated": False,
        "worst_site": (
            None
            if worst is None
            else {
                "endpoint": worst.get("endpoint", "attention.o"),
                "layer": worst.get("layer", worst.get("parent_layer")),
                "relL2": worst["relL2"],
                "cos": worst["cos"],
            }
        ),
        "evidence": {
            "trace_cache_key": TRACE.name,
            "split_authority": "matrix-campaign-capture-manifest",
            "fit_rows": 4751,
            "heldout_rows": 1584,
            "parent_model_forward_calls": harness.forward_calls + child.forward_calls,
            "conditioned": args.arm != "stale",
        },
        "residual_basis": census,
        "export_path": (
            "stock Lfm2ForCausalLM with "
            + ", ".join(
                part
                for part in (
                    f"hidden_size=block_dim=conv_dim={hidden}",
                    f"head_dim={PINNED_HEAD_DIM} (written explicitly: the fallback "
                    f"hidden_size//num_attention_heads would be {hidden // heads})",
                    None
                    if keep_groups is None
                    else (
                        f"num_attention_heads=num_heads={keep_groups * (heads // kv)}, "
                        f"num_key_value_heads={keep_groups}"
                    ),
                    None
                    if not excised
                    else f"num_hidden_layers={len(parent_config['layer_types']) - len(excised)}",
                )
                if part
            )
            + "; no trust_remote_code, no custom module, no hook"
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": round(time.time() - started, 1),
        "child_tensors_sha256": file_sha256(outdir / "model.safetensors"),
    }
    (outdir / "mx19-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    log(
        f"child {child_parameters:,} parameters = {receipt['net_ratio']:.6f}x "
        f"-> {outdir}  ({receipt['wall_time_seconds']:.0f}s)"
    )
    _ = attention_sites
    return 0


def predicted_parameters(
    hidden: int, *, excised: list[int], keep_groups: int | None
) -> int:
    """The axis's own arithmetic, checked against the loader's count at export.

    ``32*H^2 + 191541*H + 768`` at full depth and full head count, and it returns
    229,693,184 at H=1024.  A closed form that disagrees with the file is a
    closed form that is not describing this child.
    """

    parent_config = json.loads((PARENT / "config.json").read_text())
    types = parent_config["layer_types"]
    heads = int(parent_config["num_attention_heads"])
    kv = int(parent_config["num_key_value_heads"])
    groups = kv if keep_groups is None else int(keep_groups)
    query_heads = groups * (heads // kv)
    ff = int(parent_config["intermediate_size"])
    total = int(parent_config["vocab_size"]) * hidden + hidden
    for index, kind in enumerate(types):
        if index in excised:
            continue
        total += 2 * hidden + 3 * ff * hidden
        if kind == "full_attention":
            total += (
                2 * query_heads * PINNED_HEAD_DIM * hidden
                + 2 * groups * PINNED_HEAD_DIM * hidden
                + 2 * PINNED_HEAD_DIM
            )
        else:
            total += 4 * hidden * hidden + 3 * hidden
    return total


# ---------------------------------------------------------------------- build
def build(args: argparse.Namespace) -> int:
    torch.set_grad_enabled(False)
    started = time.time()
    keep_ff = args.ff
    keep_head_dim = args.head_dim
    keep_groups = args.kv_groups
    excised = sorted(int(x) for x in (args.excise or []))
    arm = args.arm
    outdir = Path(args.child)
    conditioned = arm != "stale"

    harness = Harness(PARENT, args.device, torch.bfloat16)
    if args.hidden_size is not None:
        # The residual axis cannot narrow the parent in place -- a stream of two
        # widths is not a model -- so it builds its child from a config and
        # folds into it.  Dispatched here so wave 18's path below is untouched.
        return build_width(args, harness, started)
    parent_layer_count = len(harness.layers())
    parent_config = json.loads((PARENT / "config.json").read_text())
    rope_theta = float(parent_config["rope_parameters"]["rope_theta"])

    mlp_modules = {
        f"L{l:02d}": harness.layers()[l].feed_forward for l in range(parent_layer_count)
    }
    attn_modules = {f"A{l:02d}": harness.layers()[l].self_attn for l in ATTENTION_LAYERS}
    pristine = {
        name: copy.deepcopy(module)
        for name, module in {**mlp_modules, **attn_modules}.items()
    }
    log("capturing the parent's own stream (one pass over 24 batches)")
    parent_rows = harness.capture({**mlp_modules, **attn_modules}, pristine=pristine)

    # The RoPE ladder is decided ONCE, before any attention block is folded, and
    # from every attention site pooled: `head_dim` and `rope_theta` are one config
    # field each, so a per-site choice is not expressible in a stock child.
    shim: ParentRope | None = None
    stride = None
    stride_census: dict[str, Any] = {}
    dims: torch.Tensor | None = None
    if keep_head_dim is not None:
        from transformers.models.lfm2.modeling_lfm2 import Lfm2RotaryEmbedding

        parent_head_dim = int(harness.layers()[ATTENTION_LAYERS[0]].self_attn.head_dim)
        pooled = torch.zeros(parent_head_dim // 2, dtype=torch.float64)
        heads = int(harness.model.config.num_attention_heads)
        kv_heads = int(harness.model.config.num_key_value_heads)
        group_size = heads // kv_heads
        for parent_layer in ATTENTION_LAYERS:
            attn = harness.layers()[parent_layer].self_attn
            X = parent_rows[f"A{parent_layer:02d}"]["fit"]
            q_all = X @ attn.q_proj.weight.data.to(torch.float64).cpu().T
            k_all = X @ attn.k_proj.weight.data.to(torch.float64).cpu().T
            for h in range(heads):
                g = h // group_size
                pooled += plane_energy(
                    q_all[:, h * parent_head_dim : (h + 1) * parent_head_dim],
                    k_all[:, g * parent_head_dim : (g + 1) * parent_head_dim],
                    parent_head_dim,
                )
        total = float(pooled.sum())
        options = rope_plane_sets(parent_head_dim, keep_head_dim)
        scored = {s: float(pooled[p].sum()) / total for s, p in options.items()}
        stride = max(scored, key=lambda s: scored[s])
        half = parent_head_dim // 2
        dims = torch.tensor(
            options[stride] + [p + half for p in options[stride]], dtype=torch.long
        )
        stride_census = {
            "plane_energy_share_by_stride": scored,
            "chosen_stride": stride,
            "retained_plane_energy": scored[stride],
            "plane_energy_share": [float(x / total) for x in pooled],
        }
        rope_theta = rope_theta ** (stride * keep_head_dim / parent_head_dim)
        shim = ParentRope(harness.model.model.rotary_emb)
        for parent_layer in ATTENTION_LAYERS:
            shim.attach(harness.layers()[parent_layer].self_attn)
        harness.model.config.head_dim = keep_head_dim
        harness.model.config.rope_parameters = dict(harness.model.config.rope_parameters)
        harness.model.config.rope_parameters["rope_theta"] = rope_theta
        harness.model.model.rotary_emb = Lfm2RotaryEmbedding(
            harness.model.config, device=torch.device(args.device)
        )
        log(
            f"RoPE ladder: stride {stride} keeps {scored[stride]:.4f} of the score "
            f"energy (shares {', '.join(f'{s}:{v:.4f}' for s, v in sorted(scored.items()))}); "
            f"rope_theta {rope_theta:g}"
        )

    if excised:
        for index in sorted(excised, reverse=True):
            del harness.model.model.layers[index]
        for position, layer in enumerate(harness.layers()):
            if getattr(layer, "is_attention_layer", False):
                layer.self_attn.layer_idx = position
            elif hasattr(layer, "conv"):
                layer.conv.layer_idx = position
        # MANDATORY, and it is the whole trap on this axis -- the excision
        # counterpart of the `head_dim` trap `export` documents.  `Lfm2Model`
        # chooses each block's mask by POSITION in a config list:
        #
        #     layer_mask = causal_mask if config.layer_types[i] == "full_attention"
        #                  else linear_attention
        #
        # Deleting a module without deleting its type leaves `i` addressing the
        # PARENT's fourteen, so every attention block that an excision shifts
        # down into a conv's slot is handed the padding mask -- which is None on
        # this corpus -- instead of the causal one, and attends to the future.
        # Measured by `mx18_mask_probe.py`: with block 13 alone excised nothing
        # shifts and the module's future mass is 0.0000 at all six sites; with
        # 13 and 1 excised every site shifts and 28.6% to 48.0% of the module's
        # attention mass lands on tokens that have not happened yet.  That is
        # what the wave-17 attention-recompute gate refused, and it was right.
        # The exported child was never affected -- `export` writes the child's
        # own `layer_types` -- but every solve conditioned on the live stream
        # was, which is the whole reason this line is here.
        harness.model.config.layer_types = [
            kind for i, kind in enumerate(parent_config["layer_types"]) if i not in excised
        ]
        harness.model.config.num_hidden_layers = len(harness.model.config.layer_types)
        log(
            f"excised blocks {excised}; child depth {len(harness.layers())}; "
            f"config.layer_types re-derived ({len(harness.model.config.layer_types)} "
            "entries) so the live mask follows the live block"
        )

    surviving = [l for l in range(parent_layer_count) if l not in excised]
    per_site: list[dict[str, Any]] = []
    attention_records: list[dict[str, Any]] = []
    merge_records: list[dict[str, Any]] = []

    for position, parent_layer in enumerate(surviving):
        layer = harness.layers()[position]
        name = f"L{parent_layer:02d}"

        if keep_groups is not None and parent_layer in ATTENTION_LAYERS:
            key = f"A{parent_layer:02d}"
            record = head_count_axis(
                harness,
                position,
                parent_layer,
                keep_groups,
                trajectory_target=(
                    (parent_rows[key]["local_fit"], parent_rows[key]["local_hld"])
                    if arm == "conditioned-trajectory"
                    else None
                ),
                solve_mode=args.kv_solve,
                als_sweeps=int(args.kv_als),
                identity_gate=bool(args.kv_identity_gate),
                lambda_split=bool(args.kv_lambda_split),
                gate_probe=(
                    None if args.kv_gate_probe is None else Path(args.kv_gate_probe)
                ),
            )
            merge_records.append(record)
            merged_rows = [m for m in record["census"]["merged"] if m["merged"]]
            log(
                f"  A{parent_layer:02d} kv-groups {keep_groups} ({args.kv_solve}) "
                f"merges {record['census']['chosen_merges']} -> relL2 "
                f"{record['relL2']:.4f} cos {record['cos']:.4f} "
                f"(prior-only cos {record['prior_only_cos']:.4f})"
            )
            for row in merged_rows:
                log(
                    f"    pair {row['cluster']} relation relL2 average "
                    f"{row['relation_relL2_average']:.4f}"
                    + (
                        ""
                        if "relation_relL2_joint" not in row
                        else f" -> joint {row['relation_relL2_joint']:.4f} "
                        f"(target {row['relation_relL2_joint_target']:.4f}, "
                        f"post-norm cos k "
                        f"{row['key_recovery']['post_norm_cos_vs_target']:.4f} q "
                        + " ".join(
                            f"{g['post_norm_cos_vs_target']:.4f}"
                            for g in row["query_recovery"]
                        )
                        + ")"
                    )
                )
            value = record.get("value_solve") or {}
            if value:
                log(
                    f"    value/out pair: gate "
                    f"{value.get('attention_recompute_relL2', float('nan')):.3e} "
                    f"sweeps {value.get('sweeps_run')} installed "
                    f"{value.get('installed')}"
                    + (
                        ""
                        if "fit_sse_improvement" not in value
                        else f" fit SSE {value['fit_sse_improvement']:+.4%}"
                    )
                    + (f"  REFUSED: {value['refused']}" if "refused" in value else "")
                )

        if keep_head_dim is not None and parent_layer in ATTENTION_LAYERS:
            key = f"A{parent_layer:02d}"
            record = head_dim_axis(
                harness,
                position,
                parent_layer,
                keep_head_dim,
                stride=int(stride),
                dims=dims,
                stride_census=stride_census,
                shim=shim,
                trajectory_target=(
                    (parent_rows[key]["local_fit"], parent_rows[key]["local_hld"])
                    if arm != "conditioned-local"
                    else None
                ),
            )
            attention_records.append(record)
            log(
                f"  A{parent_layer:02d} head_dim {keep_head_dim} stride "
                f"{record['census']['chosen_stride']} planes "
                f"{record['census']['retained_plane_energy']:.4f} v "
                f"{record['census']['v_retained_energy_median']:.4f} -> relL2 "
                f"{record['relL2']:.4f} cos {record['cos']:.4f} "
                f"(prior-only cos {record['prior_only_cos']:.4f})"
            )

        if keep_ff is None:
            continue

        w1 = pristine[name].w1.weight.data.to(torch.float64).cpu()
        w3 = pristine[name].w3.weight.data.to(torch.float64).cpu()
        w2 = pristine[name].w2.weight.data.to(torch.float64).cpu()
        if conditioned:
            captured = harness.capture(
                {name: layer.feed_forward}, pristine={name: pristine[name]}
            )
        else:
            captured = {name: parent_rows[name]}
        X, X_h = captured[name]["fit"], captured[name]["hld"]
        local, local_h = captured[name]["local_fit"], captured[name]["local_hld"]
        if arm == "conditioned-trajectory":
            Y, Y_h = parent_rows[name]["local_fit"], parent_rows[name]["local_hld"]
        else:
            Y, Y_h = local, local_h

        phi = swiglu(X, w1, w3)
        phi_h = swiglu(X_h, w1, w3)
        # The identity precondition, checked and not assumed (wave 14 §3): the
        # axis being selected on must be the gated intermediate w2 actually reads.
        identity = float((phi @ w2.T - local).norm() / local.norm())
        if identity > IDENTITY_TOLERANCE:
            raise SystemExit(
                f"L{parent_layer:02d}: post_swiglu @ w2^T is {identity:.3e} from the "
                "block's own output; the intermediate-axis identity does not hold "
                "and the fold is refused"
            )
        drift = float(
            (X - parent_rows[name]["fit"]).norm() / parent_rows[name]["fit"].norm()
        )
        sel = pivoted_columns(phi, keep_ff)
        w2p, guard = law_abiding_solve(phi[:, sel], Y, w2[:, sel])
        scores = write_scores(phi_h[:, sel] @ w2p.T, Y_h)
        # What the same solve looks like against the LOCAL target, always, so a
        # trajectory arm's harder objective is visible beside its own number.
        local_scores = write_scores(phi_h[:, sel] @ w2p.T, local_h)
        install_narrow_mlp(layer, keep_ff, sel, w2p, harness.dtype)
        per_site.append(
            {
                "layer": parent_layer,
                "position": position,
                "keep": keep_ff,
                "arm": arm,
                "input_drift_vs_parent": drift,
                "identity_check": identity,
                "selection_first8": [int(x) for x in sel[:8]],
                "vs_local_target_cos": local_scores["cos"],
                "vs_local_target_relL2": local_scores["relL2"],
                **guard,
                **scores,
            }
        )
        log(
            f"  L{parent_layer:02d} keep={keep_ff} drift {drift:.4f} relL2 "
            f"{scores['relL2']:.4f} cos {scores['cos']:.4f} (local cos "
            f"{local_scores['cos']:.4f}) update_scale {guard['update_scale']:.3f} "
            f"(lam {guard['lambda_relative']:g}, free "
            f"{guard['free_solve_update_scale']:.3f})"
        )

    axes: list[dict[str, Any]] = []
    if keep_ff is not None:
        axes.append(
            {
                "axis": "intermediate_size",
                "kind": "mlp",
                "sites": [r["layer"] for r in per_site],
                "config_fields": ["intermediate_size", "block_ff_dim"],
                "selection": "pivoted-Cholesky coreset of real channels; w1/w3 rows verbatim",
                "solve": (
                    f"prior-anchored ridge on w2, lambda ladder to update_scale<=1.0, arm={arm}"
                ),
                "evidence": (
                    "the parent's own stream (stale)"
                    if not conditioned
                    else "sequentially conditioned capture through the partially built child"
                ),
                "parameters_removed": 3 * 1024 * (2560 - keep_ff) * len(per_site),
                "heldout": {
                    "median_cos": float(torch.tensor([r["cos"] for r in per_site]).median()),
                    "min_cos": min(r["cos"] for r in per_site),
                    "median_relL2": float(
                        torch.tensor([r["relL2"] for r in per_site]).median()
                    ),
                    "max_input_drift": max(r["input_drift_vs_parent"] for r in per_site),
                    "per_site": per_site,
                },
            }
        )
    if keep_groups is not None:
        heads = int(parent_config["num_attention_heads"])
        kv = int(parent_config["num_key_value_heads"])
        head_dim = int(parent_config["hidden_size"]) // heads
        group_size = heads // kv
        keep_heads = keep_groups * group_size
        hidden = int(parent_config["hidden_size"])
        per_layer_removed = (
            (heads - keep_heads) * head_dim * hidden          # q_proj rows
            + 2 * (kv - keep_groups) * head_dim * hidden      # k_proj + v_proj rows
            + (heads - keep_heads) * head_dim * hidden        # out_proj columns
        )
        axes.append(
            {
                "axis": "num_attention_heads",
                "kind": "full_attention",
                "sites": [r["parent_layer"] for r in merge_records],
                "config_fields": [
                    "num_attention_heads",
                    "num_heads",
                    "num_key_value_heads",
                    "head_dim",
                ],
                "selection": (
                    f"whole KV groups merged in pairs, {kv} -> {keep_groups}; pairs "
                    "ordered per layer by a post-norm redundancy census under the "
                    "exact per-plane sign gauge (the only frame change that commutes "
                    "with RoPE AND with the shared per-head RMSNorm weight); "
                    "surviving heads keep the parent's head_dim and rope ladder"
                ),
                "solve": (
                    "ADDENDUM 14 14.3 joint two-sided evidence-weighted solve: q/k "
                    "solved together per RoPE plane against both parents' relation "
                    "(closed form, shared key across the group), weights recovered "
                    "by a direction-anchored law-abiding ridge onto wave 16's "
                    "average-in-gauge merge, then v and out_proj alternated as the "
                    "linear pair they are with the attention probabilities fixed"
                    + (
                        "; the alternation's lambda selected on an internal split "
                        "of the fit rows (12 batches solved, 6 scored) under THE "
                        "LAW's unchanged veto"
                        if args.kv_lambda_split
                        else ""
                    )
                    if args.kv_solve == "joint"
                    else "wave 16: average in the sign gauge, out_proj re-solved "
                    "against the installed context"
                ),
                "solver_variant": args.kv_solve,
                "evidence": (
                    "the parent's own stream (stale)" if not conditioned else "conditioned capture"
                ),
                "parameters_removed": per_layer_removed * len(merge_records),
                "heldout": {
                    "median_cos": float(
                        torch.tensor([r["cos"] for r in merge_records]).median()
                    ),
                    "min_cos": min(r["cos"] for r in merge_records),
                    "per_site": merge_records,
                },
            }
        )
    if keep_head_dim is not None:
        heads, kv = 16, 8
        per_layer_removed = (
            (1024 * 1024 - heads * keep_head_dim * 1024)
            + 2 * (512 * 1024 - kv * keep_head_dim * 1024)
            + (1024 * 1024 - 1024 * heads * keep_head_dim)
            + 2 * (64 - keep_head_dim)
        )
        axes.append(
            {
                "axis": "head_dim",
                "kind": "full_attention",
                "sites": [r["parent_layer"] for r in attention_records],
                "config_fields": ["head_dim", "rope_parameters.rope_theta"],
                "selection": (
                    f"whole RoPE planes at stride {stride} -- an exact subset of the "
                    f"parent's frequency ladder under rope_theta**"
                    f"{stride * keep_head_dim / 64:g}; v by per-KV-group "
                    "evidence-weighted LRA"
                ),
                "solve": "out_proj re-solved against the installed context",
                "evidence": (
                    "the parent's own stream (stale)" if not conditioned else "conditioned capture"
                ),
                "parameters_removed": per_layer_removed * len(attention_records),
                "heldout": {
                    "median_cos": float(
                        torch.tensor([r["cos"] for r in attention_records]).median()
                    ),
                    "min_cos": min(r["cos"] for r in attention_records),
                    "per_site": attention_records,
                },
            }
        )
    if excised:
        axes.append(
            {
                "axis": "num_hidden_layers",
                "kind": "block",
                "sites": excised,
                "config_fields": ["num_hidden_layers", "layer_types"],
                "selection": "F-EXCISE: whole blocks deleted",
                "solve": (
                    "absorbed by every downstream conditioned solve"
                    if conditioned
                    else "not absorbed: a stale arm re-solves nothing downstream"
                ),
                "evidence": "conditioned capture" if conditioned else "the parent's own stream",
                "parameters_removed": None,
                "heldout": {"note": "a deleted block has no local write to score"},
            }
        )

    child_parameters = export(
        harness,
        outdir,
        keep_ff=keep_ff,
        keep_head_dim=keep_head_dim,
        keep_groups=keep_groups,
        keep_hidden=None,
        excised=excised,
        rope_theta=rope_theta,
    )
    scored = per_site + attention_records + merge_records
    worst = min(scored, key=lambda r: r["cos"]) if scored else None
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "arm": arm,
        "keep_ff": keep_ff,
        "keep_head_dim": keep_head_dim,
        "keep_kv_groups": keep_groups,
        "kv_solve": None if keep_groups is None else args.kv_solve,
        "kv_als_sweeps": None if keep_groups is None else int(args.kv_als),
        "kv_lambda_split": (
            None if keep_groups is None else bool(args.kv_lambda_split)
        ),
        "excised": excised,
        "rope_theta": rope_theta,
        "parent_parameters": PARENT_PARAMS,
        "child_parameters": child_parameters,
        "net_ratio": child_parameters / PARENT_PARAMS,
        "axes": axes,
        "passed_through": [
            axis
            for axis, used in (
                ("intermediate_size", keep_ff is not None),
                ("head_dim", keep_head_dim is not None),
                ("num_attention_heads", keep_groups is not None),
                ("num_hidden_layers", bool(excised)),
                ("hidden_size", False),
                ("vocab_size", False),
                ("conv_dim (not a live field: Lfm2ShortConv hardwires hidden_size)", False),
            )
            if not used
        ],
        "solver_path": (
            "mx15_build.law_abiding_solve (prior-anchored ridge, fp64 CPU)"
            + ("" if keep_groups is None or args.kv_solve != "joint" else
               " + mx18_build.joint_relation_solve/law_abiding_stein (T7 joint "
               "two-sided, fp64 CPU)")
            + ("" if keep_groups is None or not args.kv_lambda_split else
               " + mx18_build.law_abiding_stein_split/law_abiding_solve_split "
               "(lambda by internal fit split, THE LAW still vetoing)")
        ),
        "ridge_relative": RIDGE_REL,
        "condition_number": None,
        "prior_norm": None,
        "correction_norm": None,
        "clamp_engaged": any(r["lambda_relative"] > RIDGE_REL for r in scored),
        "requested_clamp_ratio": max([r["free_solve_update_scale"] for r in scored] or [0.0]),
        "allowed_clamp_ratio": 1.0,
        "condition_cap_saturated": False,
        "worst_site": (
            None
            if worst is None
            else {
                "endpoint": "mlp.down" if "keep" in worst else "attention.o",
                "layer": worst.get("layer", worst.get("parent_layer")),
                "relL2": worst["relL2"],
                "cos": worst["cos"],
            }
        ),
        "evidence": {
            "trace_cache_key": TRACE.name,
            "split_authority": "matrix-campaign-capture-manifest",
            "fit_rows": 4751,
            "heldout_rows": 1584,
            "parent_model_forward_calls": harness.forward_calls,
            "conditioned": conditioned,
        },
        "export_path": (
            "stock Lfm2ForCausalLM with "
            + ", ".join(
                part
                for part in (
                    None if keep_ff is None else f"intermediate_size=block_ff_dim={keep_ff}",
                    None
                    if keep_head_dim is None
                    else f"head_dim={keep_head_dim}, rope_theta={rope_theta:g}",
                    None
                    if keep_groups is None
                    else (
                        f"num_attention_heads=num_heads="
                        f"{keep_groups * (16 // 8)}, num_key_value_heads={keep_groups}, "
                        f"head_dim=64"
                    ),
                    None
                    if not excised
                    else f"num_hidden_layers={parent_layer_count - len(excised)}",
                )
                if part
            )
            + "; no trust_remote_code, no custom module, no hook"
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_seconds": round(time.time() - started, 1),
        "child_tensors_sha256": file_sha256(outdir / "model.safetensors"),
    }
    (outdir / "mx19-build-receipt.json").write_text(json.dumps(receipt, indent=1))
    log(
        f"child {child_parameters:,} parameters = {receipt['net_ratio']:.6f}x "
        f"-> {outdir}  ({receipt['wall_time_seconds']:.0f}s)"
    )
    return 0


def verify(child: Path) -> int:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(child), dtype=torch.float32)
    n = sum(p.numel() for p in model.parameters())
    print(
        json.dumps(
            {
                "class": type(model).__name__,
                "parameters": int(n),
                "net_ratio": n / PARENT_PARAMS,
                "hidden_size": int(model.config.hidden_size),
                "intermediate_size": int(model.config.intermediate_size),
                "num_hidden_layers": int(model.config.num_hidden_layers),
                "num_attention_heads": int(model.config.num_attention_heads),
                "num_key_value_heads": int(model.config.num_key_value_heads),
                "head_dim": getattr(model.config, "head_dim", None),
                "attention_head_dim_live": [
                    int(layer.self_attn.head_dim)
                    for layer in model.model.layers
                    if hasattr(layer, "self_attn")
                ],
                "rope_theta": model.config.rope_parameters["rope_theta"],
            },
            indent=1,
        )
    )
    return 0


def basis_mode(args: argparse.Namespace) -> int:
    """The free go/no-go census: how much of the residual lives in a subspace.

    One pass over the corpus, both second moments, every candidate width priced
    on the fit rows AND on the held-out rows, and -- if a width is named -- the
    basis itself written as a sealed artifact so every child at that width and
    every composition on top of it is built against the same bytes.  It builds
    nothing and decides nothing; the anti-proxy rule says a census is not a
    result and the ladder runs whatever this says.
    """

    torch.set_grad_enabled(False)
    harness = Harness(PARENT, args.device, torch.bfloat16)
    moments = residual_moments(harness, keep_rows=False)
    width = int(args.hidden_size or harness.model.config.hidden_size)
    basis, census = build_basis(moments, width, args.basis)
    census["parent_forward_calls"] = harness.forward_calls
    census["endpoints"] = list(moments["endpoints"])
    if args.basis_file is not None:
        write_basis(Path(args.basis_file), basis, census)
        log(f"basis -> {args.basis_file} sha {census['basis_sha256'][:16]}")
    print(json.dumps(census, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["parity", "build", "verify", "basis"])
    ap.add_argument(
        "--arm",
        default="conditioned-local",
        choices=["stale", "conditioned-local", "conditioned-trajectory"],
    )
    ap.add_argument("--ff", type=int)
    ap.add_argument("--head-dim", type=int)
    ap.add_argument(
        "--kv-groups",
        type=int,
        help=(
            "merge whole attention heads down to this many KV groups (parent 8); "
            "num_attention_heads becomes 2x this, head_dim stays 64"
        ),
    )
    ap.add_argument(
        "--kv-solve",
        default="joint",
        choices=["average", "joint"],
        help=(
            "how a merged head pair is decided: `average` is wave 16's "
            "average-in-gauge merge (the baseline ARCH.KV6 scored 5/15 with), "
            "`joint` is ADDENDUM 14 14.3's two-sided evidence-weighted solve"
        ),
    )
    ap.add_argument(
        "--kv-als",
        type=int,
        default=1,
        help=(
            "sweeps of the v|out_proj alternation with the attention "
            "probabilities held fixed; 0 leaves wave 16's contraction in place"
        ),
    )
    ap.add_argument(
        "--kv-lambda-split",
        action="store_true",
        help=(
            "choose the value/out_proj alternation's lambda on an internal split "
            "of the FIT rows instead of on THE LAW's trust region alone -- "
            "`NOTES-MX-U` section 7's named fix for the measured overfit.  THE "
            "LAW still vetoes; it no longer selects.  The campaign's held-out "
            "block is not read by the selection"
        ),
    )
    ap.add_argument(
        "--kv-gate-probe",
        type=Path,
        help=(
            "write the falsifying measurement for the attention-recompute gate's "
            "unattributed refusal under two excisions (`NOTES-MX-U` section 7) to "
            "this directory: the module's own attention weights against the fp64 "
            "recomputation, the peakedness and top-two margin of both, and the "
            "same gate in bf16.  Installs nothing and changes no weight"
        ),
    )
    ap.add_argument(
        "--kv-identity-gate",
        action="store_true",
        help=(
            "allow --kv-groups to equal the parent's own KV head count and "
            "assert the merge path is a bit-exact identity at zero merges"
        ),
    )
    ap.add_argument(
        "--hidden-size",
        type=int,
        help=(
            "narrow the RESIDUAL STREAM to this width -- the only standard axis "
            "that reaches the tied embedding.  head_dim is pinned at 64 and "
            "written explicitly, so the attention block keeps the parent's own "
            "RoPE ladder and this axis moves one thing"
        ),
    )
    ap.add_argument(
        "--basis",
        default="reader-pca",
        choices=list(RESIDUAL_BASIS_MODES),
        help=(
            "which second moment of the residual the basis diagonalises: "
            "`reader-pca` (post-norm, gain-scaled -- the rows a consumer sees), "
            "`evidence-pca` (the raw residual), or `identity` (the gate)"
        ),
    )
    ap.add_argument(
        "--basis-file",
        type=Path,
        help=(
            "read the basis from this artifact if it exists, otherwise compute "
            "and write it here.  One width, one basis, one digest, quoted in "
            "every receipt built against it"
        ),
    )
    ap.add_argument(
        "--no-fold-solve",
        dest="fold_solve",
        action="store_false",
        help=(
            "install the closed-form fold and re-solve nothing.  The prior IS "
            "the L2-optimal projection, so this is a real control and not a "
            "crippled build"
        ),
    )
    ap.add_argument(
        "--no-head-gain",
        dest="head_gain",
        action="store_false",
        help="leave the final norm at the transported gain instead of solving it",
    )
    ap.add_argument(
        "--no-drift",
        dest="drift",
        action="store_false",
        help=(
            "skip the parent-stream capture that prices per-site drift.  The "
            "drift numbers are a diagnostic; the solve never reads them"
        ),
    )
    ap.add_argument(
        "--identity-gate",
        choices=["strict", "report"],
        help=(
            "after export, compare every tensor with the parent's own BYTES.  "
            "Meaningful at --hidden-size 1024 --basis identity, where the fold "
            "is provably the identity; `strict` refuses on any difference"
        ),
    )
    ap.add_argument("--excise", type=int, action="append")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.set_defaults(fold_solve=True, head_gain=True, drift=True)
    args = ap.parse_args()
    if args.mode == "parity":
        print(json.dumps(parity(args.device), indent=1))
        return 0
    if args.mode == "verify":
        return verify(args.child)
    if args.mode == "basis":
        return basis_mode(args)
    if (
        args.ff is None
        and args.head_dim is None
        and args.kv_groups is None
        and args.hidden_size is None
        and not args.excise
    ):
        raise SystemExit("a build must move at least one axis")
    if args.hidden_size is not None and (args.ff is not None or args.head_dim is not None):
        raise SystemExit(
            "--hidden-size already re-solves every reader and every writer in the "
            "model against a new residual basis; composing --ff or --head-dim on "
            "top of it in one build would run a second selection inside tensors "
            "the first one has already replaced, with neither axis's census "
            "having seen the other.  Compose it with --excise or --kv-groups, "
            "which are the two axes whose machinery is width-agnostic"
        )
    if args.kv_groups is not None and args.head_dim is not None:
        raise SystemExit(
            "--kv-groups and --head-dim both rewrite the attention block and the "
            "second would be solved against the first's installed context without "
            "either axis's census having seen it; wave 15 sealed --head-dim dead "
            "alone (ARCH.HD16, 0/15), so this wave runs the head-count axis only"
        )
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
