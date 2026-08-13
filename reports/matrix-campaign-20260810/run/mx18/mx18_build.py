#!/usr/bin/env python3
"""Wave 18: exploit the distance law, and fix the lambda the value path overfits.

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
SCHEMA = "neural-forge-matrix-wave18-standard-shape-child-v1"
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
    print(f"[mx18] {message}", flush=True)


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


# --------------------------------------------------------------------- export
def export(
    harness: Harness,
    outdir: Path,
    *,
    keep_ff: int | None,
    keep_head_dim: int | None,
    keep_groups: int | None,
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
        config["head_dim"] = int(parent_config["hidden_size"]) // int(
            parent_config["num_attention_heads"]
        )
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
    (outdir / "mx18-build-receipt.json").write_text(json.dumps(receipt, indent=1))
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["parity", "build", "verify"])
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
    ap.add_argument("--excise", type=int, action="append")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.mode == "parity":
        print(json.dumps(parity(args.device), indent=1))
        return 0
    if args.mode == "verify":
        return verify(args.child)
    if args.ff is None and args.head_dim is None and args.kv_groups is None and not args.excise:
        raise SystemExit("a build must move at least one axis")
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
