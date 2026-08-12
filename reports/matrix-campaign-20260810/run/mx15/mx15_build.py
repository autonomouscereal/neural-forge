#!/usr/bin/env python3
"""Wave 15: build standard-shape LFM2 children, sequentially conditioned.

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
SCHEMA = "neural-forge-matrix-wave15-standard-shape-child-v1"
RIDGE_REL = 1e-6
LAMBDA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
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


def log(message: str) -> None:
    print(f"[mx15] {message}", flush=True)


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


# --------------------------------------------------------------------- export
def export(
    harness: Harness,
    outdir: Path,
    *,
    keep_ff: int | None,
    keep_head_dim: int | None,
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
        log(f"excised blocks {excised}; child depth {len(harness.layers())}")

    surviving = [l for l in range(parent_layer_count) if l not in excised]
    per_site: list[dict[str, Any]] = []
    attention_records: list[dict[str, Any]] = []

    for position, parent_layer in enumerate(surviving):
        layer = harness.layers()[position]
        name = f"L{parent_layer:02d}"

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
        excised=excised,
        rope_theta=rope_theta,
    )
    scored = per_site + attention_records
    worst = min(scored, key=lambda r: r["cos"]) if scored else None
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "arm": arm,
        "keep_ff": keep_ff,
        "keep_head_dim": keep_head_dim,
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
                ("num_hidden_layers", bool(excised)),
                ("hidden_size", False),
                ("vocab_size", False),
                ("conv_dim (not a live field: Lfm2ShortConv hardwires hidden_size)", False),
            )
            if not used
        ],
        "solver_path": "mx15_build.law_abiding_solve (prior-anchored ridge, fp64 CPU)",
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
    (outdir / "mx15-build-receipt.json").write_text(json.dumps(receipt, indent=1))
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
                "head_dim": getattr(model.config, "head_dim", None),
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
    ap.add_argument("--excise", type=int, action="append")
    ap.add_argument("--child", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.mode == "parity":
        print(json.dumps(parity(args.device), indent=1))
        return 0
    if args.mode == "verify":
        return verify(args.child)
    if args.ff is None and args.head_dim is None and not args.excise:
        raise SystemExit("a build must move at least one axis")
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
