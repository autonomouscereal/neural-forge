#!/usr/bin/env python3
"""XCHANGE -- the GO-GATE CENSUS, `docs/ROUND3_SLATE_2026-08-28.md` §3.3.

**THIS MODULE BUILDS NOTHING.**  It is the near-free census the slate puts in
front of the conv-to-attention substitution, and it exists because the
substitution is expensive -- a new operator KIND, the first in 988 ledger rows --
while the three things that decide whether it is EXPRESSIBLE at all can be
measured on the parent in one capture pass and a page of arithmetic.

**AND ITS PRE-REGISTRATION SAYS IT MAY CLOSE THE CLASS.**  A census normally
publishes and never decides; that is the campaign's standing rule and every other
census in this tree obeys it.  This one is the exception, deliberately and in
writing, because the three quantities below are not evidence ABOUT the operator's
quality -- they are the preconditions for the operator being in the class the
substitution installs.  If a conv site's delta is mostly outside the linear-value
attention class then no solve, no ridge and no head budget puts it back; the
design document (`docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md`
§B.2) says so in the word "provably".  The registration in this wave's notes
names the thresholds BEFORE the numbers exist, states that a failure at EVERY
surviving conv site closes the class, and states that a pass at ANY site funds
one build at the cheapest passing site.

WHAT IS MEASURED, and where each quantity comes from.

* **G1 `rho_2` -- the covariance remainder.**  §B.2 expands the ShortConv branch
  bilinearly about the evidence means of its two gates.  What is left over,
  ``eps_t = W_out [Cbar (*) (sum_l k_l (*) eps_B,t-l) + eps_C,t]``, is QUADRATIC
  in the input where attention values are LINEAR in the key row, so it is not
  representable by the class at any parameter count.  ``rho_2`` is its share of
  the branch delta's energy, and ``sqrt(rho_2)`` is therefore a LOWER BOUND on
  the installed block's relative error at that site before one unit of solve
  error is counted.  Deletion's relative error at the same site is exactly 1.0.
* **G2 gate dispersion.**  §B.1's identity is EXACT when the gates are constant
  on the evidence support, and §B.2's expansion is the first-order correction
  when they are not.  The published quantity is scale-free and finite at a
  zero-mean channel: ``nd = Var_t[g] / E_t[g^2]``, the share of a gate channel's
  second moment that is NOT carried by the point the expansion is taken about.
* **G3 Stage-1 head feasibility.**  The slate's fix (b): a single-plane head
  ALIASES -- at plane 0 the score returns to cos = 0.9998 at an offset of 44
  positions -- so a lag head needs at least two incommensurate planes phase
  aligned at its own lag.  This is arithmetic on the parent's RoPE ladder and
  needs no model at all, except for one number: the score magnitude the
  construction is allowed to assume.  That number is taken from the PARENT's own
  measured pre-softmax logits rather than chosen, because a constructed head at
  the same width and the same normalisation has no claim to a larger one.

THE RECONSTRUCTION GATE IS NOT OPTIONAL.  Every number here is a decomposition of
the conv branch's own output into a part the attention class can carry and a part
it cannot.  A decomposition whose parts do not sum back to the thing decomposed
is arithmetic about nothing, so the sum is asserted against the fp64
recomputation, and the fp64 recomputation is itself scored against the live bf16
module's captured value so that "this is the operator the model runs" is a
measured statement rather than an assumption about a chunk order or a tap order.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import mx45_build

PARENT = mx45_build.PARENT
SCHEMA = "neural-forge-matrix-wave45-xchange-go-gate-v1"
#: Every conv site in the parent.  `layer_types` is read back and compared, so a
#: parent whose block layout is not this one refuses rather than quietly
#: measuring different modules than the ones this file names.
CONV_LAYERS = (0, 1, 3, 5, 7, 9, 11, 13)
#: The slate's own go-gate cell and its value cell, named here so the artifact
#: says which sites the verdict is about.
GO_GATE_SITE = 13
VALUE_SITE = 9
#: The window the Stage-1 concentration check runs over.  The slate says a 4k
#: grid; the sealed evidence sequences are 264 positions, so this is fifteen
#: times any prompt the frozen surface can present and the check is deliberately
#: harder than the deployment.
STAGE1_WINDOW = 4096
#: The plane sets the Stage-1 check tries.  Two is the slate's minimum; the
#: others are there so that a failure at two is not mistaken for a failure of the
#: idea.  Planes are indexed as the parent indexes them: plane p rotates at
#: theta^(-2p/64), so a LOW p is a HIGH frequency.
STAGE1_PLANE_SETS = {
    "p0-p1": [0, 1],
    "p0-p1-p2": [0, 1, 2],
    "p0-p3": [0, 3],
    "p0-p1-p2-p3": [0, 1, 2, 3],
    "p0-p2-p4-p6": [0, 2, 4, 6],
    "p0..p7": [0, 1, 2, 3, 4, 5, 6, 7],
}
SINGLE_PLANE = "p0 (single, the slate's alias case)"

#: THE THRESHOLDS, IN ONE TABLE, DECLARED BEFORE THE CENSUS RUNS.  A go-gate
#: whose numbers are spread through the code is a go-gate that can be moved after
#: the measurement without anybody noticing.  Each entry carries the reason its
#: number is what it is; none is tuned to an outcome, because when they were
#: written no outcome existed.
GO_GATE = {
    "rho_2_max": (
        0.50,
        "rho_2 is the share of the conv branch's delta that is PROVABLY outside "
        "the linear-value attention class, so sqrt(rho_2) is a lower bound on "
        "the installed block's relative error at the site before any solve error "
        "is added.  Deletion's relative error at the same site is exactly 1.0.  "
        "At rho_2 = 0.50 the substitution's error FLOOR is 0.707 -- within thirty "
        "per cent of simply deleting the block -- while costing 3,145,856 "
        "parameters that deletion does not spend.  Above that line the operator "
        "cannot be argued to be carrying the block's function, and the slate's "
        "own deleted falsifier (that substitution preserves strictly more than "
        "deletion) is the only thing that would have excused building it",
    ),
    "gate_dispersion_max": (
        0.50,
        "nd = Var[g] / E[g^2] is the share of a gate channel's second moment NOT "
        "carried by the point the bilinear expansion is taken about.  At nd = "
        "0.50 the expansion point and the fluctuation carry equal energy, which "
        "is exactly where 'expand about the mean' stops being a leading-order "
        "statement.  Taken as the MEDIAN over the 1,024 channels, so a handful "
        "of wild channels cannot fail a site on their own and a broadly "
        "fluctuating gate cannot pass",
    ),
    "stage1_band_mass_min": (
        0.90,
        "a lag head's whole job is to put its probability on ONE relative offset, "
        "because the per-lag value maps M_l it feeds are per-lag.  A head with "
        "more than a tenth of its mass off its assigned lag is not a lag head, "
        "it is a smear, and the design document routes that leak downstream "
        "rather than absorbing it",
    ),
    "reconstruction_rtol": (
        1e-4,
        "the linear part plus the covariance remainder must sum back to the fp64 "
        "recomputation of the branch delta.  This is not a fit tolerance -- the "
        "identity is algebraic and holds to fp64 rounding -- it is set four "
        "orders above that floor and many orders below any chunk-order or "
        "tap-order error, so it catches a wrong decomposition and nothing else",
    ),
}
#: Stated in the artifact and not only in the wave's notes, because a reader who
#: has only the artifact must be able to see what this file was allowed to do.
AUTHORITY = (
    "this census MAY close the XCHANGE class"
    ", by its own pre-registration, and only in one direction: if EVERY "
    "surviving conv site fails the conjunction of G1, G2 and G3 then the class "
    "closes, because the three gates are preconditions for the substituted "
    "operator being in the class at all and no solve restores what is provably "
    "outside it.  A pass at ANY site funds ONE build at the cheapest passing "
    "site.  This census never grades a child and never moves a floor."
)


def log(message: str) -> None:
    print(f"[mx45xchange] {message}", flush=True)


# ------------------------------------------------------------------- capture
@torch.no_grad()
def capture_conv(harness: Any, layers: tuple[int, ...]) -> dict[int, list[dict[str, Any]]]:
    """The conv branch's INPUT and its pre-``out_proj`` value, grouped by sequence.

    Both are needed and for different reasons: the input is what the whole
    decomposition is a function of, and the pre-``out_proj`` value is the live
    bf16 module's own answer, which is what the fp64 recomputation is scored
    against.  Grouped by sequence because a causal FIR must not run across a
    batch boundary.
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

    for layer in layers:
        conv = harness.layers()[layer].conv
        handles.append(conv.register_forward_pre_hook(make_hook(f"n{layer}"), with_kwargs=True))
        handles.append(
            conv.out_proj.register_forward_pre_hook(make_hook(f"y{layer}"), with_kwargs=True)
        )
    out: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    try:
        for index, ids in harness.batches:
            split = "fit" if harness.split[index] == "fit" else "hld"
            harness.model(input_ids=ids.to(harness.device), use_cache=False)
            harness.forward_calls += 1
            for layer in layers:
                out[layer].append({
                    "batch": int(index), "split": split,
                    "n": state[f"n{layer}"], "y": state[f"y{layer}"],
                })
            state.clear()
    finally:
        for handle in handles:
            handle.remove()
    return out


# ------------------------------------------------------------------ the parts
def causal_fir(u: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """``out[t] = sum_j taps[:, j] * u[t-j]``, zero before the sequence starts.

    ``taps[:, j]`` is the lag-``j`` coefficient.  `Lfm2ShortConv` runs an
    ``nn.Conv1d`` with ``padding = L-1`` and truncates to the sequence length, so
    its stored kernel index ``L-1-j`` is lag ``j``; the caller flips the stored
    kernel once and everything below stays in LAG order, which is the order every
    equation in the design document is written in.
    """

    out = u * taps[:, 0][None, :]
    for lag in range(1, taps.shape[1]):
        out[lag:] += u[:-lag] * taps[:, lag][None, :]
    return out


def assert_decomposition(delta_check: torch.Tensor, delta: torch.Tensor, where: str) -> None:
    """The linear part plus the remainder must BE the branch delta."""

    if GO_GATE["reconstruction_rtol"][0] != 1e-4:
        raise SystemExit(
            "the declared reconstruction tolerance is not the one in use; the "
            "threshold table has to be the thing that decides"
        )
    if not torch.allclose(delta_check, delta, rtol=1e-4, atol=1e-4):
        raise SystemExit(
            f"{where}: the linear part and the covariance remainder do not sum "
            "back to the branch delta, so the decomposition is not a "
            "decomposition of this operator"
        )


def remainder_share(eps: torch.Tensor, delta: torch.Tensor) -> float:
    """The share of one sequence's branch delta the attention class cannot carry.

    Published per batch as well as pooled, because a site whose ``rho_2`` is an
    average over one wild sequence and twenty-three quiet ones is a different
    finding from a site whose ``rho_2`` is flat, and a pooled number cannot tell
    those two apart.
    """

    if float(delta.pow(2).sum()) > 0.0:
        rho_2 = float(eps.pow(2).sum() / delta.pow(2).sum())
        return rho_2
    raise SystemExit(
        "a conv branch whose delta carries no energy on this sequence has no "
        "remainder share to take; that is a broken capture, not an operator"
    )


def site_census(blocks: list[dict[str, Any]], conv: Any, hidden: int) -> dict[str, Any]:
    """The §B.2 decomposition of one conv site, on the campaign's own evidence."""

    w_in = conv.in_proj.weight.data.to(torch.float64).cpu()
    w_out = conv.out_proj.weight.data.to(torch.float64).cpu()
    kernel = conv.conv.weight.data.to(torch.float64).cpu().squeeze(1)
    taps = torch.flip(kernel, dims=[1]).contiguous()

    def gates(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # `Lfm2ShortConv` transposes then chunks along the CHANNEL axis, so the
        # three streams are contiguous 1,024-column blocks of in_proj's output in
        # the order B, C, x.  The recomputation gate below is what proves this
        # reading of the chunk order rather than asserting it.
        bcx = n @ w_in.T
        return bcx[:, :hidden], bcx[:, hidden : 2 * hidden], bcx[:, 2 * hidden :]

    # ---- the expansion point: evidence means, taken on the FIT rows only.
    fit = [b for b in blocks if b["split"] == "fit"]
    sums = {k: torch.zeros(hidden, dtype=torch.float64) for k in ("b", "c", "x", "v", "b2", "c2")}
    rows = 0
    for block in fit:
        B, C, x = gates(block["n"])
        v = causal_fir(B * x, taps)
        sums["b"] += B.sum(dim=0); sums["c"] += C.sum(dim=0); sums["x"] += x.sum(dim=0)
        sums["b2"] += B.pow(2).sum(dim=0); sums["c2"] += C.pow(2).sum(dim=0)
        sums["v"] += v.sum(dim=0)
        rows += int(B.shape[0])
    b_bar, c_bar = sums["b"] / rows, sums["c"] / rows
    x_bar, v_bar = sums["x"] / rows, sums["v"] / rows
    nd_b = ((sums["b2"] / rows) - b_bar.pow(2)) / (sums["b2"] / rows).clamp_min(1e-300)
    nd_c = ((sums["c2"] / rows) - c_bar.pow(2)) / (sums["c2"] / rows).clamp_min(1e-300)

    # ---- the decomposition, per sequence, pooled per split.
    energy = {"fit": [0.0, 0.0], "hld": [0.0, 0.0]}
    per_batch: list[dict[str, Any]] = []
    recomputation: list[float] = []
    for block in blocks:
        n = block["n"]
        B, C, x = gates(n)
        u = B * x
        v = causal_fir(u, taps)
        y = C * v
        delta = y @ w_out.T
        u_lin = b_bar[None, :] * x + (B - b_bar[None, :]) * x_bar[None, :]
        v_lin = causal_fir(u_lin, taps)
        y_lin = c_bar[None, :] * v_lin + (C - c_bar[None, :]) * v_bar[None, :]
        eps_y = c_bar[None, :] * causal_fir(u - u_lin, taps) + (
            C - c_bar[None, :]
        ) * (v - v_bar[None, :])
        eps = eps_y @ w_out.T
        assert_decomposition((y_lin @ w_out.T) + eps, delta, f"batch {block['batch']}")
        recomputation.append(float((y - block["y"]).norm() / block["y"].norm()))
        per_batch.append({
            "batch": block["batch"], "split": block["split"],
            "rho_2": remainder_share(eps, delta),
        })
        energy[block["split"]][0] += float(delta.pow(2).sum())
        energy[block["split"]][1] += float(eps.pow(2).sum())

    rho_fit = energy["fit"][1] / energy["fit"][0]
    return {
        "rho_2_fit": rho_fit,
        "rho_2_held": energy["hld"][1] / energy["hld"][0],
        "error_floor_relL2_fit": rho_fit**0.5,
        "rho_2_per_batch": per_batch,
        "gate_dispersion_B_median": float(nd_b.median()),
        "gate_dispersion_C_median": float(nd_c.median()),
        "gate_dispersion_B_mean": float(nd_b.mean()),
        "gate_dispersion_C_mean": float(nd_c.mean()),
        "fp64_recomputation_relL2_max": max(recomputation),
        "fp64_recomputation_relL2_mean": sum(recomputation) / len(recomputation),
        "evidence_rows_fit": rows,
    }


def stage1_feasibility(theta: float, gamma_max: float) -> dict[str, Any]:
    """Can a multi-plane head concentrate on ONE relative offset at all?

    Arithmetic on the parent's RoPE ladder.  With constant pre-RoPE vectors
    supported on a set of planes and phase aligned at lag ``l``, the post-RoPE
    score is ``g(D) = sum_p cos(omega_p (D - l))``, which peaks at ``D = l``.  A
    SINGLE plane aliases -- omega_0 = 1, so ``g`` returns to 0.9998 at an offset
    of 44 -- and that is the slate's own objection, checked here rather than
    asserted.  The gamma sweep is capped by the parent's own measured logit
    magnitude, so this cannot pass by assuming a score the class cannot reach.
    """

    results: dict[str, Any] = {}
    offsets = torch.arange(STAGE1_WINDOW, dtype=torch.float64)
    candidates = {SINGLE_PLANE: [0]}
    candidates.update(STAGE1_PLANE_SETS)
    for name, planes in candidates.items():
        omega = torch.tensor(
            [theta ** (-2.0 * p / 64) for p in planes], dtype=torch.float64
        )
        per_lag: dict[str, Any] = {}
        for lag in (0, 1, 2):
            g = torch.cos(omega[None, :] * (offsets[:, None] - lag)).sum(dim=1) / len(planes)
            competitor = float(torch.cat([g[:lag], g[lag + 1 :]]).max())
            best_mass, best_gamma = 0.0, 0.0
            for gamma in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, gamma_max):
                if gamma > gamma_max:
                    continue
                mass = float(torch.softmax(gamma * g, dim=0)[lag])
                if mass > best_mass:
                    best_mass, best_gamma = mass, gamma
            per_lag[str(lag)] = {
                "peak": float(g[lag]),
                "largest_competitor": competitor,
                "margin": float(g[lag]) - competitor,
                "best_band_mass": best_mass,
                "at_gamma": best_gamma,
            }
        results[name] = {
            "planes": planes,
            "per_lag": per_lag,
            "worst_band_mass": min(v["best_band_mass"] for v in per_lag.values()),
        }
    return results


@torch.no_grad()
def parent_logit_scale(harness: Any) -> float:
    """The parent's own pre-softmax logit magnitude, as the cap on Stage 1's gamma.

    The Stage-1 construction needs a score scale, and CHOOSING one would let the
    check pass by assumption.  This measures what the parent's attention actually
    reaches at the same head width, the same layernorms and the same softmax
    scaling, on the campaign's own evidence.  The statistic is the MEDIAN over
    (site, sequence, head) of each block's own 99th percentile of ``|z|``: one
    pooled quantile over sixty million causal pairs is both a large allocation
    and a number one outlier head can set on its own.
    """

    import mx45_plane

    layers = mx45_build.ATTENTION_LAYERS
    targets = {f"A{layer:02d}": harness.layers()[layer].self_attn for layer in layers}
    captured = mx45_plane.capture_by_batch(harness, targets)
    heads = int(harness.model.config.num_attention_heads)
    kv_heads = int(harness.model.config.num_key_value_heads)
    group_size = heads // kv_heads
    eps = float(harness.model.config.norm_eps)
    theta = float(harness.model.config.rope_parameters["rope_theta"])
    every = torch.arange(64, dtype=torch.long)
    scale = 64**-0.5
    marks: list[float] = []
    for layer in layers:
        attn = harness.layers()[layer].self_attn
        wq = attn.q_proj.weight.data.to(torch.float64).cpu()
        wk = attn.k_proj.weight.data.to(torch.float64).cpu()
        gq = attn.q_layernorm.weight.data.to(torch.float64).cpu()
        gk = attn.k_layernorm.weight.data.to(torch.float64).cpu()
        for block in captured[f"A{layer:02d}"]:
            if block["split"] != "fit":
                continue
            X = block["rows"]
            length = X.shape[0]
            cos, sin = mx45_plane.rope_angles(list(range(32)), length, theta)
            mask = torch.tril(torch.ones(length, length, dtype=torch.bool))
            q_bands = mx45_plane.head_bands(X, wq, gq, every, heads, eps)
            k_bands = mx45_plane.head_bands(X, wk, gk, every, kv_heads, eps)
            for h in range(heads):
                z = scale * (
                    mx45_plane.rope_apply(q_bands[h][0], cos, sin)
                    @ mx45_plane.rope_apply(k_bands[h // group_size][0], cos, sin).T
                )
                pairs = z[mask].abs()
                marks.append(
                    float(torch.quantile(pairs, torch.tensor(0.99, dtype=pairs.dtype)))
                )
    return float(torch.tensor(marks).median())


# ---------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    started = time.time()
    parent_config = json.loads((PARENT / "config.json").read_text())
    hidden = int(parent_config["hidden_size"])
    theta = float(parent_config["rope_parameters"]["rope_theta"])
    declared = tuple(
        i for i, t in enumerate(parent_config["layer_types"]) if t != "full_attention"
    )
    if declared != CONV_LAYERS:
        raise SystemExit(
            f"this parent's conv sites are {declared}, not {CONV_LAYERS}; the census "
            "would be measuring different modules than the ones it names"
        )
    harness = mx45_build.Harness(PARENT, args.device, torch.bfloat16)

    gamma_max = parent_logit_scale(harness)
    log(f"gamma cap from the parent's own pre-softmax logits: {gamma_max:.4f}")

    blocks = capture_conv(harness, CONV_LAYERS)
    sites: dict[str, Any] = {}
    for layer in CONV_LAYERS:
        row = site_census(blocks[layer], harness.layers()[layer].conv, hidden)
        sites[f"L{layer:02d}"] = row
        log(
            f"  L{layer:02d} rho_2 fit {row['rho_2_fit']:.4f} held "
            f"{row['rho_2_held']:.4f} (error floor {row['error_floor_relL2_fit']:.4f}) "
            f"| dispersion B {row['gate_dispersion_B_median']:.4f} C "
            f"{row['gate_dispersion_C_median']:.4f} | fp64-vs-bf16 "
            f"{row['fp64_recomputation_relL2_max']:.2e}"
        )

    stage1 = stage1_feasibility(theta, gamma_max)
    multiplane = [v["worst_band_mass"] for v in stage1.values() if len(v["planes"]) >= 2]
    best_stage1 = max(multiplane)
    single_plane = stage1[SINGLE_PLANE]["worst_band_mass"]
    log(f"stage 1: best multi-plane worst-lag band mass {best_stage1:.4f}; "
        f"single plane {single_plane:.4f}")

    rho_max = GO_GATE["rho_2_max"][0]
    nd_max = GO_GATE["gate_dispersion_max"][0]
    mass_min = GO_GATE["stage1_band_mass_min"][0]
    g3 = bool(best_stage1 >= mass_min)
    per_site_verdict: dict[str, Any] = {}
    passing: list[int] = []
    for layer in CONV_LAYERS:
        row = sites[f"L{layer:02d}"]
        g1 = bool(row["rho_2_fit"] <= rho_max)
        g2 = bool(row["gate_dispersion_B_median"] <= nd_max
                  and row["gate_dispersion_C_median"] <= nd_max)
        verdict = bool(g1 and g2 and g3)
        per_site_verdict[f"L{layer:02d}"] = {
            "G1_rho_2": g1, "G2_gate_dispersion": g2, "G3_stage1": g3,
            "verdict": verdict,
        }
        if verdict:
            passing.append(layer)
        if verdict is False:
            log(f"  L{layer:02d} FAILS the conjunction: G1 {g1}  G2 {g2}  G3 {g3}")

    if passing:
        chosen = GO_GATE_SITE if GO_GATE_SITE in passing else passing[-1]
        consequence = (
            f"BUILD.  {len(passing)} of {len(CONV_LAYERS)} conv sites pass the "
            f"conjunction of G1, G2 and G3; the go-gate cell is block {chosen}"
            + ("." if chosen == GO_GATE_SITE else
               f", which is NOT the slate's block {GO_GATE_SITE}: that site failed "
               "and the registered contingency moves the cell to the cheapest "
               "passing site.")
        )
    else:
        consequence = (
            "CLASS CLOSED.  No surviving conv site passes the conjunction of G1, "
            "G2 and G3, so the substituted operator is not in the class at any "
            "site and no solve restores what is provably outside it.  XCHANGE "
            "closes by its own pre-registered gate."
        )
    log(consequence)

    artifact = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": AUTHORITY,
        "supersedes": (
            "docs/ATTENTION_SHORTCONV_TRANSLATION_DESIGN_2026-08-01.md (design "
            "only, never built) for its direction-B acceptance diagnostics"
        ),
        "thresholds": {k: {"value": v[0], "why": v[1]} for k, v in GO_GATE.items()},
        "evidence": mx45_build.evidence_receipt(),
        "parent": str(PARENT),
        "conv_sites": list(CONV_LAYERS),
        "go_gate_site": GO_GATE_SITE,
        "value_site": VALUE_SITE,
        "parent_logit_gamma_cap": gamma_max,
        "sites": sites,
        "stage1": stage1,
        "stage1_best_multiplane_worst_lag_band_mass": best_stage1,
        "stage1_single_plane_worst_lag_band_mass": single_plane,
        "per_site_verdict": per_site_verdict,
        "passing_sites": passing,
        "consequence": consequence,
        "forward_calls": harness.forward_calls,
        "seconds": time.time() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=1))
    reread = json.loads(args.out.read_text())
    assert reread["schema"] == SCHEMA, "the go-gate artifact did not survive the write"
    assert reread["consequence"] == consequence, "the consequence did not survive the write"
    assert len(reread["sites"]) == len(CONV_LAYERS), "a site fell out of the write"
    log(f"-> {args.out} ({len(sites)} sites, re-parsed OK)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
