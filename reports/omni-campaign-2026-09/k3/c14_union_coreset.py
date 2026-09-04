#!/usr/bin/env python3
"""C14: the expert-union channel coreset -- 896 MoE experts -> one dense FFN.

The object being folded is the ROUTED half of a K3 MoE layer:

    R(X) = W_up . RMSNorm_g( sum_{e in top16(t)} w_e(t) . W2_e . SiTU(W1_e L, W3_e L) ),
    L = W_down X

Every expert is a 3072-channel SiTU FFN in the same 3584-wide latent space, and
`SiTU`'s constants are identical across experts, so the union of all expert
channels U = {(e, j)} (896 x 3072 = 2,752,512 of them) is ONE dense FFN's worth
of channels that happen to be gated by a router.  The fold keeps K of them
VERBATIM (channel coreset -- never an SVD, LFM2 law 40/42), refits only the
read-out W2', and DELETES the router.  W_down, routed_expert_norm, W_up and
both shared experts stay verbatim.

Two feature families, and the distinction is the whole design:

    phi_c(t) = w_e(t) . 1[e in top16(t)] . SiTU(W1_e[j].L_t, W3_e[j].L_t)
               -- what the parent WRITES through channel c (router mass in it).
    psi_c(t) = SiTU(W1_e[j].L_t, W3_e[j].L_t)
               -- what the dense child COMPUTES for every token.

Selection is ranked on phi (a channel that the router almost never opens is
not worth a slot); the refit design is psi on all fit tokens (that is what the
child will actually feed W2'); the target is `y_lat`, the PRE-norm fp32
accumulator, so `routed_expert_norm` and `W_up` remain exact.

Two-stage selection, mass-allocated:
  stage 1  per expert e: pivoted Cholesky on G_e = F_e^T F_e over the tokens
           routed to e, k_e = round(K_pool . mass_e / sum mass), floor 1.
  stage 2  re-pivot the pooled union down to K on a Gram accumulated in fp64
           over token chunks.  Skipped when K_pool == K (the rung policy for
           K = 65,536 is stage-1 only) -- and `--k-pool-factor 1` now makes
           stage 1 land on EXACTLY K (largest-remainder apportionment), with
           `run_rung` raising if stage 2 ran anyway, so that policy is
           enforced rather than hoped for.

The refit is `law_abiding_solve_split` in GRAM form: the same lambda ladder,
the same `update_scale <= 1` veto, lambda chosen on an internal fit split --
because at K = 65,536 the design matrix is 65,536 columns wide and only its
Gram fits anywhere.

Rungs and the storage they buy (routed half only, PARAMETERS, parent
29,595,009,024): 65,536 -> 704.6M (42.0x), 32,768 -> 352.3M (84.0x),
17,408 -> 187.2M (158.1x), 8,704 -> 93.6M (316.2x).  Quantization is forbidden
in a fold: everything here is bf16/fp32 and the packed parent is only ever a
storage format we read through.

WHAT A RUNG COSTS IN RAM, and it is the thing that decides which rungs are
reachable.  The refit is fp64 and quadratic in K (out_dim = 3,584):

    K        one Gram   resident (2G+2X)   solve transient   estimated peak
    8,704      0.61 GB      1.71 GB            1.71 GB           3.4 GB
    17,408     2.42 GB      5.85 GB            5.85 GB          11.7 GB
    32,768     8.59 GB     19.06 GB           19.06 GB          38.1 GB
    65,536    34.36 GB     72.48 GB           72.48 GB         145.0 GB

All of that is HOST memory, and it is host memory on purpose: every fp64
[K, K] / [K, out] accumulator is built after a `.cpu()`, because a [K, K] fp64
tensor on a 25.3 GB 3090 is 34.4 GB at K = 65,536 and `--max-gram-gb` prices
the wrong pool to catch it.

`device_memory_estimate` prices what lands on `--device`, and it prices the
RESIDENT terms as well as the per-chunk transients, because the second
adversarial review caught it pricing about 4% of the truth.  The resident
terms are the layer itself (measured, `LayerWeights.device_bytes`), the
calibration cache (`latent_in` / `y_lat` / the router top-k), the `ExpertBank`
of kept rows, `score`'s held-out buffers and, when `--form-m` is given, the
fp32 [K, hidden] lift.  Two of those are large and neither used to appear:

    --experts-device cuda:0 : 896 packed experts = 14.80 GB + 0.93 GB of E8M0
                              scales, i.e. 15.7 GB before a single token moves
    --form-m                : `bank.w1.float() @ down.float()` at K = 65,536 is
                              a [65536, 7168] fp32 lift = 1.88 GB plus its
                              operands

`--experts-device` therefore DEFAULTS TO `cpu`: the packed experts stay on the
host and only the rows a rung actually keeps are moved to `--device`.  That is
what makes the README's own K = 65,536 entry point fit on a 25.3 GB card at
all; `--experts-device cuda:0` still works and is now priced honestly, which
means `--max-device-gb` will refuse it with the arithmetic.

`gram_memory_estimate` prints exactly this and `--max-gram-gb` refuses a rung
that does not fit rather than dying in an allocator.  Three things buy the
difference against the first version, which needed ~205 GB at K = 32,768 alone:
the identity in `gram + lam * eye(n)` is gone, the full-fit Gram is derived
from inner + check by addition instead of accumulated separately, and the
lambda ladder no longer stores nine [3584, K] solutions at once.  Stage-2
selection is priced separately (`selection_memory_estimate`) and is why
`--k-pool-factor 1` is the policy at the top rungs.  These are ARITHMETIC; the
receipt also records the measured peak RSS next to them.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from typing import Any

import torch

from k3_common import (
    K3Index,
    canonical_device,
    device_inventory,
    log,
    peak_rss_bytes,
    rms_norm,
    same_device,
    save_tensors,
    sha256_tensor,
    situ,
    write_receipt,
    write_scores,
)
from k3_layer import LayerWeights, dequant_expert, load_layer

RIDGE_REL = 1e-6
LAMBDA_LADDER = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_RUNGS = (65536, 32768, 17408, 8704)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def gram_memory_estimate(width: int, out_dim: int) -> dict[str, Any]:
    """What the fp64 Gram refit actually costs, in bytes, before it is spent.

    Priced honestly, because the first version of this lane's gap list priced
    only the accumulated Grams and was ~3x optimistic:

      resident  2 Grams (inner + check) + their 2 cross matrices
      transient inside ONE ridge solve: the regularised clone, its Cholesky
                factor, and the [out, K] solution (the rhs is the same shape)

    The identity matrix that `gram + lam * eye(n)` used to build is gone; at
    K = 65,536 it alone was 34.4 GB of fp64 zeros.
    """
    gram = width * width * 8
    cross = width * out_dim * 8
    resident = 2 * gram + 2 * cross
    transient = 2 * gram + 2 * cross
    return {
        "width": width,
        "out_dim": out_dim,
        "one_gram_bytes": gram,
        "peak_bytes": resident + transient,
        "one_gram_gigabytes": round(gram / 1e9, 3),
        "resident_gigabytes": round(resident / 1e9, 3),
        "solve_transient_gigabytes": round(transient / 1e9, 3),
        "peak_gigabytes": round((resident + transient) / 1e9, 3),
        "note": ("fp64; resident = inner + check Gram/cross, transient = the regularised "
                 "clone + its Cholesky factor + the [out,K] solution.  No identity is ever "
                 "materialised and no ladder rung's solution is stored."),
    }


def process_peak_rss_record(peak: int | None) -> dict[str, Any]:
    """`VmHWM` is a PROCESS-LIFETIME high-water mark, so name it that.

    Rungs run largest-first inside one process, so every rung after the first
    inherits the first one's peak and a field called `measured_peak_rss` reads
    as this rung's cost when it is nothing of the kind (0.568 -> 0.572 -> 0.572
    in the shipped receipt is one measurement and two echoes).  Fork a child
    per rung, or use `gram_memory_probe.py`, if a per-rung figure is wanted.
    """
    return {
        "process_peak_rss_so_far_gigabytes": round(peak / 1e9, 3) if peak else None,
        "process_peak_rss_note": ("VmHWM for the WHOLE process; rungs run largest-first in one "
                                  "process, so later rungs inherit the first rung's peak.  This "
                                  "is not a per-rung measurement."),
    }


def selection_memory_estimate(pool: int, k: int) -> dict[str, Any]:
    """`pivoted_columns_gram`'s own cost: the pool Gram plus the [n, k] factor.

    This is the stage-2 term the gap list did not price.  At pool = 131,072 and
    k = 65,536 the `lower` factor alone is 68.7 GB and the loop is 65,536 Python
    iterations -- which is why `--k-pool-factor 1` (stage-1 only) is the policy
    at the top rungs, not a shortcut.
    """
    gram = pool * pool * 8
    lower = pool * min(k, pool) * 8
    return {
        "pool": pool,
        "k": k,
        "peak_bytes": gram + lower,
        "pool_gram_gigabytes": round(gram / 1e9, 3),
        "cholesky_factor_gigabytes": round(lower / 1e9, 3),
        "peak_gigabytes": round((gram + lower) / 1e9, 3),
        "python_iterations": int(min(k, pool)),
    }


def expert_bank_bytes(width: int, latent_dim: int, elem_size: int) -> int:
    """`ExpertBank`'s three kept-row matrices on `--device`: w1, w3 and w2."""
    return 3 * width * latent_dim * elem_size


def score_buffer_bytes(held_tokens: int, out_dim: int, hidden: int, elem_size: int) -> int:
    """`CoresetFold.score`'s held-out buffers: preds/targets and the write copies.

    `preds` and `targets` are accumulated as chunk LISTS and then `torch.cat`,
    so each exists twice at the join; the routed-write comparison then makes
    two [held, hidden] copies in the parent's dtype.
    """
    return 4 * held_tokens * out_dim * 4 + 4 * held_tokens * hidden * elem_size


def form_m_bytes(width: int, latent_dim: int, hidden: int, elem_size: int) -> int:
    """`build_form_m`'s fp32 lift, which is the largest single device allocation.

    `bank.w1.float() @ down.float()` upcasts a [K, latent] and a [latent,
    hidden] operand and produces a [K, hidden] fp32 result -- 1.88 GB at
    K = 65,536 / hidden = 7,168 -- then casts it down.  `w3_m` is built while
    `w1_m` is still held, so one finished copy is counted twice.
    """
    return (width * latent_dim * 4
            + latent_dim * hidden * 4
            + width * hidden * 4
            + 2 * width * hidden * elem_size)


def device_memory_estimate(width: int, out_dim: int, chunk: int, intermediate: int,
                           resident: dict[str, int] | None = None) -> dict[str, Any]:
    """What lands on `--device`, which is a DIFFERENT and much smaller budget.

    Two review findings live in this function, and they pull in opposite
    directions.

    The first: the version before last accumulated the fp64 Gram on the compute
    device.  `sel.T @ sel` with `sel` on cuda allocates a [K, K] fp64 tensor
    THERE, 34.4 GB at K = 65,536 on a 25.3 GB 3090 -- it would die in the
    allocator while `--max-gram-gb` (a HOST budget) happily admitted the rung.
    Every fp64 [K, K] / [K, out] accumulator now lives on the host, and what
    remains of the STREAMING work is bounded by `--chunk` and by
    `moe_intermediate_size`.

    The second, and the reason for `resident`: pricing only that streaming work
    priced roughly 4% of the truth.  `main()` loads the layer onto `--device`,
    `CoresetFold` moves the calibration cache there, `ExpertBank` puts the kept
    rows there and `score` / `build_form_m` allocate there too.  At
    K = 65,536 with the experts on the card that is ~18.5 GB resident before
    this function's old answer of 0.94 GB was even reached, so `--max-device-gb`
    admitted a rung that then died exactly where `enforce_budget` exists to stop
    it happening.  `resident` is a byte breakdown supplied by the caller --
    measured where it can be measured (`LayerWeights.device_bytes`) and derived
    from the rung width where it cannot -- and it is ADDED to the transient
    peak rather than maxed against it, because those allocations are all live
    at the same time.

    Also worth saying out loud: fp64 matmul on sm86 runs at 1/64 rate, so
    keeping the big fp64 work off the GPU is not only a memory fix.
    """
    stage1_gram = intermediate * intermediate * 8          # per-expert, fp64, on device
    stage1_features = chunk * intermediate * (4 + 8)       # fp32 features + its fp64 copy
    stage2_chunk = chunk * width * 4 * 3                   # psi + router table + phi, fp32
    refit_chunk = chunk * width * 4 * 3 + chunk * out_dim * 4
    transient = max(stage1_gram + stage1_features, stage2_chunk, refit_chunk)
    resident = dict(resident or {})
    resident_total = sum(int(v) for v in resident.values())
    peak = resident_total + transient
    return {
        "width": width,
        "chunk": chunk,
        "stage1_expert_gram_bytes": stage1_gram,
        "stage1_feature_chunk_bytes": stage1_features,
        "stage2_phi_chunk_bytes": stage2_chunk,
        "refit_design_chunk_bytes": refit_chunk,
        "transient_peak_bytes": transient,
        "resident_bytes": resident,
        "resident_total_bytes": resident_total,
        "resident_total_gigabytes": round(resident_total / 1e9, 3),
        "peak_bytes": peak,
        "peak_gigabytes": round(peak / 1e9, 3),
        "note": ("device-side only.  The fp64 [K,K] Gram and [K,out] cross accumulate on the "
                 "HOST (`.cpu().double()` before the outer product), so no fp64 tensor of rung "
                 "width is ever allocated on --device; the device TRANSIENTS are bounded by "
                 "--chunk and by moe_intermediate_size, and the RESIDENT terms (the layer "
                 "itself, the calibration cache, the ExpertBank, score's buffers, the form-M "
                 "lift) are supplied by the caller and added on top.  An empty `resident` means "
                 "the caller priced the transients only."),
    }


def default_device_budget_gb(device: torch.device | str) -> float:
    """0.9 x the named device's total memory, measured -- or 0 (off) on CPU.

    On CPU the host budget (`--max-gram-gb`) already governs the same pool, so
    a second guard there would only double-count.

    `canonical_device` first: `--device cuda` carries no index, and the old
    `device.index or 0` read that as card 0 rather than as the CURRENT device.
    On a box where the current device is not 0 that sized the budget off the
    wrong card -- 47.6 GB of RTX 8000 against a 25.3 GB 3090, or the reverse --
    which is the same class of error as the zeroed resident estimate.
    """
    device = canonical_device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(device.index)
    return round(0.9 * props.total_memory / 1e9, 3)


def enforce_budget(estimate: dict[str, Any], limit_gb: float, what: str,
                   flag: str = "--max-gram-gb") -> None:
    """Refuse with the arithmetic instead of dying inside an allocator.

    Compares the exact byte count, not the rounded GB figure -- at synthetic
    widths `peak_gigabytes` rounds to 0.0 and a rounded comparison would make
    the guard untestable.
    """
    if limit_gb <= 0:
        return
    if estimate["peak_bytes"] / 1e9 > limit_gb:
        raise SystemExit(
            f"{what} needs a peak of {estimate['peak_gigabytes']} GB "
            f"(> {flag} {limit_gb}); details {json.dumps(estimate)}.  "
            "Drop the rung, use --k-pool-factor 1 (stage-1 only), or raise the budget "
            "deliberately after checking what else is resident on the box."
        )


def mass_allocation(mass: torch.Tensor, k_pool: int, cap: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apportion EXACTLY `k_pool` channel slots across the experts, by router mass.

    `round(mass_e / total * k_pool)` -- what this used to do -- does not sum to
    `k_pool`: rounding and the "at least 1 per live expert" floor both leak.
    The lane's own shipped receipt shows a request of 256 landing on 257, and a
    one-channel overshoot is not cosmetic: `stage2` skips only when the pool is
    at or below the rung, so a 65,537-column pool at the top rung would start a
    65,536-iteration pivoted Cholesky with a 34.4 GB fp64 factor.  The
    documented `--k-pool-factor 1 = stage-1 only` policy was therefore not
    enforced, only hoped for.

    Largest-remainder (Hamilton) apportionment fixes it: floor the quotas, then
    hand out or take back single slots by fractional remainder, respecting the
    per-expert cap and the floor of 1 for a live expert.  When there are fewer
    slots than live experts the floor cannot hold for everyone, so the `k_pool`
    HEAVIEST experts take one slot each -- the "trim to the k highest-mass
    channels" case.  Selection downstream may still land BELOW the allocation
    (a rank-starved expert yields fewer pivots than it was allocated), which is
    the safe direction: the pool can undershoot the rung, never overshoot it.
    """
    mass = mass.double().cpu()
    alloc = torch.zeros(mass.numel(), dtype=torch.long)
    live = mass > 0
    n_live = int(live.sum())
    if n_live == 0 or k_pool <= 0:
        return alloc, {"mode": "empty", "allocated": 0, "live_experts": n_live,
                       "requested": int(k_pool), "exact": k_pool <= 0}
    if k_pool <= n_live:
        order = torch.argsort(mass, descending=True)[:k_pool]
        alloc[order] = 1
        return alloc, {"mode": "top-mass experts (fewer slots than live experts)",
                       "allocated": int(alloc.sum()), "live_experts": n_live,
                       "requested": int(k_pool), "exact": int(alloc.sum()) == k_pool,
                       "rebalance_passes": 0, "capped_experts": 0}
    total = float(mass.sum())
    quota = mass / max(total, 1e-30) * k_pool
    floor = torch.floor(quota)
    remainder = quota - floor
    base = torch.clamp(floor.long(), min=1, max=cap)
    base = torch.where(live, base, torch.zeros_like(base))
    assigned = int(base.sum())
    passes = 0
    while assigned != k_pool and passes < 64:
        passes += 1
        if assigned < k_pool:
            candidates = torch.nonzero(live & (base < cap)).flatten()
            if candidates.numel() == 0:
                break
            order = candidates[torch.argsort(remainder[candidates], descending=True)]
            take = min(k_pool - assigned, int(order.numel()))
            base[order[:take]] += 1
            assigned += take
        else:
            candidates = torch.nonzero(base > 1).flatten()
            if candidates.numel() == 0:
                break
            order = candidates[torch.argsort(remainder[candidates])]
            take = min(assigned - k_pool, int(order.numel()))
            base[order[:take]] -= 1
            assigned -= take
    return base, {"mode": "largest-remainder apportionment",
                  "allocated": int(base.sum()), "live_experts": n_live,
                  "requested": int(k_pool), "exact": int(base.sum()) == k_pool,
                  "rebalance_passes": passes,
                  "capped_experts": int((base >= cap).sum())}


def pivoted_columns_gram(gram: torch.Tensor, k: int, exact_scan: bool = False) -> torch.Tensor:
    """Greedy pivoted-Cholesky column selection, straight off a Gram.

    Same algorithm as mx45_build.pivoted_columns, with the O(n) Python scan
    replaced by `argmax` -- identical except for how ties inside 1e-15 break.
    `exact_scan=True` restores the literal scan; the test suite runs both and
    asserts they agree.

    Costs `n x k` fp64 for the growing Cholesky factor on top of the Gram --
    see `selection_memory_estimate`.
    """
    n = int(gram.shape[0])
    if k >= n:
        return torch.arange(n, dtype=torch.long)
    gram = gram.to(torch.float64)
    diag = torch.diagonal(gram).clone()
    chosen: list[int] = []
    lower = torch.zeros(n, k, dtype=torch.float64)
    for step in range(k):
        if exact_scan:
            best, pick = -1.0, -1
            for i in range(n):
                if diag[i] > best + 1e-15:
                    best, pick = float(diag[i]), i
        else:
            pick = int(torch.argmax(diag))
            best = float(diag[pick])
        if pick < 0 or best <= 1e-30:
            break
        chosen.append(pick)
        col = gram[:, pick] - lower[:, :step] @ lower[pick, :step]
        col = col / max(best, 1e-300) ** 0.5
        lower[:, step] = col
        diag = diag - col * col
        diag[chosen] = -1.0
    return torch.tensor(sorted(chosen), dtype=torch.long)


# --------------------------------------------------------------------------
# ridge, in Gram form
# --------------------------------------------------------------------------
def prior_anchored_gram(gram: torch.Tensor, cross: torch.Tensor, prior: torch.Tensor,
                        rel: float = RIDGE_REL) -> torch.Tensor:
    """argmin ||A W^T - Y||^2 + lam ||W - prior||^2 given G = A^T A, B = A^T Y.

    prior [out, K]; returns W [out, K].  lam = rel * mean(diag G), the same
    normalisation mx45's `prior_anchored` uses on the dense design.

    `gram + lam * torch.eye(n)` is never written here.  It allocates a dense
    fp64 identity AND a summed copy -- two extra n x n matrices, 68.7 GB of
    them at K = 65,536 -- for a diagonal update.  The clone is unavoidable
    (Cholesky must not destroy the Gram, which the ladder reuses 19 times) but
    it is freed the moment the factorisation exists.
    """
    lam = rel * float(torch.diagonal(gram).mean())
    regularised = gram.clone()
    regularised.diagonal().add_(lam)
    rhs = cross + lam * prior.T.to(gram.dtype)
    factor = torch.linalg.cholesky(regularised)
    del regularised
    solved = torch.cholesky_solve(rhs, factor).T.contiguous()
    del factor, rhs
    return solved


def _check_sse(weight: torch.Tensor, gram: torch.Tensor, cross: torch.Tensor, y_sq: float) -> float:
    """||A_ck W^T - Y_ck||^2 from Grams only: tr(W G W^T) - 2 tr(W B) + ||Y||^2."""
    w = weight.to(gram.dtype)
    return float((w @ gram * w).sum() - 2.0 * (w * cross.T).sum() + y_sq)


def law_abiding_solve_gram(
    gram_inner: torch.Tensor, cross_inner: torch.Tensor,
    gram_check: torch.Tensor, cross_check: torch.Tensor, check_y_sq: float,
    prior: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """`law_abiding_solve_split`, Gram form.  THE LAW is a veto, not a chooser.

    CONSUMES its inputs.  `gram_inner`/`cross_inner` are turned into the
    full-fit system IN PLACE once the check column is done -- the fit split is
    inner disjoint-union check, so gram_all = gram_inner + gram_check exactly --
    and `gram_check`/`cross_check` are dead afterwards.  The caller must not
    reuse any of them.

    Three passes, none of which keeps a solution it is not returning:

      1. inner solve per lambda -> `check_sse`  (needs inner + check)
      2. full solve per lambda  -> `update_scale`, the veto  (needs full only)
      3. ONE final full solve at the chosen lambda

    The previous version accumulated three Grams, three cross matrices and
    stored all nine ladder solutions at once.  At K = 65,536, out = 3,584 that
    is ~103 GB of Gram plus ~17 GB of stored solutions before the ridge has
    allocated anything of its own; the box has 251 GiB.
    """
    reference = float(prior.norm())

    # -- 1. the check split ------------------------------------------------
    check_sse: list[float] = []
    for rel in LAMBDA_LADDER:
        inner = prior_anchored_gram(gram_inner, cross_inner, prior, rel=rel)
        check_sse.append(_check_sse(inner, gram_check, cross_check, check_y_sq))
        del inner

    # -- fold check into inner, in place: now they ARE the full-fit system --
    gram_inner.add_(gram_check)
    cross_inner.add_(cross_check)
    gram_all, cross_all = gram_inner, cross_inner

    # -- 2. the veto -------------------------------------------------------
    update_scale: list[float] = []
    for rel in LAMBDA_LADDER:
        full = prior_anchored_gram(gram_all, cross_all, prior, rel=rel)
        update_scale.append(float((full - prior).norm() / reference))
        del full
    # LAMBDA_LADDER[0] IS the free solve, so it is not paid for twice
    free_scale = update_scale[0]

    rungs = [{"lambda_relative": rel, "check_sse": sse, "update_scale": scale}
             for rel, sse, scale in zip(LAMBDA_LADDER, check_sse, update_scale)]
    admitted = [row for row in rungs if row["update_scale"] <= 1.0]
    if not admitted:
        raise SystemExit(
            f"no lambda on the ladder brings update_scale to 1.0 (free {free_scale:.3f}); "
            "THE LAW refuses the install"
        )
    chosen = min(admitted, key=lambda row: row["check_sse"])

    # -- 3. the one solution that leaves this function ---------------------
    solved = prior_anchored_gram(gram_all, cross_all, prior, rel=chosen["lambda_relative"])
    monotone = all(check_sse[i] >= check_sse[i + 1] - 1e-12 for i in range(len(check_sse) - 1))
    return solved, {
        "update_scale": chosen["update_scale"],
        "lambda_relative": chosen["lambda_relative"],
        "free_solve_update_scale": free_scale,
        "lambda_selected_on": "internal fit split",
        "lambda_law_would_have_picked": admitted[0]["lambda_relative"],
        # An `update_scale` of 0.13 reads comfortable; it is not, if the free
        # solve wanted 9.85x the prior and only the veto pulled it back.  Say
        # which constraint actually chose the answer.
        "veto_binding": bool(free_scale > 1.0),
        "lambdas_vetoed": int(len(rungs) - len(admitted)),
        "chosen_is_largest_lambda": bool(chosen["lambda_relative"] == LAMBDA_LADDER[-1]),
        "chosen_is_weakest_admitted": bool(chosen["lambda_relative"] == admitted[0]["lambda_relative"]),
        "check_sse_monotone_decreasing_in_lambda": bool(monotone),
        "interpretation": (
            "veto_binding + chosen_is_largest_lambda + monotone check_sse together mean the "
            "check split carried no signal and the installed map is the prior, shrunk -- read "
            "that as a NO-SIGNAL fold, not a healthy one"
            if (free_scale > 1.0 and chosen["lambda_relative"] == LAMBDA_LADDER[-1] and monotone)
            else "the check split discriminated between lambdas"
        ),
        "ladder": rungs,
    }


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
class ExpertBank:
    """Verbatim gate/up/down rows for a channel set, grouped by expert.

    Dequantizing one expert costs 22 MB of scratch; a channel set touches each
    expert at most once, so the bank is built in one sweep over the experts it
    actually needs.
    """

    def __init__(self, weights: LayerWeights, channels: torch.Tensor, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        experts = channels[:, 0]
        rows = channels[:, 1]
        order = torch.argsort(experts * (int(rows.max()) + 1) + rows)
        self.channels = channels[order]
        self.order = order
        w1, w3, w2 = [], [], []
        for expert in torch.unique(self.channels[:, 0]).tolist():
            picked = self.channels[:, 1][self.channels[:, 0] == expert]
            dequantized = dequant_expert(weights, int(expert), dtype=dtype)
            w1.append(dequantized["w1"][picked].to(device))
            w3.append(dequantized["w3"][picked].to(device))
            w2.append(dequantized["w2"][:, picked].to(device))
        self.w1 = torch.cat(w1, dim=0)          # [K, latent]
        self.w3 = torch.cat(w3, dim=0)          # [K, latent]
        self.w2 = torch.cat(w2, dim=1)          # [latent_out, K]

    def psi(self, latent: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
        return situ(
            torch.nn.functional.linear(latent, self.w1),
            torch.nn.functional.linear(latent, self.w3),
            beta, linear_beta,
        )


def channel_router_weights(
    channels: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor,
    rows: torch.Tensor, n_experts: int,
) -> torch.Tensor:
    """[len(rows), K] router mass per channel -- built per token chunk.

    The scatter goes into a dense [chunk, n_experts] table (3.6 MB per 1,024
    tokens at 896 experts) and the channels then index columns off it; a
    [T, |U|] phi is never materialised.  Top-k indices are unique within a row,
    so a plain `scatter_` is exact.
    """
    table = torch.zeros(rows.numel(), n_experts, dtype=torch.float32, device=topk_w.device)
    table.scatter_(1, topk_idx[rows].long(), topk_w[rows].float())
    return table[:, channels[:, 0].to(table.device)]


# --------------------------------------------------------------------------
# the fold
# --------------------------------------------------------------------------
class CoresetFold:
    def __init__(self, weights: LayerWeights, cache: dict[str, torch.Tensor], cfg: dict,
                 device: torch.device, chunk: int = 1024,
                 holdout: str = "blocks", holdout_blocks: int = 24,
                 max_gram_gb: float = 0.0, max_device_gb: float = 0.0,
                 form_m: bool = False):
        self.w = weights
        self.cfg = cfg
        # canonical (indexed) form: every byte count below compares tensor
        # devices against this one, and `cuda` != `cuda:0` to `torch.device`
        self.device = canonical_device(device)
        self.chunk = chunk
        self.beta = cfg.get("activation_situ_beta") or 1.0
        self.linear_beta = cfg.get("activation_situ_linear_beta")
        self.latent = cache["latent_in"].to(device)
        self.y_lat = cache["y_lat"].to(device).float()
        self.topk_idx = cache["router_topk_idx"].to(device).long()
        self.topk_w = cache["router_topk_w"].to(device).float()
        self.tokens = int(self.latent.shape[0])
        self.latent_dim = int(self.latent.shape[1])
        self.n_experts = int(cfg["num_experts"])
        self.intermediate = int(cfg["moe_intermediate_size"])
        self.max_gram_gb = max_gram_gb
        self.max_device_gb = max_device_gb
        # `--form-m` is known at construction because the fp32 [K, hidden] lift
        # it allocates is part of the RUNG's device peak, and a budget that
        # learns about it only when the allocation happens is not a budget.
        self.form_m = form_m
        self.hidden = int(weights.body["block_sparse_moe.routed_expert_up_proj.weight"].shape[0])
        self.holdout = holdout
        self.holdout_blocks = holdout_blocks
        self._split_positions(holdout, holdout_blocks)
        self._build_route_index()

    # -- device budget -----------------------------------------------------
    def device_resident_bytes(self, width: int, out_dim: int,
                              include_score: bool = False,
                              include_form_m: bool = False) -> dict[str, int]:
        """Every byte this fold keeps ALIVE on `--device`, measured where possible.

        The layer term is measured off the tensors themselves rather than
        derived, because it is the term that was missing and it is the largest:
        with `--experts-device cuda:0` it is 15.7 GB of packed experts and E8M0
        scales on a real K3 MoE layer, against a transient estimate of 0.94 GB.

        Every membership test is `same_device`, not `==`: the cache is moved
        with `.to(self.device)` and then counted, so an unindexed `--device
        cuda` used to move 1.4 GB onto the card and then price it at zero.
        """
        elem = self.latent.element_size()
        resident = {
            "layer_weights_on_device": self.w.device_bytes(self.device)["total"],
            "calibration_cache_on_device": sum(
                t.numel() * t.element_size()
                for t in (self.latent, self.y_lat, self.topk_idx, self.topk_w)
                if same_device(t.device, self.device)
            ),
            "expert_bank": expert_bank_bytes(width, self.latent_dim, elem),
        }
        if include_score:
            resident["score_heldout_buffers"] = score_buffer_bytes(
                int(self.held.sum()), out_dim, self.hidden, elem)
        if include_form_m:
            resident["form_m_lift"] = form_m_bytes(width, self.latent_dim, self.hidden, elem)
        return resident

    def device_budget(self, width: int, out_dim: int, include_score: bool = False,
                      include_form_m: bool = False) -> dict[str, Any]:
        return device_memory_estimate(
            width, out_dim, self.chunk, self.intermediate,
            resident=self.device_resident_bytes(width, out_dim, include_score, include_form_m),
        )

    # -- splits ------------------------------------------------------------
    def _split_positions(self, mode: str, blocks: int) -> None:
        """Held-out / inner / check masks.  A POSITION rule, never a content rule.

        `blocks` (the default) deals CONTIGUOUS chunks of the token axis
        round-robin into inner/inner/held.  `interleaved` is the original
        `arange % 3 == 2`, kept because the synthetic runs were measured with
        it: on random token ids the two are equivalent, but on a real corpus an
        interleaved holdout sits between its own neighbours, and adjacent
        tokens of one document are strongly correlated -- the held-out relL2
        comes out optimistic and there is no way to tell from the number.
        """
        idx = torch.arange(self.tokens)
        if mode == "interleaved":
            group = idx % 3
            self.held = group == 2
            self.fit = ~self.held
            fit_positions = torch.nonzero(self.fit).flatten()
            self.check = torch.zeros(self.tokens, dtype=torch.bool)
            self.check[fit_positions[2::3]] = True
        elif mode == "blocks":
            size = max(1, -(-self.tokens // max(blocks, 1)))
            chunk_id = idx // size
            self.held = (chunk_id % 3) == 2
            self.fit = ~self.held
            fit_chunks = sorted(set(chunk_id[self.fit].tolist()))
            check_chunks = torch.tensor(fit_chunks[2::3], dtype=chunk_id.dtype)
            self.check = torch.isin(chunk_id, check_chunks) & self.fit
        else:
            raise SystemExit(f"unknown holdout mode {mode!r}; use 'blocks' or 'interleaved'")
        self.inner = self.fit & ~self.check
        if not bool(self.held.any()) or not bool(self.inner.any()) or not bool(self.check.any()):
            raise SystemExit(
                f"holdout mode {mode!r} over {self.tokens} tokens left an empty split "
                f"(held {int(self.held.sum())}, inner {int(self.inner.sum())}, "
                f"check {int(self.check.sum())})"
            )

    # -- router index ------------------------------------------------------
    def _build_route_index(self) -> None:
        """One sort instead of 896 full-table comparisons.

        `moe_infer` groups tokens by expert with an argsort; the same trick
        turns "which fit tokens went to expert e, and with what mass" into a
        slice, which matters at 896 experts x 65,536 tokens.
        """
        fit_rows = torch.nonzero(self.fit).flatten().to(self.device)
        flat_expert = self.topk_idx[fit_rows].flatten()
        flat_weight = self.topk_w[fit_rows].flatten()
        flat_token = fit_rows.repeat_interleave(self.topk_idx.shape[1])
        order = torch.argsort(flat_expert, stable=True)
        self._rows_by_expert = flat_token[order]
        self._weight_by_expert = flat_weight[order]
        counts = torch.bincount(flat_expert, minlength=self.n_experts)
        self._expert_counts = counts
        self._expert_offsets = torch.cat(
            [torch.zeros(1, dtype=counts.dtype, device=counts.device), counts.cumsum(0)]
        )
        mass = torch.zeros(self.n_experts, dtype=torch.float64, device=self.device)
        mass.index_add_(0, flat_expert, flat_weight.double())
        self._expert_mass = mass

    def expert_rows(self, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self._expert_offsets[expert])
        stop = int(self._expert_offsets[expert + 1])
        return self._rows_by_expert[start:stop], self._weight_by_expert[start:stop]

    def expert_mass(self) -> torch.Tensor:
        return self._expert_mass

    # -- stage 1 -----------------------------------------------------------
    def stage1(self, k_pool: int) -> tuple[torch.Tensor, dict]:
        mass = self.expert_mass()
        total = float(mass.sum())
        allocation, allocation_stats = mass_allocation(mass, k_pool, self.intermediate)
        selected: list[torch.Tensor] = []
        per_expert = []
        for expert in range(self.n_experts):
            k_e = int(allocation[expert])
            if k_e == 0:
                continue
            rows, weight = self.expert_rows(expert)
            if rows.numel() == 0:
                continue
            dequantized = dequant_expert(self.w, expert, dtype=self.latent.dtype)
            w1 = dequantized["w1"].to(self.device)
            w3 = dequantized["w3"].to(self.device)
            # Chunked over the expert's own rows so the fp64 transient on
            # --device is bounded by --chunk rather than by however many tokens
            # the router happened to send here (65,536 x 3,072 fp64 = 1.6 GB in
            # the worst case, which nothing was pricing).
            gram = torch.zeros(w1.shape[0], w1.shape[0], dtype=torch.float64)
            for start in range(0, int(rows.numel()), self.chunk):
                piece = rows[start:start + self.chunk]
                latent = self.latent[piece]
                features = situ(
                    torch.nn.functional.linear(latent, w1),
                    torch.nn.functional.linear(latent, w3),
                    self.beta, self.linear_beta,
                ).float() * weight[start:start + self.chunk].unsqueeze(-1)
                features = features.double()
                gram += (features.T @ features).cpu()
                del features
            picked = pivoted_columns_gram(gram, min(k_e, self.intermediate))
            del gram, w1, w3
            if picked.numel() == 0:
                continue
            selected.append(torch.stack([torch.full_like(picked, expert), picked], dim=1))
            per_expert.append({"expert": expert, "rows": int(rows.numel()),
                               "allocated": k_e, "selected": int(picked.numel())})
        if not selected:
            raise SystemExit("stage 1 selected no channels: the router never opened an expert on the fit split")
        channels = torch.cat(selected, dim=0)
        stats = {
            "k_pool_requested": k_pool,
            "k_pool_allocated": int(allocation.sum()),
            "k_pool_selected": int(channels.shape[0]),
            "allocation": allocation_stats,
            "experts_with_channels": len(per_expert),
            "expert_mass_total": total,
            "rank_starved_experts": int(sum(1 for row in per_expert if row["selected"] < row["allocated"])),
            "per_expert_head": per_expert[:8],
        }
        if int(channels.shape[0]) > k_pool:
            raise SystemExit(
                f"stage 1 overshot its pool ({int(channels.shape[0])} > {k_pool}); the "
                f"apportionment is meant to make that impossible: {json.dumps(allocation_stats)}"
            )
        return channels, stats

    # -- stage 2 -----------------------------------------------------------
    def stage2(self, channels: torch.Tensor, k: int) -> tuple[torch.Tensor, dict]:
        if channels.shape[0] <= k:
            return channels, {"skipped": "pool already at or below the rung (stage-1-only policy)"}
        budget = selection_memory_estimate(int(channels.shape[0]), k)
        enforce_budget(budget, self.max_gram_gb, f"stage-2 selection of {k} from {channels.shape[0]}")
        enforce_budget(self.device_budget(int(channels.shape[0]), int(self.y_lat.shape[1])),
                       self.max_device_gb,
                       f"stage-2 selection of {k} from {channels.shape[0]} on {self.device}",
                       flag="--max-device-gb")
        log(f"  stage2 pool {int(channels.shape[0])} -> {k}: pool Gram "
            f"{budget['pool_gram_gigabytes']} GB + factor {budget['cholesky_factor_gigabytes']} GB, "
            f"{budget['python_iterations']} pivots", tag="c14")
        bank = ExpertBank(self.w, channels, self.device, dtype=self.latent.dtype)
        gram = torch.zeros(bank.w1.shape[0], bank.w1.shape[0], dtype=torch.float64)
        rows_all = torch.nonzero(self.fit).flatten().to(self.device)
        for start in range(0, rows_all.numel(), self.chunk):
            rows = rows_all[start:start + self.chunk]
            psi = bank.psi(self.latent[rows], self.beta, self.linear_beta).float()
            phi = psi * channel_router_weights(
                bank.channels, self.topk_idx, self.topk_w, rows, self.n_experts
            )
            # fp32 to the host FIRST, then fp64: the outer product is [pool, pool]
            # and must never be allocated on --device (34.4 GB at 65,536).
            host = phi.cpu().double()
            gram += host.T @ host
            del psi, phi, host
        picked = pivoted_columns_gram(gram, k)
        del gram
        peak = peak_rss_bytes()
        stats = {"pool": int(channels.shape[0]), "kept": int(picked.numel()), "memory": budget}
        stats["memory"].update(process_peak_rss_record(peak))
        selected = bank.channels[picked]
        del bank
        gc.collect()
        return selected, stats

    # -- refit -------------------------------------------------------------
    def refit(self, channels: torch.Tensor) -> tuple[torch.Tensor, ExpertBank, dict]:
        bank = ExpertBank(self.w, channels, self.device, dtype=self.latent.dtype)
        width = bank.w1.shape[0]
        out_dim = self.y_lat.shape[1]
        budget = gram_memory_estimate(width, out_dim)
        enforce_budget(budget, self.max_gram_gb, f"the K={width} Gram refit")
        # priced for the WHOLE rung: `score` runs on this bank straight after the
        # refit and `build_form_m` after that, so their allocations are live
        # alongside it and belong in the same budget.
        device_budget = self.device_budget(width, out_dim, include_score=True,
                                           include_form_m=self.form_m)
        enforce_budget(device_budget, self.max_device_gb,
                       f"the K={width} refit/score/export on {self.device}", flag="--max-device-gb")
        log(f"  refit K={width}: fp64 Gram {budget['one_gram_gigabytes']} GB on the HOST, "
            f"estimated host peak {budget['peak_gigabytes']} GB, "
            f"device peak {device_budget['peak_gigabytes']} GB "
            f"({device_budget['resident_total_gigabytes']} GB of it resident)", tag="c14")
        # TWO accumulated Grams, not three: the full-fit system is derived from
        # inner + check by addition inside the solver (the splits are disjoint),
        # which is exact and costs nothing.
        gram_inner = torch.zeros(width, width, dtype=torch.float64)
        gram_check = torch.zeros_like(gram_inner)
        cross_inner = torch.zeros(width, out_dim, dtype=torch.float64)
        cross_check = torch.zeros_like(cross_inner)
        check_y_sq = 0.0
        for start in range(0, self.tokens, self.chunk):
            stop = min(start + self.chunk, self.tokens)
            rows = torch.arange(start, stop)
            if not bool(self.fit[rows].any()):
                continue
            psi = bank.psi(self.latent[start:stop], self.beta, self.linear_beta).float()
            target = self.y_lat[start:stop]
            for mask, gram, cross in (
                (self.inner[rows], gram_inner, cross_inner),
                (self.check[rows], gram_check, cross_check),
            ):
                if not bool(mask.any()):
                    continue
                # HOST fp64.  `sel.T @ sel` with `sel` on cuda allocates the
                # [K, K] fp64 Gram on the compute device -- 34.4 GB at
                # K = 65,536 on a 25.3 GB card -- and `--max-gram-gb`, which
                # prices host RAM, would have admitted it.  The fp32 -> host
                # copy also halves the transfer against `.double()` first.
                sel = psi[mask.to(psi.device)].cpu().double()
                gram += sel.T @ sel
                cross += sel.T @ target[mask.to(psi.device)].cpu().double()
                del sel
            if bool(self.check[rows].any()):
                check_y_sq += float(target[self.check[rows].to(psi.device)].cpu().double().pow(2).sum())
            del psi

        # prior: the parent's own read-out row, scaled by the channel's mean
        # router mass -- the dense child sees every token, the parent did not.
        # m_c = E_t[w_e(t) . 1[e in top16]] over the fit split, straight off the
        # route index (no per-channel scan).
        experts = bank.channels[:, 0].to(self.device)
        n_fit = max(int(self.fit.sum()), 1)
        mass = (self.expert_mass()[experts] / n_fit).cpu()
        counts = self._expert_counts[experts].double().cpu()
        prior = (bank.w2.double().cpu() * mass.unsqueeze(0))
        # consumes the Grams; nothing below may touch them again
        solved, ladder = law_abiding_solve_gram(
            gram_inner, cross_inner, gram_check, cross_check, check_y_sq, prior.float(),
        )
        del gram_inner, gram_check, cross_inner, cross_check
        gc.collect()
        ladder["prior_mass_mean"] = float(mass.mean())
        ladder["channel_token_count_mean"] = float(counts.mean())
        ladder["memory"] = budget
        ladder["device_memory"] = device_budget
        peak = peak_rss_bytes()
        ladder["memory"].update(process_peak_rss_record(peak))
        ladder["holdout_mode"] = self.holdout
        return solved.float(), bank, ladder

    # -- scoring -----------------------------------------------------------
    def score(self, bank: ExpertBank, w2p: torch.Tensor) -> dict:
        eps = float(self.cfg["rms_norm_eps"])
        body = self.w.body
        rows = torch.nonzero(self.held).flatten().to(self.device)
        preds, targets = [], []
        for start in range(0, rows.numel(), self.chunk):
            piece = rows[start:start + self.chunk]
            psi = bank.psi(self.latent[piece], self.beta, self.linear_beta).float()
            preds.append(torch.nn.functional.linear(psi, w2p.to(psi.device)))
            targets.append(self.y_lat[piece])
        pred = torch.cat(preds)
        target = torch.cat(targets)
        scores = {"y_lat_heldout": write_scores(pred, target)}
        norm_w = body["block_sparse_moe.routed_expert_norm.weight"]
        up_w = body["block_sparse_moe.routed_expert_up_proj.weight"]
        dtype = up_w.dtype
        pred_write = torch.nn.functional.linear(rms_norm(pred.to(dtype), norm_w, eps), up_w)
        target_write = torch.nn.functional.linear(rms_norm(target.to(dtype), norm_w, eps), up_w)
        scores["routed_write_heldout"] = write_scores(pred_write.float(), target_write.float())
        return scores


# --------------------------------------------------------------------------
def parameter_accounting(cfg: dict, width: int, latent_dim: int, hidden: int) -> dict:
    n_experts = int(cfg["num_experts"])
    intermediate = int(cfg["moe_intermediate_size"])
    shared = int(cfg.get("num_shared_experts") or 0) * intermediate
    parent_routed = n_experts * 3 * intermediate * latent_dim
    child_routed = 3 * width * latent_dim
    router = n_experts * hidden + n_experts
    common = latent_dim * hidden * 2 + latent_dim + 3 * shared * hidden
    return {
        "parent_routed_expert_parameters": parent_routed,
        "child_routed_dense_parameters": child_routed,
        "routed_component_ratio": round(parent_routed / max(child_routed, 1), 3),
        "router_parameters_deleted": router,
        "verbatim_common_parameters": common,
        "parent_moe_total": parent_routed + router + common,
        "child_moe_total": child_routed + common,
        "moe_net_ratio": round((parent_routed + router + common) / max(child_routed + common, 1), 3),
        "quantization": "none -- forbidden in a fold; the packed parent is only a storage format",
    }


def run_rung(fold: CoresetFold, k: int, k_pool_factor: int, out_dir: str, layer_idx: int,
             cfg: dict, hidden: int, form_m: bool = False) -> dict:
    started = time.time()
    stage1_only = k_pool_factor <= 1
    k_pool = k if stage1_only else min(k * k_pool_factor, fold.n_experts * fold.intermediate)
    channels, stage1_stats = fold.stage1(k_pool)
    channels, stage2_stats = fold.stage2(channels, k)
    if stage1_only and "skipped" not in stage2_stats:
        # The stage-1-only policy is now ENFORCED, not hoped for: before the
        # apportionment fix a one-channel stage-1 overshoot silently started a
        # 65,536-iteration pivoted Cholesky at the top rung.
        raise SystemExit(
            f"--k-pool-factor {k_pool_factor} asked for the stage-1-only policy at K={k} but "
            f"stage 2 ran on a pool of {stage1_stats['k_pool_selected']}; "
            f"stage-1 allocation was {json.dumps(stage1_stats['allocation'])}"
        )
    w2p, bank, ladder = fold.refit(channels)
    scores = fold.score(bank, w2p)
    width = int(bank.w1.shape[0])
    payload = {
        "w1_s": bank.w1.cpu().contiguous(),
        "w3_s": bank.w3.cpu().contiguous(),
        "w2p": w2p.cpu().contiguous(),
        "channels": bank.channels.cpu().to(torch.int32),
    }
    # Named by the REQUESTED rung, because that is the key every consumer has:
    # `k3_review_stream.py --rung 17408` used to drop a file called
    # `L01_K17407_formE.safetensors` on the floor and then review the UNFOLDED
    # parent against the parent baseline, which reads as a perfect child.  The
    # width actually installed (rank starvation can land below the rung) is in
    # the metadata and in this receipt.
    path = os.path.join(out_dir, f"L{layer_idx:02d}_K{k}_formE.safetensors")
    digest = save_tensors(path, payload, metadata={
        "form": "E", "layer": str(layer_idx), "K": str(k), "width_installed": str(width),
        "stage1_only": str(bool(stage1_only)),
    })
    receipt = {
        "layer": layer_idx,
        "rung_requested": k,
        "width_installed": width,
        "k_pool": k_pool,
        "stage1_only": bool(stage1_only),
        "fold_file_key": ("named by the requested rung K; width_installed may be lower if an "
                          "expert was rank-starved"),
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "ridge": ladder,
        "heldout": scores,
        "tokens": fold.tokens,
        "fit_tokens": int(fold.fit.sum()),
        "held_tokens": int(fold.held.sum()),
        "check_tokens": int(fold.check.sum()),
        "parameters": parameter_accounting(cfg, width, fold.latent_dim, hidden),
        "form_E_path": path,
        "form_E_sha256": digest,
        "installed_tensor_sha256": {k2: sha256_tensor(v) for k2, v in payload.items() if v.is_floating_point()},
        "stored_vs_scored_dtype": (
            "w1_s/w3_s are stored in the parent's dtype; w2p is SOLVED and STORED at fp32 but the "
            "runtime dense path casts it to the activation dtype (bf16 on a real layer), so the "
            "scored child and the stored child are the same map only up to that cast"
        ),
        "peak_rss_gigabytes": (lambda b: round(b / 1e9, 3) if b else None)(peak_rss_bytes()),
        "seconds": round(time.time() - started, 1),
    }
    if form_m:
        lifted = build_form_m(fold, bank, w2p)
        m_path = os.path.join(out_dir, f"L{layer_idx:02d}_K{k}_formM.safetensors")
        receipt["form_M_path"] = m_path
        receipt["form_M_sha256"] = save_tensors(
            m_path, {k2: v.cpu().contiguous() for k2, v in lifted.items()},
            metadata={"form": "M", "layer": str(layer_idx), "K": str(k),
                      "width_installed": str(width)},
        )
        shared_channels = int(cfg.get("num_shared_experts") or 0) * int(cfg["moe_intermediate_size"])
        receipt["form_M_parameters"] = {
            "merged_ffn_width": width + shared_channels,
            "merged_ffn_parameters": 3 * (width + shared_channels) * hidden,
            "note": "form M is larger than form E; it exists for standard-export children only (D-b2)",
        }
    return receipt


def build_form_m(fold: CoresetFold, bank: ExpertBank, w2p: torch.Tensor) -> dict[str, torch.Tensor]:
    """Form M: lift the kept channels into the 7168 space for a standard export.

    W1_M = W1_S . W_down and W3_M = W3_S . W_down are EXACT (the composition of
    two linear maps); the shared expert's 6144 channels concatenate on, giving
    one SiTU FFN of width K + 6144.  W2 is left in the latent space here --
    approximating `routed_expert_norm` is a separate, measured step and this
    function does not pretend to have done it.
    """
    down = fold.w.body["block_sparse_moe.routed_expert_down_proj.weight"]
    shared_gate = fold.w.body["block_sparse_moe.shared_experts.gate_proj.weight"]
    shared_up = fold.w.body["block_sparse_moe.shared_experts.up_proj.weight"]
    return {
        "w1_m": (bank.w1.float() @ down.float()).to(down.dtype),
        "w3_m": (bank.w3.float() @ down.float()).to(down.dtype),
        "w2_latent": w2p,
        "shared_gate": shared_gate,
        "shared_up": shared_up,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--cache", required=True, help="the layer cache written by k3_stream.py")
    ap.add_argument("--out-dir", default="/home/cereal/omni-campaign/k3/folds")
    ap.add_argument("--rungs", default=",".join(str(r) for r in DEFAULT_RUNGS))
    ap.add_argument("--k-pool-factor", type=int, default=1,
                    help="1 = stage-1 only (the 65,536 policy); 2 = stage 1 to 2K then stage 2 to K")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experts-device", default="cpu",
                    help="where the 896 PACKED experts live; 'cpu' (the default) keeps 15.7 GB "
                         "off the compute card and moves only the rows a rung keeps.  Pass the "
                         "same value as --device to hold them there instead -- it is now priced "
                         "by --max-device-gb, so it will be refused with the arithmetic rather "
                         "than dying in the allocator")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--form-m", action="store_true")
    ap.add_argument("--holdout", default="blocks", choices=("blocks", "interleaved"),
                    help="'blocks' holds out CONTIGUOUS token ranges; 'interleaved' is the "
                         "original arange%%3 rule and is optimistic on a real corpus")
    ap.add_argument("--holdout-blocks", type=int, default=24,
                    help="how many contiguous chunks the token axis is dealt into")
    ap.add_argument("--max-gram-gb", type=float, default=160.0,
                    help="refuse a rung whose ESTIMATED HOST fp64 peak exceeds this, with the "
                         "arithmetic, instead of dying inside an allocator; 0 disables")
    ap.add_argument("--max-device-gb", type=float, default=-1.0,
                    help="the same guard for --device: -1 = 0.9 x the card's measured total "
                         "memory (0 on cpu, where --max-gram-gb already governs); 0 disables")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from safetensors.torch import load_file

    index = K3Index(args.model_dir)
    cfg = index.text_config
    device = canonical_device(args.device)
    experts_device = canonical_device(args.experts_device)
    cache = load_file(args.cache)
    missing = [k for k in ("latent_in", "y_lat", "router_topk_idx", "router_topk_w") if k not in cache]
    if missing:
        raise SystemExit(f"{args.cache} is missing {missing}; re-run k3_stream.py with --cache-set minimal")
    max_device_gb = default_device_budget_gb(device) if args.max_device_gb < 0 else args.max_device_gb
    # The BODY goes to --device; the packed experts go wherever --experts-device
    # says, and that defaults to the host.  `load_layer(index, layer, device,
    # device)` -- what this used to do -- put 14.80 GB of `weight_packed` plus
    # 0.93 GB of E8M0 scales on the compute card before a single token moved,
    # and none of it was priced.
    weights, timing = load_layer(index, args.layer, device, experts_device)
    fold = CoresetFold(weights, cache, cfg, device, chunk=args.chunk,
                       holdout=args.holdout, holdout_blocks=args.holdout_blocks,
                       max_gram_gb=args.max_gram_gb, max_device_gb=max_device_gb,
                       form_m=args.form_m)
    os.makedirs(args.out_dir, exist_ok=True)

    receipt: dict = {
        "model_dir": args.model_dir,
        "layer": args.layer,
        "cache": args.cache,
        "read": timing,
        "device": str(device),
        "experts_device": str(experts_device),
        "device_requested": args.device,
        "experts_device_requested": args.experts_device,
        "layer_bytes_on_device": weights.device_bytes(device),
        "devices": device_inventory(),
        "torch": torch.__version__,
        "features": {
            "selection": "phi = router-mass-weighted SiTU channel output",
            "refit_design": "psi = dense SiTU channel output on every fit token",
            "target": "y_lat, the PRE-norm fp32 routed accumulator",
        },
        "splits": {
            "holdout_mode": args.holdout,
            "holdout_blocks": args.holdout_blocks,
            "held_tokens": int(fold.held.sum()),
            "inner_tokens": int(fold.inner.sum()),
            "check_tokens": int(fold.check.sum()),
            "note": ("'blocks' holds out contiguous token ranges: on a real corpus an "
                     "interleaved holdout sits between its own neighbours and reads optimistic"),
        },
        "memory_budget": {
            "max_gram_gb": args.max_gram_gb,
            "max_device_gb": max_device_gb,
            "max_device_gb_source": "measured 0.9 x total" if args.max_device_gb < 0 else "given",
            "per_rung_estimate": {
                str(r): gram_memory_estimate(int(r), int(fold.y_lat.shape[1]))
                for r in [int(x) for x in args.rungs.split(",") if x.strip()]
            },
            # the same estimate the guard uses, resident terms included, so the
            # printed table and the refusal cannot disagree
            "per_rung_device_estimate": {
                str(r): fold.device_budget(int(r), int(fold.y_lat.shape[1]),
                                           include_score=True, include_form_m=args.form_m)
                for r in [int(x) for x in args.rungs.split(",") if x.strip()]
            },
        },
        "rungs": {},
    }
    for rung in [int(r) for r in args.rungs.split(",") if r.strip()]:
        log(f"layer {args.layer}: rung K={rung}")
        entry = run_rung(fold, rung, args.k_pool_factor, args.out_dir, args.layer, cfg,
                         index.hidden_size, form_m=args.form_m)
        receipt["rungs"][str(rung)] = entry
        gc.collect()
        log(f"  K={entry['width_installed']} routed ratio {entry['parameters']['routed_component_ratio']}x "
            f"heldout relL2 {entry['heldout']['y_lat_heldout']['relL2']:.4f} "
            f"update_scale {entry['ridge']['update_scale']:.3f} "
            f"veto_binding {entry['ridge']['veto_binding']}")
    peak = peak_rss_bytes()
    receipt["peak_rss_gigabytes"] = round(peak / 1e9, 3) if peak else None
    out = args.out or os.path.join(args.out_dir, f"L{args.layer:02d}_c14_receipt.json")
    receipt["receipt_sha256"] = write_receipt(out, receipt)
    log(f"receipt -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
