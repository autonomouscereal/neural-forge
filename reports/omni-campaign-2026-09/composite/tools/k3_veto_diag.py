#!/usr/bin/env python
"""k3_veto_diag -- the H3 (fit-mechanics) diagnostic on the real K3 layer-1 C14 refit.

Hypothesis H3: at K = 17,408 with --k-pool-factor 1 the refit Gram is
rank-deficient or badly conditioned, so THE LAW's lambda ladder cannot bring
`update_scale` to 1.0 regardless of the prior; a smaller rung, a k-pool above 1,
or an eigen-floored solve changes the outcome.

What one configuration (rung, k_pool_factor) does:

  * the SAME stage-1 / stage-2 selection and the SAME fp64 Gram accumulation as
    `c14_union_coreset.CoresetFold.refit` (CALLED through k3_dense_pass's
    `DualCoresetFold`, never copied), with `c14.law_abiding_solve_gram` swapped
    for `diag_solve_gram` for the duration of the call.  The diagnostic solver
    computes every number the law computes (the 9-rung check_sse and
    update_scale ladder, the free solve, the admission), KEEPS every ladder
    solution, adds the symmetric eigendecomposition of the full-fit Gram
    (condition number, spectrum census) and the eigen-floored solves of
    qwen-fold/qf_solve.py's `eigen_anchored_gram` (same arithmetic, several
    floors), and returns the law's chosen map when the law admits one -- or the
    FREE solve, flagged `law_refused: true`, when it does not.  Nothing is
    installed either way: no form-E file, no ffn.safetensors, no done marker.
  * every candidate map (free, each ridge rung, each eigen floor, the pinv, the
    prior itself, zero) is scored on the HELD rows against the teacher's y_lat
    (`write_scores`: relL2 / cos / cos_p05), after routed_expert_norm + up_proj
    (the 7168 write), and on the FIT split through the Gram identity
    ||A W^T - Y||^2 = tr(W G W^T) - 2 tr(W B) + ||Y||^2.

Nothing here changes THE LAW.  The receipt records what the law said next to
the measured numbers; the verdict on H3 is drawn from those numbers by the
`summarise` step and by the reader.

Modes:

  capture   run k3_dense_pass's dual stream through the first MoE layer (the
            layer-0 dense prefix is recomputed into a SCRATCH --out), stop at the
            moment `fit_layer` would build the fold, and write the fit cache
            (latent_in, y_lat, router_topk_idx, router_topk_w, fit/held masks)
            to --cache-dir.  With --then-rung the K=<rung> configuration runs in
            the same process on the in-memory weights and cache.
  diag      read the MoE layer's weights, load the cache, run one configuration.
  summarise collect the per-configuration receipts into one receipt.

Server:
    OC=/home/cereal/omni-campaign; PY=/home/cereal/model-surgery-lab/venv/bin/python
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 $PY k3_veto_diag.py capture \\
        --src-dir /mnt/lexar/models/moonshotai/Kimi-K3 \\
        --input-ids $OC/composite/corpus/k3_input_ids.safetensors \\
        --scratch-out /mnt/lexar/omni-campaign/k3/veto_diag/pass_out \\
        --cache-dir /mnt/lexar/omni-campaign/k3/veto_diag/cache \\
        --receipt-dir $OC/composite/receipts/k3_veto_diag --device cuda:0 --experts-device cpu \\
        --then-rung 17408 --then-k-pool-factor 1 --save-gram-dir /mnt/lexar/omni-campaign/k3/veto_diag/gram
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 $PY k3_veto_diag.py diag \\
        --src-dir ... --cache-dir ... --receipt-dir ... --rung 8704 --k-pool-factor 1 --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import uuid
from typing import Any

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import k3_dense_pass as kdp  # noqa: E402  (the tool under study; imports the lane by path)

if kdp.LANE_DIR not in sys.path:
    sys.path.insert(0, kdp.LANE_DIR)

import c14_union_coreset as c14  # noqa: E402
from k3_common import (  # noqa: E402
    K3Index,
    canonical_device,
    device_inventory,
    log,
    peak_rss_bytes,
    rms_norm,
    save_tensors,
    sha256_file,
    sha256_tensor,
    write_receipt,
    write_scores,
)

TOOL_NAME = "k3_veto_diag"
TOOL_VERSION = "1.0.0"
RECEIPT_SCHEMA = "omni-campaign.k3_veto_diag.v1"
HYPOTHESIS = ("H3 fit mechanics: at K=17,408 with k-pool 1 the refit Gram is rank-deficient or badly "
              "conditioned so the ladder cannot reach update_scale 1.0 regardless of the prior; a smaller "
              "rung, a k-pool above 1, or an eigen-floored solve changes the outcome")
CACHE_NAME = "k3_veto_diag_cache.safetensors"
CACHE_MANIFEST = "cache_manifest.json"
#: qf_solve.EIGEN_FLOOR_REL is 1e-12; the others show how the floored answer moves with the floor
EIGEN_FLOORS = (1e-12, 1e-10, 1e-8, 1e-6, 1e-4)
SPECTRUM_THRESHOLDS = (1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def fnum(x: float) -> float | str:
    """JSON cannot carry inf/nan; say so in words instead of writing `Infinity`."""
    x = float(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return "nan"
    return x


# --------------------------------------------------------------------------
# the eigen-floored solve (qwen-fold/qf_solve.eigen_anchored_gram, from a decomposition)
# --------------------------------------------------------------------------
def eigen_floored_from_decomposition(values: torch.Tensor, vectors: torch.Tensor, cross: torch.Tensor,
                                     prior: torch.Tensor, lam: float, floor_rel: float,
                                     ) -> tuple[torch.Tensor, dict[str, Any]]:
    """`qf_solve.eigen_anchored_gram` arithmetic on an already-decomposed Gram.

    G + lam I shares G's eigenvectors, so ONE `eigh` of the symmetrised Gram
    serves every lambda and every floor: the ridged eigenvalues are
    `values + lam`, floored at `floor_rel * max(values + lam)`, and the
    solution is V diag(1/clamped) V^T (B + lam prior^T).  Identical to qf_solve
    up to the rounding of `eigh(G) + lam` against `eigh(G + lam I)`.
    """
    ridged = values + lam
    top = float(ridged.max())
    floor = floor_rel * top
    lifted = int((ridged < floor).sum())
    clamped = ridged.clamp_min(floor)
    rhs = cross + lam * prior.T.to(cross.dtype)
    solution = vectors @ ((vectors.T @ rhs) / clamped.unsqueeze(1))
    return solution.T.contiguous(), {
        "solver": "eigen-decomposition with an explicit eigenvalue floor (qf_solve.eigen_anchored_gram arithmetic)",
        "ridge_absolute": float(lam),
        "eigen_floor_relative": float(floor_rel),
        "eigen_floor_absolute": float(floor),
        "eigenvalues_lifted_to_the_floor": lifted,
        "eigenvalue_min_before_floor": float(ridged.min()),
        "eigenvalue_max": top,
        "condition_after_floor": fnum(top / floor) if floor > 0 else "inf",
        "is_an_extra_regularisation": "yes -- the floor is a ridge the ladder did not ask for" if lifted else
                                      "no -- nothing was below the floor; this IS the ridged solve",
    }


def spectrum_record(values: torch.Tensor, diag: torch.Tensor, lam_by_rel: dict[float, float]) -> dict[str, Any]:
    v = values.double().cpu()
    d = diag.double().cpu()
    top = float(v.max())
    bottom = float(v.min())
    positive = v.clamp_min(0.0)
    p = positive / max(float(positive.sum()), 1e-300)
    entropy = float(-(p[p > 0] * p[p > 0].log()).sum())
    pct = torch.tensor([0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0], dtype=v.dtype)
    quant = torch.quantile(v, pct)
    return {
        "n": int(v.numel()),
        "eigenvalue_max": top,
        "eigenvalue_min": bottom,
        "trace": float(v.sum()),
        "nonpositive_eigenvalues": int((v <= 0).sum()),
        "count_below_rel_of_max": {f"{r:g}": int((v < r * top).sum()) for r in SPECTRUM_THRESHOLDS},
        "numerical_rank_at_rel": {f"{r:g}": int((v >= r * top).sum()) for r in SPECTRUM_THRESHOLDS},
        "percentiles": {f"{int(q * 100)}": float(x) for q, x in zip(pct.tolist(), quant.tolist())},
        "condition_number": fnum(top / bottom) if bottom > 0 else "inf (non-positive minimum eigenvalue)",
        "condition_at_floor_1e-12": fnum(top / max(bottom, 1e-12 * top)),
        "effective_rank_trace_over_max": float(v.sum() / top) if top > 0 else 0.0,
        "effective_rank_entropy": float(math.exp(entropy)),
        "condition_after_lambda": {f"{rel:g}": fnum((top + lam) / (bottom + lam)) if (bottom + lam) > 0 else "inf"
                                   for rel, lam in lam_by_rel.items()},
        "diag": {
            "min": float(d.min()), "max": float(d.max()), "mean": float(d.mean()),
            "median": float(d.median()),
            "count_zero": int((d <= 0).sum()),
            "count_below_1e-12_of_max": int((d < 1e-12 * float(d.max())).sum()),
            "count_below_1e-6_of_max": int((d < 1e-6 * float(d.max())).sum()),
        },
    }


# --------------------------------------------------------------------------
# the diagnostic solver that stands in for c14.law_abiding_solve_gram
# --------------------------------------------------------------------------
class DiagSink:
    """Where `diag_solve_gram` leaves what the law's solver throws away."""

    def __init__(self, save_gram_dir: str | None = None, tag: str = "", do_eigh: bool = True):
        self.save_gram_dir = save_gram_dir
        self.tag = tag
        self.do_eigh = do_eigh
        self.candidates: dict[str, torch.Tensor] = {}
        self.candidate_records: dict[str, dict[str, Any]] = {}
        self.record: dict[str, Any] = {}


_SINK: DiagSink | None = None


def _candidate(sink: DiagSink, name: str, solution: torch.Tensor, prior64: torch.Tensor, reference: float,
               gram_all: torch.Tensor, cross_all: torch.Tensor, y_fit_sq: float, extra: dict[str, Any] | None = None,
               ) -> dict[str, Any]:
    sol = solution.to(torch.float64)
    sse = c14._check_sse(sol, gram_all, cross_all, y_fit_sq)
    rec = {
        "norm_F": float(sol.norm()),
        "update_norm_F": float((sol - prior64).norm()),
        "update_scale": fnum((sol - prior64).norm() / reference) if reference > 0 else "inf (zero prior)",
        "fit_sse_from_gram_identity": float(sse),
        "fit_relL2_from_gram_identity": float(math.sqrt(max(sse, 0.0) / max(y_fit_sq, 1e-300))),
    }
    if extra:
        rec.update(extra)
    sink.candidates[name] = sol.float().contiguous()
    sink.candidate_records[name] = rec
    return rec


def diag_solve_gram(gram_inner: torch.Tensor, cross_inner: torch.Tensor,
                    gram_check: torch.Tensor, cross_check: torch.Tensor, check_y_sq: float,
                    prior: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """`c14.law_abiding_solve_gram`, instrumented.  Same ladder, same veto arithmetic,
    every solution kept, the Gram decomposed, nothing refused.

    CONSUMES its inputs exactly as the original does (inner += check in place).
    Returns (map, ladder_dict) so `CoresetFold.refit` proceeds; `ladder_dict`
    carries `law_refused` and the original refusal text when THE LAW would have
    raised, and the returned map is then the free solve -- for SCORING, never
    for an install.
    """
    sink = _SINK
    assert sink is not None, "diag_solve_gram needs a DiagSink"
    timings: dict[str, float] = {}
    started = time.time()
    reference = float(prior.norm())
    prior64 = prior.to(torch.float64)
    mean_diag_inner = float(torch.diagonal(gram_inner).mean())

    # -- 1. the check split, as the law does it ------------------------------
    check_sse: list[float] = []
    for rel in c14.LAMBDA_LADDER:
        inner = c14.prior_anchored_gram(gram_inner, cross_inner, prior, rel=rel)
        check_sse.append(c14._check_sse(inner, gram_check, cross_check, check_y_sq))
        del inner
    timings["check_split_ladder_seconds"] = round(time.time() - started, 1)

    # -- the full-fit system, in place, as the law does it --------------------
    gram_inner.add_(gram_check)
    cross_inner.add_(cross_check)
    gram_all, cross_all = gram_inner, cross_inner
    y_fit_sq = float(sink.record["y_fit_sq"])
    mean_diag_all = float(torch.diagonal(gram_all).mean())
    lam_by_rel = {rel: rel * mean_diag_all for rel in c14.LAMBDA_LADDER}

    if sink.save_gram_dir:
        os.makedirs(sink.save_gram_dir, exist_ok=True)
        path = os.path.join(sink.save_gram_dir, f"{sink.tag}_gram_all_fp64.safetensors")
        t0 = time.time()
        digest = save_tensors(path, {"gram_all": gram_all.contiguous(), "cross_all": cross_all.contiguous(),
                                     "prior_fp32": prior.float().contiguous()},
                              metadata={"tool": TOOL_NAME, "tag": sink.tag, "y_fit_sq": repr(y_fit_sq),
                                        "check_y_sq": repr(float(check_y_sq))})
        sink.record["gram_saved"] = {"path": path, "sha256": digest, "bytes": os.path.getsize(path),
                                     "seconds": round(time.time() - t0, 1)}
        log(f"  [{sink.tag}] Gram saved -> {path} ({os.path.getsize(path)/1e9:.2f} GB)", tag="diag")

    # -- 2. the veto ladder, every solution kept ------------------------------
    t0 = time.time()
    update_scale: list[float] = []
    ladder_records: list[dict[str, Any]] = []
    for rel in c14.LAMBDA_LADDER:
        full = c14.prior_anchored_gram(gram_all, cross_all, prior, rel=rel)
        scale = float((full - prior64).norm() / reference) if reference > 0 else float("inf")
        update_scale.append(scale)
        name = f"ridge_rel_{rel:g}"
        rec = _candidate(sink, name, full, prior64, reference, gram_all, cross_all, y_fit_sq,
                         extra={"lambda_relative": float(rel), "lambda_absolute": float(lam_by_rel[rel])})
        ladder_records.append({"lambda_relative": float(rel), "lambda_absolute": float(lam_by_rel[rel]),
                               "check_sse": float(check_sse[len(ladder_records)]), "update_scale": fnum(scale),
                               "candidate": name})
        del full
    free_scale = update_scale[0]
    sink.candidates["free"] = sink.candidates[f"ridge_rel_{c14.LAMBDA_LADDER[0]:g}"]
    sink.candidate_records["free"] = dict(sink.candidate_records[f"ridge_rel_{c14.LAMBDA_LADDER[0]:g}"],
                                          note="LAMBDA_LADDER[0] IS the free solve (rel 1e-6), as in the law")
    timings["veto_ladder_seconds"] = round(time.time() - t0, 1)
    log(f"  [{sink.tag}] ladder update_scale " + " ".join(f"{s:.3f}" for s in update_scale) +
        f" (free {free_scale:.3f})", tag="diag")

    # -- the law's own verdict, verbatim arithmetic ---------------------------
    rungs = [{"lambda_relative": rel, "check_sse": sse, "update_scale": scale}
             for rel, sse, scale in zip(c14.LAMBDA_LADDER, check_sse, update_scale)]
    admitted = [row for row in rungs if row["update_scale"] <= 1.0]
    monotone = all(check_sse[i] >= check_sse[i + 1] - 1e-12 for i in range(len(check_sse) - 1))
    if admitted:
        chosen = min(admitted, key=lambda row: row["check_sse"])
        law = {"law_refused": False, "law_message": None,
               "update_scale": chosen["update_scale"], "lambda_relative": chosen["lambda_relative"],
               "lambda_law_would_have_picked": admitted[0]["lambda_relative"]}
        returned = sink.candidates[f"ridge_rel_{chosen['lambda_relative']:g}"]
        returned_name = f"ridge_rel_{chosen['lambda_relative']:g}"
    else:
        message = (f"no lambda on the ladder brings update_scale to 1.0 (free {free_scale:.3f}); "
                   "THE LAW refuses the install")
        law = {"law_refused": True, "law_message": message,
               "update_scale": fnum(free_scale), "lambda_relative": c14.LAMBDA_LADDER[0],
               "lambda_law_would_have_picked": None}
        returned = sink.candidates["free"]
        returned_name = "free"
        log(f"  [{sink.tag}] THE LAW would have refused: {message} -- the free solve is returned for "
            "SCORING ONLY; nothing is installed", tag="diag")

    # -- 3. the spectrum and the eigen-floored solves -------------------------
    spectrum: dict[str, Any] | None = None
    eigen_records: dict[str, Any] = {}
    if sink.do_eigh:
        t0 = time.time()
        sym = 0.5 * (gram_all + gram_all.T)
        values, vectors = torch.linalg.eigh(sym)
        del sym
        timings["eigh_seconds"] = round(time.time() - t0, 1)
        spectrum = spectrum_record(values, torch.diagonal(gram_all), lam_by_rel)
        log(f"  [{sink.tag}] eigh in {timings['eigh_seconds']}s: max {spectrum['eigenvalue_max']:.4e} "
            f"min {spectrum['eigenvalue_min']:.4e} cond {spectrum['condition_number']} "
            f"rank@1e-12 {spectrum['numerical_rank_at_rel']['1e-12']}/{spectrum['n']}", tag="diag")
        t0 = time.time()
        lam_free = lam_by_rel[c14.LAMBDA_LADDER[0]]
        for floor_rel in EIGEN_FLOORS:
            sol, rec = eigen_floored_from_decomposition(values, vectors, cross_all, prior64, lam_free, floor_rel)
            name = f"eigen_floor_{floor_rel:g}"
            eigen_records[name] = rec
            _candidate(sink, name, sol, prior64, reference, gram_all, cross_all, y_fit_sq, extra=rec)
            del sol
        # the un-ridged pseudo-inverse at qf_solve's floor: lam = 0
        sol, rec = eigen_floored_from_decomposition(values, vectors, cross_all, prior64, 0.0, 1e-12)
        eigen_records["pinv_floor_1e-12"] = rec
        _candidate(sink, "pinv_floor_1e-12", sol, prior64, reference, gram_all, cross_all, y_fit_sq, extra=rec)
        del sol, values, vectors
        timings["eigen_solves_seconds"] = round(time.time() - t0, 1)
        for name in list(eigen_records):
            log(f"  [{sink.tag}] {name}: update_scale {sink.candidate_records[name]['update_scale']} "
                f"lifted {eigen_records[name]['eigenvalues_lifted_to_the_floor']}", tag="diag")

    # -- the prior itself and zero, as candidates -----------------------------
    _candidate(sink, "prior", prior64.clone(), prior64, reference, gram_all, cross_all, y_fit_sq,
               extra={"note": "the parent read-out column scaled by mean router mass; the map the law anchors to"})
    _candidate(sink, "zero", torch.zeros_like(prior64), prior64, reference, gram_all, cross_all, y_fit_sq,
               extra={"note": "predicting zero; relL2 1.0 by construction"})
    gc.collect()

    ladder = {
        **law,
        "free_solve_update_scale": fnum(free_scale),
        "lambda_selected_on": "internal fit split",
        "veto_binding": bool(free_scale > 1.0),
        "lambdas_vetoed": int(len(rungs) - len(admitted)),
        "lambdas_admitted": int(len(admitted)),
        "min_update_scale_on_ladder": fnum(min(update_scale)),
        "chosen_is_largest_lambda": bool(admitted and chosen["lambda_relative"] == c14.LAMBDA_LADDER[-1]),
        "check_sse_monotone_decreasing_in_lambda": bool(monotone),
        "ladder": ladder_records,
        "returned_candidate": returned_name,
        "returned_is_an_install": False,
        "prior_norm_F": reference,
        "mean_diag_gram_inner": mean_diag_inner,
        "mean_diag_gram_all": mean_diag_all,
        "check_y_sq": float(check_y_sq),
        "y_fit_sq": y_fit_sq,
        "spectrum": spectrum,
        "eigen_solves": eigen_records,
        "timings": timings,
        "note": ("law_abiding_solve_gram instrumented by k3_veto_diag; identical ladder arithmetic; the law's "
                 "refusal is recorded, not raised, so the held-out numbers can be measured"),
    }
    sink.record["ladder"] = ladder
    return returned.clone(), ladder


# --------------------------------------------------------------------------
# one configuration
# --------------------------------------------------------------------------
def score_candidates(fold: c14.CoresetFold, bank: c14.ExpertBank, sink: DiagSink) -> dict[str, Any]:
    """Held-row scores for every candidate off ONE psi pass (what `CoresetFold.score` does per map)."""
    eps = float(fold.cfg["rms_norm_eps"])
    body = fold.w.body
    rows = torch.nonzero(fold.held).flatten().to(fold.device)
    psi_parts, targets = [], []
    for start in range(0, rows.numel(), fold.chunk):
        piece = rows[start:start + fold.chunk]
        psi_parts.append(bank.psi(fold.latent[piece], fold.beta, fold.linear_beta).float())
        targets.append(fold.y_lat[piece])
    psi = torch.cat(psi_parts)
    target = torch.cat(targets)
    del psi_parts, targets
    norm_w = body["block_sparse_moe.routed_expert_norm.weight"]
    up_w = body["block_sparse_moe.routed_expert_up_proj.weight"]
    dtype = up_w.dtype
    target_write = F.linear(rms_norm(target.to(dtype), norm_w, eps), up_w).float()
    out: dict[str, Any] = {}
    for name, w2p in sink.candidates.items():
        pred = F.linear(psi, w2p.to(psi.device))
        rec = {"y_lat_heldout": write_scores(pred, target)}
        pred_write = F.linear(rms_norm(pred.to(dtype), norm_w, eps), up_w).float()
        rec["routed_write_heldout"] = write_scores(pred_write, target_write)
        out[name] = rec
        del pred, pred_write
    del psi, target, target_write
    return out


def run_config(weights, cache: dict[str, torch.Tensor], cfg: dict, fit_mask: torch.Tensor, held_mask: torch.Tensor,
               *, rung: int, k_pool_factor: int, device: torch.device, name: str,
               chunk: int = 1024, check_blocks: int = 24, max_gram_gb: float = 160.0, max_device_gb: float = -1.0,
               save_gram_dir: str | None = None, do_eigh: bool = True) -> tuple[dict[str, Any], DiagSink]:
    global _SINK
    started = time.time()
    device = canonical_device(device)
    budget = c14.default_device_budget_gb(device) if max_device_gb < 0 else float(max_device_gb)
    fold = kdp.DualCoresetFold(weights, cache, cfg, device, fit_mask, held_mask, check_blocks=check_blocks,
                               chunk=chunk, max_gram_gb=max_gram_gb, max_device_gb=budget)
    stage1_only = k_pool_factor <= 1
    k_pool = rung if stage1_only else min(rung * k_pool_factor, fold.n_experts * fold.intermediate)
    log(f"[{name}] rung {rung} k_pool_factor {k_pool_factor} -> k_pool {k_pool}; fit {int(fold.fit.sum())} "
        f"inner {int(fold.inner.sum())} check {int(fold.check.sum())} held {int(fold.held.sum())}", tag="diag")
    channels, stage1_stats = fold.stage1(k_pool)
    channels, stage2_stats = fold.stage2(channels, rung)
    log(f"[{name}] selected {int(channels.shape[0])} channels (stage1 {fold.timings.get('stage1_seconds')}s, "
        f"stage2 {fold.timings.get('stage2_seconds')}s)", tag="diag")

    sink = DiagSink(save_gram_dir=save_gram_dir, tag=name, do_eigh=do_eigh)
    fit_rows = torch.nonzero(fold.fit).flatten().to(fold.device)
    sink.record["y_fit_sq"] = float(fold.y_lat[fit_rows].double().pow(2).sum())
    sink.record["y_held_sq"] = float(fold.y_lat[torch.nonzero(fold.held).flatten().to(fold.device)].double().pow(2).sum())
    del fit_rows
    original = c14.law_abiding_solve_gram
    _SINK = sink
    try:
        c14.law_abiding_solve_gram = diag_solve_gram
        solved, bank, ladder = fold.refit(channels)
    finally:
        c14.law_abiding_solve_gram = original
        _SINK = None
    del solved  # never installed
    sink.record["channels"] = bank.channels.cpu().clone()

    # the prior's ingredients, recomputed from the same objects refit used
    experts = bank.channels[:, 0].to(fold.device)
    n_fit = max(int(fold.fit.sum()), 1)
    mass = (fold.expert_mass()[experts] / n_fit).cpu().double()
    counts = fold._expert_counts[experts].double().cpu()
    parent_w2 = bank.w2.double().cpu()
    prior_rec = {
        "definition": "prior = W2_e[:, j] (verbatim dequantized parent column) * m_c, "
                      "m_c = E_t[w_e(t) 1[e in top-k]] over the FIT tokens (c14 refit lines 900-908)",
        "n_fit": n_fit,
        "mass_mean": float(mass.mean()), "mass_min": float(mass.min()), "mass_max": float(mass.max()),
        "mass_median": float(mass.median()),
        "routed_fraction_mean": float((counts / n_fit).mean()),
        "mean_router_weight_when_routed": float((mass * n_fit / counts.clamp_min(1)).mean()),
        "prior_norm_F": float(ladder["prior_norm_F"]),
        "parent_w2_columns_norm_F": float(parent_w2.norm()),
        "prior_over_parent_norm_ratio": float(ladder["prior_norm_F"] / max(float(parent_w2.norm()), 1e-300)),
        "router_config": {k: cfg.get(k) for k in ("num_experts", "num_experts_per_tok", "moe_renormalize",
                                                   "routed_scaling_factor", "num_shared_experts",
                                                   "moe_intermediate_size", "routed_expert_hidden_size")},
    }
    for rec_name, rec in sink.candidate_records.items():
        rec["update_norm_over_parent_w2_norm"] = float(rec["update_norm_F"] / max(float(parent_w2.norm()), 1e-300))
        rec["norm_over_parent_w2_norm"] = float(rec["norm_F"] / max(float(parent_w2.norm()), 1e-300))
    del parent_w2

    t0 = time.time()
    held = score_candidates(fold, bank, sink)
    score_seconds = round(time.time() - t0, 1)
    for cand, rec in held.items():
        sink.candidate_records[cand]["heldout"] = rec
    free_held = held["free"]["y_lat_heldout"]["relL2"]
    log(f"[{name}] K={int(bank.w1.shape[0])} free update_scale {ladder['free_solve_update_scale']} "
        f"heldout relL2 free {free_held:.4f} prior {held['prior']['y_lat_heldout']['relL2']:.4f} "
        f"law_refused {ladder['law_refused']}", tag="diag")

    width = int(bank.w1.shape[0])
    receipt = {
        "config": {"name": name, "rung": int(rung), "k_pool_factor": int(k_pool_factor), "k_pool": int(k_pool),
                   "stage1_only": bool(stage1_only), "width_installed_would_be": width,
                   "chunk": chunk, "check_blocks": check_blocks, "device": str(device),
                   "max_gram_gb": max_gram_gb, "max_device_gb": budget},
        "tokens": {"total": fold.tokens, "fit": int(fold.fit.sum()), "inner": int(fold.inner.sum()),
                   "check": int(fold.check.sum()), "held": int(fold.held.sum()),
                   "y_fit_sq": sink.record["y_fit_sq"], "y_held_sq": sink.record["y_held_sq"]},
        "stage1": stage1_stats,
        "stage2": stage2_stats,
        "gram": {"K": width, "out_dim": int(fold.y_lat.shape[1]),
                 "memory": ladder.get("memory"), "device_memory": ladder.get("device_memory"),
                 "mean_diag_gram_all": ladder["mean_diag_gram_all"],
                 "spectrum": ladder["spectrum"], "saved": sink.record.get("gram_saved")},
        "prior": prior_rec,
        "law": {k: ladder[k] for k in ("law_refused", "law_message", "update_scale", "lambda_relative",
                                       "lambda_law_would_have_picked", "free_solve_update_scale", "veto_binding",
                                       "lambdas_vetoed", "lambdas_admitted", "min_update_scale_on_ladder",
                                       "chosen_is_largest_lambda", "check_sse_monotone_decreasing_in_lambda",
                                       "ladder", "returned_candidate", "returned_is_an_install", "note")},
        "eigen_solves": ladder["eigen_solves"],
        "candidates": sink.candidate_records,
        "timings": {**{k: v for k, v in fold.timings.items()}, **ladder["timings"],
                    "score_seconds": score_seconds, "config_seconds": round(time.time() - started, 1)},
        "peak_rss_bytes": peak_rss_bytes(),
        "device_peak_bytes": (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None),
    }
    del bank, fold
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return receipt, sink


# --------------------------------------------------------------------------
# capture: the dual stream up to the first MoE fit
# --------------------------------------------------------------------------
class CaptureDone(Exception):
    pass


class CaptureRun(kdp.DualStreamRun):
    """`DualStreamRun` that stops where `fit_layer` would build the fold and keeps its inputs."""

    captured: dict[str, Any] | None = None

    def fit_layer(self, layer_idx, weights, raw_body, cfg, t_taps, f_latent):
        self.captured = {
            "layer": layer_idx, "weights": weights, "cfg": cfg,
            "cache": {"latent_in": f_latent, "y_lat": t_taps["y_lat"],
                      "router_topk_idx": t_taps["router_topk_idx"], "router_topk_w": t_taps["router_topk_w"]},
            "router": {"experts_touched": int(t_taps.get("experts_touched", 0))},
        }
        raise CaptureDone()


def write_cache(cache_dir: str, captured: dict[str, Any], corpus, run: kdp.DualStreamRun,
                compute_dtype: str) -> dict[str, Any]:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, CACHE_NAME)
    tensors = {
        "latent_in": captured["cache"]["latent_in"].detach().cpu().contiguous(),
        "y_lat": captured["cache"]["y_lat"].detach().cpu().float().contiguous(),
        "router_topk_idx": captured["cache"]["router_topk_idx"].detach().cpu().contiguous(),
        "router_topk_w": captured["cache"]["router_topk_w"].detach().cpu().float().contiguous(),
        "fit_mask": corpus.fit.cpu().contiguous(),
        "held_mask": corpus.held.cpu().contiguous(),
    }
    digest = save_tensors(path, tensors, metadata={"tool": TOOL_NAME, "layer": str(captured["layer"]),
                                                   "corpus_sha256": corpus.sha256})
    manifest = {
        "schema": RECEIPT_SCHEMA + ".cache", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "written_utc": utc_now(), "layer": captured["layer"], "path": path, "sha256": digest,
        "bytes": os.path.getsize(path), "corpus": corpus.record(), "model": run.receipt.get("model"),
        "compute_dtype": compute_dtype,
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": sha256_tensor(v)}
                    for k, v in tensors.items()},
        "fit": run.receipt.get("fit"),
        "layer0": {k: run.receipt["layers"].get("0", {}).get(k) for k in
                   ("stream_divergence", "prefix_out_sha256", "layer_seconds")},
        "router": captured["router"],
    }
    write_receipt(os.path.join(cache_dir, CACHE_MANIFEST), manifest)
    return manifest


def load_cache(cache_dir: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    with open(os.path.join(cache_dir, CACHE_MANIFEST), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    path = os.path.join(cache_dir, CACHE_NAME)
    digest = sha256_file(path)
    if digest != manifest["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest} != manifest {manifest['sha256']}")
    return load_file(path), manifest


def base_receipt(args, mode: str) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA, "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "tool_under_study": {"path": os.path.abspath(kdp.__file__), "sha256": sha256_file(kdp.__file__),
                             "version": kdp.TOOL_VERSION},
        "lane_dir": kdp.LANE_DIR, "lane_sha256": kdp.lane_file_digests(),
        "hypothesis": HYPOTHESIS, "mode": mode, "started_utc": utc_now(),
        "arguments": {k: v for k, v in vars(args).items() if k != "func"},
        "devices": device_inventory(), "torch": torch.__version__, "python": sys.version.split()[0],
        "law_unchanged": ("c14.law_abiding_solve_gram is swapped for diag_solve_gram ONLY inside this tool's "
                          "refit call; the lane file is untouched and nothing is installed"),
    }


def finish_receipt(receipt: dict[str, Any], receipt_dir: str, name: str) -> str:
    receipt["finished_utc"] = utc_now()
    receipt["peak_rss_bytes"] = peak_rss_bytes()
    os.makedirs(receipt_dir, exist_ok=True)
    path = os.path.join(receipt_dir, f"k3_veto_diag_{name}.json")
    digest = write_receipt(path, receipt)
    log(f"receipt -> {path} (sha256 {digest[:12]}...)", tag="diag")
    return path


def config_name(rung: int, k_pool_factor: int) -> str:
    return f"K{rung}_kp{k_pool_factor}"


def cmd_capture(args) -> int:
    pass_argv = ["--src-dir", args.src_dir, "--input-ids", args.input_ids, "--layers", args.layers,
                 "--rung", str(args.then_rung or 17408), "--k-pool-factor", str(args.then_k_pool_factor),
                 "--device", args.device, "--experts-device", args.experts_device,
                 "--kda-backend", args.kda_backend, "--compute-dtype", args.compute_dtype,
                 "--out", args.scratch_out, "--receipt-dir", os.path.join(args.receipt_dir, "capture_pass"),
                 "--row-chunk", str(args.row_chunk), "--token-chunk", str(args.token_chunk),
                 "--chunk", str(args.chunk), "--holdout-blocks", str(args.check_blocks),
                 "--max-gram-gb", str(args.max_gram_gb), "--max-device-gb", str(args.max_device_gb),
                 "--snapshot-every", "0", "--no-held-rows"]
    if args.no_layer0_receipt:
        pass_argv.append("--no-layer0-receipt")
    if args.digest_receipt:
        pass_argv += ["--digest-receipt", args.digest_receipt]
    if args.layer0_import_receipt:
        pass_argv += ["--layer0-import-receipt", args.layer0_import_receipt]
    pargs = kdp.build_parser().parse_args(pass_argv)
    pargs.model_dir = pargs.src_dir
    torch.manual_seed(0)
    policy = kdp.resolve_share_policy(pargs.src_dir, no_share_read=pargs.no_share_read,
                                      allow_share_read=pargs.allow_share_read,
                                      share_rate_receipt=pargs.share_rate_receipt)
    pargs.no_share_read = policy["no_share_read"]
    if not os.path.isdir(pargs.chimera_src or ""):
        pargs.chimera_src = None
    index = K3Index(pargs.src_dir)
    corpus = kdp.Corpus(pargs.input_ids, [])
    stager = kdp.ShardStager(pargs.src_dir, None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id=uuid.uuid4().hex[:12])
    receipt = base_receipt(args, "capture")
    receipt["pass_arguments"] = pass_argv
    run = CaptureRun(pargs, index, corpus, stager, policy)
    started = time.time()
    try:
        try:
            run.run()
        except CaptureDone:
            pass
    finally:
        stager.close()
    if run.captured is None:
        raise SystemExit("the window ended without reaching a MoE fit; nothing captured")
    receipt["capture_seconds"] = round(time.time() - started, 1)
    receipt["layer"] = run.captured["layer"]
    receipt["pass_receipt_layers"] = {k: {kk: vv for kk, vv in v.items()
                                          if kk in ("layer", "moe", "read", "teacher_compute_seconds",
                                                    "dequant_seconds", "router", "stream_divergence",
                                                    "folded", "layer_seconds", "finite_checks")}
                                      for k, v in run.receipt["layers"].items()}
    compute_dtype = str(run.captured["weights"].body["block_sparse_moe.routed_expert_up_proj.weight"].dtype)
    receipt["cache"] = write_cache(args.cache_dir, run.captured, corpus, run, compute_dtype)
    log(f"cache -> {receipt['cache']['path']} ({receipt['cache']['bytes']/1e9:.2f} GB) in "
        f"{receipt['capture_seconds']}s", tag="diag")
    receipt["model"] = run.receipt.get("model")
    receipt["corpus"] = corpus.record()
    if args.then_rung:
        name = config_name(args.then_rung, args.then_k_pool_factor)
        cache = run.captured["cache"]
        cfg = run.captured["cfg"]
        config_receipt, _ = run_config(
            run.captured["weights"], cache, cfg, corpus.fit, corpus.held, rung=args.then_rung,
            k_pool_factor=args.then_k_pool_factor, device=canonical_device(args.device), name=name,
            chunk=args.chunk, check_blocks=args.check_blocks, max_gram_gb=args.max_gram_gb,
            max_device_gb=args.max_device_gb, save_gram_dir=args.save_gram_dir, do_eigh=not args.no_eigh)
        receipt.update(config_receipt)
        receipt["mode"] = "capture+diag"
        finish_receipt(receipt, args.receipt_dir, name)
    else:
        finish_receipt(receipt, args.receipt_dir, "capture")
    return 0


def cmd_diag(args) -> int:
    torch.manual_seed(0)
    device = canonical_device(args.device)
    experts_device = canonical_device(args.experts_device or args.device)
    index = K3Index(args.src_dir)
    stager = kdp.ShardStager(args.src_dir, None, enabled=False, window=1, prefetch=0,
                             delete_after_consume=False, run_id=uuid.uuid4().hex[:12])
    receipt = base_receipt(args, "diag")
    cache, manifest = load_cache(args.cache_dir)
    layer_idx = int(manifest["layer"])
    receipt["layer"] = layer_idx
    receipt["cache"] = {k: manifest[k] for k in ("path", "sha256", "bytes", "written_utc", "layer", "corpus",
                                                 "model", "compute_dtype", "layer0", "router") if k in manifest}
    fit_mask = cache.pop("fit_mask").to(torch.bool)
    held_mask = cache.pop("held_mask").to(torch.bool)
    t0 = time.time()
    weights, raw_body, timing = kdp.read_layer(index, layer_idx, stager, device, experts_device,
                                               kdp.torch_dtype(args.compute_dtype),
                                               progress_tag=f"L{layer_idx:02d}")
    del raw_body
    receipt["read"] = timing
    name = args.name or config_name(args.rung, args.k_pool_factor)
    cfg = index.text_config
    config_receipt, _ = run_config(
        weights, cache, cfg, fit_mask, held_mask, rung=args.rung, k_pool_factor=args.k_pool_factor,
        device=device, name=name, chunk=args.chunk, check_blocks=args.check_blocks,
        max_gram_gb=args.max_gram_gb, max_device_gb=args.max_device_gb, save_gram_dir=args.save_gram_dir,
        do_eigh=not args.no_eigh)
    receipt.update(config_receipt)
    receipt["wall_seconds"] = round(time.time() - t0, 1)
    finish_receipt(receipt, args.receipt_dir, name)
    stager.close()
    return 0


def cmd_summarise(args) -> int:
    rows = []
    receipts = {}
    for entry in sorted(os.listdir(args.receipt_dir)):
        if not (entry.startswith("k3_veto_diag_K") and entry.endswith(".json")):
            continue
        with open(os.path.join(args.receipt_dir, entry), "r", encoding="utf-8") as handle:
            rec = json.load(handle)
        if "config" not in rec:
            continue
        receipts[entry] = rec
        cands = rec["candidates"]
        spec = (rec.get("gram") or {}).get("spectrum") or {}
        law = rec["law"]
        row = {
            "receipt": entry, "name": rec["config"]["name"], "rung": rec["config"]["rung"],
            "k_pool_factor": rec["config"]["k_pool_factor"], "k_pool": rec["config"]["k_pool"],
            "width": rec["gram"]["K"], "device": rec["config"]["device"],
            "law_refused": law["law_refused"], "free_solve_update_scale": law["free_solve_update_scale"],
            "min_update_scale_on_ladder": law["min_update_scale_on_ladder"],
            "lambdas_admitted": law["lambdas_admitted"],
            "gram_condition_number": spec.get("condition_number"),
            "gram_eigenvalue_max": spec.get("eigenvalue_max"), "gram_eigenvalue_min": spec.get("eigenvalue_min"),
            "gram_numerical_rank_at_1e-12": (spec.get("numerical_rank_at_rel") or {}).get("1e-12"),
            "gram_numerical_rank_at_1e-8": (spec.get("numerical_rank_at_rel") or {}).get("1e-8"),
            "gram_condition_after_free_lambda": (spec.get("condition_after_lambda") or {}).get("1e-06"),
            "prior_norm_F": rec["prior"]["prior_norm_F"], "prior_mass_mean": rec["prior"]["mass_mean"],
            "parent_w2_columns_norm_F": rec["prior"]["parent_w2_columns_norm_F"],
            "free_norm_F": cands["free"]["norm_F"],
            "free_norm_over_parent_w2_norm": cands["free"]["norm_over_parent_w2_norm"],
            "heldout_relL2": {c: cands[c]["heldout"]["y_lat_heldout"]["relL2"] for c in cands if "heldout" in cands[c]},
            "heldout_cos": {c: cands[c]["heldout"]["y_lat_heldout"]["cos"] for c in cands if "heldout" in cands[c]},
            "routed_write_relL2": {c: cands[c]["heldout"]["routed_write_heldout"]["relL2"] for c in cands
                                   if "heldout" in cands[c]},
            "fit_relL2": {c: cands[c]["fit_relL2_from_gram_identity"] for c in cands},
            "update_scale": {c: cands[c]["update_scale"] for c in cands},
            "eigen_floor_1e-12_lifted": ((rec.get("eigen_solves") or {}).get("eigen_floor_1e-12") or {}).get(
                "eigenvalues_lifted_to_the_floor"),
            "timings": rec.get("timings"),
        }
        rows.append(row)
    summary = base_receipt(args, "summarise")
    summary["configurations"] = rows
    summary["per_config_receipts"] = {k: {"sha256": sha256_file(os.path.join(args.receipt_dir, k))}
                                      for k in receipts}
    if args.verdict:
        summary["verdict"] = args.verdict
    if args.note:
        summary["verdict_note"] = args.note
    summary["finished_utc"] = utc_now()
    path = os.path.join(args.receipt_dir, "k3_veto_diag_receipt.json")
    digest = write_receipt(path, summary)
    log(f"summary -> {path} (sha256 {digest[:12]}...)", tag="diag")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    def common(p: argparse.ArgumentParser, capture: bool) -> None:
        p.add_argument("--src-dir", default=kdp.DEFAULT_SRC_DIR)
        p.add_argument("--cache-dir", required=True)
        p.add_argument("--receipt-dir", required=True)
        p.add_argument("--device", default="cpu")
        p.add_argument("--experts-device", default="cpu")
        p.add_argument("--compute-dtype", default="auto", choices=("auto", "bf16", "fp32"))
        p.add_argument("--chunk", type=int, default=1024)
        p.add_argument("--check-blocks", type=int, default=24)
        p.add_argument("--max-gram-gb", type=float, default=160.0)
        p.add_argument("--max-device-gb", type=float, default=-1.0)
        p.add_argument("--save-gram-dir", default=None, help="save gram_all/cross_all/prior (fp64) here")
        p.add_argument("--no-eigh", action="store_true", help="skip the eigendecomposition and floored solves")

    cap = sub.add_parser("capture")
    common(cap, True)
    cap.add_argument("--input-ids", required=True)
    cap.add_argument("--layers", default="0-1")
    cap.add_argument("--scratch-out", required=True, help="SCRATCH --out for the dual pass (never dense/ canonical)")
    cap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    cap.add_argument("--row-chunk", type=int, default=1)
    cap.add_argument("--token-chunk", type=int, default=1024)
    cap.add_argument("--no-layer0-receipt", action="store_true")
    cap.add_argument("--layer0-import-receipt", default=None)
    cap.add_argument("--digest-receipt", default=None)
    cap.add_argument("--then-rung", type=int, default=None)
    cap.add_argument("--then-k-pool-factor", type=int, default=1)
    cap.set_defaults(func=cmd_capture)

    dg = sub.add_parser("diag")
    common(dg, False)
    dg.add_argument("--rung", type=int, required=True)
    dg.add_argument("--k-pool-factor", type=int, default=1)
    dg.add_argument("--name", default=None)
    dg.set_defaults(func=cmd_diag)

    sm = sub.add_parser("summarise")
    sm.add_argument("--receipt-dir", required=True)
    sm.add_argument("--verdict", default=None, choices=(None, "supported", "refuted", "inconclusive"))
    sm.add_argument("--note", default=None)
    sm.set_defaults(func=cmd_summarise)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
