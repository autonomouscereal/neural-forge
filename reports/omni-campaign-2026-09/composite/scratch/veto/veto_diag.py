#!/usr/bin/env python3
"""veto_diag.py -- H1 diagnostic on the c14 `update_scale` veto, real Kimi-K3 layer 1.

H1 (structural): the prior for a routed-expert union is (near) zero or on a
different scale from the least-squares answer, so update_scale =
||W - prior||_F / ||prior||_F is unbounded by construction and THE LAW cannot
pass however good the fit is.

This is a SCRATCH driver.  It imports the canonical composite/tools/k3_dense_pass.py
by path (never edited) and monkeypatches, in memory only,

  * c14.law_abiding_solve_gram -- the SAME arithmetic (same ladder, same
    prior_anchored_gram, same check split), but every operand of update_scale is
    recorded: ||prior||_F, ||W(lambda)||_F, ||W(lambda) - prior||_F, the ratio,
    the fit-split SSE of every rung, and every full-fit solution is kept (fp32,
    host) so it can be scored on the HELD rows afterwards.
  * c14.CoresetFold.refit -- records the scales of what enters the Gram
    (latent_in RMS, y_lat RMS, psi RMS off the Gram trace), the router-mass
    statistics that build the prior, the unscaled parent read-out bank ||W2||_F,
    an alternative prior scaled by E[w_e | routed], per-column direction/scale
    agreement of the free solve with the parent column, and the held-out
    y_lat / routed-write scores of every ladder rung, the prior, the alternative
    prior, the unscaled parent bank and the zero map.

THE VETO IS NOT ENFORCED IN THIS RUN.  When no rung is admitted the refusal
message is recorded verbatim under `law.refusal_message` and the free
(rel = 1e-6) solve is returned so that run_rung's own score() and the pass's
stream_divergence run on it -- into a SCRATCH --out, never the canonical one.
That is written into the receipt as `veto_bypassed_for_diagnosis: true`.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

TOOLS = os.environ.get("K3_TOOLS_DIR", "/home/cereal/omni-campaign/composite/tools")
sys.path.insert(0, TOOLS)
import k3_dense_pass as kdp  # noqa: E402

c14 = kdp.c14
DIAG_PATH = os.environ["VETO_DIAG_JSON"]

DIAG: dict = {
    "tool": "veto_diag.py (scratch; canonical k3_dense_pass.py + c14 imported by path, patched in memory)",
    "hypothesis": "H1 structural: prior (near) zero or on a different scale -> update_scale unbounded by construction",
    "veto_bypassed_for_diagnosis": True,
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "canonical_sha256": {
        "k3_dense_pass.py": kdp.sha256_file(os.path.join(TOOLS, "k3_dense_pass.py")),
        **kdp.lane_file_digests(),
    },
    "ladder": list(c14.LAMBDA_LADDER),
    "argv": sys.argv[1:],
}
CTX: dict = {}
STORE: dict[float, torch.Tensor] = {}


def say(msg: str) -> None:
    print(f"[veto_diag {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def dump() -> None:
    DIAG["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    tmp = DIAG_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(DIAG, fh, indent=1, sort_keys=True, default=lambda o: float(o))
    os.replace(tmp, DIAG_PATH)


def fnorm(t: torch.Tensor) -> float:
    return float(t.double().norm())


def rms(t: torch.Tensor) -> float:
    return float(t.double().pow(2).mean().sqrt())


def quantiles(t: torch.Tensor) -> dict:
    t = t.double().flatten().cpu()
    qs = torch.tensor([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0], dtype=t.dtype)
    v = torch.quantile(t, qs)
    return {"min": float(v[0]), "p05": float(v[1]), "p25": float(v[2]), "median": float(v[3]),
            "p75": float(v[4]), "p95": float(v[5]), "max": float(v[6]), "mean": float(t.mean())}


# --------------------------------------------------------------------------
# 1. the solver, instrumented -- same arithmetic as c14.law_abiding_solve_gram
# --------------------------------------------------------------------------
def instrumented_solve(gram_inner, cross_inner, gram_check, cross_check, check_y_sq, prior):
    ladder = c14.LAMBDA_LADDER
    reference = float(prior.norm())  # exactly c14 line 491 (fp32 prior)
    law = {
        "formula": "update_scale(lambda) = ||W_full(lambda) - prior||_F / ||prior||_F ; "
                   "W_full(lambda) = argmin ||A W^T - Y||^2 + lam ||W - prior||^2 on the full fit split, "
                   "lam = lambda_relative * mean(diag G_full); free = lambda_relative 1e-6",
        "reference_prior_fro_norm_fp32": reference,
        "prior_fro_norm_fp64": fnorm(prior),
        "prior_shape": list(prior.shape),
        "prior_abs_mean": float(prior.double().abs().mean()),
        "prior_abs_max": float(prior.double().abs().max()),
        "prior_exact_zero_fraction": float((prior == 0).double().mean()),
        "prior_column_norm_quantiles": quantiles(prior.double().norm(dim=0)),
        "gram_inner_diag_mean": float(torch.diagonal(gram_inner).mean()),
        "gram_check_diag_mean": float(torch.diagonal(gram_check).mean()),
        "cross_inner_fro_norm": fnorm(cross_inner),
        "cross_check_fro_norm": fnorm(cross_check),
        "check_y_sq": float(check_y_sq),
        "rungs": [],
    }
    say(f"prior: ||prior||_F = {reference:.6e}, abs mean {law['prior_abs_mean']:.3e}, "
        f"zero fraction {law['prior_exact_zero_fraction']:.4f}")
    # -- 1. check split (identical to c14) --------------------------------
    check_sse = []
    for rel in ladder:
        t0 = time.time()
        inner = c14.prior_anchored_gram(gram_inner, cross_inner, prior, rel=rel)
        sse = c14._check_sse(inner, gram_check, cross_check, check_y_sq)
        check_sse.append(sse)
        say(f"pass1 rel={rel:g}: check_sse {sse:.6e} (||W_inner||_F {fnorm(inner):.4e}) {time.time()-t0:.1f}s")
        del inner
    gram_inner.add_(gram_check)
    cross_inner.add_(cross_check)
    gram_all, cross_all = gram_inner, cross_inner
    law["gram_full_diag_mean"] = float(torch.diagonal(gram_all).mean())
    law["gram_full_trace"] = float(torch.diagonal(gram_all).sum())
    law["cross_full_fro_norm"] = fnorm(cross_all)
    fit_y_sq = float(CTX["fit_y_sq"])
    law["fit_y_sq"] = fit_y_sq
    # -- 2. the veto operands, full fit (identical solves) ----------------
    update_scale = []
    for rel, sse in zip(ladder, check_sse):
        t0 = time.time()
        full = c14.prior_anchored_gram(gram_all, cross_all, prior, rel=rel)
        diff_norm = float((full - prior).norm())          # exactly c14 line 509 numerator
        scale = diff_norm / reference
        update_scale.append(scale)
        w_norm = fnorm(full)
        fit_sse = c14._check_sse(full, gram_all, cross_all, fit_y_sq)
        row = {
            "lambda_relative": rel,
            "lam_absolute": rel * law["gram_full_diag_mean"],
            "check_sse": sse,
            "check_relL2_from_sse": math.sqrt(max(sse, 0.0) / max(check_y_sq, 1e-30)),
            "fit_sse_full": fit_sse,
            "fit_relL2_from_sse": math.sqrt(max(fit_sse, 0.0) / max(fit_y_sq, 1e-30)),
            "W_fro_norm": w_norm,
            "W_minus_prior_fro_norm": diff_norm,
            "W_over_prior_norm_ratio": w_norm / max(reference, 1e-30),
            "update_scale": scale,
            "admitted_by_law": bool(scale <= 1.0),
        }
        law["rungs"].append(row)
        say(f"pass2 rel={rel:g}: ||W||_F {w_norm:.4e} ||W-prior||_F {diff_norm:.4e} "
            f"update_scale {scale:.3f} fit relL2 {row['fit_relL2_from_sse']:.4f} {time.time()-t0:.1f}s")
        STORE[rel] = full.float().cpu()
        del full
    free_scale = update_scale[0]
    rungs = [{"lambda_relative": rel, "check_sse": sse, "update_scale": scale}
             for rel, sse, scale in zip(ladder, check_sse, update_scale)]
    admitted = [r for r in rungs if r["update_scale"] <= 1.0]
    law["free_solve_update_scale"] = free_scale
    law["free_solve_is_81_642_check"] = f"free {free_scale:.3f}"
    law["admitted_count"] = len(admitted)
    monotone = all(check_sse[i] >= check_sse[i + 1] - 1e-12 for i in range(len(check_sse) - 1))
    if not admitted:
        msg = (f"no lambda on the ladder brings update_scale to 1.0 (free {free_scale:.3f}); "
               "THE LAW refuses the install")
        law["law_verdict"] = "REFUSED"
        law["refusal_message"] = msg
        say("THE LAW would refuse: " + msg + "  -- bypassed for diagnosis; installing the FREE solve")
        chosen = rungs[0]
        install_rel = ladder[0]
    else:
        law["law_verdict"] = "ADMITTED"
        chosen = min(admitted, key=lambda r: r["check_sse"])
        install_rel = chosen["lambda_relative"]
    law["installed_lambda_relative"] = install_rel
    DIAG["law"] = law
    dump()
    solved = STORE[install_rel].double()
    ridge = {
        "update_scale": chosen["update_scale"],
        "lambda_relative": chosen["lambda_relative"],
        "free_solve_update_scale": free_scale,
        "lambda_selected_on": "internal fit split" if admitted else "DIAGNOSTIC: free solve installed, law bypassed",
        "lambda_law_would_have_picked": admitted[0]["lambda_relative"] if admitted else None,
        "veto_binding": bool(free_scale > 1.0),
        "lambdas_vetoed": int(len(rungs) - len(admitted)),
        "chosen_is_largest_lambda": bool(chosen["lambda_relative"] == ladder[-1]),
        "chosen_is_weakest_admitted": bool(admitted and chosen["lambda_relative"] == admitted[0]["lambda_relative"]),
        "check_sse_monotone_decreasing_in_lambda": bool(monotone),
        "interpretation": "veto_diag scratch run: see law.refusal_message; the veto was bypassed for diagnosis",
        "law_verdict": law["law_verdict"],
        "veto_bypassed_for_diagnosis": True,
        "ladder": rungs,
    }
    return solved, ridge


# --------------------------------------------------------------------------
# 2. refit, wrapped -- scales in, prior anatomy, held-out scores of every candidate
# --------------------------------------------------------------------------
_orig_refit = c14.CoresetFold.refit


def refit_wrapped(self, channels):
    dev = self.device
    fit = self.fit.to(dev)
    held = self.held.to(dev)
    cfg = self.cfg
    n_fit = int(self.fit.sum())
    inputs = {
        "tokens": int(self.tokens), "n_fit": n_fit, "n_held": int(self.held.sum()),
        "n_inner": int(self.inner.sum()), "n_check": int(self.check.sum()),
        "latent_dim": int(self.latent_dim), "n_experts": int(self.n_experts),
        "top_k": int(cfg.get("num_experts_per_token")),
        "routed_scaling_factor": float(cfg.get("routed_scaling_factor", 1.0)),
        "moe_renormalize": bool(cfg.get("moe_renormalize", True)),
        "moe_intermediate_size": int(self.intermediate),
        "latent_in_dtype": str(self.latent.dtype),
        "latent_in_rms_all": rms(self.latent), "latent_in_rms_fit": rms(self.latent[fit]),
        "latent_in_rms_held": rms(self.latent[held]),
        "latent_in_token_norm_mean": float(self.latent.double().norm(dim=1).mean()),
        "y_lat_dtype": str(self.y_lat.dtype),
        "y_lat_rms_all": rms(self.y_lat), "y_lat_rms_fit": rms(self.y_lat[fit]), "y_lat_rms_held": rms(self.y_lat[held]),
        "y_lat_token_norm_mean": float(self.y_lat.double().norm(dim=1).mean()),
        "y_lat_fro_norm_held": fnorm(self.y_lat[held]),
        "y_lat_fro_norm_fit": fnorm(self.y_lat[fit]),
    }
    CTX["fit_y_sq"] = inputs["y_lat_fro_norm_fit"] ** 2
    topk_w = self.topk_w
    inputs["router_topk_w_row_sum_mean"] = float(topk_w.double().sum(1).mean())
    inputs["router_topk_w_mean"] = float(topk_w.double().mean())
    inputs["router_topk_w_quantiles"] = quantiles(topk_w)
    mass = self.expert_mass()          # [E] fp64: sum over fit tokens of w_e * 1[e routed]
    counts = self._expert_counts.double()
    live = counts > 0
    inputs["expert_mass_total"] = float(mass.sum())
    inputs["expert_mass_over_nfit_quantiles"] = quantiles(mass / n_fit)
    inputs["expert_count_quantiles"] = quantiles(counts)
    inputs["expert_mean_weight_when_routed_quantiles"] = quantiles((mass[live] / counts[live]))
    inputs["expert_route_fraction_quantiles"] = quantiles(counts / n_fit)
    inputs["experts_live"] = int(live.sum())
    DIAG["inputs"] = inputs
    say(f"inputs: latent_in rms {inputs['latent_in_rms_fit']:.4e} y_lat rms {inputs['y_lat_rms_fit']:.4e} "
        f"topk_w row-sum mean {inputs['router_topk_w_row_sum_mean']:.4f} n_fit {n_fit} n_held {inputs['n_held']}")
    dump()

    solved, bank, ladder = _orig_refit(self, channels)

    # -- prior anatomy ------------------------------------------------------
    experts = bank.channels[:, 0].to(dev)
    K = int(bank.w2.shape[1])
    m_c = (mass[experts] / n_fit).cpu()                     # exactly c14 line 906
    cnt_c = counts[experts].cpu()
    alt_c = (mass[experts] / counts[experts].clamp(min=1)).cpu()  # E[w_e | routed]
    w2 = bank.w2.float().cpu()                              # unscaled parent read-out columns [3584, K]
    prior = (w2.double() * m_c.unsqueeze(0))
    alt_prior = (w2.double() * alt_c.unsqueeze(0))
    free = STORE[c14.LAMBDA_LADDER[0]].double()
    w2_col = w2.double().norm(dim=0)
    free_col = free.norm(dim=0)
    cos_col = (free * w2.double()).sum(0) / (free_col * w2_col).clamp(min=1e-30)
    ratio_col = free_col / w2_col.clamp(min=1e-30)
    prior_block = {
        "K": K,
        "prior_definition": "prior[:, c] = W2_e[:, j] * m_c, m_c = sum_{t in fit} w_e(t) 1[e in topk(t)] / n_fit  (c14 lines 901-908)",
        "m_c_quantiles": quantiles(m_c),
        "channel_token_count_quantiles": quantiles(cnt_c),
        "alt_scale_E_w_given_routed_quantiles": quantiles(alt_c),
        "alt_over_mc_ratio_quantiles": quantiles(alt_c / m_c.clamp(min=1e-30)),
        "W2_bank_fro_norm_unscaled": fnorm(w2),
        "prior_fro_norm": fnorm(prior),
        "alt_prior_fro_norm": fnorm(alt_prior),
        "prior_over_W2_bank_norm_ratio": fnorm(prior) / fnorm(w2),
        "free_fro_norm": fnorm(free),
        "free_over_prior_norm_ratio": fnorm(free) / fnorm(prior),
        "free_over_W2_bank_norm_ratio": fnorm(free) / fnorm(w2),
        "free_over_alt_prior_norm_ratio": fnorm(free) / fnorm(alt_prior),
        "update_scale_vs_prior": float((free - prior).norm() / prior.norm()),
        "update_scale_vs_alt_prior": float((free - alt_prior).norm() / alt_prior.norm()),
        "update_scale_vs_W2_bank_unscaled": float((free - w2.double()).norm() / w2.double().norm()),
        "update_scale_vs_zero_prior": "inf by construction (reference 0)",
        "free_column_cos_with_W2_column_quantiles": quantiles(cos_col),
        "free_column_norm_over_W2_column_norm_quantiles": quantiles(ratio_col),
        "free_column_scale_over_mc_quantiles": quantiles(ratio_col / m_c.clamp(min=1e-30)),
        "free_column_scale_over_alt_quantiles": quantiles(ratio_col / alt_c.clamp(min=1e-30)),
    }
    DIAG["prior"] = prior_block
    say(f"prior anatomy: ||W2 bank|| {prior_block['W2_bank_fro_norm_unscaled']:.4e} ||prior|| {prior_block['prior_fro_norm']:.4e} "
        f"||alt prior|| {prior_block['alt_prior_fro_norm']:.4e} ||free|| {prior_block['free_fro_norm']:.4e}; "
        f"m_c median {prior_block['m_c_quantiles']['median']:.3e}; free/W2 col ratio median "
        f"{prior_block['free_column_norm_over_W2_column_norm_quantiles']['median']:.3e}; col cos median "
        f"{prior_block['free_column_cos_with_W2_column_quantiles']['median']:.3f}")
    dump()

    # -- held-out scores of every candidate -------------------------------
    def score(name, w):
        t0 = time.time()
        with torch.no_grad():
            s = c14.CoresetFold.score(self, bank, w.float())
        s["fro_norm"] = fnorm(w)
        s["seconds"] = round(time.time() - t0, 1)
        say(f"heldout {name}: y_lat relL2 {s['y_lat_heldout']['relL2']:.4f} cos {s['y_lat_heldout']['cos']:.4f} "
            f"routed_write relL2 {s['routed_write_heldout']['relL2']:.4f}")
        return s

    heldout = {}
    for rel in c14.LAMBDA_LADDER:
        heldout[f"ridge_rel_{rel:g}"] = score(f"rel={rel:g}", STORE[rel])
        dump()
    heldout["prior_mc_scaled_W2"] = score("prior (m_c * W2)", prior)
    heldout["alt_prior_Ew_given_routed_W2"] = score("alt prior (E[w|routed] * W2)", alt_prior)
    heldout["W2_bank_unscaled"] = score("W2 bank unscaled", w2)
    heldout["zero_map"] = score("zero map", torch.zeros_like(w2))
    DIAG["heldout"] = heldout
    # what the law's own bookkeeping computed
    DIAG["c14_ridge_record"] = {k: v for k, v in ladder.items() if k != "ladder"}
    dump()
    return solved, bank, ladder


c14.law_abiding_solve_gram = instrumented_solve
c14.CoresetFold.refit = refit_wrapped

if __name__ == "__main__":
    dump()
    try:
        code = kdp.main(sys.argv[1:])
        DIAG["pass_exit"] = code
    except SystemExit as exc:
        DIAG["pass_exit"] = f"SystemExit: {exc}"
        dump()
        raise
    DIAG["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dump()
    say("done")
