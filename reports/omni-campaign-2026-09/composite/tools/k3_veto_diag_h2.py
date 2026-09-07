#!/usr/bin/env python3
"""k3_veto_diag_h2.py -- H2 on the real K3 layer 1: is the C14 fit TARGET on the wrong scale?

Hypothesis H2 (the 2026-09-07 veto diagnosis): the folded-pair ``latent_in`` and the
teacher ``y_lat`` are on incompatible scales (missing ``routed_expert_norm``, missing
shared-expert contribution, missing AttnRes residual), so the fit target is wrong
even if THE LAW were fine.

What this tool measures, on the real layer 1, from the embedding through layer 0
(dense, recomputed and sha-checked against the sealed L00 receipt) into layer 1,
with the SAME lane functions ``k3_dense_pass`` v1.1.1 used:

1. the teacher routed output FOUR ways, with RMS / cosine / relL2 between them:
     y_lat            PRE-norm fp32 routed accumulator      (what the run used as target)
     routed_write     up_proj(routed_expert_norm(y_lat))    (the 7168-wide write, no shared)
     moe_out          routed_write + shared experts         (the MoE block's write)
     prefix_out       carried + moe_out                     (post-AttnRes: the next layer's input)
   plus latent_in (the fit design), and the scale-invariance of the downstream
   norm: relL2(routed_write(alpha * y_lat), routed_write(y_lat)) for several alpha.

2. the fit at K = 17,408 (stage-1 coreset exactly as the run selected it) solved
   with a ZERO prior over the whole lambda ladder -- the unconstrained ridge --
   and every solution scored on the HELD rows against every target form above,
   with separate FP32-fit and runtime-dtype map scores.  Both reconstruct the
   downstream writes in the parent dtype; only the runtime-dtype map also casts
   W2' as the installed dense path does.  The law's own
   prior-anchored ladder is solved from the same Cholesky factors, so the run's
   ``free 81.642`` is reproduced and the nine update_scale values the refusal
   never wrote are recorded.  The prior itself (m_c * W2) is scored too.

Nothing here installs anything.  Big artefacts go to ``--scratch`` (a directory this
tool creates); the receipt goes to ``--receipt-dir``.  The existing dense/L00, the
snapshots and dense/L01 are never written.

Usage (server, lab venv, detached):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 $PY k3_veto_diag_h2.py \\
      --src-dir /mnt/lexar/models/moonshotai/Kimi-K3 \\
      --input-ids $OC/composite/corpus/k3_input_ids.safetensors \\
      --rung 17408 --device cuda:0 --experts-device cpu \\
      --scratch /mnt/lexar/omni-campaign/k3/veto_diag_h2 \\
      --receipt-dir $OC/composite/receipts/k3_veto_diag \\
      --l00-receipt $OC/composite/receipts/k3_dense_pass/k3_dense_pass_receipt.partial.json
"""
from __future__ import annotations

import argparse
import gc
import json
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

import k3_dense_pass as kdp  # noqa: E402  (brings the lane onto sys.path, imported by path, never edited)

c14 = kdp.c14
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
    situ,
    write_receipt,
    write_scores,
)
from k3_layer import StreamedLayer  # noqa: E402

TOOL_NAME = "k3_veto_diag_h2"
TOOL_VERSION = "0.2.0"
RECEIPT_NAME = "k3_veto_diag_h2_receipt.json"
ALPHAS = (0.01, 0.1, 0.5, 2.0, 10.0, 81.642)


# --------------------------------------------------------------------------
def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    xf = x.float()
    return {"rows": int(xf.shape[0]), "dim": int(xf.shape[1]),
            "rms": float(xf.pow(2).mean().sqrt()),
            "token_norm_mean": float(xf.norm(dim=-1).mean()),
            "token_norm_p05": float(torch.quantile(xf.norm(dim=-1), 0.05)),
            "token_norm_p95": float(torch.quantile(xf.norm(dim=-1), 0.95)),
            "frobenius": float(xf.norm())}


def compare(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """relL2 / cos / cos_p05 of a against b (k3_common.write_scores) + the norm ratio ||a||/||b||."""
    out = write_scores(a, b)
    out["norm_ratio"] = float(a.float().norm() / max(float(b.float().norm()), 1e-30))
    return out


class HeldTargets:
    """The teacher's held-row tensors in every target form, plus what a child needs to write its own."""

    def __init__(self, weights, cfg: dict, held_dev: torch.Tensor, carried: torch.Tensor, ffn_in: torch.Tensor,
                 y_lat: torch.Tensor, routed_write: torch.Tensor, shared: torch.Tensor, chunk: int):
        self.eps = float(cfg["rms_norm_eps"])
        body = weights.body
        self.norm_w = body["block_sparse_moe.routed_expert_norm.weight"]
        self.up_w = body["block_sparse_moe.routed_expert_up_proj.weight"]
        self.chunk = chunk
        self.rows = held_dev
        self.carried = carried[held_dev]
        self.shared = shared[held_dev]
        self.y_lat = y_lat[held_dev]
        self.routed_write = routed_write[held_dev]
        # Match moe_write -> ffn_half: both additions round in the parent dtype.
        # Promoting either sum to fp32 measures a different downstream prefix.
        self.moe_out = self.routed_write + self.shared
        self.prefix_out = self.carried + self.moe_out

    def child_write(self, pred_y: torch.Tensor) -> dict[str, torch.Tensor]:
        """From a child's y_lat prediction (fp32) to routed_write / moe_out / prefix_out on the held rows.

        pred_y is cast to the parent dtype before the norm, as ``moe_write`` does with the
        teacher's fp32 accumulator."""
        rw = torch.empty(pred_y.shape[0], self.up_w.shape[0], dtype=self.up_w.dtype, device=pred_y.device)
        for start in range(0, pred_y.shape[0], self.chunk):
            stop = min(start + self.chunk, pred_y.shape[0])
            y = pred_y[start:stop].to(self.up_w.dtype)
            rw[start:stop] = F.linear(rms_norm(y, self.norm_w, self.eps), self.up_w)
        moe = rw + self.shared
        return {"routed_write": rw, "moe_out": moe, "prefix_out": self.carried + moe}

    def score_child(self, pred_y: torch.Tensor) -> dict[str, Any]:
        got = self.child_write(pred_y)
        return {"y_lat_heldout": compare(pred_y, self.y_lat),
                "routed_write_heldout": compare(got["routed_write"], self.routed_write),
                "moe_out_heldout": compare(got["moe_out"], self.moe_out),
                "prefix_out_heldout": compare(got["prefix_out"], self.prefix_out)}


def predict_y(bank: c14.ExpertBank, fold: c14.CoresetFold, w2p: torch.Tensor, rows: torch.Tensor,
              runtime_dtype: bool = False, token_chunk: int | None = None) -> torch.Tensor:
    """FP32 fit prediction, or the installed map's activation-dtype W2' projection."""
    preds = []
    w = w2p.to(fold.device)
    chunk = token_chunk if runtime_dtype and token_chunk is not None else fold.chunk
    for start in range(0, rows.numel(), chunk):
        piece = rows[start:start + chunk]
        psi = bank.psi(fold.latent[piece], fold.beta, fold.linear_beta)
        if runtime_dtype:
            preds.append(F.linear(psi, w.to(psi.dtype)).float())
        else:
            preds.append(F.linear(psi.float(), w.float()))
        del psi
    return torch.cat(preds)


def score_maps(bank, fold, w2p: torch.Tensor, rows: torch.Tensor, targets: HeldTargets) -> dict[str, Any]:
    """Keep fit diagnostics separate from runtime-cast map scores, without two live predictions."""
    out = {}
    for key, runtime in (("heldout", False), ("heldout_runtime_dtype", True)):
        pred = predict_y(bank, fold, w2p, rows, runtime_dtype=runtime, token_chunk=targets.chunk)
        out[key] = {**targets.score_child(pred),
                    "prediction_mode": "runtime_dtype_map" if runtime else "fp32_fit_map",
                    "map_dtype": str(fold.latent.dtype if runtime else torch.float32).replace("torch.", ""),
                    "projection_chunk": targets.chunk if runtime else fold.chunk,
                    "write_dtype": str(targets.up_w.dtype).replace("torch.", "")}
        del pred
    return out


def moe_write_split(weights, x: torch.Tensor, y_lat: torch.Tensor, cfg: dict, chunk: int
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """`k3_dense_pass.moe_write` with the two terms returned separately (same arithmetic)."""
    eps = float(cfg["rms_norm_eps"])
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    body = weights.body
    routed_all = torch.empty_like(x)
    shared_all = torch.empty_like(x)
    for start in range(0, x.shape[0], chunk):
        stop = min(start + chunk, x.shape[0])
        flat = x[start:stop]
        y = rms_norm(y_lat[start:stop].to(flat.dtype), body["block_sparse_moe.routed_expert_norm.weight"], eps)
        routed_all[start:stop] = F.linear(y, body["block_sparse_moe.routed_expert_up_proj.weight"])
        shared_all[start:stop] = F.linear(
            situ(F.linear(flat, body["block_sparse_moe.shared_experts.gate_proj.weight"]),
                 F.linear(flat, body["block_sparse_moe.shared_experts.up_proj.weight"]), beta, linear_beta),
            body["block_sparse_moe.shared_experts.down_proj.weight"])
    return routed_all, shared_all


def routed_write_of(weights, y_lat: torch.Tensor, cfg: dict, chunk: int, dtype: torch.dtype) -> torch.Tensor:
    eps = float(cfg["rms_norm_eps"])
    body = weights.body
    out = torch.empty(y_lat.shape[0], body["block_sparse_moe.routed_expert_up_proj.weight"].shape[0],
                      dtype=dtype, device=y_lat.device)
    for start in range(0, y_lat.shape[0], chunk):
        stop = min(start + chunk, y_lat.shape[0])
        y = rms_norm(y_lat[start:stop].to(dtype), body["block_sparse_moe.routed_expert_norm.weight"], eps)
        out[start:stop] = F.linear(y, body["block_sparse_moe.routed_expert_up_proj.weight"])
    return out


# --------------------------------------------------------------------------
def solve_from_factor(factor: torch.Tensor, cross: torch.Tensor, lam: float, prior: torch.Tensor | None
                      ) -> torch.Tensor:
    """`c14.prior_anchored_gram`'s solve step from an existing Cholesky factor of (G + lam I).

    rhs = B + lam * prior^T (prior None = zero prior, rhs = B); returns W [out, K] fp64."""
    rhs = cross if prior is None else cross + lam * prior.T.to(cross.dtype)
    return torch.cholesky_solve(rhs, factor).T.contiguous()


def run(args) -> dict[str, Any]:
    torch.manual_seed(0)
    started_wall = time.time()
    device = canonical_device(args.device)
    os.makedirs(args.scratch, exist_ok=True)
    os.makedirs(args.receipt_dir, exist_ok=True)
    receipt_path = os.path.join(args.receipt_dir, RECEIPT_NAME)

    # -- the run's own objects, constructed the way k3_dense_pass.main does ----------
    policy = kdp.resolve_share_policy(args.src_dir, no_share_read=None, allow_share_read=False,
                                      share_rate_receipt=kdp.DEFAULT_SHARE_RATE_RECEIPT)
    index = K3Index(args.src_dir)
    corpus = kdp.Corpus(args.input_ids, [])
    stager = kdp.ShardStager(args.src_dir, None, enabled=False, window=3, prefetch=2,
                             delete_after_consume=False, run_id=uuid.uuid4().hex[:12])
    pass_args = kdp.build_parser().parse_args([
        "--src-dir", args.src_dir, "--input-ids", args.input_ids, "--layers", "0-1",
        "--rung", str(args.rung), "--k-pool-factor", "1", "--device", args.device,
        "--experts-device", args.experts_device, "--resid-device", "cpu", "--kda-backend", args.kda_backend,
        "--out", args.scratch, "--receipt-dir", args.receipt_dir, "--holdout-blocks", str(args.holdout_blocks),
        "--chunk", str(args.chunk), "--token-chunk", str(args.token_chunk), "--row-chunk", "1",
        "--max-gram-gb", str(args.max_gram_gb), "--max-device-gb", "-1"])
    pass_args.model_dir = args.src_dir
    pass_args.no_share_read = policy["no_share_read"]
    if not os.path.isdir(pass_args.chimera_src or ""):
        pass_args.chimera_src = None
    run_obj = kdp.DualStreamRun(pass_args, index, corpus, stager, policy)
    cfg = run_obj.cfg
    token_chunk = int(args.token_chunk)

    receipt: dict[str, Any] = {
        "schema": "omni-campaign.k3_veto_diag_h2.v2",
        "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "dense_pass_tool_sha256": sha256_file(kdp.__file__), "dense_pass_tool_version": kdp.TOOL_VERSION,
        "lane_dir": kdp.LANE_DIR, "lane_sha256": kdp.lane_file_digests(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arguments": dict(vars(args)),
        "hypothesis": "H2: the fit target (teacher y_lat) and design (folded latent_in) are on incompatible "
                      "scales -- missing routed_expert_norm / shared experts / AttnRes residual -- so the "
                      "target is wrong even if THE LAW were fine",
        "model": run_obj._model_identity(),
        "corpus": corpus.record(),
        "device": str(device), "devices": device_inventory(), "torch": torch.__version__,
        "card_note": "the 2026-09-07 run used GPU6 (Quadro RTX 8000, sm75); this diagnostic ran on whatever "
                     "CUDA_VISIBLE_DEVICES names (see devices) -- a different card can change bf16 kernel "
                     "rounding, so the L00 sha check may read False on another card without meaning anything "
                     "for the measurements",
        "python": sys.version.split()[0],
        "scoring_contract": {
            "heldout": "FP32 fit map: activation features and W2' projected in fp32; not installed-map parity",
            "heldout_runtime_dtype": "W2' cast to the activation dtype before projection with token_chunk, "
                                     "as dense_y_lat does",
            "downstream": "both maps cast y_lat before routed norm/up-projection, then add routed + shared "
                          "and carried + moe_out in the parent dtype, in ffn_half order; only metrics use fp32",
            "historical_caveat": "H2 v0.1.0 reconstructed held-row MoE/prefix sums in fp32; its downstream "
                                 "numbers are not runtime-dtype receipts and are not rewritten by this tool",
        },
        "layers": {},
    }
    held = corpus.held
    fit = corpus.fit
    held_rows = torch.nonzero(held).flatten()
    fit_rows = torch.nonzero(fit).flatten()
    held_dev = held_rows.to(device)
    fit_dev = fit_rows.to(device)

    # -- the sealed L00 record to check against ---------------------------------------
    l00_expected = None
    if args.l00_receipt and os.path.exists(args.l00_receipt):
        with open(args.l00_receipt, "r", encoding="utf-8") as handle:
            prior_receipt = json.load(handle)
        l00 = prior_receipt.get("layers", {}).get("0", {})
        l00_expected = {"prefix_out_sha256": l00.get("prefix_out_sha256"),
                        "refusal": prior_receipt.get("refusal"), "path": args.l00_receipt,
                        "receipt_sha256": sha256_file(args.l00_receipt)}
    receipt["reference_run"] = l00_expected

    # -- layer 0: dense, from the embedding (what the run did; the folded pair IS the teacher) --
    pair = run_obj.initial_pair()
    receipt["embedding_prefix_sha256"] = sha256_tensor(pair.prefix)
    t0 = time.time()
    weights0, raw0, timing0 = kdp.read_layer(index, 0, stager, device, run_obj.experts_device, run_obj.compute_dtype,
                                             progress_tag="L00")
    runner0 = StreamedLayer(weights0, cfg, kda_backend=args.kda_backend)
    with torch.no_grad():
        carried0, ffn_in0, blocks_after0 = kdp.attention_half(runner0, pair, corpus, device, row_chunk=1,
                                                              token_chunk=token_chunk)
        prefix0, _ = kdp.ffn_half(runner0, carried0, ffn_in0, cfg, device=device, token_chunk=token_chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    pair = kdp.Pair(prefix0.to("cpu"), blocks_after0)
    del carried0, ffn_in0, prefix0, runner0, weights0, raw0
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    l0_sha = sha256_tensor(pair.prefix)
    l0_entry = {"layer": 0, "moe": False, "read": timing0, "seconds": round(time.time() - t0, 1),
                "prefix_out_sha256": l0_sha, "block_residual_blocks": int(pair.blocks.shape[1]),
                "matches_sealed_L00_receipt": None}
    if l00_expected and l00_expected.get("prefix_out_sha256"):
        want = l00_expected["prefix_out_sha256"].get("teacher")
        l0_entry["matches_sealed_L00_receipt"] = bool(want == l0_sha)
        l0_entry["sealed_teacher_prefix_sha256"] = want
    receipt["layers"]["0"] = l0_entry
    log(f"L00 recomputed in {l0_entry['seconds']}s; prefix sha {l0_sha[:12]}... "
        f"matches sealed receipt: {l0_entry['matches_sealed_L00_receipt']}")
    write_receipt(receipt_path, receipt)

    # -- layer 1: the teacher, every target form -------------------------------------------
    t1 = time.time()
    weights, raw_body, timing1 = kdp.read_layer(index, 1, stager, device, run_obj.experts_device,
                                                run_obj.compute_dtype, progress_tag="L01")
    runner = StreamedLayer(weights, cfg, kda_backend=args.kda_backend)
    entry: dict[str, Any] = {"layer": 1, "moe": bool(weights.is_moe), "read": timing1,
                             "routed_scaling_factor": float(cfg.get("routed_scaling_factor", 1.0)),
                             "moe_renormalize": bool(cfg.get("moe_renormalize", True)),
                             "num_experts": int(cfg["num_experts"]),
                             "num_experts_per_token": int(cfg["num_experts_per_token"]),
                             "rms_norm_eps": float(cfg["rms_norm_eps"])}
    with torch.no_grad():
        carried, ffn_in, blocks_after = kdp.attention_half(runner, pair, corpus, device, row_chunk=1,
                                                           token_chunk=token_chunk)
        latent = kdp._chunked_linear(ffn_in, weights.body["block_sparse_moe.routed_expert_down_proj.weight"],
                                     token_chunk)
        compute_start = time.time()
        y_lat, taps, dequant_seconds = kdp.routed_y_lat(weights, ffn_in, latent, cfg, device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        entry["teacher_routed_seconds"] = round(time.time() - compute_start, 1)
        entry["dequant_seconds"] = round(dequant_seconds, 1)
        entry["experts_touched"] = int(taps["experts_touched"])
        routed_write, shared = moe_write_split(weights, ffn_in, y_lat, cfg, token_chunk)
        moe_out = routed_write + shared
        prefix_out = carried + moe_out
    entry["prefix_out_sha256_teacher"] = sha256_tensor(prefix_out)
    entry["finite"] = {"latent_in": int((~torch.isfinite(latent)).sum()), "y_lat": int((~torch.isfinite(y_lat)).sum()),
                       "prefix_out": int((~torch.isfinite(prefix_out)).sum())}

    # 1. scales and relations of every form, on held and fit rows
    forms = {"latent_in": latent, "y_lat": y_lat, "routed_write": routed_write, "shared": shared,
             "moe_out": moe_out, "carried": carried, "prefix_out": prefix_out}
    scales: dict[str, Any] = {}
    relations: dict[str, Any] = {}
    with torch.no_grad():
        for split_name, rows in (("held", held_dev), ("fit", fit_dev)):
            scales[split_name] = {}
            for name, t in forms.items():  # one fp32 temporary at a time (fit rows x 7168 fp32 = 2 GB)
                scales[split_name][name] = tensor_stats(t[rows])
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            if split_name == "fit":
                continue  # the relations are a held-row measurement; five fit-row temporaries do not fit a 24 GB card
            rw, sh, mo, ca, po = (routed_write[rows], shared[rows], moe_out[rows],
                                  carried[rows], prefix_out[rows])
            zero_rw = routed_write_of(weights, torch.zeros_like(y_lat[rows]), cfg, token_chunk, ffn_in.dtype)
            zero_moe = zero_rw + sh
            relations[split_name] = {
                "routed_write_vs_moe_out": compare(rw, mo),
                "shared_vs_moe_out": compare(sh, mo),
                "shared_vs_routed_write": compare(sh, rw),
                "moe_out_vs_prefix_out": compare(mo, po),
                "carried_vs_prefix_out": compare(ca, po),
                "zero_y_lat_child": {
                    "what": "a child that writes y_lat = 0 (the relL2 = 1.0 baseline of the fit target): "
                            "what the next layer would see",
                    "routed_write": compare(zero_rw, rw),
                    "moe_out": compare(zero_moe, mo),
                    "prefix_out": compare(ca + zero_moe, po),
                },
            }
            del rw, sh, mo, ca, po, zero_rw, zero_moe
        # scale invariance of the downstream norm: the target's GLOBAL scale is not consumed
        inv = {}
        base = routed_write[held_dev].float()
        for alpha in ALPHAS:
            scaled = routed_write_of(weights, y_lat[held_dev] * alpha, cfg, token_chunk, ffn_in.dtype).float()
            inv[str(alpha)] = compare(scaled, base)
            del scaled
        del base
    entry["scales"] = scales
    entry["relations"] = relations
    entry["routed_write_scale_invariance"] = {
        "what": "relL2 of up_proj(routed_expert_norm(alpha * y_lat)) against alpha = 1 on the held rows; "
                "RMSNorm makes the write invariant to a global rescale of y_lat up to eps",
        "by_alpha": inv}
    # router mass facts that size the law's prior
    topk_w = taps["router_topk_w"].float()
    entry["router"] = {"topk_w_sum_per_token_mean": float(topk_w.sum(dim=1).mean()),
                       "topk_w_mean": float(topk_w.mean()), "topk_w_max": float(topk_w.max()),
                       "topk_w_min": float(topk_w.min())}
    log(f"L01 teacher: y_lat rms {scales['held']['y_lat']['rms']:.4g} latent_in rms "
        f"{scales['held']['latent_in']['rms']:.4g} routed_write rms {scales['held']['routed_write']['rms']:.4g} "
        f"shared rms {scales['held']['shared']['rms']:.4g} prefix_out rms {scales['held']['prefix_out']['rms']:.4g}")
    log(f"L01 held: routed_write/moe_out norm ratio {relations['held']['routed_write_vs_moe_out']['norm_ratio']:.4f} "
        f"cos {relations['held']['routed_write_vs_moe_out']['cos']:.4f}; shared/moe_out "
        f"{relations['held']['shared_vs_moe_out']['norm_ratio']:.4f}; moe_out/prefix_out "
        f"{relations['held']['moe_out_vs_prefix_out']['norm_ratio']:.4f}")
    log("L01 scale invariance relL2 by alpha: " + ", ".join(f"{k}: {v['relL2']:.3e}" for k, v in inv.items()))
    receipt["layers"]["1"] = entry
    write_receipt(receipt_path, receipt)

    # the held-row targets in every form, then drop the full-corpus 7168-wide tensors
    targets = HeldTargets(weights, cfg, held_dev, carried, ffn_in, y_lat, routed_write, shared, token_chunk)
    del routed_write, shared, moe_out, prefix_out, carried, blocks_after
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -- 2. the fit: stage 1 as the run did, then the ladders with a ZERO prior and the law prior --
    cache = {"latent_in": latent, "y_lat": y_lat, "router_topk_idx": taps["router_topk_idx"],
             "router_topk_w": taps["router_topk_w"]}
    max_device_gb = c14.default_device_budget_gb(device)
    fold = kdp.DualCoresetFold(weights, cache, cfg, device, corpus.fit, corpus.held,
                               check_blocks=int(args.holdout_blocks), chunk=int(args.chunk),
                               max_gram_gb=float(args.max_gram_gb), max_device_gb=max_device_gb)
    fit_entry: dict[str, Any] = {"rung": int(args.rung), "k_pool_factor": 1,
                                 "fit_tokens": int(fold.fit.sum()), "held_tokens": int(fold.held.sum()),
                                 "inner_tokens": int(fold.inner.sum()), "check_tokens": int(fold.check.sum()),
                                 "design": "latent_in of the teacher pair (== the folded pair at layer 1: layer 0 is "
                                           "dense and the sealed L00 receipt shows the folded pair cloned from the "
                                           "teacher with relL2 0)",
                                 "target": "teacher y_lat, PRE-norm fp32 routed accumulator (the run's target)"}
    t_sel = time.time()
    channels, stage1_stats = fold.stage1(int(args.rung))
    fit_entry["stage1"] = stage1_stats
    fit_entry["stage1_seconds"] = round(time.time() - t_sel, 1)
    log(f"stage1 selected {int(channels.shape[0])} channels in {fit_entry['stage1_seconds']}s")

    bank = c14.ExpertBank(fold.w, channels, fold.device, dtype=fold.latent.dtype)
    width = int(bank.w1.shape[0])
    out_dim = int(fold.y_lat.shape[1])
    fit_entry["width"] = width
    budget = c14.gram_memory_estimate(width, out_dim)
    c14.enforce_budget(budget, float(args.max_gram_gb), f"the K={width} Gram refit")
    fit_entry["gram_memory"] = budget
    t_gram = time.time()
    gram_inner = torch.zeros(width, width, dtype=torch.float64)
    gram_check = torch.zeros_like(gram_inner)
    cross_inner = torch.zeros(width, out_dim, dtype=torch.float64)
    cross_check = torch.zeros_like(cross_inner)
    check_y_sq = 0.0
    with torch.no_grad():
        for start in range(0, fold.tokens, fold.chunk):
            stop = min(start + fold.chunk, fold.tokens)
            rows = torch.arange(start, stop)
            if not bool(fold.fit[rows].any()):
                continue
            psi = bank.psi(fold.latent[start:stop], fold.beta, fold.linear_beta).float()
            target = fold.y_lat[start:stop]
            for mask, gram, cross in ((fold.inner[rows], gram_inner, cross_inner),
                                      (fold.check[rows], gram_check, cross_check)):
                if not bool(mask.any()):
                    continue
                sel = psi[mask.to(psi.device)].cpu().double()
                gram += sel.T @ sel
                cross += sel.T @ target[mask.to(psi.device)].cpu().double()
                del sel
            if bool(fold.check[rows].any()):
                check_y_sq += float(target[fold.check[rows].to(psi.device)].cpu().double().pow(2).sum())
            del psi
    fit_entry["gram_seconds"] = round(time.time() - t_gram, 1)
    log(f"Grams accumulated in {fit_entry['gram_seconds']}s (K={width})")

    # the law's prior, exactly as CoresetFold.refit builds it
    experts = bank.channels[:, 0].to(fold.device)
    n_fit = max(int(fold.fit.sum()), 1)
    mass = (fold.expert_mass()[experts] / n_fit).cpu()
    counts = fold._expert_counts[experts].double().cpu()
    prior = (bank.w2.double().cpu() * mass.unsqueeze(0))
    prior_f = prior.float()
    prior_norm = float(prior_f.norm())
    w2_bank_norm = float(bank.w2.double().norm())
    fit_entry["prior"] = {
        "what": "m_c * W2_e[:, j] per kept channel; m_c = E_t[w_e(t) 1[e in topk]] over the fit split",
        "prior_mass_mean": float(mass.mean()), "prior_mass_min": float(mass.min()),
        "prior_mass_max": float(mass.max()), "prior_mass_median": float(mass.median()),
        "channel_token_count_mean": float(counts.mean()),
        "prior_frobenius": prior_norm, "bank_w2_frobenius": w2_bank_norm,
        "prior_over_bank_w2": prior_norm / max(w2_bank_norm, 1e-30),
        "mean_router_weight_when_routed": float((fold.expert_mass()[experts].cpu() / counts.clamp(min=1)).mean()),
    }
    with torch.no_grad():
        prior_scores = score_maps(bank, fold, prior_f, held_dev, targets)
        fit_entry["prior_scored"] = prior_scores["heldout"]
        fit_entry["prior_scored_runtime_dtype"] = prior_scores["heldout_runtime_dtype"]
    log(f"prior: mass mean {fit_entry['prior']['prior_mass_mean']:.4e}, ||prior|| {prior_norm:.4f}, "
        f"||W2 bank|| {w2_bank_norm:.4f}; prior held relL2 {fit_entry['prior_scored']['y_lat_heldout']['relL2']:.4f}")
    write_receipt(receipt_path, {**receipt, "fit": fit_entry})

    # -- pass 1: the check split (inner solve per lambda), both priors from one factor --------
    diag_mean_inner = float(torch.diagonal(gram_inner).mean())
    ladder_zero: list[dict[str, Any]] = []
    ladder_law: list[dict[str, Any]] = []
    t_l = time.time()
    for rel in c14.LAMBDA_LADDER:
        lam = rel * diag_mean_inner
        reg = gram_inner.clone()
        reg.diagonal().add_(lam)
        factor = torch.linalg.cholesky(reg)
        del reg
        w_zero = solve_from_factor(factor, cross_inner, lam, None)
        w_law = solve_from_factor(factor, cross_inner, lam, prior_f)
        del factor
        ladder_zero.append({"lambda_relative": rel, "check_sse": c14._check_sse(w_zero, gram_check, cross_check, check_y_sq)})
        ladder_law.append({"lambda_relative": rel, "check_sse": c14._check_sse(w_law, gram_check, cross_check, check_y_sq)})
        del w_zero, w_law
        log(f"  check split rel={rel:g}: zero-prior sse {ladder_zero[-1]['check_sse']:.6e}, "
            f"law-prior sse {ladder_law[-1]['check_sse']:.6e}")
    fit_entry["check_split_seconds"] = round(time.time() - t_l, 1)
    fit_entry["check_y_sq"] = check_y_sq

    # fold check into inner: now the full-fit system (exactly as law_abiding_solve_gram does)
    gram_inner.add_(gram_check)
    cross_inner.add_(cross_check)
    gram_all, cross_all = gram_inner, cross_inner
    del gram_check, cross_check
    gc.collect()
    diag_mean_all = float(torch.diagonal(gram_all).mean())
    fit_entry["gram_diag_mean"] = {"inner": diag_mean_inner, "full": diag_mean_all}

    # -- pass 2: the full-fit solutions, scored on the held rows -------------------------------
    best: dict[str, Any] | None = None
    best_w: torch.Tensor | None = None
    t_f = time.time()
    for i, rel in enumerate(c14.LAMBDA_LADDER):
        lam = rel * diag_mean_all
        reg = gram_all.clone()
        reg.diagonal().add_(lam)
        factor = torch.linalg.cholesky(reg)
        del reg
        w_zero = solve_from_factor(factor, cross_all, lam, None)
        w_law = solve_from_factor(factor, cross_all, lam, prior_f)
        del factor
        row_zero = ladder_zero[i]
        row_law = ladder_law[i]
        row_zero["solution_frobenius"] = float(w_zero.norm())
        row_zero["norm_over_prior"] = float(w_zero.norm() / max(prior_norm, 1e-30))
        row_zero["norm_over_bank_w2"] = float(w_zero.norm() / max(w2_bank_norm, 1e-30))
        row_zero["distance_to_prior_over_prior"] = float((w_zero.float() - prior_f).norm() / max(prior_norm, 1e-30))
        row_law["update_scale"] = float((w_law.float() - prior_f).norm() / max(prior_norm, 1e-30))
        row_law["solution_frobenius"] = float(w_law.norm())
        row_law["distance_to_zero_prior_solution_rel"] = float((w_law - w_zero).norm() / max(float(w_zero.norm()), 1e-30))
        with torch.no_grad():
            row_zero.update(score_maps(bank, fold, w_zero.float(), held_dev, targets))
            row_law.update(score_maps(bank, fold, w_law.float(), held_dev, targets))
        log(f"  full rel={rel:g}: ZERO prior held y_lat relL2 {row_zero['heldout']['y_lat_heldout']['relL2']:.4f} "
            f"cos {row_zero['heldout']['y_lat_heldout']['cos']:.4f} routed_write relL2 "
            f"{row_zero['heldout']['routed_write_heldout']['relL2']:.4f} prefix_out relL2 "
            f"{row_zero['heldout']['prefix_out_heldout']['relL2']:.3e} ||W||/||prior|| {row_zero['norm_over_prior']:.3f} "
            f"| LAW prior update_scale {row_law['update_scale']:.3f} held y_lat relL2 "
            f"{row_law['heldout']['y_lat_heldout']['relL2']:.4f}")
        if best is None or row_zero["heldout"]["y_lat_heldout"]["relL2"] < best["heldout"]["y_lat_heldout"]["relL2"]:
            best = row_zero
            best_w = w_zero.float().clone()
        del w_zero, w_law
        write_receipt(receipt_path, {**receipt, "fit": {**fit_entry, "ladder_zero_prior": ladder_zero,
                                                        "ladder_law_prior": ladder_law}})
    fit_entry["full_solve_seconds"] = round(time.time() - t_f, 1)
    fit_entry["ladder_zero_prior"] = ladder_zero
    fit_entry["ladder_law_prior"] = ladder_law
    free_scale = ladder_law[0]["update_scale"]
    admitted = [r for r in ladder_law if r["update_scale"] <= 1.0]
    fit_entry["law_reproduction"] = {
        "free_solve_update_scale": free_scale,
        "run_reported_free": 81.642,
        "reproduces_run": bool(abs(free_scale - 81.642) < 0.05),
        "lambdas_admitted": len(admitted),
        "would_refuse": not admitted,
        "min_update_scale_on_ladder": min(r["update_scale"] for r in ladder_law),
    }
    zero_best_check = min(ladder_zero, key=lambda r: r["check_sse"])
    fit_entry["zero_prior_summary"] = {
        "selection_metric": "diagnostic FP32-fit-map held y_lat relL2 (not a runtime acceptance decision)",
        "best_heldout_lambda_relative": best["lambda_relative"],
        "best_heldout": best["heldout"],
        "best_heldout_runtime_dtype": best["heldout_runtime_dtype"],
        "check_split_would_pick_lambda_relative": zero_best_check["lambda_relative"],
        "check_split_pick_heldout": zero_best_check["heldout"],
        "check_split_pick_heldout_runtime_dtype": zero_best_check["heldout_runtime_dtype"],
        "unconstrained_rel1e-6_heldout": ladder_zero[0]["heldout"],
        "unconstrained_rel1e-6_heldout_runtime_dtype": ladder_zero[0]["heldout_runtime_dtype"],
        "unconstrained_norm_over_prior": ladder_zero[0]["norm_over_prior"],
        "unconstrained_norm_over_bank_w2": ladder_zero[0]["norm_over_bank_w2"],
    }
    # keep the best zero-prior W2' in scratch (fp32, nothing installs it)
    best_path = os.path.join(args.scratch, f"L01_K{args.rung}_zero_prior_w2p.safetensors")
    best_sha = save_tensors(best_path, {"w2p": best_w.cpu().contiguous(),
                                        "channels": bank.channels.cpu().to(torch.int32)},
                            metadata={"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "layer": "1",
                                      "K": str(args.rung), "prior": "zero",
                                      "lambda_relative": str(best["lambda_relative"]),
                                      "input_ids_sha256": corpus.sha256, "installed": "no -- diagnostic only"})
    fit_entry["zero_prior_summary"]["best_w2p_file"] = {"path": best_path, "sha256": best_sha}

    receipt["fit"] = fit_entry
    receipt["peak_rss_bytes"] = peak_rss_bytes()
    if torch.cuda.is_available():
        receipt["device_peak_bytes"] = int(torch.cuda.max_memory_allocated())
    receipt["total_seconds"] = round(time.time() - started_wall, 1)

    # -- the numbers that decide H2, stated next to each other ---------------------------------
    h = relations["held"]
    u = ladder_zero[0]["heldout"]
    u_runtime = ladder_zero[0]["heldout_runtime_dtype"]
    receipt["h2_summary"] = {
        "target_used_by_the_run": "y_lat (PRE-norm fp32 routed accumulator); reproduced here with the same functions",
        "y_lat_rms_held": scales["held"]["y_lat"]["rms"],
        "latent_in_rms_held": scales["held"]["latent_in"]["rms"],
        "routed_write_rms_held": scales["held"]["routed_write"]["rms"],
        "shared_rms_held": scales["held"]["shared"]["rms"],
        "prefix_out_rms_held": scales["held"]["prefix_out"]["rms"],
        "routed_write_over_moe_out_norm_ratio_held": h["routed_write_vs_moe_out"]["norm_ratio"],
        "shared_over_moe_out_norm_ratio_held": h["shared_vs_moe_out"]["norm_ratio"],
        "moe_out_over_prefix_out_norm_ratio_held": h["moe_out_vs_prefix_out"]["norm_ratio"],
        "scale_invariance_max_relL2_over_alphas": max(v["relL2"] for v in inv.values()),
        "zero_child_prefix_out_relL2_held": h["zero_y_lat_child"]["prefix_out"]["relL2"],
        "zero_child_routed_write_relL2_held": h["zero_y_lat_child"]["routed_write"]["relL2"],
        "prior_child_y_lat_relL2_held": fit_entry["prior_scored"]["y_lat_heldout"]["relL2"],
        "prior_child_prefix_out_relL2_held": fit_entry["prior_scored"]["prefix_out_heldout"]["relL2"],
        "prior_child_prefix_out_relL2_held_runtime_dtype": fit_entry["prior_scored_runtime_dtype"]["prefix_out_heldout"]["relL2"],
        "unconstrained_child_y_lat_relL2_held": u["y_lat_heldout"]["relL2"],
        "unconstrained_child_y_lat_cos_held": u["y_lat_heldout"]["cos"],
        "unconstrained_child_routed_write_relL2_held": u["routed_write_heldout"]["relL2"],
        "unconstrained_child_moe_out_relL2_held": u["moe_out_heldout"]["relL2"],
        "unconstrained_child_prefix_out_relL2_held": u["prefix_out_heldout"]["relL2"],
        "unconstrained_child_y_lat_relL2_held_runtime_dtype": u_runtime["y_lat_heldout"]["relL2"],
        "unconstrained_child_routed_write_relL2_held_runtime_dtype": u_runtime["routed_write_heldout"]["relL2"],
        "unconstrained_child_moe_out_relL2_held_runtime_dtype": u_runtime["moe_out_heldout"]["relL2"],
        "unconstrained_child_prefix_out_relL2_held_runtime_dtype": u_runtime["prefix_out_heldout"]["relL2"],
        "best_zero_prior_lambda_relative": best["lambda_relative"],
        "best_zero_prior_y_lat_relL2_held": best["heldout"]["y_lat_heldout"]["relL2"],
        "best_zero_prior_prefix_out_relL2_held": best["heldout"]["prefix_out_heldout"]["relL2"],
        "best_zero_prior_prefix_out_relL2_held_runtime_dtype": best["heldout_runtime_dtype"]["prefix_out_heldout"]["relL2"],
        "law_free_update_scale": free_scale,
        "law_would_refuse": not admitted,
    }
    write_receipt(receipt_path, receipt)
    log(f"receipt -> {receipt_path}")
    return receipt


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=kdp.DEFAULT_SRC_DIR)
    ap.add_argument("--input-ids", required=True)
    ap.add_argument("--rung", type=int, default=17408)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--experts-device", default="cpu")
    ap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    ap.add_argument("--scratch", required=True, help="a directory this tool creates for its big artefacts")
    ap.add_argument("--receipt-dir", required=True)
    ap.add_argument("--l00-receipt", default=os.path.join(kdp.DEFAULT_RECEIPT_DIR, kdp.PARTIAL_RECEIPT_NAME))
    ap.add_argument("--holdout-blocks", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--token-chunk", type=int, default=1024)
    ap.add_argument("--max-gram-gb", type=float, default=160.0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
