#!/usr/bin/env python3
"""k3_b2_geometry_diag.py -- lane gpu6, plan Step 2a (STATE_2026-10-14_CODEX_HANDOFF.md section 5): WHY does
k3_teacher_cache's teacher stream leave the fold engine's at L2?

Evidence going in:
  * B2 (k3_teacher_cache 1.2.1, sha 7899023c..., review files at T = 512, 2 row groups) failed only the p99 gate.
  * The bisect (same frozen tool, the fold's own three files, --no-trim, --group-tokens 32768 = 5 row groups)
    matched the fold receipt's teacher prefix sha at L00 (b0a9b92d...) and L01 (48afb2e4...) and left it at L02
    (fold 99eb1700..., bisect 6db46e44...).
  * Codex's probe (k3_whole_corpus_probe.py, db441b84...) made ONE attention_half + ffn_half call over the whole
    143,360-position corpus and matched all three.

This tool runs L0-L2 over that corpus, ON ONE CARD IN ONE PROCESS, five ways, each arm carrying its own state from
the same embedding:

  group         (i)   the frozen tool's own CachePass.layer, unmodified: 5 row groups (the B2 / bisect geometry)
  group_nopad   (ii)  the same 5 row groups, the FFN half run over each group's REAL positions only
  single        (iii) one call over the whole corpus (the fold engine's geometry)
  single_nopad  (ii') one call, the FFN half over the corpus' REAL positions only (padding out of the batch,
                      nothing else changed)
  fp32          the frozen tool's CachePass.layer at --compute-dtype fp32, group geometry (precision reference)

plus, at L2, a MECHANISM test on the single arm's FFN input with the layer's own weights: the router's fp32 GEMM
and the top experts' bf16 GEMMs evaluated at the full-call row count M and at each row group's M, compared bit for
bit on the shared rows, with a repeated full-M call as the determinism control.

Padded positions are never read by a real position (attention is causal per row, AttnRes and every FFN op are
per token), so in the *_nopad arms they keep the attention output; only REAL positions are compared there.

Verdict (diagnostic; nothing is written outside --out):
  padding_or_state  an arm without padding restores the reference bits on real positions, or removing padding
                    shrinks the group-vs-single error by >= 10x
  bf16_batch_numerics  no such arm restores it; the same row's router / expert GEMM bits change with M while a
                    repeated same-M call is bit-identical; and the group and single arms sit at a comparable
                    distance (ratio within [1/1.5, 1.5]) from the fp32 arm
  undetermined      anything else
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Any

import torch

TOOL_NAME = "k3_b2_geometry_diag"
# 1.0.1: --mode mechanism -- the closed-loop test.  1.0.0's mechanism sampled only the 4 most-loaded experts, whose
#        M is large in both geometries (0 rows changed: the wrong population).  1.0.1 rebuilds the fold-identical
#        single-call state through L1, then at L2 runs ffn_half on each row group's slice against the full call on
#        IDENTICAL inputs (the only difference is the call's batch) and counts the positions that change -- to be
#        compared with 1.0.0's group-vs-single count at L2 -- and tests EVERY touched expert's GEMM chain at full M
#        vs each group's M.
# 1.0.2: the verdict rule.  1.0.0 counted single_nopad as a possible "restoring" arm; it is a control (it never
#        differed), and on the measured L2 numbers the rule printed padding_or_state.  verdict() now takes the closed
#        loop of the mechanism run; --mode verdict re-judges the two sealed receipts on CPU (they are not rewritten).
# 1.1.0: --mode identity -- the fold-identity GATE the B2 re-baseline needs (review 2026-10-14: the diagnosis covered
#        L00-L02, all KDA; nothing that gated compared L03-L92 with the fold).  Every layer 0..92: the single-call
#        (fold geometry) step's prefix sha must equal the fold receipt's teacher sha -- bit-exact identity of all the
#        tool shares with the fold (embedding, reads, dequant, KDA, MLA, AttnRes block pushes, MoE, the L61+
#        continuation).  At the closed-loop layers (every MLA layer, every block push, the first MoE layer, L2, L61,
#        the last) the frozen tool's OWN CachePass.layer -- the production row-group path, unmodified -- runs on
#        IDENTICAL inputs (the fold-exact state) and must stay within bf16 batch numerics of the single call
#        (IDENTITY_RULE), with the pushed block slot bit-equal and no earlier slot touched.  Fail-fast; checkpointed
#        (--resume); host state pre-touched at start so MemAvailable shows the whole footprint.
TOOL_VERSION = "1.1.0"
FROZEN_TOOL_SHA256 = "7899023cb8eb411c4b0fef005bb135970aaef65bb42d4653c688800bf7e88826"   # k3_teacher_cache 1.2.1
OC = "/home/cereal/omni-campaign"
DEFAULT_FROZEN_TOOL = f"{OC}/distill/queues/frozen/k3_teacher_cache-gpu6.py"
DEFAULT_FOLD_RECEIPT = f"{OC}/k3-run-oct09-claude/receipts/k3_dense_pass_receipt.json"
DEFAULT_BISECT = f"{OC}/distill/parity/k3cache-2026-10-13-bisect-gpu6/bisect_sha_compare.json"
CORPUS_FILES = ["k3_input_ids", "k3_review_input_ids", "k3_review_native_input_ids"]
EXPECTED_CORPUS = {"tokens": 143_360, "real": 98_797, "groups": 5,
                   "sha256": "9ff4dbf5afca5fba3227522516ee27a6e4c81f1c49927310e517309a8c9c9273"}
LAYERS = [0, 1, 2]
ARMS = ["single", "single_nopad", "group_nopad", "group", "fp32"]
BF16_ARMS = ["single", "single_nopad", "group_nopad", "group"]
MECH_TOP_EXPERTS = 4
FP32_RATIO_BAND = 1.5
PAD_SHRINK_FACTOR = 10.0
# 1.1.0 identity gate, pre-registered 2026-10-14 before any identity result (ruling
# k3-b2-rebaseline-fp32-production-geometry-2026-10-14-r2).  Closed loop = the production CachePass.layer vs the
# single call on identical inputs.  Measured bf16 batch numerics at L2 (receipts 82532ff0 / c4c3ed81): 3 of 98,797
# real positions, relL2 2.4e-6, max |d| 4.9e-4 (one ulp).  A structural defect (rope / position offset, a wrong block
# slot, a dropped row) moves whole rows: >= 6% of real positions for the T = 512 review rows alone.
IDENTITY_RULE = {"real_unequal_fraction_max": 0.01, "real_relL2_max": 1e-3, "real_cos_min_min": 0.9999,
                 "blocks_after_equal": True, "pushed_slot_bit_equal": True, "earlier_slots_untouched": True,
                 "single_call_prefix_sha_equals_fold_teacher_sha": "every layer"}
IDENTITY_EXTRA_LAYERS = (2, 61)   # L2: the measured split; L61: the first layer of the fold's continuation run


def log(msg: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    body = json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(body)
    os.replace(tmp, path)
    return hashlib.sha256(body).hexdigest()


def load_frozen(path: str, want_sha: str):
    """The frozen k3_teacher_cache, by path, only at the sha the B2 run used."""
    got = sha256_file(path)
    if got != want_sha:
        raise SystemExit(f"{path} is sha {got[:16]}, not the frozen tool {want_sha[:16]} the B2 run used; refusing")
    spec = importlib.util.spec_from_file_location("k3_teacher_cache_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fold_reference(path: str, layers: list[int]) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as handle:
        rec = json.load(handle)
    out = {}
    for layer in layers:
        got = ((rec["layers"].get(str(layer)) or {}).get("prefix_out_sha256") or {}).get("teacher")
        if not got:
            raise SystemExit(f"{path}: no teacher prefix_out_sha256 for L{layer}")
        out[layer] = got
    return out


# --------------------------------------------------------------------------
# masks and comparisons (pure; unit-tested on CPU)
# --------------------------------------------------------------------------
def real_mask(segments) -> torch.Tensor:
    """[sum rows*T] bool, token-major in segment order: position < the row's real length."""
    parts = []
    for seg in segments:
        lens = seg.real_lengths.long().flatten()
        parts.append((torch.arange(int(seg.T))[None, :] < lens[:, None]).flatten())
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.bool)


def group_spans(groups) -> list[tuple[int, int]]:
    return [(int(g0), int(g1)) for _si, _r0, _r1, g0, g1 in groups]


def compare(mine: torch.Tensor, ref: torch.Tensor, mask: torch.Tensor | None = None, *,
            spans: list[tuple[int, int]] | None = None, chunk: int = 8192) -> dict[str, Any]:
    """Position-wise comparison of two [N, H] states over ``mask`` (all positions if None), in row chunks so a
    143k x 7168 pair never needs an fp32 copy.  ``bit_equal_positions`` = positions whose every element is
    bit-identical (meaningful only for two tensors of one dtype)."""
    if mine.shape != ref.shape:
        raise ValueError(f"shape {list(mine.shape)} vs {list(ref.shape)}")
    n = int(mine.shape[0])
    mask = torch.ones(n, dtype=torch.bool) if mask is None else mask.bool()
    same_dtype = mine.dtype == ref.dtype
    sq_d = sq_r = 0.0
    max_abs, cos_min = 0.0, 2.0
    cos_sum = 0.0
    compared = equal = 0
    unequal_at = torch.zeros(n, dtype=torch.bool)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        m = mask[a:b]
        if not bool(m.any()):
            continue
        x, r = mine[a:b][m], ref[a:b][m]
        if same_dtype:
            eq = (x == r).all(dim=1) | (torch.isnan(x) & torch.isnan(r)).all(dim=1)
        else:
            eq = torch.zeros(x.shape[0], dtype=torch.bool)
        sub = torch.zeros(b - a, dtype=torch.bool)
        sub[m] = ~eq
        unequal_at[a:b] = sub
        xf, rf = x.float(), r.float()
        d = xf - rf
        sq_d += float((d.double() ** 2).sum())
        sq_r += float((rf.double() ** 2).sum())
        if d.numel():
            max_abs = max(max_abs, float(d.abs().max()))
        cos = torch.nn.functional.cosine_similarity(xf, rf, dim=1)
        cos_min = min(cos_min, float(cos.min()))
        cos_sum += float(cos.double().sum())
        compared += int(m.sum())
        equal += int(eq.sum())
    out = {"positions": compared, "bit_equal_positions": equal if same_dtype else None,
           "bit_equal_fraction": (equal / max(compared, 1)) if same_dtype else None,
           "relL2": (sq_d ** 0.5) / max(sq_r ** 0.5, 1e-30), "max_abs_diff": max_abs,
           "cos_mean": cos_sum / max(compared, 1), "cos_min": cos_min if compared else None}
    if spans:
        out["unequal_positions_per_group"] = [int(unequal_at[g0:g1].sum()) for g0, g1 in spans]
    return out


def verdict(result: dict[str, Any], closed: dict[str, Any] | None = None) -> dict[str, Any]:
    """The cause, from the L2 comparisons of the arms run and (1.0.2) the closed loop of the mechanism run.

    1.0.0 counted single_nopad as an arm that could "restore" the reference.  It cannot: single_nopad never differed
    from the reference, so its equality is a CONTROL (padding in a batch does not reach real positions), and on the
    measured L2 numbers that rule printed padding_or_state.  1.0.2 judges:
      padding_or_state     removing padding INSIDE the group geometry restores the reference bits on real positions
                           (or shrinks the group-vs-single error >= 10x), or padding in the single call changes real
                           positions (single_nopad != single)
      bf16_batch_numerics  neither; the group-vs-single error is <= 1% of the bf16-vs-fp32 error; both geometries
                           sit at a comparable distance from fp32; and the closed loop shows the positions changed by
                           the FFN call's batch shape ALONE (identical inputs) are exactly the positions the group arm
                           changed (per group)
      undetermined         anything else (a missing closed loop included)"""
    l2 = result["layers"].get("2") or {}
    cmp_ = l2.get("compare") or {}
    why = []
    ref_ok = l2.get("single_matches_fold") is True
    group_off = l2.get("group_matches_fold") is False
    if not ref_ok:
        why.append("the single-call arm did not reproduce the fold's L2 sha: no reference to judge against")
    if not group_off:
        why.append("the group arm did not leave the fold's L2 sha: nothing to explain")
    gs = cmp_.get("group_vs_single_real") or {}
    gn = cmp_.get("group_nopad_vs_single_real") or {}
    sn = cmp_.get("single_nopad_vs_single_real") or {}
    restores = gn.get("bit_equal_fraction") == 1.0
    shrink = (gs["relL2"] / max(gn["relL2"], 1e-30)) if gs.get("relL2") and gn.get("relL2") is not None else None
    leaks = sn.get("bit_equal_fraction") is not None and sn.get("bit_equal_fraction") != 1.0
    gf, sf = cmp_.get("group_vs_fp32_real") or {}, cmp_.get("single_vs_fp32_real") or {}
    ratio = (gf["relL2"] / sf["relL2"]) if gf.get("relL2") and sf.get("relL2") else None
    comparable = ratio is not None and (1.0 / FP32_RATIO_BAND) <= ratio <= FP32_RATIO_BAND
    small = bool(gs.get("relL2") is not None and sf.get("relL2") and gs["relL2"] <= 0.01 * sf["relL2"])
    arm_rows = (gs["positions"] - gs["bit_equal_positions"]) if gs.get("bit_equal_positions") is not None else None
    arm_groups = gs.get("unequal_positions_per_group")
    loop_rows = loop_groups = None
    closes = False
    if closed:
        loop_rows = closed.get("ffn_real_rows_changed")
        loop_groups = [g.get("ffn_real_rows_changed") for g in closed.get("groups", [])]
        closes = arm_rows is not None and loop_rows == arm_rows and (arm_groups is None or loop_groups == arm_groups)
    m_dep = bool(closed and (closed.get("ffn_rows_changed") or 0) > 0)
    if ref_ok and group_off and (restores or (shrink is not None and shrink >= PAD_SHRINK_FACTOR) or leaks):
        cause = "padding_or_state"
    elif ref_ok and group_off and closes and m_dep and comparable and small:
        cause = "bf16_batch_numerics"
    else:
        cause = "undetermined"
        if not closed:
            why.append("no closed-loop mechanism result")
        elif not closes:
            why.append(f"closed loop changed {loop_rows} real rows {loop_groups}, the group arm {arm_rows} {arm_groups}")
        if not comparable:
            why.append(f"group/single distance-to-fp32 ratio {ratio} outside [{1 / FP32_RATIO_BAND:.3f}, {FP32_RATIO_BAND}]")
        if not small:
            why.append("the group-vs-single error is not <= 1% of the bf16-vs-fp32 error")
    return {"cause": cause, "group_nopad_restores_reference": restores, "padding_reaches_real_positions": leaks,
            "group_over_group_nopad_error_ratio": shrink, "group_over_single_distance_to_fp32": ratio,
            "group_vs_single_over_bf16_vs_fp32": (gs["relL2"] / sf["relL2"]) if gs.get("relL2") is not None and sf.get("relL2") else None,
            "group_arm_real_rows_changed": arm_rows, "group_arm_rows_per_group": arm_groups,
            "closed_loop_real_rows_changed": loop_rows, "closed_loop_rows_per_group": loop_groups,
            "closed_loop_matches_group_arm": closes, "notes": why}


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------
def tool_argv(a, dtype: str, pass_dir: str, layers: str = "0-2") -> list[str]:
    argv = ["run", "--probe", "--layers", layers, "--no-head", "--prefix-digests", "--no-trim", "--no-write"]
    for p in a.input_ids:
        argv += ["--input-ids", p]
    argv += ["--pass-dir", pass_dir, "--device", "cuda:0", "--experts-device", "cuda:0", "--kda-backend", "torch",
             "--compute-dtype", dtype, "--row-chunk-kda", "1", "--row-chunk-mla", "1", "--token-chunk", "1024",
             "--group-tokens", str(a.group_tokens), "--no-prefetch-next", "--fold-receipt", a.fold_receipt]
    if a.engine_dir:
        argv += ["--engine-dir", a.engine_dir]
    if a.src_dir:
        argv += ["--src-dir", a.src_dir]
    return argv


def prepare(module, cp) -> Any:
    """What CachePass.run does before its layer loop, minus the warmer: the stat check against the fold receipt
    (refuse on drift), the stager, the runner."""
    kdp = cp.kdp
    before = module.source_stat(cp.index, cp.args.src_dir)
    fold_ref = module.expected_from_fold_receipt(cp.args.fold_receipt)
    if fold_ref is None:
        raise SystemExit(f"no usable fold receipt at {cp.args.fold_receipt}")
    drift = module.compare_stat(before, fold_ref, what="before vs fold receipt")
    if drift:
        raise SystemExit("K3 source drifted from the fold receipt: " + "; ".join(drift[:4]))
    cp.stager = kdp.ShardStager(cp.args.src_dir, None, enabled=False, window=1, prefetch=0,
                                delete_after_consume=False, run_id=cp.run_id)
    cp.warmer = None
    cp.saved_blocks, cp.saved_prefix = [], None
    return cp.runner()


def ffn_over(kdp, runner, cfg, device, carried, ffn_in, token_chunk, index):
    """ffn_half over every position (index None) or only ``index``; positions left out keep ``carried``."""
    if index is None:
        new, taps = kdp.ffn_half(runner, carried, ffn_in, cfg, device=device, token_chunk=token_chunk)
        return new, taps
    part, taps = kdp.ffn_half(runner, carried[index], ffn_in[index], cfg, device=device, token_chunk=token_chunk)
    new = carried.clone()
    new[index] = part
    return new, taps


def single_step(cp, runner, layer_idx, prefix, blocks, nb, snapshot, row_chunk, real_dev, nopad, keep_ffn_in=False):
    kdp, args = cp.kdp, cp.args
    pair = kdp.Pair(prefix, blocks[:, :nb])
    carried, ffn_in, _blocks_after = kdp.attention_half(runner, pair, cp.corpus, cp.device, row_chunk=row_chunk,
                                                        token_chunk=args.token_chunk)
    del _blocks_after
    index = real_dev if nopad else None
    new, taps = ffn_over(kdp, runner, cp.cfg, cp.device, carried, ffn_in, args.token_chunk, index)
    kdp.require_finite("single-call prefix", new, layer_idx)
    if snapshot:
        blocks[:, nb].copy_(prefix)
    prefix.copy_(new.to(prefix.device))
    kept = ffn_in if keep_ffn_in else None
    del carried, new, taps, pair
    if not keep_ffn_in:
        del ffn_in
    return (nb + 1 if snapshot else nb), kept


def group_nopad_step(module, cp, runner, layer_idx, prefix, blocks, nb, snapshot, row_chunk):
    kdp, args = cp.kdp, cp.args
    for si, r0, r1, g0, g1 in cp.corpus.groups:
        seg = cp.corpus.segments[si]
        pair = kdp.Pair(prefix[g0:g1], blocks[g0:g1, :nb])
        view = module.GroupView(seg, r0, r1)
        carried, ffn_in, _ = kdp.attention_half(runner, pair, view, cp.device, row_chunk=row_chunk,
                                                token_chunk=args.token_chunk)
        lens = seg.real_lengths[r0:r1].long()
        index = (torch.arange(seg.T)[None, :] < lens[:, None]).flatten().nonzero().flatten().to(cp.device)
        new, _taps = ffn_over(kdp, runner, cp.cfg, cp.device, carried, ffn_in, args.token_chunk, index)
        kdp.require_finite("group-nopad prefix", new, layer_idx)
        if snapshot:
            blocks[g0:g1, nb].copy_(prefix[g0:g1])
        prefix[g0:g1].copy_(new.to(prefix.device))
        del carried, ffn_in, new, _taps, pair
    return nb + 1 if snapshot else nb


def mechanism(cp, weights, ffn_in, spans, real_cpu) -> dict[str, Any]:
    """Same rows, same weights, different M: the router's fp32 logits GEMM and the top experts' bf16 w1/w3/w2
    chain, full-call M vs each row group's M; a second full-M call is the determinism control."""
    import torch.nn.functional as F

    kdp = cp.kdp
    cfg = cp.cfg
    gate_w = weights.body["block_sparse_moe.gate.weight"].float()
    out: dict[str, Any] = {"M_full": int(ffn_in.shape[0]), "groups": []}
    full = F.linear(ffn_in.float(), gate_w)
    again = F.linear(ffn_in.float(), gate_w)
    det = bool(torch.equal(full, again))
    del again
    idx_full, w_full, _ = kdp.route(ffn_in, weights, cfg)
    idx_rep, w_rep, _ = kdp.route(ffn_in, weights, cfg)
    det = det and bool(torch.equal(idx_full, idx_rep)) and bool(torch.equal(w_full, w_rep))
    del idx_rep, w_rep
    sorted_full = idx_full.sort(dim=1).values
    real_dev = real_cpu.to(ffn_in.device)
    logit_rows_changed = route_rows_changed = route_real_rows_changed = 0
    for g0, g1 in spans:
        sub = F.linear(ffn_in[g0:g1].float(), gate_w)
        changed = (sub != full[g0:g1]).any(dim=1)
        i_g, _w_g, _ = kdp.route(ffn_in[g0:g1], weights, cfg)
        set_changed = (i_g.sort(dim=1).values != sorted_full[g0:g1]).any(dim=1)
        rec = {"span": [g0, g1], "M": g1 - g0, "logit_rows_bit_changed": int(changed.sum()),
               "logit_max_abs_diff": float((sub - full[g0:g1]).abs().max()),
               "expert_set_changed_rows": int(set_changed.sum()),
               "expert_set_changed_real_rows": int((set_changed & real_dev[g0:g1]).sum())}
        out["groups"].append(rec)
        logit_rows_changed += rec["logit_rows_bit_changed"]
        route_rows_changed += rec["expert_set_changed_rows"]
        route_real_rows_changed += rec["expert_set_changed_real_rows"]
        del sub, changed, i_g, _w_g, set_changed
    del full
    latent = kdp._chunked_linear(ffn_in, weights.body["block_sparse_moe.routed_expert_down_proj.weight"],
                                 int(cp.args.token_chunk))
    counts = torch.bincount(idx_full.flatten(), minlength=int(cfg["num_experts"]))
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    experts = []
    expert_rows_changed = 0
    for expert in counts.topk(MECH_TOP_EXPERTS).indices.tolist():
        rows = (idx_full == expert).any(dim=1).nonzero().flatten()
        ex = {k: v.to(ffn_in.device) for k, v in kdp.dequant_expert(weights, expert, dtype=latent.dtype).items()}

        def chain(x):
            act = kdp.situ(F.linear(x, ex["w1"]), F.linear(x, ex["w3"]), beta, linear_beta)
            return F.linear(act, ex["w2"])

        y_full = chain(latent[rows])
        y_again = chain(latent[rows])
        rec = {"expert": int(expert), "M_full": int(rows.numel()),
               "full_M_repeat_bit_identical": bool(torch.equal(y_full, y_again)), "groups": []}
        det = det and rec["full_M_repeat_bit_identical"]
        for g0, g1 in spans:
            sel = (rows >= g0) & (rows < g1)
            if not bool(sel.any()):
                continue
            y_sub = chain(latent[rows[sel]])
            diff_rows = (y_sub != y_full[sel]).any(dim=1)
            rec["groups"].append({"span": [g0, g1], "M": int(sel.sum()), "rows_bit_changed": int(diff_rows.sum()),
                                  "max_abs_diff": float((y_sub.float() - y_full[sel].float()).abs().max())})
            expert_rows_changed += int(diff_rows.sum())
            del y_sub, diff_rows
        experts.append(rec)
        del ex, y_full, y_again
    out.update({"experts": experts, "full_M_repeat_bit_identical": det,
                "router_logit_rows_changed_by_M": logit_rows_changed,
                "router_expert_set_rows_changed_by_M": route_rows_changed,
                "router_expert_set_real_rows_changed_by_M": route_real_rows_changed,
                "expert_rows_changed_by_M": expert_rows_changed,
                "note": "rows are compared with themselves: identical inputs and weights, only the GEMM row count M "
                        "differs (full call vs one row group)"})
    del latent, idx_full, w_full, sorted_full
    return out


def closed_loop(cp, runner, weights, carried, ffn_in, spans, real_cpu) -> dict[str, Any]:
    """1.0.1.  Identical L2 inputs (the single-call attention output); ffn_half once over the full corpus and once
    per row group on that group's slice.  Every difference is the call's batch shape.  Then every touched expert's
    w1/w3 -> SiTU -> w2 chain at full M vs each group's M, rows compared with themselves."""
    import torch.nn.functional as F

    kdp, cfg, tc = cp.kdp, cp.cfg, int(cp.args.token_chunk)
    dev = ffn_in.device
    real_dev = real_cpu.to(dev)
    full, taps = kdp.ffn_half(runner, carried, ffn_in, cfg, device=cp.device, token_chunk=tc)
    out: dict[str, Any] = {"M_full": int(ffn_in.shape[0]), "groups": []}
    idx_full = taps["router_topk_idx"].long()
    y_full = taps["y_lat"]
    latent = taps["latent_in"]
    changed_all = torch.zeros(int(ffn_in.shape[0]), dtype=torch.bool, device=dev)
    for g0, g1 in spans:
        part, tg = kdp.ffn_half(runner, carried[g0:g1], ffn_in[g0:g1], cfg, device=cp.device, token_chunk=tc)
        changed = (part != full[g0:g1]).any(dim=1)
        changed_all[g0:g1] = changed
        y_changed = (tg["y_lat"] != y_full[g0:g1]).any(dim=1)
        lat_changed = (tg["latent_in"] != latent[g0:g1]).any(dim=1)
        route_changed = (tg["router_topk_idx"].long().sort(dim=1).values != idx_full[g0:g1].sort(dim=1).values).any(dim=1)
        out["groups"].append({"span": [g0, g1], "M": g1 - g0, "ffn_rows_changed": int(changed.sum()),
                              "ffn_real_rows_changed": int((changed & real_dev[g0:g1]).sum()),
                              "y_lat_rows_changed": int(y_changed.sum()), "latent_rows_changed": int(lat_changed.sum()),
                              "expert_set_rows_changed": int(route_changed.sum()),
                              "ffn_max_abs_diff": float((part.float() - full[g0:g1].float()).abs().max())})
        del part, tg, changed, y_changed, lat_changed, route_changed
    out["ffn_rows_changed"] = int(changed_all.sum())
    out["ffn_real_rows_changed"] = int((changed_all & real_dev).sum())
    out["ffn_padded_rows_changed"] = int((changed_all & ~real_dev).sum())
    # every touched expert, full M vs each group's M
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    counts = torch.bincount(idx_full.flatten(), minlength=int(cfg["num_experts"]))
    experts, changed_rows, tested = [], 0, 0
    for expert in torch.nonzero(counts).flatten().tolist():
        rows = (idx_full == expert).any(dim=1).nonzero().flatten()
        ex = {k: v.to(dev) for k, v in kdp.dequant_expert(weights, expert, dtype=latent.dtype).items()}

        def chain(x):
            return F.linear(kdp.situ(F.linear(x, ex["w1"]), F.linear(x, ex["w3"]), beta, linear_beta), ex["w2"])

        y_e = chain(latent[rows])
        per = []
        for g0, g1 in spans:
            sel = (rows >= g0) & (rows < g1)
            m = int(sel.sum())
            if not m:
                continue
            d = int((chain(latent[rows[sel]]) != y_e[sel]).any(dim=1).sum())
            per.append([g0, m, d])
            changed_rows += d
            tested += m
        if any(p[2] for p in per):
            experts.append({"expert": int(expert), "M_full": int(rows.numel()), "groups_g0_M_changed": per})
        del ex, y_e
    out.update({"experts_touched": int((counts > 0).sum()), "expert_rows_tested": tested,
                "expert_rows_changed_by_M": changed_rows, "experts_with_changes": experts,
                "final_prefix": full})
    return out


def run_mechanism(a) -> int:
    """1.0.1 --mode mechanism: single-call L0, L1 (checked against the fold), then the closed loop at L2."""
    module = load_frozen(a.frozen_tool, a.frozen_sha)
    ref = fold_reference(a.fold_receipt, LAYERS)
    os.makedirs(a.out, exist_ok=True)
    wait_ram(a.min_mem_available_gib)
    torch.manual_seed(0)
    cp = module.CachePass(module.build_parser().parse_args(tool_argv(a, "auto", os.path.join(a.out, "arm-bf16"))))
    corpus = {"tokens": cp.corpus.tokens, "real": cp.real_positions, "groups": len(cp.corpus.groups),
              "sha256": cp.corpus_sha}
    if corpus != EXPECTED_CORPUS:
        raise SystemExit(f"corpus {corpus} is not the fold's {EXPECTED_CORPUS}")
    run_bf16 = prepare(module, cp)
    kdp, common = cp.kdp, cp.common
    real_cpu = real_mask(cp.corpus.segments)
    spans = group_spans(cp.corpus.groups)
    result: dict[str, Any] = {
        "schema": "omni-campaign.k3_b2_geometry_diag.mechanism.v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_file(os.path.abspath(__file__)), "started_utc": utc(),
        "frozen_tool": {"path": a.frozen_tool, "sha256": a.frozen_sha, "version": module.TOOL_VERSION},
        "engine": cp.engine.record(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(cp.device) if cp.cuda else None,
        "corpus": {**corpus, "group_spans": spans}, "layers": {}}
    out_path = os.path.join(a.out, "b2_geometry_mechanism.json")
    with torch.no_grad():
        prefix, blocks, nb = cp.init_state(run_bf16)
        for layer_idx in LAYERS:
            snapshot = (layer_idx % cp.block_size) == 0
            weights, raw, _timing = kdp.read_layer(cp.index, layer_idx, cp.stager, cp.device, cp.experts_device,
                                                   cp.compute_dtype)
            del raw
            runner = cp.engine.layer.StreamedLayer(weights, cp.cfg, kda_backend=cp.args.kda_backend)
            row_chunk = cp.args.row_chunk_kda if weights.is_kda else cp.args.row_chunk_mla
            t0 = time.time()
            if layer_idx < 2:
                nb, _ = single_step(cp, runner, layer_idx, prefix, blocks, nb, snapshot, row_chunk, None, False)
            else:
                pair = kdp.Pair(prefix, blocks[:, :nb])
                carried, ffn_in, _b = kdp.attention_half(runner, pair, cp.corpus, cp.device, row_chunk=row_chunk,
                                                         token_chunk=cp.args.token_chunk)
                del _b, pair
                loop = closed_loop(cp, runner, weights, carried, ffn_in, spans, real_cpu)
                new = loop.pop("final_prefix")
                if snapshot:
                    blocks[:, nb].copy_(prefix)
                    nb += 1
                prefix.copy_(new.to(prefix.device))
                del new, carried, ffn_in
                result["closed_loop"] = loop
                log("CLOSED_LOOP " + json.dumps({k: v for k, v in loop.items() if k not in ("experts_with_changes",)}))
                log(f"experts with M-dependent rows: {len(loop['experts_with_changes'])} "
                    + json.dumps(loop["experts_with_changes"][:12]))
            digest = common.sha256_tensor(prefix)
            result["layers"][str(layer_idx)] = {"prefix_sha256": digest, "matches_fold": digest == ref[layer_idx],
                                                "moe": bool(weights.is_moe), "seconds": round(time.time() - t0, 1)}
            log(f"L{layer_idx:02d} single-call prefix {digest[:16]} matches fold {digest == ref[layer_idx]}")
            del weights, runner
            torch.cuda.empty_cache()
    result["ended_utc"] = utc()
    digest = write_json(out_path, result)
    log(f"receipt {out_path} sha256 {digest}")
    ok = all(v["matches_fold"] for v in result["layers"].values())
    return 0 if ok else 3


def run_verdict(a) -> int:
    """1.0.2 --mode verdict: re-judge the sealed arms receipt with the mechanism run's closed loop (CPU; both receipts
    are read, never rewritten)."""
    with open(a.receipt, "r", encoding="utf-8") as handle:
        result = json.load(handle)
    closed = None
    if a.mechanism:
        with open(a.mechanism, "r", encoding="utf-8") as handle:
            mech = json.load(handle)
        closed = mech.get("closed_loop")
        ok = all(v.get("matches_fold") for v in (mech.get("layers") or {}).values())
        if not ok:
            raise SystemExit(f"{a.mechanism}: the mechanism run's single-call states do not match the fold receipt")
    v = verdict(result, closed)
    out = {"schema": "omni-campaign.k3_b2_geometry_diag.verdict.v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
           "tool_sha256": sha256_file(os.path.abspath(__file__)), "at_utc": utc(),
           "arms_receipt": {"path": a.receipt, "sha256": sha256_file(a.receipt), "verdict_as_written": result.get("verdict")},
           "mechanism_receipt": {"path": a.mechanism, "sha256": sha256_file(a.mechanism)} if a.mechanism else None,
           "verdict": v}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "b2_geometry_verdict.json")
    digest = write_json(path, out)
    log("VERDICT " + json.dumps(v))
    log(f"verdict {path} sha256 {digest}")
    return 0 if v["cause"] != "undetermined" else 3


# --------------------------------------------------------------------------
# 1.1.0 --mode identity: the fold-identity gate (pure parts unit-tested on CPU with injected steps)
# --------------------------------------------------------------------------
def parse_layer_spec(spec: str) -> list[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def closed_loop_layers(spec: str, num_layers: int, is_kda: dict[int, bool], block_size: int,
                       first_moe: int | None) -> list[int]:
    """``structural``: every MLA layer, every block push, the first MoE layer, IDENTITY_EXTRA_LAYERS and the last
    layer; ``all``; ``none``; or an explicit list ``a,b,c-d``."""
    if spec == "all":
        return list(range(num_layers))
    if spec == "none":
        return []
    if spec == "structural":
        missing = [i for i in range(num_layers) if i not in is_kda]
        if missing:
            raise SystemExit(f"layer kinds unknown for {missing[:8]}: cannot pick the MLA layers")
        chosen = {i for i in range(num_layers) if not is_kda[i]}
        chosen |= {i for i in range(num_layers) if i % block_size == 0}
        chosen |= set(IDENTITY_EXTRA_LAYERS) | {num_layers - 1}
        if first_moe is not None:
            chosen.add(int(first_moe))
        return sorted(i for i in chosen if 0 <= i < num_layers)
    return [i for i in parse_layer_spec(spec) if 0 <= i < num_layers]


def judge_closed(c_real: dict[str, Any], nb_group: int | None, nb_single: int, slot_ok: bool, earlier_ok: bool,
                 rule: dict[str, Any] = IDENTITY_RULE) -> dict[str, Any]:
    """One closed-loop layer against IDENTITY_RULE (NaN never passes: every bound is written as not (x <= max))."""
    n = int(c_real.get("positions") or 0)
    eq = c_real.get("bit_equal_positions")
    unequal = (n - int(eq)) if eq is not None else None
    frac = (unequal / n) if n and unequal is not None else None
    rel, cos_min = c_real.get("relL2"), c_real.get("cos_min")
    why = []
    if not n:
        why.append("no real positions compared")
    if frac is None:
        why.append("no bit-equality count (the two states are not one dtype)")
    elif not frac <= rule["real_unequal_fraction_max"]:
        why.append(f"{unequal} of {n} real positions differ ({frac:.3%} > {rule['real_unequal_fraction_max']:.1%})")
    if rel is None or not rel <= rule["real_relL2_max"]:
        why.append(f"real relL2 {rel} > {rule['real_relL2_max']}")
    if n and (cos_min is None or not cos_min >= rule["real_cos_min_min"]):
        why.append(f"real cos_min {cos_min} < {rule['real_cos_min_min']}")
    if nb_group != nb_single:
        why.append(f"blocks after the layer: production {nb_group}, single call {nb_single}")
    if not slot_ok:
        why.append("the pushed block slot differs from the single call's")
    if not earlier_ok:
        why.append("an earlier block slot was modified")
    return {"ok": not why, "real_unequal_positions": unequal, "real_unequal_fraction": frac, "reasons": why}


def slots_equal(x: torch.Tensor, y: torch.Tensor, slots, chunk: int = 8192) -> bool:
    """Bit-equality of block slots, in row chunks (no [N, nb, H] temporary)."""
    n = int(x.shape[0])
    for b in slots:
        for a in range(0, n, chunk):
            if not torch.equal(x[a:a + chunk, b], y[a:a + chunk, b]):
                return False
    return True


def identity_verdict(layers_rec: dict[str, Any], layers: list[int], loop_layers: list[int]) -> dict[str, Any]:
    missing = [i for i in layers if str(i) not in layers_rec]
    unmatched = [i for i in layers if str(i) in layers_rec and layers_rec[str(i)].get("matches_fold") is not True]
    loop_failed = [i for i in loop_layers if str(i) in layers_rec
                   and (layers_rec[str(i)].get("closed_loop") or {}).get("ok") is False]
    loop_missing = [i for i in loop_layers if (layers_rec.get(str(i)) or {}).get("closed_loop", {}).get("ok") is None]
    ok = not (missing or unmatched or loop_failed or loop_missing)
    cause = None if ok else ("identity" if unmatched else "structural" if loop_failed else "incomplete")
    return {"pass": ok, "cause": cause, "layers_expected": len(layers), "layers_checked": len(layers) - len(missing),
            "layers_matching_fold": len(layers) - len(missing) - len(unmatched),
            "first_layer_off_fold": min(unmatched) if unmatched else None,
            "closed_loop_expected": len(loop_layers),
            "closed_loop_passed": len(loop_layers) - len(loop_failed) - len(loop_missing),
            "closed_loop_failed": loop_failed, "missing_layers": missing[:16],
            "rule": "PASS = every layer's single-call prefix sha equals the fold teacher sha AND every closed-loop "
                    "layer meets IDENTITY_RULE (structural / identity failures are functional; they block K3)"}


def identity_pass(layers, loop_layers, ref, prefix, blocks, nb, *, block_size, group_step, single_step_fn, digest,
                  real, spans, prior=None, keep_going=False, after_layer=None, rule=IDENTITY_RULE):
    """The per-layer loop of --mode identity, engine-free (the steps are injected).  ``group_step(L, p, b, nb) -> nb``
    is the production path and ``single_step_fn(L, p, b, nb) -> nb`` the fold geometry; both update their state in
    place.  The production path always gets a copy of the single call's state ENTERING the layer (identical inputs;
    the block copy ``bg`` is kept equal to ``blocks`` on the pushed slots).  Returns (layer records, nb, failed)."""
    rec = dict(prior or {})
    loop = set(loop_layers)
    bg = torch.empty_like(blocks)
    bg.copy_(blocks)                                    # touched: the whole footprint is resident from the start
    pg = torch.zeros_like(prefix) if loop else None
    failed = False
    for layer in layers:
        if str(layer) in rec:
            continue
        snapshot = (layer % block_size) == 0
        entry: dict[str, Any] = {"layer": layer, "snapshot": snapshot}
        nb_before, nb_g = nb, None
        if layer in loop:
            t0 = time.time()
            pg.copy_(prefix)
            nb_g = group_step(layer, pg, bg, nb)
            entry["group_seconds"] = round(time.time() - t0, 1)
        t1 = time.time()
        nb = single_step_fn(layer, prefix, blocks, nb)
        entry["single_seconds"] = round(time.time() - t1, 1)
        got = digest(prefix)
        entry.update({"prefix_sha256": got, "fold_sha256": ref.get(layer), "matches_fold": got == ref.get(layer),
                      "blocks_after": nb})
        if layer in loop:
            slot_ok = (not snapshot) or slots_equal(bg, blocks, [nb_before])
            earlier_ok = slots_equal(bg, blocks, range(nb_before))
            c_real = compare(pg, prefix, real, spans=spans)
            c_pad = compare(pg, prefix, ~real, spans=spans) if bool((~real).any()) else None
            judged = judge_closed(c_real, nb_g, nb, slot_ok, earlier_ok, rule)
            entry["closed_loop"] = {**judged, "real": c_real, "padded": c_pad, "pushed_slot_bit_equal": slot_ok,
                                    "earlier_slots_untouched": earlier_ok, "blocks_after_production": nb_g}
            if not (slot_ok and earlier_ok):
                bg[:, :nb].copy_(blocks[:, :nb])        # restore identical inputs for the next closed loop
        elif snapshot:
            bg[:, nb_before].copy_(blocks[:, nb_before])
        bad = not entry["matches_fold"] or (layer in loop and not entry["closed_loop"]["ok"])
        rec[str(layer)] = entry
        if after_layer is not None:
            after_layer(layer, entry, prefix, blocks, nb, bad)
        if bad:
            failed = True
            if not keep_going:
                break
    return rec, nb, failed


def save_identity_state(state_dir: str, meta: dict[str, Any], layer_done: int, prefix: torch.Tensor,
                        blocks: torch.Tensor, nb: int, prefix_sha: str, layers_rec: dict[str, Any], digest) -> None:
    """Checkpoint after a layer that passed: each pushed block slot once (slots never change), the prefix under a
    per-layer name, then state.json (written last, atomically; it names the prefix file it belongs to)."""
    from safetensors.torch import save_file

    os.makedirs(state_dir, exist_ok=True)
    path = os.path.join(state_dir, "state.json")
    prior = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            prior = json.load(handle)
    slots = dict(prior.get("slot_sha256") or {}) if all(prior.get(k) == v for k, v in meta.items()) else {}
    for b in range(nb):
        target = os.path.join(state_dir, f"block_{b}.safetensors")
        if str(b) in slots and os.path.exists(target):
            continue
        slot = blocks[:, b].contiguous()
        tmp = f"{target}.tmp{os.getpid()}"
        save_file({"block": slot}, tmp)
        os.replace(tmp, target)
        slots[str(b)] = digest(slot)
        del slot
    name = f"prefix_L{layer_done:02d}.safetensors"
    tmp = os.path.join(state_dir, f"{name}.tmp{os.getpid()}")
    save_file({"prefix": prefix.contiguous()}, tmp)
    os.replace(tmp, os.path.join(state_dir, name))
    write_json(path, {**meta, "layer_done": layer_done, "nb": nb, "prefix_file": name, "prefix_sha256": prefix_sha,
                      "slot_sha256": slots, "layer_records": layers_rec, "at_utc": utc()})
    for other in os.listdir(state_dir):
        if other.startswith("prefix_L") and other.endswith(".safetensors") and other != name:
            os.remove(os.path.join(state_dir, other))


def load_identity_state(state_dir: str, meta: dict[str, Any], prefix: torch.Tensor, blocks: torch.Tensor, digest):
    """Resume: refuse another frozen tool / corpus / fold receipt / diag build; every tensor re-hashed on load.
    Returns (layer_done, nb, layer records) or None when there is no checkpoint."""
    from safetensors.torch import load_file

    path = os.path.join(state_dir, "state.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        st = json.load(handle)
    for key, want in meta.items():
        if st.get(key) != want:
            raise SystemExit(f"refusing to resume {state_dir}: {key} is {st.get(key)!r}, this run {want!r}")
    p = load_file(os.path.join(state_dir, st["prefix_file"]))["prefix"]
    if digest(p) != st["prefix_sha256"]:
        raise SystemExit(f"refusing to resume {state_dir}: {st['prefix_file']} does not hash to state.json's sha")
    prefix.copy_(p.to(prefix.device))
    del p
    for b in range(int(st["nb"])):
        t = load_file(os.path.join(state_dir, f"block_{b}.safetensors"))["block"]
        if digest(t) != st["slot_sha256"][str(b)]:
            raise SystemExit(f"refusing to resume {state_dir}: block_{b} does not hash to state.json's sha")
        blocks[:, b].copy_(t.to(blocks.device))
        del t
    return int(st["layer_done"]), int(st["nb"]), st["layer_records"]


def run_identity(a) -> int:
    """1.1.0 --mode identity: single-call chain over every layer against the fold's teacher sha, plus the production
    CachePass.layer closed loop at the structural layers (see IDENTITY_RULE).  Exit 0 = PASS, 3 = FAIL."""
    module = load_frozen(a.frozen_tool, a.frozen_sha)
    with open(a.fold_receipt, "r", encoding="utf-8") as handle:
        fold = json.load(handle)
    num_layers = len(fold["layers"])
    layers = parse_layer_spec(a.layers) if a.layers else list(range(num_layers))
    if layers != list(range(layers[0], layers[-1] + 1)) or layers[0] != 0:
        raise SystemExit(f"--layers must be one range from 0, got {a.layers}")
    ref = fold_reference(a.fold_receipt, layers)
    is_kda = {int(k): bool(v["is_kda"]) for k, v in fold["layers"].items() if "is_kda" in v}
    os.makedirs(a.out, exist_ok=True)
    wait_ram(a.min_mem_available_gib)
    torch.manual_seed(0)
    cp = module.CachePass(module.build_parser().parse_args(
        tool_argv(a, "auto", os.path.join(a.out, "arm-bf16"), layers=f"{layers[0]}-{layers[-1]}")))
    corpus = {"tokens": cp.corpus.tokens, "real": cp.real_positions, "sha256": cp.corpus_sha}
    if corpus != {k: EXPECTED_CORPUS[k] for k in corpus}:
        raise SystemExit(f"corpus {corpus} is not the fold's {EXPECTED_CORPUS}")
    loop_layers = [i for i in closed_loop_layers(a.closed_loop_layers, num_layers, is_kda, cp.block_size,
                                                 fold.get("first_moe_layer")) if i in layers]
    run_bf16 = prepare(module, cp)
    kdp, common = cp.kdp, cp.common
    real_cpu = real_mask(cp.corpus.segments)
    spans = group_spans(cp.corpus.groups)
    plan_shards = {i: cp.index.shards_for(cp.index.layer_keys(i)) for i in layers}
    tool_sha = sha256_file(os.path.abspath(__file__))
    meta = {"tool_sha256": tool_sha, "frozen_sha256": a.frozen_sha, "corpus_sha256": cp.corpus_sha,
            "fold_receipt_sha256": sha256_file(a.fold_receipt), "layer_range": [layers[0], layers[-1]]}
    result: dict[str, Any] = {
        "schema": "omni-campaign.k3_b2_geometry_diag.identity.v1", "mode": "identity", "tool": TOOL_NAME,
        "tool_version": TOOL_VERSION, "tool_sha256": tool_sha, "started_utc": utc(),
        "frozen_tool": {"path": a.frozen_tool, "sha256": a.frozen_sha, "version": module.TOOL_VERSION},
        "fold_receipt": {"path": a.fold_receipt, "sha256": meta["fold_receipt_sha256"]},
        "engine": cp.engine.record(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(cp.device) if cp.cuda else None,
        "num_threads": torch.get_num_threads(), "fold_num_threads":
            ((fold.get("resume_context") or {}).get("runtime") or {}).get("num_threads"),
        "corpus": {**corpus, "group_tokens": a.group_tokens, "groups": len(cp.corpus.groups), "group_spans": spans,
                   "padded": int((~real_cpu).sum())},
        "rule": IDENTITY_RULE, "closed_loop_spec": a.closed_loop_layers, "closed_loop_layers": loop_layers,
        "layers": {}}
    out_path = os.path.join(a.out, "b2_identity.json")
    state_dir = os.path.join(a.out, "state")
    group_records: dict[int, Any] = {}

    def single_fn(layer, p, b, n):
        snapshot = (layer % cp.block_size) == 0
        weights, raw, _timing = kdp.read_layer(cp.index, layer, cp.stager, cp.device, cp.experts_device,
                                               cp.compute_dtype)
        del raw
        runner = cp.engine.layer.StreamedLayer(weights, cp.cfg, kda_backend=cp.args.kda_backend)
        row_chunk = cp.args.row_chunk_kda if weights.is_kda else cp.args.row_chunk_mla
        n, _ = single_step(cp, runner, layer, p, b, n, snapshot, row_chunk, None, False)
        del weights, runner
        if cp.cuda:
            torch.cuda.empty_cache()
        return n

    def group_fn(layer, p, b, n):
        n = cp.layer(run_bf16, layer, p, b, n, plan_shards)     # the frozen production path, unmodified
        r = cp.receipt["layers"][str(layer)]
        group_records[layer] = {k: r.get(k) for k in ("read_wall_seconds", "compute_seconds", "groups", "blocks_after",
                                                      "prefix_out_sha256", "device_peak_bytes", "rss_after_bytes")}
        return n

    def after(layer, entry, p, b, n, bad):
        entry["is_kda"] = is_kda.get(layer)
        if layer in group_records:
            entry["production_layer_record"] = group_records.pop(layer)
        cl = entry.get("closed_loop")
        log(f"L{layer:02d} {'KDA' if is_kda.get(layer) else 'MLA'}{' push' if entry['snapshot'] else ''} "
            f"single==fold {entry['matches_fold']} ({entry['prefix_sha256'][:16]})"
            + (f" closed-loop {'ok' if cl['ok'] else 'FAIL'} unequal {cl['real_unequal_positions']}/"
               f"{cl['real']['positions']} relL2 {cl['real']['relL2']:.3e} cos_min {cl['real']['cos_min']} "
               f"per group {cl['real'].get('unequal_positions_per_group')} {'; '.join(cl['reasons'])}" if cl else "")
            + f" seconds single {entry['single_seconds']} production {entry.get('group_seconds', '-')}")
        result["layers"] = dict(result["layers"], **{str(layer): entry})
        write_json(out_path + ".partial", result)
        last = layer == layers[-1]
        if not bad and a.checkpoint_every > 0 and ((layer + 1) % a.checkpoint_every == 0 or last):
            save_identity_state(state_dir, meta, layer, p, b, n, entry["prefix_sha256"], result["layers"],
                                common.sha256_tensor)

    with torch.no_grad():
        prefix, blocks, nb = cp.init_state(run_bf16)
        blocks.zero_()                                   # touched: the whole footprint is resident from the start
        prior = None
        if a.resume:
            got = load_identity_state(state_dir, meta, prefix, blocks, common.sha256_tensor)
            if got is not None:
                done, nb, prior = got
                result["resumed_after_layer"] = done
                log(f"resumed after L{done:02d} ({nb} blocks) from {state_dir}")
        result["layers"] = dict(prior or {})
        rec, nb, failed = identity_pass(
            layers, loop_layers, ref, prefix, blocks, nb, block_size=cp.block_size, group_step=group_fn,
            single_step_fn=single_fn, digest=common.sha256_tensor, real=real_cpu, spans=spans, prior=prior,
            keep_going=a.keep_going, after_layer=after)
    result["layers"] = rec
    result["verdict"] = identity_verdict(rec, layers, loop_layers)
    result["ended_utc"] = utc()
    digest = write_json(out_path, result)
    try:
        os.remove(out_path + ".partial")
    except OSError:
        pass
    log("IDENTITY " + json.dumps({k: v for k, v in result["verdict"].items() if k != "rule"}))
    log(f"receipt {out_path} sha256 {digest}")
    return 0 if result["verdict"]["pass"] else 3


def wait_ram(gib: float) -> None:
    n = 0
    while True:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            avail = next(int(line.split()[1]) for line in handle if line.startswith("MemAvailable")) / 1048576
        if avail >= gib:
            log(f"MemAvailable {avail:.1f} GiB >= {gib}")
            return
        if n % 10 == 0:
            log(f"MemAvailable {avail:.1f} GiB < {gib}; waiting (never competing)")
        n += 1
        time.sleep(60)


def run(a) -> int:
    module = load_frozen(a.frozen_tool, a.frozen_sha)
    ref = fold_reference(a.fold_receipt, LAYERS)
    bisect = {}
    if a.bisect and os.path.exists(a.bisect):
        with open(a.bisect, "r", encoding="utf-8") as handle:
            bisect = {int(r["layer"]): r["mine"] for r in json.load(handle)["layers"] if int(r["layer"]) in LAYERS}
    os.makedirs(a.out, exist_ok=True)
    wait_ram(a.min_mem_available_gib)
    torch.manual_seed(0)
    cp = module.CachePass(module.build_parser().parse_args(tool_argv(a, "auto", os.path.join(a.out, "arm-bf16"))))
    corpus = {"tokens": cp.corpus.tokens, "real": cp.real_positions, "groups": len(cp.corpus.groups),
              "sha256": cp.corpus_sha}
    if corpus != EXPECTED_CORPUS:
        raise SystemExit(f"corpus {corpus} is not the fold's {EXPECTED_CORPUS}")
    if cp.layers != LAYERS:
        raise SystemExit(f"layers {cp.layers}")
    run_bf16 = prepare(module, cp)
    cpf = module.CachePass(module.build_parser().parse_args(tool_argv(a, "fp32", os.path.join(a.out, "arm-fp32"))))
    run_fp32 = prepare(module, cpf)
    kdp, common = cp.kdp, cp.common
    real_cpu = real_mask(cp.corpus.segments)
    spans = group_spans(cp.corpus.groups)
    real_dev = real_cpu.nonzero().flatten().to(cp.device)
    result: dict[str, Any] = {
        "schema": "omni-campaign.k3_b2_geometry_diag.v1", "tool": TOOL_NAME, "tool_version": TOOL_VERSION,
        "tool_sha256": sha256_file(os.path.abspath(__file__)), "started_utc": utc(),
        "frozen_tool": {"path": a.frozen_tool, "sha256": a.frozen_sha, "version": module.TOOL_VERSION},
        "engine": cp.engine.record(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(cp.device) if cp.cuda else None,
        "corpus": {**corpus, "group_spans": spans, "padded": int((~real_cpu).sum())},
        "fold_reference_prefix_sha256": {str(k): v for k, v in ref.items()},
        "bisect_prefix_sha256": {str(k): v for k, v in bisect.items()},
        "arms": {"group": "(i) frozen CachePass.layer, 5 row groups", "group_nopad": "(ii) 5 row groups, FFN over real "
                 "positions only", "single": "(iii) one call over the corpus (fold geometry)", "single_nopad":
                 "(ii') one call, FFN over real positions only", "fp32": "frozen CachePass.layer at fp32, 5 row groups"},
        "layers": {}, "failed_arms": {}}
    out_path = os.path.join(a.out, "b2_geometry_diag.json")
    with torch.no_grad():
        prefix0, blocks0, nb0 = cp.init_state(run_bf16)
        state = {arm: [prefix0.clone(), blocks0.clone(), nb0] for arm in BF16_ARMS}
        del prefix0, blocks0
        pf, bf, nbf = cpf.init_state(run_fp32)
        state["fp32"] = [pf, bf, nbf]
        plan_shards = {i: cp.index.shards_for(cp.index.layer_keys(i)) for i in LAYERS}
        alive = set(ARMS)
        for layer_idx in LAYERS:
            entry: dict[str, Any] = {"seconds": {}, "device_peak_bytes": {}}
            snapshot = (layer_idx % cp.block_size) == 0
            t0 = time.time()
            weights, raw, timing = kdp.read_layer(cp.index, layer_idx, cp.stager, cp.device, cp.experts_device,
                                                  cp.compute_dtype)
            del raw
            runner = cp.engine.layer.StreamedLayer(weights, cp.cfg, kda_backend=cp.args.kda_backend)
            row_chunk = cp.args.row_chunk_kda if weights.is_kda else cp.args.row_chunk_mla
            entry.update({"is_kda": bool(weights.is_kda), "moe": bool(weights.is_moe), "snapshot": snapshot,
                          "read_seconds": round(time.time() - t0, 1), "compute_dtype": timing.get("compute_dtype")})
            for arm in ("single", "single_nopad", "group_nopad"):
                if arm not in alive:
                    continue
                torch.cuda.reset_peak_memory_stats(cp.device)
                t1 = time.time()
                try:
                    p, b, nb = state[arm]
                    if arm == "group_nopad":
                        state[arm][2] = group_nopad_step(module, cp, runner, layer_idx, p, b, nb, snapshot, row_chunk)
                    else:
                        keep = arm == "single" and layer_idx == 2 and weights.is_moe
                        state[arm][2], ffn_in = single_step(cp, runner, layer_idx, p, b, nb, snapshot, row_chunk,
                                                            real_dev, arm == "single_nopad", keep_ffn_in=keep)
                        if keep:
                            t2 = time.time()
                            try:
                                result["mechanism"] = mechanism(cp, weights, ffn_in, spans, real_cpu)
                                result["mechanism"]["seconds"] = round(time.time() - t2, 1)
                                log("MECHANISM " + json.dumps({k: v for k, v in result["mechanism"].items()
                                                               if k not in ("groups", "experts")}))
                            except Exception as exc:  # the test is evidence, never a reason to lose the arms
                                result["mechanism"] = {"error": repr(exc)}
                                log(f"mechanism test failed: {exc!r}")
                            del ffn_in
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    alive.discard(arm)
                    result["failed_arms"][arm] = {"layer": layer_idx, "error": repr(exc)[:500]}
                    log(f"arm {arm} FAILED at L{layer_idx}: {exc!r}"[:400])
                torch.cuda.synchronize(cp.device)
                entry["seconds"][arm] = round(time.time() - t1, 1)
                entry["device_peak_bytes"][arm] = int(torch.cuda.max_memory_allocated(cp.device))
            del weights, runner
            torch.cuda.empty_cache()
            for arm, owner, rn in (("group", cp, run_bf16), ("fp32", cpf, run_fp32)):
                if arm not in alive:
                    continue
                t1 = time.time()
                try:
                    p, b, nb = state[arm]
                    state[arm][2] = owner.layer(rn, layer_idx, p, b, nb, plan_shards)
                    rec = owner.receipt["layers"][str(layer_idx)]
                    entry["device_peak_bytes"][arm] = rec.get("device_peak_bytes")
                    entry.setdefault("frozen_layer_record", {})[arm] = {
                        k: rec.get(k) for k in ("read_wall_seconds", "compute_seconds", "groups", "blocks_after",
                                                "prefix_out_sha256", "dequant_seconds_engine")}
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    alive.discard(arm)
                    result["failed_arms"][arm] = {"layer": layer_idx, "error": repr(exc)[:500]}
                    log(f"arm {arm} FAILED at L{layer_idx}: {exc!r}"[:400])
                entry["seconds"][arm] = round(time.time() - t1, 1)
                torch.cuda.empty_cache()
            # identity: the fold's teacher prefix sha (k3_common.sha256_tensor over the [143360, 7168] prefix)
            digests = {}
            for arm in ("single", "group"):
                if arm in alive:
                    digests[arm] = common.sha256_tensor(state[arm][0])
            entry["prefix_sha256"] = digests
            entry["single_matches_fold"] = digests.get("single") == ref[layer_idx] if "single" in digests else None
            entry["group_matches_fold"] = digests.get("group") == ref[layer_idx] if "group" in digests else None
            entry["group_matches_bisect"] = (digests.get("group") == bisect[layer_idx]
                                             if "group" in digests and layer_idx in bisect else None)
            cmp_: dict[str, Any] = {}
            if "single" in alive:
                s = state["single"][0]
                if "group" in alive:
                    cmp_["group_vs_single_real"] = compare(state["group"][0], s, real_cpu, spans=spans)
                    cmp_["group_vs_single_padded"] = compare(state["group"][0], s, ~real_cpu, spans=spans)
                for arm in ("group_nopad", "single_nopad"):
                    if arm in alive:
                        cmp_[f"{arm}_vs_single_real"] = compare(state[arm][0], s, real_cpu, spans=spans)
            if "fp32" in alive:
                f = state["fp32"][0]
                for arm in BF16_ARMS:
                    if arm in alive:
                        cmp_[f"{arm}_vs_fp32_real"] = compare(state[arm][0], f, real_cpu)
            entry["compare"] = cmp_
            result["layers"][str(layer_idx)] = entry
            log(f"L{layer_idx:02d} {'KDA' if entry['is_kda'] else 'MLA'} moe={entry['moe']} seconds {entry['seconds']} "
                f"single==fold {entry['single_matches_fold']} group==fold {entry['group_matches_fold']} "
                f"group==bisect {entry['group_matches_bisect']} sha {json.dumps({k: v[:16] for k, v in digests.items()})}")
            for key, c in cmp_.items():
                log(f"L{layer_idx:02d} {key}: positions {c['positions']} bit_equal {c['bit_equal_positions']} "
                    f"relL2 {c['relL2']:.3e} max_abs {c['max_abs_diff']:.4g} cos_min {c['cos_min']}"
                    + (f" unequal/group {c['unequal_positions_per_group']}" if c.get('unequal_positions_per_group') else ""))
            write_json(out_path + ".partial", result)
    result["verdict"] = verdict(result)   # arms only: no closed loop yet, so never bf16_batch_numerics here
    result["ended_utc"] = utc()
    digest = write_json(out_path, result)
    try:
        os.remove(out_path + ".partial")
    except OSError:
        pass
    log("VERDICT " + json.dumps(result["verdict"]))
    log(f"receipt {out_path} sha256 {digest}")
    return 0 if result["verdict"]["cause"] != "undetermined" else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--frozen-tool", default=DEFAULT_FROZEN_TOOL)
    ap.add_argument("--frozen-sha", default=FROZEN_TOOL_SHA256)
    ap.add_argument("--input-ids", action="append", default=None, help="the fold's three files, in the fold's order")
    ap.add_argument("--fold-receipt", default=DEFAULT_FOLD_RECEIPT)
    ap.add_argument("--bisect", default=DEFAULT_BISECT)
    ap.add_argument("--engine-dir", default=None)
    ap.add_argument("--src-dir", default=None)
    ap.add_argument("--group-tokens", type=int, default=32768)
    ap.add_argument("--min-mem-available-gib", type=float, default=70.0)
    ap.add_argument("--mode", choices=("arms", "mechanism", "verdict", "identity"), default="arms")
    ap.add_argument("--receipt", default=None, help="--mode verdict: the arms receipt b2_geometry_diag.json")
    ap.add_argument("--mechanism", default=None, help="--mode verdict: the mechanism receipt b2_geometry_mechanism.json")
    ap.add_argument("--layers", default=None, help="--mode identity: 0-N (default: every layer of the fold receipt)")
    ap.add_argument("--closed-loop-layers", default="structural",
                    help="--mode identity: structural | all | none | a,b,c-d")
    ap.add_argument("--checkpoint-every", type=int, default=4, help="--mode identity: checkpoint every N layers")
    ap.add_argument("--resume", action="store_true", help="--mode identity: continue from <out>/state")
    ap.add_argument("--keep-going", action="store_true", help="--mode identity: do not stop at the first failure")
    ap.add_argument("--out", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.mode == "verdict":
        if not a.receipt:
            raise SystemExit("--mode verdict needs --receipt")
        return run_verdict(a)
    names = [os.path.splitext(os.path.basename(p))[0] for p in a.input_ids or []]
    if names != CORPUS_FILES:
        raise SystemExit(f"--input-ids must be the fold's files in order {CORPUS_FILES}, got {names}")
    if a.mode == "identity":
        return run_identity(a)
    return run_mechanism(a) if a.mode == "mechanism" else run(a)


if __name__ == "__main__":
    raise SystemExit(main())
