#!/usr/bin/env python3
"""k3_accel_profile.py -- where the time goes in a production K3 teacher-cache layer (lane k3accel, 2026-10-14).

The K3 cache (``k3_teacher_cache`` 1.2.4, frozen sha 2aae4dc3...) runs at an estimated 45-50 h per 0.5M-token pass,
host-bound.  This tool measures WHY, on a layer window of the frozen tool, without changing a single number it
computes.  It never edits the frozen tool or the engine: it imports a COPY of the frozen tool by path (sha checked),
and the only things it replaces are three engine functions, each by a statement-for-statement copy that calls the
SAME functions in the SAME order with timers between the statements:

  kdp.attention_half   -> attention_half_timed   (H2D prefix / blocks, AttnRes reads, norm, attention core, ...)
  kdp.routed_y_lat     -> routed_y_lat_timed     (router, expert select, MXFP4 dequant, expert FFN)
  kdp.read_layer       -> timed wrapper, with kdp.read_tensors (file read) and LayerWeights.to (H2D) timed inside
  k3_layer.naive_recurrent_kda / F.scaled_dot_product_attention / kdp._chunked_linear / kdp.moe_write
                       -> timed wrappers (call through unchanged)

Proof that the copies change nothing: run ``split`` and the frozen tool itself with ``--prefix-digests`` on the
same window; every layer's ``prefix_out_sha256`` must be equal (``compare`` subcommand).

Subcommands
  split     run the frozen tool's ``run`` path (``--probe`` window) with the stage timers.  ``--sync`` (default on)
            synchronises the card at every stage boundary so each stage's wall is its own; every stage also records
            its LAUNCH time (wall until the Python call returned, before the closing sync = the host-side cost) and
            the main thread's CPU time.  ``--experiments`` adds measurement-only replays (never fed back into the
            pass, timed separately and subtracted from the enclosing stages): a torch.profiler replay of the KDA
            recurrence and of the routed MoE (device kernel time vs wall), the KDA recurrence captured in a CUDA
            graph (bitwise compare + time), the MoE with sorted dispatch and with dequant-once-across-groups
            (bitwise compare + time), a pageable vs pinned H2D bench and a checkpoint-write bench.
  monitor   run a command and sample, every ``--interval`` s: per-thread CPU of the deepest python process,
            /proc/<pid>/io read_bytes of the tree, MemAvailable; SIGTERM the tree if MemAvailable drops below
            ``--mem-floor-gib`` (the host-RAM rule: never compete, stop).
  compare   prefix digests of two probe receipts (identity of the timed copies) + a stage summary.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

TOOL_NAME = "k3_accel_profile"
TOOL_VERSION = "1.0.0"
FROZEN_K3_SHA256 = "2aae4dc3134dda725e3ee35572584fbf35532e0c4a7a5721a0b0c817fc2c4f9d"
GIB = 1 << 30


def log(msg: str) -> None:
    print(f"[{TOOL_NAME} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


# --------------------------------------------------------------------------
# stage accounting
# --------------------------------------------------------------------------
class Stages:
    """Inclusive per-(layer, stage) wall / launch / CPU seconds, with nesting and replay exclusion.

    ``wall`` is synchronised at both ends when ``sync``; ``launch`` stops before the closing sync, so for a stage with
    no internal sync it is the host-side cost of issuing it.  ``exclude(seconds)`` removes measurement-only replay time
    from every stage open at that moment."""

    def __init__(self, sync: bool, synchronize=None, clock=time.perf_counter, cpu_clock=time.thread_time):
        self.sync = bool(sync)
        self.synchronize = synchronize
        self.clock, self.cpu_clock = clock, cpu_clock
        self.layer: int | None = None
        self.acc: dict[tuple, dict[str, float]] = defaultdict(
            lambda: {"calls": 0, "wall_s": 0.0, "launch_s": 0.0, "cpu_s": 0.0, "bytes": 0})
        self.stack: list[dict[str, float]] = []
        self.paused = 0

    def _sync(self) -> None:
        if self.sync and self.synchronize is not None:
            self.synchronize()

    @contextlib.contextmanager
    def stage(self, name: str, nbytes: int = 0):
        if self.paused:
            yield
            return
        self._sync()
        frame = {"excluded": 0.0}
        self.stack.append(frame)
        t0, c0 = self.clock(), self.cpu_clock()
        try:
            yield
        finally:
            t1, c1 = self.clock(), self.cpu_clock()
            self._sync()
            t2 = self.clock()
            self.stack.pop()
            rec = self.acc[(self.layer, name)]
            rec["calls"] += 1
            rec["wall_s"] += (t2 - t0) - frame["excluded"]
            rec["launch_s"] += max((t1 - t0) - frame["excluded"], 0.0)
            rec["cpu_s"] += max((c1 - c0) - frame["excluded"], 0.0)
            rec["bytes"] += int(nbytes)

    def exclude(self, seconds: float) -> None:
        for frame in self.stack:
            frame["excluded"] += seconds

    @contextlib.contextmanager
    def replay(self):
        """Measurement-only work: not counted in any stage, and subtracted from the open ones."""
        self._sync()
        t0 = self.clock()
        self.paused += 1
        try:
            yield
        finally:
            self.paused -= 1
            self._sync()
            self.exclude(self.clock() - t0)

    def wrap(self, name: str, fn):
        def wrapped(*args, **kwargs):
            with self.stage(name):
                return fn(*args, **kwargs)
        wrapped.__wrapped__ = fn
        return wrapped

    def table(self) -> dict[str, dict[str, dict[str, float]]]:
        out: dict[str, dict[str, dict[str, float]]] = {}
        for (layer, name), rec in sorted(self.acc.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            out.setdefault(str(layer), {})[name] = {k: (round(v, 4) if isinstance(v, float) else v)
                                                    for k, v in rec.items()}
        return out


def exclusive(stages: dict[str, dict[str, float]]) -> dict[str, float]:
    """Exclusive wall per dotted stage name: inclusive minus its direct children's inclusive."""
    names = set(stages)
    out = {}
    for name, rec in stages.items():
        children = [n for n in names if n.startswith(name + ".") and "." not in n[len(name) + 1:]]
        out[name] = round(rec["wall_s"] - sum(stages[c]["wall_s"] for c in children), 4)
    return out


# --------------------------------------------------------------------------
# the frozen tool, by path
# --------------------------------------------------------------------------
def load_frozen_tool(path: str, expected_sha: str):
    got = sha256_file(path)
    if got != expected_sha:
        raise SystemExit(f"{path} sha256 {got[:16]} != expected {expected_sha[:16]}; refusing to profile another tool")
    spec = importlib.util.spec_from_file_location("k3_teacher_cache_frozen", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["k3_teacher_cache_frozen"] = module
    spec.loader.exec_module(module)
    return module, got


# --------------------------------------------------------------------------
# statement-for-statement timed copies of the engine (k3_dense_pass 1.2.5 fd7dbf8d, lines 957-998 and 1001-1036)
# --------------------------------------------------------------------------
def make_attention_half_timed(kdp, st: Stages, torch):
    def attention_half_timed(runner, pair, corpus, device, *, row_chunk, token_chunk):
        w = runner.w
        eps = runner.eps
        hidden = int(pair.prefix.shape[1])
        snapshot = (w.layer_idx % runner.block_size) == 0
        with st.stage("attn.alloc"):
            carried_all = torch.empty(corpus.tokens, hidden, dtype=pair.prefix.dtype, device=device)
            ffn_in_all = torch.empty_like(carried_all)
        for seg, r0, r1, start, stop in corpus.row_chunks(row_chunk):
            n_rows = r1 - r0
            src_p = pair.prefix[start:stop]
            with st.stage("attn.h2d_prefix", src_p.numel() * src_p.element_size()):
                p = src_p.to(device, non_blocking=False)
            src_b = pair.blocks[start:stop]
            name = "attn.h2d_blocks" if src_b.is_contiguous() else "attn.h2d_blocks_noncontig"
            with st.stage(name, src_b.numel() * src_b.element_size()):
                bl = src_b.to(device)
            del src_p, src_b
            if bl.shape[1] > 0:
                with st.stage("attn.attnres_pre"):
                    hidden_states = kdp.apply_attn_res(p, bl, w.res["self_attention_res_proj.weight"],
                                                       w.res["self_attention_res_norm.weight"], eps,
                                                       token_chunk_size=token_chunk)
            else:
                hidden_states = p
            if snapshot:
                with st.stage("attn.snapshot_cat_row"):
                    bl = torch.cat([bl, p.unsqueeze(1)], dim=1)
            with st.stage("attn.norm_in"):
                normed = kdp.rms_norm(hidden_states, w.res["input_layernorm.weight"], eps).view(n_rows, seg.T, hidden)
            with st.stage("attn.core"):
                attn_out, _ = runner.attention(normed, state=None)
            with st.stage("attn.residual"):
                attn_out = attn_out.reshape(-1, hidden)
                carried = attn_out if snapshot else p + attn_out
            with st.stage("attn.attnres_post"):
                h2 = kdp.apply_attn_res(carried, bl, w.res["mlp_res_proj.weight"], w.res["mlp_res_norm.weight"], eps,
                                        token_chunk_size=token_chunk)
            with st.stage("attn.norm_post"):
                h2 = kdp.rms_norm(h2, w.res["post_attention_layernorm.weight"], eps)
            with st.stage("attn.store"):
                carried_all[start:stop] = carried
                ffn_in_all[start:stop] = h2
            del p, bl, hidden_states, normed, attn_out, carried, h2
        blocks_after = pair.blocks
        if snapshot:
            # the engine builds this host-side copy; k3_teacher_cache discards it (`carried, ffn_in, _ = ...`)
            with st.stage("attn.snapshot_cat_host_discarded",
                          (pair.blocks.numel() + pair.prefix.numel()) * pair.prefix.element_size()):
                blocks_after = torch.cat([pair.blocks, pair.prefix.unsqueeze(1)], dim=1)
        return carried_all, ffn_in_all, blocks_after

    return attention_half_timed


def make_routed_y_lat_timed(kdp, st: Stages, torch, F, hook=None):
    def routed_y_lat_timed(weights, x, latent, cfg, device):
        beta = cfg.get("activation_situ_beta") or 1.0
        linear_beta = cfg.get("activation_situ_linear_beta")
        with st.stage("ffn.moe.route"):
            topk_idx, topk_weight, _ = kdp.route(x, weights, cfg)
        with st.stage("ffn.moe.count"):
            y_lat = torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)
            counts = torch.bincount(topk_idx.flatten(), minlength=int(cfg["num_experts"]))
            touched = torch.nonzero(counts).flatten().tolist()
        dequant_seconds = 0.0
        for expert in touched:
            with st.stage("ffn.moe.select"):
                hit = (topk_idx == expert)
                rows = hit.any(dim=1).nonzero().flatten()
                weight = topk_weight[rows].masked_fill(~hit[rows], 0.0).sum(dim=1)
            started = time.time()
            with st.stage("ffn.moe.dequant"):
                ex = kdp.dequant_expert(weights, expert, dtype=latent.dtype)
                ex = {k: v.to(device) for k, v in ex.items()}
            dequant_seconds += time.time() - started
            with st.stage("ffn.moe.expert_ffn"):
                chunk = latent[rows]
                activated = kdp.situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
                y_lat[rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
            del ex, chunk, activated
        taps = {
            "router_topk_idx": topk_idx.detach().to(torch.int16),
            "router_topk_w": topk_weight.detach().float(),
            "router_counts": counts.detach().cpu(),
            "experts_touched": int(len(touched)),
        }
        if hook is not None:
            hook(weights, x, latent, cfg, device, y_lat)
        return y_lat, taps, dequant_seconds

    return routed_y_lat_timed


# --------------------------------------------------------------------------
# measurement-only variants (never fed back into the pass); each is compared BITWISE with the engine's own result
# --------------------------------------------------------------------------
def moe_sorted_dispatch(kdp, torch, F, weights, x, latent, cfg, device):
    """The routed accumulator with one stable sort instead of 896 full ``topk_idx == e`` scans and per-expert
    ``nonzero`` syncs.  For each expert the rows come out ascending (stable sort of the flattened [N, k] index; an
    expert appears at most once per token), and the weight is the single matching top-k weight -- the engine's
    masked sum over k adds exact zeros to it -- so every expert GEMM sees the same inputs in the same order."""
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    topk_idx, topk_weight, _ = kdp.route(x, weights, cfg)
    k = topk_idx.shape[1]
    flat = topk_idx.flatten()
    order = torch.sort(flat, stable=True).indices
    tokens = order // k
    counts = torch.bincount(flat, minlength=int(cfg["num_experts"]))
    count_list = counts.tolist()  # the one host sync
    y_lat = torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)
    offset = 0
    flat_w = topk_weight.flatten()
    for expert, n in enumerate(count_list):
        if n == 0:
            continue
        sel = order[offset:offset + n]
        rows = tokens[offset:offset + n]
        weight = flat_w[sel]
        offset += n
        ex = kdp.dequant_expert(weights, expert, dtype=latent.dtype)
        ex = {kk: v.to(device) for kk, v in ex.items()}
        chunk = latent[rows]
        activated = kdp.situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
        y_lat[rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
        del ex, chunk, activated
    return y_lat


def moe_expert_outer(kdp, torch, F, weights, parts, cfg, device):
    """Several row groups (``parts`` = [(x, latent), ...]) through the routed MoE with each expert dequantized ONCE
    for all of them (the engine dequantizes it once per group).  Per group every select / GEMM call is the engine's
    own, on the same rows -- only the loop nesting changes."""
    beta = cfg.get("activation_situ_beta") or 1.0
    linear_beta = cfg.get("activation_situ_linear_beta")
    routed = []
    for x, latent in parts:
        topk_idx, topk_weight, _ = kdp.route(x, weights, cfg)
        counts = torch.bincount(topk_idx.flatten(), minlength=int(cfg["num_experts"]))
        routed.append((topk_idx, topk_weight, set(torch.nonzero(counts).flatten().tolist()),
                       torch.zeros(latent.shape[0], latent.shape[1], dtype=torch.float32, device=latent.device)))
    union = sorted(set().union(*[r[2] for r in routed]))
    dequants = 0
    for expert in union:
        ex = kdp.dequant_expert(weights, expert, dtype=parts[0][1].dtype)
        ex = {kk: v.to(device) for kk, v in ex.items()}
        dequants += 1
        for (x, latent), (topk_idx, topk_weight, touched, y_lat) in zip(parts, routed):
            if expert not in touched:
                continue
            hit = (topk_idx == expert)
            rows = hit.any(dim=1).nonzero().flatten()
            weight = topk_weight[rows].masked_fill(~hit[rows], 0.0).sum(dim=1)
            chunk = latent[rows]
            activated = kdp.situ(F.linear(chunk, ex["w1"]), F.linear(chunk, ex["w3"]), beta, linear_beta)
            y_lat[rows] += F.linear(activated, ex["w2"]).float() * weight.unsqueeze(-1)
            del chunk, activated
        del ex
    return [r[3] for r in routed], dequants


def profile_call(torch, fn, label_prefix: str = "") -> dict[str, Any]:
    """Wall vs device-kernel time of one call (torch.profiler, CPU + CUDA activity)."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    kernels = 0
    kernel_us = 0.0
    memcpy_us = 0.0
    by_name: dict[str, float] = defaultdict(float)
    for evt in prof.events():
        dev_type = str(getattr(evt, "device_type", ""))
        if "CUDA" not in dev_type:
            continue
        dur = float(getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0.0) or 0.0)
        if dur <= 0:
            dur = float(getattr(evt, "time_range").elapsed_us()) if hasattr(evt, "time_range") else 0.0
        name = evt.name or ""
        if "memcpy" in name.lower() or "memset" in name.lower():
            memcpy_us += dur
        else:
            kernels += 1
            kernel_us += dur
        by_name[name[:60]] += dur
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:12]
    return {"label": label_prefix, "wall_s": round(wall, 4), "device_kernel_s": round(kernel_us / 1e6, 4),
            "device_memcpy_s": round(memcpy_us / 1e6, 4), "kernels": kernels,
            "device_busy_fraction": round((kernel_us + memcpy_us) / 1e6 / max(wall, 1e-9), 4),
            "mean_kernel_us": round(kernel_us / max(kernels, 1), 2),
            "top_kernels_s": [[n, round(v / 1e6, 4)] for n, v in top],
            "note": "wall is under the profiler (inflated host side); device times are CUPTI kernel durations"}


def kda_graph_replay(torch, rec_fn, inputs: dict[str, Any], scale: float, eager_out, eager_state) -> dict[str, Any]:
    """Capture the engine's naive_recurrent_kda on static copies of one row's inputs as a CUDA graph, replay it, and
    compare BITWISE with the eager result.  Measures capture, replay, and the per-row input copy."""
    out: dict[str, Any] = {}
    static = {k: v.clone() for k, v in inputs.items()}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # warm-up outside capture (allocator, cuBLAS handle on this stream)
        rec_fn(static["q"], static["k"], static["v"], static["g"], static["beta"], scale)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    t0 = time.perf_counter()
    with torch.cuda.graph(graph):
        g_out, g_state = rec_fn(static["q"], static["k"], static["v"], static["g"], static["beta"], scale)
    torch.cuda.synchronize()
    out["capture_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    for name, value in inputs.items():
        static[name].copy_(value)
    torch.cuda.synchronize()
    out["input_copy_s"] = round(time.perf_counter() - t0, 4)
    walls = []
    for _ in range(3):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t0)
    out["replay_s"] = [round(w, 4) for w in walls]
    out["bitwise_equal_out"] = bool(torch.equal(g_out, eager_out))
    out["bitwise_equal_state"] = bool(torch.equal(g_state, eager_state))
    out["max_abs_diff_out"] = float((g_out.float() - eager_out.float()).abs().max())
    del graph, g_out, g_state, static
    torch.cuda.empty_cache()
    return out


def h2d_bench(torch, device, nbytes: int) -> dict[str, Any]:
    """Pageable vs pinned host->device copy of one AttnRes block-slice-sized buffer, and a strided (non-contiguous)
    host slice of the same size, which torch first gathers on the host."""
    n = nbytes // 2
    host = torch.empty(n, dtype=torch.bfloat16).fill_(1.0)
    res: dict[str, Any] = {"bytes": nbytes}
    for name, src in (("pageable", host), ("pinned", host.pin_memory())):
        dev = torch.empty(n, dtype=torch.bfloat16, device=device)
        dev.copy_(src)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            dev.copy_(src, non_blocking=(name == "pinned"))
        torch.cuda.synchronize()
        res[f"{name}_GB_per_s"] = round(3 * nbytes / 1e9 / (time.perf_counter() - t0), 2)
        del dev
    hidden = 7168
    rows = max(1, n // (hidden * 8))
    big = torch.empty(rows, 9, hidden, dtype=torch.bfloat16).fill_(1.0)
    strided = big[:, :8]
    t0 = time.perf_counter()
    for _ in range(3):
        _ = strided.to(device)
    torch.cuda.synchronize()
    res["noncontig_to_device_GB_per_s"] = round(3 * strided.numel() * 2 / 1e9 / (time.perf_counter() - t0), 2)
    t0 = time.perf_counter()
    for _ in range(3):
        _ = strided.contiguous()
    res["noncontig_host_gather_GB_per_s"] = round(3 * strided.numel() * 2 / 1e9 / (time.perf_counter() - t0), 2)
    return res


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------
def cmd_split(args, tool_argv: list[str]) -> int:
    import torch
    import torch.nn.functional as F

    tool, tool_sha = load_frozen_tool(args.tool, args.tool_sha)
    targs = tool.build_parser().parse_args(tool_argv)
    if targs.command != "run" or not targs.probe:
        raise SystemExit("split profiles a `run --probe` window only (never a cache pass)")
    pass_obj = tool.CachePass(targs)
    kdp = pass_obj.kdp
    import k3_layer  # the lane module the engine bound (already on sys.path)

    cuda = pass_obj.cuda
    st = Stages(sync=args.sync, synchronize=(lambda: torch.cuda.synchronize(pass_obj.device)) if cuda else None)
    experiments: dict[str, Any] = {"kda": [], "moe": [], "mla_sdpa": []}
    state = {"kda_done": set(), "moe_done": set()}

    # -- read_layer: file read + H2D inside it
    orig_read_tensors = kdp.read_tensors
    kdp.read_tensors = st.wrap("read.file", orig_read_tensors)
    orig_to = k3_layer.LayerWeights.to
    k3_layer.LayerWeights.to = st.wrap("read.h2d", orig_to)
    kdp.read_layer = st.wrap("read", kdp.read_layer)

    # -- attention half (timed copy), its core, the KDA recurrence and SDPA
    kdp.attention_half = st.wrap("attn", make_attention_half_timed(kdp, st, torch))
    k3_layer.StreamedLayer.attention = st.wrap("attn.core.dispatch", k3_layer.StreamedLayer.attention)
    orig_rec = k3_layer.naive_recurrent_kda

    def rec_timed(q, k, v, g, beta, scale, initial_state=None):
        with st.stage("attn.core.dispatch.kda_recurrence"):
            out, s = orig_rec(q, k, v, g, beta, scale, initial_state=initial_state)
        layer = st.layer
        if args.experiments and cuda and layer in args.kda_exp_layers and layer not in state["kda_done"]                 and initial_state is None:
            state["kda_done"].add(layer)
            with st.replay():
                experiments["kda"].append(kda_experiments(torch, orig_rec, layer, q, k, v, g, beta, scale, out, s))
        return out, s

    k3_layer.naive_recurrent_kda = rec_timed
    orig_sdpa = F.scaled_dot_product_attention
    k3_layer.F.scaled_dot_product_attention = st.wrap("attn.core.dispatch.sdpa", orig_sdpa)

    # -- FFN half pieces
    kdp._chunked_linear = st.wrap("ffn.latent_proj", kdp._chunked_linear)
    kdp.moe_write = st.wrap("ffn.moe_write_shared", kdp.moe_write)
    orig_routed = kdp.routed_y_lat

    def moe_hook(weights, x, latent, cfg, device, y_lat):
        layer = st.layer
        if not args.experiments or not cuda or layer not in args.moe_exp_layers or layer in state["moe_done"]:
            return
        state["moe_done"].add(layer)
        with st.replay():
            experiments["moe"].append(moe_experiments(torch, F, kdp, orig_routed, layer, weights, x, latent, cfg,
                                                       device, y_lat, args))

    routed_timed = make_routed_y_lat_timed(kdp, st, torch, F, hook=moe_hook)
    kdp.routed_y_lat = lambda *a, **kw: _stage_call(st, "ffn.moe", routed_timed, *a, **kw)
    kdp.ffn_half = st.wrap("ffn", kdp.ffn_half)
    kdp.require_finite = st.wrap("finite_check", kdp.require_finite)

    # -- the layer: everything the tool does per layer (incl. D2H of the new prefix and the block push)
    orig_layer = tool.CachePass.layer

    def layer_timed(self, run, layer_idx, prefix, blocks, nb, plan_shards):
        st.layer = layer_idx
        t0 = time.perf_counter()
        with st.stage("layer_total"):
            nb_out = orig_layer(self, run, layer_idx, prefix, blocks, nb, plan_shards)
        log(f"L{layer_idx:02d} layer_total {time.perf_counter() - t0:.1f}s (replays excluded from stages)")
        write_json(args.out + ".partial", {"stages": st.table(), "experiments": experiments,
                                           "layers_receipt": self.receipt["layers"]})
        return nb_out

    tool.CachePass.layer = layer_timed

    started = time.time()
    receipt = pass_obj.run()
    wall = time.time() - started
    if args.experiments and cuda:
        experiments["h2d"] = h2d_bench(torch, pass_obj.device, args.h2d_bench_mb << 20)
        experiments["checkpoint"] = checkpoint_bench(pass_obj, args)
    tables = st.table()
    out = {"tool": TOOL_NAME, "tool_version": TOOL_VERSION, "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "profiled_tool": {"path": os.path.abspath(args.tool), "sha256": tool_sha,
                             "version": tool.TOOL_VERSION, "engine": receipt.get("engine")},
           "argv": tool_argv, "sync": args.sync, "wall_s": round(wall, 1),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "device": str(pass_obj.device), "device_name": torch.cuda.get_device_name(pass_obj.device) if cuda else None,
           "num_threads": torch.get_num_threads(),
           "corpus": {k: receipt["corpus"][k] for k in ("positions", "real_positions", "groups", "group_tokens")},
           "layers_receipt": receipt["layers"], "stages": tables,
           "stages_exclusive_wall": {layer: exclusive(rec) for layer, rec in tables.items()},
           "experiments": experiments, "probe_receipt_dir": pass_obj.pass_dir}
    digest = write_json(args.out, out)
    log(f"split -> {args.out} (sha256 {digest[:12]})")
    return 0


def _stage_call(st: Stages, name: str, fn, *a, **kw):
    with st.stage(name):
        return fn(*a, **kw)


def kda_experiments(torch, rec_fn, layer, q, k, v, g, beta, scale, eager_out, eager_state) -> dict[str, Any]:
    res: dict[str, Any] = {"layer": layer, "shape_q": list(q.shape)}
    try:
        res["determinism_eager_rerun_equal"] = bool(torch.equal(rec_fn(q, k, v, g, beta, scale)[0], eager_out))
        res["profile"] = profile_call(torch, lambda: rec_fn(q, k, v, g, beta, scale), "kda_recurrence_one_row")
        # rows batched (B=2, the same row twice): is the RECURRENCE alone bitwise row-independent?
        out2, _ = rec_fn(torch.cat([q, q]), torch.cat([k, k]), torch.cat([v, v]), torch.cat([g, g]),
                         torch.cat([beta, beta]), scale)
        res["batched_b2_recurrence_bitwise_equal"] = bool(torch.equal(out2[:1], eager_out) and
                                                          torch.equal(out2[1:], eager_out))
        del out2
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        need = 3 * sum(t.numel() * t.element_size() for t in (q, k, v, g, beta))
        if free < need:  # a failed allocation inside a capture can leave the stream unusable: never risk the probe
            res["graph"] = {"skipped": f"free {free / GIB:.1f} GiB < {need / GIB:.1f} GiB"}
            return res
        res["graph"] = kda_graph_replay(torch, rec_fn, {"q": q, "k": k, "v": v, "g": g, "beta": beta}, scale,
                                        eager_out, eager_state)
    except Exception as exc:  # measurement only: recorded, never fails the probe
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    torch.cuda.empty_cache()
    return res


def moe_experiments(torch, F, kdp, orig_routed, layer, weights, x, latent, cfg, device, y_lat, args) -> dict[str, Any]:
    res: dict[str, Any] = {"layer": layer, "positions": int(x.shape[0])}
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_ref, _, dq = orig_routed(weights, x, latent, cfg, device)
        torch.cuda.synchronize()
        res["engine_replay_s"] = round(time.perf_counter() - t0, 3)
        res["engine_replay_dequant_launch_s"] = round(dq, 3)
        res["determinism_engine_rerun_equal"] = bool(torch.equal(y_ref, y_lat))
        del y_ref
        res["profile_engine"] = profile_call(torch, lambda: orig_routed(weights, x, latent, cfg, device),
                                             "routed_y_lat_engine")
        # dequant alone: every touched expert once (what one group pays)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for expert in range(int(cfg["num_experts"])):
            ex = kdp.dequant_expert(weights, expert, dtype=latent.dtype)
            del ex
        torch.cuda.synchronize()
        res["dequant_all_experts_once_s"] = round(time.perf_counter() - t0, 3)
        def dequant_each() -> None:  # one expert resident at a time, as in the engine
            for e in range(int(cfg["num_experts"])):
                ex = kdp.dequant_expert(weights, e, dtype=latent.dtype)
                del ex

        res["profile_dequant_all"] = profile_call(torch, dequant_each, "dequant_all")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_sorted = moe_sorted_dispatch(kdp, torch, F, weights, x, latent, cfg, device)
        torch.cuda.synchronize()
        res["sorted_dispatch_s"] = round(time.perf_counter() - t0, 3)
        res["sorted_dispatch_bitwise_equal"] = bool(torch.equal(y_sorted, y_lat))
        del y_sorted
        # two pseudo-groups (halves): engine per group vs expert-outer with one dequant per expert
        half = x.shape[0] // 2
        parts = [(x[:half], latent[:half]), (x[half:], latent[half:])]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        refs = [orig_routed(weights, px, pl, cfg, device)[0] for px, pl in parts]
        torch.cuda.synchronize()
        res["two_groups_engine_s"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter()
        outs, dequants = moe_expert_outer(kdp, torch, F, weights, parts, cfg, device)
        torch.cuda.synchronize()
        res["two_groups_expert_outer_s"] = round(time.perf_counter() - t0, 3)
        res["two_groups_expert_outer_dequants"] = dequants
        res["two_groups_expert_outer_bitwise_equal"] = bool(all(torch.equal(a, b) for a, b in zip(outs, refs)))
        del refs, outs
    except Exception as exc:  # measurement only: recorded, never fails the probe
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    torch.cuda.empty_cache()
    return res


def checkpoint_bench(pass_obj, args) -> dict[str, Any]:
    """What one production checkpoint of the prefix costs: the tool's own save (safetensors write + sha256 re-read)
    of this probe's prefix-sized tensor into the scratch dir, then removed."""
    import torch

    hidden = int(pass_obj.index.hidden_size)
    positions = int(pass_obj.corpus.tokens)
    tensor = torch.zeros(positions, hidden, dtype=torch.bfloat16)
    path = os.path.join(args.bench_dir, f"ckpt_bench_{os.getpid()}.safetensors")
    os.makedirs(args.bench_dir, exist_ok=True)
    t0 = time.perf_counter()
    pass_obj.common.save_tensors(path, {"prefix": tensor}, metadata={"tool": TOOL_NAME})
    wall = time.perf_counter() - t0
    size = os.path.getsize(path)
    os.remove(path)
    return {"positions": positions, "bytes": size, "save_plus_sha_s": round(wall, 2),
            "MB_per_s": round(size / 1e6 / max(wall, 1e-9), 1), "path": path}


# --------------------------------------------------------------------------
# monitor
# --------------------------------------------------------------------------
def mem_available() -> int:
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def descendants(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        cur = todo.pop()
        out.append(cur)
        try:
            with open(f"/proc/{cur}/task/{cur}/children", "r", encoding="utf-8") as handle:
                todo += [int(c) for c in handle.read().split()]
        except OSError:
            pass
    return out


def proc_comm(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/comm", "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def thread_ticks(pid: int) -> dict[int, tuple[str, int]]:
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        try:
            with open(f"/proc/{pid}/task/{tid}/stat", "r", encoding="utf-8") as handle:
                raw = handle.read()
            comm = raw[raw.index("(") + 1: raw.rindex(")")]
            fields = raw[raw.rindex(")") + 2:].split()
            out[int(tid)] = (comm, int(fields[11]) + int(fields[12]))  # utime + stime
        except (OSError, ValueError, IndexError):
            pass
    return out


def io_read_bytes(pids: list[int]) -> int:
    total = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/io", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("read_bytes:"):
                        total += int(line.split()[1])
        except OSError:
            pass
    return total


def cmd_monitor(args, command: list[str]) -> int:
    hz = os.sysconf("SC_CLK_TCK")
    floor = int(args.mem_floor_gib * GIB)
    proc = subprocess.Popen(command, start_new_session=True)
    samples = open(args.out, "w", encoding="utf-8")
    prev_t, prev_ticks, prev_io = time.time(), {}, 0
    thread_total: dict[int, list] = {}
    killed = None
    while proc.poll() is None:
        time.sleep(args.interval)
        now = time.time()
        tree = descendants(proc.pid)
        pythons = [p for p in tree if proc_comm(p).startswith("python")]
        target = pythons[-1] if pythons else proc.pid
        ticks = thread_ticks(target)
        dt = max(now - prev_t, 1e-6)
        per_thread = {tid: round((tk - prev_ticks.get(tid, (c, tk))[1]) / hz / dt * 100.0, 1)
                      for tid, (c, tk) in ticks.items()}
        for tid, (c, tk) in ticks.items():
            thread_total[tid] = [c, tk]
        io = io_read_bytes(tree)
        avail = mem_available()
        rec = {"t": round(now, 1), "target_pid": target, "cpu_pct_total": round(sum(per_thread.values()), 1),
               "top_threads": sorted(([tid, ticks[tid][0], pct] for tid, pct in per_thread.items() if pct > 0),
                                     key=lambda r: -r[2])[:8],
               "read_MB_per_s": round((io - prev_io) / 1e6 / dt, 1) if prev_io else None,
               "mem_available_gib": round(avail / GIB, 1)}
        samples.write(json.dumps(rec) + "\n")
        samples.flush()
        prev_t, prev_ticks, prev_io = now, ticks, io
        if avail < floor and killed is None:
            killed = f"MemAvailable {avail / GIB:.1f} GiB < floor {args.mem_floor_gib} GiB"
            log(f"STOP: {killed}; SIGTERM to the tree")
            os.killpg(proc.pid, signal.SIGTERM)
    samples.close()
    totals = sorted(([tid, c, round(tk / hz, 1)] for tid, (c, tk) in thread_total.items()), key=lambda r: -r[2])
    write_json(args.out + ".summary.json", {"rc": proc.returncode, "killed": killed,
                                            "thread_cpu_seconds_top": totals[:16], "threads": len(totals)})
    log(f"monitor: rc={proc.returncode} killed={killed}")
    return 75 if killed else proc.returncode


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
def cmd_compare(args, _rest) -> int:
    with open(args.a, "r", encoding="utf-8") as handle:
        a = json.load(handle)
    with open(args.b, "r", encoding="utf-8") as handle:
        b = json.load(handle)
    la, lb = a.get("layers") or a.get("layers_receipt"), b.get("layers") or b.get("layers_receipt")
    rows, ok = [], True
    for key in sorted(set(la) | set(lb), key=int):
        sa, sb = (la.get(key) or {}).get("prefix_out_sha256"), (lb.get(key) or {}).get("prefix_out_sha256")
        equal = bool(sa) and sa == sb
        ok &= equal
        rows.append({"layer": int(key), "a": sa, "b": sb, "equal": equal})
        print(f"L{int(key):02d} {'EQUAL' if equal else 'DIFFER'} {str(sa)[:16]} {str(sb)[:16]}")
    print(f"PREFIX DIGESTS {'ALL EQUAL' if ok else 'DIFFER'} over {len(rows)} layers")
    if args.out:
        write_json(args.out, {"a": args.a, "b": args.b, "layers": rows, "all_equal": ok})
    return 0 if ok else 3


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("split", help="stage timers on a `run --probe` window of the frozen tool; tool args after --")
    s.add_argument("--tool", required=True, help="a COPY of the frozen k3_teacher_cache")
    s.add_argument("--tool-sha", default=FROZEN_K3_SHA256)
    s.add_argument("--out", required=True)
    s.add_argument("--sync", dest="sync", action="store_true", default=True)
    s.add_argument("--no-sync", dest="sync", action="store_false")
    s.add_argument("--experiments", action="store_true")
    s.add_argument("--kda-exp-layers", type=lambda v: {int(x) for x in v.split(",") if x}, default=set(),
                   help="layers whose first KDA row gets the replays (profile, batched, CUDA graph)")
    s.add_argument("--moe-exp-layers", type=lambda v: {int(x) for x in v.split(",") if x}, default=set(),
                   help="layers whose routed MoE gets the replays (profile, dequant, sorted, expert-outer)")
    s.add_argument("--h2d-bench-mb", type=int, default=448)
    s.add_argument("--bench-dir", default="/mnt/lexar/omni-campaign/scratch/k3accel/bench")
    s.set_defaults(func=cmd_split)
    m = sub.add_parser("monitor", help="run a command (after --) and sample per-thread CPU, read MB/s, MemAvailable")
    m.add_argument("--out", required=True)
    m.add_argument("--interval", type=float, default=5.0)
    m.add_argument("--mem-floor-gib", type=float, default=40.0)
    m.set_defaults(func=cmd_monitor)
    c = sub.add_parser("compare", help="prefix digests of two probe receipts / split outputs")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--out", default=None)
    c.set_defaults(func=cmd_compare)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, rest = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)
    return int(args.func(args, rest) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
