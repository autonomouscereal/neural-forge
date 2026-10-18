#!/usr/bin/env python3
"""d1_dp_bench.py -- time one D1 data-parallel training step: the 1.2.3 path vs the d1_trunk_recovery path.

Lane d1, Claude, 2026-10-13.  The review of d1_trunk_recovery 1.2.3 found that its host-reduced data-parallel
step walked the cards one after another (eager gathers on the main thread, pageable blocking uploads), so two
cards gave ~1.33x one card.  1.2.4 runs every CUDA card's shard on its own thread with lazy gathers and pinned
non_blocking uploads.  This bench runs both paths on synthetic tensors at the real D1 shapes (hidden 5120, trunk
4352, 32,768 tokens per step, 8,192 per micro-batch by default) on the given --devices and reports:

    seconds per step (median of --steps after --warmup) for each path
    parity: the host-reduced gradients and the summed loss of the two paths (same devices, same micro-batches)
            -- torch.equal and max |diff|; the per-device order is the same, so they should be bit-equal

Nothing is trained or written except the --out JSON.  Synthetic numbers only: no capability claim.

    d1_dp_bench.py --devices cuda:0 --out bench_1card.json
    d1_dp_bench.py --devices cuda:0,cuda:1 --out bench_2card.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_trunk_recovery as d1  # noqa: E402

TOOL_NAME = "d1_dp_bench"
TOOL_VERSION = "1.0.0"   # 1.0.0 (2026-10-13): first version, lane d1 review fix (1.2.3 vs 1.2.4 DP step)


def legacy_shard_grads(trainer: Any, shards: list[list[tuple[Any, Any]]], n_total: int) -> tuple[dict[str, Any], float]:
    """The 1.2.3 step, kept verbatim in behaviour: cards one after another, pageable blocking uploads."""
    import torch

    per_dev, losses = [], []
    for d, (dev, micro) in enumerate(zip(trainer.devices, shards)):
        acc = None
        for xm, y in micro:
            xd = xm.to(dev, non_blocking=False)
            yd = y.to(dev, non_blocking=False)
            loss, g, _st = d1.d1_loss_and_grads(xd, yd, trainer.replicas[d], trainer.norm_eps, 1.0 / n_total,
                                                residual=trainer.residual, mm_dtype=trainer.mm_dtype)
            losses.append(loss)
            if acc is None:
                acc = g
            else:
                for k in d1.PARAM_NAMES:
                    acc[k] += g[k]
        per_dev.append(acc)
    host = {}
    for k in d1.PARAM_NAMES:
        parts = [g[k].detach().to("cpu", torch.float64 if g[k].dtype == torch.float64 else torch.float32) for g in per_dev]
        total = parts[0].clone()
        for extra in parts[1:]:
            total += extra
        host[k] = total
    return host, float(sum(float(l) for l in losses))


def sync_all(devices: list[str]) -> None:
    import torch

    for d in devices:
        if d.startswith("cuda"):
            torch.cuda.synchronize(torch.device(d))


def run(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    if len(set(devices)) != len(devices) and not all(d == "cpu" for d in devices):
        raise d1.Refusal(f"--devices repeats a card: {devices}")
    h, q, per_step, micro = int(args.hidden), int(args.trunk), int(args.tokens_per_step), int(args.micro_tokens)
    buf_dtype = getattr(torch, args.buffer_dtype)
    g = torch.Generator().manual_seed(int(args.seed))
    pool = int(args.pool_steps) * per_step
    pin = any(d.startswith("cuda") for d in devices)
    C = torch.empty(pool, h, dtype=buf_dtype, pin_memory=pin)
    P = torch.empty(pool, h, dtype=buf_dtype, pin_memory=pin)
    for lo in range(0, pool, 8192):   # chunked fill: no whole-buffer fp32 temporary
        x = torch.randn(min(8192, pool - lo), h, generator=g) * 3.0
        C[lo:lo + x.shape[0]].copy_(x.to(buf_dtype))
        P[lo:lo + x.shape[0]].copy_((x + 0.4 * torch.randn(x.shape, generator=g)).to(buf_dtype))
    init = {"gate": torch.randn(q, h, generator=g) * 0.02, "up": torch.randn(q, h, generator=g) * 0.02,
            "down": torch.randn(h, q, generator=g) * 0.02, "norm": 1.0 + 0.01 * torch.randn(h, generator=g)}
    mm_dtype = None if args.train_dtype == "float32" else getattr(torch, args.train_dtype)
    kw = dict(lr=0.0, betas=(0.9, 0.95), eps=1e-8, clip=1.0, norm_eps=1e-6, residual=True, mm_dtype=mm_dtype,
              master_dtype=torch.float32)
    trainer = d1.TrunkTrainer(init, devices, parallel=None if args.threaded == "auto" else args.threaded == "on", **kw)
    D = len(devices)
    steps = int(args.warmup) + int(args.steps)
    orders = [torch.randperm(pool, generator=g)[:per_step] for _ in range(steps)]

    def bounds(n: int) -> list[int]:
        return [round(n * d / D) for d in range(D + 1)]

    def eager(idx: Any) -> list[list[tuple[Any, Any]]]:
        b = bounds(int(idx.numel()))
        return [[(C.index_select(0, part[m:m + micro]), P.index_select(0, part[m:m + micro]))
                 for m in range(0, int(part.numel()), micro)] for part in (idx[b[d]:b[d + 1]] for d in range(D))]

    def lazy(idx: Any) -> list[list[Any]]:   # exactly d1_trunk_recovery.train_layer's shards
        b = bounds(int(idx.numel()))
        return [[(lambda sel=part[m:m + micro]: (C.index_select(0, sel), P.index_select(0, sel)))
                 for m in range(0, int(part.numel()), micro)] for part in (idx[b[d]:b[d + 1]] for d in range(D))]

    times: dict[str, list[float]] = {"legacy_1_2_3": [], "current": []}
    parity = {"steps": 0, "all_equal": True, "max_abs_diff": 0.0, "loss_abs_diff": 0.0}
    for s, idx in enumerate(orders):
        sync_all(devices)
        t0 = time.perf_counter()
        h_old, l_old = legacy_shard_grads(trainer, eager(idx), per_step)   # gathers inside the timed step, as in 1.2.3
        sync_all(devices)
        t1 = time.perf_counter()
        h_new, l_new, _st = trainer.shard_grads(lazy(idx), per_step)
        sync_all(devices)
        t2 = time.perf_counter()
        if s >= int(args.warmup):
            times["legacy_1_2_3"].append(t1 - t0)
            times["current"].append(t2 - t1)
        parity["steps"] += 1
        for k in d1.PARAM_NAMES:
            parity["all_equal"] = parity["all_equal"] and bool(torch.equal(h_old[k], h_new[k]))
            parity["max_abs_diff"] = max(parity["max_abs_diff"], float((h_old[k] - h_new[k]).abs().max()))
        parity["loss_abs_diff"] = max(parity["loss_abs_diff"], abs(l_old - l_new))
        print(f"[{TOOL_NAME}] step {s + 1}/{steps}: legacy {t1 - t0:.3f}s current {t2 - t1:.3f}s "
              f"grads equal {parity['all_equal']}", flush=True)
    result = {
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION, "sha256": d1.sha256_file(__file__)},
        "d1_tool": {"version": d1.TOOL_VERSION, "sha256": d1.sha256_file(d1.__file__)},
        "torch": torch.__version__, "devices": devices, "threaded": trainer.parallel,
        "cards": {d: torch.cuda.get_device_name(torch.device(d)) for d in devices if d.startswith("cuda")},
        "shape": {"hidden": h, "trunk": q, "tokens_per_step": per_step, "micro_tokens": micro,
                  "buffer_dtype": args.buffer_dtype, "train_dtype": args.train_dtype, "pinned_pool": pin},
        "seconds_per_step_median": {k: statistics.median(v) for k, v in times.items() if v},
        "seconds_per_step": times, "parity": parity,
        "claim": "synthetic tensors; timing + host-reduce parity of the DP step only",
    }
    med = result["seconds_per_step_median"]
    if med.get("legacy_1_2_3") and med.get("current"):
        result["speedup_current_over_legacy"] = med["legacy_1_2_3"] / med["current"]
    if args.out:
        d1.write_json_atomic(Path(args.out), result, with_sha=False)
    print(f"[{TOOL_NAME}] {TOOL_VERSION} devices {devices}: median s/step legacy {med.get('legacy_1_2_3')} current "
          f"{med.get('current')}; grads bit-equal {parity['all_equal']} (max |diff| {parity['max_abs_diff']})", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--hidden", type=int, default=5120)
    ap.add_argument("--trunk", type=int, default=4352)
    ap.add_argument("--tokens-per-step", type=int, default=32768)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--buffer-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--train-dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--threaded", default="auto", choices=("auto", "on", "off"))
    ap.add_argument("--pool-steps", type=int, default=2, help="host pool = this many steps of tokens")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


if __name__ == "__main__":
    run()
    sys.exit(0)
