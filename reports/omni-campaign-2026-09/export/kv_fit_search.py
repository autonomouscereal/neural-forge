#!/usr/bin/env python3
"""How much context actually fits: the largest -c a (model, KV type, card) triple loads at.

Why this exists. `envelope_bench.py --kv q8_0 --ctx 262144` on a 24 GB card answered the
question it was asked -- it does not fit -- and the log said exactly where it died:

    E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 1360.28 MiB on device 0:
      cudaMalloc failed: out of memory
    E graph_reserve: failed to allocate compute buffers

"Does not fit at 256k" is a measurement. "Fits up to N" is the number a plan can be built
on, and nothing in the lane produced it. This script bisects N by LOADING the server, which
is the only honest way to get it: the failure above is not a KV-size calculation, it is the
compute buffers that the KV type and context together imply, and no arithmetic on this box
has predicted those correctly yet.

Every attempt is recorded -- the context tried, whether the server became healthy, the wall
clock, the per-PID peak VRAM this run itself caused, and the verbatim CUDA lines when it
failed. There is no pass/fail anywhere: the output is "largest context observed to load"
and "smallest context observed to fail", with the gap between them left visible rather than
interpolated.

Process safety: host runner only; it stops only the server it started itself.

  python3 kv_fit_search.py --model /home/cereal/models/gguf/Qwen3.6-27B-Q4_K_M.gguf \
      --kv q8_0 --gpu-uuid GPU-08b615f9-... --gpu-index 3 --tag Q36-27B-q8kv-p40 \
      --lo 8192 --hi 262144 --port 18096
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    GpuSampler,
    LlamaServer,
    analyse_server_log,
    build_identity,
    host_facts,
    props_n_ctx,
    safe_tag,
    sha256_file,
    utc_now,
    write_receipt,
)
from envelope_bench import (  # noqa: E402
    DEFAULT_LLAMACPP_ROOT,
    KV_TYPES,
    ServerHandle,
    launch_host,
    wait_healthy_watching_child,
)

# llama.cpp allocates the KV cache in blocks; searching at finer granularity than this
# reports precision the allocator does not have.
DEFAULT_GRANULARITY = 4096

_FAILURE_MARKERS = (
    "cudaMalloc failed",
    "failed to allocate",
    "out of memory",
    "failed to load model",
    "failed to create context",
    "compute buffers",
)


def round_down(value, granularity):
    """Largest multiple of `granularity` not above `value` (never below one block)."""
    return max(granularity, (int(value) // granularity) * granularity)


def failure_lines(log_text, limit=12):
    """The verbatim allocator lines, which are the whole point of a failed attempt."""
    found = []
    for line in log_text.splitlines():
        if any(marker in line for marker in _FAILURE_MARKERS):
            found.append(line.strip())
            if len(found) >= limit:
                break
    return found


def next_context_to_try(largest_fit, smallest_fail, lo, hi, granularity):
    """The next context to test, or None when the search is finished.

    Kept separate from the launching so the bisect logic can be tested without a GPU.
    """
    if smallest_fail is None:
        return None                       # the ceiling itself loaded; nothing to bisect
    if largest_fit is None:
        candidate = round_down(lo, granularity)
        return candidate if candidate < smallest_fail else None
    if smallest_fail - largest_fit <= granularity:
        return None                       # boundary located to the allocator's resolution
    candidate = round_down((largest_fit + smallest_fail) // 2, granularity)
    if candidate <= largest_fit or candidate >= smallest_fail:
        return None
    return candidate


def summarise(largest_fit, smallest_fail, kv, granularity):
    if largest_fit is not None and smallest_fail is not None:
        return ("at %s KV on this card, -c %d loaded and -c %d did not; the boundary lies "
                "between them and was searched to %d-token resolution"
                % (kv, largest_fit, smallest_fail, granularity))
    if smallest_fail is None:
        return "every context tried loaded, up to -c %s" % largest_fit
    return ("nothing in the searched range loaded; the smallest context that failed was "
            "-c %s" % smallest_fail)


def try_context(args, ctx, sampler, out_dir):
    """Launch the server at -c ctx and report whether it came up. Always tears down."""
    handle = ServerHandle()
    attempt_args = types.SimpleNamespace(
        llamacpp_root=args.llamacpp_root,
        model=args.model,
        ctx=int(ctx),
        kv=args.kv,
        port=args.port,
        fa=args.fa,
        extra_args=args.extra_args,
        bind=args.bind,
        gpu_uuid=args.gpu_uuid,
        gpu_index=args.gpu_index,
        tag="%s-c%d" % (args.tag, int(ctx)),
    )
    started = time.time()
    record = {"ctx": int(ctx), "started_at": utc_now(),
              "load_timeout_s": int(args.load_timeout)}
    try:
        launch_host(handle, attempt_args, out_dir)
        record["command"] = handle.command
        if handle.process is not None:
            sampler.watch_pid(handle.process.pid)
            record["pid"] = handle.process.pid
        server = LlamaServer("%s:%d" % (args.host_url, args.port), timeout=60.0)
        health = wait_healthy_watching_child(server, handle, args.load_timeout, poll_s=1.0)
        record["healthy"] = bool(health.get("healthy"))
        record["wait_s"] = health.get("wait_s")
        record["child_exit_code"] = health.get("child_exit_code")
        # Measured 2026-09-02: the -c 262144 attempt waited 1,709 s against a requested
        # --load-timeout of 420 while another lane's 128k prefill had the host's CPUs. The
        # poll loop is wall-clock bounded, so the overrun is real and worth seeing in the
        # receipt rather than reconstructing it later from timestamps.
        record["wait_exceeded_load_timeout"] = bool(
            record["wait_s"] is not None and record["wait_s"] > args.load_timeout * 1.1
        )
        log_text = handle.logs(4000)
        analysis = analyse_server_log(log_text)
        record["listening_at_log_s"] = analysis.get("listening_at_log_s")
        record["fallback_evidence"] = analysis.get("fallback_evidence")
        record["failure_lines"] = failure_lines(log_text)
        if record["healthy"]:
            record["props_n_ctx"] = props_n_ctx(server.props())
        record["log_path"] = handle.log_path
    except Exception as exc:  # noqa: BLE001 - a crash at one context is a measurement
        record["error"] = repr(exc)
        record["healthy"] = False
    finally:
        record["teardown"] = handle.stop()
        record["wall_s"] = round(time.time() - started, 2)
        # Per-attempt footprint. The run-level own_peak sums over every tracked PID, which
        # for a bisect means eight sequential servers added together -- a number that is
        # not any one server's footprint. This is.
        pid = record.get("pid")
        slot = None
        if pid is not None:
            slot = (sampler.peak_by_pid.get(pid)
                    or sampler.peak_by_pid.get(str(pid))
                    or sampler.peak_by_pid.get(int(pid)))
        record["own_peak_memory_used_mib"] = (
            slot.get("peak_memory_used_mib") if isinstance(slot, dict) else None
        )
        # The card does not free instantly; without a settle the next attempt can inherit
        # this one's allocation and report a false OOM.
        time.sleep(args.settle_s)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kv", default="q8_0", choices=KV_TYPES)
    parser.add_argument("--fa", default="on", choices=("on", "off", "auto"))
    parser.add_argument("--lo", type=int, default=8192, help="smallest context worth trying")
    parser.add_argument("--hi", type=int, default=262144, help="context tried first")
    parser.add_argument("--granularity", type=int, default=DEFAULT_GRANULARITY)
    parser.add_argument("--max-attempts", type=int, default=9)
    parser.add_argument("--port", type=int, default=18096)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--host-url", default="http://127.0.0.1")
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--llamacpp-root", default=DEFAULT_LLAMACPP_ROOT)
    parser.add_argument("--out-dir", default="/home/cereal/omni-campaign/export/envelope")
    parser.add_argument("--load-timeout", type=int, default=600)
    parser.add_argument("--settle-s", type=float, default=8.0)
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--extra-args", default=None)
    args = parser.parse_args(argv)
    args.tag = safe_tag(args.tag)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.lo >= args.hi:
        raise SystemExit("--lo (%d) must be below --hi (%d)" % (args.lo, args.hi))

    started_all = time.time()
    sampler = GpuSampler(interval=args.sample_interval,
                         indices=[args.gpu_index] if args.gpu_index is not None else None)
    sampler.start()

    receipt = {
        "kind": "kv_fit_search",
        "created_at": utc_now(),
        "tag": args.tag,
        "host": host_facts(),
        "build_identity": build_identity(llamacpp_root=args.llamacpp_root,
                                         source="own_build_receipt"),
        "config": {
            "model": args.model,
            "kv_type_k": args.kv,
            "kv_type_v": args.kv,
            "flash_attn_requested": args.fa,
            "n_parallel": 1,
            "n_gpu_layers": 999,
            "gpu_uuid": args.gpu_uuid,
            "gpu_index": args.gpu_index,
            "granularity": args.granularity,
            "search_lo": args.lo,
            "search_hi": args.hi,
            "load_timeout_s": args.load_timeout,
            "settle_s": args.settle_s,
        },
        "model_sha256": sha256_file(args.model) if os.path.exists(args.model) else None,
        "attempts": [],
        "note": (
            "Every row is a real server launch. largest_fitting_ctx is the largest context "
            "OBSERVED to load; smallest_failing_ctx is the smallest OBSERVED to fail. "
            "Nothing between them was tried, and nothing here is a gate."
        ),
    }

    largest_fit = None
    smallest_fail = None
    lo = round_down(args.lo, args.granularity)
    hi = round_down(args.hi, args.granularity)

    ctx = hi                              # probe the ceiling first
    attempts = 0
    while ctx is not None and attempts < args.max_attempts:
        attempts += 1
        print("attempt %d: -c %d -ctk %s -ctv %s ..." % (attempts, ctx, args.kv, args.kv))
        record = try_context(args, ctx, sampler, args.out_dir)
        receipt["attempts"].append(record)
        write_receipt(os.path.join(args.out_dir, "kvfit-%s.partial.json" % args.tag), receipt)
        if record.get("healthy"):
            largest_fit = ctx if largest_fit is None else max(largest_fit, ctx)
            print("  LOADED at %d (healthy after %ss)" % (ctx, record.get("wait_s")))
        else:
            smallest_fail = ctx if smallest_fail is None else min(smallest_fail, ctx)
            first = (record.get("failure_lines") or ["(no allocator line in the log)"])[0]
            print("  did not load at %d: %s" % (ctx, first[:150]))
        ctx = next_context_to_try(largest_fit, smallest_fail, lo, hi, args.granularity)

    receipt["largest_fitting_ctx"] = largest_fit
    receipt["smallest_failing_ctx"] = smallest_fail
    receipt["attempts_made"] = attempts
    receipt["attempt_budget"] = args.max_attempts
    receipt["search_exhausted_budget"] = bool(ctx is not None and attempts >= args.max_attempts)
    receipt["resolution_tokens"] = args.granularity
    receipt["reading"] = summarise(largest_fit, smallest_fail, args.kv, args.granularity)
    receipt["gpu_sampling"] = sampler.stop()
    receipt["gpu_sampling"]["own_peak_caveat"] = (
        "own_peak_memory_used_mib here is the SUM over every server this search started, "
        "one after another; it is not a footprint. Read attempts[].own_peak_memory_used_mib "
        "for the per-attempt number and peak_by_index for the whole card."
    )
    receipt["peak_by_attempt_mib"] = {
        str(item["ctx"]): item.get("own_peak_memory_used_mib") for item in receipt["attempts"]
    }
    receipt["total_wall_s"] = round(time.time() - started_all, 2)

    path = os.path.join(args.out_dir, "kvfit-%s.json" % args.tag)
    digest = write_receipt(path, receipt)
    partial = os.path.join(args.out_dir, "kvfit-%s.partial.json" % args.tag)
    if os.path.exists(partial):
        os.remove(partial)
    print("\n%s" % receipt["reading"])
    print("kv fit receipt: %s sha256=%s" % (path, digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
