#!/usr/bin/env python3
"""The 256k envelope bench: does a Q4 child actually run at -c 262144 on a 24 GB card,
and what does it cost.

Protocol (all numbers are REPORTED MEASUREMENTS; there is no pass/fail anywhere in this
file, by campaign rule):

  1. start (or attach to) a llama-server at `-c <ctx> -fa on -np 1 -ctk <kv> -ctv <kv>`,
  2. read the server log and record every flash-attention / KV / offload signal, including
     any CPU-fallback or "not supported" line verbatim,
  3. sample nvidia-smi throughout and record PEAK VRAM per GPU index,
  4. prefill wall clock at each context point (default 4k / 32k / 128k / 256k), measured
     with cache_prompt=false so the whole prefix is really processed,
  5. decode tok/s at each context point (a second request over the same, now cached, prefix),
  6. needle-in-a-haystack at each needle point x each depth, code retrieval scored by exact
     containment, with the model's answer recorded verbatim.

Container safety: this script only ever stops a container that IT started, whose name it
generated. `--attach` mode touches no container at all.

Examples
--------
  # attach to the already-running P40 envelope server and bench it (starts/stops nothing)
  python3 envelope_bench.py --tag P40-Q36-27B-q4kv --attach http://127.0.0.1:18080 \
      --container-name p40_256k --gpu-index 1 --kv q4_0 --ctx 262144 \
      --prefill-points 4096,32768 --needle-points 32768 --needle-depths 0.1,0.9

  # launch our own server on GPU4 (RTX 3090) from the built binaries, q8_0 KV
  python3 envelope_bench.py --tag OC01-Q4KM-gpu4-q8kv \
      --model /home/cereal/omni-campaign/export/gguf/OC01.Q4_K_M.gguf \
      --runner host --gpu-uuid GPU-2ce6a95c-7b1f-644a-3808-d781732a0582 --gpu-index 4 \
      --kv q8_0 --port 18093

  # same, in a container image instead of the host binaries
  python3 envelope_bench.py ... --runner docker --image ghcr.io/ggml-org/llama.cpp:server-cuda
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    GpuSampler,
    LlamaServer,
    NEEDLE_PREDICT_TOKENS,
    analyse_server_log,
    build_identity,
    build_needle_prompt,
    calibrate_line_count,
    context_point_headroom,
    container_launch_command,
    docker_logs,
    flash_attn_evidence,
    host_facts,
    load_r3_control,
    parse_server_flags,
    props_n_ctx,
    parse_float_list,
    parse_int_list,
    run,
    safe_tag,
    score_needle,
    sha256_file,
    utc_now,
    write_receipt,
)

KV_TYPES = ("f16", "q8_0", "q4_0", "q5_0", "q5_1", "q4_1", "iq4_nl")
DEFAULT_LLAMACPP_ROOT = "/home/cereal/omni-campaign/export/llamacpp"
NEEDLE_CODES = ["ZQ7-4418", "MB3-9052", "TR8-1177", "KX2-6630", "VD5-3094",
                "PL9-7726", "HC4-2851", "NW6-5503", "JF1-8340", "RS0-4962"]


class ServerHandle(object):
    """Owns at most one llama-server we started ourselves."""

    def __init__(self):
        self.mode = None          # "attach" | "docker" | "host"
        self.container = None
        self.process = None
        self.log_path = None
        self.started_by_us = False
        self.command = None

    def logs(self, tail=4000, head=None):
        """head=N reads the LOAD banner (flash_attn / KV / offload print once at startup)."""
        if self.container and self.mode in ("docker", "attach"):
            return docker_logs(self.container, head=head, tail=None if head else tail)
        if self.log_path and os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()
            return "".join(lines[:head]) if head else "".join(lines[-tail:])
        return ""

    def stop(self):
        """Stop ONLY what we started."""
        if not self.started_by_us:
            return {"stopped": False, "reason": "not started by this process"}
        if self.mode == "docker" and self.container:
            rc, out, err = run(["docker", "rm", "-f", self.container], timeout=120)
            return {"stopped": rc == 0, "container": self.container, "rc": rc, "err": err[-200:]}
        if self.mode == "host" and self.process is not None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.process.kill()
            return {"stopped": True, "pid": self.process.pid}
        return {"stopped": False, "reason": "nothing to stop"}


def server_args(model_path, ctx, kv, port, fa, extra, bind="127.0.0.1"):
    """llama-server flags. `bind` defaults to loopback.

    This box also fronts Keycloak, mailcow and the SOC dashboard; llama-server has no auth
    and prints its own warning that CORS is open and no API key is set, so binding it to
    every interface for a multi-hour sweep is not something a bench tool should do by
    default. The docker runner publishes as `-p 127.0.0.1:<port>:<port>`, which keeps the
    HOST side on loopback while the server still binds 0.0.0.0 inside its own netns (a
    container-internal loopback bind would be unreachable through the port publish).
    """
    args = [
        "-m", model_path,
        "-c", str(ctx),
        "-fa", fa,
        "-np", "1",
        "-ngl", "999",
        "-ctk", kv,
        "-ctv", kv,
        "--host", bind,
        "--port", str(port),
    ]
    if extra:
        args.extend(extra.split())
    return args


def launch_docker(handle, args, out_dir):
    handle.mode = "docker"
    handle.container = "oc-env-%s-%d" % (args.tag.lower().replace("_", "-"), args.port)
    rc, out, _ = run(["docker", "ps", "-a", "--format", "{{.Names}}"], timeout=30)
    if handle.container in out.split():
        raise RuntimeError(
            "container %s already exists; refusing to touch a container we did not create"
            % handle.container
        )
    gpu_selector = "device=%s" % (args.gpu_uuid or args.gpu_index)
    cmd = [
        "docker", "run", "-d", "--name", handle.container,
        "--gpus", gpu_selector,
        # published on loopback only; the server binds 0.0.0.0 INSIDE the container netns
        # because a container-internal 127.0.0.1 bind is unreachable through -p.
        "-p", "127.0.0.1:%d:%d" % (args.port, args.port),
        "-v", "%s:%s:ro" % (os.path.dirname(args.model), os.path.dirname(args.model)),
        args.image,
    ] + server_args(args.model, args.ctx, args.kv, args.port, args.fa, args.extra_args,
                    bind="0.0.0.0")
    handle.command = cmd
    rc, out, err = run(cmd, timeout=300)
    if rc != 0:
        raise RuntimeError("docker run failed rc=%d: %s" % (rc, err[-500:]))
    handle.started_by_us = True
    return handle


def launch_host(handle, args, out_dir):
    handle.mode = "host"
    binary = os.path.join(args.llamacpp_root, "bin", "llama-server")
    if not os.path.exists(binary):
        raise RuntimeError("built llama-server not found at %s (run build_llamacpp_cuda.sh)" % binary)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(args.llamacpp_root, "bin") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    elif args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    handle.log_path = os.path.join(out_dir, "server-%s.log" % args.tag)
    cmd = [binary] + server_args(args.model, args.ctx, args.kv, args.port, args.fa,
                                 args.extra_args, bind=args.bind)
    handle.command = cmd
    # The child inherits the fd; this process closes its own copy right after the fork so a
    # long sweep does not sit on a write handle it never uses again.
    log_handle = open(handle.log_path, "w", encoding="utf-8")
    try:
        handle.process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT,
                                          env=env)
    finally:
        log_handle.close()
    handle.started_by_us = True
    return handle


def wait_healthy_watching_child(server, handle, deadline_s, poll_s=1.0):
    """Poll /health, but treat OUR child exiting as the answer.

    `LlamaServer.wait_healthy` only knows about the URL. When we started the server
    ourselves, its process is the ground truth: if it exits, whatever answers that port is
    not ours, and the measurement is "our server exited rc=N", not a benchmark.
    """
    process = handle.process if handle.started_by_us and handle.mode == "host" else None
    started = time.time()
    while time.time() - started < deadline_s:
        if process is not None:
            code = process.poll()
            if code is not None:
                tail = handle.logs(60).splitlines()[-25:]
                return {
                    "healthy": False,
                    "wait_s": round(time.time() - started, 2),
                    "child_exit_code": code,
                    "error": (
                        "the llama-server WE launched exited rc=%s before becoming healthy; "
                        "anything answering port %s is not ours" % (code, server.base_url)
                    ),
                    "log_tail": tail,
                }
        if server.health().get("status") == "ok":
            return {
                "healthy": True,
                "wait_s": round(time.time() - started, 2),
                "child_exit_code": None,
                "child_alive": process is None or process.poll() is None,
            }
        time.sleep(poll_s)
    return {"healthy": False, "wait_s": round(time.time() - started, 2),
            "child_exit_code": process.poll() if process is not None else None,
            "error": "server never became healthy within %ss" % deadline_s}


def served_model_check(props, props_n_ctx_value, model_path, ctx):
    """Is the server that answered serving the model and context WE asked for?

    `parity_check.py` already compares its --gguf-path against `/props.model_path`; this is
    the same comparison for the bench, which previously captured `server_props.model_path`
    and never looked at it.
    """
    served = props.get("model_path") if isinstance(props, dict) else None
    check = {
        "served_model_path": served,
        "requested_model_path": model_path,
        "served_model_matches_model_path": None,
        "served_n_ctx": props_n_ctx_value,
        "requested_n_ctx": ctx,
        "served_n_ctx_matches_requested": None,
        "reasons": [],
    }
    if model_path and served:
        check["served_model_matches_model_path"] = (
            os.path.basename(str(served)) == os.path.basename(str(model_path)))
        if not check["served_model_matches_model_path"]:
            check["reasons"].append(
                "server is serving %r but --model is %r" % (served, model_path))
    elif model_path and not served:
        check["reasons"].append("/props reported no model_path to compare against")
    if props_n_ctx_value is not None and ctx:
        check["served_n_ctx_matches_requested"] = int(props_n_ctx_value) == int(ctx)
        if not check["served_n_ctx_matches_requested"]:
            check["reasons"].append(
                "server reports n_ctx %s but --ctx is %s" % (props_n_ctx_value, ctx))
    return check


def _request_or_error(server, prompt, **kwargs):
    """server.completion(), but a refusal comes back as data instead of an exception.

    llama-server answers 400 when prompt + n_predict does not fit the slot context, and
    urllib raises on that. Losing the whole sweep to one refused request is how the
    sibling lane ended up with two `HTTPError 400` rows and nothing after them.
    """
    started = time.time()
    try:
        return server.completion(prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure IS the measurement here
        detail = None
        body = getattr(exc, "read", None)
        if callable(body):
            try:
                detail = body().decode("utf-8", "replace")[:2000]
            except Exception:  # noqa: BLE001
                detail = None
        return {
            "error": repr(exc),
            "error_body": detail,
            "content": "",
            "wall_s": round(time.time() - started, 3),
        }


def measure_context_point(server, target_tokens, code, decode_tokens, depths, do_needle,
                          request_timeout, label="K7", decode_ignore_eos=True, n_ctx=None):
    """Prefill wall + decode tok/s + optional needle retrieval at one context length.

    `n_ctx` is the context the server was launched with. It is not decoration: the prompt
    has to leave room for the tokens we ask the server to generate, or llama-server answers
    `400 Bad Request` and the point is lost. That is exactly how the sibling lane's two
    250k needle rows came back as errors. With `n_ctx` given, the prompt is calibrated
    under `n_ctx - reserve` and the receipt records the cap it hit.
    """
    point = {"target_tokens": target_tokens, "code": code}
    headroom = context_point_headroom(n_ctx, decode_tokens)
    point["context_headroom"] = headroom
    started = time.time()
    calibration = calibrate_line_count(server, target_tokens, code, label=label,
                                       depth_fraction=0.5,
                                       max_tokens=headroom["prompt_ceiling"])
    point["calibration"] = {
        "lines": calibration["lines"],
        "tokens": calibration["tokens"],
        "rounds": calibration["rounds"],
        "requested_target_tokens": calibration["requested_target_tokens"],
        "effective_target_tokens": calibration["target_tokens"],
        "prompt_ceiling": calibration["max_tokens"],
        "capped_by_ctx": calibration["capped"],
        "over_cap": calibration["over_cap"],
        "wall_s": round(time.time() - started, 2),
    }
    point["capped_by_ctx"] = calibration["capped"]
    if calibration["capped"]:
        print("    NOTE: %d-token point capped to %d tokens by -c %s (reserve %d for decode)"
              % (target_tokens, calibration["tokens"], n_ctx, headroom["reserve_tokens"]))
    prompt = calibration["prompt"]

    prefill = _request_or_error(server, prompt, n_predict=1, cache_prompt=False,
                                timeout=request_timeout)
    point["prefill"] = prefill
    point["prefill_wall_s"] = prefill.get("wall_s")
    point["prefill_tokens_per_s"] = prefill.get("prompt_per_second")
    if prefill.get("error"):
        # A refused request is a measurement too: record it and keep the sweep going, so
        # one bad context point cannot cost the points after it.
        point["error"] = prefill["error"]
        # The decode keys are written EVEN THOUGH no decode ran. A refused point that
        # simply omits them is how this early return used to destroy the very receipt it
        # exists to protect: finish()'s summary indexed point["decode_tokens_per_s"],
        # raised KeyError after the complete receipt had already been written, and the
        # `except` in main() then re-ran finish() with partial=True and the KeyError as
        # the receipt's error -- overwriting a good sweep with a Python bug.
        point["decode"] = None
        point["decode_tokens_per_s"] = None
        point["decode_ignore_eos"] = bool(decode_ignore_eos)
        point["decode_tokens_requested"] = int(decode_tokens)
        point["needle"] = []
        return point

    # The calibration prompt contains the needle AND its question, so the model answers and
    # hits EOS after a handful of tokens. Without --ignore-eos the 4k point decodes 9 tokens
    # and the 32k point decodes 63, and those two tok/s numbers are not comparable. ignore_eos
    # makes every context point decode exactly `decode_tokens`.
    decode = _request_or_error(
        server, prompt, n_predict=decode_tokens, cache_prompt=True, timeout=request_timeout,
        extra={"ignore_eos": True} if decode_ignore_eos else None,
    )
    point["decode"] = decode
    point["decode_tokens_per_s"] = decode.get("predicted_per_second")
    point["decode_ignore_eos"] = bool(decode_ignore_eos)
    point["decode_tokens_requested"] = int(decode_tokens)

    point["needle"] = []
    if do_needle:
        for depth in depths:
            needle_prompt, insert_at = build_needle_prompt(
                calibration["lines"], depth, code, label=label
            )
            answer = _request_or_error(
                server, needle_prompt, n_predict=NEEDLE_PREDICT_TOKENS, cache_prompt=False,
                timeout=request_timeout,
            )
            scored = score_needle(answer.get("content", ""), code)
            point["needle"].append({
                "depth_fraction": depth,
                "error": answer.get("error"),
                "insert_line": insert_at,
                "of_lines": calibration["lines"],
                "prompt_tokens": answer.get("prompt_n"),
                "hit": scored["hit"],
                "answer_verbatim": answer.get("content", ""),
                "answer_normalized": scored["normalized"],
                "prefill_tokens_per_s": answer.get("prompt_per_second"),
                "decode_tokens_per_s": answer.get("predicted_per_second"),
                "wall_s": answer.get("wall_s"),
            })
            print("    needle depth=%.2f tokens=%s hit=%s answer=%r" % (
                depth, answer.get("prompt_n"), scored["hit"], scored["normalized"][:80]))
    return point


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", default="/home/cereal/omni-campaign/export/envelope")
    parser.add_argument("--attach", default=None, help="base URL of an already running server")
    parser.add_argument("--container-name", default=None,
                        help="in --attach mode, read this container's logs (never stopped)")
    parser.add_argument("--runner", default="host", choices=("host", "docker"))
    parser.add_argument("--image", default="ghcr.io/ggml-org/llama.cpp:server-cuda")
    parser.add_argument("--llamacpp-root", default=DEFAULT_LLAMACPP_ROOT)
    parser.add_argument("--model", default=None, help="GGUF path (required unless --attach)")
    parser.add_argument("--ctx", type=int, default=262144)
    parser.add_argument("--kv", default="q4_0", choices=KV_TYPES,
                        help="matched pair: applied to BOTH -ctk and -ctv")
    parser.add_argument("--fa", default="on", choices=("on", "off", "auto"))
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--bind", default="127.0.0.1",
                        help="host-runner --host value; loopback by default (llama-server "
                             "has no auth and this box fronts other services)")
    parser.add_argument("--host-url", default="http://127.0.0.1")
    parser.add_argument("--gpu-index", type=int, default=None, help="for nvidia-smi peak sampling")
    parser.add_argument("--gpu-uuid", default=None, help="GPU-<uuid>; CUDA_VISIBLE_DEVICES / --gpus")
    parser.add_argument("--prefill-points", default="4096,32768,131072,262144")
    parser.add_argument("--needle-points", default="32768,65536,131072,262144")
    parser.add_argument("--needle-depths", default="0.1,0.5,0.9")
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--no-decode-ignore-eos", dest="decode_ignore_eos",
                        action="store_false", default=True,
                        help="let the decode leg stop at EOS (context points then decode "
                             "different token counts and their tok/s are not comparable)")
    parser.add_argument("--r3-control", default=None,
                        help="fa_negative_control.py receipt that licenses (or voids) the "
                             "R3 quantized-KV inference on THIS build")
    parser.add_argument("--load-timeout", type=int, default=2400)
    parser.add_argument("--request-timeout", type=float, default=5400.0)
    parser.add_argument("--max-wall-s", type=int, default=0,
                        help="stop cleanly after this many seconds and write a partial receipt")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--log-head", type=int, default=600,
                        help="lines of the load banner to analyse for FA/KV/offload signals")
    parser.add_argument("--extra-args", default=None, help="extra llama-server flags, space separated")
    parser.add_argument("--keep-server", action="store_true", help="leave our server running")
    args = parser.parse_args(argv)
    # --tag lands in every output path and (docker runner) in a container name. Only
    # export_hf_to_gguf.py used to validate it, so `--tag ../../x` wrote outside --out-dir
    # through write_receipt's makedirs + os.replace.
    args.tag = safe_tag(args.tag)

    prefill_points = parse_int_list(args.prefill_points)
    needle_points = set(parse_int_list(args.needle_points))
    depths = parse_float_list(args.needle_depths)
    all_points = sorted(set(prefill_points) | needle_points)
    for point in all_points:
        if point > args.ctx:
            parser.error("context point %d exceeds --ctx %d" % (point, args.ctx))

    os.makedirs(args.out_dir, exist_ok=True)
    handle = ServerHandle()
    sampler = GpuSampler(
        interval=args.sample_interval,
        indices=[args.gpu_index] if args.gpu_index is not None else None,
    ).start()

    receipt = {
        "kind": "envelope_bench",
        "created_at": utc_now(),
        "tag": args.tag,
        "config": {
            "ctx": args.ctx,
            "kv_type_k": args.kv,
            "kv_type_v": args.kv,
            "flash_attn_requested": args.fa,
            "n_parallel": 1,
            "runner": "attach" if args.attach else args.runner,
            "image": args.image if args.runner == "docker" else None,
            "model": args.model,
            "gpu_index": args.gpu_index,
            "gpu_uuid": args.gpu_uuid,
            "prefill_points": prefill_points,
            "needle_points": sorted(needle_points),
            "needle_depths": depths,
            "decode_tokens": args.decode_tokens,
            "decode_ignore_eos": bool(args.decode_ignore_eos),
            "bind": args.bind if not args.attach else None,
        },
        "host": host_facts(),
        # How many context points this run SET OUT to measure. finish() compares it against
        # len(points) so "the sweep stopped early" is a fact in the receipt rather than
        # something a reader has to infer by counting.
        "points_requested": len(all_points),
        "points": [],
        "grading_note": (
            "Every number here is a reported measurement. No threshold in this file is a "
            "gate; decision lines are Cyril's to set."
        ),
    }
    if args.model and os.path.exists(args.model):
        receipt["model_bytes"] = os.path.getsize(args.model)
        if os.path.getsize(args.model) < 40e9:  # hashing a huge GGUF is minutes of I/O
            receipt["model_sha256"] = sha256_file(args.model)

    deadline = time.time() + args.max_wall_s if args.max_wall_s else None
    started_all = time.time()
    try:
        # ------------------------------------------------------------------ server
        if args.attach:
            handle.mode = "attach"
            handle.container = args.container_name
            base_url = args.attach
        else:
            if not args.model:
                parser.error("--model is required unless --attach is given")
            print("launching %s runner on port %d ..." % (args.runner, args.port))
            if args.runner == "docker":
                launch_docker(handle, args, args.out_dir)
            else:
                launch_host(handle, args, args.out_dir)
            base_url = "%s:%d" % (args.host_url, args.port)
            receipt["launch_command"] = handle.command
            if handle.process is not None:
                receipt["server_pid"] = handle.process.pid
                sampler.watch_pid(handle.process.pid)

        server = LlamaServer(base_url, timeout=args.request_timeout)
        # 1 s polling: wait_s is a health-poll upper bound, NOT the model load time. The
        # load time is read off the server's own log clock below (listening_at_log_s).
        #
        # The wait also watches OUR CHILD. Polling a URL alone cannot tell "my server came
        # up" from "my server died on bind and a stranger on that port answered": 18080 and
        # 18081 are occupied on this box, and a port collision would have produced a full
        # sweep of a foreign model under a receipt claiming our own -fa on / -ctk q4_0 as
        # observed flags. That is precisely the fabrication the --attach path was hardened
        # against, arriving through the front door.
        wait = wait_healthy_watching_child(server, handle, args.load_timeout, poll_s=1.0)
        wait["note"] = (
            "wait_s is when OUR poll first saw /health ok, rounded up by poll_s=1.0; "
            "read log_analysis.listening_at_log_s for the server's own load time"
        )
        receipt["server_wait"] = wait
        receipt["server_url"] = base_url
        print("server health: %s" % wait)
        if not wait["healthy"]:
            receipt["error"] = wait.get("error") or (
                "server never became healthy within %ss" % args.load_timeout)
            receipt["log_tail"] = handle.logs(200).splitlines()[-60:]
            print("SERVER DID NOT COME UP: %s" % receipt["error"])
            return finish(receipt, args, handle, sampler, started_all, partial=True)

        load_log = handle.logs(head=args.log_head)
        receipt["log_analysis"] = analyse_server_log(load_log)
        receipt["log_head"] = load_log.splitlines()[:60]
        receipt["log_tail"] = handle.logs(200).splitlines()[-40:]
        props = server.props()
        receipt["server_props"] = _slim_props(props)
        receipt["server_props_n_ctx"] = props_n_ctx(props)
        identity_check = served_model_check(
            props, receipt["server_props_n_ctx"], args.model, args.ctx)
        receipt["served_model_check"] = identity_check
        if identity_check["served_model_matches_model_path"] is False:
            receipt["error"] = (
                "the server answering %s is serving a DIFFERENT model than --model: %s"
                % (base_url, "; ".join(identity_check["reasons"]))
            )
            print("SERVED MODEL MISMATCH: %s" % receipt["error"])
            return finish(receipt, args, handle, sampler, started_all, partial=True)
        if identity_check["served_n_ctx_matches_requested"] is False:
            # Not fatal (llama.cpp clamps n_ctx to the model's trained maximum and says so),
            # but a receipt headed "-c 262144" whose server runs 32768 is a false headline.
            print("WARNING: %s" % "; ".join(identity_check["reasons"]))

        # Requested envelope, WITH PROVENANCE. Only two sources are observations of the
        # server that is actually answering: our own launch command, and `docker inspect`
        # of the attached container. Falling back to our own CLI defaults would let an
        # --attach run against a peer's `-fa off -ctk f16` server produce a receipt
        # asserting a q4_0 flash-attention run, so that case is labelled and the flags are
        # discarded downstream by flash_attn_evidence().
        requested = parse_server_flags(handle.command) if handle.command else {}
        flags_source = "own_launch_command" if requested else None
        if not requested and handle.container:
            command = container_launch_command(handle.container)
            receipt["attached_container_command"] = command
            requested = parse_server_flags(command)
            if requested:
                flags_source = "docker_inspect"
        if not requested:
            requested = {
                "flash_attn": args.fa, "type_k": args.kv, "type_v": args.kv,
                "n_ctx": args.ctx, "n_parallel": 1,
            }
            flags_source = "cli_defaults_unverified"
        receipt["requested_flags"] = requested
        receipt["requested_flags_source"] = flags_source
        # The control licenses R3 for ONE binary. In host mode we know exactly which one
        # (--llamacpp-root), so the control is joined to it on commit + llama-server
        # sha256; in --attach mode we do not, and the join comes back unchecked, which
        # flash_attn_evidence() treats as "not licensed here" rather than as a licence.
        identity = None
        if not args.attach and args.runner == "host":
            identity = build_identity(llamacpp_root=args.llamacpp_root,
                                      source="own_build_receipt")
        receipt["build_identity"] = identity
        r3_control = load_r3_control(args.r3_control, identity=identity)
        receipt["r3_control"] = r3_control
        receipt["flash_attn_evidence"] = flash_attn_evidence(
            receipt["log_analysis"], requested, flags_source=flags_source, r3_control=r3_control
        )
        print("requested (%s): -fa %s -ctk %s -ctv %s -c %s -np %s" % (
            flags_source,
            requested.get("flash_attn"), requested.get("type_k"), requested.get("type_v"),
            requested.get("n_ctx"), requested.get("n_parallel")))
        if flags_source == "cli_defaults_unverified":
            print("WARNING: launch flags were NOT observed off the running server; "
                  "they are our own CLI defaults and are excluded from the FA evidence.")
        print("FA reading: %s" % receipt["flash_attn_evidence"]["reading"])
        print("kv buffer MiB from log: %s | n_ctx log/props: %s/%s" % (
            receipt["log_analysis"]["kv_buffer_mib_total"],
            receipt["log_analysis"]["n_ctx"], receipt["server_props_n_ctx"]))
        if receipt["log_analysis"]["fallback_evidence"]:
            print("FALLBACK EVIDENCE (recorded, not a verdict):")
            for item in receipt["log_analysis"]["fallback_evidence"][:10]:
                print("   %s :: %s" % (item["signal"], item["line"][:160]))

        # ------------------------------------------------------------------ sweep
        for index, point in enumerate(all_points):
            if deadline and time.time() > deadline:
                receipt["stopped_early"] = "max_wall_s reached before %d-token point" % point
                print(receipt["stopped_early"])
                break
            code = NEEDLE_CODES[index % len(NEEDLE_CODES)]
            print("context point %d tokens (code %s) ..." % (point, code))
            measured = measure_context_point(
                server, point, code, args.decode_tokens, depths,
                do_needle=point in needle_points,
                request_timeout=args.request_timeout,
                decode_ignore_eos=args.decode_ignore_eos,
                n_ctx=args.ctx,
            )
            receipt["points"].append(measured)
            if measured.get("error"):
                print("  REFUSED by the server (recorded, sweep continues): %s"
                      % measured["error"])
            else:
                print("  prefill %.1fs (%s tok/s), decode %s tok/s" % (
                    measured["prefill_wall_s"], measured["prefill_tokens_per_s"],
                    measured["decode_tokens_per_s"]))
            # keep_versions=False: the in-flight receipt is rewritten after every context
            # point, and an immutable sidecar per point (5 for a 5-point sweep) is clutter,
            # not provenance. The FINAL receipt still gets its sidecar.
            write_receipt(os.path.join(args.out_dir, "envelope-%s.partial.json" % args.tag),
                          receipt, keep_versions=False)

        receipt["log_analysis_final"] = analyse_server_log(handle.logs(600))
        # A sweep the --max-wall-s deadline cut short is NOT a complete sweep. It used to
        # be written with partial=false, which is the exact field EXPORT_LANE_SPEC.md
        # calls the completeness marker, and finish() then deleted the .partial.json that
        # would have shown the truncation. A budget-truncated 3-of-4-point run must not be
        # indistinguishable from a finished one.
        return finish(receipt, args, handle, sampler, started_all,
                      partial=sweep_is_partial(receipt))
    except KeyboardInterrupt:
        receipt["error"] = "interrupted"
        return finish(receipt, args, handle, sampler, started_all, partial=True)
    except Exception as exc:  # noqa: BLE001 - a crash is also a measurement
        receipt["error"] = repr(exc)
        receipt["log_tail"] = handle.logs(200).splitlines()[-60:]
        return finish(receipt, args, handle, sampler, started_all, partial=True)


def _slim_props(props):
    if not isinstance(props, dict):
        return {}
    keep = {}
    for key in ("model_path", "n_ctx", "total_slots", "chat_template", "build_info"):
        if key in props:
            value = props[key]
            keep[key] = value[:200] if isinstance(value, str) else value
    return keep


def sweep_is_partial(receipt):
    """Was this sweep cut short, whatever the exit path?

    `partial` is the field EXPORT_LANE_SPEC.md cites as the completeness marker, so the
    decision cannot live at one call site and be forgotten at another. A --max-wall-s
    break sets `stopped_early` and used to fall through to `partial=False`, which wrote a
    3-of-4-point run that reads exactly like a finished one AND deleted the .partial.json
    that showed the truncation.
    """
    return bool(receipt.get("stopped_early")) or bool(receipt.get("error"))


def finish(receipt, args, handle, sampler, started_all, partial):
    receipt["gpu_sampling"] = sampler.stop()
    receipt["total_wall_s"] = round(time.time() - started_all, 2)
    receipt["partial"] = partial
    # Two readers of this receipt ask different questions: "did the process finish
    # cleanly" (partial) and "was every requested point measured" (complete_sweep). A
    # refused point keeps the sweep complete-but-flawed; a deadline break makes it neither.
    points = receipt.get("points") or []
    refused = [point.get("target_tokens") for point in points if point.get("error")]
    receipt["points_measured"] = len(points)
    receipt["points_refused"] = refused
    requested_count = receipt.get("points_requested")
    receipt["complete_sweep"] = bool(
        not partial
        and not receipt.get("stopped_early")
        and not receipt.get("error")
        and (requested_count is None or len(points) == requested_count)
    )
    if not args.keep_server:
        receipt["teardown"] = handle.stop()
    else:
        receipt["teardown"] = {
            "stopped": False,
            "reason": "--keep-server",
            "warning": (
                "the server this run started is STILL RUNNING; whatever stops it later is "
                "outside this receipt. Prefer omitting --keep-server so the shutdown is "
                "recorded, and never pattern-kill llama-server on this box (peer lanes own "
                "p40_256k and gpu0_256k)."
            ),
            "pid": getattr(handle.process, "pid", None),
            "container": handle.container if handle.mode == "docker" else None,
        }
    path = os.path.join(args.out_dir, "envelope-%s.json" % args.tag)
    digest = write_receipt(path, receipt)
    partial_path = os.path.join(args.out_dir, "envelope-%s.partial.json" % args.tag)
    if os.path.exists(partial_path) and not partial:
        os.remove(partial_path)
    print("\nenvelope receipt: %s sha256=%s%s" % (path, digest, "  (PARTIAL)" if partial else ""))
    sampling = receipt["gpu_sampling"]
    peaks = sampling["peak_by_index"]
    for index, info in sorted(peaks.items()):
        print("  GPU%s whole-card peak %.0f / %s MiB" % (
            index, info["peak_memory_used_mib"], info.get("memory_total_mib")))
    if sampling.get("own_peak_memory_used_mib") is not None:
        print("  this run's OWN measured peak: %.0f MiB (pids %s)" % (
            sampling["own_peak_memory_used_mib"], ",".join(sampling.get("tracked_pids") or [])))
    for pid, info in sorted(sampling.get("peak_by_pid", {}).items()):
        print("    pid %-8s %8.0f MiB  gpu%s  %s%s" % (
            pid, info["peak_memory_used_mib"], info.get("gpu_index"),
            info.get("process_name"), "  <- ours" if info.get("tracked_as_ours") else ""))
    for point in receipt["points"]:
        hits = ["%.2f:%s" % (n["depth_fraction"], "HIT" if n["hit"] else "miss")
                for n in point.get("needle", [])]
        if point.get("error"):
            # A refused point has no decode leg. Printing it as a refusal (rather than
            # indexing keys it never had) is what keeps one bad point from taking the
            # whole receipt down at the very last step.
            print("  %8d tok | REFUSED after %ss: %s" % (
                point.get("target_tokens"), point.get("prefill_wall_s"),
                str(point["error"])[:120]))
            continue
        print("  %8d tok | prefill %8s tok/s %8ss | decode %6s tok/s | %s" % (
            point.get("target_tokens"),
            point.get("prefill_tokens_per_s"), point.get("prefill_wall_s"),
            point.get("decode_tokens_per_s"), " ".join(hits)))
    return 0 if not partial else 1


if __name__ == "__main__":
    raise SystemExit(main())
