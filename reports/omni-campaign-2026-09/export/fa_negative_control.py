#!/usr/bin/env python3
"""Negative control for the R3 flash-attention inference, run against THIS build.

The whole FA evidence chain in this lane leans on one premise:

    "llama.cpp refuses -ctk/-ctv != f16 unless flash attention is on, so a server that
     loaded with q4_0 KV ran the FA path."

That is a claim about a specific binary's behaviour, and it was never exercised against
commit b96806d9 -- the same build that turned out to print no flash_attn/KV banner at all.
An untested premise is not evidence, so this script goes and looks:

    trial NEG   -fa off  -ctk <kv> -ctv <kv>   (does it refuse?)
    trial POS   -fa on   -ctk <kv> -ctv <kv>   (does it load?)

`premise_holds` is True only when ALL THREE hold:

    1. NEG failed to come up,
    2. POS came up,
    3. NEG's log actually shows a flash-attn / quantized-KV refusal.

(3) is not decoration. Without it, ANY failure of the negative trial licensed R3: something
already listening on the port, an OOM on the chosen card or a moved model path all exit
rc=1 in about a second, `became_healthy` is False, POS then binds the freed port and comes
up, and every downstream receipt reads "flash-attention path ran ... which this build
refuses without flash_attn" on the strength of a bind error. The script also refuses to
start at all when the port is already answering, which is that scenario's usual trigger.

If NEG loads anyway, the premise is VOID; if NEG fails for an unrelated reason, the result
is INCONCLUSIVE (`premise_holds=None`). `flash_attn_evidence()` reads this receipt and
downgrades every reading in the lane to R1 (the launch flag) in both cases.

The receipt records the llama-server sha256 it exercised. `load_r3_control()` joins on that
plus the commit, so a control left on disk across a rebuild cannot license a binary it
never touched.

Both trials use a small model and a small context; the question is load-time acceptance,
not throughput. Only servers this script started are ever stopped.

Example
-------
  python3 fa_negative_control.py \
      --model /home/cereal/omni-campaign/export/gguf/Q35-08B-stock.Q4_K_M.gguf \
      --gpu-uuid GPU-a92b0dfe-83f8-11ff-db0d-c65921f365b4 --gpu-index 2 \
      --kv q4_0 --ctx 4096 --port 18097 \
      --out-dir /home/cereal/omni-campaign/export/fa-control
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from llama_probe import (  # noqa: E402
    LlamaServer,
    analyse_server_log,
    host_facts,
    parse_server_flags,
    safe_tag,
    sha256_file,
    utc_now,
    write_receipt,
)

DEFAULT_LLAMACPP_ROOT = "/home/cereal/omni-campaign/export/llamacpp"
# Lines that, if present, MENTION flash attention or KV quantization. Broad on purpose:
# they are recorded verbatim so a human can read them. They are NOT the licence -- see
# refusal_signals() for that, which is narrow on purpose.
REFUSAL_MARKERS = (
    "cache quantization requires",
    "requires flash_attn",
    "flash attention",
    "flash_attn",
)
# The licence condition. A negative trial that merely FAILED proves nothing: a port already
# in use, an OOM on the chosen card or a moved model path all exit rc=1 in about a second,
# and the earlier version of this script read any such failure as "the build refused
# quantized KV", which then licensed R3 in every downstream receipt. The trial must be seen
# refusing FOR THE STATED REASON.
_RE_REFUSAL = re.compile(
    r"(?=.*(?:flash.?attn|flash.?attention))(?=.*(?:cache|kv|ctk|ctv|quantiz))",
    re.IGNORECASE,
)


def refusal_signals(analysis, lines):
    """Positive evidence that the negative trial refused because of quantized KV.

    Returns (signals, evidence_lines). Empty means: whatever went wrong, it was not the
    refusal R3 depends on.
    """
    signals = []
    evidence = []
    for item in analysis.get("fallback_evidence", []):
        if item.get("signal") in ("kv_quant_refused", "flash_attn_unsupported"):
            signals.append(item["signal"])
            evidence.append(item["line"])
    for line in lines:
        if _RE_REFUSAL.search(line):
            text = line.strip()
            if text not in evidence:
                signals.append("flash_attn_kv_line")
                evidence.append(text)
    ordered = []
    for signal in signals:
        if signal not in ordered:
            ordered.append(signal)
    return ordered, evidence[:20]


def port_is_answering(port, timeout=3.0):
    """Is something already bound to this port? A pre-existing server would make the
    negative trial fail on bind and the positive trial answer for a stranger."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        return sock.connect_ex(("127.0.0.1", int(port))) == 0
    finally:
        sock.close()


def decide_premise(negative, positive):
    """(premise_holds, observed) from the two trials. Pure; unit-tested.

    The licence needs THREE things, not two: NEG must fail, POS must come up, AND NEG's log
    must actually say it refused over quantized KV / flash attention. Without the third,
    any unrelated NEG failure (occupied port, OOM, moved model path) licensed R3, and every
    downstream receipt then read "flash-attention path ran ... which this build refuses
    without flash_attn" on the strength of a bind error.
    """
    neg_refused = bool(negative.get("refusal_signals"))
    if negative["became_healthy"] and positive["became_healthy"]:
        return False, (
            "this build LOADS with quantized KV and -fa off: quantized-KV acceptance is NOT "
            "evidence that the flash-attention path ran (R3 VOID)"
        )
    if not positive["became_healthy"]:
        return None, (
            "INCONCLUSIVE: the positive control (-fa on) also failed to come up, so the "
            "negative result says nothing about flash attention"
        )
    if negative["became_healthy"]:
        return None, "INCONCLUSIVE"
    if not neg_refused:
        return None, (
            "INCONCLUSIVE: the negative trial failed for a reason unrelated to KV "
            "quantization -- no flash-attn/KV refusal line appears in its log (exit=%s, "
            "first error line: %r). R3 is NOT licensed by this run."
            % (negative.get("process_exit_code"), (negative.get("error_lines") or [None])[0])
        )
    return True, (
        "this build REFUSES quantized KV with -fa off and accepts it with -fa on: "
        "quantized-KV acceptance is evidence that flash attention was enabled "
        "(R3 LICENSED; refusal signals %s)" % ",".join(negative["refusal_signals"])
    )


def run_trial(name, fa, args, out_dir):
    """Start one llama-server, watch it, stop it. Returns a measurement dict."""
    binary = os.path.join(args.llamacpp_root, "bin", "llama-server")
    log_path = os.path.join(out_dir, "fa-control-%s-%s.log" % (args.tag, name))
    cmd = [
        binary,
        "-m", args.model,
        "-c", str(args.ctx),
        "-fa", fa,
        "-np", "1",
        "-ngl", "999",
        "-ctk", args.kv,
        "-ctv", args.kv,
        "--host", "127.0.0.1",
        "--port", str(args.port),
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(args.llamacpp_root, "bin") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu_uuid:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    elif args.gpu_index is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    print("trial %s: -fa %s -ctk %s -ctv %s -c %d" % (name, fa, args.kv, args.kv, args.ctx))
    handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env)
    server = LlamaServer("http://127.0.0.1:%d" % args.port, timeout=30.0)

    started = time.time()
    healthy = False
    exited = None
    while time.time() - started < args.trial_timeout:
        exited = process.poll()
        if exited is not None:
            break
        if server.health(timeout=5.0).get("status") == "ok":
            healthy = True
            break
        time.sleep(1.0)
    wall = round(time.time() - started, 2)

    stopped = {"stopped": False, "reason": "process had already exited"}
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        stopped = {"stopped": True, "pid": process.pid}
    handle.close()

    with open(log_path, "r", encoding="utf-8", errors="replace") as reader:
        text = reader.read()
    lines = text.splitlines()
    analysis = analyse_server_log(text)
    marker_lines = [
        line.strip() for line in lines
        if any(marker.lower() in line.lower() for marker in REFUSAL_MARKERS)
    ]
    error_lines = [
        line.strip() for line in lines
        if ("error" in line.lower() or "failed" in line.lower() or "unsupported" in line.lower())
    ]

    signals, signal_lines = refusal_signals(analysis, lines)

    return {
        "trial": name,
        "flash_attn_flag": fa,
        "kv": args.kv,
        "command": cmd,
        "requested_flags": parse_server_flags(cmd),
        "refusal_signals": signals,
        "refusal_evidence_lines": signal_lines,
        "log_path": log_path,
        "log_sha256": sha256_file(log_path),
        "log_line_count": len(lines),
        "log_head": lines[:40],
        "log_tail": lines[-40:],
        "process_exit_code": process.returncode,
        "exited_before_ready": exited is not None,
        "became_healthy": healthy,
        "wall_s": wall,
        "log_analysis": analysis,
        "flash_attn_or_kv_lines": marker_lines[:20],
        "error_lines": error_lines[:20],
        "teardown": stopped,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="small GGUF; load-time behaviour only")
    parser.add_argument("--llamacpp-root", default=DEFAULT_LLAMACPP_ROOT)
    parser.add_argument("--out-dir", default="/home/cereal/omni-campaign/export/fa-control")
    parser.add_argument("--tag", default="build")
    parser.add_argument("--kv", default="q4_0")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=18097)
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--gpu-uuid", default=None)
    parser.add_argument("--trial-timeout", type=float, default=180.0)
    parser.add_argument("--receipt-name", default="fa-negative-control.json",
                        help="the name load_r3_control() looks for")
    args = parser.parse_args(argv)

    args.tag = safe_tag(args.tag)
    binary = os.path.join(args.llamacpp_root, "bin", "llama-server")
    for path in (binary, args.model):
        if not os.path.exists(path):
            print("MISSING: %s" % path, file=sys.stderr)
            return 3
    # Refuse to run into an occupied port. Both trials would then be about someone else's
    # server: NEG dies on bind (which used to read as a refusal) and POS "comes up" because
    # the stranger answers /health.
    if port_is_answering(args.port):
        print("PORT %d IS ALREADY ANSWERING on 127.0.0.1 -- refusing to run: the negative "
              "trial would fail on bind and the positive trial would be measuring whatever "
              "is already there. Pass a free --port." % args.port, file=sys.stderr)
        return 4
    os.makedirs(args.out_dir, exist_ok=True)

    negative = run_trial("neg-fa-off", "off", args, args.out_dir)
    print("  neg: healthy=%s exit=%s wall=%ss" % (
        negative["became_healthy"], negative["process_exit_code"], negative["wall_s"]))
    positive = run_trial("pos-fa-on", "on", args, args.out_dir)
    print("  pos: healthy=%s exit=%s wall=%ss" % (
        positive["became_healthy"], positive["process_exit_code"], positive["wall_s"]))

    premise_holds, observed = decide_premise(negative, positive)
    neg_refused_for_the_right_reason = bool(negative["refusal_signals"])

    # The refusal line is only ever taken from the REFUSAL evidence. Falling back to any
    # error line is what let a bind failure or an OOM be published as a refusal.
    refusal_line = (negative["refusal_evidence_lines"] or [None])[0]

    build_receipt = os.path.join(args.llamacpp_root, "build-receipt.json")
    receipt = {
        "kind": "fa_negative_control",
        "created_at": utc_now(),
        "tag": args.tag,
        "premise": (
            "llama.cpp refuses -ctk/-ctv != f16 unless flash attention is on "
            "(the R3 inference used by flash_attn_evidence)"
        ),
        "premise_holds": premise_holds,
        "observed": observed,
        "refusal_line": refusal_line,
        "negative_refusal_signals": negative["refusal_signals"],
        "negative_refused_for_the_stated_reason": neg_refused_for_the_right_reason,
        "licence_conditions": {
            "negative_trial_failed_to_come_up": not negative["became_healthy"],
            "positive_trial_came_up": positive["became_healthy"],
            "negative_log_shows_a_flash_attn_or_kv_refusal": neg_refused_for_the_right_reason,
        },
        "port_was_free_before_the_run": True,
        "kv": args.kv,
        "ctx": args.ctx,
        "model": args.model,
        "model_sha256": sha256_file(args.model),
        "gpu_index": args.gpu_index,
        "gpu_uuid": args.gpu_uuid,
        "trials": [negative, positive],
        "host": host_facts(),
        "grading_note": (
            "A control, not a gate. It licenses or voids one inference; it says nothing "
            "about any model's quality."
        ),
    }
    # The binary this control actually exercised, by hash. `load_r3_control()` joins on
    # this: a control is a measurement of ONE binary, and without the hash a stale control
    # keeps licensing R3 across a rebuild at a new commit.
    receipt["llama_server_path"] = binary
    receipt["llama_server_sha256"] = sha256_file(binary)
    if os.path.exists(build_receipt):
        receipt["build_receipt_sha256"] = sha256_file(build_receipt)
        with open(build_receipt, "r", encoding="utf-8") as reader:
            built = json.load(reader)
        receipt["llama_cpp_commit"] = built.get("llama_cpp_commit")
        receipt["llama_cpp_describe"] = built.get("llama_cpp_describe")
        recorded = ((built.get("binaries") or {}).get("llama-server") or {}).get("sha256")
        receipt["llama_server_sha256_matches_build_receipt"] = (
            None if not recorded else recorded == receipt["llama_server_sha256"])

    path = os.path.join(args.out_dir, args.receipt_name)
    digest = write_receipt(path, receipt)
    print("\npremise_holds=%s" % premise_holds)
    print("observed: %s" % observed)
    print("refusal/marker line: %r" % refusal_line)
    print("fa control receipt: %s sha256=%s" % (path, digest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
