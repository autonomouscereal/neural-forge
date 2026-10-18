#!/usr/bin/env python3
"""Stub of selfgen_sampler.py for test_vmath_queue.sh: same CLI surface the queue calls (serve / stop / eog-probe / run),
no model, no GPU.  Every call is appended to $STUB_LOG.  `run` writes journal records in the real schema the verifier
reads, sleeps $STUB_SLEEP s per record, exits 2 on SIGTERM (as the real sampler does when stopped early) and seals a
manifest.json once the token target is reached or every prompt is sampled."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

LOG = os.environ.get("STUB_LOG", "/dev/null")


def note(*parts):
    with open(LOG, "a") as fh:
        fh.write(" ".join(str(p) for p in parts) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command")
    p.add_argument("--gpu-index"); p.add_argument("--port"); p.add_argument("--name"); p.add_argument("--receipt-dir")
    p.add_argument("--launch-receipt"); p.add_argument("--outcome"); p.add_argument("--out"); p.add_argument("--out-dir")
    p.add_argument("--prompts"); p.add_argument("--token-target", type=int, default=0)
    args, _ = p.parse_known_args()
    note(args.command, args.gpu_index or args.launch_receipt or args.out_dir or "", args.outcome or "")
    if args.command == "serve":
        Path(args.receipt_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.receipt_dir) / f"serve-{args.name}.json").write_text(json.dumps({"port": args.port, "gpu": args.gpu_index}))
        return 0
    if args.command in ("stop",):
        return 0
    if args.command == "eog-probe":
        Path(args.out).write_text(json.dumps({"ok": True}))
        return 0
    if args.command == "run":
        signal.signal(signal.SIGTERM, lambda *_: (note("run got SIGTERM"), sys.exit(2)))
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        prompts = [json.loads(l) for l in open(args.prompts)]
        journal = out / "journal.jsonl"
        done = {json.loads(l)["index"] for l in open(journal)} if journal.exists() else set()
        tokens = 100 * len(done)
        for i, pr in enumerate(prompts):
            if i in done:
                continue
            if args.token_target and tokens >= args.token_target:
                break
            time.sleep(float(os.environ.get("STUB_SLEEP", "0.1")))
            text = "work</think>\n\\boxed{1}" if pr["stratum"] == "R" else "work</think>\n\\boxed{2}"
            with open(journal, "a") as fh:
                fh.write(json.dumps({"index": i, "rid": pr["rid"], "text": text, "end": "eog", "response_tokens": 100, "seed": 1}) + "\n")
            tokens += 100
        (out / "progress.json").write_text(json.dumps({"rate_tokens_per_s_session": 1.0, "response_tokens_done": tokens}))
        (out / "manifest.json").write_text(json.dumps({"stub": True, "tokens": tokens}))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
