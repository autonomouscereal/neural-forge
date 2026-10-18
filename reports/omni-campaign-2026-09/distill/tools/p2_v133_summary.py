#!/usr/bin/env python3
"""p2_v133_summary.py -- the P2 rerun summary (lane p2p11; called by run_p2_v133.sh, also on an early exit).

Reads, per planned cell and per attempt (the cell, then its `-ffn2048` OOM retry), what run_p2_v133.sh leaves in
--out: `<tag>.rc` (written by the driver after the trainer exits: the attempt RAN), `<tag>.log` (its `OOM {...}`
line names the phase) and the train receipt `p2-trainer-v133-omni1-<tag>-gpu<N>.json` (written only when train
reaches its end; an OOM re-raises before it).  Every planned cell appears in the summary with its state, even
with no receipt.

Reading (STAGE_D_RULINGS G5-1: >= 151 tok/s -> streamed; else the measured split cell; < 60 -> token cut) is taken
from COMPLETED cells only.  No completed cell -> "NO READING" (a rate of 0 is never invented).  When the long rung
did not complete, the reading says so and names its phase: the 4k cells alone do not show the streamed engine can
run the 32k rung.

1.0.0 (2026-10-14): replaces the inline summary of run_p2_v133.sh (bd28b35f), which globbed receipts only -- an
OOM'd cell vanished from the summary, and with no receipt at all `best = 0` printed "tokens may be cut".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

TOOL_NAME = "p2_v133_summary"
TOOL_VERSION = "1.0.0"
RETRY_SUFFIX = "-ffn2048"
STREAMED_MIN = 151.0
TOKEN_CUT_BELOW = 60.0


def receipt_path(out: Path, tag: str, gpu: int) -> Path:
    return out / f"p2-trainer-v133-omni1-{tag}-gpu{gpu}.json"


def last_oom(log: Path) -> dict[str, Any] | None:
    """The last `OOM {...}` line of an attempt log (streamed_kd_trainer 1.3.3 prints it before re-raising)."""
    if not log.is_file():
        return None
    found = None
    for line in log.read_text(errors="replace").splitlines():
        if line.startswith("OOM "):
            try:
                found = json.loads(line[4:])
            except ValueError:
                found = {"phase": "unparsed OOM line", "raw": line[:300]}
    return found


def read_attempt(out: Path, tag: str, gpu: int) -> dict[str, Any] | None:
    """None when the attempt never ran (no .rc); else its state, rc, OOM phase and (completed) rates."""
    rc_file = out / f"{tag}.rc"
    if not rc_file.is_file():
        return None
    text = rc_file.read_text().strip()
    rc = int(text) if text.lstrip("-").isdigit() else None
    oom = last_oom(out / f"{tag}.log")
    rpath = receipt_path(out, tag, gpu)
    a: dict[str, Any] = {"tag": tag, "rc": rc, "log": str(out / f"{tag}.log"), "receipt": str(rpath) if rpath.is_file() else None,
                         "oom_phase": oom.get("phase") if oom else None, "oom_step": oom.get("step") if oom else None}
    receipt = None
    if rpath.is_file():
        try:
            receipt = json.loads(rpath.read_text())
        except ValueError as exc:
            a.update(state="failed", reason=f"receipt unreadable: {exc}")
            return a
    if rc == 0 and receipt is not None and receipt.get("verdict") == "completed":
        rates = [float(s["tok_per_s"]) for s in receipt.get("steps") or []]
        steady = rates[1:] or rates
        me, mm = receipt.get("memory_engine") or {}, receipt.get("memory_model") or {}
        a.update(state="completed", reason=None, tok_per_s_by_step=rates,
                 steady_tok_per_s_mean_steps_2_on=round(sum(steady) / len(steady), 2) if steady else None,
                 measured_peak_gib=me.get("peak_allocated_gib"), predicted_peak_gib=round(mm.get("peak_bytes", 0) / 2**30, 3),
                 measured_phases_gib=me.get("phases_peak_allocated_gib"),
                 predicted_phases_gib={k: round(v / 2**30, 3) for k, v in (mm.get("phases_bytes") or {}).items()},
                 trainable=(receipt.get("adapter_census") or {}).get("parameters_trainable"), streamed_gb=receipt.get("streamed_gb"),
                 layer_load_seconds=receipt.get("layer_load_seconds"), host_rss_gb=receipt.get("host_rss_gb"))
        if a["steady_tok_per_s_mean_steps_2_on"] is None:
            a.update(state="failed", reason="completed receipt with no steps")
        return a
    if oom:
        reason = f"OOM in {oom.get('phase')} at step {oom.get('step')}"
    elif receipt is not None:
        reason = f"rc={rc}, receipt verdict {receipt.get('verdict')!r} halt {receipt.get('halt_reason')!r}"
    elif rc == 0:
        reason = "rc=0 but no receipt"
    else:
        reason = f"rc={rc}, no OOM line and no receipt (see the log)"
    a.update(state="failed", reason=reason)
    return a


def summarize(out: Path, gpu: int, cells: list[str], long_cell: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for cell in cells:
        attempts = [a for a in (read_attempt(out, t, gpu) for t in (cell, cell + RETRY_SUFFIX)) if a is not None]
        done = [a for a in attempts if a["state"] == "completed"]
        if done:
            entry = {"state": "completed", "used_attempt": done[-1]["tag"], **{k: v for k, v in done[-1].items() if k not in ("tag", "state")}}
        elif attempts:
            last = attempts[-1]
            entry = {"state": "failed", "reason": last["reason"], "oom_phase": last["oom_phase"], "rc": last["rc"]}
        else:
            entry = {"state": "not_run", "reason": "no attempt ran (driver stopped or refused first)"}
        entry["attempts"] = [{k: a.get(k) for k in ("tag", "state", "rc", "reason", "oom_phase", "oom_step", "log", "receipt")} for a in attempts]
        table[cell] = entry
    completed = {c: e for c, e in table.items() if e["state"] == "completed"}
    if not completed:
        best = None
        reading = "NO READING: no cell completed (" + "; ".join(f"{c}: {e['reason']}" for c, e in table.items()) + ")"
    else:
        best = max(e["steady_tok_per_s_mean_steps_2_on"] for e in completed.values())
        reading = ("streamed path (>= 151 tok/s)" if best >= STREAMED_MIN
                   else "split cell (< 151 tok/s)" if best >= TOKEN_CUT_BELOW else "split cell; tokens may be cut (< 60 tok/s)")
        reading += f" from {', '.join(c for c, e in completed.items() if e['steady_tok_per_s_mean_steps_2_on'] == best)}"
        long_entry = table.get(long_cell)
        if long_entry is not None and long_entry["state"] != "completed":
            reading += f"; long rung failed: {long_cell} {long_entry['reason']}"
    return {"probe": "S0 P2 rerun (DISTILL_DESIGN section 4): T(ii) streamed bf16 fwd+bwd, one card, lr = 0, streamed_kd_trainer 1.3.3",
            "tool": TOOL_NAME, "tool_version": TOOL_VERSION, "gpu_index": gpu, "cells": table,
            "cells_completed": sorted(completed), "cells_failed": sorted(c for c, e in table.items() if e["state"] != "completed"),
            "best_steady_tok_per_s": best,
            "engine_rule": "STAGE_D_RULINGS G5-1: >= 151 tok/s -> streamed; else the measured split cell; < 60 -> token cut",
            "engine_rule_reading": reading, "note": "a reading for the ruling, not the ruling; read only from completed cells"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", required=True, type=int)
    p.add_argument("--cells", required=True, help="planned cells in order, comma-separated")
    p.add_argument("--long-cell", required=True)
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    args = p.parse_args(argv)
    cells = [c for c in args.cells.split(",") if c]
    summary = summarize(args.out, args.gpu, cells, args.long_cell)
    path = args.out / f"p2-summary-v133-gpu{args.gpu}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, path)
    brief = {c: {"state": e["state"], "steady": e.get("steady_tok_per_s_mean_steps_2_on"), "oom_phase": e.get("oom_phase"),
                 "peak_gib": e.get("measured_peak_gib"), "pred_gib": e.get("predicted_peak_gib")} for c, e in summary["cells"].items()}
    print("P2 SUMMARY " + json.dumps(brief) + f" best {summary['best_steady_tok_per_s']} -> {summary['engine_rule_reading']}")
    print(f"P2 SUMMARY FILE {path}")
    return 0 if not summary["cells_failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
