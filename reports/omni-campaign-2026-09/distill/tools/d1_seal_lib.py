#!/usr/bin/env python3
"""d1_seal_lib.py -- the decisions $Q/d1_seal_queue.sh and $Q/post_d1_queue.sh take, as testable functions.

Lane d1seal, Claude, 2026-10-14.  Each subcommand reads files, prints one result line and exits with a code the
queue branches on; nothing here starts, stops or locks anything.

  d1-state       --run-dir RUN --unit-state STATE [--unit-result RESULT --unit-load LOADSTATE]
                 0 = the hardened D1 finished and succeeded (status completed / succeeded / exit 0, systemd finalize
                 SERVICE_RESULT=success, launch ids equal, the unit not active), 10 = wait (running, restarting, a
                 failed attempt systemd will restart, no status yet), 2 = down for good (an alert: a refusal or the
                 restart budget spent -- the seal never starts on that).  1.1.0: when the unit is still LOADED its
                 systemd Result is a second witness: anything but `success` beside a "succeeded" status is a
                 contradiction (2).  A garbage-collected unit (LoadState not-found) leaves the finalize record alone.
  phase-a-receipt --out FILE --checkpoint-id ID --queue VERSION --omni1-s7 OMNI1_S7.json KEY=PATH...
                 1.1.0: the phase-A receipt (every KEY=PATH hashed; KEY `s7` is the composed child's s7 receipt) with
                 the R-Q 210 top-1 readings vs the parent and vs OMNI1, and OMNI1's own vs-parent count READ from its
                 s7 receipt (1.0.0's queue wrote a constant).  Diagnostics only.
  sealed-marker  --receipt PHASEA.json --out $Q/d1.sealed
                 1.1.0: line 1 = UTC + "D1 sealed" + the phase-A receipt path + sha256 (what p2p11_card_guard and
                 j6_vl.sh read); then one `<sha256>  <key>  <path>` line per receipt it lists.  Atomic.
  p40-free       --claims FILE --gpus 1,2,3
                 0 = no live claim names any of those GPUs (the last line naming a GPU is a release, or none names it);
                 10 = held (prints the holder lines).  Per lane the last line naming the card decides; a line is a
                 release only when it says RELEASE / released.
  register-check --log DRYRUN.log --checkpoint-id ID --base-id BASE --base-sha256 SHA --replace-ops N --targets N
                 --replaced N --parameters N --parent-parameters N
                 0 = `oc_checkpoint.py register` DRY RUN printed exactly the expected plan (stage distill, base pinned at
                 the sealed digest, replace-only ops, element counts, composed = base parameters); 3 = it did not.
  p2-verdict     --summary p2-summary-v133-gpu0.json --out $Q/p2.verdict
                 the ruling's reading of the P2 cells (STAGE_D_RULINGS G5-1: >= 151 tok/s -> streamed; else the
                 measured split cell; < 60 -> tokens may be cut; NEVER a depth cut), written with the inputs' sha256.
  handoff        --out FILE --lane L --state S --running R --next N --untouchable U
                 the five-line Relay hand-off body (plan section 7.3).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_seal_lib"
TOOL_VERSION = "1.1.0"   # 1.0.0 (2026-10-14): first version, lane d1seal; 1.1.0 (2026-10-14, lane d1seal resume):
                         # d1-state --unit-result / --unit-load, phase-a-receipt, sealed-marker
STREAMED_MIN, TOKEN_CUT_BELOW = 151.0, 60.0
ACTIVE = {"active", "activating", "deactivating", "reloading", "refreshing"}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------------- d1-state
def d1_state(run: Path, unit_state: str, unit_result: str = "", unit_load: str = "") -> tuple[int, str]:
    status, launch = read_json(run / "runtime-status.json"), read_json(run / "launch.json")
    if unit_state in ACTIVE:
        return 10, f"running: unit {unit_state}"
    if not isinstance(status, dict):
        return 10, "waiting: no readable runtime-status.json"
    if status.get("alert"):
        return 2, f"down: {status['alert']}"
    if not status.get("completed"):
        return 10, f"running: status {status.get('state')} (unit {unit_state or 'gone'})"
    ok = status.get("state") == "succeeded" and status.get("exit_code") == 0
    if not ok:
        return 10, (f"waiting: attempt {status.get('attempt')} ended {status.get('state')} exit {status.get('exit_code')} "
                    "with no alert (systemd restarts it)")
    if (status.get("systemd") or {}).get("service_result") != "success":
        return 10, "waiting: succeeded, the ExecStopPost finalize has not recorded SERVICE_RESULT=success yet"
    if not isinstance(launch, dict) or launch.get("launch_id") != status.get("launch_id"):
        return 2, f"down: runtime-status launch {status.get('launch_id')} != launch.json {(launch or {}).get('launch_id')}"
    if unit_load == "loaded" and unit_result and unit_result != "success":
        return 2, (f"down: runtime-status says succeeded but the loaded unit's systemd Result is {unit_result!r} "
                   "(contradiction; a human reads it)")
    witness = f"systemd Result {unit_result}" if unit_load == "loaded" and unit_result else f"unit {unit_load or 'gone'}"
    return 0, f"succeeded: attempt {status.get('attempt')} finished {status.get('finished_at')}, unit {unit_state or 'gone'}, {witness}"


# ---------------------------------------------------------------------------------------------------- phase-a-receipt
def phase_a_receipt(out: Path, checkpoint_id: str, queue: str, omni1_s7: Path, pairs: list[str]) -> dict[str, Any]:
    items: dict[str, dict[str, str]] = {}
    for kv in pairs:
        key, _, path = kv.partition("=")
        if not key or not path:
            raise SystemExit(f"[{TOOL_NAME}] REFUSED: receipt item {kv!r} is not KEY=PATH")
        if not Path(path).is_file():
            raise SystemExit(f"[{TOOL_NAME}] REFUSED: receipt {key} {path} is missing")
        items[key] = {"path": str(path), "sha256": sha256_file(Path(path))}
    if "s7" not in items:
        raise SystemExit(f"[{TOOL_NAME}] REFUSED: no s7=<receipt> item")
    s7 = read_json(Path(items["s7"]["path"])) or {}
    t = s7.get("text_forward") or {}
    o = (read_json(omni1_s7) or {}).get("text_forward") or {}
    doc = {"schema": "omni-campaign.d1-seal-phaseA.v1", "checkpoint_id": checkpoint_id, "queue": queue,
           "tool": f"{TOOL_NAME} {TOOL_VERSION}", "at_utc": utc(), "receipts": items,
           "s7_verdict": s7.get("verdict"), "s7_refusals": s7.get("refusals"),
           "rq_210": {"positions": t.get("positions"), "top1_vs_parent": t.get("rq_top1_omni1_vs_parent"),
                      "top1_vs_omni1": t.get("rq_top1_child_vs_omni1"),
                      "omni1_equivalence_holds": (t.get("omni1_equivalence") or {}).get("holds"),
                      "omni1_top1_vs_parent_at_s7": o.get("rq_top1_omni1_vs_parent"),
                      "omni1_positions_at_s7": o.get("positions"),
                      "omni1_s7_receipt": {"path": str(omni1_s7), "sha256": sha256_file(omni1_s7) if omni1_s7.is_file() else None},
                      "note": "diagnostics (owner policy 2026-10-06): top-1 over the base vocab; the D1 reading 'R-Q vs parent "
                              ">= S8' is recorded, never a refusal"},
           "next": "phase B: bank v1 review (both surfaces) -> record-review -> clone OMNI1D1.clone-pre-D2; GPU0 / GPU4 "
                   "released to post_d1_queue.sh"}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return doc


def sealed_marker(receipt: Path, out: Path) -> str:
    doc = read_json(receipt)
    if not isinstance(doc, dict) or doc.get("schema") != "omni-campaign.d1-seal-phaseA.v1":
        raise SystemExit(f"[{TOOL_NAME}] REFUSED: {receipt} is not a phase-A receipt")
    digest = sha256_file(receipt)
    lines = [f"{utc()} D1 sealed (phase A: verified, registered {doc.get('checkpoint_id')}, composed, s7 PASS) receipt "
             f"{receipt} sha256 {digest}"]
    for key, item in sorted((doc.get("receipts") or {}).items()):
        lines.append(f"{item['sha256']}  {key}  {item['path']}")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return digest


# ---------------------------------------------------------------------------------------------------- p40-free
GPU_RE = re.compile(r"\bGPU(\d+)\b")


LANE_RE = re.compile(r"\blane=(\S+)")


def claim_state(lines: list[str], gpu: int) -> tuple[bool, str | None]:
    """(held, a deciding claim line) for one GPU.  Lanes interleave on one card (2026-10-14: p2p11 claimed and
    released GPU1 inside the toolcall lane's live GPU1 claim), so the last line PER LANE decides that lane's hold,
    and the card is held while any lane's last line for it is not a release."""
    last: dict[str, str] = {}
    for line in lines:
        fields = line.split()
        if len(fields) < 2:
            continue
        head = fields[2] if fields[1] in ("CLAIM", "RELEASE") and len(fields) > 2 else fields[1]
        if gpu in {int(g) for g in GPU_RE.findall(head)}:
            m = LANE_RE.search(line)
            last[m.group(1) if m else "?"] = line
    for line in last.values():
        released = " RELEASE " in f" {line} " or re.search(r"\breleased\b", line) is not None
        if not released:
            return True, line
    return False, None


def p40_free(claims: Path, gpus: list[int]) -> tuple[int, list[str]]:
    lines = claims.read_text(encoding="utf-8", errors="replace").splitlines() if claims.is_file() else []
    held = []
    for g in gpus:
        busy, line = claim_state(lines, g)
        if busy:
            held.append(f"GPU{g}: {line}")
    return (10 if held else 0), held


# ---------------------------------------------------------------------------------------------------- register-check
def register_check(log: str, a: argparse.Namespace) -> list[str]:
    def num(n: int) -> str:
        return f"{int(n):,}"

    want = [
        rf"=== REGISTER {re.escape(a.checkpoint_id)} ===",
        r"stage\s+distill$",
        rf"directory_sha256\s+[0-9a-f]{{64}}$",
        rf"parameters\s+{re.escape(num(a.parameters))}\s+\[derived-from-safetensors-headers\]",
        rf"parent parameters\s+{re.escape(num(a.parent_parameters))}\s+\({re.escape(a.parent_code)}\)\s+\[derived-from-safetensors-headers\]",
        rf"delta base\s+{re.escape(a.base_id)} pinned at {a.base_sha256[:16]} \(matches its seal\)",
        rf"delta chain\s+.*{re.escape(a.base_id)} -> {re.escape(a.checkpoint_id)}$",
        rf"delta ops\s+\{{'replace': {a.replace_ops}, 'lora': 0, 'scale': 0\}} on {a.targets} target tensor\(s\); "
        rf"replaced {re.escape(num(a.replaced))}, added 0$",
        r"status\s+candidate\b",
        r"DRY RUN: nothing was written\.",
    ]
    lines = [ln.strip() for ln in log.splitlines()]
    return [w for w in want if not any(re.search(w, ln) for ln in lines)]


# ---------------------------------------------------------------------------------------------------- p2-verdict
def p2_verdict(summary_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        return {"verdict": "none", "reason": f"no readable P2 summary at {summary_path}"}
    cells = summary.get("cells") or {}
    rates = {c: e.get("steady_tok_per_s_mean_steps_2_on") for c, e in cells.items()
             if e.get("state") == "completed" and isinstance(e.get("steady_tok_per_s_mean_steps_2_on"), (int, float))}
    if not rates:
        return {"verdict": "none", "reason": "no P2 cell completed: " + "; ".join(f"{c}: {e.get('reason')}" for c, e in cells.items()),
                "cells": {c: e.get("state") for c, e in cells.items()}}
    best_cell = max(rates, key=lambda c: rates[c])
    best = float(rates[best_cell])
    engine = "streamed" if best >= STREAMED_MIN else "split"
    long_cell = cells.get("L32768-mb1") or {}
    return {"verdict": engine, "best_steady_tok_per_s": best, "best_cell": best_cell, "rates": rates,
            "token_cut_allowed": best < TOKEN_CUT_BELOW, "depth_cut": False,
            "long_rung": long_cell.get("state"), "long_rung_reason": long_cell.get("reason"),
            "rule": "STAGE_D_RULINGS G5-1 / D-R5-1: >= 151 tok/s -> the streamed path (GPU0 + GPU4 bf16 DP); otherwise the "
                    "measured split cell (GPU6 + GPU4, fp16 LoRA); no depth cut; tokens are cut only below 60 tok/s",
            "standing_go": "applied automatically under Cyril's standing GO; recorded for review, nobody has to be awake"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("d1-state")
    s.add_argument("--run-dir", required=True, type=Path)
    s.add_argument("--unit-state", default="")
    s.add_argument("--unit-result", default="")
    s.add_argument("--unit-load", default="")
    s = sub.add_parser("phase-a-receipt")
    s.add_argument("--out", required=True, type=Path)
    s.add_argument("--checkpoint-id", required=True)
    s.add_argument("--queue", required=True)
    s.add_argument("--omni1-s7", required=True, type=Path)
    s.add_argument("items", nargs="+", metavar="KEY=PATH")
    s = sub.add_parser("sealed-marker")
    s.add_argument("--receipt", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("p40-free")
    s.add_argument("--claims", required=True, type=Path)
    s.add_argument("--gpus", required=True)
    s = sub.add_parser("register-check")
    s.add_argument("--log", required=True, type=Path)
    for k in ("checkpoint-id", "base-id", "base-sha256"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--parent-code", default="Q38-27BU")
    for k in ("replace-ops", "targets", "replaced", "parameters", "parent-parameters"):
        s.add_argument(f"--{k}", required=True, type=int)
    s = sub.add_parser("p2-verdict")
    s.add_argument("--summary", required=True, type=Path)
    s.add_argument("--out", required=True, type=Path)
    s = sub.add_parser("handoff")
    for k in ("out", "lane", "state", "running", "next", "untouchable"):
        s.add_argument(f"--{k}", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "d1-state":
        code, line = d1_state(args.run_dir, args.unit_state.strip(), args.unit_result.strip(), args.unit_load.strip())
        print(f"D1 STATE {line}")
        return code
    if args.cmd == "phase-a-receipt":
        doc = phase_a_receipt(args.out, args.checkpoint_id, args.queue, args.omni1_s7, args.items)
        print(f"PHASE A RECEIPT {args.out} sha256 {sha256_file(args.out)}; R-Q top-1 vs parent "
              f"{doc['rq_210']['top1_vs_parent']}/{doc['rq_210']['positions']} (OMNI1 at S7: "
              f"{doc['rq_210']['omni1_top1_vs_parent_at_s7']}/{doc['rq_210']['omni1_positions_at_s7']}), vs OMNI1 "
              f"{doc['rq_210']['top1_vs_omni1']}")
        return 0
    if args.cmd == "sealed-marker":
        digest = sealed_marker(args.receipt, args.out)
        print(f"SEALED MARKER {args.out} (receipt {args.receipt} sha256 {digest})")
        return 0
    if args.cmd == "p40-free":
        code, held = p40_free(args.claims, [int(g) for g in args.gpus.split(",") if g.strip()])
        print("P40 FREE" if code == 0 else "P40 HELD " + " | ".join(h[:220] for h in held))
        return code
    if args.cmd == "register-check":
        missing = register_check(args.log.read_text(encoding="utf-8", errors="replace"), args)
        if missing:
            print("REGISTER CHECK FAILED: the dry run did not print " + " | ".join(missing))
            return 3
        print(f"REGISTER CHECK OK: {args.checkpoint_id} on {args.base_id} @{args.base_sha256[:16]}, {args.replace_ops} replace "
              f"ops on {args.targets} tensors, {args.replaced:,} elements, {args.parameters:,} parameters")
        return 0
    if args.cmd == "p2-verdict":
        doc = p2_verdict(args.summary)
        doc.update({"tool": f"{TOOL_NAME} {TOOL_VERSION}", "at_utc": utc(), "summary": str(args.summary),
                    "summary_sha256": sha256_file(args.summary) if args.summary.is_file() else None})
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(f"{doc['verdict']} {utc()} best {doc.get('best_steady_tok_per_s')} tok/s ({doc.get('best_cell')})\n"
                       + json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, args.out)
        print(f"P2 VERDICT {doc['verdict']} best {doc.get('best_steady_tok_per_s')} -> {args.out}")
        return 0
    lines = [f"1. lane: {args.lane}", f"2. state: {args.state}", f"3. running: {args.running}",
             f"4. next: {args.next}", f"5. do not touch: {args.untouchable}"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"HANDOFF {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
