#!/usr/bin/env python3
"""Exact process control for remote Neural Forge experiment jobs.

This exists to avoid broad ``pgrep -f`` / ``kill`` one-liners that can match the
SSH-launched shell carrying the control command itself. Tag matching is limited
to Neural Forge CLI processes; exact PID matching can be further constrained
with ``--expect-contains``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


NEURAL_FORGE_FOLD_MARKER = "-m neural_forge.cli hf-xfamily-chain-anchor-compress"


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    ppid: int
    stat: str
    command: str
    args: str


def parse_ps_rows(text: str) -> list[ProcessRow]:
    rows: list[ProcessRow] = []
    for line in text.splitlines():
        parts = line.strip().split(maxsplit=4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        rows.append(
            ProcessRow(
                pid=pid,
                ppid=ppid,
                stat=parts[2],
                command=parts[3],
                args=parts[4],
            )
        )
    return rows


def read_ps_rows() -> list[ProcessRow]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,stat=,comm=,args="],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return parse_ps_rows(result.stdout)


def ancestor_pids(pid: int | None = None) -> set[int]:
    current = int(pid or os.getpid())
    protected = {current}
    while True:
        stat_path = Path(f"/proc/{current}/stat")
        if not stat_path.exists():
            break
        try:
            fields = stat_path.read_text(encoding="utf-8").split()
            parent = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
        if parent <= 1 or parent in protected:
            break
        protected.add(parent)
        current = parent
    return protected


def read_pid_files(paths: Iterable[Path]) -> set[int]:
    pids: set[int] = set()
    for path in paths:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        token = "".join(ch if ch.isdigit() else " " for ch in raw).split()
        if token:
            pids.add(int(token[0]))
    return pids


def is_neural_forge_fold(row: ProcessRow) -> bool:
    return (
        "python" in row.command.lower()
        and NEURAL_FORGE_FOLD_MARKER in row.args
    )


def select_processes(
    rows: Iterable[ProcessRow],
    *,
    pids: set[int],
    tag: str | None,
    expect_contains: tuple[str, ...],
    protected_pids: set[int],
) -> list[ProcessRow]:
    selected: list[ProcessRow] = []
    for row in rows:
        if row.pid in protected_pids or row.ppid in protected_pids:
            continue
        matched = False
        if row.pid in pids:
            matched = True
        if tag and is_neural_forge_fold(row) and tag in row.args:
            matched = True
        if not matched:
            continue
        if expect_contains and not all(fragment in row.args for fragment in expect_contains):
            continue
        selected.append(row)
    return selected


def signal_for_action(action: str) -> int | None:
    if action == "list":
        return None
    return {
        "term": signal.SIGTERM,
        "kill": signal.SIGKILL,
        "stop": signal.SIGSTOP,
        "cont": signal.SIGCONT,
    }[action]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "term", "kill", "stop", "cont"))
    parser.add_argument("--pid", action="append", type=int, default=[])
    parser.add_argument("--pid-file", action="append", type=Path, default=[])
    parser.add_argument("--tag")
    parser.add_argument("--expect-contains", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-match", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    args = parser.parse_args(argv)

    pids = set(args.pid) | read_pid_files(args.pid_file)
    if not pids and not args.tag:
        parser.error("provide --pid, --pid-file, or --tag")

    protected = ancestor_pids()
    rows = select_processes(
        read_ps_rows(),
        pids=pids,
        tag=args.tag,
        expect_contains=tuple(args.expect_contains),
        protected_pids=protected,
    )
    sig = signal_for_action(args.action)
    signaled: list[int] = []
    if sig is not None and not args.dry_run:
        for row in rows:
            os.kill(row.pid, sig)
            signaled.append(row.pid)
        if args.wait_seconds > 0 and args.action in {"term", "kill"}:
            deadline = time.monotonic() + args.wait_seconds
            while time.monotonic() < deadline:
                alive = {row.pid for row in read_ps_rows()}
                if not any(pid in alive for pid in signaled):
                    break
                time.sleep(0.25)

    payload = {
        "action": args.action,
        "dry_run": args.dry_run,
        "protected_pids": sorted(protected),
        "matched": [asdict(row) for row in rows],
        "signaled": signaled,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"matched={len(rows)} action={args.action} dry_run={int(args.dry_run)}")
        for row in rows:
            print(f"{row.pid}\t{row.stat}\t{row.command}\t{row.args}")
    if args.require_match and not rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
