#!/usr/bin/env python3
"""d1_hardened_entry.py -- the process that runs D1 on the server (state doc 2026-10-14 step 4a).

Lane d1, Claude, 2026-10-14.  Replaces Codex's allocator attestation (a Windows lease monitor refreshing a 45 s
proof over SSH; every monitor death killed the trainer) with things that live and die with the trainer itself:

  * the card locks ($Q/gpu0.lock, $Q/gpu4.lock -- the locks gpu0.sh / d1_launch.sh use) are taken with flock
    BEFORE the trainer starts and held by this process until it exits; the kernel drops them on any exit,
    including SIGKILL / OOM.  A busy lock is waited for (--lock-wait-s, default 30 min), then exit 4;
  * --gpu-uuid: lanes that ignore the flock (Codex's allocator containers) are seen through nvidia-smi: the
    trainer starts only when its cards carry no other compute process, within the same wait budget, else exit 4;
  * runtime-status.json (atomic) carries the attempt, state, exit code, error, a heartbeat (time, the trainer's
    PROGRESS layer / phase, host + cgroup memory) and the history of earlier attempts;
  * ``--finalize`` is the systemd ExecStopPost hook: it records SERVICE_RESULT / EXIT_CODE / EXIT_STATUS, and
    when the process died without writing its exit state (SIGKILL, OOM kill) it writes it.

Nothing here talks to PostgreSQL, Docker, Windows or SSH.  The launcher (d1_hardened_launch.sh) runs this under
``systemd-run --user`` (a cgroup with MemoryMax, outside any SSH session, Restart=on-failure).

Exit codes: 0 done; 1 error (restartable: CUDA OOM, I/O, anything unexpected); 3 refused (a functional refusal
or a bad argument -- systemd's RestartPreventExitStatus, a human looks); 4 card lock not obtained in time
(restartable); 143 SIGTERM.

    d1_hardened_entry.py --status RUN/runtime-status.json --lock Q/gpu0.lock --lock Q/gpu4.lock -- <d1 args>
    d1_hardened_entry.py --finalize --status RUN/runtime-status.json
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

TOOL_NAME = "d1_hardened_entry"
TOOL_VERSION = "1.1.0"   # 1.0.0 (2026-10-14): first version, lane d1 (state doc step 4a); 1.1.0: --gpu-uuid
                         # (Codex relaunched D1 as r5/r6 on GPU0+GPU4 without the flock, 2026-10-14 15:29Z/15:58Z)
SCHEMA = "omni-campaign.d1-runtime-status.v1"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_LOCK_TIMEOUT, EXIT_TERM = 0, 1, 3, 4, 143
HERE = Path(__file__).resolve().parent


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [{TOOL_NAME}] {message}", flush=True)


class Status:
    """runtime-status.json, rewritten atomically under a lock (main thread + heartbeat thread)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.doc: dict[str, Any] = {}

    def begin(self, argv: list[str], locks: list[str]) -> None:
        previous = self.read()
        history = list(previous.pop("history", []) or []) if previous else []
        if previous:
            history.append(previous)
        self.doc = {"schema": SCHEMA, "tool": [TOOL_NAME, TOOL_VERSION], "attempt": len(history) + 1,
                    "execution_id": str(uuid.uuid4()), "state": "starting", "completed": False, "exit_code": None,
                    "pid": os.getpid(), "host": socket.gethostname(), "invocation_id": os.environ.get("INVOCATION_ID"),
                    "started_at": utc(), "finished_at": None, "heartbeat_at": utc(), "progress": {}, "memory": {},
                    "locks": locks, "argv": argv, "error": None, "traceback_tail": None, "history": history}
        self.write()

    def read(self) -> dict[str, Any] | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def update(self, **fields: Any) -> None:
        with self.lock:
            self.doc.update(fields)
            self._write()

    def write(self) -> None:
        with self.lock:
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self.doc, indent=1, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, self.path)


def take_locks(paths: list[str], wait_s: float, status: Status, poll_s: float = 15.0) -> list[int] | None:
    """flock(LOCK_EX) every path, in order, non-blocking with polling; the fds stay open for the process lifetime.
    None after ``wait_s`` without all of them (whatever was taken is released)."""
    fds: list[int] = []
    deadline = time.time() + wait_s
    for p in paths:
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o664)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() >= deadline:
                    os.close(fd)
                    for held in fds:
                        os.close(held)
                    return None
                status.update(state="waiting_for_locks", heartbeat_at=utc(), waiting_on=p)
                time.sleep(min(poll_s, max(0.1, deadline - time.time())))
        fds.append(fd)
    return fds


def foreign_gpu_processes(uuids: list[str]) -> list[str] | None:
    """Compute processes on the given cards other than this one (``pid@uuid``); None when nvidia-smi fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    busy = []
    for line in out.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 2 and parts[1] in uuids and parts[0] != str(os.getpid()):
            busy.append(f"{parts[0]}@{parts[1]}")
    return busy


def wait_for_free_cards(uuids: list[str], deadline: float, status: Status, poll_s: float) -> list[str] | None:
    """[] once the cards are free of foreign compute processes; the last busy list (or None: nvidia-smi failed)
    at the deadline."""
    while True:
        busy = foreign_gpu_processes(uuids)
        if busy == []:
            return []
        if time.time() >= deadline:
            return busy
        status.update(state="waiting_for_cards", heartbeat_at=utc(), waiting_on=busy if busy is not None else "nvidia-smi failed")
        time.sleep(min(poll_s, max(0.1, deadline - time.time())))


class Terminated(BaseException):
    pass


def _on_term(signum: int, frame: Any) -> None:
    raise Terminated(f"signal {signum}")


def heartbeat(status: Status, every_s: float, stop: threading.Event) -> None:
    import d1_trunk_recovery as d1

    while not stop.wait(every_s):
        try:
            status.update(heartbeat_at=utc(), progress=dict(d1.PROGRESS), memory=d1.host_memory())
        except Exception as error:  # noqa: BLE001 - a heartbeat failure must never touch the run
            log(f"heartbeat write failed: {error!r}")


def run(args: argparse.Namespace, trainer_argv: list[str]) -> int:
    status = Status(Path(args.status).absolute())
    status.begin(trainer_argv, args.lock)
    log(f"attempt {status.doc['attempt']} pid {os.getpid()} execution {status.doc['execution_id']}; locks {args.lock}")
    signal.signal(signal.SIGTERM, _on_term)
    deadline = time.time() + float(args.lock_wait_s)
    fds = take_locks(args.lock, float(args.lock_wait_s), status, float(args.lock_poll_s))
    if fds is not None and args.gpu_uuid:
        busy = wait_for_free_cards(args.gpu_uuid, deadline, status, float(args.lock_poll_s))
        if busy != []:
            for fd in fds:
                os.close(fd)
            status.update(state="gpu_busy", completed=True, exit_code=EXIT_LOCK_TIMEOUT, finished_at=utc(),
                          error=f"cards {args.gpu_uuid} still carry foreign compute processes {busy} after {args.lock_wait_s}s")
            log(f"cards busy ({busy}) after {args.lock_wait_s}s; exit {EXIT_LOCK_TIMEOUT}")
            return EXIT_LOCK_TIMEOUT
    if fds is None:
        status.update(state="lock_timeout", completed=True, exit_code=EXIT_LOCK_TIMEOUT, finished_at=utc(),
                      error=f"card locks {args.lock} not free within {args.lock_wait_s}s")
        log(f"card locks not free within {args.lock_wait_s}s; exit {EXIT_LOCK_TIMEOUT}")
        return EXIT_LOCK_TIMEOUT
    status.update(state="running", locks_held_at=utc(), waiting_on=None)
    log(f"card locks held: {args.lock}")
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat, args=(status, float(args.heartbeat_s), stop), daemon=True)
    beat.start()
    code, state, error, tail = EXIT_OK, "succeeded", None, None
    try:
        import d1_trunk_recovery as d1

        d1.run(trainer_argv)
    except Terminated as exc:
        code, state, error = EXIT_TERM, "terminated", str(exc)
    except SystemExit as exc:   # d1.Refusal (functional) or argparse: a human looks, systemd does not restart
        code, state, error = EXIT_REFUSED, "refused", str(exc.code if exc.code is not None else exc)
        tail = traceback.format_exc()[-4000:]
    except BaseException as exc:  # noqa: BLE001 - CUDA OOM, I/O, bugs: restartable
        code, state, error = EXIT_ERROR, "failed", f"{type(exc).__name__}: {exc}"
        tail = traceback.format_exc()[-4000:]
    finally:
        stop.set()
    try:
        import d1_trunk_recovery as d1

        last = {"progress": dict(d1.PROGRESS), "memory": d1.host_memory()}
    except Exception:  # noqa: BLE001
        last = {}
    status.update(state=state, completed=True, exit_code=code, finished_at=utc(), error=error, traceback_tail=tail, **last)
    log(f"trainer ended: {state} (exit {code}){'; ' + error if error else ''}")
    for fd in fds:
        os.close(fd)
    return code


def finalize(status_path: Path) -> int:
    """systemd ExecStopPost: SERVICE_RESULT / EXIT_CODE / EXIT_STATUS into the status; the exit state itself when
    the process never wrote one (SIGKILL / OOM kill / a crash before the except clauses)."""
    status = Status(status_path)
    doc = status.read() or {"schema": SCHEMA, "tool": [TOOL_NAME, TOOL_VERSION], "history": []}
    result = {"service_result": os.environ.get("SERVICE_RESULT"), "exit_code": os.environ.get("EXIT_CODE"),
              "exit_status": os.environ.get("EXIT_STATUS"), "at": utc(), "invocation_id": os.environ.get("INVOCATION_ID")}
    doc["systemd"] = result
    if not doc.get("completed"):
        raw = result["exit_status"]
        doc.update(completed=True, finished_at=utc(),
                   state="killed" if result["exit_code"] in ("killed", "dumped") else "failed",
                   exit_code=int(raw) if raw and raw.lstrip("-").isdigit() else raw,
                   error=f"the process ended without writing its exit state: SERVICE_RESULT={result['service_result']} "
                         f"EXIT_CODE={result['exit_code']} EXIT_STATUS={raw}")
    status.doc = doc
    status.write()
    log(f"finalize: {result}; state {doc.get('state')} exit {doc.get('exit_code')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    trainer_argv: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, trainer_argv = argv[:cut], argv[cut + 1:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--status", required=True)
    ap.add_argument("--lock", action="append", default=[], help="card lock file, flock'ed for the process lifetime")
    ap.add_argument("--gpu-uuid", action="append", default=[], help="card that must carry no foreign compute process")
    ap.add_argument("--lock-wait-s", type=float, default=1800.0)
    ap.add_argument("--lock-poll-s", type=float, default=15.0)
    ap.add_argument("--heartbeat-s", type=float, default=60.0)
    ap.add_argument("--finalize", action="store_true", help="systemd ExecStopPost mode")
    args = ap.parse_args(argv)
    if args.finalize:
        return finalize(Path(args.status).absolute())
    if not trainer_argv:
        ap.error("the trainer arguments follow --")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return run(args, trainer_argv)


if __name__ == "__main__":
    sys.exit(main())
