#!/usr/bin/env python3
"""d1_hardened_entry.py -- the process that runs D1 on the server (state doc 2026-10-14 step 4a).

Lane d1, Claude, 2026-10-14.  Replaces Codex's allocator attestation (a Windows lease monitor refreshing a 45 s
proof over SSH; every monitor death killed the trainer) with things that live and die with the trainer itself:

  * the card locks ($Q/gpu0.lock, $Q/gpu4.lock -- the locks gpu0.sh / d1_launch.sh use) are taken with flock
    BEFORE the trainer starts and held by this process until it exits; the kernel drops them on any exit,
    including SIGKILL / OOM.  A busy lock is waited for (--lock-wait-s, default 30 min), then exit 4;
  * --gpu-uuid: lanes that ignore the flock (Codex's allocator containers) are seen through nvidia-smi: the
    trainer starts only when its cards carry no other compute process, within the same wait budget, else exit 4;
  * --min-mem-available-gib (1.2.0): with locks and cards held, the trainer starts only once host MemAvailable is
    at least that much (D1 allocates ~82 GB of P/C streams + ~10 GB of model state at once).  Checked on EVERY
    start, so a systemd restart after a CUDA OOM waits for a K3 pass / HostResidentReader job that started in the
    gap instead of piling onto it.  Waited for up to --mem-wait-s (default 6 h), then exit 4;
  * runtime-status.json (atomic) carries the attempt, state, exit code, error, a heartbeat (time, the trainer's
    PROGRESS layer / phase, host + cgroup memory) and the history of earlier attempts;
  * ``--finalize`` is the systemd ExecStopPost hook: it records SERVICE_RESULT / EXIT_CODE / EXIT_STATUS, and
    when the process died without writing its exit state (SIGKILL, OOM kill) it writes it.  1.2.0: when systemd
    will NOT bring the trainer back -- a refusal (exit 3, RestartPreventExitStatus) or the launch's start budget
    spent (--start-limit-burst starts of this launch, D1_LAUNCH_ID, within --start-limit-interval-s) -- it writes
    ``alert`` into the status, an ``ALERT`` log line and a JSON line to --alert-file: D1 is down and a human looks.

Nothing here talks to PostgreSQL, Docker, Windows or SSH.  The launcher (d1_hardened_launch.sh) runs this under
a system-level transient unit (``sudo systemd-run --uid``: system.slice, a cgroup with MemoryMax, outside any SSH
session and outside systemd-oomd's user@1000.service pressure kill, Restart=on-failure).

Exit codes: 0 done; 1 error (restartable: CUDA OOM, I/O, anything unexpected); 3 refused (a functional refusal
or a bad argument -- systemd's RestartPreventExitStatus, a human looks); 4 card lock / free cards / host memory
not obtained in time (restartable); 143 SIGTERM.

    d1_hardened_entry.py --status RUN/runtime-status.json --lock Q/gpu0.lock --lock Q/gpu4.lock -- <d1 args>
    d1_hardened_entry.py --finalize --status RUN/runtime-status.json
"""
from __future__ import annotations

import argparse
import calendar
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
TOOL_VERSION = "1.2.0"   # 1.0.0 (2026-10-14): first version, lane d1 (state doc step 4a); 1.1.0: --gpu-uuid
                         # (Codex relaunched D1 as r5/r6 on GPU0+GPU4 without the flock, 2026-10-14 15:29Z/15:58Z)
                         # 1.2.0 (review): MemAvailable gate on every start (--min-mem-available-gib, --mem-wait-s);
                         # finalize alerts when systemd will not restart (refusal / start budget spent)
SCHEMA = "omni-campaign.d1-runtime-status.v1"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_LOCK_TIMEOUT, EXIT_TERM = 0, 1, 3, 4, 143
HERE = Path(__file__).resolve().parent
MEMINFO = Path("/proc/meminfo")


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
                    "launch_id": os.environ.get("D1_LAUNCH_ID"),
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


def mem_available_gib(path: Path | None = None) -> float | None:
    """Host MemAvailable in GiB from /proc/meminfo; None when unreadable."""
    try:
        for line in (path or MEMINFO).read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / 2 ** 30
    except (OSError, ValueError, IndexError):
        pass
    return None


def wait_for_memory(min_gib: float, wait_s: float, status: Status, poll_s: float) -> float | None:
    """The MemAvailable reading once it is >= ``min_gib``; None after ``wait_s`` without it (an unreadable meminfo
    counts as not enough: the gate never opens on a guess)."""
    deadline = time.time() + wait_s
    while True:
        avail = mem_available_gib()
        if avail is not None and avail >= min_gib:
            return avail
        if time.time() >= deadline:
            return None
        status.update(state="waiting_for_memory", heartbeat_at=utc(),
                      waiting_on=f"MemAvailable {avail if avail is None else round(avail, 1)} GiB < {min_gib} GiB")
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
    if args.min_mem_available_gib > 0:   # 1.2.0: on every start, restarts included; the card locks stay held
        avail = wait_for_memory(float(args.min_mem_available_gib), float(args.mem_wait_s), status, float(args.lock_poll_s))
        if avail is None:
            for fd in fds:
                os.close(fd)
            status.update(state="memory_low", completed=True, exit_code=EXIT_LOCK_TIMEOUT, finished_at=utc(),
                          error=f"host MemAvailable {mem_available_gib()} GiB stayed below {args.min_mem_available_gib} "
                                f"GiB for {args.mem_wait_s}s")
            log(f"MemAvailable below {args.min_mem_available_gib} GiB for {args.mem_wait_s}s; exit {EXIT_LOCK_TIMEOUT}")
            return EXIT_LOCK_TIMEOUT
        status.update(mem_available_at_start_gib=round(avail, 2), waiting_on=None)
        log(f"MemAvailable {avail:.1f} GiB >= {args.min_mem_available_gib} GiB")
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


def _epoch(stamp: Any) -> float | None:
    try:
        return float(calendar.timegm(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")))
    except (TypeError, ValueError):
        return None


def restart_alert(doc: dict[str, Any], burst: int, interval_s: float, now: float) -> str | None:
    """Why systemd will not start this trainer again (None: it will, or it is done).  Only attempts that wrote a
    status count toward the start budget: a crash before Status.begin is not seen (undercount -> a late alert)."""
    if doc.get("state") == "succeeded" and doc.get("exit_code") == EXIT_OK:
        return None
    if doc.get("exit_code") == EXIT_REFUSED:
        return f"refused (exit {EXIT_REFUSED}, RestartPreventExitStatus): systemd will not restart; {doc.get('error')}"
    if burst > 0:
        launch = doc.get("launch_id")
        starts = [a for a in [*(doc.get("history") or []), doc]
                  if isinstance(a, dict) and launch is not None and a.get("launch_id") == launch
                  and (t := _epoch(a.get("started_at"))) is not None and now - t <= interval_s]
        if len(starts) >= burst:
            return (f"start budget spent: {len(starts)} starts of launch {launch} within {interval_s:.0f}s "
                    f"(StartLimitBurst {burst}): systemd will not restart; last: {doc.get('state')} exit "
                    f"{doc.get('exit_code')} {doc.get('error')}")
    return None


def finalize(status_path: Path, burst: int = 0, interval_s: float = 86400.0, alert_file: str | None = None) -> int:
    """systemd ExecStopPost: SERVICE_RESULT / EXIT_CODE / EXIT_STATUS into the status; the exit state itself when
    the process never wrote one (SIGKILL / OOM kill / a crash before the except clauses); an alert when systemd
    will not restart it."""
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
    alert = restart_alert(doc, burst, interval_s, time.time())
    doc["alert"] = alert
    status.doc = doc
    status.write()
    log(f"finalize: {result}; state {doc.get('state')} exit {doc.get('exit_code')}")
    if alert:
        log(f"ALERT D1 is down: {alert}")
        if alert_file:
            try:
                with open(alert_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"at": utc(), "status": str(status_path), "attempt": doc.get("attempt"),
                                             "launch_id": doc.get("launch_id"), "alert": alert}, default=str) + "\n")
            except OSError as error:
                log(f"alert file {alert_file} not written: {error!r}")
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
    ap.add_argument("--min-mem-available-gib", type=float, default=0.0,
                    help="start only once host MemAvailable >= this (0: no gate; the launcher passes 100)")
    ap.add_argument("--mem-wait-s", type=float, default=21600.0)
    ap.add_argument("--finalize", action="store_true", help="systemd ExecStopPost mode")
    ap.add_argument("--start-limit-burst", type=int, default=0, help="finalize: the unit's StartLimitBurst (0: no check)")
    ap.add_argument("--start-limit-interval-s", type=float, default=86400.0)
    ap.add_argument("--alert-file", default=None, help="finalize: JSON lines appended when D1 will not come back")
    args = ap.parse_args(argv)
    if args.finalize:
        return finalize(Path(args.status).absolute(), args.start_limit_burst, args.start_limit_interval_s, args.alert_file)
    if not trainer_argv:
        ap.error("the trainer arguments follow --")
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return run(args, trainer_argv)


if __name__ == "__main__":
    sys.exit(main())
