#!/usr/bin/env python3
"""x3_v1_yield_watchdog.py -- the fold pilot (x3_queue.sh, lane L-RESEARCH) yields host RAM and GPU3 to v1.
Written 2026-10-15 by Claude (Chimera Omni lead, research lane).  Standard library only: run with /usr/bin/python3.

Every POLL seconds it reads the v1 lanes' own files and the host, and computes PRESSURE (any one is enough):
  gpu6_wait_ram     the gpu6 queue (bash $Q/gpu6.sh, alive) last printed "<utc> MemAvailable N GiB < M; waiting"
                    (queue_gpu6.sh wait_ram, read from the queue's stdout log via /proc/<pid>/fd/1), not yet followed
                    by its "<utc> MemAvailable N GiB" line
  d1_wait_mem       D1's runtime-status.json (fresh < 1 h) has waiting_on "MemAvailable ... GiB < ... GiB"
                    (d1_hardened_entry's RAM gate)
  post_d1_wait_mem  the last $Q/post_d1.status line of the armed (alive) post_d1_queue.sh says state=waiting|deferred
                    with MemAvailable_<n>GiB_lt_<m>
  d1_seal_wait_mem  the same for $Q/d1_seal.status (d1_seal_queue.sh wait_mem)
  d2_wait_mem       the last $Q/d2.status line of a live d2_launch.sh says "MemAvailable N GiB < M" or "MemAvailable N
                    GiB (need M)" with N < M
  k3_queued_low_mem MemAvailable < 75 GiB while the gpu6 queue is alive and its last $Q/gpu6.status line is a WAIT
                    (a K3-chain job queued behind its gates)
  swap_full         swap used > 90 % of SwapTotal and MemAvailable < 90 GiB
  gpu3_wait         the armed (alive) d1_seal_queue.sh is in its bank v1 review card phase for the P40s: its last B1
                    line is "step=B1-cards state=waiting|retrying", or "state=held placement=p40" not yet followed by
                    "step=B event=released"; or a live post_d1 / d2 status line waits on gpu3.lock / GPU3
On pressure it stops the fold job: writes $X/STOP_QUEUE (marked as its own) and $X/V1_YIELD.json, then SIGTERM to
the x3 queue and to every fold job process (fold_eval.py / x3_ladder.py / qf_build_p40.py / x3_summary.py with $X in
its command line), once per pid.  The trainer saves a resume point and exits 75, the queue writes its release line and
exits at STOP_QUEUE.  It never sends SIGKILL.
RESUME (only a job it stopped itself, STOP_QUEUE still its own): every sample for RESUME_HOLD_S (600 s) shows no
pressure, MemAvailable >= RESUME_GIB (110), no fold process alive, and the d1_seal review not waiting for / holding the
P40s (i.e. finished and released GPU3).  Then it removes its STOP_QUEUE and relaunches `bash x3_queue.sh` (setsid,
>> $X/queue.log), whose own gates (claims, card, gpu3.lock, MemAvailable >= 80) still apply.
CLAIM HYGIENE: x3_queue.sh 1.0.0's claim line says "claim released within ~3 min", which d1_seal_lib.claim_state (the
seal queue's p40-free) reads as a release.  While such a line is the fold lane's live claim (pid alive) it appends a
restated "<utc> CLAIM GPU3 ..." line for the same pid, and it holds $Q/gpu3.lock (flock, never inherited) while the
fold lane's GPU3 claim is live and the queue does not hold the lock itself (1.0.0 takes it only if the file exists).
Events heartbeat lane L-RESEARCH (detail only).  Files: $X/v1_yield_watchdog.{log,state.json,lock}.

    setsid nohup /usr/bin/python3 x3_v1_yield_watchdog.py >> $X/v1_yield_watchdog.out 2>&1 < /dev/null &
    python3 x3_v1_yield_watchdog.py --probe      # print the signals once; act on nothing, take nothing
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

TOOL = ("x3_v1_yield_watchdog", "1.0.0")
OC = "/home/cereal/omni-campaign"
Q = f"{OC}/distill/queues"
R = "/mnt/lexar/omni-campaign/research/fold"
MARK = "v1-yield-watchdog"
FOLD_TOOLS = ("fold_eval.py", "x3_ladder.py", "qf_build_p40.py", "x3_summary.py")
LANE = "fold-research"
MEMWAIT_RE = re.compile(r"MemAvailable_\d+GiB_lt_\d+|MemAvailable \d+(?:\.\d+)? GiB < \d+")
NEED_RE = re.compile(r"MemAvailable (\d+) GiB \(need (\d+)\)")
GPU6_RAM_RE = re.compile(r"^\S+ MemAvailable (\d+) GiB(?: < (\d+); waiting)?$")
PID_RE = re.compile(r"\bpid=(\d+)\b")
GPU3_WAIT_RE = re.compile(r"gpu3\.lock|\bGPU3_\d+MiB\b")


def utc(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(time.time() if epoch is None else epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------------------------------------ host and processes
def meminfo(path: str) -> dict:
    out = {}
    with open(path) as handle:
        for line in handle:
            k, _, v = line.partition(":")
            out[k.strip()] = int(v.split()[0])
    return out


def cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def pids_where(test) -> list[int]:
    me = os.getpid()
    out = []
    for name in os.listdir("/proc"):
        if name.isdigit() and int(name) != me:
            cmd = cmdline(int(name))
            if cmd and TOOL[0] not in cmd and test(cmd):
                out.append(int(name))
    return sorted(out)


def lines_of(path: str, tail_bytes: int = 1 << 16) -> list[str]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - tail_bytes))
            return handle.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def armed_pid(status_path: str, script: str) -> int | None:
    """pid of the last `state=armed ... pid=N` line, if that process is alive and runs `script`."""
    for line in reversed(lines_of(status_path)):
        if "state=armed" in line:
            m = PID_RE.search(line)
            if m and script in cmdline(int(m.group(1))):
                return int(m.group(1))
            return None
    return None


def since_armed(status_path: str) -> list[str]:
    lines = lines_of(status_path, 1 << 20)
    for i in range(len(lines) - 1, -1, -1):
        if "state=armed" in lines[i]:
            return lines[i:]
    return lines


# ------------------------------------------------------------------------------------------------ signals
def seal_gpu3(a) -> tuple[str, str]:
    """('none' | 'waiting' | 'holding', deciding line) for the armed d1_seal_queue.sh and the P40s."""
    if armed_pid(a.seal_status, "d1_seal_queue.sh") is None:
        return "none", ""
    state, why = "none", ""
    for line in since_armed(a.seal_status):
        if re.search(r"step=B1-cards state=(waiting|retrying)", line):
            state, why = "waiting", line
        elif "step=B1-cards state=held" in line:
            state, why = ("holding", line) if "placement=p40" in line else ("none", line)
        elif ("step=B event=released" in line or "step=B1-cards state=gave_up" in line
              or re.search(r"state=(end|stopped)\b", line)):
            state, why = "none", line
    return state, why


def mem_wait_line(line: str) -> bool:
    if MEMWAIT_RE.search(line):
        return bool(re.search(r"state=(waiting|deferred)|\bWAIT\b|waiting", line))
    m = NEED_RE.search(line)
    return bool(m and int(m.group(1)) < int(m.group(2)))


def signals(a, mi: dict) -> dict:
    """name -> reason (str) for every active pressure signal, plus the probe facts under '_facts'."""
    out: dict = {}
    avail = mi.get("MemAvailable", 0) / 1048576
    swap_t, swap_f = mi.get("SwapTotal", 0), mi.get("SwapFree", 0)
    swap_frac = (swap_t - swap_f) / swap_t if swap_t else 0.0
    gpu6 = pids_where(lambda c: c.startswith("bash ") and a.gpu6_script in c)
    # gpu6 queue RAM wait, from its own stdout log
    gpu6_log = ""
    for pid in gpu6:
        try:
            gpu6_log = os.readlink(f"/proc/{pid}/fd/1")
            break
        except OSError:
            continue
    if gpu6_log:
        for line in reversed(lines_of(gpu6_log)):
            m = GPU6_RAM_RE.match(line.strip())
            if m:
                if m.group(2):
                    out["gpu6_wait_ram"] = f"gpu6 queue: {line.strip()[:200]}"
                break
    # D1 entry RAM gate
    try:
        if time.time() - os.stat(a.d1_runtime).st_mtime < 3600:
            with open(a.d1_runtime) as handle:
                w = json.load(handle).get("waiting_on")
            if isinstance(w, str) and w.startswith("MemAvailable"):
                out["d1_wait_mem"] = f"D1 runtime-status waiting_on: {w[:160]}"
    except (OSError, ValueError):
        pass
    # post_d1 / d1_seal / d2 status files
    for name, path, script in (("post_d1", a.post_d1_status, "post_d1_queue.sh"),
                               ("d1_seal", a.seal_status, "d1_seal_queue.sh")):
        if armed_pid(path, script) is None:
            continue
        last = (lines_of(path) or [""])[-1]
        if mem_wait_line(last):
            out[f"{name}_wait_mem"] = f"{name}.status: {last[:200]}"
        if name == "post_d1" and "waiting" in last and GPU3_WAIT_RE.search(last):
            out["gpu3_wait"] = f"post_d1.status: {last[:200]}"
    if pids_where(lambda c: "d2_launch.sh" in c and " run" in c):
        last = (lines_of(a.d2_status) or [""])[-1]
        if mem_wait_line(last):
            out["d2_wait_mem"] = f"d2.status: {last[:200]}"
        if "wait" in last.lower() and GPU3_WAIT_RE.search(last):
            out["gpu3_wait"] = f"d2.status: {last[:200]}"
    # a K3-chain job queued (gpu6.status last line WAIT) while MemAvailable < 75
    if gpu6 and avail < a.k3_queued_gib:
        last = (lines_of(a.gpu6_status) or [""])[-1].split()
        if len(last) >= 3 and last[2] == "WAIT":
            out["k3_queued_low_mem"] = f"MemAvailable {avail:.1f} GiB < {a.k3_queued_gib} while gpu6 job {last[1]} is WAIT"
    if swap_t and swap_frac > a.swap_frac and avail < a.swap_mem_gib:
        out["swap_full"] = f"swap {100 * swap_frac:.1f} % used and MemAvailable {avail:.1f} GiB < {a.swap_mem_gib}"
    st, why = seal_gpu3(a)
    if st != "none":
        out["gpu3_wait"] = f"d1_seal {st}: {why[:200]}"
    out["_facts"] = {"mem_available_gib": round(avail, 2), "swap_used_frac": round(swap_frac, 4), "seal_gpu3": st,
                     "gpu6_queue_pids": gpu6, "gpu6_log": gpu6_log}
    return out


# ------------------------------------------------------------------------------------------------ fold job
def fold_procs(a) -> tuple[list[int], list[int]]:
    queues = pids_where(lambda c: c.startswith("bash ") and a.queue_script in c)
    jobs = pids_where(lambda c: a.x in c and any(t in c for t in FOLD_TOOLS) and a.queue_script not in c)
    return queues, jobs


def fold_claim(a) -> dict:
    """The fold lane's last GPU3 line: live (not a release, pid alive), its pid, and whether d1_seal_lib reads it held."""
    last = None
    for line in lines_of(a.claims, 1 << 20):
        f = line.split()
        if len(f) < 2:
            continue
        head = f[2] if f[1] in ("CLAIM", "RELEASE") and len(f) > 2 else f[1]
        if "GPU3" in head.split(",") and f"lane={LANE}" in line:
            last = line
    if last is None:
        return {"live": False, "pid": None, "line": None, "seal_reads_held": False}
    f = last.split()
    released = f[1] == "RELEASE" or re.search(r"\bjob=\S+ released\b", last) is not None
    m = PID_RE.search(last)
    pid = int(m.group(1)) if m else None
    alive = bool(pid and a.queue_script in cmdline(pid))
    seal_held = not (" RELEASE " in f" {last} " or re.search(r"\breleased\b", last))
    return {"live": (not released) and alive, "pid": pid, "line": last, "seal_reads_held": seal_held}


def lock_holders(path: str) -> list[int]:
    try:
        st = os.stat(path)
        with open("/proc/locks") as handle:
            rows = handle.read().splitlines()
    except OSError:
        return []
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    return [int(r.split()[4]) for r in rows if "->" not in r.split() and len(r.split()) >= 6 and r.split()[5] == key]


class Watchdog:
    def __init__(self, a):
        self.a = a
        self.lock_fd = None
        self.termed: set[int] = set()
        self.ok_since: float | None = None
        self.children: list[subprocess.Popen] = []
        self.last_sig: tuple = ()
        self.noted: set[str] = set()
        self.yield_path = os.path.join(a.x, "V1_YIELD.json")
        self.stop_path = os.path.join(a.x, "STOP_QUEUE")

    def log(self, text: str) -> None:
        line = f"{utc()} {text}"
        print(line, flush=True)
        with open(self.a.log, "a") as handle:
            handle.write(line + "\n")

    def note_once(self, key: str, text: str) -> None:
        if key not in self.noted:
            self.noted.add(key)
            self.log(text)

    def heartbeat(self, detail: str) -> None:
        if not self.a.lanes:
            return
        try:
            subprocess.run(["/usr/bin/python3", self.a.lanes, "heartbeat", "--lane", "L-RESEARCH", "--agent", "claude",
                            "--detail", detail[:480], "--source", f"{TOOL[0]}-{TOOL[1]}"],
                           capture_output=True, timeout=45, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass

    # ---- gpu3.lock on behalf of a live fold claim
    def mirror_lock(self, claim: dict) -> str:
        if not claim["live"]:
            if self.lock_fd is not None:
                fcntl.flock(self.lock_fd, fcntl.LOCK_UN)
                os.close(self.lock_fd)
                self.lock_fd = None
                self.log(f"gpu3.lock released (fold claim ended)")
            return "not held (no live fold claim)"
        if self.lock_fd is not None:
            return f"held by this watchdog (pid {os.getpid()}) for fold pid {claim['pid']}"
        fd = os.open(self.a.lock3, os.O_RDWR | os.O_CREAT, 0o664)  # non-inheritable (PEP 446)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            holders = lock_holders(self.a.lock3)
            # a bash queue locks with flock(1): /proc/locks names that helper's pid, gone while its parent keeps the fd
            alive = [p for p in holders if cmdline(p)]
            fold = [p for p in alive if self.a.queue_script in cmdline(p) or self.a.x in cmdline(p)]
            if len(fold) == len(alive):
                return (f"held by the fold job itself (holder pid {','.join(map(str, holders)) or '?'}"
                        f"{', a flock(1) helper that exited' if len(alive) < len(holders) else ''})")
            self.note_once(f"lock-anomaly-{holders}", f"ANOMALY gpu3.lock held by {holders} "
                           f"({'; '.join(cmdline(p)[:80] for p in holders)}) while the fold claim is live")
            return f"held by non-fold pid {holders}"
        self.lock_fd = fd
        self.log(f"gpu3.lock taken for the live fold claim (fold pid {claim['pid']})")
        return f"held by this watchdog (pid {os.getpid()}) for fold pid {claim['pid']}"

    def restate_claim(self, claim: dict) -> None:
        if claim["live"] and not claim["seal_reads_held"]:
            p = claim["pid"]
            line = (f"{utc()} CLAIM GPU3 owner=claude lane={LANE} job=x3-08b pid={p} log={self.a.x}/queue.log "
                    f"(restated by {TOOL[0]} {TOOL[1]}: the queue's own claim line reads as a release to d1_seal_lib) "
                    f"preempt: kill -TERM {p}")
            with open(self.a.claims, "a") as handle:
                handle.write(line + "\n")
            self.log(f"claim restated for pid {p}: {line}")

    # ---- yield and resume
    def yielded(self) -> dict | None:
        try:
            with open(self.yield_path) as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return None

    def stop_is_ours(self) -> bool:
        try:
            with open(self.stop_path) as handle:
                return handle.read().startswith(MARK)
        except OSError:
            return False

    def do_yield(self, sig: dict, queues: list[int], jobs: list[int]) -> None:
        reasons = {k: v for k, v in sig.items() if not k.startswith("_")}
        y = self.yielded()
        if y is None:
            ours = not os.path.exists(self.stop_path)
            if ours:
                with open(self.stop_path, "w") as handle:
                    handle.write(f"{MARK} {utc()} pid {os.getpid()}: {'; '.join(reasons)}\n")
            y = {"schema": "omni-research.x3-v1-yield.v1", "tool": list(TOOL), "at_utc": utc(), "reasons": reasons,
                 "facts": sig.get("_facts"), "queue_pids": queues, "job_pids": jobs, "stop_queue_ours": ours,
                 "relaunch": ours}
            tmp = self.yield_path + ".tmp"
            with open(tmp, "w") as handle:
                json.dump(y, handle, indent=1)
            os.replace(tmp, self.yield_path)
            self.log(f"YIELD: {json.dumps(reasons)[:600]}; STOP_QUEUE {'written' if ours else 'was already there (foreign: no relaunch)'}")
            self.heartbeat(f"{utc()} watchdog YIELD fold job (queue {queues}, jobs {jobs}): {'; '.join(reasons)}")
        for pid in queues + jobs:
            if pid not in self.termed:
                try:
                    os.kill(pid, signal.SIGTERM)
                    self.log(f"SIGTERM pid {pid}: {cmdline(pid)[:160]}")
                except ProcessLookupError:
                    pass
                self.termed.add(pid)

    def maybe_resume(self, sig: dict, mem: float) -> None:
        y = self.yielded()
        reasons = [k for k in sig if not k.startswith("_")]
        if not y.get("relaunch", True) or not self.stop_is_ours():
            self.note_once("no-relaunch", "STOP_QUEUE is not this watchdog's: the fold job stays stopped (no relaunch)")
            return
        good = not reasons and mem >= self.a.resume_gib
        now = time.time()
        if not good:
            if self.ok_since is not None:
                self.log(f"resume window reset (MemAvailable {mem:.1f} GiB; pressure {reasons})")
            self.ok_since = None
            return
        if self.ok_since is None:
            self.ok_since = now
            self.log(f"resume window opened: MemAvailable {mem:.1f} GiB >= {self.a.resume_gib}, no pressure; "
                     f"relaunch after {self.a.resume_hold_s} s")
            return
        if now - self.ok_since < self.a.resume_hold_s:
            return
        os.remove(self.stop_path)
        with open(self.a.queue_log, "a") as out:
            proc = subprocess.Popen(["bash", self.a.queue_script], stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        self.children.append(proc)
        os.replace(self.yield_path, self.yield_path + f".done-{utc().replace(':', '')}")
        self.ok_since = None
        self.termed.clear()
        self.log(f"RESUME: relaunched {self.a.queue_script} pid {proc.pid} (MemAvailable {mem:.1f} GiB for "
                 f">= {self.a.resume_hold_s} s, no pressure, seal review not on the P40s)")
        self.heartbeat(f"{utc()} watchdog RESUME: fold queue relaunched pid {proc.pid}")

    def step(self) -> dict:
        a = self.a
        for c in list(self.children):
            if c.poll() is not None:
                self.children.remove(c)
        mi = meminfo(a.meminfo)
        sig = signals(a, mi)
        mem = sig["_facts"]["mem_available_gib"]
        queues, jobs = fold_procs(a)
        claim = fold_claim(a)
        self.restate_claim(claim)
        lock_state = self.mirror_lock(fold_claim(a))
        active = tuple(sorted(k for k in sig if not k.startswith("_")))
        if active != self.last_sig:
            self.log(f"pressure {list(active) or 'none'}; MemAvailable {mem:.1f} GiB; swap "
                     f"{100 * sig['_facts']['swap_used_frac']:.1f} %; seal {sig['_facts']['seal_gpu3']}; "
                     + "; ".join(f"{k}: {sig[k]}" for k in active)[:900])
            self.last_sig = active
        y = self.yielded()
        if y is not None and ((y.get("stop_queue_ours") and not self.stop_is_ours())
                              or (not y.get("stop_queue_ours") and not os.path.exists(self.stop_path))):
            # someone removed / replaced the STOP_QUEUE this yield rests on: they took the fold job back
            os.replace(self.yield_path, self.yield_path + f".void-{utc().replace(':', '')}")
            self.log("yield marker VOID: STOP_QUEUE was removed or replaced by someone else; this watchdog will "
                     "not stop or relaunch on that yield (new pressure still yields)")
            y = None
        if active and (queues or jobs):
            self.do_yield(sig, queues, jobs)
        elif y is not None and (queues or jobs):
            self.do_yield(sig, queues, jobs)  # stragglers (a stage started by a queue that had a TERM pending)
        elif y is not None:
            self.maybe_resume(sig, mem)
        state = {"utc": utc(), "tool": list(TOOL), "pid": os.getpid(), "pressure": {k: sig[k] for k in active},
                 "facts": sig["_facts"], "fold_queue_pids": queues, "fold_job_pids": jobs, "fold_claim": claim,
                 "gpu3_lock": lock_state, "yielded": self.yielded() is not None,
                 "resume_window_s": None if self.ok_since is None else round(time.time() - self.ok_since)}
        tmp = a.state + ".tmp"
        with open(tmp, "w") as handle:
            json.dump(state, handle, indent=1)
        os.replace(tmp, a.state)
        return state

    def done(self) -> bool:
        queues, jobs = fold_procs(self.a)
        return (os.path.exists(os.path.join(self.a.x, "SUMMARY.md")) and not queues and not jobs
                and self.yielded() is None)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog=TOOL[0], description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=" ".join(TOOL))
    p.add_argument("--x", default=f"{R}/x3-08b")
    p.add_argument("--queue-script", default=f"{R}/tools/x3_queue.sh")
    p.add_argument("--q", default=Q)
    p.add_argument("--claims")
    p.add_argument("--lock3")
    p.add_argument("--gpu6-script")
    p.add_argument("--gpu6-status")
    p.add_argument("--seal-status")
    p.add_argument("--post-d1-status")
    p.add_argument("--d2-status")
    p.add_argument("--d1-runtime",
                   default="/mnt/lexar/omni-campaign/distill/d1/run-OMNI1D1-hardened-claude-20261014/runtime-status.json")
    p.add_argument("--meminfo", default="/proc/meminfo")
    p.add_argument("--lanes", default=f"{OC}/distill/tools/lanes.py", help="'' disables the L-RESEARCH heartbeat")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--resume-gib", type=float, default=110.0)
    p.add_argument("--resume-hold-s", type=float, default=600.0)
    p.add_argument("--k3-queued-gib", type=float, default=75.0)
    p.add_argument("--swap-frac", type=float, default=0.90)
    p.add_argument("--swap-mem-gib", type=float, default=90.0)
    p.add_argument("--probe", action="store_true", help="print the signals and the fold claim once; act on nothing")
    a = p.parse_args(argv)
    a.claims = a.claims or f"{a.q}/p40.claims"
    a.lock3 = a.lock3 or f"{a.q}/gpu3.lock"
    a.gpu6_script = a.gpu6_script or f"{a.q}/gpu6.sh"
    a.gpu6_status = a.gpu6_status or f"{a.q}/gpu6.status"
    a.seal_status = a.seal_status or f"{a.q}/d1_seal.status"
    a.post_d1_status = a.post_d1_status or f"{a.q}/post_d1.status"
    a.d2_status = a.d2_status or f"{a.q}/d2.status"
    a.log = os.path.join(a.x, "v1_yield_watchdog.log")
    a.state = os.path.join(a.x, "v1_yield_watchdog.state.json")
    a.queue_log = os.path.join(a.x, "queue.log")
    if a.probe:
        sig = signals(a, meminfo(a.meminfo))
        queues, jobs = fold_procs(a)
        print(json.dumps({"pressure": {k: v for k, v in sig.items() if not k.startswith("_")}, "facts": sig["_facts"],
                          "fold_queue_pids": queues, "fold_job_pids": jobs, "fold_claim": fold_claim(a)}, indent=1))
        return 0
    lk = os.open(os.path.join(a.x, "v1_yield_watchdog.lock"), os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another {TOOL[0]} holds {a.x}/v1_yield_watchdog.lock", file=sys.stderr)
        return 5
    w = Watchdog(a)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    w.log(f"ARMED {TOOL[0]} {TOOL[1]} pid {os.getpid()}: x={a.x} queue={a.queue_script} poll={a.poll}s "
          f"resume>={a.resume_gib} GiB for {a.resume_hold_s}s; k3-queued<{a.k3_queued_gib}; swap>{a.swap_frac} & "
          f"mem<{a.swap_mem_gib}; claims={a.claims} lock={a.lock3}")
    w.heartbeat(f"{utc()} watchdog {TOOL[1]} armed pid {os.getpid()} (keys: gpu6 wait_ram, d1/post_d1/d1_seal/d2 "
                f"MemAvailable waits, K3 WAIT & mem<75, swap>90% & mem<90, d1_seal B1 P40 wait/hold)")
    while not stop["now"]:
        try:
            w.step()
        except Exception as exc:  # a bad read must not end the watchdog; it is logged and retried next poll
            w.log(f"ERROR in step: {type(exc).__name__}: {exc}")
        if w.done():
            w.log("fold pipeline DONE (SUMMARY.md, no fold process, no yield): exiting")
            break
        end = time.time() + a.poll
        while not stop["now"] and time.time() < end:
            time.sleep(min(1.0, a.poll))
    if w.lock_fd is not None:
        fcntl.flock(w.lock_fd, fcntl.LOCK_UN)
        os.close(w.lock_fd)
    w.log(f"EXIT pid {os.getpid()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
