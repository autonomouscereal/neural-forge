#!/usr/bin/env python3
"""lanes.py -- the Stage D lane registry, $OC/distill/queues/LANES.json (card X-T2; STAGE_D_RECONCILED_PLAN_2026-10-14
sections 7.1 lane ownership, 7.2 single sources of truth, 7.4 locks and takeover).  Written 2026-10-14 by Claude, lane
infra.  Standard library only: queues call it with /usr/bin/python3.

Cyril alternates between Claude and Codex as usage runs out, so no lane may depend on an agent being awake.  Queues
run unattended and hold their own locks; this file is where either agent reads who owns a lane, which locks it holds,
whether its process is alive, its newest receipt, and whether it may be taken over.

Commands (exit codes: 0 ok, 2 refused input, 3 not the lane's owner, 4 takeover refused, 5 lock busy, 6 registry
missing or corrupt):
  show       [--lane L] [--json]   every lane: owner, status, heartbeat age, stale, takeover rule, lock holders (read
             from /proc/locks, never by taking the lock), open GPU claims in its *.claims lock paths, newest receipt
  heartbeat  --lane L --agent A [--status S] [--detail T] [--receipt PATH] [--transferable yes|no] [--source S]
             the lane's OWNER only; a receipt is stored with its sha256
  pulse      [--dry-run]  for every lane with a probe (systemd unit active / pid cmdline alive / file fresh, all must
             pass): heartbeat it on the owner's behalf and record its newest receipt from the probe's receipt globs.
             Run by cron every 10 min, so a lane whose queue runs unattended stays fresh while its process lives and
             goes stale once it stops.  A failing probe never heartbeats; it is recorded in probe_last.  A probe may
             carry status_ok / status_down: the lane's status follows the probe (a lane with no heartbeating owner
             would otherwise keep showing "idle" while its job runs).  1.1.1: the probe's line is MERGED with the
             owner's own status line (status_note: the text of the last heartbeat --status / register / takeover,
             with its utc, author and the probe verdict it was written under) as "<probe line> | <by> <utc>: <note>",
             never replacing it; the note is retired to status_note_dropped only when the probe verdict flips (the
             owner's words then describe a run that is gone).  A status written before 1.1.1 that is neither probe
             line is adopted as the note on the first pulse.
  set-probe  --lane L --agent A --probe JSON|@FILE   the lane's OWNER only: set (or with '{}' clear) the lane's probe;
             validated like a seed probe
  takeover   --lane L --agent A --reason T --state-file PATH --live-check T --synced [--cyril "verbatim quote"]
             plan 7.4.5: allowed only if Cyril said so (--cyril) or the lane is transferable and its heartbeat (or,
             if it never had one, its registration) is older than 6 h.  Writes a takeover receipt next to the
             registry (lanes_takeovers/), runs the lane's probe and prints DO NOT RELAUNCH when the process is alive.
             A lane with NO probe is recorded as process_alive null ("unknown (no probe)"), never false, and the open
             claims of its *.claims lock paths are printed and recorded: only the --live-check says it is idle.
  init       --seed SEED.json [--replace]   create the registry (an existing one only with --replace, backed up first)
  hold-lock  --lock PATH --while-pid PID --cmd-match S [--also-match M ...] [--join-helper] [--poll S] [--wait S]
             [--ready-file PATH]
             hold an exclusive flock on PATH while pid PID lives with S in its cmdline OR any process has an
             --also-match M in its cmdline (the card's jobs: a job that outlives its queue keeps the card locked);
             a busy lock is waited for up to --wait s, then refused (exit 5).  --join-helper: when every holder is
             an ORPHANED `lanes.py hold-lock` (its own --while-pid queue is gone; it holds only for jobs, which this
             queue adopts), write the ready file at once and keep waiting -- without a deadline, while PID lives --
             to take the lock when that helper releases it; the lock is never free in between for longer than one
             0.1 s wait poll.  A helper whose queue still lives is never joined (two queues on one card).  Used
             (a) by queue_gpu6.sh >= 1.3.2 for its card lock, so the jobs it starts never inherit the lock fd, and
             (b) as a bridge for a queue that started without one (queue_gpu6.sh 1.3.0)
  cards      [--json]   the card map: GPU index -> the lanes that may claim it (cards = primary, flex_cards = shared)
             and whether it is ask-first (a serving card: Cyril's approval before anything else runs on it)
  register   --agent A --lane-json JSON|@FILE --reason T [--cyril "verbatim quote"]
             add a lane, or re-scope one you own: the given fields are merged, the values they replace are kept in
             the lane's history (utc, by, reason, previous).  Heartbeat / receipt / takeover fields are not settable
  set-rule   --agent A --key K --text T --reason T   set a registry rule; the previous text goes to rules_history
  claim-run  --lane L --agent A --gpus 3,5 --job J --log PATH [--borrow REASON] [--cyril QUOTE] [--wait S]
             [--busy-limit MIB] [--mem-floor-gib G] [--nice N] [--ready-file F] [--exit-file F] [--omni-lab-kind K |
             --no-omni-lab] [--dry-run] -- CMD ...
             the one launcher for a lane's GPU job.  Refuses (exit 2 policy, 4 serving card without --cyril, 5 busy)
             unless every card is the lane's (cards / flex_cards), another lane's P40 is borrowed with --borrow REASON,
             or a serving card comes with Cyril's words.  Then: flock on every $Q/gpu<N>.lock (lowest index first,
             waits up to --wait s), no open p40.claims line on those cards, every card <= --busy-limit MiB, host
             MemAvailable >= --mem-floor-gib; starts CMD under nice in its own session with CUDA_DEVICE_ORDER=PCI_BUS_ID
             and CUDA_VISIBLE_DEVICES=<the cards' UUIDs> (the lock fds are not inherited), appends the CLAIM line
             (pid = the job, preempt = kill -TERM this wrapper), registers the run with omni_lab log-run (best effort),
             forwards TERM/INT/HUP to the job's session, waits, appends the RELEASE line, ends the omni_lab run and
             exits with the job's code.  --ready-file / --exit-file get JSON (job pid, cards, UUIDs, omni_lab run id;
             then rc and status) for a queue runner.  Nothing on a P40 is preempted by this tool: it waits or refuses
  release-stale  --agent A --cards GPU1,GPU2 --claim-lane X --job J --reason T
             append a RELEASE line for an open p40.claims claim whose pid is dead (a claim with a live or unknown pid
             is refused, exit 5); the claim's cards / lane / job must match its CLAIM line exactly

Writes take <registry>.lock (flock, up to 30 s), then write a temp file, fsync and rename: a reader never sees half a
file, and a queue's heartbeat never blocks it for long (callers wrap it in `timeout`).
"""

from __future__ import annotations

import argparse
import fcntl
import glob
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

TOOL_NAME = "lanes"
TOOL_VERSION = "1.1.1"   # 1.0.0 (2026-10-14): first version, card X-T2
# 1.0.1 (2026-10-15, review of 1.0.0): takeover records process_alive null (not false) for a lane with no probe and
#   prints/records open *.claims lines; show lists open claims; probe status_ok / status_down; set-probe; hold-lock
#   --also-match (a job outliving its queue keeps the card lock) and --join-helper (a new queue adopts without a gap)
# 1.0.2 (2026-10-15): set-probe @FILE that cannot be read is refused (exit 2), not a traceback
# 1.1.0 (2026-10-21, AI_SERVER_SERVING_DESIGN_2026-10-21 section 2.6; Cyril 2026-10-21: P40s for training and
#   experiments, GPU0/4/6 for Sprite + data generation "[Cyril quote withheld]"): per-lane card policy
#   (cards / flex_cards / ask_first), cards, register, set-rule, claim-run (the one launcher for lane GPU jobs),
#   release-stale.  GPU5 is no longer reserved
# 1.1.1 (2026-10-21, storage lane): pulse merges the probe's status line with the owner's status note instead of
#   overwriting it (the 10-minute cron replaced the training / experiments lines written by their owners); the note
#   is retired only when the probe verdict flips
SCHEMA = "omni-campaign.lanes.v1"
TAKEOVER_SCHEMA = "omni-campaign.lane-takeover.v1"
DEFAULT_FILE = "/home/cereal/omni-campaign/distill/queues/LANES.json"
AGENTS = ("claude", "codex")
STALE_S = 6 * 3600
LOCK_WAIT_S = 30.0
REQUIRED = ("lane", "owner", "scope", "lock_paths", "status", "transferable", "transferable_reason")
PROBE_KINDS = ("systemd_unit", "pid_cmdline", "file_fresh")
PROBE_CHECK_KEYS = {"systemd_unit": ("unit",), "pid_cmdline": ("match",), "file_fresh": ("path", "max_age_s")}
WAIT_POLL_S = 0.1  # hold-lock's poll while waiting for a busy lock (the longest gap a handover can leave)
CLAIMS_FILE = "p40.claims"  # beside the registry, like the gpu<N>.lock files
MAX_GPU_INDEX = 15
JOB_RE = r"[A-Za-z0-9][A-Za-z0-9._:-]{0,79}"
# Words that make a p40.claims line a release for local-hermes/remote_runtime.py; never allowed in a job name
CLAIM_RELEASE_WORDS = ("released", "stopped", "finished", "done", "freed", "ended", "completed")
REGISTER_PROTECTED = ("heartbeat_utc", "heartbeat_by", "heartbeat_source", "last_receipt", "takeover", "probe_last",
                      "status_note", "status_note_dropped",
                      "registered_utc", "history", "previous_owner", "owner_confirmed")
OMNI_LAB_PY = "/home/cereal/.venvs/chimera-intake/bin/python"
OMNI_LAB_CLI = "/home/cereal/omni-campaign/lab/omni_lab.py"
OMNI_LAB_ENV = "/home/cereal/.config/omni-lab/env"
DEFAULT_BUSY_MIB = 2000
DEFAULT_MEM_FLOOR_GIB = 40.0


class Refused(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------------------------------------ time and files
def now_epoch() -> float:
    forced = os.environ.get("OC_LANES_NOW", "")  # tests only: an ISO UTC instant
    return parse_utc(forced) if forced else time.time()


def utc(epoch: float | None = None) -> str:
    return datetime.fromtimestamp(now_epoch() if epoch is None else epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text: str) -> float:
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt_info(path: str) -> dict:
    p = os.path.abspath(path)
    if not os.path.isfile(p):
        raise Refused(2, f"receipt {p} is not a file")
    st = os.stat(p)
    return {"path": p, "sha256": sha256_file(p), "bytes": st.st_size, "mtime_utc": utc(st.st_mtime)}


def atomic_write_json(path: str, obj: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w") as handle:
        json.dump(obj, handle, indent=1, sort_keys=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class RegistryLock:
    def __init__(self, path: str):
        self.path = path + ".lock"
        self.fd = -1

    def __enter__(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        deadline = time.monotonic() + LOCK_WAIT_S
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except BlockingIOError:
                if time.monotonic() > deadline:
                    os.close(self.fd)
                    raise Refused(5, f"registry lock {self.path} busy for {LOCK_WAIT_S:.0f} s")
                time.sleep(0.2)

    def __exit__(self, *exc):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        os.close(self.fd)


def load(path: str) -> dict:
    try:
        with open(path) as handle:
            reg = json.load(handle)
    except FileNotFoundError:
        raise Refused(6, f"no registry at {path} (create it with `lanes.py init --seed ...`)")
    except json.JSONDecodeError as exc:
        raise Refused(6, f"registry {path} is not JSON: {exc}")
    if reg.get("schema") != SCHEMA or not isinstance(reg.get("lanes"), dict):
        raise Refused(6, f"registry {path} has schema {reg.get('schema')!r}, want {SCHEMA}")
    return reg


def save(path: str, reg: dict, by: str) -> None:
    reg["revision"] = int(reg.get("revision", 0)) + 1
    reg["updated_utc"] = utc()
    reg["updated_by"] = by
    atomic_write_json(path, reg)


def lane_of(reg: dict, name: str) -> dict:
    if name not in reg["lanes"]:
        raise Refused(2, f"unknown lane {name!r}; lanes: {', '.join(reg['lanes'])}")
    return reg["lanes"][name]


def agent_arg(value: str) -> str:
    v = value.strip().lower()
    if v not in AGENTS:
        raise argparse.ArgumentTypeError(f"agent must be one of {AGENTS}")
    return v


# ------------------------------------------------------------------------------------------------ liveness
def lock_holders(path: str) -> list[dict] | None:
    """Holders of a flock/POSIX lock on path, from /proc/locks (read-only; never takes the lock). None if unknowable."""
    try:
        st = os.stat(path)
        with open("/proc/locks") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    out = []
    for line in lines:
        f = line.split()
        if "->" in f:  # a blocked waiter: "1: -> FLOCK ..."
            continue
        if len(f) >= 6 and f[5] == key:
            out.append({"pid": int(f[4]), "kind": f[1], "mode": f[3], "cmd": pid_cmdline(int(f[4]))[:160]})
    return out


def pid_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return handle.read().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def pids_matching(match: str) -> list[int]:
    me = os.getpid()
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        cmd = pid_cmdline(int(name))
        if match in cmd and "lanes.py" not in cmd:  # never our own helpers (hold-lock names the match in its argv)
            out.append(int(name))
    return sorted(out)


def pid_alive(pid: int) -> bool:
    return os.path.isdir(f"/proc/{pid}")


def claims_open(path: str) -> list[dict] | None:
    """Open claims in a *.claims ledger: per (cards, lane, job) the last line, if it is not a release.  Two line
    formats are in use: "<utc> GPU3 lane=X owner=Y job=Z pid=P ..." / "<utc> GPU3 lane=X job=Z released" and
    "<utc> CLAIM GPU1,GPU2 owner=.. lane=.. job=.. pid=.." / "<utc> RELEASE GPU1,GPU2 ... released".  None if
    unreadable."""
    try:
        with open(path, errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return None
    last: dict[tuple, tuple[bool, dict, str]] = {}
    for line in lines:
        f = line.split()
        if len(f) < 3:
            continue
        verb = f[1] if f[1] in ("CLAIM", "RELEASE") else ""
        cards = f[2] if verb else f[1]
        kv = dict(t.split("=", 1) for t in f if "=" in t)
        released = verb == "RELEASE" or "released" in f
        last[(cards, kv.get("lane", ""), kv.get("job", ""))] = (released, kv, line)
    out = []
    for (cards, lane, job), (released, kv, line) in last.items():
        if released:
            continue
        pid = int(kv["pid"]) if kv.get("pid", "").isdigit() else None
        out.append({"cards": cards, "lane": lane, "job": job, "pid": pid,
                    "pid_alive": None if pid is None else pid_alive(pid), "line": line[:300]})
    return out


def claims_of(lane: dict) -> dict:
    return {p: claims_open(p) for p in lane.get("lock_paths", []) if p.endswith(".claims")}


def run_check(check: dict) -> tuple[bool, str]:
    kind = check.get("kind")
    if kind == "systemd_unit":
        unit = check["unit"]
        try:
            r = subprocess.run(["systemctl", "show", "-p", "ActiveState", "--value", unit],
                               capture_output=True, text=True, timeout=15)
            state = r.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"{unit}: systemctl failed ({exc})"
        return state == "active", f"{unit} {state or 'unknown'}"
    if kind == "pid_cmdline":
        pids = pids_matching(check["match"])
        return bool(pids), f"pid {','.join(map(str, pids[:4]))} '{check['match']}'" if pids else f"no pid '{check['match']}'"
    if kind == "file_fresh":
        try:
            age = now_epoch() - os.stat(check["path"]).st_mtime
        except OSError:
            return False, f"{check['path']} missing"
        return age <= float(check["max_age_s"]), f"{os.path.basename(check['path'])} {age:.0f}s old (max {check['max_age_s']})"
    return False, f"unknown probe kind {kind!r}"


def run_probe(probe: dict) -> dict:
    results = [(c, *run_check(c)) for c in probe.get("checks", [])]
    ok = bool(results) and all(r[1] for r in results)
    return {"utc": utc(), "ok": ok, "checks": [{"kind": c["kind"], "ok": good, "detail": d} for c, good, d in results]}


def newest_receipt(globs: list[str]) -> str | None:
    best, best_m = None, -1.0
    for pattern in globs:
        for p in glob.glob(pattern):
            try:
                m = os.stat(p).st_mtime
            except OSError:
                continue
            if os.path.isfile(p) and m > best_m:
                best, best_m = p, m
    return best


def detail_from(spec: dict | None) -> str | None:
    if not spec:
        return None
    try:
        if spec.get("mode") == "json_keys":
            with open(spec["path"]) as handle:
                d = json.load(handle)
            return " ".join(f"{k}={d.get(k)}" for k in spec.get("keys", []))
        with open(spec["path"], "rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 4096))
            lines = handle.read().decode(errors="replace").strip().splitlines()
        return lines[-1][:300] if lines else ""
    except (OSError, ValueError) as exc:
        return f"detail unreadable: {exc}"


def note_status(lane: dict, text: str, by: str, source: str) -> None:
    """Set the lane's status to the owner's (or a takeover's) own words and keep them as the status note that pulse
    merges with its probe line (1.1.1).  probe_ok = the probe verdict the words were written under."""
    lane["status"] = text
    lane["status_note"] = {"text": text, "utc": utc(), "by": by, "source": source,
                           "probe_ok": (lane.get("probe_last") or {}).get("ok")}


def merged_status(lane: dict, follow: str, ok: bool) -> str:
    """The pulse's status line: the probe's follow line, then the owner's note (1.1.1: merge, never replace).
    A pre-1.1.1 status that is neither probe line is adopted as the note; a note written under the other probe
    verdict is retired to status_note_dropped (its run is gone or has come back since the owner wrote it)."""
    probe = lane.get("probe") or {}
    note = lane.get("status_note")
    current = str(lane.get("status") or "").strip()
    if note is None and current and current not in (probe.get("status_ok"), probe.get("status_down")):
        note = {"text": current, "utc": lane.get("heartbeat_utc") or lane.get("registered_utc"),
                "by": lane.get("heartbeat_by") or lane.get("owner"),
                "source": f"adopted by pulse {TOOL_VERSION}", "probe_ok": ok}
        lane["status_note"] = note
    if note is None:
        return follow
    if note.get("probe_ok") is None:
        note["probe_ok"] = ok                       # written before the lane had a verdict: bind it to this one
    elif bool(note["probe_ok"]) != ok:
        lane["status_note_dropped"] = dict(note, dropped_utc=utc(), why=f"probe turned {'ok' if ok else 'down'}")
        del lane["status_note"]
        return follow
    if note.get("text") == follow:
        return follow
    return f"{follow} | {note.get('by')} {note.get('utc')}: {note.get('text')}"


# ------------------------------------------------------------------------------------------------ rules
def staleness(lane: dict) -> tuple[float, str]:
    """Seconds since the lane last showed life: its heartbeat, or its registration if it never had one."""
    ref = lane.get("heartbeat_utc") or lane.get("registered_utc")
    basis = "heartbeat" if lane.get("heartbeat_utc") else "registration"
    if not ref:
        return float("inf"), "none"
    return now_epoch() - parse_utc(ref), basis


def takeover_rule(lane: dict, cyril: str | None) -> tuple[bool, str]:
    if cyril and cyril.strip():
        return True, "cyril"
    age, basis = staleness(lane)
    if not lane.get("transferable"):
        return False, "not transferable (needs Cyril's word)"
    if age <= STALE_S:
        return False, f"{basis} {age / 3600:.1f} h old <= 6 h"
    return True, f"transferable and {basis} {age / 3600:.1f} h old > 6 h"


# ------------------------------------------------------------------------------------------------ commands
def alive_word(flag: bool | None) -> str:
    return "alive" if flag else "dead" if flag is False else "unknown"


def cmd_show(args) -> int:
    reg = load(args.file)
    names = [args.lane] if args.lane else list(reg["lanes"])
    rows = []
    for name in names:
        lane = lane_of(reg, name)
        age, basis = staleness(lane)
        allowed, why = takeover_rule(lane, None)
        locks = {p: lock_holders(p) for p in lane.get("lock_paths", []) if not p.endswith(".claims")}
        rows.append({"lane": name, "owner": lane["owner"], "status": lane["status"],
                     "cards": lane.get("cards", []), "flex_cards": lane.get("flex_cards", []),
                     "ask_first": bool(lane.get("ask_first")),
                     "heartbeat_utc": lane.get("heartbeat_utc"), "heartbeat_source": lane.get("heartbeat_source"),
                     "age_s": None if age == float("inf") else round(age), "age_basis": basis,
                     "stale": age > STALE_S, "transferable": lane["transferable"],
                     "takeover_allowed_without_cyril": allowed, "takeover_rule": why,
                     "probe_ok": (lane.get("probe_last") or {}).get("ok"), "lock_holders": locks,
                     "claims_open": claims_of(lane),
                     "last_receipt": lane.get("last_receipt"), "detail": lane.get("detail")})
    if args.json:
        print(json.dumps({"file": args.file, "revision": reg.get("revision"), "updated_utc": reg.get("updated_utc"),
                          "lanes": rows}, indent=1))
        return 0
    print(f"{args.file} revision {reg.get('revision')} updated {reg.get('updated_utc')} by {reg.get('updated_by')}")
    for r in rows:
        age = "never" if r["age_s"] is None else f"{r['age_s'] / 3600:.1f}h"
        locks = "; ".join(f"{os.path.basename(p)}:{'?' if h is None else ','.join(str(x['pid']) for x in h) or 'free'}"
                          for p, h in r["lock_holders"].items()) or "-"
        rc = r["last_receipt"] or {}
        lane = reg["lanes"][r["lane"]]
        cards = ",".join(map(str, lane.get("cards", []))) or "-"
        flex = ("+flex " + ",".join(map(str, lane["flex_cards"]))) if lane.get("flex_cards") else ""
        print(f"{r['lane']:<14} owner={r['owner']:<6} cards={cards}{flex}{' ask-first' if lane.get('ask_first') else ''}"
              f" status={r['status']}")
        print(f"{'':<14} heartbeat {r['heartbeat_utc']} ({age} by {r['age_basis']}, {r['heartbeat_source']}) "
              f"stale={'yes' if r['stale'] else 'no'} probe_ok={r['probe_ok']} takeover: {r['takeover_rule']}")
        print(f"{'':<14} locks {locks}; receipt {rc.get('path', '-')} {str(rc.get('sha256', ''))[:16]}")
        for p, cl in r["claims_open"].items():
            for c in cl or []:
                print(f"{'':<14} open claim in {os.path.basename(p)}: {c['cards']} lane={c['lane']} job={c['job']} "
                      f"pid={c['pid']} ({alive_word(c['pid_alive'])})")
        if r["detail"]:
            print(f"{'':<14} detail {str(r['detail'])[:200]}")
    return 0


def cmd_heartbeat(args) -> int:
    with RegistryLock(args.file):
        reg = load(args.file)
        lane = lane_of(reg, args.lane)
        if lane["owner"] != args.agent:
            raise Refused(3, f"{args.lane} is owned by {lane['owner']}; {args.agent} may read and enqueue, "
                             "or take it over by the takeover rule (plan 7.4.5)")
        lane["heartbeat_utc"] = utc()
        lane["heartbeat_by"] = args.agent
        lane["heartbeat_source"] = args.source or "cli"
        if args.status:
            note_status(lane, args.status, args.agent, args.source or "cli")
        if args.detail is not None:
            lane["detail"] = args.detail[:500]
        if args.receipt:
            lane["last_receipt"] = receipt_info(args.receipt)
        if args.transferable:
            lane["transferable"] = args.transferable == "yes"
            if args.transferable_reason:
                lane["transferable_reason"] = args.transferable_reason
        lane["owner_confirmed"] = True
        save(args.file, reg, f"{args.agent} heartbeat {args.lane}")
    print(f"HEARTBEAT {args.lane} {lane['heartbeat_utc']} by {args.agent} status={lane['status']}")
    return 0


def cmd_pulse(args) -> int:
    with RegistryLock(args.file):
        reg = load(args.file)
        lines, changed = [], False
        for name, lane in reg["lanes"].items():
            probe = lane.get("probe")
            if not probe or not probe.get("checks"):
                continue
            res = run_probe(probe)
            detail = "; ".join(c["detail"] for c in res["checks"])
            if args.dry_run:
                lines.append(f"PULSE {name} {'ok' if res['ok'] else 'DOWN'} (dry run) {detail}")
                continue
            lane["probe_last"] = res
            changed = True
            follow = probe.get("status_ok") if res["ok"] else probe.get("status_down")
            if follow:
                lane["status"] = merged_status(lane, follow, bool(res["ok"]))
            if res["ok"]:
                lane["heartbeat_utc"] = res["utc"]
                lane["heartbeat_by"] = lane["owner"]
                lane["heartbeat_source"] = "pulse"
                d = detail_from(probe.get("detail"))
                if d is not None:
                    lane["detail"] = d[:500]
                newest = newest_receipt(probe.get("receipt_globs", []))
                old = lane.get("last_receipt") or {}
                if newest and (old.get("path") != os.path.abspath(newest)
                               or old.get("mtime_utc") != utc(os.stat(newest).st_mtime)):
                    lane["last_receipt"] = receipt_info(newest)
            lines.append(f"PULSE {name} {'ok' if res['ok'] else 'DOWN (no heartbeat)'} {detail}")
        if changed:
            save(args.file, reg, f"pulse {TOOL_VERSION}")
    print(f"{utc()} {TOOL_NAME} {TOOL_VERSION} pulse: " + (" | ".join(lines) if lines else "no lane has a probe"))
    return 0


def gpu_snapshot() -> str:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else f"nvidia-smi rc {r.returncode}: {r.stderr.strip()[:200]}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable ({exc})"


def cmd_takeover(args) -> int:
    if not args.synced:
        raise Refused(2, "run `session.py --agent <you> sync` first and pass --synced (plan 7.4.5 step 1)")
    if not os.path.isfile(args.state_file):
        raise Refused(2, f"--state-file {args.state_file} is not a file: read the newest STATE first (step 2)")
    if not args.live_check.strip() or not args.reason.strip():
        raise Refused(2, "--live-check and --reason must say what was checked and why")
    with RegistryLock(args.file):
        reg = load(args.file)
        lane = lane_of(reg, args.lane)
        if lane["owner"] == args.agent:
            raise Refused(2, f"{args.agent} already owns {args.lane}")
        allowed, why = takeover_rule(lane, args.cyril)
        if not allowed:
            raise Refused(4, f"takeover of {args.lane} refused: {why}")
        age, basis = staleness(lane)
        probe = run_probe(lane["probe"]) if (lane.get("probe") or {}).get("checks") else None
        claims = claims_of(lane)
        stamp = utc()
        rec = {"schema": TAKEOVER_SCHEMA, "tool": f"{TOOL_NAME} {TOOL_VERSION}", "lane": args.lane, "utc": stamp,
               "from_owner": lane["owner"], "to_owner": args.agent, "rule": "cyril" if args.cyril else "stale-transferable",
               "rule_detail": why, "cyril_quote": args.cyril or None, "reason": args.reason,
               "heartbeat_utc_before": lane.get("heartbeat_utc"), "staleness_s": None if age == float("inf") else round(age),
               "staleness_basis": basis, "transferable": lane["transferable"],
               "state_file": receipt_info(args.state_file), "synced_attested": True,
               "live_check_note": args.live_check, "probe": probe,
               "process_alive": None if probe is None else bool(probe["ok"]),
               "liveness": "unknown (no probe)" if probe is None else ("probe ok" if probe["ok"] else "probe down"),
               "claims_open": claims,
               "lock_holders": {p: lock_holders(p) for p in lane.get("lock_paths", []) if not p.endswith(".claims")},
               "gpu_compute_apps": "skipped" if args.no_gpu_snapshot else gpu_snapshot()}
        d = os.path.join(os.path.dirname(os.path.abspath(args.file)), "lanes_takeovers")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{stamp.replace(':', '').replace('-', '')}_{args.lane}_{args.agent}.json")
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
        with os.fdopen(fd, "w") as handle:
            json.dump(rec, handle, indent=1)
            handle.write("\n")
        info = receipt_info(path)
        lane["previous_owner"] = lane["owner"]
        lane["owner"] = args.agent
        lane["owner_confirmed"] = True
        note_status(lane, f"taken over by {args.agent} at {stamp}", args.agent, "takeover")
        lane["heartbeat_utc"] = stamp
        lane["heartbeat_by"] = args.agent
        lane["heartbeat_source"] = "takeover"
        lane["takeover"] = {"utc": stamp, "receipt": info}
        save(args.file, reg, f"{args.agent} takeover {args.lane}")
    print(f"TAKEOVER {args.lane} {rec['from_owner']} -> {args.agent} ({why}); receipt {info['path']} sha256 {info['sha256']}")
    if rec["process_alive"]:
        print(f"DO NOT RELAUNCH: the lane's process is alive ({'; '.join(c['detail'] for c in probe['checks'])})")
    elif probe is None:
        print("LIVENESS UNKNOWN: this lane has no probe, so nothing here shows its process is gone -- only your "
              "--live-check does.  Check systemctl / pgrep / nvidia-smi again before relaunching anything")
    for p, cl in claims.items():
        for c in cl or []:
            print(f"OPEN CLAIM in {p}: {c['line']} (pid {alive_word(c['pid_alive'])})")
    print("next: post the five-line hand-off on Relay (lane; state + receipt path/sha; running; next command; do not touch)")
    return 0


def validate_lane(name: str, lane: dict) -> None:
    missing = [k for k in REQUIRED if k not in lane]
    if missing:
        raise Refused(2, f"lane {name}: missing {missing}")
    if lane["lane"] != name or lane["owner"] not in AGENTS or not isinstance(lane["lock_paths"], list) \
            or not isinstance(lane["transferable"], bool):
        raise Refused(2, f"lane {name}: lane/owner/lock_paths/transferable malformed")
    for key in ("cards", "flex_cards"):
        cards = lane.get(key, [])
        if not isinstance(cards, list) or any(type(c) is not int or not 0 <= c <= MAX_GPU_INDEX for c in cards) \
                or len(set(cards)) != len(cards):
            raise Refused(2, f"lane {name}: {key} must be a list of distinct GPU indexes 0-{MAX_GPU_INDEX}")
    if set(lane.get("cards", [])) & set(lane.get("flex_cards", [])):
        raise Refused(2, f"lane {name}: a card is in both cards and flex_cards")
    if "ask_first" in lane and not isinstance(lane["ask_first"], bool):
        raise Refused(2, f"lane {name}: ask_first must be true or false")
    validate_probe(name, lane.get("probe") or {})


def validate_probe(name: str, probe: dict) -> None:
    if not isinstance(probe, dict):
        raise Refused(2, f"lane {name}: probe must be an object")
    for c in probe.get("checks", []):
        if c.get("kind") not in PROBE_KINDS:
            raise Refused(2, f"lane {name}: probe kind {c.get('kind')!r} not in {PROBE_KINDS}")
        missing = [k for k in PROBE_CHECK_KEYS[c["kind"]] if not c.get(k)]
        if missing:
            raise Refused(2, f"lane {name}: probe check {c['kind']} needs {missing}")
    if probe and not probe.get("checks"):
        raise Refused(2, f"lane {name}: a probe needs at least one check ('{{}}' clears the probe)")
    for k in ("status_ok", "status_down"):
        if k in probe and not (isinstance(probe[k], str) and probe[k].strip()):
            raise Refused(2, f"lane {name}: probe {k} must be a non-empty string")
    if not isinstance(probe.get("receipt_globs", []), list):
        raise Refused(2, f"lane {name}: probe receipt_globs must be a list")


def cmd_set_probe(args) -> int:
    text = args.probe
    if text.startswith("@"):
        try:
            with open(text[1:]) as handle:
                text = handle.read()
        except OSError as exc:
            raise Refused(2, f"--probe {text}: {exc}")
    try:
        probe = json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(2, f"--probe is not JSON: {exc}")
    validate_probe(args.lane, probe)
    with RegistryLock(args.file):
        reg = load(args.file)
        lane = lane_of(reg, args.lane)
        if lane["owner"] != args.agent:
            raise Refused(3, f"{args.lane} is owned by {lane['owner']}; only the owner sets its probe")
        lane["probe"] = probe or None
        lane.pop("probe_last", None)
        save(args.file, reg, f"{args.agent} set-probe {args.lane}")
    note = f"dry probe {'ok' if run_probe(probe)['ok'] else 'DOWN'}" if probe else "probe cleared"
    print(f"SET-PROBE {args.lane}: {len(probe.get('checks', []))} check(s); {note}")
    return 0


def cmd_init(args) -> int:
    with open(args.seed) as handle:
        seed = json.load(handle)
    lanes = seed.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        raise Refused(2, "seed needs a non-empty 'lanes' list")
    names = [ln.get("lane") for ln in lanes]
    if len(set(names)) != len(names):
        raise Refused(2, f"duplicate lane ids in seed: {names}")
    stamp = utc()
    reg = {"schema": SCHEMA, "about": seed.get("about", ""), "rules": seed.get("rules", {}), "revision": 0,
           "created_utc": stamp, "created_by": seed.get("created_by", "unknown"), "lanes": {}}
    for ln in lanes:
        validate_lane(ln.get("lane"), ln)
        ln.setdefault("registered_utc", stamp)
        ln.setdefault("heartbeat_utc", None)
        ln.setdefault("last_receipt", None)
        if ln.get("last_receipt_path"):
            ln["last_receipt"] = receipt_info(ln.pop("last_receipt_path"))
        reg["lanes"][ln["lane"]] = ln
    with RegistryLock(args.file):
        if os.path.exists(args.file):
            if not args.replace:
                raise Refused(2, f"{args.file} exists; --replace backs it up and replaces it")
            backup = f"{args.file}.{stamp.replace(':', '')}.bak"
            os.replace(args.file, backup)
            print(f"backup {backup} sha256 {sha256_file(backup)}")
        save(args.file, reg, f"init from {os.path.basename(args.seed)}")
    print(f"INIT {args.file}: {len(reg['lanes'])} lanes ({', '.join(reg['lanes'])}) sha256 {sha256_file(args.file)}")
    return 0


def pid_argv(pid: int) -> list[str]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            return [a.decode(errors="replace") for a in handle.read().split(b"\0") if a]
    except OSError:
        return []


def orphaned_helper(pid: int) -> bool:
    """True if pid is a `lanes.py hold-lock` whose own --while-pid process is gone: it holds only for --also-match
    jobs, so a successor queue may adopt them.  A helper whose queue still lives is NOT joinable (two queues on one
    card); neither is anything that is not a helper."""
    argv = pid_argv(pid)
    if not (any(a.endswith("lanes.py") for a in argv) and "hold-lock" in argv):
        return False
    opts = {argv[i]: argv[i + 1] for i in range(len(argv) - 1) if argv[i] in ("--while-pid", "--cmd-match")}
    if not opts.get("--while-pid", "").isdigit() or "--cmd-match" not in opts:
        return False
    return opts["--cmd-match"] not in pid_cmdline(int(opts["--while-pid"]))


def cmd_hold_lock(args) -> int:
    also = [m for m in args.also_match if m.strip()]

    def owner_alive() -> bool:
        return args.cmd_match in pid_cmdline(args.while_pid)

    def jobs_alive() -> list[int]:
        return sorted({p for m in also for p in pids_matching(m)})

    def write_ready(note: str) -> None:
        if args.ready_file:
            with open(args.ready_file, "w") as handle:
                handle.write(f"{os.getpid()}{note}\n")

    if not owner_alive():
        raise Refused(2, f"pid {args.while_pid} is not alive with '{args.cmd_match}' in its cmdline")
    fd = os.open(args.lock, os.O_RDWR | os.O_CREAT, 0o664)
    deadline = time.monotonic() + args.wait
    joined: list[int] = []
    next_join_check = 0.0
    while True:  # wait, never compete: the lock is only ever taken when free
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not owner_alive():
                os.close(fd)
                if joined:
                    print(f"{utc()} GIVE UP {args.lock}: pid {args.while_pid} is gone while waiting behind helper "
                          f"{joined}", flush=True)
                    return 0
                raise Refused(5, f"{args.lock} is held by {lock_holders(args.lock)}; pid {args.while_pid} is gone")
            if args.join_helper and not joined and time.monotonic() >= next_join_check:
                next_join_check = time.monotonic() + 1.0
                holders = lock_holders(args.lock) or []
                if holders and all(orphaned_helper(h["pid"]) for h in holders):
                    joined = [h["pid"] for h in holders]
                    print(f"{utc()} JOIN {args.lock}: held by hold-lock helper pid {joined}; this helper (pid "
                          f"{os.getpid()}) takes it when that one releases", flush=True)
                    write_ready(f" joined {','.join(map(str, joined))}")
            if not joined and time.monotonic() >= deadline:
                os.close(fd)
                raise Refused(5, f"{args.lock} is held by {lock_holders(args.lock)} (waited {args.wait:.0f} s)")
            time.sleep(WAIT_POLL_S)
    print(f"{utc()} HOLD {args.lock} for pid {args.while_pid} ('{args.cmd_match}')"
          + (f" or any process matching {also}" if also else "") + f", poll {args.poll}s, holder pid {os.getpid()}",
          flush=True)
    if not joined:
        write_ready("")
    while owner_alive() or jobs_alive():
        time.sleep(args.poll)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    print(f"{utc()} RELEASE {args.lock}: pid {args.while_pid} is gone"
          + (f" and no process matches {also}" if also else ""), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ card policy (1.1.0)
def card_map(reg: dict) -> dict[int, list[dict]]:
    """GPU index -> the lanes that list it: role primary (cards) or flex (flex_cards), and the lane's ask_first."""
    out: dict[int, list[dict]] = {}
    for name, lane in reg["lanes"].items():
        for key, role in (("cards", "primary"), ("flex_cards", "flex")):
            for card in lane.get(key, []):
                out.setdefault(card, []).append({"lane": name, "role": role, "ask_first": bool(lane.get("ask_first"))})
    return dict(sorted(out.items()))


def card_verdicts(reg: dict, lane_name: str, cards: list[int], borrow: str | None, cyril: str | None) -> list[str]:
    """One note per card if lane_name may claim it; Refused otherwise.  A lane's own cards and flex cards are free to
    it; another lane's ask-first (serving) card needs Cyril's words (exit 4); any other lane's card needs --borrow
    REASON (exit 2); a card no lane lists needs Cyril's words (exit 2)."""
    lane = lane_of(reg, lane_name)
    own = {c: "own" for c in lane.get("cards", [])}
    own.update({c: "flex" for c in lane.get("flex_cards", [])})
    cmap = card_map(reg)
    notes = []
    for card in cards:
        if card in own:
            notes.append(f"GPU{card} {own[card]}")
            continue
        holders = cmap.get(card, [])
        serving = sorted({h["lane"] for h in holders if h["ask_first"]})
        if serving:
            if not (cyril and cyril.strip()):
                raise Refused(4, f"GPU{card} is a serving card ({', '.join(serving)}): pulling it needs Cyril's "
                                 "approval (--cyril with his words)")
            notes.append(f"GPU{card} serving card of {','.join(serving)} pulled on Cyril's word")
        elif holders:
            if not (borrow and borrow.strip()):
                raise Refused(2, f"GPU{card} belongs to {','.join(sorted({h['lane'] for h in holders}))}, not "
                                 f"{lane_name}: borrow it with --borrow REASON (it still waits for the card's lock)")
            notes.append(f"GPU{card} borrowed from {','.join(sorted({h['lane'] for h in holders}))}")
        else:
            if not (cyril and cyril.strip()):
                raise Refused(2, f"GPU{card} is in no lane's cards: only on Cyril's word (--cyril)")
            notes.append(f"GPU{card} unassigned, on Cyril's word")
    return notes


def parse_cards(text: str) -> list[int]:
    """'3,5' or 'GPU3,GPU5' -> [3, 5] (sorted, distinct)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        part = part[3:] if part.upper().startswith("GPU") else part
        if not part.isdigit() or int(part) > MAX_GPU_INDEX:
            raise Refused(2, f"bad card {part!r} in {text!r}: use indexes like 3,5")
        out.append(int(part))
    if not out or len(set(out)) != len(out):
        raise Refused(2, f"cards {text!r}: name each card once")
    return sorted(out)


def cards_token(cards: list[int]) -> str:
    return ",".join(f"GPU{c}" for c in sorted(cards))


def free_text(text: str, what: str, claim_line: bool = True) -> str:
    """Free text for a claims line: one line, no key=value (it would shadow the line's fields) and, in a CLAIM line,
    no release word (remote_runtime.py and claims_open would read the claim as a release)."""
    t = " ".join(text.split()).replace("=", ":")
    if claim_line and any(w.strip(".,;:()!").lower() in CLAIM_RELEASE_WORDS or w.strip(".,;:()!").lower() == "release"
           for w in t.split()):
        raise Refused(2, f"{what} {t!r} contains a release word ({', '.join(CLAIM_RELEASE_WORDS)}): reword it")
    return t


def smi_cards() -> dict[int, dict]:
    """nvidia-smi index -> {index, uuid, name, used_mib}.  OC_LANES_SMI_JSON (tests only) replaces the query."""
    forced = os.environ.get("OC_LANES_SMI_JSON", "")
    if forced:
        return {int(c["index"]): c for c in json.loads(forced)}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(5, f"nvidia-smi unavailable ({exc}): card state unknown")
    if r.returncode != 0:
        raise Refused(5, f"nvidia-smi rc {r.returncode}: {r.stderr.strip()[:200]}")
    out = {}
    for line in r.stdout.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) == 4 and f[0].isdigit():
            out[int(f[0])] = {"index": int(f[0]), "uuid": f[1], "name": f[2], "used_mib": int(float(f[3]))}
    return out


def mem_available_gib() -> float | None:
    forced = os.environ.get("OC_LANES_MEMAVAIL_GIB", "")  # tests only
    if forced:
        return float(forced)
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return None


def claims_path(registry: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(registry)), CLAIMS_FILE)


def card_lock_path(registry: str, card: int) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(registry)), f"gpu{card}.lock")


def append_claims_line(path: str, line: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
    try:
        os.write(fd, (line.rstrip("\n") + "\n").encode())
    finally:
        os.close(fd)


def open_claims_on(path: str, cards: list[int]) -> list[dict]:
    """Open claims in the ledger that name any of cards (both line formats)."""
    out = []
    for c in claims_open(path) or []:
        named = {int(t[3:]) for t in c["cards"].split(",") if t.upper().startswith("GPU") and t[3:].isdigit()}
        if named & set(cards):
            out.append(c)
    return out


def omni_lab(argv: list[str]) -> tuple[bool, str]:
    """omni_lab.py log-run, best effort (a database outage never stops a job).  OC_LANES_OMNI_LAB (tests): a JSON
    command prefix replacing `<intake venv python> omni_lab.py`; OC_LANES_OMNI_LAB_ENV: the env file to load."""
    forced = os.environ.get("OC_LANES_OMNI_LAB", "")
    cmd = json.loads(forced) if forced else [OMNI_LAB_PY, OMNI_LAB_CLI]
    env = dict(os.environ)
    try:
        with open(os.environ.get("OC_LANES_OMNI_LAB_ENV", OMNI_LAB_ENV)) as handle:
            for raw in handle:
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                        v = v[1:-1]
                    env.setdefault(k.strip().removeprefix("export "), v)
    except OSError:
        pass
    try:
        r = subprocess.run([*cmd, "log-run", *argv], capture_output=True, text=True, timeout=30, env=env,
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"log-run failed: {exc}"
    lines = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
    if r.returncode != 0:
        return False, f"log-run rc {r.returncode}: {(r.stderr.strip() or r.stdout.strip())[-300:]}"
    return True, lines[-1] if lines else ""


def lane_note(registry: str, lane_name: str, agent: str, detail: str) -> None:
    """Heartbeat the lane with a claim-run detail line; best effort (never fails the job)."""
    try:
        with RegistryLock(registry):
            reg = load(registry)
            lane = lane_of(reg, lane_name)
            if lane["owner"] != agent:
                return
            lane["heartbeat_utc"] = utc()
            lane["heartbeat_by"] = agent
            lane["heartbeat_source"] = f"claim-run {TOOL_VERSION}"
            lane["detail"] = detail[:500]
            save(registry, reg, f"{agent} claim-run {lane_name}")
    except (Refused, OSError, ValueError) as exc:
        print(f"{utc()} WARN lane heartbeat skipped: {exc}", file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------------ commands (1.1.0)
def cmd_cards(args) -> int:
    reg = load(args.file)
    cmap = card_map(reg)
    if args.json:
        print(json.dumps({str(k): v for k, v in cmap.items()}, indent=1))
        return 0
    for card, holders in cmap.items():
        print(f"GPU{card}: " + "; ".join(f"{h['lane']} ({h['role']}{', ask-first' if h['ask_first'] else ''})"
                                          for h in holders))
    return 0


def json_arg(text: str, what: str):
    if text.startswith("@"):
        try:
            with open(text[1:]) as handle:
                text = handle.read()
        except OSError as exc:
            raise Refused(2, f"{what} {text}: {exc}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise Refused(2, f"{what} is not JSON: {exc}")


def cmd_register(args) -> int:
    obj = json_arg(args.lane_json, "--lane-json")
    if not isinstance(obj, dict) or not isinstance(obj.get("lane"), str) or not obj["lane"].strip():
        raise Refused(2, "--lane-json must be an object with a 'lane' id")
    bad = [k for k in obj if k in REGISTER_PROTECTED]
    if bad:
        raise Refused(2, f"register cannot set {bad} (heartbeat / pulse / takeover fields)")
    if not args.reason.strip():
        raise Refused(2, "--reason must say why")
    name = obj["lane"]
    with RegistryLock(args.file):
        reg = load(args.file)
        stamp = utc()
        if name in reg["lanes"]:
            lane = reg["lanes"][name]
            if lane["owner"] != args.agent and not (args.cyril and args.cyril.strip()):
                raise Refused(3, f"{name} is owned by {lane['owner']}; only the owner re-scopes it (or --cyril)")
            previous = {k: lane.get(k) for k, v in obj.items() if k != "lane" and lane.get(k) != v}
            if not previous:
                print(f"REGISTER {name}: no change")
                return 0
            merged = {**lane, **obj}
            validate_lane(name, merged)
            if "probe" in previous:
                merged.pop("probe_last", None)
            if "status" in previous:
                note_status(merged, obj["status"], args.agent, "register")
            merged.setdefault("history", []).append({"utc": stamp, "by": args.agent, "reason": args.reason,
                                                     "cyril": args.cyril or None, "previous": previous})
            reg["lanes"][name] = merged
            verb = f"RESCOPED ({', '.join(previous)})"
        else:
            lane = dict(obj)
            validate_lane(name, lane)
            note_status(lane, lane["status"], args.agent, "register")
            lane["registered_utc"] = stamp
            lane.setdefault("heartbeat_utc", None)
            lane.setdefault("last_receipt", None)
            lane["history"] = [{"utc": stamp, "by": args.agent, "reason": args.reason, "cyril": args.cyril or None,
                                "previous": None}]
            reg["lanes"][name] = lane
            verb = "ADDED"
        save(args.file, reg, f"{args.agent} register {name}")
    print(f"REGISTER {name} {verb}; registry revision {reg['revision']}")
    return 0


def cmd_set_rule(args) -> int:
    if not args.text.strip() or not args.reason.strip():
        raise Refused(2, "--text and --reason must not be empty")
    with RegistryLock(args.file):
        reg = load(args.file)
        rules = reg.setdefault("rules", {})
        old = rules.get(args.key)
        if old == args.text:
            print(f"SET-RULE {args.key}: no change")
            return 0
        rules[args.key] = args.text
        reg.setdefault("rules_history", []).append({"utc": utc(), "by": args.agent, "key": args.key,
                                                     "previous": old, "reason": args.reason})
        save(args.file, reg, f"{args.agent} set-rule {args.key}")
    print(f"SET-RULE {args.key}; registry revision {reg['revision']}")
    return 0


def cmd_release_stale(args) -> int:
    path = claims_path(args.file)
    token = cards_token(parse_cards(args.cards))
    reason = free_text(args.reason, "--reason", claim_line=False) if args.reason.strip() else ""
    if not reason:
        raise Refused(2, "--reason must say why")
    match = [c for c in claims_open(path) or []
             if c["cards"] == token and c["lane"] == args.claim_lane and c["job"] == args.job]
    if not match:
        raise Refused(2, f"no open claim {token} lane={args.claim_lane} job={args.job} in {path}")
    claim = match[0]
    if claim["pid"] is None:
        raise Refused(5, f"the claim has no pid, so nothing proves it stale: {claim['line']}")
    if claim["pid_alive"]:
        raise Refused(5, f"pid {claim['pid']} is alive: the claim is not stale ({pid_cmdline(claim['pid'])[:120]})")
    owner = dict(t.split("=", 1) for t in claim["line"].split() if "=" in t).get("owner", "unknown")
    line = (f"{utc()} RELEASE {token} owner={owner} lane={args.claim_lane} job={args.job} released "
            f"(stale: pid {claim['pid']} dead; by {args.agent} via lanes.py {TOOL_VERSION} release-stale: {reason})")
    append_claims_line(path, line)
    print(line)
    return 0


def cmd_claim_run(args) -> int:
    cmd = list(args.command)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd and not args.dry_run:
        raise Refused(2, "no command: claim-run ... -- CMD ARGS")
    if not re.fullmatch(JOB_RE, args.job) or args.job.lower() in CLAIM_RELEASE_WORDS:
        raise Refused(2, f"--job {args.job!r}: 1-80 of [A-Za-z0-9._:-], starting alphanumeric, not a release word")
    if not args.log or any(ch.isspace() for ch in args.log) or "=" in args.log:
        raise Refused(2, "--log must be a path without spaces or '='")
    borrow = free_text(args.borrow, "--borrow") if args.borrow else None
    cyril = " ".join(args.cyril.split()) if args.cyril else None
    cards = parse_cards(args.gpus)
    token = cards_token(cards)
    reg = load(args.file)
    lane = lane_of(reg, args.lane)
    if lane["owner"] != args.agent and not cyril:
        raise Refused(3, f"{args.lane} is owned by {lane['owner']}: its GPU jobs start from the owner's launcher")
    notes = card_verdicts(reg, args.lane, cards, borrow, cyril)
    claims = claims_path(args.file)
    fds: list[int] = []
    saved_handlers: dict = {}

    def unlock() -> None:
        while fds:
            fd = fds.pop()
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    try:
        deadline = time.monotonic() + max(0.0, args.wait)
        for card in cards:  # lowest index first: two launchers never deadlock on each other's half
            path = card_lock_path(args.file, card)
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fds.append(fd)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        os.close(fd)
                        raise Refused(5, f"{path} is held by {lock_holders(path)} (waited {args.wait:.0f} s)")
                    time.sleep(1.0)
        open_ = open_claims_on(claims, cards)
        if open_:
            why = "; ".join(f"{c['line'][:200]} (pid {alive_word(c['pid_alive'])})" for c in open_)
            hint = ("; a dead pid is cleared with `lanes.py release-stale`" if any(c["pid_alive"] is False for c in open_)
                    else "")
            raise Refused(5, f"open claim on {token}: {why}{hint}")
        smi = smi_cards()
        missing = [c for c in cards if c not in smi]
        if missing:
            raise Refused(2, f"GPU{missing[0]} is not in nvidia-smi")
        busy = [f"GPU{c} {smi[c]['used_mib']} MiB" for c in cards if smi[c]["used_mib"] > args.busy_limit]
        if busy:
            raise Refused(5, f"not idle (> {args.busy_limit} MiB): {', '.join(busy)}")
        avail = mem_available_gib()
        if avail is not None and avail < args.mem_floor_gib:
            raise Refused(5, f"host MemAvailable {avail:.1f} GiB < floor {args.mem_floor_gib} GiB")
        uuids = [smi[c]["uuid"] for c in cards]
        if args.dry_run:
            print(f"CLAIM-RUN DRY OK {token} lane={args.lane} job={args.job}: {'; '.join(notes)}; "
                  f"MemAvailable {'?' if avail is None else f'{avail:.1f}'} GiB; uuids {','.join(uuids)}")
            return 0
        env = dict(os.environ)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
        os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
        stopped = {"signal": None}
        child_box: dict = {}

        def forward(signum, _frame):
            stopped["signal"] = signum
            child = child_box.get("child")
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signum)
                except ProcessLookupError:
                    pass

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            saved_handlers[sig] = signal.signal(sig, forward)
        nice = args.nice
        note = free_text("; ".join(notes) + (f"; borrow: {borrow}" if borrow else "")
                         + ("; Cyril approved" if cyril else ""), "claim note")

        def best_effort(what: str, fn, *a) -> None:
            """Bookkeeping after the job started must never abandon it (the locks stay held until it exits)."""
            try:
                fn(*a)
            except OSError as exc:
                print(f"{utc()} WARN {what} failed: {exc}", file=sys.stderr, flush=True)

        try:
            with open(args.log, "ab") as logf:
                child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                         start_new_session=True, preexec_fn=(lambda: os.nice(nice)) if nice else None)
        except OSError as exc:
            raise Refused(2, f"cannot start {cmd[0]!r}: {exc}")
        child_box["child"] = child
        started = utc()
        claim_line = (f"{started} CLAIM {token} owner={args.agent} lane={args.lane} job={args.job} pid={child.pid} "
                      f"wrapper={os.getpid()} log={os.path.abspath(args.log)} preempt: kill -TERM {os.getpid()} "
                      f"({note})")
        best_effort("CLAIM line", append_claims_line, claims, claim_line)
        print(claim_line, flush=True)
        run_id = None
        if not args.no_omni_lab:
            config = {"cmd": cmd, "cards": cards, "uuids": uuids, "claim": claim_line, "tool": f"{TOOL_NAME} {TOOL_VERSION}",
                      "borrow": borrow, "cyril": cyril}
            ok, out = omni_lab(["start", "--kind", args.omni_lab_kind, "--name", args.job, "--lane", args.lane,
                                "--gpus", ",".join(map(str, cards)), "--log", os.path.abspath(args.log),
                                "--owner", args.agent, "--config", json.dumps(config)])
            run_id = int(out) if ok and out.isdigit() else None
            if run_id is None:
                print(f"{utc()} WARN omni_lab log-run start: {out}", file=sys.stderr, flush=True)
        if args.ready_file:
            best_effort("ready file", atomic_write_json, args.ready_file,
                        {"wrapper_pid": os.getpid(), "pid": child.pid, "cards": cards, "uuids": uuids,
                         "lane": args.lane, "job": args.job, "claim_line": claim_line, "omni_lab_run_id": run_id,
                         "started_utc": started})
        lane_note(args.file, args.lane, args.agent, f"{started} claim-run {args.job} on {token} pid {child.pid}")
        rc = child.wait()
        status = "pass" if rc == 0 else "stopped" if stopped["signal"] is not None else "fail"
        finished = utc()
        sig_note = f", signal {stopped['signal']} forwarded" if stopped["signal"] is not None else ""
        best_effort("RELEASE line", append_claims_line, claims, f"{finished} RELEASE {token} owner={args.agent} "
                    f"lane={args.lane} job={args.job} released (rc {rc}{sig_note})")
        if run_id is not None:
            ok, out = omni_lab(["end", str(run_id), "--status", status, "--exit-code", str(rc),
                                "--log", os.path.abspath(args.log)])
            if not ok:
                print(f"{utc()} WARN omni_lab log-run end: {out}", file=sys.stderr, flush=True)
        if args.exit_file:
            best_effort("exit file", atomic_write_json, args.exit_file,
                        {"rc": rc, "status": status, "finished_utc": finished, "omni_lab_run_id": run_id,
                         "pid": child.pid})
        lane_note(args.file, args.lane, args.agent, f"{finished} claim-run {args.job} ended {status} rc {rc}")
        print(f"{finished} RELEASE {token} job={args.job} rc {rc} status {status}", flush=True)
        return rc if rc >= 0 else 128 - rc
    finally:
        unlock()
        for sig, handler in saved_handlers.items():
            signal.signal(sig, handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="lanes.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"{TOOL_NAME} {TOOL_VERSION}")
    ap.add_argument("--file", default=os.environ.get("OC_LANES_FILE", DEFAULT_FILE), help="registry (LANES.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show")
    p.add_argument("--lane")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("heartbeat")
    p.add_argument("--lane", required=True)
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--status")
    p.add_argument("--detail")
    p.add_argument("--receipt")
    p.add_argument("--transferable", choices=("yes", "no"))
    p.add_argument("--transferable-reason")
    p.add_argument("--source", help="who heartbeats: cli (default), a queue's VERSION string, ...")
    p = sub.add_parser("pulse")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("takeover")
    p.add_argument("--lane", required=True)
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--reason", required=True)
    p.add_argument("--state-file", required=True, help="the newest STATE_<date>_<tag>.md, read before acting")
    p.add_argument("--live-check", required=True, help="what systemctl / docker ps / nvidia-smi / pgrep showed")
    p.add_argument("--synced", action="store_true", help="attest that `session.py --agent <you> sync` ran first")
    p.add_argument("--cyril", help="Cyril's words authorising the takeover, verbatim")
    p.add_argument("--no-gpu-snapshot", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("set-probe")
    p.add_argument("--lane", required=True)
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--probe", required=True, help="probe JSON, or @FILE; '{}' clears it")
    p = sub.add_parser("init")
    p.add_argument("--seed", required=True)
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("hold-lock")
    p.add_argument("--lock", required=True)
    p.add_argument("--while-pid", required=True, type=int)
    p.add_argument("--cmd-match", required=True)
    p.add_argument("--also-match", action="append", default=[],
                   help="keep holding while any process has this in its cmdline (repeatable; the card's jobs)")
    p.add_argument("--join-helper", action="store_true",
                   help="a lock held only by other hold-lock helpers counts as ready; wait (no deadline) to take it")
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a busy lock before refusing (exit 5)")
    p.add_argument("--ready-file", help="written (holder pid) once the lock is held, or '<pid> joined <helper pids>'")
    p = sub.add_parser("cards")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("register")
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--lane-json", required=True, help="lane object JSON, or @FILE")
    p.add_argument("--reason", required=True)
    p.add_argument("--cyril", help="Cyril's words, verbatim, when re-scoping a lane you do not own")
    p = sub.add_parser("set-rule")
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--key", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--reason", required=True)
    p = sub.add_parser("release-stale")
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--cards", required=True, help="exactly as in the CLAIM line, e.g. GPU1,GPU2")
    p.add_argument("--claim-lane", required=True, help="the lane= of the CLAIM line")
    p.add_argument("--job", required=True, help="the job= of the CLAIM line")
    p.add_argument("--reason", required=True)
    p = sub.add_parser("claim-run")
    p.add_argument("--lane", required=True)
    p.add_argument("--agent", required=True, type=agent_arg)
    p.add_argument("--gpus", required=True, help="nvidia-smi indexes, e.g. 3,5")
    p.add_argument("--job", required=True)
    p.add_argument("--log", required=True, help="the job's stdout+stderr (appended)")
    p.add_argument("--borrow", help="why this job takes another lane's (non-serving) card")
    p.add_argument("--cyril", help="Cyril's words, verbatim: needed for a serving card or an unassigned one")
    p.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a busy card lock (default: refuse)")
    p.add_argument("--busy-limit", type=int, default=DEFAULT_BUSY_MIB, help="refuse a card using more MiB")
    p.add_argument("--mem-floor-gib", type=float, default=DEFAULT_MEM_FLOOR_GIB)
    p.add_argument("--nice", type=int, default=10)
    p.add_argument("--ready-file", help="JSON written once the job runs (pid, cards, uuids, omni_lab run id)")
    p.add_argument("--exit-file", help="JSON written when the job ends (rc, status)")
    p.add_argument("--omni-lab-kind", default="other",
                   choices=("fold", "distill", "cache", "gen", "eval", "probe", "build", "export", "review", "smoke",
                            "other"))
    p.add_argument("--no-omni-lab", action="store_true", help="do not register the run in omni_lab")
    p.add_argument("--dry-run", action="store_true", help="every check, no job")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- CMD ARGS")
    args = ap.parse_args(argv)
    fn = {"show": cmd_show, "heartbeat": cmd_heartbeat, "pulse": cmd_pulse, "takeover": cmd_takeover,
          "set-probe": cmd_set_probe, "init": cmd_init, "hold-lock": cmd_hold_lock, "cards": cmd_cards,
          "register": cmd_register, "set-rule": cmd_set_rule, "release-stale": cmd_release_stale,
          "claim-run": cmd_claim_run}[args.cmd]
    try:
        return fn(args)
    except Refused as exc:
        print(f"REFUSED ({exc.code}): {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
