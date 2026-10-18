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
             from /proc/locks, never by taking the lock), newest receipt
  heartbeat  --lane L --agent A [--status S] [--detail T] [--receipt PATH] [--transferable yes|no] [--source S]
             the lane's OWNER only; a receipt is stored with its sha256
  pulse      [--dry-run]  for every lane with a probe (systemd unit active / pid cmdline alive / file fresh, all must
             pass): heartbeat it on the owner's behalf and record its newest receipt from the probe's receipt globs.
             Run by cron every 10 min, so a lane whose queue runs unattended stays fresh while its process lives and
             goes stale once it stops.  A failing probe never heartbeats; it is recorded in probe_last.
  takeover   --lane L --agent A --reason T --state-file PATH --live-check T --synced [--cyril "verbatim quote"]
             plan 7.4.5: allowed only if Cyril said so (--cyril) or the lane is transferable and its heartbeat (or,
             if it never had one, its registration) is older than 6 h.  Writes a takeover receipt next to the
             registry (lanes_takeovers/), runs the lane's probe and prints DO NOT RELAUNCH when the process is alive.
  init       --seed SEED.json [--replace]   create the registry (an existing one only with --replace, backed up first)
  hold-lock  --lock PATH --while-pid PID --cmd-match S [--poll S] [--wait S] [--ready-file PATH]
             hold an exclusive flock on PATH while pid PID lives with S in its cmdline; a busy lock is waited for up
             to --wait s, then refused (exit 5).  Used (a) by queue_gpu6.sh >= 1.3.1 for its card lock, so the
             jobs it starts never inherit the lock fd and an adopted job does not block the next queue, and (b) as
             a bridge for a queue that started without one (queue_gpu6.sh 1.3.0 holds no gpu6.lock)

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
import subprocess
import sys
import time
from datetime import datetime, timezone

TOOL_NAME = "lanes"
TOOL_VERSION = "1.0.0"   # 1.0.0 (2026-10-14): first version, card X-T2
SCHEMA = "omni-campaign.lanes.v1"
TAKEOVER_SCHEMA = "omni-campaign.lane-takeover.v1"
DEFAULT_FILE = "/home/cereal/omni-campaign/distill/queues/LANES.json"
AGENTS = ("claude", "codex")
STALE_S = 6 * 3600
LOCK_WAIT_S = 30.0
REQUIRED = ("lane", "owner", "scope", "lock_paths", "status", "transferable", "transferable_reason")
PROBE_KINDS = ("systemd_unit", "pid_cmdline", "file_fresh")


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
                     "heartbeat_utc": lane.get("heartbeat_utc"), "heartbeat_source": lane.get("heartbeat_source"),
                     "age_s": None if age == float("inf") else round(age), "age_basis": basis,
                     "stale": age > STALE_S, "transferable": lane["transferable"],
                     "takeover_allowed_without_cyril": allowed, "takeover_rule": why,
                     "probe_ok": (lane.get("probe_last") or {}).get("ok"), "lock_holders": locks,
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
        print(f"{r['lane']:<14} owner={r['owner']:<6} status={r['status']}")
        print(f"{'':<14} heartbeat {r['heartbeat_utc']} ({age} by {r['age_basis']}, {r['heartbeat_source']}) "
              f"stale={'yes' if r['stale'] else 'no'} probe_ok={r['probe_ok']} takeover: {r['takeover_rule']}")
        print(f"{'':<14} locks {locks}; receipt {rc.get('path', '-')} {str(rc.get('sha256', ''))[:16]}")
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
            lane["status"] = args.status
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
        stamp = utc()
        rec = {"schema": TAKEOVER_SCHEMA, "tool": f"{TOOL_NAME} {TOOL_VERSION}", "lane": args.lane, "utc": stamp,
               "from_owner": lane["owner"], "to_owner": args.agent, "rule": "cyril" if args.cyril else "stale-transferable",
               "rule_detail": why, "cyril_quote": args.cyril or None, "reason": args.reason,
               "heartbeat_utc_before": lane.get("heartbeat_utc"), "staleness_s": None if age == float("inf") else round(age),
               "staleness_basis": basis, "transferable": lane["transferable"],
               "state_file": receipt_info(args.state_file), "synced_attested": True,
               "live_check_note": args.live_check, "probe": probe,
               "process_alive": bool(probe and probe["ok"]),
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
        lane["status"] = f"taken over by {args.agent} at {stamp}"
        lane["heartbeat_utc"] = stamp
        lane["heartbeat_by"] = args.agent
        lane["heartbeat_source"] = "takeover"
        lane["takeover"] = {"utc": stamp, "receipt": info}
        save(args.file, reg, f"{args.agent} takeover {args.lane}")
    print(f"TAKEOVER {args.lane} {rec['from_owner']} -> {args.agent} ({why}); receipt {info['path']} sha256 {info['sha256']}")
    if rec["process_alive"]:
        print(f"DO NOT RELAUNCH: the lane's process is alive ({'; '.join(c['detail'] for c in probe['checks'])})")
    print("next: post the five-line hand-off on Relay (lane; state + receipt path/sha; running; next command; do not touch)")
    return 0


def validate_lane(name: str, lane: dict) -> None:
    missing = [k for k in REQUIRED if k not in lane]
    if missing:
        raise Refused(2, f"lane {name}: missing {missing}")
    if lane["lane"] != name or lane["owner"] not in AGENTS or not isinstance(lane["lock_paths"], list) \
            or not isinstance(lane["transferable"], bool):
        raise Refused(2, f"lane {name}: lane/owner/lock_paths/transferable malformed")
    for c in (lane.get("probe") or {}).get("checks", []):
        if c.get("kind") not in PROBE_KINDS:
            raise Refused(2, f"lane {name}: probe kind {c.get('kind')!r} not in {PROBE_KINDS}")


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


def cmd_hold_lock(args) -> int:
    if args.cmd_match not in pid_cmdline(args.while_pid):
        raise Refused(2, f"pid {args.while_pid} is not alive with '{args.cmd_match}' in its cmdline")
    fd = os.open(args.lock, os.O_RDWR | os.O_CREAT, 0o664)
    deadline = time.monotonic() + args.wait
    while True:  # wait, never compete: the lock is only ever taken when free
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline or args.cmd_match not in pid_cmdline(args.while_pid):
                os.close(fd)
                raise Refused(5, f"{args.lock} is held by {lock_holders(args.lock)} (waited {args.wait:.0f} s)")
            time.sleep(1.0)
    print(f"{utc()} HOLD {args.lock} for pid {args.while_pid} ('{args.cmd_match}'), poll {args.poll}s, holder pid {os.getpid()}",
          flush=True)
    if args.ready_file:
        with open(args.ready_file, "w") as handle:
            handle.write(f"{os.getpid()}\n")
    while args.cmd_match in pid_cmdline(args.while_pid):
        time.sleep(args.poll)
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
    print(f"{utc()} RELEASE {args.lock}: pid {args.while_pid} is gone", flush=True)
    return 0


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
    p = sub.add_parser("init")
    p.add_argument("--seed", required=True)
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("hold-lock")
    p.add_argument("--lock", required=True)
    p.add_argument("--while-pid", required=True, type=int)
    p.add_argument("--cmd-match", required=True)
    p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--wait", type=float, default=0.0, help="seconds to wait for a busy lock before refusing (exit 5)")
    p.add_argument("--ready-file", help="written (holder pid) once the lock is held")
    args = ap.parse_args(argv)
    fn = {"show": cmd_show, "heartbeat": cmd_heartbeat, "pulse": cmd_pulse, "takeover": cmd_takeover,
          "init": cmd_init, "hold-lock": cmd_hold_lock}[args.cmd]
    try:
        return fn(args)
    except Refused as exc:
        print(f"REFUSED ({exc.code}): {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
