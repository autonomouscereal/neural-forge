"""Tests for lanes.py 1.1.0 (card X-T2, lane infra, Claude, 2026-10-14; 1.0.1-1.0.2 cases 2026-10-15; 1.1.0 cases
2026-10-21, P40 lane policy).  AI server (Linux: /proc/locks, flock), CPU:

    cd /home/cereal/omni-campaign/distill/tools
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider -q test_lanes.py

What is pinned (every registry here is a temp file; the live $Q/LANES.json is never read or written):
  * init writes the schema, stamps registration, hashes a seeded receipt, refuses to overwrite without --replace and
    backs up with it; a malformed seed (duplicate lane, bad owner, unknown probe kind) is refused;
  * heartbeat is owner-only (exit 3), stores status / detail / receipt sha256 / transferable, bumps the revision;
  * takeover follows plan 7.4.5: refused for a fresh lane, a non-transferable lane, the owner itself, no --synced,
    a missing STATE file; allowed when transferable and stale > 6 h (from heartbeat, or registration if none) or
    with Cyril's quote; it writes a receipt (sha in the registry), moves ownership, and says DO NOT RELAUNCH when
    the lane's probe finds its process alive;
  * pulse heartbeats a lane whose probe passes (pid cmdline + fresh file), records its newest receipt by glob,
    leaves a failing lane's heartbeat alone (probe_last.ok False), ignores lanes without a probe;
  * lock holders come from /proc/locks without taking the lock; hold-lock holds while the pid lives, then releases;
    a busy lock is refused (exit 5);
  * the registry write is atomic and serialised: 8 concurrent heartbeats all land (revision +8);
  * 1.0.1: a takeover of a lane with no probe records process_alive null + "unknown (no probe)" and prints LIVENESS
    UNKNOWN, never a false "not alive"; open *.claims lines (both ledger formats) are parsed, printed and recorded;
    a probe's status_ok / status_down drive the lane's status; set-probe is owner-only and validated; hold-lock
    --also-match keeps the lock while a job outlives its queue; --join-helper joins only an ORPHANED helper (its
    queue gone) and takes the lock when that helper releases it; a helper whose queue lives is never joined.
  * 1.1.0: register adds a lane or re-scopes an owned one (history keeps the replaced values; protected fields and
    malformed cards are refused; another owner's lane only with --cyril); set-rule keeps rules_history; the card map
    and card_verdicts (own / flex free, another lane's P40 only with --borrow, a serving card only with --cyril,
    an unlisted card only with --cyril); release-stale releases only a dead-pid claim; claim-run starts the job with
    the cards' UUIDs, holds every gpu<N>.lock without the job inheriting it, writes CLAIM then RELEASE lines that
    claims_open AND local-hermes remote_runtime.active_p40_claims both read correctly, passes the exit code
    through, registers start/end with omni_lab, forwards TERM (status stopped), and refuses (nothing written) on a
    held lock, an open claim, a busy card, the MemAvailable floor, a serving card, a bad job name or a non-owner.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lanes  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists("/proc/locks"), reason="Linux only (/proc, flock)")


def seed(tmp_path: Path, **over) -> Path:
    receipt = tmp_path / "r0.json"
    receipt.write_text('{"x": 1}\n')
    lanes_ = [
        {"lane": "L-A", "owner": "claude", "scope": "a", "lock_paths": [str(tmp_path / "gpu9.lock")], "status": "running",
         "transferable": True, "transferable_reason": "test", "last_receipt_path": str(receipt)},
        {"lane": "L-B", "owner": "codex", "scope": "b", "lock_paths": [], "status": "registered",
         "transferable": False, "transferable_reason": "test"},
    ]
    for ln in lanes_:
        ln.update(over.get(ln["lane"], {}))
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"created_by": "test", "lanes": lanes_}))
    return p


def run(reg: Path, *argv, now: str | None = None, capsys=None):
    env_before = os.environ.get("OC_LANES_NOW")
    if now:
        os.environ["OC_LANES_NOW"] = now
    try:
        rc = lanes.main(["--file", str(reg), *argv])
    finally:
        if now:
            if env_before is None:
                os.environ.pop("OC_LANES_NOW", None)
            else:
                os.environ["OC_LANES_NOW"] = env_before
    out = capsys.readouterr() if capsys else None
    return rc, out


def read(reg: Path) -> dict:
    return json.loads(reg.read_text())


@pytest.fixture
def reg(tmp_path, capsys):
    r = tmp_path / "LANES.json"
    rc, _ = run(r, "init", "--seed", str(seed(tmp_path)), now="2026-10-14T20:00:00Z", capsys=capsys)
    assert rc == 0
    return r


def state_file(tmp_path: Path) -> Path:
    s = tmp_path / "STATE_2026-10-14_TEST.md"
    s.write_text("state\n")
    return s


def test_init_writes_schema_and_receipt(reg, tmp_path, capsys):
    d = read(reg)
    assert d["schema"] == lanes.SCHEMA and d["revision"] == 1 and list(d["lanes"]) == ["L-A", "L-B"]
    a = d["lanes"]["L-A"]
    assert a["registered_utc"] == "2026-10-14T20:00:00Z" and a["heartbeat_utc"] is None
    assert a["last_receipt"]["sha256"] == lanes.sha256_file(str(tmp_path / "r0.json"))
    rc, out = run(reg, "init", "--seed", str(seed(tmp_path)), capsys=capsys)
    assert rc == 2 and "exists" in out.err
    rc, out = run(reg, "init", "--seed", str(seed(tmp_path)), "--replace", capsys=capsys)
    assert rc == 0 and list(tmp_path.glob("LANES.json.*.bak"))


@pytest.mark.parametrize("over,why", [
    ({"L-B": {"lane": "L-A"}}, "duplicate"),
    ({"L-A": {"owner": "gpt"}}, "malformed"),
    ({"L-A": {"probe": {"checks": [{"kind": "ping"}]}}}, "probe kind"),
])
def test_init_refuses_bad_seed(tmp_path, capsys, over, why):
    rc, out = run(tmp_path / "LANES.json", "init", "--seed", str(seed(tmp_path, **over)), capsys=capsys)
    assert rc == 2 and why in out.err
    assert not (tmp_path / "LANES.json").exists()


def test_heartbeat_owner_only(reg, tmp_path, capsys):
    rc, out = run(reg, "heartbeat", "--lane", "L-B", "--agent", "claude", capsys=capsys)
    assert rc == 3 and "owned by codex" in out.err
    r1 = tmp_path / "r1.json"
    r1.write_text("{}\n")
    rc, _ = run(reg, "heartbeat", "--lane", "L-A", "--agent", "Claude", "--status", "waiting", "--detail", "d",
                "--receipt", str(r1), "--transferable", "no", "--source", "queue_x 1.0", now="2026-10-14T21:00:00Z",
                capsys=capsys)
    assert rc == 0
    d = read(reg)
    a = d["lanes"]["L-A"]
    assert (a["heartbeat_utc"], a["heartbeat_by"], a["heartbeat_source"], a["status"], a["detail"]) == \
        ("2026-10-14T21:00:00Z", "claude", "queue_x 1.0", "waiting", "d")
    assert a["last_receipt"]["path"] == str(r1) and a["transferable"] is False and d["revision"] == 2
    rc, out = run(reg, "heartbeat", "--lane", "L-A", "--agent", "claude", "--receipt", str(tmp_path / "nope"), capsys=capsys)
    assert rc == 2 and "not a file" in out.err


def takeover(reg, tmp_path, capsys, lane="L-A", agent="codex", now="2026-10-15T03:00:00Z", extra=()):
    return run(reg, "takeover", "--lane", lane, "--agent", agent, "--reason", "owner out of usage",
               "--state-file", str(state_file(tmp_path)), "--live-check", "pgrep: none", "--synced",
               "--no-gpu-snapshot", *extra, now=now, capsys=capsys)


def test_takeover_rules(reg, tmp_path, capsys):
    # registered 20:00Z, never heartbeated: 5 h later is fresh by registration
    rc, out = takeover(reg, tmp_path, capsys, now="2026-10-15T01:00:00Z")
    assert rc == 4 and "registration 5.0 h old" in out.err
    rc, out = takeover(reg, tmp_path, capsys, agent="claude")
    assert rc == 2 and "already owns" in out.err
    rc, out = takeover(reg, tmp_path, capsys, lane="L-B", agent="claude", now="2026-10-16T00:00:00Z")
    assert rc == 4 and "not transferable" in out.err
    rc, out = run(reg, "takeover", "--lane", "L-A", "--agent", "codex", "--reason", "r", "--state-file",
                  str(state_file(tmp_path)), "--live-check", "x", "--no-gpu-snapshot", capsys=capsys)
    assert rc == 2 and "sync" in out.err
    rc, out = run(reg, "takeover", "--lane", "L-A", "--agent", "codex", "--reason", "r", "--state-file",
                  str(tmp_path / "missing.md"), "--live-check", "x", "--synced", "--no-gpu-snapshot", capsys=capsys)
    assert rc == 2 and "STATE" in out.err
    # a heartbeat restarts the clock: 20:30Z heartbeat, 03:00Z is 6.5 h later -> allowed
    run(reg, "heartbeat", "--lane", "L-A", "--agent", "claude", now="2026-10-14T20:30:00Z", capsys=capsys)
    rc, out = takeover(reg, tmp_path, capsys, now="2026-10-15T02:00:00Z")
    assert rc == 4 and "heartbeat 5.5 h old" in out.err
    rc, out = takeover(reg, tmp_path, capsys, now="2026-10-15T03:00:00Z")
    assert rc == 0 and "TAKEOVER L-A claude -> codex" in out.out and "DO NOT RELAUNCH" not in out.out
    a = read(reg)["lanes"]["L-A"]
    assert a["owner"] == "codex" and a["previous_owner"] == "claude" and a["heartbeat_source"] == "takeover"
    rec_path = Path(a["takeover"]["receipt"]["path"])
    assert rec_path.parent.name == "lanes_takeovers"
    assert lanes.sha256_file(str(rec_path)) == a["takeover"]["receipt"]["sha256"]
    rec = json.loads(rec_path.read_text())
    assert rec["rule"] == "stale-transferable" and rec["staleness_basis"] == "heartbeat" and rec["staleness_s"] == 23400
    assert rec["state_file"]["sha256"] == lanes.sha256_file(str(state_file(tmp_path)))
    # the old owner is now refused a heartbeat
    rc, out = run(reg, "heartbeat", "--lane", "L-A", "--agent", "claude", capsys=capsys)
    assert rc == 3


def test_takeover_with_cyril_and_live_process(reg, tmp_path, capsys):
    sleeper = subprocess.Popen(["sleep", "600.4271"])  # a duration nothing else on the host uses
    try:
        d = read(reg)
        d["lanes"]["L-B"]["probe"] = {"checks": [{"kind": "pid_cmdline", "match": "sleep 600.4271"}]}
        reg.write_text(json.dumps(d))
        rc, out = takeover(reg, tmp_path, capsys, lane="L-B", agent="claude", now="2026-10-14T20:10:00Z",
                           extra=("--cyril", "Claude, take the data lane"))
        assert rc == 0 and "DO NOT RELAUNCH" in out.out
        b = read(reg)["lanes"]["L-B"]
        rec = json.loads(Path(b["takeover"]["receipt"]["path"]).read_text())
        assert rec["rule"] == "cyril" and rec["cyril_quote"] == "Claude, take the data lane" and rec["process_alive"]
    finally:
        sleeper.kill()
        sleeper.wait()


def test_pulse(reg, tmp_path, capsys):
    status = tmp_path / "status.json"
    status.write_text('{"accepted": 3, "eta_hours": 1.5}')
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    (rdir / "old.json").write_text("{}")
    os.utime(rdir / "old.json", (time.time() - 100, time.time() - 100))
    (rdir / "new.json").write_text('{"n": 2}')
    sleeper = subprocess.Popen(["sleep", "601.3917"])
    try:
        d = read(reg)
        d["lanes"]["L-A"]["probe"] = {
            "checks": [{"kind": "pid_cmdline", "match": "sleep 601.3917"},
                       {"kind": "file_fresh", "path": str(status), "max_age_s": 600}],
            "receipt_globs": [str(rdir / "*.json")],
            "detail": {"path": str(status), "mode": "json_keys", "keys": ["accepted", "eta_hours"]}}
        d["lanes"]["L-B"]["probe"] = {"checks": [{"kind": "pid_cmdline", "match": "no-such-process-oc-lanes"}]}
        reg.write_text(json.dumps(d))
        rc, out = run(reg, "pulse", capsys=capsys)
        assert rc == 0 and "PULSE L-A ok" in out.out and "PULSE L-B DOWN" in out.out
        d = read(reg)
        a, b = d["lanes"]["L-A"], d["lanes"]["L-B"]
        assert a["heartbeat_source"] == "pulse" and a["heartbeat_by"] == "claude" and a["probe_last"]["ok"]
        assert a["detail"] == "accepted=3 eta_hours=1.5" and a["last_receipt"]["path"] == str(rdir / "new.json")
        assert b["heartbeat_utc"] is None and b["probe_last"]["ok"] is False
        rev = d["revision"]
        rc, out = run(reg, "pulse", "--dry-run", capsys=capsys)
        assert rc == 0 and "(dry run)" in out.out and read(reg)["revision"] == rev
    finally:
        sleeper.kill()
        sleeper.wait()
    os.utime(status, (time.time() - 3600, time.time() - 3600))
    rc, out = run(reg, "pulse", capsys=capsys)
    assert "PULSE L-A DOWN" in out.out


def test_lock_holders_and_hold_lock(tmp_path, capsys):
    lock = tmp_path / "gpu9.lock"
    lock.touch()
    assert lanes.lock_holders(str(lock)) == []
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        held = lanes.lock_holders(str(lock))
        assert [h["pid"] for h in held] == [os.getpid()] and held[0]["kind"] == "FLOCK"
        rc = lanes.main(["hold-lock", "--lock", str(lock), "--while-pid", str(os.getpid()), "--cmd-match", "python"])
        assert rc == 5
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    target = subprocess.Popen(["sleep", "4.0731"])
    holder = subprocess.Popen([sys.executable, str(HERE / "lanes.py"), "hold-lock", "--lock", str(lock), "--while-pid",
                               str(target.pid), "--cmd-match", "sleep 4.0731", "--poll", "0.5"],
                              stdout=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not lanes.lock_holders(str(lock)):
            time.sleep(0.1)
        assert [h["pid"] for h in lanes.lock_holders(str(lock))] == [holder.pid]
        target.wait()
        out, _ = holder.communicate(timeout=15)
        assert holder.returncode == 0 and "HOLD" in out and "RELEASE" in out
        assert lanes.lock_holders(str(lock)) == []
    finally:
        target.kill()
        holder.kill()
    rc = lanes.main(["hold-lock", "--lock", str(lock), "--while-pid", "999999999", "--cmd-match", "x"])
    assert rc == 2


def test_hold_lock_waits_then_takes(tmp_path):
    lock, ready = tmp_path / "gpu9.lock", tmp_path / "ready"
    lock.touch()
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    target = subprocess.Popen(["sleep", "30.5813"])
    holder = subprocess.Popen([sys.executable, str(HERE / "lanes.py"), "hold-lock", "--lock", str(lock), "--while-pid",
                               str(target.pid), "--cmd-match", "sleep 30.5813", "--poll", "0.5", "--wait", "20",
                               "--ready-file", str(ready)], stdout=subprocess.PIPE, text=True)
    try:
        time.sleep(1.5)
        assert not ready.exists() and holder.poll() is None  # waiting, not competing
        fcntl.flock(fd, fcntl.LOCK_UN)
        deadline = time.time() + 10
        while time.time() < deadline and not ready.exists():
            time.sleep(0.1)
        assert ready.read_text().strip() == str(holder.pid)
        assert [h["pid"] for h in lanes.lock_holders(str(lock))] == [holder.pid]
    finally:
        os.close(fd)
        target.kill()
        target.wait()
    out, _ = holder.communicate(timeout=15)
    assert holder.returncode == 0 and "RELEASE" in out
    # waiting is bounded: a lock held past --wait is refused
    fd = os.open(lock, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        t0 = time.time()
        rc = lanes.main(["hold-lock", "--lock", str(lock), "--while-pid", str(os.getpid()), "--cmd-match", "python",
                         "--wait", "2"])
        assert rc == 5 and 1.5 < time.time() - t0 < 10
    finally:
        os.close(fd)


def test_concurrent_heartbeats_all_land(reg, tmp_path):
    rev = read(reg)["revision"]
    procs = [subprocess.Popen([sys.executable, str(HERE / "lanes.py"), "--file", str(reg), "heartbeat", "--lane", "L-A",
                               "--agent", "claude", "--detail", f"writer {i}"], stdout=subprocess.DEVNULL)
             for i in range(8)]
    assert [p.wait(timeout=60) for p in procs] == [0] * 8
    d = read(reg)
    assert d["revision"] == rev + 8 and d["lanes"]["L-A"]["detail"].startswith("writer ")
    assert not list(tmp_path.glob(".LANES.json.*.tmp"))


def test_show(reg, tmp_path, capsys):
    rc, out = run(reg, "show", "--json", now="2026-10-15T03:00:00Z", capsys=capsys)
    assert rc == 0
    rows = {r["lane"]: r for r in json.loads(out.out)["lanes"]}
    assert rows["L-A"]["stale"] and rows["L-A"]["takeover_allowed_without_cyril"]
    assert rows["L-A"]["lock_holders"] == {str(tmp_path / "gpu9.lock"): None}  # lock file absent -> unknowable
    assert not rows["L-B"]["takeover_allowed_without_cyril"]
    rc, out = run(reg, "show", "--lane", "L-A", capsys=capsys)
    assert rc == 0 and "L-A" in out.out and "L-B" not in out.out
    rc, out = run(tmp_path / "none.json", "show", capsys=capsys)
    assert rc == 6


# ------------------------------------------------------------------------------------------------ 1.0.1
def claims_file(tmp_path: Path, live_pid: int) -> Path:
    c = tmp_path / "p40.claims"
    c.write_text("\n".join([
        "2026-10-14T19:43:02Z GPU3 lane=k3accel owner=claude job=k3-v130-micro pid=999999991 log=x",
        "2026-10-14T19:43:09Z GPU3 lane=k3accel job=k3-v130-micro released",
        f"2026-10-14T23:18:14Z GPU3 lane=k3accel owner=claude job=k3-v130-parity pid={live_pid} log=y",
        "2026-10-14T23:20:00Z CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=999999992 log=z",
        "2026-10-14T23:21:00Z CLAIM GPU1 owner=codex lane=sp job=s0 pid=999999993",
        "2026-10-14T23:22:00Z RELEASE GPU1 owner=codex lane=sp job=s0 released",
        "",
    ]))
    return c


def test_claims_open_both_formats(tmp_path):
    c = claims_file(tmp_path, os.getpid())
    got = {(x["cards"], x["job"]): (x["pid"], x["pid_alive"]) for x in lanes.claims_open(str(c))}
    assert got == {("GPU3", "k3-v130-parity"): (os.getpid(), True),
                   ("GPU1,GPU2,GPU3", "d1-review"): (999999992, False)}
    assert lanes.claims_open(str(tmp_path / "absent.claims")) is None


def test_takeover_without_probe_is_unknown_not_dead(tmp_path, capsys):
    sleeper = subprocess.Popen(["sleep", "602.6113"])
    try:
        c = claims_file(tmp_path, sleeper.pid)
        r = tmp_path / "LANES.json"
        rc, _ = run(r, "init", "--seed", str(seed(tmp_path, **{"L-A": {"lock_paths": [str(c)]}})),
                    now="2026-10-14T20:00:00Z", capsys=capsys)
        assert rc == 0
        rc, out = run(r, "show", "--lane", "L-A", capsys=capsys)
        assert rc == 0
        assert f"open claim in p40.claims: GPU3 lane=k3accel job=k3-v130-parity pid={sleeper.pid} (alive)" in out.out
        rc, out = takeover(r, tmp_path, capsys)
        assert rc == 0 and "LIVENESS UNKNOWN" in out.out and "DO NOT RELAUNCH" not in out.out
        assert f"OPEN CLAIM in {c}:" in out.out and "k3-v130-parity" in out.out
        rec = json.loads(Path(read(r)["lanes"]["L-A"]["takeover"]["receipt"]["path"]).read_text())
        assert rec["process_alive"] is None and rec["liveness"] == "unknown (no probe)" and rec["probe"] is None
        assert [x["job"] for x in rec["claims_open"][str(c)]] == ["k3-v130-parity", "d1-review"]
    finally:
        sleeper.kill()
        sleeper.wait()


def test_takeover_with_a_down_probe_is_false(reg, tmp_path, capsys):
    d = read(reg)
    d["lanes"]["L-A"]["probe"] = {"checks": [{"kind": "pid_cmdline", "match": "no-such-process-oc-lanes"}]}
    reg.write_text(json.dumps(d))
    rc, out = takeover(reg, tmp_path, capsys)
    assert rc == 0 and "LIVENESS UNKNOWN" not in out.out
    rec = json.loads(Path(read(reg)["lanes"]["L-A"]["takeover"]["receipt"]["path"]).read_text())
    assert rec["process_alive"] is False and rec["liveness"] == "probe down"


def test_pulse_status_follows_probe(reg, capsys):
    sleeper = subprocess.Popen(["sleep", "603.2297"])
    try:
        rc, _ = run(reg, "set-probe", "--lane", "L-A", "--agent", "claude", "--probe", json.dumps(
            {"checks": [{"kind": "pid_cmdline", "match": "sleep 603.2297"}],
             "status_ok": "running (probe ok)", "status_down": "idle (probe down)"}), capsys=capsys)
        assert rc == 0
        run(reg, "pulse", capsys=capsys)
        assert read(reg)["lanes"]["L-A"]["status"] == "running (probe ok)"
    finally:
        sleeper.kill()
        sleeper.wait()
    run(reg, "pulse", capsys=capsys)
    a = read(reg)["lanes"]["L-A"]
    assert a["status"] == "idle (probe down)" and a["probe_last"]["ok"] is False


def test_set_probe(reg, tmp_path, capsys):
    good = {"checks": [{"kind": "file_fresh", "path": str(tmp_path / "seed.json"), "max_age_s": 600}],
            "receipt_globs": [str(tmp_path / "*.json")]}
    rc, out = run(reg, "set-probe", "--lane", "L-B", "--agent", "claude", "--probe", json.dumps(good), capsys=capsys)
    assert rc == 3 and "only the owner" in out.err
    for bad, why in [({"checks": [{"kind": "ping"}]}, "probe kind"),
                     ({"checks": [{"kind": "pid_cmdline"}]}, "needs ['match']"),
                     ({"receipt_globs": []}, "at least one check"),
                     ({"checks": [{"kind": "pid_cmdline", "match": "x"}], "status_ok": " "}, "status_ok"),
                     ("not json", "not JSON"),
                     (f"@{tmp_path / 'absent.json'}", "No such file")]:  # 1.0.2: refused, not a traceback
        text = bad if isinstance(bad, str) else json.dumps(bad)
        rc, out = run(reg, "set-probe", "--lane", "L-A", "--agent", "claude", "--probe", text, capsys=capsys)
        assert rc == 2 and why in out.err, (bad, out.err)
    rev = read(reg)["revision"]
    f = tmp_path / "probe.json"
    f.write_text(json.dumps(good))
    rc, out = run(reg, "set-probe", "--lane", "L-A", "--agent", "claude", "--probe", f"@{f}", capsys=capsys)
    assert rc == 0 and "SET-PROBE L-A: 1 check(s); dry probe ok" in out.out
    d = read(reg)
    assert d["lanes"]["L-A"]["probe"] == good and d["revision"] == rev + 1
    rc, out = run(reg, "set-probe", "--lane", "L-A", "--agent", "claude", "--probe", "{}", capsys=capsys)
    assert rc == 0 and "probe cleared" in out.out and read(reg)["lanes"]["L-A"]["probe"] is None


def helper(lock: Path, while_pid: int, match: str, *extra: str) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, str(HERE / "lanes.py"), "hold-lock", "--lock", str(lock), "--while-pid",
                             str(while_pid), "--cmd-match", match, "--poll", "0.5", *extra],
                            stdout=subprocess.PIPE, text=True)


def wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.05)
    return False


def holder_pids(lock: Path) -> list[int]:
    return [h["pid"] for h in lanes.lock_holders(str(lock)) or []]


def test_hold_lock_also_match_outlives_the_queue(tmp_path):
    lock = tmp_path / "gpu9.lock"
    lock.touch()
    queue, job = subprocess.Popen(["sleep", "604.1101"]), subprocess.Popen(["sleep", "604.2202"])
    h = helper(lock, queue.pid, "sleep 604.1101", "--also-match", "sleep 604.2202")
    try:
        assert wait_for(lambda: holder_pids(lock) == [h.pid])
        queue.kill()
        queue.wait()
        time.sleep(1.5)  # three polls with the queue gone: the job still holds the card
        assert h.poll() is None and holder_pids(lock) == [h.pid]
        job.kill()
        job.wait()
        out, _ = h.communicate(timeout=15)
        assert h.returncode == 0 and "no process matches ['sleep 604.2202']" in out
        assert holder_pids(lock) == []
    finally:
        for p in (queue, job, h):
            p.kill()


def test_hold_lock_join_helper_adopts_without_a_gap(tmp_path):
    lock, ready = tmp_path / "gpu9.lock", tmp_path / "ready"
    lock.touch()
    old_q, job, new_q = (subprocess.Popen(["sleep", f"605.{n}"]) for n in ("1101", "2202", "3303"))
    h1 = helper(lock, old_q.pid, "sleep 605.1101", "--also-match", "sleep 605.2202")
    h2 = None
    try:
        assert wait_for(lambda: holder_pids(lock) == [h1.pid])
        # the old queue still lives: never joined, refused after --wait
        rc = subprocess.run([sys.executable, str(HERE / "lanes.py"), "hold-lock", "--lock", str(lock), "--while-pid",
                             str(new_q.pid), "--cmd-match", "sleep 605.3303", "--join-helper", "--wait", "2"],
                            capture_output=True, text=True, timeout=30)
        assert rc.returncode == 5 and "JOIN" not in rc.stdout
        old_q.kill()
        old_q.wait()  # h1 is now orphaned: it holds only for the job
        h2 = helper(lock, new_q.pid, "sleep 605.3303", "--also-match", "sleep 605.2202", "--join-helper",
                    "--wait", "2", "--ready-file", str(ready))
        assert wait_for(lambda: ready.exists() and ready.read_text().strip())
        assert ready.read_text().strip() == f"{h2.pid} joined {h1.pid}"
        time.sleep(2.5)  # past --wait: a joined helper keeps waiting, it is not refused
        assert h2.poll() is None and holder_pids(lock) == [h1.pid]
        job.kill()
        job.wait()  # h1 releases; h2 takes the lock
        assert wait_for(lambda: holder_pids(lock) == [h2.pid])
        out1, _ = h1.communicate(timeout=15)
        assert h1.returncode == 0 and "RELEASE" in out1
        new_q.kill()
        new_q.wait()
        out2, _ = h2.communicate(timeout=15)
        assert h2.returncode == 0 and "JOIN" in out2 and "HOLD" in out2 and "RELEASE" in out2
        assert holder_pids(lock) == []
    finally:
        for p in (old_q, job, new_q, h1, h2):
            if p is not None:
                p.kill()


def test_hold_lock_join_gives_up_when_its_queue_dies(tmp_path):
    lock = tmp_path / "gpu9.lock"
    lock.touch()
    old_q, job, new_q = (subprocess.Popen(["sleep", f"606.{n}"]) for n in ("1101", "2202", "3303"))
    h1 = helper(lock, old_q.pid, "sleep 606.1101", "--also-match", "sleep 606.2202")
    try:
        assert wait_for(lambda: holder_pids(lock) == [h1.pid])
        old_q.kill()
        old_q.wait()
        h2 = helper(lock, new_q.pid, "sleep 606.3303", "--join-helper")
        time.sleep(1.5)
        new_q.kill()
        new_q.wait()
        out2, _ = h2.communicate(timeout=15)
        assert h2.returncode == 0 and "GIVE UP" in out2 and "HOLD" not in out2
        assert holder_pids(lock) == [h1.pid]  # the job still holds the card through the old helper
    finally:
        for p in (old_q, job, new_q, h1):
            p.kill()


# ------------------------------------------------------------------------------------------------ 1.1.0
SMI = [{"index": i, "uuid": f"GPU-test-{i}", "name": "P40" if i in (1, 2, 3, 5) else "RTX", "used_mib": 10}
       for i in range(7)]
LANES_PY = str(HERE / "lanes.py")
REMOTE_RUNTIME = Path("/home/cereal/local-hermes/remote_runtime.py")


def policy_seed(tmp_path: Path) -> Path:
    base = {"scope": "s", "status": "idle", "transferable": True, "transferable_reason": "test"}
    lanes_ = [
        {"lane": "L-TRAIN", "owner": "claude", "lock_paths": [], "cards": [1, 2], "flex_cards": [3], **base},
        {"lane": "L-EXP", "owner": "claude", "lock_paths": [], "cards": [5], "flex_cards": [3], **base},
        {"lane": "L-SERVE", "owner": "claude", "lock_paths": [], "cards": [0, 6], "ask_first": True, **base},
        {"lane": "L-CODEX", "owner": "codex", "lock_paths": [], **base},
    ]
    p = tmp_path / "seed.json"
    p.write_text(json.dumps({"created_by": "test", "lanes": lanes_}))
    return p


@pytest.fixture
def preg(tmp_path, capsys):
    r = tmp_path / "LANES.json"
    rc, _ = run(r, "init", "--seed", str(policy_seed(tmp_path)), capsys=capsys)
    assert rc == 0
    return r


def fake_omni_lab(tmp_path: Path) -> tuple[Path, str]:
    calls = tmp_path / "omni_calls.jsonl"
    script = tmp_path / "fake_omni_lab.py"
    script.write_text("import json, sys\n"
                      f"open({str(calls)!r}, 'a').write(json.dumps(sys.argv[1:]) + '\\n')\n"
                      "print('4242' if sys.argv[2] == 'start' else 'run 4242: 0 metric rows')\n")
    return calls, json.dumps([sys.executable, str(script)])


def claim_env(tmp_path: Path, **over) -> dict:
    _calls, cmd = fake_omni_lab(tmp_path)
    env = dict(os.environ, OC_LANES_SMI_JSON=json.dumps(over.pop("smi", SMI)), OC_LANES_MEMAVAIL_GIB="200",
               OC_LANES_OMNI_LAB=cmd, OC_LANES_OMNI_LAB_ENV=str(tmp_path / "no-env"))
    env.update(over)
    return env


def claim_run(reg: Path, tmp_path: Path, *argv, env=None, timeout=60):
    return subprocess.run([sys.executable, LANES_PY, "--file", str(reg), "claim-run", *argv],
                          capture_output=True, text=True, env=env or claim_env(tmp_path), timeout=timeout)


def test_register_adds_and_rescopes_with_history(preg, tmp_path, capsys):
    rc, out = run(preg, "register", "--agent", "claude", "--reason", "new lane", "--lane-json", json.dumps(
        {"lane": "L-NEW", "owner": "claude", "scope": "n", "lock_paths": [], "status": "idle", "transferable": True,
         "transferable_reason": "t", "cards": [4]}), capsys=capsys)
    assert rc == 0 and "ADDED" in out.out
    d = read(preg)
    assert d["lanes"]["L-NEW"]["registered_utc"] and d["lanes"]["L-NEW"]["history"][0]["reason"] == "new lane"
    rc, out = run(preg, "register", "--agent", "claude", "--reason", "rescope", "--lane-json",
                  json.dumps({"lane": "L-TRAIN", "cards": [1], "scope": "s2"}), capsys=capsys)
    assert rc == 0 and "RESCOPED" in out.out
    t = read(preg)["lanes"]["L-TRAIN"]
    assert t["cards"] == [1] and t["scope"] == "s2" and t["history"][-1]["previous"] == {"cards": [1, 2], "scope": "s"}
    rc, out = run(preg, "register", "--agent", "claude", "--reason", "x", "--lane-json",
                  json.dumps({"lane": "L-TRAIN", "cards": [1], "scope": "s2"}), capsys=capsys)
    assert rc == 0 and "no change" in out.out
    for obj, code, why in (({"lane": "L-TRAIN", "heartbeat_utc": "x"}, 2, "cannot set"),
                           ({"lane": "L-TRAIN", "cards": [1, 1]}, 2, "distinct"),
                           ({"lane": "L-TRAIN", "cards": [3]}, 2, "both"),
                           ({"lane": "L-TRAIN", "ask_first": "yes"}, 2, "ask_first"),
                           ({"lane": "L-CODEX", "scope": "mine"}, 3, "owned by codex")):
        rc, out = run(preg, "register", "--agent", "claude", "--reason", "x", "--lane-json", json.dumps(obj),
                      capsys=capsys)
        assert rc == code and why in out.err, (obj, out.err)
    rc, out = run(preg, "register", "--agent", "claude", "--reason", "x", "--cyril", "take it",
                  "--lane-json", json.dumps({"lane": "L-CODEX", "scope": "mine"}), capsys=capsys)
    assert rc == 0 and read(preg)["lanes"]["L-CODEX"]["history"][-1]["cyril"] == "take it"


def test_set_rule_keeps_history(preg, capsys):
    rc, _ = run(preg, "set-rule", "--agent", "claude", "--key", "gpu", "--text", "rule 1", "--reason", "r", capsys=capsys)
    assert rc == 0
    rc, _ = run(preg, "set-rule", "--agent", "claude", "--key", "gpu", "--text", "rule 2", "--reason", "r2", capsys=capsys)
    d = read(preg)
    assert rc == 0 and d["rules"]["gpu"] == "rule 2" and d["rules_history"][-1]["previous"] == "rule 1"


def test_card_map_and_verdicts(preg, capsys):
    reg_ = read(preg)
    m = lanes.card_map(reg_)
    assert [h["lane"] for h in m[3]] == ["L-TRAIN", "L-EXP"] and m[0][0]["ask_first"] and m[1][0]["role"] == "primary"
    assert lanes.card_verdicts(reg_, "L-TRAIN", [1, 2, 3], None, None) == ["GPU1 own", "GPU2 own", "GPU3 flex"]
    assert lanes.card_verdicts(reg_, "L-SERVE", [0], None, None) == ["GPU0 own"]
    for lane_, cards, borrow, cyril, code in (("L-TRAIN", [5], None, None, 2), ("L-EXP", [0], "hard job", None, 4),
                                              ("L-EXP", [9], "x", None, 2)):
        with pytest.raises(lanes.Refused) as e:
            lanes.card_verdicts(reg_, lane_, cards, borrow, cyril)
        assert e.value.code == code
    assert "borrowed from L-EXP" in lanes.card_verdicts(reg_, "L-TRAIN", [5], "hard job", None)[0]
    assert "Cyril" in lanes.card_verdicts(reg_, "L-EXP", [0], None, "yes pull GPU0")[0]
    rc, out = run(preg, "cards", capsys=capsys)
    assert rc == 0 and "GPU3: L-TRAIN (flex); L-EXP (flex)" in out.out and "GPU0: L-SERVE (primary, ask-first)" in out.out


def test_release_stale_only_for_a_dead_pid(preg, tmp_path, capsys):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    claims = tmp_path / "p40.claims"
    claims.write_text(f"2026-10-15T02:41:34Z CLAIM GPU1,GPU2 owner=claude lane=toolcall job=ext pid={dead.pid} log=/x\n"
                      f"2026-10-15T02:41:35Z CLAIM GPU3 owner=claude lane=live job=j pid={os.getpid()}\n"
                      "2026-10-15T02:41:36Z CLAIM GPU5 owner=claude lane=nopid job=k (docker)\n")
    for cards, lane_, job, code in (("GPU3", "live", "j", 5), ("GPU5", "nopid", "k", 5), ("GPU1", "toolcall", "ext", 2)):
        rc, out = run(preg, "release-stale", "--agent", "claude", "--cards", cards, "--claim-lane", lane_, "--job", job,
                      "--reason", "test", capsys=capsys)
        assert rc == code, out.err
    rc, out = run(preg, "release-stale", "--agent", "claude", "--cards", "GPU1,GPU2", "--claim-lane", "toolcall",
                  "--job", "ext", "--reason", "driver gone, it never wrote its own RELEASE line", capsys=capsys)
    assert rc == 0 and "released (stale" in out.out
    assert {c["lane"] for c in lanes.claims_open(str(claims))} == {"live", "nopid"}


def test_claim_run_runs_and_releases(preg, tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import fcntl, json, os, sys\n"
        "fds = [os.readlink(f'/proc/self/fd/{f}') for f in os.listdir('/proc/self/fd')\n"
        "       if os.path.exists(f'/proc/self/fd/{f}')]\n"
        f"lock = open({str(tmp_path / 'gpu5.lock')!r}, 'a')\n"
        "try:\n    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB); held = False\n"
        "except BlockingIOError:\n    held = True\n"
        "print(json.dumps({'cvd': os.environ.get('CUDA_VISIBLE_DEVICES'), 'order': os.environ.get('CUDA_DEVICE_ORDER'),"
        " 'held': held, 'inherited': [f for f in fds if f.endswith('.lock')]}))\n"
        "sys.exit(3)\n")
    log = tmp_path / "job.log"
    ready, exit_ = tmp_path / "ready.json", tmp_path / "exit.json"
    r = claim_run(preg, tmp_path, "--lane", "L-EXP", "--agent", "claude", "--gpus", "5,3", "--job", "exp-1",
                  "--log", str(log), "--ready-file", str(ready), "--exit-file", str(exit_), "--omni-lab-kind", "probe",
                  "--", sys.executable, str(probe))
    assert r.returncode == 3, r.stderr
    job = json.loads(log.read_text().strip().splitlines()[-1])
    assert job == {"cvd": "GPU-test-3,GPU-test-5", "order": "PCI_BUS_ID", "held": True, "inherited": []}
    lines = (tmp_path / "p40.claims").read_text().splitlines()
    assert len(lines) == 2 and " CLAIM GPU3,GPU5 owner=claude lane=L-EXP job=exp-1 pid=" in lines[0]
    assert " RELEASE GPU3,GPU5 owner=claude lane=L-EXP job=exp-1 released (rc 3)" in lines[1]
    assert lanes.claims_open(str(tmp_path / "p40.claims")) == []
    rd, ex = json.loads(ready.read_text()), json.loads(exit_.read_text())
    assert rd["cards"] == [3, 5] and rd["omni_lab_run_id"] == 4242 and ex["rc"] == 3 and ex["status"] == "fail"
    calls = [json.loads(x) for x in (tmp_path / "omni_calls.jsonl").read_text().splitlines()]
    assert calls[0][:4] == ["log-run", "start", "--kind", "probe"] and "--lane" in calls[0] and "3,5" in calls[0]
    assert calls[1][:3] == ["log-run", "end", "4242"] and calls[1][calls[1].index("--status") + 1] == "fail"
    lane_ = read(preg)["lanes"]["L-EXP"]
    assert lane_["heartbeat_source"].startswith("claim-run") and "ended fail rc 3" in lane_["detail"]
    if REMOTE_RUNTIME.is_file():  # the controller's own parser must read the same lines the same way
        import importlib.util
        spec = importlib.util.spec_from_file_location("rr_claims_probe", REMOTE_RUNTIME)
        rr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rr)
        assert rr.active_p40_claims(lines[0] + "\n") == {"3": [lines[0]], "5": [lines[0]]}
        assert rr.active_p40_claims("\n".join(lines) + "\n") == {}


def test_claim_run_forwards_term(preg, tmp_path):
    ready, exit_ = tmp_path / "ready.json", tmp_path / "exit.json"
    proc = subprocess.Popen([sys.executable, LANES_PY, "--file", str(preg), "claim-run", "--lane", "L-TRAIN",
                             "--agent", "claude", "--gpus", "1,2", "--job", "train-1", "--log", str(tmp_path / "t.log"),
                             "--ready-file", str(ready), "--exit-file", str(exit_), "--no-omni-lab", "--",
                             sys.executable, "-c", "import time; time.sleep(60)"], env=claim_env(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for _ in range(100):
        if ready.exists():
            break
        time.sleep(0.1)
    assert ready.exists()
    job_pid = json.loads(ready.read_text())["pid"]
    assert lanes.pid_alive(job_pid)
    proc.send_signal(15)
    out, err = proc.communicate(timeout=30)
    assert proc.returncode == 143, err
    assert json.loads(exit_.read_text())["status"] == "stopped"
    last = (tmp_path / "p40.claims").read_text().splitlines()[-1]
    assert "RELEASE GPU1,GPU2" in last and "signal 15 forwarded" in last
    for _ in range(50):
        if not lanes.pid_alive(job_pid):
            break
        time.sleep(0.1)
    assert not lanes.pid_alive(job_pid)
    assert not (tmp_path / "omni_calls.jsonl").exists()


def test_claim_run_refusals_write_nothing(preg, tmp_path):
    base = ["--agent", "claude", "--log", str(tmp_path / "x.log"), "--no-omni-lab"]
    job = ["--", sys.executable, "-c", "pass"]
    busy = [dict(c, used_mib=5000) if c["index"] == 5 else c for c in SMI]
    cases = [
        (["--lane", "L-EXP", "--gpus", "5", "--job", "j"], claim_env(tmp_path, smi=busy), 5, "not idle"),
        (["--lane", "L-EXP", "--gpus", "5", "--job", "j"], claim_env(tmp_path, OC_LANES_MEMAVAIL_GIB="12"), 5,
         "MemAvailable"),
        (["--lane", "L-EXP", "--gpus", "0", "--job", "j", "--borrow", "need it"], None, 4, "serving card"),
        (["--lane", "L-EXP", "--gpus", "1", "--job", "j"], None, 2, "--borrow"),
        (["--lane", "L-EXP", "--gpus", "5", "--job", "done"], None, 2, "--job"),
        (["--lane", "L-EXP", "--gpus", "1", "--job", "j", "--borrow", "old job finished"], None, 2, "release word"),
        (["--lane", "L-CODEX", "--gpus", "5", "--job", "j"], None, 3, "owned by codex"),
    ]
    for argv, env, code, why in cases:
        r = claim_run(preg, tmp_path, *argv, *base, *job, env=env)
        assert r.returncode == code and why in r.stderr, (argv, r.returncode, r.stderr)
    assert not (tmp_path / "p40.claims").exists()
    with open(tmp_path / "gpu5.lock", "a") as held:  # a card lock held by someone else
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        r = claim_run(preg, tmp_path, "--lane", "L-EXP", "--gpus", "3,5", "--job", "j", *base, *job)
        assert r.returncode == 5 and "gpu5.lock is held" in r.stderr
    (tmp_path / "p40.claims").write_text(
        f"2026-10-21T00:00:00Z CLAIM GPU3 owner=claude lane=other job=o pid={os.getpid()}\n")
    r = claim_run(preg, tmp_path, "--lane", "L-EXP", "--gpus", "3", "--job", "j", *base, *job)
    assert r.returncode == 5 and "open claim on GPU3" in r.stderr
    assert len((tmp_path / "p40.claims").read_text().splitlines()) == 1
    r = claim_run(preg, tmp_path, "--lane", "L-EXP", "--gpus", "5", "--job", "j", "--dry-run", *base)
    assert r.returncode == 0 and "CLAIM-RUN DRY OK GPU5" in r.stdout
    assert len((tmp_path / "p40.claims").read_text().splitlines()) == 1


def test_claim_run_pulls_a_serving_card_only_with_cyril(preg, tmp_path):
    r = claim_run(preg, tmp_path, "--lane", "L-EXP", "--gpus", "6", "--job", "big", "--agent", "claude",
                  "--log", str(tmp_path / "b.log"), "--no-omni-lab", "--cyril", "yes take GPU6 for this",
                  "--", sys.executable, "-c", "pass")
    assert r.returncode == 0, r.stderr
    first = (tmp_path / "p40.claims").read_text().splitlines()[0]
    assert "CLAIM GPU6 owner=claude lane=L-EXP job=big" in first and "Cyril approved" in first
