"""Tests for ``d1_hardened_entry.py`` + ``d1_hardened_launch.sh`` -- CPU, the real d1_trunk_recovery on the tiny
synthetic fixture of test_d1_trunk_recovery (no stand-in trainer).

  test_entry_success_status_and_locks       exit 0; runtime-status.json: succeeded / completed / exit 0 / attempt /
                                            progress "done" / memory; the card locks are free afterwards; a second
                                            attempt keeps the first in its history
  test_entry_refusal_exit_3                 a functional refusal -> exit 3, state refused, the message recorded
  test_entry_lock_busy_exit_4               a card lock held by another process -> waits, then exit 4 (lock_timeout)
  test_entry_waits_for_foreign_gpu_processes  1.1.0: a card with another compute process (fake nvidia-smi) ->
                                            exit 4 gpu_busy, nothing started; a free card -> the run proceeds
  test_finalize_after_sigkill               ExecStopPost mode: a status left "running" becomes killed with systemd's
                                            SERVICE_RESULT; a completed status keeps its own state
  test_fault_ssh_session_killed             (opt-in: D1_FAULT_INJECTION=1) launched through d1_hardened_launch.sh
                                            (systemd-run --user) from a throwaway session; the session is SIGKILLed
                                            mid-training (the SSH drop); the trainer is alive and heart-beating for
                                            D1_FAULT_SECONDS (default 300) after, then finishes: succeeded, delta
  test_fault_sigkill_restart_resumes        (opt-in) the trainer is SIGKILLed during L01 (an OOM kill): systemd
                                            restarts it, it resumes from the L00 boundary (no L00 recompute), and
                                            its delta bytes == an uninterrupted in-process run's

Server (CPU only):
    cd $OC/distill/tools && EF=$OC/import/rq27b/tree_ef1ffb9 && CHIMERA_SRC=$EF/src CHIMERA_OMNI_SRC=$EF \\
        PYTHONPATH=$PWD:$EF/src CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 /home/cereal/model-surgery-lab/venv/bin/python \\
        -m pytest -o addopts="" -p no:cacheprovider test_d1_hardened_entry.py -q [-s]
    fault injection: D1_FAULT_INJECTION=1 [D1_FAULT_SECONDS=300] [D1_FAULT_EPOCHS=N] (takes 10-20 min; run detached)
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_hardened_entry as entry  # noqa: E402
import d1_trunk_recovery as d1  # noqa: E402
from test_d1_trunk_recovery import argv, build_fixture  # noqa: E402

FAULT = os.environ.get("D1_FAULT_INJECTION") == "1"
FAULT_SECONDS = float(os.environ.get("D1_FAULT_SECONDS", "300"))


def status_of(path: Path) -> dict:
    for _ in range(50):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            time.sleep(0.1)
    raise AssertionError(f"{path} unreadable")


def lock_is_free(path: Path) -> bool:
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


@pytest.fixture(scope="module")
def fx(tmp_path_factory) -> dict[str, Path]:
    return build_fixture(tmp_path_factory.mktemp("entryfx"))


def test_entry_success_status_and_locks(fx, tmp_path: Path) -> None:
    st, locks = tmp_path / "runtime-status.json", [tmp_path / "gpu0.lock", tmp_path / "gpu4.lock"]
    lock_args = [a for l in locks for a in ("--lock", str(l))]
    code = entry.main(["--status", str(st), *lock_args, "--heartbeat-s", "0.2", "--", *argv(fx, tmp_path / "out")])
    doc = status_of(st)
    assert code == 0 and doc["state"] == "succeeded" and doc["completed"] is True and doc["exit_code"] == 0
    assert doc["attempt"] == 1 and doc["history"] == [] and doc["progress"]["phase"] == "done"
    assert doc["memory"]["rss"] > 0 and doc["locks"] == [str(l) for l in locks] and doc["finished_at"]
    assert all(lock_is_free(l) for l in locks)
    assert (tmp_path / "out" / "delta" / "D1_delta.safetensors").is_file()
    code = entry.main(["--status", str(st), *lock_args, "--", *argv(fx, tmp_path / "out", "--resume")])
    doc = status_of(st)
    assert code == 0 and doc["attempt"] == 2 and doc["history"][0]["execution_id"] != doc["execution_id"]
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_entry_refusal_exit_3(fx, tmp_path: Path) -> None:
    st = tmp_path / "s.json"
    code = entry.main(["--status", str(st), "--lock", str(tmp_path / "g.lock"), "--",
                       *argv(fx, tmp_path / "out", "--held-rows", "6")])
    doc = status_of(st)
    assert code == entry.EXIT_REFUSED and doc["state"] == "refused" and "leaves no fit rows" in doc["error"]
    assert lock_is_free(tmp_path / "g.lock")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_entry_lock_busy_exit_4(fx, tmp_path: Path) -> None:
    lock = tmp_path / "gpu0.lock"
    holder = subprocess.Popen([sys.executable, "-c", (
        "import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
        "print('held', flush=True); time.sleep(60)"), str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        st = tmp_path / "s.json"
        t0 = time.time()
        code = entry.main(["--status", str(st), "--lock", str(lock), "--lock-wait-s", "1.5", "--lock-poll-s", "0.2",
                           "--", *argv(fx, tmp_path / "out")])
        doc = status_of(st)
        assert code == entry.EXIT_LOCK_TIMEOUT and doc["state"] == "lock_timeout" and time.time() - t0 >= 1.4
        assert not (tmp_path / "out").exists(), "the trainer must not start without its card locks"
    finally:
        holder.kill()
        holder.wait()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_entry_waits_for_foreign_gpu_processes(fx, tmp_path: Path, monkeypatch) -> None:
    """--gpu-uuid: a card carrying another compute process (a fake nvidia-smi on PATH reports one) keeps the
    trainer from starting -> exit 4 gpu_busy; once the card is free the run proceeds."""
    fake = tmp_path / "bin"
    fake.mkdir()
    report = tmp_path / "apps.csv"
    (fake / "nvidia-smi").write_text(f"#!/bin/sh\ncat {report}\n")
    os.chmod(fake / "nvidia-smi", 0o755)
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    report.write_text("424242, GPU-aaaa\n")
    st = tmp_path / "s.json"
    code = entry.main(["--status", str(st), "--lock", str(tmp_path / "g.lock"), "--gpu-uuid", "GPU-aaaa",
                       "--lock-wait-s", "1", "--lock-poll-s", "0.2", "--", *argv(fx, tmp_path / "out")])
    doc = status_of(st)
    assert code == entry.EXIT_LOCK_TIMEOUT and doc["state"] == "gpu_busy" and "424242@GPU-aaaa" in doc["error"]
    assert not (tmp_path / "out").exists() and lock_is_free(tmp_path / "g.lock")
    report.write_text("424242, GPU-bbbb\n")              # another card is busy: ours is free
    code = entry.main(["--status", str(st), "--lock", str(tmp_path / "g.lock"), "--gpu-uuid", "GPU-aaaa",
                       "--lock-wait-s", "5", "--lock-poll-s", "0.2", "--", *argv(fx, tmp_path / "out")])
    assert code == 0 and status_of(st)["state"] == "succeeded"
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_finalize_after_sigkill(tmp_path: Path, monkeypatch) -> None:
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"schema": entry.SCHEMA, "state": "running", "completed": False, "attempt": 1, "history": []}))
    monkeypatch.setenv("SERVICE_RESULT", "oom-kill")
    monkeypatch.setenv("EXIT_CODE", "killed")
    monkeypatch.setenv("EXIT_STATUS", "KILL")
    assert entry.main(["--finalize", "--status", str(st)]) == 0
    doc = status_of(st)
    assert doc["completed"] is True and doc["state"] == "killed" and doc["exit_code"] == "KILL"
    assert doc["systemd"]["service_result"] == "oom-kill" and "without writing its exit state" in doc["error"]
    st.write_text(json.dumps({"state": "refused", "completed": True, "exit_code": 3}))
    monkeypatch.setenv("SERVICE_RESULT", "exit-code")
    monkeypatch.setenv("EXIT_CODE", "exited")
    monkeypatch.setenv("EXIT_STATUS", "3")
    entry.main(["--finalize", "--status", str(st)])
    doc = status_of(st)
    assert doc["state"] == "refused" and doc["exit_code"] == 3 and doc["systemd"]["exit_status"] == "3"


# ------------------------------------------------------------------ fault injection through the production launcher
def launch(unit: str, run_dir: Path, lock_dir: Path, trainer_argv: list[str], restart_sec: int = 300,
           session: bool = True) -> subprocess.Popen | None:
    """d1_hardened_launch.sh from a throwaway session (setsid bash ... ; sleep): the stand-in SSH session."""
    ef = os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9")
    cmd = ["bash", str(HERE / "d1_hardened_launch.sh"), "--unit", unit, "--run-dir", str(run_dir), "--memory-max", "16G",
           "--restart-sec", str(restart_sec), "--cuda-visible", "", "--lock", str(lock_dir / "gpu0.lock"),
           "--lock", str(lock_dir / "gpu4.lock"), "--threads", os.environ.get("OMP_NUM_THREADS", "4"),
           "--python", sys.executable, "--tools", str(HERE), "--chimera", ef, "--heartbeat-s", "0.5", "--", *trainer_argv]
    if not session:
        subprocess.run(cmd, check=True)
        return None
    shell = " ".join(f"'{c}'" for c in cmd) + "; sleep 3600"
    return subprocess.Popen(["setsid", "bash", "-c", shell], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def unit_show(unit: str, *props: str) -> dict[str, str]:
    out = subprocess.run(["systemctl", "--user", "show", f"{unit}.service", *[f"-p{p}" for p in props]],
                         capture_output=True, text=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def wait_for(pred, timeout: float, what: str, every: float = 1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        got = pred()
        if got:
            return got
        time.sleep(every)
    raise AssertionError(f"timed out after {timeout:.0f}s waiting for {what}")


def cleanup(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", f"{unit}.service"], capture_output=True)
    subprocess.run(["systemctl", "--user", "reset-failed", f"{unit}.service"], capture_output=True)


def long_argv(fx, out: Path, epochs: str, *extra: str) -> list[str]:
    a = argv(fx, out, *extra)
    a[a.index("--epochs") + 1] = epochs
    a[a.index("--log-every") + 1] = "5000"
    return a


@pytest.mark.skipif(not FAULT, reason="fault injection is opt-in: D1_FAULT_INJECTION=1 (takes minutes)")
def test_fault_ssh_session_killed(fx, tmp_path: Path) -> None:
    unit = f"claude-d1h-test-{uuid.uuid4().hex[:8]}"
    run_dir, st = tmp_path / "run", tmp_path / "run" / "runtime-status.json"
    epochs = os.environ.get("D1_FAULT_EPOCHS", "60000")
    sess = launch(unit, run_dir, tmp_path, long_argv(fx, run_dir / "trainer", epochs))
    try:
        doc = wait_for(lambda: st.is_file() and (d := status_of(st)) and str(d.get("progress", {}).get("phase", "")).startswith("train")
                       and d, 180, "the trainer to reach a training step", every=0.5)
        pid = int(unit_show(unit, "MainPID")["MainPID"])
        assert pid == doc["pid"] and os.getsid(pid) != os.getsid(sess.pid), "the trainer must not live in the launching session"
        t_kill = time.time()
        os.killpg(sess.pid, signal.SIGKILL)               # the SSH session dies (every process in it)
        sess.wait(timeout=10)
        print(f"FAULT session {sess.pid} SIGKILLed at {time.strftime('%H:%M:%SZ', time.gmtime(t_kill))}; trainer pid {pid} "
              f"phase {doc['progress']}", flush=True)
        beats = set()
        while time.time() - t_kill < FAULT_SECONDS:
            d = status_of(st)
            os.kill(pid, 0)                                # raises if the trainer died
            assert d["state"] == "running" and d["pid"] == pid, d["state"]
            beats.add(d["heartbeat_at"])
            time.sleep(10)
        alive_for = time.time() - t_kill
        print(f"FAULT trainer alive {alive_for:.0f}s after the session kill; heartbeats seen {len(beats)}; "
              f"progress {status_of(st)['progress']}", flush=True)
        assert len(beats) >= max(2, int(FAULT_SECONDS // 60) - 1)
        done = wait_for(lambda: (d := status_of(st)).get("completed") and d, 3600, "the trainer to finish", every=5)
        assert done["state"] == "succeeded" and done["exit_code"] == 0 and done["attempt"] == 1, done
        assert (run_dir / "trainer" / "delta" / "D1_delta.safetensors").is_file()
        show = unit_show(unit, "Result", "ExecMainStatus")
        print(f"FAULT done: status {done['state']} exit {done['exit_code']} finished {done['finished_at']}; systemd {show}", flush=True)
        assert show.get("Result") == "success"
    finally:
        try:
            os.killpg(sess.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        cleanup(unit)


@pytest.mark.skipif(not FAULT, reason="fault injection is opt-in: D1_FAULT_INJECTION=1 (takes minutes)")
def test_fault_sigkill_restart_resumes(fx, tmp_path: Path, monkeypatch) -> None:
    epochs = os.environ.get("D1_FAULT_RESTART_EPOCHS", "8000")   # ~30 s a layer: the 0.5 s heartbeat sees L01 train
    monkeypatch.setenv("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))
    ref_out = tmp_path / "ref"
    d1.run(long_argv(fx, ref_out, epochs))
    ref_delta = json.loads((ref_out / "d1_receipt.json").read_text())["delta"]["file_sha256"]
    unit = f"claude-d1h-test-{uuid.uuid4().hex[:8]}"
    run_dir, st = tmp_path / "run", tmp_path / "run" / "runtime-status.json"
    bd = tmp_path / "boundary"
    launch(unit, run_dir, tmp_path, long_argv(fx, run_dir / "trainer", epochs, "--boundary-dir", str(bd), "--resume"),
           restart_sec=5, session=False)
    try:
        wait_for(lambda: (bd / "LATEST.json").is_file() and str(status_of(st).get("progress", {}).get("phase", "")).startswith("train")
                 and status_of(st)["progress"].get("layer") == 1, 600, "L01 training after the L00 boundary", every=0.5)
        pid = int(unit_show(unit, "MainPID")["MainPID"])
        os.kill(pid, signal.SIGKILL)                       # the OOM killer's signal
        print(f"FAULT SIGKILL pid {pid} during {status_of(st)['progress']}", flush=True)
        done = wait_for(lambda: (d := status_of(st)).get("completed") and d.get("attempt", 0) >= 2 and d, 1800,
                        "the restarted attempt to finish", every=2)
        assert done["state"] == "succeeded" and done["exit_code"] == 0, done
        first = done["history"][0]
        assert first["state"] == "killed" and first["systemd"]["exit_status"] == "KILL", first
        rec = json.loads((run_dir / "trainer" / "d1_receipt.json").read_text())
        assert rec["boundary"]["resumed_from"]["layer"] == 0 and rec["layers"][0]["from_boundary"]
        assert "rung" in rec["layers"][1]
        print(f"FAULT restart: attempt {done['attempt']}, first attempt {first['state']} ({first['systemd']}), resumed from "
              f"boundary L{rec['boundary']['resumed_from']['layer']:02d}; delta {rec['delta']['file_sha256'][:16]} vs "
              f"uninterrupted {ref_delta[:16]}", flush=True)
        assert rec["delta"]["file_sha256"] == ref_delta
    finally:
        cleanup(unit)
