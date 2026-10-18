"""Tests for d2clean_recache.py (lane d2clean, 2026-10-15).  CPU only; no card, no nvidia-smi, no real teacher.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2clean_recache.py

Pure parts: the GO file (cards now and cards_after a declared stop), which v1 signals a card class obeys (swap_full is
recorded, never obeyed; the P40 review signal only on a P40), the file pick, and the p40.claims lines as d1_seal_lib
reads them (a claim line must read as HELD -- the x3 lesson -- and the release line as released).  The runner's yield:
a fake build that traps SIGTERM is stopped when a v1 signal appears, the claim line is released and both flocks are
free afterwards; refused cards never construct a runner.
"""

from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_seal_lib as SL  # noqa: E402
import d2clean_recache as RC  # noqa: E402

SHA = "a" * 64


def test_go_file():
    assert RC.read_go(None)["ok"] is False
    assert RC.read_go("nope\nmanifest_sha256=" + SHA)["ok"] is False
    assert RC.read_go("d2clean-recache-go\ncards=3")["ok"] is False  # no manifest sha
    go = RC.read_go(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards=3\ncards_after=2026-10-15T21:40:00Z:1,2\n",
                    now="2026-10-15T03:00:00Z")
    assert go["ok"] and go["cards"] == [3] and go["pending"] == [{"after": "2026-10-15T21:40:00Z", "cards": [1, 2]}]
    go = RC.read_go(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards=3\ncards_after=2026-10-15T21:40:00Z:1,2\n",
                    now="2026-10-15T21:40:00Z")
    assert go["cards"] == [1, 2, 3] and go["pending"] == []
    assert RC.read_go(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards_after=tomorrow:1\n")["ok"] is False
    assert RC.read_go(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards=x\n")["ok"] is False


def test_pressure_by_card_class():
    sig = {"swap_full": "swap 99 %", "gpu3_wait": "d1_seal waiting: step=B1-cards state=waiting", "_facts": {}}
    assert RC.pressure(sig, "p40") == ["gpu3_wait: d1_seal waiting: step=B1-cards state=waiting"]
    assert RC.pressure(sig, "rtx3090") == []
    sig = {"gpu6_wait_ram": "gpu6 queue: MemAvailable 80 GiB < 93; waiting", "swap_full": "x"}
    assert RC.pressure(sig, "rtx3090") == ["gpu6_wait_ram: gpu6 queue: MemAvailable 80 GiB < 93; waiting"]
    for name in RC.OBEYED:
        assert RC.pressure({name: "r"}, "p40") == [f"{name}: r"]


def test_pick_file():
    files = [{"file": f"f{i}", "sha256": str(i)} for i in range(4)]
    assert [f["file"] for f in RC.pick_file(files, {"0": True, "2": True}, {"3"})] == ["f1"]


def test_claim_lines_read_by_d1_seal_lib():
    claim = RC.claim_line(3, 4242, "/x/gpu3.log")
    held, line = SL.claim_state([claim], 3)
    assert held and "lane=d2clean" in line
    assert SL.claim_state([claim, RC.release_line(3)], 3) == (False, None)
    other = "2026-10-15T00:00:00Z CLAIM GPU1,GPU2 owner=claude lane=toolcall job=t pid=1 log=/l (release=this driver)"
    # another lane's live claim on the same card keeps it held after this lane releases
    held, line = SL.claim_state([claim, "2026-10-15T00:00:01Z CLAIM GPU3 owner=claude lane=toolcall job=t pid=1 log=/l",
                                 RC.release_line(3)], 3)
    assert held and "lane=toolcall" in line
    assert SL.claim_state([other], 3) == (False, None)


def test_refused_cards_never_build_a_runner(capsys):
    for card in (5, 6, 7):
        assert RC.main(["probe", "--card", str(card)]) == 3
    assert "REFUSED card 6" in capsys.readouterr().out


@pytest.fixture()
def world(tmp_path, monkeypatch):
    q = tmp_path / "queues"
    q.mkdir()
    monkeypatch.setattr(RC, "Q", q)
    monkeypatch.setattr(RC, "STATUS", q / "d2clean.status")
    monkeypatch.setattr(RC, "CLAIMS", q / "p40.claims")
    monkeypatch.setattr(RC, "GO_FILE", q / "d2clean.go")
    monkeypatch.setattr(RC, "OUT_ROOT", tmp_path / "cache")
    monkeypatch.setattr(RC, "POLL_S", 1)
    (q / "d2clean.go").write_text(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards=3\n")

    class A:
        card = 3
        clean_dir = str(tmp_path / "clean")

    runner = RC.Runner(A())
    monkeypatch.setattr(runner, "heartbeat", lambda detail: None)
    return runner, q, tmp_path


def test_gates_report_v1_pressure_and_ram(world, monkeypatch):
    runner, q, _ = world
    monkeypatch.setattr(runner, "signals", lambda: {"swap_full": "x", "_facts": {}})
    monkeypatch.setattr(RC, "mem_available_gib", lambda: 80.0)
    assert runner.gates(starting=True)[0] == []
    monkeypatch.setattr(RC, "mem_available_gib", lambda: 50.0)
    assert runner.gates(starting=True)[0] == [f"MemAvailable 50.0 GiB < {RC.START_GIB}"]
    assert runner.gates(starting=False)[0] == []  # above the floor while building
    monkeypatch.setattr(runner, "signals", lambda: {"d1_seal_wait_mem": "d1_seal.status: ... MemAvailable_40GiB_lt_70", "_facts": {}})
    monkeypatch.setattr(RC, "mem_available_gib", lambda: 80.0)
    assert runner.gates(starting=False)[0][0].startswith("d1_seal_wait_mem")

    def boom():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(runner, "signals", boom)
    assert runner.gates(starting=True)[0][0].startswith("v1 pressure reader failed")
    (q / "d2clean.go").write_text(f"d2clean-recache-go\nmanifest_sha256={SHA}\ncards=1\n")
    assert runner.gates(starting=True)[0] == ["go: card 3 not in cards=[1]"]


def test_build_yields_on_v1_pressure_and_frees_everything(world, monkeypatch):
    runner, q, tmp = world
    marker = tmp / "got_sigterm"
    fake = tmp / "fake_build.py"
    fake.write_text("import signal, sys, time\n"
                    f"signal.signal(signal.SIGTERM, lambda *_: (open({str(marker)!r}, 'w').write('143'), sys.exit(143)))\n"
                    "time.sleep(120)\n")
    monkeypatch.setattr(runner, "build_command", lambda f: ([sys.executable, str(fake)], dict(os.environ)))
    calls = {"n": 0}

    def gates(starting):
        calls["n"] += 1
        return (["gpu3_wait: d1_seal waiting"] if calls["n"] >= 2 else []), {}

    monkeypatch.setattr(runner, "gates", gates)
    f = {"file": "text-clean-00000.safetensors", "sha256": "b" * 64}
    assert runner.take_file_lock(f) and runner.take_card_lock()
    with open(RC.CLAIMS, "a") as handle:
        handle.write(RC.claim_line(3, os.getpid(), "x") + "\n")
    runner.claimed = True
    started = time.time()
    assert runner.run_file(f) == "yielded"
    assert marker.read_text() == "143" and time.time() - started < 100
    runner.release_all("yielded")
    assert SL.claim_state(RC.CLAIMS.read_text().splitlines(), 3) == (False, None)
    for path in (q / "gpu3.lock", q / "d2clean" / "claims" / f"{f['file']}.lock"):
        fd = os.open(str(path), os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # free: would raise BlockingIOError if still held
        os.close(fd)
    assert "YIELD text-clean-00000.safetensors: gpu3_wait" in RC.STATUS.read_text()


def test_a_sealed_build_is_verified(world, monkeypatch):
    runner, q, tmp = world
    f = {"file": "text-clean-00001.safetensors", "sha256": "c" * 64}
    cache = RC.OUT_ROOT / RC.TEACHER / f["sha256"]
    cache.mkdir(parents=True)
    (cache / "manifest.json").write_text(json.dumps({"all_finite": True, "all_structural_ok": True,
                                                     "teacher": {"equal_before_and_after_every_session": True}}))
    monkeypatch.setattr(runner, "build_command", lambda f: ([sys.executable, "-c", "pass"], dict(os.environ)))
    monkeypatch.setattr(runner, "gates", lambda starting: ([], {}))
    monkeypatch.setattr(runner, "verify_cache", lambda f, env: (0, "VERIFY OK"))
    assert runner.run_file(f) == "sealed"
    monkeypatch.setattr(runner, "verify_cache", lambda f, env: (1, "VERIFY FAILED"))
    assert runner.run_file(f) == "failed"
