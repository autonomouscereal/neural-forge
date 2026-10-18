"""Tests for d2refill_recache.py (lane refill, 2026-10-15).  CPU only; no card, no nvidia-smi, no real teacher.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2refill_recache.py

Pinned: the GO file (cards now / after a declared stop, no manifest line), the P40 micro-batch rule (never more tokens
per pass than d2clean's measured mb2 x 4,096), the pick (eligible + not sealed-and-verified, rank order; ineligible and
twice-failed corpora never), the claim / release lines as d1_seal_lib reads them (a claim reads HELD, never a release),
the build command (the frozen 1.0.5 tool, the sealed flags, the corpus path), a yield under v1 pressure that frees the
claim and both flocks, an exception inside a build that still releases the claim and stops the build (review d2clean
finding 1), and the verify receipt written after the card is free.
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
import d2refill_recache as RR  # noqa: E402

SHA = "b" * 64


def test_go_file():
    assert RR.read_go(None)["ok"] is False
    assert RR.read_go("d2clean-recache-go\ncards=1")["ok"] is False
    go = RR.read_go("d2refill-recache-go\n# comment\ncards_after=2026-10-15T21:40:00Z:1,2\n", now="2026-10-15T05:00:00Z")
    assert go["ok"] and go["cards"] == [] and go["pending"] == [{"after": "2026-10-15T21:40:00Z", "cards": [1, 2]}]
    go = RR.read_go("d2refill-recache-go\ncards=3\ncards_after=2026-10-15T21:40:00Z:1,2\n", now="2026-10-15T21:40:00Z")
    assert go["cards"] == [1, 2, 3] and go["pending"] == []
    assert RR.read_go("d2refill-recache-go\ncards_after=tonight:1\n")["ok"] is False
    assert RR.read_go("d2refill-recache-go\ncards=x\n")["ok"] is False


def test_p40_micro_batch_never_exceeds_the_measured_pass():
    assert RR.micro_batch("p40", 296, 2) == 16 and RR.micro_batch("p40", 817, 2) == 10 and RR.micro_batch("p40", 4096, 2) == 2
    assert RR.micro_batch("p40", 4841, 2) == 2  # wider than the probe: never below 2, the measured micro-batch
    assert RR.micro_batch("rtx3090", 296, 8) == 8
    for width in (100, 296, 817, 2048, 4096):
        assert RR.micro_batch("p40", width, 2) * width <= max(RR.PASS_TOKENS, 2 * width)


def test_pick_order_and_exclusions():
    c = [{"rank": 3, "name": "refill-codex-v8-train", "eligible": True, "corpus_sha256": "8", "cache_sealed": False},
         {"rank": 3, "name": "refill-codex-v7-train", "eligible": True, "corpus_sha256": "7", "cache_sealed": True, "verified": False},
         {"rank": 2, "name": "refill-selfgen", "eligible": False, "corpus_sha256": "2", "cache_sealed": False},
         {"rank": 2, "name": "other", "eligible": True, "corpus_sha256": "9", "cache_sealed": True, "verified": True}]
    assert [x["name"] for x in RR.pick(c, set())] == ["refill-codex-v7-train", "refill-codex-v8-train"]
    assert [x["name"] for x in RR.pick(c, {"7"})] == ["refill-codex-v8-train"]


def test_claim_lines_read_by_d1_seal_lib():
    claim = RR.claim_line(1, 4242, "/x/gpu1.log")
    held, line = SL.claim_state([claim], 1)
    assert held and "lane=d2refill" in line
    assert SL.claim_state([claim, RR.release_line(1)], 1) == (False, None)
    # the d2clean runner's live claim on the same card keeps it held after this lane releases (and vice versa)
    held, line = SL.claim_state([RC.claim_line(1, 7, "y"), claim, RR.release_line(1)], 1)
    assert held and "lane=d2clean" in line


@pytest.fixture()
def world(tmp_path, monkeypatch):
    q = tmp_path / "queues"
    q.mkdir()
    for mod in (RC, RR):
        monkeypatch.setattr(mod, "Q", q)
    monkeypatch.setattr(RC, "CLAIMS", q / "p40.claims")
    monkeypatch.setattr(RC, "OUT_ROOT", tmp_path / "cache")
    monkeypatch.setattr(RR, "STATUS", q / "d2refill.status")
    monkeypatch.setattr(RR, "GO_FILE", q / "d2refill.go")
    monkeypatch.setattr(RR, "REFILL_DIR", tmp_path / "d2_refill")
    monkeypatch.setattr(RR, "POLL_S", 1)
    (q / "d2refill.go").write_text("d2refill-recache-go\ncards=1\n")

    class A:
        card = 1
        refill_dir = str(tmp_path / "d2_refill")

    runner = RR.RefillRunner(A())
    monkeypatch.setattr(runner, "heartbeat", lambda detail: None)
    return runner, q, tmp_path


F = {"rank": 3, "name": "refill-codex-v7-train", "file": "codex-native-v7-train.safetensors", "corpus": "/x/codex-native-v7-train.safetensors",
     "corpus_sha256": SHA, "width": 817, "eligible": True, "cache_sealed": False}


def test_build_command_runs_the_frozen_tool_with_the_sealed_flags(world):
    runner, _, _ = world
    cmd, env = runner.build_command(F)
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[3] == RC.PASCAL_PY and cmd[4] == str(RC.FROZEN) and cmd[5] == "build-text"
    assert cmd[cmd.index("--corpus") + 1] == F["corpus"]
    assert cmd[cmd.index("--micro-batch") + 1] == "10" and cmd[cmd.index("--linear-attn-kernel") + 1] == "torch"
    joined = " ".join(cmd)
    assert " ".join(RC.SEALED_FLAGS) in joined and "--out-root " + str(RC.OUT_ROOT) in joined
    assert env["CUDA_VISIBLE_DEVICES"] == RC.CARDS[1][0] and env["QTEXT_TPS_PROBE"] == str(RC.ENGINE)


def fake_build(tmp: Path, body: str) -> Path:
    fake = tmp / "fake_build.py"
    fake.write_text(body)
    return fake


def free(path: Path) -> bool:
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def test_yield_on_v1_pressure_frees_claim_and_locks(world, monkeypatch):
    runner, q, tmp = world
    marker = tmp / "got_sigterm"
    fake = fake_build(tmp, "import signal, sys, time\n"
                           f"signal.signal(signal.SIGTERM, lambda *_: (open({str(marker)!r}, 'w').write('143'), sys.exit(143)))\n"
                           "time.sleep(120)\n")
    monkeypatch.setattr(runner, "build_command", lambda f: ([sys.executable, str(fake), "--micro-batch", "10"], dict(os.environ)))
    calls = {"n": 0}

    def gates(starting):
        calls["n"] += 1
        return (["gpu3_wait: d1_seal waiting"] if calls["n"] >= 2 else []), {}

    monkeypatch.setattr(runner, "gates", gates)
    assert runner.take_file_lock({"file": F["file"]}) and runner.take_card_lock()
    with open(RC.CLAIMS, "a") as handle:
        handle.write(RR.claim_line(1, os.getpid(), "x") + "\n")
    runner.claimed = True
    started = time.time()
    try:
        result = runner.run_file(F)
    finally:
        runner.release_all("test")
    assert result == "yielded" and marker.read_text() == "143" and time.time() - started < 100
    assert SL.claim_state(RC.CLAIMS.read_text().splitlines(), 1) == (False, None)
    assert free(q / "gpu1.lock") and free(q / "d2refill" / "claims" / f"{F['file']}.lock")
    assert "YIELD codex-native-v7-train.safetensors: gpu3_wait" in RR.STATUS.read_text()


def test_an_exception_in_the_build_loop_still_stops_the_build_and_releases(world, monkeypatch):
    runner, q, tmp = world
    marker = tmp / "got_sigterm"
    fake = fake_build(tmp, "import signal, sys, time\n"
                           f"signal.signal(signal.SIGTERM, lambda *_: (open({str(marker)!r}, 'w').write('143'), sys.exit(143)))\n"
                           "time.sleep(120)\n")
    monkeypatch.setattr(runner, "build_command", lambda f: ([sys.executable, str(fake), "--micro-batch", "10"], dict(os.environ)))

    def boom(starting):
        raise ValueError("cache manifest mid-write")

    monkeypatch.setattr(runner, "gates", boom)
    # the run() pattern: claim, run_file inside try/finally release_all
    assert runner.take_file_lock({"file": F["file"]}) and runner.take_card_lock()
    with open(RC.CLAIMS, "a") as handle:
        handle.write(RR.claim_line(1, os.getpid(), "x") + "\n")
    runner.claimed = True
    with pytest.raises(ValueError):
        try:
            runner.run_file(F)
        finally:
            runner.release_all("exception")
    assert marker.read_text() == "143" and runner.proc.poll() is not None
    assert SL.claim_state(RC.CLAIMS.read_text().splitlines(), 1) == (False, None) and free(q / "gpu1.lock")


def test_a_sealed_build_is_verified_after_the_card_is_free(world, monkeypatch):
    runner, q, tmp = world
    cache = RC.OUT_ROOT / RC.TEACHER / SHA
    cache.mkdir(parents=True)
    (cache / "manifest.json").write_text(json.dumps({"all_finite": True, "all_structural_ok": True,
                                                     "teacher": {"equal_before_and_after_every_session": True}}))
    monkeypatch.setattr(runner, "build_command", lambda f: ([sys.executable, "-c", "pass", "--micro-batch", "10"], dict(os.environ)))
    monkeypatch.setattr(runner, "gates", lambda starting: ([], {}))
    assert runner.run_file(F) == "sealed"
    seen = {}

    def verify_cache(f, env):
        seen["card_lock_free"] = free(q / "gpu1.lock")
        return 0, "VERIFY OK"

    monkeypatch.setattr(runner, "verify_cache", verify_cache)
    assert runner.verify(F) == 0 and seen["card_lock_free"]
    receipt = json.loads(RR.verify_receipt_path(SHA).read_text())
    assert receipt["rc"] == 0 and receipt["cache_manifest_sha256"] == RC.sha256_file(cache / "manifest.json")
    monkeypatch.setattr(runner, "verify_cache", lambda f, env: (1, "VERIFY FAILED"))
    assert runner.verify(F) == 1 and json.loads(RR.verify_receipt_path(SHA).read_text())["rc"] == 1


def test_run_loop_waits_for_the_go_time_then_stops_when_off_the_go_file(world, monkeypatch):
    runner, q, _ = world
    monkeypatch.setattr(runner, "refill_state", lambda: [dict(F)])
    monkeypatch.setattr(runner, "gates", lambda starting: (["go: card 1 not in cards=[]"], {}))
    (q / "d2refill.go").write_text("d2refill-recache-go\ncards=3\n")  # card 1 neither now nor later
    assert runner.run() == 0 and "STOP: card 1 is not in the GO file" in RR.STATUS.read_text()


def test_refused_cards_never_build_a_runner(capsys):
    for card in (5, 6, 7):
        assert RR.main(["probe", "--card", str(card)]) == 3
    assert "REFUSED card 5" in capsys.readouterr().out
