"""Tests for ``d1_seal_lib.py`` -- CPU / filesystem only.

  test_d1_state                 running / restarting / no status -> wait (10); succeeded but not finalized -> wait;
                                finalized success -> 0; an alert -> 2; a launch-id mismatch -> 2
  test_p40_claims_live_file     the server's p40.claims as of 2026-10-14T19:43Z: GPU1 / GPU2 held by lane toolcall
                                (a p2p11 claim + release on GPU1 inside it does NOT free it), GPU3 / GPU5 free;
                                then the toolcall RELEASE frees GPU1 / GPU2
  test_register_check_on_a_real_dry_run   the ledger's own TestDelta fixture (a reviewed fold child, its R2 clone) and
                                a replace-only delta: `oc_checkpoint.py register --delta-base` DRY RUN output passes
                                with the right numbers and fails with a wrong base sha / element count / op count
  test_p2_verdict               >= 151 streamed, < 151 split, < 60 token cut allowed, no completed cell -> none;
                                never a depth cut
  test_handoff                  five lines, in order
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parents[1] / "ledger" if (HERE.parents[1] / "ledger").is_dir() else Path("/home/cereal/omni-campaign/ledger")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LEDGER))

import d1_seal_lib as lib  # noqa: E402

LIVE_CLAIMS = """\
2026-10-14T14:28:20Z CLAIM GPU1,GPU2 owner=claude lane=toolcall what=llama-server parent Q4_K_M (re-export in progress) tool-use sampler, starts after export (~16:00Z), ETA ~8-10 h; release=selfgen_sampler.py stop via $OC/distill/toolcall/receipts/serve-toolcall-q38-q4km-gpu{1,2}.json; GPU3 NOT claimed (left for tests)
2026-10-14T14:52:12Z GPU1 lane=p2p11 owner=claude job=p2-memory-proof trainer=b6706f450418ac9d pid=4018805 log=/home/cereal/omni-campaign/distill/tps/p2p11/driver.log
2026-10-14T14:52:45Z GPU1 lane=p2p11 job=p2-memory-proof released (moved to GPU3: GPU1 is the toolcall lane claim of 14:28:20Z)
2026-10-14T14:53:08Z GPU3 lane=p2p11 owner=claude job=p2-memory-proof trainer=b6706f450418ac9d pid=4023993 log=/home/cereal/omni-campaign/distill/tps/p2p11/driver.log
2026-10-14T15:05:01Z GPU3 lane=p2p11 job=p2-memory-proof paused (operator stop before the C cells: P40 SDPA math kernel; relaunch with --sdpa-gqa repeat)
2026-10-14T15:06:08Z GPU3 lane=p2p11 owner=claude job=p2-memory-proof trainer=ffa81dd7aefd386f pid=4049666 log=/home/cereal/omni-campaign/distill/tps/p2p11/driver.log
2026-10-14T15:34:30Z GPU3 lane=p2p11 job=p2-memory-proof v2 stopped after R4 (R5 dropped); v3 = C cells, 1 step
2026-10-14T15:34:30Z GPU3 lane=p2p11 owner=claude job=p2-memory-proof trainer=ffa81dd7aefd386f pid=4099771 log=/home/cereal/omni-campaign/distill/tps/p2p11/driver.log
2026-10-14T15:51:59Z GPU3 lane=p2p11 job=p2-memory-proof released
2026-10-14T18:16:43Z GPU3 lane=k3accel owner=claude job=k3-profile-1.2.4 pid=163899 log=/mnt/lexar/omni-campaign/scratch/k3accel/runs/20261014T1830Z-p40/driver.log
2026-10-14T18:18:50Z GPU3 lane=k3accel job=k3-profile-1.2.4 released
2026-10-14T19:43:02Z GPU3 lane=k3accel owner=claude job=k3-v130-micro pid=284684 log=/mnt/lexar/omni-campaign/scratch/k3accel/runs/micro.log
2026-10-14T19:43:09Z GPU3 lane=k3accel job=k3-v130-micro released
"""


def test_d1_state(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "launch.json").write_text(json.dumps({"launch_id": "L1"}))
    assert lib.d1_state(run, "active")[0] == 10
    assert lib.d1_state(run, "")[0] == 10                                      # no status yet
    base = {"launch_id": "L1", "attempt": 1}
    (run / "runtime-status.json").write_text(json.dumps({**base, "completed": False, "state": "running"}))
    assert lib.d1_state(run, "inactive")[0] == 10
    (run / "runtime-status.json").write_text(json.dumps({**base, "completed": True, "state": "failed", "exit_code": 1}))
    code, line = lib.d1_state(run, "activating")
    assert code == 10 and "unit activating" in line                            # auto-restart pending
    assert lib.d1_state(run, "inactive")[0] == 10                              # no alert: systemd restarts it
    (run / "runtime-status.json").write_text(json.dumps({**base, "completed": True, "state": "succeeded", "exit_code": 0}))
    assert "finalize" in lib.d1_state(run, "inactive")[1]                      # ExecStopPost not run yet
    done = {**base, "completed": True, "state": "succeeded", "exit_code": 0, "systemd": {"service_result": "success"}}
    (run / "runtime-status.json").write_text(json.dumps(done))
    assert lib.d1_state(run, "inactive")[0] == 0
    assert lib.d1_state(run, "")[0] == 0                                       # transient unit already collected
    assert lib.d1_state(run, "active")[0] == 10                                # Result=success is printed while running
    (run / "runtime-status.json").write_text(json.dumps({**done, "launch_id": "L0"}))
    assert lib.d1_state(run, "inactive")[0] == 2
    (run / "runtime-status.json").write_text(json.dumps({**base, "completed": True, "state": "refused", "exit_code": 3,
                                                         "alert": "refused (exit 3)"}))
    assert lib.d1_state(run, "inactive")[0] == 2


def test_p40_claims_live_file(tmp_path: Path) -> None:
    claims = tmp_path / "p40.claims"
    claims.write_text(LIVE_CLAIMS)
    code, held = lib.p40_free(claims, [1, 2, 3])
    assert code == 10 and len(held) == 2 and all("lane=toolcall" in h for h in held)
    assert lib.p40_free(claims, [3]) == (0, [])
    assert lib.p40_free(claims, [5]) == (0, [])
    with claims.open("a") as fh:
        fh.write("2026-10-15T04:00:00Z RELEASE GPU1,GPU2 owner=claude lane=toolcall (toolcall self-gen complete)\n")
    assert lib.p40_free(claims, [1, 2, 3]) == (0, [])
    with claims.open("a") as fh:
        fh.write("2026-10-15T05:00:00Z CLAIM GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review pid=1\n")
    assert lib.p40_free(claims, [2])[0] == 10
    with claims.open("a") as fh:
        fh.write("2026-10-15T15:00:00Z RELEASE GPU1,GPU2,GPU3 owner=claude lane=d1seal job=d1-review released\n")
    assert lib.p40_free(claims, [1, 2, 3])[0] == 0


def test_register_check_on_a_real_dry_run() -> None:
    oc_checkpoint = pytest.importorskip("oc_checkpoint")
    tol = pytest.importorskip("test_oc_ledger")
    import oc_seal
    t = tol.TestDelta("test_version_string")
    t.setUp()
    try:
        directory = t.make_delta("d1", entries=[{"op": "replace", "target": "w", "tensor": "w.rows", "slice": [[0, 0, 4]]}],
                                 tensors={"w.rows": ("BF16", [4, 16])})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = t.run_checkpoint("register", "--checkpoint-id", t.D1, "--stage", "distill", "--directory", str(directory),
                                  "--delta-base", t.base, "--delta-base-sha256", t.base_sha,
                                  "--parent-directory", str(t.root / "parents" / "Q35-4B"), "--parent-code", "Q35-4B",
                                  "--recipe-command", "d1.sh")
        assert rc == 0
        log = buf.getvalue()
        parent_count, _ = oc_seal.safetensors_parameter_count(t.root / "parents" / "Q35-4B")
        ns = lambda **kw: type("A", (), {"checkpoint_id": t.D1, "base_id": t.base, "base_sha256": t.base_sha, "replace_ops": 1,
                                         "targets": 1, "replaced": 64, "parameters": 256, "parent_parameters": parent_count,
                                         "parent_code": "Q35-4B", **kw})()
        assert lib.register_check(log, ns()) == [], log
        assert lib.register_check(log, ns(base_sha256="f" * 64))
        assert lib.register_check(log, ns(replaced=65))
        assert lib.register_check(log, ns(replace_ops=2))
        assert lib.register_check(log.replace("DRY RUN", "WROTE"), ns())
        assert oc_checkpoint.history(t.registry, t.D1) == []                   # a dry run wrote nothing
    finally:
        t.tearDown()


def test_p2_verdict(tmp_path: Path) -> None:
    def summary(rates: dict[str, float | None]) -> Path:
        cells = {c: ({"state": "completed", "steady_tok_per_s_mean_steps_2_on": r} if r is not None
                     else {"state": "failed", "reason": "OOM backward.mixer"}) for c, r in rates.items()}
        p = tmp_path / f"s{len(list(tmp_path.iterdir()))}.json"
        p.write_text(json.dumps({"cells": cells}))
        return p

    v = lib.p2_verdict(summary({"L4096-mb8": 170.2, "L4096-mb4": 160.0, "L32768-mb1": None}))
    assert v["verdict"] == "streamed" and v["best_cell"] == "L4096-mb8" and v["depth_cut"] is False and v["long_rung"] == "failed"
    v = lib.p2_verdict(summary({"L4096-mb8": 120.0}))
    assert v["verdict"] == "split" and v["token_cut_allowed"] is False
    assert lib.p2_verdict(summary({"L4096-mb8": 50.0}))["token_cut_allowed"] is True
    assert lib.p2_verdict(summary({"L4096-mb8": None}))["verdict"] == "none"
    assert lib.p2_verdict(tmp_path / "absent.json")["verdict"] == "none"
    out = tmp_path / "p2.verdict"
    assert lib.main(["p2-verdict", "--summary", str(summary({"L4096-mb4": 151.0})), "--out", str(out)]) == 0
    first = out.read_text().splitlines()[0]
    assert first.startswith("streamed ") and json.loads(out.read_text().splitlines()[1])["summary_sha256"]


def test_handoff(tmp_path: Path) -> None:
    out = tmp_path / "h.md"
    assert lib.main(["handoff", "--out", str(out), "--lane", "d1seal", "--state", "s", "--running", "r", "--next", "n",
                     "--untouchable", "u"]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) == 5 and [ln.split(".")[0] for ln in lines] == ["1", "2", "3", "4", "5"]
