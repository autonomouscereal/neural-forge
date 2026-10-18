"""Tests for ``run_toolcall_extend_2026-10-15.sh`` (L-TOOL: tool-call self-gen 1.5M -> 3.0M, stop by 22:00Z).

The driver runs for real (bash, the real toolcall_selfgen.py subprocesses, the real finalize screen and
qtext_teacher_cache load); only the llama-servers are stood in by the in-process scripted /completion of
test_toolcall_selfgen, and every path is redirected into tmp through the driver's TOOLCALL_EXT_* hooks
(the launch receipts name pids that have exited and allocator names that do not exist, so
`selfgen_sampler.py stop` touches nothing real).

  test_driver_target_path          run 1 sealed, continuation reaches 3.0M-equivalent, one corpus over both runs
                                   at C1's path, RELEASE line after the export, transient Q4 removed
  test_driver_hard_stop_path       the sampler hangs: the hard stop kills run 2, the servers are released before
                                   finalize, run 2 sealed --accept-short --cut-at-gap (or run 1 alone if it journaled nothing)
  test_driver_past_deadline_path   started after --stop-by: no continuation run, run 1 alone exported, cards released

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
    CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src \\
    $PY -m pytest -o addopts="" -p no:cacheprovider test_run_toolcall_extend.py -q
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import toolcall_selfgen as tc  # noqa: E402
from test_toolcall_selfgen import PARENT, Scripted, _Handler  # noqa: E402

DRIVER = HERE / "run_toolcall_extend_2026-10-15.sh"
pytestmark = pytest.mark.skipif(not (PARENT / "tokenizer.json").is_file(), reason="parent tokenizer not on this machine")


class Slow(Scripted):
    """Scripted, plus a hang after `hang_after` requests (a sampler that stops answering)."""

    hang_after = None

    def respond(self, prompt_ids):
        if self.hang_after is not None and self.requests >= self.hang_after:
            time.sleep(3600)
        return super().respond(prompt_ids)


def utc_in(seconds: float) -> str:
    return (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    """A schema library, an 80-item plan, the bank and a finished (unsealed) run 1 of 5,000 tokens."""
    renderer = tc.Renderer(PARENT)
    script = Slow(renderer)
    handler = type("H", (_Handler,), {"script": script})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    root = tmp_path_factory.mktemp("staged")
    tcd = root / "toolcall"
    domains = tc.DOMAINS
    tc.DOMAINS = tc.DOMAINS[:4]
    try:
        assert tc.main(["schemas", "--server", url, "--out-dir", str(tcd / "schemas"), "--per-domain", "2", "--workers-per-server", "2"]) == 0
    finally:
        tc.DOMAINS = domains
    plan = tcd / "plan-2026-10-14.jsonl"
    assert tc.main(["plan", "--schemas", str(tcd / "schemas" / "schemas.jsonl"), "--out", str(plan), "--items", "80"]) == 0
    bank = root / "bank.json"
    assert tc.main(["bank-build", "--out", str(bank)]) == 0
    q4 = root / "scratch" / "fake.Q4_K_M.gguf"
    q4.parent.mkdir(parents=True)
    q4.write_bytes(b"not a model")
    receipts = tcd / "receipts"
    receipts.mkdir(parents=True)
    (receipts / "export_check_Q38-27BU-parent_toolcall.json").write_text(json.dumps(
        {"parent_unchanged": True, "q4": {"path": str(q4), "sha256": "0" * 64}}))
    for g in (1, 2):
        (receipts / f"serve-toolcall-q38-q4km-gpu{g}.json").write_text(json.dumps(
            {"pid": dead_pid(), "name": f"pytest-toolcall-extend-gpu{g}", "gguf": str(q4)}))
    assert tc.main(["run", "--plan", str(plan), "--server", url, "--out-dir", str(tcd / "run-q38-q4km-2026-10-14"), "--workers-per-server", "3",
                    "--progress-every", "0", "--token-target", "5000", "--gguf", str(q4), "--gguf-sha256", "0" * 64]) == 0
    yield {"root": root, "url": url, "script": script, "bank": bank}
    script.hang_after = None
    server.shutdown()


def drive(staged, tmp_path, *, total: int, stop_by: str, hard_stop: str, timeout: float = 900):
    root = tmp_path / "w"
    shutil.copytree(staged["root"], root)
    # the copied receipts still name the staged Q4: point them at this copy's
    q4 = root / "scratch" / "fake.Q4_K_M.gguf"
    check = root / "toolcall" / "receipts" / "export_check_Q38-27BU-parent_toolcall.json"
    check.write_text(json.dumps({"parent_unchanged": True, "q4": {"path": str(q4), "sha256": "0" * 64}}))
    env = dict(os.environ, TOOLCALL_EXT_PY=sys.executable, TOOLCALL_EXT_T=str(HERE), TOOLCALL_EXT_Q=str(root / "queues"),
               TOOLCALL_EXT_TC=str(root / "toolcall"), TOOLCALL_EXT_SCRATCH=str(root / "scratch"), TOOLCALL_EXT_BANK=str(root / "bank.json"),
               TOOLCALL_EXT_S1=staged["url"], TOOLCALL_EXT_S2=staged["url"], TOOLCALL_EXT_TARGET1="5000", TOOLCALL_EXT_TOTAL=str(total),
               TOOLCALL_STOP_BY=stop_by, TOOLCALL_HARD_STOP=hard_stop, CUDA_VISIBLE_DEVICES="")
    (root / "queues").mkdir()
    proc = subprocess.run(["bash", str(DRIVER), str(dead_pid())], env=env, capture_output=True, text=True, timeout=timeout)
    log = proc.stdout + proc.stderr
    tcd = root / "toolcall"
    claims = (root / "queues" / "p40.claims").read_text().splitlines()
    return proc.returncode, log, tcd, claims, q4


def sidecar_of(tcd: Path) -> dict:
    return tc.read_json(tcd / "rescore" / "toolcall-q38-2026-10-14.rows.json")


def released_after_export(claims: list[str], tcd: Path) -> None:
    assert claims[0].split()[1] == "CLAIM" and "lane=toolcall" in claims[0]
    assert sum("RELEASE GPU1,GPU2 owner=claude lane=toolcall" in c for c in claims) == 1  # post_d1_queue job_c1's grep
    sys.path.insert(0, str(HERE))
    import d1_seal_lib
    assert d1_seal_lib.claim_state(claims, 1) == (False, None) and d1_seal_lib.claim_state(claims, 2) == (False, None)


def test_driver_target_path(staged, tmp_path):
    staged["script"].hang_after = None
    rc, log, tcd, claims, q4 = drive(staged, tmp_path, total=9000, stop_by=utc_in(3600), hard_stop=utc_in(7200))
    assert rc == 0, log[-4000:]
    m1 = tc.read_json(tcd / "run-q38-q4km-2026-10-14" / "manifest.json")
    m2 = tc.read_json(tcd / "run-q38-q4km-2026-10-15-cont" / "manifest.json")
    assert not m1["short_of_target"] and m2["token_target"] == 9000 - m1["tokens"] and not m2["short_of_target"]
    assert m2["prior_manifests"][0]["manifest_sha256"] == m1["manifest_sha256"]
    assert tc.read_json(tcd / "run-q38-q4km-2026-10-15-cont" / "run_config.json")["base_seed"] == 20261015
    side = sidecar_of(tcd)
    assert [s["manifest_sha256"] for s in side["source_manifests"]] == [m1["manifest_sha256"], m2["manifest_sha256"]]
    assert side["rows"] == m1["conversations"] + m2["conversations"] and side["tokens"] == m1["tokens"] + m2["tokens"]
    assert side["corpus"]["sha256"] == tc.file_sha256(tcd / "rescore" / "toolcall-q38-2026-10-14.safetensors")
    released_after_export(claims, tcd)
    assert not q4.exists() and "removed transient" in log
    assert "export check: rows" in log and (tcd / "logs" / "extend.status").read_text().splitlines()[-1].split()[2] == "done"


def test_driver_hard_stop_path(staged, tmp_path):
    script = staged["script"]
    script.hang_after = script.requests + 30
    try:
        rc, log, tcd, claims, q4 = drive(staged, tmp_path, total=10 ** 9, stop_by=utc_in(7200), hard_stop=utc_in(45))
    finally:
        script.hang_after = None
    assert rc == 0, log[-4000:]
    assert "hard stop" in log and "run 2 rc=137" in log
    release = next(i for i, c in enumerate(claims) if "RELEASE" in c)
    assert "ended at/after" in claims[release]  # the cards were released before finalize / export
    side = sidecar_of(tcd)
    run2 = tcd / "run-q38-q4km-2026-10-15-cont"
    if (run2 / "manifest.json").exists() and tc.read_json(run2 / "manifest.json")["conversations"]:
        m2 = tc.read_json(run2 / "manifest.json")
        assert m2["short_of_target"] and len(side["source_manifests"]) == 2
    else:
        assert len(side["source_manifests"]) == 1
    released_after_export(claims, tcd)
    assert not q4.exists()


def test_driver_past_deadline_path(staged, tmp_path):
    staged["script"].hang_after = None
    rc, log, tcd, claims, q4 = drive(staged, tmp_path, total=10 ** 9, stop_by=utc_in(-60), hard_stop=utc_in(-30))
    assert rc == 0, log[-4000:]
    assert "no continuation run" in log and not (tcd / "run-q38-q4km-2026-10-15-cont").exists()
    assert len(sidecar_of(tcd)["source_manifests"]) == 1
    released_after_export(claims, tcd)
    assert not q4.exists()
