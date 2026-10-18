"""Tests for p2_v133_summary.py 1.0.0 (lane p2p11, 2026-10-14). CPU, no GPU, no model: the inputs are the files
run_p2_v133.sh 1.1.0 leaves per attempt (<tag>.rc, <tag>.log with the trainer's `OOM {...}` line, the train receipt
with the fields streamed_kd_trainer 1.3.3 writes).  Run: python -m pytest -q test_p2_v133_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import p2_v133_summary as S

CELLS = ["L4096-mb8", "L4096-mb4", "L32768-mb1"]
GPU = 0
OOM_GDN = {"step": 0, "phase": "backward.mixer.linear_attention", "engine": {"peak_allocated_gib": 23.4},
           "predicted_phases_gib": {"backward.mixer.linear_attention": 25.55}}


def attempt(out: Path, tag: str, rc: int, *, oom: dict | None = None, rates: list[float] | None = None,
            verdict: str = "completed", receipt: bool = True) -> None:
    (out / f"{tag}.rc").write_text(f"{rc}\n")
    lines = [f"== {tag} 2026-10-14T18:00:00Z ffn 4096 head 256 MiB", "TRAIN {\"step\": 0}"]
    if oom is not None:
        lines += ["OOM " + json.dumps(oom, sort_keys=True), "Traceback (most recent call last):", "torch.OutOfMemoryError: CUDA out of memory."]
    (out / f"{tag}.log").write_text("\n".join(lines) + "\n")
    if rates is not None and receipt:
        r = {"verdict": verdict, "halt_reason": None if verdict == "completed" else "nonfinite", "steps": [{"tok_per_s": x} for x in rates],
             "memory_engine": {"peak_allocated_gib": 19.5, "phases_peak_allocated_gib": {"forward.mixer": 12.0}},
             "memory_model": {"peak_bytes": 20 * 2**30, "phases_bytes": {"forward.mixer": 12 * 2**30}},
             "adapter_census": {"parameters_trainable": 123}, "streamed_gb": 54.0, "layer_load_seconds": 30.0, "host_rss_gb": 17.0}
        S.receipt_path(out, tag, GPU).write_text(json.dumps(r))


def run(out: Path, capsys) -> tuple[int, dict, str]:
    rc = S.main(["--out", str(out), "--gpu", str(GPU), "--cells", ",".join(CELLS), "--long-cell", "L32768-mb1"])
    printed = capsys.readouterr().out
    return rc, json.loads((out / f"p2-summary-v133-gpu{GPU}.json").read_text()), printed


def test_all_cells_oom_gives_no_reading_never_a_token_cut(tmp_path, capsys):
    """The 1.0.0 inline summary printed 'best 0 -> split cell; tokens may be cut' here."""
    for c in CELLS:
        attempt(tmp_path, c, 1, oom=OOM_GDN)
        attempt(tmp_path, c + "-ffn2048", 1, oom=OOM_GDN)
    rc, s, printed = run(tmp_path, capsys)
    assert rc == 1
    assert s["best_steady_tok_per_s"] is None
    assert s["engine_rule_reading"].startswith("NO READING: no cell completed")
    assert "tokens may be cut" not in s["engine_rule_reading"] and "tokens may be cut" not in printed
    assert set(s["cells"]) == set(CELLS) and s["cells_completed"] == []
    for c in CELLS:
        e = s["cells"][c]
        assert e["state"] == "failed" and e["oom_phase"] == "backward.mixer.linear_attention"
        assert [a["tag"] for a in e["attempts"]] == [c, c + "-ffn2048"]
        assert all(a["receipt"] is None for a in e["attempts"])
    assert "best None -> NO READING" in printed


def test_long_rung_oom_is_named_in_the_reading(tmp_path, capsys):
    """The likely case: the 4k cells complete, L32768-mb1 OOMs in the GDN half on both attempts."""
    attempt(tmp_path, "L4096-mb8", 0, rates=[90.0, 160.0, 170.0, 165.0])
    attempt(tmp_path, "L4096-mb4", 0, rates=[80.0, 140.0, 150.0, 145.0])
    attempt(tmp_path, "L32768-mb1", 1, oom=OOM_GDN)
    attempt(tmp_path, "L32768-mb1-ffn2048", 1, oom=OOM_GDN)
    rc, s, printed = run(tmp_path, capsys)
    assert rc == 1
    assert s["best_steady_tok_per_s"] == 165.0
    reading = s["engine_rule_reading"]
    assert reading.startswith("streamed path (>= 151 tok/s) from L4096-mb8")
    assert "long rung failed: L32768-mb1 OOM in backward.mixer.linear_attention at step 0" in reading
    assert s["cells"]["L32768-mb1"]["state"] == "failed" and s["cells_failed"] == ["L32768-mb1"]
    assert "long rung failed" in printed and '"L32768-mb1": {"state": "failed"' in printed


def test_all_complete_reads_from_the_best_completed_cell(tmp_path, capsys):
    attempt(tmp_path, "L4096-mb8", 0, rates=[50.0, 100.0, 110.0, 120.0])
    attempt(tmp_path, "L4096-mb4", 0, rates=[50.0, 90.0, 90.0, 90.0])
    attempt(tmp_path, "L32768-mb1", 0, rates=[40.0, 70.0, 72.0])
    rc, s, _ = run(tmp_path, capsys)
    assert rc == 0 and s["cells_failed"] == []
    assert s["best_steady_tok_per_s"] == 110.0
    assert s["engine_rule_reading"] == "split cell (< 151 tok/s) from L4096-mb8"
    assert s["cells"]["L4096-mb8"]["tok_per_s_by_step"] == [50.0, 100.0, 110.0, 120.0]
    assert s["cells"]["L4096-mb8"]["predicted_peak_gib"] == 20.0


def test_retry_that_completes_is_used_and_the_first_oom_kept(tmp_path, capsys):
    attempt(tmp_path, "L4096-mb8", 1, oom={"step": 1, "phase": "forward.head"})
    attempt(tmp_path, "L4096-mb8-ffn2048", 0, rates=[30.0, 55.0, 55.0, 55.0])
    attempt(tmp_path, "L4096-mb4", 0, rates=[30.0, 50.0, 50.0, 50.0])
    attempt(tmp_path, "L32768-mb1", 0, rates=[30.0, 40.0, 40.0])
    rc, s, _ = run(tmp_path, capsys)
    e = s["cells"]["L4096-mb8"]
    assert rc == 0 and e["state"] == "completed" and e["used_attempt"] == "L4096-mb8-ffn2048"
    assert e["attempts"][0]["state"] == "failed" and e["attempts"][0]["oom_phase"] == "forward.head"
    assert s["engine_rule_reading"] == "split cell; tokens may be cut (< 60 tok/s) from L4096-mb8"


def test_driver_stopped_mid_run_marks_unrun_cells_and_reads_only_what_completed(tmp_path, capsys):
    """card_clear refuses before cell 2 (a D1 attempt appeared): the EXIT trap still writes the summary."""
    attempt(tmp_path, "L4096-mb8", 0, rates=[90.0, 160.0, 160.0, 160.0])
    rc, s, _ = run(tmp_path, capsys)
    assert rc == 1
    assert s["cells"]["L4096-mb4"]["state"] == "not_run" and s["cells"]["L32768-mb1"]["state"] == "not_run"
    assert "long rung failed: L32768-mb1 no attempt ran" in s["engine_rule_reading"]


@pytest.mark.parametrize("case,expect", [
    ("rc0-no-receipt", "rc=0 but no receipt"),
    ("crash-no-oom", "rc=1, no OOM line and no receipt"),
    ("halted-receipt", "receipt verdict 'halted'"),
])
def test_failures_without_an_oom_line_still_carry_a_reason(tmp_path, capsys, case, expect):
    for c in CELLS[:2]:
        attempt(tmp_path, c, 0, rates=[60.0, 70.0, 70.0, 70.0])
    if case == "rc0-no-receipt":
        attempt(tmp_path, "L32768-mb1", 0)
    elif case == "crash-no-oom":
        attempt(tmp_path, "L32768-mb1", 1)
    else:
        attempt(tmp_path, "L32768-mb1", 4, rates=[10.0], verdict="halted")
    _, s, _ = run(tmp_path, capsys)
    assert s["cells"]["L32768-mb1"]["state"] == "failed"
    assert expect in s["cells"]["L32768-mb1"]["reason"] and expect in s["engine_rule_reading"]


def test_a_receipt_without_an_rc_file_is_ignored(tmp_path, capsys):
    """A stale receipt from an earlier run (the driver moves those to prev/, but never trust a lone receipt)."""
    attempt(tmp_path, "L4096-mb8", 0, rates=[90.0, 300.0, 300.0, 300.0])
    (tmp_path / "L4096-mb8.rc").unlink()
    _, s, _ = run(tmp_path, capsys)
    assert s["cells"]["L4096-mb8"]["state"] == "not_run" and s["best_steady_tok_per_s"] is None
    assert s["engine_rule_reading"].startswith("NO READING")


def test_the_driver_calls_this_tool_with_its_cells():
    """run_p2_v133.sh 1.1.0: the summary is this tool, its cell list matches the cells it runs, and the pin is this file."""
    import hashlib
    here = Path(S.__file__).resolve().parent
    drv = (here / "run_p2_v133.sh").read_text()
    assert "<<'PY'\nimport glob" not in drv, "the 1.0.0 inline receipt-glob summary is gone"
    assert "CELLS=(L4096-mb8 L4096-mb4 L32768-mb1)" in drv and "LONG_CELL=L32768-mb1" in drv
    for c in CELLS:
        assert f"run_cell {c} " in drv
    pin = [ln for ln in drv.splitlines() if ln.startswith("WANT_S=")][0].split("=", 1)[1]
    assert pin == hashlib.sha256(Path(S.__file__).read_bytes()).hexdigest()
