"""Tests for run_media_a_wave2.sh 1.0.3 (lane media-a, Claude, 2026-10-13; 1.0.3 Lexar mount test by lane gpu4). Server, no GPU, no docker:

    cd /home/cereal/omni-campaign/distill/tools
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_run_media_a_wave2.py

The driver runs against a fake `nvidia-smi` (thresholds and a live temperature read from files) and a fake `docker`
(`run` sleeps, `stop`/`kill` end it and are logged) placed first on PATH, with OC pointed at a temp dir. Covers the
card refusals (GPU5, step-1 cards, H3 on GPU6 without a thermal ruling, a busy card), the thermal refusals (no
thresholds, a stop line at/above shutdown, a card already near the stop line), the allocator receipt's thermal
block, and the watchdog actually stopping the container when the card crosses the stop line.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = HERE / "run_media_a_wave2.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None or os.name == "nt", reason="needs a POSIX bash")

FAKE_SMI = r"""#!/usr/bin/env bash
D="$FAKE_GPU_DIR"; gpu=""; prev=""
for a in "$@"; do [ "$prev" = "-i" ] && gpu="$a"; prev="$a"; done
t() { v="${1:-NA}"; [ "$v" = NA ] && echo "N/A" || echo "$v C"; }
case "$*" in
  *"-q -d TEMPERATURE"*)
    echo "GPU 00000000:88:00.0"; echo "    Temperature"
    echo "        GPU Current Temp                               : $(cat "$D/temp") C"
    echo "        GPU T.Limit Temp                               : N/A"
    echo "        GPU Shutdown Temp                              : $(t "$FAKE_SHUT")"
    echo "        GPU Slowdown Temp                              : $(t "$FAKE_SLOW")"
    echo "        GPU Max Operating Temp                         : $(t "$FAKE_MAXOP")"
    echo "        Memory Current Temp                            : N/A"
    echo "        Memory Max Operating Temp                      : 95 C";;
  *query-gpu=name*) echo "${FAKE_NAME:-Tesla P40}";;
  *query-gpu=uuid*) echo "GPU-fake-$gpu";;
  *query-gpu=memory.used*) echo "${FAKE_USED:-5}";;
  *query-gpu=temperature.gpu*) cat "$D/temp";;
  *) echo "fake nvidia-smi: unhandled $*" >&2; exit 9;;
esac
"""

FAKE_DOCKER = r"""#!/usr/bin/env bash
D="$FAKE_GPU_DIR"
case "$1" in
  run) echo "$*" > "$D/docker.args"; echo $$ > "$D/docker.pid"; exec sleep "${FAKE_RUN_SECONDS:-60}";;
  stop|kill) echo "$*" >> "$D/docker.stops"; kill "$(cat "$D/docker.pid")" 2>/dev/null; exit 0;;
  *) echo "fake docker: unhandled $*" >&2; exit 9;;
esac
"""


@pytest.fixture()
def rig(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("nvidia-smi", FAKE_SMI), ("docker", FAKE_DOCKER)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    gpu_dir = tmp_path / "gpu"
    gpu_dir.mkdir()
    (gpu_dir / "temp").write_text("50\n")
    oc = tmp_path / "oc"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "H3_GPU6", "THERMAL_", "ALLOW_STEP1"))}
    env.update({"PATH": f"{bin_dir}:{env.get('PATH', '/usr/bin:/bin')}", "FAKE_GPU_DIR": str(gpu_dir), "OC": str(oc),
                "FAKE_SHUT": "87", "FAKE_SLOW": "84", "FAKE_MAXOP": "82", "WATCHDOG_INTERVAL": "1",
                "ITEMS_QIMG": "/nonexistent/items.jsonl", "ITEMS_H3": "/nonexistent/items.jsonl",
                "LEXAR_TEACHER_CACHE": str(tmp_path / "lexar" / "teacher_cache")})

    def run(step, gpu, **extra):
        e = dict(env, **{k: str(v) for k, v in extra.items()})
        return subprocess.run(["bash", str(DRIVER), step, str(gpu)], env=e, capture_output=True, text=True, timeout=60)

    return {"run": run, "gpu": gpu_dir, "oc": oc, "lexar": tmp_path / "lexar" / "teacher_cache"}


def _alloc(rig, step, gpu, card="teslap40"):
    return rig["oc"] / "composite" / "receipts" / "allocator" / f"media-a-{step}-gpu{gpu}-{card}-claude.json"


def _wait_ended(path, seconds=40.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if path.is_file():
            rec = json.loads(path.read_text())
            if rec.get("state") == "ended":
                return rec
        time.sleep(0.5)
    raise AssertionError(f"{path} never reached state ended")


def test_refuses_gpu5_and_step1_cards(rig):
    r = rig["run"]("c-qimg-encode", 5)
    assert r.returncode == 2 and "GPU5 is reserved" in r.stdout
    for gpu in (0, 4, 6):
        r = rig["run"]("c-qimg-encode", gpu)
        assert r.returncode == 2 and "belongs to step 1" in r.stdout


@pytest.mark.parametrize("step", ["p4-h3-bf16", "p4-h3-fp16", "p4-h3-int8", "c-h3-velocity", "c-h3-encode"])
def test_refuses_h3_on_gpu6_without_thermal_ruling(rig, step):
    r = rig["run"](step, 6, ALLOW_STEP1_CARDS=1, FAKE_NAME="Quadro RTX 8000", H3_VAE_CACHE="/x", H3_TE_CACHE="/y")
    assert r.returncode == 2 and "H3 on GPU6 is denied" in r.stdout, r.stdout + r.stderr
    assert not (rig["oc"] / "composite" / "receipts" / "allocator").exists() or not list(
        (rig["oc"] / "composite" / "receipts" / "allocator").glob("*.json"))


def test_qimg_on_gpu6_is_not_an_h3_refusal(rig):
    r = rig["run"]("c-qimg-encode", 6, ALLOW_STEP1_CARDS=1, FAKE_NAME="Quadro RTX 8000", FAKE_RUN_SECONDS=1)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = _wait_ended(_alloc(rig, "c-qimg-encode", 6, "quadrortx8000"))
    assert rec["exit"] == 0 and rec["thermal"]["h3_gpu6_thermal_ruling"] == ""


def test_busy_card_and_thermal_refusals(rig):
    r = rig["run"]("c-qimg-encode", 3, FAKE_USED=2500)
    assert r.returncode == 2 and "MiB in use" in r.stdout
    r = rig["run"]("c-qimg-encode", 3, FAKE_SHUT="NA", FAKE_SLOW="NA", FAKE_MAXOP="NA")
    assert r.returncode == 2 and "no temperature thresholds" in r.stdout
    r = rig["run"]("c-qimg-encode", 3, THERMAL_STOP_C=87)
    assert r.returncode == 2 and "not below shutdown" in r.stdout
    (rig["gpu"] / "temp").write_text("78\n")
    r = rig["run"]("c-qimg-encode", 3)
    assert r.returncode == 2 and "within 5 C of the 82 C stop line" in r.stdout
    assert not list(rig["oc"].rglob("media-a-*-claude.json"))


def test_stop_line_fallbacks(rig):
    r = rig["run"]("c-qimg-encode", 3, FAKE_MAXOP="NA", FAKE_RUN_SECONDS=1)
    assert r.returncode == 0 and "thermal stop 81 C (slowdown-3" in r.stdout, r.stdout
    _wait_ended(_alloc(rig, "c-qimg-encode", 3))
    r = rig["run"]("c-qimg-encode", 3, FAKE_MAXOP="NA", FAKE_SLOW="NA", FAKE_RUN_SECONDS=1)
    assert r.returncode == 0 and "thermal stop 81 C (shutdown-6" in r.stdout, r.stdout
    _wait_ended(_alloc(rig, "c-qimg-encode", 3))


def test_receipt_records_thresholds_and_clean_exit(rig):
    r = rig["run"]("c-qimg-encode", 3, FAKE_RUN_SECONDS=2)
    assert r.returncode == 0, r.stdout + r.stderr
    path = _alloc(rig, "c-qimg-encode", 3)
    rec = _wait_ended(path)
    assert rec["thermal"] == {"current_c": "50", "slowdown_c": "84", "max_operating_c": "82", "shutdown_c": "87",
                              "stop_c": 82, "stop_from": "max_operating", "h3_gpu6_thermal_ruling": ""}
    assert rec["exit"] == 0 and rec["thermal_stop_fired"] == 0 and rec["peak_temp_c"] == 50
    assert rec["driver"] == "run_media_a_wave2.sh 1.0.3"
    args = (rig["gpu"] / "docker.args").read_text()
    assert "--name media_a_c_qimg_encode_gpu3" in args and "qimg-encode --items /nonexistent/items.jsonl" in args
    assert not (rig["gpu"] / "docker.stops").exists()


def test_watchdog_stops_the_container_at_the_stop_line(rig):
    r = rig["run"]("c-h3-encode", 3, FAKE_RUN_SECONDS=60)
    assert r.returncode == 0, r.stdout + r.stderr
    path = _alloc(rig, "c-h3-encode", 3)
    time.sleep(3)
    running = json.loads(path.read_text())
    assert running["state"] == "running" and running["thermal"]["stop_c"] == 82
    (rig["gpu"] / "temp").write_text("83\n")
    rec = _wait_ended(path, 30)
    assert rec["thermal_stop_fired"] == 1 and rec["peak_temp_c"] == 83 and rec["exit"] != 0
    assert "stop -t 20 media_a_c_h3_encode_gpu3" in (rig["gpu"] / "docker.stops").read_text()
    assert "[thermal watchdog]" in Path(rec["log"]).read_text()


def test_h3_on_gpu6_with_ruling_launches_and_records_it(rig):
    r = rig["run"]("p4-h3-bf16", 6, ALLOW_STEP1_CARDS=1, FAKE_NAME="Quadro RTX 8000", H3_GPU6_THERMAL_RULING="TR-test-1",
                   FAKE_RUN_SECONDS=1)
    assert r.returncode == 0, r.stdout + r.stderr
    rec = _wait_ended(_alloc(rig, "p4-h3-bf16", 6, "quadrortx8000"))
    assert rec["thermal"]["h3_gpu6_thermal_ruling"] == "TR-test-1" and rec["exit"] == 0


def test_lexar_teacher_cache_is_mounted_read_write_only_when_present(rig):
    r = rig["run"]("c-qimg-text", 3, FAKE_RUN_SECONDS=1)
    assert r.returncode == 0, r.stdout + r.stderr
    _wait_ended(_alloc(rig, "c-qimg-text", 3))
    args = (rig["gpu"] / "docker.args").read_text()
    assert str(rig["lexar"]) not in args
    rig["lexar"].mkdir(parents=True)
    r = rig["run"]("c-qimg-text", 3, FAKE_RUN_SECONDS=1)
    assert r.returncode == 0, r.stdout + r.stderr
    _wait_ended(_alloc(rig, "c-qimg-text", 3))
    args = (rig["gpu"] / "docker.args").read_text()
    lexar = str(rig["lexar"])
    assert f"-v {lexar}:{lexar} " in args and f"{lexar}:{lexar}:ro" not in args
