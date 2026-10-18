"""Tests for conftest.py (oc_cuda_guard 1.0.1, card X-T1, lane infra, Claude, 2026-10-14).  AI server, CPU only:

    cd /home/cereal/omni-campaign/distill/tools
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -p no:cacheprovider -q test_oc_cuda_guard.py

What is pinned:
  * under pytest the process sees CUDA_VISIBLE_DEVICES == "" and torch reports no CUDA device;
  * a subprocess a test starts (a trainer, an entry point) inherits the hiding;
  * every directory under distill/tools that holds test files has a conftest.py byte-equal to this directory's one;
  * OC_TEST_ALLOW_GPU=1 without a named card is refused, and (1.0.1) a named card without OC_TEST_ALLOW_GPU=1 is
    refused instead of silently hidden; "" and "-1" from the caller still run, hidden.
Every assertion that would touch the CUDA driver checks the variable FIRST, so the negative control
(`env -u CUDA_VISIBLE_DEVICES python -m pytest --noconftest test_oc_cuda_guard.py`) fails without opening any card.
With OC_TEST_ALLOW_GPU=1 the hiding tests are skipped (the caller named its own card).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ALLOWED = os.environ.get("OC_TEST_ALLOW_GPU", "") == "1"
SKIP_DIRS = {"__pycache__", ".pytest_cache"}
needs_hiding = pytest.mark.skipif(ALLOWED, reason="OC_TEST_ALLOW_GPU=1: the caller opted in to its own card")


def assert_hidden():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}: conftest.py did not hide the cards")


@needs_hiding
def test_cuda_visible_devices_is_empty():
    assert_hidden()


@needs_hiding
def test_torch_sees_no_cuda_device():
    assert_hidden()  # before any torch.cuda call: an unguarded run must fail here, not open a card
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available() is False
    assert torch.cuda.device_count() == 0
    assert torch.cuda.is_initialized() is False


@needs_hiding
def test_subprocess_inherits_hiding():
    assert_hidden()
    pytest.importorskip("torch")
    code = ("import os, torch; print(repr(os.environ.get('CUDA_VISIBLE_DEVICES')), torch.cuda.is_available(), "
            "torch.cuda.device_count())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split() == ["''", "False", "0"]


def _test_dirs(root: Path):
    found = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        if any(n.startswith("test_") and n.endswith(".py") for n in filenames):
            found.add(Path(dirpath))
    return sorted(found)


def test_every_test_directory_has_the_guard():
    want = hashlib.sha256((HERE / "conftest.py").read_bytes()).hexdigest()
    dirs = _test_dirs(HERE)
    assert HERE in dirs
    bad = []
    for d in dirs:
        c = d / "conftest.py"
        if not c.is_file():
            bad.append(f"{d}: no conftest.py")
        elif hashlib.sha256(c.read_bytes()).hexdigest() != want:
            bad.append(f"{d}: conftest.py differs from {HERE / 'conftest.py'}")
    assert not bad, "\n".join(bad)


def child_session(tmp_path: Path, cuda: str | None, allow: bool):
    # a child pytest session that collects one trivial test from a temp dir holding a copy of this conftest; the test
    # prints the variable it saw, so a passing run shows what the child was given
    (tmp_path / "conftest.py").write_bytes((HERE / "conftest.py").read_bytes())
    (tmp_path / "test_trivial.py").write_text(
        "import os\ndef test_ok():\n    print('SAW', repr(os.environ.get('CUDA_VISIBLE_DEVICES')))\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "OC_TEST_ALLOW_GPU")}
    if cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda
    if allow:
        env["OC_TEST_ALLOW_GPU"] = "1"
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", str(tmp_path)],
                          capture_output=True, text=True, env=env, cwd=tmp_path, timeout=300)


def test_conftest_refuses_allow_without_a_named_card(tmp_path):
    # OC_TEST_ALLOW_GPU=1 with no CUDA_VISIBLE_DEVICES must stop the session before collection (it would expose every
    # card)
    out = child_session(tmp_path, None, allow=True)
    assert out.returncode != 0
    assert "needs an explicit CUDA_VISIBLE_DEVICES" in out.stdout + out.stderr


def test_conftest_refuses_a_named_card_without_allow(tmp_path):
    # 1.0.1: the L-K3ACC driver exported CUDA_VISIBLE_DEVICES=3 and ran its GPU unit tests without the opt-in; 1.0.0
    # hid the card and the run passed on CPU.  Now the session stops before collection (UsageError, exit 4).
    out = child_session(tmp_path, "3", allow=False)
    text = out.stdout + out.stderr
    assert out.returncode == 4, text[-2000:]
    assert "the caller named card(s) CUDA_VISIBLE_DEVICES='3' without OC_TEST_ALLOW_GPU=1" in text
    assert "SAW" not in text  # no test ran


@pytest.mark.parametrize("cuda", ["", "-1"])
def test_conftest_runs_hidden_when_the_caller_names_no_card(tmp_path, cuda):
    out = child_session(tmp_path, cuda, allow=False)
    text = out.stdout + out.stderr
    assert out.returncode == 0, text[-2000:]
    assert "SAW ''" in text and "1 passed" in text
