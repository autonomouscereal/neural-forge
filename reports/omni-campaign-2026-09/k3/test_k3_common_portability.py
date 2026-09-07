"""CPU-only portability contracts for K3's shared receipt and RSS helpers."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import k3_common as common


def test_receipt_sha_matches_exact_utf8_bytes_without_text_translation(tmp_path, monkeypatch):
    original = common.tempfile.NamedTemporaryFile

    def translated_tempfile(mode="w+b", *args, **kwargs):
        # Reproduce Windows text translation even when this test runs on Linux.
        if "b" not in mode:
            kwargs["newline"] = "\r\n"
        return original(mode, *args, **kwargs)

    monkeypatch.setattr(common.tempfile, "NamedTemporaryFile", translated_tempfile)
    payload = {"nested": {"number": 2}, "label": "naïve λ", "line": "a\nb"}
    path = tmp_path / "nested" / "receipt.json"
    digest = common.write_receipt(str(path), payload)
    expected = json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8")
    assert path.read_bytes() == expected
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_bytes()) == payload


def test_native_peak_rss_is_a_real_positive_measurement():
    if sys.platform not in {"win32", "linux", "darwin"}:
        pytest.skip("platform has no native peak RSS contract")
    assert common.peak_rss_bytes() > 0


@pytest.mark.parametrize("api_ok,peak,expected", [(True, 5 * (1 << 30), 5 * (1 << 30)),
                                                 (False, 1234, None), (True, 0, None)])
def test_windows_rss_reads_size_t_peak_working_set(monkeypatch, api_ok, peak, expected):
    import ctypes

    if ctypes.sizeof(ctypes.c_size_t) < 8:
        pytest.skip("model inference requires a 64-bit address space")

    def memory_info(_handle, pointer, _size):
        pointer._obj.PeakWorkingSetSize = peak
        pointer._obj.WorkingSetSize = 11
        pointer._obj.PeakPagefileUsage = 22
        return int(api_ok)

    kernel = SimpleNamespace(GetCurrentProcess=lambda: -1)
    psapi = SimpleNamespace(GetProcessMemoryInfo=memory_info)
    monkeypatch.setattr(ctypes, "WinDLL", lambda name, **_kw: kernel if name == "kernel32" else psapi,
                        raising=False)
    monkeypatch.setattr(common, "sys", SimpleNamespace(platform="win32"), raising=False)
    assert common.peak_rss_bytes() == expected


def test_windows_rss_native_api_failure_is_unknown(monkeypatch):
    import ctypes

    def unavailable(*_args, **_kwargs):
        raise OSError("native memory counter unavailable")

    monkeypatch.setattr(ctypes, "WinDLL", unavailable, raising=False)
    monkeypatch.setattr(common, "sys", SimpleNamespace(platform="win32"), raising=False)
    assert common.peak_rss_bytes() is None


def test_linux_rss_keeps_proc_high_water_mark_first(monkeypatch):
    monkeypatch.setattr(common, "sys", SimpleNamespace(platform="linux"), raising=False)
    monkeypatch.setattr(common, "open", lambda *_a, **_kw: io.StringIO("Name: test\nVmHWM:\t1234 kB\n"),
                        raising=False)
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_maxrss=9999)))
    assert common.peak_rss_bytes() == 1234 * 1024


@pytest.mark.parametrize("platform,value,expected", [("linux", 1234, 1234 * 1024), ("darwin", 1234, 1234),
                                                      ("linux", 0, None), ("linux", -1, None)])
def test_unix_rss_fallback_units_and_unknown_values(monkeypatch, platform, value, expected):
    def no_proc(*_args, **_kwargs):
        raise OSError("no proc mount")

    monkeypatch.setattr(common, "sys", SimpleNamespace(platform=platform), raising=False)
    monkeypatch.setattr(common, "open", no_proc, raising=False)
    monkeypatch.setitem(sys.modules, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _who: SimpleNamespace(ru_maxrss=value)))
    assert common.peak_rss_bytes() == expected
