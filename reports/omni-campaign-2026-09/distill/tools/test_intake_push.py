"""test_intake_push.py -- intake_push.py 1.0.0: copy, verify, never overwrite, never delete without the flag."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_push  # noqa: E402


def make(path: Path, data: bytes, age: float = 600.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    old = time.time() - age
    os.utime(path, (old, old))
    return path


def test_push_copies_verifies_and_is_idempotent(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    dst.mkdir()
    make(src / "speech" / "own-1" / "a.wav", b"RIFFaaaa")
    make(src / "speech" / "own-1" / "a.wav.part", b"partial")
    make(src / "speech" / "fresh.wav", b"new", age=0)
    make(src / "elsewhere" / "x.wav", b"ignored")
    assert intake_push.main(["--src", str(src), "--dst", str(dst)]) == 0
    assert (dst / "speech" / "own-1" / "a.wav").read_bytes() == b"RIFFaaaa"
    assert (dst / "speech" / "own-1" / "a.wav.sha256").read_text().startswith(intake_push.sha256_file(src / "speech" / "own-1" / "a.wav"))
    assert not (dst / "speech" / "own-1" / "a.wav.part").exists()
    assert not (dst / "speech" / "fresh.wav").exists()                       # still being written
    assert not (dst / "elsewhere").exists()
    assert (src / "speech" / "own-1" / "a.wav").exists()                     # local kept by default
    before = sorted(p.name for p in dst.rglob("*"))
    assert intake_push.main(["--src", str(src), "--dst", str(dst)]) == 0
    assert sorted(p.name for p in dst.rglob("*")) == before                    # idempotent


def test_push_never_overwrites_a_different_file(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src / "text" / "notes.txt", b"version two")
    make(dst / "text" / "notes.txt", b"version one")
    assert intake_push.main(["--src", str(src), "--dst", str(dst)]) == 0
    assert (dst / "text" / "notes.txt").read_bytes() == b"version one"
    pushed = [p for p in (dst / "text").iterdir() if p.name.startswith("notes.") and p.suffix == ".txt" and p.name != "notes.txt"]
    assert len(pushed) == 1 and pushed[0].read_bytes() == b"version two"


def test_delete_after_verify_only_with_flag(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    dst.mkdir()
    make(src / "images" / "p.png", b"\x89PNGdata")
    assert intake_push.main(["--src", str(src), "--dst", str(dst), "--delete-after-verify"]) == 0
    assert not (src / "images" / "p.png").exists() and (dst / "images" / "p.png").read_bytes() == b"\x89PNGdata"


def test_unreachable_drop_zone_refuses(tmp_path):
    assert intake_push.main(["--src", str(tmp_path), "--dst", str(tmp_path / "missing")]) == 2
