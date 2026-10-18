"""Tests for codex_inflight_adopt.py (lane inflight).  CPU only, no torch needed.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_codex_inflight_adopt.py

Small fake Codex outputs (a tree with a manifest using relative and container-absolute references, and a collection
tarball with its sha index) are adopted; every refusal path is exercised as a negative control, and the source is
proven byte-unchanged after every run.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import sys
import tarfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import codex_inflight_adopt as A  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def tree_digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture
def meminfo(tmp_path):
    def make(kib: int) -> str:
        p = tmp_path / f"meminfo-{kib}"
        p.write_text(f"MemTotal: 999999999 kB\nMemAvailable: {kib} kB\n")
        return str(p)
    return make


def make_tree(root: Path) -> dict[str, str]:
    (root / "cache" / "shard-00000").mkdir(parents=True)
    a = b"tensor-bytes-a" * 100
    b_ = b"tensor-bytes-b" * 50
    (root / "cache" / "shard-00000" / "a.bin").write_bytes(a)
    (root / "cache" / "b.bin").write_bytes(b_)
    manifest = {"shards": [{"path": "shard-00000/a.bin", "sha256": sha(a)}],
                "corpus": {"path": "/work/cache/b.bin", "sha256": sha(b_)},
                "parent": {"path": "/models/parent/model.safetensors", "sha256": "0" * 64}}
    mb = json.dumps(manifest).encode()
    (root / "cache" / "manifest.json").write_bytes(mb)
    return {"cache/manifest.json": sha(mb)}


def make_tar(tmp: Path, tamper: str | None = None, evil: str | None = None) -> tuple[Path, Path]:
    members = {"codex-x/native/000.safetensors": b"x" * 300, "codex-x/replay-index.json": b'{"entries": []}'}
    index = {"files": {k: {"bytes": len(v), "sha256": sha(v)} for k, v in members.items()}}
    if tamper:
        members[tamper] = b"tampered"
    tar_path = tmp / "collection.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        for name, data in list(members.items()) + ([(evil, b"evil")] if evil else []):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    index["archive_sha256"] = sha(tar_path.read_bytes())
    idx_path = tmp / "collection.tar.json"
    idx_path.write_text(json.dumps(index))
    return tar_path, idx_path


def run(plan_items, dest: Path, tmp: Path, mem: str, extra=()):
    plan = tmp / "plan.json"
    plan.write_text(json.dumps({"schema": A.PLAN_SCHEMA, "items": plan_items}))
    return A.main(["run", "--plan", str(plan), "--dest", str(dest), "--meminfo", mem, "--mem-wait-s", "0", *extra])


def test_tree_adopt_pass_and_readonly(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    pins = make_tree(src)
    before = tree_digest(src)
    dest = tmp_path / "dest"
    item = {"name": "toolcache", "kind": "tree", "src": str(src), "pins": pins,
            "manifests": ["cache/manifest.json"], "prefix_map": {"/work": "."}}
    assert run([item], dest, tmp_path, meminfo(50 << 20)) == 0
    out = capsys.readouterr().out
    assert "ADOPT PASS toolcache: files 3" in out and "refs 3 ok 2 external 1" in out
    rec = json.loads((dest / "toolcache" / "ADOPTION.json").read_text())
    assert rec["status"] == "PASS" and rec["source_pins"]["cache/manifest.json"]["source_sha256"] == pins["cache/manifest.json"]
    copied = dest / "toolcache" / "payload" / "cache" / "b.bin"
    assert not (stat.S_IMODE(copied.stat().st_mode) & 0o222)
    assert tree_digest(src) == before
    assert A.main(["verify-item", "--dir", str(dest / "toolcache")]) == 0
    assert "VERIFY toolcache: 3/3 files match" in capsys.readouterr().out


def test_rerun_is_idempotent_and_never_overwrites(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    pins = make_tree(src)
    dest = tmp_path / "dest"
    item = {"name": "toolcache", "kind": "tree", "src": str(src), "pins": pins}
    assert run([item], dest, tmp_path, meminfo(50 << 20)) == 0
    first = (dest / "toolcache" / "ADOPTION.json").read_text()
    assert run([item], dest, tmp_path, meminfo(50 << 20)) == 0
    assert (dest / "toolcache" / "ADOPTION.json").read_text() == first
    (dest / "other").mkdir()
    (dest / "other" / "x").write_text("someone else's")
    other = dict(item, name="other")
    assert run([other], dest, tmp_path, meminfo(50 << 20)) == 1
    assert "exists without a PASS adoption record" in capsys.readouterr().out
    assert (dest / "other" / "x").read_text() == "someone else's"


def test_source_pin_mismatch_refuses_before_copy(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    make_tree(src)
    dest = tmp_path / "dest"
    item = {"name": "toolcache", "kind": "tree", "src": str(src), "pins": {"cache/manifest.json": "f" * 64}}
    assert run([item], dest, tmp_path, meminfo(50 << 20)) == 1
    assert "source pin" in capsys.readouterr().out
    assert not (dest / "toolcache").exists() and not (dest / "toolcache.partial").exists()


def test_manifest_reference_mismatch_fails(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    make_tree(src)
    (src / "cache" / "b.bin").write_bytes(b"changed after sealing")
    dest = tmp_path / "dest"
    item = {"name": "toolcache", "kind": "tree", "src": str(src), "manifests": ["cache/manifest.json"],
            "prefix_map": {"/work": "."}}
    assert run([item], dest, tmp_path, meminfo(50 << 20)) == 1
    assert "1 mismatched" in capsys.readouterr().out
    assert not (dest / "toolcache").exists()


def test_symlink_leaving_tree_fails(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    make_tree(src)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    try:
        os.symlink(outside, src / "cache" / "escape")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this host")
    assert run([{"name": "t", "kind": "tree", "src": str(src)}], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "symlink leaves the tree" in capsys.readouterr().out


def test_memory_gate_refuses(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    make_tree(src)
    assert run([{"name": "t", "kind": "tree", "src": str(src)}], tmp_path / "dest", tmp_path, meminfo(1 << 20)) == 1
    assert "MemAvailable 1.0 GiB < gate 40.0 GiB" in capsys.readouterr().out


def test_tar_adopt_pass_with_payload_pin(tmp_path, meminfo, capsys):
    tar_path, idx = make_tar(tmp_path)
    before = sha(tar_path.read_bytes())
    item = {"name": "speech", "kind": "tar", "src": str(tar_path), "index": idx.name,
            "payload_pins": {"codex-x/replay-index.json": sha(b'{"entries": []}')}}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 0
    assert "ADOPT PASS speech: files 2" in capsys.readouterr().out
    assert sha(tar_path.read_bytes()) == before


def test_tar_tampered_member_fails(tmp_path, meminfo, capsys):
    tar_path, idx = make_tar(tmp_path, tamper="codex-x/native/000.safetensors")
    item = {"name": "speech", "kind": "tar", "src": str(tar_path), "index": idx.name}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "differ from the index" in capsys.readouterr().out


def test_tar_traversal_member_refused(tmp_path, meminfo, capsys):
    tar_path, idx = make_tar(tmp_path, evil="../escape.txt")
    item = {"name": "speech", "kind": "tar", "src": str(tar_path), "index": idx.name}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "unsafe tar member" in capsys.readouterr().out
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path / "dest" / "escape.txt").exists()


def test_archive_sha_must_match_index(tmp_path, meminfo, capsys):
    tar_path, idx = make_tar(tmp_path)
    d = json.loads(idx.read_text())
    d["archive_sha256"] = "e" * 64
    idx.write_text(json.dumps(d))
    item = {"name": "speech", "kind": "tar", "src": str(tar_path), "index": idx.name}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "index archive_sha256" in capsys.readouterr().out


def test_keyed_manifest_style(tmp_path, meminfo, capsys):
    src = tmp_path / "src"
    (src / "shard-00000").mkdir(parents=True)
    ids = b"\x01\x00\x00\x00" * 64
    (src / "shard-00000" / "ids.u32").write_bytes(ids)
    shard = {"files": {"ids.u32": {"bytes": len(ids), "sha256": sha(ids)}}, "checks": {"finite": True}}
    (src / "shard-00000" / "shard.json").write_text(json.dumps(shard))
    item = {"name": "q", "kind": "tree", "src": str(src),
            "manifests": [{"rel": "shard-00000/shard.json", "files_key": "files"}]}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 0
    assert "shard-00000/shard.json refs 1 ok 1 external 0" in capsys.readouterr().out
    (src / "shard-00000" / "ids.u32").write_bytes(b"\x02" * 256)
    assert run([dict(item, name="q2")], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "1 mismatched" in capsys.readouterr().out


def write_sums(root: Path, out: Path, star: bool = False) -> None:
    lines = [f"{sha(p.read_bytes())} {'*' if star else ' '}./{p.relative_to(root).as_posix()}"
             for p in sorted(root.rglob("*")) if p.is_file()]
    out.write_text("\n".join(lines) + "\n")


def test_source_sums_tie_staged_copy_to_original(tmp_path, meminfo, capsys):
    original = tmp_path / "original"
    make_tree(original)
    sums = tmp_path / "SOURCE_SHA256SUMS"
    write_sums(original, sums, star=True)
    staged = tmp_path / "staged"
    import shutil
    shutil.copytree(original, staged)
    item = {"name": "rootcopy", "kind": "tree", "src": str(staged), "source_sums": str(sums), "original": str(original)}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 0
    assert "ADOPT PASS rootcopy: files 3" in capsys.readouterr().out
    rec = json.loads((tmp_path / "dest" / "rootcopy" / "ADOPTION.json").read_text())
    assert rec["source_sums"]["files"] == 3 and rec["source_sums"]["original"] == str(original)


def test_source_sums_refuse_drift_and_extra_files(tmp_path, meminfo, capsys):
    original = tmp_path / "original"
    make_tree(original)
    sums = tmp_path / "SOURCE_SHA256SUMS"
    write_sums(original, sums)
    import shutil
    staged = tmp_path / "staged"
    shutil.copytree(original, staged)
    (staged / "cache" / "b.bin").write_bytes(b"changed in staging")
    item = {"name": "rc1", "kind": "tree", "src": str(staged), "source_sums": str(sums)}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "differs from the original's root-side sums at 2 entries" in capsys.readouterr().out
    staged2 = tmp_path / "staged2"
    shutil.copytree(original, staged2)
    (staged2 / "extra.txt").write_text("not in the original")
    assert run([dict(item, name="rc2", src=str(staged2))], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "at 1 entries" in capsys.readouterr().out
    assert not (tmp_path / "dest" / "rc1").exists() and not (tmp_path / "dest" / "rc2").exists()


def test_move_staged_moves_our_copy_and_keeps_original(tmp_path, meminfo, capsys):
    import shutil
    original = tmp_path / "original"
    make_tree(original)
    before = tree_digest(original)
    sums = tmp_path / "SOURCE_SHA256SUMS"
    write_sums(original, sums)
    staged = tmp_path / "staged"
    shutil.copytree(original, staged)
    item = {"name": "big", "kind": "tree", "src": str(staged), "source_sums": str(sums), "original": str(original),
            "move_staged": True, "manifests": ["cache/manifest.json"], "prefix_map": {"/work": "."}}
    assert run([item], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 0
    assert "ADOPT PASS big: files 3" in capsys.readouterr().out
    assert not staged.exists() and tree_digest(original) == before
    rec = json.loads((tmp_path / "dest" / "big" / "ADOPTION.json").read_text())
    assert rec["moved_from_staging"] == str(staged)
    assert A.main(["verify-item", "--dir", str(tmp_path / "dest" / "big")]) == 0


def test_move_staged_refusals_give_the_copy_back(tmp_path, meminfo, capsys):
    import shutil
    original = tmp_path / "original"
    make_tree(original)
    staged = tmp_path / "staged"
    shutil.copytree(original, staged)
    no_sums = {"name": "m1", "kind": "tree", "src": str(staged), "move_staged": True}
    assert run([no_sums], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "move_staged needs source_sums" in capsys.readouterr().out and staged.is_dir()
    sums = tmp_path / "SOURCE_SHA256SUMS"
    write_sums(original, sums)
    inside = dict(no_sums, name="m2", src=str(original), source_sums=str(sums), original=str(original))
    assert run([inside], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "is inside the original" in capsys.readouterr().out and original.is_dir()
    # a manifest reference that fails after the move: the staged copy must be renamed back, intact
    bad = dict(no_sums, name="m3", source_sums=str(sums), original=str(original),
               manifests=["cache/manifest.json"], prefix_map={"/work": "nowhere"})
    before = tree_digest(staged)
    assert run([bad], tmp_path / "dest", tmp_path, meminfo(50 << 20)) == 1
    assert "missing references" in capsys.readouterr().out
    assert staged.is_dir() and tree_digest(staged) == before and not (tmp_path / "dest" / "m3").exists()


def test_read_sums_rejects_garbage_and_duplicates(tmp_path):
    p = tmp_path / "s"
    p.write_text("nothex  a\n")
    with pytest.raises(A.Refusal):
        A.read_sums(p)
    p.write_text(f"{'a' * 64}  ./x\n{'b' * 64}  x\n")
    with pytest.raises(A.Refusal, match="listed twice"):
        A.read_sums(p)


def test_cuda_is_hidden():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
