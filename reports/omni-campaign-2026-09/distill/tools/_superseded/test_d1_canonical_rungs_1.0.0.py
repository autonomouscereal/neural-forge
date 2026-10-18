"""Tests for ``d1_canonical_rungs.py`` -- pure CPU / filesystem, no torch.

  test_sync_two_attempts_then_verify     v35-style receipt (L00-L02 trained) then a c36-style receipt (L00-L02
                                         imported, L03-L04 trained): contiguous L00..L04, read-only copies,
                                         SHA256SUMS re-hashes, PROVENANCE records both syncs; a re-sync is a no-op
  test_refusals                          receipt sha != file, rung json disagreeing, a gap, an imported record pin
                                         that is not the canonical json, a tampered canonical file, a stale sidecar
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d1_canonical_rungs as cr  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_attempt(root: Path, trained: list[int], imported: dict[int, str] | None = None, tag: str = "a") -> Path:
    """An attempt dir: rungs/L{i}.safetensors + .json for the trained layers, d1_receipt.json (+ sidecar)."""
    rungs = root / "rungs"
    rungs.mkdir(parents=True)
    layers = []
    for i, pin in sorted((imported or {}).items()):
        layers.append({"text_layer": i, "trained": True, "resumed_from": f"/ancestor/L{i:02d}.safetensors",
                       "import_provenance": {"old_record_pin": {"bytes": 1, "sha256": pin}}})
    for i in trained:
        payload = f"{tag}-rung-{i}".encode() * 100
        (rungs / f"L{i:02d}.safetensors").write_bytes(payload)
        doc = {"text_layer": i, "file_sha256": sha(payload), "input_chain": f"chain-{tag}-{i}", "tool": {"version": tag}}
        (rungs / f"L{i:02d}.json").write_text(json.dumps(doc))
        layers.append({"text_layer": i, "trained": True, "rung": {"file": f"/runout/rungs/L{i:02d}.safetensors",
                                                                  "sha256": sha(payload)}})
    raw = json.dumps({"layers": layers}).encode()
    (root / "d1_receipt.json").write_bytes(raw)
    (root / "d1_receipt.json.sha256").write_text(sha(raw) + "\n")
    return root


def test_sync_two_attempts_then_verify(tmp_path: Path) -> None:
    canon = tmp_path / "canon"
    v35 = make_attempt(tmp_path / "v35", [0, 1, 2], tag="v35")
    res = cr.sync(canon, v35 / "d1_receipt.json", v35 / "rungs", "v35")
    assert res["layers"] == 3 and res["added"] == [0, 1, 2]
    pins = {i: sha((canon / f"L{i:02d}.json").read_bytes()) for i in range(3)}
    c36 = make_attempt(tmp_path / "c36", [3, 4], imported=pins, tag="c36")
    res = cr.sync(canon, c36 / "d1_receipt.json", c36 / "rungs", "c36")
    assert res["layers"] == 5 and res["added"] == [3, 4] and res["lineage_checked"] == [0, 1, 2]
    for p in canon.glob("L*"):
        assert not os.access(p, os.W_OK) or os.geteuid() == 0, f"{p} is writable"
    v = cr.verify(canon, expect_layers=5)
    assert v["layers"] == 5 and v["sha256sums_sha256"] == res["sha256sums_sha256"]
    assert (canon / cr.SUMS).read_text().count("\n") == 10
    prov = json.loads((canon / cr.PROVENANCE).read_text())
    assert sorted(prov["layers"]) == [f"L{i:02d}" for i in range(5)] and len(prov["syncs"]) == 2
    assert prov["layers"]["L03"]["label"] == "c36"
    again = cr.sync(canon, c36 / "d1_receipt.json", c36 / "rungs", "c36 again")
    assert again["added"] == [] and again["already_equal"] == [3, 4] and again["sha256sums_sha256"] == res["sha256sums_sha256"]


def test_refusals(tmp_path: Path) -> None:
    canon = tmp_path / "canon"
    bad = make_attempt(tmp_path / "bad", [0])
    (bad / "rungs" / "L00.safetensors").write_bytes(b"tampered")
    doc = json.loads((bad / "rungs" / "L00.json").read_text())
    with pytest.raises(SystemExit, match="hashes"):
        cr.sync(canon, bad / "d1_receipt.json", bad / "rungs")
    doc["file_sha256"] = "0" * 64
    (bad / "rungs" / "L00.json").write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="file_sha256"):
        cr.sync(canon, bad / "d1_receipt.json", bad / "rungs")
    gap = make_attempt(tmp_path / "gap", [1])
    with pytest.raises(SystemExit, match="gap"):
        cr.sync(tmp_path / "canon_gap", gap / "d1_receipt.json", gap / "rungs")
    good = make_attempt(tmp_path / "good", [0, 1])
    cr.sync(canon, good / "d1_receipt.json", good / "rungs")
    wrong = make_attempt(tmp_path / "wrong", [2], imported={0: "f" * 64, 1: "f" * 64})
    with pytest.raises(SystemExit, match="imported a rung record"):
        cr.sync(canon, wrong / "d1_receipt.json", wrong / "rungs")
    other = make_attempt(tmp_path / "other", [0], tag="other")      # a different L00 than the canonical one
    with pytest.raises(SystemExit, match="canonical"):
        cr.sync(canon, other / "d1_receipt.json", other / "rungs")
    target = canon / "L01.safetensors"
    os.chmod(target, 0o644)
    target.write_bytes(b"changed")
    with pytest.raises(SystemExit, match="hashes"):
        cr.verify(canon)
    stale = make_attempt(tmp_path / "stale", [0])
    (stale / "d1_receipt.json.sha256").write_text("0" * 64 + "\n")
    with pytest.raises(SystemExit, match="sidecar"):
        cr.read_receipt(stale / "d1_receipt.json", tries=2)
