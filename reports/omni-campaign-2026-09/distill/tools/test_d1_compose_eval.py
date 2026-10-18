"""Tests for ``d1_compose_eval.py`` -- numpy + oc_seal only (no torch, no GPU, no chimera tree).

A two-shard base in the chimera layout (BF16 + F32 tensors, index with chimera tensor_sha256 pins), a delta file and a
sealed delta plan in oc_checkpoint's schema.

  test_compose_is_base_plus_delta     the composed tensor = base with the region overwritten, every other byte equal,
                                      shard headers byte-identical, index pins = chimera digests of the new bytes,
                                      an untouched shard copied, receipt + sidecar written, nothing left in .partial
  test_refusals                       a lora op, a delta file off its plan sha, a base tensor off its pin, an --out
                                      outside the scratch root / already present, a plan root digest that the bytes
                                      do not re-derive (the partial dir is removed every time), a broken seal
  test_split                          the contiguous split with the lowest worst fill; head on card 0; no fit -> None
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
HERE = Path(__file__).resolve().parent
LEDGER = HERE.parents[1] / "ledger" if (HERE.parents[1] / "ledger").is_dir() else Path("/home/cereal/omni-campaign/ledger")
os.environ["OC_LEDGER_ROOT"] = str(LEDGER)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LEDGER))

import d1_compose_eval as ce  # noqa: E402
import oc_seal  # noqa: E402


def write_st(path: Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header, off, blobs = {}, 0, []
    for name in sorted(tensors):
        dtype, arr = tensors[name]
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blobs.append(raw)
    text = json.dumps(header, separators=(",", ":")).encode()
    text += b" " * ((8 - len(text) % 8) % 8)
    path.write_bytes(len(text).to_bytes(8, "little") + text + b"".join(blobs))


def base_dir(tmp: Path) -> tuple[Path, dict[str, tuple[str, np.ndarray]]]:
    rng = np.random.default_rng(0)
    base = tmp / "base"
    base.mkdir()
    s1 = {"blocks.0.ffn.up.weight": ("BF16", rng.integers(0, 2 ** 15, (6, 4), dtype=np.uint16)),
          "blocks.0.norm2.weight": ("F32", rng.standard_normal(4).astype("<f4"))}
    s2 = {"embed.weight": ("BF16", rng.integers(0, 2 ** 15, (5, 4), dtype=np.uint16))}
    write_st(base / "model-00001-of-00002.safetensors", s1)
    write_st(base / "model-00002-of-00002.safetensors", s2)
    allt = {**s1, **s2}
    wm = {n: ("model-00001-of-00002.safetensors" if n in s1 else "model-00002-of-00002.safetensors") for n in allt}
    pins = {n: ce.chimera_tensor_sha256(d, list(a.shape), a.tobytes()) for n, (d, a) in allt.items()}
    (base / ce.INDEX).write_text(json.dumps({"metadata": {"tensor_sha256": pins, "total_size": 1}, "weight_map": wm}))
    (base / "config.json").write_text("{}")
    return base, allt


def plan_for(tmp: Path, base: Path, entries: list[dict], delta: dict[str, tuple[str, np.ndarray]], *, root_sha: str | None = None) -> Path:
    ddir = tmp / "delta"
    ddir.mkdir(exist_ok=True)
    write_st(ddir / "D1_delta.safetensors", delta)
    digest, files = oc_seal.directory_digest(base)
    plan = oc_seal.seal({"schema": ce.PLAN_SCHEMA, "checkpoint_id": "T-D1", "deep": True,
                         "root": {"checkpoint_id": "T-BASE", "record_sha": "0" * 64, "directory": str(base),
                                  "directory_sha256": root_sha or digest,
                                  "files": [f for f in files if f["path"].endswith(".safetensors")]},
                         "deltas": [{"checkpoint_id": "T-D1", "directory": str(ddir), "entries": entries,
                                     "files": [{"path": "D1_delta.safetensors", "bytes": 1,
                                                "sha256": oc_seal.file_sha256(ddir / "D1_delta.safetensors")}]}]})
    path = tmp / "plan.json"
    path.write_text(json.dumps(plan))
    return path


REGION = [[0, 1, 3], [1, 0, 4]]


def test_compose_is_base_plus_delta(tmp_path: Path) -> None:
    base, allt = base_dir(tmp_path)
    d = np.full((2, 4), 0x3F80, dtype=np.uint16)                       # bf16 1.0
    plan = plan_for(tmp_path, base, [{"op": "replace", "target": "blocks.0.ffn.up.weight", "tensor": "w", "slice": REGION}],
                    {"w": ("BF16", d)})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    out = scratch / "composed"
    assert ce.main(["compose", "--plan", str(plan), "--out", str(out), "--scratch-root", str(scratch)]) == 0
    assert not (scratch / "composed.partial").exists()
    h0, raw0, b0 = ce.read_header(base / "model-00001-of-00002.safetensors")
    h1, raw1, b1 = ce.read_header(out / "model-00001-of-00002.safetensors")
    assert raw0 == raw1 and h0 == h1
    data = (out / "model-00001-of-00002.safetensors").read_bytes()
    lo, hi = h1["blocks.0.ffn.up.weight"]["data_offsets"]
    got = np.frombuffer(data[b1 + lo: b1 + hi], dtype="<u2").reshape(6, 4)
    want = allt["blocks.0.ffn.up.weight"][1].copy()
    want[1:3] = d
    assert np.array_equal(got, want)
    lo, hi = h1["blocks.0.norm2.weight"]["data_offsets"]
    assert data[b1 + lo: b1 + hi] == allt["blocks.0.norm2.weight"][1].tobytes()
    assert (out / "model-00002-of-00002.safetensors").read_bytes() == (base / "model-00002-of-00002.safetensors").read_bytes()
    pins = json.loads((out / ce.INDEX).read_text())["metadata"]["tensor_sha256"]
    assert pins["blocks.0.ffn.up.weight"] == ce.chimera_tensor_sha256("BF16", [6, 4], want.tobytes())
    assert pins["embed.weight"] == json.loads((base / ce.INDEX).read_text())["metadata"]["tensor_sha256"]["embed.weight"]
    receipt = json.loads((scratch / "composed.compose.json").read_text())
    assert receipt["tensors_composed"] == 1 and receipt["replaced_elements"] == 8 and receipt["readback_verified"]
    assert receipt["out_directory_sha256"] == oc_seal.directory_digest(out)[0]
    assert (scratch / "composed.compose.json.sha256").read_text().split()[0] == ce.sha256_file(scratch / "composed.compose.json")


def test_refusals(tmp_path: Path) -> None:
    base, allt = base_dir(tmp_path)
    d = np.full((2, 4), 0x3F80, dtype=np.uint16)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rep = {"op": "replace", "target": "blocks.0.ffn.up.weight", "tensor": "w", "slice": REGION}

    def refuse(plan: Path, match: str, out: Path | None = None) -> None:
        out = out or scratch / "c"
        with pytest.raises(ce.Refusal, match=match):
            ce.compose(plan, out, scratch)
        assert not out.with_name(out.name + ".partial").exists()

    refuse(plan_for(tmp_path, base, [{"op": "lora", "target": "blocks.0.ffn.up.weight", "a": "w", "b": "w"}], {"w": ("BF16", d)}), "only `replace`")
    plan = plan_for(tmp_path, base, [rep], {"w": ("BF16", d)})
    refuse(plan, "not under the declared scratch", out=tmp_path / "outside")
    (scratch / "exists").mkdir()
    refuse(plan, "exists", out=scratch / "exists")
    refuse(plan_for(tmp_path, base, [rep], {"w": ("F32", d.astype("<f4"))}), "is F32")
    refuse(plan_for(tmp_path, base, [rep], {"w": ("BF16", d)}, root_sha="a" * 64), "re-derives")
    plan = plan_for(tmp_path, base, [rep], {"w": ("BF16", d)})
    (tmp_path / "delta" / "D1_delta.safetensors").write_bytes(b"x" * 64)
    refuse(plan, "hashes")
    doc = json.loads(plan.read_text())
    doc["checkpoint_id"] = "edited"
    plan.write_text(json.dumps(doc))
    refuse(plan, "seal")
    idx = json.loads((base / ce.INDEX).read_text())
    idx["metadata"]["tensor_sha256"]["blocks.0.ffn.up.weight"] = "f" * 64
    (base / ce.INDEX).write_text(json.dumps(idx))
    refuse(plan_for(tmp_path, base, [rep], {"w": ("BF16", d)}), "index pin")


def test_split(tmp_path: Path) -> None:
    gib = 1 << 30
    blocks = [gib] * 10
    best = ce.best_split(blocks, 4 * gib, [10.0, 8.0, 8.0], head_extra=1.0, block_extra=0.0)
    assert sum(best["split"]) == 10 and best["worst_fill"] <= 1.0
    assert best["loads_gib"][0] == best["split"][0] + 5.0
    assert ce.best_split(blocks, 4 * gib, [5.0, 5.0], head_extra=0.0, block_extra=0.0) is None
    base, _ = base_dir(tmp_path)
    b, head = ce.block_bytes(base)
    assert b == [6 * 4 * 2 + 4 * 4] and head == 5 * 4 * 2
