"""Tests for d1_delta_recipe.py.  CPU only, a few seconds.

The D1 delta is written by d1_trunk_recovery's OWN writer (`write_delta`, on a minimal run state), the recipe is
checked by the ledger's OWN validator (`oc_checkpoint.validate_delta_entries` against both directories' headers),
and the composition is done the way the P1 reader composes a plan (`p1_qwen35_reader._region_index`) and compared
with d1's reference reader `apply_delta`, bit for bit.

    cd $OC/distill/tools && CUDA_VISIBLE_DEVICES= PYTHONPATH=$OC/distill/tools:$OC/ledger \\
        $PY -m pytest -o addopts="" -p no:cacheprovider test_d1_delta_recipe.py -q
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
LEDGER = HERE.parent.parent / "ledger"
sys.path[:0] = [str(HERE), str(LEDGER)]

import d1_delta_recipe as R  # noqa: E402
import d1_trunk_recovery as d1  # noqa: E402
import oc_checkpoint  # noqa: E402
import p1_qwen35_reader as p1  # noqa: E402

H, W, Q, K3 = 16, 24, 8, 16          # hidden, union FFN width, trunk channels, K3 channels


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def base(tmp_path: Path) -> Path:
    g = torch.Generator().manual_seed(0)
    root = tmp_path / "OMNI1.clone-pre-D1"
    root.mkdir()
    tensors = {"embed.weight": torch.randn(32, H, generator=g).bfloat16()}
    for j in range(3):
        tensors[f"blocks.{j}.ffn.up.weight"] = torch.randn(2 * W, H, generator=g).bfloat16()
        tensors[f"blocks.{j}.ffn.down.weight"] = torch.randn(H, W, generator=g).bfloat16()
        tensors[f"blocks.{j}.norm2.weight"] = torch.randn(H, generator=g).bfloat16()
    save_file(tensors, str(root / "model.safetensors"))
    (root / "config.json").write_text(json.dumps({"schema": "test", "hidden": H}))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "model.safetensors" for n in tensors}}))
    return root


def write_d1_delta(base: Path, out: Path, layers: dict[int, int]) -> Path:
    """d1_trunk_recovery.write_delta on a run state with `layers` {text layer: block} trained.

    2026-10-14 (lane d1seal): d1_trunk_recovery >= 1.3.0 keeps a trained layer ON DISK (`rs.trained[i]` =
    trained_entry(block, base shas, rung file, sha), re-read by trained_tensors), so each layer is first written as a
    rung with the trainer's own write_rung; the 1.2.x fixture handed it in-memory tensors and failed with
    KeyError 'file' against 1.3.x."""
    g = torch.Generator().manual_seed(1)
    trained = {}
    state = load_file(str(base / "model.safetensors"))
    rs = SimpleNamespace(trained=trained, layout=d1.UnionLayout(W, Q, K3, 0), child=SimpleNamespace(hidden=H), chain="test",
                         args=SimpleNamespace(delta_out=str(out), out=str(out.parent / "run"), min_free_gib=0.0))
    for i, j in layers.items():
        stored = {"gate": torch.randn(Q, H, generator=g).bfloat16(), "up": torch.randn(Q, H, generator=g).bfloat16(),
                  "down": torch.randn(H, Q, generator=g).bfloat16(), "norm": torch.randn(H, generator=g).bfloat16()}
        shas = {f"blocks.{j}.{n}": d1.tensor_bytes_sha256(state[f"blocks.{j}.{n}"])
                for n in ("ffn.up.weight", "ffn.down.weight", "norm2.weight")}
        if hasattr(d1, "trained_entry"):
            doc = d1.write_rung(rs, i, j, stored, {}, shas)
            trained[i] = d1.trained_entry(j, shas, doc["file"], doc["file_sha256"])
        else:                                            # d1_trunk_recovery < 1.3.0
            trained[i] = {"block": j, "stored": stored, "base_shas": shas}
    identity = {"path": str(base), "config_sha256": sha(base / "config.json"),
                "index_sha256": sha(base / "model.safetensors.index.json"), "chimera_commit": None,
                "files": {"model.safetensors": {"sha256": sha(base / "model.safetensors")}}, "mode": "full"}
    d1.write_delta(rs, identity)
    return out


def base_record(base: Path, tmp_path: Path, checkpoint_id: str = "OC02-F-Q38-27BU-OMNI1.clone-pre-D1") -> Path:
    path = tmp_path / "base_record.json"
    path.write_text(json.dumps({"checkpoint_id": checkpoint_id,
                                "checkpoint": {"directory": str(base), "directory_sha256": "ab" * 32}}))
    return path


def test_recipe_validates_in_the_ledger_and_composes_like_d1(base, tmp_path):
    delta = write_d1_delta(base, tmp_path / "delta", {0: 0, 2: 2})
    assert R.main(["--delta-dir", str(delta), "--base-record", str(base_record(base, tmp_path))]) == 0
    recipe = json.loads((delta / "DELTA.json").read_text())
    assert recipe["schema"] == oc_checkpoint.DELTA_RECIPE_SCHEMA and R.RECIPE_NAME == oc_checkpoint.DELTA_RECIPE_NAME
    assert {e["op"] for e in recipe["entries"]} == {"replace"} and len(recipe["entries"]) == 8
    accounting = oc_checkpoint.validate_delta_entries(
        recipe, base_index=oc_checkpoint._tensor_index(base), delta_index=oc_checkpoint._tensor_index(delta), where="test")
    assert accounting is not None
    manifest = json.loads((delta / "D1_delta.json").read_text())
    values = load_file(str(delta / "D1_delta.safetensors"))
    state = load_file(str(base / "model.safetensors"))
    by_key = {e["key"]: e for e in manifest["entries"]}
    composed = {n: t.clone() for n, t in state.items()}
    for entry in recipe["entries"]:      # the P1 reader's region rule, entry by entry, in recipe order
        target = composed[entry["target"]]
        target[p1._region_index(entry["slice"], target.dim())] = values[entry["tensor"]]
    expected = {n: t.clone() for n, t in state.items()}
    for key, entry in by_key.items():    # d1's own reference reader
        expected[entry["base_tensor"]] = d1.apply_delta(expected[entry["base_tensor"]], entry, values[key])
    for name in state:
        assert torch.equal(composed[name], expected[name]), name
    assert torch.equal(composed["blocks.1.ffn.up.weight"], state["blocks.1.ffn.up.weight"])   # untrained block
    # the K3 rows / columns are untouched: gate rows [Q, W), up rows [W+Q, 2W), down columns [Q, W)
    for j in (0, 2):
        assert torch.equal(composed[f"blocks.{j}.ffn.up.weight"][Q:W], state[f"blocks.{j}.ffn.up.weight"][Q:W])
        assert torch.equal(composed[f"blocks.{j}.ffn.up.weight"][W + Q:], state[f"blocks.{j}.ffn.up.weight"][W + Q:])
        assert torch.equal(composed[f"blocks.{j}.ffn.down.weight"][:, Q:], state[f"blocks.{j}.ffn.down.weight"][:, Q:])
    assert recipe["source"]["d1_file_sha256"] == manifest["file_sha256"]


def test_refusals(base, tmp_path):
    delta = write_d1_delta(base, tmp_path / "delta", {1: 1})
    record = base_record(base, tmp_path)
    # another base directory
    other = tmp_path / "other"
    other.mkdir()
    (tmp_path / "rec_other.json").write_text(json.dumps({"checkpoint_id": "X", "checkpoint": {
        "directory": str(other), "directory_sha256": "cd" * 32}}))
    assert R.main(["--delta-dir", str(delta), "--base-record", str(tmp_path / "rec_other.json")]) == 2
    # the wrong id asked for
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record), "--base-checkpoint-id", "OC02-F-X"]) == 2
    # the base's config changed after D1 read it
    cfg = (base / "config.json").read_text()
    (base / "config.json").write_text(cfg + " ")
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record)]) == 2
    (base / "config.json").write_text(cfg)
    # a delta file edited after d1 wrote it
    f = delta / "D1_delta.safetensors"
    raw = f.read_bytes()
    f.write_bytes(raw[:-2] + bytes([raw[-2] ^ 1]) + raw[-1:])
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record)]) == 2
    f.write_bytes(raw)
    # the manifest edited (its sidecar no longer matches)
    m = delta / "D1_delta.json"
    m.write_text(m.read_text().replace('"bytes"', '"bytes" ', 1))
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record)]) == 2
    assert not (delta / "DELTA.json").exists()


def test_existing_recipe_is_kept_unless_forced(base, tmp_path):
    delta = write_d1_delta(base, tmp_path / "delta", {0: 0})
    record = base_record(base, tmp_path)
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record)]) == 0
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record)]) == 2
    assert R.main(["--delta-dir", str(delta), "--base-record", str(record), "--force"]) == 0
