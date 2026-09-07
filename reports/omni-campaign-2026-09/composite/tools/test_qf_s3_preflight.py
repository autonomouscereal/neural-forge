"""CPU-only S3 launch/configuration regressions, never real checkpoint loads."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qf_stream27 as qs


def checkpoint(path, config, *, mtp=True, sharded=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    tensors = {"model.embed_tokens.weight": torch.zeros(2, 3)}
    if mtp:
        tensors["mtp.a.weight"] = torch.zeros(4, 3)
        tensors["mtp.b.weight"] = torch.zeros(2)
    if sharded:
        save_file(tensors, str(path / "part.safetensors"))
        (path / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "part.safetensors" for name in tensors}}))
    else:
        save_file(tensors, str(path / "model.safetensors"))
    return path


@pytest.mark.parametrize("nested", [True, False])
@pytest.mark.parametrize("sharded", [True, False])
def test_mtp_drop_canonical_config_and_exact_no_credit_accounting(tmp_path, nested, sharded, monkeypatch):
    text = {"mtp_num_hidden_layers": 1, "intermediate_size": 4352}
    config = {"model_type": "qwen3_5", "text_config": text} if nested else {
        "model_type": "qwen3_5_text", **text}
    parent = checkpoint(tmp_path / "parent", config, sharded=sharded)
    child = checkpoint(tmp_path / "child", config, mtp=False, sharded=sharded)
    before = (parent / "config.json").read_bytes()
    monkeypatch.setattr(qs.SingleOrShardedReader, "get", lambda *a: pytest.fail("materialized a weight"))
    report = qs.drop_exported_mtp(parent, child)
    loaded = json.loads((child / "config.json").read_text())
    canonical = loaded["text_config"] if nested else loaded
    assert canonical["mtp_num_hidden_layers"] == 0
    assert canonical["intermediate_size"] == 4352
    assert (parent / "config.json").read_bytes() == before
    assert report["previous_config_value"] == 1
    assert report["dropped_parameters"] == 14
    assert len(report["dropped_tensors"]) == 2
    assert report["credited_to_fold_ratio"] is False
    if nested:
        assert "mtp_num_hidden_layers" not in loaded
        assert report["config_field"] == "text_config.mtp_num_hidden_layers"


def test_mtp_drop_refuses_existing_child_tensors_without_config_write(tmp_path):
    config = {"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1}
    parent = checkpoint(tmp_path / "parent", config)
    child = checkpoint(tmp_path / "child", config)
    before = (child / "config.json").read_bytes()
    with pytest.raises(SystemExit, match="still contains MTP"):
        qs.drop_exported_mtp(parent, child)
    assert (child / "config.json").read_bytes() == before


def test_mtp_absent_parent_receipt_does_not_claim_dropped_parameters(tmp_path):
    config = {"model_type": "qwen3_5", "text_config": {"mtp_num_hidden_layers": 1},
              "mtp_num_hidden_layers": 1}
    parent = checkpoint(tmp_path / "parent", config, mtp=False)
    child = checkpoint(tmp_path / "child", config, mtp=False)
    report = qs.drop_exported_mtp(parent, child)
    loaded = json.loads((child / "config.json").read_text())
    assert report["dropped_parameters"] == 0
    assert report["mtp"] == "not present in the parent checkpoint"
    assert loaded["mtp_num_hidden_layers"] == loaded["text_config"]["mtp_num_hidden_layers"] == 0


@pytest.mark.parametrize("nested", [None, [], 1])
def test_mtp_drop_refuses_malformed_vl_config_before_writing(tmp_path, nested):
    config = {"model_type": "qwen3_5", "text_config": nested}
    parent = checkpoint(tmp_path / "parent", config)
    child = checkpoint(tmp_path / "child", config, mtp=False)
    before = (child / "config.json").read_bytes()
    with pytest.raises(SystemExit, match="canonical text_config"):
        qs.drop_exported_mtp(parent, child)
    assert (child / "config.json").read_bytes() == before
