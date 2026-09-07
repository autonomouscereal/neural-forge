"""CPU-only S3 launch/configuration regressions, never real checkpoint loads."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qf_stream27 as qs
import preflight_s3_tracked as pf


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
    assert qs.qf_arch.text_config(qs.qf_arch.load_config(child)).mtp_num_hidden_layers == 0
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


def test_header_inventory_counts_but_never_claims_full_digest(tmp_path):
    parent = checkpoint(tmp_path / "parent", {}, sharded=True)
    result = pf.header_inventory(parent)
    assert result["stock_model_parameters"] == 6
    assert result["mtp_parameters_excluded_from_fold_ratio"] == 14
    assert result["indexed_tensors"] == 3
    assert "NOT a full weight digest" in result["method"]
    assert "sha256" not in result["files"][0]
    assert len(result["files"][0]["header_sha256"]) == 64


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:\\escape", "nested/file", "..", "bad:name"])
def test_header_inventory_refuses_traversal(tmp_path, name):
    parent = checkpoint(tmp_path / "parent", {}, sharded=True)
    (parent / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": name}}))
    with pytest.raises(ValueError, match="unsafe"):
        pf.header_inventory(parent)


def sealed_evidence(parent, out, *, duplicate=False):
    import hashlib
    raw = b'{"fixture": "tokenizer only"}'
    (parent / "tokenizer.json").write_bytes(raw)
    body = {"schema": "neural-forge-omni-evidence-v1", "sequence_length": 2,
            "tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
            "rows": [{"batch_index": 0 if duplicate else i, "split": "held" if i == 3 else "fit", "ids": [1, 2]} for i in range(4)]}
    digest = hashlib.sha256(pf.canonical(body)).hexdigest()
    out.write_text(json.dumps({**body, "manifest_sha256": digest}))
    return digest


def test_evidence_validates_tokenizer_seal_ids_and_partitions(tmp_path):
    parent = checkpoint(tmp_path / "parent", {"vocab_size": 3}, sharded=True)
    evidence = tmp_path / "evidence.json"
    digest = sealed_evidence(parent, evidence)
    report = pf.inspect_evidence(parent, evidence, expected=digest)
    assert (report["fit_rows"], report["held_rows"], report["internal_check_rows"]) == (6, 2, 2)
    with pytest.raises(ValueError, match="pinned evidence"):
        pf.inspect_evidence(parent, evidence)
    (parent / "tokenizer.json").write_text("different tokenizer")
    with pytest.raises(ValueError, match="tokenizer mismatch"):
        pf.inspect_evidence(parent, evidence, expected=digest)


def test_evidence_rejects_sealed_duplicate_batch_ids(tmp_path):
    parent = checkpoint(tmp_path / "parent", {"vocab_size": 3}, sharded=True)
    evidence = tmp_path / "evidence.json"
    digest = sealed_evidence(parent, evidence, duplicate=True)
    with pytest.raises(ValueError, match="batch identity"):
        pf.inspect_evidence(parent, evidence, expected=digest)


def test_memory_accounting_c2_only_and_fp32_export_peak():
    config = {"hidden_size": 5120, "intermediate_size": 17408, "num_hidden_layers": 64,
              "vocab_size": 248320, "num_attention_heads": 24}
    report = pf.memory_components(config, {"batches": 24, "sequence_length": 4096}, 27356728560)
    assert report["parameters_removed_by_C2"] == 12834570240
    assert report["component_gib"]["child_fp32_reload_weights"] == 54.099
    assert report["component_gib"]["gpu_bf16_untied_embedding_and_head"] == 4.736


def test_existing_child_refused_before_input_reads(tmp_path):
    child = tmp_path / "existing"
    child.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        pf.main(["--parent", "missing", "--evidence", "missing", "--out", str(child)])
