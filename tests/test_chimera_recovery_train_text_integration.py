"""Real CPU command/engine/Adam/checkpoint integration, using synthetic rows.

Requires the exact audited original in CHIMERA_STREAMED_BASELINE and its real
qtext_teacher_cache.py sibling (or QTEXT_TEACHER_CACHE_PATH). This is a tiny
2026 integration fixture, not campaign training or an intelligence result.
Only the preregistration peer is replaced with a fixture-specific sealed pin;
the candidate command, source/plugin reader, cache reader, layerwise engine,
assistant mask, optimizer, safety serializer and resume checks are production.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import pytest


PLUGIN = '''import hashlib, json, math
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm

class TinySource:
    text_prefix = "model.language_model"
    def __init__(self):
        self.bytes_read = 0
        config = Qwen3_5TextConfig(vocab_size=32, hidden_size=16, intermediate_size=24,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=32, layer_types=["full_attention"]*2,
            linear_key_head_dim=8, linear_value_head_dim=8,
            linear_num_key_heads=1, linear_num_value_heads=2,
            rope_parameters={"rope_type":"default", "rope_theta":10000., "mrope_section":[1,1,2]})
        config._attn_implementation = "sdpa"
        self.config = config.to_dict()
        self.weights = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261013)
            self.weights[self.text_prefix+".embed_tokens.weight"] = torch.randn(32,16)/4
            self.weights["lm_head.weight"] = torch.randn(32,16)/4
            norm = Qwen3_5RMSNorm(16, eps=config.rms_norm_eps)
            self.weights[self.text_prefix+".norm.weight"] = norm.weight.detach().clone()
            for i in range(2):
                layer = Qwen3_5DecoderLayer(config, i).eval().requires_grad_(False)
                for name, value in layer.state_dict().items():
                    self.weights[f"{self.text_prefix}.layers.{i}.{name}"] = value.detach().clone()
        h = hashlib.sha256()
        for name, value in sorted(self.weights.items()):
            h.update(name.encode()); h.update(value.numpy().tobytes())
        self.weight_sha256 = h.hexdigest()
    def get(self, name):
        value = self.weights[name]
        self.bytes_read += value.numel()*value.element_size()
        return value
    def has(self, name): return name in self.weights
    def names_under(self, prefix): return sorted(n for n in self.weights if n.startswith(prefix))
    def config_dict(self): return self.config
    def identity(self): return {"kind":"synthetic_cpu_2026_fixture", "weights_sha256":self.weight_sha256, "config":self.config}
def factory(): return TinySource()
'''


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def production_fixture(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from safetensors.torch import save_file
    from experiments.chimera_recovery import train_text as T

    raw = os.environ.get("CHIMERA_STREAMED_BASELINE")
    if not raw:
        pytest.skip("exact audited streamed baseline required")
    baseline = Path(raw)
    peer = Path(os.environ.get("QTEXT_TEACHER_CACHE_PATH", str(baseline.parent / "qtext_teacher_cache.py")))
    assert peer.is_file(), "actual qtext cache peer required"
    monkeypatch.setenv("QTEXT_TEACHER_CACHE_PATH", str(peer))
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.setenv(name, "1" if name == "WORLD_SIZE" else "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    candidate, _ = T.patch_trainer(baseline)
    candidate_path = tmp_path / "tools" / "streamed_candidate.py"
    candidate_path.parent.mkdir()
    candidate_path.write_bytes(candidate.encode())
    module = load_module("chimera_recovery_cpu_integration_candidate", candidate_path)

    # The real preregistration function still invokes and pins its peer. This
    # peer is explicitly scoped to our synthetic test document, not OMNI1.
    amendment = tmp_path / "amendment.md"
    amendment.write_bytes(b"2026-10-13 synthetic CPU integration; no campaign promotion.\n")
    prereg = tmp_path / "prereg.md"
    prereg.write_bytes(b"# Synthetic CPU test only\n## Amendments\n" + amendment.read_bytes())
    receipt_peer = tmp_path / "fixture_preregistration_peer.py"
    receipt_peer.write_text(
        "import hashlib\nTOOL_VERSION='synthetic-cpu-2026-fixture'\n"
        "class Stage2TrainingReceiptError(RuntimeError): pass\n"
        "def assert_preregistration_unedited(path):\n"
        f"    if hashlib.sha256(path.read_bytes()).hexdigest() != {sha(prereg)!r}:\n"
        "        raise Stage2TrainingReceiptError('fixture preregistration changed')\n"
        "    return {'fixture_only':True}\n", encoding="utf-8")
    monkeypatch.setenv("STAGE2_TRAINING_RECEIPT_PATH", str(receipt_peer))
    plugin = tmp_path / "tiny_source.py"
    plugin.write_text(PLUGIN, encoding="utf-8")

    lengths = [4, 7, 3, 5, 6, 4, 8]
    width = 12
    ids = torch.zeros((len(lengths), width), dtype=torch.int64)
    masks = torch.zeros_like(ids, dtype=torch.bool)
    for row, length in enumerate(lengths):
        ids[row, :length] = (torch.arange(length) + row * 3 + 1) % 30 + 1
        masks[row, length - min(3, length - 1):length] = True
    corpus = tmp_path / "synthetic_corpus.safetensors"
    save_file({"input_ids": ids, "real_lengths": torch.tensor(lengths), "assistant_mask": masks}, str(corpus))
    cache = tmp_path / "synthetic_qtext_cache"
    shard = cache / "shard-00000"
    cache.mkdir()
    row_table, positions, cached_ids, top_ids, offset = [], [], [], [], 0
    for row, length in enumerate(lengths):
        count = length - 1
        row_table.append({"row": row, "length": length, "offset": offset, "count": count})
        positions.extend(range(count)); cached_ids.extend(ids[row, :count].tolist())
        top_ids.extend([[int(ids[row, pos + 1]), (int(ids[row, pos + 1]) + 1) % 32] for pos in range(count)])
        offset += count
    q = module.load_peer_module("qtext_teacher_cache")
    # Use the real shard writer/checker, including all five raw-file seals.
    shard_record = q.write_shard(cache, 0, {"ids": torch.tensor(cached_ids),
        "pos": torch.tensor(positions), "topk_ids": torch.tensor(top_ids),
        "topk_logprobs": torch.tensor([[-.5, -2.]] * offset, dtype=torch.bfloat16),
        "lse": torch.zeros(offset)}, {"row_table": row_table}, 32, 2)
    assert q.verify_shard(shard)["ok"]
    write_json(cache / "manifest.json", {"schema": q.MANIFEST_SCHEMA, "fixture_only": True,
        "all_finite": True, "all_structural_ok": True,
        "teacher": {"equal_before_and_after_every_session": True, "kind": "synthetic_fixture_targets_no_teacher_run"},
        "cache_config": {"top_k": 2, "vocab_rows": 32},
        "corpus": {"kind": "text", "path": str(corpus), "sha256": sha(corpus)},
        "shards": [{"shard": "shard-00000", "shard_json_sha256": sha(shard / "shard.json"),
                    "files": shard_record["files"]} ]})
    data = tmp_path / "data.json"
    write_json(data, {"sources": [{"name": "fixture", "kind": "text", "supervision": "assistant_only", "qtext_cache": str(cache), "corpus": str(corpus)}]})
    base_args = ["train", "--structured-recovery", "--source-plugin", str(plugin) + ":factory",
        "--devices", "cpu", "--dtype", "float32", "--kernels", "torch", "--no-pin", "--no-prefetch",
        "--text-vocab-rows", "32", "--seq-len", str(width), "--micro-batch", "2", "--accumulation", "2",
        "--lora-rank", "2", "--k3-lora-rank", "1", "--trunk-ffn-width", "16",
        "--w-q38", "1", "--w-ce", ".1", "--w-eos", ".1", "--w-ul", ".1", "--eos-ids", "2",
        "--lr", ".002", "--gain-lr", ".003", "--warmup-steps", "0", "--deterministic",
        "--preregistration", str(prereg), "--recovery-amendment", str(amendment), "--data-config", str(data),
        "--safety-every-steps", "1", "--safety-keep", "3"]
    def run(label, extra=()):
        out, safety = tmp_path / label, tmp_path / (label + "-safety")
        code = module.main([*base_args, "--out", str(out), "--safety-dir", str(safety), *extra])
        return code, out, safety
    yield module, run, lengths, masks, tmp_path, data
    torch.set_num_threads(old_threads)


def assert_nested_equal(torch, a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_nested_equal(torch, a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(torch, left, right)
    else:
        assert a == b


def test_actual_command_optimizer_exact_resume_and_resealed_bad_cursor(production_fixture):
    import torch
    from safetensors.torch import load_file
    from experiments.chimera_recovery.training import ReadinessError
    module, run, lengths, masks, root, _ = production_fixture
    code, full, full_safety = run("full")
    assert code == 0
    code, first, split_safety = run("split", ["--max-steps-this-run", "1"])
    assert code == 0
    first_receipt = json.loads((first / "training-receipt.json").read_text())
    assert first_receipt["verdict"] == "segment_done" and first_receipt["step"] == 1
    checkpoint = split_safety / "ckpt-00000001"
    code, resumed, _ = run("split", ["--resume", str(checkpoint)])
    assert code == 0
    full_receipt = json.loads((full / "training-receipt.json").read_text())
    resumed_receipt = json.loads((resumed / "training-receipt.json").read_text())
    assert full_receipt["verdict"] == resumed_receipt["verdict"] == "completed"
    assert full_receipt["step"] == resumed_receipt["step"] == 2
    assert full_receipt["tokens_seen"] == resumed_receipt["tokens_seen"] == sum(lengths)
    progress = full_receipt["cumulative_applied_progress"]
    assert progress == resumed_receipt["cumulative_applied_progress"]
    assert progress["rows"] == 7 and progress["supervised_positions"] == int(masks.sum())
    assert full_receipt["segment_counts"]["compute_tokens"] < 7 * 12
    assert full_receipt["row_plan"]["cursor"] == resumed_receipt["row_plan"]["cursor"] == 7
    assert len(full_receipt["steps"]) == 2 and len(resumed_receipt["steps"]) == 1
    assert_nested_equal(torch, load_file(str(full / "adapters.safetensors")), load_file(str(resumed / "adapters.safetensors")))
    assert_nested_equal(torch, torch.load(full_safety / "ckpt-00000002" / "optimizer.pt", weights_only=False),
                        torch.load(split_safety / "ckpt-00000002" / "optimizer.pt", weights_only=False))
    initial = json.loads((resumed / "initial-receipt-segment-1.json").read_text())
    assert initial["before_first_update"] and initial["sampler"]["cursor"] == 4
    assert initial["data_identity"]["recovery_amendment"]["document_sha256"]

    # Checksums alone cannot prove semantically consistent progress. Reseal a
    # genuinely saved checkpoint with a cursor moved backward; actual resume
    # must refuse before an optimizer update.
    corrupt = root / "resealed-inconsistent-checkpoint"
    shutil.copytree(checkpoint, corrupt)
    state = json.loads((corrupt / "trainer_state.json").read_text())
    state["sampler"]["cursor"] = 0
    write_json(corrupt / "trainer_state.json", state)
    complete = json.loads((corrupt / "COMPLETE.json").read_text())
    complete["sha256"]["trainer_state.json"] = sha(corrupt / "trainer_state.json")
    write_json(corrupt / "COMPLETE.json", complete)
    assert module.safety_checkpoint_problem(corrupt) is None
    with pytest.raises(ReadinessError, match="cursor"):
        run("bad-cursor", ["--resume", str(corrupt)])
    assert not (root / "bad-cursor" / "adapters.safetensors").exists()


def test_actual_command_nonfinite_retry_reuses_rows_and_optimizer_matches_clean(production_fixture, monkeypatch):
    import torch
    from safetensors.torch import load_file
    module, run, lengths, _, root, _ = production_fixture
    code, clean, clean_safety = run("clean")
    assert code == 0
    real = module.LayerwiseEngine.forward_backward
    calls = []
    def fail_first_accumulation_window(self, ids, targets, objective, **kwargs):
        calls.append(ids.detach().clone())
        result = real(self, ids, targets, objective, **kwargs)
        if len(calls) == 2:
            # Inject one finite-check failure after the real engine produced
            # gradients. The real retry path must discard the entire window.
            next(p for p in self.bank.trainable() if p.grad is not None).grad.fill_(float("nan"))
        return result
    monkeypatch.setattr(module.LayerwiseEngine, "forward_backward", fail_first_accumulation_window)
    code, retried, retry_safety = run("retry")
    assert code == 0
    receipt = json.loads((retried / "training-receipt.json").read_text())
    assert receipt["step"] == 2 and receipt["tokens_seen"] == sum(lengths)
    assert receipt["skipped_steps"] == 1 and receipt["applied_steps"] == 2
    assert receipt["cumulative_applied_progress"]["rows"] == 7
    assert receipt["steps"][0]["applied"] is False and receipt["steps"][0]["step"] == 0
    assert torch.equal(calls[0], calls[2]) and torch.equal(calls[1], calls[3])
    assert_nested_equal(torch, load_file(str(clean / "adapters.safetensors")), load_file(str(retried / "adapters.safetensors")))
    assert_nested_equal(torch, torch.load(clean_safety / "ckpt-00000002" / "optimizer.pt", weights_only=False),
                        torch.load(retry_safety / "ckpt-00000002" / "optimizer.pt", weights_only=False))


def test_actual_resume_refuses_same_size_raw_teacher_target_mutation(production_fixture):
    module, run, _, _, root, _ = production_fixture
    code, _, safety = run("target-seal", ["--max-steps-this-run", "1"])
    assert code == 0
    cache = root / "synthetic_qtext_cache"
    manifest_before = sha(cache / "manifest.json")
    shard_before = sha(cache / "shard-00000" / "shard.json")
    target = cache / "shard-00000" / "topk_logprobs.bf16"
    changed = bytearray(target.read_bytes())
    changed[0] ^= 1  # Same length, finite nearby BF16 logprob; metadata stays fixed.
    target.write_bytes(changed)
    assert sha(cache / "manifest.json") == manifest_before
    assert sha(cache / "shard-00000" / "shard.json") == shard_before
    assert not module.load_peer_module("qtext_teacher_cache").verify_shard(cache / "shard-00000")["ok"]
    try:
        code, _, _ = run("target-seal-refused", ["--resume", str(safety / "ckpt-00000001")])
    except (RuntimeError, ValueError) as error:
        assert any(term in str(error).lower() for term in ("sha", "digest", "verify", "integrity", "shard"))
        code = 2
    assert code == 2, "metadata identity must not permit changed raw teacher targets on resume"
    assert not (root / "target-seal-refused" / "adapters.safetensors").exists()
