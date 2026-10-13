"""Native code-band conversion and authentic cache admission contracts."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/music_cache.py"
spec = importlib.util.spec_from_file_location("recovery_music_cache_test", PATH)
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
spec.loader.exec_module(C)


def source(torch):
    n = 3
    return {"tokens": torch.tensor([2, 5, 8], dtype=torch.int32),
            "topk_ids": torch.arange(64, dtype=torch.int32).repeat(n, 1),
            "topk_logprobs": torch.full((n, 64), -7., dtype=torch.bfloat16),
            "lse_full": torch.full((n,), 10.), "lse_band": torch.full((n,), 9.),
            "eos_logprob": torch.full((n,), -12.), "prompt_ids": torch.tensor([200000, 200001], dtype=torch.int32)}


def sidecar():
    return {"recorded": C.RECORDED, "band": {"start": 151669, "end": 215669}, "lm_metadata": {"min_tokens": 3}}


def test_recorded_full_to_band_normalization_and_previous_code_only_teacher_forcing():
    torch = pytest.importorskip("torch")
    tensors = source(torch)
    batch, proof = C.native_music_batch(tensors, sidecar(), [1, 2])
    assert batch["input_ids"].tolist() == [[1, 2, 252418, 252421]]
    assert torch.equal(batch["topk_logprobs_band"], torch.full((3, 64), -6.))
    assert proof["teacher_positions"] == 3 and proof["prompt_length"] == 2
    assert proof["target_code_in_same_input_position"] is False
    assert not any(token in tensors["prompt_ids"].tolist() for token in batch["input_ids"].flatten().tolist())


def test_band_overmass_duplicate_codes_bad_context_and_parent_ids_fail_closed():
    torch = pytest.importorskip("torch")
    tensors = source(torch)
    tensors["topk_logprobs"].fill_(-4.)
    with pytest.raises(C.Error, match="mass exceeds one"):
        C.native_music_batch(tensors, sidecar(), [1, 2])
    tensors = source(torch)
    tensors["topk_ids"][0, 1] = 0
    with pytest.raises(C.Error, match="code IDs invalid"):
        C.native_music_batch(tensors, sidecar(), [1, 2])
    with pytest.raises(C.Error, match="no truncation"):
        C.native_music_batch(source(torch), sidecar(), [1, 2], max_length=3)
    with pytest.raises(C.Error, match="code IDs invalid"):
        C.native_music_batch(source(torch), sidecar(), [252416])


def test_partial_source_refuses_before_any_torch_or_model_load(tmp_path):
    with pytest.raises(C.Error, match="not sealed"):
        C.sealed_source(tmp_path, tmp_path / "missing-items.json", "0" * 64)


def test_native_prefix_uses_native_template_and_binds_actual_teacher_defaults():
    class Tokenizer:
        chat_template = "fixture-template"
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            assert kwargs == {"add_generation_prompt": True, "enable_thinking": False}
            assert '"seconds":60' in messages[0]["content"]
            assert '"bpm":120' in messages[0]["content"]
            return {"input_ids": [11, 12]} if tokenize else "fixture"
        def __call__(self, rendered, *, add_special_tokens):
            assert rendered == "fixture" and add_special_tokens is False
            return {"input_ids": [11, 12]}
    config = {"knobs": {"min_seconds": 10, "max_seconds": 120, "node_defaults": {"bpm": 120}, "sampling": {"cfg_scale": 2}}}
    ids, proof = C.native_music_prompt(Tokenizer(), {"caption": "music", "lyrics": "", "seconds": 60},
                                      {"seconds": 60, "sampling": {"cfg_scale": 2}}, config)
    assert ids == [11, 12] and proof["parent_prompt_ids_reused"] is False


def test_sealed_source_checks_exact_corpus_count_and_configuration(tmp_path):
    config = {"schema": "omni-campaign.teacher_cache_config.v1", "teacher": C.TEACHER, "knobs": {}}
    corpus = {"schema": "omni-campaign.teacher_corpus.v1", "kind": "ace_prompts", "corpus_sha256": "a" * 64,
              "items": [{"id": "sample", "sha256": "b" * 64}], "n_items": 1, "contamination_screen": {"applied": True}}
    items = tmp_path / "items.json"
    items.write_bytes(C.R.canonical(corpus))
    (tmp_path / "cache_config.json").write_bytes(C.R.canonical(config))
    immutable = {"mode": "full", "digest": "c" * 64, "files": [{"sha256": "d" * 64}]}
    manifest = {"schema": "omni-campaign.teacher_cache.v2", "teacher": C.TEACHER, "finite": True,
                "teacher_integrity": {"equal": True, "mode": "full", "before": [immutable], "after": immutable},
                "run_config": config, "config_sha256": C.R.digest(config),
                "corpus": {"file_sha256": C.R.sha(items), "sha256": "a" * 64, "n_items": 1},
                "shards": [{"name": "shard-00000.safetensors"}], "shard_read_order": ["shard-00000.safetensors"]}
    path = tmp_path / "manifest.json"
    path.write_bytes(C.R.canonical(manifest))
    assert C.sealed_source(tmp_path, items, C.R.sha(path))[1] == corpus
    manifest["shards"] = []
    path.write_bytes(C.R.canonical(manifest))
    with pytest.raises(C.Error, match="corpus/count"):
        C.sealed_source(tmp_path, items, C.R.sha(path))
