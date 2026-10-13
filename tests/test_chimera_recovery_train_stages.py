"""Stage dispatch uses actual native engines and preserves wrapper resume policy."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
D = load("recovery_stages_test", ROOT / "experiments/chimera_recovery/train_stages.py")


def test_vision_only_cache_is_not_falsely_ready_and_preflight_does_not_import_torch(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": "chimera_recovery.native_stage_config.v1", "stage": "D3d"}))
    already = "torch" in sys.modules
    result = D.preflight(config)
    assert not result["production_ready"] and not result["training_started"]
    assert any("ViT-only" in value and "assistant-answer KD" in value for value in result["blockers"])
    assert ("torch" in sys.modules) == already


def test_actual_music_dispatch_adam_and_wrapper_safety_exact_resume(tmp_path):
    fixture = load("recovery_stages_music_fixture", ROOT / "tests/test_chimera_recovery_train_music.py")
    torch, engine, construct, segment, kwargs = fixture.setup(tmp_path)
    batch = {"input_ids": segment["ids"], "topk_codes": kwargs["topk_codes"], "topk_logprobs_band": kwargs["topk_logprobs_band"]}
    receipt = {"prompt_length": kwargs["prompt_length"]}
    assert not D.dispatch_step(engine, "D3c", batch, receipt)["skipped"]
    policy = {"consecutive_skips": 0, "window": [True, False], "decode_review_passed": True}
    D.save_dispatch_checkpoint(engine, tmp_path / "dispatch-interrupted", policy)
    expected = D.dispatch_step(engine, "D3c", batch, receipt)
    resumed = construct()
    resumed.resume(tmp_path / "dispatch-interrupted")
    assert D.resume_dispatch_policy(resumed, tmp_path / "dispatch-interrupted") == policy
    actual = D.dispatch_step(resumed, "D3c", batch, receipt)
    assert expected == actual and torch.equal(resumed.head_master, engine.head_master)
    assert torch.equal(resumed.embed_master, engine.embed_master) and resumed.sampler_counter == engine.sampler_counter
    assert all(torch.equal(value, resumed.census[key]) for key, value in engine.census.items())
    (tmp_path / "dispatch-interrupted/dispatch_policy.json").write_text("{}")
    with pytest.raises(D.Error, match="policy missing/changed"):
        D.resume_dispatch_policy(resumed, tmp_path / "dispatch-interrupted")


def test_actual_refit_dispatch_owns_independent_outward_chart(tmp_path):
    fixture = load("recovery_stages_refit_fixture", ROOT / "tests/test_chimera_recovery_train_refit.py")
    torch, engine, _, tensors, kwargs = fixture.setup(tmp_path)
    batch = dict(zip(("host_rows", "parent_in", "parent_out", "host_write_target"), tensors))
    batch.update({"levels": kwargs["levels"], "row_types": kwargs["row_types"]})
    before = engine.hybrid.units[engine.placement.container].chart_out.weight.detach().clone()
    result = D.dispatch_step(engine, "D3a", batch, {})
    assert not result["skipped"] and not torch.equal(before, engine.hybrid.units[engine.placement.container].chart_out.weight)


def test_actual_vision_dispatch_owns_answer_and_feature_gradients(tmp_path):
    fixture = load("recovery_stages_vision_fixture", ROOT / "tests/test_chimera_recovery_train_vision.py")
    _, _, engine, _, segment, targets, features = fixture.setup(tmp_path)
    batch = {"input_ids": segment["ids"], "native_media_valid_mask": __import__("torch").ones_like(segment["types"], dtype=__import__("torch").bool),
             "native_media_features": segment["features"], "native_media_type_ids": segment["types"], "native_media_positions": segment["positions"],
             "native_media_indices": segment["indices"], "native_media_noise_levels": segment["sigma"], "teacher_features": features}
    batch.update({"answer." + key: value for key, value in targets.items()})
    result = D.dispatch_step(engine, "D3d", batch, {"answer_positions": 3})
    assert not result["skipped"] and result["feature_kd_weighted"] > 0 and result["text_weight"] == 1.
