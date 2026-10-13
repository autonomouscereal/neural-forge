"""D3b preparation checks and genuine tiny HybridD3 interrupted-resume proof."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/train_media.py"
spec = importlib.util.spec_from_file_location("recovery_train_media", PATH)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def test_d3_production_preflight_is_blocked_without_real_prerequisites(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"schema": "chimera_recovery.d3b_config.v1", "stage": "D3d"}))
    result = M.preflight(config)
    assert result["production_ready"] is False and result["training_started"] is False
    assert any("only D3b" in blocker for blocker in result["blockers"])
    assert result["unsupported_stages"] == ["D3a", "D3c", "D3d"]


def test_d3_optimizer_has_no_invented_production_defaults():
    with pytest.raises(M.Error, match="must pin exactly"):
        M.optimizer_config({"unit_lr": 0.001})
    valid = {"unit_lr": 1e-4, "emission_lr": 1e-4, "lora_lr": 1e-4, "betas": [0.9, 0.95], "eps": 1e-8,
             "weight_decay": 0., "clip": 1., "warmup_steps": 20, "min_lr_ratio": .1}
    assert M.optimizer_config(valid) == valid
    with pytest.raises(M.Error, match="positive"):
        M.optimizer_config({**valid, "lora_lr": 0.})


def test_d3_receipts_refuse_changed_bytes_and_frozen_body(tmp_path):
    path = tmp_path / "receipt.json"
    path.write_bytes(b"{}")
    pin = {"path": path.name, "sha256": M.R.sha(path)}
    assert M.pinned_file(pin, tmp_path) == path.resolve()
    path.write_bytes(b"changed")
    with pytest.raises(M.Error, match="changed"):
        M.pinned_file(pin, tmp_path)
    doc = tmp_path / "prereg.md"
    doc.write_bytes(b"## 1. Scope\nchanged frozen body\n\n## Amendments\n2026-10-13 guided_velocity\n")
    with pytest.raises(M.Error, match="frozen D1-D4"):
        M.verify_preregistration({"path": doc.name, "sha256": M.R.sha(doc), "amendment_sha256": "a" * 64}, tmp_path)


def test_rolling_checkpoints_keep_two_and_do_not_touch_other_artifacts(tmp_path):
    for i in range(4):
        folder = tmp_path / f"checkpoint-safety-{i:08d}"
        folder.mkdir()
        (folder / "state.pt").write_bytes(b"fixture")
    important = tmp_path / "checkpoint-initial"
    important.mkdir()
    M.prune_safety_checkpoints(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["checkpoint-initial", "checkpoint-safety-00000002", "checkpoint-safety-00000003"]


def test_actual_tiny_hybrid_velocity_updates_and_exact_resume(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    source = os.environ.get("CHIMERA_D3_BASELINE")
    if not source:
        pytest.skip("set CHIMERA_D3_BASELINE to the actual D3 module; fixture uses no production weights")
    foundation = os.environ.get("CHIMERA_OMNI_SRC")
    if foundation:
        sys.path.insert(0, str(Path(foundation) / "src"))
    out = tmp_path / "proof"
    out.mkdir()
    result = M.cpu_fixture(Path(source), out)
    assert result["passed"] and result["actual_hybrid_forward_backward"]
    assert result["resume_tensor_exact"] and result["changed_identity_refused"]
    assert result["updated_parameters"] and result["optimizer_updates"] == 3
    assert result["production_model_weights_loaded"] is False
    assert result["production_ready"] is False
    assert (out / "initial" / "receipt.json").is_file()
    assert (out / "delta" / "manifest.json").is_file()


def test_h3_source_positions_are_dispatched_by_source_and_media_ids_match_base():
    torch = pytest.importorskip("torch")
    batch = {"input_ids": torch.tensor([[1, 2, 50, 50]]), "native_media_features": torch.zeros(1, 2, 6144),
             "native_media_type_ids": torch.tensor([[2, 3]]), "native_media_positions": torch.zeros(1, 2, 4),
             "native_media_indices": torch.tensor([[2, 3]]), "native_media_valid_mask": torch.ones(1, 2, dtype=torch.bool),
             "native_media_noise_levels": torch.ones(1, 2), "charted_source_positions": torch.zeros(1, 4, 3)}
    segment = M.make_segment(batch, media_placeholder=50)
    assert segment["source_positions"]["h3"] is batch["charted_source_positions"]
    with pytest.raises(M.Error, match="immutable base ledger"):
        M.make_segment(batch, media_placeholder=51)
