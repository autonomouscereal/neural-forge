"""ACE conditioning, native layout and odd-frame objective contracts."""
import importlib.util
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/ace_cache.py"
spec = importlib.util.spec_from_file_location("recovery_ace_test", PATH)
A = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = A
spec.loader.exec_module(A)


def fixture():
    torch = pytest.importorskip("torch")
    noisy = torch.arange(64 * 3).reshape(64, 3).float() / 50
    hints = -noisy - 2  # clean targets are never used as code conditioning.
    velocity = torch.ones_like(noisy)
    codes = torch.tensor([17], dtype=torch.int32)
    batch = A.native_ace_batch(noisy, velocity, hints, codes, sigma=.5, prefix_ids=[1, 2], max_length=8)
    condition = {"ace_source_latents": "authentic_code_detokenizer", "ace_chunk_masks": "ones_with_right_padding_zero",
                 "regime": "teacher_audio_code_conditioned_generation", "full_clean_latent_in_inputs": False,
                 "source_band_sha256": A.M.tensor_digest(batch["native_media_features"][..., :128][batch["native_media_type_ids"] == 4]),
                 "chunk_band_sha256": A.M.tensor_digest(batch["native_media_features"][..., 128:256][batch["native_media_type_ids"] == 4]),
                 "codes_sha256": A.M.tensor_digest(codes), "conditioning_manifest_sha256": "a" * 64,
                 "ace_step15_sha256": A.SOURCE_PINS["comfy/ldm/ace/ace_step15.py"], "model_base_sha256": A.SOURCE_PINS["comfy/model_base.py"]}
    receipt = {"schema": "chimera_recovery.native_flow_cache.v1", "sealed": True, "all_finite": True, "all_structural_ok": True,
               "guided_teacher": True, "no_clean_target_in_inputs": True,
               **{key: "a" * 64 for key in ("cache_sha256", "corpus_sha256", "teacher_sha256", "recovery_amendment_sha256")},
               "route": "music_generate", "target_kind": "canonical_velocity", "prediction_kind": "raw", "parent_signs": {"4": -1.},
               "input_layout": "feature", "target_layout": "emission", "noisy_feature_bands": A.NOISE_BANDS,
               "noisy_feature_bands_sha256": A.R.digest(A.NOISE_BANDS), "conditioning_contract": condition}
    return batch, receipt, noisy, hints


def test_native_ace_codes_context_patch_order_and_odd_frame_channel_mask():
    torch = pytest.importorskip("torch")
    batch, _, noisy, hints = fixture()
    assert batch["input_ids"].tolist() == [[1, 2, 252433, A.M.MEDIA_PLACEHOLDER, A.M.MEDIA_PLACEHOLDER]]
    assert batch["native_media_indices"].tolist() == [[3, 4]] and batch["ace_code_indices"].tolist() == [[2]]
    assert torch.equal(batch["native_media_features"][0, 0, :128], hints[:, :2].flatten())
    assert torch.equal(batch["native_media_features"][0, 1, 256:384:2], noisy[:, 2])
    assert not batch["native_media_features"][0, 1, 257:384:2].any()
    assert batch["native_media_channel_loss_mask"][0, 1, :128].sum() == 64
    assert batch["native_media_loss_mask"].tolist() == [[1., .5]]
    assert torch.allclose(batch["native_media_positions"][0, :, 0], torch.tensor([.04, .12]))
    stored, contract = A.M.compact_native_batch(batch, A.BANDS)
    expanded = A.M.inflate_native_batch(stored, contract)
    assert all(torch.equal(value, expanded[key]) for key, value in batch.items())


def test_ace_velocity_gradient_only_actual_frames_and_conditioning_digest_refusals():
    torch = pytest.importorskip("torch")
    batch, receipt, _, _ = fixture()
    objective = A.R.ReadyObjectiveAdapter(receipt).bind("music_generate", batch, A.BANDS, parent_signs={4: -1.})
    prediction = torch.zeros_like(batch["native_media_targets"]).requires_grad_()
    prediction.data[0, 1, 1:128:2] = 999  # missing313th frame partner must have zero influence.
    loss = objective(prediction)
    assert float(loss) == 1.
    loss.backward()
    assert not prediction.grad[0, 1, 1:128:2].any() and not prediction.grad[..., 128:].any()
    assert torch.allclose(prediction.grad[0, 0, :128], torch.full((128,), -1 / 96))
    assert torch.allclose(prediction.grad[0, 1, :128:2], torch.full((64,), -1 / 96))
    batch["input_ids"][0, 2] += 1
    with pytest.raises(A.Error, match="code IDs differ"):
        A.R.ReadyObjectiveAdapter(receipt).bind("music_generate", batch, A.BANDS, parent_signs={4: -1.})
    batch["input_ids"][0, 2] -= 1
    batch["native_media_features"][0, 0, 1] += 1
    with pytest.raises(A.Error, match="band digest moved"):
        A.R.ReadyObjectiveAdapter(receipt).bind("music_generate", batch, A.BANDS, parent_signs={4: -1.})


def test_native_ace_context_budget_and_code_band_fail_closed():
    torch = pytest.importorskip("torch")
    batch, _, noisy, hints = fixture()
    with pytest.raises(A.Error, match="no truncation"):
        A.native_ace_batch(noisy, noisy, hints, torch.tensor([1]), sigma=.5, prefix_ids=[1, 2], max_length=4)
    with pytest.raises(A.Error, match="64000 written rows"):
        A.native_ace_batch(noisy, noisy, hints, torch.tensor([64000]), sigma=.5, prefix_ids=[1, 2], max_length=8)


def test_ace_sealed_cuda_noise_recipe_checks_scheduler_seed_and_targets():
    torch = pytest.importorskip("torch")
    batch, _, noisy, _ = fixture()
    item = {"sha256": "a" * 64}
    sigmas = [1., .9, .75, .5]  # actual shifted-simple scheduler for source item0.
    sidecar = {"latent_frames": 3, "code_frames": 1, "cfg": 1., "sigmas": sigmas}
    seeds = [int(__import__("hashlib").sha256(f"ace-noise-{row}:{item['sha256']}".encode()).hexdigest()[:8], 16) for row in range(4)]
    tensors = {"x0": noisy, "x_t": noisy[None, :, :].repeat(4, 1, 1), "v_guided": noisy[None, :, :].repeat(4, 1, 1),
               "codes": torch.tensor([17], dtype=torch.int32), "sigma": torch.tensor(sigmas), "noise_seed": torch.tensor(seeds)}
    assert A.validate_ace_tensor_recipe(tensors, sidecar, index=0, item=item) == 3
    tensors["noise_seed"][0] += 1
    with pytest.raises(A.Error, match="noise seeds"):
        A.validate_ace_tensor_recipe(tensors, sidecar, index=0, item=item)
    tensors["noise_seed"][0] -= 1
    with pytest.raises(A.Error, match="shifted scheduler"):
        A.validate_ace_tensor_recipe(tensors, sidecar, index=1, item=item)
    tensors["v_guided"][0, 0, 0] = float("nan")
    with pytest.raises(A.Error, match="shape/finite"):
        A.validate_ace_tensor_recipe(tensors, sidecar, index=0, item=item)
