"""Meaningful CPU checks for deterministic teacher reconstruction and admission.

These labeled fixtures prove mechanics, never real parent capabilities. Root's
bounded real-cache preparation supplies the independent genuine cache evidence.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/media_cache.py"
spec = importlib.util.spec_from_file_location("recovery_media_cache", PATH)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def source_target():
    torch = pytest.importorskip("torch")
    gen = torch.Generator().manual_seed(11)
    sources = torch.randn(2, 16, 1, 4, 4, generator=gen)
    noise = torch.randn(1, 16, 1, 4, 4, generator=torch.Generator().manual_seed(7))
    sigma, cfg = .4, 4.
    noisy = (1 - sigma) * sources[1:2] + sigma * noise
    vc = torch.randn_like(noisy)
    vu = torch.randn_like(noisy)
    velocity = vu + cfg * (vc - vu)
    x0hat = (noisy.double() - sigma * velocity.double()).float()
    tensors = {"item.x0": sources, "item.index": torch.tensor([41, 42]), "seed": torch.tensor([7]),
               "sigma": torch.tensor([sigma]), "v": velocity, "v_cond": vc, "v_uncond": vu, "x0hat": x0hat,
               "target_item": torch.tensor([42])}
    sidecar = {"targets": [{"item_index": 42, "seed": 7, "sigma": sigma,
                             "noise_sha256": hashlib.sha256(M.tensor_digest(noise).encode()).hexdigest()}]}
    return tensors, sidecar, noisy


def test_reconstruction_joins_source_item_instead_of_target_row_and_checks_exact_noise():
    torch = pytest.importorskip("torch")
    tensors, sidecar, expected = source_target()
    noisy, velocity, proof = M.reconstruct_qimg_target(tensors, sidecar, 0, 4.)
    assert torch.equal(noisy, expected) and torch.equal(velocity, tensors["v"])
    assert proof["denoising_max_abs"] < 1e-5
    sidecar["targets"][0]["noise_sha256"] = "0" * 64
    with pytest.raises(M.Error, match="exact CPU teacher noise"):
        M.reconstruct_qimg_target(tensors, sidecar, 0, 4.)


def test_reconstruction_refuses_wrong_guidance_or_endpoint():
    tensors, sidecar, _ = source_target()
    tensors["v_cond"] = tensors["v_cond"] + 1
    with pytest.raises(M.Error, match="CFG branches"):
        M.reconstruct_qimg_target(tensors, sidecar, 0, 4.)
    tensors, sidecar, _ = source_target()
    tensors["x0hat"] = tensors["x0hat"] + 1
    with pytest.raises(M.Error, match="algebra disagrees"):
        M.reconstruct_qimg_target(tensors, sidecar, 0, 4.)


def test_native_grid_packing_separates_noisy_and_velocity_bands_without_truncation():
    torch = pytest.importorskip("torch")
    tensors, _, noisy = source_target()
    batch = M.native_qimg_batch(noisy, tensors["v"], sigma=.4, prefix_ids=[1, 2, 3], max_length=10)
    assert batch["input_ids"].tolist() == [[1, 2, 3] + [M.MEDIA_PLACEHOLDER] * 4]
    assert batch["native_media_indices"].tolist() == [[3, 4, 5, 6]]
    assert torch.equal(batch["native_media_features"][0, 0, 512:576], noisy[0, :, 0, :2, :2].flatten())
    assert torch.equal(batch["native_media_targets"][..., 256:320], M.qimg_pack(tensors["v"]))
    assert not batch["native_media_features"][..., :512].any() and not batch["native_media_features"][..., 576:].any()
    assert batch["native_media_positions"].tolist() == [[[0., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 1., 1., 0.]]]
    with pytest.raises(M.Error, match="no row truncation"):
        M.native_qimg_batch(noisy, tensors["v"], sigma=.4, prefix_ids=[1, 2, 3], max_length=6)


def test_media_admission_refuses_placeholder_train_and_cross_split_groups(tmp_path):
    rows = [{"schema": "recovery_media_admission.v1", "kind": "qimg", "item_id": "a", "recovery_group_id": "same",
             "split": "train", "train_admitted": True, "blockers": [],
             **{key: "a" * 64 for key in ("asset_sha256", "prompt_sha256", "conditioning_sha256")}}]
    data = tmp_path / "admission.jsonl"
    data.write_text(json.dumps(rows[0]) + "\n")
    manifest = {"schema": "recovery_media_admission.v1", "connected_groups_verified": True, "eval_disjoint": True,
                "sources": {"qimg_items": {"sha256": "b" * 64}, "eval_rows": {"sha256": "c" * 64}},
                "outputs": {"admission": {"path": "C:/old-root/admission.jsonl", "sha256": M.R.sha(data), "rows": 1}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert M.media_admission(path, "b" * 64)[1]["a"]["train_admitted"]
    manifest["connected_groups_verified"] = False
    path.write_text(json.dumps(manifest))
    with pytest.raises(M.Error, match="source-group admission"):
        M.media_admission(path, "b" * 64)
    manifest["connected_groups_verified"] = True
    rows.append({**rows[0], "item_id": "b", "split": "test", "train_admitted": False})
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest["outputs"]["admission"].update(sha256=M.R.sha(data), rows=2)
    path.write_text(json.dumps(manifest))
    with pytest.raises(M.Error, match="crosses train/holdout"):
        M.media_admission(path, "b" * 64)
