"""HY3D source math, exact pixel processing and unapproved native route refusal."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

PATH = Path(__file__).resolve().parents[1] / "experiments/chimera_recovery/hy3d_cache.py"
spec = importlib.util.spec_from_file_location("recovery_hy3d_cache_test", PATH)
H = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = H
spec.loader.exec_module(H)


def trajectory(torch):
    """Small deterministic mathematical fixture, never a production teacher."""
    time = torch.linspace(0., 1., 30)
    next_time = torch.cat((time[1:], torch.ones(1)))
    velocity = torch.arange(4 * 64).reshape(4, 64).float() / 256.
    rows = [velocity * 2.]
    for index in range(29):
        rows.append(rows[-1] + (next_time[index] - time[index]) * velocity)
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tensors = {"x_t": torch.stack(rows), "v_guided": velocity.repeat(30, 1, 1), "t_native": time,
               "noise_level": 1. - time, "sigma_next": next_time, "x0_final": rows[-1].clone(),
               "context_cond": torch.zeros(1370, 1024), "image_input": torch.zeros(3, 512, 512),
               "image_mask": torch.zeros(1, 512, 512), "mesh_vertices": vertices,
               "mesh_faces": torch.tensor([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=torch.int32)}
    sidecar = {"steps": 30, "guidance": 5., "time_convention": H.TIME_CONVENTION,
               "checks": {"recording_euler_chain_residual_rel": 0., "reference_upstream_pipeline": {
                   "pipeline_sha256": H.PIPELINE_SHA, "bitwise_equal": True, "max_abs_diff": 0.}},
               "mesh": {"vertices": 4, "faces": 4, "octree_resolution": 256, "num_chunks": 8000}}
    return tensors, sidecar


def test_real_schedule_sign_euler_chain_and_endpoint_exclusion():
    torch = pytest.importorskip("torch")
    tensors, sidecar = trajectory(torch)
    proof = H.validate_trajectory(tensors, sidecar, require_native_shape=False)
    assert proof["positive_noise_steps"] == list(range(29))
    assert proof["excluded_clean_endpoint_steps"] == [29]
    assert proof["target_conversion"] == "canonical_velocity = -v_guided"
    assert proof["euler_chain_residual_rel"] == 0. and proof["geometry_semantic_quality_claim"] is False
    tensors["x_t"][15, 0, 0] += .5
    with pytest.raises(H.Error, match="Euler chain"):
        H.validate_trajectory(tensors, sidecar, require_native_shape=False)
    tensors, sidecar = trajectory(torch)
    tensors["noise_level"][0] = .5
    with pytest.raises(H.Error, match="noise_level"):
        H.validate_trajectory(tensors, sidecar, require_native_shape=False)


def test_invalid_mesh_and_missing_independent_parent_comparison_refuse():
    torch = pytest.importorskip("torch")
    tensors, sidecar = trajectory(torch)
    sidecar["checks"]["reference_upstream_pipeline"]["bitwise_equal"] = False
    with pytest.raises(H.Error, match="upstream pipeline"):
        H.validate_trajectory(tensors, sidecar, require_native_shape=False)
    tensors, sidecar = trajectory(torch)
    tensors["mesh_vertices"][:, 2] = 0.
    with pytest.raises(H.Error, match="planar"):
        H.validate_trajectory(tensors, sidecar, require_native_shape=False)
    tensors, sidecar = trajectory(torch)
    tensors["mesh_faces"][0, 0] = 4
    with pytest.raises(H.Error, match="vertex/face"):
        H.validate_trajectory(tensors, sidecar, require_native_shape=False)


def test_chart_icons_and_held_objects_cannot_receive_geometry_admission():
    item = {"sha256": "a" * 64, "held": False}
    row = {"asset_sha256": item["sha256"], "teacher_item_sha256": item["sha256"],
           "geometry_source_category": "physical_object_photo", "genuine_geometry_target": True,
           "source_is_chart_or_icon": False, "train_admitted": True, "split": "train", "blockers": []}
    assert H.geometry_admission(row, item)
    with pytest.raises(H.Error, match="charts/icons"):
        H.geometry_admission({**row, "source_is_chart_or_icon": True}, item)
    with pytest.raises(H.Error, match="held"):
        H.geometry_admission(row, {**item, "held": True})


def test_experimental_pack_has_no_targets_in_inputs_and_zero_context_gradients():
    torch = pytest.importorskip("torch")
    noisy = torch.arange(4 * 64).float().reshape(4, 64)
    velocity = torch.ones_like(noisy) * 3.
    positions = torch.arange(16).float().reshape(4, 4)
    image = torch.ones(2, 1536) * 5.
    batch, proof = H.experimental_native_pack(noisy, velocity, prefix_ids=[1, 2], latent_positions=positions,
                                             image_features=image, image_positions=torch.zeros(2, 4), sigma=.5, max_length=9)
    assert batch["native_media_type_ids"].tolist() == [[6, 6, 7, 7, 7, 7, 7]]
    assert torch.equal(batch["native_media_features"][0, 3:, 576:640], noisy)
    assert torch.equal(batch["native_media_targets"][0, 3:, 320:384], -velocity)
    assert torch.equal(batch["native_media_positions"][0, 2], torch.full((4,), -1.))
    assert not bool(batch["native_media_features"][..., 320:384].any())
    assert proof["production_ready"] is False and proof["position_policy_established"] is False
    prediction = torch.zeros_like(batch["native_media_targets"], requires_grad=True)
    H.R.native_velocity_loss(prediction, batch["native_media_targets"], batch["native_media_type_ids"],
                             batch["native_media_loss_mask"], {7: (320, 384)},
                             prediction_kind="canonical_velocity", target_kind="canonical_velocity").backward()
    assert not bool(prediction.grad[0, :3].any())
    assert bool(prediction.grad[0, 3:, 320:384].abs().sum() > 0)
    assert not bool(prediction.grad[..., :320].any()) and not bool(prediction.grad[..., 384:].any())
    with pytest.raises(H.Error, match="type dispatch"):
        H.R.validate_native_batch("volume_generate", batch)
    with pytest.raises(H.Error, match="no truncation"):
        H.experimental_native_pack(noisy, velocity, prefix_ids=[1, 2], latent_positions=positions,
                                   image_features=image, image_positions=torch.zeros(2, 4), sigma=.5, max_length=8)


def test_partial_preflight_cannot_select_zero_position_policy(tmp_path):
    result = H.preflight(tmp_path, tmp_path / "missing.json", "0" * 64)
    assert result["production_ready"] is False
    assert result["experimental_proposal"]["selected"] is False
    assert result["experimental_proposal"]["parent_learned_query_slots"] is False
    assert any("not sealed" in blocker for blocker in result["blockers"])
    assert any("query-position" in blocker for blocker in result["blockers"])


def test_source_pins_are_bytes_not_claimed_booleans(tmp_path, monkeypatch):
    texts = {"hy3dshape/hy3dshape/preprocessors.py": "fixture preprocessor\n",
             "configs/dit_config_2_1.yaml": "use_pos_emb: false\nuse_attention_pooling: false\n",
             "hy3dshape/hy3dshape/models/denoisers/hunyuandit.py": "x = torch.cat([c, x], dim=1)\n"}
    pins = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in texts.items()}
    monkeypatch.setattr(H, "SOURCE_PINS", pins)
    data = {"hy3d_sources": {name: {"text": text, "sha256": pins[name]} for name, text in texts.items()}}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(data))
    assert H.source_contract(path)["parent_latent_permutation_equivariant"] is True
    data["hy3d_sources"]["configs/dit_config_2_1.yaml"]["text"] += "changed\n"
    path.write_text(json.dumps(data))
    with pytest.raises(H.Error, match="source bytes differ"):
        H.source_contract(path)


def test_actual_pinned_preprocessor_reconstruction_matches_source_and_refuses_mutation(tmp_path):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    image_path = tmp_path / "asset.png"
    original = np.zeros((23, 31, 4), dtype=np.uint8)
    original[2:21, 4:27, :3] = [20, 70, 190]  # BGR: verifies RGB conversion too.
    original[2:21, 4:27, 3] = 180
    assert cv2.imwrite(str(image_path), original)
    sha = H.R.sha(image_path)
    image, mask = H.preprocess_image(image_path, sha)
    assert image.shape == (3, 512, 512) and mask.shape == (1, 512, 512)
    assert image[0, 256, 256] > image[2, 256, 256]
    expected = {"image_input": image, "image_mask": mask}
    assert H.reconstruct_image(image_path, sha, expected)["image_bitwise_equal"] is True
    changed = image.clone()
    changed[0, 256, 256] += .01
    with pytest.raises(H.Error, match="image/mask differ"):
        H.reconstruct_image(image_path, sha, {**expected, "image_input": changed})
    image_path.write_bytes(b"changed")
    with pytest.raises(H.Error, match="image bytes changed"):
        H.preprocess_image(image_path, sha)
