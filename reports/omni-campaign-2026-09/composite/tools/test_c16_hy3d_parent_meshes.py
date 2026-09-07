"""Synthetic fixtures only; optional genuine source pipeline, never parent proof."""
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import c16_hy3d_parent_meshes as mod


SETTINGS = {**mod.FIXED, "steps": 4, "guidance_scale": 5., "bounds": 1.01,
            "octree_resolution": 8, "query_chunk": 128, "capture_sigma": .5}
DIMS = {"tokens": 4, "channels": 2, "context_tokens": 3, "context_width": 2}


def put_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return mod.digest_file(path)


@pytest.fixture
def plan(tmp_path):
    rows, parent = [], []
    for i, split in enumerate(("fit", "held")):
        image = tmp_path / f"image{i}.png"
        # Only read_plan hash tests; explicitly not a real PNG/production input.
        image.write_bytes(f"synthetic provenance fixture {i}".encode())
        row = {"id": f"image{i}", "conditioning_file": image.name,
               "conditioning_sha256": mod.digest_file(image), "split": split, "seed": 100 + i}
        rows.append(row)
        parent.append({"id": row["id"], "sha256": row["conditioning_sha256"],
                       "split": "fit" if split == "fit" else "holdout"})
    receipt = tmp_path / "qimg.json"
    digest = put_json(receipt, {"schema": "neural_forge.qimg_parent_images.v1", "status": "complete", "images": parent})
    body = {"schema": mod.PLAN_SCHEMA, "generation": dict(SETTINGS), "images": rows,
            "qimg_receipt_file": receipt.name, "qimg_receipt_sha256": digest}
    path = tmp_path / "plan.json"
    put_json(path, body)
    return path, body


def test_hash_bound_plan_preserves_qimg_split_and_seed(plan):
    path, body = plan
    actual, rows, identity = mod.read_plan(path)
    assert actual == body and [r["seed"] for r in rows] == [100, 101]
    assert rows[1]["qimg_image"]["split"] == "holdout"
    assert identity["plan_sha256"] == mod.digest_file(path)


@pytest.mark.parametrize("key,value", [("steps", 1), ("steps", True), ("steps", 51),
    ("query_chunk", 2049), ("bounds", float("nan")), ("capture_sigma", -1),
    ("guidance_scale", -1), ("dtype", "bfloat16"), ("postprocess", "repair"),
    ("mc_level", 0), ("octree_resolution", 512), ("seed_device", "cuda")])
def test_unreviewed_generation_refused(plan, key, value):
    path, body = plan
    body["generation"][key] = value
    put_json(path, body)
    with pytest.raises(ValueError):
        mod.read_plan(path)


@pytest.mark.parametrize("change", ["seed", "split", "id", "image", "receipt", "missing_receipt", "single"])
def test_changed_or_incomplete_provenance_refused(plan, change):
    path, body = plan
    if change == "seed": body["images"][0]["seed"] = -1
    elif change == "split": body["images"][1]["split"] = "fit"
    elif change == "id": body["images"][1]["id"] = body["images"][0]["id"]
    elif change == "image": (path.parent / "image0.png").write_bytes(b"changed fixture")
    elif change == "receipt": (path.parent / "qimg.json").write_text("{}")
    elif change == "missing_receipt": del body["qimg_receipt_file"]
    elif change == "single": body["images"].pop()
    put_json(path, body)
    with pytest.raises(ValueError):
        mod.read_plan(path)


def test_same_image_cannot_cross_splits_even_with_other_id(plan):
    path, body = plan
    body["images"][1].update(conditioning_file="image0.png", conditioning_sha256=body["images"][0]["conditioning_sha256"])
    receipt = path.parent / "qimg.json"
    parent = json.loads(receipt.read_text())
    parent["images"][1]["sha256"] = parent["images"][0]["sha256"]
    body["qimg_receipt_sha256"] = put_json(receipt, parent)
    put_json(path, body)
    with pytest.raises(ValueError, match="leaked"):
        mod.read_plan(path)


def test_source_digest_refuses_mutation(tmp_path):
    file = tmp_path / "source.py"
    file.write_text("# fixture\n")
    pins = {file.name: mod.digest_file(file)}
    mod.verify_files(tmp_path, pins)
    file.write_text("# changed fixture\n")
    with pytest.raises(ValueError, match="re-audit"):
        mod.verify_files(tmp_path, pins)


class TinyDiT(torch.nn.Module):
    """Pure fixture model with observable native-shaped API, not a trained DiT."""
    guidance_embed = False
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(2)])
        self.t_embedder = torch.nn.Identity()
        self.bad = False
    def forward(self, latent, timestep, context, guidance=None):
        time = self.t_embedder(timestep[:, None, None])
        value = latent * .125 + context["main"].mean((1, 2))[:, None, None] + time
        for block in self.blocks: value = block(value)
        return value * float("nan") if self.bad else value


class FixtureScheduler:
    def __init__(self): self.step_index = None
    def step(self, value, time, sample):
        index = 0 if self.step_index is None else self.step_index
        result = sample.float() + (self.sigmas[index + 1] - self.sigmas[index]) * value
        self.step_index = index + 1
        return SimpleNamespace(prev_sample=result.to(value.dtype))


class FixturePipeline:
    def __init__(self):
        self.model = TinyDiT().eval().requires_grad_(False)
        self.scheduler = FixtureScheduler()
        self.image_processor = lambda image: {"image": torch.ones(1, 3, 2, 2), "mask": torch.ones(1, 1, 2, 2)}
    def __call__(self, image, num_inference_steps, guidance_scale, generator, enable_pbar, callback, callback_steps):
        self.image_processor(image)
        latent = torch.randn((1, 4, 2), generator=generator, dtype=torch.float16)
        self.scheduler.sigmas = torch.cat((torch.linspace(0, 1, num_inference_steps), torch.ones(1)))
        self.scheduler.step_index = None
        context = {"main": torch.cat((torch.ones(1, 3, 2), torch.zeros(1, 3, 2))).half()}
        for i, time in enumerate(self.scheduler.sigmas[:-1] * 1000):
            prediction = self.model(torch.cat([latent] * 2), time.expand(2).half() / 1000, context, guidance=None)
            cond, uncond = prediction.chunk(2)
            output = self.scheduler.step(uncond + guidance_scale * (cond - uncond), time, latent)
            latent = output.prev_sample
            callback(i, time, output)
        return latent


def test_observation_does_not_change_fixture_sampling(monkeypatch):
    monkeypatch.setattr(mod, "CAPTURE_DIMS", DIMS)
    pipeline = FixturePipeline()
    expected = pipeline(None, 4, 5., torch.Generator().manual_seed(321), False, lambda *a: None, 1)
    actual, trace = mod.sample_native(pipeline, None, 321, SETTINGS)
    assert torch.equal(actual, expected)
    assert trace["context"].shape == (2, 3, 2)
    assert trace["trajectory_model_output"].shape == (4, 2, 4, 2)
    assert trace["trajectory_sigma_pair"][-1].tolist() == [1., 1.]
    assert torch.equal(trace["trajectory_latent_before"][-1], trace["trajectory_latent_after"][-1])
    assert trace["step_index_before"].tolist() == [-1, 1, 2, 3]
    assert "step" not in pipeline.scheduler.__dict__
    assert not pipeline.model._forward_hooks and not pipeline.model._forward_pre_hooks


def test_nonfinite_fixture_restores_observers(monkeypatch):
    monkeypatch.setattr(mod, "CAPTURE_DIMS", DIMS)
    pipeline = FixturePipeline()
    processor = pipeline.image_processor
    pipeline.model.bad = True
    with pytest.raises(ValueError, match="nonfinite"):
        mod.sample_native(pipeline, None, 321, SETTINGS)
    assert pipeline.image_processor is processor
    assert "step" not in pipeline.scheduler.__dict__
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in pipeline.model.modules())


def test_frozen_fixture_required(monkeypatch):
    monkeypatch.setattr(mod, "CAPTURE_DIMS", DIMS)
    pipeline = FixturePipeline()
    pipeline.model.train()
    with pytest.raises(ValueError, match="frozen"):
        mod.sample_native(pipeline, None, 321, SETTINGS)


def test_actual_ply_roundtrip_preserves_fixture_geometry(tmp_path):
    pytest.importorskip("trimesh")
    import numpy as np
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0, 1., 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    identity = mod.canonical_geometry(vertices, faces)
    result = mod.write_mesh(tmp_path / "fixture.ply", vertices, faces, identity)
    assert result["geometry_identity"] == identity
    with pytest.raises(ValueError, match="replace"):
        mod.write_mesh(tmp_path / "fixture.ply", vertices, faces, identity)


def native_decode_fixture(native):
    class Geometry(torch.nn.Module):
        label_type = "binary"
        def forward(self, queries, latents):
            return .5 - queries.square().sum(-1, keepdim=True)
    class TinyVAE(torch.nn.Module):
        latent_shape = (4, 2)
        scale_factor = 1.0039506158752403
        enable_flashvdm_decoder = native["vae"].ShapeVAE.enable_flashvdm_decoder
        latents2mesh = native["vae"].ShapeVAE.latents2mesh
        def __init__(self):
            super().__init__()
            self.geo_decoder = Geometry()
            self.decode_input = None
        def decode(self, latent):
            self.decode_input = latent.clone()
            return latent
    return TinyVAE().eval().requires_grad_(False)


@pytest.mark.skipif(not os.environ.get("HY3D_NATIVE_TEST_SOURCE"), reason="real pinned native source/runtime not mounted")
def test_native_scheduler_and_mc_without_comfy_import(tmp_path, monkeypatch):
    """Genuine scheduler/volume/MC, synthetic field: no parent capability claim."""
    import numpy as np
    monkeypatch.setenv("HY3DGEN_DEBUG", "0")
    monkeypatch.setenv("USE_SAGEATTN", "0")
    root = os.environ["HY3D_NATIVE_TEST_SOURCE"]
    mod.verify_files(root, mod.EXTRA_SOURCE)
    with mod.native_modules(root, include_dit=False) as native:
        prefix = native["vae"].__package__.rsplit(".models.autoencoders", 1)[0]
        scheduler_module = importlib.import_module(prefix + ".schedulers")
        scheduler = scheduler_module.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(sigmas=np.linspace(0, 1, 4), device="cpu")
        assert scheduler.sigmas[-2:].tolist() == [1., 1.]
        latent = torch.ones(1, 4, 2).half()
        prediction = torch.full_like(latent, .25)
        for index, timestep in enumerate(scheduler.timesteps):
            expected = (latent.float() + (scheduler.sigmas[index+1] - scheduler.sigmas[index]) * prediction).half()
            latent = scheduler.step(prediction, timestep, latent).prev_sample
            assert torch.equal(latent, expected)
        vae = native_decode_fixture(native)
        vertices, faces, field, extraction = mod.extract_native(vae, latent, SETTINGS)
        assert torch.equal(vae.decode_input, latent)
        assert field["field_logits"].shape == (1, 9, 9, 9)
        assert extraction["query_count"] == 729 and extraction["query_calls"] == 6
        assert not vae.geo_decoder._forward_hooks
        mod.write_mesh(tmp_path / "synthetic-native-field.ply", vertices, faces, extraction["geometry_identity"])


@pytest.mark.skipif(not os.environ.get("HY3D_NATIVE_TEST_PIPELINE"),
                    reason="whole Comfy import requires allocated GPU-driver runtime; CPU-only Triton import fails")
def test_original_pipeline_scheduler_and_extraction_with_tiny_fixtures(monkeypatch, tmp_path):
    """Actual unedited algorithms, synthetic model/field. Never parent evidence."""
    from PIL import Image
    monkeypatch.setattr(mod, "CAPTURE_DIMS", DIMS)
    monkeypatch.setenv("HY3DGEN_DEBUG", "0")
    monkeypatch.setenv("USE_SAGEATTN", "0")
    with mod.generation_modules(os.environ["HY3D_NATIVE_TEST_SOURCE"], "/opt/ComfyUI", "cpu") as native:
        class Conditioner(torch.nn.Module):
            def forward(self, image, **kwargs): return {"main": torch.ones(1, 3, 2).half()}
            def unconditional_embedding(self, size, **kwargs): return {"main": torch.zeros(size, 3, 2).half()}
        vae = native_decode_fixture(native)
        pipeline = native["pipeline"].Hunyuan3DDiTFlowMatchingPipeline(
            vae=vae, model=TinyDiT().eval(), conditioner=Conditioner().eval(),
            scheduler=native["scheduler"].FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000),
            image_processor=native["image"].ImageProcessorV2(size=512, border_ratio=.15), device="cpu", dtype=torch.float16)
        image = Image.new("RGB", (512, 512), "white")
        expected = pipeline(image=image, num_inference_steps=4, guidance_scale=5.,
                            generator=torch.Generator().manual_seed(321), enable_pbar=False)
        actual, trace = mod.sample_native(pipeline, image, 321, SETTINGS)
        assert torch.equal(actual, expected)
        assert trace["trajectory_sigma_pair"][-1].tolist() == [1., 1.]
        assert torch.equal(trace["trajectory_latent_before"][-1], trace["trajectory_latent_after"][-1])
        assert torch.equal(trace["trajectory_model_timestep"], (trace["trajectory_scheduler_timestep"].half() / 1000)[:, None].expand(-1, 2))
        vertices, faces, field, extraction = mod.extract_native(vae, actual, SETTINGS)
        assert torch.equal(vae.decode_input, actual)  # native nodes.py: no /scale_factor
        assert field["field_logits"].shape == (1, 9, 9, 9)
        assert extraction["query_count"] == 729 and extraction["query_calls"] == 6
        assert not vae.geo_decoder._forward_hooks
        mod.write_mesh(tmp_path / "synthetic-native-field.ply", vertices, faces, extraction["geometry_identity"])
