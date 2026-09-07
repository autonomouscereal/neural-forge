"""Synthetic preparation mechanics only; these tests are not teacher evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_prepare_inputs as tool
import c16_qimg_capture as capture


def make_plan(tmp_path):
    image = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), (180, 80, 10)).save(image)
    plan = {"schema": tool.PLAN_SCHEMA, "sigmas": [.25, .75], "entries": [{"id": "fixture",
        "split": "fit", "seed": 5, "prompt": "Fixture image, not actual source evidence.",
        "image": {"path": image.name, "sha256": tool.file_sha(image)},
        "image_provenance": {"kind": "test_fixture", "description": "Only the validator is tested."}}]}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


def test_plan_preserves_actual_pixel_identity_and_explicit_provenance(tmp_path):
    path, plan = make_plan(tmp_path)
    got = tool.read_plan(path)
    pixels, meta = tool.image_pixels(tmp_path / "fixture.png")
    assert got["entries"][0]["pixel_preflight"] == meta
    assert pixels.shape == (1, 32, 32, 3)
    torch.testing.assert_close(pixels[0, 0, 0], torch.tensor([180, 80, 10]) / 255)
    assert got["entries"][0]["image_provenance"]["kind"] == "test_fixture"


@pytest.mark.parametrize("mutation", ["sigma_zero", "sigma_nan", "sigma_bool", "sigma_duplicate", "seed_bool",
    "missing_provenance", "template", "changed_image", "split_leak", "too_many"])
def test_plan_refuses_ambiguous_or_unbounded_inputs(tmp_path, mutation):
    path, plan = make_plan(tmp_path)
    if mutation.startswith("sigma_"):
        plan["sigmas"] = {"sigma_zero": [0], "sigma_nan": [float("nan")], "sigma_bool": [True],
                          "sigma_duplicate": [.5, .5]}[mutation]
    elif mutation == "seed_bool":
        plan["entries"][0]["seed"] = True
    elif mutation == "missing_provenance":
        del plan["entries"][0]["image_provenance"]
    elif mutation == "template":
        plan["entries"][0]["prompt"] = "<|im_start|>system"
    elif mutation == "changed_image":
        plan["entries"][0]["image"]["sha256"] = "0" * 64
    elif mutation == "split_leak":
        plan["entries"].append({**copy.deepcopy(plan["entries"][0]), "id": "other", "split": "holdout"})
    else:
        plan["entries"] *= 17
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        tool.read_plan(path)


def test_pixels_refuse_implicit_resize_and_composite_alpha_explicitly(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (32, 16), (0, 0, 0, 0)).save(path)
    pixels, meta = tool.image_pixels(path)
    assert pixels.eq(1).all() and meta["alpha_composite"] == "white"
    Image.new("RGB", (33, 16)).save(path)
    with pytest.raises(ValueError, match="no implicit crop"):
        tool.image_pixels(path)


def fixture_tokens():
    ids = [151644, 8948, 198, 500, 151645, 198, 151644, 872, 198, 123, 124, 151645, 198, 151644, 77091, 198]
    return {"qwen25_7b": [[(t, 1.0) for t in ids]]}


def test_native_user_template_crop_keeps_assistant_suffix_and_alignment():
    ids, meta = tool.aligned_context_ids(fixture_tokens(), torch.zeros(1, 7, 3584))
    assert meta["template_crop_start"] == 9
    assert ids.tolist() == [[123, 124, 151645, 198, 151644, 77091, 198]]
    with pytest.raises(ValueError, match="exact source token crop"):
        tool.aligned_context_ids(fixture_tokens(), torch.zeros(1, 6, 3584))


def test_alignment_refuses_embedded_or_weighted_tokens():
    tokens = fixture_tokens()
    tokens["qwen25_7b"][0][3] = ({"image": "fixture"}, 1.0)
    with pytest.raises(ValueError, match="embedded/weighted"):
        tool.aligned_context_ids(tokens, torch.zeros(1, 7, 3584))


def flow(sigma, noise, clean):
    s = sigma.reshape(-1, 1, 1, 1, 1)
    return s * noise + (1 - s) * clean


def test_calibration_noise_is_seeded_shared_and_not_a_sampler_claim():
    clean = torch.arange(256).float().reshape(1, 16, 1, 4, 4) / 100
    first, sigma, receipt = tool.calibration_inputs(clean, [.25, .75], 10, flow)
    second, _, _ = tool.calibration_inputs(clean, [.25, .75], 10, flow)
    assert torch.equal(first, second)
    inferred = (first - (1 - sigma[:, None, None, None, None]) * clean) / sigma[:, None, None, None, None]
    torch.testing.assert_close(inferred[0], inferred[1])
    assert "not captured sampler trajectory" in receipt["sigma_semantics"]
    assert receipt["sigma_shift_applied"] is False
    with pytest.raises(ValueError, match="noise scaling differs"):
        tool.calibration_inputs(clean, [.25], 10, lambda s, n, c: flow(s, n, c) + 1)


def test_end_to_end_fixture_output_passes_capture_identity_validator(tmp_path, monkeypatch):
    plan_path, _ = make_plan(tmp_path)
    source = tmp_path / "fixture_source.safetensors"
    save_file({"weight": torch.ones(1)}, str(source))
    class TextModel(torch.nn.Module):
        pass
    TextModel.__module__ = "comfy.text_encoders.qwen_image"
    class VaeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
    VaeModel.__module__ = "comfy.ldm.wan.vae"
    clip = types.SimpleNamespace(cond_stage_model=TextModel(), tokenize=lambda prompt: fixture_tokens(),
        encode_from_tokens=lambda tokens, **kwargs: {"cond": torch.ones(1, 7, 3584, dtype=torch.float16)})
    vae = types.SimpleNamespace(first_stage_model=VaeModel(), latent_channels=16,
                              encode=lambda pixels: torch.ones(1, 16, 1, 4, 4))
    sd = types.SimpleNamespace(load_clip=lambda *a, **k: clip, CLIPType=types.SimpleNamespace(QWEN_IMAGE=18),
                               VAE=lambda **kwargs: vae)
    utils = types.SimpleNamespace(load_torch_file=lambda *a, **k: ({"weight": torch.ones(1)}, {}))
    latent_format = types.SimpleNamespace(process_in=lambda x: x / 2)
    monkeypatch.setattr(tool, "init_comfy", lambda *args: (sd, utils, latent_format,
                        types.SimpleNamespace(noise_scaling=flow), {"fixture": "explicit"}))
    package = types.ModuleType("comfy")
    package.__path__ = []
    management = types.ModuleType("comfy.model_management")
    management.unload_all_models = lambda: None
    package.model_management = management
    monkeypatch.setitem(sys.modules, "comfy", package)
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    out = tmp_path / "encoded"
    tool.encode_plan(plan_path, source, source, tmp_path, "cpu", out)
    entries, receipt = capture.load_inputs(out / "capture_inputs.json")
    assert receipt["forwards"] == 2 and len(entries) == 1
    body = json.loads((out / "encoding_receipt.json").read_text())
    assert body["status"] == "complete"
    assert body["details"]["fixture"]["alignment"]["teacher_output_dtype"] == "torch.float16"
    assert body["details"]["fixture"]["image_provenance"]["kind"] == "test_fixture"
    with pytest.raises(ValueError, match="already exists"):
        tool.encode_plan(plan_path, source, source, tmp_path, "cpu", out)
