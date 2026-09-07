"""Explicitly synthetic hook/contract tests; no fixture stands in for source output."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_capture as tool


def test_loader_does_not_request_model_bytes_as_extra_inference_scratch(monkeypatch, tmp_path):
    """Exercise the production loader call, without importing Comfy or a parent."""
    calls = []
    dit = nn.Linear(2, 2, dtype=torch.bfloat16)
    model = types.SimpleNamespace(diffusion_model=dit,
        model_config=types.SimpleNamespace(manual_cast_dtype=torch.bfloat16))
    patcher = types.SimpleNamespace(model=model)
    package = types.ModuleType("comfy")
    package.__path__ = []
    options = types.ModuleType("comfy.options")
    cli = types.ModuleType("comfy.cli_args")
    cli.args = types.SimpleNamespace(cpu=False)
    management = types.ModuleType("comfy.model_management")
    management.in_training = False
    management.get_torch_device = lambda: torch.device("cpu")
    management.load_models_gpu = lambda models, **kwargs: calls.append((models, kwargs))
    sd = types.ModuleType("comfy.sd")
    sd.load_diffusion_model = lambda path, **kwargs: patcher
    for module in (package, options, cli, management, sd):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    package.options, package.sd, package.model_management = options, sd, management
    old_path = sys.path[:]
    try:
        got, loaded, dtype, receipt = tool.load_parent(tmp_path / "fixture.safetensors", tmp_path, "cpu")
    finally:
        sys.path[:] = old_path
    assert got is patcher and loaded is dit and dtype == torch.bfloat16
    assert calls == [([patcher], {"force_full_load": True})]
    assert receipt["full_residency_verified"] is True
    assert not dit.training


class FixturePositions(nn.Module):
    def forward(self, ids):
        angles = ids[..., 0:1].expand(-1, -1, 4) / 10
        return torch.stack((angles.cos(), -angles.sin(), angles.sin(), angles.cos()), -1).view(1, 1, ids.shape[1], 4, 2, 2)


class FixtureAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.img_out = nn.Linear(8, 8)
        self.txt_out = nn.Linear(8, 8)

    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        joint = torch.cat((encoder_hidden_states, hidden_states), 1)
        q, k, v = self.qkv(joint).chunk(3, -1)
        out = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        nt = encoder_hidden_states.shape[1]
        return self.img_out(out[:, nt:]), self.txt_out(out[:, :nt])


class FixtureBlock(nn.Module):
    attention_head_dim = 8

    def __init__(self):
        super().__init__()
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))
        self.img_mlp = nn.Sequential(nn.Linear(8, 12), nn.GELU(approximate="tanh"), nn.Linear(12, 8))
        self.txt_mlp = copy.deepcopy(self.img_mlp)
        self.attn = FixtureAttention()
        self.fail = False

    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask, temb,
                image_rotary_emb=None, timestep_zero_index=None, transformer_options={}):
        im, tm = self.img_mod(temb).chunk(6, -1), self.txt_mod(temb).chunk(6, -1)
        # Deliberate in-place mutation tests whether the tap preserves entry values.
        hidden_states.add_(0.125)
        ia, ta = self.attn(hidden_states, encoder_hidden_states,
                           encoder_hidden_states_mask=encoder_hidden_states_mask, image_rotary_emb=image_rotary_emb)
        img = torch.addcmul(hidden_states, im[2][:, None], ia)
        txt = encoder_hidden_states + tm[2][:, None] * ta
        img = torch.addcmul(img, im[5][:, None], self.img_mlp(img))
        txt = torch.addcmul(txt, tm[5][:, None], self.txt_mlp(txt))
        if self.fail:
            raise RuntimeError("fixture source failure")
        return txt, img


class FixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe_embedder = FixturePositions()
        self.transformer_blocks = nn.ModuleList([FixtureBlock()])
        self.img_in = nn.Linear(64, 8)
        self.txt_in = nn.Linear(6, 8)
        self.time = nn.Linear(1, 8)
        self.zero_index = None

    def forward(self, x, timestep, context, attention_mask=None, transformer_options={}):
        rows = x[:, :, 0].reshape(1, 16, 2, 2, 2, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, 4, 64)
        img, txt = self.img_in(rows), self.txt_in(context)
        ids = torch.arange(txt.shape[1] + img.shape[1], dtype=torch.float32).view(1, -1, 1).expand(-1, -1, 3)
        pe = self.pe_embedder(ids).to(x.dtype)
        txt, img = self.transformer_blocks[0](hidden_states=img, encoder_hidden_states=txt,
            encoder_hidden_states_mask=attention_mask, temb=self.time(timestep[:, None].to(x.dtype)),
            image_rotary_emb=pe, timestep_zero_index=self.zero_index, transformer_options=transformer_options)
        return img


def forward_fixture():
    torch.manual_seed(18)
    model = FixtureModel().eval()
    x, sigma, context = torch.randn(1, 16, 1, 4, 4), torch.tensor([.6]), torch.randn(1, 3, 6)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
    return model, x, sigma, context, mask


def assert_no_hooks(model):
    for module in model.modules():
        assert not module._forward_pre_hooks and not module._forward_hooks


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hooks_record_actual_fixture_forward_without_replacing_computation(dtype):
    model, x, sigma, context, mask = forward_fixture()
    model, x, context = model.to(dtype), x.to(dtype), context.to(dtype)
    with torch.no_grad():
        expected = model(x, sigma, context, attention_mask=mask)
    records, meta = tool.capture_forward(model, 0, x, sigma, context, mask, fixture=True)
    assert torch.equal(records["prediction"], expected)
    assert meta["fixture"] is True and meta["complete_context"] is True
    assert meta["image_tokens"] == 4 and meta["text_tokens"] == 3
    assert records["source_ids"].shape == (1, 7, 3)
    assert records["rotation_used"].shape == (1, 1, 7, 4, 2, 2)
    assert records["rotation_generated"].dtype == torch.float32
    assert records["rotation_used"].dtype == dtype
    assert records["sigma"].dtype == torch.float32
    assert records["joint_row_kind"].tolist() == [0, 0, 0, 1, 1, 1, 1]
    assert records["image_patch_indices"].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    assert torch.equal(records["text_attention_mask"], mask)
    for stream in ("img", "txt"):
        assert records[stream + "_modulation"].shape == (1, 48)
        with torch.no_grad():
            torch.testing.assert_close(records[stream + "_mlp_out"], getattr(model.transformer_blocks[0], stream + "_mlp")(records[stream + "_mlp_in"]))
    assert_no_hooks(model)


def test_pre_hook_preserves_entry_values_before_inplace_source_write():
    model, x, sigma, context, mask = forward_fixture()
    before = []
    handle = model.img_in.register_forward_hook(lambda m, a, out: before.append(out.detach().clone()))
    records, _ = tool.capture_forward(model, 0, x, sigma, context, mask, fixture=True)
    handle.remove()
    assert torch.equal(records["img_in"], before[0])
    assert not torch.equal(records["img_in"], records["img_out"])
    assert_no_hooks(model)


def test_production_tap_refuses_fixture_class():
    model, x, sigma, context, mask = forward_fixture()
    with pytest.raises(ValueError, match="actual Qwen-Image"):
        tool.capture_forward(model, 0, x, sigma, context, mask)
    assert_no_hooks(model)


def test_partial_hook_registration_is_cleaned_up():
    model, *_ = forward_fixture()
    del model.transformer_blocks[0].txt_mod
    with pytest.raises(AttributeError):
        with tool.QimgBlockTap(model, 0, fixture=True):
            pass
    assert_no_hooks(model)


def test_missing_submodule_write_is_not_filled_with_fake_values():
    model, x, sigma, context, mask = forward_fixture()
    with tool.QimgBlockTap(model, 0, fixture=True) as tap, torch.no_grad():
        model(x, sigma, context, attention_mask=mask)
        del tap.records["txt_mlp_out"]
        with pytest.raises(ValueError, match="incomplete source hooks"):
            tap.finish()
    assert_no_hooks(model)


@pytest.mark.parametrize("variant", ["source_error", "reference", "nonfinite"])
def test_hooks_are_removed_even_when_source_or_contract_fails(variant):
    model, x, sigma, context, mask = forward_fixture()
    if variant == "source_error":
        model.transformer_blocks[0].fail = True
    elif variant == "reference":
        model.zero_index = 1
    else:
        x.fill_(float("nan"))
    with pytest.raises((ValueError, RuntimeError)):
        tool.capture_forward(model, 0, x, sigma, context, mask, fixture=True)
    assert_no_hooks(model)


def test_captures_do_not_alias_input_or_parameter_storage():
    model, x, sigma, context, mask = forward_fixture()
    records, _ = tool.capture_forward(model, 0, x, sigma, context, mask, fixture=True)
    old = {name: value.clone() for name, value in records.items()}
    x.zero_()
    sigma.zero_()
    context.zero_()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    assert all(torch.equal(value, old[name]) for name, value in records.items())


def input_bundle(tmp_path):
    # Files here are explicitly test fixtures. The production input validator is
    # being exercised, not used to claim these are real image encodings.
    image = tmp_path / "fixture_image.bin"
    image.write_bytes(b"fixture image identity")
    source = tmp_path / "fixture_encoder.safetensors"
    save_file({"weight": torch.ones(2, 2)}, str(source))
    _, identity = tool.read_header(source)
    tensor_path = tmp_path / "fixture_inputs.safetensors"
    state = {"model_inputs": torch.zeros(2, 16, 1, 4, 4), "sigma": torch.tensor([.3, .8]),
             "context": torch.ones(1, 3, 3584), "context_mask": torch.tensor([[True, True, False]]),
             "context_token_ids": torch.tensor([[2, 3, 0]])}
    save_file(state, str(tensor_path))
    prompt = "Fixture only"
    entry = {"id": "fixture01", "split": "fit", "prompt": prompt,
             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
             "image": {"path": image.name, "sha256": tool.file_sha(image)},
             "tensors": {"path": tensor_path.name, "sha256": tool.file_sha(tensor_path)}}
    upstream = {"status": "complete", "kind": "real_image_vae_qwen25_context", "latent_space": "wan21_model_input",
                "vae": {"path": source.name, "header_sha256": identity["header_sha256"]},
                "text_encoder": {"path": source.name, "header_sha256": identity["header_sha256"]},
                "entries": {entry["id"]: {"image_sha256": entry["image"]["sha256"],
                    "prompt_sha256": entry["prompt_sha256"], "tensor_file_sha256": entry["tensors"]["sha256"]}}}
    upstream_path = tmp_path / "fixture_encoding_receipt.json"
    upstream_path.write_text(json.dumps(upstream))
    body = {"schema": tool.INPUT_SCHEMA, "kind": "real_image_vae_qwen25_context",
            "latent_space": "wan21_model_input", "context_kind": "qwen25_vl_7b_hidden_states",
            "encoding_receipt": {"path": upstream_path.name, "sha256": tool.file_sha(upstream_path)}, "entries": [entry]}
    manifest = tmp_path / "fixture_manifest.json"
    manifest.write_text(json.dumps(body))
    return manifest, body, tensor_path


def test_input_identity_linkage_and_bounded_full_context(tmp_path):
    path, body, _ = input_bundle(tmp_path)
    entries, receipt = tool.load_inputs(path)
    assert entries[0]["token_count"] == 7 and receipt["forwards"] == 2
    assert "not numerically re-run" in receipt["scope"]
    estimate = tool.output_size_estimate({"block_parameters": 100, "retained_global_time_parameters": 20,
                                          "width": 3072, "head_dim": 128}, entries)
    assert estimate > 2 * 5 * 7 * 3072 * 4
    with pytest.raises(ValueError, match="token count"):
        tool.load_inputs(path, max_tokens=6)
    with pytest.raises(ValueError, match="forward count"):
        tool.load_inputs(path, max_forwards=1)


@pytest.mark.parametrize("mutation", ["fixture_kind", "image_hash", "prompt_hash", "tensor_hash", "upstream_link", "split_leak"])
def test_input_refusals_do_not_create_output(tmp_path, mutation):
    path, body, _ = input_bundle(tmp_path)
    if mutation == "fixture_kind":
        body["kind"] = "fixture"
    elif mutation in {"image_hash", "tensor_hash"}:
        name = "image" if mutation == "image_hash" else "tensors"
        body["entries"][0][name]["sha256"] = "0" * 64
    elif mutation == "prompt_hash":
        body["entries"][0]["prompt"] += "changed"
    elif mutation == "upstream_link":
        body["entries"][0]["id"] = "unlinked"
    else:
        body["entries"].append({**copy.deepcopy(body["entries"][0]), "id": "fixture02", "split": "holdout"})
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        tool.load_inputs(path)
    assert not (tmp_path / "out").exists()


def test_writer_records_dtypes_and_tensor_digests_without_narrowing(tmp_path):
    state = {"sigma": torch.tensor([.631578922]), "activation": torch.randn(1, 4, 8).bfloat16(),
             "source_fp8": torch.tensor([1., 2.]).to(torch.float8_e4m3fn)}
    body = tool.write_tensors(tmp_path / "fixture.safetensors", state)
    loaded = tool.load_file(str(tmp_path / "fixture.safetensors"))
    assert body["dtypes"]["source_fp8"] == "torch.float8_e4m3fn"
    for name, value in loaded.items():
        assert tool.tensor_sha(value) == body["tensor_sha256"][name] == tool.tensor_sha(state[name])
