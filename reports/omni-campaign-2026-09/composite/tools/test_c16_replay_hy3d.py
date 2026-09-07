"""Explicit synthetic replay fixtures; no parent/child quality evidence."""
from contextlib import ExitStack
import json
import os

import pytest
import torch
from safetensors.torch import save_file

import c16_replay_hy3d as mod


class Time(torch.nn.Module):
    def forward(self, sigma, condition=None):
        return sigma[:, None, None].expand(-1, 1, 4)


class Attention(torch.nn.Module):
    def forward(self, x, context=None):
        return x * .125 if context is None else context.mean(1)[:, None].expand_as(x) * .125


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2, self.gelu = torch.nn.Linear(4, 16), torch.nn.Linear(16, 4), torch.nn.GELU()
    def forward(self, x): return self.fc2(self.gelu(self.fc1(x)))


class Block(torch.nn.Module):
    use_moe = timested_modulate = False
    def __init__(self, skip=False):
        super().__init__()
        self.attn1, self.attn2, self.mlp = Attention(), Attention(), MLP()
        self.skip_linear = torch.nn.Linear(8, 4) if skip else None
    def forward(self, x, c, text_states, skip_value=None):
        if self.skip_linear is not None: x = self.skip_linear(torch.cat((skip_value, x), -1))
        x = x + self.attn1(x)
        x = x + self.attn2(x, text_states)
        return x + self.mlp(x)


@torch.inference_mode()
def fixture_capture(block, time, embedding, layer=0, width=4, context_width=4):
    for module in (block, time, embedding): module.half().eval().requires_grad_(False)
    latent = torch.arange(6).reshape(1, 3, 2).half() / 20
    sigma = torch.tensor([.5], dtype=torch.float16)
    token = time(sigma, condition=None)
    first = torch.cat((token, embedding(latent)), 1)
    if layer: first = first + .125  # evolved token0, never overwrite it
    tensors = {"block_input": first, "sigma": sigma, "timestep": sigma.clone(), "time_token": token,
               "context": torch.arange(2 * context_width).reshape(1, 2, context_width).half() / 10,
               "dit_latent_input": latent}
    if layer > 10: tensors["skip_value"] = first * .5
    with ExitStack() as hooks:
        for name, module in (("self_attention_write", block.attn1), ("cross_attention_write", block.attn2), ("mlp_output", block.mlp)):
            hooks.callback(module.register_forward_hook(lambda m, a, out, name=name: tensors.__setitem__(name, out.clone())).remove)
        hooks.callback(block.mlp.register_forward_pre_hook(lambda m, a: tensors.__setitem__("mlp_input", a[0].clone())).remove)
        tensors["block_output"] = block(first, token, tensors["context"], skip_value=tensors.get("skip_value"))
    return {name: tensor.clone() for name, tensor in tensors.items()}


@pytest.mark.parametrize("layer", [0, 4, 11, 14])
def test_full_context_f16_replay_and_skip(layer):
    torch.manual_seed(17)
    block, time, embedding = Block(layer > 10), Time(), torch.nn.Linear(2, 4)
    tensors = fixture_capture(block, time, embedding, layer)
    _, metrics = mod.replay_one(block, time, embedding, tensors, 0, layer, "cpu")
    assert all(row["exact"] for row in metrics.values())
    assert ("initial_block_input" in metrics) == (layer == 0)
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in block.modules())


def test_changed_complete_context_produces_baseline_mismatch():
    block, time, embedding = Block(), Time(), torch.nn.Linear(2, 4)
    tensors = fixture_capture(block, time, embedding)
    tensors["context"] = tensors["context"] + .25
    _, metrics = mod.replay_one(block, time, embedding, tensors, 0, 0, "cpu")
    assert metrics["self_attention_write"]["exact"]
    assert not metrics["cross_attention_write"]["exact"] and not metrics["block_output"]["exact"]


def test_fold_replays_native_mlp_and_reports_residual_only():
    block, time, embedding = Block(), Time(), torch.nn.Linear(2, 4)
    tensors = fixture_capture(block, time, embedding)
    with torch.no_grad(): block.mlp.fc2.bias.add_(.25)
    _, metrics = mod.replay_one(block, time, embedding, tensors, 0, 0, "cpu")
    assert all(metrics[name]["exact"] for name in ("time_token", "self_attention_write", "cross_attention_write", "mlp_input"))
    assert metrics["mlp_output"]["relative_l2"] > 0 and not metrics["block_output"]["exact"]


def test_missing_skip_or_changed_f16_refused():
    block, time, embedding = Block(), Time(), torch.nn.Linear(2, 4)
    tensors = fixture_capture(block, time, embedding)
    with pytest.raises(ValueError, match="skip ownership"):
        mod.replay_one(block, time, embedding, tensors, 0, 11, "cpu")
    block.float()
    with pytest.raises(ValueError, match="F16"):
        mod.replay_one(block, time, embedding, tensors, 0, 0, "cpu")


def test_zero_reference_residual_has_no_invented_floor():
    result = mod.residual(torch.ones(2).half(), torch.zeros(2).half())
    assert result["relative_l2"] is None and result["reference_l2"] == 0


@pytest.fixture
def candidate(tmp_path):
    mlp = MLP().half().state_dict()
    path = tmp_path / "fold.safetensors"
    save_file(mlp, str(path))
    source = {"source_identity": {"fixture": True}}
    body = {"block_index": 0, "file_sha256": "a" * 64}
    rows = [{"object_sha256": "b" * 64, "split": "fit"}, {"object_sha256": "c" * 64, "split": "held"}]
    receipt = {"schema": mod.FOLD_SCHEMA, **source, "block_index": 0, "capture_sha256": body["file_sha256"],
               "fit_object_sha256": [rows[0]["object_sha256"]], "file": path.name, "file_sha256": mod.capture.digest_file(path)}
    target = tmp_path / "fold.json"
    target.write_text(json.dumps(receipt))
    return target, receipt, source, body, rows


def test_supplied_frozen_artifact_does_not_fit_or_accept_quality(candidate):
    path, _, source, body, rows = candidate
    first, second, info, _ = mod.load_candidate(path, source, body, rows, 4, "cpu")
    assert first.weight.dtype == second.weight.dtype == torch.float16
    assert not first.weight.requires_grad and info["quality_accepted"] is False


@pytest.mark.parametrize("change", ["fit", "source", "block", "capture", "hash", "dtype", "extra"])
def test_misattributed_or_unreviewed_candidate_refused(candidate, change):
    path, receipt, source, body, rows = candidate
    if change == "fit": receipt["fit_object_sha256"].append(rows[1]["object_sha256"])
    elif change == "source": receipt["source_identity"] = {}
    elif change == "block": receipt["block_index"] = 1
    elif change == "capture": receipt["capture_sha256"] = "d" * 64
    elif change == "hash": receipt["file_sha256"] = "d" * 64
    else:
        state = MLP().state_dict()
        if change == "extra": state = {**{k: v.half() for k, v in state.items()}, "extra": torch.ones(1).half()}
        tensor_path = path.parent / receipt["file"]
        save_file(state, str(tensor_path))
        receipt["file_sha256"] = mod.capture.digest_file(tensor_path)
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): mod.load_candidate(path, source, body, rows, 4, "cpu")


@pytest.mark.parametrize("layer", [-1, 15, 20])
def test_non_dense_layer_refused_without_any_source_read(layer):
    with pytest.raises(ValueError, match="dense"):
        mod.load_capture("missing", layer, {})


@pytest.fixture
def sealed_capture_fixture(tmp_path, monkeypatch):
    """File/provenance plumbing fixture: metadata validator explicitly mocked."""
    monkeypatch.setattr(mod, "CAPTURE_DIMS", {"tokens": 3, "hidden": 4, "channels": 2})
    source = {"source_identity": {"fixture": True}}
    checked = {"contract_valid": True, "fixture": True}
    monkeypatch.setattr(mod, "validate_capture", lambda directory, source: checked)
    block, time, embedding = Block(), Time(), torch.nn.Linear(2, 4)
    one = fixture_capture(block, time, embedding)
    names = {"block_input", "context", "sigma", "timestep", "time_token", "mlp_input", "mlp_output", "block_output"}
    joined = {name: torch.cat([one[name]] * 2) for name in names}
    surface = torch.arange(35).reshape(1, 5, 7).half()
    indices = torch.tensor([[0, 2, 4]])
    anchors = surface[..., :3].gather(1, indices[..., None].expand(-1, -1, 3))
    joined.update(surface=torch.cat([surface] * 2), query_source_index=torch.cat([indices] * 2), latent_xyz=torch.cat([anchors] * 2))
    rows, block_rows = [], []
    for index, split in enumerate(("fit", "held")):
        row = {"object_sha256": str(index + 1) * 64, "conditioning_sha256": "3" * 64, "split": split, "seed": index,
               "geometry_identity": {"schema": mod.capture.GEOMETRY_SCHEMA, "sha256": str(index + 4) * 64}}
        block_rows.append(dict(row))
        tensor_path = tmp_path / f"sample-{index:04d}.safetensors"
        state = {name: one[name] for name in ("self_attention_write", "cross_attention_write", "dit_latent_input")}
        state.update(surface=surface, query_source_index=indices, anchor_xyz=anchors)
        save_file(state, str(tensor_path))
        row.update(file=tensor_path.name, file_sha256=mod.capture.digest_file(tensor_path), requested_sigma=.5)
        (tmp_path / f"sample-{index:04d}.json").write_text(json.dumps(row))
        rows.append(row)
    directory = tmp_path / "block-00"
    directory.mkdir()
    save_file(joined, str(directory / "capture.safetensors"))
    body = {"kind": "dit_block", "block_index": 0, "samples": block_rows, "file": "capture.safetensors",
            "file_sha256": mod.capture.digest_file(directory / "capture.safetensors")}
    (directory / "capture.json").write_text(json.dumps(body))
    run = {"schema": mod.capture.SCHEMA, **source, "producer": mod.capture.PRODUCER, "samples": rows, "dit_capture": checked}
    (tmp_path / "capture-run.json").write_text(json.dumps(run))
    return tmp_path, source, run


def test_sealed_capture_joins_actual_attention_and_anchor_files(sealed_capture_fixture):
    path, source, _ = sealed_capture_fixture
    tensors, rows, tracked, _ = mod.load_capture(path, 0, source)
    assert tensors["self_attention_write"].shape == (2, 4, 4)
    assert [r["split"] for r in rows] == ["fit", "held"]
    assert len(tracked) == 7


@pytest.mark.parametrize("change", ["row_order", "sample_hash", "receipt", "sigma", "anchors", "missing_write", "dtype"])
def test_mismatched_capture_join_refused(sealed_capture_fixture, change):
    path, source, run = sealed_capture_fixture
    if change == "row_order": run["samples"].reverse()
    elif change == "sample_hash": run["samples"][0]["file_sha256"] = "0" * 64
    elif change == "receipt": (path / "sample-0000.json").write_text("{}")
    elif change == "sigma":
        run["samples"][0]["requested_sigma"] = .6
        (path / "sample-0000.json").write_text(json.dumps(run["samples"][0]))
    else:
        target = path / "sample-0000.safetensors"
        with mod.safe_open(str(target), framework="pt", device="cpu") as handle:
            state = {name: handle.get_tensor(name) for name in handle.keys()}
        if change == "anchors": state["anchor_xyz"] = state["anchor_xyz"] + 1
        elif change == "missing_write": del state["cross_attention_write"]
        elif change == "dtype": state["self_attention_write"] = state["self_attention_write"].float()
        save_file(state, str(target))
        run["samples"][0]["file_sha256"] = mod.capture.digest_file(target)
        (path / "sample-0000.json").write_text(json.dumps(run["samples"][0]))
    (path / "capture-run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError): mod.load_capture(path, 0, source)


@pytest.mark.skipif(not os.environ.get("HY3D_NATIVE_TEST_SOURCE"), reason="real pinned source/runtime not mounted")
def test_original_native_dense_and_skip_blocks_with_synthetic_weights(monkeypatch):
    """Runs actual unmodified source class, tiny random weights; never parent proof."""
    monkeypatch.setenv("USE_SAGEATTN", "0")
    monkeypatch.setenv("HY3DGEN_DEBUG", "0")
    with mod.capture.native_modules(os.environ["HY3D_NATIVE_TEST_SOURCE"], include_dit=True) as native:
        config = {"model": {"params": {"depth": 21, "num_moe_layers": 6, "use_pos_emb": False,
            "use_attention_pooling": False, "with_decoupled_ca": False, "hidden_size": 16, "num_heads": 2,
            "context_dim": 6, "qk_norm": True, "qkv_bias": False, "in_channels": 2}}}
        for layer in (0, 11):
            block, time, embedding = mod.native_components(native, config, layer)
            tensors = fixture_capture(block, time, embedding, layer, width=16, context_width=6)
            _, metrics = mod.replay_one(block, time, embedding, tensors, 0, layer, "cpu")
            assert all(row["exact"] for row in metrics.values())
            if layer:
                assert not torch.equal(tensors["block_input"][:, :1], tensors["time_token"])
            with torch.no_grad(): block.mlp.fc2.bias.add_(.03125)
            _, changed = mod.replay_one(block, time, embedding, tensors, 0, layer, "cpu")
            assert changed["mlp_input"]["exact"] and not changed["mlp_output"]["exact"]
