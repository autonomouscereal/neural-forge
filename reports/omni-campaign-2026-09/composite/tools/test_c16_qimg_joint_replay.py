"""Explicit synthetic contracts; only the optional pinned-Comfy test uses native code."""
import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_qimg_joint_replay as tool
from test_c16_qimg_fold import bundle, bounded_threads, dump


class SyntheticJointBlock:
    """A cross-stream test double, never a source or numerical-parity claim."""
    def __init__(self, state, data):
        self.state, self.data = state, data

    def forward(self, state, record):
        values = {"time_recomputed": state["temb"].clone(), "rotation_recomputed": state["rotation_generated"].clone()}
        valid = state["context_valid"].flatten()
        joint = (state["img_in"].float().mean() + state["txt_in"][:, valid].float().mean()).bfloat16()
        for stream in tool.fold.STREAMS:
            x = state[stream + "_in"]
            attn = torch.ones_like(x) * joint
            mlp_in = x + attn
            _, mlp_out = tool.fold.mlp_forward(mlp_in, tool.fold.mlp_weights(self.state, 0, stream, "cpu"))
            values.update({stream + "_attn_out": attn, stream + "_mlp_in": mlp_in,
                           stream + "_mlp_out": mlp_out, stream + "_out": mlp_in + mlp_out * .25,
                           stream + "_modulation": state["temb"].repeat(1, 6)})
        return values


def prepared(tmp_path):
    path, receipt, policy_path, _ = bundle(tmp_path)
    data = tool.fold.load_captures([path], fixture=True)
    native = SyntheticJointBlock(data["frozen"], data)
    for record in receipt["captures"]:
        state = tool.fold.load_file(str(tmp_path / record["file"]))
        ni, nt = record["image_tokens"], record["text_tokens"]
        row, col = torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij")
        grid = torch.stack((torch.zeros(ni, dtype=torch.int64), row.flatten(), col.flatten()), 1)
        image_ids = grid.float(); image_ids[:, 1] -= 2; image_ids[:, 2] -= 3
        txt_ids = torch.arange(3, 3 + nt).float()[:, None].expand(-1, 3)
        state["image_patch_indices"] = grid
        state["source_ids"] = torch.cat((txt_ids, image_ids))[None]
        state["text_attention_mask"][~state["context_valid"]] = -float("inf")
        record["text_mask_present"] = True
        actual = native.forward(state, record)
        state.update({k: actual[k] for k in tool.TAPS})
        record.update(tool.fold.write_tensors(tmp_path / record["file"], state))
    dump(path, receipt)
    tool.fold.convert([path], policy_path, tool.fold.file_sha(policy_path), tmp_path / "fold", fixture=True)
    data = tool.fold.load_captures([path], fixture=True)
    folded_path = tmp_path / "fold/fold_receipt.json"
    candidate, _ = tool.load_candidate(folded_path, tool.fold.file_sha(folded_path), data)
    return path, folded_path, data, candidate


def test_complete_joint_replay_reports_both_streams_without_acceptance(tmp_path):
    path, folded, data, candidate = prepared(tmp_path)
    before = {p: tool.fold.file_sha(p) for p in (path, folded)}
    out = tmp_path / "joint_report.json"
    result = tool.replay([path], folded, tool.fold.file_sha(folded), tmp_path, "cpu", 0,
                         out, fixture=True, factory=SyntheticJointBlock)
    assert result["status"] == "measured_native_joint_block" and result["fixture"] is True
    assert result["source_baseline_passed"] is True
    assert set(result["folded_pooled_valid_rows"]) == {"img", "txt"}
    assert result["folded_pooled_valid_rows"]["txt"]["fit"]["rows"] == 10
    assert result["folded_per_forward"][0]["scores"]["txt"]["all_context_rows"]["rows"] == 12
    assert "not evaluated" in result["folded_numerical_acceptance"]
    assert {p: tool.fold.file_sha(p) for p in before} == before
    assert json.loads(out.read_text())["source_baseline_passed"] is True


def test_baseline_failure_prevents_candidate_instantiation(tmp_path):
    _, _, data, candidate = prepared(tmp_path)
    calls = []
    class WrongSource(SyntheticJointBlock):
        def forward(self, state, record):
            actual = super().forward(state, record)
            actual["txt_out"] = actual["txt_out"] + 1
            return actual
    def factory(state, metadata):
        calls.append(state)
        assert state is data["frozen"], "candidate constructed before all baseline forwards passed"
        return WrongSource(state, metadata)
    with pytest.raises(tool.fold.FoldRejected) as caught:
        tool.evaluate(data, candidate, factory, tool.capture_jobs(data)[0], 0)
    assert len(calls) == 1
    assert caught.value.receipt["folded_block_evaluated"] is False


def test_failed_baseline_writes_no_success_report_and_keeps_provenance(tmp_path):
    path, folded, _, _ = prepared(tmp_path)
    class WrongSource(SyntheticJointBlock):
        def forward(self, state, record):
            output = super().forward(state, record)
            output["img_out"] += 1
            return output
    out = tmp_path / "not_created.json"
    with pytest.raises(tool.fold.FoldRejected) as caught:
        tool.replay([path], folded, tool.fold.file_sha(folded), tmp_path, "cpu", 0,
                    out, fixture=True, factory=WrongSource)
    assert not out.exists()
    assert caught.value.receipt["capture_receipts"][0]["sha256"] == tool.fold.file_sha(path)
    assert caught.value.receipt["fold_receipt"]["sha256"] == tool.fold.file_sha(folded)


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf")])
def test_invalid_baseline_tolerance_refuses_before_factory(value):
    def factory(*args):
        pytest.fail("invalid baseline tolerance reached native construction")
    with pytest.raises(ValueError, match="tolerance"):
        tool.evaluate({}, {}, factory, [], value)


@pytest.mark.parametrize("target", ["time_recomputed", "rotation_recomputed", "img_mlp_in", "txt_modulation"])
def test_every_source_intermediate_is_gated_before_folded_replay(tmp_path, target):
    _, _, data, candidate = prepared(tmp_path)
    class WrongSource(SyntheticJointBlock):
        def forward(self, state, record):
            actual = super().forward(state, record)
            actual[target] = actual[target] + .25
            return actual
    with pytest.raises(tool.fold.FoldRejected):
        tool.evaluate(data, candidate, WrongSource, tool.capture_jobs(data)[0], 0)


@pytest.mark.parametrize("field", ["source_ids", "image_patch_indices", "joint_row_kind", "rotation_used", "text_attention_mask"])
def test_changed_complete_context_refuses(tmp_path, field):
    _, _, data, _ = prepared(tmp_path)
    root, record = tool.capture_jobs(data)[0][0]
    state, _ = tool.fold.verified_tensors(root, record, 1 << 28)
    if field == "text_attention_mask": state[field][0, 0] = float("nan")
    else: state[field].flatten()[0] += 1
    with pytest.raises(ValueError):
        tool.context_contract(state, record, data)


def test_additive_mask_preserves_negative_infinity_and_finite_valid_bias(tmp_path):
    _, _, data, _ = prepared(tmp_path)
    root, record = tool.capture_jobs(data)[0][0]
    state, _ = tool.fold.verified_tensors(root, record, 1 << 28)
    state["text_attention_mask"][0, 0] = -.75
    tool.context_contract(state, record, data)
    assert torch.isneginf(state["text_attention_mask"][0, -1])
    record = {**record, "text_mask_present": False}
    with pytest.raises(ValueError, match="absent"):
        tool.context_contract(state, record, data)


@pytest.mark.parametrize("mutation", ["frozen", "dtype", "selection", "capture_identity", "receipt_hash"])
def test_candidate_source_identity_and_mixed_storage_fail_closed(tmp_path, mutation):
    _, path, data, _ = prepared(tmp_path)
    receipt = json.loads(path.read_text())
    if mutation in ("frozen", "dtype"):
        state, target = tool.fold.verified_tensors(path.parent, receipt["output"], 1 << 28)
        name = "time_text_embed.timestep_embedder.linear_1.weight" if mutation == "frozen" else "transformer_blocks.0.img_mlp.net.2.weight"
        state[name] = state[name].float() + 1 if mutation == "frozen" else state[name].to(torch.float8_e4m3fn)
        receipt["output"] = tool.fold.write_tensors(target, state)
    elif mutation == "selection":
        state, target = tool.fold.verified_tensors(path.parent, receipt["selection"], 1 << 20)
        state["img.selected_channels"][1] = state["img.selected_channels"][0]
        receipt["selection"] = tool.fold.write_tensors(target, state)
    elif mutation == "capture_identity": receipt["capture_receipts"][0]["sha256"] = "0" * 64
    dump(path, receipt)
    with pytest.raises(ValueError):
        tool.load_candidate(path, "0" * 64 if mutation == "receipt_hash" else tool.fold.file_sha(path), data)


def test_existing_report_is_preserved_before_source_loading(tmp_path, monkeypatch):
    out = tmp_path / "old.json"; out.write_text("preserve")
    monkeypatch.setattr(tool.fold, "load_captures", lambda *a, **k: pytest.fail("existing report must refuse first"))
    with pytest.raises(ValueError, match="already exists"):
        tool.replay([], tmp_path / "none", "0" * 64, tmp_path, "cpu", 0, out)
    assert out.read_text() == "preserve"


def test_source_pin_mismatch_precedes_native_import(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        tool.init_native(tmp_path, "cpu", {"torch": torch.__version__})


@pytest.mark.skipif(not os.environ.get("QIMG_COMFY_ROOT"), reason="requires the separately pinned native Comfy runtime")
def test_optional_actual_native_tiny_block_preserves_mixed_dtype_and_joint_context():
    root = Path(os.environ["QIMG_COMFY_ROOT"])
    device = os.environ.get("QIMG_NATIVE_TEST_DEVICE", "cpu")
    assert device in {"cpu", "cuda:0"}
    runtime = {"torch": torch.__version__, "float32_matmul_precision": "highest",
               "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": True}
    tool.init_native(root, device, runtime)
    from comfy.ldm.qwen_image import model
    from comfy import ops
    torch.manual_seed(913)
    data = {"block": 0, "width": 8}
    state = {k: (torch.randn(shape) * .1).to(torch.float8_e4m3fn)
             for k, shape in tool.fold.frozen_shapes(0, 8, 32, 8).items()}
    for stream, keep in (("img", 3), ("txt", 2)):
        p = "transformer_blocks.0." + stream + "_mlp."
        for suffix in ("net.0.proj.weight", "net.0.proj.bias"):
            state[p + suffix] = state[p + suffix].view(torch.uint8)[:keep].clone().view(torch.float8_e4m3fn)
        state[p + "net.2.weight"] = state[p + "net.2.weight"].bfloat16()[:, :keep].contiguous()
        state[p + "net.2.bias"] = state[p + "net.2.bias"].bfloat16()
    native = tool.NativeBlock(model, ops.manual_cast, state, data, device, axes=(2, 2, 4))
    assert native.block.img_mlp.net[0].proj.weight.dtype == torch.float8_e4m3fn
    assert native.block.img_mlp.net[2].weight.dtype == torch.bfloat16
    ids = torch.arange(6).float()[None, :, None].expand(1, 6, 3)
    record = {"text_mask_present": True}
    capture = {"img_in": torch.randn(1, 4, 8).bfloat16(), "txt_in": torch.randn(1, 2, 8).bfloat16(),
        "temb": torch.randn(1, 8).bfloat16(), "source_ids": ids, "sigma": torch.tensor([.3]),
        "rotation_used": native.embed(ids).bfloat16(), "text_attention_mask": torch.tensor([[0., -float('inf')]]).bfloat16()}
    before = {k: v.clone() for k, v in capture.items()}
    output = native.forward(capture, record)
    assert set(output) == set(tool.TAPS) | {"time_recomputed", "rotation_recomputed"}
    assert output["img_out"].shape == (1, 4, 8) and output["txt_out"].shape == (1, 2, 8)
    assert all(tool.same_bytes(capture[k], before[k]) for k in capture)
    assert all(bool(torch.isfinite(v).all()) for v in output.values())
