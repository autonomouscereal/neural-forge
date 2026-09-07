"""Synthetic converter contracts; no fixture proves H3 media capability."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c16_convert_h3 as converter
from chimera_omni.charted import ChartedBlockSpec
from chimera_omni.modeling.charted import ChartedSourceBlock


def fixture(tmp_path):
    torch.manual_seed(37)
    width, inner, keep, heads, head_dim = 12, 12, 3, 2, 8
    prefix = "blocks.25."
    block = {
        "norm1.weight": torch.ones(width), "norm2.weight": torch.ones(width),
        "attn.q_norm.weight": torch.ones(head_dim), "attn.k_norm.weight": torch.ones(head_dim),
        "attn.qkv_proj.weight": torch.randn(3 * heads * head_dim, width) / 10,
        "attn.out_proj.weight": torch.randn(width, heads * head_dim) / 10,
        "mlp.fc1.weight": torch.randn(inner * 2, width) / 10,
        "mlp.fc2.weight": torch.randn(width, inner) / 10,
        "adaln_proj.linear.weight": torch.randn(18 * width, 3) / 10,
        "adaln_proj.linear.bias": torch.randn(18 * width) / 10,
    }
    original = {prefix + k: t for k, t in block.items()}
    folded = {prefix + "mlp.fc1.weight": torch.cat([block["mlp.fc1.weight"][:keep],
                                                       block["mlp.fc1.weight"][inner:inner + keep]]),
              prefix + "mlp.fc2.weight": block["mlp.fc2.weight"][:, :keep].contiguous()}
    globals_ = {"adaln_t_table": torch.randn(5, 3), "rope.inv_freq": torch.tensor([.31])}
    paths = {}
    for name, state in (("block_state", original), ("folded_mlp", folded), ("source_weights", globals_)):
        paths[name] = tmp_path / (name + ".safetensors")
        save_file(state, str(paths[name]))
    sv = torch.tensor([.6, .6, .8])
    base = sv / (12 - 11 * sv)
    sa = 3 * base / (1 + 2 * base)
    ts = 1 - torch.stack((sv, sa), 1).sort(1, descending=True).values
    p = ts.flatten() * 4
    lo = p.floor().long().clamp(max=3)
    curve = torch.lerp(globals_["adaln_t_table"][lo], globals_["adaln_t_table"][lo + 1], (p - lo)[:, None])
    mods = F.linear(curve, block["adaln_proj.linear.weight"], block["adaln_proj.linear.bias"])
    mods = mods.view(3, 6, 6, width).permute(0, 2, 1, 3).contiguous()
    capture = tmp_path / "capture.safetensors"
    save_file({"sigma_video": sv, "sigma_audio": sa, "temb": curve.view(3, 2, 3), "adaln": mods}, str(capture))
    receipt = {"block": 25, "child": {"file_sha256": converter.file_sha(paths["folded_mlp"])},
               "capture": {"block_weights_sha256": converter.file_sha(paths["block_state"])},
               "header_counts": {"parent_mlp_parameters": 3 * width * inner,
                                  "child_mlp_parameters": 3 * width * keep},
               "measured": {"holdout_scores_replace": {"rel_l2": .3}},
               "capture_deviations": [{"name": "fixture_not_real_media"}]}
    receipt_path = tmp_path / "coreset_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    charts = tmp_path / "charts.safetensors"
    save_file({"chart_in.weight": torch.randn(width, 10) / 10, "chart_in.bias": torch.zeros(width),
               "chart_out.weight": torch.randn(10, width) / 10}, str(charts))
    return dict(**paths, coreset_receipt=receipt_path, captures=[capture], charts=charts,
                heads=heads, head_dim=head_dim, out=tmp_path / "out"), original, folded


def test_converter_preserves_all_source_parts_and_validates_time_reference(tmp_path):
    args, original, folded = fixture(tmp_path)
    body = converter.convert(**args)
    state = load_file(str(args["out"] / "charted_block.safetensors"))
    assert torch.equal(torch.cat([state[f"attn.{k}_proj.weight"] for k in ("q", "k", "v")]),
                       original["blocks.25.attn.qkv_proj.weight"])
    assert torch.equal(torch.cat([state["mlp.gate_proj.weight"], state["mlp.up_proj.weight"]]),
                       folded["blocks.25.mlp.fc1.weight"])
    assert torch.equal(state["h3_time.proj.weight"], original["blocks.25.adaln_proj.linear.weight"])
    spec = ChartedBlockSpec.from_manifest({**body["spec"], "layer_index": 0})
    block = ChartedSourceBlock(spec, 10)
    block.load_state_dict({**state, **load_file(str(args["charts"]))}, strict=True)
    reference = load_file(str(args["out"] / "time_reference.safetensors"))
    assert len(reference["sigma"]) == 4  # two repeated forwards were verified then deduplicated
    torch.testing.assert_close(block.time_path(reference["sigma"]), reference["modulation"], rtol=1e-5, atol=1e-7)
    assert body["time_reference"]["verified"] is True
    assert body["accounting"]["parent_mlp_parameters"] == 4 * body["accounting"]["child_mlp_parameters"]
    assert body["accounting"]["retained_shared_coefficients"] == 5 * 3 + 1
    assert (body["accounting"]["child_block_parameters"] + body["accounting"]["retained_shared_coefficients"]
            == sum(t.numel() for t in state.values()))
    assert body["runtime_requirements"]["source_positions"].startswith("explicit")
    assert body["capture_deviations"] == [{"name": "fixture_not_real_media"}]
    assert "no context tower" in body["spec"]["metadata"]["context"]
    assert block.parameter_count() == spec.parameter_count(10)


def test_converter_without_charts_still_requires_real_paired_taps(tmp_path):
    args, _, _ = fixture(tmp_path)
    args["charts"] = None
    body = converter.convert(**args)
    assert body["runtime_requirements"]["charts"] == "paired host/source tap rows still required"
    assert not (args["out"] / "charts.safetensors").exists()


@pytest.mark.parametrize("mutation", ["curve", "modulation", "folded_digest", "block_digest", "counts", "extra_frozen"])
def test_converter_refuses_mismatches_before_output(tmp_path, mutation):
    args, _, _ = fixture(tmp_path)
    if mutation in {"curve", "modulation"}:
        path = args["captures"][0]
        state = load_file(str(path))
        key = "temb" if mutation == "curve" else "adaln"
        state[key] += .2
        save_file(state, str(path))
    elif mutation in {"folded_digest", "block_digest", "counts"}:
        path = args["coreset_receipt"]
        body = json.loads(path.read_text())
        if mutation == "folded_digest":
            body["child"]["file_sha256"] = "0" * 64
        elif mutation == "block_digest":
            body["capture"]["block_weights_sha256"] = "0" * 64
        else:
            body["header_counts"]["child_mlp_parameters"] += 1
        path.write_text(json.dumps(body), encoding="utf-8")
    else:
        state = load_file(str(args["block_state"]))
        state["blocks.25.unaccounted.weight"] = torch.ones(1)
        save_file(state, str(args["block_state"]))
        body = json.loads(args["coreset_receipt"].read_text())
        body["capture"]["block_weights_sha256"] = converter.file_sha(args["block_state"])
        args["coreset_receipt"].write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        converter.convert(**args)
    assert not args["out"].exists()


def test_converter_preserves_existing_outputs(tmp_path):
    args, _, _ = fixture(tmp_path)
    args["out"].mkdir()
    sentinel = args["out"] / "keep.txt"
    sentinel.write_text("user work")
    with pytest.raises(ValueError, match="never overwritten"):
        converter.convert(**args)
    assert sentinel.read_text() == "user work"


def test_converter_refuses_previously_narrowed_source_coefficients(tmp_path):
    args, _, _ = fixture(tmp_path)
    state = load_file(str(args["source_weights"]))
    state["adaln_t_table"] = state["adaln_t_table"].bfloat16()
    save_file(state, str(args["source_weights"]))
    with pytest.raises(ValueError, match="must be stored in FP32"):
        converter.convert(**args)
    assert not args["out"].exists()
