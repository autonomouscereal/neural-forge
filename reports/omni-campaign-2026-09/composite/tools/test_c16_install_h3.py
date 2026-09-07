"""H3 S7 container mechanics only; no synthetic fixture is media capability evidence."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file
import torch.nn.functional as F

from chimera_omni.charted import ChartedBlockSpec, H3_FP32_PARAMETERS
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint, read_safetensors
from chimera_omni.modeling.charted import ChartedSourceBlock
from test_c16_install_media import tool, fixtures, run_tool, manifest_of, header_total


@pytest.fixture
def h3_container(tmp_path):
    """Source-contract-shaped state and independently calculated capture reference."""
    spec = ChartedBlockSpec(layer_index=0, source="h3", component="h3.dit.block.mlp", type_ids=(2, 3),
        width=12, heads=2, kv_heads=2, head_dim=8, mlp_width=3, modulation="h3_curve6",
        norm_eps=1e-5, h3_curve_grid=5, h3_curve_dim=3, h3_rope_freqs=1,
        metadata={"source_position_schema": "h3_thw_v1"})
    hidden = fixtures.DEFAULT_HIDDEN
    torch.manual_seed(437)
    module = ChartedSourceBlock(spec, hidden).eval()
    with torch.no_grad():
        for name, value in module.named_parameters(): value.copy_(torch.randn_like(value) * .13)
        for norm in (module.attn_norm, module.mlp_norm, module.attn.q_norm, module.attn.k_norm): norm.weight.add_(1)
        module.attn.rope_inv_freq.fill_(.313)
    all_state = module.state_dict()
    state = {name: value for name, value in all_state.items() if not name.startswith("chart_")}
    charts = {name: value for name, value in all_state.items() if name.startswith("chart_")}
    sigma = torch.tensor([.2, .3, .6, .9], dtype=torch.float32)
    pos = (1 - sigma) * (spec.h3_curve_grid - 1)
    lower = pos.floor().long().clamp(max=spec.h3_curve_grid - 2)
    curve = torch.lerp(state["h3_time.curve_table"][lower], state["h3_time.curve_table"][lower + 1], (pos - lower)[:, None])
    modulation = F.linear(curve, state["h3_time.proj.weight"], state["h3_time.proj.bias"]).view(-1, 3, 6, spec.width)
    reference = {"sigma": sigma, "curve": curve, "modulation": modulation}
    directory = tmp_path / "h3_native"
    hashes = {name: fixtures.save_single(tensors, directory / name) for name, tensors in
              (("charted_block.safetensors", state), ("charts.safetensors", charts), (tool.TIME_REFERENCE_FILENAME, reference))}
    shared = sum(state[name].numel() for name in tool.H3_SHARED_COEFFICIENTS)
    block_count = sum(value.numel() for value in state.values()) - shared
    body = {"schema": tool.BLOCK_SCHEMA, "spec": {key: value for key, value in spec.to_manifest().items() if key != "layer_index"},
        "provenance": {"fixture": True, "source_globals_tensor_sha256": {
            source: hashes["charted_block.safetensors"][target] for target, source in tool.H3_SHARED_COEFFICIENTS.items()}},
        "accounting": {"child_mlp_parameters": spec.mlp_parameter_count(), "parent_mlp_parameters": 4 * spec.mlp_parameter_count(),
            "child_block_parameters": block_count, "parent_block_parameters": block_count + 3 * spec.mlp_parameter_count(),
            "retained_shared_coefficients": shared},
        "time_reference": {"file": tool.TIME_REFERENCE_FILENAME, "modality_order": ["video", "text", "audio"]},
        "runtime_requirements": {"source_positions": tool.H3_SOURCE_POSITIONS, "coordinate_inference": tool.H3_COORDINATE_INFERENCE,
            "capability_claim": "none: synthetic fixture"},
        "capture_deviations": [{"name": "fixture_not_real_media"}], "tensor_sha256": hashes, "insert_after": 1}
    fixtures.write_json(directory / "charted_block.json", body)
    return directory, body, spec, hidden


def load(fixture):
    directory, _body, _spec, hidden = fixture
    return tool.load_container(directory, SimpleNamespace(hidden_size=hidden, native_type_count=9))


def rewrite(fixture, filename, tensors):
    directory, body, _spec, _hidden = fixture
    body["tensor_sha256"][filename] = fixtures.save_single(tensors, directory / filename)
    fixtures.write_json(directory / "charted_block.json", body)


def test_h3_capture_curve_and_three_modality_reference_are_verified(h3_container):
    assert tool.H3_FP32_TENSORS == frozenset(H3_FP32_PARAMETERS)
    value = load(h3_container)
    report = tool.verify_time_reference(value, h3_container[3], rel_tol=1e-5)
    assert report["verified"] is True
    assert report["reference_shape"] == [4, 3, 6, 12]
    assert report["modality_order"] == ["video", "text", "audio"]
    assert report["curve_rel_l2"] < 1e-6
    assert len(report["reference_chunk_norms"]) == 3
    assert all(len(row) == 6 for row in report["reference_chunk_norms"])
    assert value["h3_source_contract"]["retained_shared_coefficients"] == 16
    assert value["h3_source_contract"]["position_mapping_verified"] is False


@pytest.mark.parametrize("mutation", ["modulation", "curve", "missing_curve", "ace_shape", "sigma_rank", "sigma_nan", "sigma_range", "curve_shape", "modulation_nan", "curve_nan"])
def test_h3_reference_tampering_cannot_be_hidden_by_updated_hashes(h3_container, mutation):
    directory, _body, _spec, hidden = h3_container
    state = load_file(str(directory / tool.TIME_REFERENCE_FILENAME))
    if mutation in ("modulation", "curve"): state[mutation] += .2
    elif mutation == "missing_curve": del state["curve"]
    elif mutation == "ace_shape": state["modulation"] = state["modulation"][:, 0].contiguous()
    elif mutation == "sigma_rank": state["sigma"] = state["sigma"].view(2, 2)
    elif mutation == "sigma_nan": state["sigma"][0] = float("nan")
    elif mutation == "sigma_range": state["sigma"][0] = 1.1
    elif mutation == "curve_shape": state["curve"] = state["curve"][:, :2].contiguous()
    elif mutation == "modulation_nan": state["modulation"][0, 0, 0, 0] = float("nan")
    else: state["curve"][0, 0] = float("nan")
    rewrite(h3_container, tool.TIME_REFERENCE_FILENAME, state)
    with pytest.raises(tool.Refusal): tool.verify_time_reference(load(h3_container), hidden, rel_tol=1e-5)


@pytest.mark.parametrize("mutation", ["source_schema", "source_positions", "inference", "order", "shared_count", "missing_count", "child_count", "mlp_count", "parent_count", "source_digest", "missing_source_digest", "missing_tensor_digest"])
def test_h3_requires_coordinate_contract_and_header_counted_coefficients(h3_container, mutation):
    directory, body, _spec, _hidden = h3_container
    if mutation == "source_schema": body["spec"]["metadata"]["source_position_schema"] = "thwr_v1"
    elif mutation == "source_positions": body["runtime_requirements"].pop("source_positions")
    elif mutation == "inference": body["runtime_requirements"]["coordinate_inference"] = "infer from native positions"
    elif mutation == "order": body["time_reference"]["modality_order"] = ["text", "video", "audio"]
    elif mutation == "shared_count": body["accounting"]["retained_shared_coefficients"] *= 4  # bytes are not elements
    elif mutation == "missing_count": del body["accounting"]["retained_shared_coefficients"]
    elif mutation == "child_count": body["accounting"]["child_block_parameters"] += 16
    elif mutation == "mlp_count": body["accounting"]["child_mlp_parameters"] += 1
    elif mutation == "parent_count": body["accounting"]["parent_block_parameters"] += 1
    elif mutation == "source_digest": body["provenance"]["source_globals_tensor_sha256"]["rope.inv_freq"] = "0" * 64
    elif mutation == "missing_source_digest": body["provenance"].pop("source_globals_tensor_sha256")
    else: body["tensor_sha256"]["charted_block.safetensors"].pop("h3_time.curve_table")
    fixtures.write_json(directory / "charted_block.json", body)
    with pytest.raises(tool.Refusal): load(h3_container)


@pytest.mark.parametrize("key", sorted(tool.H3_FP32_TENSORS))
def test_h3_cannot_relabel_narrowed_time_or_rope_as_source_fp32(h3_container, key):
    directory, body, _spec, _hidden = h3_container
    state = load_file(str(directory / "charted_block.safetensors"))
    state[key] = state[key].to(torch.bfloat16)
    rewrite(h3_container, "charted_block.safetensors", state)
    if key in tool.H3_SHARED_COEFFICIENTS:
        body["provenance"]["source_globals_tensor_sha256"][tool.H3_SHARED_COEFFICIENTS[key]] = body["tensor_sha256"]["charted_block.safetensors"][key]
        fixtures.write_json(directory / "charted_block.json", body)
    with pytest.raises(tool.Refusal, match="stored F32"): load(h3_container)


@pytest.mark.parametrize("tolerance", [float("nan"), float("inf"), 0, -1, 1e-4])
def test_h3_time_evidence_tolerance_cannot_be_waived(h3_container, tolerance):
    with pytest.raises(tool.Refusal, match="tolerance"):
        tool.verify_time_reference(load(h3_container), h3_container[3], rel_tol=tolerance)


def test_h3_insert_counts_and_preserves_fp32_coefficients_inside_bf16_child(h3_container, tmp_path):
    directory, body, spec, hidden = h3_container
    child, out = tmp_path / "child", tmp_path / "installed"
    fixtures.write_child(child, hidden=hidden)
    assert run_tool("--child", child, "--out", out, "--blocks", directory, "--pair-by", "explicit") == 0
    manifest = manifest_of(out)
    placement = manifest["blocks"]["placements"][0]
    assert placement["parameters"] == spec.parameter_count(hidden)
    assert header_total(out) == header_total(child) + spec.parameter_count(hidden)
    assert manifest["accounting"]["charted_dtype"] == "mixed"
    assert placement["runtime_requirements"] == body["runtime_requirements"]
    assert placement["capture_deviations"] == body["capture_deviations"]
    assert placement["h3_source_contract"]["source_positions_required"] is True
    assert placement["h3_source_contract"]["position_mapping_verified"] is False
    views = read_safetensors(out)
    source = read_safetensors(directory / "charted_block.safetensors")
    prefix = placement["tensor_prefix"]
    for key in tool.H3_FP32_TENSORS:
        assert views[prefix + key].dtype == "F32"
        assert views[prefix + key].sha256() == source[key].sha256()
        assert placement["narrowing"][key]["max_abs_narrowing"] == 0
    assert views[prefix + "mlp.gate_proj.weight"].dtype == "BF16"
    whole = manifest["components"][0]["whole_block"]
    assert whole["retained_shared_coefficients"] == 16
    assert whole["child_with_retained_coefficients_params"] == sum(v.numel() for v in load(h3_container)["tensors"].values())
    model, cfg, _ = load_chimera_checkpoint(out)
    loaded = model.blocks[placement["layer_index"]].charted
    reference = load_file(str(directory / tool.TIME_REFERENCE_FILENAME))
    torch.testing.assert_close(loaded.time_path(reference["sigma"]), reference["modulation"], rtol=1e-5, atol=1e-7)
    # Text-only still works, while actual H3 writes cannot infer source THW positions.
    ids = torch.randint(0, 200, (1, 5))
    with torch.no_grad(): assert torch.isfinite(model.eval()(ids)["logits"]).all()
    with pytest.raises(ValueError, match="positions"):
        loaded(torch.randn(1, 5, hidden), torch.tensor([[0, 0, 3, 2, 2]]),
               torch.tensor([[0, 0, .3, .6315789, .6315789]]), torch.ones(1, 5, dtype=torch.bool))


def test_h3_blend_refuses_before_any_output(h3_container, tmp_path):
    fixtures.write_child(tmp_path / "child", hidden=h3_container[3])
    out = tmp_path / "blend"
    assert run_tool("--child", tmp_path / "child", "--out", out, "--blocks", h3_container[0],
                    "--pair-by", "explicit", "--arm", "blend", "--blend-uniform", "pad") == tool.EXIT_REFUSED
    assert not out.exists()


def test_peak_rss_is_actual_platform_measurement():
    value = tool.peak_rss_bytes()
    assert value is None or isinstance(value, int) and value > 0
    if sys.platform in ("linux", "win32", "darwin"): assert value is not None


def test_actual_converter_fixture_contract_is_accepted_when_available(tmp_path, monkeypatch):
    path = Path(os.environ.get("C16_H3_CONVERTER_TESTS", str(Path(__file__).with_name("test_c16_convert_h3.py"))))
    if not path.is_file(): pytest.skip("separate H3 converter not present in this checkout")
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("h3_converter_fixture_contract", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    arguments, _original, _folded = module.fixture(tmp_path)
    module.converter.convert(**arguments)
    container = tool.load_container(arguments["out"], SimpleNamespace(hidden_size=10, native_type_count=9))
    report = tool.verify_time_reference(container, 10, rel_tol=1e-5)
    assert report["verified"] and report["curve_rel_l2"] <= 1e-5
