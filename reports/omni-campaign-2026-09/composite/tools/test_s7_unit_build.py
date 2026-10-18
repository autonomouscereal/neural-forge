"""Tests for s7_unit_build.py 1.2.0 (lane M3, Claude, 2026-10-10) and 1.3.0 (PREP-B). Server, CPU only:

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 PYTHONPATH=$CHIMERA_OMNI_SRC/src \
    OMP_NUM_THREADS=8 nice -n 19 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \
        -p no:cacheprovider -q test_s7_unit_build.py

Synthetic: the qvl parent-width reader (true width from the header, refusal on mixed widths,
refusal without --parent), the HY3D time token against an independent restatement of the
Hunyuan3D-2.1 formula. Real artefacts (skipped when absent): every rebuilt unit under
$OC/composite/s7/units loads through c16_install_media.load_container, the H3 and QIMG
time gates verify at c16's default 1e-5, QVL32 routes only type 6, HY3D ships its
time-token row contract and reference.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402

HERE = Path(__file__).resolve().parent
os.environ.setdefault("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9")
UNITS = Path(os.environ.get("S7_UNITS", "/home/cereal/omni-campaign/composite/s7/units"))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build = _load(HERE / "s7_unit_build.py", "s7_unit_build")
C16 = build.C16


def _vision_parent(path: Path, widths: list[int], prefix: str = "visual.") -> Path:
    state = {}
    for i, w in enumerate(widths):
        state[f"{prefix}blocks.{i}.mlp.linear_fc1.weight"] = torch.zeros(w, 8)
        state[f"{prefix}blocks.{i}.mlp.linear_fc2.weight"] = torch.zeros(8, w)
    state["model.embed_tokens.weight"] = torch.zeros(4, 8)
    save_file(state, str(path))
    return path


def test_parent_vision_mlp_reads_the_true_width(tmp_path) -> None:
    info = build.parent_vision_mlp(_vision_parent(tmp_path / "p.safetensors", [12] * 5), 3)
    assert info["intermediate_size"] == 12 and info["hidden_size"] == 8 and info["depth"] == 5
    assert info["prefix"] == "visual."
    info = build.parent_vision_mlp(_vision_parent(tmp_path / "q.safetensors", [6] * 3, prefix="model.visual."), 1)
    assert info["intermediate_size"] == 6 and info["prefix"] == "model.visual."


def test_parent_vision_mlp_refuses_mixed_widths(tmp_path) -> None:
    with pytest.raises(C16.Refusal):
        build.parent_vision_mlp(_vision_parent(tmp_path / "p.safetensors", [12, 12, 10]), 0)


def test_build_qvl_refuses_without_a_parent(tmp_path) -> None:
    with pytest.raises(C16.Refusal):
        build.build_qvl(tmp_path / "nochild", tmp_path / "out", 13, parent=None)


def test_hy3d_time_token_matches_the_source_formula(tmp_path) -> None:
    g = torch.Generator().manual_seed(0)
    dims, inner = 16, 24
    w0, b0 = torch.randn(inner, dims, generator=g), torch.randn(inner, generator=g)
    w2, b2 = torch.randn(dims, inner, generator=g), torch.randn(dims, generator=g)
    parent = tmp_path / "hy3d.safetensors"
    save_file({"model.t_embedder.mlp.0.weight": w0.half(), "model.t_embedder.mlp.0.bias": b0.half(),
               "model.t_embedder.mlp.2.weight": w2.half(), "model.t_embedder.mlp.2.bias": b2.half()}, str(parent))
    sigma = torch.tensor([0.0, 0.37, 1.0])
    got = build.hy3d_time_token(parent, sigma)["time_token"]
    # independent restatement: sin block first, then cos; frequencies 1e4^(-k/half); no *1000
    half = dims // 2
    freqs = torch.tensor([math.pow(10000.0, -k / half) for k in range(half)], dtype=torch.float64)
    phase = sigma.double()[:, None] * freqs[None]
    emb = torch.cat((phase.sin(), phase.cos()), dim=-1)
    h = torch.nn.functional.gelu(emb @ w0.half().double().T + b0.half().double())
    want = h @ w2.half().double().T + b2.half().double()
    assert torch.allclose(got.double(), want, rtol=1e-5, atol=1e-4)


def test_hy3d_reference_vs_capture_is_a_recorded_diagnostic(tmp_path) -> None:
    g = torch.Generator().manual_seed(1)
    dims, inner = 16, 24
    parent = tmp_path / "hy3d.safetensors"
    save_file({"model.t_embedder.mlp.0.weight": torch.randn(inner, dims, generator=g).half(),
               "model.t_embedder.mlp.0.bias": torch.randn(inner, generator=g).half(),
               "model.t_embedder.mlp.2.weight": torch.randn(dims, inner, generator=g).half(),
               "model.t_embedder.mlp.2.bias": torch.randn(dims, generator=g).half()}, str(parent))
    sigma = torch.tensor([0.5, 0.5, 0.25])
    token = build.hy3d_time_token(parent, sigma)["time_token"]
    capture = tmp_path / "block0.safetensors"
    save_file({"time_token": token.view(3, 1, dims).half(), "sigma": sigma.half()}, str(capture))
    report = build.hy3d_reference_vs_capture(parent, capture)
    assert report["checked"] is True and report["sigma_key"] == "sigma"
    assert max(report["rel_l2_fp32_reference_vs_captured_f16_time_token"]) < 1e-2  # only the F16 rounding of the capture
    wrong = tmp_path / "wrong.safetensors"
    save_file({"time_token": (token * -1).view(3, 1, dims).half(), "sigma": sigma.half()}, str(wrong))
    assert min(build.hy3d_reference_vs_capture(parent, wrong)["rel_l2_fp32_reference_vs_captured_f16_time_token"]) > 1.0
    assert build.hy3d_reference_vs_capture(parent, tmp_path / "absent.safetensors")["checked"] is False


# ---------------------------------------------------------------- real rebuilt units


def _unit(path: str) -> Path:
    unit = UNITS / path
    if not (unit / "charted_block.json").is_file():
        pytest.skip(f"{unit} not built here")
    return unit


def _cfg(hidden: int = 5120):
    class Cfg:
        hidden_size = hidden
        native_type_count = 8
    return Cfg()


def test_rebuilt_h3_unit_gates_at_default_tolerance() -> None:
    unit = _unit("h3/h3_dit_L25_curve6")
    assert (unit / C16.TIME_REFERENCE_FILENAME).is_file()
    assert not (unit / "h3_time_reference.safetensors").exists()
    container = C16.load_container(unit, _cfg())
    container["charts"] = {k: torch.zeros(v) for k, v in container["spec"].parameter_shapes("", 5120).items() if k.startswith("chart_")}
    container["hidden"] = 5120
    report = C16.verify_time_reference(container, 5120, rel_tol=1.0e-5)
    assert report["verified"] is True and report["modulation"] == "h3_curve6"
    assert container["tensor_sha256_check"][C16.TIME_REFERENCE_FILENAME]["verified"] is True


def test_rebuilt_qimg_unit_gates_at_default_tolerance_against_the_fp32_reference() -> None:
    unit = _unit("qimg/qimg_mmdit_L30_img")
    body = json.loads((unit / "charted_block.json").read_text(encoding="utf-8"))
    assert "FP32" in body["time_reference"]["source"]
    container = C16.load_container(unit, _cfg())
    container["charts"] = {k: torch.zeros(v) for k, v in container["spec"].parameter_shapes("", 5120).items() if k.startswith("chart_")}
    report = C16.verify_time_reference(container, 5120, rel_tol=1.0e-5)
    assert report["verified"] is True
    assert body["time_reference"]["bf16_capture_gate_diagnostic"]["rel_l2"] > 1.0e-3  # the old floor, kept as a diagnostic


def test_rebuilt_qvl_unit_routes_ocr_only_with_the_parent_width() -> None:
    unit = _unit("qvl32/qvl32_vit_L13")
    body = json.loads((unit / "charted_block.json").read_text(encoding="utf-8"))
    assert body["spec"]["type_ids"] == [6]
    assert body["accounting"]["parent_intermediate"] == body["accounting"]["parent_intermediate_source"]["intermediate_size"]
    assert "parent_intermediate_assumed" not in body["accounting"]
    container = C16.load_container(unit, _cfg())
    assert container["spec"].type_ids == (6,)


def test_rebuilt_hy3d_unit_ships_its_time_token_row_contract() -> None:
    unit = _unit("hy3d/hy3d_dit_L10")
    body = json.loads((unit / "charted_block.json").read_text(encoding="utf-8"))
    row = body["runtime_requirements"]["hy3d_time_token_row"]
    views = C16.read_single(unit / row["reference_file"])
    assert {k: C16.view_hash(v) for k, v in views.items()} == row["reference_tensor_sha256"]
    assert tuple(views["time_token"].shape) == (21, int(body["spec"]["width"]))
    assert body["s7_unit"]["structural_limitation_owner_decision"]["decision_needed_from"] == "Cyril"
    C16.load_container(unit, _cfg())
    # PREP-B (1.3.0): the pairing is implemented, and charts present on this unit were fitted WITH the time row
    if body["s7_unit"]["tool_version"] >= "1.3.0":
        assert set(row["pairing_implemented_by"]) == {"s7_pairs_parent", "s7_host_rows", "s7_chart_fit"}
    if "charts" in body:
        assert body["charts"]["time_rows"]["rows"] >= 1 and body["charts"]["time_rows"]["requirement"] == "hy3d_time_token_row"
