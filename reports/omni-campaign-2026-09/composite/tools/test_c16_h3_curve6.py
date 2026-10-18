"""Tests for c16_install_media.py 1.2.0's h3_curve6 support (lane M3, Claude, 2026-10-10).

Run on the AI server, CPU only, against the tree that HAS h3_curve6 (tree_ef1ffb9):

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 \
    MEDIAFOLD_ROOT=/home/cereal/omni-campaign/media OC_LEDGER_ROOT=/home/cereal/omni-campaign/ledger \
    C16_V110=<path to the 1.1.0 copy> PYTHONPATH=$CHIMERA_OMNI_SRC/src OMP_NUM_THREADS=8 \
    nice -n 19 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_c16_h3_curve6.py

What is asserted:
  (1) an h3_curve6 container WITH time_reference.safetensors installs under --charted-dtype match;
      its four H3_FP32_PARAMETERS are F32 in the written shards (everything else in the child's
      storage dtype), the manifest lists them, the time gate verified, and the output LOADS
      through the tree's load_chimera_checkpoint (which refuses BF16 H3 coefficients) with the
      text residual bit-identical to the child;
  (2) a tampered or missing h3 reference, and a wrong-shape one, are refused (exit 3) under
      --strict-time-reference; since c16 1.3.0 (owner policy 2026-10-06, D5) the tampered one
      (a finite rel_l2 above the tolerance) installs by default with a recorded warning, while
      missing / wrong-shape stay refused without the flag;
  (3) the blend arm refuses an h3_curve6 block;
  (4) 1.1.0 refused the same container (the gap this version closes);
  (5) adaLN6 and modulation-none installs are byte-identical between 1.1.0 and 1.2.0
      (every written tensor's sha256, the interface/band records and the block placements).
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "c16_install_media.py"
FIXTURES = HERE / "c16_media_fixtures.py"
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9"))
MEDIAFOLD_ROOT = Path(os.environ.get("MEDIAFOLD_ROOT", "/home/cereal/omni-campaign/media"))
LEDGER_ROOT = Path(os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))
C16_V110 = Path(os.environ.get("C16_V110", str(HERE / ".bak-m3-20261010" / "c16_install_media.py")))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from chimera_omni import charted as charted_spec  # noqa: E402

if "h3_curve6" not in getattr(charted_spec, "MODULATION_KINDS", ()):
    pytest.skip("this chimera tree has no h3_curve6 (use tree_ef1ffb9)", allow_module_level=True)

from chimera_omni.charted import H3_FP32_PARAMETERS, ChartedBlockSpec  # noqa: E402
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint, read_index, read_safetensors  # noqa: E402
from chimera_omni.modeling.charted import ChartedSourceBlock  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load(TOOL, "c16_install_media_v120")
fixtures = _load(FIXTURES, "c16_media_fixtures")
COMMON = ["--chimera-src", str(CHIMERA_SRC), "--mediafold-root", str(MEDIAFOLD_ROOT), "--ledger-root", str(LEDGER_ROOT)]

H3_FIELDS = {
    "source": "h3", "component": "h3.dit.block.mlp", "type_ids": [2, 3], "width": 16, "heads": 2, "kv_heads": 2,
    "head_dim": 8, "mlp_width": 12, "mlp_kind": "swiglu", "modulation": "h3_curve6", "norm_kind": "rms",
    "norm_affine": True, "norm_eps": 1.0e-5, "qk_norm": True, "attn_bias": False, "rope_theta": 0.0,
    "sliding_window": 0, "time_freq_dim": 256, "time_hidden": 0, "time_factor": 1000.0, "time_embed_r": False,
    "h3_curve_grid": 9, "h3_curve_dim": 3, "h3_rope_freqs": 1, "delta_scale": 1.0, "chart_in_bias": True,
    "rung": "fixture", "source_depth_fraction": 0.5, "metadata": {"fixture": "h3_curve6"},
}
SIGMAS = (0.3, 0.6315789222717285, 0.9)


def run(module, *args) -> int:
    return module.main([*COMMON, *(str(a) for a in args)])


def write_h3_container(root: Path, hidden: int, *, seed: int = 11, reference: str = "good") -> Path:
    spec = ChartedBlockSpec(layer_index=0, **{**H3_FIELDS, "type_ids": tuple(H3_FIELDS["type_ids"])})
    spec.validate()
    gen = torch.Generator().manual_seed(seed)
    state, charts = {}, {}
    for name, shape in spec.parameter_shapes("", hidden).items():
        target = charts if name.startswith("chart_") else state
        if name.endswith("norm.weight"):
            value = torch.ones(shape)
        elif name == "attn.rope_inv_freq":
            value = torch.rand(shape, generator=gen) + 0.1
        else:
            value = torch.randn(shape, generator=gen) * 0.3 / (shape[-1] ** 0.5)
        # the tensors a real container ships: FP32 coefficients, BF16-representable rest
        target[name] = value if name in H3_FP32_PARAMETERS else value.to(torch.bfloat16).to(torch.float32)
    block = ChartedSourceBlock(spec, hidden).eval()
    block.load_state_dict({**state, **charts}, strict=True)
    sigma = torch.tensor(SIGMAS, dtype=torch.float32)
    with torch.no_grad():
        modulation = block.time_path(sigma).to(torch.float32)
    if reference == "tampered":
        modulation = modulation * 1.05
    elif reference == "wrong_shape":
        modulation = modulation[:, 0]
    out = root / "h3_curve6_fixture"
    block_table = fixtures.save_single(state, out / "charted_block.safetensors")
    chart_table = fixtures.save_single(charts, out / "charts.safetensors")
    tables = {"charted_block.safetensors": block_table, "charts.safetensors": chart_table}
    if reference != "missing":
        tables[tool.TIME_REFERENCE_FILENAME] = fixtures.save_single(
            {"sigma": sigma, "modulation": modulation}, out / tool.TIME_REFERENCE_FILENAME)
    mlp = spec.mlp_parameter_count()
    body = {
        "schema": tool.BLOCK_SCHEMA, "spec": dict(H3_FIELDS),
        "provenance": {"fixture": True}, "holdout": {},
        "accounting": {"parent_mlp_parameters": 4 * mlp, "child_mlp_parameters": mlp},
        "time_reference": {"file": tool.TIME_REFERENCE_FILENAME, "sigma": list(SIGMAS), "modality_order": ["video", "text", "audio"]},
        "insert_after": None, "tensor_sha256": tables,
    }
    fixtures.write_json(out / "charted_block.json", body)
    return out


@pytest.fixture(scope="module")
def fixture_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("m3")
    report = fixtures.write_fixture_set(root, seed=0, residual_mode="none")
    report["accounting_parent"] = fixtures.write_child(root / "accounting_parent", seed=0, intermediate=128)
    return root, report


def assert_json_close(a, b, where: str) -> None:
    if isinstance(a, float) or isinstance(b, float):
        assert isinstance(a, (int, float)) and isinstance(b, (int, float)), where
        assert a == pytest.approx(b, rel=1e-9, abs=1e-12), (where, a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), (where, sorted(set(a) ^ set(b)))
        for key in a:
            assert_json_close(a[key], b[key], f"{where}.{key}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), where
        for index, (x, y) in enumerate(zip(a, b)):
            assert_json_close(x, y, f"{where}[{index}]")
    else:
        assert a == b, (where, a, b)


def header_dtypes(directory: Path) -> dict[str, str]:
    return {name: str(view.dtype) for name, view in read_safetensors(directory, verify=False).items()}


def test_h3_curve6_installs_under_match_with_fp32_coefficients_and_loads(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    hidden = report["hidden"]
    container = write_h3_container(tmp_path / "blocks", hidden)
    out = tmp_path / "h3_match"
    assert run(tool, "--child", root / "child", "--out", out, "--blocks", container,
               "--charted-dtype", "match", "--accounting-parent-directory", root / "accounting_parent") == 0
    manifest = json.loads((out / "composite_manifest.json").read_text(encoding="utf-8"))
    placement = manifest["blocks"]["placements"][0]
    prefix = placement["tensor_prefix"]
    dtypes = header_dtypes(out)
    for name in H3_FP32_PARAMETERS:
        assert "F32" in dtypes[prefix + name].upper() or "FLOAT32" in dtypes[prefix + name].upper(), (name, dtypes[prefix + name])
        assert placement["narrowing"][name]["kept_fp32"] == "H3_FP32_PARAMETERS"
    others = [k for k in dtypes if k.startswith(prefix) and k[len(prefix):] not in H3_FP32_PARAMETERS]
    assert others and all("BF16" in dtypes[k].upper() or "BFLOAT16" in dtypes[k].upper() for k in others)
    assert manifest["accounting"]["charted_fp32_kept"] == sorted(prefix + n for n in H3_FP32_PARAMETERS)
    gate = placement["time_reference"]
    assert gate["verified"] is True and gate["modulation"] == "h3_curve6" and len(gate["reference_chunk_norms"]) == 6
    assert gate["rel_l2"] < 1e-6 and len(gate["modality_rel_l2"]) == 3
    assert placement["tensor_sha256_check"][tool.TIME_REFERENCE_FILENAME]["verified"] is True
    model, cfg, _ = load_chimera_checkpoint(out)  # the tree's loader refuses BF16 H3 coefficients
    layer = placement["layer_index"]
    assert cfg.block_kind(layer) == "charted"
    assert model.blocks[layer].charted.h3_time.curve_table.dtype == torch.float32
    child, _, _ = load_chimera_checkpoint(root / "child")
    ids = torch.randint(0, 256, (1, 9), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        assert torch.equal(child.eval()(ids)["hidden"], model.eval()(ids)["hidden"])


@pytest.mark.parametrize("variant", ["tampered", "missing", "wrong_shape"])
def test_bad_h3_references_are_refused_under_strict(fixture_set, tmp_path, variant) -> None:
    root, report = fixture_set
    container = write_h3_container(tmp_path / "blocks", report["hidden"], reference=variant)
    out = tmp_path / f"h3_{variant}"
    assert run(tool, "--child", root / "child", "--out", out, "--blocks", container, "--charted-dtype", "match",
               "--strict-time-reference") == 3
    assert not (out / "composite_manifest.json").exists()
    if variant == "tampered":
        # c16 1.3.0 (owner policy 2026-10-06, D5): a finite rel_l2 above the tolerance is RECORDED by default
        out = tmp_path / "h3_tampered_default"
        assert run(tool, "--child", root / "child", "--out", out, "--blocks", container, "--charted-dtype", "match") == 0
        manifest = json.loads((out / "composite_manifest.json").read_text(encoding="utf-8"))
        gate = manifest["blocks"]["placements"][0]["time_reference"]
        assert gate["verified"] is False and gate["policy"] == "diagnostic" and gate["modulation"] == "h3_curve6"
        assert gate["rel_l2"] == pytest.approx(0.05 / 1.05, rel=1e-3) and len(gate["modality_rel_l2"]) == 3
        assert [w["kind"] for w in manifest["warnings"]] == ["time_reference"]
    else:
        # missing / wrong-shape references are structural: refused without --strict too
        out = tmp_path / f"h3_{variant}_default"
        assert run(tool, "--child", root / "child", "--out", out, "--blocks", container, "--charted-dtype", "match") == 3
        assert not (out / "composite_manifest.json").exists()


def test_blend_refuses_an_h3_curve6_block(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    container = write_h3_container(tmp_path / "blocks", report["hidden"])
    assert run(tool, "--child", root / "child", "--out", tmp_path / "blend", "--blocks", container, "--arm", "blend",
               "--blend-uniform", "pad") == 3


def _v110():
    if not C16_V110.is_file():
        pytest.skip(f"no c16 1.1.0 copy at {C16_V110}")
    module = _load(C16_V110, "c16_install_media_v110")
    assert module.TOOL_VERSION == "1.1.0"
    return module


def test_v110_refused_the_same_h3_container(fixture_set, tmp_path) -> None:
    old = _v110()
    root, report = fixture_set
    container = write_h3_container(tmp_path / "blocks", report["hidden"])
    assert run(old, "--child", root / "child", "--out", tmp_path / "old", "--blocks", container, "--charted-dtype", "match") == 3


def test_adaln6_and_none_installs_are_byte_identical_to_v110(fixture_set, tmp_path) -> None:
    old = _v110()
    root, _report = fixture_set
    blocks = [root / "blocks" / n for n in ("ace_L12", "qvl32_vit_Lmid", "qimg_Lmid")]
    common = ["--child", root / "child", "--blocks", *blocks, "--tap-rows", root / "tap",
              "--c15", root / "c15", "--c11", root / "c11", "--c18", root / "c18" / "ace_lm_music_codes.safetensors",
              "--accounting-parent-directory", root / "accounting_parent"]
    for dtype in ("match", "fp32"):
        out_old, out_new = tmp_path / f"old_{dtype}", tmp_path / f"new_{dtype}"
        assert run(old, *common, "--out", out_old, "--charted-dtype", dtype) == 0
        assert run(tool, *common, "--out", out_new, "--charted-dtype", dtype) == 0
        a, b = read_index(out_old), read_index(out_new)
        assert a["metadata"]["tensor_sha256"] == b["metadata"]["tensor_sha256"]
        assert a["weight_map"] == b["weight_map"]
        ma = json.loads((out_old / "composite_manifest.json").read_text(encoding="utf-8"))
        mb = json.loads((out_new / "composite_manifest.json").read_text(encoding="utf-8"))
        # tensors are byte-equal (above); the records may differ only in the last bits of
        # multithreaded fp64 lstsq residual floats, so floats compare at rel 1e-9
        for key in ("interface", "band", "components", "narrowing"):
            assert_json_close(ma[key], mb[key], key)
        assert_json_close(ma["blocks"]["placements"], mb["blocks"]["placements"], "placements")
        assert "charted_fp32_kept" not in mb["accounting"]
        assert ma["accounting"]["output_header_parameters"] == mb["accounting"]["output_header_parameters"]
