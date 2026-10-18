"""Tests for c16_install_media.py (run on the AI server, CPU only).

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/omni-composite/tree_2ec15a8 \
    MEDIAFOLD_ROOT=/home/cereal/omni-campaign/media OC_LEDGER_ROOT=/home/cereal/omni-campaign/ledger \
    PYTHONPATH=$CHIMERA_OMNI_SRC/src /home/cereal/model-surgery-lab/venv/bin/python -m pytest \
        -o addopts="" -p no:cacheprovider -q test_c16_install_media.py

Run-book row tests (tool table, c16_install_media.py):
  (1) `--arm blend` output is qwen35-exportable through the gguf_names oracle
      (`require_qwen35_exportable` + `qwen35_export_plan(cfg).covers(state names)`);
  (2) `--arm insert` keeps the chimera-omni suite green (that suite runs on the tree; here
      the insert output LOADS through `load_chimera_checkpoint` and runs);
  (3) header count = S5 child + blocks + charts exactly (re-derived from the written headers);
  (4) band rows written == C18 table rows.
Task tests: synthetic fixtures round-trip into a loadable checkpoint whose header count
equals the accounting; the zero-media control equals the input trunk byte-for-byte (which
is also the streamed writer's parity proof against `write_safetensors`).
Review findings (2026-09-06), one test each: the captured-modulation gate (a dropped
`time_embed_r`, a tampered or missing reference), blend components below 4x registering as
`--passed-through`, the reachable register line against an accounting parent, the free-space
guard, non-zero type scale rows, pairwise band overlaps, declared tensor_sha256 verification,
`--overwrite` scope, the SiTU-trunk blend refusal, the H3 / HY3D presets with explicit
pairing. Plus the ACE c16_transplant_run converter against mediafold's own TransplantedBlock
driven by BOTH timestep paths, and (server-only, skipped elsewhere) the real c16_L12_v4 run.
1.3.0 (owner policy 2026-10-06, S7 decision D5): the captured-modulation measurement is a
diagnostic -- the tampered / dropped-time_embed_r containers install with `verified: false`
and a manifest warning by default, and refuse under `--strict-time-reference`
(test_a_time_path_..._is_recorded_unless_strict, test_verify_time_reference_records_by_default_
and_raises_when_strict, test_ace_converter_records_a_time_path_mismatch_unless_strict).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
TOOL = HERE / "c16_install_media.py"
FIXTURES = HERE / "c16_media_fixtures.py"
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/omni-composite/tree_2ec15a8"))
MEDIAFOLD_ROOT = Path(os.environ.get("MEDIAFOLD_ROOT", "/home/cereal/omni-campaign/media"))
LEDGER_ROOT = Path(os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))
REAL_ACE_RUN = Path(os.environ.get("ACE_C16_RUN", "/home/cereal/omni-campaign/media/out/c16_L12_v4"))
REAL_ACE_WEIGHTS = Path(os.environ.get("ACE_WEIGHTS", "/home/cereal/comfyui/models/diffusion_models/acestep_v1.5_turbo.safetensors"))
PY = sys.executable

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from chimera_omni.charted import ChartedBlockSpec  # noqa: E402
from chimera_omni.config import ChimeraConfig  # noqa: E402
from chimera_omni.export.gguf_names import chimera_parameter_names, qwen35_export_plan, require_qwen35_exportable  # noqa: E402
from chimera_omni.importing.safetensors_io import (  # noqa: E402
    load_chimera_checkpoint,
    read_checkpoint_config,
    read_index,
    read_safetensors,
    write_safetensors,
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load(TOOL, "c16_install_media")
fixtures = _load(FIXTURES, "c16_media_fixtures")

COMMON = [
    "--chimera-src", str(CHIMERA_SRC),
    "--mediafold-root", str(MEDIAFOLD_ROOT),
    "--ledger-root", str(LEDGER_ROOT),
]


def run_tool(*args: str) -> int:
    return tool.main([*COMMON, *(str(a) for a in args)])


@pytest.fixture(scope="module")
def fixture_set(tmp_path_factory):
    root = tmp_path_factory.mktemp("s6")
    report = fixtures.write_fixture_set(root, seed=0)
    # the accounting parent: a wider trunk whose header count exceeds every insert output here
    report["accounting_parent"] = fixtures.write_child(root / "accounting_parent", seed=0, intermediate=128)
    return root, report


def header_total(directory: Path) -> int:
    return sum(tool.header_counts(directory).values())


def manifest_of(directory: Path) -> dict:
    return json.loads((directory / "composite_manifest.json").read_text(encoding="utf-8"))


def tampered_child(src: Path, out: Path, mutate) -> Path:
    """A copy of the fixture child with `mutate(state)` applied, written by the reference writer."""

    document = read_checkpoint_config(src)
    state = {name: view.torch() for name, view in read_safetensors(src, verify=False).items()}
    mutate(state)
    write_safetensors(state, out, config=document["config"], ledger=document["ledger"], chimera_commit=document["chimera_commit"])
    return out


# ---------------------------------------------------------------- zero-media control


def test_zero_media_control_is_byte_for_byte_the_input_child(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    out = tmp_path / "control"
    assert run_tool("--child", root / "child", "--out", out) == 0
    child_index = read_index(root / "child")
    out_index = read_index(out)
    assert out_index["weight_map"] == child_index["weight_map"]
    assert out_index["metadata"]["tensor_sha256"] == child_index["metadata"]["tensor_sha256"]
    for name in sorted(set(child_index["weight_map"].values())):
        assert (out / name).read_bytes() == (root / "child" / name).read_bytes()
    assert header_total(out) == header_total(root / "child")
    manifest = manifest_of(out)
    assert manifest["components"] == []
    assert manifest["blocks"]["arm"] == "none"
    assert manifest["accounting"]["charted_or_blend_added"] == 0
    assert manifest["accounting"]["output_header_parameters"] == manifest["accounting"]["child_header_parameters"]
    # every input tensor was streamed, none materialised; the child was sha-verified on open
    assert manifest["disk"]["materialised_bytes"] == 0
    assert manifest["disk"]["streamed_bytes"] == manifest["child"]["header_bytes"]
    assert manifest["child"]["tensors_sha256_verified"] == manifest["child"]["tensors"]
    sidecar = (out / "composite_manifest.json.sha256").read_text().split()[0]
    assert len(sidecar) == 64


def test_lazy_views_hash_exactly_like_tensor_views(fixture_set) -> None:
    root, _report = fixture_set
    views, index = tool.open_lazy_checkpoint(root / "child")
    eager = read_safetensors(root / "child", verify=False)
    for name, lazy in views.items():
        assert lazy.sha256() == eager[name].sha256() == index["metadata"]["tensor_sha256"][name]
        assert torch.equal(lazy.torch(), eager[name].torch())
    assert tool.verify_lazy_checkpoint(views, index) == len(views)


# ---------------------------------------------------------------- the full insert


@pytest.fixture(scope="module")
def full_insert(fixture_set, tmp_path_factory):
    root, report = fixture_set
    out = tmp_path_factory.mktemp("omni") / "OMNI1"
    code = run_tool(
        "--child", root / "child", "--out", out,
        "--c15", root / "c15", "--c11", root / "c11",
        "--c18", root / "c18" / "ace_lm_music_codes.safetensors",
        "--blocks", root / "blocks" / "ace_L12", root / "blocks" / "qvl32_vit_Lmid", root / "blocks" / "qimg_Lmid",
        "--tap-rows", root / "tap",
        "--pair-by", "depth-fraction", "--arm", "insert",
        "--accounting-parent-directory", root / "accounting_parent",
    )
    assert code == 0
    return root, report, out


def test_full_insert_round_trips_into_a_loadable_checkpoint(full_insert) -> None:
    root, _report, out = full_insert
    model, cfg, receipt = load_chimera_checkpoint(out)
    assert receipt["tensors"] > 0
    assert [cfg.block_kind(i) for i in range(cfg.num_layers)].count("charted") == 3
    assert cfg.num_trunk_layers() == 8
    assert {spec.source: spec.time_embed_r for spec in cfg.charted_blocks} == {"ace": True, "qvl32": False, "qimg": False}
    ids = torch.randint(0, 256, (1, 7))
    with torch.no_grad():
        logits = model.eval()(ids)["logits"]
    assert torch.isfinite(logits).all()


def test_streamed_shards_equal_the_reference_writer_bytes(full_insert, tmp_path) -> None:
    """The streamed writer is `write_safetensors` byte for byte, on an output that mixes streamed and materialised tensors."""

    _root, _report, out = full_insert
    manifest = manifest_of(out)
    assert manifest["disk"]["materialised_bytes"] > 0 and manifest["disk"]["streamed_bytes"] > 0
    model, cfg, _receipt = load_chimera_checkpoint(out, dtype=torch.bfloat16)
    reference = tmp_path / "reference"
    write_safetensors(model.state_dict(), reference, config=cfg.to_manifest(), ledger=cfg.ledger().to_manifest())
    out_index, reference_index = read_index(out), read_index(reference)
    assert out_index["weight_map"] == reference_index["weight_map"]
    assert out_index["metadata"]["tensor_sha256"] == reference_index["metadata"]["tensor_sha256"]
    assert out_index["metadata"]["total_size"] == reference_index["metadata"]["total_size"]
    for name in sorted(set(out_index["weight_map"].values())):
        assert (out / name).read_bytes() == (reference / name).read_bytes()


def test_header_count_equals_child_plus_blocks_and_charts_exactly(full_insert) -> None:
    root, _report, out = full_insert
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    expected = header_total(root / "child") + sum(spec.parameter_count(cfg.hidden_size) for spec in cfg.charted_blocks)
    assert header_total(out) == expected
    manifest = manifest_of(out)
    assert manifest["accounting"]["output_header_parameters"] == expected
    assert manifest["accounting"]["oc_seal_header_parameters"] in (None, expected)
    assert manifest["accounting"]["charted_or_blend_added"] == expected - header_total(root / "child")
    counts = tool.header_counts(out)
    for component, placement in zip(manifest["components"], manifest["blocks"]["placements"]):
        prefix = placement["mlp_tensor_prefix"]
        assert component["child_params"] == sum(v for k, v in counts.items() if k.startswith(prefix))
        assert component["child_params"] == ChartedBlockSpec.from_manifest(placement["spec"]).mlp_parameter_count()
        assert component["ratio"] == pytest.approx(0.25)
        assert component["meets_4x"] is True
        assert component["register_form"] == "--component"
        assert component["register_spec"].split(":")[0] == component["component"]
    assert set(chimera_parameter_names(cfg)) == set(counts)


def test_band_rows_written_equal_the_c18_table(full_insert) -> None:
    root, _report, out = full_insert
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    band = cfg.ledger().range("music_codes")
    table = read_safetensors(root / "c18" / "ace_lm_music_codes.safetensors", verify=False)
    written = read_safetensors(out, ["embed.weight", "lm_head.weight"])
    for name in ("embed.weight", "lm_head.weight"):
        rows = written[name].torch()[band.begin : band.end]
        assert torch.equal(rows, table[name].torch().to(rows.dtype))
    child = read_safetensors(root / "child", ["embed.weight", "lm_head.weight"])
    for name in ("embed.weight", "lm_head.weight"):
        assert torch.equal(child[name].torch()[: band.begin], written[name].torch()[: band.begin])
    manifest = manifest_of(out)
    assert manifest["band"]["rows_written"] == band.width
    assert manifest["band"]["writes"]["embed.weight"]["previous_nonzero_values"] == 0
    assert manifest["band"]["table_tensor_sha256_check"]["verified"] is True


def test_interface_rows_land_in_the_counted_bridge_and_head(full_insert) -> None:
    root, report, out = full_insert
    written = read_safetensors(
        out,
        [
            "native_media_bridge.feature_proj.weight", "native_media_bridge.type_embed.weight",
            "native_media_head.out.weight", "native_media_head.type_affine.weight",
        ],
    )
    child = read_safetensors(root / "child", list(written))
    hidden = report["hidden"]
    entries = report["c15"]["entries"] + report["c11"]["entries"]
    assert len(entries) == 3
    feature = written["native_media_bridge.feature_proj.weight"].torch().float()
    head = written["native_media_head.out.weight"].torch().float()
    type_embed = written["native_media_bridge.type_embed.weight"].torch().float()
    affine = written["native_media_head.type_affine.weight"].torch().float()
    child_type_embed = child["native_media_bridge.type_embed.weight"].torch().float()
    child_affine = child["native_media_head.type_affine.weight"].torch().float()
    for entry, set_dir in zip(entries, ("c15", "c15", "c11")):
        rows = read_safetensors(root / set_dir / entry["rows"], verify=False)
        b, e = entry["feature_band"]
        E = rows["bridge.feature_proj.weight"].torch()
        D = rows["head.out.weight"].torch()
        assert torch.equal(feature[:, b:e], E.to(torch.bfloat16).float())
        assert torch.equal(head[b:e, :], D.to(torch.bfloat16).float())
        t = entry["type_id"]
        expected_row = (child_type_embed[t] + rows["bridge.type_embed_delta"].torch()).to(torch.bfloat16).float()
        assert torch.equal(type_embed[t], expected_row)
        # the shift half moved, the scale half did not
        assert not torch.equal(affine[t, hidden:], child_affine[t, hidden:])
        assert torch.equal(affine[t, :hidden], child_affine[t, :hidden])
    manifest = manifest_of(out)
    assert {e["type_id"] for e in manifest["interface"]["entries"]} == {4, 3, 1}
    for e in manifest["interface"]["entries"]:
        assert e["writes"]["native_media_head.type_affine.weight"]["relative_residual"] < 1.0e-6
        assert e["preconditions"]["violated"] is False
        assert e["tensor_sha256_check"]["verified"] is True and e["tensor_sha256_check"]["declared"] == 4


def test_depth_fraction_pairs_each_block_with_the_nearest_refinement_layer(full_insert) -> None:
    _root, _report, out = full_insert
    manifest = manifest_of(out)
    by_name = {p["container"]: p for p in manifest["blocks"]["placements"]}
    # trunk: 8 layers, gated_gqa at 3 and 7; depth fractions 3/7 = 0.43 and 1.0
    assert by_name["ace_L12"]["pairing"]["insert_after"] == 3       # 12/23 = 0.52 -> layer 3
    assert by_name["qimg_Lmid"]["pairing"]["insert_after"] == 3     # 0.25 -> layer 3
    assert by_name["qvl32_vit_Lmid"]["pairing"]["insert_after"] == 7  # 0.75 -> layer 7
    layers = sorted(p["layer_index"] for p in manifest["blocks"]["placements"])
    assert layers == [4, 5, 10]
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    assert [cfg.block_kind(i) for i in range(cfg.num_layers)] == [
        "kda", "kda", "kda", "gated_gqa", "charted", "charted", "kda", "kda", "kda", "gated_gqa", "charted",
    ]
    assert by_name["qimg_Lmid"]["chart_source"].startswith("fit_charts")
    # parent = host @ W + noise: the forward chart recovers it; the round trip through a
    # narrower parent width cannot (and its number is reported, not gated)
    assert by_name["qimg_Lmid"]["chart_fit"]["forward_scores"]["rel_l2"] < 0.2
    assert "rel_l2" in by_name["qimg_Lmid"]["chart_fit"]["roundtrip_scores"]
    assert by_name["ace_L12"]["chart_source"] == "container"
    # the adaLN blocks carry a verified captured-modulation reference; the ViT block has none
    assert by_name["ace_L12"]["time_reference"]["verified"] is True and by_name["ace_L12"]["time_reference"]["time_embed_r"] is True
    assert by_name["qimg_Lmid"]["time_reference"]["verified"] is True and by_name["qimg_Lmid"]["time_reference"]["rel_l2"] < 1.0e-5
    assert by_name["qvl32_vit_Lmid"]["time_reference"] is None


def test_text_is_bit_identical_and_routed_media_moves(full_insert) -> None:
    root, _report, out = full_insert
    child_model, _cfg, _r = load_chimera_checkpoint(root / "child")
    model, cfg, _receipt = load_chimera_checkpoint(out)
    child_model.eval()
    model.eval()
    torch.manual_seed(9)
    ids = torch.randint(0, 256, (2, 11))
    band_begin = cfg.ledger().range("music_codes").begin
    with torch.no_grad():
        before_text = child_model(ids)
        after_text = model(ids)
    # the residual stream is bit-identical; the logits differ ONLY on the music band
    # columns, which C18 filled (the child's rows there were zero)
    assert torch.equal(before_text["hidden"], after_text["hidden"])
    assert torch.equal(before_text["logits"][..., :band_begin], after_text["logits"][..., :band_begin])
    assert not torch.equal(before_text["logits"][..., band_begin:], after_text["logits"][..., band_begin:])
    feature = cfg.native_feature_dim
    media = dict(
        native_media_features=torch.randn(2, 3, feature),
        native_media_type_ids=torch.full((2, 3), 4, dtype=torch.long),
        native_media_positions=torch.rand(2, 3, 4),
        native_media_indices=torch.tensor([[8, 9, 10], [8, 9, 10]]),
        native_media_valid_mask=torch.ones(2, 3, dtype=torch.bool),
        return_aux=True,
    )
    with torch.no_grad():
        before = child_model(ids, **media)
        after = model(ids, **media)
    # the interface rows changed too (music band rows in the bridge), so the media rows differ
    # for both reasons; the text prefix is causally untouched by either
    assert torch.equal(before["hidden"][:, :8], after["hidden"][:, :8])
    assert not torch.allclose(before["aux"]["native_media_prediction"], after["aux"]["native_media_prediction"])


def test_the_manifest_names_every_source_and_a_reachable_register_line(full_insert) -> None:
    root, report, out = full_insert
    manifest = manifest_of(out)
    assert manifest["schema"] == "chimera_omni.composite_manifest.v1"
    sources = {s["source"] for s in manifest["sources"]}
    assert sources == {"ACE", "H3", "QVL32", "ACE-LM", "ace", "qvl32", "qimg"}
    for source in manifest["sources"]:
        for mechanism in source["mechanisms"]:
            assert mechanism["parameters"] > 0
            assert mechanism["sha256"]
    register = manifest["register"]
    command = register["command"]
    assert "--component ace.dit.block12.mlp:" in command
    assert "--passed-through" not in command
    assert f"--parent-directory {root / 'accounting_parent'}" in command
    assert "--parent-parameter-count" not in command and "--allow-asserted" not in command
    assert "--accounting-slack" not in command
    assert f"--recipe {out / 'composite_manifest.json'} --recipe-command " in command
    assert "--recipe-command 'c16_install_media.py " in command and f"--child {root / 'child'}" in command
    assert command.endswith("--confirm")
    assert register["reachable"] is True
    assert register["accounting_slack_required"] == 0
    parent_total = header_total(root / "accounting_parent")
    assert register["accounting_parent"]["parameter_count"] == parent_total
    assert register["measured_drop"] == parent_total - manifest["accounting"]["output_header_parameters"] > 0
    assert register["claimed_parent"] <= parent_total
    assert register["lineage_parent_checkpoint_id"] == tool.DEFAULT_PARENT_ID
    assert register["directory"]["known"] is True and register["directory"]["registerable"] is True
    assert manifest["tree"]["package_sha256"]
    assert manifest["accounting"]["quantization"] == "none"


def test_the_register_line_is_unreachable_without_an_accounting_parent(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    assert run_tool("--child", root / "child", "--out", tmp_path / "bare", "--blocks", root / "blocks" / "ace_L12") == 0
    register = manifest_of(tmp_path / "bare")["register"]
    assert register["reachable"] is False
    assert register["measured_drop"] is None
    assert any("measured_drop <= 0" in note for note in register["notes"])
    # a typed count is carried as an assertion the ledger will label as such
    parent_total = header_total(root / "accounting_parent")
    assert run_tool("--child", root / "child", "--out", tmp_path / "typed", "--blocks", root / "blocks" / "ace_L12", "--parent-parameter-count", str(parent_total)) == 0
    register = manifest_of(tmp_path / "typed")["register"]
    assert register["reachable"] is True
    assert f"--parent-parameter-count {parent_total} --allow-asserted-parameter-counts" in register["command"]
    # a typed count that disagrees with the directory's headers is refused: the bytes decide
    assert run_tool(
        "--child", root / "child", "--out", tmp_path / "disagree", "--blocks", root / "blocks" / "ace_L12",
        "--accounting-parent-directory", root / "accounting_parent", "--parent-parameter-count", str(parent_total + 1),
    ) == 3


# ---------------------------------------------------------------- the captured-modulation gate


def _assert_recorded_not_refused(out: Path, name: str) -> dict:
    """1.3.0 default: the (single) block installs; its gate report and the manifest carry the warning."""

    manifest = manifest_of(out)
    assert len(manifest["blocks"]["placements"]) == 1
    gate = manifest["blocks"]["placements"][0]["time_reference"]
    assert gate["verified"] is False and gate["rel_l2"] > gate["tolerance_rel_l2"] == 1.0e-5
    assert gate["policy"] == "diagnostic" and "Recorded, not refused" in gate["warning"]
    assert manifest["policy"]["time_reference"] == "diagnostic"
    warnings = [w for w in manifest["warnings"] if w["kind"] == "time_reference"]
    assert len(warnings) == 1 and warnings[0]["rel_l2"] == gate["rel_l2"] and warnings[0]["tolerance_rel_l2"] == 1.0e-5
    return gate


def test_a_time_path_that_does_not_reproduce_the_captured_modulation_is_recorded_unless_strict(fixture_set, tmp_path) -> None:
    """Owner policy 2026-10-06 (D5): the rel_l2 is a diagnostic; --strict-time-reference restores the refusal."""

    root, report = fixture_set
    child = root / "child"
    hidden = report["hidden"]
    from safetensors.torch import save_file

    # (a) a tampered reference: the block's own path no longer matches it
    fixtures.write_block_container(tmp_path / "tampered", "ace_L12", hidden=hidden, seed=1)
    container = tmp_path / "tampered" / "ace_L12"
    reference = {k: v.torch() for k, v in read_safetensors(container / tool.TIME_REFERENCE_FILENAME, verify=False).items()}
    reference["modulation"] = reference["modulation"] * 1.1
    save_file({k: v.contiguous() for k, v in reference.items()}, str(container / tool.TIME_REFERENCE_FILENAME))
    body = json.loads((container / "charted_block.json").read_text(encoding="utf-8"))
    body["tensor_sha256"].pop(tool.TIME_REFERENCE_FILENAME)
    (container / "charted_block.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o_a_strict", "--blocks", container, "--strict-time-reference") == 3
    assert not (tmp_path / "o_a_strict").exists()
    assert run_tool("--child", child, "--out", tmp_path / "o_a", "--blocks", container) == 0
    gate = _assert_recorded_not_refused(tmp_path / "o_a", "ace_L12")
    assert gate["rel_l2"] == pytest.approx(0.1 / 1.1, rel=1e-3)   # reference x1.1: |p - 1.1p| / |1.1p|
    # (b) no reference at all on an adaLN block: structural, refused with or without --strict
    fixtures.write_block_container(tmp_path / "missing", "qimg_Lmid", hidden=hidden, seed=2)
    (tmp_path / "missing" / "qimg_Lmid" / tool.TIME_REFERENCE_FILENAME).unlink()
    assert run_tool("--child", child, "--out", tmp_path / "o_b", "--blocks", tmp_path / "missing" / "qimg_Lmid") == 3
    assert not (tmp_path / "o_b").exists()
    # (c) THE regression: the ACE block converted with time_embed alone. The reference was
    # captured from both paths; a container that drops time_embed_r cannot reproduce it --
    # strict refuses it, the default installs it and says so in the manifest.
    fixtures.write_block_container(tmp_path / "dropped", "ace_L12", hidden=hidden, seed=3)
    container = tmp_path / "dropped" / "ace_L12"
    state = {k: v.torch() for k, v in read_safetensors(container / "charted_block.safetensors", verify=False).items()}
    state = {k: v for k, v in state.items() if not k.startswith("time_embed_r.")}
    save_file({k: v.contiguous() for k, v in state.items()}, str(container / "charted_block.safetensors"))
    body = json.loads((container / "charted_block.json").read_text(encoding="utf-8"))
    body["spec"]["time_embed_r"] = False
    body["tensor_sha256"].pop("charted_block.safetensors")
    (container / "charted_block.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o_c_strict", "--blocks", container, "--strict-time-reference") == 3
    assert not (tmp_path / "o_c_strict").exists()
    assert run_tool("--child", child, "--out", tmp_path / "o_c", "--blocks", container) == 0
    gate = _assert_recorded_not_refused(tmp_path / "o_c", "ace_L12")
    assert gate["time_embed_r"] is False
    # the same container with both paths installs clean, with or without --strict: no warning keys
    fixtures.write_block_container(tmp_path / "good", "ace_L12", hidden=hidden, seed=3)
    for flag, out in (([], "o_d"), (["--strict-time-reference"], "o_d_strict")):
        assert run_tool("--child", child, "--out", tmp_path / out, "--blocks", tmp_path / "good" / "ace_L12", *flag) == 0
        manifest = manifest_of(tmp_path / out)
        placement = manifest["blocks"]["placements"][0]
        assert placement["time_reference"]["verified"] is True and placement["time_reference"]["sigma"] == [0.25, 0.5, 0.75]
        assert "warning" not in placement["time_reference"] and "policy" not in placement["time_reference"]
        assert manifest["warnings"] == []
        assert manifest["policy"]["time_reference"] == ("strict" if flag else "diagnostic")


def test_verify_time_reference_records_by_default_and_raises_when_strict(fixture_set, tmp_path) -> None:
    """The function-level contract s7_m1_config_check.py relies on: a report, never a raise, unless strict."""

    root, report = fixture_set
    hidden = report["hidden"]

    class Cfg:
        hidden_size = hidden
        native_type_count = 8

    container = tool.load_container(root / "blocks" / "ace_L12", Cfg())
    if container["charts"] is None:
        container["charts"] = {k: torch.zeros(v) for k, v in container["spec"].parameter_shapes("", hidden).items() if k.startswith("chart_")}
    clean = tool.verify_time_reference(container, hidden, rel_tol=1.0e-5)
    assert clean["verified"] is True and "warning" not in clean
    container["time_reference"] = {**container["time_reference"], "modulation": container["time_reference"]["modulation"] * 2.0}
    loose = tool.verify_time_reference(container, hidden, rel_tol=1.0e-5)
    assert loose["verified"] is False and loose["rel_l2"] == pytest.approx(0.5, rel=1e-4) and loose["policy"] == "diagnostic"
    assert container["time_reference_check"] is loose
    with pytest.raises(tool.Refusal, match="does not reproduce"):
        tool.verify_time_reference(container, hidden, rel_tol=1.0e-5, strict=True)
    # structural refusals hold without strict: non-finite and all-zero references
    nan = container["time_reference"]["modulation"].clone()
    nan.view(-1)[0] = float("nan")
    container["time_reference"] = {**container["time_reference"], "modulation": nan}
    with pytest.raises(tool.Refusal, match="non-finite"):
        tool.verify_time_reference(container, hidden, rel_tol=1.0e-5)
    container["time_reference"] = {**container["time_reference"], "modulation": torch.zeros_like(nan)}
    with pytest.raises(tool.Refusal, match="all zero"):
        tool.verify_time_reference(container, hidden, rel_tol=1.0e-5)


# ---------------------------------------------------------------- refusals


def test_refusals(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    child = root / "child"
    hidden, feature = report["hidden"], report["native_feature_dim"]
    # overlapping bands
    bad_c15 = tmp_path / "bad_c15"
    fixtures.write_interface_set(
        bad_c15,
        [{"name": "a", "type_id": 4, "feature_band": [0, 20]}, {"name": "b", "type_id": 3, "feature_band": [10, 30]}],
        hidden=hidden, feature=feature,
    )
    assert run_tool("--child", child, "--out", tmp_path / "o1", "--c15", bad_c15) == 3
    assert not (tmp_path / "o1").exists()
    assert run_tool("--child", child, "--out", tmp_path / "o1b", "--c15", bad_c15, "--allow-band-overlap") == 0
    assert manifest_of(tmp_path / "o1b")["interface"]["bands"]["overlaps"]
    # band table of the wrong width
    fixtures.write_band_table(tmp_path / "wrong.safetensors", width=report["c18"]["band_width"] + 1, hidden=hidden)
    assert run_tool("--child", child, "--out", tmp_path / "o2", "--c18", tmp_path / "wrong.safetensors") == 3
    # non-zero band rows: fill once, then refuse to fill the output again
    good = root / "c18" / "ace_lm_music_codes.safetensors"
    assert run_tool("--child", child, "--out", tmp_path / "o3", "--c18", good) == 0
    assert run_tool("--child", tmp_path / "o3", "--out", tmp_path / "o4", "--c18", good) == 3
    assert run_tool("--child", tmp_path / "o3", "--out", tmp_path / "o4", "--c18", good, "--allow-nonzero-band") == 0
    # a container whose tensor set is not the canonical set
    fixtures.write_block_container(tmp_path / "blocks", "ace_L12", hidden=hidden, seed=1)
    tampered = tmp_path / "blocks" / "ace_L12" / "charted_block.safetensors"
    state = {k: v.torch() for k, v in read_safetensors(tampered, verify=False).items()}
    state.pop("mlp.up_proj.weight")
    fixtures.save_single(state, tampered)
    assert run_tool("--child", child, "--out", tmp_path / "o5", "--blocks", tmp_path / "blocks" / "ace_L12") == 3
    # a container without charts and no tap rows
    fixtures.write_block_container(tmp_path / "nocharts", "qimg_Lmid", hidden=hidden, with_charts=False)
    assert run_tool("--child", child, "--out", tmp_path / "o6", "--blocks", tmp_path / "nocharts" / "qimg_Lmid") == 3
    # out overlaps child; out already populated (by this tool: --overwrite is allowed)
    assert run_tool("--child", child, "--out", child / "sub") == 3
    assert run_tool("--child", child, "--out", tmp_path / "o3") == 3
    assert run_tool("--child", child, "--out", tmp_path / "o3", "--overwrite") == 0
    # explicit pairing without insert_after
    assert run_tool("--child", child, "--out", tmp_path / "o7", "--blocks", root / "blocks" / "ace_L12", "--pair-by", "explicit") == 3


def test_overwrite_only_removes_this_tools_own_output(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    stray = foreign / "model-00001-of-00001.safetensors"
    shutil.copy(next((root / "child").glob("*.safetensors")), stray)
    before = stray.read_bytes()
    assert run_tool("--child", root / "child", "--out", foreign, "--overwrite") == 3
    assert stray.read_bytes() == before
    assert not (foreign / "composite_manifest.json").exists()
    # a manifest from another tool does not unlock it either
    (foreign / "composite_manifest.json").write_text(json.dumps({"tool": {"name": "someone_else"}}), encoding="utf-8")
    assert run_tool("--child", root / "child", "--out", foreign, "--overwrite") == 3
    assert stray.read_bytes() == before


def test_the_free_space_guard_refuses_before_anything_is_written(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    out = tmp_path / "never"
    assert run_tool("--child", root / "child", "--out", out, "--min-free-gib", "1e9") == 3
    assert not out.exists()
    assert run_tool("--child", root / "child", "--out", out, "--min-free-gib", "0") == 0
    disk = manifest_of(out)["disk"]
    assert disk["estimated_output_bytes"] == disk["streamed_bytes"] + disk["materialised_bytes"]
    assert disk["free_bytes_before"] > disk["estimated_output_bytes"]


def test_nonzero_type_scale_rows_are_refused_unless_waived(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    hidden = report["hidden"]

    def scale_type_4(state: dict) -> None:
        state["native_media_head.type_affine.weight"][4, :hidden] = 0.25

    scaled = tampered_child(root / "child", tmp_path / "scaled", scale_type_4)
    assert run_tool("--child", scaled, "--out", tmp_path / "o1", "--c15", root / "c15") == 3
    assert not (tmp_path / "o1").exists()
    assert run_tool("--child", scaled, "--out", tmp_path / "o2", "--c15", root / "c15", "--allow-nonzero-type-affine") == 0
    entries = {e["type_id"]: e for e in manifest_of(tmp_path / "o2")["interface"]["entries"]}
    assert entries[4]["preconditions"]["violated"] is True and entries[4]["preconditions"]["waived"] is True
    assert entries[4]["preconditions"]["nonzero_scale_values"]["native_media_head.type_affine.weight[type, scale half]"] == hidden
    assert entries[3]["preconditions"]["violated"] is False

    def scale_bridge_band(state: dict) -> None:
        b, e = report["c15"]["entries"][1]["feature_band"]
        state["native_media_bridge.type_feature_scale.weight"][3, b:e] = 0.5

    bridge = tampered_child(root / "child", tmp_path / "bridge", scale_bridge_band)
    assert run_tool("--child", bridge, "--out", tmp_path / "o3", "--c15", root / "c15") == 3


def test_band_overlap_record_is_pairwise() -> None:
    entries = [
        {"name": "wide", "type_id": 1, "feature_band": [0, 100], "emission_band": [0, 100]},
        {"name": "a", "type_id": 2, "feature_band": [10, 20], "emission_band": [10, 20]},
        {"name": "b", "type_id": 3, "feature_band": [30, 40], "emission_band": [30, 40]},
    ]
    with pytest.raises(tool.Refusal):
        tool.check_bands(entries, allow_overlap=False)
    report = tool.check_bands(entries, allow_overlap=True)
    pairs = {(o["side"], o["a"], o["b"], tuple(o["span"])) for o in report["overlaps"]}
    assert pairs == {
        ("feature_band", "wide", "a", (10, 20)), ("feature_band", "wide", "b", (30, 40)),
        ("emission_band", "wide", "a", (10, 20)), ("emission_band", "wide", "b", (30, 40)),
    }
    assert report["duplicate_type_ids"] == {}


def test_declared_tensor_sha256_is_verified_against_the_files(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    child = root / "child"
    # an interface row set whose declared digest is stale
    shutil.copytree(root / "c11", tmp_path / "c11")
    companions = json.loads((tmp_path / "c11" / "companions.json").read_text(encoding="utf-8"))
    companions["entries"][0]["tensor_sha256"]["head.out.weight"] = "0" * 64
    (tmp_path / "c11" / "companions.json").write_text(json.dumps(companions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o1", "--c11", tmp_path / "c11") == 3
    # a declared name the file does not carry
    companions["entries"][0]["tensor_sha256"] = {"nope": "0" * 64}
    (tmp_path / "c11" / "companions.json").write_text(json.dumps(companions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o2", "--c11", tmp_path / "c11") == 3
    # a container whose charts digest is stale
    fixtures.write_block_container(tmp_path / "blocks", "ace_L12", hidden=report["hidden"], seed=5)
    body_path = tmp_path / "blocks" / "ace_L12" / "charted_block.json"
    body = json.loads(body_path.read_text(encoding="utf-8"))
    body["tensor_sha256"]["charts.safetensors"]["chart_in.weight"] = "f" * 64
    body_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o3", "--blocks", tmp_path / "blocks" / "ace_L12") == 3
    # the band table's sidecar digest
    shutil.copy(root / "c18" / "ace_lm_music_codes.safetensors", tmp_path / "band.safetensors")
    sidecar = json.loads((root / "c18" / "ace_lm_music_codes.safetensors.json").read_text(encoding="utf-8"))
    sidecar["tensor_sha256"]["embed.weight"] = "1" * 64
    (tmp_path / "band.safetensors.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert run_tool("--child", child, "--out", tmp_path / "o4", "--c18", tmp_path / "band.safetensors") == 3
    assert not any(p.exists() for p in (tmp_path / "o1", tmp_path / "o2", tmp_path / "o3", tmp_path / "o4"))


def test_a_post_write_count_disagreement_unlinks_the_output(fixture_set, tmp_path, monkeypatch) -> None:
    root, _report = fixture_set
    out = tmp_path / "refused"
    monkeypatch.setattr(tool.Installer, "load_blocks", lambda self: [])
    original = tool.Installer.write

    def poisoned(self):
        original(self)
        self.added_parameters = 7  # the headers will not agree

    monkeypatch.setattr(tool.Installer, "write", poisoned)
    assert run_tool("--child", root / "child", "--out", out) == 3
    assert (out / "REFUSED.json").is_file()
    assert not list(out.glob("*.safetensors"))


def test_the_cli_refuses_without_a_child() -> None:
    env = {**os.environ, "PYTHONPATH": str(CHIMERA_SRC / "src")}
    completed = subprocess.run([PY, str(TOOL), "--out", "/tmp/nowhere"], capture_output=True, text=True, env=env)
    assert completed.returncode == 3
    assert "REFUSED" in completed.stdout + completed.stderr


# ---------------------------------------------------------------- explicit pairing: H3 / HY3D presets


def test_explicit_pairing_inserts_the_h3_and_hy3d_containers(fixture_set, tmp_path) -> None:
    root, report = fixture_set
    hidden = report["hidden"]
    fixtures.write_block_container(tmp_path / "blocks", "h3_Lmid", hidden=hidden, seed=7, insert_after=1)
    fixtures.write_block_container(tmp_path / "blocks", "hy3d_Lmid", hidden=hidden, seed=8, insert_after=5)
    out = tmp_path / "explicit"
    assert run_tool(
        "--child", root / "child", "--out", out,
        "--blocks", tmp_path / "blocks" / "h3_Lmid", tmp_path / "blocks" / "hy3d_Lmid",
        "--pair-by", "explicit", "--arm", "insert",
        "--accounting-parent-directory", root / "accounting_parent",
    ) == 0
    manifest = manifest_of(out)
    by_name = {p["container"]: p for p in manifest["blocks"]["placements"]}
    assert by_name["h3_Lmid"]["pairing"] == {"rule": "explicit", "insert_after": 1}
    assert by_name["h3_Lmid"]["layer_index"] == 2 and by_name["hy3d_Lmid"]["layer_index"] == 7
    assert by_name["h3_Lmid"]["type_ids"] == [2, 3] and by_name["hy3d_Lmid"]["type_ids"] == [7]
    for name in ("h3_Lmid", "hy3d_Lmid"):
        assert by_name[name]["time_reference"]["verified"] is True
        assert by_name[name]["tensor_sha256_check"]["charted_block.safetensors"]["verified"] is True
    assert {c["component"]: c["register_form"] for c in manifest["components"]} == {"h3.dit.block.mlp": "--component", "hy3d.dit.block.mlp": "--component"}
    assert manifest["register"]["reachable"] is True
    model, cfg, _receipt = load_chimera_checkpoint(out)
    assert [cfg.block_kind(i) for i in range(cfg.num_layers)] == [
        "kda", "kda", "charted", "kda", "gated_gqa", "kda", "kda", "charted", "kda", "gated_gqa",
    ]
    assert header_total(out) == header_total(root / "child") + sum(spec.parameter_count(hidden) for spec in cfg.charted_blocks)
    torch.manual_seed(3)
    ids = torch.randint(0, 256, (1, 9))
    for type_id in (2, 7):
        media = dict(
            native_media_features=torch.randn(1, 3, cfg.native_feature_dim),
            native_media_type_ids=torch.full((1, 3), type_id, dtype=torch.long),
            native_media_positions=torch.rand(1, 3, 4),
            native_media_indices=torch.tensor([[6, 7, 8]]),
            native_media_valid_mask=torch.ones(1, 3, dtype=torch.bool),
            return_aux=True,
        )
        with torch.no_grad():
            assert torch.isfinite(model.eval()(ids, **media)["aux"]["native_media_prediction"]).all()


# ---------------------------------------------------------------- blend arm


def test_blend_arm_is_qwen35_exportable_and_registers_below_4x_as_passed_through(tmp_path) -> None:
    root = tmp_path / "s6"
    fixtures.write_child(root / "child", seed=2, residual_mode="none", tie_word_embeddings=True)
    fixtures.write_child(root / "parent", seed=2, residual_mode="none", tie_word_embeddings=True, intermediate=128)
    fixtures.write_block_container(root / "blocks", "ace_L12", hidden=fixtures.DEFAULT_HIDDEN, seed=3)
    fixtures.write_block_container(root / "blocks", "qvl32_vit_Lmid", hidden=fixtures.DEFAULT_HIDDEN, seed=4)
    child_total = header_total(root / "child")
    # per-layer widths are refused by default
    assert run_tool("--child", root / "child", "--out", tmp_path / "b0", "--blocks", root / "blocks" / "ace_L12", "--arm", "blend") == 3
    # a gelu block cannot enter a silu-glu FFN
    assert run_tool("--child", root / "child", "--out", tmp_path / "b1", "--blocks", root / "blocks" / "qvl32_vit_Lmid", "--arm", "blend", "--blend-uniform", "pad") == 3
    out = tmp_path / "blend"
    assert run_tool(
        "--child", root / "child", "--out", out, "--blocks", root / "blocks" / "ace_L12", "--arm", "blend", "--blend-uniform", "pad",
        "--accounting-parent-directory", root / "parent",
    ) == 0
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    require_qwen35_exportable(cfg)
    counts = tool.header_counts(out)
    names = list(counts) + list(read_index(out)["metadata"]["tied_weights"])
    qwen35_export_plan(cfg).covers(names)
    manifest = manifest_of(out)
    keep = 20
    hidden = fixtures.DEFAULT_HIDDEN
    assert cfg.intermediate_size == 48 + keep
    assert manifest["blocks"]["padding_parameters"] == 7 * 3 * keep * hidden
    assert manifest["blocks"]["activation_kind"] == "silu"
    assert manifest["accounting"]["output_header_parameters"] == child_total + 8 * 3 * keep * hidden
    assert header_total(out) == manifest["accounting"]["output_header_parameters"]
    model, _cfg, _receipt = load_chimera_checkpoint(out)
    with torch.no_grad():
        assert torch.isfinite(model.eval()(torch.randint(0, 256, (1, 5)))["logits"]).all()
    component = manifest["components"][0]
    assert component["tensor_prefix"] is None
    assert component["child_params"] == 3 * keep * hidden
    # 3 x 20 x 32 = 1,920 against a 4x-coreset parent of 5,760: ratio 1/3 > 0.25, so the ledger
    # would refuse it as --component; the line carries it as --passed-through and is NOT reachable
    assert component["ratio"] == pytest.approx(1 / 3)
    assert component["meets_4x"] is False and component["register_form"] == "--passed-through"
    register = manifest["register"]
    assert "--passed-through ace.dit.block12.mlp" in register["command"]
    assert "--component" not in register["command"]
    assert register["reachable"] is False and register["passed_through"] == ["ace.dit.block12.mlp"]


def test_blend_refuses_a_situ_trunk(tmp_path) -> None:
    root = tmp_path / "s6"
    fixtures.write_child(root / "child", seed=2, residual_mode="none", tie_word_embeddings=True, activation_kind="situ", situ_gate_beta=8.0, situ_up_beta=8.0)
    fixtures.write_block_container(root / "blocks", "ace_L12", hidden=fixtures.DEFAULT_HIDDEN, seed=3)
    assert run_tool("--child", root / "child", "--out", tmp_path / "b", "--blocks", root / "blocks" / "ace_L12", "--arm", "blend", "--blend-uniform", "pad") == 3
    assert not (tmp_path / "b").exists()
    # the insert arm does not touch the trunk FFN, so a SiTU trunk is fine there
    assert run_tool("--child", root / "child", "--out", tmp_path / "i", "--blocks", root / "blocks" / "ace_L12", "--arm", "insert") == 0


# ---------------------------------------------------------------- ACE converter


def _mediafold():
    if str(MEDIAFOLD_ROOT) not in sys.path:
        sys.path.insert(0, str(MEDIAFOLD_ROOT))
    return pytest.importorskip("mediafold.ace_dit"), pytest.importorskip("mediafold.c16_transplant"), pytest.importorskip("mediafold.linalg")


def test_ace_run_converter_reproduces_mediafold_transplanted_block_with_both_time_paths(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    ace_dit, c16, linalg = _mediafold()
    from safetensors.torch import save_file

    torch.manual_seed(21)
    hidden, width, heads, kv, hd, inner = 32, 24, 2, 1, 8, 12
    block = ace_dit.AceStepDiTLayer(width, heads, kv, hd, inner, 1.0e-6, layer_type="sliding_attention", sliding_window=3)
    with torch.no_grad():
        for p in block.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3 / p.shape[-1] ** 0.5)
        block.scale_shift_table.normal_(0.0, 0.1)
    time_embed = ace_dit.TimestepEmbedding(16, width)
    time_embed_r = ace_dit.TimestepEmbedding(16, width)
    sigma = 0.3
    t = torch.tensor([sigma, sigma])
    with torch.no_grad():
        _temb_t, proj_t = time_embed(t)
        _temb_r, proj_r = time_embed_r(t - 0.0)  # the parent's t - t_r with t_r = 0, as captured
        projected = proj_t + proj_r
    run = tmp_path / "run"
    run.mkdir()
    save_file({k: v.contiguous() for k, v in block.state_dict().items()}, str(run / f"folded_block_L12_keep{inner}.safetensors"))
    chart_in = linalg.LinearMap(weight=torch.randn(width, hidden, dtype=torch.float64) * 0.2, bias=torch.randn(width, dtype=torch.float64) * 0.1)
    chart_out = linalg.LinearMap(weight=torch.randn(hidden, width, dtype=torch.float64) * 0.2, bias=torch.zeros(hidden, dtype=torch.float64))
    pair = c16.ChartPair(chart_in=chart_in, chart_out=chart_out, method="ridge")
    save_file({k: v.contiguous() for k, v in pair.state().items()}, str(run / "chart_ridge.safetensors"))
    # the capture: the parent's temb (both paths summed) on every captured row
    save_file({"temb": projected[:1].expand(4, 6, width).contiguous(), "x_in": torch.zeros(4, 5, width)}, str(run / "capture_L12.safetensors"))
    (run / "c16_receipt.json").write_text(json.dumps({
        "kind": "c16_transplant",
        "block": {"source": "fixture", "prefix": "decoder.layers.12.", "layer_type": "sliding_attention", "mlp_parameters": 3 * width * 4 * inner},
        "capture": {"shell": {"depth": 12, "config": {"hidden_size": width, "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd, "intermediate_size": 4 * inner}}, "stimulus": {"rung": "latents", "sigma": sigma}},
        "coreset": {"keep": inner, "component": {"parent_parameters": 3 * width * 4 * inner, "child_parameters": 3 * width * inner}, "whole_block": {"parent_parameters": 1000, "child_parameters": 500}, "holdout_scores": {"rel_l2": 0.5, "cos": 0.8}, "split_by": "clip"},
    }), encoding="utf-8")
    weights = tmp_path / "ace.safetensors"
    save_file({
        **{f"decoder.time_embed.{k}": v.contiguous() for k, v in time_embed.state_dict().items()},
        **{f"decoder.time_embed_r.{k}": v.contiguous() for k, v in time_embed_r.state_dict().items()},
    }, str(weights))

    # a weights file without the second path is refused: the parent has two, the capture used two
    half = tmp_path / "half.safetensors"
    save_file({f"decoder.time_embed.{k}": v.contiguous() for k, v in time_embed.state_dict().items()}, str(half))
    with pytest.raises(tool.Refusal, match="time_embed_r"):
        tool.convert_ace_run(run, half, tmp_path / "half_container", sliding_window=3)

    container = tmp_path / "container"
    body = tool.convert_ace_run(run, weights, container, sliding_window=3)
    assert body["spec"]["mlp_width"] == inner and body["spec"]["sliding_window"] == 3
    assert body["spec"]["component"] == "ace.dit.block12.mlp"
    assert body["spec"]["time_embed_r"] is True
    assert body["spec"]["source_depth_fraction"] == pytest.approx(12 / 23)
    assert body["time_reference"]["sigma"] == [sigma] and body["time_reference"]["conversion_rel_l2"] < 1.0e-5
    assert set(body["tensor_sha256"]) == {"charted_block.safetensors", "charts.safetensors", tool.TIME_REFERENCE_FILENAME}

    from chimera_omni.modeling.charted import ChartedSourceBlock

    spec = ChartedBlockSpec(layer_index=0, **{**body["spec"], "type_ids": tuple(body["spec"]["type_ids"])})
    charted = ChartedSourceBlock(spec, hidden).eval()
    state = {k: v.torch() for k, v in read_safetensors(container / "charted_block.safetensors", verify=False).items()}
    state.update({k: v.torch() for k, v in read_safetensors(container / "charts.safetensors", verify=False).items()})
    charted.load_state_dict(state, strict=True)

    reference = c16.TransplantedBlock(block, pair, arm="insert", rope_theta=1.0e6).eval()
    x = torch.randn(2, 7, hidden)
    with torch.no_grad():
        expected = reference(x, projected)
        row_types = torch.full((2, 7), 4, dtype=torch.long)
        levels = torch.full((2, 7), sigma)
        got = x + charted(x, row_types, levels, torch.ones(2, 7, dtype=torch.bool))
        # the charted time path IS the captured temb; time_embed alone is not
        assert torch.allclose(charted.time_path(torch.tensor([sigma])), projected[:1], atol=1.0e-5, rtol=1.0e-5)
        assert not torch.allclose(charted.time_embed(torch.tensor([sigma])).view(1, 6, width), projected[:1], atol=1.0e-3)
    assert torch.allclose(expected, got, atol=1.0e-4, rtol=1.0e-4)
    assert not torch.allclose(expected, x)
    # and the converted container passes the installer's gate onto the fixture child
    assert run_tool("--child", root / "child", "--out", tmp_path / "installed", "--blocks", container) == 0
    assert manifest_of(tmp_path / "installed")["blocks"]["placements"][0]["time_reference"]["verified"] is True


def test_ace_converter_records_a_time_path_mismatch_unless_strict(tmp_path) -> None:
    """A capture whose temb is time_embed ALONE cannot be reproduced by the two-path container:
    1.3.0 writes the container with `conversion_warning`; strict refuses and writes nothing."""

    ace_dit, c16, linalg = _mediafold()
    from safetensors.torch import save_file

    torch.manual_seed(23)
    hidden, width, heads, kv, hd, inner = 32, 24, 2, 1, 8, 12
    block = ace_dit.AceStepDiTLayer(width, heads, kv, hd, inner, 1.0e-6, layer_type="sliding_attention", sliding_window=3)
    with torch.no_grad():
        for p in block.parameters():
            if p.dim() >= 2:
                p.normal_(0.0, 0.3 / p.shape[-1] ** 0.5)
        block.scale_shift_table.normal_(0.0, 0.1)
    time_embed = ace_dit.TimestepEmbedding(16, width)
    time_embed_r = ace_dit.TimestepEmbedding(16, width)
    sigma = 0.3
    with torch.no_grad():
        _temb_t, proj_t = time_embed(torch.tensor([sigma, sigma]))
    run = tmp_path / "run"
    run.mkdir()
    save_file({k: v.contiguous() for k, v in block.state_dict().items()}, str(run / f"folded_block_L12_keep{inner}.safetensors"))
    chart_in = linalg.LinearMap(weight=torch.randn(width, hidden, dtype=torch.float64) * 0.2, bias=torch.randn(width, dtype=torch.float64) * 0.1)
    chart_out = linalg.LinearMap(weight=torch.randn(hidden, width, dtype=torch.float64) * 0.2, bias=torch.zeros(hidden, dtype=torch.float64))
    pair = c16.ChartPair(chart_in=chart_in, chart_out=chart_out, method="ridge")
    save_file({k: v.contiguous() for k, v in pair.state().items()}, str(run / "chart_ridge.safetensors"))
    # the capture carries time_embed ALONE (the pre-2026-09-06 bug), the weights carry both paths
    save_file({"temb": proj_t[:1].expand(4, 6, width).contiguous(), "x_in": torch.zeros(4, 5, width)}, str(run / "capture_L12.safetensors"))
    (run / "c16_receipt.json").write_text(json.dumps({
        "kind": "c16_transplant",
        "block": {"source": "fixture", "prefix": "decoder.layers.12.", "layer_type": "sliding_attention", "mlp_parameters": 3 * width * 4 * inner},
        "capture": {"shell": {"depth": 12, "config": {"hidden_size": width, "num_attention_heads": heads, "num_key_value_heads": kv, "head_dim": hd, "intermediate_size": 4 * inner}}, "stimulus": {"rung": "latents", "sigma": sigma}},
        "coreset": {"keep": inner, "component": {"parent_parameters": 3 * width * 4 * inner, "child_parameters": 3 * width * inner}, "whole_block": {"parent_parameters": 1000, "child_parameters": 500}, "holdout_scores": {"rel_l2": 0.5, "cos": 0.8}, "split_by": "clip"},
    }), encoding="utf-8")
    weights = tmp_path / "ace.safetensors"
    save_file({
        **{f"decoder.time_embed.{k}": v.contiguous() for k, v in time_embed.state_dict().items()},
        **{f"decoder.time_embed_r.{k}": v.contiguous() for k, v in time_embed_r.state_dict().items()},
    }, str(weights))
    with pytest.raises(tool.Refusal, match="does not reproduce"):
        tool.convert_ace_run(run, weights, tmp_path / "strict", sliding_window=3, strict=True)
    assert not (tmp_path / "strict").exists()
    body = tool.convert_ace_run(run, weights, tmp_path / "recorded", sliding_window=3)
    ref = body["time_reference"]
    assert ref["conversion_rel_l2"] > 1.0e-5 and ref["policy"] == "diagnostic"
    assert ref["conversion_tolerance_rel_l2"] == 1.0e-5 and "does not reproduce" in ref["conversion_warning"]
    assert (tmp_path / "recorded" / "charted_block.json").is_file()
    # CLI: --strict-time-reference reaches the converter
    assert tool.main(["--convert-ace-run", str(run), "--ace-weights", str(weights), "--convert-out", str(tmp_path / "cli_strict"),
                      "--convert-sliding-window", "3", "--strict-time-reference", *COMMON]) == 3
    assert not (tmp_path / "cli_strict").exists()


@pytest.mark.skipif(not (REAL_ACE_RUN.is_dir() and REAL_ACE_WEIGHTS.is_file()), reason="real ACE c16 run / weights absent")
def test_the_real_ace_c16_l12_v4_run_converts(tmp_path) -> None:
    body = tool.convert_ace_run(REAL_ACE_RUN, REAL_ACE_WEIGHTS, tmp_path / "ace_L12")
    assert body["spec"]["width"] == 2048 and body["spec"]["mlp_width"] == 1536
    assert body["spec"]["heads"] == 16 and body["spec"]["kv_heads"] == 8 and body["spec"]["head_dim"] == 128
    assert body["spec"]["sliding_window"] == 128
    assert body["spec"]["time_embed_r"] is True
    assert body["accounting"]["parent_mlp_parameters"] == 37748736
    assert body["accounting"]["child_mlp_parameters"] == 9437184
    # the captured temb at sigma 0.5 (review probe: chunk norms [4.9, 375.8, 32.2, 5.9, 124.8, 33.9];
    # time_embed alone gave [2.1, 41.4, 5.0, 1.8, 8.3, 6.4]) is reproduced by both paths
    assert body["time_reference"]["sigma"] == [0.5]
    assert body["time_reference"]["conversion_rel_l2"] < 1.0e-5
    assert body["time_reference"]["chunk_norms"][1] == pytest.approx(375.8, rel=0.02)
    assert body["time_reference"]["chunk_norms"][4] == pytest.approx(124.8, rel=0.02)
    spec = ChartedBlockSpec(layer_index=0, **{**body["spec"], "type_ids": tuple(body["spec"]["type_ids"])})
    views = read_safetensors(tmp_path / "ace_L12" / "charted_block.safetensors", verify=False)
    shapes = spec.parameter_shapes("", 128)
    assert set(views) == {name for name in shapes if not name.startswith("chart_")}
    for name, view in views.items():
        assert tuple(view.shape) == shapes[name]
    assert spec.mlp_parameter_count() == 9437184
    # the folded MLP rows are the run's bytes, untouched
    folded = read_safetensors(next(REAL_ACE_RUN.glob("folded_block_L12_keep1536.safetensors")), verify=False)
    for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"):
        assert torch.equal(views[name].torch().float(), folded[name].torch().float())


# ---------------------------------------------------------------- 1.4.0: the all-source table (OMNI1 assembly lane)

FIXTURE_ALIASES = {"ACE-Step": ["ACE", "ace", "ACE-LM"], "H3": ["H3"], "QVL": ["QVL32", "qvl32"], "QIMG": ["qimg"]}


def _lineage_file(tmp_path: Path, aliases: dict, *, tamper: bool = False) -> Path:
    receipt = tmp_path / "trunk_receipt.json"
    receipt.write_text(json.dumps({"header_parameters": 123}), encoding="utf-8")
    digest = tool.sha256_file(receipt)
    body = {
        "schema": tool.LINEAGE_SCHEMA,
        "sources": [{"source": "Trunk", "role": "the child's trunk", "accounting": {"components": 1, "parent_params": 10, "child_params": 2},
                     "receipts": [{"path": str(receipt), "sha256": ("0" * 64) if tamper else digest}]}],
        "media_aliases": aliases,
    }
    path = tmp_path / ("lineage_bad.json" if tamper else "lineage.json")
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _full_insert_args(root: Path, out: Path) -> list:
    return [
        "--child", root / "child", "--out", out, "--c15", root / "c15", "--c11", root / "c11",
        "--c18", root / "c18" / "ace_lm_music_codes.safetensors",
        "--blocks", root / "blocks" / "ace_L12", root / "blocks" / "qvl32_vit_Lmid", root / "blocks" / "qimg_Lmid",
        "--tap-rows", root / "tap", "--pair-by", "depth-fraction", "--arm", "insert",
        "--accounting-parent-directory", root / "accounting_parent",
    ]


def test_lineage_sources_name_every_source_and_media_accounting_comes_from_the_headers(fixture_set, full_insert, tmp_path) -> None:
    root, _report = fixture_set
    out = tmp_path / "OMNI1L"
    lineage = _lineage_file(tmp_path, FIXTURE_ALIASES)
    assert run_tool(*_full_insert_args(root, out), "--lineage-sources", lineage) == 0
    manifest = manifest_of(out)
    plain = manifest_of(full_insert[2])
    # without the flag the manifest carries no 1.4.0 keys; with it, exactly two more
    assert "sources_all" not in plain and "lineage_sources" not in plain
    assert set(manifest) - set(plain) == {"sources_all", "lineage_sources"}
    # the weights do not depend on the flag
    assert read_index(out)["metadata"]["tensor_sha256"] == read_index(full_insert[2])["metadata"]["tensor_sha256"]
    table = manifest["sources_all"]
    assert table["count"] == 5 and table["names"] == ["Trunk", "ACE-Step", "H3", "QVL", "QIMG"]
    assert table["sources"][0]["accounting"] == {"components": 1, "parent_params": 10, "child_params": 2}
    assert manifest["lineage_sources"]["sha256"] == tool.sha256_file(lineage)
    assert manifest["lineage_sources"]["receipts_verified"][0]["sha256"] == tool.sha256_file(tmp_path / "trunk_receipt.json")
    counts = tool.header_counts(out)
    by_name = {s["source"]: s for s in table["sources"]}
    placements = {p["source"]: p for p in manifest["blocks"]["placements"]}
    for display, label in (("ACE-Step", "ace"), ("QVL", "qvl32"), ("QIMG", "qimg")):
        acc = by_name[display]["accounting"]
        prefix = placements[label]["tensor_prefix"]
        total = sum(v for k, v in counts.items() if k.startswith(prefix))
        assert acc["header_parameters_added"] == total == placements[label]["parameters"]
        assert acc["mlp_child_parameters"] + acc["chart_parameters"] + acc["non_mlp_block_parameters"] == total
        assert acc["chart_parameters"] == placements[label]["chart_parameters"]
        assert acc["mlp_child_parameters"] == acc["component_child_parameters"] == placements[label]["mlp_parameters"]
        assert [c["component"] for c in acc["components"]] == [placements[label]["component"]]
        assert acc["passed_through_added_parameters"] == total - acc["mlp_child_parameters"]
    # interface-only and band rows are written in place: they add nothing
    assert by_name["H3"]["accounting"]["header_parameters_added"] == 0
    assert by_name["H3"]["accounting"]["written_in_place_parameters"] > 0
    assert any(m["label"] == "ACE-LM" for m in by_name["ACE-Step"]["mechanisms"])
    assert table["media_header_parameters_added"] == manifest["accounting"]["charted_or_blend_added"]


def test_lineage_sources_refusals(fixture_set, tmp_path) -> None:
    root, _report = fixture_set
    # a receipt whose bytes do not match the declared sha refuses before anything is written
    bad = _lineage_file(tmp_path, FIXTURE_ALIASES, tamper=True)
    assert run_tool(*_full_insert_args(root, tmp_path / "r1"), "--lineage-sources", bad) == 3
    assert not (tmp_path / "r1").exists()
    # an installed label without an alias refuses (and leaves nothing loadable behind)
    missing = dict(FIXTURE_ALIASES)
    missing["QIMG"] = ["not-qimg"]
    assert run_tool(*_full_insert_args(root, tmp_path / "r2"), "--lineage-sources", _lineage_file(tmp_path, missing)) == 3
    assert not list((tmp_path / "r2").glob("*.safetensors"))
    assert (tmp_path / "r2" / "REFUSED.json").is_file()
    # an alias that installed nothing refuses too
    extra = {**FIXTURE_ALIASES, "Ghost": ["ghost"]}
    assert run_tool(*_full_insert_args(root, tmp_path / "r3"), "--lineage-sources", _lineage_file(tmp_path, extra)) == 3
    assert not list((tmp_path / "r3").glob("*.safetensors"))
    # a label claimed twice refuses at load
    twice = {**FIXTURE_ALIASES, "Again": ["ace"]}
    assert run_tool(*_full_insert_args(root, tmp_path / "r4"), "--lineage-sources", _lineage_file(tmp_path, twice)) == 3
    assert not (tmp_path / "r4").exists()


def test_register_directory_reads_the_ledgers_registrable_under_protected(tmp_path) -> None:
    sys.path.insert(0, str(LEDGER_ROOT))
    oc_checkpoint = pytest.importorskip("oc_checkpoint")
    if not getattr(oc_checkpoint, "REGISTRABLE_UNDER_PROTECTED", ()):
        pytest.skip("this ledger has no REGISTRABLE_UNDER_PROTECTED")
    installer = tool.Installer.__new__(tool.Installer)
    installer.args = tool.build_parser().parse_args(["--ledger-root", str(LEDGER_ROOT)])
    installer.out = Path("/mnt/lexar/omni-campaign/omni1/X")
    lexar = installer.directory_registerable()
    assert lexar["registerable"] is True and lexar["allowed_by"] == "/mnt/lexar" and lexar["matched_prefix"] == "/mnt"
    installer.out = Path("/mnt/ai_data/omni-campaign/X")
    share = installer.directory_registerable()
    assert share["registerable"] is False and share["matched_prefix"] == "/mnt"
    installer.out = tmp_path / "x"
    assert installer.directory_registerable()["registerable"] is True
