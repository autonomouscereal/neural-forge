"""Tests for s7_omni1_check.py (run on the AI server, CPU only).

    cd /home/cereal/omni-campaign/composite/tools
    CHIMERA_OMNI_SRC=/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9 MEDIAFOLD_ROOT=/home/cereal/omni-campaign/media \
    OC_LEDGER_ROOT=/home/cereal/omni-campaign/ledger PYTHONPATH=$CHIMERA_OMNI_SRC/src CUDA_VISIBLE_DEVICES= \
    /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_s7_omni1_check.py

A synthetic composite (c16_media_fixtures child + c16_install_media insert of two charted blocks + C15/C11/C18) is
checked against its own S5 child and a wider "parent"; synthetic pair sets / host rows in the s7_host_rows formats
give every charted block one native media segment.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
CHIMERA_SRC = Path(os.environ.get("CHIMERA_OMNI_SRC", "/home/cereal/omni-campaign/import/rq27b/tree_ef1ffb9"))
LEDGER_ROOT = Path(os.environ.get("OC_LEDGER_ROOT", "/home/cereal/omni-campaign/ledger"))
MEDIAFOLD_ROOT = Path(os.environ.get("MEDIAFOLD_ROOT", "/home/cereal/omni-campaign/media"))

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


c16 = _load(HERE / "c16_install_media.py", "c16_install_media_o1t")
fixtures = _load(HERE / "c16_media_fixtures.py", "c16_media_fixtures_o1t")
check = _load(HERE / "s7_omni1_check.py", "s7_omni1_check_t")


def write_segment(tap_root: Path, pairs_root: Path, unit: str, *, type_id: int, width: int, hidden: int,
                  position_dim: int, sigma: float, rows: int = 6) -> None:
    g = torch.Generator().manual_seed(7)
    n = 2 * rows  # two forwards: the first is fit, the second held
    pair_dir = pairs_root / unit
    pair_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "features": torch.randn(n, width, generator=g),
        "feature_width": torch.full((n,), width, dtype=torch.long),
        "type_ids": torch.full((n,), type_id, dtype=torch.long),
        "positions": torch.zeros(n, position_dim),
        "sigma": torch.full((n,), sigma),
        "forward": torch.arange(n) // rows,
        "token": torch.arange(n) % rows,
        "held": (torch.arange(n) // rows) == 1,
    }
    save_file(state, str(pair_dir / "parent_capture.safetensors"))
    (pair_dir / "parent_capture.json").write_text(json.dumps({"forwards": [
        {"index": 0, "prompt": "p0", "rows": rows}, {"index": 1, "prompt": "p1", "rows": rows}]}), encoding="utf-8")
    unit_dir = tap_root / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    tap = {"host_rows": torch.randn(n, hidden, generator=g), "forward": state["forward"], "held": state["held"]}
    save_file(tap, str(unit_dir / "paired_rows.safetensors"))
    (unit_dir / "paired_rows.json").write_text(json.dumps({
        "file_sha256": check.sha256_file(unit_dir / "paired_rows.safetensors"),
        "pairs": {"file": str(pair_dir / "parent_capture.safetensors"), "sha256": check.sha256_file(pair_dir / "parent_capture.safetensors")},
        "sequence_layout": {"tokenizer": "unused", "tokenizer_sha256": "unused"}}), encoding="utf-8")


@pytest.fixture(scope="module")
def composite(tmp_path_factory):
    root = tmp_path_factory.mktemp("o1")
    report = fixtures.write_fixture_set(root, seed=0, residual_mode="none")
    fixtures.write_child(root / "parent", seed=3, intermediate=128, residual_mode="none")
    out = root / "OMNI1"
    code = c16.main([
        "--chimera-src", str(CHIMERA_SRC), "--mediafold-root", str(MEDIAFOLD_ROOT), "--ledger-root", str(LEDGER_ROOT),
        "--child", str(root / "child"), "--out", str(out), "--c15", str(root / "c15"), "--c11", str(root / "c11"),
        "--c18", str(root / "c18" / "ace_lm_music_codes.safetensors"),
        "--blocks", str(root / "blocks" / "ace_L12"), str(root / "blocks" / "qvl32_vit_Lmid"),
        "--tap-rows", str(root / "tap"), "--pair-by", "depth-fraction", "--arm", "insert",
        "--accounting-parent-directory", str(root / "parent"),
    ])
    assert code == 0
    manifest = json.loads((out / "composite_manifest.json").read_text())
    bands = {int(e["type_id"]): e["feature_band"] for e in manifest["interface"]["entries"]}
    from chimera_omni.config import ChimeraConfig
    from chimera_omni.importing.safetensors_io import read_checkpoint_config
    cfg = ChimeraConfig.from_manifest(read_checkpoint_config(out)["config"])
    tap, pairs = root / "tap_s5", root / "pairs"
    for p in manifest["blocks"]["placements"]:
        t = next(t for t in p["type_ids"] if t in bands)
        b, e = bands[t]
        sigma = 0.5 if p["spec"]["modulation"] == "adaln6" else -1.0
        write_segment(tap, pairs, p["container"], type_id=t, width=e - b, hidden=report["hidden"],
                      position_dim=int(cfg.native_position_dim), sigma=sigma)
    (root / "prompts.json").write_text(json.dumps([[1, 2, 3, 4, 5], [6, 7, 8]]))
    (root / "media_prompt.json").write_text(json.dumps([9, 10, 11]))
    return root, out, manifest


def args_for(root: Path, out: Path, receipt: Path, *extra: str) -> list[str]:
    return ["--omni1", str(out), "--s5", str(root / "child"), "--parent", str(root / "parent"), "--tap-rows", str(root / "tap_s5"),
            "--prompt-ids", str(root / "prompts.json"), "--media-prompt-ids", str(root / "media_prompt.json"),
            "--chimera-src", str(CHIMERA_SRC), "--ledger-root", str(LEDGER_ROOT), "--insert-after", "3",
            "--parent-parameter-count", str(10 ** 12), "--dtype", "float32", "--out", str(receipt), *extra]


def test_the_check_passes_on_a_synthetic_composite(composite, tmp_path) -> None:
    root, out, manifest = composite
    receipt_path = tmp_path / "check.json"
    assert check.main(args_for(root, out, receipt_path)) == 0
    r = json.loads(receipt_path.read_text())
    assert r["structural"]["strict_equivalent"] is True
    assert r["strict_load_coverage"]["complete"] is True and r["strict_load_coverage"]["sha256_verified_on_read"] is True
    assert r["accounting"]["omni1_equals_s5_plus_charted"] is True
    ident = r["identity_trunk_vs_s5"]
    assert ident["bit_identical"] == ident["blocks"] == 8
    changed = r["identity_nonblock_vs_s5"]["changed"]
    assert changed and all(v["within_declared_writes"] for v in changed.values())
    assert "embed.weight" in changed and "native_media_bridge.feature_proj.weight" in changed
    text = r["text_forward"]
    assert text["rq_top1_omni1_vs_s5"] == text["positions"] == 8
    assert text["max_abs_logit_diff_omni1_vs_s5_base_vocab"] == 0.0
    assert all(text["logits_finite"].values())
    assert "rq_top1_omni1_vs_parent" in text and "rq_top1_s5_vs_parent" in text
    assert len(r["charted"]) == 2
    for rep in r["charted"].values():
        assert rep["text"]["identity_on_text"] is True
        own = rep["segments"][rep["container"]]
        assert own["own_segment"] and own["rows_owned"] == 6 and own["write_finite"] and own["write_norm_owned"] > 0
        assert own["zeros_off_owned"] is True
        for unit, rec in rep["segments"].items():
            if unit != rep["container"]:
                assert rec["write_norm_owned"] == 0.0
    for rec in r["media_forward"].values():
        assert rec["final_logits_finite"] and rec["native_media_head"]["finite"]
        assert "x31_vs_s5_host_rows" in rec
    assert r["refusals"] == []


def test_a_charted_block_without_host_rows_refuses(composite, tmp_path) -> None:
    root, out, manifest = composite
    unit = manifest["blocks"]["placements"][0]["container"]
    moved = root / "tap_s5" / unit
    hidden = root / "tap_s5_hidden"
    hidden.mkdir(exist_ok=True)
    moved.rename(hidden / unit)
    try:
        with pytest.raises(check.Refusal):
            check.main(args_for(root, out, tmp_path / "r.json"))
    finally:
        (hidden / unit).rename(moved)


def test_a_changed_trunk_tensor_is_caught(composite, tmp_path) -> None:
    """The S5 side read through a copy whose block-0 tensor differs: identity refuses (exit 3), nothing else breaks."""

    root, out, _manifest = composite
    from chimera_omni.importing.safetensors_io import read_checkpoint_config, read_safetensors, write_safetensors
    document = read_checkpoint_config(root / "child")
    state = {name: view.torch() for name, view in read_safetensors(root / "child", verify=False).items()}
    name = next(n for n in sorted(state) if n.startswith("blocks.0.") and state[n].is_floating_point() and state[n].ndim == 2)
    state[name] = state[name].clone()
    state[name][0, 0] += 1.0
    other = tmp_path / "s5_other"
    write_safetensors(state, other, config=document["config"], ledger=document["ledger"], chimera_commit=document["chimera_commit"])
    receipt_path = tmp_path / "bad.json"
    a = args_for(root, out, receipt_path)
    a[a.index("--s5") + 1] = str(other)
    assert check.main(a) == 3
    r = json.loads(receipt_path.read_text())
    assert any("is not bit-identical to S5 block 0" in x for x in r["refusals"])


# ----------------------------------------------------------------------------- 1.1.0: a child composed from a delta plan

def _compose_tool():
    return _load(HERE.parent.parent / "distill" / "tools" / "d1_compose_eval.py", "d1_compose_eval_o1t")


def _plan(out: Path, delta_dir: Path, target: str, rows: int, cols: int, tensor: str = "d0") -> Path:
    sys.path.insert(0, str(LEDGER_ROOT))
    import oc_seal
    digest, entries = oc_seal.directory_digest(out)
    files = [{"path": "delta.safetensors", "bytes": (delta_dir / "delta.safetensors").stat().st_size,
              "sha256": oc_seal.file_sha256(delta_dir / "delta.safetensors")}]
    plan = oc_seal.seal({
        "schema": "neural-forge-omni-delta-plan-v1", "campaign": oc_seal.CAMPAIGN, "tool_version": "test", "checkpoint_id": "TEST-D1",
        "deep": True, "root": {"checkpoint_id": "TEST-ROOT", "record_sha": "0" * 64, "directory": str(out), "directory_sha256": digest,
                               "files": [{"path": e["path"], "bytes": e["bytes"], "sha256": e["sha256"]} for e in entries if e["path"].endswith(".safetensors")]},
        "deltas": [{"checkpoint_id": "TEST-D1", "record_sha": "0" * 64, "directory": str(delta_dir), "directory_sha256": "0" * 64,
                    "base_checkpoint_id": "TEST-ROOT", "base_directory_sha256": digest, "recipe_file_sha256": "0" * 64,
                    "entries": [{"op": "replace", "target": target, "tensor": tensor, "slice": [[0, 0, rows], [1, 0, cols]]}],
                    "files": files, "added_parameters": 0}],
        "composed_parameter_count": 0, "apply_rule": "test"})
    path = delta_dir / f"plan-{rows}.json"
    path.write_text(json.dumps(plan))
    return path


@pytest.fixture(scope="module")
def composed(composite, tmp_path_factory):
    """OMNI1 + a delta replacing rows [0, 2) of a block-0 2-D trunk tensor, composed by d1_compose_eval."""
    root, out, _manifest = composite
    from safetensors import safe_open
    index = json.loads((out / "model.safetensors.index.json").read_text())
    target = next(n for n in sorted(index["weight_map"]) if n.startswith("blocks.0.") and n.endswith("weight")
                  and "norm" not in n)
    with safe_open(str(out / index["weight_map"][target]), "pt") as h:
        base = h.get_tensor(target)
    assert base.ndim == 2 and base.shape[0] >= 2
    work = tmp_path_factory.mktemp("d1compose")
    delta_dir = work / "delta"
    delta_dir.mkdir()
    value = (base[0:2].float() + 1.0).to(base.dtype)
    save_file({"d0": value.contiguous()}, str(delta_dir / "delta.safetensors"))
    plan2 = _plan(out, delta_dir, target, 2, int(base.shape[1]))
    tool = _compose_tool()
    scratch = work / "scratch"
    scratch.mkdir()
    child = scratch / "composed"
    assert tool.main(["compose", "--plan", str(plan2), "--out", str(child), "--scratch-root", str(scratch)]) == 0
    # a second plan that claims only row 0 (its delta = the first row of the first): row 1 then differs outside it
    narrow_dir = work / "narrow"
    narrow_dir.mkdir()
    save_file({"d0": value[0:1].contiguous()}, str(narrow_dir / "delta.safetensors"))
    plan1 = _plan(out, narrow_dir, target, 1, int(base.shape[1]))
    return root, out, child, plan2, plan1, target


def test_compose_receipt_and_index_pins(composed) -> None:
    root, out, child, plan2, _plan1, target = composed
    receipt = json.loads(child.with_name(child.name + ".compose.json").read_text())
    assert receipt["tensors_composed"] == 1 and receipt["root_rederived_from_every_byte"] and receipt["readback_verified"]
    from chimera_omni.importing.safetensors_io import LazySafetensorsMapping
    views = LazySafetensorsMapping(child, verify=True)       # every pin in the rewritten index holds
    assert views[target].sha256() == receipt["changed"][target]["composed_sha256"] != receipt["changed"][target]["base_sha256"]


def test_delta_plan_identity_passes_and_reports_vs_omni1(composite, composed, tmp_path) -> None:
    root, out, child, plan2, _plan1, target = composed
    base_receipt = tmp_path / "omni1.json"
    assert check.main(args_for(root, out, base_receipt)) == 0                  # the fixture's OMNI1 itself
    receipt_path = tmp_path / "d1.json"
    a = args_for(root, out, receipt_path, "--delta-plan", str(plan2), "--omni1-equivalence-receipt", str(base_receipt))
    a[a.index("--omni1") + 1] = str(child)
    assert check.main(a) == 0
    r = json.loads(receipt_path.read_text())
    assert r["refusals"] == [] and r["delta_plan"]["replace_targets"] == 1
    rows = [x for x in r["identity_trunk_vs_s5"]["per_block"] if not x["bit_identical"]]
    assert len(rows) == 1 and rows[0]["delta_tensors"] == 1 and rows[0]["region_elements_differing_from_s5"] > 0
    assert rows[0]["bit_identical_outside_delta_regions"] is True
    eq = r["text_forward"]["omni1_equivalence"]
    assert eq["holds"] is True and r["text_forward"]["rq_top1_child_vs_omni1"] == r["text_forward"]["rq_top1_omni1_vs_s5"]


def test_delta_child_without_plan_or_with_a_narrower_plan_refuses(composed, tmp_path) -> None:
    root, out, child, _plan2, plan1, _target = composed
    a = args_for(root, out, tmp_path / "noplan.json")
    a[a.index("--omni1") + 1] = str(child)
    assert check.main(a) == 3
    assert any("is not bit-identical to S5 block 0" in x for x in json.loads((tmp_path / "noplan.json").read_text())["refusals"])
    b = args_for(root, out, tmp_path / "narrow.json", "--delta-plan", str(plan1))
    b[b.index("--omni1") + 1] = str(child)
    assert check.main(b) == 3
    refusals = json.loads((tmp_path / "narrow.json").read_text())["refusals"]
    assert any("outside the delta plan's regions" in x and "changed outside its delta regions" in x for x in refusals), refusals
