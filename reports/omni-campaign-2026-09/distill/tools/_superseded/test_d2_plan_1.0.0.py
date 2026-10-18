"""Tests for d2_plan.py (the decisions behind queues/d2_launch.sh).  CPU only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_plan.py

A fake campaign tree (registry manifest, sealed / unsealed Q38 cache manifests, mask files, K3 pass manifests) in a
temp dir; the real prereg (beside this tools directory) for the A2 draft -> in-force step.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d2_plan as P  # noqa: E402

PREREG = HERE.parent / "PREREGISTRATION_OMNI_STAGE_D_v1.md"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_cache(root: Path, corpus: Path, tokens: int, rows: int, width: int) -> Path:
    d = root / sha(corpus)
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"all_finite": True, "all_structural_ok": True,
                                                 "teacher": {"equal_before_and_after_every_session": True},
                                                 "corpus": {"path": str(corpus), "sha256": sha(corpus), "tokens": tokens,
                                                            "rows": rows, "row_length": width}}))
    return d


def ids_file(path: Path, rows: int, width: int, lengths: list[int] | None = None) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    t = {"input_ids": np.zeros((rows, width), dtype=np.int32)}
    if lengths is not None:
        t["lengths"] = np.array(lengths, dtype=np.int64)
    save_file(t, str(path))
    return path


@pytest.fixture()
def world(tmp_path, monkeypatch):
    oc, lex = tmp_path / "oc", tmp_path / "lex"
    layout = P.Layout(oc, lex)
    shards = []
    for i in range(2):
        f = ids_file(oc / "distill" / "corpus" / "text" / f"text-{i:05d}.safetensors", 3, 8)
        shards.append({"path": str(f), "sha256": sha(f), "rows": [3 * i, 3 * i + 3]})
    layout.text_manifest.write_text(json.dumps({"shards": shards}))
    for s in shards:
        sealed_cache(layout.cache_root, Path(s["path"]), 24, 3, 8)
    # self-gen and long are pinned by content in production; the test pins its own files
    sg = ids_file(lex / "inputs" / "rescore_target.safetensors", 2, 12, [10, 7])
    lg = ids_file(oc / "distill" / "corpus" / "long" / "long-00000.safetensors", 1, 32)
    monkeypatch.setattr(P, "SELFGEN_CORPUS_SHA", sha(sg))
    monkeypatch.setattr(P, "LONG_CORPUS_SHA", sha(lg))
    monkeypatch.setattr(P, "SEQ", 8)
    sealed_cache(layout.cache_root, sg, 17, 2, 12)
    sealed_cache(layout.cache_root, lg, 32, 1, 32)
    tc = ids_file(layout.toolcall_corpus, 2, 6, [6, 5])
    bf = ids_file(layout.bfcl_corpus, 1, 5, [5])
    sealed_cache(layout.cache_root, tc, 11, 2, 6)
    for k in range(4):
        q = ids_file(layout.k3_corpus / f"pass-{k}" / "qwen_input_ids.safetensors", 2, 8, [8, 6])
        sealed_cache(layout.cache_root, q, 14, 2, 8)
    (layout.k3_corpus / "manifest.json").write_text(json.dumps({"manifest_sha256": "ab" * 32}))
    layout.masks.mkdir(parents=True)
    pins = {}
    for name in P.MASK_SHAS:
        body = {"exclude_rows.json": {"schema": "omni-d2-row-exclusion-v1", "corpora": {shards[1]["sha256"]: {"rows": [1]}}}}.get(name, {"x": name})
        (layout.masks / name).write_text(json.dumps(body))
        pins[name] = sha(layout.masks / name)
    monkeypatch.setattr(P, "MASK_SHAS", pins)
    (layout.masks / "build_receipt.json").write_text(json.dumps({"doc_index_check": {"ok": True}}))
    layout.k3_frozen_tool.parent.mkdir(parents=True)
    layout.k3_frozen_tool.write_text("# frozen k3 tool\n")
    return layout, bf


def test_inventory_waits_for_the_bfcl_cache_and_then_opens(world, capsys):
    layout, bf = world
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "inventory"]) == P.EXIT_WAITING
    assert "bfcl_c2" in capsys.readouterr().out
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "inventory"]) == 0
    # a mask whose sha is not the pinned one closes the gate again
    (layout.masks / "gt_mask.json").write_text("{}")
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "inventory"]) == P.EXIT_WAITING


def test_data_config_budgets_exclusion_deferral_and_width(world, tmp_path):
    layout, bf = world
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    out = tmp_path / "data.json"
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "data-config", "--out", str(out)]) == 0
    cfg = json.loads(out.read_text())
    names = [s["name"] for s in cfg["sources"]]
    assert names[:2] == ["registry-text-00000", "registry-text-00001"] and names[2:5] == ["selfgen", "tool-selfgen", "tool-bfcl-native"]
    reg = cfg["sources"][:2]
    assert cfg["registry_positions_after_exclusion"] == (6 - 1) * 8  # one excluded row
    assert sum(s["tokens"] for s in reg) == cfg["registry_budget_positions"] == 40  # capped at one epoch
    assert reg[1]["tokens"] == 16 and all(s["exclude_rows"].endswith("exclude_rows.json") for s in reg)
    tool = cfg["sources"][3]
    assert tool["supervision"] == "assistant_only" and tool["w_ul"] == 0.0 and tool["budget_unit"] == "real_tokens"
    k3 = [s for s in cfg["sources"] if s["kind"] == "k3"]
    assert len(k3) == 4 and all(s["defer_until_tokens"] == P.K3_DEFER_TOKENS and s["sealed_marker"] == str(layout.k3_marker) for s in k3)
    assert k3[0]["tokens"] == 14 and cfg["deferred_rows"] == 8
    assert cfg["max_width"] == 12  # the self-gen rows are the widest


def test_verdict_rule_and_file_validation(world, tmp_path, capsys):
    layout, _ = world
    args = ["--oc", str(layout.oc), "--lex", str(layout.lex), "verdict"]
    assert P.main(args) == P.EXIT_WAITING  # no summary yet
    layout.p2_summaries.mkdir(parents=True)

    def summary(gpu, cells):
        (layout.p2_summaries / f"p2-summary-v133-gpu{gpu}.json").write_text(json.dumps({"gpu_index": gpu, "cells": cells}))

    summary(0, {"L4096-mb8": {"state": "completed", "steady_tok_per_s_mean_steps_2_on": 160.0},
                "L4096-mb4": {"state": "completed", "steady_tok_per_s_mean_steps_2_on": 170.5},
                "L32768-mb1": {"state": "failed"}})
    assert P.main(args) == 0 and not layout.p2_verdict.exists()  # derived, not written
    assert P.main(args + ["--write"]) == 0
    v = P.read_verdict_file(layout.p2_verdict)
    assert v["engine"] == "streamed" and (v["micro_batch"], v["accumulation"]) == ("4", "2") and v["long_rung_cell_completed"] == "no"
    assert v["by"].startswith("auto")
    layout.p2_verdict.unlink()
    summary(0, {"L4096-mb8": {"state": "completed", "steady_tok_per_s_mean_steps_2_on": 120.0}, "L32768-mb1": {"state": "completed"}})
    d = P.derive_verdict(sorted(layout.p2_summaries.glob("*.json")))
    assert d["engine"] == "split" and (d["micro_batch"], d["accumulation"]) == ("4", "4") and d["long_rung_cell_completed"] == "yes"
    layout.p2_verdict.write_text("engine=fast\nmicro_batch=8\naccumulation=1\n")
    capsys.readouterr()
    assert P.main(args) == 2 and "engine must be" in capsys.readouterr().err


def test_a2_draft_is_found_and_finalized_only_with_sealed_c1_c2(world, tmp_path, capsys):
    layout, bf = world
    if "### A2, DRAFT" not in PREREG.read_text(encoding="utf-8"):
        pytest.skip("the prereg beside the tools no longer carries the A2 draft")
    prereg = tmp_path / "PREREG.md"
    shutil.copy2(PREREG, prereg)
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex)]
    assert P.main(base + ["a2-state", "--prereg", str(prereg)]) == P.EXIT_WAITING
    state = P.a2_state(prereg)
    assert state["a2_draft"] and state["placeholders"] == ["C1", "C2"]
    assert P.main(base + ["a2-finalize", "--prereg", str(prereg)]) == P.EXIT_WAITING  # C2 not sealed
    monkey_tokens = P.BFCL_EXPECT["tokens"]
    sealed_cache(layout.cache_root, bf, monkey_tokens, 1, 5)
    capsys.readouterr()
    assert P.main(base + ["a2-finalize", "--prereg", str(prereg)]) == 0
    after = P.a2_state(prereg)
    assert after["in_force"] and after["placeholders"] == []
    text = prereg.read_text(encoding="utf-8")
    assert "### A2, in force" in text and "`<C1>`" not in text
    assert list(tmp_path.glob("PREREG.md.pre-A2final-*"))  # backed up
    pin = P.check_prereg(prereg, HERE)
    assert pin["preregistration_sha256"] == "c24d21e08815d7acd2ce14aea989d16f84f167eb24a40597966be19284cb1370"


def test_k3_state_counts_only_passes_sealed_by_the_frozen_tool(world, capsys):
    layout, _ = world
    tool_sha = sha(layout.k3_frozen_tool)
    out = layout.k3_out()
    for k in range(4):
        (out / f"pass-{k}-of-4").mkdir(parents=True)
        by = tool_sha if k < 3 else "other"
        (out / f"pass-{k}-of-4" / "manifest.json").write_text(json.dumps({"verdict": "sealed", "tool_sha256": by}))
    state = P.k3_state(layout)
    assert state["sealed"] == 3 and not state["marker_exists"]
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "k3-seal"]) == P.EXIT_WAITING
    assert "3 of 4 K3 passes" in capsys.readouterr().out
