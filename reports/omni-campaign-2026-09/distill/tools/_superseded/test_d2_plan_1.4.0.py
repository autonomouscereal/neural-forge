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


def ids_file(path: Path, rows: int, width: int, lengths: list[int] | None = None, fill: int = 0) -> Path:
    import numpy as np
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    t = {"input_ids": np.full((rows, width), fill, dtype=np.int32)}
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
        # distinct content per shard: exclusions are keyed by corpus sha256, so identical shards would share one
        f = ids_file(oc / "distill" / "corpus" / "text" / f"text-{i:05d}.safetensors", 3, 8, fill=i + 1)
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
    k3q = []
    for k in range(4):
        q = ids_file(layout.k3_corpus / f"pass-{k}" / "qwen_input_ids.safetensors", 2, 8, [8, 6 + k])
        sealed_cache(layout.cache_root, q, 14 + k, 2, 8)
        k3q.append(q)
    (layout.k3_corpus / "manifest.json").write_text(json.dumps({"manifest_sha256": "ab" * 32}))
    layout.masks.mkdir(parents=True)
    pins = {}
    for name in P.MASK_SHAS:
        # registry file 1 loses its row 1; K3 pass 1 loses its row 1 (a twin row: 6 + 1 = 7 real tokens)
        body = {"exclude_rows.json": {"schema": "omni-d2-row-exclusion-v1", "corpora": {shards[1]["sha256"]: {"rows": [1]},
                                                                                       sha(k3q[1]): {"rows": [1]}}}}.get(name, {"x": name})
        (layout.masks / name).write_text(json.dumps(body))
        pins[name] = sha(layout.masks / name)
    monkeypatch.setattr(P, "MASK_SHAS", pins)
    (layout.masks / "build_receipt.json").write_text(json.dumps({"doc_index_check": {"ok": True}}))
    layout.k3_frozen_tool.parent.mkdir(parents=True, exist_ok=True)
    layout.k3_frozen_tool.write_text("# frozen k3 tool\n")
    layout.test_shards = shards
    layout.test_tc = tc
    write_decisions(layout, monkeypatch, tc)
    return layout, bf


def write_decisions(layout: P.Layout, monkeypatch, tc: Path, *, extra_rows: dict | None = None, screen: bool = True) -> None:
    """1.3.0: a D2_DECISIONS-shaped file (pinned by monkeypatch), a d2_masks_v2 build as d2_mask_build 1.2.0 leaves it
    (v1 exclusion + `extra_rows` {corpus sha: [rows]}; 10 trained positions removed of 40), the tool-call screen receipt."""

    d = layout.decisions
    d.parent.mkdir(parents=True, exist_ok=True)
    d.write_text(json.dumps({"schema": "omni-campaign.d2-decisions.v1", "id": "D2_DECISIONS_TEST", "rulings": {"ids": ["x1", "x2"]},
                             "applied_receipt": "/srv/omni-campaign/" + P.APPLIED_SUFFIX + " (written by the D2 lane: this file's sha256)"}))
    monkeypatch.setattr(P, "DECISIONS_SHA", sha(d))
    excl = json.loads((layout.masks / "exclude_rows.json").read_text())
    for corpus, rows in (extra_rows or {}).items():
        entry = excl["corpora"].setdefault(corpus, {"rows": []})
        entry["rows"] = sorted(set(entry["rows"]) | set(rows))
    v2 = layout.masks_v2
    v2.mkdir(parents=True, exist_ok=True)
    files = {"exclude_rows.json": excl, "all_mask.json": {"schema": "omni-d2-position-mask-v1", "kind": "all", "corpora": {}},
             "gt_mask.json": {"schema": "omni-d2-position-mask-v1", "kind": "gt", "corpora": {}}, "heldout_rids.json": {"rids": []}}
    outputs = {}
    for name, body in files.items():
        (v2 / name).write_text(json.dumps(body))
        outputs[name] = sha(v2 / name)
    receipt = {"kind": "d2-decisions", "tool_version": "1.2.0", "decisions": {"sha256": sha(d)}, "outputs": outputs,
               "inputs": {f"v1_{n}": P.MASK_SHAS[n] for n in P.MASK_SHAS}, "rechecks": {"hits": 0}, "count_check": {"ok": True},
               "doc_index_check": {"ok": True}, "counts": {"no_loss_rows": sum(len(r) for r in (extra_rows or {}).values())},
               "refill": {"heldout_trained_positions_removed": 10, "positions_after_v1_exclusion": 40, "kd_positions_v1": 38,
                          "kd_positions_v2": 28}}
    (v2 / "build_receipt.json").write_text(json.dumps(receipt))
    if screen:
        layout.toolcall_screen.write_text(json.dumps({"schema": P.SCREEN_SCHEMA, "decisions_sha256": sha(d), "corpus_sha256": sha(tc),
                                                      "ngram": 13, "hits_in_corpus": 0}))


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
    # the K3 twin row leaves pass 1's budget and the deferred row count; the other passes keep every row
    assert [s["tokens"] for s in k3] == [14, 8, 16, 17] and cfg["deferred_rows"] == 7
    assert all(s["exclude_rows"].endswith("exclude_rows.json") for s in k3 + [cfg["sources"][2]])  # K3 + self-gen carry it
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


# ---------------------------------------------------------------------------------------------- K3 acceptance (1.3.0)
QUADRO = [{"index": 0, "name": "Quadro RTX 8000", "capability": "sm75"}]
RTX3090 = [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "capability": "sm86"}]
P40 = [{"index": 0, "name": "Tesla P40", "capability": "sm61"}]
V124, V130 = P.K3_V124_SHA, P.K3_V130_SHA


def k3_b2(layout: P.Layout, *, parity: str = "pass") -> None:
    """The B2 dir as queue_gpu6 leaves a PASS: parity receipt, the bf16 and fp32 review passes and the 93-layer identity."""

    b2 = layout.k3_b2_base
    for d in ("review-pass", "review-pass-fp32", "identity"):
        (b2 / d).mkdir(parents=True, exist_ok=True)
    (b2 / "queue_tool.sha256").write_text(V124 + "\n")
    for d in ("review-pass", "review-pass-fp32"):
        (b2 / d / "manifest.json").write_text(json.dumps({"verdict": "sealed", "tool_sha256": V124}))
    (b2 / "parity_b2_receipt.json").write_text(json.dumps({"verdict": parity}))
    (b2 / "identity" / "b2_identity.json").write_text(json.dumps({"mode": "identity", "frozen_tool": {"sha256": V124},
                                                                  "verdict": {"pass": True, "layers_checked": 93, "layers_expected": 93}}))
    layout.k3_rebase_ruling.write_text("k3-b2-rebaseline-fp32-production-geometry-2026-10-14-r2\n")


def seal_pass(layout: P.Layout, k: int, tool: str, devices: list, *, gt: int = 65536, record: bool = True, cls: str | None = None,
              gate: str | None = None, geometry_gt: int | None = None) -> Path:
    """A sealed K3 pass dir, its <pd>.geometry and (record=True) its k3_lib k3_record_pass entry."""

    pd = layout.k3_out() / f"pass-{k}-of-4"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "manifest.json").write_text(json.dumps({"verdict": "sealed", "tool_sha256": tool, "device": "cuda:0", "devices": devices,
                                                  "corpus": {"group_tokens": gt}, "arguments": {"group_tokens": gt}}))
    ran_on = P.card_class_of(devices[0]["name"], devices[0]["capability"])
    Path(f"{pd}.geometry").write_text(json.dumps({"tool_sha256": tool, "card_class": ran_on,
                                                  "group_tokens": gt if geometry_gt is None else geometry_gt}))
    if record:
        rec = json.loads(layout.k3_pass_tools.read_text()) if layout.k3_pass_tools.is_file() else {"schema": "omni-k3-pass-tools-v1", "passes": {}}
        rec["passes"][str(k)] = {"dir": str(pd), "tool_sha256": tool, "card_class": cls or ran_on, "gate_receipt": gate,
                                 "manifest_sha256": sha(pd / "manifest.json")}
        layout.k3_pass_tools.write_text(json.dumps(rec))
    return pd


def v130_gate(layout: P.Layout, *, verdict: str = "pass", neg_identical: bool = False) -> str:
    """queue_gpu6 1.4.0's $Q/k3_v130_gate/<class>/verdict.json and its four compare receipts."""

    run = layout.lex / "scratch" / "gpu6-gate"
    run.mkdir(parents=True, exist_ok=True)
    shas = {}
    for name, identical in P.K3_V130_COMPARES.items():
        same = neg_identical if name == "compare_w1_A_NEG.json" else identical
        (run / name).write_text(json.dumps({"bit_identical": same, "a": {"sha256": V124}, "b": {"sha256": V130}, "tensors": 12}))
        shas[name] = sha(run / name)
    g = layout.k3_v130_gate / P.K3_HOME_CLASS / "verdict.json"
    g.parent.mkdir(parents=True, exist_ok=True)
    g.write_text(json.dumps({"schema": "omni-k3-v130-gate-v1", "verdict": verdict, "tool_sha256": V130, "reference_tool_sha256": V124,
                             "card_class": P.K3_HOME_CLASS, "run_dir": str(run), "compare_receipts_sha256": shas,
                             "compare_exits": f"w1=0 w2p=0 w4=0 w1_NEG={0 if neg_identical else 3}"}))
    return str(g)


def cardclass(layout: P.Layout, cls: str, tool: str, *, verdict: str = "pass") -> str:
    """k3_multicard.sh's $Q/k3_cardclass/<class>/verdict.json and its parity receipt against the B2 fp32 reference."""

    eq = layout.lex / "eq" / cls
    eq.mkdir(parents=True, exist_ok=True)
    (eq / "parity_vs_b2_fp32.json").write_text(json.dumps({"verdict": verdict}))
    v = layout.k3_cardclass / cls / "verdict.json"
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_text(json.dumps({"schema": "omni-k3-cardclass-v1", "ruling": P.K3_CARD_CLASS_RULING, "card_class": cls, "tool_sha256": tool,
                             "parity_code_tool_sha256": V124, "verdict": verdict,
                             "gate": {"path": str(eq / "parity_vs_b2_fp32.json"), "sha256": sha(eq / "parity_vs_b2_fp32.json"),
                                      "verdict": verdict}}))
    return str(v)


C3090 = "nvidia-geforce-rtx-3090-sm86"


def four_routes(layout: P.Layout) -> None:
    """Pass 0: 1.2.4 on GPU6 (recorded, B2); 1: 1.2.4 on GPU6 sealed by a queue that kept no record (legacy);
    2: 1.3.0 on GPU6 behind the k3-v130-gate; 3: 1.2.4 on a 3090 behind its card-class PASS."""

    k3_b2(layout)
    seal_pass(layout, 0, V124, QUADRO)
    seal_pass(layout, 1, V124, QUADRO, record=False)
    seal_pass(layout, 2, V130, QUADRO, gate=v130_gate(layout))
    seal_pass(layout, 3, V124, RTX3090, gt=32768, gate=cardclass(layout, C3090, V124))


def test_k3_state_accepts_the_three_gated_routes_and_the_legacy_pass(world, capsys):
    layout, _ = world
    four_routes(layout)
    state = P.k3_state(layout)
    assert state["sealed"] == 4 and not state["refused"] and not state["pending"], state["refused"]
    kinds = [p["gate"]["kind"] for p in state["passes"]]
    assert kinds == ["b2", "b2", "k3-v130-gate", "card-class"]
    assert state["passes"][1]["source"].startswith("legacy") and state["passes"][0]["source"] == "k3_pass_tools.json"
    assert [p["card_class"] for p in state["passes"]] == [P.K3_HOME_CLASS] * 3 + [C3090]
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "k3-state"]) == P.EXIT_WAITING  # no marker yet
    # the xtok builder must be the B2-proven 1.2.4 itself (the fixture's frozen file is not)
    capsys.readouterr()
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "k3-seal"]) == 2
    assert "not the B2-proven 1.2.4" in capsys.readouterr().err


def problems_of(layout: P.Layout, k: int) -> str:
    st = P.k3_state(layout)["passes"][k]
    assert st["state"] == "refused", st
    return " | ".join(st["problems"])


def test_k3_negative_controls_missing_receipt_wrong_sha_failed_gate(world, capsys):
    layout, _ = world
    four_routes(layout)
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex)]
    # MISSING RECEIPT: the 3090's card-class verdict is gone
    v = layout.k3_cardclass / C3090 / "verdict.json"
    body = v.read_text()
    v.unlink()
    assert "not a PASS of tool" in problems_of(layout, 3)
    capsys.readouterr()
    assert P.main(base + ["k3-seal"]) == 2 and "not acceptable" in capsys.readouterr().err  # a refusal, never a wait
    v.write_text(body)
    # MISSING RECEIPT: the card-class parity receipt changed after the verdict
    (layout.lex / "eq" / C3090 / "parity_vs_b2_fp32.json").write_text(json.dumps({"verdict": "pass", "x": 1}))
    assert "missing, changed" in problems_of(layout, 3)
    cardclass(layout, C3090, V124)
    # WRONG SHA: the 3090 class passed with 1.3.0, its pass ran 1.2.4
    cardclass(layout, C3090, V130)
    assert "not a PASS of tool 2aae4dc3134dda72" in problems_of(layout, 3)
    cardclass(layout, C3090, V124)
    # WRONG SHA: a pass sealed by an unpinned tool
    seal_pass(layout, 0, "f" * 64, QUADRO)
    assert "not a pinned K3 tool" in problems_of(layout, 0)
    seal_pass(layout, 0, V124, QUADRO)
    # WRONG SHA: the manifest changed after the record
    man = layout.k3_out() / "pass-0-of-4" / "manifest.json"
    man.write_text(man.read_text().replace('"sealed"', '"sealed" ', 1))
    assert "record manifest sha" in problems_of(layout, 0)
    seal_pass(layout, 0, V124, QUADRO)
    # FAILED GATE: k3-v130-gate DIFFER, and a negative control that did not DIFFER
    v130_gate(layout, verdict="differ")
    assert "not a PASS of 1.3.0" in problems_of(layout, 2)
    v130_gate(layout, neg_identical=True)
    p = problems_of(layout, 2)
    assert "the +1-ulp control DIFFER" in p and "compare_w1_A_NEG.json: bit_identical True" in p
    v130_gate(layout)
    # FAILED GATE: B2 did not pass -> no pass counts, the legacy one included
    k3_b2(layout, parity="fail")
    st = P.k3_state(layout)
    assert st["sealed"] == 0 and len(st["refused"]) == 4 and all("no B2 PASS" in " ".join(r["problems"]) for r in st["refused"])
    k3_b2(layout)
    assert P.k3_state(layout)["sealed"] == 4
    # a 1.3.0 pass no record names is pending (never accepted as legacy); a P40 class is out of scope
    seal_pass(layout, 2, V130, QUADRO, record=False)
    rec = json.loads(layout.k3_pass_tools.read_text())
    del rec["passes"]["2"]
    layout.k3_pass_tools.write_text(json.dumps(rec))
    st = P.k3_state(layout)
    assert st["passes"][2]["state"] == "pending" and "not (yet) recorded" in st["passes"][2]["why"]
    seal_pass(layout, 2, V124, P40, gate=cardclass(layout, "tesla-p40-sm61", V124))
    assert "out of scope" in problems_of(layout, 2)


def test_k3_mixed_geometry_within_a_pass_is_refused(world):
    layout, _ = world
    four_routes(layout)
    # the 1.3.x OOM switch: a pass begun at 65,536 group tokens and resumed at 32,768 (the manifest and the legacy
    # geometry record only the last session) -- the runner log's attempt lines show both
    seal_pass(layout, 1, V124, QUADRO, gt=32768, record=False)
    log = layout.k3_logs / "gpu6" / "k3-pass-1.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("== k3-pass-1 attempt 1 group_tokens 65536 2026-10-16T01:00:00Z\nOutOfMemoryError\n"
                   "== k3-pass-1 attempt 2 group_tokens 32768 2026-10-16T01:10:00Z\n")
    assert "mixed geometry within the pass: runner logs show group tokens [32768, 65536]" in problems_of(layout, 1)
    log.write_text("== k3-pass-1 attempt 1 group_tokens 32768 2026-10-16T01:00:00Z\n")
    assert P.k3_state(layout)["passes"][1]["state"] == "accepted"
    # the geometry recorded at the first start differs from the manifest
    seal_pass(layout, 0, V124, QUADRO, geometry_gt=32768)
    assert ".geometry group_tokens 32768 != 65536" in problems_of(layout, 0)
    seal_pass(layout, 0, V124, QUADRO)
    # a pass that moved card class: the record says GPU6's class, the manifest ran on a 3090
    seal_pass(layout, 0, V124, RTX3090, cls=P.K3_HOME_CLASS)
    assert "record card class quadro-rtx-8000-sm75 != the class the manifest ran on (nvidia-geforce-rtx-3090-sm86)" in problems_of(layout, 0)
    seal_pass(layout, 0, V124, QUADRO)
    # a 1.4.0 attempt line naming another tool or class in the same pass
    log0 = layout.k3_logs / "k3mc-gpu0" / "k3-pass-0.log"
    log0.parent.mkdir(parents=True, exist_ok=True)
    log0.write_text(f"== k3-pass-0 attempt 1 group_tokens 65536 tool {V130[:16]} (v130) class {P.K3_HOME_CLASS} gpu 6 x\n")
    assert "mixed tools within the pass" in problems_of(layout, 0)
    log0.unlink()
    assert P.k3_state(layout)["sealed"] == 4


def test_k3_marker_is_valid_only_for_the_passes_it_sealed(world, capsys):
    layout, _ = world
    four_routes(layout)
    st = P.k3_state(layout)
    built = []
    for p in st["passes"]:
        xt = Path(p["xtok"])
        xt.parent.mkdir(parents=True, exist_ok=True)
        xt.write_bytes(b"xtok-%d" % p["pass"])
        built.append({"pass": p["pass"], "k3_manifest_sha256": p["manifest_sha256"], "xtok_sha256": sha(xt)})
    layout.k3_marker.parent.mkdir(parents=True, exist_ok=True)
    layout.k3_marker.write_text(json.dumps({"schema": "omni-d2-k3-seal-v1", "passes": built}))
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex)]
    assert P.main(base + ["k3-state"]) == 0 and P.k3_state(layout)["marker_valid"]
    assert P.main(base + ["k3-seal"]) == 0  # valid: nothing to do
    # pass 2 re-sealed after the marker (its record follows): accepted again, but the marker no longer covers it
    seal_pass(layout, 2, V130, QUADRO, gate=v130_gate(layout))
    man = layout.k3_out() / "pass-2-of-4" / "manifest.json"
    man.write_text(man.read_text() + "\n")
    rec = json.loads(layout.k3_pass_tools.read_text())
    rec["passes"]["2"]["manifest_sha256"] = sha(man)
    layout.k3_pass_tools.write_text(json.dumps(rec))
    st = P.k3_state(layout)
    assert st["sealed"] == 4 and not st["marker_valid"] and "manifest changed since the seal" in st["marker_check"]
    assert P.main(base + ["k3-state"]) == 2
    capsys.readouterr()
    assert P.main(base + ["k3-seal"]) == 2 and "exists but does not match" in capsys.readouterr().err
    # a 1.2.0 marker (no manifest shas) is never taken as valid
    layout.k3_marker.write_text(json.dumps({"schema": "omni-d2-k3-seal-v1", "passes": [{"pass": k, "xtok_sha256": "x"} for k in range(4)]}))
    assert not P.k3_state(layout)["marker_valid"]


# ---------------------------------------------------------------------------------------------- D2_DECISIONS (1.3.0)
def short_config(layout: P.Layout, tmp_path: Path, name: str = "data.json") -> dict:
    out = tmp_path / name
    assert P.main(["--oc", str(layout.oc), "--lex", str(layout.lex), "data-config", "--out", str(out)]) == 0
    return json.loads(out.read_text())


def test_data_config_applies_the_v2_masks_and_declares_the_refill(world, tmp_path, monkeypatch):
    layout, bf = world
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    shards = layout.test_shards
    write_decisions(layout, monkeypatch, layout.test_tc, extra_rows={shards[0]["sha256"]: [2]})  # one no-loss row
    cfg = short_config(layout, tmp_path)
    reg = [s for s in cfg["sources"] if s["name"].startswith("registry-")]
    assert all(s["exclude_rows"] == str(layout.masks_v2 / "exclude_rows.json") and s["gt_mask"] == str(layout.masks_v2 / "gt_mask.json")
               and s["all_mask"] == str(layout.masks_v2 / "all_mask.json") for s in reg)
    assert cfg["registry_positions_after_exclusion"] == (6 - 2) * 8 and sum(s["tokens"] for s in reg) == 32
    # the K3 and self-gen sources keep the A2 3(a) file (their twin rows; D2_DECISIONS leaves them unchanged)
    assert all(s["exclude_rows"] == str(layout.masks / "exclude_rows.json") for s in cfg["sources"] if s["kind"] == "k3" or s["name"] == "selfgen")
    d = cfg["decisions"]
    assert d["sha256"] == P.DECISIONS_SHA and d["variant"] == "span-mask" and "no clean registry version" in d["variant_reason"]
    assert d["registry_masks"]["exclude_rows"]["sha256"] == sha(layout.masks_v2 / "exclude_rows.json")
    r = cfg["refill"]
    assert r["target_tokens"] == 10 and r["supplied_estimate"] == 0 and r["shortfall_estimate"] == 10
    assert [x["rank"] for x in r["ranks"]] == [1, 2, 3, 3, 4] and "not used" in r["ranks"][1]["why"]
    # a sealed rank-2 corpus is taken, capped at the remainder; it never makes D2 wait
    rf = ids_file(layout.refill_dir / "selfgen-refill.safetensors", 2, 8, [8, 4])
    sealed_cache(layout.cache_root, rf, 12, 2, 8)
    # 1.4.0 NEGATIVE: a sealed refill corpus without its eligibility screen (decisions refill_policy), or with a hit,
    # supplies nothing, whatever its cache
    cfg = short_config(layout, tmp_path, "data_noscreen.json")
    r = cfg["refill"]
    assert r["supplied_estimate"] == 0 and "no eligibility screen receipt" in r["ranks"][1]["why"]
    assert not any(s["name"] == "refill-selfgen" for s in cfg["sources"])
    screen = rf.with_name(rf.name + ".heldout_screen.json")
    good = {"schema": P.REFILL_SCREEN_SCHEMA, "decisions_sha256": P.DECISIONS_SHA, "corpus_sha256": sha(rf), "ngram": 13,
            "hits_in_corpus": 0, "split": "train", "excluded_id_hits": 0, "verified_rows_only": True}
    for bad in ({"hits_in_corpus": 1}, {"split": "validation"}, {"excluded_id_hits": 2}, {"corpus_sha256": "0" * 64}):
        screen.write_text(json.dumps({**good, **bad}))
        cfg = short_config(layout, tmp_path, "data_badscreen.json")
        assert cfg["refill"]["supplied_estimate"] == 0 and "not eligible" in cfg["refill"]["ranks"][1]["why"], bad
    screen.write_text(json.dumps(good))
    cfg = short_config(layout, tmp_path, "data2.json")
    r = cfg["refill"]
    assert r["ranks"][1]["tokens_budgeted"] == 10 and r["supplied_estimate"] == 10 and r["shortfall_estimate"] == 0
    assert r["ranks"][1]["eligibility"]["ok"] is True
    src = next(s for s in cfg["sources"] if s["name"] == "refill-selfgen")
    assert src["tokens"] == 10 and src["budget_unit"] == "real_tokens" and "defer_until_tokens" not in src
    # 1.4.0: the span form records the switch (no clean version: `d2_clean_repack.py state` exits 10) and its label
    sw = cfg["decisions"]["clean_switch"]
    assert sw["taken"] is False and sw["d2_clean_repack_state"]["rc"] == P.EXIT_WAITING
    assert cfg["decisions"]["heldout_label"] == "loss-excluded in D2; context-exposed in shared packed rows"
    assert cfg["decisions"]["clean_cache_manifests"] is None


def runner_sealed(layout: P.Layout, files: list[Path], verify_rc: int = 0) -> None:
    """d2clean_recache 1.0.1's status line after a seal (the frozen qtext tool's verify rc)."""

    layout.clean_status.parent.mkdir(parents=True, exist_ok=True)
    with open(layout.clean_status, "a") as handle:
        for f in files:
            handle.write(f"2026-10-16T10:00:00Z d2clean gpu3 SEALED {f.name} rc 0; verify rc {verify_rc}: VERIFY ok\n")


def write_clean(layout: P.Layout, *, seal: bool = True, verified: bool = True, decisions: str | None = None) -> list[Path]:
    """Lane d2clean's version as d2_clean_repack 1.0.0 writes it ($OC/distill/corpus/text_clean_v1): two re-pack files and
    their Q38 caches; clean_exclude_rows.json drops ALL of sealed file 0 and row 0 of file 1; verify_receipt.json.
    1.4.0: read by the real d2_clean_repack.py `state` (beside d2_plan), not by a stand-in."""

    d = layout.clean_dir
    d.mkdir(parents=True, exist_ok=True)
    text = json.loads(layout.text_manifest.read_text())
    text["manifest_sha256"] = "5e" * 32
    layout.text_manifest.write_text(json.dumps(text))
    files = [ids_file(d / f"text-clean-{i:05d}.safetensors", 2, 8, fill=10 + i) for i in range(2)]
    if seal:
        for f in files:
            sealed_cache(layout.cache_root, f, 16, 2, 8)
    shards = layout.test_shards
    masks = {}
    for name, body in (("clean_exclude_rows.json", {"schema": "omni-d2-row-exclusion-v1",
                                                    "corpora": {shards[0]["sha256"]: {"rows": [0, 1, 2]}, shards[1]["sha256"]: {"rows": [0]}}}),
                       ("clean_all_mask.json", {"schema": "omni-d2-position-mask-v1", "kind": "all", "corpora": {}}),
                       ("clean_gt_mask.json", {"schema": "omni-d2-position-mask-v1", "kind": "gt", "corpora": {}})):
        (d / name).write_text(json.dumps(body))
        masks[name] = {"path": str(d / name), "sha256": sha(d / name)}
    man = {"repack_schema": "omni-d2-clean-repack-v1", "manifest_sha256": "c1" * 32,
           "inputs": {"decisions": {"sha256": decisions or P.DECISIONS_SHA}}, "sealed_corpus": {"manifest_sha256": "5e" * 32},
           "masks": masks, "counts": {"kd_positions_clean_total": 20},
           "shards": [{"file": f.name, "path": str(f), "sha256": sha(f), "rows": [2 * i, 2 * i + 2]} for i, f in enumerate(files)]}
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "verify_receipt.json").write_text(json.dumps({"ok": verified, "manifest_sha256": man["manifest_sha256"],
                                                       "manifest_file_sha256": sha(d / "manifest.json")}))
    return files


def test_data_config_prefers_the_verified_cached_clean_registry_and_falls_back_recorded(world, tmp_path):
    layout, bf = world
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    files = write_clean(layout, seal=False)
    cfg = short_config(layout, tmp_path, "pending.json")  # re-pack caches not sealed: span-mask, with the reason
    assert cfg["decisions"]["variant"] == "span-mask" and "2 Q38 cache(s) of the clean registry not sealed" in cfg["decisions"]["variant_reason"]
    assert "`d2_clean_repack.py state` rc 10" in cfg["decisions"]["variant_reason"]
    assert cfg["decisions"]["clean_switch"]["d2_clean_repack_state"]["caches_sealed"] == 0
    for f in files:
        sealed_cache(layout.cache_root, f, 16, 2, 8)
    # sealed by their manifests and `state` exits 0, but the recache runner never logged a passed byte re-verify
    cfg = short_config(layout, tmp_path, "noverify.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "byte re-verify did not return 0" in cfg["decisions"]["variant_reason"]
    assert cfg["decisions"]["clean_switch"]["d2_clean_repack_state"]["rc"] == 0
    runner_sealed(layout, files[:1], verify_rc=0)
    runner_sealed(layout, files[1:], verify_rc=1)
    cfg = short_config(layout, tmp_path, "verify1.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "verify rc 1" in cfg["decisions"]["variant_reason"]
    runner_sealed(layout, files[1:], verify_rc=0)  # the LAST seal line of a file counts (a re-seal after a failure)
    cfg = short_config(layout, tmp_path, "clean.json")
    sw = cfg["decisions"]["clean_switch"]
    assert sw["taken"] is True and sw["d2_clean_repack_state"]["rc"] == 0 and sw["d2_clean_repack_state"]["tool_version"] == "1.0.1"
    assert sw["d2_clean_repack_state"]["tool_sha256"] == sha(HERE / "d2_clean_repack.py")
    assert cfg["decisions"]["heldout_label"] == "excluded from D2 (re-packed)"
    cms = cfg["decisions"]["clean_cache_manifests"]
    assert sorted(cms) == [f.name for f in files] and all(c["manifest_sha256"] == sha(layout.cache_root / sha(f) / "manifest.json")
                                                          for f, c in zip(files, (cms[f.name] for f in files)))
    reg = [s for s in cfg["sources"] if s["name"].startswith("registry-")]
    # sealed file 0 lost every row to the clean exclusion: it is not a source (the trainer refuses an empty one)
    assert cfg["decisions"]["variant"] == "clean" and [s["name"] for s in reg] == ["registry-text-00001", "registry-text-clean-00000",
                                                                                   "registry-text-clean-00001"]
    assert cfg["registry_sources_dropped"] == ["registry-text-00000"]
    assert [Path(s["qtext_cache"]).name for s in reg[1:]] == [sha(f) for f in files]
    assert all(s["exclude_rows"] == str(layout.clean_dir / "clean_exclude_rows.json") and s["all_mask"] == str(layout.clean_dir / "clean_all_mask.json")
               for s in reg)
    assert cfg["registry_positions_after_exclusion"] == 6 * 8 and sum(s["tokens"] for s in reg) == 48
    # refill: KD of the v1 draw (38) minus KD of the clean draw (20); the span form's target (10) is printed beside it
    assert cfg["refill"]["target_tokens"] == 18 and cfg["decisions"]["clean_manifest_sha256"] == "c1" * 32
    assert cfg["refill"]["span_form_target_tokens"] == 10 and "the span form would refill 10" in cfg["refill"]["target_basis"]
    # NEGATIVE: a version that is not verified, or built for another decisions file, is never used (fallback, recorded)
    write_clean(layout, verified=False)
    cfg = short_config(layout, tmp_path, "unverified.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "verify_receipt.json is missing, not OK" in cfg["decisions"]["variant_reason"]
    write_clean(layout, decisions="0" * 64)
    cfg = short_config(layout, tmp_path, "otherdec.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "built for decisions 0000000000000000" in cfg["decisions"]["variant_reason"]


def clean_world(world, tmp_path) -> tuple[P.Layout, list[Path]]:
    layout, bf = world
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    files = write_clean(layout)
    runner_sealed(layout, files)
    return layout, files


def test_the_clean_switch_is_the_state_exit_code_not_d2_plans_own_checks(world, tmp_path, monkeypatch):
    """1.4.0 (D2_CLEAN_REPACK_PLAN section 5): everything d2_plan checks itself passes, but `d2_clean_repack.py state`
    does not exit 0 -- a missing tool, a tool below 1.0.1, a tool whose state exits 10 -- so the span form runs."""

    layout, files = clean_world(world, tmp_path)
    assert short_config(layout, tmp_path, "ok.json")["decisions"]["variant"] == "clean"  # the control: the real tool says 0
    real = (HERE / "d2_clean_repack.py").read_text(encoding="utf-8")
    tools = tmp_path / "tools-copy"
    tools.mkdir()
    for name in ("d2_mask_build.py", "distill_corpus_build.py"):
        shutil.copy(HERE / name, tools / name)
    monkeypatch.setattr(P, "TOOLS_DIR", tools)
    cfg = short_config(layout, tmp_path, "notool.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "no " in cfg["decisions"]["clean_switch"]["d2_clean_repack_state"]["why"]
    exit_line = 'return 0 if body["ready"] else EXIT_WAITING'
    assert real.count(exit_line) == 1
    (tools / "d2_clean_repack.py").write_text(real.replace(exit_line, "return EXIT_WAITING"), encoding="utf-8")
    cfg = short_config(layout, tmp_path, "state10.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "clean registry not taken: `d2_clean_repack.py state` rc 10" in cfg["decisions"]["variant_reason"]
    # a state that exits 0 for another manifest than the one d2_plan reads (a tool reading another place)
    other_line = 'body["manifest_sha256"] = man.get("manifest_sha256")'
    assert real.count(other_line) == 1
    (tools / "d2_clean_repack.py").write_text(real.replace(other_line, 'body["manifest_sha256"] = "ff" * 32'), encoding="utf-8")
    cfg = short_config(layout, tmp_path, "otherman.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "`state` judged another manifest" in cfg["decisions"]["variant_reason"]
    old = real.replace('TOOL_VERSION = "1.0.1"', 'TOOL_VERSION = "1.0.0"')
    assert old != real
    (tools / "d2_clean_repack.py").write_text(old, encoding="utf-8")
    cfg = short_config(layout, tmp_path, "old.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "is below 1.0.1" in cfg["decisions"]["variant_reason"]
    (tools / "d2_clean_repack.py").write_text("import sys\nTOOL_VERSION = \"1.0.1\"\nsys.exit(3)\n", encoding="utf-8")
    cfg = short_config(layout, tmp_path, "crash.json")
    assert cfg["decisions"]["variant"] == "span-mask" and "printed no state line" in cfg["decisions"]["variant_reason"]
    assert cfg["decisions"]["clean_switch"]["d2_clean_repack_state"]["rc"] == 3


def test_clean_form_applied_receipt_names_mode_manifest_masks_and_caches(world, tmp_path, capsys):
    layout, files = clean_world(world, tmp_path)
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex)]
    data = tmp_path / "data.json"
    assert P.main(base + ["data-config", "--out", str(data)]) == 0
    assert P.main(base + ["decisions-apply", "--data", str(data)]) == 0
    rec = json.loads(layout.applied.read_text())
    mode = rec["registry_mode"]
    assert rec["variant"] == "clean" and mode["d2_clean_repack_state_rc"] == 0 and mode["clean_manifest_sha256"] == "c1" * 32
    assert mode["codex_heldout_label"] == "excluded from D2 (re-packed)"
    assert {k: v["sha256"] for k, v in mode["registry_masks"].items()} == {k: sha(layout.clean_dir / n) for k, n in P.CLEAN_MASKS.items()}
    assert mode["clean_cache_manifest_sha256"] == {f.name: sha(layout.cache_root / sha(f) / "manifest.json") for f in files}
    assert P.main(base + ["decisions-state", "--data", str(data)]) == 0
    # a re-pack cache manifest rewritten after the receipt: the train-time check (d2_launch decisions_ok) refuses
    m = layout.cache_root / sha(files[1]) / "manifest.json"
    body = json.loads(m.read_text())
    m.write_text(json.dumps({**body, "note": "rewritten"}))
    capsys.readouterr()
    assert P.main(base + ["decisions-state", "--data", str(data)]) == 2 and "re-pack cache of text-clean-00001" in capsys.readouterr().out
    m.write_text(json.dumps(body))
    # NEGATIVE: a clean config whose recorded switch is not exit 0, or that lost a re-pack source, is not applied
    cfg = json.loads(data.read_text())
    bad = json.loads(json.dumps(cfg))
    bad["decisions"]["clean_switch"]["d2_clean_repack_state"]["rc"] = 10
    (tmp_path / "bad1.json").write_text(json.dumps(bad))
    assert P.main(base + ["decisions-apply", "--data", str(tmp_path / "bad1.json"), "--replace"]) == 2
    assert "without `d2_clean_repack.py state` exit 0" in capsys.readouterr().err
    bad = json.loads(json.dumps(cfg))
    bad["sources"] = [s for s in bad["sources"] if s["name"] != "registry-text-clean-00001"]
    (tmp_path / "bad2.json").write_text(json.dumps(bad))
    assert P.main(base + ["decisions-apply", "--data", str(tmp_path / "bad2.json"), "--replace"]) == 2
    assert "text-clean-00001.safetensors is not a registry source" in capsys.readouterr().err
    bad = json.loads(json.dumps(cfg))
    bad["decisions"]["heldout_label"] = P.HELDOUT_LABEL["span-mask"]
    (tmp_path / "bad3.json").write_text(json.dumps(bad))
    assert P.main(base + ["decisions-apply", "--data", str(tmp_path / "bad3.json"), "--replace"]) == 2
    assert "Codex val/test label" in capsys.readouterr().err


def test_decisions_gate_waits_refuses_and_the_applied_receipt_binds_the_config(world, tmp_path, monkeypatch, capsys):
    layout, bf = world
    sealed_cache(layout.cache_root, bf, 5, 1, 5)
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex)]
    # no tool-call screen receipt: the short config waits
    layout.toolcall_screen.unlink()
    assert P.main(base + ["data-config", "--out", str(tmp_path / "x.json")]) == P.EXIT_WAITING
    assert "tool-call screen" in capsys.readouterr().out and not (tmp_path / "x.json").exists()
    assert P.main(base + ["decisions-state"]) == P.EXIT_WAITING
    write_decisions(layout, monkeypatch, layout.test_tc)
    # v2 masks whose check-decisions failed: refused, and so is a config
    r = json.loads((layout.masks_v2 / "build_receipt.json").read_text())
    r["doc_index_check"] = {"ok": False, "problems": ["gt_mask differs"]}
    (layout.masks_v2 / "build_receipt.json").write_text(json.dumps(r))
    capsys.readouterr()
    assert P.main(base + ["decisions-state"]) == 2
    assert P.main(base + ["data-config", "--out", str(tmp_path / "x.json")]) == 2 and "check-decisions FAILED" in capsys.readouterr().err
    write_decisions(layout, monkeypatch, layout.test_tc)
    # a decisions file other than the pinned one: refused
    layout.decisions.write_text(layout.decisions.read_text() + " ")
    assert P.main(base + ["decisions-state"]) == 2
    write_decisions(layout, monkeypatch, layout.test_tc)
    assert P.main(base + ["decisions-state"]) == 0
    data = tmp_path / "data.json"
    assert P.main(base + ["data-config", "--out", str(data)]) == 0
    assert P.main(base + ["decisions-state", "--data", str(data)]) == P.EXIT_WAITING  # no receipt yet
    assert P.main(base + ["decisions-apply", "--data", str(data)]) == 0
    rec = json.loads(layout.applied.read_text())
    assert rec["data_config"]["sha256"] == sha(data) and rec["decisions"]["sha256"] == P.DECISIONS_SHA and rec["variant"] == "span-mask"
    assert rec["registry_mode"]["d2_clean_repack_state_rc"] == P.EXIT_WAITING and "clean_manifest_sha256" not in rec["registry_mode"]
    assert rec["registry_mode"]["codex_heldout_label"] == "loss-excluded in D2; context-exposed in shared packed rows"
    assert rec["refill"]["target_tokens"] == 10 and rec["apply_before_first_step"]["5_a2_revised"]["prereg"] == str(layout.prereg)
    assert P.main(base + ["decisions-state", "--data", str(data)]) == 0
    # the config changed after the receipt: mismatch (2); a receipt for another config is never overwritten silently
    other = tmp_path / "data_other.json"
    cfg = json.loads(data.read_text())
    cfg["note"] = "rewritten"
    other.write_text(json.dumps(cfg))
    assert P.main(base + ["decisions-state", "--data", str(other)]) == 2
    capsys.readouterr()
    assert P.main(base + ["decisions-apply", "--data", str(other)]) == 2 and "give --replace" in capsys.readouterr().err
    assert P.main(base + ["decisions-apply", "--data", str(other), "--replace"]) == 0
    assert list(layout.applied.parent.glob("D2_DECISIONS_APPLIED.json.superseded-*"))
    # a config that does not carry the decisions (a 1.2.0-style config) is refused
    old = tmp_path / "old.json"
    cfg.pop("decisions")
    old.write_text(json.dumps(cfg))
    assert P.main(base + ["decisions-apply", "--data", str(old), "--replace"]) == 2
    # a registry mask edited after the receipt breaks it
    (layout.masks_v2 / "gt_mask.json").write_text("{}")
    assert P.main(base + ["decisions-state", "--data", str(other)]) == 2


# ---------------------------------------------------------------------------------------------- memory-fit (1.2.0)
#: the OMNI1 text geometry at 4 layers (3 GDN + 1 gated-GQA): the memory model is arithmetic (layers built on `meta`),
#: so the test runs at the real GiB scale, where a row group of one 4,841-token row and the micro-batch's I/O compete
GEOM = {"model_type": "qwen3_5_text", "hidden_size": 5120, "intermediate_size": 16640, "num_hidden_layers": 4,
        "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256, "linear_num_value_heads": 48,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4, "vocab_size": 248320, "rms_norm_eps": 1e-6, "max_position_embeddings": 262144,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}
P2_ARGV = ["/x/streamed_kd_trainer.py", "train", "--engine", "streamed", "--model", "/nonexistent-p2-source", "--devices", "cuda:0",
           "--dtype", "bfloat16", "--data-config", "/x/data.json", "--seq-len", "4096", "--micro-batch", "8", "--accumulation", "1",
           "--total-steps", "4", "--lr", "0", "--gain-lr", "0", "--recompute-tokens", "4096", "--out", "/x/run", "--safety-dir", "/x/safety"]
D2_RANKS = {"lora_rank": 64, "k3_lora_rank": 32, "trunk_ffn_width": 4352}


def tiny_loader(ns):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig.from_dict(dict(GEOM))


@pytest.fixture()
def p2(world, tmp_path):
    """A P2 summary whose mb8 cell completed only on its OOM retry (ffn 2048 / head 128), a long cell, and the receipts."""

    layout, _ = world
    T = P.load_trainer(HERE)
    out = layout.p2_summaries
    out.mkdir(parents=True)
    argv = P2_ARGV + ["--ffn-chunk-tokens", "2048", "--head-chunk-mib", "128"]
    ns = T.parser().parse_args(argv[1:])
    trainable = P.trainable_parameters(T, tiny_loader(ns), ns)
    kernels = {"chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk_gated_delta_rule"}
    pred = P.predicted_peak(T, ns, tiny_loader(ns), kernels, trainable, micro=8, length=4096)
    receipt = out / "p2-trainer-v133-omni1-L4096-mb8-ffn2048-gpu0.json"
    receipt.write_text(json.dumps({"argv": argv, "kernels": kernels, "adapter_census": {"parameters_trainable": trainable},
                                   "memory_model": {"peak_bytes": pred}, "memory_engine": {"peak_allocated_gib": pred / P.GIB}}))
    long_argv = [a for a in P2_ARGV] + ["--ffn-chunk-tokens", "1024", "--head-chunk-mib", "64"]
    long_receipt = out / "p2-trainer-v133-omni1-L32768-mb1-ffn2048-gpu0.json"
    long_receipt.write_text(json.dumps({"argv": long_argv}))
    cells = {"L4096-mb8": {"state": "completed", "used_attempt": "L4096-mb8-ffn2048", "steady_tok_per_s_mean_steps_2_on": 160.0,
                           "receipt": str(receipt), "measured_peak_gib": pred / P.GIB},
             "L4096-mb4": {"state": "failed"},
             "L32768-mb1": {"state": "completed", "used_attempt": "L32768-mb1-ffn2048", "receipt": str(long_receipt),
                            "steady_tok_per_s_mean_steps_2_on": 90.0}}
    (out / "p2-summary-v133-gpu0.json").write_text(json.dumps({"gpu_index": 0, "cells": cells}))
    return layout, T, pred, receipt


def test_memory_fit_carries_the_retry_knobs_and_halves_the_micro_batch_only_when_the_width_needs_it(p2):
    layout, T, pred, receipt = p2
    fit_args = dict(tools=HERE, d2_ranks=D2_RANKS, config_loader=tiny_loader)
    # at P2's own width the verdict's micro-batch fits by proof, however small the card
    code, fit = P.memory_fit(layout, width=4096, card_mib=1, **fit_args)
    assert code == 0 and fit["modeled"] and (fit["micro_batch"], fit["accumulation"]) == (8, 1)
    assert fit["knobs"]["ffn_chunk_tokens"] == 2048 and fit["knobs"]["head_chunk_mib"] == 128.0  # NOT the trainer defaults
    assert fit["long_knobs"]["ffn_chunk_tokens"] == 1024 and fit["long_knobs"]["head_chunk_mib"] == 64.0
    assert fit["calibration"]["used_attempt"] == "L4096-mb8-ffn2048" and fit["ratio_measured_to_predicted"] == 1.0
    assert fit["d2_trainable"] == fit["p2_trainable"] and fit["geometry_equal_to_p2"]
    # wider rows: micro 8 needs more than P2 measured; a card with room for it keeps 8, a tight card drops to 4 x 2
    code, fit = P.memory_fit(layout, width=4841, card_mib=1, **fit_args)
    est = {r["micro_batch"]: r["estimate_gib"] for r in fit["rows"]}
    assert est[8] > pred / P.GIB
    assert code == 0 and (fit["micro_batch"], fit["accumulation"]) == (4, 2) and fit["rows"][-1]["fits"] and not fit["rows"][0]["fits"]
    roomy = int((est[8] + P.MEM_HEADROOM_GIB) * 1024) + 1
    code, fit = P.memory_fit(layout, width=4841, card_mib=roomy, **fit_args)
    assert code == 0 and (fit["micro_batch"], fit["accumulation"]) == (8, 1) and len(fit["rows"]) == 1
    # NEGATIVE: nothing fits -> refused (exit 2), never a silent OOM loop
    code, fit = P.memory_fit(layout, width=65536, card_mib=1, **fit_args)
    assert code == 2 and [r["micro_batch"] for r in fit["rows"]] == [8, 4, 2, 1] and "no micro-batch" in fit["reason"]


def test_memory_fit_does_not_guess_when_it_cannot_reproduce_p2(p2):
    layout, T, pred, receipt = p2

    def gone(ns):
        raise FileNotFoundError("the P2 source was moved")

    code, fit = P.memory_fit(layout, width=4841, card_mib=1, tools=HERE, d2_ranks=D2_RANKS, config_loader=gone)
    assert code == 0 and not fit["modeled"] and "cannot be opened" in fit["reason"] and fit["knobs"]["ffn_chunk_tokens"] == 2048
    body = json.loads(receipt.read_text())
    body["adapter_census"]["parameters_trainable"] += 1
    receipt.write_text(json.dumps(body))
    code, fit = P.memory_fit(layout, width=4841, card_mib=1, tools=HERE, d2_ranks=D2_RANKS, config_loader=tiny_loader)
    assert code == 0 and not fit["modeled"] and fit["micro_batch"] == 8 and "not reproducing P2" in fit["reason"]
    assert fit["knobs"]["ffn_chunk_tokens"] == 2048  # the knobs are still carried
    # a split verdict is not modelled either; the long cell's knobs still come through
    layout.p2_verdict.write_text("engine=split\nmicro_batch=4\naccumulation=4\n")
    code, fit = P.memory_fit(layout, width=4841, card_mib=1, tools=HERE, config_loader=tiny_loader)
    assert code == 0 and not fit["modeled"] and fit["engine"] == "split" and fit["long_knobs"]["ffn_chunk_tokens"] == 1024


def test_memory_fit_cli_waits_without_a_verdict_and_writes_only_a_fit(world, p2, tmp_path, monkeypatch, capsys):
    layout, T, pred, receipt = p2
    data = tmp_path / "data.json"
    data.write_text(json.dumps({"max_width": 4841, "sources": []}))
    monkeypatch.setattr(P, "default_config_loader", lambda T_, ns: tiny_loader(ns))
    base = ["--oc", str(layout.oc), "--lex", str(layout.lex), "memory-fit", "--data", str(data)]
    out = tmp_path / "fit.json"
    assert P.main(base + ["--card-mib", "1", "--out", str(out)]) == 0
    fit = json.loads(out.read_text())
    assert (fit["micro_batch"], fit["accumulation"], fit["width"]) == (4, 2, 4841) and fit["data_config"]["sha256"] == sha(data)
    out.unlink()
    data.write_text(json.dumps({"max_width": 65536, "sources": []}))
    capsys.readouterr()
    assert P.main(base + ["--card-mib", "1", "--out", str(out)]) == 2 and not out.exists()
    assert "REFUSED: memory-fit" in capsys.readouterr().err
    shutil.rmtree(layout.p2_summaries)
    assert P.main(base + ["--card-mib", "1"]) == P.EXIT_WAITING
