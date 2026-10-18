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
