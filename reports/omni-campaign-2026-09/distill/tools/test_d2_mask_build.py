"""Tests for d2_mask_build.py (card D2-M1; 1.1.0 adds the BFCL-62 task-twin exclusions; 1.2.0 the D2_DECISIONS v2
masks).  CPU only, no torch needed.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_mask_build.py

A tiny packed stream (3 files x 2 rows x SEQ positions, SEQ patched to 16) with hand-placed documents is labelled
the way Codex's normalized.jsonl labels the real one; the build must produce exactly the hand-derived exclusions and
spans, and check-doc-index must agree with the files' own doc_index tensors -- and must FAIL on a moved span
(negative control).  A hand-written bfcl62_rids.json names one text twin, one K3 twin and one self-gen twin; a holdout
that disagrees with the group join, or whose twin rows differ, is refused.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d2_mask_build as M  # noqa: E402

SEQ = 16


@pytest.fixture(autouse=True)
def small_rows(monkeypatch):
    monkeypatch.setattr(M, "SEQ", SEQ)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_world(tmp: Path) -> dict[str, Path]:
    """Six rows of 16 positions.  Documents (global token spans):
       0 [0,10)  plain            1 [10,20) BFCL test source (crosses rows 0|1)   2 [20,30) BFCL prose
       3 [30,50) rewrite (crosses rows 1|2|3)  4 [50,60) flattened tool  5 [60,70) BFCL prose AND flattened, and a
       TEXT TWIN of the BFCL-62 task (rows 3|4)   6 [70,96) plain (to the end)
    K3: 4 passes x 3 rows; the K3 twin rid sits in pass 1 row 2.  Self-gen: 4 rescored rows, rows 1 and 3 from the twin
    prompt."""

    from safetensors.numpy import save_file

    docs_spec = [(0, 10, "plain", ()), (10, 20, M.BFCL_DATASET, ()), (20, 30, M.BFCL_DATASET, ()),
                 (30, 50, "x", (M.GT_BLOCKERS[0],)), (50, 60, "y", (M.GT_BLOCKERS[1],)),
                 (60, 70, M.BFCL_DATASET, (M.GT_BLOCKERS[1],)), (70, 96, "plain", ())]
    docs, normalized = [], []
    doc_index = np.full(96, -1, dtype=np.int32)
    for i, (s, e, ds, blockers) in enumerate(docs_spec):
        rid = f"rid-{i}"
        docs.append({"doc": i, "rid": rid, "uid": f"uid-{i}", "dataset_id": ds, "start": s, "end": e, "kept": e - s, "length": e - s})
        normalized.append({"id": f"uid-{i}", "split": "test" if i == 1 else "train",
                           "provenance": {"registry_origin": {"rid": rid}, "recovery_group_id": f"group-{i}"},
                           "quality": {"blockers": list(blockers)}})
        doc_index[s:e] = i
    paths = {"documents": tmp / "documents.jsonl", "normalized": tmp / "normalized.jsonl", "tasks": tmp / "tasks.jsonl"}
    paths["documents"].write_text("".join(json.dumps(d) + "\n" for d in docs))
    paths["normalized"].write_text("".join(json.dumps(n) + "\n" for n in normalized))
    tasks = [{"id": "bfcl-heldout:abc", "group_id": "group-1"}, {"id": "tool-seed-1", "group_id": "nowhere"}]
    paths["tasks"].write_text("".join(json.dumps(t) + "\n" for t in tasks))
    shards = []
    for f in range(3):
        rows = slice(2 * f * SEQ, 2 * (f + 1) * SEQ)
        path = tmp / f"text-{f:05d}.safetensors"
        save_file({"doc_index": doc_index[rows].reshape(2, SEQ).copy()}, str(path))
        shards.append({"path": str(path), "sha256": sha(path), "rows": [2 * f, 2 * f + 2]})
    paths["manifest"] = tmp / "manifest.json"
    paths["manifest"].write_text(json.dumps({"shards": shards, "documents_file": {"sha256": sha(paths["documents"])}}))
    # K3 passes
    k3 = tmp / "k3"
    for k in range(4):
        (k3 / f"pass-{k}").mkdir(parents=True, exist_ok=True)
        rids = [[f"k3-{k}-{r}-a", f"k3-{k}-{r}-b"] for r in range(3)]
        if k == 1:
            rids[2].append("k3-twin")
        (k3 / f"pass-{k}" / "row_meta.json").write_text(json.dumps({"rows": [{"row_index": r, "rids": x} for r, x in enumerate(rids)]}))
        save_file({"input_ids": np.full((3, 8), k, dtype=np.int32)}, str(k3 / f"pass-{k}" / "qwen_input_ids.safetensors"))
    paths["k3"] = k3
    # self-gen rescore export rows + the target corpus built from it
    export_sha = "e" * 64
    paths["sg_rows"] = tmp / "rescore_export.rows.json"
    paths["sg_rows"].write_text(json.dumps({"corpus": {"sha256": export_sha}, "row_ids": ["p0", "sg-twin", "p2", "sg-twin"]}))
    paths["sg_corpus"] = tmp / "rescore_target.safetensors"
    save_file({"input_ids": np.zeros((4, 8), dtype=np.int32)}, str(paths["sg_corpus"]), metadata={"source_export_sha256": export_sha})
    paths["holdout"] = tmp / "bfcl62_rids.json"
    write_holdout(paths["holdout"])
    return paths


def write_holdout(path: Path, **override) -> None:
    body = {"schema": M.HOLDOUT_SCHEMA, "tool": "bfcl62_holdout", "tool_version": "test", "rids": ["rid-1"], "global_rows": [0, 1],
            "twin_exposure": {"twin_ids_by_source": {"text": ["rid-5"], "k3": ["k3-twin"], "selfgen": ["sg-twin"], "long": [],
                                                     "native_train": ["uid-native"]},
                              "extra_text_rows_if_text_twins_excluded": [3, 4]}}
    for key, value in override.items():
        if key in body:
            body[key] = value
        else:
            body["twin_exposure"][key] = value
    path.write_text(json.dumps(body))


def run_build(paths: dict[str, Path], out: Path, k3: bool = True, selfgen: bool = True) -> int:
    argv = ["build", "--documents", str(paths["documents"]), "--corpus-manifest", str(paths["manifest"]),
            "--normalized", str(paths["normalized"]), "--eval-tasks", str(paths["tasks"]), "--holdout", str(paths["holdout"]),
            "--out", str(out), "--no-expect"]
    if k3:
        argv += ["--k3-dir", str(paths["k3"])]
    if selfgen:
        argv += ["--selfgen-rows", str(paths["sg_rows"]), "--selfgen-corpus", str(paths["sg_corpus"])]
    return M.main(argv)


def global_spans(mask: dict) -> list[tuple[int, int, int]]:
    out = []
    for entry in mask["corpora"].values():
        out += [(entry["row_offset"] + r, a, b) for r, a, b in entry["spans"]]
    return sorted(out)


def test_build_writes_the_hand_derived_exclusions_and_spans(tmp_path):
    paths = write_world(tmp_path)
    assert run_build(paths, tmp_path / "out") == 0
    excl = json.loads((tmp_path / "out" / "exclude_rows.json").read_text())
    # doc 1 [10,20) touches rows 0 and 1; its text twin doc 5 [60,70) rows 3 and 4
    assert excl["bfcl62_rows"] == [0, 1] and excl["bfcl62_rids"] == ["rid-1"]
    assert excl["text_twin_rids"] == ["rid-5"] and excl["text_twin_rows"] == [3, 4]
    assert excl["global_rows"] == [0, 1, 3, 4]
    files = {e["file"]: e["rows"] for e in excl["corpora"].values() if e["source"] == "registry"}
    assert files == {"text-00000.safetensors": [0, 1], "text-00001.safetensors": [1], "text-00002.safetensors": [0]}
    # the K3 twin row (pass 1, row 2) and the self-gen rows 1, 3, keyed by their corpus sha
    k3_sha = sha(paths["k3"] / "pass-1" / "qwen_input_ids.safetensors")
    assert excl["corpora"][k3_sha]["rows"] == [2] and excl["corpora"][k3_sha]["source"] == "k3"
    assert excl["corpora"][sha(paths["sg_corpus"])]["rows"] == [1, 3]
    assert excl["other_sources"]["native_train"]["twin_uids"] == ["uid-native"]
    allm = json.loads((tmp_path / "out" / "all_mask.json").read_text())
    # doc 2 [20,30) only: doc 5 is a twin (row-excluded, not masked): positions [19,29) -> row 1
    assert global_spans(allm) == [(1, 3, 13)]
    assert allm["documents"] == 1 and allm["positions"] == 10 and allm["terms_off"] == ["q38", "k3", "ce", "eos", "ul"]
    gt = json.loads((tmp_path / "out" / "gt_mask.json").read_text())
    # docs 3 [30,50), 4 [50,60), 5 [60,70) are contiguous: positions [29, 69) merged
    assert global_spans(gt) == [(1, 13, 16), (2, 0, 16), (3, 0, 16), (4, 0, 5)]
    assert gt["documents"] == 3 and gt["positions"] == 40 and gt["terms_off"] == ["ce", "eos", "ul"]
    entry = next(e for e in gt["corpora"].values() if e["file"] == "text-00001.safetensors")
    assert entry["row_offset"] == 2 and entry["spans"] == [[0, 0, 16], [1, 0, 16]]
    receipt = json.loads((tmp_path / "out" / "build_receipt.json").read_text())
    c = receipt["counts"]
    assert (c["excluded_rows"], c["text_twin_rows"], c["k3_twin_rows"], c["selfgen_twin_rows"], c["all_mask_docs"]) == (4, 2, 1, 2, 1)


def test_a_selfgen_twin_that_was_never_rescored_excludes_nothing_and_says_so(tmp_path):
    paths = write_world(tmp_path)
    paths["sg_rows"].write_text(json.dumps({"corpus": {"sha256": "e" * 64}, "row_ids": ["p0", "p1", "p2", "p3"]}))
    assert run_build(paths, tmp_path / "out") == 0
    excl = json.loads((tmp_path / "out" / "exclude_rows.json").read_text())
    assert sha(paths["sg_corpus"]) not in excl["corpora"]
    assert excl["other_sources"]["selfgen"]["rows"] == [] and "nothing to exclude" in excl["other_sources"]["selfgen"]["note"]


def test_check_doc_index_agrees_and_the_negative_control_fails(tmp_path, capsys):
    paths = write_world(tmp_path)
    out = tmp_path / "out"
    assert run_build(paths, out) == 0
    assert M.main(["check-doc-index", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]) == 0
    receipt = json.loads((out / "build_receipt.json").read_text())
    assert receipt["doc_index_check"]["ok"] is True
    assert receipt["doc_index_check"]["gt_mask"]["equal_to_doc_index_derivation"] is True
    assert receipt["doc_index_check"]["exclusion"]["text_twin_docs_inside_excluded_rows"] == 1
    # negative control: one gt span moved by one column -> refused
    gt = json.loads((out / "gt_mask.json").read_text())
    first = next(iter(gt["corpora"].values()))
    first["spans"][0][1] += 1
    (out / "gt_mask.json").write_text(json.dumps(gt))
    capsys.readouterr()
    assert M.main(["check-doc-index", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]) == 1
    assert "gt_mask: doc_index derivation differs" in capsys.readouterr().out


def test_check_doc_index_fails_when_a_twin_row_is_dropped_from_the_exclusion(tmp_path, capsys):
    paths = write_world(tmp_path)
    out = tmp_path / "out"
    assert run_build(paths, out) == 0
    excl = json.loads((out / "exclude_rows.json").read_text())
    excl["global_rows"] = [0, 1]  # the 1.0.0 exclusion: the twin's rows 3 and 4 stay in D2

    (out / "exclude_rows.json").write_text(json.dumps(excl))
    capsys.readouterr()
    assert M.main(["check-doc-index", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]) == 1
    assert "BFCL-62 / text twin documents are not inside the excluded rows" in capsys.readouterr().out


def test_holdout_that_disagrees_or_needs_a_missing_input_is_refused(tmp_path, capsys):
    paths = write_world(tmp_path)
    write_holdout(paths["holdout"], rids=["rid-2"], global_rows=[1])
    assert run_build(paths, tmp_path / "o1") == 2
    assert "differ from this tool's group join" in capsys.readouterr().err
    write_holdout(paths["holdout"], extra_text_rows_if_text_twins_excluded=[3])
    assert run_build(paths, tmp_path / "o2") == 2
    assert "text twin rows [3, 4]" in capsys.readouterr().err
    write_holdout(paths["holdout"])
    assert run_build(paths, tmp_path / "o3", k3=False) == 2
    assert "give --k3-dir" in capsys.readouterr().err
    write_holdout(paths["holdout"], twin_ids_by_source={"text": ["rid-5"], "k3": ["not-a-k3-rid"], "selfgen": [], "long": []})
    assert run_build(paths, tmp_path / "o4") == 2
    assert "K3 twin rids not found" in capsys.readouterr().err
    assert not (tmp_path / "o1").exists() and not (tmp_path / "o4" / "exclude_rows.json").exists()


def test_build_refuses_a_partial_join_and_pinned_inputs(tmp_path, capsys):
    paths = write_world(tmp_path)
    lines = paths["normalized"].read_text().splitlines()
    paths["normalized"].write_text("\n".join(lines[:-1]) + "\n")  # one document loses its label
    assert run_build(paths, tmp_path / "out") == 2
    assert "join must be total" in capsys.readouterr().err
    paths = write_world(tmp_path)
    args = ["build", "--documents", str(paths["documents"]), "--corpus-manifest", str(paths["manifest"]),
            "--normalized", str(paths["normalized"]), "--eval-tasks", str(paths["tasks"]), "--holdout", str(paths["holdout"]),
            "--out", str(tmp_path / "o2")]
    assert M.main(args) == 2  # not the pinned X04 inputs
    assert "is not the pinned" in capsys.readouterr().err


def test_self_test():
    assert M.self_test() == 0


# ---------------------------------------------------------------------------------------------- 1.2.0 decisions
def write_decisions_world(tmp: Path, expected_override: dict | None = None) -> dict[str, Path]:
    """write_world + input_ids in the packed files, the v1 build, and a D2_DECISIONS-shaped file over it.
    Held-out (D2-X1): rid-0 (short:test, joined through its staging record_id 'rec-0'; row 0 is v1-excluded) and rid-6
    (short:validation; [70,96): row 4 is v1-excluded, row 5 is trained and ends with position 95, which has no target).
    Two executed-tool ids join nothing.  Truncated (D2-X2): doc 2 (08), doc 4 (14) and doc 6 (06, held-out) end in
    <|im_end|> newline <|endoftext|>; doc 3 is flagged 14 but cut (kept < length) and so is not.
    Hand-derived: X1 positions [0,9) + [69,95) = 9 + 11 excluded + 15 trained; row 5 is left with no loss-carrying
    position (15 held + 1 without target) and joins the exclusion; row 2 (v1 gt-mask, KL kept) stays."""

    from safetensors.numpy import load_file, save_file

    tmp.mkdir(parents=True, exist_ok=True)
    paths = write_world(tmp)
    docs = [json.loads(line) for line in paths["documents"].read_text().splitlines()]
    docs[3]["length"] = 25  # cut at its end: not a D2-X2 document
    paths["documents"].write_text("".join(json.dumps(d) + "\n" for d in docs))
    ids = np.arange(96, dtype=np.int32) + 1000
    for doc in (2, 4, 6):
        e = docs[doc]["end"]
        ids[e - 3:e] = M.EOS_TAIL
    manifest = json.loads(paths["manifest"].read_text())
    for f, s in enumerate(manifest["shards"]):
        t = load_file(s["path"])
        save_file({"doc_index": t["doc_index"], "input_ids": ids[2 * f * SEQ: 2 * (f + 1) * SEQ].reshape(2, SEQ).copy()}, s["path"])
        s["sha256"] = sha(Path(s["path"]))
    manifest["documents_file"] = {"sha256": sha(paths["documents"])}
    paths["manifest"].write_text(json.dumps(manifest))
    assert run_build(paths, tmp / "v1") == 0
    paths["v1"] = tmp / "v1"
    paths["rows"] = tmp / "rows.jsonl"
    paths["rows"].write_text("".join(json.dumps({"rid": f"rid-{i}", "record_id": f"rec-{i}"}) + "\n" for i in range(7)))
    packs = []
    for name, row_ids in (("original_short_validation", ["rid-6"]), ("original_short_test", ["rec-0"]),
                          ("executed_tools_test", ["x-tool-1", "x-tool-2"])):
        m = tmp / "packs" / name / "manifest.json"
        m.parent.mkdir(parents=True)
        m.write_text(json.dumps({"row_receipts": [{"row_id": r} for r in row_ids]}))
        packs.append({"name": name, "manifest": str(m), "manifest_sha256": sha(m), "rows": len(row_ids)})
    paths["scores"] = tmp / "scores.jsonl"
    paths["scores"].write_text("".join(json.dumps({"rid": r, "class": c}) + "\n" for r, c in
                                       (("rid-0", "15"), ("rid-2", "08"), ("rid-3", "14"), ("rid-4", "14"), ("rid-6", "06"))))
    paths["long_tasks"] = tmp / "tasks_long.json"
    paths["long_tasks"].write_text(json.dumps({"tasks": [{"doc": 0, "rows": ["long-a"]}]}))
    paths["bfcl_rows"] = tmp / "bfcl_rows.json"
    paths["bfcl_rows"].write_text(json.dumps({"rows": [{"row_id": "nat-1", "out_row": 0}]}))
    expected = {"source": "hand-derived",
                "heldout_registry_docs": {"short:test": 1, "short:validation": 1},
                "heldout_positions": {"trained_ce_kl": 15, "excluded_row": 20, "all_mask": 0, "total": 35},
                "packed_rows_touched": 3, "packed_rows_touched_not_already_excluded": 1,
                "rows_by_heldout_share": {">=50%": 2, ">=90%": 1},
                "eos_flagged_docs": {"06": 1, "08": 1, "14": 1, "total": 3, "of_which_heldout": 1},
                "eos_positions": {"train_docs_trained": 0, "heldout_docs_trained": 3, "in_excluded_rows": 6, "docs_trained_today": 1}}
    expected.update(expected_override or {})
    decisions = {"schema": M.DECISIONS_SCHEMA, "id": "D2_DECISIONS_TEST", "rulings": {"ids": ["x1", "x2"]},
                 "canonical_split": {"heldout_packs": packs, "cross_checks": {"heldout_ids_total": 4, "joined_to_registry_rids": 2}},
                 "exclusion": {"join_rule": "test", "sets": {"a2_3a_row_exclusion_unchanged": {"sha256": sha(tmp / "v1" / "exclude_rows.json")}}},
                 "eos_mask": {"flag": {"sha256": sha(paths["scores"])}}, "expected_counts": expected}
    paths["decisions"] = tmp / "D2_DECISIONS.json"
    paths["decisions"].write_text(json.dumps(decisions))
    return paths


def run_decisions(paths: dict[str, Path], out: Path, *extra: str) -> int:
    return M.main(["decisions", "--decisions", str(paths["decisions"]), "--v1", str(paths["v1"]), "--rows", str(paths["rows"]),
                   "--documents", str(paths["documents"]), "--corpus-manifest", str(paths["manifest"]), "--scores", str(paths["scores"]),
                   "--k3-dir", str(paths["k3"]), "--long-tasks", str(paths["long_tasks"]), "--selfgen-rows", str(paths["sg_rows"]),
                   "--bfcl-native-rows", str(paths["bfcl_rows"]), "--holdout", str(paths["holdout"]), "--out", str(out), "--no-expect", *extra])


def test_decisions_writes_the_hand_derived_v2_masks_and_exclusion(tmp_path):
    paths = write_decisions_world(tmp_path)
    out = tmp_path / "v2"
    assert run_decisions(paths, out) == 0
    allm = json.loads((out / "all_mask.json").read_text())
    # v1 all [19,29) + X1 doc 0 [0,9) + X1 doc 6 [69,95)
    assert global_spans(allm) == [(0, 0, 9), (1, 3, 13), (4, 5, 16), (5, 0, 15)]
    assert allm["kind"] == "all" and allm["components"]["d2_x1"]["documents"] == 2 and allm["components"]["d2_x1"]["positions"] == 35
    gt = json.loads((out / "gt_mask.json").read_text())
    # v1 gt [29,69) + the end-of-turn targets of docs 2 [26,29), 4 [56,59) (inside v1) and 6 [92,95)
    assert global_spans(gt) == [(1, 10, 16), (2, 0, 16), (3, 0, 16), (4, 0, 5), (5, 12, 15)]
    assert gt["components"]["d2_x2"]["docs"] == [2, 4, 6] and gt["terms_off"] == ["ce", "eos", "ul"]
    excl = json.loads((out / "exclude_rows.json").read_text())
    assert excl["global_rows"] == [0, 1, 3, 4, 5] and excl["no_loss_rows"] == [5] and excl["v1"]["global_rows"] == [0, 1, 3, 4]
    reg = {e["file"]: e["rows"] for e in excl["corpora"].values() if e["source"] == "registry"}
    assert reg == {"text-00000.safetensors": [0, 1], "text-00001.safetensors": [1], "text-00002.safetensors": [0, 1]}
    k3_sha = sha(paths["k3"] / "pass-1" / "qwen_input_ids.safetensors")
    assert excl["corpora"][k3_sha]["rows"] == [2]  # the A2 3(a) K3 twin row is carried unchanged
    held = json.loads((out / "heldout_rids.json").read_text())
    assert held["rids"] == ["rid-0", "rid-6"] and held["unjoined_by_label"] == {"executed_tools:test": 2} and held["registry_docs"] == [0, 6]
    r = json.loads((out / "build_receipt.json").read_text())
    assert r["kind"] == "d2-decisions" and r["count_check"]["ok"] and r["rechecks"]["hits"] == 0
    assert r["refill"]["heldout_trained_positions_removed"] == 15
    # KD-carrying positions: v1 = rows 2 (16) + 5 (15; position 95 has no target); v2 = row 2 only
    assert r["refill"]["kd_positions_v1"] == 31 and r["refill"]["kd_positions_v2"] == 16
    assert r["counts"]["positions_after_v2_exclusion"] == 16 and r["counts"]["loss_positions_after_v2"] == 16


def test_check_decisions_agrees_and_negative_controls_fail(tmp_path, capsys):
    paths = write_decisions_world(tmp_path)
    out = tmp_path / "v2"
    assert run_decisions(paths, out) == 0
    args = ["check-decisions", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]
    assert M.main(args) == 0
    check = json.loads((out / "build_receipt.json").read_text())["doc_index_check"]
    assert check["ok"] and check["gt_mask"]["eos_docs"] == 3 and check["exclusion"]["no_loss_rows"] == 1
    # NEGATIVE: a file edited after the build is refused outright
    gt = json.loads((out / "gt_mask.json").read_text())
    gt["corpora"][next(iter(gt["corpora"]))]["spans"][0][1] += 1
    (out / "gt_mask.json").write_text(json.dumps(gt))
    capsys.readouterr()
    assert M.main(args) == 2 and "changed since the build" in capsys.readouterr().err
    # NEGATIVE: the same edit re-sealed into the receipt is caught by the doc_index derivation
    r = json.loads((out / "build_receipt.json").read_text())
    r["outputs"]["gt_mask.json"] = sha(out / "gt_mask.json")
    (out / "build_receipt.json").write_text(json.dumps(r))
    assert M.main(args) == 1 and "gt_mask: doc_index derivation differs" in capsys.readouterr().out
    # NEGATIVE: the no-loss row dropped from the exclusion
    assert run_decisions(paths, out) == 0
    excl = json.loads((out / "exclude_rows.json").read_text())
    excl["global_rows"], excl["no_loss_rows"] = [0, 1, 3, 4], []
    (out / "exclude_rows.json").write_text(json.dumps(excl))
    r = json.loads((out / "build_receipt.json").read_text())
    r["outputs"]["exclude_rows.json"] = sha(out / "exclude_rows.json")
    (out / "build_receipt.json").write_text(json.dumps(r))
    capsys.readouterr()
    assert M.main(args) == 1 and "no-loss rows by doc_index" in capsys.readouterr().out


def test_decisions_refuses_counts_tails_hits_and_unpinned_packs(tmp_path, capsys):
    from safetensors.numpy import load_file, save_file

    # counts that differ from expected_counts: refused, unless --reason (recorded)
    paths = write_decisions_world(tmp_path / "a", {"packed_rows_touched": 4})
    assert run_decisions(paths, tmp_path / "a" / "v2") == 2
    assert "packed_rows_touched" in capsys.readouterr().err
    assert run_decisions(paths, tmp_path / "a" / "v2", "--reason", "test: recount") == 0
    r = json.loads((tmp_path / "a" / "v2" / "build_receipt.json").read_text())
    assert not r["count_check"]["ok"] and r["count_check"]["reason"] == "test: recount"
    assert r["count_check"]["differences"] == {"packed_rows_touched": {"got": 3, "want": 4}}
    # a truncated document whose tail is not <|im_end|> newline <|endoftext|>
    paths = write_decisions_world(tmp_path / "b")
    manifest = json.loads(paths["manifest"].read_text())
    s = manifest["shards"][1]  # rows 2-3: doc 4 ends at 60 (row 3, column 11 is its last token)
    t = load_file(s["path"])
    t["input_ids"][1, 59 - 48] = 7
    save_file(t, s["path"])
    s["sha256"] = sha(Path(s["path"]))
    paths["manifest"].write_text(json.dumps(manifest))
    capsys.readouterr()
    assert run_decisions(paths, tmp_path / "b" / "v2") == 2
    assert "do not end in" in capsys.readouterr().err
    # a held-out rid inside a K3 row: refused (decisions per_input)
    paths = write_decisions_world(tmp_path / "c")
    meta = paths["k3"] / "pass-0" / "row_meta.json"
    body = json.loads(meta.read_text())
    body["rows"][0]["rids"].append("rid-6")
    meta.write_text(json.dumps(body))
    assert run_decisions(paths, tmp_path / "c" / "v2") == 2
    assert "held-out row(s) found in another D2 input" in capsys.readouterr().err
    assert not (tmp_path / "c" / "v2" / "all_mask.json").exists()
    # a held-out pack manifest that is not the one the decisions file pinned
    paths = write_decisions_world(tmp_path / "d")
    d = json.loads(paths["decisions"].read_text())
    Path(d["canonical_split"]["heldout_packs"][0]["manifest"]).write_text(json.dumps({"row_receipts": [{"row_id": "rid-5"}]}))
    assert run_decisions(paths, tmp_path / "d" / "v2") == 2
    assert "is not the decisions file's" in capsys.readouterr().err
    # --out may never be the v1 directory
    paths = write_decisions_world(tmp_path / "e")
    assert run_decisions(paths, paths["v1"]) == 2
    assert "must not be the d2_masks_v1" in capsys.readouterr().err
