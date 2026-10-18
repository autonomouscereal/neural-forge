"""Tests for d2_mask_build.py (card D2-M1; 1.1.0 adds the BFCL-62 task-twin exclusions).  CPU only, no torch needed.

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
