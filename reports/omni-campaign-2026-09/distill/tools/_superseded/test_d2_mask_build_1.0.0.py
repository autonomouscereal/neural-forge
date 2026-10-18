"""Tests for d2_mask_build.py (card D2-M1).  CPU only, no torch needed.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_mask_build.py

A tiny packed stream (3 files x 2 rows x SEQ positions, SEQ patched to 16) with hand-placed documents is labelled
the way Codex's normalized.jsonl labels the real one; the build must produce exactly the hand-derived exclusions and
spans, and check-doc-index must agree with the files' own doc_index tensors -- and must FAIL on a moved span
(negative control).
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
       3 [30,50) rewrite (crosses rows 1|2|3)  4 [50,60) flattened tool  5 [60,70) BFCL prose AND flattened
       6 [70,96) plain (to the end)"""

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
    return paths


def run_build(paths: dict[str, Path], out: Path) -> int:
    return M.main(["build", "--documents", str(paths["documents"]), "--corpus-manifest", str(paths["manifest"]),
                   "--normalized", str(paths["normalized"]), "--eval-tasks", str(paths["tasks"]), "--out", str(out), "--no-expect"])


def global_spans(mask: dict) -> list[tuple[int, int, int]]:
    out = []
    for entry in mask["corpora"].values():
        out += [(entry["row_offset"] + r, a, b) for r, a, b in entry["spans"]]
    return sorted(out)


def test_build_writes_the_hand_derived_exclusions_and_spans(tmp_path):
    paths = write_world(tmp_path)
    assert run_build(paths, tmp_path / "out") == 0
    excl = json.loads((tmp_path / "out" / "exclude_rows.json").read_text())
    assert excl["global_rows"] == [0, 1] and excl["bfcl62_rids"] == ["rid-1"]  # doc 1 [10,20) touches rows 0 and 1
    files = {e["file"]: e["rows"] for e in excl["corpora"].values()}
    assert files == {"text-00000.safetensors": [0, 1]}
    allm = json.loads((tmp_path / "out" / "all_mask.json").read_text())
    # docs 2 [20,30) and 5 [60,70): positions [19,29) and [59,69) -> rows 1 / 3,4
    assert global_spans(allm) == [(1, 3, 13), (3, 11, 16), (4, 0, 5)]
    assert allm["documents"] == 2 and allm["positions"] == 20 and allm["terms_off"] == ["q38", "k3", "ce", "eos", "ul"]
    gt = json.loads((tmp_path / "out" / "gt_mask.json").read_text())
    # docs 3 [30,50), 4 [50,60), 5 [60,70) are contiguous: positions [29, 69) merged
    assert global_spans(gt) == [(1, 13, 16), (2, 0, 16), (3, 0, 16), (4, 0, 5)]
    assert gt["documents"] == 3 and gt["positions"] == 40 and gt["terms_off"] == ["ce", "eos", "ul"]
    # local rows per file: row 3 is file 1's local row 1
    entry = next(e for e in gt["corpora"].values() if e["file"] == "text-00001.safetensors")
    assert entry["row_offset"] == 2 and entry["spans"] == [[0, 0, 16], [1, 0, 16]]


def test_check_doc_index_agrees_and_the_negative_control_fails(tmp_path, capsys):
    paths = write_world(tmp_path)
    out = tmp_path / "out"
    assert run_build(paths, out) == 0
    assert M.main(["check-doc-index", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]) == 0
    receipt = json.loads((out / "build_receipt.json").read_text())
    assert receipt["doc_index_check"]["ok"] is True
    assert receipt["doc_index_check"]["gt_mask"]["equal_to_doc_index_derivation"] is True
    # negative control: one gt span moved by one column -> refused
    gt = json.loads((out / "gt_mask.json").read_text())
    first = next(iter(gt["corpora"].values()))
    first["spans"][0][1] += 1
    (out / "gt_mask.json").write_text(json.dumps(gt))
    capsys.readouterr()
    assert M.main(["check-doc-index", "--out", str(out), "--corpus-manifest", str(paths["manifest"])]) == 1
    assert "gt_mask: doc_index derivation differs" in capsys.readouterr().out


def test_build_refuses_a_partial_join_and_pinned_inputs(tmp_path, capsys):
    paths = write_world(tmp_path)
    lines = paths["normalized"].read_text().splitlines()
    paths["normalized"].write_text("\n".join(lines[:-1]) + "\n")  # one document loses its label
    assert run_build(paths, tmp_path / "out") == 2
    assert "join must be total" in capsys.readouterr().err
    paths = write_world(tmp_path)
    args = ["build", "--documents", str(paths["documents"]), "--corpus-manifest", str(paths["manifest"]),
            "--normalized", str(paths["normalized"]), "--eval-tasks", str(paths["tasks"]), "--out", str(tmp_path / "o2")]
    assert M.main(args) == 2  # not the pinned X04 inputs
    assert "is not the pinned" in capsys.readouterr().err


def test_self_test():
    assert M.self_test() == 0
