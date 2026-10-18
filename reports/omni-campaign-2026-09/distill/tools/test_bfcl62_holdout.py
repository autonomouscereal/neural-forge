"""Tests for bfcl62_holdout.py (lane L-DATA, BFCL-62 rid list).  CPU only, stdlib only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_bfcl62_holdout.py

A tiny packed stream (SEQ patched to 16) carries two BFCL held-out source docs, one BFCL train twin of the first
(same request, other category, other recovery group) and plain docs.  The build must find the hand-derived rids and
rows by both derivation paths, flag the twin, and refuse every broken input.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bfcl62_holdout as M  # noqa: E402

SEQ = 16
B = M.BFCL_DATASET


@pytest.fixture(autouse=True)
def small_rows(monkeypatch):
    monkeypatch.setattr(M, "SEQ", SEQ)
    monkeypatch.setattr(M, "EXPECT", {"tasks": 2, "docs": 2, "rows": 3, "excluded_positions": 3 * SEQ, "bfcl_docs": 3})


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_msg(text, category="bfcl_v4_simple_python"):
    return [{"role": "user", "content": f"Produce the exact tool call set.\nCategory: {category}\n\nUser request:\n"
                                        f"Turn 1: user: {text}\n\nAvailable tool schemas:\n[]"},
            {"role": "assistant", "content": "{}"}]


def norm(uid, rid, group, split, text, calls, category="bfcl_v4_simple_python"):
    return {"id": uid, "split": split, "messages": request_msg(text, category), "quality": {"train_ready": True, "blockers": []},
            "provenance": {"recovery_group_id": group, "registry_origin": {"rid": rid, "dataset_id": B, "task_type": category},
                           "bfcl_schema_target_adaptation": {"original_acceptable_targets": {"expected_tool_calls": calls}}}}


GCD = "Find the greatest common divisor (GCD) of 12 and 18 please"
FX = "Convert 150 Euros to Canadian dollars at todays exchange rate"


def write_world(tmp: Path, *, twin=True, native_leak=False, bad_group=False) -> dict[str, Path]:
    """Rows of 16 positions.  Docs (global token spans):
       0 [0,10) plain        1 [10,20) BFCL-62 a (rows 0,1)     2 [20,30) plain (row 1)
       3 [30,40) BFCL twin of a (rows 1,2)                      4 [40,50) BFCL-62 b (rows 2,3)   5 [50,64) plain
    """
    spans = [(0, 10, "000:0:0", "other"), (10, 10, "547:b-000000:1", B), (20, 10, "000:0:2", "other"),
             (30, 10, "547:b-000000:3", B), (40, 10, "547:b-000000:4", B), (50, 14, "000:0:5", "other")]
    docs = [{"doc": i, "start": s, "kept": k, "end": s + k, "length": k, "rid": rid, "uid": f"u{i}", "dataset_id": ds,
             "record_id": f"r{i}", "row_first": s // SEQ, "row_last": (s + k - 1) // SEQ} for i, (s, k, rid, ds) in enumerate(spans)]
    gcd_call = [{"math.gcd": {"num1": [12], "num2": [18]}}]
    normalized = [norm("u0", "000:0:0", "g0", "train", "plain zero text that is long enough here", []),
                  norm("u1", "547:b-000000:1", "g1", "test", GCD, gcd_call),
                  norm("u2", "000:0:2", "g2", "train", "plain two text that is long enough here", []),
                  norm("u3", "547:b-000000:3", "g3", "train", GCD if twin else "an unrelated BFCL question about weather",
                       gcd_call, "bfcl_v4_multiple"),
                  norm("u4", "547:b-000000:4", "g1x" if bad_group else "g4", "test", FX,
                       [{"fx.convert": {"amount": [150], "to": ["CAD"]}}]),
                  norm("u5", "000:0:5", "g5", "train", "plain five text that is long enough here", [])]
    if bad_group:
        normalized[2]["provenance"]["recovery_group_id"] = "g4"  # group join now lands on another document
    tasks = [{"id": "bfcl-heldout:u1", "group_id": "g1", "route": "tool_call", "split": "test",
              "provenance": {"source_ids": ["547:b-000000:1"], "source_row_id": "u1"}},
             {"id": "bfcl-heldout:u4", "group_id": "g4", "route": "tool_parallel", "split": "test",
              "provenance": {"source_ids": ["547:b-000000:4"], "source_row_id": "u4"}},
             {"id": "tool-seed-x", "group_id": "gx", "route": "tool_call", "split": "test", "provenance": {}}]
    paths = {}

    def put(name, value, jsonl=False):
        path = tmp / name
        path.write_text("".join(json.dumps(v) + "\n" for v in value) if jsonl else json.dumps(value), encoding="utf-8")
        paths[name] = path

    put("tasks.jsonl", tasks, True)
    put("pin_tasks.jsonl", tasks[:2], True)
    put("training_groups.json", ["g0", "g2", "g3", "g5"])
    put("documents.jsonl", docs, True)
    put("normalized.jsonl", normalized, True)
    put("allocation.json", {"long": {"docs": []}, "k3": {"rows": []}, "selfgen": {"prompts": []},
                            "text": {"docs": [d["rid"] for d in docs]}})
    put("native.json", {"sha256": "n" * 64, "row_receipts": [{"row_id": "u3"}, {"row_id": "u0"}] + ([{"row_id": "u4"}] if native_leak else [])})
    put("manifest.json", {"shards": [{"path": "/x/text-00000.safetensors", "sha256": "a" * 64, "rows": [0, 2]},
                                     {"path": "/x/text-00001.safetensors", "sha256": "b" * 64, "rows": [2, 4]}]})
    rows, rids = [0, 1, 2, 3], ["547:b-000000:1", "547:b-000000:4"]
    put("exclude_rows.json", {"tool": "d2_mask_build", "tool_version": "1.0.0", "global_rows": rows, "global_rows_sha256": canonical(rows),
                              "bfcl62_rids": rids, "bfcl62_rids_sha256": canonical(rids)})
    return paths


def run(paths, out, *extra):
    argv = ["build", "--eval-tasks", str(paths["tasks.jsonl"]), "--bfcl-tasks", str(paths["pin_tasks.jsonl"]),
            "--bfcl-training-groups", str(paths["training_groups.json"]), "--documents", str(paths["documents.jsonl"]),
            "--normalized", str(paths["normalized.jsonl"]), "--allocation", str(paths["allocation.json"]),
            "--native-train-manifest", str(paths["native.json"]), "--corpus-manifest", str(paths["manifest.json"]),
            "--exclude-rows", str(paths["exclude_rows.json"]), "--out", str(out), "--no-expect", *extra]
    return M.main(argv)


def test_world_rows_rids_and_twin(tmp_path, monkeypatch):
    # rows: doc 1 -> {0,1}; doc 4 -> {2,3}.  EXPECT counts rows 3 by default; this world has 4.
    monkeypatch.setattr(M, "EXPECT", {"tasks": 2, "docs": 2, "rows": 4, "excluded_positions": 4 * SEQ, "bfcl_docs": 3})
    paths = write_world(tmp_path)
    out = tmp_path / "bfcl62_rids.json"
    assert run(paths, out) == 0
    o = json.loads(out.read_text(encoding="utf-8"))
    assert o["rids"] == ["547:b-000000:1", "547:b-000000:4"]
    assert o["global_rows"] == [0, 1, 2, 3] and o["global_rows_sha256"] == canonical([0, 1, 2, 3])
    assert o["corpora"]["a" * 64]["rows"] == [0, 1] and o["corpora"]["b" * 64]["rows"] == [0, 1]
    assert all(c["ok"] for k, c in o["cross_checks"].items() if c["hard"])
    te = o["twin_exposure"]
    # the twin (doc 3) sits in rows 1-2, inside the excluded rows, but it is also a native-train row -> still exposed
    assert te["tasks_exposed"] == ["bfcl-heldout:u1"] and te["clean_task_ids"] == ["bfcl-heldout:u4"]
    (hit,) = te["hits"]
    assert hit["uid"] == "u3" and hit["sources"] == ["text", "native_train"] and hit["same_answer"] is True
    assert o["counts"]["collateral_docs_in_excluded_rows"] == 4


def test_twin_only_inside_excluded_text_rows_is_covered(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "EXPECT", {"tasks": 2, "docs": 2, "rows": 4, "excluded_positions": 4 * SEQ, "bfcl_docs": 3})
    paths = write_world(tmp_path)
    native = json.loads(paths["native.json"].read_text())
    native["row_receipts"] = [{"row_id": "u0"}]
    paths["native.json"].write_text(json.dumps(native))
    out = tmp_path / "o.json"
    assert run(paths, out) == 0
    te = json.loads(out.read_text(encoding="utf-8"))["twin_exposure"]
    assert te["tasks_exposed"] == [] and te["hits"][0]["covered_by_excluded_rows"] is True


def test_count_mismatch_refuses(tmp_path):
    assert run(write_world(tmp_path), tmp_path / "o.json") == 2  # default EXPECT says 3 rows; the world has 4
    assert not (tmp_path / "o.json").exists()


def test_path_a_disagreeing_with_path_b_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "EXPECT", {"tasks": 2, "docs": 2, "rows": 4, "excluded_positions": 4 * SEQ, "bfcl_docs": 3})
    assert run(write_world(tmp_path, bad_group=True), tmp_path / "o.json") == 2


def test_held_out_uid_in_native_train_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "EXPECT", {"tasks": 2, "docs": 2, "rows": 4, "excluded_positions": 4 * SEQ, "bfcl_docs": 3})
    assert run(write_world(tmp_path, native_leak=True), tmp_path / "o.json") == 2


def test_pinned_sha_is_enforced_without_no_expect(tmp_path):
    paths = write_world(tmp_path)
    argv = ["build", "--eval-tasks", str(paths["tasks.jsonl"]), "--bfcl-tasks", str(paths["pin_tasks.jsonl"]),
            "--bfcl-training-groups", str(paths["training_groups.json"]), "--documents", str(paths["documents.jsonl"]),
            "--normalized", str(paths["normalized.jsonl"]), "--allocation", str(paths["allocation.json"]),
            "--native-train-manifest", str(paths["native.json"]), "--out", str(tmp_path / "o.json")]
    assert M.main(argv) == 2


def test_self_test_passes():
    assert M.self_test() == 0
