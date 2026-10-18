"""Tests for d2_bfcl_native_copy.py (card D2-M2; 1.1.0 adds the BFCL-62 task-twin drop).  CPU only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_bfcl_native_copy.py

A fake Codex preparation dir (6 packed rows, 3 of them BFCL) in Codex's own layout, plus one held-out BFCL test task
and a bfcl62_rids.json naming it.  BFCL row 5 asks for the held-out task's exact call (a twin: dropped); BFCL row 3
calls the same function with one other argument value (not a twin: kept -- the rule is sharp).  The copy must keep
the rest byte for byte, trim to the longest kept row, and refuse a selection that touches a held-out group, a listed
twin the answer check cannot confirm, or a mask that marks column 0.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import d2_bfcl_native_copy as B  # noqa: E402


def expected(value: int) -> dict:
    return {"bfcl_schema_target_adaptation": {"original_acceptable_targets": {"expected_tool_calls": [{"math.gcd": {"a": [value], "b": [18, ""]}}]}}}


def make_codex(tmp: Path, *, heldout_group: str = "g-none", bad_col0: bool = False, listed: tuple[str, ...] = ("r5",)) -> dict[str, Path]:
    from safetensors.torch import save_file

    width = 12
    lengths = torch.tensor([10, 6, 12, 5, 8, 9])
    ids = torch.randint(0, 1000, (6, width), generator=torch.Generator().manual_seed(1))
    target = torch.zeros(6, width, dtype=torch.bool)
    for r, n in enumerate(lengths.tolist()):
        target[r, n - 3: n] = True
    if bad_col0:
        target[1, 0] = True
    d = tmp / "codex"
    (d / "native-text-v3-train").mkdir(parents=True)
    (d / "corpus").mkdir()
    src = d / "native-text-v3-train" / "assistant_text.safetensors"
    save_file({"input_ids": ids, "lengths": lengths, "target_mask": target, "assistant_mask": target.clone()}, str(src))
    datasets = ["a", "x_bfcl_v4_y", "b", "x_bfcl_v4_y", "c", "x_bfcl_v4_y"]
    answers = {1: 99, 3: 13, 5: 12}  # the held-out task's answer is a=12
    receipts = [{"row_id": f"r{i}", "route": "tool_call" if "bfcl" in ds else "text_code", "rendered_sha256": f"h{i}"}
                for i, ds in enumerate(datasets)]
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    (d / "native-text-v3-train" / "manifest.json").write_text(json.dumps({"sha256": sha, "template_sha256": "t", "row_receipts": receipts}))
    lines = []
    for i, ds in enumerate(datasets):
        prov = {"registry_origin": {"dataset_id": ds, "rid": f"rid{i}"}, "recovery_group_id": heldout_group if i == 3 else f"g{i}"}
        if i in answers:
            prov.update(expected(answers[i]))
        lines.append(json.dumps({"id": f"r{i}", "provenance": prov}))
    (d / "corpus" / "train.jsonl").write_text("\n".join(lines) + "\n")
    (d / "corpus" / "validation.jsonl").write_text("")
    held = {"id": "held1", "provenance": {"registry_origin": {"dataset_id": "x_bfcl_v4_y", "rid": "ridH"}, "recovery_group_id": "g-held",
                                         **expected(12)}}
    (d / "corpus" / "test.jsonl").write_text(json.dumps(held) + "\n")
    tasks = tmp / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "bfcl-heldout:held1", "group_id": "g-held"}) + "\n" + json.dumps({"id": "tool-seed-1", "group_id": "g3"}) + "\n")
    holdout = tmp / "bfcl62_rids.json"
    holdout.write_text(json.dumps({"schema": B.HOLDOUT_SCHEMA, "tasks": [{"task_id": "bfcl-heldout:held1", "source_row_id": "held1",
                                                                         "group_id": "g-held"}],
                                   "twin_exposure": {"twin_ids_by_source": {"native_train": list(listed)}}}))
    return {"codex": d, "tasks": tasks, "holdout": holdout}


def argv(w: dict[str, Path], out: Path, *, expect: bool = False) -> list[str]:
    a = ["--codex-dir", str(w["codex"]), "--eval-tasks", str(w["tasks"]), "--holdout", str(w["holdout"]), "--out", str(out)]
    return a if expect else a + ["--no-expect"]


def test_copy_keeps_the_bfcl_rows_byte_for_byte_and_drops_the_task_twin(tmp_path):
    from safetensors.torch import load_file

    w = make_codex(tmp_path)
    out = tmp_path / "out"
    assert B.main(argv(w, out)) == 0
    got = load_file(str(out / B.out_name(2)))
    src = load_file(str(w["codex"] / "native-text-v3-train" / "assistant_text.safetensors"))
    # BFCL rows 1, 3, 5; row 5 is the twin -> rows 1 and 3 (lengths 6 and 5): trimmed to 6
    assert got["input_ids"].dtype == torch.int32 and got["input_ids"].shape == (2, 6)
    rows = json.loads((out / "rows.json").read_text())
    assert [r["source_row"] for r in rows["rows"]] == [1, 3]
    assert torch.equal(got["input_ids"].long(), src["input_ids"][[1, 3], :6])
    assert torch.equal(got["target_mask"], src["target_mask"][[1, 3], :6])
    assert [(d["row_id"], d["twin_of_tasks"], d["listed_by_holdout"]) for d in rows["dropped_twins"]] == [("r5", ["bfcl-heldout:held1"], True)]
    c = rows["counts"]
    assert c["rows"] == 2 and c["targets"] == 6 and c["dropped_twin_rows"] == 1 and c["dropped_twin_tokens"] == 9
    assert rows["selection_counts"] == {"rows": 3, "tokens": 20, "targets": 9, "max_length": 9}
    assert rows["output"]["sha256"] == hashlib.sha256((out / B.out_name(2)).read_bytes()).hexdigest()


def test_the_twin_rule_is_sharp_one_other_argument_value_is_not_a_twin(tmp_path):
    w = make_codex(tmp_path, listed=())
    assert B.main(argv(w, tmp_path / "out")) == 0
    rows = json.loads((tmp_path / "out" / "rows.json").read_text())
    # row 3 (a=13) stays; row 5 (a=12) is found by the answer check even though the holdout did not list it
    assert [r["source_row"] for r in rows["rows"]] == [1, 3]
    assert [(d["row_id"], d["listed_by_holdout"]) for d in rows["dropped_twins"]] == [("r5", False)]


def test_copy_refuses_a_heldout_group_an_unconfirmed_twin_and_a_column0_target(tmp_path, capsys):
    w = make_codex(tmp_path / "h", heldout_group="g-held")
    assert B.main(argv(w, tmp_path / "o1")) == 2
    assert "BFCL held-out task group" in capsys.readouterr().err
    assert not (tmp_path / "o1").exists()
    w = make_codex(tmp_path / "t", listed=("r5", "r3"))  # the holdout lists r3, whose answer differs
    assert B.main(argv(w, tmp_path / "o2")) == 2
    assert "answer check does not find them" in capsys.readouterr().err
    w = make_codex(tmp_path / "c", bad_col0=True)
    assert B.main(argv(w, tmp_path / "o3")) == 2
    assert "column 0" in capsys.readouterr().err
    w = make_codex(tmp_path / "p")
    assert B.main(argv(w, tmp_path / "o4", expect=True)) == 2  # not the pinned Codex corpus
    assert "pinned Codex corpus" in capsys.readouterr().err


def test_find_twins_matches_any_heldout_task():
    records = {"h1": {"expected": [{"f": {"x": [1]}}]}, "h2": {"expected": [{"g": {"y": ["a"]}}]},
               "u1": {"expected": [{"g": {"y": ["a", "b"]}}]}, "u2": {"expected": [{"f": {"x": [2]}}]}, "u3": {"expected": None}}
    assert B.find_twins(["u1", "u2", "u3"], records, {"h1": "t1", "h2": "t2"}) == {"u1": ["t2"]}


def test_the_output_is_byte_reproducible_and_carries_one_metadata_key(tmp_path):
    from safetensors import safe_open

    w = make_codex(tmp_path)
    shas = []
    for k in range(3):
        out = tmp_path / f"out{k}"
        assert B.main(argv(w, out)) == 0
        shas.append(hashlib.sha256((out / B.out_name(2)).read_bytes()).hexdigest())
    assert len(set(shas)) == 1
    with safe_open(str(tmp_path / "out0" / B.out_name(2)), "pt") as handle:
        meta = handle.metadata()
    assert list(meta) == [B.METADATA_KEY] and json.loads(meta[B.METADATA_KEY])["holdout_sha256"] == hashlib.sha256(w["holdout"].read_bytes()).hexdigest()
