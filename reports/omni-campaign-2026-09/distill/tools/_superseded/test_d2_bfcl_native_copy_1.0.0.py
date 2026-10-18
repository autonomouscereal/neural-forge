"""Tests for d2_bfcl_native_copy.py (card D2-M2).  CPU only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_bfcl_native_copy.py

A fake Codex preparation dir (5 packed rows, 2 of them BFCL) in Codex's own layout: the copy must select exactly the
BFCL rows, keep their ids and masks byte-for-byte, trim to the longest selected row, and refuse a selection that
touches a BFCL held-out group (negative control) or a mask that marks column 0.
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


def make_codex(tmp: Path, *, heldout_group: str = "g-none", bad_col0: bool = False) -> tuple[Path, Path]:
    from safetensors.torch import save_file

    width = 12
    lengths = torch.tensor([10, 6, 12, 5, 8])
    ids = torch.randint(0, 1000, (5, width), generator=torch.Generator().manual_seed(1))
    target = torch.zeros(5, width, dtype=torch.bool)
    for r, n in enumerate(lengths.tolist()):
        target[r, n - 3: n] = True
    if bad_col0:
        target[1, 0] = True
    d = tmp / "codex"
    (d / "native-text-v3-train").mkdir(parents=True)
    (d / "corpus").mkdir()
    src = d / "native-text-v3-train" / "assistant_text.safetensors"
    save_file({"input_ids": ids, "lengths": lengths, "target_mask": target, "assistant_mask": target.clone()}, str(src))
    datasets = ["a", "x_bfcl_v4_y", "b", "x_bfcl_v4_y", "c"]
    receipts = [{"row_id": f"r{i}", "route": "tool_call" if "bfcl" in ds else "text_code", "rendered_sha256": f"h{i}"}
                for i, ds in enumerate(datasets)]
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    (d / "native-text-v3-train" / "manifest.json").write_text(json.dumps({"sha256": sha, "template_sha256": "t", "row_receipts": receipts}))
    lines = [json.dumps({"id": f"r{i}", "provenance": {"registry_origin": {"dataset_id": ds, "rid": f"rid{i}"},
                                                         "recovery_group_id": heldout_group if i == 3 else f"g{i}"}}) for i, ds in enumerate(datasets)]
    (d / "corpus" / "train.jsonl").write_text("\n".join(lines) + "\n")
    (d / "corpus" / "validation.jsonl").write_text("")
    (d / "corpus" / "test.jsonl").write_text("")
    tasks = tmp / "tasks.jsonl"
    tasks.write_text(json.dumps({"id": "bfcl-heldout:1", "group_id": "g-held"}) + "\n" + json.dumps({"id": "tool-seed-1", "group_id": "g3"}) + "\n")
    return d, tasks


def test_copy_selects_the_bfcl_rows_byte_for_byte(tmp_path):
    from safetensors.torch import load_file

    codex, tasks = make_codex(tmp_path)
    out = tmp_path / "out"
    assert B.main(["--codex-dir", str(codex), "--eval-tasks", str(tasks), "--out", str(out), "--no-expect"]) == 0
    got = load_file(str(out / B.OUT_NAME))
    src = load_file(str(codex / "native-text-v3-train" / "assistant_text.safetensors"))
    assert got["input_ids"].dtype == torch.int32 and got["input_ids"].shape == (2, 6)  # rows 1 and 3 (lengths 6 and 5): trimmed to 6
    rows = json.loads((out / "rows.json").read_text())
    assert [r["source_row"] for r in rows["rows"]] == [1, 3]
    assert torch.equal(got["input_ids"].long(), src["input_ids"][[1, 3], :6])
    assert torch.equal(got["target_mask"], src["target_mask"][[1, 3], :6])
    assert rows["counts"]["rows"] == 2 and rows["counts"]["targets"] == 6 and rows["output"]["sha256"] == hashlib.sha256((out / B.OUT_NAME).read_bytes()).hexdigest()


def test_copy_refuses_a_heldout_group_and_a_column0_target(tmp_path, capsys):
    codex, tasks = make_codex(tmp_path / "h", heldout_group="g-held")
    assert B.main(["--codex-dir", str(codex), "--eval-tasks", str(tasks), "--out", str(tmp_path / "o1"), "--no-expect"]) == 2
    assert "BFCL held-out task group" in capsys.readouterr().err
    assert not (tmp_path / "o1" / B.OUT_NAME).exists()
    codex, tasks = make_codex(tmp_path / "c", bad_col0=True)
    assert B.main(["--codex-dir", str(codex), "--eval-tasks", str(tasks), "--out", str(tmp_path / "o2"), "--no-expect"]) == 2
    assert "column 0" in capsys.readouterr().err
    codex, tasks = make_codex(tmp_path / "p")
    assert B.main(["--codex-dir", str(codex), "--out", str(tmp_path / "o3")]) == 2  # not the pinned Codex corpus
    assert "pinned Codex corpus" in capsys.readouterr().err
