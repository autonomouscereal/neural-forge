"""Tests for media_a_corpus_items.py 1.0.0 (lane gpu4, Claude, 2026-10-13). Server or workstation, no GPU:

    cd /home/cereal/omni-campaign/distill/tools
    nice -n 15 /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" -p no:cacheprovider -q test_media_a_corpus_items.py

The converted file must be accepted by stage_c_media_a.read_items unchanged (the contract is read from the consumer,
not copied), keep corpus order and count, carry sha256 + negative, and refuse missing uids, duplicates, missing files
and changed bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import media_a_corpus_items as mci  # noqa: E402
import stage_c_media_a as media_a  # noqa: E402


def _corpus(tmp_path: Path, n: int = 3, *, negative: bool = True) -> tuple[Path, list[dict]]:
    rows = []
    for i in range(n):
        img = tmp_path / "store" / f"img{i}.png"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(f"png-bytes-{i}".encode())
        row = {"uid": f"u{i:03d}", "rid": f"1:0:{i}", "dataset_id": "d", "tier": "description", "task_type": "t",
               "route": "image", "prompt": f"prompt {i}", "prompt_source": "row response verbatim",
               "image": {"path": f"/nas/img{i}.png", "stored_path": str(img), "bytes": img.stat().st_size,
                         "sha256": hashlib.sha256(img.read_bytes()).hexdigest(), "width": 8, "height": 8}}
        if negative:
            row["negative_prompt"] = ""
        rows.append(row)
    src = tmp_path / "items.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return src, rows


def test_converted_items_are_read_by_stage_c_media_a(tmp_path):
    src, rows = _corpus(tmp_path)
    out = tmp_path / "out" / "qimg_items.jsonl"
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out), "--check-files", "--hash"]) == 0
    items = media_a.read_items(out, media_key="image")
    assert [i["id"] for i in items] == [r["uid"] for r in rows]
    assert [i["image"] for i in items] == [r["image"]["stored_path"] for r in rows]
    assert all(i["sha256"] == r["image"]["sha256"] for i, r in zip(items, rows))
    assert all(i["negative"] == "" for i in items)
    assert items[1]["corpus"]["line"] == 2 and items[1]["corpus"]["image_source_path"] == "/nas/img1.png"
    # prompts + the one shared empty negative: what qimg-text will encode
    texts = media_a.texts_for(items, negatives=True, default_negative="")
    assert texts == ["prompt 0", "", "prompt 1", "prompt 2"]
    receipt = json.loads((out.parent / "qimg_items.jsonl.receipt.json").read_text())
    assert receipt["items"] == 3 and receipt["with_negative"] == 3 and receipt["file_check"]["rehashed"] is True
    assert receipt["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_nas_path_key_and_no_negative(tmp_path):
    src, rows = _corpus(tmp_path, negative=False)
    out = tmp_path / "o.jsonl"
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out), "--media-path-key", "path"]) == 0
    items = media_a.read_items(out, media_key="image")
    assert items[0]["image"] == "/nas/img0.png" and "negative" not in items[0]


def test_rerun_is_idempotent_and_refuses_silent_change(tmp_path):
    src, rows = _corpus(tmp_path)
    out = tmp_path / "o.jsonl"
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out)]) == 0
    first = out.read_bytes()
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out)]) == 0
    assert out.read_bytes() == first
    rows[0]["prompt"] = "changed"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out)]) == 2
    assert out.read_bytes() == first
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(out), "--overwrite"]) == 0
    assert media_a.read_items(out, media_key="image")[0]["prompt"] == "changed"


@pytest.mark.parametrize("mutate, message", [
    (lambda r: r.pop("uid"), "no uid"),
    (lambda r: r.__setitem__("prompt", "  "), "no prompt"),
    (lambda r: r.__setitem__("image", "/a/path.png"), "no 'image' dict"),
    (lambda r: r["image"].pop("stored_path"), "stored_path is missing"),
    (lambda r: r.__setitem__("negative_prompt", None), "negative_prompt is not a string"),
])
def test_bad_rows_refuse(tmp_path, mutate, message, capsys):
    src, rows = _corpus(tmp_path)
    mutate(rows[1])
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(tmp_path / "o.jsonl")]) == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "o.jsonl").exists()


def test_duplicate_missing_and_changed_files_refuse(tmp_path, capsys):
    src, rows = _corpus(tmp_path)
    rows[2]["uid"] = rows[0]["uid"]
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(tmp_path / "o.jsonl")]) == 2
    assert "duplicate uid" in capsys.readouterr().err
    src, rows = _corpus(tmp_path)
    Path(rows[1]["image"]["stored_path"]).unlink()
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(tmp_path / "o.jsonl"), "--check-files"]) == 2
    assert "1 missing file(s)" in capsys.readouterr().err
    src, rows = _corpus(tmp_path)
    Path(rows[2]["image"]["stored_path"]).write_bytes(b"other bytes")
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(tmp_path / "o.jsonl"), "--check-files"]) == 0
    assert mci.main(["convert", "--route", "image", "--src", str(src), "--out", str(tmp_path / "o2.jsonl"), "--check-files", "--hash"]) == 2
    assert "1 sha256 mismatch(es)" in capsys.readouterr().err
