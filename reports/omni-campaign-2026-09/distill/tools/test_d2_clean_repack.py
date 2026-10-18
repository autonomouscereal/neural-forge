"""Tests for d2_clean_repack.py (lane d2clean, 2026-10-15).  CPU only, no torch needed.

    CUDA_VISIBLE_DEVICES= python -m pytest -q test_d2_clean_repack.py

A tiny sealed registry (2 files x 3 rows x 8 positions, no padding) with hand-placed documents, a decisions file in the
real schema, v1 masks in d2_mask_build's format and a quality scores file.  Documents (global token spans):
    0 [0,5)   plain                       row 0
    1 [5,12)  HELD-OUT (Codex test, also a BFCL-62 source)   rows 0|1
    2 [12,14) BFCL JSON-in-prose (v1 all_mask)               row 1
    3 [14,27) v1 gt_mask + EOS-flagged (class 08)            rows 1|2|3   -> a BOUNDARY document (row 2 is kept)
    4 [27,33) BFCL-62 text TWIN                              rows 3|4
    5 [33,40) EOS-flagged (class 14)                         row 4
    6 [40,48) EOS-flagged (class 06)                         row 5        -> stays whole in kept row 5 (its EOS
              positions are masked there, in the sealed file)
Affected rows = rows of docs 1 and 4 = {0, 1, 3, 4}; kept rows = {2, 5}.  The build must produce exactly the
hand-derived re-pack, exclusion and masks; verify must agree; a planner that forgets the boundary document's kept-row
positions must be caught by verify (negative control); a mask that is not a union of whole documents, a flagged
document without the EOS tail and an existing --out are refused.
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

import d2_clean_repack as R  # noqa: E402

SEQ = 8
SEP = 248044
TAIL = [248046, 198, 248044]
SPEC = [(0, 5, None), (5, 12, None), (12, 14, None), (14, 27, "08"), (27, 33, None), (33, 40, "14"), (40, 48, "06")]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def doc_tokens(i: int, n: int, flagged: bool) -> list[int]:
    body = [100 + 10 * i + (k % 7) for k in range(n)]
    if flagged:
        body[-3:] = TAIL
    else:
        body[-1] = SEP
    return body


def write_world(tmp: Path, *, bad_tail: bool = False, partial_all_mask: bool = False) -> dict[str, Path]:
    from safetensors.numpy import save_file

    corpus = tmp / "corpus" / "text"
    corpus.mkdir(parents=True)
    ids = np.zeros(48, dtype=np.int32)
    doc_index = np.zeros(48, dtype=np.int32)
    docs = []
    routes = {"chat": 0, "code": 1}
    route_of = []
    for i, (s, e, cls) in enumerate(SPEC):
        toks = doc_tokens(i, e - s, cls is not None)
        if bad_tail and i == 5:
            toks[-3] = 77
        ids[s:e] = toks
        doc_index[s:e] = i
        route = "code" if i % 2 else "chat"
        route_of += [routes[route]] * (e - s)
        docs.append({"doc": i, "rid": f"r{i}", "uid": f"uid-{i}", "record_id": f"rec-{i}", "dataset_id": "bfcl" if i == 2 else "ds",
                     "route": route, "render": "chat", "start": s, "end": e, "length": e - s, "kept": e - s,
                     "row_first": s // SEQ, "row_last": (e - 1) // SEQ})
    route_ids = np.array(route_of, dtype=np.uint8)
    (corpus / "documents.jsonl").write_text("".join(json.dumps(d, sort_keys=True) + "\n" for d in docs))
    shards = []
    for f in range(2):
        rows = slice(3 * f * SEQ, 3 * (f + 1) * SEQ)
        path = corpus / f"text-{f:05d}.safetensors"
        save_file({"input_ids": ids[rows].reshape(3, SEQ).copy(), "doc_index": doc_index[rows].reshape(3, SEQ).copy(),
                   "route_ids": route_ids[rows].reshape(3, SEQ).copy(), "real_lengths": np.full(3, SEQ, dtype=np.int32)}, str(path))
        shards.append({"path": str(path), "sha256": sha(path), "rows": [3 * f, 3 * f + 3], "tokens": 3 * SEQ})
    manifest = {"sequence_length": SEQ, "rows": 6, "shards": shards, "documents_file": {"path": str(corpus / "documents.jsonl"), "sha256": sha(corpus / "documents.jsonl")},
                "tokenizer": {"sha256": "tok", "separator_id": SEP}, "route_table": routes,
                "tensors": {"input_ids": "int32 [rows, 8]"}}
    manifest["manifest_sha256"] = R.DCB.canonical_sha256(manifest)
    (corpus / "manifest.json").write_text(json.dumps(manifest, indent=1))
    # Codex side: rows.jsonl join and a held-out pack
    (tmp / "rows.jsonl").write_text("".join(json.dumps({"rid": d["rid"], "uid": d["uid"], "record_id": d["record_id"]}) + "\n" for d in docs))
    pack = tmp / "pack-test" / "manifest.json"
    pack.parent.mkdir()
    pack.write_text(json.dumps({"row_receipts": [{"row_id": "uid-1"}, {"row_id": "not-in-registry"}]}))
    (tmp / "bfcl62_rids.json").write_text(json.dumps({"rids": ["r1"]}))
    masks = tmp / "corpus" / "d2_masks_v1"
    masks.mkdir()
    table = [{"file": Path(s["path"]).name, "path": s["path"], "sha256": s["sha256"], "row_offset": s["rows"][0], "rows": 3} for s in shards]
    excl = {"schema": R.MB.EXCLUDE_SCHEMA, "global_rows": [0, 1, 3, 4], "text_twin_rids": ["r4"],
            "corpora": {shards[0]["sha256"]: {"rows": [0, 1], "source": "registry"}, shards[1]["sha256"]: {"rows": [0, 1], "source": "registry"},
                        "k3sha": {"file": "k3/pass-2/qwen_input_ids.safetensors", "row_offset": None, "rows": [10], "source": "k3"}}}
    (masks / "exclude_rows.json").write_text(json.dumps(excl))
    all_spans = [(1, 3, 4)] if partial_all_mask else [(1, 3, 5)]  # doc 2 owns positions [11, 13)
    gt_spans = R.MB.merge_spans(R.MB.split_by_row(13, 26, SEQ))  # doc 3 owns [13, 26)
    for name, kind, spans in (("all_mask.json", "all", all_spans), ("gt_mask.json", "gt", gt_spans)):
        (masks / name).write_text(json.dumps({"schema": R.MB.MASK_SCHEMA, "kind": kind, "corpora": R.MB.per_file(spans, table)}))
    scores = tmp / "scores.jsonl"
    scores.write_text("".join(json.dumps({"rid": f"r{i}", "class": cls or "15"}) + "\n" for i, (_, _, cls) in enumerate(SPEC)))
    decisions = {"schema": "omni-campaign.d2-decisions.v1",
                 "canonical_split": {"heldout_packs": [{"name": "short_test", "manifest": str(pack), "manifest_sha256": sha(pack), "rows": 2}]},
                 "exclusion": {"sets": {"bfcl62": {"path": str(tmp / "bfcl62_rids.json"), "sha256": sha(tmp / "bfcl62_rids.json")},
                                        "a2_3a_row_exclusion_unchanged": {"path": str(masks / "exclude_rows.json"), "sha256": sha(masks / "exclude_rows.json")}}},
                 "eos_mask": {"flag": {"source": f"{scores} (quality_scorer 1.0.4)", "sha256": sha(scores),
                                       "classes": {"06": "a", "08": "b", "14": "c"}}}}  # the real file's "<path> (<note>)" form
    (tmp / "decisions.json").write_text(json.dumps(decisions))
    return {"corpus": corpus, "masks": masks, "rows": tmp / "rows.jsonl", "decisions": tmp / "decisions.json", "out": tmp / "clean",
            "cache_root": tmp / "cache"}


def build_argv(w: dict[str, Path]) -> list[str]:
    return ["build", "--decisions", str(w["decisions"]), "--corpus-dir", str(w["corpus"]), "--rows", str(w["rows"]),
            "--masks-v1", str(w["masks"]), "--out", str(w["out"]), "--cache-root", str(w["cache_root"]), "--no-screen", "--no-expect"]


def spans_of(mask: dict, sha_: str) -> list[list[int]]:
    return (mask["corpora"].get(sha_) or {}).get("spans", [])


def test_self_test():
    assert R.self_test() == 0


def test_build_matches_the_hand_derivation_and_verifies(tmp_path):
    from safetensors.numpy import load_file

    w = write_world(tmp_path)
    assert R.main(build_argv(w)) == 0
    out = w["out"]
    man = json.loads((out / "manifest.json").read_text())
    c = man["counts"]
    assert (c["affected_rows"], c["kept_rows"], c["x_docs"], c["repack_docs"], c["boundary_docs"]) == (4, 2, 2, 4, 1)
    assert c["x_tokens"] == 7 + 6 and c["repack_tokens"] == 5 + 2 + 13 + 7 and c["kept_whole_docs"] == 1
    assert man["conservation"]["sum"] == 48 and c["kept_row_runs"] == 2
    # re-pack: docs 0, 2, 3, 5 whole and in order, 27 tokens -> 4 rows of 8, the last 3 real
    sealed = np.concatenate([load_file(str(w["corpus"] / f"text-{f:05d}.safetensors"))["input_ids"].reshape(-1) for f in range(2)])
    want = np.concatenate([sealed[s:e] for s, e, _ in (SPEC[0], SPEC[2], SPEC[3], SPEC[5])])
    t = load_file(str(out / "text-clean-00000.safetensors"))
    assert t["input_ids"].shape == (4, SEQ)
    assert np.array_equal(t["input_ids"].reshape(-1)[:27], want)
    assert t["real_lengths"].tolist() == [8, 8, 8, 3]
    assert (t["doc_index"].reshape(-1)[27:] == -1).all() and (t["input_ids"].reshape(-1)[27:] == SEP).all()
    assert (t["route_ids"].reshape(-1)[27:] == 255).all()
    assert t["doc_index"].reshape(-1)[:27].tolist() == [0] * 5 + [1] * 2 + [2] * 13 + [3] * 7
    table = [json.loads(x) for x in (out / "documents.jsonl").read_text().splitlines()]
    assert [d["source_doc"] for d in table] == [0, 2, 3, 5]
    assert [(d["start"], d["end"]) for d in table] == [(0, 5), (5, 7), (7, 20), (20, 27)]
    assert [d["source_class"] for d in table] == ["inside_affected", "inside_affected", "boundary", "inside_affected"]
    # exclusion: rows 0,1 | 3,4 per sealed file, the K3 entry carried unchanged
    ex = json.loads((out / "clean_exclude_rows.json").read_text())
    s0, s1 = (json.loads((w["corpus"] / "manifest.json").read_text())["shards"][f]["sha256"] for f in range(2))
    assert ex["global_rows"] == [0, 1, 3, 4] and ex["corpora"][s0]["rows"] == [0, 1] and ex["corpora"][s1]["rows"] == [0, 1]
    assert ex["corpora"]["k3sha"]["rows"] == [10]
    # all mask: kept row 2 is all doc 3 (re-packed boundary) -> masked; kept row 5 is doc 6 (whole in K) -> trained
    allm = json.loads((out / "clean_all_mask.json").read_text())
    assert spans_of(allm, s0) == [[2, 0, 8]] and spans_of(allm, s1) == []
    new_sha = man["shards"][0]["sha256"]
    assert spans_of(allm, new_sha) == [[0, 4, 6]]  # doc 2 at re-pack tokens [5, 7) owns positions [4, 6)
    gtm = json.loads((out / "clean_gt_mask.json").read_text())
    # doc 3 at re-pack [7, 20) owns [6, 19) (its EOS [16, 19) inside); doc 5 at [20, 27): EOS [23, 26);
    # doc 6 stays in kept row 5: EOS positions [44, 47) = sealed file 1 local row 2 cols [4, 7)
    assert spans_of(gtm, new_sha) == [[0, 6, 8], [1, 0, 8], [2, 0, 3], [2, 7, 8], [3, 0, 2]]
    assert spans_of(gtm, s0) == [] and spans_of(gtm, s1) == [[2, 4, 7]]
    assert man["counts"]["eos_flagged_docs_masked"] == 3
    assert R.main(["verify", "--out", str(out), "--corpus-dir", str(w["corpus"])]) == 0
    rec = json.loads((out / "verify_receipt.json").read_text())
    assert rec["ok"] and rec["x_docs"] == 2 and rec["conservation"]
    # state: built + verified, no cache yet -> 10; a sealed cache manifest per file -> 0
    assert R.main(["state", "--out", str(out), "--cache-root", str(w["cache_root"])]) == R.EXIT_WAITING
    for f in man["shards"]:
        d = w["cache_root"] / f["sha256"]
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"all_finite": True, "all_structural_ok": True,
                                                     "teacher": {"equal_before_and_after_every_session": True}}))
    assert R.main(["state", "--out", str(out), "--cache-root", str(w["cache_root"])]) == 0


def test_existing_out_is_refused(tmp_path):
    w = write_world(tmp_path)
    w["out"].mkdir()
    assert R.main(build_argv(w)) == 2


def test_negative_control_planner_that_forgets_boundary_positions_is_caught_by_verify(tmp_path, monkeypatch):
    w = write_world(tmp_path)
    real = R.plan_repack

    def broken(*a, **k):
        p = real(*a, **k)
        p["kept_row_all_spans"] = []  # the bug: kept-row positions of re-packed / X documents left trainable
        return p

    monkeypatch.setattr(R, "plan_repack", broken)
    assert R.main(build_argv(w)) == 0
    monkeypatch.setattr(R, "plan_repack", real)
    assert R.main(["verify", "--out", str(w["out"]), "--corpus-dir", str(w["corpus"])]) == 1
    rec = json.loads((w["out"] / "verify_receipt.json").read_text())
    assert any("misses" in p for p in rec["problems"])


def test_tampered_repack_file_fails_verify(tmp_path):
    w = write_world(tmp_path)
    assert R.main(build_argv(w)) == 0
    path = w["out"] / "text-clean-00000.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(bytes(data))
    assert R.main(["verify", "--out", str(w["out"]), "--corpus-dir", str(w["corpus"])]) == 1


def test_mask_that_is_not_whole_documents_is_refused(tmp_path):
    w = write_world(tmp_path, partial_all_mask=True)
    assert R.main(build_argv(w)) == 2
    assert not w["out"].exists()


def test_flagged_document_without_eos_tail_is_refused(tmp_path):
    w = write_world(tmp_path, bad_tail=True)
    assert R.main(build_argv(w)) == 2
    assert not w["out"].exists()


def test_pins_refuse_other_inputs_without_no_expect(tmp_path):
    w = write_world(tmp_path)
    argv = [a for a in build_argv(w) if a != "--no-expect"]
    assert R.main(argv) == 2  # the synthetic decisions file is not the pinned one
