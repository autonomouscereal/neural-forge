"""Tests for distill_corpus_build.py (run on the server with PYTHONPATH and CHIMERA_OMNI_SRC exported).

    CHIMERA_OMNI_SRC=$OC/import/rq27b/tree_ef1ffb9/src PYTHONPATH=$OC/distill/tools \\
    $PY -m pytest -q $OC/distill/tools/test_distill_corpus_build.py

Tests that need the 27B tokenizer, transformers, the chimera tree or the K3 tokenizer files skip
(and say so) where those are absent; on the server none of them skips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import distill_corpus_build as dcb  # noqa: E402

QWEN_DIR = Path(dcb.DEFAULT_QWEN_DIR)
K3_DIR = Path(dcb.DEFAULT_K3_DIR)
CHIMERA_SRC = dcb.resolve_chimera_src(None)
HAVE_QWEN = (QWEN_DIR / "tokenizer.json").is_file()
HAVE_K3 = (K3_DIR / "tiktoken.model").is_file()
HAVE_CHIMERA = (CHIMERA_SRC / "chimera_omni" / "data" / "curated_shards.py").is_file()
try:
    import transformers  # noqa: F401

    HAVE_TRANSFORMERS = True
except Exception:  # pragma: no cover - environment
    HAVE_TRANSFORMERS = False

LONG_ITEM = "the winter delivery window moves to thursday at six forty and all drivers must enter through the north gate of the depot"
SHORT_ITEM = "measure the gap never guess it"


def test_version_string_present():
    assert dcb.VERSION.count(".") == 2
    assert dcb.parse_args(["screen"]).step == "screen"


def test_words_cjk_and_case():
    assert dcb.words("Hello, 今天 World_x 42") == ["hello", "今", "天", "world", "x", "42"]
    assert dcb.words("ＡＢＣ") == ["abc"]  # NFKC


def make_index() -> dcb.ReviewIndex:
    index = dcb.ReviewIndex()
    index.add_text("bank", "long", LONG_ITEM)
    index.add_text("bank", "short", SHORT_ITEM)
    index.add_literal("codes", "c1", "ZQ7-4418")
    index.add_media("assets", "a.png", "ab" * 32)
    return index


def test_review_index_rules():
    index = make_index()
    assert index.hits("Preface. " + LONG_ITEM.upper() + " Afterword.")[0]["rule"] == "ngram13"
    assert index.hits("\n".join(LONG_ITEM.split()))  # re-wrapped
    assert index.hits("He said: " + SHORT_ITEM + ", then left.")[0]["rule"] == "short_item"
    assert index.hits("ticket zq7-4418 closed")[0]["rule"] == "literal"
    exact = index.hits(LONG_ITEM)
    assert {h["rule"] for h in exact} >= {"exact", "ngram13"}
    fragment = " ".join(LONG_ITEM.split()[3:15])  # 12 words: below the n-gram
    assert not index.hits("note: " + fragment)
    assert not index.hits("The pump was serviced on the 3rd and the reading settled at 9 units.")


def test_liveness_fragment_control_not_rebuilt_by_neighbour():
    # the item's word after the 12-word middle fragment is "and": an ordinary splice
    # "... [fragment] and closed it." rebuilt a real 13-gram in 1.0.0 and failed the control
    toks = [f"w{i}" for i in range(30)]
    toks[27] = "and"
    toks[14] = "wrote"
    index = dcb.ReviewIndex()
    index.add_text("bank", "item", " ".join(toks))
    live = dcb.screen_liveness(index)
    assert live["totals"]["fragment_controls"] == 1
    assert live["totals"].get("fragment_controls_falsely_caught", 0) == 0
    assert live["screen_is_live"], live


def test_exact_needs_four_words():
    index = dcb.ReviewIndex()
    index.add_text("bank", "a", "A")
    index.add_text("bank", "yes", "yes it is")
    assert not index.hits("A")
    assert not index.hits("yes it is")


def test_screen_liveness_live_and_dead():
    index = make_index()
    live = dcb.screen_liveness(index)
    assert live["screen_is_live"], live

    class Noisy(dcb.ReviewIndex):
        def hits(self, text, exact_keys=None, toks=None):
            return [{"rule": "x", "source": "y", "item": "z"}]

    noisy = Noisy()
    noisy.add_text("bank", "long", LONG_ITEM)
    assert not dcb.screen_liveness(noisy)["screen_is_live"]


def test_extract_keeps_newlines_and_no_cut():
    body = "def f(x):\n    return x\n" * 1000
    row = {"instruction": "write f", "response": body}
    assert dcb.extract_response(row) == body.strip()
    assert len(dcb.extract_response(row)) > 8192
    assert dcb.raw_render("a", "b") == "a\n\nb"
    assert dcb.raw_render("", "b") == "b"


@pytest.mark.skipif(not HAVE_CHIMERA, reason="chimera-omni tree absent")
def test_extract_field_order_parity_with_loader():
    cs = dcb.import_curated_shards(CHIMERA_SRC)
    shapes = [
        {"instruction": "  Do X\n now ", "response": "Done.\n\nYes"},
        {"prompt": "P", "target_json": {"answer": "A  b"}},
        {"input_json": {"messages": [{"role": "user", "content": "hi\nthere"}]}, "target_json": {"messages": [{"role": "assistant", "content": "yo"}]}},
        {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a\n1"}]},
        {"question": "q?", "output": "o"},
    ]
    for row in shapes:
        assert " ".join(dcb.extract_instruction(row).split()) == cs._extract_instruction(row)
        assert " ".join(dcb.extract_response(row).split()) == cs._extract_response(row)


def row(rid, route="text", text="alpha beta", family="text", sidecars=(), uid=None, evidence="e"):
    r = {"rid": rid, "uid": uid or rid, "route": route, "family": family, "text_key": dcb.text_sha256(text), "evidence_key": evidence + rid, "sidecars": list(sidecars)}
    r["dedup_key"] = dcb.dedup_key(r)
    return r


def test_dedup_key_media_rows_distinct():
    a = row("1", route="image", family="media", text="t", sidecars=[{"sha256": "a" * 64}])
    b = row("2", route="image", family="media", text="t", sidecars=[{"sha256": "b" * 64}])
    c = row("3", route="text", text="t")
    d = row("4", route="text", text="t")
    assert a["dedup_key"] != b["dedup_key"]
    assert c["dedup_key"] == d["dedup_key"]


def test_carve_eval_rules():
    rows = [row(str(i), text=f"text {i}") for i in range(2000)]
    rows.append(row("dupA", text="shared text"))
    rows.append(row("dupB", text="shared text"))
    forced = {"dupA"}
    assignment = dcb.carve_eval(rows, forced_eval_uids=forced, evidence_keys=set())
    assert assignment["dupA"] == "eval" and assignment["dupB"] == "eval"
    share = sum(1 for v in assignment.values() if v == "eval") / len(rows)
    assert 0.01 < share < 0.04
    again = dcb.carve_eval(rows, forced_eval_uids=forced, evidence_keys=set())
    assert again == assignment
    # fold-evidence documents never land in eval
    evidence_keys = {r["evidence_key"] for r in rows if assignment[r["rid"]] == "eval" and r["rid"] not in forced}
    moved = dcb.carve_eval(rows, forced_eval_uids=forced, evidence_keys=evidence_keys)
    assert all(moved[r["rid"]] == "train" for r in rows if r["evidence_key"] in evidence_keys and r["rid"] not in forced)


def test_carve_eval_route_floor():
    rows = [row(f"s{i}", route="tiny", text=f"tiny {i}") for i in range(40)]
    assignment = dcb.carve_eval(rows, forced_eval_uids=set(), evidence_keys=set())
    assert sum(1 for v in assignment.values() if v == "eval") >= 4


def test_largest_remainder():
    out = dcb.largest_remainder(31, {"a": 45.0, "b": 15.0, "c": 13.0, "d": 7.0})
    assert sum(out.values()) == 31
    assert out["a"] > out["b"] > out["d"]


def test_interleave_by_deficit_prefix_shares():
    queues = {"a": [("a", 10)] * 300, "b": [("b", 10)] * 100}
    taken = dcb.interleave_by_deficit(queues, lambda item: item[1], 2000)
    assert sum(i[1] for i in taken) >= 2000
    prefix = taken[:40]
    share_a = sum(1 for i in prefix if i[0] == "a") / len(prefix)
    assert abs(share_a - 0.75) <= 0.05


def test_pack_rows_first_fit():
    items = [("x", n) for n in (3000, 2000, 1000, 1096, 4096, 50)]
    bins, unused = dcb.pack_rows_first_fit(items, lambda i: i[1], rows=3, capacity=4096, lookahead=10)
    for b in bins:
        assert sum(i[1] for i in b) <= 4096
    assert [sum(i[1] for i in b) for b in bins] == [4050, 3096, 4096]
    assert unused == []


def test_stream_pack():
    docs = [[1, 2, 3, 9], [4, 5, 9], [6, 7, 8, 8, 8, 9]]
    ids, doc_of, spans = dcb.stream_pack(docs, rows=3, seq=4, pad_id=9)
    assert ids.shape == (3, 4)
    flat = ids.reshape(-1).tolist()
    assert flat == [1, 2, 3, 9, 4, 5, 9, 6, 7, 8, 8, 8]
    assert spans[-1]["kept"] == 5 and spans[-1]["length"] == 6
    assert (doc_of >= 0).all()
    ids2, doc_of2, _ = dcb.stream_pack(docs[:1], rows=2, seq=4, pad_id=9)
    assert (doc_of2[1] == -1).all()


def test_compose_long_tasks():
    sections = [f"Row {i} begins with these twelve plain words that make section {i} unique here. More text." for i in range(20)]
    composed = dcb.compose_long(sections, "text", 3)
    again = dcb.compose_long(sections, "text", 3)
    assert composed == again
    assert len(composed["needles"]) == 3
    for needle in composed["needles"]:
        assert needle["text"] in composed["user"]
        assert needle["value"] in composed["assistant"]
    assert composed["passage"] is not None
    assert composed["answers"][-1] == f"Section {composed['passage']['section']}"
    assert len(composed["questions"]) == len(composed["answers"]) == 4
    review = make_index()
    assert not review.hits(composed["user"] + composed["assistant"])


def test_ltx_suspect_and_centered_start():
    v9 = {"dataset_id": "v9_base_validated_28025", "sidecars": [{"media": "video"}]}
    other = {"dataset_id": "v36_addenda", "sidecars": [{"media": "video"}]}
    icon = {"dataset_id": "v9_base_validated_28025", "sidecars": [{"media": "image"}]}
    assert dcb.ltx_suspect(v9) and not dcb.ltx_suspect(other) and not dcb.ltx_suspect(icon)
    assert dcb.centered_start(4.0, 2.0) == 1.0
    assert dcb.centered_start(1.0, 2.0) == 0.0
    assert dcb.centered_start(None, 2.0) == 0.0


def test_media_kind():
    assert dcb.media_kind("/a/b.PNG") == "image"
    assert dcb.media_kind("/a/b.mp4") == "video"
    assert dcb.media_kind("/a/b.flac") == "audio"
    assert dcb.media_kind("/a/b.json") is None


def test_parse_prompt_pack(tmp_path):
    script = tmp_path / "s.py"
    script.write_text('import torch\nPROMPT_PACK: tuple[str, ...] = (\n    "a",\n    "b c",\n)\n', encoding="utf-8")
    assert dcb.parse_prompt_pack(script) == ["a", "b c"]


@pytest.mark.skipif(not HAVE_CHIMERA, reason="chimera-omni tree absent")
def test_rq_prompt_pack_is_eight_prompts():
    prompts = dcb.parse_prompt_pack(CHIMERA_SRC.parent / "scripts" / "import_receipt_qwen35.py")
    assert len(prompts) == 8


def test_write_json_seal(tmp_path):
    path = tmp_path / "m.json"
    seal = dcb.write_json(path, {"a": 1, "b": [1, 2]}, seal_field="manifest_sha256")
    body = json.loads(path.read_text())
    assert body["manifest_sha256"] == seal == dcb.canonical_sha256({"a": 1, "b": [1, 2]})


@pytest.mark.skipif(not HAVE_TRANSFORMERS, reason="transformers absent")
def test_visual_tokens_image_and_video():
    processor = {
        "image_processor": {"patch_size": 16, "merge_size": 2, "size": {"shortest_edge": 65536, "longest_edge": 16777216}},
        "video_processor": {"patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "fps": 2, "min_frames": 4, "max_frames": 768, "size": {"shortest_edge": 4096, "longest_edge": 25165824}},
    }
    image = dcb.visual_tokens_image(768, 540, processor)
    assert (image["resized_height"], image["resized_width"]) == (544, 768)
    assert image["visual_tokens"] == (544 // 16) * (768 // 16) // 4 == 408
    small = dcb.visual_tokens_image(100, 100, processor)  # below min_pixels: scaled up
    assert small["resized_height"] * small["resized_width"] >= 65536
    video = dcb.visual_tokens_video(640, 360, 64, 16.0, processor)
    assert video["sampled_frames"] == 8 and video["grid_t"] == 4
    assert video["visual_tokens"] == 4 * (video["resized_height"] // 16) * (video["resized_width"] // 16) // 4


@pytest.mark.skipif(not (HAVE_QWEN and HAVE_TRANSFORMERS), reason="27B tokenizer or transformers absent")
def test_qwen_text_chat_render():
    qwen = dcb.QwenText(QWEN_DIR)
    assert qwen.separator_id == 248044
    text = qwen.chat([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Yo"}])
    assert text.startswith("<|im_start|>user\nHi<|im_end|>") and "<think>\n\n</think>" in text
    ids = qwen.encode(text)
    assert qwen.separator_id not in ids and max(ids) < dcb.TEXT_VOCAB_ROWS
    assert qwen.decode(ids) == text


@pytest.mark.skipif(not (HAVE_QWEN and HAVE_K3 and HAVE_CHIMERA), reason="K3 / 27B tokenizer / chimera tree absent")
def test_k3_row_round_trip():
    args = dcb.parse_args(["k3"])
    kcb = dcb.bind_k3_corpus_build(args)
    qwen = dcb.QwenText(QWEN_DIR)
    q_side, k_side, _ = kcb.load_sides(qwen_dir=QWEN_DIR, k3_dir=K3_DIR, qwen_separator_id=qwen.separator_id, k3_separator_id=None, k3_pad_id=None)
    docs = ["def f(x):\n    return x * 2\n\nExplain f.", "今天的天气很好。\n\nThe weather is nice today."]
    ids = []
    for doc in docs:
        ids.extend(qwen.encode(doc) + [qwen.separator_id])
    padded = kcb.pad(ids, 256, qwen.separator_id)
    build = kcb.build_row(padded, qwen=q_side, k3=k_side, sequence_length=256, label="test", require_terminated=True)
    assert k_side.decode_bytes(build.k3_ids) == bytes(build.raw) == "".join(docs).encode("utf-8")
    assert build.separator_pairs == 2
    pairs = build.pairs
    assert all(q1 > q0 and k1 > k0 for (q0, k0), (q1, k1) in zip(pairs, pairs[1:]))


def synthetic_rows(n_per_route: dict[str, int], tokens: int) -> list[dict]:
    rows = []
    for route, n in n_per_route.items():
        for i in range(n):
            rid = f"{route}:{i}"
            rows.append({"rid": rid, "uid": rid, "route": route, "family": "text", "tok_raw": tokens, "tok_chat": tokens + 20, "tok_prompt": 50, "tok_response": tokens // 2, "sep_in_chat": False, "sep_in_raw": False, "max_id_chat": 1000, "instruction": "q", "response": "a", "dataset_id": "d", "training_status": "s"})
    return rows


def test_allocate_disjoint_and_sized(tmp_path):
    out = dcb.out_paths(tmp_path)
    rows = synthetic_rows({"text": 8000, "code": 4000, "math": 2500}, 1900)
    dcb.write_jsonl(out["rows"], rows)
    assignment = {r["rid"]: "train" for r in rows}
    dcb.write_json(out["carve"], {"rows_sha256": dcb.file_sha256(out["rows"]), "assignment": assignment}, seal_field="carve_sha256")
    allocation = dcb.allocate(dcb.parse_args(["allocate", "--out-dir", str(tmp_path)]))
    assert len(allocation["long"]["docs"]) == dcb.LONG_DOCS
    assert all(d["estimated_tokens"] <= dcb.LONG_SEQ - dcb.LONG_RESERVE for d in allocation["long"]["docs"])
    assert len(allocation["k3"]["rows"]) == dcb.K3_PASSES * dcb.K3_ROWS_PER_PASS
    assert allocation["text"]["tokens_available"] >= allocation["text"]["target"]
    assert allocation["selfgen"]["budget_tokens"] >= dcb.SELFGEN_BUDGET
    sets = [{r for d in allocation["long"]["docs"] for r in d["rows"]}, {r for b in allocation["k3"]["rows"] for r in b}, set(allocation["text"]["docs"]), set(allocation["selfgen"]["prompts"])]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not sets[i] & sets[j]


def test_allocate_refuses_stale_carve(tmp_path):
    out = dcb.out_paths(tmp_path)
    rows = synthetic_rows({"text": 10}, 100)
    dcb.write_jsonl(out["rows"], rows)
    dcb.write_json(out["carve"], {"rows_sha256": "0" * 64, "assignment": {}}, seal_field="carve_sha256")
    with pytest.raises(SystemExit):
        dcb.allocate(dcb.parse_args(["allocate", "--out-dir", str(tmp_path)]))


def test_verify_detects_tampering(tmp_path):
    shard = tmp_path / "text" / "text-00000.safetensors"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"abc")
    dcb.write_json(tmp_path / "text" / "manifest.json", {"shards": [{"path": str(shard), "sha256": dcb.file_sha256(shard)}]}, seal_field="manifest_sha256")
    assert dcb.verify(dcb.parse_args(["verify", "--out-dir", str(tmp_path)]))["ok"]
    shard.write_bytes(b"abd")
    with pytest.raises(SystemExit):
        dcb.verify(dcb.parse_args(["verify", "--out-dir", str(tmp_path)]))


def test_onpolicy_pack_pinned():
    pack = dcb.DEFAULT_ONPOLICY_PACK
    assert pack.is_file()
    assert dcb.file_sha256(pack) == dcb.ONPOLICY_PACK_SHA256
    assert len(json.loads(pack.read_text(encoding="utf-8"))["prompts"]) == 120
