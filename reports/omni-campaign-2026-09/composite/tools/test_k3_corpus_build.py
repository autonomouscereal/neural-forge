"""Tests for k3_corpus_build.py (run-book S1 row, tests (1)-(4) plus the
mechanics they rest on, plus one test per review finding).  CPU only.  Uses the
REAL tokenizers -- the 27B tokenizer.json and the K3 tiktoken.model -- on a
small synthetic manifest sealed by qf_corpus.write_manifest, so what is tested
is the actual byte behaviour of the two BPEs, not a stub of it.

Run on the server:

    cd $OC/composite/tools && /home/cereal/model-surgery-lab/venv/bin/python \
        -m pytest -o addopts="" -p no:cacheprovider test_k3_corpus_build.py -q

Paths come from K3_DIR / QWEN_DIR (defaults: the Lexar K3 copy and the 27B
parent under /home/cereal/models).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import k3_corpus_build as kcb  # noqa: E402

K3_DIR = Path(os.environ.get("K3_DIR", kcb.DEFAULT_K3_DIR))
QWEN_DIR = Path(os.environ.get("QWEN_DIR", kcb.DEFAULT_QWEN_DIR))

if not (K3_DIR / "tiktoken.model").is_file() or not (QWEN_DIR / "tokenizer.json").is_file():
    pytest.skip(
        f"real tokenizers required: {K3_DIR}/tiktoken.model and {QWEN_DIR}/tokenizer.json",
        allow_module_level=True,
    )

SEQ = 256
REVIEW_SEQ = 96
QWEN_SEP = 248044
K3_EOS = 163585
K3_PAD = 163839

DOCS = [
    "Hello world, this is a plain English document about pumps and valves.\n",
    "def page_count(n, page_size):\n\treturn (n + page_size - 1) // page_size\n\nprint(page_count(10, 3))\n",
    "Café — naïve façade; 日本語のテキストです。Emoji: 🚀🔥 and a tab\there.\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n",
    "Numbers 1234567890 3.14159 and   multiple   spaces\n\n\nand blank lines.\n",
    "The quick brown fox jumps over the lazy dog. " * 12 + "\n",
]


@pytest.fixture(scope="module")
def lanes():
    kcb.bind_lanes()
    return kcb


@pytest.fixture(scope="module")
def sides(lanes):
    qwen, k3, report = kcb.load_sides(qwen_dir=QWEN_DIR, k3_dir=K3_DIR, qwen_separator_id=QWEN_SEP, k3_separator_id=None)
    return qwen, k3, report


def _lone_byte_id(qwen: kcb.QwenSide, byte: bytes) -> int:
    """The Qwen id that spells exactly one raw byte (the byte alphabet)."""

    for token_id in range(qwen.vocab_size):
        if token_id in qwen.special_ids or token_id == qwen.separator_id:
            continue
        if qwen.token_bytes(token_id) == byte:
            return token_id
    raise AssertionError(f"no single-byte token for {byte!r}")


def _pack(doc_ids: list[list[int]], terminated_last: bool = True) -> list[int]:
    ids: list[int] = []
    for index, doc in enumerate(doc_ids):
        ids.extend(doc)
        if index < len(doc_ids) - 1 or terminated_last:
            ids.append(QWEN_SEP)
    assert len(ids) <= SEQ, len(ids)
    return ids + [QWEN_SEP] * (SEQ - len(ids))


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory, lanes, sides):
    """A sealed 6-row manifest (4 fit / 2 held) plus fake bank files."""

    qwen, k3, _ = sides
    root = tmp_path_factory.mktemp("k3corpus")
    enc = qwen.encode
    partial = _lone_byte_id(qwen, b"\xe6")  # first byte of a 3-byte CJK character
    rows_ids = [
        _pack([enc(DOCS[0]), enc(DOCS[1])]),
        _pack([enc(DOCS[2])]),
        _pack([enc(DOCS[3]), enc(DOCS[4])]),                     # held
        _pack([enc("abc 日本") + [partial]]),                     # doc cut mid-character
        _pack([enc(DOCS[5])]),
        _pack([enc(DOCS[0]), enc(DOCS[3]), enc(DOCS[4])]),       # held
    ]
    rows = [
        {"batch_index": i, "split": "held" if i in (2, 5) else "fit", "ids": ids}
        for i, ids in enumerate(rows_ids)
    ]
    manifest_path = root / "evidence.json"
    kcb.qf_corpus.write_manifest(
        rows,
        tokenizer_sha256=kcb.file_sha256(QWEN_DIR / "tokenizer.json"),
        source="test_k3_corpus_build fixture",
        out=manifest_path,
    )
    bank_v1 = {
        "schema": "neural-forge-whole-response-review-prompts-v1",
        "prompts": [
            {"id": "conv-greeting", "category": "conversation", "messages": [{"role": "user", "content": "Hi! Ask me one follow-up question."}]},
            {"id": "sys-user", "category": "instruction", "messages": [{"role": "system", "content": "Answer in one sentence."}, {"role": "user", "content": "Why does bread rise?"}]},
        ],
    }
    bank_v2 = {
        "schema": "neural-forge-omni-review-bank-v2-draft",
        "prompts": [
            {"id": "code-A", "category": "code", "lane": "code-produce", "modality_in": "text", "modality_out": "text", "prompt_tokens_target": 512,
             "messages": [{"role": "user", "content": "Write run_length(s) with no imports."}]},
            {"id": "lc-8k-A", "category": "long-context", "lane": "long-context-retrieval", "modality_in": "text", "modality_out": "text", "prompt_tokens_target": 8192,
             "haystack_spec": {"generator": "oc-filler-v1", "seed": 1, "sentences": 400, "needles": [], "haystack_sha256": "0" * 64},
             "messages": [{"role": "user", "content": "DOCUMENT:\n<<HAYSTACK>>\n\nQuestion: which gate?"}]},
            {"id": "vis-A", "category": "vision", "lane": "vision-ocr", "modality_in": "text+image", "modality_out": "text", "prompt_tokens_target": 512,
             "messages": [{"role": "user", "content": [{"type": "image", "asset_id": "x"}, {"type": "text", "text": "Read the total."}]}]},
            {"id": "img-gen-B", "category": "media-out", "lane": "media-out-image", "modality_in": "text", "modality_out": "image", "prompt_tokens_target": 512,
             "messages": [{"role": "user", "content": "Draw a shelf."}]},
        ],
    }
    probe = {"schema": "neural-forge-matrix-prompt-set-v1", "kind": "free-run-probe", "prompts": ["The maintenance log for the north pump station reads as follows:"]}
    smoke = {"schema": "neural-forge-matrix-prompt-set-v1", "kind": "decode-smoke", "prompts": ["Explain in plain language what a circuit breaker does."]}
    for name, payload in (("bank_v1.json", bank_v1), ("bank_v2.json", bank_v2), ("probe.json", probe), ("smoke.json", smoke)):
        (root / name).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return root


def _args(root: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--evidence", str(root / "evidence.json"),
        "--qwen-dir", str(QWEN_DIR),
        "--k3-dir", str(K3_DIR),
        "--out-dir", str(out),
        "--append-bank-rows",
        "--review-sequence-length", str(REVIEW_SEQ),
        "--bank-v1", str(root / "bank_v1.json"),
        "--bank-v2", str(root / "bank_v2.json"),
        "--probe", str(root / "probe.json"),
        "--smoke", str(root / "smoke.json"),
        "--bank-render", "plain",
        *extra,
    ]


def _load(out: Path) -> dict:
    from safetensors.numpy import load_file

    row_meta = json.loads((out / "row_meta.json").read_text(encoding="utf-8"))
    boundary = json.loads((out / "boundary_map.json").read_text(encoding="utf-8"))
    tensors = {name: load_file(str(out / f"{name}.safetensors")) for name in row_meta["files"]}
    return {"out": out, "row_meta": row_meta, "boundary": boundary, "tensors": tensors}


def _rows(built: dict, file: str) -> list[dict]:
    return [m for m in built["row_meta"]["row_list"] if m["file"] == file]


@pytest.fixture(scope="module")
def built(fixture_dir):
    out = fixture_dir / "out_plain"
    receipt = kcb.build_corpus(kcb.parse_args(_args(fixture_dir, out)))
    return {"receipt": receipt, **_load(out)}


# --------------------------------------------------------------------------
# (1) tokenizer-sha refusal
# --------------------------------------------------------------------------


def test_refuses_manifest_tokenized_by_another_tokenizer(fixture_dir, lanes, tmp_path):
    body = json.loads((fixture_dir / "evidence.json").read_text(encoding="utf-8"))
    foreign = tmp_path / "foreign.json"
    kcb.qf_corpus.write_manifest(
        body["rows"],
        tokenizer_sha256="5f9e4d49" + "0" * 56,  # the 4B tokenizer's prefix: the wrong manifest
        source="foreign",
        out=foreign,
    )
    argv = _args(fixture_dir, tmp_path / "out") + ["--evidence", str(foreign), "--k3-bank-render", "none"]
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(argv))
    assert "was tokenized by 5f9e4d49" in str(refusal.value)
    assert "4B evidence" in str(refusal.value)
    assert not (tmp_path / "out").exists()
    receipt = kcb.build_corpus(kcb.parse_args(argv + ["--allow-tokenizer-mismatch"]))
    assert receipt["tokenizer_check"]["matches"] is False
    assert receipt["tokenizer_check"]["allow_mismatch"] is True


def test_refuses_unpinned_parent_and_k3_tokenizer(fixture_dir, lanes, tmp_path, monkeypatch):
    argv = _args(fixture_dir, tmp_path / "out", "--k3-bank-render", "none")
    monkeypatch.setattr(kcb, "EXPECTED_QWEN_TOKENIZER_SHA256", "f" * 64)
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(argv))
    assert "pinned 27B tokenizer" in str(refusal.value)
    monkeypatch.setattr(kcb, "EXPECTED_QWEN_TOKENIZER_SHA256", kcb.file_sha256(QWEN_DIR / "tokenizer.json"))
    monkeypatch.setattr(kcb, "EXPECTED_K3_TIKTOKEN_SHA256", "e" * 64)
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(argv))
    assert "pinned K3 rank file" in str(refusal.value)
    receipt = kcb.build_corpus(kcb.parse_args(argv + ["--allow-unpinned-k3-tokenizer"]))
    assert receipt["tokenizer_check"]["k3_pinned"] is False and receipt["tokenizer_check"]["k3_allow_unpinned"] is True


def test_refuses_share_reads_by_default(fixture_dir, lanes, tmp_path):
    argv = _args(fixture_dir, tmp_path / "out") + ["--evidence", "/mnt/ai_data/anything/evidence.json"]
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(argv))
    assert "share reads are OFF" in str(refusal.value)


# --------------------------------------------------------------------------
# (2) K3 detokenize == raw text bytes for every row
# --------------------------------------------------------------------------


def _qwen_raw_bytes(qwen_row: list[int], qwen: kcb.QwenSide) -> bytes:
    """Independent reconstruction: tokenizers' own decode of each id (specials
    decode to their literal names), separators contribute nothing."""

    hf = qwen.base._tokenizer
    byte_decoder = {char: byte for byte, char in kcb.ctok.bytes_to_unicode().items()}
    out = bytearray()
    for token_id in qwen_row:
        token_id = int(token_id)
        if token_id == QWEN_SEP:
            continue
        piece = hf.id_to_token(token_id)
        if token_id in qwen.special_ids:
            out += piece.encode("utf-8")
        else:
            out += bytes(byte_decoder[c] for c in piece)
    return bytes(out)


def _paired(built: dict):
    """(meta, k3_row, qwen_row) for every row of the two paired files."""

    for k3_file, qwen_file in ((kcb.FILE_EVIDENCE, kcb.FILE_EVIDENCE_QWEN), (kcb.FILE_REVIEW, kcb.FILE_REVIEW_QWEN)):
        k3_ids = built["tensors"][k3_file]["input_ids"]
        qwen_ids = built["tensors"][qwen_file]["input_ids"]
        rows = _rows(built, k3_file)
        assert len(rows) == k3_ids.shape[0] == qwen_ids.shape[0]
        for meta in rows:
            yield meta, k3_ids[meta["row_index"]], qwen_ids[meta["row_index"]]


def test_round_trip_every_row(built, sides):
    qwen, k3, _ = sides
    checked = 0
    for meta, k3_row, qwen_row in _paired(built):
        real = k3_row[: meta["k3_tokens_real"]].tolist()
        decoded = k3.decode_bytes(real)
        expected = _qwen_raw_bytes(qwen_row[: meta["qwen_tokens_real"]].tolist(), qwen)
        if meta["dropped_trailing_partial_utf8_bytes"]:
            expected = expected[: len(expected) - meta["dropped_trailing_partial_utf8_bytes"]]
        assert decoded == expected, (meta["file"], meta["row_index"])
        assert hashlib.sha256(decoded).hexdigest() == meta["raw_text_sha256"]
        assert meta["k3_covered_bytes"] == len(decoded)
        if meta["source"] != "evidence":
            assert decoded == meta["rendered_text"].encode("utf-8")
        # tiktoken's own decoder agrees with the per-token bytes
        assert k3.encoding.decode_bytes([i for i in real if i not in k3.special_ids]) == decoded
        checked += 1
    assert checked == 6 + 5
    assert all(m["k3_round_trip_exact"] for m in built["row_meta"]["row_list"])


def test_partial_trailing_utf8_byte_is_dropped_and_counted(built):
    row = _rows(built, kcb.FILE_EVIDENCE)[3]
    assert row["dropped_trailing_partial_utf8_bytes"] == 1
    assert row["raw_bytes_total"] == row["k3_covered_bytes"] + 1
    assert row["segments"] == row["segments_kept"] == 1
    assert built["row_meta"]["counts"]["dropped_trailing_partial_utf8_bytes_total"] == 1


# --------------------------------------------------------------------------
# (3) boundary map symmetric and monotone
# --------------------------------------------------------------------------


def _end_offsets(row: list[int], token_bytes) -> dict[int, int]:
    """token index -> end byte offset (separators are zero-width boundaries)."""

    offsets = {}
    consumed = 0
    for index, token in enumerate(row):
        piece = token_bytes(int(token))
        consumed += len(piece or b"")
        offsets[index] = consumed
    return offsets


def test_boundary_map_symmetric_and_monotone(built, sides):
    qwen, k3, _ = sides
    entries = {(e["file"], e["row_index"]): e for e in built["boundary"]["rows"]}
    assert len(entries) == 11  # the two paired files; the native file has no Qwen side
    total_pairs = 0
    for meta, k3_row, qwen_row in _paired(built):
        entry = entries[(meta["file"], meta["row_index"])]
        pairs = [tuple(p) for p in entry["pairs"]]
        assert entry["aligned_pairs"] == len(pairs) > 0
        q_positions = [q for q, _ in pairs]
        k_positions = [k for _, k in pairs]
        # strictly monotone on both axes
        assert all(b > a for a, b in zip(q_positions, q_positions[1:]))
        assert all(b > a for a, b in zip(k_positions, k_positions[1:]))
        # symmetric: q->k and k->q are inverse bijections
        q_to_k = dict(pairs)
        k_to_q = {k: q for q, k in pairs}
        assert len(q_to_k) == len(k_to_q) == len(pairs)
        assert all(k_to_q[q_to_k[q]] == q for q in q_to_k)
        # inside the real tokens
        assert max(q_positions) < meta["qwen_tokens_real"] and max(k_positions) < meta["k3_tokens_real"]
        # every pair sits on a common byte boundary, recomputed here from the ids
        q_off = _end_offsets(qwen_row.tolist(), qwen.token_bytes)
        k_off = _end_offsets(k3_row.tolist(), k3.token_bytes)
        for q, k in pairs:
            assert q_off[q] == k_off[k], (meta["file"], meta["row_index"], q, k)
        # a boundary common to both sides is never missed (completeness)
        q_boundaries = {q_off[i] for i in range(meta["qwen_tokens_real"]) if int(qwen_row[i]) != QWEN_SEP}
        k_boundaries = {k_off[i] for i in range(meta["k3_tokens_real"]) if int(k3_row[i]) != k3.separator_id}
        assert len(q_boundaries & k_boundaries) == len(pairs) - entry["separator_pairs"]
        assert entry["aligned_fraction"] == pytest.approx(len(pairs) / max(meta["qwen_tokens_real"], meta["k3_tokens_real"]))
        assert 0 < entry["aligned_fraction"] <= 1
        total_pairs += len(pairs)
    aggregate = built["boundary"]["aggregate"]
    assert aggregate["all"]["aligned_pairs"] == total_pairs
    assert aggregate["evidence"]["rows"] == 6 and aggregate["review"]["rows"] == 5
    assert aggregate["evidence_fit"]["rows"] == 4 and aggregate["evidence_held"]["rows"] == 2


def test_separators_pair_when_both_sides_close_the_same_document(built):
    row0 = built["boundary"]["rows"][0]
    assert row0["file"] == kcb.FILE_EVIDENCE and row0["row_index"] == 0
    assert row0["separator_pairs"] == 2  # two documents in the row
    meta0 = _rows(built, kcb.FILE_EVIDENCE)[0]
    assert row0["pairs"][-1] == [meta0["qwen_tokens_real"] - 1, meta0["k3_tokens_real"] - 1]


def test_align_positions_unit():
    pairs = kcb.align_positions([(0, 2), (1, 5), (2, 9)], [(0, 1), (1, 2), (2, 9), (3, 9)])
    assert pairs == [(0, 1), (2, 3)]  # the LAST zero-width token carries a shared boundary


# --------------------------------------------------------------------------
# (4) review rows never in the fit split -- and the split is IN-BAND (finding 1)
# --------------------------------------------------------------------------


def test_review_rows_never_in_fit_split(built, fixture_dir):
    rows = built["row_meta"]["row_list"]
    manifest = json.loads((fixture_dir / "evidence.json").read_text(encoding="utf-8"))
    declared = {int(r["batch_index"]): r["split"] for r in manifest["rows"]}
    for meta in rows:
        if meta["source"] == "evidence":
            assert meta["file"] == kcb.FILE_EVIDENCE
            assert meta["split"] == declared[meta["origin"]["batch_index"]]
            assert meta["is_review_prompt"] is False and meta["masked"] is False
        else:
            assert meta["file"] in (kcb.FILE_REVIEW, kcb.FILE_REVIEW_NATIVE)
            assert meta["split"] == "review"
            assert meta["is_review_prompt"] is True and meta["masked"] is True
    fit_rows = [m for m in rows if m["split"] == "fit"]
    assert len(fit_rows) == 4 and all(m["source"] == "evidence" for m in fit_rows)
    counts = built["row_meta"]["counts"]
    assert counts["review_rows_in_fit_split"] == 0 and counts["review_rows_in_evidence_file"] == 0
    assert counts["rows_by_split"] == {"fit": 4, "held": 2, "review": 5 + 4}
    assert counts["rows_by_source"] == {"bank_v1": 2 + 2, "bank_v2": 1 + 1, "evidence": 6, "probe": 1, "smoke": 1 + 1}


def test_finding1_split_is_in_band_and_fit_mask_counts_fit_tokens(built):
    """Finding 1: a consumer that reads only input_ids and deals positions by
    index would fit on review rows and padding; the split, real lengths and
    fit/held masks travel IN the safetensors file beside input_ids."""

    receipt = built["receipt"]
    for name, tensors in built["tensors"].items():
        assert set(tensors) == set(kcb.IN_BAND_TENSORS), name
        rows, seq = tensors["input_ids"].shape
        assert tensors["split"].shape == (rows,) and tensors["split"].dtype == np.int8
        assert tensors["real_lengths"].shape == (rows,) and tensors["real_lengths"].dtype == np.int64
        assert tensors["fit_mask"].shape == tensors["held_mask"].shape == (rows, seq)
        assert tensors["fit_mask"].dtype == tensors["held_mask"].dtype == np.bool_
        position = np.arange(seq)[None, :]
        real = position < tensors["real_lengths"][:, None]
        assert (tensors["fit_mask"] == (real & (tensors["split"] == 0)[:, None])).all()
        assert (tensors["held_mask"] == (real & (tensors["split"] == 1)[:, None])).all()
        assert not (tensors["fit_mask"] & tensors["held_mask"]).any()
        # the receipt names every in-band tensor of every file
        assert set(receipt["outputs"][name]["tensors"]) == set(kcb.IN_BAND_TENSORS)
        assert receipt["outputs"][name]["fit_mask_true"] == int(tensors["fit_mask"].sum())
    # (4) in-band: fit_mask.sum() == real K3 tokens of the fit rows, no more
    evidence = built["tensors"][kcb.FILE_EVIDENCE]
    metas = _rows(built, kcb.FILE_EVIDENCE)
    fit_tokens = sum(m["k3_tokens_real"] for m in metas if m["split"] == "fit")
    held_tokens = sum(m["k3_tokens_real"] for m in metas if m["split"] == "held")
    assert int(evidence["fit_mask"].sum()) == fit_tokens == built["row_meta"]["counts"]["k3_fit_tokens"]
    assert int(evidence["held_mask"].sum()) == held_tokens == built["row_meta"]["counts"]["k3_held_tokens"]
    assert evidence["split"].tolist() == [0, 0, 1, 0, 0, 1]
    assert evidence["real_lengths"].tolist() == [m["k3_tokens_real"] for m in metas]
    # padding positions are outside every mask
    for meta, row_fit in zip(metas, evidence["fit_mask"]):
        assert not row_fit[meta["k3_tokens_real"]:].any()
    # the review files carry split 2 everywhere and an all-False fit_mask
    for name in (kcb.FILE_REVIEW, kcb.FILE_REVIEW_QWEN, kcb.FILE_REVIEW_NATIVE):
        review = built["tensors"][name]
        assert (review["split"] == 2).all()
        assert not review["fit_mask"].any() and not review["held_mask"].any()
        assert receipt["outputs"][name]["fit_mask_true"] == 0
    # split codes are recorded in the file metadata and the receipt
    assert receipt["split_codes"] == {"fit": 0, "held": 1, "review": 2}


# --------------------------------------------------------------------------
# finding 2: the review surface is a separate SHORT file with its own T
# --------------------------------------------------------------------------


def test_finding2_review_surface_is_a_separate_short_file(built, sides):
    qwen, k3, _ = sides
    tensors = built["tensors"]
    assert tensors[kcb.FILE_EVIDENCE]["input_ids"].shape == (6, SEQ)
    assert tensors[kcb.FILE_EVIDENCE_QWEN]["input_ids"].shape == (6, SEQ)
    assert tensors[kcb.FILE_REVIEW]["input_ids"].shape == (5, REVIEW_SEQ)
    assert tensors[kcb.FILE_REVIEW_QWEN]["input_ids"].shape == (5, REVIEW_SEQ)
    assert tensors[kcb.FILE_REVIEW_NATIVE]["input_ids"].shape == (4, REVIEW_SEQ)
    assert all(m["k3_tokens_real"] <= REVIEW_SEQ for m in built["row_meta"]["row_list"] if m["split"] == "review")
    # no review row in the evidence file, no evidence row in a review file
    assert all(m["split"] in ("fit", "held") for m in _rows(built, kcb.FILE_EVIDENCE))
    assert all(m["split"] == "review" for m in _rows(built, kcb.FILE_REVIEW) + _rows(built, kcb.FILE_REVIEW_NATIVE))
    # both T's and the per-file positions are recorded for the S2 pricing
    pricing = built["receipt"]["pricing"]
    assert pricing["files"][kcb.FILE_EVIDENCE]["sequence_length"] == SEQ
    assert pricing["files"][kcb.FILE_REVIEW]["sequence_length"] == REVIEW_SEQ
    assert pricing["files"][kcb.FILE_REVIEW_NATIVE]["sequence_length"] == REVIEW_SEQ
    assert pricing["evidence_positions"] == 6 * SEQ
    assert pricing["review_positions"] == (5 + 4) * REVIEW_SEQ
    assert pricing["positions_total"] == 6 * SEQ + 9 * REVIEW_SEQ
    for name, entry in pricing["files"].items():
        assert entry["positions"] == entry["rows"] * entry["sequence_length"]
        assert entry["k3_padding_tokens"] == entry["positions"] - entry["k3_tokens_real"]
    assert built["receipt"]["counts"]["review_sequence_length"] == REVIEW_SEQ
    assert built["row_meta"]["files"][kcb.FILE_REVIEW]["role"] == "review (masked)"
    # the review files sit in the boundary map by (file, row_index) and the sha table names every ids file
    assert set(built["boundary"]["ids_files_sha256"]) == set(tensors)


def test_finding2_overlong_is_judged_against_the_review_length(built):
    skipped = {(s["source"], s["id"]): s["reason"] for s in built["row_meta"]["skipped_review_prompts"]}
    assert f"prompt_tokens_target 8192 > review sequence_length {REVIEW_SEQ}" in skipped[("bank_v2", "lc-8k-A")]
    assert "not a text case" in skipped[("bank_v2", "vis-A")]
    assert "media artifact" in skipped[("bank_v2", "img-gen-B")]
    assert len(skipped) == 3


# --------------------------------------------------------------------------
# finding 3: the K3-native review set and the render statements
# --------------------------------------------------------------------------


def test_finding3_k3_native_review_rows(built, sides):
    qwen, k3, _ = sides
    receipt = built["receipt"]
    native = built["tensors"][kcb.FILE_REVIEW_NATIVE]
    rows = _rows(built, kcb.FILE_REVIEW_NATIVE)
    assert len(rows) == 4 and native["input_ids"].shape[0] == 4
    assert {m["source"] for m in rows} == {"bank_v1", "bank_v2", "smoke"}  # every CHAT prompt, no raw probe
    reserved = set(k3.special_ids)
    for meta, row in zip(rows, native["input_ids"]):
        real = row[: meta["k3_tokens_real"]].tolist()
        specials = [i for i in real if i in reserved]
        assert specials, "a K3-native chat render carries K3's own specials"
        assert meta["k3_special_tokens"] == len(specials)
        assert sorted(set(specials)) == meta["k3_special_ids_present"]
        # the ids decode (through the campaign's own K3 encoding) to the rendered text
        assert k3.decode_with_specials(real).decode("utf-8") == meta["rendered_text"]
        assert hashlib.sha256(meta["rendered_text"].encode("utf-8")).hexdigest() == meta["raw_text_sha256"]
        assert meta["render"] == "k3-native" and meta["split"] == "review" and meta["masked"] is True
        assert "K3-native" in meta["k3_review_render"]
        assert (row[meta["k3_tokens_real"]:] == K3_PAD).all()
        # the Qwen chat markup never appears in a native render
        assert "<|im_start|>" not in meta["rendered_text"]
    # the paired review rows are Qwen/plain markup spelled as literal text and say so
    for meta in _rows(built, kcb.FILE_REVIEW):
        assert meta["k3_review_render"] == kcb.K3_REVIEW_RENDER_STATEMENTS[meta["render"]]
        assert "not K3-native" in meta["k3_review_render"] or meta["render"] == "raw"
        real = built["tensors"][kcb.FILE_REVIEW]["input_ids"][meta["row_index"]][: meta["k3_tokens_real"]].tolist()
        assert not any(i in reserved for i in real)
    statements = receipt["k3_review_render"]
    assert statements["paired_review_rows"] == kcb.K3_REVIEW_RENDER_STATEMENTS["plain"]
    assert statements["native_review_rows"] == kcb.K3_REVIEW_RENDER_STATEMENTS["k3-native"]
    assert statements["s2_bank_row_logits_are_k3_native_baseline"] is True
    assert receipt["k3_bank_render"]["thinking"] is False and receipt["k3_bank_render"]["tokenization_kimi_sha256"] == kcb.file_sha256(K3_DIR / "tokenization_kimi.py")
    not_rendered = {(s["source"], s["id"]): s["reason"] for s in receipt["native_not_rendered"]}
    assert "raw prompt" in not_rendered[("probe", "probe-0")]
    assert receipt["counts"]["review_native_rows"] == 4


def test_finding3_no_native_set_is_stated(fixture_dir):
    out = fixture_dir / "out_no_native"
    receipt = kcb.build_corpus(kcb.parse_args(_args(fixture_dir, out, "--k3-bank-render", "none")))
    assert not (out / f"{kcb.FILE_REVIEW_NATIVE}.safetensors").exists()
    assert kcb.FILE_REVIEW_NATIVE not in receipt["outputs"]
    assert receipt["k3_review_render"]["s2_bank_row_logits_are_k3_native_baseline"] is False
    assert "NOT a K3-native baseline" in receipt["k3_review_render"]["native_review_rows"]
    assert receipt["k3_bank_render"] == {"mode": "none", "statement": kcb.K3_REVIEW_RENDER_STATEMENTS["none"]}
    assert receipt["counts"]["review_native_rows"] == 0 and receipt["counts"]["review_rows"] == 5


# --------------------------------------------------------------------------
# finding 4: [EOS] separates documents, [PAD] pads rows, both recorded
# --------------------------------------------------------------------------


def test_finding4_pad_is_pad_and_eos_is_the_separator(built, sides):
    qwen, k3, report = sides
    assert k3.separator_id == K3_EOS and k3.pad_id == K3_PAD
    assert report["k3"]["separator_name"] == "[EOS]" and report["k3"]["pad_name"] == "[PAD]"
    for file in (kcb.FILE_EVIDENCE, kcb.FILE_REVIEW):
        ids = built["tensors"][file]["input_ids"]
        assert ids.dtype == np.int64 and int(ids.max()) < k3.vocab_size and int(ids.min()) >= 0
        for meta, row in zip(_rows(built, file), ids):
            real = row[: meta["k3_tokens_real"]]
            assert (row[meta["k3_tokens_real"]:] == K3_PAD).all()
            assert K3_PAD not in real.tolist()
            if meta["source"] == "evidence":
                assert real[-1] == K3_EOS  # last document closed
                assert meta["terminated"] is True
                assert list(real).count(K3_EOS) == meta["segments_kept"]
            else:
                assert real[-1] != K3_EOS  # a prompt awaiting its answer
                assert meta["terminated"] is False
    # the Qwen side keeps the manifest's own padding (its separator), stated as such
    qwen_ids = built["tensors"][kcb.FILE_EVIDENCE_QWEN]["input_ids"]
    for meta, row in zip(_rows(built, kcb.FILE_EVIDENCE), qwen_ids):
        assert (row[meta["qwen_tokens_real"]:] == QWEN_SEP).all()
    outputs = built["receipt"]["outputs"]
    assert outputs[kcb.FILE_EVIDENCE]["separator_id"] == K3_EOS and outputs[kcb.FILE_EVIDENCE]["pad_id"] == K3_PAD
    assert outputs[kcb.FILE_EVIDENCE_QWEN]["separator_id"] == outputs[kcb.FILE_EVIDENCE_QWEN]["pad_id"] == QWEN_SEP
    assert built["receipt"]["tokenizers"]["k3"]["pad_id"] == K3_PAD and built["receipt"]["tokenizers"]["k3"]["separator_id"] == K3_EOS
    # a --k3-pad-id outside the reserved band is refused
    with pytest.raises(SystemExit) as refusal:
        kcb.load_sides(qwen_dir=QWEN_DIR, k3_dir=K3_DIR, qwen_separator_id=QWEN_SEP, k3_separator_id=None, k3_pad_id=5)
    assert "pad id 5 is not a reserved special" in str(refusal.value)


# --------------------------------------------------------------------------
# finding 5: the default manifest is the spec path; boundary_map carries the seals
# --------------------------------------------------------------------------


def test_finding5_default_evidence_resolution(fixture_dir, tmp_path):
    spec = tmp_path / "composite" / "corpus" / "omni-evidence-27b-v1.json"
    lane = tmp_path / "qwen-fold" / "evidence" / "omni-evidence-27bu-v1.json"
    with pytest.raises(SystemExit) as refusal:
        kcb.resolve_evidence(None, spec=spec, lane=lane)
    assert "no evidence manifest" in str(refusal.value)
    lane.parent.mkdir(parents=True)
    shutil.copyfile(fixture_dir / "evidence.json", lane)
    path, how = kcb.resolve_evidence(None, spec=spec, lane=lane)
    assert path == lane and how["resolved_from"].startswith("lane-copy")
    spec.parent.mkdir(parents=True)
    shutil.copyfile(fixture_dir / "evidence.json", spec)
    path, how = kcb.resolve_evidence(None, spec=spec, lane=lane)
    assert path == spec and how["resolved_from"].startswith("spec-path")
    spec.write_text(spec.read_text(encoding="utf-8") + "\n", encoding="utf-8")  # differs by a byte
    with pytest.raises(SystemExit) as refusal:
        kcb.resolve_evidence(None, spec=spec, lane=lane)
    assert "differ" in str(refusal.value) and "--evidence" in str(refusal.value)
    path, how = kcb.resolve_evidence(str(lane), spec=spec, lane=lane)
    assert path == lane and how["resolved_from"] == "argument"
    assert kcb.DEFAULT_EVIDENCE_SPEC.name == "omni-evidence-27b-v1.json" and kcb.DEFAULT_EVIDENCE_SPEC.parent.name == "corpus"


def test_content_sha_is_deterministic_across_builds(built, fixture_dir):
    """The safetensors FILE sha can differ between two identical builds (the
    header's metadata map is serialized in hash order); the receipt therefore
    also carries the sha of the ids themselves, in the lane's fp32 convention."""

    out = fixture_dir / "out_plain_again"
    again = kcb.build_corpus(kcb.parse_args(_args(fixture_dir, out)))
    for name, tensors in built["tensors"].items():
        first, second = built["receipt"]["outputs"][name], again["outputs"][name]
        assert first["input_ids_sha256"] == second["input_ids_sha256"] == kcb.content_sha256(tensors["input_ids"])
        assert first["fit_mask_sha256"] == second["fit_mask_sha256"]
        assert first["input_ids_sha256"]["fp32"] == hashlib.sha256(tensors["input_ids"].astype(np.float32).tobytes()).hexdigest()
        assert built["boundary"]["ids_content_sha256"][name] == first["input_ids_sha256"]
    assert again["outputs"]["row_meta"]["row_meta_sha256"] == built["row_meta"]["row_meta_sha256"]
    again_boundary = json.loads((out / "boundary_map.json").read_text(encoding="utf-8"))
    assert again_boundary["ids_content_sha256"] == built["boundary"]["ids_content_sha256"]
    assert again_boundary["rows"] == built["boundary"]["rows"] and again_boundary["aggregate"] == built["boundary"]["aggregate"]


def test_finding5_boundary_map_carries_the_manifest_seal_and_ids_shas(built, fixture_dir):
    manifest = json.loads((fixture_dir / "evidence.json").read_text(encoding="utf-8"))
    boundary = built["boundary"]
    assert boundary["evidence_manifest_sha256"] == manifest["manifest_sha256"]
    assert boundary["evidence_manifest_file_sha256"] == kcb.file_sha256(fixture_dir / "evidence.json")
    for name, sha in boundary["ids_files_sha256"].items():
        assert kcb.file_sha256(built["out"] / f"{name}.safetensors") == sha == built["receipt"]["outputs"][name]["sha256"]
    assert set(boundary["ids_files_sha256"]) == set(built["tensors"])
    from safetensors import safe_open

    with safe_open(str(built["out"] / f"{kcb.FILE_EVIDENCE}.safetensors"), framework="np") as handle:
        metadata = handle.metadata()
    assert metadata["evidence_manifest_sha256"] == manifest["manifest_sha256"]
    assert metadata["in_band_tensors"] == ",".join(kcb.IN_BAND_TENSORS)
    assert built["receipt"]["evidence"]["resolution"]["resolved_from"] == "argument"


# --------------------------------------------------------------------------
# finding 6: the Qwen separator comes from the build receipt / generation_config, never a literal
# --------------------------------------------------------------------------


def test_finding6_qwen_separator_from_authorities_only(fixture_dir, tmp_path):
    evidence = tmp_path / "ev.json"
    shutil.copyfile(fixture_dir / "evidence.json", evidence)
    manifest = json.loads(evidence.read_text(encoding="utf-8"))
    # no sibling receipt: the parent's generation_config is the one authority
    separator, report = kcb.qwen_separator_from_sources(evidence, QWEN_DIR, manifest_sha256=manifest["manifest_sha256"])
    assert separator == QWEN_SEP and report["build_receipt"] is None
    assert report["source"] == "generation_config pad_token_id"
    # a sibling build receipt that agrees is recorded as a second authority
    sibling = tmp_path / "ev-build-receipt.json"
    sibling.write_text(json.dumps({"manifest": {"manifest_sha256": manifest["manifest_sha256"]}, "packing": {"padding_token_id": QWEN_SEP}, "tokenizer": {"separator_id": QWEN_SEP}}), encoding="utf-8")
    separator, report = kcb.qwen_separator_from_sources(evidence, QWEN_DIR, manifest_sha256=manifest["manifest_sha256"])
    assert separator == QWEN_SEP and report["build_receipt"]["packing.padding_token_id"] == QWEN_SEP
    assert set(report["candidates"]) == {"build receipt packing.padding_token_id", "build receipt tokenizer.separator_id", "generation_config pad_token_id"}
    # a sibling receipt that disagrees with generation_config is a refusal
    sibling.write_text(json.dumps({"manifest": {"manifest_sha256": manifest["manifest_sha256"]}, "packing": {"padding_token_id": 999}}), encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        kcb.qwen_separator_from_sources(evidence, QWEN_DIR, manifest_sha256=manifest["manifest_sha256"])
    assert "ambiguous" in str(refusal.value)
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(_args(fixture_dir, tmp_path / "out", "--evidence", str(evidence), "--k3-bank-render", "none")))
    assert "ambiguous" in str(refusal.value) and not (tmp_path / "out").exists()
    # a sibling receipt describing ANOTHER manifest is a refusal
    sibling.write_text(json.dumps({"manifest": {"manifest_sha256": "0" * 64}, "packing": {"padding_token_id": QWEN_SEP}}), encoding="utf-8")
    with pytest.raises(SystemExit) as refusal:
        kcb.qwen_separator_from_sources(evidence, QWEN_DIR, manifest_sha256=manifest["manifest_sha256"])
    assert "not this manifest's" in str(refusal.value)
    sibling.unlink()
    # no authority at all: refusal, no literal default
    bare = tmp_path / "bare_parent"
    bare.mkdir()
    with pytest.raises(SystemExit) as refusal:
        kcb.qwen_separator_from_sources(evidence, bare, manifest_sha256=manifest["manifest_sha256"])
    assert "no authority" in str(refusal.value) and "--qwen-separator-id" in str(refusal.value)
    separator, report = kcb.qwen_separator_from_sources(evidence, bare, manifest_sha256=manifest["manifest_sha256"], override=QWEN_SEP)
    assert separator == QWEN_SEP and report["source"] == "--qwen-separator-id"
    # no literal separator id anywhere in the build path
    import inspect

    assert "248044" not in inspect.getsource(kcb.build_corpus)
    assert "248044" not in inspect.getsource(kcb.qwen_separator_from_sources)
    # no workstation path in the chimera src candidates; env / argument only
    assert all(not str(c).startswith("C:") for c in kcb.CHIMERA_SRC_CANDIDATES)
    assert len(kcb.CHIMERA_SRC_CANDIDATES) == 1 and kcb.CHIMERA_SRC_ENV == "CHIMERA_OMNI_SRC"
    assert kcb.resolve_chimera_src("/x/src") == Path("/x/src")


def test_finding6_separator_source_recorded(built):
    sources = built["receipt"]["tokenizers"]["qwen"]["separator_sources"]
    assert sources["source"] == "generation_config pad_token_id"
    assert sources["generation_config"]["pad_token_id"] == QWEN_SEP
    assert built["receipt"]["arguments"]["qwen_separator_id"] is None


# --------------------------------------------------------------------------
# finding 7: the share guard covers every input path
# --------------------------------------------------------------------------


def test_finding7_share_guard_covers_every_input(fixture_dir, tmp_path):
    for role, flag in (("bank_v2", "--bank-v2"), ("bank_v1", "--bank-v1"), ("probe", "--probe"), ("smoke", "--smoke"), ("calibration_dir", "--calibration-dir"), ("qwen_fold_dir", "--qwen-fold-dir"), ("chimera_src", "--chimera-src")):
        argv = _args(fixture_dir, tmp_path / "out") + [flag, f"/mnt/ai_data/anywhere/{role}"]
        with pytest.raises(SystemExit) as refusal:
            kcb.build_corpus(kcb.parse_args(argv))
        assert "share reads are OFF" in str(refusal.value) and f"{role}=/mnt/ai_data/anywhere/{role}" in str(refusal.value), role
        assert not (tmp_path / "out").exists()
    report = kcb.guard_share_reads({"evidence": Path("/tmp/e.json"), "bank_v2": Path("/mnt/ai_data/b.json"), "probe": None}, allow_share_read=True)
    assert report["inputs_on_share"] == ["bank_v2=/mnt/ai_data/b.json"] and set(report["guarded_inputs"]) == {"evidence", "bank_v2"}


def test_finding7_every_input_is_recorded_under_share_reads(built):
    share = built["receipt"]["share_reads"]
    assert share["inputs_on_share"] == [] and share["allow_share_read"] is False
    assert set(share["guarded_inputs"]) >= {"evidence", "qwen_dir", "k3_dir", "qwen_fold_dir", "chimera_src", "calibration_dir", "bank_v1", "bank_v2", "probe", "smoke"}
    assert share["guarded_inputs"]["k3_dir"] == str(K3_DIR)


# --------------------------------------------------------------------------
# finding 8: interior invalid UTF-8 and an unterminated evidence row are named refusals
# --------------------------------------------------------------------------


def test_finding8_interior_invalid_utf8_is_a_named_refusal(sides):
    qwen, k3, _ = sides
    bad = _lone_byte_id(qwen, b"\xff")
    prefix = qwen.encode("abc")
    ids = prefix + [bad] + qwen.encode("def") + [QWEN_SEP]
    with pytest.raises(SystemExit) as refusal:
        kcb.build_row(ids, qwen=qwen, k3=k3, sequence_length=64, label="evidence batch_index 7")
    message = str(refusal.value)
    assert "evidence batch_index 7" in message and "invalid UTF-8 inside document 0" in message
    assert "row byte offset 3" in message  # 'abc' is three bytes; the bad byte is the fourth
    # in a second document the offset counts the row's whole raw stream
    first = qwen.encode(DOCS[0])
    ids = first + [QWEN_SEP] + prefix + [bad] + [QWEN_SEP]
    with pytest.raises(SystemExit) as refusal:
        kcb.build_row(ids, qwen=qwen, k3=k3, sequence_length=128, label="evidence batch_index 8")
    assert f"row byte offset {len(DOCS[0].encode('utf-8')) + 3}" in str(refusal.value) and "document 1" in str(refusal.value)


def test_finding8_evidence_rows_must_be_terminated(fixture_dir, sides, lanes, tmp_path):
    qwen, k3, _ = sides
    open_doc = qwen.encode(DOCS[4])
    with pytest.raises(SystemExit) as refusal:
        kcb.build_row(open_doc, qwen=qwen, k3=k3, sequence_length=64, label="evidence batch_index 3", require_terminated=True)
    assert "evidence batch_index 3" in str(refusal.value) and "not closed by separator" in str(refusal.value)
    build = kcb.build_row(open_doc, qwen=qwen, k3=k3, sequence_length=64, label="review x")
    assert build.terminated is False  # a prompt row: allowed, and the fact is carried
    build = kcb.build_row(open_doc + [QWEN_SEP], qwen=qwen, k3=k3, sequence_length=64, label="evidence batch_index 3", require_terminated=True)
    assert build.terminated is True
    # end to end: a manifest whose last document is cut open is refused before anything is written
    body = json.loads((fixture_dir / "evidence.json").read_text(encoding="utf-8"))
    rows = [dict(r) for r in body["rows"]]
    # the packer pads with its separator, so an open document can only be one that
    # fills the row to its last position with no separator after it
    filling = qwen.encode(DOCS[5] * 3)[:SEQ]
    assert len(filling) == SEQ and QWEN_SEP not in filling
    rows[1] = {**rows[1], "ids": filling}
    broken = tmp_path / "unterminated.json"
    kcb.qf_corpus.write_manifest(rows, tokenizer_sha256=body["tokenizer_sha256"], source="unterminated", out=broken)
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(_args(fixture_dir, tmp_path / "out", "--evidence", str(broken), "--k3-bank-render", "none")))
    assert "evidence batch_index 1" in str(refusal.value) and not (tmp_path / "out").exists()


# --------------------------------------------------------------------------
# truncation, receipt, overwrite, qwen-chat render
# --------------------------------------------------------------------------


def test_k3_truncation_follows_the_packer_rule(sides):
    qwen, k3, _ = sides
    long_doc = qwen.encode(DOCS[5] * 3)
    ids = long_doc + [QWEN_SEP]
    build = kcb.build_row(ids, qwen=qwen, k3=k3, sequence_length=40)
    assert build.truncated and build.k3_real == 40 and build.k3_ids[-1] == k3.separator_id
    assert build.k3_tokens_dropped > 0 and build.separator_pairs == 0
    assert all(k < 39 for _, k in build.pairs)
    assert k3.decode_bytes(build.k3_ids) == bytes(build.raw)
    # an unterminated prompt is cut to the row with no separator appended
    build = kcb.build_row(long_doc, qwen=qwen, k3=k3, sequence_length=40)
    assert build.truncated and build.k3_real == 40 and build.k3_ids[-1] != k3.separator_id


def test_receipt_seals_and_names_every_output(built):
    receipt = json.loads((built["out"] / "k3_corpus_build_receipt.json").read_text(encoding="utf-8"))
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    assert kcb.canonical_sha256(body) == receipt["receipt_sha256"] == built["receipt"]["receipt_sha256"]
    for name in (kcb.FILE_EVIDENCE, kcb.FILE_EVIDENCE_QWEN, kcb.FILE_REVIEW, kcb.FILE_REVIEW_QWEN, kcb.FILE_REVIEW_NATIVE, "row_meta", "boundary_map"):
        entry = receipt["outputs"][name]
        assert kcb.file_sha256(entry["path"]) == entry["sha256"]
    row_meta = built["row_meta"]
    assert kcb.canonical_sha256({k: v for k, v in row_meta.items() if k != "row_meta_sha256"}) == row_meta["row_meta_sha256"]
    boundary = built["boundary"]
    assert kcb.canonical_sha256({k: v for k, v in boundary.items() if k != "boundary_map_sha256"}) == boundary["boundary_map_sha256"]
    assert receipt["tokenizer_check"]["matches"] and receipt["tokenizer_check"]["pinned"] and receipt["tokenizer_check"]["k3_pinned"]
    assert receipt["share_reads"]["inputs_on_share"] == []
    assert receipt["counts"]["rows_by_file"] == {kcb.FILE_EVIDENCE: 6, kcb.FILE_REVIEW: 5, kcb.FILE_REVIEW_NATIVE: 4}


def test_refuses_to_overwrite_without_force(built, fixture_dir):
    with pytest.raises(SystemExit) as refusal:
        kcb.build_corpus(kcb.parse_args(_args(fixture_dir, built["out"])))
    assert "--force" in str(refusal.value)


def test_qwen_chat_render_thinking_off(fixture_dir):
    pytest.importorskip("transformers")
    out = fixture_dir / "out_chat"
    receipt = kcb.build_corpus(kcb.parse_args(_args(fixture_dir, out) + ["--bank-render", "qwen-chat", "--k3-bank-render", "none"]))
    assert receipt["bank_render"]["mode"] == "qwen-chat" and receipt["bank_render"]["enable_thinking"] is False
    row_meta = json.loads((out / "row_meta.json").read_text(encoding="utf-8"))
    chat_rows = [m for m in row_meta["row_list"] if m["source"] in ("bank_v1", "bank_v2", "smoke")]
    assert chat_rows and all(m["render"] == "qwen-chat" for m in chat_rows)
    assert all("<|im_start|>assistant" in m["rendered_text"] for m in chat_rows)
    assert all("<think>" not in m["rendered_text"].split("<|im_start|>assistant")[-1] or "</think>" in m["rendered_text"] for m in chat_rows)
    assert all(m["k3_review_render"] == kcb.K3_REVIEW_RENDER_STATEMENTS["qwen-chat"] for m in chat_rows)
    assert receipt["k3_review_render"]["paired_review_rows"] == kcb.K3_REVIEW_RENDER_STATEMENTS["qwen-chat"]
    probe_rows = [m for m in row_meta["row_list"] if m["source"] == "probe"]
    assert probe_rows and probe_rows[0]["render"] == "raw"
