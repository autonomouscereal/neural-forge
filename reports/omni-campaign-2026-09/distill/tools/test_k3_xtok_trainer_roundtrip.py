"""E5 round trip: the K3 cross-tokenizer target, from the REAL writers to one REAL training step.  CPU only.

    K3 ids + Qwen ids + boundary map, in the DISTILL corpus layout (distill_corpus_build 1.0.1: pairs as int32
    arrays in boundary_map.safetensors, the JSON rows carry no "file" / "pairs", the Qwen ids file states its
    padding as in-band `real_lengths` and has no `lengths`)
      -> k3_teacher_cache.py run    the real streamed teacher-forced pass, on the K3 lane's synthetic K3
      -> k3_teacher_cache.py xtok   the real cross-tokenizer writer (tokenizers: two tiny byte tables)
      -> qtext_teacher_cache.py build-text over the SAME Qwen ids file (the K3-row Q38 anchors)
      -> streamed_kd_trainer.py     load_data_config / build_batch / chunked_objective / train (one step)

Every expectation is computed independently, from the raw files and the byte tables (never through the
tools' own join code):

  test_xtok_target_is_the_independent_join             ids -1 for unmatched K3 tokens, logprobs bit-equal to
                                                       the cache shard, aligned boundaries only (both sides have
                                                       an in-row next token), tail = 1 - matched mass, and the
                                                       position convention (K3 next token == Qwen next token
                                                       wherever their bytes are equal)
  test_trainer_targets_anchor_and_padding              the trainer's batch: K3 term on aligned positions only,
                                                       Q38 anchor lambda*a / K3 lambda*(1-a), no target on the
                                                       separator padding the Q-text cache stored as real text
  test_a_cache_that_stored_padding_is_cut_to_real_lengths   a Q-text cache that stored the separator padding
                                                       as text (qtext <= 1.0.3 on these files) is cut back to the
                                                       corpus's real_lengths: no KL / CE / EOS on the padding
  test_objective_on_real_targets_matches_reference     chunked_objective == the trainer tests' independent loop
  test_one_training_step_with_the_k3_gain              one step of the real trainer: the K3 term is non-zero,
                                                       the gain starts at exactly 1.0 fp32 and moves (stays 1.0
                                                       with --freeze-k3-gain); the pre-registration pin is
                                                       recorded and a moved pre-registration is refused
  test_a_boundary_map_of_another_pass_is_refused       the K3 side of the join: every distill pass names its
                                                       file k3_input_ids, so only the file sha tells them apart
  test_two_pass_caches_with_one_file_name_are_refused  CacheReader never lets one pass overwrite another's rows
  test_the_real_distill_boundary_map_reads             (server) pass-0 of $OC/distill/corpus/k3 through the loader
  test_the_real_k3_vocab_map_is_injective              (server) the K3 -> Qwen byte map xtok wrote in parity

Server (needs the K3 engine dir, as test_k3_teacher_cache.py does):
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES= PYTHONPATH=$OC/distill/tools \\
      $PY -m pytest -o addopts="" -p no:cacheprovider test_k3_xtok_trainer_roundtrip.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k3_teacher_cache as ktc  # noqa: E402

ENGINE = ktc.Engine(os.environ.get("K3_ENGINE_DIR", ktc.DEFAULT_ENGINE_DIR))
import k3_common  # noqa: E402
import k3_synth  # noqa: E402
from safetensors.numpy import save_file as save_numpy  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

import streamed_kd_trainer as T  # noqa: E402
from test_streamed_kd_trainer import reference_objective, save_tiny_model  # noqa: E402

T.select_linear_attention_kernels("torch")

OC = Path(os.environ.get("OC", "/home/cereal/omni-campaign"))
PREREG = HERE.parent / "PREREGISTRATION_OMNI_STAGE_D_v1.md"
K3_VOCAB = 128          # the synthetic K3's vocabulary
QWEN_VOCAB = 300        # the tiny Qwen3.5 student / teacher
TEXT_ROWS = 290         # rows [290, 300) play the ledger / music bands
SEQ = 40                # both sides' row length (the trainer needs one seq_len per run)
ROWS = 6
LAMBDA, ANCHOR = 0.3, 0.5

# ------------------------------------------------------------------ the two byte tables
ALPHA = b"abcdefgh"
K3_SEP, K3_PAD, K3_SPECIAL, K3_UNDEFINED = 127, 125, 124, 126
Q_SEP = 280             # also the Qwen pad, as in the distill corpus (pad_id = separator)
Q_SPECIALS = set(range(281, 286))


def _tables() -> tuple[dict[int, bytes], dict[int, bytes]]:
    rng = random.Random(20261013)
    letters = [bytes([c]) for c in ALPHA]
    pairs = [a + b for a in letters for b in letters]
    triples = [a + b + c for a in letters for b in letters for c in letters]
    rng.shuffle(triples)
    k3 = {i: letters[i] for i in range(8)}
    k3.update({8 + i: p for i, p in enumerate(pairs)})                  # ids 8..71
    k3_triples = triples[:52]
    k3.update({72 + i: t for i, t in enumerate(k3_triples)})             # ids 72..123
    qwen = {100 + i: letters[7 - i] for i in range(8)}                   # another order
    order = list(range(64))
    rng.shuffle(order)
    qwen.update({10 + j: pairs[i] for j, i in enumerate(order)})        # ids 10..73
    shared = k3_triples[::2]                                             # half the K3 triples exist on Qwen
    qwen.update({150 + i: t for i, t in enumerate(shared + triples[52:72])})  # Qwen-only triples too
    quads = sorted({a + b for a in rng.sample(pairs, 12) for b in rng.sample(pairs, 2)})[:20]
    qwen.update({200 + i: q for i, q in enumerate(quads)})               # Qwen-only 4-byte tokens
    return k3, qwen


K3_TABLE, QWEN_TABLE = _tables()


class Side:
    """The tokenizer surface k3_teacher_cache uses (k3_corpus_build's QwenSide / K3Side): token_bytes,
    special_ids, separator_id, vocab_size; the separator spells no bytes (structural)."""

    def __init__(self, table: dict[int, bytes], specials: set[int], separator: int, vocab: int):
        self.table, self.special_ids, self.separator_id, self.vocab_size = table, frozenset(specials), separator, vocab

    def token_bytes(self, i: int):
        if i == self.separator_id:
            return None
        if i in self.special_ids:
            return b"<|special %d|>" % i
        if i not in self.table:
            raise SystemExit("undefined id")
        return self.table[i]

    def encode(self, raw: bytes) -> list[int]:
        """Greedy longest match (the test's own segmentation; only consistency with token_bytes matters)."""
        by_bytes = {v: k for k, v in self.table.items()}
        longest = max(len(v) for v in by_bytes)
        out, i = [], 0
        while i < len(raw):
            for n in range(min(longest, len(raw) - i), 0, -1):
                if raw[i:i + n] in by_bytes:
                    out.append(by_bytes[raw[i:i + n]])
                    i += n
                    break
            else:
                raise AssertionError(f"cannot encode {raw[i:i + 1]!r}")
        return out


K3_SIDE = Side(K3_TABLE, {K3_SPECIAL, K3_PAD}, K3_SEP, K3_VOCAB)
QWEN_SIDE = Side(QWEN_TABLE, Q_SPECIALS, Q_SEP, QWEN_VOCAB)


def independent_map() -> dict[int, int]:
    """K3 id -> Qwen id by identical bytes, straight from the two tables (-1 when none)."""
    reverse = {v: k for k, v in QWEN_TABLE.items()}
    out = {kid: reverse.get(raw, -1) for kid, raw in K3_TABLE.items()}
    out[K3_SEP] = Q_SEP
    for kid in (K3_PAD, K3_SPECIAL, K3_UNDEFINED):
        out[kid] = -1
    return out


# ------------------------------------------------------------------ the distill corpus layout
def build_pass(directory: Path, seed: int) -> dict:
    """One pass dir exactly as distill_corpus_build writes it (k3_input_ids / qwen_input_ids / boundary_map.*).
    Documents are random byte strings, each closed by the separator on both sides; pairs are where the
    consumed bytes end at the same offset (k3_corpus_build.align_positions), separators paired explicitly."""
    rng = random.Random(seed)
    directory.mkdir(parents=True, exist_ok=True)
    k3_rows, q_rows, k3_real, q_real, row_pairs, row_docs = [], [], [], [], [], []
    for _ in range(ROWS):
        k_ids, q_ids, pairs, offset, docs = [], [], [], 0, 0
        while True:
            raw = bytes(rng.choice(ALPHA) for _ in range(rng.randint(8, 30)))
            kd, qd = K3_SIDE.encode(raw), QWEN_SIDE.encode(raw)
            if len(k_ids) + len(kd) + 1 > SEQ or len(q_ids) + len(qd) + 1 > SEQ:
                break
            k_off, q_off, acc = {}, {}, 0
            for j, t in enumerate(kd):
                acc += len(K3_TABLE[t])
                k_off[offset + acc] = len(k_ids) + j
            acc = 0
            for j, t in enumerate(qd):
                acc += len(QWEN_TABLE[t])
                q_off[offset + acc] = len(q_ids) + j
            pairs += [(q_off[o], k_off[o]) for o in sorted(set(k_off) & set(q_off))]
            k_ids += kd + [K3_SEP]
            q_ids += qd + [Q_SEP]
            pairs.append((len(q_ids) - 1, len(k_ids) - 1))      # the separators close the same document
            offset += len(raw)
            docs += 1
        assert pairs and all(b[0] > a[0] and b[1] > a[1] for a, b in zip(pairs, pairs[1:]))
        row_docs.append(docs)
        k3_real.append(len(k_ids))
        q_real.append(len(q_ids))
        k3_rows.append(k_ids + [K3_PAD] * (SEQ - len(k_ids)))
        q_rows.append(q_ids + [Q_SEP] * (SEQ - len(q_ids)))
        row_pairs.append(pairs)
    assert any(n < SEQ for n in q_real), "the fixture must carry Qwen padding"
    files = {}
    for name, table, real, sep, pad in (("k3_input_ids", k3_rows, k3_real, K3_SEP, K3_PAD),
                                        ("qwen_input_ids", q_rows, q_real, Q_SEP, Q_SEP)):
        path = directory / f"{name}.safetensors"
        save_file({"input_ids": torch.tensor(table, dtype=torch.int64), "split": torch.zeros(ROWS, dtype=torch.int64),
                   "real_lengths": torch.tensor(real, dtype=torch.int64),
                   "fit_mask": torch.ones(ROWS, dtype=torch.int64), "held_mask": torch.zeros(ROWS, dtype=torch.int64)},
                  str(path), metadata={"file": name, "rows": str(ROWS), "sequence_length": str(SEQ),
                                       "separator_id": str(sep), "pad_id": str(pad)})
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / "row_meta.json").write_text(json.dumps({"schema": "neural-forge-omni-distill-corpus-v1/k3_row_meta", "pass": 0,
                                                         "rows": [{"row_index": r, "routes": ["text"] * n}
                                                                  for r, n in enumerate(row_docs)]}))
    offsets = np.zeros(ROWS + 1, dtype=np.int64)
    for r, p in enumerate(row_pairs):
        offsets[r + 1] = offsets[r] + len(p)
    flat = [(r, q, k) for r, p in enumerate(row_pairs) for q, k in p]
    bpath = directory / "boundary_map.safetensors"
    save_numpy({"row": np.asarray([f[0] for f in flat], dtype=np.int32),
                "qwen_position": np.asarray([f[1] for f in flat], dtype=np.int32),
                "k3_position": np.asarray([f[2] for f in flat], dtype=np.int32), "row_offsets": offsets},
               str(bpath), metadata={"schema": "neural-forge-omni-distill-corpus-v1/k3_boundary_map",
                                     "qwen_file_sha256": files["qwen_input_ids"]["sha256"],
                                     "k3_file_sha256": files["k3_input_ids"]["sha256"]})
    pairs_file = {"path": str(bpath), "sha256": hashlib.sha256(bpath.read_bytes()).hexdigest(), "pairs": len(flat)}
    rows = [{"row_index": r, "qwen_tokens_real": q_real[r], "k3_tokens_real": k3_real[r], "aligned_pairs": len(p),
             "separator_pairs": sum(1 for q, _ in p if q_rows[r][q] == Q_SEP),
             "aligned_fraction": len(p) / max(q_real[r], k3_real[r])} for r, p in enumerate(row_pairs)]
    bm = {"schema": "neural-forge-k3-boundary-map-v2", "distill_schema": "neural-forge-omni-distill-corpus-v1/k3",
          "pass": 0, "aggregate": {"rows": ROWS, "aligned_pairs": len(flat)}, "pairs_file": pairs_file,
          "ids_files_sha256": {n: f["sha256"] for n, f in files.items()}, "rows": rows}
    (directory / "boundary_map.json").write_text(json.dumps(bm, indent=1, sort_keys=True))
    return {"dir": directory, "files": files, "pairs": row_pairs, "k3_real": k3_real, "q_real": q_real,
            "k3_ids": torch.tensor(k3_rows), "q_ids": torch.tensor(q_rows), "boundary_map": directory / "boundary_map.json"}


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("k3_xtok_roundtrip")


@pytest.fixture(scope="module")
def synth(work: Path) -> dict:
    out = work / "synthetic-k3"
    argv = ["--out-dir", str(out), "--layers", "6", "--hidden", "64", "--experts", "8", "--top-k", "2",
            "--attn-res-block-size", "3", "--latent", "32", "--moe-intermediate", "64", "--vocab-size", str(K3_VOCAB)]
    k3_synth.build(k3_synth.build_parser().parse_args(argv))
    index = k3_common.K3Index(str(out))
    stat = ktc.source_stat(index, str(out))
    fold = work / "synthetic_fold_receipt.json"
    fold.write_text(json.dumps({"resume_context": {"model": {"source_file_metadata": stat["files"],
                                                             "config_sha256": stat["config_sha256"],
                                                             "index_sha256": stat["index_sha256"]}}}))
    return {"dir": str(out), "fold_receipt": str(fold)}


def run_cache(synth: dict, ids_file: str, out_root: Path) -> Path:
    argv = ["run", "--src-dir", synth["dir"], "--out-root", str(out_root), "--device", "cpu", "--experts-device", "cpu",
            "--compute-dtype", "fp32", "--kda-backend", "torch", "--fold-receipt", synth["fold_receipt"],
            "--digest-receipt", "", "--no-prefetch-next", "--min-free-gib", "0", "--token-chunk", "7",
            "--input-ids", ids_file]
    receipt = ktc.CachePass(ktc.build_parser().parse_args(argv)).run()
    assert receipt["verdict"] == "sealed"
    return Path(receipt["shards"][0]["path"]).parent.parent


@pytest.fixture(scope="module")
def pass0(work: Path, synth: dict) -> dict:
    corpus = build_pass(work / "corpus" / "pass-0", seed=1)
    corpus["cache"] = run_cache(synth, corpus["files"]["k3_input_ids"]["path"], work / "k3cache-0")
    return corpus


@pytest.fixture(scope="module")
def xtok(pass0: dict, work: Path) -> Path:
    out = work / "xtok-0"
    saved = ktc.bind_tokenizers
    ktc.bind_tokenizers = lambda args: (QWEN_SIDE, K3_SIDE, {"tokenizers": "test byte tables"})
    try:
        rc = ktc.main(["xtok", "--cache-dir", str(pass0["cache"]), "--boundary-map", str(pass0["boundary_map"]),
                       "--k3-ids", pass0["files"]["k3_input_ids"]["path"], "--qwen-ids", pass0["files"]["qwen_input_ids"]["path"],
                       "--k3-vocab", str(K3_VOCAB), "--out", str(out)])
    finally:
        ktc.bind_tokenizers = saved
    assert rc == 0
    return out / "xtok_target.safetensors"


@pytest.fixture(scope="module")
def teacher_dir(tmp_path_factory) -> Path:
    return save_tiny_model(tmp_path_factory.mktemp("rt_teacher") / "teacher", seed=23)


@pytest.fixture(scope="module")
def student_dir(tmp_path_factory) -> Path:
    return save_tiny_model(tmp_path_factory.mktemp("rt_student") / "student", seed=11)


@pytest.fixture(scope="module")
def qcache(pass0: dict, teacher_dir: Path, work: Path) -> Path:
    q = T.load_peer_module("qtext_teacher_cache")
    corpus = pass0["files"]["qwen_input_ids"]["path"]
    rc = q.main(["build-text", "--model-dir", str(teacher_dir), "--teacher", "tiny-t", "--corpus", corpus,
                 "--device", "cpu", "--dtype", "float32", "--linear-attn-kernel", "torch", "--top-k", "8",
                 "--vocab-rows", str(TEXT_ROWS), "--micro-batch", "2", "--rows-per-shard", "4",
                 "--head-chunk-rows", "64", "--out-root", str(work / "qtext"), "--teacher-digest", "full"])
    assert rc == 0
    return work / "qtext" / "tiny-t" / pass0["files"]["qwen_input_ids"]["sha256"]


@pytest.fixture(scope="module")
def data_config(qcache: Path, xtok: Path, work: Path) -> Path:
    path = work / "data_k3.json"
    path.write_text(json.dumps({"sources": [{"name": "k3rows", "kind": "k3", "qtext_cache": str(qcache), "xtok": str(xtok)}]}))
    return path


def shard_rows(cache_dir: Path) -> dict[int, dict[str, torch.Tensor]]:
    """The K3 cache's per-row arrays read straight from the shard files (not through CacheReader)."""
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    out = {}
    for shard in manifest["shards"]:
        got = load_file(shard["path"])
        pos = 0
        for i in range(got["real_lengths"].numel()):
            n = int(got["real_lengths"][i])
            out[int(got["row_index"][i])] = {k: got[k][pos:pos + n] for k in ("topk_ids", "topk_logprobs")}
            pos += n
    return out


def expected_targets(pass0: dict) -> dict[int, dict[int, tuple[list[int], torch.Tensor]]]:
    """row -> qwen position -> (mapped ids, K3 logprobs) at every aligned pair that predicts on both sides."""
    mapping = independent_map()
    rows = shard_rows(pass0["cache"])
    out: dict[int, dict[int, tuple[list[int], torch.Tensor]]] = {}
    for r, pairs in enumerate(pass0["pairs"]):
        for q, k in pairs:
            if q < pass0["q_real"][r] - 1 and k < pass0["k3_real"][r] - 1:
                ids = [mapping[int(i)] for i in rows[r]["topk_ids"][k]]
                out.setdefault(r, {})[q] = (ids, rows[r]["topk_logprobs"][k])
    return out


# ------------------------------------------------------------------ tests
def test_xtok_target_is_the_independent_join(pass0, xtok):
    got = load_file(str(xtok))
    want = expected_targets(pass0)
    n = sum(len(v) for v in want.values())
    assert int(got["q_pos"].numel()) == n > 0
    k3_rows = shard_rows(pass0["cache"])
    seen_unmatched = seen_matched = 0
    for i in range(n):
        r, q, k = int(got["q_row"][i]), int(got["q_pos"][i]), int(got["k3_pos"][i])
        ids, lp = want[r][q]
        assert got["qwen_ids"][i].tolist() == ids, (r, q)
        assert torch.equal(got["k3_logprobs"][i], lp), (r, q)                         # bf16, bit-equal
        assert (r, q, k) in {(r, a, b) for a, b in pass0["pairs"][r]}
        p = lp.float().exp()
        valid = torch.tensor(ids) >= 0
        assert math.isclose(float(got["matched_mass"][i]), float(p[valid].sum()), rel_tol=1e-6, abs_tol=1e-7)
        assert math.isclose(float(got["tail_mass"][i]), max(0.0, 1.0 - float(p[valid].sum())), abs_tol=1e-6)
        seen_unmatched += int((~valid).sum())
        seen_matched += int(valid.sum())
        # the position convention: K3 position k and Qwen position q both predict the token after the same
        # bytes, so where the two next tokens spell the same bytes, the map must take one to the other
        kn, qn = int(pass0["k3_ids"][r, k + 1]), int(pass0["q_ids"][r, q + 1])
        same = (kn == K3_SEP and qn == Q_SEP) or (K3_TABLE.get(kn) is not None and K3_TABLE.get(kn) == QWEN_TABLE.get(qn))
        assert int(got["next_token_identical"][i]) == int(same)
    assert seen_unmatched > 0 and seen_matched > 0          # both id kinds are exercised
    assert int(got["next_token_identical"].sum()) > 0
    assert int(got["qwen_ids"].max()) < TEXT_ROWS and int(got["qwen_ids"].min()) == -1
    receipt = json.loads((xtok.parent / "xtok_receipt.json").read_text())
    assert receipt["boundary_map"]["path"] == str(pass0["boundary_map"])
    assert receipt["boundary_map"]["pairs_file"]["sha256"] == hashlib.sha256(
        Path(pass0["boundary_map"]).with_suffix(".safetensors").read_bytes()).hexdigest()
    assert receipt["k3_join"]["k3_input_ids"]["verified"] is True
    assert got["k3_pos"].shape[0] == n and set(k3_rows) == set(range(ROWS))


def test_trainer_targets_anchor_and_padding(pass0, qcache, data_config):
    sources, identity = T.load_data_config(data_config)
    (source,) = sources
    assert identity["sources"][0]["xtok_provenance"]["method"] == "receipt_chain"
    assert identity["sources"][0]["length_source"] == "corpus real_lengths"
    padded = sum(1 for n in pass0["q_real"] if n < SEQ)
    # qtext <= 1.0.3 read only `lengths` and stored the padding (the trainer cuts every padded row); qtext 1.0.4
    # reads `real_lengths` itself (nothing left to cut).  Either way the trainer's rows end at real_lengths.
    assert padded > 0 and identity["sources"][0]["rows_cut_to_real_lengths"] in (0, padded)
    assert all(source.cache.index[r][3] == pass0["q_real"][r] for r in range(ROWS))
    objective = T.ObjectiveConfig(lambda_k3=LAMBDA, k3_q38_anchor=ANCHOR, text_vocab_rows=TEXT_ROWS, eos_ids=(Q_SEP,),
                                  head_chunk_positions=16)
    picks = [(0, r) for r in range(ROWS)]
    ids, tg = T.build_batch(sources, picks, objective)
    assert torch.equal(ids, pass0["q_ids"])
    want = expected_targets(pass0)
    k_ids, k_lp = tg["k_ids"].view(ROWS, SEQ, -1), tg["k_lp"].view(ROWS, SEQ, -1)
    k_w, q_w = tg["k_w"].view(ROWS, SEQ), tg["q_w"].view(ROWS, SEQ)
    ce_w, eos_w = tg["ce_w"].view(ROWS, SEQ), tg["eos_w"].view(ROWS, SEQ)
    for r in range(ROWS):
        real = pass0["q_real"][r]
        for p in range(SEQ):
            hit = want.get(r, {}).get(p)
            if hit is not None and any(i >= 0 for i in hit[0]):
                assert k_ids[r, p].tolist() == hit[0] and torch.equal(k_lp[r, p], hit[1].float())
                assert float(k_w[r, p]) == pytest.approx(LAMBDA * (1 - ANCHOR))
            else:
                assert float(k_w[r, p]) == 0.0, (r, p)
            # the Q38 anchor on every real position; nothing on the padding the cache stored as text
            assert float(q_w[r, p]) == pytest.approx(LAMBDA * ANCHOR if p < real else 0.0), (r, p, real)
            assert float(ce_w[r, p]) == pytest.approx(0.1 if p < real - 1 else 0.0)
            is_eos = p < real - 1 and int(pass0["q_ids"][r, p + 1]) == Q_SEP
            assert float(eos_w[r, p]) == pytest.approx(0.1 if is_eos else 0.0)
    assert float((k_w > 0).sum()) > 0


def test_a_cache_that_stored_padding_is_cut_to_real_lengths(pass0, teacher_dir, work):
    """The qtext <= 1.0.3 situation, rebuilt with qtext 1.0.4: the cache is told the rows are full width
    (`--lengths-key full_lengths`), so it stores the separator padding as text; the corpus file still states
    `real_lengths` and carries no `lengths`, so the trainer (>= 1.3.1) cuts every padded row back."""
    corpus = work / "padded" / "qwen_rows_full.safetensors"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": pass0["q_ids"].clone(), "real_lengths": torch.tensor(pass0["q_real"], dtype=torch.int64),
               "full_lengths": torch.full((ROWS,), SEQ, dtype=torch.int64)}, str(corpus))
    q = T.load_peer_module("qtext_teacher_cache")
    rc = q.main(["build-text", "--model-dir", str(teacher_dir), "--teacher", "tiny-t", "--corpus", str(corpus),
                 "--lengths-key", "full_lengths", "--device", "cpu", "--dtype", "float32", "--linear-attn-kernel", "torch",
                 "--top-k", "8", "--vocab-rows", str(TEXT_ROWS), "--micro-batch", "2", "--rows-per-shard", "4",
                 "--head-chunk-rows", "64", "--out-root", str(work / "padded" / "qtext"), "--teacher-digest", "full"])
    assert rc == 0
    cache = T.QTextCache(work / "padded" / "qtext" / "tiny-t" / hashlib.sha256(corpus.read_bytes()).hexdigest())
    padded = [r for r in range(ROWS) if pass0["q_real"][r] < SEQ]
    assert cache.length_source == "corpus real_lengths" and cache.rows_cut_to_real_lengths == len(padded) > 0
    objective = T.ObjectiveConfig(text_vocab_rows=TEXT_ROWS, eos_ids=(Q_SEP,), head_chunk_positions=16)
    src = [T.DataSource("t", "text", cache, row_ids=sorted(cache.index))]
    _, tg = T.build_batch(src, [(0, r) for r in padded], objective)
    for b, r in enumerate(padded):
        real = pass0["q_real"][r]
        row = cache.row(r)
        assert row["length"] == real and not bool(row["stored"][real:].any()) and bool(row["stored"][:real].all())
        assert not bool((tg["q_w"].view(len(padded), SEQ)[b, real:] > 0).any())
        assert not bool((tg["ce_w"].view(len(padded), SEQ)[b, real - 1:] > 0).any())
        assert not bool((tg["eos_w"].view(len(padded), SEQ)[b, real - 1:] > 0).any())   # no "<sep> after <sep>"


def test_objective_on_real_targets_matches_reference(data_config):
    sources, _ = T.load_data_config(data_config)
    objective = T.ObjectiveConfig(lambda_k3=LAMBDA, k3_q38_anchor=ANCHOR, text_vocab_rows=TEXT_ROWS, eos_ids=(Q_SEP,),
                                  head_chunk_positions=7)
    ids, tg = T.build_batch(sources, [(0, 1), (0, 4)], objective)
    g = torch.Generator().manual_seed(5)
    hidden = torch.randn(2, SEQ, 12, generator=g, dtype=torch.float64)
    head = torch.randn(TEXT_ROWS, 12, generator=g, dtype=torch.float64)
    denominator = 2 * (SEQ - 1)
    parts, grad = T.chunked_objective(hidden, head, tg, objective, denominator=denominator)
    ref = reference_objective(hidden, head, tg, objective, denominator)
    assert parts["k3"] > 0 and parts["k3_positions"] > 0
    assert math.isclose(parts["total"], float(ref), rel_tol=1e-9)
    leaf = hidden.clone().requires_grad_(True)
    reference_objective(leaf, head, tg, objective, denominator).backward()
    assert torch.allclose(grad, leaf.grad, rtol=1e-7, atol=1e-10)


def _train(student_dir, data, out, *extra):
    return T.main(["train", "--model", str(student_dir), "--devices", "cpu", "--dtype", "float32", "--kernels", "torch",
                   "--data-config", str(data), "--seq-len", str(SEQ), "--micro-batch", "2", "--total-steps", "1",
                   "--text-vocab-rows", str(TEXT_ROWS), "--trunk-ffn-width", "32", "--lora-rank", "4", "--k3-lora-rank", "2",
                   "--lr", "5e-3", "--gain-lr", "5e-3", "--warmup-steps", "1", "--eos-ids", str(Q_SEP),
                   "--head-chunk-positions", "16", "--out", str(out), "--safety-dir", str(out) + "-safety", *extra])


def test_one_training_step_with_the_k3_gain(student_dir, data_config, work):
    receipts = T.load_peer_module("stage2_training_receipt")
    # the gain is born at exactly 1.0, fp32, one value per K3 channel (inter 48 - trunk 32 = 16)
    config = T.text_config_from_source(T.SafetensorsSource(student_dir), torch.float32, "sdpa")
    plan = T.build_site_plan(config, T.SitePlanConfig(rank=4, alpha=4.0, k3_rank=2, k3_alpha=2.0, trunk_ffn_width=32))
    bank = T.AdapterBank(plan, [str(t) for t in config.layer_types], "cpu", 20261012)
    gains = {n: t for n, t in bank.tensors.items() if n.endswith(".gain")}
    assert len(gains) == config.num_hidden_layers
    assert all(t.dtype == torch.float32 and t.shape == (16,) and bool((t == 1.0).all()) and t.requires_grad for t in gains.values())

    assert _train(student_dir, data_config, work / "train", "--preregistration", str(PREREG)) == 0
    receipt = json.loads((work / "train" / "training-receipt.json").read_text())
    (step,) = receipt["steps"]
    assert step["applied"] and math.isfinite(step["loss"]) and step["k3"] > 0 and step["q38"] > 0
    assert receipt["objective"]["lambda_k3"] == LAMBDA and receipt["objective"]["k3_q38_anchor"] == ANCHOR
    assert receipt["preregistration"]["pinned"] is True
    assert receipt["preregistration"]["preregistration_sha256"] == receipts.STAGE2_PREREGISTRATION_SHA256
    assert receipt["data_identity"]["sources"][0]["xtok_provenance"]["structure"]["max_id"] < TEXT_ROWS
    trained = load_file(str(work / "train" / "adapters.safetensors"))
    moved = [n for n in gains if not torch.equal(trained[n], torch.ones(16))]
    assert all(trained[n].dtype == torch.float32 for n in gains) and moved          # the gain trains

    assert _train(student_dir, data_config, work / "frozen", "--freeze-k3-gain", "--preregistration", str(PREREG)) == 0
    frozen = load_file(str(work / "frozen" / "adapters.safetensors"))
    assert all(torch.equal(frozen[n], torch.ones(16)) for n in gains)               # G3 fallback: held at 1.0
    assert json.loads((work / "frozen" / "training-receipt.json").read_text())["steps"][0]["k3"] > 0

    moved_doc = work / PREREG.name
    text = PREREG.read_text(encoding="utf-8")
    moved_doc.write_text(text.replace("lambda_K3 = 0.3", "lambda_K3 = 0.4", 1), encoding="utf-8", newline="\n")
    assert _train(student_dir, data_config, work / "moved", "--preregistration", str(moved_doc)) == 2
    assert not (work / "moved" / "training-receipt.json").exists()
    # trainer 1.3.2: no document and no waiver on a weight-moving run is refused too
    assert _train(student_dir, data_config, work / "unpinned") == 2
    assert not (work / "unpinned" / "training-receipt.json").exists()


def test_a_boundary_map_of_another_pass_is_refused(pass0, work):
    other = build_pass(work / "corpus" / "pass-1", seed=2)        # same file names, other rows
    ktc.bind_tokenizers, saved = (lambda args: (QWEN_SIDE, K3_SIDE, {})), ktc.bind_tokenizers
    try:
        with pytest.raises(SystemExit, match="k3_input_ids"):
            ktc.main(["xtok", "--cache-dir", str(pass0["cache"]), "--boundary-map", str(other["boundary_map"]),
                      "--k3-vocab", str(K3_VOCAB), "--out", str(work / "xtok-mixed")])
    finally:
        ktc.bind_tokenizers = saved
    # and a pairs file that moved after the map was written
    bm = json.loads(Path(other["boundary_map"]).read_text())
    Path(bm["pairs_file"]["path"]).write_bytes(Path(bm["pairs_file"]["path"]).read_bytes() + b" ")
    with pytest.raises(SystemExit, match="sha256"):
        ktc.load_boundary_map(str(other["boundary_map"]))


def test_two_pass_caches_with_one_file_name_are_refused(pass0, synth, work):
    other = build_pass(work / "corpus" / "pass-1b", seed=3)
    cache1 = run_cache(synth, other["files"]["k3_input_ids"]["path"], work / "k3cache-1")
    with pytest.raises(SystemExit, match="k3_input_ids"):
        ktc.CacheReader([str(pass0["cache"]), str(cache1)])


REAL_PASS0 = OC / "distill" / "corpus" / "k3" / "pass-0" / "boundary_map.json"
REAL_MAP = OC / "distill" / "parity" / "k3cache-2026-10-12" / "rerun-1.2.0" / "xtok-review" / "k3_to_qwen_map.safetensors"


@pytest.mark.skipif(not REAL_PASS0.is_file(), reason="the distill K3 corpus is on the server only")
def test_the_real_distill_boundary_map_reads():
    doc = json.loads(REAL_PASS0.read_text())
    got = ktc.load_boundary_map(str(REAL_PASS0))
    assert got["format"] == "distill-safetensors"
    assert len(got["rows"]) == len(doc["rows"])
    assert sum(int(r["pairs"].shape[0]) for r in got["rows"]) == int(doc["aggregate"]["aligned_pairs"])
    for row in got["rows"]:
        assert row["file"] == "k3_input_ids"
        p = row["pairs"]
        assert bool((p[1:, 0] > p[:-1, 0]).all()) and bool((p[1:, 1] > p[:-1, 1]).all())
        assert int(p[:, 0].max()) < int(row["qwen_tokens_real"]) and int(p[:, 1].max()) < int(row["k3_tokens_real"])


@pytest.mark.skipif(not REAL_MAP.is_file(), reason="the parity xtok output is on the server only")
def test_the_real_k3_vocab_map_is_injective():
    mapping = load_file(str(REAL_MAP))["k3_to_qwen"].long()
    mapped = mapping[mapping >= 0]
    assert mapped.numel() > 100_000
    assert int(mapped.max()) < ktc.QWEN_TEXT_ROWS
    assert int(torch.unique(mapped).numel()) == int(mapped.numel())   # the bucket KL's uniqueness assumption
