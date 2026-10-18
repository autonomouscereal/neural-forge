"""Tests for ``qtext_teacher_cache.py`` 1.0.5.  CPU only, tiny random-init fixture.

The fixture is a real ``Qwen3_5ForConditionalGeneration`` (both GDN and gated-GQA
layer kinds, a two-block vision tower, the full 248,320-row vocabulary so the
parent's own processor and chat template apply), saved sharded so the streamed
engine's ``ShardReader`` reads it exactly as it reads the 27B.  fp32 on the CPU with
the torch GDN kernels forced.  Every claim here is about MECHANICS: the writer, the
shard format, resume, the locks, and both parity arms end to end at toy scale.  The
real-checkpoint parity runs are the server runs in the lane report, not these.

    cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
        /home/cereal/model-surgery-lab/venv/bin/python -m pytest -o addopts="" \\
        -p no:cacheprovider test_qtext_teacher_cache.py -q
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import qtext_teacher_cache as qc  # noqa: E402

PROCESSOR_SOURCE = Path("/home/cereal/models/Qwen3.5-0.8B")
PROCESSOR_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json",
                   "video_preprocessor_config.json", "vocab.json", "merges.txt")


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory) -> Path:
    if not PROCESSOR_SOURCE.is_dir():
        pytest.skip(f"{PROCESSOR_SOURCE} (processor + tokenizer source) is not on this host")
    qc.select_linear_attention_kernel("torch")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

    text = dict(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, layer_types=["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4, linear_conv_kernel_dim=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
        vocab_size=qc.TEXT_VOCAB_ROWS, max_position_embeddings=4096, rms_norm_eps=1e-6,
        attn_output_gate=True, tie_word_embeddings=False, mtp_num_hidden_layers=0,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000, "partial_rotary_factor": 0.25,
                         "mrope_interleaved": True, "mrope_section": [2, 1, 1]},
    )
    vision = dict(depth=2, hidden_size=32, num_heads=2, intermediate_size=64, out_hidden_size=32,
                  patch_size=16, spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                  num_position_embeddings=2304, deepstack_visual_indexes=[], hidden_act="gelu_pytorch_tanh")
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=248056, video_token_id=248057,
                           vision_start_token_id=248053, vision_end_token_id=248054, tie_word_embeddings=False)
    torch.manual_seed(20261012)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32)
    # Random init leaves norms at 0 (they are 1-centred); give every tensor some spread so
    # top-k is not decided by ties.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.05)
    root = tmp_path_factory.mktemp("tiny-qwen35")
    model.save_pretrained(str(root), max_shard_size="8MB")
    for name in PROCESSOR_FILES:
        if (PROCESSOR_SOURCE / name).is_file():
            shutil.copy2(PROCESSOR_SOURCE / name, root / name)
    assert (root / "model.safetensors.index.json").is_file()
    return root


def build_args(tiny_dir: Path, out_root: Path, corpus: Path, **overrides) -> argparse.Namespace:
    base = dict(model_dir=str(tiny_dir), processor_dir=None, device="cpu", dtype="float32",
                attn_implementation="sdpa", linear_attn_kernel="torch", top_k=8, vocab_rows=qc.TEXT_VOCAB_ROWS,
                micro_batch=2, head_chunk_rows=128, rows_per_shard=2, max_pixels=None, teacher_digest="full",
                digest_threads=2, allocator_receipt=None, purpose="test", ids_key="input_ids",
                teacher="tiny", corpus=[str(c) for c in corpus] if isinstance(corpus, list) else str(corpus),
                out_root=str(out_root), verify_existing=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def write_corpus(path: Path, rows: int = 5, length: int = 24, seed: int = 3, bump: bool = False) -> Path:
    from safetensors.torch import save_file

    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 248_044, (rows, length), generator=generator, dtype=torch.int64)
    if bump:
        ids[0, 0] = (ids[0, 0] + 1) % 248_044
    lengths = torch.tensor([length - (3 * r) % 9 for r in range(rows)], dtype=torch.int64)
    save_file({"input_ids": ids, "lengths": lengths}, str(path))
    return path


# ------------------------------------------------------------------ unit
def test_score_hidden_is_log_softmax_top_k_over_text_rows_only():
    torch.manual_seed(0)
    hidden = torch.randn(10, 16)
    weight = torch.randn(40, 16)
    weight[30:] = hidden.mean(0) * 1000.0  # rows past V would win every top-k if they were scored
    ids, lp, lse = qc.score_hidden(hidden, weight, vocab_rows=30, top_k=5, chunk_rows=3)
    reference = torch.log_softmax(hidden @ weight[:30].T, dim=-1)
    values, indices = torch.topk(reference, k=5, dim=-1)
    assert torch.equal(ids.to(torch.int64), indices)
    assert int(ids.max()) < 30
    assert torch.allclose(lp.float(), values, atol=2e-2)
    assert torch.allclose(lse, torch.logsumexp(hidden @ weight[:30].T, dim=-1), atol=1e-5)
    assert lp.dtype == torch.bfloat16 and ids.dtype == torch.int32 and lse.dtype == torch.float32


def test_score_hidden_refuses_a_head_narrower_than_vocab_rows():
    with pytest.raises(qc.CacheError, match="fewer than"):
        qc.score_hidden(torch.randn(2, 4), torch.randn(5, 4), vocab_rows=6, top_k=2, chunk_rows=2)


def test_stored_mask_rule():
    lengths = torch.tensor([4, 2])
    store = torch.ones(2, 5, dtype=torch.bool)
    store[0, 1] = False
    types = torch.zeros(2, 5, dtype=torch.int64)
    types[0, 2] = 1
    mask = qc.stored_mask_for(lengths, 5, store, types)
    assert mask.tolist() == [[True, False, False, True, False], [True, True, False, False, False]]


def arrays_for(p: int, k: int, vocab: int = 100) -> dict:
    torch.manual_seed(p)
    logits = torch.randn(p, vocab)
    lse = torch.logsumexp(logits, -1)
    values, indices = torch.topk(logits, k, -1)
    return {"ids": torch.arange(p) % 7, "pos": torch.arange(p), "topk_ids": indices.to(torch.int32),
            "topk_logprobs": (values - lse[:, None]).to(torch.bfloat16), "lse": lse}


def test_shard_roundtrip_is_byte_exact_and_verify_catches_a_flipped_byte(tmp_path):
    arrays = arrays_for(13, 4)
    shard = qc.write_shard(tmp_path, 0, arrays, {"rows": [0, 1], "row_table": []}, vocab_rows=100, top_k=4)
    assert shard["positions"] == 13
    assert shard["files"]["topk_logprobs.bf16"]["bytes"] == 13 * 4 * 2
    back = qc.read_shard(tmp_path / "shard-00000")
    for key in ("ids", "pos", "topk_ids", "lse"):
        assert torch.equal(back[key].to(arrays[key].dtype), arrays[key]), key
    assert torch.equal(back["topk_logprobs"].view(torch.int16), arrays["topk_logprobs"].view(torch.int16))
    assert qc.verify_shard(tmp_path / "shard-00000")["ok"]
    target = tmp_path / "shard-00000" / "lse.f32"
    data = bytearray(target.read_bytes())
    data[5] ^= 0x01
    target.write_bytes(bytes(data))
    result = qc.verify_shard(tmp_path / "shard-00000")
    assert not result["ok"] and "lse.f32 sha256 mismatch" in result["problems"]


def test_non_finite_shard_is_refused_and_never_published(tmp_path):
    arrays = arrays_for(6, 4)
    arrays["lse"][3] = float("nan")
    with pytest.raises(qc.NonFiniteShard):
        qc.write_shard(tmp_path, 2, arrays, {"rows": [4, 6], "row_table": []}, vocab_rows=100, top_k=4)
    assert not (tmp_path / "shard-00002").exists()
    failure = json.loads((tmp_path / "failures" / "shard-00002.json").read_text())
    assert failure["checks"]["nonfinite_lse"] == 1 and failure["checks"]["finite"] is False


def test_out_of_range_topk_id_is_a_structural_failure(tmp_path):
    arrays = arrays_for(5, 4, vocab=100)
    with pytest.raises(qc.NonFiniteShard, match="topk_ids_out_of_range"):
        qc.write_shard(tmp_path, 0, arrays, {"rows": [0, 1], "row_table": []}, vocab_rows=50, top_k=4)


def test_text_corpus_is_content_addressed(tmp_path):
    a = qc.load_text_corpus(write_corpus(tmp_path / "a.safetensors"))
    b = qc.load_text_corpus(write_corpus(tmp_path / "b.safetensors"))
    c = qc.load_text_corpus(write_corpus(tmp_path / "c.safetensors", bump=True))
    assert a["sha256"] == b["sha256"] != c["sha256"]
    assert a["tokens"] == int(a["lengths"].sum())


def test_text_corpus_refuses_bad_lengths(tmp_path):
    from safetensors.torch import save_file

    save_file({"input_ids": torch.zeros(2, 4, dtype=torch.int64), "lengths": torch.tensor([4, 5])},
              str(tmp_path / "bad.safetensors"))
    with pytest.raises(qc.CacheError, match="lengths"):
        qc.load_text_corpus(tmp_path / "bad.safetensors")


def test_teacher_digest_moves_with_one_byte(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"abc")
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "lock").write_bytes(b"x")
    first = qc.teacher_digest(tmp_path, "full")
    (tmp_path / ".cache" / "lock").write_bytes(b"changed")
    assert qc.teacher_digest(tmp_path, "full")["tree_sha256"] == first["tree_sha256"]
    (tmp_path / "a.bin").write_bytes(b"abd")
    assert qc.teacher_digest(tmp_path, "full")["tree_sha256"] != first["tree_sha256"]
    assert first["file_count"] == 1


def test_writer_lock_refuses_a_live_holder_and_takes_over_a_dead_one(tmp_path):
    lock = qc.WriterLock(tmp_path)
    lock.acquire()
    with pytest.raises(qc.CacheError, match="another writer"):
        qc.WriterLock(tmp_path).acquire()
    lock.release()
    (tmp_path / "writer.lock").write_text(json.dumps({"host": qc.socket.gethostname(), "pid": 2**22 + 12345}))
    second = qc.WriterLock(tmp_path)
    second.acquire()
    assert second.taken_over["pid"] == 2**22 + 12345
    second.release()


def test_collate_vl_pads_right_and_concatenates_media_in_row_order():
    a = {"input_ids": torch.tensor([[1, 2, 3]]), "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
         "pixel_values": torch.full((4, 6), 1.0), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    b = {"input_ids": torch.tensor([[4, 5, 6, 7, 8]]), "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]]),
         "pixel_values": torch.full((8, 6), 2.0), "image_grid_thw": torch.tensor([[1, 2, 4]])}
    batch = qc.collate_vl([a, b], [0, 1], pad_id=99)
    assert batch["input_ids"].tolist() == [[1, 2, 3, 99, 99], [4, 5, 6, 7, 8]]
    assert batch["mm_token_type_ids"].tolist() == [[0, 1, 0, 0, 0], [0, 1, 1, 0, 0]]
    assert batch["lengths"].tolist() == [3, 5]
    assert batch["pixel_values"][:4].eq(1.0).all() and batch["pixel_values"][4:].eq(2.0).all()
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]


# ------------------------------------------------------------------ end to end on the tiny model
def cache_bytes(cache_dir: Path) -> dict:
    return {p.relative_to(cache_dir).as_posix(): p.read_bytes()
            for p in sorted(cache_dir.glob("shard-*/*")) if p.name != "shard.json"}


def test_build_text_seals_verifies_and_resumes_byte_identically(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    out = tmp_path / "cache"
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0
    sha = qc.load_text_corpus(corpus)["sha256"]
    cache_dir = out / "tiny" / sha
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["shard_read_order"] == ["shard-00000", "shard-00001", "shard-00002"]
    assert manifest["all_finite"] and manifest["all_structural_ok"]
    assert manifest["teacher"]["equal_before_and_after_every_session"]
    assert manifest["positions_total"] == qc.load_text_corpus(corpus)["tokens"]
    assert manifest["cache_config"]["vocab_rows"] == qc.TEXT_VOCAB_ROWS
    assert manifest["engine"]["sha256"] == qc.file_sha256(qc.ENGINE_PATH)
    assert qc.verify_cache(cache_dir) == 0
    first = cache_bytes(cache_dir)

    # Resume: lose the seal and one shard; only that shard is rewritten, byte-identical.
    (cache_dir / "manifest.json").unlink()
    shutil.rmtree(cache_dir / "shard-00001")
    kept = (cache_dir / "shard-00000" / "shard.json").read_bytes()
    assert qc.build_cache(build_args(tiny_dir, out, corpus, verify_existing=True), "text") == 0
    assert (cache_dir / "shard-00000" / "shard.json").read_bytes() == kept
    assert cache_bytes(cache_dir) == first
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert len(manifest["sessions"]) == 2
    assert len({s["session"] for s in manifest["shards"]}) == 2


def test_a_sealed_cache_is_verified_not_rewritten(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=2)
    out = tmp_path / "cache"
    args = build_args(tiny_dir, out, corpus)
    assert qc.build_cache(args, "text") == 0
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    sealed = (cache_dir / "manifest.json").read_bytes()
    assert qc.build_cache(args, "text") == 0
    assert (cache_dir / "manifest.json").read_bytes() == sealed


def test_resume_with_different_semantics_is_refused(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=3)
    out = tmp_path / "cache"
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    (cache_dir / "manifest.json").unlink()
    with pytest.raises(qc.CacheError, match="top_k"):
        qc.build_cache(build_args(tiny_dir, out, corpus, top_k=16), "text")
    assert not (cache_dir / "writer.lock").exists()


def test_corpus_id_past_the_embedding_is_refused(tiny_dir, tmp_path):
    from safetensors.torch import save_file

    save_file({"input_ids": torch.full((1, 4), qc.TEXT_VOCAB_ROWS, dtype=torch.int64)},
              str(tmp_path / "big.safetensors"))
    with pytest.raises(qc.CacheError, match="embedding rows"):
        qc.build_cache(build_args(tiny_dir, tmp_path / "cache", tmp_path / "big.safetensors"), "text")


def parity_args(tiny_dir: Path, work: Path, **overrides) -> argparse.Namespace:
    base = dict(model_dir=str(tiny_dir), processor_dir=None, device="cpu", dtype="float32",
                attn_implementation="sdpa", linear_attn_kernel="torch", top_k=16, vocab_rows=qc.TEXT_VOCAB_ROWS,
                micro_batch=2, head_chunk_rows=128, rows_per_shard=2, max_pixels=None, teacher_digest="full",
                digest_threads=2, allocator_receipt=None, purpose="test", ids_key="input_ids",
                work_dir=str(work), receipt=None, corpus=None, parity_rows=3, parity_length=48, seed=7)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_text_parity_arm_passes_on_the_tiny_model(tiny_dir, tmp_path):
    assert qc.run_parity(parity_args(tiny_dir, tmp_path / "text"), "text") == 0
    receipt = json.loads((tmp_path / "text" / "parity-text.json").read_text())
    c = receipt["comparison_batched_same_inputs"]
    assert receipt["passed"] and c["topk_ids_equal_all_positions"] and c["same_positions_and_input_ids"]
    assert c["positions"] > 0
    # Streaming changes no arithmetic: the stored bf16 equals the full load rounded to bf16.
    assert c["logprob_bitwise_equal_to_reference_bf16"], c
    assert c["lse_max_abs_diff"] == 0.0, c


def test_vl_parity_arm_passes_on_the_tiny_model(tiny_dir, tmp_path):
    assert qc.run_parity(parity_args(tiny_dir, tmp_path / "vl"), "vl") == 0
    receipt = json.loads((tmp_path / "vl" / "parity-vl.json").read_text())
    c = receipt["comparison_batched_same_inputs"]
    assert receipt["passed"] and c["topk_ids_equal_all_positions"]
    assert c["logprob_bitwise_equal_to_reference_bf16"], c
    cache = qc.load_cache_arrays(Path(receipt["cache_dir"]))
    # Placeholder positions are not stored: no stored input id is an image pad.
    assert not bool((cache["ids"] == 248056).any())
    # Row 1 carries two images; both rows are present in the row table.
    assert [r["row_id"] for r in cache["row_table"]] == ["vl-parity-0", "vl-parity-1"]
    assert receipt["full_load_vision_buffers"] is not None


def write_video_corpus(work: Path) -> Path:
    from PIL import Image

    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    names = []
    for i in range(4):
        Image.new("RGB", (256, 192), (i * 50, 120, 60)).save(frames_dir / f"f{i}.png")
        names.append(f"frames/f{i}.png")
    rows = [{"row_id": "video-0", "messages": [
        {"role": "user", "content": [{"type": "video", "frames": names, "fps": 2.0},
                                     {"type": "text", "text": "What changes over the clip?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "The red channel rises frame by frame."}]}]}]
    path = work / "video.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_video_row_uses_the_corpus_fps_for_the_timestamps(tiny_dir, tmp_path):
    corpus = qc.load_vl_corpus(write_video_corpus(tmp_path))
    processor = qc.load_processor(tiny_dir, None)
    encoded = qc.encode_vl_row(processor, corpus["rows_data"][0], tmp_path)
    assert "pixel_values_videos" in encoded and "video_grid_thw" in encoded
    text = processor.tokenizer.decode([t for t in encoded["input_ids"][0].tolist() if t != 248057])
    # 4 frames at 2 fps sit at 0.0/0.5/1.0/1.5 s; temporal patch 2 merges pairs and the
    # processor stamps each group with its mean time: 0.25 -> "0.2", 1.25 -> "1.2".  The
    # processor's fps-24 fallback (metadata ignored) would stamp "0.0" and "0.1".
    assert "<0.2 seconds>" in text and "<1.2 seconds>" in text, text
    assert "<0.1 seconds>" not in text


def test_video_parity_arm_passes_on_the_tiny_model(tiny_dir, tmp_path):
    corpus = write_video_corpus(tmp_path)
    assert qc.run_parity(parity_args(tiny_dir, tmp_path / "work", corpus=str(corpus)), "vl") == 0
    receipt = json.loads((tmp_path / "work" / "parity-vl.json").read_text())
    assert receipt["passed"] and receipt["comparison_batched_same_inputs"]["topk_ids_equal_all_positions"]
    cache = qc.load_cache_arrays(Path(receipt["cache_dir"]))
    assert not bool((cache["ids"] == 248057).any())


def test_a_refused_card_start_leaves_no_session_and_no_lock(tiny_dir, tmp_path, monkeypatch):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=2)

    def busy(*_a, **_k):
        raise qc.CacheError("card GPU-x already holds 9999 MiB (> 2000); refusing to start")

    monkeypatch.setattr(qc, "visible_card_check", busy)
    with pytest.raises(qc.CacheError, match="refusing to start"):
        qc.build_cache(build_args(tiny_dir, tmp_path / "cache", corpus, device="cuda:0"), "text")
    cache_dir = tmp_path / "cache" / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    assert not (cache_dir / "sessions").exists() and not (cache_dir / "writer.lock").exists()


def test_seal_records_but_does_not_judge_a_session_that_stopped_before_its_digest(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=2)
    out = tmp_path / "cache"
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    (cache_dir / "sessions").mkdir(parents=True)
    # The v1.0.1 shape of a refused start: a session record with an "after" digest only.
    (cache_dir / "sessions" / "session-20261013T015909Z-1.json").write_text(json.dumps(
        {"session": "20261013T015909Z-1", "teacher_unchanged_in_session": False,
         "teacher_digest_after": {"tree_sha256": "x" * 64, "mode": "full", "files": []}}))
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert [a["session"] for a in manifest["sessions_aborted_before_start"]] == ["20261013T015909Z-1"]
    assert len(manifest["sessions"]) == 1 and manifest["teacher"]["equal_before_and_after_every_session"]


# ------------------------------------------------------------------ stopped sessions (1.0.3)
STOP_SCRIPT = r"""
import argparse, json, os, signal, sys
sys.path.insert(0, sys.argv[1])
import qtext_teacher_cache as qc
how, args_json = sys.argv[2], sys.argv[3]
real = qc.fsync_directory


def stop_after_staging_shard_1(path):
    real(path)
    # Shard 0 is published; shard 1 is fully staged but not yet renamed into place.
    if path.name.startswith(".shard-00001.tmp-"):
        os.kill(os.getpid(), signal.SIGKILL if how == "kill" else signal.SIGTERM)


qc.fsync_directory = stop_after_staging_shard_1
raise SystemExit(qc.build_cache(argparse.Namespace(**json.loads(args_json)), "text"))
"""


def run_stopped_writer(tmp_path: Path, how: str, args: argparse.Namespace) -> subprocess.CompletedProcess:
    script = tmp_path / "stop_writer.py"
    script.write_text(STOP_SCRIPT, encoding="utf-8")
    return subprocess.run([sys.executable, str(script), str(HERE), how, json.dumps(vars(args))],
                          capture_output=True, text=True, timeout=600, env=dict(os.environ))


def session_records(cache_dir: Path) -> list:
    return [json.loads(p.read_text()) for p in sorted((cache_dir / "sessions").glob("session-*.json"))]


@pytest.mark.skipif(not hasattr(qc.signal, "SIGKILL"), reason="needs POSIX signals")
def test_a_hard_killed_session_is_interrupted_not_a_teacher_change(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    out = tmp_path / "cache"
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    killed = run_stopped_writer(tmp_path, "kill", build_args(tiny_dir, out, corpus))
    assert killed.returncode == -9, killed.stdout + killed.stderr
    # The shape a SIGKILL leaves: a "before"-only session, one shard, a staged shard, the lock.
    (record,) = session_records(cache_dir)
    assert "teacher_digest_before" in record and "teacher_digest_after" not in record
    assert (cache_dir / "shard-00000").is_dir() and not (cache_dir / "shard-00001").exists()
    assert len(list(cache_dir.glob(qc.STAGING_GLOB))) == 1
    assert (cache_dir / "writer.lock").is_file()
    # v1.0.2 refused to seal here ("teacher digests differ ... None") on every resume.
    assert qc.build_cache(build_args(tiny_dir, out, corpus, verify_existing=True), "text") == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    (interrupted,) = manifest["sessions_interrupted"]
    (complete,) = manifest["sessions"]
    assert interrupted["session"] == record["session"] and interrupted["shards_written"] == 1
    assert interrupted["vouched_by"] == complete["session"]
    assert manifest["teacher"]["interrupted_sessions_vouched_for"] == 1
    assert manifest["teacher"]["equal_before_and_after_every_session"]
    resume = session_records(cache_dir)[-1]
    assert resume["lock_taken_over_from"]["pid"] == int(record["session"].rsplit("-", 1)[1])
    assert len(resume["stale_staging_removed"]) == 1 and not list(cache_dir.glob(qc.STAGING_GLOB))
    assert not (cache_dir / "writer.lock").exists()
    # The shards are the shards an uninterrupted writer produces, byte for byte.
    clean = tmp_path / "clean"
    assert qc.build_cache(build_args(tiny_dir, clean, corpus), "text") == 0
    assert cache_bytes(cache_dir) == cache_bytes(clean / "tiny" / qc.load_text_corpus(corpus)["sha256"])


@pytest.mark.skipif(not hasattr(qc.signal, "SIGKILL"), reason="needs POSIX signals")
def test_sigterm_still_writes_the_after_digest_and_records_exit_143(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    out = tmp_path / "cache"
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    receipt = tmp_path / "allocator" / "term.json"
    termed = run_stopped_writer(tmp_path, "term", build_args(tiny_dir, out, corpus, allocator_receipt=str(receipt)))
    assert termed.returncode == 143, termed.stdout + termed.stderr
    (record,) = session_records(cache_dir)
    assert record["teacher_unchanged_in_session"] is True and "teacher_digest_after" in record
    assert record["failure"] == "SystemExit(143)"
    assert not (cache_dir / "writer.lock").exists()
    allocator = json.loads(receipt.read_text())
    assert allocator["outcome"] == "exit 143" and allocator["failure"] == "SystemExit(143)"
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert manifest["sessions_interrupted"] == [] and len(manifest["sessions"]) == 2
    assert manifest["sessions"][0]["failure"] == "SystemExit(143)"


def test_a_crashed_session_records_the_failure_and_a_nonzero_exit(tiny_dir, tmp_path, monkeypatch):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=3)
    out = tmp_path / "cache"
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    receipt = tmp_path / "allocator" / "oom.json"

    def oom(*_a, **_k):
        raise RuntimeError("CUDA out of memory (simulated)")

    with monkeypatch.context() as patch:
        patch.setattr(qc, "run_batch", oom)
        with pytest.raises(RuntimeError, match="simulated"):
            qc.build_cache(build_args(tiny_dir, out, corpus, allocator_receipt=str(receipt)), "text")
    allocator = json.loads(receipt.read_text())
    assert allocator["outcome"] == "exit 1" and "simulated" in allocator["failure"]  # v1.0.2: "exit 0"
    (record,) = session_records(cache_dir)
    assert "simulated" in record["failure"] and record["teacher_unchanged_in_session"] is True
    assert not (cache_dir / "writer.lock").exists()
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0


def test_seal_judges_an_interrupted_session_by_the_sessions_around_it(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=2)
    out = tmp_path / "cache"
    args = build_args(tiny_dir, out, corpus)
    assert qc.build_cache(args, "text") == 0
    loaded = qc.load_text_corpus(corpus)
    cache_dir = out / "tiny" / loaded["sha256"]
    (good,) = session_records(cache_dir)
    before = good["teacher_digest_before"]

    def seal_with(name: str, tree: str) -> int:
        (cache_dir / "manifest.json").unlink(missing_ok=True)
        for p in (cache_dir / "sessions").glob("session-*-99.json"):
            p.unlink()
        (cache_dir / "sessions" / f"session-{name}-99.json").write_text(json.dumps(
            {"session": f"{name}-99", "teacher_digest_before": {**before, "tree_sha256": tree}}))
        return qc.seal_manifest(cache_dir, loaded, Path(args.model_dir), 0.0)

    # The reviewer's repro: a before-only session that wrote nothing, earlier than a complete one.
    assert seal_with("20000101T000000Z", before["tree_sha256"]) == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert [x["session"] for x in manifest["sessions_interrupted"]] == ["20000101T000000Z-99"]
    assert manifest["sessions_interrupted"][0]["vouched_by"] == good["session"]
    # It started on another tree: a teacher change, refused.
    assert seal_with("20000101T000000Z", "y" * 64) == 2
    # It is the latest session: nothing after it vouches for the teacher, not sealed (yet).
    assert seal_with("29991231T235959Z", before["tree_sha256"]) == 1
    assert not (cache_dir / "manifest.json").exists()


# ------------------------------------------------------------------ 1.0.4: L2 target_mask
def write_prompt_completion_corpus(path: Path, prompt_lens=(5, 9), lengths=(14, 20), width: int = 24,
                                   with_store: str | None = None) -> dict:
    """Prompt+completion rows: target_mask marks the completion tokens [prompt_len, length)."""
    from safetensors.torch import save_file

    rows = len(prompt_lens)
    ids = torch.randint(0, 248_044, (rows, width), generator=torch.Generator().manual_seed(11), dtype=torch.int64)
    target = torch.zeros((rows, width), dtype=torch.uint8)
    for r, (p, n) in enumerate(zip(prompt_lens, lengths)):
        target[r, p:n] = 1
    tensors = {"input_ids": ids, "lengths": torch.tensor(lengths, dtype=torch.int64), "target_mask": target}
    if with_store == "right":
        store = torch.zeros_like(target)
        for r, (p, n) in enumerate(zip(prompt_lens, lengths)):
            store[r, p - 1 : n - 1] = 1
        tensors["store_mask"] = store
    elif with_store == "wrong":  # the off-by-one a producer makes when it marks the targets as store positions
        tensors["store_mask"] = target.clone()
    save_file(tensors, str(path))
    return {"ids": ids, "prompt_lens": prompt_lens, "lengths": lengths}


def test_target_mask_is_shifted_inside_the_tool():
    lengths = torch.tensor([6, 4])
    target = torch.zeros(2, 6, dtype=torch.bool)
    target[0, 2:6] = True
    target[1, 3:4] = True
    store = qc.store_mask_from_target_mask(target, lengths)
    assert store.tolist() == [[False, True, True, True, True, False], [False, False, True, False, False, False]]


def test_prompt_completion_first_stored_position_is_the_last_prompt_token(tiny_dir, tmp_path):
    facts = write_prompt_completion_corpus(tmp_path / "pc.safetensors")
    corpus = qc.load_text_corpus(tmp_path / "pc.safetensors")
    assert corpus["mask_source"].startswith("target_mask shifted by one")
    out = tmp_path / "cache"
    assert qc.build_cache(build_args(tiny_dir, out, tmp_path / "pc.safetensors"), "text") == 0
    cache = qc.load_cache_arrays(out / "tiny" / corpus["sha256"])
    assert cache["manifest"]["cache_config"]["corpus_read"]["mask_source"] == corpus["mask_source"]
    for entry in cache["row_table"]:
        r = entry["row"]
        p, n = facts["prompt_lens"][r], facts["lengths"][r]
        pos = cache["pos"][entry["offset"] : entry["offset"] + entry["count"]].tolist()
        ids = cache["ids"][entry["offset"] : entry["offset"] + entry["count"]].tolist()
        # One distribution per completion token: the first is read at the LAST PROMPT token,
        # the last at the second-to-last completion token.
        assert entry["count"] == n - p
        assert pos[0] == p - 1 and ids[0] == int(facts["ids"][r, p - 1])
        assert pos == list(range(p - 1, n - 1))


def test_target_mask_refusals(tmp_path):
    from safetensors.torch import save_file

    write_prompt_completion_corpus(tmp_path / "ok.safetensors", with_store="right")
    assert qc.load_text_corpus(tmp_path / "ok.safetensors")["mask_source"].startswith("store_mask, checked equal")
    write_prompt_completion_corpus(tmp_path / "wrong.safetensors", with_store="wrong")
    with pytest.raises(qc.CacheError, match="disagree"):
        qc.load_text_corpus(tmp_path / "wrong.safetensors")
    target = torch.zeros(1, 8, dtype=torch.uint8)
    target[0, 0] = 1
    save_file({"input_ids": torch.zeros(1, 8, dtype=torch.int64), "target_mask": target}, str(tmp_path / "c0.safetensors"))
    with pytest.raises(qc.CacheError, match="column 0"):
        qc.load_text_corpus(tmp_path / "c0.safetensors")
    target = torch.zeros(1, 8, dtype=torch.uint8)
    target[0, 6] = 1
    save_file({"input_ids": torch.zeros(1, 8, dtype=torch.int64), "lengths": torch.tensor([5]), "target_mask": target},
              str(tmp_path / "past.safetensors"))
    with pytest.raises(qc.CacheError, match="past their row"):
        qc.load_text_corpus(tmp_path / "past.safetensors")


def test_lengths_key_auto_reads_the_corpus_lanes_real_lengths(tmp_path):
    from safetensors.torch import save_file

    save_file({"input_ids": torch.ones(2, 8, dtype=torch.int32), "real_lengths": torch.tensor([8, 3], dtype=torch.int32)},
              str(tmp_path / "rl.safetensors"))
    corpus = qc.load_text_corpus(tmp_path / "rl.safetensors")
    assert corpus["lengths_key"] == "real_lengths" and corpus["lengths"].tolist() == [8, 3] and corpus["tokens"] == 11
    with pytest.raises(qc.CacheError, match="no 'lengths'"):
        qc.load_text_corpus(tmp_path / "rl.safetensors", lengths_key="lengths")


# ------------------------------------------------------------------ 1.0.4: L3 head precision
def test_fp32_head_is_an_exact_upcast_and_the_model_head_is_not():
    torch.manual_seed(5)
    hidden = (torch.randn(64, 32) * 4).to(torch.bfloat16)
    weight = (torch.randn(300, 32) * 4).to(torch.bfloat16)
    exact = hidden.double() @ weight.double().T
    fp32 = qc.prepare_head(weight, 256, "fp32")
    assert fp32.dtype == torch.float32 and fp32.shape == (256, 32)
    assert torch.equal(fp32, weight[:256].float())
    _ids, _lp, lse32 = qc.score_hidden(hidden, fp32, 256, 8, 16)
    _ids, _lp, lse16 = qc.score_hidden(hidden, qc.prepare_head(weight, 256, "model"), 256, 8, 16)
    reference = torch.logsumexp(exact[:, :256], dim=-1)
    err32 = float((lse32.double() - reference).abs().max())
    err16 = float((lse16.double() - reference).abs().max())
    assert err32 < 1e-4 and err16 > 10 * err32, (err32, err16)
    with pytest.raises(qc.CacheError, match="head-precision"):
        qc.prepare_head(weight, 256, "fp16")


def test_head_precision_is_locked_semantics(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=2)
    out = tmp_path / "cache"
    assert qc.build_cache(build_args(tiny_dir, out, corpus, head_precision="fp32"), "text") == 0
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    config = json.loads((cache_dir / "cache_config.json").read_text())
    assert config["head_precision"] == "fp32" and "TF32 off" in config["logit_precision"]
    with pytest.raises(qc.CacheError, match="head_precision"):
        qc.build_cache(build_args(tiny_dir, out, corpus, head_precision="model"), "text")


# ------------------------------------------------------------------ 1.0.4: batching
def test_plan_batches_longest_first_under_a_token_cap():
    lengths = [5, 30, 7, 30, 2, 12]
    assert qc.plan_batches(lengths, 2, 0, "corpus") == [[0, 1], [2, 3], [4, 5]]
    assert qc.plan_batches(lengths, 4, 0, "length") == [[1, 3, 5, 2], [0, 4]]
    # cap 60: the two 30-wide rows fill a batch; the rest are at most 12 wide, 4 x 12 = 48 <= 60
    assert qc.plan_batches(lengths, 8, 60, "length") == [[1, 3], [5, 2, 0, 4]]
    # a row longer than the cap runs alone
    assert qc.plan_batches([100, 3], 8, 50, "length") == [[0], [1]]
    with pytest.raises(qc.CacheError, match="batch-order"):
        qc.plan_batches(lengths, 2, 0, "random")


def test_shard_layout_does_not_depend_on_the_batch_plan(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    sha = qc.load_text_corpus(corpus)["sha256"]
    caches = {}
    for name, order, cap in (("corpus", "corpus", 0), ("length", "length", 40)):
        out = tmp_path / name
        assert qc.build_cache(build_args(tiny_dir, out, corpus, batch_order=order, max_batch_tokens=cap,
                                         rows_per_shard=5, micro_batch=3), "text") == 0
        caches[name] = qc.load_cache_arrays(out / "tiny" / sha)
    a, b = caches["corpus"], caches["length"]
    assert [(r["row"], r["offset"], r["count"]) for r in a["row_table"]] == [(r["row"], r["offset"], r["count"]) for r in b["row_table"]]
    assert torch.equal(a["ids"], b["ids"]) and torch.equal(a["pos"], b["pos"])
    assert torch.equal(a["topk_ids"], b["topk_ids"])
    shard = json.loads((tmp_path / "length" / "tiny" / sha / "shard-00000" / "shard.json").read_text())
    assert shard["batch_order"] == "length" and shard["padded_tokens"] >= shard["forward_tokens"]
    assert shard["batches"] >= 3  # 24-wide rows under a 40-token cap: at most one per batch


def test_max_new_shards_stops_unsealed_and_a_later_session_seals(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    out = tmp_path / "cache"
    cache_dir = out / "tiny" / qc.load_text_corpus(corpus)["sha256"]
    assert qc.build_cache(build_args(tiny_dir, out, corpus, max_new_shards=1), "text") == 0
    assert (cache_dir / "shard-00000").is_dir() and not (cache_dir / "shard-00001").exists()
    assert not (cache_dir / "manifest.json").exists()
    assert qc.build_cache(build_args(tiny_dir, out, corpus), "text") == 0
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    assert len(manifest["sessions"]) == 2 and manifest["all_finite"]


# ------------------------------------------------------------------ 1.0.4: L1 processor files, corpus-lane rows, video files
def test_processor_files_are_locked_into_a_vl_cache(tiny_dir, tmp_path):
    processor_dir = tmp_path / "processor"
    shutil.copytree(tiny_dir, processor_dir)
    work = tmp_path / "work"
    jsonl = qc.make_parity_vl_corpus(work, None)
    out = tmp_path / "cache"
    args = build_args(tiny_dir, out, jsonl, processor_dir=str(processor_dir), chat_template_kwargs='{"enable_thinking": false}')
    assert qc.build_cache(args, "vl") == 0
    cache_dir = out / "tiny" / qc.load_vl_corpus(jsonl)["sha256"]
    config = json.loads((cache_dir / "cache_config.json").read_text())
    files = config["processor"]["files_sha256"]
    assert files["chat_template.jinja"] == qc.file_sha256(processor_dir / "chat_template.jinja")
    assert config["processor"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert config["processor"]["dir"] == str(processor_dir.resolve())
    template = processor_dir / "chat_template.jinja"
    template.write_text(template.read_text() + "\n{#- an edit #}", encoding="utf-8")
    with pytest.raises(qc.CacheError, match="processor"):
        qc.build_cache(args, "vl")


def write_corpus_lane_records(work: Path) -> Path:
    from PIL import Image

    store = work / "media_store"
    store.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 96), (200, 30, 30)).save(store / "a.png")
    sha = qc.file_sha256(store / "a.png")
    record = {"uid": "u-1", "rid": "000:0:1", "instruction": "user: What colour is the square?",
              "media": [{"media": "image", "path": "/elsewhere/a.png", "stored_path": str(store / "a.png"), "sha256": sha,
                         "visual_tokens": 6}],
              "reference_response": "It is red.", "route": "image"}
    path = work / "vision_in.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


def test_corpus_lane_records_map_to_one_user_and_one_assistant_turn(tiny_dir, tmp_path):
    path = write_corpus_lane_records(tmp_path)
    corpus = qc.load_vl_corpus(path)
    (row,) = corpus["rows_data"]
    assert row["row_id"] == "u-1" and corpus["row_formats"] == {"corpus-lane vision_in record": 1}
    user, assistant = row["messages"]
    assert [c["type"] for c in user["content"]] == ["image", "text"]
    assert user["content"][1]["text"] == "user: What colour is the square?"
    assert assistant["content"][0]["text"] == "It is red."
    encoded = qc.encode_vl_row(qc.load_processor(tiny_dir, None), row, tmp_path)
    assert int(encoded["mm_token_type_ids"].sum()) > 0
    record = json.loads(path.read_text())
    record["media"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(qc.CacheError, match="digests to"):
        qc.load_vl_corpus(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs the ffmpeg CLI")
def test_a_video_file_is_decoded_and_sampled_by_the_processor_rule(tiny_dir, tmp_path):
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=128x96:rate=16:duration=2",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", str(clip)], check=True, timeout=120)
    frames, facts = qc.decode_video(clip, 4)
    assert facts["total_num_frames"] == 32 and facts["fps"] == 16.0 and facts["frames_indices"] == [0, 10, 21, 31]
    assert len(frames) == 4 and frames[0].size == (128, 96)
    again, _ = qc.decode_video(clip, 4)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(frames, again))
    row = {"row_id": "v", "messages": [
        {"role": "user", "content": [{"type": "video", "path": str(clip), "num_frames": 4},
                                     {"type": "text", "text": "What moves?"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Bars."}]}]}
    processor = qc.load_processor(tiny_dir, None)
    encoded = qc.encode_vl_row(processor, row, tmp_path)
    assert "pixel_values_videos" in encoded and int(encoded["video_grid_thw"][0, 0]) == 2
    text = processor.tokenizer.decode([t for t in encoded["input_ids"][0].tolist() if t != 248057])
    # indices (0, 10) and (21, 31) at 16 fps: group means 5/16 = 0.3125 s and 26/16 = 1.625 s
    assert "<0.3 seconds>" in text and "<1.6 seconds>" in text, text


def test_smoke_vl_greedy_first_token_is_the_full_load_argmax(tiny_dir, tmp_path):
    work = tmp_path / "work"
    jsonl = qc.make_parity_vl_corpus(work, None)
    receipt = tmp_path / "smoke.json"
    args = argparse.Namespace(**{**vars(build_args(tiny_dir, tmp_path / "unused", jsonl)), "rows": "0,1",
                                 "max_new_tokens": 3, "eos_ids": "248046,248044", "receipt": str(receipt),
                                 "chat_template_kwargs": '{"enable_thinking": false}'})
    assert qc.run_smoke_vl(args) == 0
    out = json.loads(receipt.read_text())
    assert out["teacher_unchanged"] and [r["row"] for r in out["rows"]] == [0, 1]
    assert all(1 <= len(r["answer_ids"]) <= 3 for r in out["rows"])
    assert out["rows"][0]["reference_response"].startswith("A red square")
    # Independent arm: the stock full load, one forward on row 0's generation prompt.
    corpus = qc.load_vl_corpus(jsonl)
    processor = qc.load_processor(tiny_dir, None)
    enc = qc.encode_vl_row(processor, corpus["rows_data"][0], jsonl.parent, {"enable_thinking": False},
                           generation_prompt=True)
    assert processor.tokenizer.decode(enc["input_ids"][0, -6:].tolist()).endswith("<think>\n\n</think>\n\n")
    full = qc.build_full_load_teacher(tiny_dir, "cpu", torch.float32, "sdpa", qc.TEXT_VOCAB_ROWS, "fp32")
    with torch.no_grad():
        hidden = full.vl_hidden({k: v for k, v in enc.items()})
        logits = hidden[0, -1].float() @ full.head_weight.T
    assert int(logits.argmax()) == out["rows"][0]["answer_ids"][0]


# ------------------------------------------------------------------ 1.0.5: host-resident teacher, several corpora per process
def test_host_resident_teacher_writes_the_same_bytes_as_the_streamed_mmap(tiny_dir, tmp_path):
    corpus = write_corpus(tmp_path / "corpus.safetensors", rows=5)
    sha = qc.load_text_corpus(corpus)["sha256"]
    assert qc.build_cache(build_args(tiny_dir, tmp_path / "mmap", corpus, host_resident="off"), "text") == 0
    assert qc.build_cache(build_args(tiny_dir, tmp_path / "resident", corpus, host_resident="pageable"), "text") == 0
    assert cache_bytes(tmp_path / "mmap" / "tiny" / sha) == cache_bytes(tmp_path / "resident" / "tiny" / sha)
    (session,) = session_records(tmp_path / "resident" / "tiny" / sha)
    assert session["host_resident"]["mode"] == "pageable" and session["host_resident"]["tensors"] > 0
    (off,) = session_records(tmp_path / "mmap" / "tiny" / sha)
    assert off["host_resident"] is None


def test_host_resident_reader_holds_owned_copies_of_the_layer_tensors(tiny_dir):
    tps = qc.engine_module()
    reader = tps.ShardReader(tiny_dir, device="cpu")
    resident = qc.HostResidentReader(reader, [f"{tps.TEXT_TRUNK_PREFIX}.layers."])
    names = reader.names_under(f"{tps.TEXT_TRUNK_PREFIX}.layers.")
    assert resident.resident_tensors == len(names) > 0
    for name in names[:6]:
        assert torch.equal(resident.get(name), reader.get(name))
    embed = f"{tps.TEXT_TRUNK_PREFIX}.embed_tokens.weight"
    assert embed not in resident.store and torch.equal(resident.get(embed), reader.get(embed))
    assert resident.names_under(f"{tps.TEXT_TRUNK_PREFIX}.layers.0.") == reader.names_under(f"{tps.TEXT_TRUNK_PREFIX}.layers.0.")


def test_several_corpora_in_one_process_equal_separate_builds(tiny_dir, tmp_path):
    a = write_corpus(tmp_path / "a.safetensors", rows=5, seed=3)
    b = write_corpus(tmp_path / "b.safetensors", rows=3, seed=4)
    sha_a, sha_b = qc.load_text_corpus(a)["sha256"], qc.load_text_corpus(b)["sha256"]
    together = tmp_path / "together"
    assert qc.build_cache(build_args(tiny_dir, together, [str(a), str(b)]), "text") == 0
    for corpus, sha in ((a, sha_a), (b, sha_b)):
        alone = tmp_path / f"alone-{sha[:8]}"
        assert qc.build_cache(build_args(tiny_dir, alone, corpus), "text") == 0
        assert cache_bytes(together / "tiny" / sha) == cache_bytes(alone / "tiny" / sha)
    (sa,) = session_records(together / "tiny" / sha_a)
    (sb,) = session_records(together / "tiny" / sha_b)
    assert sa["session"] == sb["session"]
    assert sa["teacher_digest_before"]["tree_sha256"] == sb["teacher_digest_before"]["tree_sha256"]
    assert sa["teacher_digest_after"]["tree_sha256"] == sb["teacher_digest_after"]["tree_sha256"]
    assert sorted(sa["caches_in_process"]) == sorted([str(together / "tiny" / sha_a), str(together / "tiny" / sha_b)])
    for sha in (sha_a, sha_b):
        manifest = json.loads((together / "tiny" / sha / "manifest.json").read_text())
        assert manifest["all_finite"] and manifest["teacher"]["equal_before_and_after_every_session"]
    # A refused config for the second corpus touches neither cache.
    c = write_corpus(tmp_path / "c.safetensors", rows=2, seed=5)
    d = write_corpus(tmp_path / "d.safetensors", rows=2, seed=6)
    sha_d = qc.load_text_corpus(d)["sha256"]
    fresh = tmp_path / "fresh"
    assert qc.build_cache(build_args(tiny_dir, fresh, d, top_k=16), "text") == 0
    (fresh / "tiny" / sha_d / "manifest.json").unlink()
    with pytest.raises(qc.CacheError, match="top_k"):
        qc.build_cache(build_args(tiny_dir, fresh, [str(c), str(d)]), "text")
    sha_c = qc.load_text_corpus(c)["sha256"]
    assert not (fresh / "tiny" / sha_c / "sessions").exists() and not (fresh / "tiny" / sha_c / "writer.lock").exists()


@pytest.mark.skipif(not hasattr(qc.signal, "SIGKILL"), reason="needs POSIX signals")
def test_a_hard_killed_multi_corpus_process_resumes_and_seals_every_cache(tiny_dir, tmp_path):
    a = write_corpus(tmp_path / "a.safetensors", rows=5, seed=3)
    b = write_corpus(tmp_path / "b.safetensors", rows=5, seed=4)
    sha_a, sha_b = qc.load_text_corpus(a)["sha256"], qc.load_text_corpus(b)["sha256"]
    out = tmp_path / "cache"
    args = build_args(tiny_dir, out, [str(a), str(b)])
    killed = run_stopped_writer(tmp_path, "kill", args)
    assert killed.returncode == -9, killed.stdout + killed.stderr
    # Corpus a died staging its shard 1; corpus b was opened (locked, "before" written) but never read.
    for sha in (sha_a, sha_b):
        (record,) = session_records(out / "tiny" / sha)
        assert "teacher_digest_before" in record and "teacher_digest_after" not in record
        assert (out / "tiny" / sha / "writer.lock").is_file()
    assert qc.build_cache(build_args(tiny_dir, out, [str(a), str(b)], verify_existing=True), "text") == 0
    for corpus, sha in ((a, sha_a), (b, sha_b)):
        manifest = json.loads((out / "tiny" / sha / "manifest.json").read_text())
        (interrupted,) = manifest["sessions_interrupted"]
        assert interrupted["vouched_by"] == manifest["sessions"][0]["session"]
        clean = tmp_path / f"clean-{sha[:8]}"
        assert qc.build_cache(build_args(tiny_dir, clean, corpus), "text") == 0
        assert cache_bytes(out / "tiny" / sha) == cache_bytes(clean / "tiny" / sha)
