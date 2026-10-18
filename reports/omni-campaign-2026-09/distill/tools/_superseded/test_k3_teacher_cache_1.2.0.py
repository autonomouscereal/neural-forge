"""Tests for ``k3_teacher_cache.py`` on the lane's synthetic K3 (``k3_synth``), CPU, fp32.

The fixture is 6 layers at attn_res_block_size 3: layers 0/3 are snapshot layers, 3 and 5 are MLA
(full attention every 4th one-based layer and the last), the rest KDA; layer 0 is the dense MLP, 1-5
are latent MoE with MXFP4 experts -- every branch the real 93-layer pass takes.

  test_pass_matches_the_lane_whole_layer_reference   the cache vs StreamedLayer.forward chained over the
                                                     whole stack + the output AttnRes head, per row
  test_groups_trim_and_row_chunks_do_not_change_the_cache
  test_kill_after_a_layer_and_resume_reproduces_the_single_run
  test_pass_plan_is_contiguous_balanced_and_the_token_guard_refuses
  test_cache_layout_is_196_bytes_core_and_the_manifest_seals
  test_source_change_during_the_pass_is_a_refusal / test_fold_receipt_drift_is_refused_up_front
  test_inline_digest_matches_and_a_wrong_sha_is_named
  test_inline_digest_refuses_a_missing_or_incomplete_receipt / test_a_missing_fold_receipt_is_refused
  test_resume_never_rereads_a_finished_shard
  test_vocab_map_and_xtok_tail_bucket
  test_probe_bits_per_byte_on_the_same_span
  test_parity_identity_passes_and_a_perturbed_cache_fails
  test_probe_mode_with_synthetic_blocks_records_rss_and_writes_no_cache
  test_engine_is_pinned

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
      $PY -m pytest -o addopts="" -p no:cacheprovider test_k3_teacher_cache.py -q
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import k3_teacher_cache as ktc  # noqa: E402

ENGINE = ktc.Engine(os.environ.get("K3_ENGINE_DIR", ktc.DEFAULT_ENGINE_DIR))
import k3_common  # noqa: E402
import k3_layer  # noqa: E402
import k3_synth  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

VOCAB = 128


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("k3_teacher_cache")


@pytest.fixture(scope="module")
def synth(work: Path) -> dict:
    out = work / "synthetic-k3"
    argv = ["--out-dir", str(out), "--layers", "6", "--hidden", "64", "--experts", "8", "--top-k", "2",
            "--attn-res-block-size", "3", "--latent", "32", "--moe-intermediate", "64", "--vocab-size", str(VOCAB)]
    k3_synth.build(k3_synth.build_parser().parse_args(argv))
    index = k3_common.K3Index(str(out))
    # the fold receipt of record for the synthetic source: its resume_context, as k3_dense_pass writes it
    stat = ktc.source_stat(index, str(out))
    fold = work / "synthetic_fold_receipt.json"
    fold.write_text(json.dumps({"resume_context": {"model": {"source_file_metadata": stat["files"],
                                                             "config_sha256": stat["config_sha256"],
                                                             "index_sha256": stat["index_sha256"]}}}))
    return {"dir": str(out), "index": index, "fold_receipt": str(fold)}


def digest_receipt(synth: dict) -> dict:
    """k3_lexar_digest_verify.json's layout for the synthetic shards, every file VERIFIED."""
    files = []
    for p in sorted(Path(synth["dir"]).glob("model-*.safetensors")):
        sha = ktc.sha256_file(str(p))
        files.append({"file": p.name, "hf_sha256": sha, "hf_size": p.stat().st_size, "local_sha256": sha,
                      "local_size": p.stat().st_size, "status": "VERIFIED"})
    return {"all_verified": True, "files": files}


def write_ids(path: Path, rows: int, seq: int, lengths: list[int], seed: int) -> Path:
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (rows, seq), generator=gen, dtype=torch.long)
    lens = torch.tensor(lengths, dtype=torch.long)
    for r, n in enumerate(lengths):
        ids[r, n:] = 0
    save_file({"input_ids": ids, "real_lengths": lens}, str(path))
    return path


@pytest.fixture(scope="module")
def ids_files(work: Path) -> list[str]:
    a = write_ids(work / "k3_input_ids.safetensors", 3, 24, [24, 17, 9], seed=1)
    b = write_ids(work / "k3_review_input_ids.safetensors", 4, 16, [5, 16, 2, 11], seed=2)
    return [str(a), str(b)]


def run_args(synth: dict, ids: list[str], out: Path, *extra: str) -> list[str]:
    argv = ["run", "--src-dir", synth["dir"], "--out-root", str(out), "--device", "cpu", "--experts-device", "cpu",
            "--compute-dtype", "fp32", "--kda-backend", "torch", "--fold-receipt", synth["fold_receipt"],
            "--digest-receipt", "",
            "--no-prefetch-next", "--min-free-gib", "0", "--token-chunk", "7"]
    for p in ids:
        argv += ["--input-ids", p]
    return argv + list(extra)


def run_pass(argv: list[str]) -> dict:
    args = ktc.build_parser().parse_args(argv)
    return ktc.CachePass(args).run()


def read_cache(receipt: dict) -> dict:
    out = {}
    names = [f["name"] for f in receipt["corpus"]["files"]]
    for shard in receipt["shards"]:
        got = load_file(shard["path"])
        pos = 0
        for i in range(got["real_lengths"].numel()):
            n = int(got["real_lengths"][i])
            out[(names[int(got["row_file"][i])], int(got["row_index"][i]))] = {
                k: got[k][pos:pos + n] for k in ("topk_ids", "topk_logprobs", "lse", "target_logprob")}
            pos += n
    return out


# ------------------------------------------------------------------ the independent reference
def reference_logits(synth: dict, ids: torch.Tensor) -> torch.Tensor:
    """The lane's WHOLE-layer forward (StreamedLayer.forward, not the engine's halves), one row at a time."""
    index = synth["index"]
    cfg = index.text_config
    eps = float(cfg["rms_norm_eps"])

    def model_tensor(name):
        return k3_common.read_tensors(os.path.join(synth["dir"], index.weight_map[name]), [name])[name].float()

    table = model_tensor("language_model.model.embed_tokens.weight")
    prefix = table[ids].unsqueeze(0)  # [1, T, H]
    blocks = prefix.new_zeros(prefix.shape[1], 0, prefix.shape[2])
    for layer in range(index.num_layers):
        weights, _ = k3_layer.load_layer(index, layer, "cpu", "cpu", log_reads=False)
        for group in (weights.attn, weights.res, weights.body):
            for key, t in group.items():
                if t.is_floating_point():
                    group[key] = t.float()
        runner = k3_layer.StreamedLayer(weights, cfg, kda_backend="torch")
        prefix, blocks, _ = runner.forward(prefix, blocks, collect=False)
    hidden = k3_common.apply_attn_res(prefix.view(-1, prefix.shape[-1]), blocks,
                                      model_tensor("language_model.model.output_attn_res_proj.weight"),
                                      model_tensor("language_model.model.output_attn_res_norm.weight"), eps)
    hidden = k3_common.rms_norm(hidden, model_tensor("language_model.model.norm.weight"), eps)
    return torch.nn.functional.linear(hidden, model_tensor("language_model.lm_head.weight"))


# ------------------------------------------------------------------ tests
def test_pass_matches_the_lane_whole_layer_reference(synth, ids_files, work):
    receipt = run_pass(run_args(synth, ids_files, work / "c1", "--group-tokens", "20", "--row-chunk-kda", "2"))
    assert receipt["verdict"] == "sealed"
    cache = read_cache(receipt)
    worst = 0.0
    for path in ids_files:
        got = load_file(path)
        name = Path(path).stem
        for r in range(got["input_ids"].shape[0]):
            n = int(got["real_lengths"][r])
            logits = reference_logits(synth, got["input_ids"][r, :n])
            lse = torch.logsumexp(logits, -1)
            lp = logits - lse[:, None]
            top = lp.topk(ktc.TOPK, -1)
            row = cache[(name, r)]
            assert row["lse"].numel() == n
            assert torch.allclose(row["lse"], lse, atol=2e-5, rtol=0)
            # bf16 storage of the logprobs: compare at bf16 resolution
            assert torch.allclose(row["topk_logprobs"].float(), top.values, atol=2e-2, rtol=0)
            assert (row["topk_ids"][:, 0].long() == top.indices[:, 0]).all()
            target = got["input_ids"][r, 1:n]
            want = lp[: n - 1].gather(1, target[:, None]).squeeze(1)
            assert torch.allclose(row["target_logprob"][: n - 1], want, atol=2e-5, rtol=0)
            assert float(row["target_logprob"][n - 1]) == 0.0
            worst = max(worst, float((row["lse"] - lse).abs().max()))
    assert worst < 2e-5


def test_groups_trim_and_row_chunks_do_not_change_the_cache(synth, ids_files, work):
    a = read_cache(run_pass(run_args(synth, ids_files, work / "g1", "--group-tokens", "100000", "--no-trim")))
    b = read_cache(run_pass(run_args(synth, ids_files, work / "g2", "--group-tokens", "16", "--row-chunk-kda", "3",
                                     "--row-chunk-mla", "2", "--head-row-chunk", "2")))
    assert a.keys() == b.keys()
    for key in a:
        for k in ("lse", "target_logprob"):
            assert torch.allclose(a[key][k], b[key][k], atol=2e-5, rtol=0), (key, k)
        assert torch.allclose(a[key]["topk_logprobs"].float(), b[key]["topk_logprobs"].float(), atol=1e-2)


def test_kill_after_a_layer_and_resume_reproduces_the_single_run(synth, ids_files, work, monkeypatch):
    whole = run_pass(run_args(synth, ids_files, work / "r1", "--group-tokens", "20", "--prefix-digests"))
    out = work / "r2"
    real_layer = ktc.CachePass.layer

    def dying_layer(self, run, layer_idx, *a, **k):
        if layer_idx == 4:
            raise SystemExit("killed after L03 (test)")
        return real_layer(self, run, layer_idx, *a, **k)

    monkeypatch.setattr(ktc.CachePass, "layer", dying_layer)
    with pytest.raises(SystemExit, match="killed after L03"):
        run_pass(run_args(synth, ids_files, out, "--group-tokens", "20", "--prefix-digests"))
    monkeypatch.setattr(ktc.CachePass, "layer", real_layer)
    state = json.loads((next(out.glob("*/pass-0-of-1")) / "state" / "state.json").read_text())
    assert state["layer_done"] == 3 and state["n_blocks"] == 2  # pushes at L00 and L03
    resumed = run_pass(run_args(synth, ids_files, out, "--group-tokens", "20", "--prefix-digests", "--resume"))
    assert resumed["start"] == {"from": "checkpoint", "layer_done": 3}
    assert resumed["layers"]["5"]["prefix_out_sha256"] == whole["layers"]["5"]["prefix_out_sha256"]
    a, b = read_cache(whole), read_cache(resumed)
    for key in a:
        for k in ("topk_ids", "topk_logprobs", "lse", "target_logprob"):
            assert torch.equal(a[key][k], b[key][k]), (key, k)
    assert not (next(out.glob("*/pass-0-of-1")) / "state").exists()  # sealed: transient state removed


def test_pass_plan_is_contiguous_balanced_and_the_token_guard_refuses(synth, ids_files, work):
    files = [ktc.load_ids_file(p) for p in ids_files]
    plan = ktc.plan_passes(files, 3)
    flat = [x for p in plan for x in p]
    assert flat == [(fi, r) for fi, f in enumerate(files) for r in range(f["rows"])]
    tokens = [sum(int(files[fi]["real_lengths"][r]) for fi, r in p) for p in plan]
    assert max(tokens) - min(tokens) <= 24
    with pytest.raises(SystemExit, match="max-pass-tokens"):
        run_pass(run_args(synth, ids_files, work / "guard", "--max-pass-tokens", "50"))
    receipt = run_pass(run_args(synth, ids_files, work / "p3", "--pass-count", "2", "--pass-index", "1"))
    rows = {(s["segment"], s["rows"]) for s in receipt["shards"]}
    assert receipt["corpus"]["pass_index"] == 1 and rows


def test_cache_layout_is_196_bytes_core_and_the_manifest_seals(synth, ids_files, work):
    assert ktc.CORE_BYTES_PER_TOKEN == 196
    receipt = run_pass(run_args(synth, ids_files, work / "lay"))
    real = sum(int(load_file(p)["real_lengths"].sum()) for p in ids_files)
    assert receipt["cache"]["positions"] == real
    assert receipt["cache"]["core_bytes"] == real * 196
    shard = load_file(receipt["shards"][0]["path"])
    assert shard["topk_ids"].dtype == torch.int32 and shard["topk_ids"].shape[1] == 32
    assert shard["topk_logprobs"].dtype == torch.bfloat16 and shard["lse"].dtype == torch.float32
    assert shard["target_logprob"].dtype == torch.float32
    per_token = sum(shard[k].element_size() * shard[k][0].numel() for k in ("topk_ids", "topk_logprobs", "lse"))
    assert per_token == 196
    manifest = Path(receipt["shards"][0]["path"]).parent.parent / "manifest.json"
    sidecar = (manifest.parent / "manifest.json.sha256").read_text().split()[0]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == sidecar
    sealed = json.loads(manifest.read_text())
    assert sealed["verdict"] == "sealed" and sealed["integrity"]["after_equals_before"]
    assert all(s["finite"]["positions_match_real_lengths"] for s in sealed["shards"])
    assert sealed["engine"]["sha256"] == ktc.ENGINE_EXPECTED_SHA256


def test_source_change_during_the_pass_is_a_refusal(synth, ids_files, work, monkeypatch):
    src = work / "synthetic-k3-touched"
    shutil.copytree(synth["dir"], src)
    real_head = ktc.CachePass.head

    def touching_head(self, *a, **k):
        shard = sorted(p for p in src.glob("model-*.safetensors"))[1]
        st = shard.stat()
        os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        return real_head(self, *a, **k)

    monkeypatch.setattr(ktc.CachePass, "head", touching_head)
    argv = run_args({"dir": str(src), "fold_receipt": synth["fold_receipt"]}, ids_files, work / "touched")
    with pytest.raises(SystemExit, match="teacher integrity failed"):
        run_pass(argv)


def test_fold_receipt_drift_is_refused_up_front(synth, ids_files, work):
    index = synth["index"]
    stat = ktc.source_stat(index, synth["dir"])
    files = {k: dict(v) for k, v in stat["files"].items()}
    first = sorted(files)[0]
    files[first]["mtime_ns"] += 1
    fake = work / "fold_receipt.json"
    fake.write_text(json.dumps({"resume_context": {"model": {"source_file_metadata": files,
                                                             "config_sha256": stat["config_sha256"],
                                                             "index_sha256": stat["index_sha256"]}}}))
    argv = run_args(synth, ids_files, work / "drift")
    argv[argv.index("--fold-receipt") + 1] = str(fake)
    with pytest.raises(SystemExit, match="differs from the fold receipt"):
        run_pass(argv)
    files[first]["mtime_ns"] -= 1
    fake.write_text(json.dumps({"resume_context": {"model": {"source_file_metadata": files}}}))
    assert run_pass(argv)["integrity"]["drift_vs_fold_receipt"] == []


def test_inline_digest_matches_and_a_wrong_sha_is_named(synth, ids_files, work):
    shards = sorted(Path(synth["dir"]).glob("model-*.safetensors"))
    good = digest_receipt(synth)
    receipt_path = work / "digest_ok.json"
    receipt_path.write_text(json.dumps(good))
    argv = run_args(synth, ids_files, work / "dig1", "--inline-digest", "--prefetch-next")
    argv[argv.index("--digest-receipt") + 1] = str(receipt_path)
    got = run_pass(argv)["integrity"]["inline_digest"]
    assert got["all_hashed_match"] and got["checked_against_receipt"] == len(shards) == got["hashed"]
    assert not got["mismatches"] and not got["unchecked"] and not got["not_hashed"] and got["compares"] == "hf_sha256"
    # a receipt that is itself wrong (VERIFIED, hf == local, but not these bytes): named at the seal
    bad = json.loads(json.dumps(good))
    bad["files"][2]["hf_sha256"] = bad["files"][2]["local_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(bad))
    argv[argv.index("--out-root") + 1] = str(work / "dig2")
    with pytest.raises(SystemExit, match=f"inline digest mismatch {shards[2].name}"):
        run_pass(argv)


class FakeSide:
    def __init__(self, table: dict[int, bytes | None], specials: set[int], separator: int, vocab: int):
        self.table, self.special_ids, self.separator_id, self.vocab_size = table, frozenset(specials), separator, vocab

    def token_bytes(self, i):
        if i == self.separator_id:
            return None
        if i in self.special_ids:
            return b"<|special|>" if i == 7 else b""
        if i not in self.table:
            raise SystemExit("undefined")
        return self.table[i]


def test_vocab_map_and_xtok_tail_bucket():
    k3 = FakeSide({0: b"a", 1: b"bc", 2: b"zz", 3: b"<|special|>", 5: b"q"}, {4}, 6, 8)
    # qwen: 7 is a special spelling the literal name; 9 is >= the text-row limit
    qwen = FakeSide({0: b"bc", 1: b"a", 2: b"x", 9: b"q"}, {7}, 8, 10)
    mapping, stats = ktc.build_vocab_map(k3, qwen, 8, qwen_limit=9)
    assert mapping.tolist() == [1, 0, -1, -1, -1, -1, 8, -1]
    assert stats["k3_mapped"] == 3

    class Cache:
        rows = {("k3_x", 0): {"topk_ids": torch.tensor([[0, 2, 1], [1, 6, 2], [5, 0, 1], [2, 2, 2]], dtype=torch.int32),
                              "topk_logprobs": torch.log(torch.tensor([[0.5, 0.3, 0.1], [0.6, 0.2, 0.1],
                                                                       [0.9, 0.05, 0.01], [0.1, 0.1, 0.1]])).bfloat16(),
                              "lse": torch.zeros(4), "target_logprob": torch.zeros(4)}}

    boundary = {"rows": [{"file": "k3_x", "row_index": 0, "qwen_tokens_real": 5, "k3_tokens_real": 4,
                          "pairs": [[0, 0], [1, 1], [3, 2], [4, 3]]}]}
    got = ktc.xtok_rows(Cache(), boundary, mapping, None, None)
    t = got["tensors"]
    assert t["k3_pos"].tolist() == [0, 1, 2] and t["q_pos"].tolist() == [0, 1, 3]  # last pair has no next token
    assert t["qwen_ids"].tolist() == [[1, -1, 0], [0, 8, -1], [-1, 1, 0]]
    p = torch.tensor([[0.5, 0.3, 0.1], [0.6, 0.2, 0.1], [0.9, 0.05, 0.01]]).bfloat16().float()
    want = torch.stack([p[0, 0] + p[0, 2], p[1, 0] + p[1, 1], p[2, 1] + p[2, 2]])
    assert torch.allclose(t["matched_mass"], want, atol=1e-2)
    assert torch.allclose(t["tail_mass"], 1 - t["matched_mass"])
    assert got["qwen_files"] == ["qwen_x"]


def test_probe_bits_per_byte_on_the_same_span():
    k3_len = torch.tensor([1, 2, 3, 0, 4])
    q_len = torch.tensor([2, 1, 3, 4, 0, 0])
    k3_ids = {"k3_x": torch.tensor([[0, 1, 2, 4, 3]])}
    qwen_ids = {"qwen_x": torch.tensor([[1, 0, 2, 3, 4]])}

    class Cache:
        rows = {("k3_x", 0): {"target_logprob": torch.tensor([-1.0, -2.0, -0.5, -0.25, 0.0])}}

    boundary = {"rows": [{"file": "k3_x", "row_index": 0, "qwen_tokens_real": 5, "k3_tokens_real": 5,
                          "pairs": [[0, 0], [2, 2], [3, 3]]}]}
    q_target = {"qwen_x": torch.tensor([[-0.5, -0.5, -1.0, 0.0, 0.0]])}
    got = ktc.probe_rows(Cache(), boundary, k3_ids, k3_len, {("k3_x", 0): "code"}, q_target, qwen_ids, q_len)
    rec = got["rows"][0]
    # span = tokens 1..3 on K3: bytes 2 + 3 + 4 = 9; Q tokens 1..3: 2 + 3 + 4 = 9
    assert rec["bytes"] == 9 and got["q_span_byte_mismatches"] == 0
    assert math.isclose(rec["k3_bpb"], 3.5 / math.log(2) / 9)
    assert math.isclose(rec["q_bpb"], 2.0 / math.log(2) / 9)
    agg = got["per_route"]["code"]
    assert math.isclose(agg["k3_minus_q_bpb"], (3.5 - 2.0) / math.log(2) / 9)


def test_parity_identity_passes_and_a_perturbed_cache_fails(synth, ids_files, work):
    receipt = run_pass(run_args(synth, ids_files, work / "par"))
    pass_dir = str(Path(receipt["shards"][0]["path"]).parent.parent)
    cache = ktc.CacheReader([pass_dir])
    name = "k3_review_input_ids"
    got = load_file(ids_files[1])
    payload = {k: [] for k in ("teacher_topk_ids", "teacher_topk_logprobs", "teacher_argmax",
                               "teacher_next_token_logprob", "row_index", "position")}
    for r in range(got["input_ids"].shape[0]):
        n = int(got["real_lengths"][r])
        if n < 2:
            continue
        row = cache.rows[(name, r)]
        payload["teacher_topk_ids"].append(row["topk_ids"][: n - 1, :16])
        payload["teacher_topk_logprobs"].append(row["topk_logprobs"][: n - 1, :16].float())
        payload["teacher_argmax"].append(row["topk_ids"][: n - 1, 0])
        payload["teacher_next_token_logprob"].append(row["target_logprob"][: n - 1])
        payload["row_index"].append(torch.full((n - 1,), r, dtype=torch.int32))
        payload["position"].append(torch.arange(n - 1, dtype=torch.int32))
    ref = work / f"{name}_teacher_forced.safetensors"
    save_file({k: torch.cat(v).contiguous() for k, v in payload.items()}, str(ref))
    m = ktc.parity_metrics(cache, str(ref))
    verdict = ktc.parity_verdict([m])
    assert verdict["verdict"] == "pass" and m["top1_agreement"] == 1.0 and m["target_logprob_abs_diff"]["max"] == 0.0
    assert m["positions_compared"] == sum(int(x) - 1 for x in got["real_lengths"] if int(x) >= 2)
    bent = {k: torch.cat(v).contiguous() for k, v in payload.items()}
    bent["teacher_argmax"] = (bent["teacher_argmax"] + 1) % VOCAB
    bent["teacher_next_token_logprob"] = bent["teacher_next_token_logprob"] - 2.0
    save_file(bent, str(ref))
    verdict = ktc.parity_verdict([ktc.parity_metrics(cache, str(ref))])
    assert verdict["verdict"] == "fail" and not verdict["gates"]["top1_agreement"]
    assert not verdict["gates"]["target_logprob_median"]


def test_probe_mode_with_synthetic_blocks_records_rss_and_writes_no_cache(synth, ids_files, work):
    with pytest.raises(SystemExit, match="PROBE"):
        run_pass(run_args(synth, ids_files, work / "pr0", "--layers", "1-2"))
    receipt = run_pass(run_args(synth, ids_files, work / "pr", "--probe", "--layers", "1-2",
                                "--synthetic-blocks", "3", "--no-head"))
    assert set(receipt["layers"]) == {"1", "2"}
    assert all(v["blocks_after"] == 3 for v in receipt["layers"].values())
    assert receipt["host_state"]["block_slots"] == 3
    assert all(v["peak_rss_bytes"] for v in receipt["layers"].values())
    assert "shards" not in receipt
    assert list((work / "pr").glob("*/probe-*/probe_receipt.json"))
    assert not list((work / "pr").glob("*/pass-*"))


def test_engine_is_pinned(work):
    assert ENGINE.version == ktc.ENGINE_EXPECTED_VERSION and ENGINE.sha256 == ktc.ENGINE_EXPECTED_SHA256
    other = work / "other_engine"
    other.mkdir(exist_ok=True)
    shutil.copy(ENGINE.path, other / "k3_dense_pass.py")
    with pytest.raises(SystemExit, match="already imported"):
        ktc.Engine(str(other))


def test_parity_start_from_a_reference_snapshot_reproduces_the_window(synth, ids_files, work, monkeypatch):
    """--start-snapshot (the fold engine's before_L{l} layout) over a window + head == the full pass."""
    whole = read_cache(run_pass(run_args(synth, ids_files, work / "s1")))
    out = work / "s2"
    real_layer = ktc.CachePass.layer

    def dying_layer(self, run, layer_idx, *a, **k):
        if layer_idx == 3:
            raise SystemExit("killed before L03 (test)")
        return real_layer(self, run, layer_idx, *a, **k)

    monkeypatch.setattr(ktc.CachePass, "layer", dying_layer)
    with pytest.raises(SystemExit, match="killed before L03"):
        run_pass(run_args(synth, ids_files, out))
    monkeypatch.setattr(ktc.CachePass, "layer", real_layer)
    state_dir = next(out.glob("*/pass-0-of-1")) / "state"
    state = json.loads((state_dir / "state.json").read_text())
    assert state["layer_done"] == 2 and state["n_blocks"] == 1
    prefix = load_file(state["prefix"]["path"])["prefix"]
    blocks = torch.stack([load_file(b["path"])["block"] for b in state["blocks"]], dim=1)
    snap = work / "before_L03.safetensors"
    save_file({"prefix_sum": prefix, "block_residual": blocks}, str(snap),
              metadata={"layer": "3", "stream": "teacher", "tool": "k3_dense_pass"})
    rows_a = load_file(ids_files[0])["input_ids"].shape[0]
    layout = f"k3_input_ids=0:24,k3_review_input_ids={rows_a * 24}:16"
    with pytest.raises(SystemExit, match="PARITY probe"):
        run_pass(run_args(synth, ids_files, work / "s3", "--start-snapshot", str(snap), "--snapshot-layout", layout))
    receipt = run_pass(run_args(synth, ids_files, work / "s3", "--probe", "--layers", "3-5",
                                "--start-snapshot", str(snap), "--snapshot-layout", layout))
    assert receipt["kind"] == "parity_from_reference_snapshot" and receipt["start"]["from"] == "reference_snapshot"
    window = read_cache(receipt)
    assert window.keys() == whole.keys()
    for key in whole:
        for k in ("lse", "target_logprob"):
            assert torch.allclose(window[key][k], whole[key][k], atol=1e-6, rtol=0), (key, k)
        assert torch.equal(window[key]["topk_ids"][:, 0], whole[key]["topk_ids"][:, 0])


def test_inline_digest_refuses_a_missing_or_incomplete_receipt(synth, ids_files, work, monkeypatch):
    """--inline-digest verifies EVERY shard of the pass or refuses: up front for a receipt that cannot, at the
    seal for a shard that was never hashed."""
    shards = sorted(Path(synth["dir"]).glob("model-*.safetensors"))
    receipt_path = work / "digest_partial.json"
    argv = run_args(synth, ids_files, work / "digm", "--inline-digest", "--prefetch-next")
    argv[argv.index("--digest-receipt") + 1] = str(work / "no_such_digest_receipt.json")
    with pytest.raises(SystemExit, match="no digest receipt"):
        run_pass(argv)
    assert not list((work / "digm").glob("*/pass-*/pass_receipt.partial.json"))  # refused before any layer
    argv[argv.index("--digest-receipt") + 1] = str(receipt_path)
    cases = []
    incomplete = digest_receipt(synth)
    del incomplete["files"][3]
    cases.append((incomplete, f"{shards[3].name} is not in the digest receipt"))
    mismatch = digest_receipt(synth)  # the verifier's own MISMATCH row: local_sha256 is the corrupt bytes' hash
    mismatch["files"][1].update({"status": "MISMATCH", "hf_sha256": "f" * 64})
    cases.append((mismatch, f"{shards[1].name} has status 'MISMATCH'"))
    split = digest_receipt(synth)
    split["files"][4]["hf_sha256"] = "e" * 64
    cases.append((split, f"{shards[4].name}: the digest receipt's local_sha256 != hf_sha256"))
    for i, (payload, message) in enumerate(cases):
        receipt_path.write_text(json.dumps(payload))
        argv[argv.index("--out-root") + 1] = str(work / f"digm{i}")
        with pytest.raises(SystemExit, match=message):
            run_pass(argv)
        assert not list((work / f"digm{i}").glob("*/pass-*/pass_receipt.partial.json"))
    # a complete receipt, but the warmer never reads one shard: not sealed
    receipt_path.write_text(json.dumps(digest_receipt(synth)))
    argv[argv.index("--out-root") + 1] = str(work / "digm-unhashed")
    real_init = ktc.ShardWarmer.__init__

    def lossy_init(self, *a, **k):
        real_init(self, *a, **k)
        self.skip.add(shards[5].name)

    monkeypatch.setattr(ktc.ShardWarmer, "__init__", lossy_init)
    with pytest.raises(SystemExit, match=f"inline digest never hashed {shards[5].name}"):
        run_pass(argv)


def test_a_missing_fold_receipt_is_refused(synth, ids_files, work):
    argv = run_args(synth, ids_files, work / "nofold")
    for missing in (str(work / "no_such_fold_receipt.json"), ""):
        argv[argv.index("--fold-receipt") + 1] = missing
        with pytest.raises(SystemExit, match="no usable fold receipt"):
            run_pass(argv)
    receipt = run_pass(argv + ["--allow-source-drift"])
    assert receipt["verdict"] == "sealed" and receipt["integrity"]["fold_receipt_unusable_override"] is True


def test_resume_never_rereads_a_finished_shard(synth, ids_files, work, monkeypatch):
    """Killed before L04, resumed with the warmer on: no shard only L00-03 read is opened again (with
    --inline-digest its digest comes from the checkpoint) and the seal still covers every shard of the pass."""
    index = synth["index"]
    receipt_path = work / "digest_resume.json"
    receipt_path.write_text(json.dumps(digest_receipt(synth)))
    plan = {i: index.shards_for(index.layer_keys(i)) for i in range(index.num_layers)}
    head = {index.weight_map[k] for k in ("language_model.lm_head.weight", "language_model.model.norm.weight",
                                          "language_model.model.embed_tokens.weight")}
    later = {s for i in (4, 5) for s in plan[i]} | head
    earlier_only = {s for i in range(4) for s in plan[i]} - later
    assert earlier_only  # the fixture has shards no remaining layer reads
    real_layer = ktc.CachePass.layer

    def dying_layer(self, run, layer_idx, *a, **k):
        if layer_idx == 4:
            raise SystemExit("killed after L03 (test)")
        return real_layer(self, run, layer_idx, *a, **k)

    for label, extra in (("prefetch", ["--prefetch-next"]),
                         ("digest", ["--prefetch-next", "--inline-digest", "--digest-receipt", str(receipt_path)])):
        out = work / f"resume-{label}"
        whole = run_pass(run_args(synth, ids_files, work / f"resume-{label}-whole", "--group-tokens", "20", *extra))
        monkeypatch.setattr(ktc.CachePass, "layer", dying_layer)
        with pytest.raises(SystemExit, match="killed after L03"):
            run_pass(run_args(synth, ids_files, out, "--group-tokens", "20", *extra))
        monkeypatch.setattr(ktc.CachePass, "layer", real_layer)
        state = json.loads((next(out.glob("*/pass-0-of-1")) / "state" / "state.json").read_text())
        resumed = run_pass(run_args(synth, ids_files, out, "--group-tokens", "20", "--resume", *extra))
        assert resumed["start"] == {"from": "checkpoint", "layer_done": 3}
        warm = resumed["integrity"]["warmer"]
        read = set(warm["read"])
        if label == "prefetch":
            assert read <= later and not read & earlier_only, (label, sorted(read))
            assert set(warm["skipped"]) == earlier_only
        else:
            prior = {s for s, r in (state["digests"] or {}).items() if r and r.get("sha256")}
            assert prior & earlier_only  # the checkpoint holds at least one finished shard's digest
            assert set(warm["skipped"]) == set(warm["digests_from_checkpoint"]) == prior & earlier_only
            assert read <= later | (earlier_only - prior), (label, sorted(read))
            inline = resumed["integrity"]["inline_digest"]
            assert inline["all_hashed_match"] and inline["hashed"] == inline["checked_against_receipt"] == warm["order"]
            want = whole["integrity"]["inline_digest"]["results"]
            assert {s: r["sha256"] for s, r in inline["results"].items()} == {s: r["sha256"] for s, r in want.items()}
        a, b = read_cache(whole), read_cache(resumed)
        for key in a:
            for k in ("topk_ids", "topk_logprobs", "lse", "target_logprob"):
                assert torch.equal(a[key][k], b[key][k]), (label, key, k)
