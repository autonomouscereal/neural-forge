"""Tests for ``k3_teacher_cache_v130.py`` (k3_teacher_cache 1.3.0, lane k3accel) on the lane's synthetic K3.

Run this file in its OWN pytest process: it binds the module name ``k3_teacher_cache`` to 1.3.0 and then imports
the whole 1.2.4 suite (``test_k3_teacher_cache.py``), so every 1.2.x test runs again against 1.3.0 with --accel on
(the default) -- resume, digests, state scoping, probe mode, parity, xtok, export-reference.  Added for 1.3.0:

  test_accel_is_bit_identical_to_the_frozen_124       fp32 and bf16, several group / row-chunk geometries: per-layer
                                                      prefix sha and every cache tensor's bits vs the 1.2.4 file
  test_moe_group_batches_and_the_oom_fallback_keep_the_bits
  test_a_non_finite_prefix_is_still_the_engines_refusal
  test_accel_off_is_the_124_path
  test_exact_dequant_matches_the_engine_on_every_byte_and_scale
  test_sorted_dispatch_routed_moe_matches_the_engine_bitwise
  test_layer_loader_reads_the_engine_bytes_in_the_engine_order
  test_streaming_writer_round_trips_and_hashes_in_line
  test_async_checkpoint_never_overwrites_an_unwritten_row
  test_resumed_loader_never_reads_a_finished_layer
  test_stage_timers_are_a_probe_option
  test_kda_graph_replay_is_bitwise_and_a_mismatch_falls_back   (CUDA only; skipped on CPU)

Server:
  cd $OC/distill/tools && CUDA_VISIBLE_DEVICES="" PYTHONPATH=$OC/distill/tools \\
      $PY -m pytest -o addopts="" -p no:cacheprovider test_k3_teacher_cache_v130.py -q
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if "k3_teacher_cache" in sys.modules and not sys.modules["k3_teacher_cache"].__file__.endswith("_v130.py"):
    raise RuntimeError("run test_k3_teacher_cache_v130.py in its own pytest process (k3_teacher_cache is bound)")
_spec = importlib.util.spec_from_file_location("k3_teacher_cache", HERE / "k3_teacher_cache_v130.py")
v130 = importlib.util.module_from_spec(_spec)
sys.modules["k3_teacher_cache"] = v130
_spec.loader.exec_module(v130)

from test_k3_teacher_cache import *  # noqa: E402,F401,F403 -- the 1.2.4 suite, bound to 1.3.0
import test_k3_teacher_cache as base  # noqa: E402

assert base.ktc is v130 and v130.TOOL_VERSION == "1.3.0"
_spec124 = importlib.util.spec_from_file_location("k3_teacher_cache_124", HERE / "k3_teacher_cache.py")
v124 = importlib.util.module_from_spec(_spec124)
sys.modules["k3_teacher_cache_124"] = v124
_spec124.loader.exec_module(v124)
assert v124.TOOL_VERSION == "1.2.4"

import k3_common  # noqa: E402
import k3_layer  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

KDP = base.ENGINE.kdp


def run_with(module, argv: list[str]) -> dict:
    return module.CachePass(module.build_parser().parse_args(argv)).run()


def assert_cache_bits_equal(a: dict, b: dict) -> None:
    ca, cb = base.read_cache(a), base.read_cache(b)
    assert ca.keys() == cb.keys() and ca
    for key in ca:
        for k in ("topk_ids", "topk_logprobs", "lse", "target_logprob"):
            assert v130.bits_equal(ca[key][k], cb[key][k]), (key, k)


# ------------------------------------------------------------------ 1.3.0 vs the frozen 1.2.4
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
@pytest.mark.parametrize("geometry", [("--group-tokens", "20"),
                                      ("--group-tokens", "16", "--row-chunk-kda", "3", "--row-chunk-mla", "2"),
                                      ("--group-tokens", "100000", "--no-trim"),
                                      ("--group-tokens", "20", "--moe-group-batch", "1"),
                                      ("--group-tokens", "20", "--moe-group-batch", "3")])
def test_accel_is_bit_identical_to_the_frozen_124(synth, ids_files, work, dtype, geometry):
    tag = f"{dtype}-{'-'.join(geometry).replace('--', '')}"
    only130 = [a for i, a in enumerate(geometry) if a == "--moe-group-batch" or (i and geometry[i - 1] == "--moe-group-batch")]
    shared = [a for a in geometry if a not in only130] if only130 else list(geometry)
    argv = base.run_args(synth, ids_files, work / f"bx124-{tag}", "--prefix-digests", *shared)
    argv[argv.index("--compute-dtype") + 1] = dtype
    old = run_with(v124, argv)
    argv[argv.index("--out-root") + 1] = str(work / f"bx130-{tag}")
    new = run_with(v130, argv + only130)
    assert new["accel"]["enabled"] and new["accel"]["layer_loader"] and new["accel"]["async_checkpoint"]
    assert old["layers"].keys() == new["layers"].keys()
    for layer in old["layers"]:
        assert old["layers"][layer]["prefix_out_sha256"] == new["layers"][layer]["prefix_out_sha256"], (tag, layer)
    assert_cache_bits_equal(old, new)
    assert new["accel"]["loader_record"]["read"]  # the loader did the reads


def test_moe_group_batches_and_the_oom_fallback_keep_the_bits(synth, ids_files, work, monkeypatch):
    """auto on CPU batches every group of a layer; a card out-of-memory inside a multi-group expert loop restarts the
    accumulators one group at a time -- every bit still equal to 1.2.4."""
    argv = base.run_args(synth, ids_files, work / "mgb124", "--prefix-digests", "--group-tokens", "20")
    old = run_with(v124, argv)
    argv[argv.index("--out-root") + 1] = str(work / "mgb130")
    new = run_with(v130, argv)
    moe = [v["accel"]["moe_group_batch"] for v in new["layers"].values() if v["moe"]]
    assert moe and all(b["batches"] == [new["corpus"]["groups"]] for b in moe)  # one batch of all 7 groups
    assert_cache_bits_equal(old, new)
    real = v130.CachePass.moe_experts_accel
    calls = {"oom": 0}

    def flaky(self, weights, plans, cfg, device):
        if len(plans) > 1:
            calls["oom"] += 1
            raise torch.cuda.OutOfMemoryError("test: CUDA out of memory")
        return real(self, weights, plans, cfg, device)

    monkeypatch.setattr(v130.CachePass, "moe_experts_accel", flaky)
    argv[argv.index("--out-root") + 1] = str(work / "mgb130oom")
    fell = run_with(v130, argv)
    assert calls["oom"] == 1 and any(e.get("action") == "moe_group_batch_oom_fallback" for e in fell["events"])
    for layer in old["layers"]:
        assert old["layers"][layer]["prefix_out_sha256"] == fell["layers"][layer]["prefix_out_sha256"]
    assert_cache_bits_equal(old, fell)


def test_a_non_finite_prefix_is_still_the_engines_refusal(synth, ids_files, work, monkeypatch):
    real = KDP.moe_write

    def poisoned(weights, x, y_lat, cfg, chunk):
        out = real(weights, x, y_lat, cfg, chunk)
        if weights.layer_idx == 2:
            out[1, 3] = float("nan")
        return out

    monkeypatch.setattr(KDP, "moe_write", poisoned)
    with pytest.raises(SystemExit, match="layer 2: teacher prefix after the layer carries 1 non-finite values"):
        run_with(v130, base.run_args(synth, ids_files, work / "nanpass", "--group-tokens", "20"))


def test_accel_off_is_the_124_path(synth, ids_files, work):
    argv = base.run_args(synth, ids_files, work / "off124", "--prefix-digests", "--group-tokens", "20")
    old = run_with(v124, argv)
    argv[argv.index("--out-root") + 1] = str(work / "off130")
    new = run_with(v130, argv + ["--accel", "off"])
    assert not new["accel"]["enabled"] and "loader_record" not in new["accel"]
    assert all("accel" not in v for v in new["layers"].values())
    for layer in old["layers"]:
        assert old["layers"][layer]["prefix_out_sha256"] == new["layers"][layer]["prefix_out_sha256"]
    assert_cache_bits_equal(old, new)


# ------------------------------------------------------------------ the pieces
def dequant_domain(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Every packed byte under every E8M0 scale: rows 2s and 2s+1 hold bytes 0..127 and 128..255, scale s."""
    rows = []
    for _ in range(256):
        rows.append(torch.arange(0, 128, dtype=torch.int32))
        rows.append(torch.arange(128, 256, dtype=torch.int32))
    packed = torch.stack(rows).to(torch.uint8).to(device)                       # [512, 128] -> 256 columns
    scale = torch.arange(256, dtype=torch.int32).repeat_interleave(2).to(torch.uint8)
    scale = scale.unsqueeze(1).expand(512, 256 // 32).contiguous().to(device)  # [512, 8]
    return packed, scale


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_exact_dequant_matches_the_engine_on_every_byte_and_scale(device, dtype):
    packed, scale = dequant_domain(device)
    want = k3_common.dequant_mxfp4(packed, scale, dtype=dtype)
    got = v130.dequant_mxfp4_exact(packed, scale, v130.e2m1_pair_table(k3_common, torch.device(device)), dtype=dtype)
    assert v130.bits_equal(got, want)
    # the domain really holds -0.0 (code 8), the largest magnitude, an overflowing scale and NaN (0 x inf)
    bits = want.float()
    assert bool(torch.signbit(bits[bits == 0]).any()) and bool(torch.isinf(bits).any()) and bool(torch.isnan(bits).any())
    # and a one-ulp change of one table value (1.5 -> 1.5 + 2^-7: one bf16 ulp, exact in all three dtypes) is caught
    bad = v130.e2m1_pair_table(k3_common, torch.device(device)).clone()
    assert float(bad[3, 0]) == 1.5
    bad[3, 0] = 1.5 + 2.0 ** -7
    assert not v130.bits_equal(v130.dequant_mxfp4_exact(packed, scale, bad, dtype=dtype), want)


def moe_layer(synth: dict, layer: int = 1):
    index = synth["index"]
    weights, _, _ = KDP.read_layer(index, layer, KDP.ShardStager(synth["dir"], None, enabled=False, window=1,
                                                                    prefetch=0, delete_after_consume=False,
                                                                    run_id="t"),
                                   torch.device("cpu"), torch.device("cpu"), torch.float32)
    assert weights.is_moe
    return weights, index.text_config


def test_sorted_dispatch_routed_moe_matches_the_engine_bitwise(synth):
    weights, cfg = moe_layer(synth)
    gen = torch.Generator().manual_seed(7)
    hidden = int(cfg["hidden_size"])
    latent_dim = int(weights.body["block_sparse_moe.routed_expert_down_proj.weight"].shape[0])
    owner = types.SimpleNamespace(kdp=KDP, clock=v130.StageClock(False), pair_tables={}, common=k3_common)
    owner.pair_table = lambda device: v130.CachePass.pair_table(owner, device)
    owner.moe_route_accel = lambda *a: v130.CachePass.moe_route_accel(owner, *a)
    owner.moe_experts_accel = lambda *a: v130.CachePass.moe_experts_accel(owner, *a)
    for n_tokens in (1, 3, 37, 200):
        x = torch.randn(n_tokens, hidden, generator=gen)
        latent = torch.randn(n_tokens, latent_dim, generator=gen)
        want, _, _ = KDP.routed_y_lat(weights, x, latent, cfg, torch.device("cpu"))
        got, _ = v130.CachePass.routed_y_lat_accel(owner, weights, x, latent, cfg, torch.device("cpu"))
        assert v130.bits_equal(got, want), n_tokens
    # a 1-ulp change in one expert's packed scale is caught (the comparison is not vacuous)
    e = 0
    weights.experts[e]["w1.weight_scale"].view(-1)[0] += 1
    want2, _, _ = KDP.routed_y_lat(weights, x, latent, cfg, torch.device("cpu"))
    assert not v130.bits_equal(want2, got) or int((KDP.route(x, weights, cfg)[0] == e).sum()) == 0


def test_layer_loader_reads_the_engine_bytes_in_the_engine_order(synth):
    index = synth["index"]
    path_for = lambda s: os.path.join(synth["dir"], s)  # noqa: E731
    layers = list(range(index.num_layers))
    loader = v130.LayerLoader(index, k3_common, path_for, layers, pinned=False, fadvise_drop=True, warmer=None,
                              plan_shards={})
    loader.start()
    kept = {}
    try:
        for i in layers:
            loader.request(i)
            got, timing, _ = loader.take(i)
            want = {}
            for shard in index.shards_for(index.layer_keys(i)):
                wanted = [k for k in index.layer_keys(i) if index.weight_map[k] == shard]
                want.update(k3_common.read_tensors(path_for(shard), wanted))
            assert list(got) == list(want), i  # the engine's dict order (shard order, then on-disk offset)
            for k in want:
                assert got[k].dtype == want[k].dtype and v130.bits_equal(got[k], want[k]), (i, k)
            assert timing["bytes"] == sum(t.numel() * t.element_size() for t in want.values())
            kept[i] = {k: v.clone() for k, v in got.items()}
            loader.release()
        # the arena is reused: earlier clones are untouched, and a re-read gives the same bits
        loader.request(0)
        again, _, _ = loader.take(0)
        assert all(v130.bits_equal(again[k], kept[0][k]) for k in kept[0])
        loader.release()
    finally:
        loader.close()
    assert loader.arena_bytes >= max(sum(t.numel() * t.element_size() for t in kept[i].values()) for i in layers)


def test_streaming_writer_round_trips_and_hashes_in_line(work):
    names = {v: k for k, v in k3_common.DTYPES.items()}
    gen = torch.Generator().manual_seed(3)
    stack = torch.randn(1001, 3, 37, generator=gen).to(torch.bfloat16)
    cases = {"bf16": torch.randn(1001, 37, generator=gen).to(torch.bfloat16),
             "fp32": torch.randn(77, 5, generator=gen), "strided_slot": stack[:, 1]}
    for label, tensor in cases.items():
        path = str(work / f"stream_{label}.safetensors")
        seen = []
        digest = v130.write_safetensors_streaming(path, "prefix", tensor, {"b": "2", "a": "1"}, names,
                                                  on_rows=seen.append, chunk_bytes=1000)
        assert digest == v130.sha256_file(path)
        assert v130.bits_equal(load_file(path)["prefix"], tensor.contiguous())
        with safe_open(path, framework="pt") as handle:
            assert handle.metadata() == {"a": "1", "b": "2"}
        assert seen == sorted(seen) and seen[-1] == tensor.shape[0] and len(seen) > 1, label
        assert not list(work.glob(f"stream_{label}.safetensors.tmp*"))


def test_async_checkpoint_never_overwrites_an_unwritten_row(synth, ids_files, work, monkeypatch):
    """A slowed writer: the next layer must wait for it (row waits recorded), the checkpoint after L03 holds exactly
    the prefix after L03, and the cache equals the 1.2.4 run's bits."""
    whole = run_with(v124, base.run_args(synth, ids_files, work / "slow124", "--group-tokens", "20", "--prefix-digests"))
    real_write = v130.write_safetensors_streaming

    def slow_write(path, name, tensor, metadata, dtype_names, on_rows=None, chunk_bytes=0):
        def slow_rows(r):
            time.sleep(0.005)
            if on_rows is not None:
                on_rows(r)
        return real_write(path, name, tensor, metadata, dtype_names, on_rows=slow_rows, chunk_bytes=256)

    monkeypatch.setattr(v130, "write_safetensors_streaming", slow_write)
    real_layer = v130.CachePass.layer

    def dying_layer(self, run, layer_idx, *a, **k):
        if layer_idx == 4:
            raise SystemExit("killed after L03 (test)")
        return real_layer(self, run, layer_idx, *a, **k)

    out = work / "slow130"
    monkeypatch.setattr(v130.CachePass, "layer", dying_layer)
    with pytest.raises(SystemExit, match="killed after L03"):
        run_with(v130, base.run_args(synth, ids_files, out, "--group-tokens", "20", "--prefix-digests"))
    monkeypatch.setattr(v130.CachePass, "layer", real_layer)
    state = json.loads(base.state_json(out).read_text())
    assert state["layer_done"] == 3
    saved = load_file(state["prefix"]["path"])["prefix"]
    assert k3_common.sha256_tensor(saved) == whole["layers"]["3"]["prefix_out_sha256"]
    resumed = run_with(v130, base.run_args(synth, ids_files, out, "--group-tokens", "20", "--prefix-digests",
                                           "--resume"))
    waits = [v["accel"]["checkpoint_row_wait_seconds"] for v in resumed["layers"].values()]
    assert max(waits) > 0  # the main thread did wait for the slowed writer
    assert all(c.get("sha256_inline") for c in resumed["checkpoints"] if "layer_done" in c and "action" not in c)
    assert resumed["layers"]["5"]["prefix_out_sha256"] == whole["layers"]["5"]["prefix_out_sha256"]
    assert_cache_bits_equal(whole, resumed)


def test_resumed_loader_never_reads_a_finished_layer(synth, ids_files, work, monkeypatch):
    index = synth["index"]
    plan = {i: index.shards_for(index.layer_keys(i)) for i in range(index.num_layers)}
    later = {s for i in (4, 5) for s in plan[i]}
    earlier_only = {s for i in range(4) for s in plan[i]} - later
    real_layer = v130.CachePass.layer

    def dying_layer(self, run, layer_idx, *a, **k):
        if layer_idx == 4:
            raise SystemExit("killed after L03 (test)")
        return real_layer(self, run, layer_idx, *a, **k)

    out = work / "loader-resume"
    monkeypatch.setattr(v130.CachePass, "layer", dying_layer)
    with pytest.raises(SystemExit, match="killed after L03"):
        run_with(v130, base.run_args(synth, ids_files, out, "--group-tokens", "20"))
    monkeypatch.setattr(v130.CachePass, "layer", real_layer)
    resumed = run_with(v130, base.run_args(synth, ids_files, out, "--group-tokens", "20", "--resume"))
    read = set(resumed["accel"]["loader_record"]["read"])
    assert read <= later and not read & earlier_only, sorted(read)


def test_stage_timers_are_a_probe_option(synth, ids_files, work):
    with pytest.raises(SystemExit, match="PROBE"):
        run_with(v130, base.run_args(synth, ids_files, work / "st0", "--stage-timers"))
    receipt = run_with(v130, base.run_args(synth, ids_files, work / "st1", "--probe", "--layers", "1-2",
                                           "--synthetic-blocks", "1", "--no-head", "--stage-timers"))
    stages = receipt["layers"]["1"]["stages"]
    for name in ("read", "attn.core", "ffn.moe.route", "ffn.moe.dequant", "ffn.moe.expert_ffn", "finite_push_d2h"):
        assert stages[name]["calls"] >= 1 and stages[name]["wall_s"] >= 0, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a card")
def test_kda_graph_replay_is_bitwise_and_a_mismatch_falls_back():
    gen = torch.Generator().manual_seed(11)
    B, T, H, K = 1, 96, 4, 16
    dev = torch.device("cuda")

    def inputs():
        q = k3_layer.l2norm(torch.randn(B, T, H, K, generator=gen)).float().to(dev)
        k = k3_layer.l2norm(torch.randn(B, T, H, K, generator=gen)).float().to(dev)
        v = torch.randn(B, T, H, K, generator=gen).to(dev)
        g = (-torch.rand(B, T, H, K, generator=gen)).to(dev)
        beta = torch.rand(B, T, H, generator=gen).to(dev)
        return q, k, v, g, beta

    graphed = v130.KdaGraphRecurrence(k3_layer.naive_recurrent_kda, verify="layer", min_free_bytes=0)
    for layer in range(3):
        graphed.layer = layer
        for _ in range(2):
            q, k, v, g, beta = inputs()
            out, state = graphed(q, k, v, g, beta, K ** -0.5)
            e_out, e_state = k3_layer.naive_recurrent_kda(q, k, v, g, beta, K ** -0.5)
            assert v130.bits_equal(out, e_out) and v130.bits_equal(state, e_state)
    rec = graphed.record()
    assert rec["graphs"] == 1 and rec["counts"]["replays"] == 6 and rec["all_verified_equal"] and not rec["disabled"]

    def drifting(q, k, v, g, beta, scale, initial_state=None):  # 1 ulp off whenever it runs OUTSIDE the capture
        out, state = k3_layer.naive_recurrent_kda(q, k, v, g, beta, scale, initial_state=initial_state)
        if not torch.cuda.is_current_stream_capturing():
            out = out.clone()
            out.view(-1)[:1].view(torch.int32).add_(1)
        return out, state

    graphed = v130.KdaGraphRecurrence(drifting, verify="layer", min_free_bytes=0)
    graphed.layer = 7
    q, k, v, g, beta = inputs()
    out, _ = graphed(q, k, v, g, beta, K ** -0.5)
    want, _ = drifting(q, k, v, g, beta, K ** -0.5)
    assert graphed.disabled and graphed.disabled["layer"] == 7 and v130.bits_equal(out, want)
    graphed(q, k, v, g, beta, K ** -0.5)
    assert graphed.counts["eager"] >= 1  # disabled: eager from then on
