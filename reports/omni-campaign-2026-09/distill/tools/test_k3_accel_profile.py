"""Tests for k3_accel_profile.py: the timed copies compute what the engine computes, bit for bit, and the stage
accounting is right.  The engine tests import the real fold engine (k3_dense_pass 1.2.5) by path and run on CPU, so
they need the server tree; they skip elsewhere.

  CUDA_VISIBLE_DEVICES= /home/cereal/model-surgery-lab/venv/bin/python -m pytest -q test_k3_accel_profile.py
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import k3_accel_profile as kap  # noqa: E402

ENGINE_DIR = os.environ.get(
    "K3_ENGINE_DIR",
    "/home/cereal/omni-campaign/runtime/k3-continuation-oct08-claude/code/reports/omni-campaign-2026-09/composite/tools")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_stages_nesting_exclusive_and_replay():
    clock = FakeClock()
    st = kap.Stages(sync=False, clock=clock, cpu_clock=clock)
    st.layer = 7
    with st.stage("attn"):
        clock.t += 1.0
        with st.stage("attn.core"):
            clock.t += 2.0
            with st.replay():
                clock.t += 5.0  # measurement-only: must vanish from both open stages
            with st.stage("attn.core.kda"):
                clock.t += 0.5
        clock.t += 0.25
    table = st.table()["7"]
    assert table["attn"]["wall_s"] == pytest.approx(3.75)
    assert table["attn.core"]["wall_s"] == pytest.approx(2.5)
    assert table["attn.core.kda"]["wall_s"] == pytest.approx(0.5)
    ex = kap.exclusive(table)
    assert ex["attn"] == pytest.approx(1.25)
    assert ex["attn.core"] == pytest.approx(2.0)
    assert ex["attn.core.kda"] == pytest.approx(0.5)


def test_stage_inside_replay_is_not_counted():
    clock = FakeClock()
    st = kap.Stages(sync=False, clock=clock, cpu_clock=clock)
    with st.replay():
        with st.stage("x"):
            clock.t += 1.0
    assert st.table() == {}


def test_exclusive_does_not_treat_name_prefix_as_child():
    stages = {"ffn": {"wall_s": 10.0}, "ffn.moe": {"wall_s": 6.0}, "ffn.moe_write_shared": {"wall_s": 1.0},
              "ffn.moe.dequant": {"wall_s": 2.0}}
    ex = kap.exclusive(stages)
    assert ex["ffn"] == pytest.approx(3.0)
    assert ex["ffn.moe"] == pytest.approx(4.0)


def test_load_frozen_tool_refuses_other_sha(tmp_path):
    path = tmp_path / "tool.py"
    path.write_text("TOOL_VERSION = 'x'\n")
    with pytest.raises(SystemExit):
        kap.load_frozen_tool(str(path), "0" * 64)


def test_compare_digests(tmp_path, capsys):
    a = {"layers": {"83": {"prefix_out_sha256": "aa"}, "84": {"prefix_out_sha256": "bb"}}}
    b = {"layers_receipt": {"83": {"prefix_out_sha256": "aa"}, "84": {"prefix_out_sha256": "bb"}}}
    c = {"layers": {"83": {"prefix_out_sha256": "aa"}, "84": {"prefix_out_sha256": "cc"}}}
    for name, payload in (("a", a), ("b", b), ("c", c)):
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    args = kap.build_parser().parse_args(["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert kap.cmd_compare(args, []) == 0
    args = kap.build_parser().parse_args(["compare", str(tmp_path / "a.json"), str(tmp_path / "c.json")])
    assert kap.cmd_compare(args, []) == 3
    assert "DIFFER" in capsys.readouterr().out


# --------------------------------------------------------------------------
# against the real engine (server)
# --------------------------------------------------------------------------
def engine():
    if not os.path.isfile(os.path.join(ENGINE_DIR, "k3_dense_pass.py")):
        pytest.skip("the K3 fold engine is not on this machine")
    if ENGINE_DIR not in sys.path:
        sys.path.insert(0, ENGINE_DIR)
    import torch
    import torch.nn.functional as F

    import k3_dense_pass as kdp

    return kdp, torch, F


class FakeWeights:
    def __init__(self, torch, layer_idx, hidden, experts, latent, inter, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.layer_idx = layer_idx
        self.is_kda, self.is_moe = True, True
        rnd = lambda *shape: (torch.randn(*shape, generator=g) * 0.1).to(torch.bfloat16)  # noqa: E731
        self.res = {"self_attention_res_proj.weight": rnd(1, hidden), "self_attention_res_norm.weight": rnd(hidden) + 1,
                    "mlp_res_proj.weight": rnd(1, hidden), "mlp_res_norm.weight": rnd(hidden) + 1,
                    "input_layernorm.weight": rnd(hidden) + 1, "post_attention_layernorm.weight": rnd(hidden) + 1}
        self.body = {"block_sparse_moe.gate.weight": rnd(experts, hidden),
                     "block_sparse_moe.gate.e_score_correction_bias": torch.randn(experts, generator=g).float() * 0.01}
        self.experts = {}
        for e in range(experts):
            rec = {}
            for name, (o, i) in (("w1", (inter, latent)), ("w3", (inter, latent)), ("w2", (latent, inter))):
                rec[f"{name}.weight_packed"] = torch.randint(0, 256, (o, i // 2), generator=g, dtype=torch.uint8)
                rec[f"{name}.weight_scale"] = torch.randint(120, 130, (o, i // 32), generator=g, dtype=torch.uint8)
            self.experts[e] = rec
        self.attn_w = rnd(hidden, hidden)


class FakeRunner:
    def __init__(self, torch, F, w, block_size):
        self.w, self.eps, self.block_size = w, 1e-5, block_size
        self.F = F

    def attention(self, x, state=None):
        return self.F.linear(x, self.w.attn_w), {}


class Seg:
    def __init__(self, rows, T):
        self.rows, self.T = rows, T


class Corpus:
    def __init__(self, rows, T):
        self.seg = Seg(rows, T)
        self.tokens = rows * T

    def row_chunks(self, row_chunk):
        for r0 in range(0, self.seg.rows, row_chunk):
            r1 = min(r0 + row_chunk, self.seg.rows)
            yield self.seg, r0, r1, r0 * self.seg.T, r1 * self.seg.T


@pytest.mark.parametrize("layer_idx,nb,slots", [(13, 2, 8), (12, 2, 8), (85, 8, 8)])
def test_attention_half_timed_is_bitwise_the_engine(layer_idx, nb, slots):
    kdp, torch, F = engine()
    hidden, rows, T = 64, 3, 40
    w = FakeWeights(torch, layer_idx, hidden, 4, 32, 32)
    runner = FakeRunner(torch, F, w, 12)
    g = torch.Generator().manual_seed(1)
    prefix = torch.randn(rows * T, hidden, generator=g).to(torch.bfloat16)
    blocks = torch.randn(rows * T, slots, hidden, generator=g).to(torch.bfloat16)
    pair = kdp.Pair(prefix, blocks[:, :nb])  # non-contiguous when nb < slots, as in the tool
    corpus = Corpus(rows, T)
    st = kap.Stages(sync=False)
    st.layer = layer_idx
    timed = kap.make_attention_half_timed(kdp, st, torch)
    ref = kdp.attention_half(runner, pair, corpus, torch.device("cpu"), row_chunk=1, token_chunk=16)
    got = timed(runner, pair, corpus, torch.device("cpu"), row_chunk=1, token_chunk=16)
    for a, b in zip(ref, got):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    names = set(st.table()[str(layer_idx)])
    assert ("attn.snapshot_cat_host_discarded" in names) == (layer_idx % 12 == 0)
    assert ("attn.h2d_blocks_noncontig" in names) == (nb < slots)


def moe_setup(seed=0, positions=96):
    kdp, torch, F = engine()
    hidden, experts, latent, inter = 64, 8, 64, 32
    w = FakeWeights(torch, 5, hidden, experts, latent, inter, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(positions, hidden, generator=g).to(torch.bfloat16)
    lat = torch.randn(positions, latent, generator=g).to(torch.bfloat16)
    cfg = {"num_experts": experts, "num_experts_per_token": 3, "moe_router_activation_func": "sigmoid",
           "moe_renormalize": True, "routed_scaling_factor": 1.0, "activation_situ_beta": 4.0,
           "activation_situ_linear_beta": 25.0}
    return kdp, torch, F, w, x, lat, cfg


def test_routed_timed_and_sorted_dispatch_are_bitwise_the_engine():
    kdp, torch, F, w, x, lat, cfg = moe_setup()
    dev = torch.device("cpu")
    ref, ref_taps, _ = kdp.routed_y_lat(w, x, lat, cfg, dev)
    st = kap.Stages(sync=False)
    seen = []
    timed = kap.make_routed_y_lat_timed(kdp, st, torch, F, hook=lambda *a: seen.append(a[-1]))
    got, taps, _ = timed(w, x, lat, cfg, dev)
    assert torch.equal(ref, got) and seen and torch.equal(seen[0], got)
    assert torch.equal(ref_taps["router_topk_idx"], taps["router_topk_idx"])
    assert ref_taps["experts_touched"] == taps["experts_touched"]
    assert torch.equal(kap.moe_sorted_dispatch(kdp, torch, F, w, x, lat, cfg, dev), ref)


def test_expert_outer_is_bitwise_the_engine_per_group_and_dequantizes_once():
    kdp, torch, F, w, x, lat, cfg = moe_setup(seed=3, positions=128)
    dev = torch.device("cpu")
    parts = [(x[:64], lat[:64]), (x[64:], lat[64:])]
    refs = [kdp.routed_y_lat(w, px, pl, cfg, dev)[0] for px, pl in parts]
    outs, dequants = kap.moe_expert_outer(kdp, torch, F, w, parts, cfg, dev)
    assert all(torch.equal(a, b) for a, b in zip(outs, refs))
    assert dequants <= cfg["num_experts"]


def test_split_refuses_a_non_probe_run(tmp_path):
    engine()
    tool = os.path.join(HERE, "k3_teacher_cache.py")
    if not os.path.isfile(tool) or kap.sha256_file(tool) != kap.FROZEN_K3_SHA256:
        pytest.skip("the frozen k3_teacher_cache 1.2.4 is not beside this test")
    args = kap.build_parser().parse_args(["split", "--tool", tool, "--out", str(tmp_path / "o.json")])
    with pytest.raises(SystemExit):
        kap.cmd_split(args, ["run", "--input-ids", "x.safetensors"])
