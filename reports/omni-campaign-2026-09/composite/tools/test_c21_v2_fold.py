"""CPU tests for the v2 fold-init tools (c19 alignment, c20 stream chart, c21 fold).

The load-bearing test is the identity fold: a synthetic Qwen3.5 trunk folded into a v2 config of
its own width and depth, keeping every head and channel.  Every identity c21 relies on -- the
charted reads with the norm weight folded in, the rho scale, grouped GDN keys duplicated into 1:1
v2 heads, the per-head conv/decay/gate-norm transplant, the ridge returning its prior on exact
data, the head-side tied vocabulary, the zero-init AttnRes reading a plain residual -- must then
reproduce the source's logits.  A wrong slice or a transposed chart shows up as a logit error.

Run: CHIMERA_SRC=<arch-v2 worktree>/src python -m pytest test_c21_v2_fold.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import c19_v2_alignment as c19  # noqa: E402
import c20_stream_chart as c20  # noqa: E402
import c21_v2_fold as c21  # noqa: E402
from chimera_omni.config import _v2_config  # noqa: E402
from chimera_omni.importing.profiles import qwen35_chimera_config  # noqa: E402
from chimera_omni.importing.qwen35 import QWEN_TEXT_PREFIX, import_qwen35_state  # noqa: E402
from chimera_omni.importing.safetensors_io import load_chimera_checkpoint  # noqa: E402

HIDDEN, LAYERS, INTERMEDIATE = 32, 8, 24
KEY_HEADS, VALUE_HEADS, KEY_DIM, VALUE_DIM, CONV = 2, 4, 8, 8, 4
Q_HEADS, KV_HEADS, HEAD_DIM, ROPE_DIM, THETA = 4, 2, 8, 4, 1.0e7
VOCAB, EPS = 64, 1.0e-6


def hf_config() -> dict:
    return {"text_config": {
        "vocab_size": VOCAB, "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
        "num_hidden_layers": LAYERS, "num_attention_heads": Q_HEADS, "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM, "linear_num_key_heads": KEY_HEADS, "linear_num_value_heads": VALUE_HEADS,
        "linear_key_head_dim": KEY_DIM, "linear_value_head_dim": VALUE_DIM, "linear_conv_kernel_dim": CONV,
        "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (LAYERS // 4),
        "full_attention_interval": 4,
        "rope_parameters": {"rope_theta": THETA, "partial_rotary_factor": ROPE_DIM / HEAD_DIM},
        "rms_norm_eps": EPS, "max_position_embeddings": 4096, "tie_word_embeddings": True, "hidden_act": "silu",
    }}


def random_state(seed: int = 0, final_norm_zero: bool = True) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)

    def n(*shape, scale=0.2):
        return torch.randn(*shape, generator=g) * scale

    kd, vd = KEY_HEADS * KEY_DIM, VALUE_HEADS * VALUE_DIM
    p = QWEN_TEXT_PREFIX
    state = {f"{p}embed_tokens.weight": n(VOCAB, HIDDEN, scale=1.0),
             f"{p}norm.weight": torch.zeros(HIDDEN) if final_norm_zero else n(HIDDEN)}
    for i in range(LAYERS):
        L = f"{p}layers.{i}."
        state[f"{L}input_layernorm.weight"] = n(HIDDEN)
        state[f"{L}post_attention_layernorm.weight"] = n(HIDDEN)
        state[f"{L}mlp.gate_proj.weight"] = n(INTERMEDIATE, HIDDEN)
        state[f"{L}mlp.up_proj.weight"] = n(INTERMEDIATE, HIDDEN)
        state[f"{L}mlp.down_proj.weight"] = n(HIDDEN, INTERMEDIATE)
        if i % 4 == 3:
            state[f"{L}self_attn.q_proj.weight"] = n(Q_HEADS * 2 * HEAD_DIM, HIDDEN)
            state[f"{L}self_attn.k_proj.weight"] = n(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{L}self_attn.v_proj.weight"] = n(KV_HEADS * HEAD_DIM, HIDDEN)
            state[f"{L}self_attn.o_proj.weight"] = n(HIDDEN, Q_HEADS * HEAD_DIM)
            state[f"{L}self_attn.q_norm.weight"] = n(HEAD_DIM)
            state[f"{L}self_attn.k_norm.weight"] = n(HEAD_DIM)
        else:
            state[f"{L}linear_attn.in_proj_qkv.weight"] = n(2 * kd + vd, HIDDEN)
            state[f"{L}linear_attn.in_proj_z.weight"] = n(vd, HIDDEN)
            state[f"{L}linear_attn.in_proj_a.weight"] = n(VALUE_HEADS, HIDDEN)
            state[f"{L}linear_attn.in_proj_b.weight"] = n(VALUE_HEADS, HIDDEN)
            state[f"{L}linear_attn.conv1d.weight"] = n(2 * kd + vd, 1, CONV)
            state[f"{L}linear_attn.dt_bias"] = n(VALUE_HEADS, scale=1.0)
            state[f"{L}linear_attn.A_log"] = torch.log(torch.empty(VALUE_HEADS).uniform_(1.0, 16.0, generator=g))
            state[f"{L}linear_attn.norm.weight"] = n(VALUE_DIM)
            state[f"{L}linear_attn.out_proj.weight"] = n(HIDDEN, vd)
    return state


def source_of(state: dict[str, torch.Tensor]) -> c21.QwenTrunkSource:
    return c21.QwenTrunkSource(hf_config(), lambda name: state[name], state.keys(), label="synthetic-qwen35")


def target_config(*, hidden, intermediate, layers, q_heads, kv_heads, kda_heads):
    return _v2_config(name="test-fold", hidden_size=hidden, intermediate_size=intermediate, num_layers=layers,
                      num_heads=q_heads, num_kv_heads=kv_heads, attention_head_dim=HEAD_DIM, gqa_rope_dim=ROPE_DIM,
                      kda_num_heads=kda_heads, kda_num_key_heads=kda_heads, kda_head_dim=KEY_DIM,
                      init_source="fold (test)", text_vocab=VOCAB, control_band=8, max_sequence_length=4096,
                      gqa_mrope_section=(), rope_theta=THETA)


def calibration(rows: int = 8, length: int = 24, seed: int = 1) -> c21.Calibration:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (rows, length), generator=g)
    lengths = torch.tensor([length - (i % 3) * 3 for i in range(rows)])
    valid = torch.arange(length).unsqueeze(0) < lengths.unsqueeze(1)
    return c21.Calibration(ids, valid, fit_rows=[0, 1, 2, 4, 5, 6], check_rows=[5], held_rows=[3, 7],
                           record={"synthetic": True})


def text_logits_source(state, ids, valid):
    cfg = qwen35_chimera_config(hf_config())
    model, _ = import_qwen35_state(state, cfg)
    model.eval()
    with torch.no_grad():
        out = model(ids, attention_mask=valid.long())
    return out["logits"][..., :VOCAB]


def text_logits(model, ids, valid):
    with torch.no_grad():
        out = model(ids, attention_mask=valid.long())
    return out["logits"][..., :VOCAB]


# ------------------------------------------------------------------------------------------- c19


def test_alignment_floor_intervals_match_c17_and_cover_every_source_layer():
    rows = c19.period_alignment(64, 24, 4, ("kda", "kda", "kda", "gated_gqa"))
    periods = [r["source_periods"] for r in rows if r["slot"] == 0]
    # floor(16 p / 6): the plan's section-4.2 table printed {8,9,10} / {11,12} for periods 3 and 4;
    # the floor rule it names gives {8,9} / {10,11,12} (16*4/6 = 10.67).
    assert periods == [[0, 1], [2, 3, 4], [5, 6, 7], [8, 9], [10, 11, 12], [13, 14, 15]]
    assert sorted(l for r in rows for l in r["source_layers"]) == list(range(64))
    assert all(l % 4 == r["slot"] for r in rows for l in r["source_layers"])         # kind-preserving
    proxy = c19.period_alignment(64, 16, 4, ("kda", "kda", "kda", "gated_gqa"))
    assert {len(r["source_layers"]) for r in proxy} == {4}
    k3 = c19.k3_alignment(92, 24)
    sizes = [len([l for l in r["k3_layers"] if l != 0]) for r in k3]
    assert sorted(sizes).count(4) == 20 and sorted(sizes).count(3) == 4
    assert k3[0]["k3_layers"][0] == 0 and sorted(l for r in k3 for l in r["k3_layers"]) == list(range(93))
    with pytest.raises(ValueError):
        c19.floor_interval(3, 4)


# ------------------------------------------------------------------------------------------- c20


def test_stream_chart_is_orthonormal_deterministic_and_keeps_the_mean():
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(500, 12, generator=g) + 3.0 * torch.ones(12)
    moment = c20.SecondMoment(12)
    moment.add(rows, layer=0)
    chart, record = c20.fit_chart(moment, 4)
    assert record["orthonormality_error"] < 1e-10
    again, _ = c20.fit_chart(moment, 4)
    assert torch.equal(chart, again)
    assert record["mean_retained_fraction"] > 0.99      # the dominant mean direction is in the chart
    full, full_record = c20.fit_chart(moment, 12)
    assert abs(full_record["retained_energy"] - 1.0) < 1e-12


# ------------------------------------------------------------------------------------------- c21


@pytest.mark.parametrize("target_mode,final_norm", [("writes", "scalar"), ("block_end", "scalar"), ("block_end", "diag")])
def test_identity_fold_reproduces_the_source_logits(target_mode, final_norm):
    state = random_state()
    src = source_of(state)
    cfg = target_config(hidden=HIDDEN, intermediate=INTERMEDIATE, layers=LAYERS, q_heads=Q_HEADS,
                        kv_heads=KV_HEADS, kda_heads=VALUE_HEADS)
    calib = calibration()
    # an identity fold is 1x by construction, so the 4x rule is the one thing switched off here
    fold = c21.Fold(src, cfg, calib, c21.FoldOptions(device="cpu", batch=3, require_4x=False, target_mode=target_mode,
                                                       final_norm=final_norm))
    model, receipt = fold.run()
    assert receipt["four_x_rule"].startswith("OFF")
    assert abs(receipt["stream_scale_alpha"] - 1.0) < 1e-5
    want = text_logits_source(state, calib.input_ids, calib.valid)
    got = text_logits(model, calib.input_ids, calib.valid)
    v = calib.valid
    err = float((got[v] - want[v]).abs().max() / want[v].abs().max())
    assert err < 1e-4, err
    # every ridge refit returned (numerically) its charted prior: exact data, exact transplant
    for row in receipt["components"]:
        for key in ("out_ridge", "down_ridge"):
            if key in row["diagnostics"]:
                assert row["diagnostics"][key]["held_rel_l2"] < 1e-4, (row["target_layer"], key)
    assert receipt["assembly_check"]["normalised_stream_rel_l2"] < 1e-5
    assert receipt["copy_methods"] == 0


def test_width_and_depth_fold_is_finite_accounted_and_reloads(tmp_path):
    state = random_state(seed=3, final_norm_zero=False)
    src = source_of(state)
    cfg = target_config(hidden=16, intermediate=12, layers=4, q_heads=2, kv_heads=1, kda_heads=2)
    calib = calibration(rows=8, length=20, seed=5)
    fold = c21.Fold(src, cfg, calib, c21.FoldOptions(device="cpu", batch=4))
    model, receipt = fold.run()
    # every parameter has a fold method; nothing is a copy; selections clear 4x from shapes alone
    assert set(receipt["parameters_by_method"]) <= set(c21.FOLD_METHODS)
    assert sum(receipt["parameters_by_method"].values()) == sum(p.numel() for p in model.parameters())
    selections = [c for c in receipt["components"] if c["selection_component"]]
    assert len(selections) == 2 * 4 and all(c["compression"] >= 4.0 for c in selections)
    for c in selections:
        assert c["installed_params"] == sum(
            p.numel() for n, p in model.named_parameters()
            if n.startswith(f"blocks.{c['target_layer']}.") and
            (n.split(".")[2] == ("mixer" if c["component"].startswith("mixer") else "ffn")))
    # the depth map pooled 2 source layers per v2 layer, kind-preserving
    assert [r["source_layers"] for r in receipt["alignment"]["trunk_table"]] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    # a selected GDN head carries ITS key head's charted rows (grouped source keys -> 1:1 v2 heads)
    layer0 = receipt["layers"][0]
    (l_src, h_src) = layer0["mixer"]["heads"][0]
    period_model, _ = src.period_model((l_src // 4) * 4, 4, torch.device("cpu"))
    st = c21.block_state(period_model.blocks[l_src % 4])
    rho = layer0["mixer"]["rho"][l_src] / layer0["mixer"]["norm_scalar"]
    kh = h_src // (VALUE_HEADS // KEY_HEADS)
    expected = c21.read_rows(st["mixer.q_proj.weight"][kh * KEY_DIM:(kh + 1) * KEY_DIM], st["norm1.weight"], fold.chart, rho)
    assert torch.allclose(model.blocks[0].mixer.q_proj.weight[:KEY_DIM], expected, atol=1e-6)
    # placeholders are exact zeros, never random
    assert all(float(p.detach().abs().max()) == 0.0 for n, p in model.named_parameters()
               if receipt and fold.param_methods[n] == "zero_placeholder")
    # the checkpoint round-trips through the chimera loader with identical logits
    info = c21.save_checkpoint(model, tmp_path / "child", receipt)
    reloaded, cfg2, _ = load_chimera_checkpoint(tmp_path / "child", dtype=torch.float32)
    assert cfg2.to_manifest() == cfg.to_manifest()
    reloaded.eval()
    c21.use_chunked_gdn_scan(reloaded, 64)          # the fold ran the chunked GDN scan; compare like with like
    a = text_logits(model, calib.input_ids, calib.valid)
    b = text_logits(reloaded, calib.input_ids, calib.valid)
    assert torch.equal(a, b)
    assert "model.safetensors" in "".join(info["files"]) or any(f.endswith(".safetensors") for f in info["files"])
    json.dumps(receipt, default=str)        # the receipt serialises
    assert math.isfinite(receipt["source_reference"]["held_nll_per_token"])


def test_geometry_mismatches_are_refused():
    src = source_of(random_state())
    calib = calibration()
    wrong_head = _v2_config(name="bad", hidden_size=16, intermediate_size=12, num_layers=4, num_heads=2,
                            num_kv_heads=1, attention_head_dim=16, gqa_rope_dim=ROPE_DIM, kda_num_heads=2,
                            kda_num_key_heads=2, kda_head_dim=KEY_DIM, init_source="x", text_vocab=VOCAB,
                            control_band=8, max_sequence_length=4096, gqa_mrope_section=(), rope_theta=THETA)
    with pytest.raises(SystemExit, match="per-head shapes"):
        c21.Fold(src, wrong_head, calib, c21.FoldOptions())
    grouped = _v2_config(name="bad", hidden_size=16, intermediate_size=12, num_layers=4, num_heads=2,
                         num_kv_heads=1, attention_head_dim=HEAD_DIM, gqa_rope_dim=ROPE_DIM, kda_num_heads=4,
                         kda_num_key_heads=2, kda_head_dim=KEY_DIM, init_source="x", text_vocab=VOCAB,
                         control_band=8, max_sequence_length=4096, gqa_mrope_section=(), rope_theta=THETA)
    with pytest.raises(SystemExit, match="1:1 GDN key heads"):
        c21.Fold(src, grouped, calib, c21.FoldOptions())
    rope = _v2_config(name="bad", hidden_size=16, intermediate_size=12, num_layers=4, num_heads=2,
                      num_kv_heads=1, attention_head_dim=HEAD_DIM, gqa_rope_dim=ROPE_DIM, kda_num_heads=2,
                      kda_num_key_heads=2, kda_head_dim=KEY_DIM, init_source="x", text_vocab=VOCAB,
                      control_band=8, max_sequence_length=4096, gqa_mrope_section=(), rope_theta=1.0e4)
    with pytest.raises(SystemExit, match="rotary differs"):
        c21.Fold(src, rope, calib, c21.FoldOptions())


def test_fold_proxy_profile_keeps_the_2b_head_geometry():
    from chimera_omni.config import V2_PROFILES

    proxy, two = V2_PROFILES["co2-fold-proxy"](), V2_PROFILES["co2-2b"]()
    for name in ("attention_head_dim", "gqa_rope_dim", "gqa_rope_theta", "gqa_mrope_section", "kda_head_dim",
                 "kda_value_head_dim", "vocab_size", "ledger_base_vocab_size", "block_pattern", "residual_mode"):
        assert getattr(proxy, name) == getattr(two, name), name
    assert proxy.resolved_kda_num_heads() == proxy.resolved_kda_num_key_heads()
