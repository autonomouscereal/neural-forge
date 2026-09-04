#!/usr/bin/env python3
"""Tests for the K3 streamed harness.  Run them; do not trust the edits.

Everything here runs against the synthetic K3 built by `k3_synth.py` -- real
safetensors shards, real MXFP4 packing, real one-based `kda_layers` -- so the
same code paths the 1.42 TiB checkpoint will take are exercised without
touching the CIFS share.

The cross-references are deliberately to things this lane did NOT write:

  * `fla.ops.kda.naive.naive_recurrent_kda` and
    `fla.ops.kda.gate.naive_kda_lowerbound_gate` are fla's own pure-torch
    references and run on CPU, so the KDA recurrence and the gate are checked
    without a GPU.
  * `modeling_kimi_linear.SituAndMul`, `KimiRMSNorm`, `_apply_attn_res`,
    `KimiMLAAttention`, `KimiSparseMoeBlock` and `KimiDecoderLayer` are the
    repo's own modules.
  * `compressed_tensors`' own unpacking helpers check the MXFP4 layout.
  * `prior_anchored_gram` is checked against a dense re-implementation of
    mx45_build's `prior_anchored`.

A test that cannot run (no GPU, library missing) is reported SKIP with the
reason, never silently passed.

    python test_k3_harness.py --work-dir /home/cereal/omni-campaign/k3/test \\
        --src-dir /home/cereal/omni-campaign/k3/k3_src [--device cuda:4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import torch

import k3_common
import k3_layer
import c14_union_coreset as c14
from k3_common import K3Index, apply_attn_res, dequant_mxfp4, rms_norm, situ, write_receipt
from k3_layer import LayerWeights, StreamedLayer, load_layer

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS: list = []
CONTEXT: dict = {}


def test(fn):
    TESTS.append(fn)
    return fn


class Skip(Exception):
    pass


def close(a: torch.Tensor, b: torch.Tensor, tol: float, label: str) -> dict:
    a = a.float()
    b = b.float()
    rel = float((a - b).norm() / max(float(b.norm()), 1e-30))
    mx = float((a - b).abs().max())
    if not rel <= tol:
        raise AssertionError(f"{label}: relL2 {rel:.3e} > {tol:.1e} (max abs {mx:.3e})")
    return {"relL2": rel, "max_abs": mx}


# --------------------------------------------------------------------------
@test
def t01_synthetic_checkpoint_round_trips() -> dict:
    index: K3Index = CONTEXT["index"]
    assert index.num_layers == CONTEXT["layers"], index.num_layers
    assert index.is_kda_layer(0) is True, "layer 0 should be KDA (one-based kda_layers)"
    assert index.is_kda_layer(index.num_layers - 1) is False, "the last layer is forced full-attn"
    assert index.is_moe_layer(0) is False, "first_k_dense_replace = 1 keeps layer 0 dense"
    assert index.is_moe_layer(1) is True
    weights, timing = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    counts = weights.parameter_counts()
    assert counts["expert_count"] == CONTEXT["experts"], counts
    return {"read": timing, "parameters": counts, "shards": len(set(index.weight_map.values()))}


@test
def t02_mxfp4_layout_is_low_nibble_even_column() -> dict:
    # one byte, one group: low nibble 1 -> 0.5, high nibble 2 -> 1.0, scale 2^0
    packed = torch.full((1, 16), 0x21, dtype=torch.uint8)
    scale = torch.tensor([[127]], dtype=torch.uint8)
    out = dequant_mxfp4(packed, scale, group=32, dtype=torch.float32)
    assert out.shape == (1, 32), out.shape
    assert float(out[0, 0]) == 0.5, out[0, :4]
    assert float(out[0, 1]) == 1.0, out[0, :4]
    # bit 3 is the sign: 0x9A -> low nibble 0xA = sign + magnitude 2 = -1.0,
    # high nibble 0x9 = sign + magnitude 1 = -0.5
    signed = dequant_mxfp4(torch.full((1, 16), 0x9A, dtype=torch.uint8), scale,
                           group=32, dtype=torch.float32)
    assert float(signed[0, 0]) == -1.0, signed[0, :2]
    assert float(signed[0, 1]) == -0.5, signed[0, :2]
    # a scale byte of 121 is 2**-6
    scaled = dequant_mxfp4(torch.full((1, 16), 0x11, dtype=torch.uint8),
                           torch.tensor([[121]], dtype=torch.uint8), group=32, dtype=torch.float32)
    assert float(scaled[0, 0]) == 0.5 * 2 ** -6, float(scaled[0, 0])
    return {"even": float(out[0, 0]), "odd": float(out[0, 1]),
            "signed_low": float(signed[0, 0]), "signed_high": float(signed[0, 1]),
            "scaled": float(scaled[0, 0])}


@test
def t03_mxfp4_matches_compressed_tensors() -> dict:
    index: K3Index = CONTEXT["index"]
    weights, _ = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    expert = weights.experts[0]
    fragment = k3_common.mxfp4_reference_check(expert["w1.weight_packed"], expert["w1.weight_scale"])
    if not fragment.get("available"):
        raise Skip("compressed_tensors is not installed in this interpreter")
    if "error" in fragment:
        raise AssertionError(f"cross-check raised: {fragment['error']}")
    assert fragment.get("bit_exact"), fragment
    return fragment


@test
def t04_situ_matches_the_repo() -> dict:
    modeling = CONTEXT.get("modeling")
    if modeling is None:
        raise Skip(CONTEXT["modeling_error"])
    generator = torch.Generator().manual_seed(7)
    gate = torch.randn(64, 32, generator=generator).to(torch.bfloat16)
    up = torch.randn(64, 32, generator=generator).to(torch.bfloat16)
    reference = modeling.SituAndMul(beta=4.0, linear_beta=25.0)(torch.cat([gate, up], dim=-1))
    ours = situ(gate, up, 4.0, 25.0)
    assert torch.equal(ours, reference), (ours - reference).abs().max()
    return {"bit_exact": True, "shape": list(ours.shape)}


@test
def t05_norm_and_attn_res_match_the_repo() -> dict:
    modeling = CONTEXT.get("modeling")
    if modeling is None:
        raise Skip(CONTEXT["modeling_error"])
    generator = torch.Generator().manual_seed(11)
    hidden = 64
    x = torch.randn(8, hidden, generator=generator).to(torch.bfloat16)
    weight = torch.randn(hidden, generator=generator).to(torch.bfloat16)
    module = modeling.KimiRMSNorm(hidden, eps=1e-5)
    module.weight.data = weight.clone()
    assert torch.equal(rms_norm(x, weight, 1e-5), module(x)), "KimiRMSNorm mismatch"

    blocks = torch.randn(8, 3, hidden, generator=generator).to(torch.bfloat16)
    proj = torch.nn.Linear(hidden, 1, bias=False).to(torch.bfloat16)
    proj.weight.data = torch.randn(1, hidden, generator=generator).to(torch.bfloat16)
    reference = modeling._apply_attn_res(x, blocks, proj, module)
    ours = apply_attn_res(x, blocks, proj.weight, module.weight, 1e-5)
    assert torch.equal(ours, reference), (ours.float() - reference.float()).abs().max()
    return {"rms_norm": "bit exact", "apply_attn_res": "bit exact"}


@test
def t05b_the_fixture_can_tell_the_norm_gains_apart() -> dict:
    """Every RMSNorm gain in the fixture is DISTINCT, and mis-wiring one moves the layer.

    Until this test the fixture built every gain as `torch.ones(...)`, and that
    made the whole suite structurally blind to WHICH norm the harness reads.
    K3 has six distinct gains per layer and `StreamedLayer.forward` wires four
    of them by name; with unit gains you can swap `self_attention_res_norm`
    with `mlp_res_norm`, or `input_layernorm` with `post_attention_layernorm`,
    or point `moe_forward` at the wrong norm, and t05/t08/t09/t10/t10b/t12/
    t12b/t17 and both bs3 exactness receipts still report bit-exact block
    stacks and fp32 relL2 ~1e-7 -- because the repo module is loaded from the
    same all-ones state dict.  On the real checkpoint the gains differ and
    prefix_out, y_lat and the C14 target would all be silently wrong.  Same
    class as the eps = 1e-6 bug, and the fixture could not see it.

    So: assert the gains are pairwise distinct and none of them is a unit
    vector, then MUTATE the state dict (swap a pair, flatten one to ones) and
    require the layer output to move.  A mutation that does not move the output
    is a name the harness never reads.
    """
    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    layer_idx = index.num_layers - 1
    if index.is_kda_layer(layer_idx):
        raise Skip("the last layer is KDA in this configuration")
    hidden = index.hidden_size
    block_size = index.attn_res_block_size or 1
    n_blocks = k3_common.blocks_before(layer_idx, block_size)
    generator = torch.Generator().manual_seed(77)
    x = (torch.randn(1, 12, hidden, generator=generator) * 0.05).to(torch.float32)
    blocks = (torch.randn(12, n_blocks, hidden, generator=generator) * 0.05).to(torch.float32)

    def run(mutate=None) -> torch.Tensor:
        weights, _ = load_layer(index, layer_idx, "cpu", "cpu", log_reads=False)
        for group in (weights.attn, weights.res, weights.body):
            for key, tensor in group.items():
                if tensor.is_floating_point():
                    group[key] = tensor.to(torch.float32)
        if mutate is not None:
            mutate(weights)
        out, _, _ = StreamedLayer(weights, cfg, kda_backend="torch").forward(
            x, blocks.clone(), collect=False)
        return out

    # -- 1. the gains are distinct ----------------------------------------
    weights, _ = load_layer(index, layer_idx, "cpu", "cpu", log_reads=False)
    gains = {}
    for group in (weights.attn, weights.res, weights.body):
        for key, tensor in group.items():
            if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
                gains[key] = tensor.float()
    assert len(gains) >= 6, sorted(gains)
    for key, tensor in gains.items():
        assert not torch.equal(tensor, torch.ones_like(tensor)), f"{key} is still a unit gain"
    names = sorted(gains)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if gains[a].shape == gains[b].shape:
                assert not torch.equal(gains[a], gains[b]), f"{a} and {b} are the same vector"

    # -- 2. mutating a name the layer reads must move the output ----------
    base = run()
    scale = float(base.float().norm())

    def swap(first: str, second: str):
        def apply(w):
            w.res[first], w.res[second] = w.res[second], w.res[first]
        return apply

    def flatten(group_name: str, key: str):
        def apply(w):
            group = getattr(w, group_name)
            group[key] = torch.ones_like(group[key])
        return apply

    mutations = {
        "swap self_attention_res_norm <-> mlp_res_norm":
            swap("self_attention_res_norm.weight", "mlp_res_norm.weight"),
        "swap input_layernorm <-> post_attention_layernorm":
            swap("input_layernorm.weight", "post_attention_layernorm.weight"),
        "input_layernorm -> ones": flatten("res", "input_layernorm.weight"),
        "post_attention_layernorm -> ones": flatten("res", "post_attention_layernorm.weight"),
        "self_attention_res_norm -> ones": flatten("res", "self_attention_res_norm.weight"),
        "mlp_res_norm -> ones": flatten("res", "mlp_res_norm.weight"),
        "routed_expert_norm -> ones":
            flatten("body", "block_sparse_moe.routed_expert_norm.weight"),
        "kv_a_layernorm -> ones": flatten("attn", "kv_a_layernorm.weight"),
        "q_a_layernorm -> ones": flatten("attn", "q_a_layernorm.weight"),
    }
    # Two floors, because the effect sizes are genuinely two different sizes.
    # The four gains `StreamedLayer.forward` wires by name, plus
    # `routed_expert_norm`, sit directly on the hidden stream and must move it a
    # LOT -- those are the ones the review named and the ones a swap corrupts.
    # The two MLA latent gains sit inside a rank-24/32 bottleneck that the
    # attention output is projected back out of, so their measured effect on
    # this fixture is ~9e-5; the claim being made about them is only that the
    # harness READS them, and at fp32 with a deterministic forward any nonzero
    # movement is that proof (fp32 round-off here is ~1e-7 of the output norm).
    STRONG = 1e-3
    WEAK = 1e-6
    strong_labels = {
        "swap self_attention_res_norm <-> mlp_res_norm",
        "swap input_layernorm <-> post_attention_layernorm",
        "input_layernorm -> ones",
        "post_attention_layernorm -> ones",
        "self_attention_res_norm -> ones",
        "mlp_res_norm -> ones",
        "routed_expert_norm -> ones",
    }
    moved = {}
    for label, mutate in mutations.items():
        out = run(mutate)
        rel = float((out.float() - base.float()).norm() / max(scale, 1e-30))
        floor = STRONG if label in strong_labels else WEAK
        assert rel > floor, (
            f"{label} did not move the layer (relL2 {rel:.3e} <= {floor:.0e}) -- either the gain "
            "is still a unit vector or the harness never reads that name")
        moved[label] = float(f"{rel:.6g}")
    return {"distinct_gains": names,
            "mutation_relL2_vs_unmutated": moved,
            "floor_on_the_four_wired_by_name_and_routed_expert_norm": STRONG,
            "floor_on_the_two_MLA_latent_gains": WEAK,
            "why_two_floors": ("the MLA latent gains act inside a rank-24/32 bottleneck, so their "
                               "effect is ~9e-5 on this fixture; the claim about them is only that "
                               "the harness reads them at all")}


@test
def t06_kda_gate_and_recurrence_match_fla() -> dict:
    try:
        from fla.ops.kda.gate import naive_kda_lowerbound_gate
        from fla.ops.kda.naive import naive_recurrent_kda as fla_naive
    except Exception as exc:
        raise Skip(f"fla reference not importable: {exc!r}")
    generator = torch.Generator().manual_seed(13)
    B, T, H, K = 1, 12, 4, 24
    g_raw = torch.randn(B, T, H, K, generator=generator)
    a_log = torch.log(torch.empty(H).uniform_(1, 16, generator=generator))
    dt_bias = torch.randn(H * K, generator=generator) * 0.1
    ours_gate = k3_layer.kda_gate(g_raw, a_log, dt_bias, -5.0, axis="heads")
    ref_gate = naive_kda_lowerbound_gate(g_raw, a_log, dt_bias, lower_bound=-5.0)
    gate_scores = close(ours_gate, ref_gate, 1e-6, "kda gate (lower-bound form)")
    # the shipped K3 checkpoint carries A_log with head_dim elements, not
    # num_heads (k3_layer1_shapes.json: f32[128] against num_heads 96), so the
    # key-dim broadcast has to exist and has to be selected by shape
    assert k3_layer.kda_gate_axis(a_log, H, K) == "heads", "axis resolution picked the wrong form"
    a_key = torch.log(torch.empty(K).uniform_(1, 16, generator=generator))
    assert k3_layer.kda_gate_axis(a_key, H, K) == "key_dim", "key-dim A_log was not detected"
    key_gate = k3_layer.kda_gate(g_raw, a_key, dt_bias, -5.0, axis="key_dim")
    manual = -5.0 * torch.sigmoid(a_key.exp().view(1, K) * (g_raw.float() + dt_bias.view(H, K)))
    close(key_gate, manual, 1e-6, "key-dim kda gate")
    try:
        k3_layer.kda_gate_axis(torch.zeros(H + K + 1), H, K)
        raise AssertionError("an A_log matching neither axis must be refused")
    except SystemExit:
        pass

    q = torch.randn(B, T, H, K, generator=generator)
    k = torch.randn(B, T, H, K, generator=generator)
    v = torch.randn(B, T, H, K, generator=generator)
    beta = torch.sigmoid(torch.randn(B, T, H, generator=generator))
    ours_o, ours_state = k3_layer.naive_recurrent_kda(q, k, v, ref_gate, beta, scale=K ** -0.5)
    ref_o, ref_state = fla_naive(q, k, v, ref_gate, beta, scale=K ** -0.5, output_final_state=True)
    return {
        "gate": gate_scores,
        "output": close(ours_o, ref_o, 1e-5, "kda recurrence output"),
        "state": close(ours_state, ref_state, 1e-5, "kda recurrence final state"),
        "A_log_axis_fla_form": "num_heads (fla does A_log.view(H, 1))",
        "A_log_axis_real_k3": "head_dim -- f32[128] against num_heads 96; see k3_layer.kda_gate_axis",
    }


@test
def t06b_key_dim_A_log_runs_end_to_end() -> dict:
    """The real checkpoint's [head_dim] A_log must drive a whole KDA forward."""
    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    heads = cfg["linear_attn_config"]["num_heads"]
    head_dim = cfg["linear_attn_config"]["head_dim"]
    assert heads != head_dim, "this test needs heads != head_dim to tell the axes apart"
    weights, _ = load_layer(index, 0, "cpu", "cpu", log_reads=False)
    assert weights.is_kda
    x = (torch.randn(1, 12, index.hidden_size,
                     generator=torch.Generator().manual_seed(47)) * 0.05).to(torch.bfloat16)
    head_out, head_taps = k3_layer.kda_forward(weights, x, cfg, backend="torch")
    assert head_taps["kda_gate_axis"] == "heads", head_taps["kda_gate_axis"]
    weights.attn["A_log"] = torch.log(
        torch.empty(head_dim).uniform_(1, 16, generator=torch.Generator().manual_seed(48))
    ).to(torch.float32)
    key_out, key_taps = k3_layer.kda_forward(weights, x, cfg, backend="torch")
    assert key_taps["kda_gate_axis"] == "key_dim", key_taps["kda_gate_axis"]
    assert key_out.shape == head_out.shape
    assert torch.isfinite(key_out.float()).all()
    difference = float((key_out.float() - head_out.float()).norm() / head_out.float().norm())
    assert difference > 1e-6, "the two broadcasts produced the same answer; the axis is not being used"
    return {"heads_axis_sha256": k3_common.sha256_tensor(head_out)[:16],
            "key_dim_axis_sha256": k3_common.sha256_tensor(key_out)[:16],
            "relative_difference_between_axes": difference}


@test
def t07_short_conv_is_causal_and_matches_a_loop() -> dict:
    generator = torch.Generator().manual_seed(17)
    B, T, D, W = 1, 9, 6, 4
    x = torch.randn(B, T, D, generator=generator)
    weight = torch.randn(D, 1, W, generator=generator)
    ours, cache = k3_layer.short_conv(x, weight, activation=None)
    manual = torch.zeros_like(ours)
    for t in range(T):
        for j in range(W):
            src = t - (W - 1) + j
            if src >= 0:
                manual[:, t] += x[:, src] * weight[:, 0, j]
    scores = close(ours, manual, 1e-5, "short conv vs explicit causal loop")
    # streaming the same sequence one token at a time must give the same answer
    streamed, running = k3_layer.short_conv(x[:, :5], weight, activation=None)
    tail, _ = k3_layer.short_conv(x[:, 5:], weight, cache=running, activation=None)
    scores_stream = close(torch.cat([streamed, tail], dim=1), ours, 1e-6, "short conv cache continuity")
    assert cache.shape == (B, D, W - 1), cache.shape
    return {"causal": scores, "cache_continuity": scores_stream}


def layer_at(dtype: torch.dtype, layer_idx: int, index: K3Index | None = None):
    """The layer's weights cast to `dtype` -- experts included, dequantized to it."""
    index = index or CONTEXT["index"]
    weights, _ = load_layer(index, layer_idx, "cpu", "cpu", log_reads=False)
    for group in (weights.attn, weights.res, weights.body):
        for key, tensor in group.items():
            if tensor.is_floating_point():
                group[key] = tensor.to(dtype)
    return weights


@test
def t08_mla_matches_the_repo_module() -> dict:
    """Exact at fp32; at bf16 the gap must stay inside the repo's OWN bf16 floor.

    The first version of this test asserted a bf16 tolerance and failed at
    relL2 6.06e-2 -- which turned out to be a real algebra bug (K3's two MLA
    latent norms run at KimiRMSNorm's default eps=1e-6, not
    `config.rms_norm_eps`), not noise.  So the claim is now made at fp32, where
    arithmetic cannot hide anything, and the bf16 number is measured against
    the reference module's own bf16-vs-fp32 self-difference.
    """
    modeling, config_module = CONTEXT.get("modeling"), CONTEXT.get("config_module")
    if modeling is None:
        raise Skip(CONTEXT["modeling_error"])
    index: K3Index = CONTEXT["index"]
    layer_idx = index.num_layers - 1
    cfg = index.text_config
    hf_config = config_module.KimiLinearConfig(**cfg)
    hf_config._attn_implementation = "eager"

    def run(dtype):
        weights = layer_at(dtype, layer_idx)
        with torch.device("meta"):
            module = modeling.KimiMLAAttention(config=hf_config, layer_idx=layer_idx)
        missing, unexpected = module.load_state_dict(dict(weights.attn), strict=False, assign=True)
        assert not missing and not unexpected, (missing, unexpected)
        module.eval()
        generator = torch.Generator().manual_seed(19)
        x = (torch.randn(1, 16, index.hidden_size, generator=generator) * 0.05).to(dtype)
        mask = torch.full((1, 1, 16, 16), float("-inf")).triu(1).to(dtype)
        with torch.no_grad():
            reference = module(x, attention_mask=mask)
        ours, _ = k3_layer.mla_forward(weights, x, cfg)
        return ours, reference

    ours32, ref32 = run(torch.float32)
    ours16, ref16 = run(torch.bfloat16)
    exact = close(ours32, ref32, 1e-5, "MLA vs KimiMLAAttention at fp32")
    floor = float((ref16.float() - ref32.float()).norm() / ref32.float().norm())
    gap = float((ours16.float() - ref16.float()).norm() / ref16.float().norm())
    assert gap <= max(2 * floor, 1e-3), f"bf16 gap {gap:.3e} exceeds the module's own bf16 floor {floor:.3e}"
    return {"fp32": exact, "bf16_gap": gap, "reference_bf16_floor": floor,
            "latent_norm_eps": k3_layer.MLA_LATENT_NORM_EPS}


@test
def t09_moe_matches_the_repo_module() -> dict:
    modeling, config_module = CONTEXT.get("modeling"), CONTEXT.get("config_module")
    if modeling is None:
        raise Skip(CONTEXT["modeling_error"])
    index: K3Index = CONTEXT["index"]
    weights, _ = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    cfg = index.text_config
    hf_config = config_module.KimiLinearConfig(**cfg)
    hf_config._attn_implementation = "eager"
    with torch.device("meta"):
        module = modeling.KimiSparseMoeBlock(hf_config)
    from k3_receipt import module_state_dict

    state = {k[len("block_sparse_moe."):]: v for k, v in module_state_dict(weights, True).items()
             if k.startswith("block_sparse_moe.")}
    missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
    assert not missing, missing
    module.eval()
    generator = torch.Generator().manual_seed(23)
    x = (torch.randn(1, 24, index.hidden_size, generator=generator) * 0.05).to(torch.bfloat16)
    with torch.no_grad():
        reference = module(x)
    ours, taps = k3_layer.moe_forward(weights, x, cfg, collect=True)
    with torch.no_grad():
        ref_idx, ref_w = module.gate(x)
    our_idx, our_w, _ = k3_layer.route(x, weights, cfg)
    agreement = float(torch.stack([
        torch.isin(our_idx[t], ref_idx[t]).float().mean() for t in range(our_idx.shape[0])
    ]).mean())
    assert agreement == 1.0, f"router top-k set agreement {agreement}"
    assert taps["y_lat"].dtype == torch.float32, taps["y_lat"].dtype
    return {"scores": close(ours, reference, 5e-3, "MoE vs KimiSparseMoeBlock"),
            "router_topk_set_agreement": agreement,
            "experts_touched": int((taps["router_counts"] > 0).sum())}


def whole_layer_against_repo(index: K3Index, layer_idx: int, tokens: int = 16) -> dict:
    """One layer vs `KimiDecoderLayer._forward_attn_residual`, fp32 and bf16.

    Parameterised by layer because K3's residual has TWO branches and they are
    not interchangeable:

      * ``layer_idx % attn_res_block_size == 0`` -- the snapshot branch.  The
        prefix sum is pushed onto the block stack and reset to None, so the
        attention output BECOMES the new prefix.
      * otherwise -- ``carried = prefix_sum + attn_out``, the ordinary residual.
        On the real checkpoint (93 layers, block size 12) this is the branch 85
        of 93 layers take, and a block-size-1 fixture can never reach it.

    Returns the scores plus which branch ran, so the receipt says so out loud.
    """
    modeling, config_module = CONTEXT["modeling"], CONTEXT["config_module"]
    cfg = index.text_config
    hf_config = config_module.KimiLinearConfig(**cfg)
    hf_config._attn_implementation = "eager"
    from k3_receipt import module_state_dict

    block_size = index.attn_res_block_size or 1
    n_blocks = k3_common.blocks_before(layer_idx, block_size)
    snapshot_layer = (layer_idx % block_size == 0)

    def run(dtype):
        weights = layer_at(dtype, layer_idx, index)
        with torch.device("meta"):
            layer = modeling.KimiDecoderLayer(hf_config, layer_idx)
        missing, _ = layer.load_state_dict(
            module_state_dict(weights, True, expert_dtype=dtype), strict=False, assign=True)
        assert not missing, missing
        layer.eval()
        generator = torch.Generator().manual_seed(29)
        x = (torch.randn(1, tokens, index.hidden_size, generator=generator) * 0.05).to(dtype)
        blocks = (torch.randn(tokens, n_blocks, index.hidden_size, generator=generator) * 0.05).to(dtype)
        mask = torch.full((1, 1, tokens, tokens), float("-inf")).triu(1).to(dtype)
        with torch.no_grad():
            ref_prefix, ref_blocks = layer(x, attention_mask=mask, block_residual=blocks.clone())
        our_prefix, our_blocks, _ = StreamedLayer(weights, cfg, kda_backend="torch").forward(
            x, blocks.clone(), collect=True)
        assert our_blocks.shape == ref_blocks.shape, (our_blocks.shape, ref_blocks.shape)
        assert torch.equal(our_blocks, ref_blocks), "the block snapshot stack diverged"
        # the branch under test has to be the one we think it is
        assert (our_blocks.shape[1] == n_blocks + 1) == snapshot_layer, (
            f"L{layer_idx} B={block_size}: expected snapshot={snapshot_layer} but the stack went "
            f"{n_blocks} -> {our_blocks.shape[1]}")
        return our_prefix, ref_prefix

    ours32, ref32 = run(torch.float32)
    ours16, ref16 = run(torch.bfloat16)
    exact = close(ours32, ref32, 1e-5, f"L{layer_idx} prefix_out at fp32")
    floor = float((ref16.float() - ref32.float()).norm() / ref32.float().norm())
    gap = float((ours16.float() - ref16.float()).norm() / ref16.float().norm())
    # The ONLY assert here is self-consistency against the reference module's own
    # bf16-vs-fp32 difference.  The D-a2 line is a PROPOSED DECISION LINE for Cyril
    # and is REPORTED below, never enforced -- a harness that goes red on a proposed
    # threshold has quietly turned it into a gate on his behalf.
    assert gap <= max(2 * floor, 1e-3), f"bf16 gap {gap:.3e} exceeds the layer's own bf16 floor {floor:.3e}"
    return {"layer": layer_idx, "attn_res_block_size": block_size,
            "residual_branch": "snapshot (prefix reset to None)" if snapshot_layer
                               else "carried = prefix_sum + attn_out",
            "blocks_in": n_blocks, "blocks_out": n_blocks + (1 if snapshot_layer else 0),
            "fp32": exact, "bf16_gap": gap, "reference_bf16_floor": floor,
            "block_stack": "bit exact", "proposed_D_a2_prefix_relL2_max": 2e-2,
            "proposed_D_a2_note": "reported, NOT gated; the decision line is Cyril's",
            "bf16_gap_inside_proposed_D_a2": bool(gap < 2e-2)}


@test
def t10_whole_layer_matches_the_repo_layer() -> dict:
    """MLA + MoE layer end to end, including the attn-residual plumbing.

    The KDA layer cannot be referenced on CPU -- `KimiDeltaAttention.forward`
    goes straight to `chunk_kda`, a triton kernel -- so this is the MLA layer;
    `k3_receipt.py --kda-backend fla` is the GPU counterpart.

    The primary fixture is built at `attn_res_block_size = 2`, so the last layer
    (index 1) is a NON-snapshot layer and this test exercises
    `carried = prefix_sum + attn_out`.  t10b covers the snapshot branch and a
    two-snapshot stack on a second fixture.
    """
    if CONTEXT.get("modeling") is None:
        raise Skip(CONTEXT["modeling_error"])
    index: K3Index = CONTEXT["index"]
    layer_idx = index.num_layers - 1
    if index.is_kda_layer(layer_idx):
        raise Skip("the last layer is KDA in this configuration; needs a GPU")
    return whole_layer_against_repo(index, layer_idx)


@test
def t10b_whole_layer_covers_both_residual_branches() -> dict:
    """Both branches of `_forward_attn_residual`, on a block-size-3 fixture.

    Layer 3 is a snapshot layer (3 % 3 == 0, one block already on the stack);
    layer 5 is not (5 % 3 == 2, TWO blocks on the stack).  Without this the
    whole-layer evidence covers only the branch 8 of K3's 93 layers take.
    """
    if CONTEXT.get("modeling") is None:
        raise Skip(CONTEXT["modeling_error"])
    index: K3Index = CONTEXT.get("block_index")
    if index is None:
        raise Skip("the block-size fixture was not built")
    block_size = index.attn_res_block_size
    assert block_size > 1, block_size
    covered = {}
    for layer_idx in range(index.num_layers):
        if index.is_kda_layer(layer_idx) or not index.is_moe_layer(layer_idx):
            continue
        branch = "snapshot" if layer_idx % block_size == 0 else "carried"
        if branch in covered:
            continue
        covered[branch] = whole_layer_against_repo(index, layer_idx)
    assert set(covered) == {"snapshot", "carried"}, (
        f"the fixture did not offer both branches on an MLA+MoE layer: {sorted(covered)}")
    assert covered["carried"]["blocks_in"] >= 2, (
        "the non-snapshot layer must be reached with more than one snapshot on the stack; "
        f"got {covered['carried']['blocks_in']}")
    return covered


@test
def t11_streamed_pass_writes_a_resumable_cache() -> dict:
    work = CONTEXT["work"]
    cache = os.path.join(work, "cache")
    model_dir = CONTEXT["model_dir"]
    layers = CONTEXT["layers"]
    tokens = CONTEXT["tokens"]
    common = [sys.executable, os.path.join(HERE, "k3_stream.py"), "--model-dir", model_dir,
              "--cache-dir", cache, "--layers", f"0-{layers - 1}", "--random-tokens", str(tokens),
              "--vocab-size", str(CONTEXT["vocab"]), "--cache-set", "full", "--snapshot-every", "1",
              "--emit-logits", "--tokens-tag", str(tokens), "--body-dir", os.path.join(work, "body")]
    first = subprocess.run(common, cwd=HERE, capture_output=True, text=True, timeout=1800)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    cache_dir = os.path.join(cache, f"T{tokens}")
    receipt_path = os.path.join(cache_dir, "stream_receipt.json")
    receipt = json.load(open(receipt_path, encoding="utf-8"))
    assert len(receipt["layers"]) == layers, receipt["layers"].keys()
    for layer in range(1, layers):
        tensors = os.path.join(cache_dir, f"L{layer:02d}", "tensors.safetensors")
        assert os.path.exists(tensors), tensors
        assert os.path.exists(os.path.join(cache_dir, f"L{layer:02d}", "router_stats.json"))
    first_sha = receipt["final_prefix_sha256"]
    assert os.path.exists(os.path.join(work, "body", "L01.safetensors")), "body staging did not happen"

    # resume: the snapshots written on the first pass must reproduce the run
    resumed = subprocess.run(common + ["--resume"], cwd=HERE, capture_output=True, text=True, timeout=1800)
    assert resumed.returncode == 0, resumed.stdout[-2000:] + resumed.stderr[-2000:]
    again = json.load(open(receipt_path, encoding="utf-8"))
    assert again["final_prefix_sha256"] == first_sha, "resume changed the answer"
    last = str(layers - 1)
    assert again["layers"][last]["read"].get("staged_body"), "the resumed pass did not reuse the staged body"
    CONTEXT["cache_dir"] = cache_dir
    return {
        "layers_cached": len(receipt["layers"]),
        "final_prefix_sha256": first_sha[:16],
        "resume_reproduces": True,
        "logits_top_ids": receipt.get("logits", {}).get("last_token_top_ids", [])[:5],
        "staged_body_reused": bool(again["layers"]["1"]["read"].get("staged_body")),
    }


@test
def t11b_segmented_resume_reproduces_a_single_pass() -> dict:
    """Stream 0..m, stop, stream m+1..n with --resume, land on the same answer.

    Two separate defects met here.  A window that ended before the next
    snapshot boundary left nothing to resume from -- `resume_point` found no
    snapshot inside the new `--layers` range and `load_snapshot` raised -- so
    `k3_stream` now writes a snapshot for the layer after the window.  And a
    resumed pass used to rebuild `receipt["layers"]` from empty, discarding the
    per-layer read/sha record of every earlier segment; a 93-layer pass resumed
    three times kept only the last one.  Both are checked here.
    """
    work = CONTEXT["work"]
    cache = os.path.join(work, "cache-segmented")
    layers = CONTEXT["layers"]
    tokens = CONTEXT["tokens"]
    base = [sys.executable, os.path.join(HERE, "k3_stream.py"), "--model-dir", CONTEXT["model_dir"],
            "--cache-dir", cache, "--random-tokens", str(tokens), "--vocab-size", str(CONTEXT["vocab"]),
            "--cache-set", "minimal", "--snapshot-every", "1", "--tokens-tag", str(tokens)]
    cut = max(0, layers // 2 - 1)  # both windows must be non-empty (layers=2 -> 0-0 then 1-1)
    first = subprocess.run(base + ["--layers", f"0-{cut}"], cwd=HERE,
                           capture_output=True, text=True, timeout=1800)
    assert first.returncode == 0, first.stdout[-2000:] + first.stderr[-2000:]
    receipt_path = os.path.join(cache, f"T{tokens}", "stream_receipt.json")
    segment_one = json.load(open(receipt_path, encoding="utf-8"))
    assert sorted(int(k) for k in segment_one["layers"]) == list(range(cut + 1)), segment_one["layers"]
    assert segment_one["end_of_window_snapshot"]["layer"] == cut + 1, segment_one.get("end_of_window_snapshot")

    second = subprocess.run(base + ["--layers", f"{cut + 1}-{layers - 1}", "--resume"], cwd=HERE,
                            capture_output=True, text=True, timeout=1800)
    assert second.returncode == 0, second.stdout[-2000:] + second.stderr[-2000:]
    merged = json.load(open(receipt_path, encoding="utf-8"))
    assert sorted(int(k) for k in merged["layers"]) == list(range(layers)), merged["layers"].keys()
    assert merged["resumed_from_receipt"]["layers_carried_forward"] == list(range(cut + 1)), \
        merged["resumed_from_receipt"]
    # every earlier segment's read record survived, not just the last window's
    for layer in range(layers):
        assert "bytes" in merged["layers"][str(layer)]["read"], merged["layers"][str(layer)]

    single = json.load(open(os.path.join(CONTEXT["cache_dir"], "stream_receipt.json"), encoding="utf-8"))
    assert merged["final_prefix_sha256"] == single["final_prefix_sha256"], (
        "the segmented pass did not reproduce the single pass")
    return {"cut_after_layer": cut,
            "layers_in_merged_receipt": sorted(int(k) for k in merged["layers"]),
            "final_prefix_sha256": merged["final_prefix_sha256"][:16],
            "matches_single_pass": True}


@test
def t12_streamed_pass_equals_a_resident_pass() -> dict:
    """The streamed loop and the all-in-RAM loop must agree bit for bit."""
    from safetensors.torch import load_file
    from k3_review_stream import ReviewModel

    index: K3Index = CONTEXT["index"]
    receipt = json.load(open(os.path.join(CONTEXT["cache_dir"], "stream_receipt.json"), encoding="utf-8"))
    args = argparse.Namespace(device="cpu", kda_backend="torch", dense_dir=None, dense_layers=None, rung=None)
    model = ReviewModel(index, args)
    model.load()
    ids = torch.randint(0, CONTEXT["vocab"], (1, CONTEXT["tokens"]),
                        generator=torch.Generator().manual_seed(0), dtype=torch.long)
    assert k3_common.sha256_tensor(ids) == receipt["input_ids_sha256"], "the token seed drifted"
    with torch.no_grad():
        logits = model.logits(ids)
    streamed = load_file(os.path.join(CONTEXT["cache_dir"], "logits.safetensors"))["logits"]
    return {"logits": close(logits[0], streamed, 1e-6, "streamed vs resident logits"),
            "top1_agree": float((logits[0].argmax(-1) == streamed.argmax(-1)).float().mean())}


@test
def t12b_streamed_proxy_equals_the_resident_proxy() -> dict:
    """`ReviewModel.stream_proxy` must answer exactly what the resident loop does.

    This is the correctness half of the streaming fix; `t16` checks that the
    proxy entry point actually takes the streamed path.  Both cases go through
    one shared layer-major loop, so this also checks that carrying several
    residual pairs across the same layer does not cross-contaminate them.
    """
    from k3_review_stream import ReviewModel

    index: K3Index = CONTEXT["index"]
    args = argparse.Namespace(device="cpu", kda_backend="torch", dense_dir=None,
                              dense_layers=None, rung=None)
    generator = torch.Generator().manual_seed(53)
    ids = [torch.randint(0, CONTEXT["vocab"], (1, n), generator=generator, dtype=torch.long)
           for n in (11, 7)]

    streamed_model = ReviewModel(index, args)
    streamed, report = streamed_model.stream_proxy(ids)
    assert report["layers_streamed"] == index.num_layers, report
    assert report["cases_per_layer"] == len(ids), report
    # one report row per layer, i.e. each layer was read exactly once for BOTH cases
    assert len(streamed_model.load_report) == index.num_layers, len(streamed_model.load_report)

    resident = ReviewModel(index, args)
    resident.load()
    with torch.no_grad():
        expected = [resident.logits(case) for case in ids]
    scores = [close(a, b, 1e-6, f"streamed vs resident proxy logits, case {i}")
              for i, (a, b) in enumerate(zip(streamed, expected))]
    return {"cases": scores, "layer_reads": len(streamed_model.load_report),
            "gigabytes_read": report["gigabytes_read"]}


@test
def t13_pivoted_selection_and_gram_ridge_match_mx45() -> dict:
    generator = torch.Generator().manual_seed(31)
    features = torch.randn(64, 20, generator=generator, dtype=torch.float64)
    gram = features.T @ features
    fast = c14.pivoted_columns_gram(gram, 7)
    slow = c14.pivoted_columns_gram(gram, 7, exact_scan=True)
    assert torch.equal(fast, slow), (fast, slow)
    keep_all = c14.pivoted_columns_gram(gram, 20)
    assert torch.equal(keep_all, torch.arange(20)), "K >= |U| must be the identity selection"

    # `prior_anchored_gram` vs mx45_build's dense `prior_anchored`, verbatim
    design = torch.randn(80, 12, generator=generator, dtype=torch.float64)
    target = torch.randn(80, 5, generator=generator, dtype=torch.float64)
    prior = torch.randn(5, 12, generator=generator, dtype=torch.float64)

    def prior_anchored_dense(A, Y, P, rel):
        G = A.T @ A
        lam = rel * float(torch.diagonal(G).mean())
        G = G + lam * torch.eye(G.shape[0], dtype=G.dtype)
        B = A.T @ Y + lam * P.T
        return torch.cholesky_solve(B, torch.linalg.cholesky(G)).T.contiguous()

    dense = prior_anchored_dense(design, target, prior, 1e-3)
    gram_form = c14.prior_anchored_gram(design.T @ design, design.T @ target, prior, 1e-3)
    # the ridge must not touch the Gram it was handed -- the ladder reuses it 19 times
    reused = design.T @ design
    before = reused.clone()
    c14.prior_anchored_gram(reused, design.T @ target, prior, 1e-2)
    assert torch.equal(reused, before), "prior_anchored_gram mutated its Gram"
    return {"selection_identical": True,
            "gram_not_mutated": True,
            "ridge": close(gram_form, dense, 1e-10, "gram ridge vs mx45 dense ridge")}


@test
def t13b_law_abiding_solve_matches_a_three_gram_reference() -> dict:
    """The two-Gram solver must answer exactly what the three-Gram one did.

    `law_abiding_solve_gram` now accumulates only the inner and check systems
    and derives the full-fit system by adding them in place -- the splits are
    disjoint, so that is an identity, but "is an identity" is a claim and this
    is the measurement.  The reference here is a literal re-implementation of
    the previous three-Gram, nine-stored-solutions version.
    """
    generator = torch.Generator().manual_seed(37)
    n, width, out_dim = 120, 14, 6
    design = torch.randn(n, width, generator=generator, dtype=torch.float64)
    target = torch.randn(n, out_dim, generator=generator, dtype=torch.float64)
    prior = (torch.randn(out_dim, width, generator=generator, dtype=torch.float64) * 0.3).float()
    inner = torch.zeros(n, dtype=torch.bool)
    inner[: 2 * n // 3] = True
    check = ~inner

    def gram_of(mask):
        a = design[mask]
        return a.T @ a, a.T @ target[mask]

    gram_inner, cross_inner = gram_of(inner)
    gram_check, cross_check = gram_of(check)
    gram_all, cross_all = gram_of(torch.ones(n, dtype=torch.bool))
    check_y_sq = float(target[check].pow(2).sum())

    # reference: three independent Grams, every ladder solution stored
    reference_rungs = []
    for rel in c14.LAMBDA_LADDER:
        inner_w = c14.prior_anchored_gram(gram_inner, cross_inner, prior, rel=rel)
        full_w = c14.prior_anchored_gram(gram_all, cross_all, prior, rel=rel)
        reference_rungs.append({
            "lambda_relative": rel,
            "check_sse": c14._check_sse(inner_w, gram_check, cross_check, check_y_sq),
            "update_scale": float((full_w - prior).norm() / float(prior.norm())),
            "solved": full_w,
        })
    admitted = [r for r in reference_rungs if r["update_scale"] <= 1.0]
    reference = min(admitted, key=lambda r: r["check_sse"])

    solved, ladder = c14.law_abiding_solve_gram(
        gram_inner.clone(), cross_inner.clone(), gram_check.clone(), cross_check.clone(),
        check_y_sq, prior,
    )
    assert ladder["lambda_relative"] == reference["lambda_relative"], (
        ladder["lambda_relative"], reference["lambda_relative"])
    for key in ("veto_binding", "lambdas_vetoed", "chosen_is_largest_lambda",
                "check_sse_monotone_decreasing_in_lambda"):
        assert key in ladder, key
    # the in-place fold of check into inner must reproduce the separately
    # accumulated full-fit Gram
    assert torch.allclose(gram_inner + gram_check, gram_all, atol=1e-9), "inner + check != all"
    return {"solution": close(solved.double(), reference["solved"], 1e-10,
                              "two-Gram solve vs three-Gram reference"),
            "lambda_relative": ladder["lambda_relative"],
            "veto_binding": ladder["veto_binding"],
            "free_solve_update_scale": round(ladder["free_solve_update_scale"], 5)}


@test
def t13c_memory_estimates_match_the_documented_table() -> dict:
    """The fp64 budget in the README/module docstring is checked, not asserted.

    A memory table nobody recomputes is how the first version of this lane came
    to understate the refit by ~3x.  Real K3 out_dim is 3,584.
    """
    expected = {8704: 3.4, 17408: 11.7, 32768: 38.1, 65536: 145.0}
    got = {}
    for width, peak in expected.items():
        estimate = c14.gram_memory_estimate(width, 3584)
        got[width] = estimate["peak_gigabytes"]
        assert abs(estimate["peak_gigabytes"] - peak) < 0.06, (width, estimate)
        # the identity that used to be built would have added a whole Gram
        assert estimate["peak_bytes"] == 4 * (width * width * 8) + 4 * (width * 3584 * 8), estimate
    selection = c14.selection_memory_estimate(131072, 65536)
    assert abs(selection["cholesky_factor_gigabytes"] - 68.719) < 0.01, selection
    assert selection["python_iterations"] == 65536, selection
    # the DEVICE side is a different pool.  Its TRANSIENT half must be bounded by
    # --chunk, not by K: a [K, K] fp64 Gram on the compute device is 34.4 GB at
    # K = 65,536, which no card here has, and the host budget above cannot see it.
    # The RESIDENT half -- the packed experts, the cache, the bank, score's
    # buffers, the form-M lift -- is t13d's job, and it is ~20x this number.
    device = c14.device_memory_estimate(65536, 3584, 1024, 3072)
    assert device["peak_bytes"] < 1e9, device
    doubled = c14.device_memory_estimate(2 * 65536, 3584, 1024, 3072)
    assert doubled["peak_bytes"] < 2.1 * device["peak_bytes"], (device, doubled)
    assert device["stage1_expert_gram_bytes"] == 3072 * 3072 * 8, device
    return {"refit_peak_gigabytes": got,
            "stage2_at_pool_131072": {k: selection[k] for k in
                                      ("pool_gram_gigabytes", "cholesky_factor_gigabytes",
                                       "peak_gigabytes", "python_iterations")},
            "device_transient_peak_gigabytes_at_K65536_chunk1024": device["peak_gigabytes"],
            "device_transients_are_chunk_bounded_not_quadratic_in_K": True,
            "resident_terms_are_checked_by": "t13d"}


@test
def t13d_the_device_budget_prices_what_actually_lands_on_the_device() -> dict:
    """`--max-device-gb` must see the RESIDENT terms, not 4% of the truth.

    The shipped estimator priced only the per-chunk transients.  At K3 scale
    that is ~0.94 GB against ~18.5 GB actually resident, because
    `load_layer(index, layer, device, device)` put all 896 PACKED experts on
    the compute device (896 x 3 x 5,505,024 B = 14.80 GB of `weight_packed`
    plus 0.93 GB of E8M0 scales), `CoresetFold` moved the calibration cache
    there, `ExpertBank` put the kept rows there and `--form-m` allocated a
    [65536, 7168] fp32 lift.  So the README's own K = 65,536 entry point on a
    25.3 GB 3090 was told the rung cost 0.94 GB, was ADMITTED by a 22.77 GB
    budget, and then died inside the allocator -- which is exactly what
    `enforce_budget` exists to prevent.

    Checked here: the arithmetic of each resident term at real K3 widths, that
    the old transient-only answer would have been admitted where the new one is
    refused, and that a real fold on the fixture refuses with the resident
    terms NAMED in the message.
    """
    latent, hidden, intermediate, experts = 3584, 7168, 3072, 896
    packed_bytes = experts * 3 * (intermediate * latent // 2)
    scale_bytes = experts * 3 * (intermediate * latent // 32)
    assert abs(packed_bytes - 14.80e9) < 0.05e9, packed_bytes
    assert abs(scale_bytes - 0.93e9) < 0.02e9, scale_bytes

    bank = c14.expert_bank_bytes(65536, latent, 2)
    assert bank == 3 * 65536 * 3584 * 2, bank
    lift = c14.form_m_bytes(65536, latent, hidden, 2)
    assert lift > 65536 * hidden * 4, lift            # the fp32 [K, hidden] lift alone is 1.88 GB

    resident = {
        "layer_weights_on_device": packed_bytes + scale_bytes,
        "calibration_cache_on_device": 65536 * latent * 2 + 65536 * latent * 4,
        "expert_bank": bank,
        "form_m_lift": lift,
    }
    priced = c14.device_memory_estimate(65536, latent, 1024, intermediate, resident=resident)
    naive = c14.device_memory_estimate(65536, latent, 1024, intermediate)
    assert naive["resident_total_bytes"] == 0, naive
    assert priced["peak_bytes"] == naive["transient_peak_bytes"] + sum(resident.values()), priced
    # a 25.3 GB 3090 at the default 0.9x
    budget_gb = round(0.9 * 25.3, 3)
    assert naive["peak_bytes"] / 1e9 < budget_gb, naive          # the old answer: admitted
    assert priced["peak_bytes"] / 1e9 > budget_gb, priced        # the honest one: refused
    try:
        c14.enforce_budget(priced, budget_gb, "the K=65536 rung", flag="--max-device-gb")
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("the priced estimate was admitted by a 22.77 GB budget")
    assert "layer_weights_on_device" in message and "form_m_lift" in message, message[:400]

    # and end to end on the fixture: the guard must fire with the terms named
    cache_file = os.path.join(CONTEXT["cache_dir"], "L01", "tensors.safetensors")
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "c14_union_coreset.py"), "--model-dir", CONTEXT["model_dir"],
         "--layer", "1", "--cache", cache_file,
         "--out-dir", os.path.join(CONTEXT["work"], "folds-refused"),
         "--rungs", "32", "--k-pool-factor", "1", "--chunk", "8",
         "--max-device-gb", "0.000001",
         "--out", os.path.join(CONTEXT["work"], "c14_refused.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert proc.returncode != 0, proc.stdout[-1500:]
    assert "--max-device-gb" in proc.stderr, proc.stderr[-800:]
    assert "layer_weights_on_device" in proc.stderr, proc.stderr[-800:]
    assert "expert_bank" in proc.stderr, proc.stderr[-800:]
    return {
        "packed_expert_bytes_on_a_real_moe_layer": packed_bytes,
        "e8m0_scale_bytes": scale_bytes,
        "transient_only_peak_gigabytes_at_K65536": naive["peak_gigabytes"],
        "priced_peak_gigabytes_at_K65536": priced["peak_gigabytes"],
        "under_pricing_factor": round(priced["peak_bytes"] / naive["peak_bytes"], 1),
        "budget_gigabytes_on_a_25_3_GB_3090": budget_gb,
        "old_estimate_admitted": True,
        "new_estimate_refused": True,
        "fixture_guard_names_the_resident_terms": True,
    }


@test
def t13e_packed_experts_default_to_the_host() -> dict:
    """`--experts-device` defaults to cpu, and that is what makes the top rung fit.

    `load_layer(index, layer, args.device, args.device)` -- the old call --
    put 15.7 GB of packed experts and E8M0 scales on the compute card before a
    token moved.  `LayerWeights.device_bytes` is measured, so this test can
    check the placement rather than the intention.
    """
    index: K3Index = CONTEXT["index"]
    both_here, _ = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    counts = both_here.device_bytes("cpu")
    assert counts["experts_packed"] > 0, counts
    assert counts["total"] == sum(v for k, v in counts.items() if k != "total"), counts
    # a device the layer is NOT on must price at zero, which is the property the
    # cpu default relies on
    elsewhere = both_here.device_bytes("meta")
    assert elsewhere["total"] == 0, elsewhere
    import inspect

    signature = inspect.signature(c14.CoresetFold.__init__)
    assert "form_m" in signature.parameters, signature
    return {"fixture_layer_bytes_on_cpu": counts,
            "device_bytes_of_an_absent_device": elsewhere["total"]}


@test
def t13f_a_device_alias_does_not_zero_the_resident_estimate() -> dict:
    """`cuda` and `cuda:0` are one card; `torch.device` says they are not.

    `LayerWeights.device_bytes` and `CoresetFold.device_resident_bytes` count a
    tensor only when its device equals the device they were handed, and every
    such test used to be a raw `==`.  A tensor moved to `cuda` reports
    `cuda:0`, so `--device cuda` priced a fully resident layer at ZERO bytes:
    the 15.7 GB of packed experts, the calibration cache and the expert bank
    all vanished from the estimate and `--max-device-gb` was back to admitting
    exactly the rung t13d proves must be refused -- silently, because a budget
    that under-counts does not raise, it lets the allocator do the raising
    later.

    The same alias exists on the host (`cpu:0` vs `cpu`, and a tensor built on
    `'cpu:0'` reports `cpu`), so the first arm needs no GPU and still fails on
    the pre-fix code.  `raw_equality_hits` asserts the alias is genuinely
    invisible to `==`, i.e. that this test is not vacuous on some future torch
    that starts indexing CPU tensors.
    """
    from safetensors.torch import load_file

    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    cache = load_file(os.path.join(CONTEXT["cache_dir"], "L01", "tensors.safetensors"))
    out_dim = int(cache["y_lat"].shape[1])
    width = 32

    # -- the helper's own contract
    assert k3_common.canonical_device("cpu:0") == torch.device("cpu"), "cpu tensors carry no index"
    assert k3_common.canonical_device("meta") == torch.device("meta")
    assert k3_common.same_device("cpu", "cpu:0") and not k3_common.same_device("cpu", "meta")

    # -- host arm.  Fails on the pre-fix code with aliased["total"] == 0.
    layer, _ = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    raw_equality_hits = sum(
        1
        for group in (layer.attn, layer.res, layer.body)
        for t in group.values()
        if t.device == torch.device("cpu:0")
    )
    assert raw_equality_hits == 0, "cpu:0 is no longer an alias here; pick another pair"
    plain = layer.device_bytes("cpu")
    aliased = layer.device_bytes("cpu:0")
    assert plain["total"] > 0 and plain["experts_packed"] > 0, plain
    assert aliased == plain, (aliased, plain)

    def resident(device) -> dict:
        fold = c14.CoresetFold(layer, cache, cfg, device, chunk=8)
        return fold.device_resident_bytes(width, out_dim, include_score=True, include_form_m=True)

    host_plain, host_alias = resident(torch.device("cpu")), resident(torch.device("cpu:0"))
    assert host_plain["calibration_cache_on_device"] > 0, host_plain
    assert host_plain["layer_weights_on_device"] == plain["total"], (host_plain, plain)
    assert host_alias == host_plain, (host_alias, host_plain)

    detail = {
        "raw_equality_hits_on_the_alias": raw_equality_hits,
        "layer_bytes_cpu": plain["total"],
        "layer_bytes_cpu0_alias": aliased["total"],
        "resident_cpu": host_plain,
        "resident_cpu0_alias": host_alias,
        "cuda_arm": "no CUDA device visible",
    }

    # -- device arm.  Same property where it actually costs VRAM.
    if torch.cuda.is_available():
        layer.to("cuda", "cuda")            # in place; the fixture layer is tiny
        indexed = layer.device_bytes("cuda:0")
        alias = layer.device_bytes("cuda")
        assert indexed["total"] > 0 and indexed["experts_packed"] > 0, indexed
        assert alias == indexed, (alias, indexed)
        card_alias = resident(torch.device("cuda"))
        card_indexed = resident(torch.device("cuda:0"))
        assert card_alias == card_indexed, (card_alias, card_indexed)
        assert card_alias["layer_weights_on_device"] == indexed["total"], (card_alias, indexed)
        assert card_alias["calibration_cache_on_device"] > 0, card_alias
        # and the number the guard compares against a budget is the priced one
        budget = c14.CoresetFold(layer, cache, cfg, torch.device("cuda"), chunk=8).device_budget(
            width, out_dim, include_score=True, include_form_m=True)
        assert budget["resident_total_bytes"] == sum(card_alias.values()), budget
        assert budget["peak_bytes"] > budget["transient_peak_bytes"], budget
        budget_gb_alias = c14.default_device_budget_gb(torch.device("cuda"))
        budget_gb_indexed = c14.default_device_budget_gb(torch.device("cuda:0"))
        assert budget_gb_alias == budget_gb_indexed > 0, (budget_gb_alias, budget_gb_indexed)
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        detail["cuda_arm"] = {
            "card": properties.name,
            "capability": f"sm{properties.major}{properties.minor}",
            "layer_bytes_cuda0": indexed["total"],
            "layer_bytes_cuda_alias": alias["total"],
            "resident_cuda_alias": card_alias,
            "priced_peak_bytes": budget["peak_bytes"],
            "transient_only_peak_bytes": budget["transient_peak_bytes"],
            "default_budget_gigabytes": budget_gb_alias,
        }
        layer.to("cpu", "cpu")
        del budget
        torch.cuda.empty_cache()
    return detail


@test
def t14_c14_coreset_folds_a_layer() -> dict:
    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    cache_file = os.path.join(CONTEXT["cache_dir"], "L01", "tensors.safetensors")
    out_dir = os.path.join(CONTEXT["work"], "folds")
    union = cfg["num_experts"] * cfg["moe_intermediate_size"]
    rungs = [union // 4, union // 8, union // 16]
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "c14_union_coreset.py"), "--model-dir", CONTEXT["model_dir"],
         "--layer", "1", "--cache", cache_file, "--out-dir", out_dir,
         "--rungs", ",".join(str(r) for r in rungs), "--k-pool-factor", "2", "--form-m", "--chunk", "8"],
        cwd=HERE, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    receipt = json.load(open(os.path.join(out_dir, "L01_c14_receipt.json"), encoding="utf-8"))
    # the placement of the packed experts is part of the record, and the printed
    # per-rung device table must be the SAME estimate the guard uses -- resident
    # terms included -- or the table and the refusal can disagree
    assert receipt["experts_device"] == "cpu", receipt["experts_device"]
    assert receipt["layer_bytes_on_device"]["experts_packed"] >= 0, receipt["layer_bytes_on_device"]
    for rung, entry in receipt["memory_budget"]["per_rung_device_estimate"].items():
        assert entry["resident_total_bytes"] > 0, (rung, entry)
        assert "layer_weights_on_device" in entry["resident_bytes"], (rung, entry)
        assert "expert_bank" in entry["resident_bytes"], (rung, entry)
        assert "form_m_lift" in entry["resident_bytes"], (rung, entry)   # --form-m is on here
        assert entry["peak_bytes"] == entry["transient_peak_bytes"] + entry["resident_total_bytes"], entry
    # the holdout must be contiguous blocks and the splits must be non-empty
    assert receipt["splits"]["holdout_mode"] == "blocks", receipt["splits"]
    assert min(receipt["splits"]["held_tokens"], receipt["splits"]["inner_tokens"],
               receipt["splits"]["check_tokens"]) > 0, receipt["splits"]
    assert receipt["peak_rss_gigabytes"] is None or receipt["peak_rss_gigabytes"] > 0, receipt
    widths, errors = [], []
    for rung in rungs:
        entry = receipt["rungs"][str(rung)]
        assert entry["ridge"]["update_scale"] <= 1.0, entry["ridge"]
        # the diagnostic that tells a no-signal fold from a healthy one
        assert isinstance(entry["ridge"]["veto_binding"], bool), entry["ridge"]
        assert "memory" in entry["ridge"], entry["ridge"].keys()
        assert entry["ridge"]["memory"]["peak_bytes"] > 0, entry["ridge"]["memory"]
        assert entry["parameters"]["quantization"].startswith("none")
        widths.append(entry["width_installed"])
        errors.append(entry["heldout"]["y_lat_heldout"]["relL2"])
        assert errors[-1] < 1.0, f"K={widths[-1]} fits worse than predicting zero: relL2 {errors[-1]}"
        assert os.path.exists(entry["form_E_path"]), entry["form_E_path"]
        assert os.path.exists(entry["form_M_path"]), entry["form_M_path"]
    # a wider coreset must not fit the held-out routed write worse than a narrow one
    assert widths[0] > widths[-1], widths
    assert errors[0] <= errors[-1] + 1e-3, f"widths {widths} gave errors {errors}"
    # the channels a fold installs must be distinct (e, j) pairs
    from safetensors.torch import load_file

    installed = load_file(receipt["rungs"][str(rungs[1])]["form_E_path"])
    pairs = installed["channels"].to(torch.int64)
    keyed = pairs[:, 0] * cfg["moe_intermediate_size"] + pairs[:, 1]
    assert int(torch.unique(keyed).numel()) == int(keyed.numel()), "duplicate channels installed"
    assert installed["w1_s"].shape[0] == widths[1], installed["w1_s"].shape
    # an unaffordable rung must be refused WITH the arithmetic, not OOM-killed
    refused = subprocess.run(
        [sys.executable, os.path.join(HERE, "c14_union_coreset.py"), "--model-dir", CONTEXT["model_dir"],
         "--layer", "1", "--cache", cache_file, "--out-dir", os.path.join(CONTEXT["work"], "folds-refused"),
         "--rungs", str(rungs[0]), "--k-pool-factor", "1", "--chunk", "8",
         "--max-gram-gb", "0.0000001"],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert refused.returncode != 0, refused.stdout[-1000:]
    assert "--max-gram-gb" in refused.stderr and "peak_gigabytes" in refused.stderr, refused.stderr[-800:]

    # the fold FILE is keyed by the requested rung; the installed width is a
    # separate number and the two must not be conflated by any consumer
    for rung in rungs:
        entry = receipt["rungs"][str(rung)]
        assert os.path.basename(entry["form_E_path"]).endswith(f"_K{rung}_formE.safetensors"), (
            entry["form_E_path"], rung)
        assert entry["rung_requested"] == rung, entry
    CONTEXT["fold_dir"] = out_dir
    CONTEXT["fold_rung"] = rungs[1]
    CONTEXT["fold_width"] = widths[1]
    return {"widths": widths,
            "rungs_requested": rungs,
            "fold_files_named_by_requested_rung": True, "heldout_relL2": [round(e, 5) for e in errors],
            "veto_binding": [receipt["rungs"][str(r)]["ridge"]["veto_binding"] for r in rungs],
            "estimated_peak_gigabytes": [receipt["rungs"][str(r)]["ridge"]["memory"]["peak_gigabytes"]
                                         for r in rungs],
            "process_peak_rss_so_far_gigabytes": receipt["peak_rss_gigabytes"],
            "unaffordable_rung_refused_with_arithmetic": True,
            "routed_ratio": [receipt["rungs"][str(r)]["parameters"]["routed_component_ratio"] for r in rungs],
            "update_scale": [round(receipt["rungs"][str(r)]["ridge"]["update_scale"], 4) for r in rungs]}


@test
def t14b_stage1_only_lands_on_the_rung_exactly() -> dict:
    """`--k-pool-factor 1` must SKIP stage 2, not merely usually skip it.

    `round(mass_e / total * k_pool)` with a floor of 1 does not sum to k_pool --
    the lane's own shipped receipt has 256 -> 257 -- and `stage2` skips only
    when the pool is at or below the rung.  At K = 65,536 that one extra column
    starts a 65,536-iteration pivoted Cholesky with a 34.4 GB fp64 factor,
    admitted by `--max-gram-gb 160` because 68.7 GB < 160.  So the apportionment
    is exact now, and `run_rung` raises if stage 2 ran under the stage-1-only
    policy anyway.

    Two measurements: the apportionment on its own over a spread of requests,
    and a real stage-1-only run of the tool.
    """
    generator = torch.Generator().manual_seed(53)
    mass = torch.rand(64, generator=generator).double()
    mass[torch.tensor([3, 17, 40])] = 0.0            # dead experts get nothing
    allocations = {}
    capacity = 61 * 12          # 61 live experts, cap 12 channels each
    for k_pool in (7, 61, 64, 65, 128, 257, 500, capacity):
        alloc, stats = c14.mass_allocation(mass, k_pool, 12)
        assert int(alloc.sum()) == k_pool, (k_pool, stats)
        assert int(alloc[torch.tensor([3, 17, 40])].sum()) == 0, "a dead expert was allocated slots"
        assert int(alloc.max()) <= 12, (k_pool, int(alloc.max()))
        assert stats["exact"], stats
        allocations[k_pool] = stats["mode"]
    # at capacity every live expert is at its cap and every dead one is at zero
    capped, _ = c14.mass_allocation(mass, capacity, 12)
    assert int(capped.min()) == 0 and int(capped[capped > 0].min()) == 12, capped
    # and a request BEYOND capacity undershoots honestly rather than overshooting:
    # the pool may only ever land at or below the rung, never above it
    over, over_stats = c14.mass_allocation(mass, capacity + 200, 12)
    assert int(over.sum()) == capacity, over_stats
    assert over_stats["exact"] is False, over_stats

    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    cache_file = os.path.join(CONTEXT["cache_dir"], "L01", "tensors.safetensors")
    out_dir = os.path.join(CONTEXT["work"], "folds-stage1-only")
    rung = cfg["num_experts"] * cfg["moe_intermediate_size"] // 8
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "c14_union_coreset.py"), "--model-dir", CONTEXT["model_dir"],
         "--layer", "1", "--cache", cache_file, "--out-dir", out_dir,
         "--rungs", str(rung), "--k-pool-factor", "1", "--chunk", "8"],
        cwd=HERE, capture_output=True, text=True, timeout=1800)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    entry = json.load(open(os.path.join(out_dir, "L01_c14_receipt.json"),
                           encoding="utf-8"))["rungs"][str(rung)]
    assert entry["stage1_only"] is True, entry
    assert entry["stage1"]["k_pool_requested"] == rung, entry["stage1"]
    assert entry["stage1"]["k_pool_allocated"] == rung, entry["stage1"]
    assert entry["stage1"]["allocation"]["exact"] is True, entry["stage1"]["allocation"]
    assert entry["stage1"]["k_pool_selected"] <= rung, entry["stage1"]
    assert "skipped" in entry["stage2"], entry["stage2"]
    assert os.path.basename(entry["form_E_path"]) == f"L01_K{rung}_formE.safetensors", entry
    # and the device budget is priced separately from the host one
    assert entry["ridge"]["device_memory"]["peak_bytes"] > 0, entry["ridge"]["device_memory"]
    return {"apportionment_modes": allocations,
            "stage1_only_rung": rung,
            "k_pool_allocated": entry["stage1"]["k_pool_allocated"],
            "k_pool_selected": entry["stage1"]["k_pool_selected"],
            "stage2": entry["stage2"]["skipped"],
            "width_installed": entry["width_installed"],
            "device_peak_gigabytes": entry["ridge"]["device_memory"]["peak_gigabytes"]}


@test
def t15_dense_fold_installs_and_changes_the_layer() -> dict:
    """The form-E child must run in place of the MoE half and be close, not identical."""
    from safetensors.torch import load_file

    index: K3Index = CONTEXT["index"]
    cfg = index.text_config
    weights, _ = load_layer(index, 1, "cpu", "cpu", log_reads=False)
    fold_path = [os.path.join(CONTEXT["fold_dir"], f) for f in os.listdir(CONTEXT["fold_dir"])
                 if f.endswith("formE.safetensors") and f"_K{CONTEXT['fold_rung']}_" in f][0]
    fold = load_file(fold_path)
    dense = {k: fold[k] for k in ("w1_s", "w3_s", "w2p")}
    cache = load_file(os.path.join(CONTEXT["cache_dir"], "L01", "tensors.safetensors"))
    x = cache["moe_x"].unsqueeze(0)
    parent, _ = k3_layer.moe_forward(weights, x, cfg, collect=False)
    child, taps = k3_layer.moe_forward(weights, x, cfg, collect=False, dense=dense)
    rel = float((child.float() - parent.float()).norm() / parent.float().norm())
    assert taps["dense_channels"] == CONTEXT["fold_width"], taps
    assert rel < 1.0, f"the dense child is further from the parent than zero would be (relL2 {rel})"
    return {"moe_out_relL2_child_vs_parent": rel, "dense_channels": taps["dense_channels"]}


@test
def t16_review_runner_proxy_and_decode() -> dict:
    work = CONTEXT["work"]
    model_dir = CONTEXT["model_dir"]
    cases = os.path.join(work, "cases.json")
    generator = torch.Generator().manual_seed(41)
    with open(cases, "w", encoding="utf-8") as handle:
        json.dump([{"id": f"case-{i}",
                    "input_ids": torch.randint(0, CONTEXT["vocab"], (12,), generator=generator).tolist(),
                    "reference_ids": torch.randint(0, CONTEXT["vocab"], (6,), generator=generator).tolist()}
                   for i in range(3)], handle)

    baseline = os.path.join(work, "parent_logits.safetensors")
    parent = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", model_dir,
         "--mode", "proxy", "--case-file", cases, "--emit-baseline", baseline,
         "--out", os.path.join(work, "review_parent.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert parent.returncode == 0, parent.stdout[-2000:] + parent.stderr[-2000:]

    child = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", model_dir,
         "--mode", "proxy", "--case-file", cases, "--dense-dir", CONTEXT["fold_dir"],
         "--rung", str(CONTEXT["fold_rung"]), "--baseline", baseline,
         "--out", os.path.join(work, "review_child.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert child.returncode == 0, child.stdout[-2000:] + child.stderr[-2000:]
    parent_receipt = json.load(open(os.path.join(work, "review_parent.json"), encoding="utf-8"))
    # the parent baseline has NO folds, so this is the run a resident proxy could
    # not do on the real model; it must have gone through the streamed path
    assert parent_receipt["layers_folded"] == [], parent_receipt["layers_folded"]
    assert parent_receipt["stream"]["mode"].startswith("streamed"), parent_receipt["stream"]
    assert len(parent_receipt["load"]["per_layer"]) == CONTEXT["layers"], parent_receipt["load"]

    child_receipt = json.load(open(os.path.join(work, "review_child.json"), encoding="utf-8"))
    assert child_receipt["layers_folded"] == [1], child_receipt["layers_folded"]
    assert child_receipt["moe_layers_unfolded"] == 0, child_receipt
    assert child_receipt["stream"]["mode"].startswith("streamed"), child_receipt["stream"]
    drift = [case["vs_parent"] for case in child_receipt["cases"]]
    assert all("kl_parent_child_mean" in d for d in drift), drift
    assert all("vs_parent_continuation_only" in case for case in child_receipt["cases"]), (
        "the continuation-only agreement is missing")
    # a folded layer must not read that layer's packed experts at all
    folded_rows = [r for r in child_receipt["load"]["per_layer"] if r["folded"]]
    unfolded_rows = [r for r in parent_receipt["load"]["per_layer"] if r["layer"] == 1]
    assert folded_rows and unfolded_rows, (folded_rows, unfolded_rows)
    assert folded_rows[0]["bytes"] < unfolded_rows[0]["bytes"], (folded_rows, unfolded_rows)
    expert_bytes_skipped = unfolded_rows[0]["bytes"] - folded_rows[0]["bytes"]

    decode = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", model_dir,
         "--mode", "decode", "--case-file", cases, "--dense-dir", CONTEXT["fold_dir"],
         "--rung", str(CONTEXT["fold_rung"]), "--max-new-tokens", "6",
         "--out", os.path.join(work, "review_decode.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert decode.returncode == 0, decode.stdout[-2000:] + decode.stderr[-2000:]
    decoded = json.load(open(os.path.join(work, "review_decode.json"), encoding="utf-8"))
    assert all(len(case["tokens"]) > 0 for case in decoded["cases"]), decoded["cases"]

    refused = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", model_dir,
         "--mode", "decode", "--case-file", cases, "--out", os.path.join(work, "review_refuse.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert refused.returncode != 0 and "cannot generate" in refused.stderr, refused.stderr[-500:]
    # the refusal must fire BEFORE any weight is read.  On the real model the
    # resident load it was guarding is 1.44 TB, so a guard placed after load()
    # is a guard that never runs -- the process dies first.  `_load_one_layer`
    # is the only thing that logs "loaded", so its absence is the evidence.
    assert "loaded" not in refused.stdout, refused.stdout[-800:]
    return {
        "proxy_drift": [{k: round(v, 5) for k, v in d.items()} for d in drift],
        "proxy_streamed": True,
        "expert_bytes_skipped_by_the_fold": int(expert_bytes_skipped),
        "decoded_tokens": [case["tokens"] for case in decoded["cases"]],
        "decode_tokens_per_second": [case["tokens_per_second"] for case in decoded["cases"]],
        "decode_without_folds_refused": True,
        "decode_refusal_read_no_weights": True,
    }


@test
def t16b_a_fold_request_that_matches_nothing_is_refused() -> dict:
    """A `--rung` that selects zero folds must NOT quietly review the parent.

    This is the failure mode the fold-file rename exists to prevent: with the
    file named by the width INSTALLED and `--rung` naming the width REQUESTED,
    a single rank-starved expert made `discover_folds` return {} and proxy mode
    then scored the unfolded parent against the parent baseline -- KL ~ 0,
    top-1 agreement 1.0, i.e. a perfect child.  Only `layers_folded: []` said
    otherwise and nothing read it.
    """
    work = CONTEXT["work"]
    cases = os.path.join(work, "cases.json")
    assert os.path.exists(cases), "t16 must have run first"
    missing = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", CONTEXT["model_dir"],
         "--mode", "proxy", "--case-file", cases, "--dense-dir", CONTEXT["fold_dir"],
         "--rung", "999999", "--out", os.path.join(work, "review_missing.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert missing.returncode != 0, missing.stdout[-1500:]
    assert "NO fold file matched" in missing.stderr, missing.stderr[-800:]
    assert "Rungs present" in missing.stderr, missing.stderr[-800:]
    # and it refuses before reading a weight, like the decode refusal
    assert "loaded" not in missing.stdout, missing.stdout[-800:]
    # a layer that has no fold at all is refused too
    bad_layer = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"), "--model-dir", CONTEXT["model_dir"],
         "--mode", "proxy", "--case-file", cases, "--dense-dir", CONTEXT["fold_dir"],
         "--dense-layers", "0", "--rung", str(CONTEXT["fold_rung"]),
         "--out", os.path.join(work, "review_badlayer.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert bad_layer.returncode != 0, bad_layer.stdout[-1500:]
    assert "no fold matched" in bad_layer.stderr or "NO fold file matched" in bad_layer.stderr, (
        bad_layer.stderr[-800:])
    return {"unmatched_rung_refused": True,
            "unmatched_dense_layer_refused": True,
            "refusal_read_no_weights": True,
            "refusal_lists_available_rungs": True}


@test
def t16c_a_folded_run_cannot_be_written_or_read_as_a_parent_baseline() -> dict:
    """`--emit-baseline` must refuse a folded run, and `--baseline` must refuse a child.

    The silent-parent failure in the other direction from t16b, and it needs
    exactly one forgotten flag: copy the child command line, leave
    `--dense-dir`/`--rung` on, add `--emit-baseline parent_logits.safetensors`.
    `_check_requested_folds` is SATISFIED (folds were found), the run streams a
    child, and that child is now the "parent" every later comparison is scored
    against -- KL ~ 1e-4, high top-1 agreement, a far better child than the
    truth, and nothing in `review_child.json` said which baseline it was.

    Three things are checked: the emit refusal, the read refusal off the file's
    own stamp, and that a legitimate comparison now RECORDS `baseline_path`,
    `baseline_sha256` and the baseline's fold census in the receipt.
    """
    work = CONTEXT["work"]
    cases = os.path.join(work, "cases.json")
    assert os.path.exists(cases), "t16 must have run first"
    baseline = os.path.join(work, "parent_logits.safetensors")
    assert os.path.exists(baseline), "t16 must have emitted the parent baseline"

    def review(*extra, out: str):
        return subprocess.run(
            [sys.executable, os.path.join(HERE, "k3_review_stream.py"),
             "--model-dir", CONTEXT["model_dir"], "--mode", "proxy", "--case-file", cases,
             *extra, "--out", os.path.join(work, out)],
            cwd=HERE, capture_output=True, text=True, timeout=900)

    # 1. a folded run refuses to write a baseline -- before reading a weight
    child_as_parent = os.path.join(work, "child_as_parent.safetensors")
    refused = review("--dense-dir", CONTEXT["fold_dir"], "--rung", str(CONTEXT["fold_rung"]),
                     "--emit-baseline", child_as_parent, out="review_emit_refused.json")
    assert refused.returncode != 0, refused.stdout[-1500:]
    assert "refuses to write a FOLDED run as a baseline" in refused.stderr, refused.stderr[-800:]
    assert "loaded" not in refused.stdout, refused.stdout[-800:]
    assert not os.path.exists(child_as_parent), "the child baseline was written anyway"

    # 2. deliberately emit one, then prove the read side refuses it
    allowed = review("--dense-dir", CONTEXT["fold_dir"], "--rung", str(CONTEXT["fold_rung"]),
                     "--emit-baseline", child_as_parent, "--allow-folded-baseline",
                     out="review_emit_allowed.json")
    assert allowed.returncode == 0, allowed.stdout[-2000:] + allowed.stderr[-2000:]
    stamp = k3_common.read_metadata(child_as_parent)
    assert stamp.get("is_parent") == "False", stamp
    assert json.loads(stamp["layers_folded"]) == [1], stamp

    read_refused = review("--baseline", child_as_parent, out="review_bad_baseline.json")
    assert read_refused.returncode != 0, read_refused.stdout[-1500:]
    assert "was itself produced with layers" in read_refused.stderr, read_refused.stderr[-800:]
    assert "loaded" not in read_refused.stdout, read_refused.stdout[-800:]

    # 3. the legitimate comparison records WHICH baseline it used
    good = review("--dense-dir", CONTEXT["fold_dir"], "--rung", str(CONTEXT["fold_rung"]),
                  "--baseline", baseline, out="review_child_provenance.json")
    assert good.returncode == 0, good.stdout[-2000:] + good.stderr[-2000:]
    receipt = json.load(open(os.path.join(work, "review_child_provenance.json"), encoding="utf-8"))
    provenance = receipt["baseline_provenance"]
    assert provenance["baseline_path"] == baseline, provenance
    assert provenance["baseline_sha256"] == k3_common.sha256_file(baseline), provenance
    assert provenance["baseline_layers_folded"] == [], provenance
    assert provenance["baseline_is_stamped"] is True, provenance
    assert provenance["this_run_layers_folded"] == [1], provenance

    # 4. --emit-baseline in decode mode would have written nothing at all
    decode_emit = subprocess.run(
        [sys.executable, os.path.join(HERE, "k3_review_stream.py"),
         "--model-dir", CONTEXT["model_dir"], "--mode", "decode", "--case-file", cases,
         "--dense-dir", CONTEXT["fold_dir"], "--rung", str(CONTEXT["fold_rung"]),
         "--emit-baseline", os.path.join(work, "never.safetensors"),
         "--out", os.path.join(work, "review_decode_emit.json")],
        cwd=HERE, capture_output=True, text=True, timeout=900)
    assert decode_emit.returncode != 0, decode_emit.stdout[-1500:]
    assert "only has anything to write in --mode proxy" in decode_emit.stderr, decode_emit.stderr[-600:]

    return {"folded_run_refused_to_emit_a_baseline": True,
            "folded_baseline_refused_on_read": True,
            "allow_folded_baseline_is_the_only_way_through": True,
            "emitted_baseline_stamp": stamp,
            "consuming_receipt_records": {k: provenance[k] for k in
                                          ("baseline_path", "baseline_sha256",
                                           "baseline_layers_folded", "baseline_is_stamped")},
            "decode_mode_emit_baseline_refused": True}


@test
def t17_incremental_decode_state_matches_a_full_pass() -> dict:
    """Prefill n-1 tokens, step the last one, and land on the full-pass logits.

    This is the check that the KDA conv cache + recurrent state and the MLA
    K/V cache are actually carrying what a full forward computes -- the thing
    that silently breaks first when a decode path is bolted onto a prefill one.
    """
    from k3_review_stream import ReviewModel

    index: K3Index = CONTEXT["index"]
    args = argparse.Namespace(device="cpu", kda_backend="torch", dense_dir=CONTEXT["fold_dir"],
                              dense_layers=None, rung=CONTEXT["fold_rung"])
    model = ReviewModel(index, args)
    model.load()
    ids = torch.randint(0, CONTEXT["vocab"], (1, 10),
                        generator=torch.Generator().manual_seed(43), dtype=torch.long)
    with torch.no_grad():
        full = model.logits(ids, use_state=False)
        model.reset_states()
        model.logits(ids[:, :-1], use_state=True)
        stepped = model.logits(ids[:, -1:], use_state=True)
    return {"last_position": close(stepped[0, -1], full[0, -1], 5e-3, "incremental vs full logits"),
            "top1_matches": bool(int(stepped[0, -1].argmax()) == int(full[0, -1].argmax()))}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default=os.path.join(HERE, "_test_work"))
    ap.add_argument("--src-dir", default="/home/cereal/omni-campaign/k3/k3_src")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=256, help="calibration tokens for the streamed pass")
    ap.add_argument("--block-size", type=int, default=2,
                    help="attn_res_block_size of the PRIMARY fixture; must be > 1 or the "
                         "non-snapshot residual branch is never executed")
    ap.add_argument("--branch-layers", type=int, default=6,
                    help="layers in the second fixture, built at block size 3 for t10b")
    ap.add_argument("--branch-block-size", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma list of test name prefixes")
    args = ap.parse_args()

    work = os.path.abspath(args.work_dir)
    os.makedirs(work, exist_ok=True)
    import k3_synth

    def fixture(name: str, layers: int, block_size: int) -> tuple[str, dict]:
        """Build one synthetic checkpoint, forwarding EVERY shape knob.

        The previous version forwarded only --out-dir/--layers/--hidden/--experts,
        so `--attn-res-block-size` silently kept its default of 1 and every
        fixture layer was a snapshot layer.  Anything the fixture can vary is
        passed explicitly here.
        """
        out_dir = os.path.join(work, name)
        argv = ["--out-dir", out_dir, "--layers", str(layers), "--hidden", str(args.hidden),
                "--experts", str(args.experts), "--attn-res-block-size", str(block_size)]
        return out_dir, k3_synth.build(k3_synth.build_parser().parse_args(argv))

    if args.block_size < 2:
        raise SystemExit("--block-size 1 makes every layer a snapshot layer; use 2 or more")
    model_dir, synth = fixture("synthetic-k3", args.layers, args.block_size)
    branch_dir, branch_synth = fixture("synthetic-k3-blocks", args.branch_layers, args.branch_block_size)

    CONTEXT.update({
        "work": work, "model_dir": model_dir, "index": K3Index(model_dir),
        "layers": args.layers, "experts": args.experts, "tokens": args.tokens, "synth": synth,
        "block_model_dir": branch_dir, "block_index": K3Index(branch_dir),
        "block_synth": branch_synth,
    })
    CONTEXT["vocab"] = CONTEXT["index"].text_config["vocab_size"]
    try:
        from k3_receipt import TRANSFORMERS_COMPAT, import_k3_modeling

        modeling, config_module = import_k3_modeling(args.src_dir)
        CONTEXT["modeling"], CONTEXT["config_module"] = modeling, config_module
        CONTEXT["transformers_compat"] = dict(TRANSFORMERS_COMPAT)
    except Exception as exc:
        CONTEXT["modeling"] = None
        CONTEXT["modeling_error"] = f"modeling_kimi_linear.py not importable here: {exc!r}"
        print(f"[warn] {CONTEXT['modeling_error']}")

    results = []
    failures = 0
    for fn in TESTS:
        if args.only and not any(fn.__name__.startswith(p) for p in args.only.split(",")):
            continue
        started = time.time()
        try:
            detail = fn()
            status = "PASS"
        except Skip as exc:
            detail, status = {"reason": str(exc)}, "SKIP"
        except Exception:
            detail, status = {"traceback": traceback.format_exc()[-2500:]}, "FAIL"
            failures += 1
        entry = {"test": fn.__name__, "status": status, "seconds": round(time.time() - started, 2),
                 "detail": detail}
        results.append(entry)
        print(f"[{status}] {fn.__name__} ({entry['seconds']}s)")
        if status != "PASS":
            print(json.dumps(detail, indent=1, default=str)[:2500])

    receipt = {
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "devices": k3_common.device_inventory(),
        "synthetic": {k: v for k, v in synth.items() if k != "shard_sha256"},
        "synthetic_attn_res_block_size": args.block_size,
        "synthetic_branch_fixture": {
            **{k: v for k, v in branch_synth.items() if k != "shard_sha256"},
            "layers": args.branch_layers,
            "attn_res_block_size": args.branch_block_size,
        },
        "peak_rss_gigabytes": (lambda b: round(b / 1e9, 3) if b else None)(k3_common.peak_rss_bytes()),
        "modeling_import": CONTEXT.get("modeling_error", "ok"),
        "transformers_compat": CONTEXT.get("transformers_compat", {}),
        "transformers": __import__("transformers").__version__,
        "results": results,
        "passed": sum(1 for r in results if r["status"] == "PASS"),
        "skipped": sum(1 for r in results if r["status"] == "SKIP"),
        "failed": failures,
    }
    out = args.out or os.path.join(work, "test_receipt.json")
    receipt["receipt_sha256"] = write_receipt(out, receipt)
    print(f"\n{receipt['passed']} passed, {receipt['skipped']} skipped, {failures} failed -> {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
