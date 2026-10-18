#!/usr/bin/env python3
"""Build a tiny, REAL K3-shaped checkpoint so the harness can be tested.

Not a mock: this writes actual safetensors shards, an actual
`model.safetensors.index.json` and an actual nested `config.json` with the same
key names, the same one-based `kda_layers` rule, the same MXFP4 packing and the
same `attn_res_block_size` plumbing as `/mnt/ai_data/models/moonshotai/Kimi-K3`.
Every tool in this lane -- the streamed pass, the exactness receipt against the
repo's own modules, the C14 coreset, the review runner -- runs against it
unmodified.

It exists because the real layer 1 costs 53 minutes of CIFS read (measured:
16.99 GB at 5.3 MB/s) and starting a second 30 GB shard read while another
process is on the share is the fastest way to make both of them useless.

Defaults are the task's cell: 2 layers, hidden 64, 8 experts, attn-res block
size 2.  The layout still exercises every branch --

    layer 0: KDA attention + dense KimiMLP   (first_k_dense_replace = 1)
                                             0 % 2 == 0 -> SNAPSHOT layer
    layer 1: gated MLA + latent MoE          (the final layer is forced full-attn,
                                             exactly as K3 forces layer 93)
                                             1 % 2 == 1 -> NON-snapshot layer

-- because the group sizes are chosen so MXFP4's 32-column groups divide, which
is the constraint that decides `routed_expert_hidden_size` and
`moe_intermediate_size` here.

`--attn-res-block-size` defaults to 2, NOT 1, and that is load-bearing.  At 1
every layer satisfies ``layer_idx % block_size == 0``, so `StreamedLayer.forward`
always takes the snapshot branch (`carried = None -> carried = attn_out`) and
the line `carried = prefix_sum + attn_out` never runs.  On the real checkpoint
`attn_res_block_size = 12` over 93 layers, so 85 of 93 layers take exactly the
branch a block-size-1 fixture cannot reach.  `test_k3_harness.py` additionally
builds a 6-layer, block-size-3 fixture and runs the whole-layer comparison on
BOTH a snapshot layer and a non-snapshot one.

Every RMSNorm gain is a DISTINCT random vector (`norm_gain`), never
`torch.ones`.  With unit gains the fixture cannot tell the six per-layer norms
apart and a mis-wired name is invisible to every comparison in the suite --
see `norm_gain`'s own note and `t05b`.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from k3_common import log, save_tensors


def build_config(args) -> dict:
    layers = args.layers
    # K3's rule: full attention on every fourth layer (one-based) and on the
    # last one; everything else is KDA.
    full = sorted({i for i in range(1, layers + 1) if i % 4 == 0} | {layers})
    kda = [i for i in range(1, layers + 1) if i not in full]
    text = {
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden,
        "intermediate_size": args.hidden + 32,
        "num_hidden_layers": layers,
        "num_attention_heads": args.heads,
        "num_key_value_heads": args.heads,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1024,
        "attn_res_block_size": args.attn_res_block_size,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts": args.experts,
        "num_experts_per_token": args.top_k,
        "num_shared_experts": 2,
        "moe_intermediate_size": args.moe_intermediate,
        "routed_expert_hidden_size": args.latent,
        "latent_moe_use_norm": True,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "num_expert_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "q_lora_rank": args.q_lora,
        "kv_lora_rank": args.kv_lora,
        "qk_nope_head_dim": args.nope,
        "qk_rope_head_dim": args.rope,
        "v_head_dim": args.v_dim,
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "tie_word_embeddings": False,
        "linear_attn_config": {
            "kda_layers": kda,
            "full_attn_layers": full,
            "num_heads": args.kda_heads,
            "head_dim": args.kda_head_dim,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    }
    return {
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": text,
        "quantization_config": {
            "format": "mxfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {
                "group_size": 32, "num_bits": 4, "type": "float", "symmetric": True,
            }}},
            "ignore": ["re:.*self_attn.*", "re:.*shared_experts.*",
                       "re:.*mlp\\.(gate|up|gate_up|down)_proj.*", "lm_head"],
        },
        "synthetic": True,
    }


def randn(*shape, generator, scale=0.02, dtype=torch.bfloat16) -> torch.Tensor:
    return (torch.randn(*shape, generator=generator) * scale).to(dtype)


def norm_gain(width: int, generator, dtype=torch.bfloat16) -> torch.Tensor:
    """A DISTINCT random RMSNorm gain -- never `torch.ones`, and that is load-bearing.

    Every gain in this fixture used to be a unit vector, which made the whole
    suite structurally blind to WHICH norm the harness reads.  K3 has six
    distinct gains per layer (`input_layernorm`, `post_attention_layernorm`,
    `self_attention_res_norm`, `mlp_res_norm`, `o_norm` / the two MLA latent
    norms, `routed_expert_norm`) and `StreamedLayer.forward` wires four of them
    by name; with all-ones gains you can swap `self_attention_res_norm` with
    `mlp_res_norm` -- or point `moe_forward` at the wrong norm -- and t05, t08,
    t09, t10, t10b, t12, t12b and t17 all still report bit-exact block stacks
    and fp32 relL2 ~1e-7, because the repo module is loaded from the same
    all-ones state dict.  On the real checkpoint the gains differ and every
    downstream number (prefix_out, y_lat, the C14 target) would be silently
    wrong.  Same class of bug as the eps = 1e-6 one, and the fixture could not
    see it.

    Centred on 1.0 so the fixture's activation scale is unchanged; the spread
    is what makes a mis-wired name move the output (`t05b` mutates the state
    dict and checks that it does).
    """
    return (torch.randn(width, generator=generator) * 0.1 + 1.0).to(dtype)


def pack_expert(rows: int, cols: int, generator) -> dict[str, torch.Tensor]:
    """Random MXFP4-packed weight: codes are uniform, scales a narrow exponent band.

    The dequantized value IS the ground truth here -- nothing is quantized from
    a bf16 original -- so a dequant bug shows up as an arithmetic mismatch, not
    as rounding noise that could be argued away.
    """
    if cols % 32:
        raise SystemExit(f"MXFP4 groups are 32 wide; {cols} input columns do not divide")
    packed = torch.randint(0, 256, (rows, cols // 2), generator=generator, dtype=torch.uint8)
    scale = torch.randint(119, 124, (rows, cols // 32), generator=generator, dtype=torch.uint8)
    return {"weight_packed": packed, "weight_scale": scale}


def build(args) -> dict:
    cfg = build_config(args)
    text = cfg["text_config"]
    generator = torch.Generator().manual_seed(args.seed)
    hidden = text["hidden_size"]
    latent = text["routed_expert_hidden_size"]
    moe_i = text["moe_intermediate_size"]
    shared_i = moe_i * text["num_shared_experts"]
    kda_heads = text["linear_attn_config"]["num_heads"]
    kda_dim = text["linear_attn_config"]["head_dim"]
    kda_proj = kda_heads * kda_dim
    conv = text["linear_attn_config"]["short_conv_kernel_size"]
    q_head_dim = text["qk_nope_head_dim"] + text["qk_rope_head_dim"]
    heads = text["num_attention_heads"]

    os.makedirs(args.out_dir, exist_ok=True)
    shards: dict[str, dict[str, torch.Tensor]] = {}
    weight_map: dict[str, str] = {}

    def put(shard: str, name: str, tensor: torch.Tensor) -> None:
        shards.setdefault(shard, {})[name] = tensor
        weight_map[name] = shard

    n_shards = text["num_hidden_layers"] + 1
    head_shard = f"model-00001-of-{n_shards:05d}.safetensors"

    put(head_shard, "language_model.model.embed_tokens.weight",
        randn(text["vocab_size"], hidden, generator=generator))
    put(head_shard, "language_model.model.norm.weight", norm_gain(hidden, generator))
    put(head_shard, "language_model.model.output_attn_res_norm.weight",
        norm_gain(hidden, generator))
    put(head_shard, "language_model.model.output_attn_res_proj.weight",
        randn(1, hidden, generator=generator, scale=0.1))
    put(head_shard, "language_model.lm_head.weight",
        randn(text["vocab_size"], hidden, generator=generator))

    for layer in range(text["num_hidden_layers"]):
        shard = f"model-{layer + 2:05d}-of-{n_shards:05d}.safetensors"
        prefix = f"language_model.model.layers.{layer}."
        is_kda = (layer + 1) in text["linear_attn_config"]["kda_layers"]
        is_moe = layer >= text["first_k_dense_replace"]

        for name in ("input_layernorm", "post_attention_layernorm",
                     "self_attention_res_norm", "mlp_res_norm"):
            put(shard, f"{prefix}{name}.weight", norm_gain(hidden, generator))
        for name in ("self_attention_res_proj", "mlp_res_proj"):
            put(shard, f"{prefix}{name}.weight", randn(1, hidden, generator=generator, scale=0.1))

        if is_kda:
            for name in ("q_proj", "k_proj"):
                put(shard, f"{prefix}self_attn.{name}.weight", randn(kda_proj, hidden, generator=generator))
            put(shard, f"{prefix}self_attn.v_proj.weight", randn(kda_proj, hidden, generator=generator))
            put(shard, f"{prefix}self_attn.g_proj.weight", randn(kda_proj, hidden, generator=generator))
            put(shard, f"{prefix}self_attn.o_proj.weight", randn(hidden, kda_proj, generator=generator))
            put(shard, f"{prefix}self_attn.f_a_proj.weight", randn(kda_dim, hidden, generator=generator))
            put(shard, f"{prefix}self_attn.f_b_proj.weight", randn(kda_proj, kda_dim, generator=generator))
            put(shard, f"{prefix}self_attn.b_proj.weight", randn(kda_heads, hidden, generator=generator))
            # A_log indexes HEADS -- fla does A_log.view(H, 1).  dt_bias is H*K.
            # A_log's axis is a fixture choice because the SHIPPED Kimi-K3
            # disagrees with its own modeling code: the checkpoint carries
            # f32[head_dim] while `KimiDeltaAttention` declares [num_heads]
            # (k3_layer1_shapes.json).  `--a-log-axis key_dim` reproduces the
            # real checkpoint so the harness's precomputed-gate path is exercised.
            a_log_width = kda_heads if args.a_log_axis == "heads" else kda_dim
            put(shard, f"{prefix}self_attn.A_log",
                torch.log(torch.empty(a_log_width).uniform_(1, 16, generator=generator)).to(torch.float32))
            put(shard, f"{prefix}self_attn.dt_bias",
                (torch.randn(kda_proj, generator=generator) * 0.1).to(torch.float32))
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                put(shard, f"{prefix}self_attn.{name}.weight",
                    randn(kda_proj, 1, conv, generator=generator, scale=0.3))
            put(shard, f"{prefix}self_attn.o_norm.weight", norm_gain(kda_dim, generator))
        else:
            put(shard, f"{prefix}self_attn.q_a_proj.weight",
                randn(text["q_lora_rank"], hidden, generator=generator))
            put(shard, f"{prefix}self_attn.q_a_layernorm.weight",
                norm_gain(text["q_lora_rank"], generator))
            put(shard, f"{prefix}self_attn.q_b_proj.weight",
                randn(heads * q_head_dim, text["q_lora_rank"], generator=generator))
            put(shard, f"{prefix}self_attn.kv_a_proj_with_mqa.weight",
                randn(text["kv_lora_rank"] + text["qk_rope_head_dim"], hidden, generator=generator))
            put(shard, f"{prefix}self_attn.kv_a_layernorm.weight",
                norm_gain(text["kv_lora_rank"], generator))
            put(shard, f"{prefix}self_attn.kv_b_proj.weight",
                randn(heads * (text["qk_nope_head_dim"] + text["v_head_dim"]),
                      text["kv_lora_rank"], generator=generator))
            put(shard, f"{prefix}self_attn.o_proj.weight",
                randn(hidden, heads * text["v_head_dim"], generator=generator))
            put(shard, f"{prefix}self_attn.g_proj.weight",
                randn(heads * text["v_head_dim"], hidden, generator=generator))

        if is_moe:
            moe = f"{prefix}block_sparse_moe."
            put(shard, f"{moe}gate.weight", randn(text["num_experts"], hidden, generator=generator, scale=0.1))
            put(shard, f"{moe}gate.e_score_correction_bias",
                (torch.randn(text["num_experts"], generator=generator) * 0.05).to(torch.float32))
            put(shard, f"{moe}routed_expert_down_proj.weight", randn(latent, hidden, generator=generator))
            put(shard, f"{moe}routed_expert_up_proj.weight", randn(hidden, latent, generator=generator))
            put(shard, f"{moe}routed_expert_norm.weight", norm_gain(latent, generator))
            put(shard, f"{moe}shared_experts.gate_proj.weight", randn(shared_i, hidden, generator=generator))
            put(shard, f"{moe}shared_experts.up_proj.weight", randn(shared_i, hidden, generator=generator))
            put(shard, f"{moe}shared_experts.down_proj.weight", randn(hidden, shared_i, generator=generator))
            for expert in range(text["num_experts"]):
                for name, (rows, cols) in (("w1", (moe_i, latent)), ("w3", (moe_i, latent)),
                                           ("w2", (latent, moe_i))):
                    packed = pack_expert(rows, cols, generator)
                    for suffix, tensor in packed.items():
                        put(shard, f"{moe}experts.{expert}.{name}.{suffix}", tensor)
        else:
            put(shard, f"{prefix}mlp.gate_proj.weight",
                randn(text["intermediate_size"], hidden, generator=generator))
            put(shard, f"{prefix}mlp.up_proj.weight",
                randn(text["intermediate_size"], hidden, generator=generator))
            put(shard, f"{prefix}mlp.down_proj.weight",
                randn(hidden, text["intermediate_size"], generator=generator))

    digests = {}
    for shard, tensors in shards.items():
        digests[shard] = save_tensors(os.path.join(args.out_dir, shard), tensors)
    total = sum(t.numel() * t.element_size() for tensors in shards.values() for t in tensors.values())
    with open(os.path.join(args.out_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as handle:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, handle, indent=1)
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=1)
    log(f"synthetic K3 at {args.out_dir}: {len(weight_map)} tensors, {total/1e6:.2f} MB, "
        f"{len(shards)} shards", tag="synth")
    return {"out_dir": args.out_dir, "tensors": len(weight_map), "bytes": total, "shard_sha256": digests}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--vocab-size", type=int, default=128)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--moe-intermediate", type=int, default=64, help="multiple of 32 (MXFP4 group)")
    ap.add_argument("--latent", type=int, default=32, help="routed_expert_hidden_size, multiple of 32")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--kda-heads", type=int, default=4)
    ap.add_argument("--kda-head-dim", type=int, default=24)
    ap.add_argument("--q-lora", type=int, default=24)
    ap.add_argument("--kv-lora", type=int, default=32)
    ap.add_argument("--nope", type=int, default=16)
    ap.add_argument("--rope", type=int, default=8)
    ap.add_argument("--v-dim", type=int, default=16)
    ap.add_argument("--attn-res-block-size", type=int, default=2,
                    help="MUST be > 1 or every layer is a snapshot layer and the "
                         "`carried = prefix_sum + attn_out` branch -- the one 85 of K3's "
                         "93 real layers take -- is never executed (real K3: 12)")
    ap.add_argument("--a-log-axis", default="heads", choices=("heads", "key_dim"),
                    help="'key_dim' reproduces the shipped Kimi-K3, whose A_log is f32[head_dim]")
    ap.add_argument("--seed", type=int, default=2026090101)
    return ap


if __name__ == "__main__":
    build(build_parser().parse_args())
