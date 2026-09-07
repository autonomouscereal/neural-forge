#!/usr/bin/env python3
"""K3-1(b): the exactness receipt for ONE layer against the repo's own modules.

The harness in `k3_layer.py` is a transcription.  A transcription is worth
nothing until it is measured against the thing it transcribes, so this script
builds the real `KimiDecoderLayer` (or its halves) out of
`modeling_kimi_linear.py`, feeds both it and the harness the same seeded input,
and writes down every difference.

Five checks, each recorded whether it passes or not:

  1. MXFP4  -- our `dequant_mxfp4` vs compressed_tensors' own
     `unpack_fp4_from_uint8` / `decompress_mx_scale` on expert 0, bit-exact or
     not, both sha256s.
  2. KDA gate axis -- `A_log` must have `num_heads` elements for fla's
     `A_log.view(H, 1)`; the real on-disk shape is read and printed.  R6's
     first pass reported f32[128] from a header read while the modeling code
     declares [num_heads=96]; this is the check that settles it.
  3. KDA backends -- our pure-torch recurrence vs fla's `chunk_kda` (GPU), and
     both orders of the gated output norm, so `NORM_BEFORE_GATE` is a measured
     constant rather than a guess.
  4. Halves -- the real `KimiDeltaAttention`/`KimiMLAAttention` and the real
     `KimiSparseMoeBlock` vs ours, including top-k router set agreement.
  5. Whole layer -- the real `_forward_attn_residual` vs `StreamedLayer.forward`
     on the same (prefix_sum, block_residual) pair.

PROPOSED DECISION LINE for Cyril (D-a2, not a gate this script enforces):
prefix_out relL2 <= 2e-2 and top-16 router agreement >= 99%.  The script
reports `meets_proposed_line` and exits 0 either way; a real mismatch is
information, not something to hide behind an exit code.

Building the reference MoE materialises all experts at bf16 (59.2 GB for a
real K3 layer) -- that is the price of the reference and it is opt-in via
--reference-moe.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import torch

from k3_common import (
    K3Index,
    blocks_before,
    canonical_device,
    device_inventory,
    dequant_mxfp4,
    log,
    mxfp4_reference_check,
    sha256_tensor,
    write_receipt,
    write_scores,
)
import k3_layer
from k3_layer import LayerWeights, StreamedLayer, dequant_expert, load_layer, moe_forward, route

PROPOSED_PREFIX_RELL2 = 2e-2
PROPOSED_TOPK_AGREEMENT = 0.99


# --------------------------------------------------------------------------
TRANSFORMERS_COMPAT: dict[str, str] = {}


def _transformers_compat() -> dict[str, str]:
    """Re-export symbols transformers 5.12 moved, so the repo file imports.

    `modeling_kimi_linear.py` was written against transformers 4.56-4.57 and
    does `from transformers.utils.generic import OutputRecorder`; 5.12.1 moved
    that class to `transformers.utils.output_capturing`.  This binds the SAME
    class back under its old name -- it is a re-export, not a stub, and every
    substitution is recorded in the receipt.  Everything else the file imports
    (create_causal_mask, check_model_inputs, ALL_LAYERNORM_LAYERS,
    ALL_ATTENTION_FUNCTIONS, Unpack, TransformersKwargs) is still where it was.
    """
    notes: dict[str, str] = {}
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder
        notes["OutputRecorder"] = "re-exported from transformers.utils.output_capturing (moved in 5.12)"
    return notes


def import_k3_modeling(src_dir: str, model_dir: str | None = None):
    """Import the repo's modeling file as a package (it uses relative imports).

    `modeling_kimi_linear.py` does `from .configuration_kimi_k3 import ...`, so
    the directory has to be a package.  We only ever write the `__init__.py`
    into OUR staging copy -- never into the share, and now that is ENFORCED
    rather than promised: passing `--src-dir` equal to (or inside) the model
    directory is refused, because the docstring's guarantee was previously
    nothing but a docstring.
    """
    TRANSFORMERS_COMPAT.update(_transformers_compat())
    src_dir = os.path.abspath(src_dir)
    if model_dir:
        model_dir = os.path.abspath(model_dir)
        try:  # different drives on Windows make commonpath raise; that is "not inside"
            inside = os.path.commonpath([src_dir, model_dir]) == model_dir
        except ValueError:
            inside = False
        if inside:
            raise SystemExit(
                f"--src-dir {src_dir} is inside the model directory {model_dir}; this function "
                "writes an __init__.py and must never write to the checkpoint (least of all a "
                "CIFS share).  Copy the modeling files to a local staging directory first."
            )
    init = os.path.join(src_dir, "__init__.py")
    if not os.path.exists(init):
        with open(init, "w", encoding="utf-8") as handle:
            handle.write("# created by k3_receipt.py so the repo's relative imports resolve\n")
    parent, package = os.path.split(src_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    config_module = importlib.import_module(f"{package}.configuration_kimi_k3")
    modeling = importlib.import_module(f"{package}.modeling_kimi_linear")
    return modeling, config_module


def build_config(text_config: dict, config_module, attn_implementation: str):
    cfg = config_module.KimiLinearConfig(**{k: v for k, v in text_config.items() if k != "torch_dtype"})
    cfg._attn_implementation = attn_implementation
    return cfg


def module_state_dict(weights: LayerWeights, dequantize_experts: bool,
                      expert_dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """Layer tensors renamed to the repo module's own parameter names."""
    state: dict[str, torch.Tensor] = {}
    for short, tensor in weights.attn.items():
        state[f"self_attn.{short}"] = tensor
    state.update(weights.res)
    for short, tensor in weights.body.items():
        state[short] = tensor
    if dequantize_experts:
        for expert in sorted(weights.experts):
            dequantized = dequant_expert(weights, expert, dtype=expert_dtype)
            for name in ("w1", "w2", "w3"):
                state[f"block_sparse_moe.experts.{expert}.{name}.weight"] = dequantized[name]
    return state


# --------------------------------------------------------------------------
def causal_mask_for(weights: LayerWeights, tokens: int, dtype: torch.dtype,
                    device: torch.device) -> torch.Tensor | None:
    """The additive causal mask the repo's eager/sdpa path needs -- or None.

    KDA takes a 2-D 0/1 padding mask (or None) and is causal by construction;
    handing it a 4-D additive mask raises.  MLA with `attention_mask=None`
    silently runs NON-CAUSAL, which is how the first version of this receipt
    reported relL2 3.24 on a transcription that is actually exact.
    """
    if weights.is_kda:
        return None
    return torch.full((1, 1, tokens, tokens), float("-inf"), dtype=dtype, device=device).triu(1)


def check_mxfp4(weights: LayerWeights) -> dict:
    if not weights.experts:
        return {"skipped": "layer has no packed experts"}
    expert = weights.experts[sorted(weights.experts)[0]]
    fragment = mxfp4_reference_check(expert["w1.weight_packed"], expert["w1.weight_scale"])
    dequantized = dequant_mxfp4(expert["w1.weight_packed"], expert["w1.weight_scale"])
    fragment["expert0_w1"] = {
        "shape": list(dequantized.shape),
        "std": round(float(dequantized.float().std()), 6),
        "absmax": round(float(dequantized.float().abs().max()), 6),
        "zero_fraction": round(float((dequantized == 0).float().mean()), 5),
        "sha256": sha256_tensor(dequantized),
    }
    return fragment


def check_kda_gate_axis(weights: LayerWeights, cfg: dict) -> dict:
    if not weights.is_kda:
        return {"skipped": "not a KDA layer"}
    linear = cfg["linear_attn_config"]
    a_log = weights.attn["A_log"]
    dt_bias = weights.attn.get("dt_bias")
    heads = int(linear["num_heads"])
    head_dim = int(linear["head_dim"])
    axis = k3_layer.kda_gate_axis(a_log, heads, head_dim)
    verdict = {
        "heads": ("A_log indexes HEADS -- fla's own form (A_log.view(H, 1)); the kernel's fused "
                  "gate can be used and Chimera's decay_log_scale[heads] maps 1:1"),
        "key_dim": ("A_log indexes the KEY DIM, not heads.  The repo's KimiDeltaAttention declares "
                    "torch.empty(num_heads) and fla's kernel hardcodes A_log.view(H, 1), so BOTH "
                    "refuse this tensor: strict load_state_dict fails and use_gate_in_kernel=True "
                    "is unusable.  The harness precomputes the decay with A_log.view(1, K) and "
                    "passes use_gate_in_kernel=False.  WHICH BROADCAST IS SEMANTICALLY CORRECT IS "
                    "NOT SETTLED HERE -- decision line D-a2b, needs a behavioural receipt."),
    }[axis]
    return {
        "A_log_shape": list(a_log.shape),
        "A_log_dtype": str(a_log.dtype),
        "A_log_numel": int(a_log.numel()),
        "num_heads": heads,
        "head_dim": head_dim,
        "dt_bias_numel": int(dt_bias.numel()) if dt_bias is not None else None,
        "b_proj_shape": list(weights.attn["b_proj.weight"].shape) if "b_proj.weight" in weights.attn else None,
        "o_norm_numel": int(weights.attn["o_norm.weight"].numel()) if "o_norm.weight" in weights.attn else None,
        "resolved_axis": axis,
        "axis_is_num_heads": axis == "heads",
        "verdict": verdict,
    }


def check_o_norm_order(weights: LayerWeights, cfg: dict, device: torch.device) -> dict:
    """Settle `NORM_BEFORE_GATE` against fla's own module, in isolation.

    Comparing it through a whole KDA forward buries the answer under the conv
    and recurrence kernels' bf16 noise (both orders scored ~6.5e-3 there).
    Calling `FusedRMSNormGated` directly at fp32 separates them by orders of
    magnitude, which is the difference between a measured constant and a guess.
    """
    if device.type != "cuda":
        return {"skipped": "FusedRMSNormGated is a triton kernel; CPU has no backend"}
    try:
        from fla.modules import FusedRMSNormGated
    except Exception as exc:
        return {"error": repr(exc)[:200]}
    linear = cfg["linear_attn_config"]
    head_dim = int(linear["head_dim"])
    eps = float(cfg["rms_norm_eps"])
    weight = weights.attn["o_norm.weight"].to(device)
    out: dict = {}
    for dtype, tag in ((torch.float32, "fp32"), (torch.bfloat16, "bf16")):
        try:
            module = FusedRMSNormGated(head_dim, eps=eps, activation="sigmoid").to(device=device, dtype=dtype)
            module.weight.data = weight.to(dtype).clone()
            generator = torch.Generator(device="cpu").manual_seed(101)
            x = torch.randn(1, 32, 2, head_dim, generator=generator).to(device=device, dtype=dtype)
            gate = torch.randn(1, 32, 2, head_dim, generator=generator).to(device=device, dtype=dtype)
            with torch.no_grad():
                reference = module(x, gate)
            before = k3_layer.rms_norm_gated(x, gate, module.weight, eps, norm_before_gate=True)
            after = k3_layer.rms_norm_gated(x, gate, module.weight, eps, norm_before_gate=False)
            out[tag] = {
                "norm_before_gate_relL2": write_scores(before, reference)["relL2"],
                "gate_before_norm_relL2": write_scores(after, reference)["relL2"],
            }
            out[tag]["matched"] = ("norm_before_gate"
                                   if out[tag]["norm_before_gate_relL2"] <= out[tag]["gate_before_norm_relL2"]
                                   else "gate_before_norm")
        except Exception as exc:
            out[tag] = {"error": repr(exc)[:250]}
    out["harness_constant_NORM_BEFORE_GATE"] = k3_layer.NORM_BEFORE_GATE
    return out


def check_kda_backends(weights: LayerWeights, cfg: dict, hidden: torch.Tensor) -> dict:
    """torch recurrence vs fla chunk_kda, and both gated-norm orders."""
    if not weights.is_kda:
        return {"skipped": "not a KDA layer"}
    result: dict = {}
    torch_out, _ = k3_layer.kda_forward(weights, hidden, cfg, backend="torch")
    result["torch_sha256"] = sha256_tensor(torch_out)
    result["o_norm_order"] = check_o_norm_order(weights, cfg, hidden.device)
    if hidden.is_cuda:
        try:
            fla_out, _ = k3_layer.kda_forward(weights, hidden, cfg, backend="fla")
            result["fla_vs_torch"] = write_scores(torch_out.float(), fla_out.float())
            result["fla_vs_torch"]["max_abs"] = float((torch_out.float() - fla_out.float()).abs().max())
            saved = k3_layer.NORM_BEFORE_GATE
            try:
                k3_layer.NORM_BEFORE_GATE = not saved
                flipped, _ = k3_layer.kda_forward(weights, hidden, cfg, backend="torch")
            finally:
                k3_layer.NORM_BEFORE_GATE = saved
            result["norm_before_gate_relL2"] = write_scores(torch_out.float(), fla_out.float())["relL2"]
            result["gate_before_norm_relL2"] = write_scores(flipped.float(), fla_out.float())["relL2"]
            result["order_that_matched_fla"] = (
                "norm_before_gate" if result["norm_before_gate_relL2"] <= result["gate_before_norm_relL2"]
                else "gate_before_norm"
            )
            result["harness_constant_NORM_BEFORE_GATE"] = saved
        except Exception as exc:
            result["fla_error"] = repr(exc)[:400]
    else:
        result["fla_skipped"] = "chunk_kda is a triton kernel; CPU has no backend"
    return result


def check_halves(weights: LayerWeights, cfg: dict, hidden: torch.Tensor,
                 modeling, config_module, args) -> dict:
    """The repo's own attention/MoE modules vs ours on the same input."""
    out: dict = {}
    hf_config = build_config(cfg, config_module, args.attn_implementation)
    device = hidden.device

    # -- attention ---------------------------------------------------------
    try:
        with torch.device("meta"):
            if weights.is_kda:
                module = modeling.KimiDeltaAttention(config=hf_config, layer_idx=weights.layer_idx)
            else:
                module = modeling.KimiMLAAttention(config=hf_config, layer_idx=weights.layer_idx)
        attn_state = {k[len("self_attn."):]: v for k, v in module_state_dict(weights, False).items()
                      if k.startswith("self_attn.")}
        missing, unexpected = module.load_state_dict(attn_state, strict=False, assign=True)
        module = module.to(device).eval()
        mask = causal_mask_for(weights, hidden.shape[1], hidden.dtype, device)
        with torch.no_grad():
            reference = module(hidden) if mask is None else module(hidden, attention_mask=mask)
        if weights.is_kda:
            ours, _ = k3_layer.kda_forward(weights, hidden, cfg, backend=args.kda_backend)
        else:
            ours, _ = k3_layer.mla_forward(weights, hidden, cfg)
        out["attention"] = write_scores(ours.float(), reference.float())
        out["attention"]["max_abs"] = float((ours.float() - reference.float()).abs().max())
        out["attention"]["missing_keys"] = list(missing)[:8]
        out["attention"]["unexpected_keys"] = list(unexpected)[:8]
        out["attention"]["reference_sha256"] = sha256_tensor(reference)
        out["attention"]["harness_sha256"] = sha256_tensor(ours)
    except Exception as exc:
        out["attention_error"] = repr(exc)[:500]

    # -- MoE ---------------------------------------------------------------
    if weights.is_moe and args.reference_moe:
        try:
            with torch.device("meta"):
                module = modeling.KimiSparseMoeBlock(hf_config)
            state = {k: v for k, v in module_state_dict(weights, True).items()
                     if k.startswith("block_sparse_moe.")}
            state = {k[len("block_sparse_moe."):]: v for k, v in state.items()}
            missing, unexpected = module.load_state_dict(state, strict=False, assign=True)
            module = module.to(device).eval()
            started = time.time()
            with torch.no_grad():
                reference = module(hidden)
            reference_seconds = time.time() - started
            started = time.time()
            ours, taps = moe_forward(weights, hidden, cfg, collect=True)
            out["moe"] = write_scores(ours.float(), reference.float())
            out["moe"]["max_abs"] = float((ours.float() - reference.float()).abs().max())
            out["moe"]["reference_seconds"] = round(reference_seconds, 2)
            out["moe"]["harness_seconds"] = round(time.time() - started, 2)
            out["moe"]["missing_keys"] = list(missing)[:8]
            out["moe"]["unexpected_keys"] = list(unexpected)[:8]
            # router set agreement, token by token
            with torch.no_grad():
                ref_idx, ref_w = module.gate(hidden)
            our_idx, our_w, _ = route(hidden, weights, cfg)
            agree = torch.stack([
                torch.isin(our_idx[t], ref_idx[t]).float().mean() for t in range(our_idx.shape[0])
            ])
            out["moe"]["topk_set_agreement"] = float(agree.mean())
            out["moe"]["topk_weight_relL2"] = float(
                (our_w.sort(dim=-1).values - ref_w.sort(dim=-1).values).norm() / max(float(ref_w.norm()), 1e-30)
            )
        except Exception as exc:
            out["moe_error"] = repr(exc)[:500]
    elif weights.is_moe:
        out["moe_skipped"] = "pass --reference-moe to materialise every expert at bf16 (59.2 GB on a real K3 layer)"
    return out


def check_whole_layer(weights: LayerWeights, cfg: dict, hidden: torch.Tensor,
                      block_residual: torch.Tensor, modeling, config_module, args) -> dict:
    if weights.is_moe and not args.reference_moe:
        return {"skipped": "the reference layer needs every expert at bf16; pass --reference-moe"}
    hf_config = build_config(cfg, config_module, args.attn_implementation)
    try:
        with torch.device("meta"):
            layer = modeling.KimiDecoderLayer(hf_config, weights.layer_idx)
        state = module_state_dict(weights, weights.is_moe)
        missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
        layer = layer.to(hidden.device).eval()
        mask = causal_mask_for(weights, hidden.shape[1], hidden.dtype, hidden.device)
        started = time.time()
        with torch.no_grad():
            ref_prefix, ref_blocks = layer(hidden, attention_mask=mask,
                                           block_residual=block_residual.clone())
        arithmetic_note = (
            "at bf16 the harness and the reference differ by the reference's own bf16-vs-fp32 "
            "floor (measured 6.3e-3 relL2 on the synthetic checkpoint, dominated by SDPA vs the "
            "repo's eager softmax); the algebra itself is exact -- fp32 gives relL2 1.6e-7. The "
            "proposed D-a2 line of 2e-2 was set knowing that floor."
        )
        reference_seconds = time.time() - started
        runner = StreamedLayer(weights, cfg, kda_backend=args.kda_backend)
        started = time.time()
        our_prefix, our_blocks, taps = runner.forward(hidden, block_residual.clone(), collect=True)
        scores = write_scores(our_prefix.float(), ref_prefix.float())
        scores["max_abs"] = float((our_prefix.float() - ref_prefix.float()).abs().max())
        return {
            "prefix_out": scores,
            "arithmetic_note": arithmetic_note,
            "reference_seconds": round(reference_seconds, 2),
            "harness_seconds": round(time.time() - started, 2),
            "block_stack_matches": bool(our_blocks.shape == ref_blocks.shape),
            "block_stack_relL2": float(
                (our_blocks.float() - ref_blocks.float()).norm() / max(float(ref_blocks.float().norm()), 1e-30)
            ) if our_blocks.shape == ref_blocks.shape else None,
            "reference_prefix_sha256": sha256_tensor(ref_prefix),
            "harness_prefix_sha256": sha256_tensor(our_prefix),
            "missing_keys": list(missing)[:8],
            "unexpected_keys": list(unexpected)[:8],
        }
    except Exception as exc:
        return {"error": repr(exc)[:800]}


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--src-dir", default="/home/cereal/omni-campaign/k3/k3_src")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--experts-device", default=None)
    ap.add_argument("--kda-backend", default="torch", choices=("torch", "fla"))
    ap.add_argument("--attn-implementation", default="eager", choices=("eager", "sdpa"))
    ap.add_argument("--reference-moe", action="store_true",
                    help="materialise every expert at bf16 for the reference MoE/layer")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = canonical_device(args.device)
    index = K3Index(args.model_dir)
    cfg = index.text_config
    receipt: dict = {
        "model_dir": args.model_dir,
        "layer": args.layer,
        "tokens": args.tokens,
        "seed": args.seed,
        "device": str(device),
        "device_requested": args.device,
        "kda_backend": args.kda_backend,
        "attn_implementation": args.attn_implementation,
        "torch": torch.__version__,
        "devices": device_inventory(),
        "is_kda_layer": index.is_kda_layer(args.layer),
        "is_moe_layer": index.is_moe_layer(args.layer),
        "proposed_decision_line": {
            "prefix_out_relL2_max": PROPOSED_PREFIX_RELL2,
            "router_topk_agreement_min": PROPOSED_TOPK_AGREEMENT,
            "status": "PROPOSED DECISION LINE for Cyril -- this script never gates on it",
        },
    }

    started = time.time()
    weights, timing = load_layer(index, args.layer, args.device, args.experts_device or args.device)
    receipt["read"] = timing
    receipt["parameters"] = weights.parameter_counts()
    log(f"layer {args.layer} loaded in {time.time()-started:.0f}s: {timing['gigabytes']} GB "
        f"at {timing['MB_per_s']} MB/s")

    hidden_size = index.hidden_size
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    hidden = torch.randn(1, args.tokens, hidden_size, generator=generator, dtype=torch.float32)
    hidden = (hidden * 0.02).to(torch.bfloat16).to(device)
    receipt["input_sha256"] = sha256_tensor(hidden)
    blocks = index.attn_res_block_size or 1
    # ceil, not floor: the stack entering layer L holds ceil(L / block_size)
    # snapshots.  `L // block_size` is only right at block_size 1 and undercounts
    # every real K3 layer that is not a multiple of 12.
    n_blocks = blocks_before(args.layer, blocks)
    receipt["attn_res"] = {
        "block_size": blocks,
        "blocks_entering_this_layer": n_blocks,
        "residual_branch": ("snapshot (prefix reset to None)" if args.layer % blocks == 0
                            else "carried = prefix_sum + attn_out"),
    }
    block_residual = (torch.randn(args.tokens, n_blocks, hidden_size, generator=generator) * 0.02).to(
        torch.bfloat16).to(device)
    receipt["block_residual_shape"] = list(block_residual.shape)

    receipt["mxfp4"] = check_mxfp4(weights)
    receipt["kda_gate_axis"] = check_kda_gate_axis(weights, cfg)
    receipt["kda_backends"] = check_kda_backends(weights, cfg, hidden)

    try:
        modeling, config_module = import_k3_modeling(args.src_dir, args.model_dir)
        receipt["modeling_import"] = {"ok": True, "file": modeling.__file__,
                                      "transformers_compat": dict(TRANSFORMERS_COMPAT)}
    except Exception as exc:
        modeling = config_module = None
        receipt["modeling_import"] = {"ok": False, "error": repr(exc)[:600]}

    if modeling is not None:
        receipt["halves"] = check_halves(weights, cfg, hidden, modeling, config_module, args)
        receipt["whole_layer"] = check_whole_layer(
            weights, cfg, hidden, block_residual, modeling, config_module, args
        )
        prefix = receipt["whole_layer"].get("prefix_out", {})
        agreement = receipt["halves"].get("moe", {}).get("topk_set_agreement")
        receipt["meets_proposed_line"] = {
            "prefix_out_relL2": prefix.get("relL2"),
            "prefix_out_ok": (prefix.get("relL2") is not None and prefix["relL2"] <= PROPOSED_PREFIX_RELL2),
            "router_topk_agreement": agreement,
            "router_topk_ok": (agreement is not None and agreement >= PROPOSED_TOPK_AGREEMENT),
        }

    receipt["total_seconds"] = round(time.time() - started, 1)
    out = args.out or os.path.join(".", f"k3_layer{args.layer}_exactness_receipt.json")
    receipt["receipt_sha256"] = write_receipt(out, receipt)
    log(f"receipt -> {out}")
    print(f"prefix_out: {receipt.get('whole_layer', {}).get('prefix_out')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
