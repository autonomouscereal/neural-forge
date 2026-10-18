#!/usr/bin/env python3
"""qtext_vl_divergence_diag.py -- where does T(i) stop being bit-identical to the stock VL forward?

The 0.8B parity-vl arm on a P40 (2026-10-12, receipt
$OC/distill/tps/qtext-parity-vl-08b-fp32-p40.json) matched top-1 at 101/101 positions but
not every top-64 order (1 position) and not the lse (3.6e-5), while the text arm on the
same card was bit-identical.  This script runs both engines in ONE process on the SAME
collated batch and compares every stage bitwise: vision parameters and buffers, the
vision tower output, inputs_embeds after the scatter, the M-RoPE position ids, the final
hidden state -- and repeats the stock forward to measure run-to-run determinism, which
is the first thing a non-zero distance has to be checked against.  Diagnostic only; it
writes one JSON and changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOL_NAME = "qtext_vl_divergence_diag"
TOOL_VERSION = "1.0.0"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qtext_teacher_cache as qc  # noqa: E402


def dist(a, b) -> dict:
    import torch

    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "equal": False}
    return {"equal": bool(torch.equal(a, b)), "max_abs": float((a - b).abs().max()) if a.numel() else 0.0}


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--linear-attn-kernel", default="auto")
    ap.add_argument("--deterministic", action="store_true", help="torch.use_deterministic_algorithms + cudnn.deterministic")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    kernel = qc.select_linear_attention_kernel(args.linear_attn_kernel)
    dtype = qc.resolve_dtype(args.dtype)
    corpus = qc.load_vl_corpus(Path(args.corpus))
    processor = qc.load_processor(Path(args.model_dir), None)
    pad = int(processor.tokenizer.pad_token_id or 0)
    batch = next(qc.vl_batches(corpus, processor, 0, corpus["rows"], corpus["rows"], pad))
    moved = {k: (v.to(args.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
             if k not in ("rows", "lengths", "store_mask")}
    streamed = qc.build_streamed_teacher(Path(args.model_dir), args.device, dtype, "sdpa", with_vision=True)
    stock = qc.build_full_load_teacher(Path(args.model_dir), args.device, dtype, "sdpa")
    out: dict = {"tool": TOOL_NAME, "version": TOOL_VERSION, "kernel": kernel, "dtype": args.dtype,
                 "deterministic_flags": args.deterministic, "cudnn_version": torch.backends.cudnn.version()}

    sv, rv = streamed.vl_model.visual, stock.vl_model.visual
    sp, rp = dict(sv.named_parameters()), dict(rv.named_parameters())
    out["vision_param_names_equal"] = sorted(sp) == sorted(rp)
    out["vision_params_not_bit_equal"] = [n for n in sp if not torch.equal(sp[n].float().cpu(), rp[n].float().cpu())]
    sb, rb = dict(sv.named_buffers()), dict(rv.named_buffers())
    out["vision_buffers"] = {n: {"streamed": str(sb[n].dtype), "stock": str(rb.get(n).dtype) if n in rb else None,
                                 **dist(sb[n], rb[n])} for n in sb if n in rb}
    out["vision_attn_impl"] = {"streamed": sv.config._attn_implementation, "stock": rv.config._attn_implementation}
    out["text_attn_impl"] = {"streamed": streamed.text_trunk.config._attn_implementation,
                             "stock": stock.text_trunk.config._attn_implementation}

    with torch.no_grad():
        feats = {}
        for name, teacher in (("streamed", streamed), ("stock", stock), ("stock_again", stock)):
            f = teacher.vl_model.get_image_features(moved["pixel_values"], moved["image_grid_thw"], return_dict=True)
            feats[name] = torch.cat(f.pooler_output, dim=0)
        out["vision_output_streamed_vs_stock"] = dist(feats["streamed"], feats["stock"])
        out["vision_output_stock_run_to_run"] = dist(feats["stock"], feats["stock_again"])
        # Same features into both: does the remaining (text) path agree?
        pos = {}
        hidden = {}
        for name, teacher in (("streamed", streamed), ("stock", stock), ("stock_again", stock)):
            teacher.vl_model.rope_deltas = None
            pos[name] = teacher.vl_model.compute_3d_position_ids(
                input_ids=moved["input_ids"], inputs_embeds=None, image_grid_thw=moved["image_grid_thw"],
                mm_token_type_ids=moved["mm_token_type_ids"])
            hidden[name] = teacher.vl_hidden(moved)
        out["position_ids_streamed_vs_stock"] = dist(pos["streamed"], pos["stock"])
        out["hidden_streamed_vs_stock"] = dist(hidden["streamed"], hidden["stock"])
        out["hidden_stock_run_to_run"] = dist(hidden["stock"], hidden["stock_again"])
        # Inject the stock vision features into both text paths: isolates the LLM half.
        emb_s = streamed.vl_model.get_input_embeddings()(moved["input_ids"])
        emb_r = stock.vl_model.get_input_embeddings()(moved["input_ids"])
        mask = (moved["input_ids"] == stock.config.image_token_id).unsqueeze(-1)
        emb_s = emb_s.masked_scatter(mask, feats["stock"].to(emb_s.dtype))
        emb_r = emb_r.masked_scatter(mask, feats["stock"].to(emb_r.dtype))
        out["inputs_embeds_same_features"] = dist(emb_s, emb_r)
        h_s = streamed.text_trunk(inputs_embeds=emb_s, position_ids=pos["stock"], use_cache=False).last_hidden_state
        h_r = stock.text_trunk(inputs_embeds=emb_r, position_ids=pos["stock"], use_cache=False).last_hidden_state
        out["llm_hidden_same_features_streamed_vs_stock"] = dist(h_s, h_r)
    Path(args.out).write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in out.items() if k not in ("vision_buffers",)}, indent=1, sort_keys=True))
    print("vision_buffers", json.dumps(out["vision_buffers"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
