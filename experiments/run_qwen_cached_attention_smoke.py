#!/usr/bin/env python3
"""Run one bounded real-Qwen cached residual-attention trace.

This is intentionally separate from the unified surgery runner.  It loads one
dense Hugging Face model or one standalone Neural Forge Qwen3 codebook artifact,
collects a single prompt/layer lane, and writes a compact JSON evidence record.
It never installs a candidate, edits weights, or writes into a model checkout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch

from neural_forge.hf_cached_attention_trace import (
    HFCachedAttentionTraceLimits,
    collect_hf_cached_attention_trace,
)
from neural_forge.hf_qwen_codebook_model import load_qwen3_codebook_checkpoint


def _comma_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers.") from error
    if not result:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    return result


def _hook_inventory(model: torch.nn.Module) -> tuple[int, int]:
    return (
        sum(len(module._forward_hooks) for module in model.modules()),
        sum(len(module._forward_pre_hooks) for module in model.modules()),
    )


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _load(args: argparse.Namespace) -> torch.nn.Module:
    if args.codebook_checkpoint is not None:
        model = load_qwen3_codebook_checkpoint(
            args.codebook_checkpoint.expanduser().resolve(),
            device="cpu",
        )
        return model.to(device=args.device, dtype=_dtype(args.dtype)).eval()
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Dense smoke loading requires Transformers.") from error
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_dtype(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    return model.to(args.device).eval()


def _tokenizer(args: argparse.Namespace) -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Smoke tokenization requires Transformers.") from error
    source = args.tokenizer or args.model
    if source is None:
        raise ValueError("--tokenizer is required with --codebook-checkpoint.")
    return transformers.AutoTokenizer.from_pretrained(
        source,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.stage-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    model = _load(args)
    tokenizer = _tokenizer(args)
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape[:1]) != (1,):
        raise RuntimeError("Tokenizer did not return one input_ids row.")
    groups = _comma_ints(args.group_indices, "group indices")
    forced = (
        None
        if args.forced_token_ids is None
        else _comma_ints(args.forced_token_ids, "forced token IDs")
    )
    if args.lane == "forced" and forced is None:
        raise ValueError("--forced-token-ids is required for --lane forced.")
    if args.lane == "greedy" and forced is not None:
        raise ValueError("--forced-token-ids is valid only for --lane forced.")
    hooks_before = _hook_inventory(model)
    backend_before = str(
        getattr(getattr(model, "config", None), "_attn_implementation", None)
        or "eager"
    )
    trace = collect_hf_cached_attention_trace(
        model,
        input_ids,
        layer_index=args.layer_index,
        group_indices=groups,
        horizon=args.horizon,
        sequence_id=args.sequence_id,
        lane=args.lane,
        forced_token_ids=forced,
        limits=HFCachedAttentionTraceLimits(
            max_horizon=args.max_horizon,
            max_prompt_tokens=args.max_prompt_tokens,
            max_groups=args.max_groups,
            max_capture_bytes=args.max_capture_bytes,
            logit_rtol=args.logit_rtol,
            logit_atol=args.logit_atol,
            context_rtol=args.context_rtol,
            context_atol=args.context_atol,
        ),
    )
    hooks_after = _hook_inventory(model)
    backend_after = str(
        getattr(getattr(model, "config", None), "_attn_implementation", None)
        or "eager"
    )
    if hooks_after != hooks_before:
        raise RuntimeError("Cached smoke collector leaked hooks.")
    if backend_after != backend_before:
        raise RuntimeError("Cached smoke collector changed the attention backend.")
    return {
        "schema": "neural-forge-qwen-cached-attention-smoke-v1",
        "source": {
            "model": args.model,
            "codebook_checkpoint": (
                None
                if args.codebook_checkpoint is None
                else str(args.codebook_checkpoint.expanduser().resolve())
            ),
            "tokenizer": args.tokenizer or args.model,
            "device": args.device,
            "dtype": args.dtype,
        },
        "prompt": {
            "text": args.prompt,
            "token_count": int(input_ids.shape[1]),
        },
        "hook_inventory_before": list(hooks_before),
        "hook_inventory_after": list(hooks_after),
        "attention_backend_before": backend_before,
        "attention_backend_after": backend_after,
        "trace": trace.as_dict(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--codebook-checkpoint", type=Path)
    result.add_argument("--tokenizer")
    result.add_argument("--prompt", required=True)
    result.add_argument("--layer-index", type=int, required=True)
    result.add_argument("--group-indices", required=True)
    result.add_argument("--horizon", type=int, default=2)
    result.add_argument("--sequence-id", default="real-smoke-0")
    result.add_argument("--lane", choices=("forced", "greedy"), default="greedy")
    result.add_argument("--forced-token-ids")
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    result.add_argument("--max-horizon", type=int, default=4)
    result.add_argument("--max-prompt-tokens", type=int, default=64)
    result.add_argument("--max-groups", type=int, default=2)
    result.add_argument("--max-capture-bytes", type=int, default=64 * 1024 * 1024)
    result.add_argument("--logit-rtol", type=float, default=5e-3)
    result.add_argument("--logit-atol", type=float, default=5e-3)
    result.add_argument("--context-rtol", type=float, default=8e-3)
    result.add_argument("--context-atol", type=float, default=8e-3)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--local-files-only", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        _write_json(args.output, payload)
    except Exception as error:
        print(f"cached attention smoke failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
