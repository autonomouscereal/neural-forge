#!/usr/bin/env python3
"""Run dense-parent K/V/residual-delta causal interventions on one Qwen child.

The parent and candidate are loaded sequentially, so a 4B parent and p34/p35
codebook child never need to coexist on the GPU.  Output is JSON evidence only;
the candidate checkpoint is never saved or edited.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import torch

from neural_forge.hf_qwen_codebook_model import load_qwen3_codebook_checkpoint
from neural_forge.hf_residual_attention_oracle import (
    HFResidualAttentionOracleLimits,
    capture_parent_residual_attention_oracle_trace,
    fingerprint_hf_oracle_tokenizer,
    run_residual_attention_oracle_suite,
)


RUN_SCHEMA = "neural-forge-qwen-residual-attention-oracle-run-v2"


def _comma_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{name} must be comma-separated integers."
        ) from error
    if not result:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty.")
    return result


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _dense_model(
    source: str,
    *,
    args: argparse.Namespace,
) -> torch.nn.Module:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Dense Qwen loading requires Transformers.") from error
    model = transformers.AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=_dtype(args.dtype),
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    return model.to(args.device).eval()


def _candidate_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.candidate_codebook_checkpoint is not None:
        model = load_qwen3_codebook_checkpoint(
            args.candidate_codebook_checkpoint.expanduser().resolve(),
            device="cpu",
        )
        return model.to(device=args.device, dtype=_dtype(args.dtype)).eval()
    return _dense_model(args.candidate_model, args=args)


def _candidate_parent_token_rows(path: Path | None) -> torch.Tensor | None:
    if path is None:
        return None
    destination = path.expanduser().resolve()
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Codebook checkpoint lacks manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_name = manifest.get("tensor_file")
    if not isinstance(tensor_name, str) or Path(tensor_name).name != tensor_name:
        raise RuntimeError("Codebook manifest has an invalid tensor_file.")
    tensor_path = destination / tensor_name
    if not tensor_path.is_file():
        raise RuntimeError("Codebook manifest tensor_file is unavailable.")
    try:
        from safetensors import safe_open
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Codebook row loading requires safetensors.") from error
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        if "parent_token_rows" not in handle.keys():
            raise RuntimeError("Codebook checkpoint lacks parent_token_rows.")
        rows = handle.get_tensor("parent_token_rows")
    return rows.detach().to(device="cpu", dtype=torch.int64).contiguous().clone()


def _tokenizer(args: argparse.Namespace) -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - optional dependency guard
        raise RuntimeError("Oracle tokenization requires Transformers.") from error
    return transformers.AutoTokenizer.from_pretrained(
        args.tokenizer or args.parent_model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )


def _release_runtime() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.stage-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = _tokenizer(args)
    tokenizer_identity = fingerprint_hf_oracle_tokenizer(tokenizer)
    parent_token_rows = _candidate_parent_token_rows(args.candidate_codebook_checkpoint)
    encoded = tokenizer(
        args.prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=args.max_prompt_tokens,
    )
    input_ids = encoded.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape[:1]) != (1,):
        raise RuntimeError("Tokenizer did not return exactly one input_ids row.")
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
    if args.feedback == "autoregressive" and args.lane != "greedy":
        raise ValueError("--feedback autoregressive requires --lane greedy.")
    limits = HFResidualAttentionOracleLimits(
        max_horizon=args.max_horizon,
        max_prompt_tokens=args.max_prompt_tokens,
        max_trace_bytes=args.max_trace_bytes,
        max_replay_bytes=args.max_replay_bytes,
        fingerprint_chunk_bytes=args.fingerprint_chunk_bytes,
        cache_rtol=args.cache_rtol,
        cache_atol=args.cache_atol,
        residual_rtol=args.residual_rtol,
        residual_atol=args.residual_atol,
        logit_rtol=args.logit_rtol,
        logit_atol=args.logit_atol,
    )

    parent = _dense_model(args.parent_model, args=args)
    try:
        parent_trace = capture_parent_residual_attention_oracle_trace(
            parent,
            input_ids,
            tokenizer_identity_sha256=tokenizer_identity,
            layer_index=args.layer_index,
            group_indices=groups,
            horizon=args.horizon,
            sequence_id=args.sequence_id,
            lane=args.lane,
            forced_token_ids=forced,
            parent_token_rows=parent_token_rows,
            limits=limits,
        )
    finally:
        del parent
        _release_runtime()

    candidate = _candidate_model(args)
    try:
        suite = run_residual_attention_oracle_suite(
            candidate,
            parent_trace,
            tokenizer_identity_sha256=tokenizer_identity,
            feedback=args.feedback,
            limits=limits,
        )
    finally:
        del candidate
        _release_runtime()

    return {
        "schema": RUN_SCHEMA,
        "source": {
            "parent_model": args.parent_model,
            "candidate_model": args.candidate_model,
            "candidate_codebook_checkpoint": (
                None
                if args.candidate_codebook_checkpoint is None
                else str(args.candidate_codebook_checkpoint.expanduser().resolve())
            ),
            "tokenizer": args.tokenizer or args.parent_model,
            "device": args.device,
            "dtype": args.dtype,
            "models_loaded_sequentially": True,
            "parent_token_rows_loaded_before_parent_capture": (
                parent_token_rows is not None
            ),
        },
        "prompt": {
            "text": args.prompt,
            "token_count": int(input_ids.shape[1]),
            "tokenizer_identity_sha256": tokenizer_identity,
        },
        "parent_trace": parent_trace.as_dict(),
        "suite": suite.as_dict(),
        "checkpoint_mutated": False,
        "checkpoint_written": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parent-model", required=True)
    candidate = result.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-model")
    candidate.add_argument("--candidate-codebook-checkpoint", type=Path)
    result.add_argument("--tokenizer")
    result.add_argument("--prompt", required=True)
    result.add_argument("--layer-index", type=int, required=True)
    result.add_argument("--group-indices", required=True)
    result.add_argument("--horizon", type=int, default=8)
    result.add_argument("--sequence-id", default="oracle-0")
    result.add_argument("--lane", choices=("forced", "greedy"), default="greedy")
    result.add_argument("--forced-token-ids")
    result.add_argument(
        "--feedback",
        choices=("forced-cache-recurrent", "autoregressive"),
        default="forced-cache-recurrent",
    )
    result.add_argument("--device", default="cuda:0")
    result.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    result.add_argument("--max-horizon", type=int, default=32)
    result.add_argument("--max-prompt-tokens", type=int, default=256)
    result.add_argument("--max-trace-bytes", type=int, default=512 * 1024 * 1024)
    result.add_argument("--max-replay-bytes", type=int, default=256 * 1024 * 1024)
    result.add_argument("--fingerprint-chunk-bytes", type=int, default=16 * 1024 * 1024)
    result.add_argument("--cache-rtol", type=float, default=1e-4)
    result.add_argument("--cache-atol", type=float, default=1e-5)
    result.add_argument("--residual-rtol", type=float, default=8e-3)
    result.add_argument("--residual-atol", type=float, default=8e-3)
    result.add_argument("--logit-rtol", type=float, default=5e-3)
    result.add_argument("--logit-atol", type=float, default=5e-3)
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
        print(f"residual-attention oracle failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(payload, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
