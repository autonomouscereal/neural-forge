#!/usr/bin/env python3
"""Compile and fully evaluate one in-memory Qwen/Llama depth-basis control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_forge.hf_depth_basis_ir import PROJECTION_ROLES, compile_hf_depth_basis_ir
from neural_forge.hf_tools import evaluate_loaded_text_model


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dtype(torch: Any, name: str) -> Any:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers

    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    role_ranks = {role: int(getattr(args, f"{role}_rank")) for role in PROJECTION_ROLES}
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.source_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.source_model,
        dtype=_dtype(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        compiled = compile_hf_depth_basis_ir(
            model,
            role_ranks=role_ranks,
            feature_chunk_size=args.feature_chunk_size,
        )
    compile_report = compiled.as_dict()
    _write_json(output / "compile_report.json", compile_report)

    candidate_eval_path = output / "candidate_eval.json"
    candidate_eval = evaluate_loaded_text_model(
        model,
        tokenizer,
        candidate_eval_path,
        model_label=f"depth-basis:{args.source_model}",
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )

    from compare_eval_retention import compare

    comparison = compare(
        args.cached_parent_eval,
        candidate_eval_path,
        max_samples=args.manual_sample_count,
    )
    _write_json(output / "parent_candidate_comparison.json", comparison)
    report = {
        "schema": "neural-forge-qwen-depth-basis-memory-run-v1",
        "source_model": args.source_model,
        "role_ranks": role_ranks,
        "compile": compile_report,
        "candidate_eval": candidate_eval,
        "comparison": str(output / "parent_candidate_comparison.json"),
        "zero_recovery": True,
        "standalone_checkpoint": False,
        "status": "complete",
    }
    _write_json(output / "run_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    for role in PROJECTION_ROLES:
        parser.add_argument(f"--{role}-rank", type=int, default=1)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-chunk-size", type=int, default=65_536)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-text-evals", type=int, default=1000)
    parser.add_argument("--max-choice-probes", type=int, default=250)
    parser.add_argument("--max-generation-prompts", type=int, default=1000)
    parser.add_argument("--max-behavior-probes", type=int, default=1000)
    parser.add_argument("--manual-sample-count", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
