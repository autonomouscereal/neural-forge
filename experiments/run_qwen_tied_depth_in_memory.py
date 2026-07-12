#!/usr/bin/env python3
"""Build and fully evaluate one zero-gradient Qwen3 tied-depth child."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_forge.hf_tied_depth import prepare_qwen3_tied_depth_model
from neural_forge.hf_tools import evaluate_loaded_text_model


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _progress(event: str, **values: Any) -> None:
    fields = " ".join(f"{key}={values[key]}" for key in sorted(values))
    print(f"neural_forge_progress event={event} {fields}".rstrip(), flush=True)


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
    _progress("load_parent")
    donor_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.source_model,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    target_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.target_tokenizer,
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
    _progress(
        "tie_depth_start",
        attention=args.attention_source_layer,
        mlp=args.mlp_source_layer,
    )
    report = prepare_qwen3_tied_depth_model(
        model,
        donor_tokenizer,
        target_tokenizer,
        attention_source_layer=args.attention_source_layer,
        mlp_source_layer=args.mlp_source_layer,
    )
    _write_json(output / "compile_report.json", report.as_dict())
    _progress(
        "tie_depth_done",
        compression=f"{report.unique_compression_ratio:.6f}",
        unique=report.child_unique_parameter_count,
    )

    candidate_eval_path = output / "candidate_eval.json"
    candidate_eval = evaluate_loaded_text_model(
        model,
        target_tokenizer,
        candidate_eval_path,
        model_label=(
            f"qwen3-tied-depth:a{args.attention_source_layer}:m{args.mlp_source_layer}"
        ),
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )

    from compare_eval_retention import compare

    comparison_path = output / "parent_candidate_comparison.json"
    comparison = compare(
        args.cached_parent_eval,
        candidate_eval_path,
        max_samples=args.manual_sample_count,
    )
    _write_json(comparison_path, comparison)
    run_report = {
        "schema": "neural-forge-qwen-tied-depth-memory-run-v1",
        "source_model": args.source_model,
        "target_tokenizer": args.target_tokenizer,
        "compile": report.as_dict(),
        "candidate_eval": candidate_eval,
        "comparison": str(comparison_path),
        "zero_recovery": True,
        "standalone_checkpoint": False,
        "status": "complete",
    }
    _write_json(output / "run_report.json", run_report)
    _progress(
        "complete",
        behavior=comparison["child"]["behavior_correct"],
        choice=comparison["child"]["choice_correct"],
        generation=comparison["child"]["generation_correct"],
    )
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument("--attention-source-layer", type=int, required=True)
    parser.add_argument("--mlp-source-layer", type=int, required=True)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--device", default="cuda:0")
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
