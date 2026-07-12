#!/usr/bin/env python3
"""Export, reload, and fully evaluate one standalone Qwen3 basis checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from neural_forge.hf_qwen_basis_checkpoint import (
    export_qwen3_basis_checkpoint,
    load_qwen3_basis_checkpoint,
)
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
    checkpoint = output / "checkpoint"
    ranks = {
        "q": args.q_rank,
        "k": args.k_rank,
        "v": args.v_rank,
        "o": args.o_rank,
        "gate": args.gate_rank,
        "up": args.up_rank,
        "down": args.down_rank,
        "vocab": args.vocab_rank,
    }
    _progress("qwen_basis_load_parent", ranks=json.dumps(ranks, sort_keys=True))
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
    _progress("qwen_basis_export_start", budget=args.parameter_budget)
    manifest = export_qwen3_basis_checkpoint(
        model,
        checkpoint,
        rank_plan=ranks,
        parameter_budget=args.parameter_budget,
        feature_chunk_size=args.feature_chunk_size,
        vocab_algorithm=args.vocab_algorithm,
        vocab_row_chunk_size=args.vocab_row_chunk_size,
        vocab_oversampling=args.vocab_oversampling,
        vocab_power_iterations=args.vocab_power_iterations,
        vocab_seed=args.seed,
    )
    tokenizer.save_pretrained(checkpoint)
    _progress(
        "qwen_basis_export_done",
        unique=manifest["parameter_counts"]["actual_unique_parameter_count"],
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _progress("qwen_basis_reload_start")
    candidate = load_qwen3_basis_checkpoint(checkpoint, device=args.device)
    _progress("qwen_basis_reload_done")
    candidate_eval_path = output / "candidate_eval.json"
    candidate_eval = evaluate_loaded_text_model(
        candidate,
        tokenizer,
        candidate_eval_path,
        model_label=str(checkpoint),
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
    report = {
        "schema": "neural-forge-qwen-basis-checkpoint-run-v1",
        "source_model": args.source_model,
        "checkpoint": str(checkpoint),
        "rank_plan": ranks,
        "manifest": manifest,
        "candidate_eval": candidate_eval,
        "comparison": str(comparison_path),
        "zero_recovery": True,
        "parent_loaded_at_runtime": False,
        "status": "complete",
    }
    _write_json(output / "run_report.json", report)
    _progress(
        "qwen_basis_complete",
        behavior=comparison["child"]["behavior_correct"],
        choice=comparison["child"]["choice_correct"],
        generation=comparison["child"]["generation_correct"],
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument("--parameter-budget", type=int, default=134_445_568)
    parser.add_argument("--q-rank", type=int, default=1)
    parser.add_argument("--k-rank", type=int, default=6)
    parser.add_argument("--v-rank", type=int, default=6)
    parser.add_argument("--o-rank", type=int, default=1)
    parser.add_argument("--gate-rank", type=int, default=1)
    parser.add_argument("--up-rank", type=int, default=1)
    parser.add_argument("--down-rank", type=int, default=1)
    parser.add_argument("--vocab-rank", type=int, default=46)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-chunk-size", type=int, default=65_536)
    parser.add_argument("--vocab-algorithm", choices=["auto", "full", "randomized"], default="randomized")
    parser.add_argument("--vocab-row-chunk-size", type=int, default=4096)
    parser.add_argument("--vocab-oversampling", type=int, default=16)
    parser.add_argument("--vocab-power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
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
