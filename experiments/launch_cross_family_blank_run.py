#!/usr/bin/env python3
"""Run, evaluate, and compare one zero-recovery cross-family blank fold."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from neural_forge.hf_blank_checkpoint import bootstrap_blank_decoder_checkpoint
from neural_forge.hf_tools import evaluate_text_model


def _progress(event: str, **values: Any) -> None:
    fields = " ".join(f"{key}={values[key]}" for key in sorted(values))
    print(f"neural_forge_progress event={event} {fields}".rstrip(), flush=True)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def load_text_corpus(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}.") from exc
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("prompt")
            else:
                text = None
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"Calibration row {path}:{line_number} has no nonempty text."
                )
            texts.append(text.strip())
    if not texts:
        raise ValueError("Calibration corpus is empty.")
    return texts


def deterministic_prompt_split(
    texts: Sequence[str],
    calibration_count: int,
    validation_count: int,
) -> tuple[list[str], list[str], dict[str, Any]]:
    if calibration_count <= 0 or validation_count <= 0:
        raise ValueError("Prompt split counts must be positive.")
    total = calibration_count + validation_count
    if total > len(texts):
        raise ValueError(f"Need {total} prompts, but corpus contains {len(texts)}.")
    if total == len(texts):
        indices = list(range(total))
    else:
        denominator = total - 1 if total > 1 else 1
        indices = [(slot * (len(texts) - 1)) // denominator for slot in range(total)]
    if len(set(indices)) != total:
        raise RuntimeError("Deterministic prompt sampling produced duplicate indices.")
    calibration_indices = indices[::2]
    validation_indices = indices[1::2]
    ordered = calibration_indices + validation_indices
    calibration_indices = ordered[:calibration_count]
    validation_indices = ordered[calibration_count:total]
    if set(calibration_indices) & set(validation_indices):
        raise RuntimeError("Calibration and validation prompt splits overlap.")
    calibration = [texts[index] for index in calibration_indices]
    validation = [texts[index] for index in validation_indices]
    report = {
        "corpus_count": len(texts),
        "corpus_sha256": _canonical_sha256(list(texts)),
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "calibration_indices": calibration_indices,
        "validation_indices": validation_indices,
        "calibration_sha256": _canonical_sha256(calibration),
        "validation_sha256": _canonical_sha256(validation),
    }
    return calibration, validation, report


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _release_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _compare_evals(
    parent_eval: Path,
    child_eval: Path,
    output: Path,
    max_samples: int,
) -> dict[str, Any]:
    experiments_dir = str(Path(__file__).resolve().parent)
    if experiments_dir not in sys.path:
        sys.path.insert(0, experiments_dir)
    from compare_eval_retention import compare

    report = compare(parent_eval, child_eval, max_samples=max_samples)
    _write_json(output, report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = (args.run_root / args.tag).resolve()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    texts = load_text_corpus(args.calibration_file)
    calibration, validation, split_report = deterministic_prompt_split(
        texts,
        args.calibration_count,
        args.validation_count,
    )
    _write_json(run_dir / "prompt_split.json", split_report)
    _progress(
        "cross_family_blank_start",
        calibration=len(calibration),
        validation=len(validation),
        target_family=args.target_family,
        target_hidden=args.hidden_size,
        target_layers=args.num_hidden_layers,
    )

    checkpoint_dir = run_dir / "checkpoint"
    checkpoint_report = bootstrap_blank_decoder_checkpoint(
        args.source_model,
        target_family=args.target_family,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        head_dim=args.head_dim,
        max_position_embeddings=args.max_position_embeddings,
        output_dir=checkpoint_dir,
        calibration_prompts=calibration,
        validation_prompts=validation,
        tokenizer=None,
        seed=args.seed,
        dtype=args.dtype,
        donor_dtype=args.donor_dtype,
        target_dtype=args.target_dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        exact_parameter_budget=args.exact_parameter_budget,
        trace_pooling=args.trace_pooling,
        residual_map_method=args.residual_map_method,
        ridge=args.ridge,
        max_operator_delta_ratio=args.max_operator_delta_ratio,
        min_final_state_relative_improvement=args.min_final_state_improvement,
        min_logit_relative_improvement=args.min_logit_improvement,
        min_top1_agreement_gain=args.min_top1_gain,
        fit_full_lm_head=args.fit_full_lm_head,
        max_full_head_fit_elements=args.max_full_head_fit_elements,
        max_coordinate_rows=args.max_coordinate_rows,
        vocab_projection_chunk_size=args.vocab_projection_chunk_size,
    )
    _progress(
        "cross_family_blank_fold_done",
        accepted=checkpoint_report["accepted"],
        compression=(
            checkpoint_report["compression"]["unique"]["source_to_child_ratio"]
        ),
    )

    run_report: dict[str, Any] = {
        "schema": "neural-forge-cross-family-blank-run-v1",
        "tag": args.tag,
        "run_dir": str(run_dir),
        "source_model": args.source_model,
        "prompt_split": split_report,
        "checkpoint_report": checkpoint_report,
        "parent_eval": str(args.cached_parent_eval.resolve()),
        "eval_suite": str(args.eval_suite.resolve()),
        "zero_recovery": True,
    }
    if not checkpoint_report["accepted"]:
        run_report["status"] = "fold-rejected"
        _write_json(run_dir / "run_report.json", run_report)
        return run_report

    _release_cuda()
    candidate_eval_path = run_dir / "candidate_eval.json"
    _progress("cross_family_blank_eval_start")
    evaluate_text_model(
        str(checkpoint_dir),
        candidate_eval_path,
        torch_dtype=args.dtype,
        device=args.device,
        trust_remote_code=False,
        max_new_tokens=args.max_new_tokens,
        eval_suite=args.eval_suite,
        max_text_evals=args.max_text_evals,
        max_choice_probes=args.max_choice_probes,
        max_generation_prompts=args.max_generation_prompts,
        max_behavior_probes=args.max_behavior_probes,
    )
    comparison_path = run_dir / "parent_candidate_comparison.json"
    comparison = _compare_evals(
        args.cached_parent_eval,
        candidate_eval_path,
        comparison_path,
        args.manual_sample_count,
    )
    run_report.update(
        {
            "status": "complete",
            "candidate_eval": str(candidate_eval_path),
            "comparison": str(comparison_path),
            "aggregate": {
                "parent": comparison["parent"],
                "child": comparison["child"],
                "param_compression_ratio": comparison["param_compression_ratio"],
                "behavior": comparison["behavior"]["overall"],
                "generation": comparison["generation"]["overall"],
                "choice": comparison["choice"]["overall"],
            },
            "manual_samples": comparison["manual_samples"],
        }
    )
    _write_json(run_dir / "run_report.json", run_report)
    _progress(
        "cross_family_blank_complete",
        behavior_child=comparison["child"]["behavior_correct"],
        choice_child=comparison["child"]["choice_correct"],
        generation_child=comparison["child"]["generation_correct"],
    )
    return run_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument(
        "--target-family", choices=["gpt2", "llama", "qwen3"], default="llama"
    )
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--num-hidden-layers", type=int, required=True)
    parser.add_argument("--num-attention-heads", type=int, required=True)
    parser.add_argument("--num-key-value-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--max-position-embeddings", type=int, required=True)
    parser.add_argument("--exact-parameter-budget", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument(
        "--donor-dtype", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--target-dtype", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--trace-pooling",
        choices=["token_rows", "mean", "last_valid"],
        default="token_rows",
    )
    parser.add_argument(
        "--residual-map-method", choices=["sparse_fold", "pca"], default="sparse_fold"
    )
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--max-operator-delta-ratio", type=float, default=4.0)
    parser.add_argument("--min-final-state-improvement", type=float, default=0.0)
    parser.add_argument("--min-logit-improvement", type=float, default=0.0)
    parser.add_argument("--min-top1-gain", type=float, default=0.0)
    parser.add_argument("--fit-full-lm-head", action="store_true")
    parser.add_argument("--max-full-head-fit-elements", type=int, default=10_000_000)
    parser.add_argument("--max-coordinate-rows", type=int, default=4096)
    parser.add_argument("--vocab-projection-chunk-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-text-evals", type=int, default=1000)
    parser.add_argument("--max-choice-probes", type=int, default=250)
    parser.add_argument("--max-generation-prompts", type=int, default=1000)
    parser.add_argument("--max-behavior-probes", type=int, default=1000)
    parser.add_argument("--manual-sample-count", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
