#!/usr/bin/env python3
"""Fresh-reload whole-output evaluation for a dense structural LFM2 child.

The parent is never loaded or executed.  Parent behavior comes from an
authenticated response envelope that was generated once previously.  The
candidate is loaded directly from local JSON+safetensors, its exact dense
logical parameter count and state hash are rebound to the structural-fold
report, and the existing 28-case whole-output gate generates coherent-output
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from experiments import run_lfm2_dynamic_structural_compression as structural
from experiments import run_lfm2_residual_attention_fold as residual
from experiments.run_lfm2_dynamic_catalog import fingerprint_local_tokenizer
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_behavior import (
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
)
from neural_forge.dynamic_sequence_horizon import (
    evaluation_rollout_contract,
)


SCHEMA = "neural-forge-lfm2-dynamic-structural-whole-output-v1"


class Lfm2DynamicStructuralEvaluationError(RuntimeError):
    """The local child or authenticated parent envelope failed closed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2DynamicStructuralEvaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


def _object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Lfm2DynamicStructuralEvaluationError(
            f"Could not read {name}: {path}."
        ) from exc
    if not isinstance(value, Mapping):
        raise Lfm2DynamicStructuralEvaluationError(f"{name} must be an object.")
    return dict(value)


def _load_structural_report(checkpoint: Path) -> dict[str, Any]:
    report = _object(checkpoint / "structural-fold-report.json", "fold report")
    seal = report.pop("run_sha256", None)
    if not isinstance(seal, str) or structural._json_sha256(report) != seal:
        raise Lfm2DynamicStructuralEvaluationError(
            "Structural fold report seal is missing or invalid."
        )
    report["run_sha256"] = seal
    if (
        report.get("schema") != structural.RUN_SCHEMA
        or report.get("training_steps") != 0
        or report.get("optimizer_steps") != 0
        or report.get("backward_calls") != 0
        or report.get("dtype_or_packing_reduction_claimed") is not False
    ):
        raise Lfm2DynamicStructuralEvaluationError(
            "Fold report is not a zero-training dense structural run."
        )
    fold = report.get("fold")
    blank = report.get("blank_first")
    horizons = report.get("sequence_horizon_contract")
    if (
        not isinstance(fold, Mapping)
        or fold.get("complete_target_parameter_overwrite") is not True
        or fold.get("parent_immutable") is not True
        or not isinstance(blank, Mapping)
        or blank.get("folded_states_identical") is not True
        or not isinstance(horizons, Mapping)
        or horizons.get("evaluation_rollout") is not None
        or horizons.get("evaluation_limit_can_cap_folding") is not False
        or horizons.get("limits_are_independently_named_and_bound")
        is not True
    ):
        raise Lfm2DynamicStructuralEvaluationError(
            "Fold report lacks complete overwrite/seed-erasure/horizon proof."
        )
    return report


def evaluate(
    *,
    checkpoint: Path,
    parent_tokenizer_directory: Path,
    parent_envelope: Path,
    expected_parent_envelope_sha256: str,
    output: Path,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Reload and evaluate one completed structural child without its parent."""

    checkpoint = checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_dir():
        raise Lfm2DynamicStructuralEvaluationError("Checkpoint must be a directory.")
    output = output.expanduser().absolute()
    if output.exists():
        raise Lfm2DynamicStructuralEvaluationError(
            f"Refusing to replace existing evaluation: {output}."
        )
    expected_envelope_file = _digest(
        expected_parent_envelope_sha256,
        "expected_parent_envelope_sha256",
    )
    if _file_sha256(parent_envelope) != expected_envelope_file:
        raise Lfm2DynamicStructuralEvaluationError(
            "Parent response envelope file fingerprint changed."
        )
    envelope = _object(parent_envelope, "parent response envelope")
    expected_cache_key = _digest(
        str(envelope.get("parent_cache_key", "")),
        "envelope parent_cache_key",
    )
    expected_parent_state = _digest(
        str(envelope.get("parent_state_fingerprint", "")),
        "envelope parent_state_fingerprint",
    )
    expected_tokenizer = _digest(
        str(envelope.get("tokenizer_fingerprint", "")),
        "envelope tokenizer_fingerprint",
    )
    generation = envelope.get("generation")
    if not isinstance(generation, Mapping):
        raise Lfm2DynamicStructuralEvaluationError(
            "Parent envelope lacks generation settings."
        )
    maximum_new_tokens = int(generation.get("max_new_tokens", 0))
    if maximum_new_tokens <= 0:
        raise Lfm2DynamicStructuralEvaluationError(
            "Parent envelope has an invalid generation limit."
        )

    report = _load_structural_report(checkpoint)
    expected_count = int(
        report["child"]["exact_unique_logical_parameters"]
    )
    expected_state = str(report["fold"]["folded_state_sha256"])
    child = structural.load_local_lfm2(
        checkpoint,
        device=torch.device(device),
        dtype=dtype,
    )
    child_count = structural.dense_logical_parameter_audit(
        child,
        require_any_nonzero=True,
    )["unique_logical_parameters"]
    child_state = structural.model_state_sha256(child)
    if child_count != expected_count or child_state != expected_state:
        raise Lfm2DynamicStructuralEvaluationError(
            "Freshly reloaded child differs from its sealed fold report."
        )

    tokenizer_fingerprint = fingerprint_local_tokenizer(
        parent_tokenizer_directory
    )
    if tokenizer_fingerprint != expected_tokenizer:
        raise Lfm2DynamicStructuralEvaluationError(
            "Local tokenizer differs from the sealed parent behavior envelope."
        )
    baseline = load_lfm2_cached_whole_output_baseline(
        parent_envelope,
        expected_sha256=expected_envelope_file,
        expected_parent_cache_key=expected_cache_key,
        expected_parent_state_fingerprint=expected_parent_state,
        expected_tokenizer_fingerprint=expected_tokenizer,
        expected_generation_limit=maximum_new_tokens,
        fragment_minimum_characters=12,
    )
    tokenizer = residual.load_local_tokenizer(parent_tokenizer_directory)
    evaluator = Lfm2WholeOutputBehaviorEvaluator(
        baseline=baseline,
        tokenizer=tokenizer,
        maximum_prompt_tokens=512,
        fragment_minimum_characters=12,
        maximum_new_empty_responses=2,
        maximum_new_fragment_responses=2,
        minimum_parent_nonempty_retention=0.90,
        minimum_task_score_retention=0.90,
        maximum_parent_task_score_regressions=2,
        maximum_new_generation_limit_saturations=2,
        maximum_repeated_ngram_share_regressions=2,
        maximum_repeated_bigram_share_delta=0.05,
    )
    decision = evaluator.evaluate_model(
        child,
        phase="dynamic_structural_fresh_reload_whole_output",
    )
    parent_count = int(
        report["parent"]["dense_parameter_audit"]["unique_logical_parameters"]
    )
    body = {
        "schema": SCHEMA,
        "status": "accepted" if decision.accepted else "rejected",
        "decision": decision.as_dict(),
        "events": evaluator.events,
        "checkpoint": str(checkpoint),
        "child_state_sha256": child_state,
        "child_unique_parameters": child_count,
        "parent_unique_parameters": parent_count,
        "parameters_removed": parent_count - child_count,
        "fraction_removed": 1.0 - child_count / parent_count,
        "fold_run_sha256": report["run_sha256"],
        "parent_envelope_file_sha256": expected_envelope_file,
        "parent_behavior_envelope_sha256": envelope.get("envelope_sha256"),
        "parent_forward_passes": 0,
        "parent_model_loaded": False,
        "candidate_forward_passes": len(baseline.cases),
        "evaluation_rollout": evaluation_rollout_contract(
            maximum_new_tokens
        ),
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "metric_interpretation": {
            "exact_text_equality": "diagnostic-only",
            "task_score_retention": "weak automatic diagnostic",
            "coherence_and_intelligence": "manual whole-output review required",
        },
    }
    document = {**body, "report_sha256": canonical_sha256(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def _dtype(value: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[value.strip().lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            "dtype must be float16, bfloat16, or float32"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--parent-tokenizer-directory", required=True, type=Path)
    parser.add_argument("--parent-envelope", required=True, type=Path)
    parser.add_argument("--expected-parent-envelope-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", type=_dtype, default=torch.bfloat16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate(
        checkpoint=args.checkpoint,
        parent_tokenizer_directory=args.parent_tokenizer_directory,
        parent_envelope=args.parent_envelope,
        expected_parent_envelope_sha256=(
            args.expected_parent_envelope_sha256
        ),
        output=args.output,
        device=args.device,
        dtype=args.dtype,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "fraction_removed": report["fraction_removed"],
                "report_sha256": report["report_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
