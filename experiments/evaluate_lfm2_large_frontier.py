"""Evaluate a completed local LFM2 fold against cached whole responses.

This runner never executes the parent and never trains the child.  It binds a
freshly loaded standalone child to its sealed fold report, then applies the
project's whole-output behavior gate.  The reported task-retention value is a
weak semantic/task diagnostic, not a claim of retained intelligence; empties,
fragments, generation-limit saturation, and repetition remain independent
promotion blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from experiments import run_lfm2_residual_attention_fold as residual
from experiments.run_lfm2_dynamic_catalog import fingerprint_local_tokenizer
from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_lfm2_behavior import (
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
)


class Lfm2LargeFrontierEvaluationError(RuntimeError):
    """Raised when authenticated frontier evaluation cannot proceed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise Lfm2LargeFrontierEvaluationError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return normalized


def _load_sealed_fold_report(path: Path) -> tuple[dict[str, Any], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report must be a JSON object."
        )
    body = dict(document)
    seal = body.pop("report_sha256", None)
    if not isinstance(seal, str) or canonical_sha256(body) != seal:
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report seal is missing or invalid."
        )
    if document.get("status") != "complete":
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report is not complete."
        )
    completed = document.get("completed_layers")
    if not isinstance(completed, list) or not completed:
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report has no completed layer ledger."
        )
    return document, seal


def _checkpoint_accounting(
    fold_document: dict[str, Any],
) -> tuple[Path, str, int, int]:
    checkpoint = fold_document.get("final_checkpoint")
    accounting = fold_document.get("accounting")
    if not isinstance(checkpoint, dict) or not isinstance(accounting, dict):
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report lacks checkpoint/accounting evidence."
        )
    try:
        child_path = Path(checkpoint["path"])
        child_state_sha256 = _digest(
            str(checkpoint["state_sha256"]),
            "final_checkpoint.state_sha256",
        )
        child_parameters = int(checkpoint["unique_parameters"])
        parent_parameters = int(accounting["observed_parent_unique_parameters"])
        target_parameters = int(accounting["observed_target_unique_parameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2LargeFrontierEvaluationError(
            "The fold checkpoint/accounting fields are malformed."
        ) from exc
    if child_parameters <= 0 or parent_parameters <= child_parameters:
        raise Lfm2LargeFrontierEvaluationError(
            "The fold report does not describe positive parameter removal."
        )
    if child_parameters != target_parameters:
        raise Lfm2LargeFrontierEvaluationError(
            "Checkpoint and accounting target parameter counts differ."
        )
    if not child_path.is_dir():
        raise Lfm2LargeFrontierEvaluationError(
            f"The final child checkpoint does not exist: {child_path}"
        )
    return (
        child_path,
        child_state_sha256,
        child_parameters,
        parent_parameters,
    )


def evaluate_frontier(
    *,
    parent_model: Path,
    evidence_root: Path,
    run_root: Path,
    envelope: Path,
    envelope_sha256: str,
    parent_cache_key: str,
    output: Path,
    device: str,
    dtype: str,
) -> dict[str, Any]:
    """Evaluate one completed fold while executing zero parent forwards."""

    envelope_sha256 = _digest(envelope_sha256, "envelope_sha256")
    parent_cache_key = _digest(parent_cache_key, "parent_cache_key")
    if output.exists():
        raise Lfm2LargeFrontierEvaluationError(
            f"Refusing to replace an existing evaluation: {output}"
        )
    if _file_sha256(envelope) != envelope_sha256:
        raise Lfm2LargeFrontierEvaluationError(
            "The sealed parent response envelope changed."
        )

    fold_report = run_root / "report.json"
    fold_document, fold_report_sha256 = _load_sealed_fold_report(fold_report)
    (
        child_path,
        expected_child_state,
        expected_child_parameters,
        parent_parameters,
    ) = _checkpoint_accounting(fold_document)

    manifest_path = (
        evidence_root
        / "circuit-cache"
        / "parent"
        / parent_cache_key
        / "manifest.json"
    )
    cache_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parent_identity = cache_manifest.get("parent", {}).get("identity_sha256")
    if not isinstance(parent_identity, str):
        raise Lfm2LargeFrontierEvaluationError(
            "The sealed parent cache lacks an authenticated parent identity."
        )

    tokenizer_fingerprint = fingerprint_local_tokenizer(parent_model)
    baseline = load_lfm2_cached_whole_output_baseline(
        envelope,
        expected_sha256=envelope_sha256,
        expected_parent_cache_key=parent_cache_key,
        expected_parent_state_fingerprint=parent_identity,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
        expected_generation_limit=64,
        fragment_minimum_characters=12,
    )
    tokenizer = residual.load_local_tokenizer(parent_model)
    torch_dtype = residual._torch_dtype(dtype)
    child = residual.load_local_lfm2_model(
        child_path,
        device=torch.device(device),
        dtype=torch_dtype,
    )
    child_state = residual.model_state_sha256(child)
    if child_state != expected_child_state:
        raise Lfm2LargeFrontierEvaluationError(
            "The freshly loaded child differs from the fold report."
        )
    child_parameters = residual.unique_parameter_count(child)
    if child_parameters != expected_child_parameters:
        raise Lfm2LargeFrontierEvaluationError(
            "The freshly loaded child has the wrong unique parameter count."
        )

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
        phase="large_ratio_fresh_reload_weak_semantic_gate",
    )
    parameters_removed = parent_parameters - child_parameters
    body = {
        "schema": "neural-forge-lfm2-large-frontier-whole-output-v1",
        "status": "accepted" if decision.accepted else "rejected",
        "decision": decision.as_dict(),
        "events": evaluator.events,
        "metric_interpretation": {
            "task_score_retention": (
                "weak semantic/task relevance diagnostic; not intelligence"
            ),
            "exact_text_equality": "diagnostic only",
            "manual_open_ended_review_required": True,
            "promotion_requires_all_independent_gates": True,
        },
        "fold_report": str(fold_report),
        "fold_report_sha256": fold_report_sha256,
        "child_path": str(child_path),
        "child_state_sha256": child_state,
        "child_unique_parameters": child_parameters,
        "parent_unique_parameters": parent_parameters,
        "parameters_removed": parameters_removed,
        "fraction_removed": parameters_removed / parent_parameters,
        "parent_cache_key": parent_cache_key,
        "baseline_fingerprint": baseline.fingerprint,
        "parent_forward_passes": 0,
        "training_steps": 0,
    }
    document = {**body, "report_sha256": canonical_sha256(body)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fresh-reload whole-output evaluation for a completed local-only "
            "LFM2 large-ratio fold."
        )
    )
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--envelope-sha256", required=True)
    parser.add_argument("--parent-cache-key", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    document = evaluate_frontier(
        parent_model=args.parent_model,
        evidence_root=args.evidence_root,
        run_root=args.run_root,
        envelope=args.envelope,
        envelope_sha256=args.envelope_sha256,
        parent_cache_key=args.parent_cache_key,
        output=args.output,
        device=args.device,
        dtype=args.dtype,
    )
    print(
        json.dumps(
            {
                "status": document["status"],
                "fraction_removed": document["fraction_removed"],
                "decision": document["decision"],
                "report_sha256": document["report_sha256"],
                "output": str(args.output),
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
