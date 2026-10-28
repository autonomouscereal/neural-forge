from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from neural_forge.hf_tools import _generation_health, evaluate_loaded_text_model

import experiments.run_qwen_compensated_neuron_fold as runner


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected an object in {path}.")
    return value


def _bind_cached_certification(report: dict, identity: dict) -> dict:
    result = dict(report)
    generations = list(result.get("generations") or ())
    diagnostics = [
        _generation_health(
            item,
            runner.GENERATION_HEALTH_CONTRACT["min_continuation_chars"],
            runner.GENERATION_HEALTH_CONTRACT["min_unique_token_ratio"],
        )
        for item in generations
    ]
    correct = sum(1 for item in diagnostics if item["passed"])
    result["generation_diagnostics"] = diagnostics
    result["generation_health"] = {
        "correct": correct,
        "total": len(diagnostics),
        "pass_rate": correct / max(1, len(diagnostics)),
        "contract": dict(runner.GENERATION_HEALTH_CONTRACT),
    }
    result["identity_binding"] = dict(identity)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--certification-suite", type=Path, required=True)
    parser.add_argument("--selection-suite", type=Path, required=True)
    parser.add_argument("--existing-certification", type=Path, required=True)
    parser.add_argument("--bound-certification", type=Path, required=True)
    parser.add_argument("--bound-selection", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    load_args = SimpleNamespace(
        source_model=args.model,
        source_tokenizer=args.model,
        source_revision=args.revision,
        trust_remote_code=False,
        local_files_only=True,
        dtype="float16",
        device=args.device,
    )
    model, tokenizer, tokenizer_name = runner._load_source_runtime(load_args)
    certification_identity = runner.build_exact_parent_identity(
        model,
        tokenizer,
        source_model=args.model,
        source_revision=args.revision,
        source_tokenizer=tokenizer_name,
        eval_suite=args.certification_suite,
    )
    selection_identity = runner.derive_parent_identity_for_eval_suite(
        certification_identity, args.selection_suite
    )

    certification = _bind_cached_certification(
        _load_json(args.existing_certification), certification_identity
    )
    runner._write_json(args.bound_certification, certification)
    runner.validate_bound_parent_report(
        certification,
        certification_identity,
        label="Offline-bound certification parent",
    )

    selection = evaluate_loaded_text_model(
        model,
        tokenizer,
        args.bound_selection,
        model_label=args.model,
        device=args.device,
        max_new_tokens=96,
        eval_suite=args.selection_suite,
        max_text_evals=32,
        max_choice_probes=128,
        max_generation_prompts=256,
        max_behavior_probes=256,
        min_continuation_chars=runner.GENERATION_HEALTH_CONTRACT[
            "min_continuation_chars"
        ],
        min_unique_token_ratio=runner.GENERATION_HEALTH_CONTRACT[
            "min_unique_token_ratio"
        ],
        identity_binding=selection_identity,
    )
    runner.validate_bound_parent_report(
        selection,
        selection_identity,
        label="Prepared selection parent",
    )
    runner._write_json(
        args.report,
        {
            "schema": "neural-forge-reused-parent-cache-binding-v1",
            "existing_certification": str(args.existing_certification),
            "bound_certification": str(args.bound_certification),
            "bound_certification_sha256": runner._file_sha256(args.bound_certification),
            "bound_selection": str(args.bound_selection),
            "bound_selection_sha256": runner._file_sha256(args.bound_selection),
            "certification_identity_sha256": certification_identity["identity_sha256"],
            "selection_identity_sha256": selection_identity["identity_sha256"],
            "certification_parent_forward_passes": 0,
            "selection_parent_evaluations": 1,
            "strict_certification_generation_rescored_offline": len(
                certification.get("generations") or ()
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
