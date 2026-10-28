#!/usr/bin/env python3
"""Generate final-only whole-output review evidence for compact LFM2 children."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch


REMOTE_ROOT = Path("/home/cereal/model-surgery-lab")
SOURCE_ROOT = REMOTE_ROOT / "artifacts/lfm2_residual_attention_fold_20260716"
DYNAMIC_ROOT = REMOTE_ROOT / "artifacts/lfm2_dynamic_fold_20260716"
DEFAULT_OUTPUT = DYNAMIC_ROOT / "final-only-whole-output-review-20260716"
PROMPTS_PATH = SOURCE_ROOT / "heldout_parent_responses.json"
PARENT_RESPONSE_SOURCE = (
    SOURCE_ROOT / "max1-holistic/parent-review-responses.json"
)
PARENT_MODEL = REMOTE_ROOT / "models/LFM2.5-230M"
RUNNER_PATH = (
    REMOTE_ROOT
    / "neural-forge-src/experiments/run_lfm2_residual_attention_fold.py"
)
MAX_NEW_TOKENS = 80
LABELS = {
    "m2559": {
        "run": DYNAMIC_ROOT / "m2559_provisional_gpu0",
        "description": "LFM2 child with intermediate size 2559",
    },
    "m2558": {
        "run": DYNAMIC_ROOT / "m2558_provisional_gpu6",
        "description": "LFM2 child with intermediate size 2558",
    },
    "m2556": {
        "run": DYNAMIC_ROOT / "m2556_provisional_gpu0",
        "description": "LFM2 child with intermediate size 2556",
    },
    "m2552": {
        "run": DYNAMIC_ROOT / "m2552_provisional_gpu4",
        "description": "LFM2 child with intermediate size 2552",
    },
    "m2544": {
        "run": DYNAMIC_ROOT / "m2544_provisional_gpu6",
        "description": "LFM2 child with intermediate size 2544",
    },
    "m2536": {
        "run": DYNAMIC_ROOT / "m2536_provisional_gpu0",
        "description": "LFM2 child with intermediate size 2536",
    },
    "m2528": {
        "run": DYNAMIC_ROOT / "m2528_provisional_gpu4",
        "description": "LFM2 child with intermediate size 2528",
    },
    "m2496": {
        "run": DYNAMIC_ROOT / "m2496_provisional_gpu6",
        "description": "LFM2 child with intermediate size 2496",
    },
}


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location(
        "neural_forge_lfm2_review_runner",
        RUNNER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import review runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def response_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("Duplicate response ids.")
    return result


def prepare(output: Path, labels: tuple[str, ...]) -> None:
    runner = load_runner()
    if output.exists():
        raise RuntimeError(f"Review output already exists: {output}")
    if not labels or len(set(labels)) != len(labels):
        raise RuntimeError("Review labels must be nonempty and unique.")
    unknown = set(labels) - set(LABELS)
    if unknown:
        raise RuntimeError(f"Unknown review labels: {sorted(unknown)}")
    output.mkdir(parents=True)

    prompts = runner.load_review_prompts(PROMPTS_PATH)
    prompt_identity = runner.review_prompt_identity(prompts)
    source_parent = read_json(PARENT_RESPONSE_SOURCE)
    source_body = dict(source_parent)
    source_seal = source_body.pop("artifact_sha256", None)
    if source_seal != runner._json_sha256(source_body):
        raise RuntimeError("Existing parent whole-response cache seal is invalid.")
    if (
        source_body.get("review_prompts_sha256")
        != prompt_identity["review_prompts_sha256"]
    ):
        raise RuntimeError("Parent response cache does not match held-out prompts.")
    if int(source_body.get("max_new_tokens", -1)) != MAX_NEW_TOKENS:
        raise RuntimeError("Parent response cache generation length changed.")

    parent_fingerprint = str(source_body["parent_fingerprint"])
    children: dict[str, Any] = {}
    for label in labels:
        values = LABELS[label]
        run = Path(values["run"])
        report = read_json(run / "report.json")
        state = read_json(run / "run_state.json")
        if report.get("status") != "complete":
            raise RuntimeError(f"{label} run is not complete.")
        if state.get("parent_fingerprint") != parent_fingerprint:
            raise RuntimeError(f"{label} parent fingerprint does not match cache.")
        final_checkpoint = dict(report["final_checkpoint"])
        children[label] = {
            "description": values["description"],
            "run_directory": str(run),
            "report_path": str(run / "report.json"),
            "report_sha256": report["report_sha256"],
            "child_directory": final_checkpoint["path"],
            "child_state_sha256": final_checkpoint["state_sha256"],
            "unique_parameters": final_checkpoint["unique_parameters"],
            "final_gate": report["final_gate"],
        }

    shared_parent_path = output / "shared-parent-review-responses.json"
    shutil.copy2(PARENT_RESPONSE_SOURCE, shared_parent_path)
    validated = runner.load_or_build_parent_review_responses(
        shared_parent_path,
        None,
        None,
        prompts,
        parent_fingerprint=parent_fingerprint,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    if len(validated) != len(prompts):
        raise RuntimeError("Validated parent response count changed.")

    context = {
        "schema": "neural-forge-lfm2-final-whole-output-review-context-v1",
        "review_mode": "final-only-whole-output",
        "heldout_prompts_path": str(PROMPTS_PATH),
        "prompt_identity": prompt_identity,
        "parent_model_directory": str(PARENT_MODEL),
        "parent_fingerprint": parent_fingerprint,
        "shared_parent_response_source": str(PARENT_RESPONSE_SOURCE),
        "shared_parent_response_cache": str(shared_parent_path),
        "shared_parent_response_artifact_sha256": source_seal,
        "parent_responses_reused": True,
        "parent_response_forward_passes_this_review": 0,
        "parent_trace_forward_passes_this_review": 0,
        "parent_traces_regenerated": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "direct_local_io": runner.direct_local_io_audit(PARENT_MODEL),
        "children": children,
    }
    runner._atomic_write_json(output / "review-context.json", context)
    print(json.dumps(context, indent=2, sort_keys=True))


def render_candidate_markdown(
    *,
    label: str,
    context: dict[str, Any],
    prompt_rows: list[dict[str, Any]],
    parent_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> str:
    parent = response_map(parent_rows)
    candidate = response_map(candidate_rows)
    lines = [
        f"# Final whole-output evidence: {label}",
        "",
        "No human score or promotion decision is assigned here. The parent "
        "wording is context only; exact token matching is not the objective.",
        "",
        f"- Prompt identity: `{context['prompt_identity']['review_prompts_sha256']}`",
        f"- Child state: `{context['children'][label]['child_state_sha256']}`",
        f"- Max generated tokens: {context['max_new_tokens']}",
        "",
    ]
    for index, prompt in enumerate(prompt_rows, start=1):
        prompt_id = str(prompt["id"])
        parent_row = parent[prompt_id]
        candidate_row = candidate[prompt_id]
        lines.extend(
            [
                f"## {index:02d}. {prompt_id}",
                "",
                f"Category: `{prompt.get('category', 'unspecified')}`",
                "",
                "Prompt:",
                "",
                "```text",
                str(prompt["prompt"]),
                "```",
                "",
                "Expected response metadata:",
                "",
                "```text",
                str(prompt.get("expected_response", "")),
                "```",
                "",
                "Cached parent response (context only):",
                "",
                "```text",
                str(parent_row["response"]),
                "```",
                "",
                f"{label} child response:",
                "",
                "```text",
                str(candidate_row["response"]),
                "```",
                "",
                "Candidate mechanical diagnostics:",
                "",
                "```json",
                json.dumps(
                    candidate_row["diagnostics"],
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def generate(label: str, output: Path, device: str) -> None:
    runner = load_runner()
    context = read_json(output / "review-context.json")
    if label not in context["children"]:
        raise RuntimeError(f"Unknown child label: {label}")
    completion_path = output / f"{label}-completion.json"
    if completion_path.exists():
        raise RuntimeError(f"Review already completed for {label}.")

    prompts = runner.load_review_prompts(PROMPTS_PATH)
    prompt_identity = runner.review_prompt_identity(prompts)
    if prompt_identity != context["prompt_identity"]:
        raise RuntimeError("Held-out prompt identity changed after preparation.")
    parent_responses = runner.load_or_build_parent_review_responses(
        Path(context["shared_parent_response_cache"]),
        None,
        None,
        prompts,
        parent_fingerprint=context["parent_fingerprint"],
        max_new_tokens=int(context["max_new_tokens"]),
    )
    tokenizer = runner.load_local_tokenizer(PARENT_MODEL)

    child_context = context["children"][label]
    child = runner.load_local_lfm2_model(
        child_context["child_directory"],
        device=torch.device(device),
        dtype=torch.bfloat16,
    )
    state_sha256 = runner.model_state_sha256(child)
    if state_sha256 != child_context["child_state_sha256"]:
        raise RuntimeError(f"{label} loaded child state hash changed.")
    responses = runner.generate_independent_responses(
        child,
        tokenizer,
        prompts,
        max_new_tokens=int(context["max_new_tokens"]),
    )
    response_body = {
        "schema": "neural-forge-lfm2-final-child-review-responses-v1",
        "label": label,
        "child_directory": child_context["child_directory"],
        "child_state_sha256": state_sha256,
        "report_sha256": child_context["report_sha256"],
        "parent_fingerprint": context["parent_fingerprint"],
        "review_prompts_sha256": prompt_identity["review_prompts_sha256"],
        "max_new_tokens": context["max_new_tokens"],
        "responses": list(responses),
        "deterministic_greedy_generation": True,
        "exact_token_match_is_not_a_review_objective": True,
        "direct_local_model_loader_used": True,
        "direct_local_tokenizer_loader_used": True,
        "parent_forward_passes": 0,
        "parent_trace_regeneration": False,
    }
    response_document = {
        **response_body,
        "artifact_sha256": runner._json_sha256(response_body),
    }
    response_path = output / f"{label}-candidate-responses.json"
    runner._atomic_write_json(response_path, response_document)

    packet = runner.build_manual_review_packet(
        layer_index=14,
        candidate_state_sha256=state_sha256,
        parent_fingerprint=context["parent_fingerprint"],
        prompt_identity=prompt_identity,
        parent_responses=parent_responses,
        incumbent_responses=parent_responses,
        candidate_responses=responses,
        internal_diagnostics=child_context["final_gate"],
    )
    packet_path = output / f"{label}-final-{packet['packet_sha256']}.json"
    runner._atomic_write_json(packet_path, packet)

    prompt_rows = read_json(PROMPTS_PATH)
    markdown_path = output / f"{label}-whole-output-evidence.md"
    markdown_path.write_text(
        render_candidate_markdown(
            label=label,
            context=context,
            prompt_rows=prompt_rows,
            parent_rows=list(parent_responses),
            candidate_rows=list(responses),
        ),
        encoding="utf-8",
    )
    completion = {
        "schema": "neural-forge-lfm2-final-review-completion-v1",
        "label": label,
        "candidate_response_path": str(response_path),
        "candidate_response_file_sha256": file_sha256(response_path),
        "manual_review_packet_path": str(packet_path),
        "manual_review_packet_sha256": packet["packet_sha256"],
        "whole_output_evidence_path": str(markdown_path),
        "whole_output_evidence_file_sha256": file_sha256(markdown_path),
        "response_count": len(responses),
        "empty_response_count": sum(
            bool(row["diagnostics"]["empty"]) for row in responses
        ),
        "fragment_like_response_count": sum(
            bool(row["diagnostics"]["fragment_like"]) for row in responses
        ),
        "parent_forward_passes": 0,
        "parent_trace_regeneration": False,
        "manual_score_assigned": False,
    }
    runner._atomic_write_json(completion_path, completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


def finalize(output: Path) -> None:
    runner = load_runner()
    context = read_json(output / "review-context.json")
    prompt_rows = read_json(PROMPTS_PATH)
    parent_document = read_json(
        Path(context["shared_parent_response_cache"])
    )
    parent = response_map(parent_document["responses"])
    children: dict[str, dict[str, dict[str, Any]]] = {}
    completions = {}
    labels = tuple(context["children"])
    if not labels:
        raise RuntimeError("Review context contains no children.")
    for label in labels:
        completion = read_json(output / f"{label}-completion.json")
        candidate_document = read_json(
            Path(completion["candidate_response_path"])
        )
        children[label] = response_map(candidate_document["responses"])
        completions[label] = completion

    ordered_ids = [str(row["id"]) for row in prompt_rows]
    all_models_empty = [
        prompt_id
        for prompt_id in ordered_ids
        if not str(parent[prompt_id]["response"]).strip()
        and all(
            not str(children[label][prompt_id]["response"]).strip()
            for label in labels
        )
    ]
    child_divergence_ids = [
        prompt_id
        for prompt_id in ordered_ids
        if len(
            {
                str(children[label][prompt_id]["response"])
                for label in labels
            }
        )
        > 1
    ]
    math_rows = [
        row
        for row in prompt_rows
        if row.get("category") == "math"
        and row.get("expected_response") is not None
    ]

    def mechanical_summary(
        values: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        rows = [values[prompt_id] for prompt_id in ordered_ids]
        return {
            "response_count": len(rows),
            "empty_response_count": sum(
                bool(row["diagnostics"]["empty"]) for row in rows
            ),
            "fragment_like_response_count": sum(
                bool(row["diagnostics"]["fragment_like"]) for row in rows
            ),
            "mean_character_count": (
                sum(int(row["diagnostics"]["characters"]) for row in rows)
                / len(rows)
            ),
            "mean_generated_token_count": (
                sum(int(row["diagnostics"]["tokens"]) for row in rows)
                / len(rows)
            ),
        }

    def critical_regression_gate(label: str) -> dict[str, Any]:
        new_empty = [
            prompt_id
            for prompt_id in ordered_ids
            if str(parent[prompt_id]["response"]).strip()
            and not str(children[label][prompt_id]["response"]).strip()
        ]
        new_fragments = [
            prompt_id
            for prompt_id in ordered_ids
            if not bool(
                parent[prompt_id]["diagnostics"]["fragment_like"]
            )
            and bool(
                children[label][prompt_id]["diagnostics"][
                    "fragment_like"
                ]
            )
        ]
        return {
            "accepted": not new_empty and not new_fragments,
            "new_empty_response_ids": new_empty,
            "new_fragment_like_response_ids": new_fragments,
            "exact_text_equality_required": False,
        }

    objective_comparison = {
        "schema": "neural-forge-lfm2-objective-review-comparison-v1",
        "review_prompts_sha256": context["prompt_identity"][
            "review_prompts_sha256"
        ],
        "prompt_count": len(ordered_ids),
        "exact_text_equality_is_diagnostic_only": True,
        "manual_score_assigned": False,
        "pairwise_child_exact_response_counts": {
            f"{left}__{right}": sum(
                children[left][prompt_id]["response"]
                == children[right][prompt_id]["response"]
                for prompt_id in ordered_ids
            )
            for left_index, left in enumerate(labels)
            for right in labels[left_index + 1 :]
        },
        "child_equals_parent_response_counts": {
            label: sum(
                children[label][prompt_id]["response"]
                == parent[prompt_id]["response"]
                for prompt_id in ordered_ids
            )
            for label in labels
        },
        "child_divergence_ids": child_divergence_ids,
        "all_models_empty_response_ids": all_models_empty,
        "cached_parent": mechanical_summary(parent),
        "children": {
            label: mechanical_summary(children[label])
            for label in labels
        },
        "automatic_critical_regression_gate": {
            label: critical_regression_gate(label)
            for label in labels
        },
        "heldout_math_expected_text_present": {
            "prompt_count": len(math_rows),
            "parent_count": sum(
                str(row["expected_response"])
                in str(parent[str(row["id"])]["response"])
                for row in math_rows
            ),
            "child_counts": {
                label: sum(
                    str(row["expected_response"])
                    in str(
                        children[label][str(row["id"])]["response"]
                    )
                    for row in math_rows
                )
                for label in labels
            },
            "diagnostic_only": True,
        },
        "parent_forward_passes": 0,
        "parent_trace_regeneration": False,
    }
    objective_path = output / "objective-comparison.json"
    runner._atomic_write_json(objective_path, objective_comparison)

    categories_seen: set[str] = set()
    representative_ids = []
    for row in prompt_rows:
        category = str(row.get("category", "unspecified"))
        if category not in categories_seen:
            categories_seen.add(category)
            representative_ids.append(str(row["id"]))

    prompt_by_id = {str(row["id"]): row for row in prompt_rows}
    representatives = []
    for prompt_id in representative_ids:
        row = prompt_by_id[prompt_id]
        representatives.append(
            {
                "id": prompt_id,
                "category": row.get("category"),
                "prompt": row["prompt"],
                "expected_response": row.get("expected_response"),
                "parent_response_context_only": parent[prompt_id]["response"],
                "child_responses": {
                    label: children[label][prompt_id]["response"]
                    for label in labels
                },
                "selection_rule": "first held-out prompt in each category",
            }
        )
    representative_body = {
        "schema": "neural-forge-lfm2-representative-whole-output-evidence-v1",
        "selection_rule": "first held-out prompt in each category",
        "review_prompts_sha256": context["prompt_identity"][
            "review_prompts_sha256"
        ],
        "outputs": representatives,
        "manual_score_assigned": False,
    }
    representative_path = output / "representative-full-outputs.json"
    runner._atomic_write_json(representative_path, representative_body)

    lines = [
        "# Representative full outputs",
        "",
        "Selection is deterministic: the first held-out prompt in each "
        "category. No human score is assigned.",
        "",
    ]
    for value in representatives:
        lines.extend(
            [
                f"## {value['category']}: {value['id']}",
                "",
                "Prompt:",
                "",
                "```text",
                str(value["prompt"]),
                "```",
                "",
                f"Expected metadata: `{value['expected_response']}`",
                "",
                "Parent (context only):",
                "",
                "```text",
                str(value["parent_response_context_only"]),
                "```",
                "",
            ]
        )
        for label in labels:
            lines.extend(
                [
                    f"{label}:",
                    "",
                    "```text",
                    str(value["child_responses"][label]),
                    "```",
                    "",
                ]
            )
    representative_markdown = output / "representative-full-outputs.md"
    representative_markdown.write_text("\n".join(lines), encoding="utf-8")

    index = {
        "schema": "neural-forge-lfm2-final-review-index-v1",
        "review_directory": str(output),
        "context_path": str(output / "review-context.json"),
        "shared_parent_response_cache": context[
            "shared_parent_response_cache"
        ],
        "parent_responses_reused": True,
        "parent_forward_passes": 0,
        "parent_trace_regeneration": False,
        "heldout_prompt_count": context["prompt_identity"]["prompt_count"],
        "review_prompts_sha256": context["prompt_identity"][
            "review_prompts_sha256"
        ],
        "children": completions,
        "representative_full_outputs_path": str(representative_path),
        "representative_full_outputs_file_sha256": file_sha256(
            representative_path
        ),
        "representative_markdown_path": str(representative_markdown),
        "representative_markdown_file_sha256": file_sha256(
            representative_markdown
        ),
        "objective_comparison_path": str(objective_path),
        "objective_comparison_file_sha256": file_sha256(objective_path),
        "manual_score_assigned": False,
    }
    runner._atomic_write_json(output / "review-index.json", index)
    print(json.dumps(index, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument(
        "--label",
        action="append",
        choices=tuple(LABELS),
        dest="labels",
    )
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--label", choices=tuple(LABELS), required=True)
    generate_parser.add_argument("--device", default="cuda:0")
    sub.add_parser("finalize")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = args.output.expanduser().absolute()
    if args.command == "prepare":
        prepare(output, tuple(args.labels or LABELS))
    elif args.command == "generate":
        generate(args.label, output, args.device)
    elif args.command == "finalize":
        finalize(output)
    else:
        raise RuntimeError(f"Unhandled command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
