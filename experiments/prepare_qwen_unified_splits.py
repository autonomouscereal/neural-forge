#!/usr/bin/env python3
"""Create four deterministic, prompt-disjoint unified-surgery input splits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from neural_forge.hf_tools import _load_eval_suite


class UnifiedSplitError(ValueError):
    pass


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _prompt_from_calibration(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, Mapping):
        raise UnifiedSplitError("Calibration row has no prompt text.")
    messages = value.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        rendered = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                role = str(message.get("role") or "user").strip()
                rendered.append(f"{role}: {content.strip()}")
        if rendered:
            return "\n".join(rendered)
    input_json = value.get("input_json")
    if isinstance(input_json, Mapping):
        try:
            return _prompt_from_calibration(input_json)
        except UnifiedSplitError:
            pass
    for key in ("text", "prompt", "content"):
        item = value.get(key)
        if isinstance(item, str) and item.strip():
            return item.strip()
    raise UnifiedSplitError("Calibration row has no prompt text.")


def _load_calibration(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                result.append(_prompt_from_calibration(json.loads(line)))
            except (json.JSONDecodeError, UnifiedSplitError) as exc:
                raise UnifiedSplitError(
                    f"Invalid calibration row {path}:{line_number}."
                ) from exc
    unique = list(dict.fromkeys(result))
    if not unique:
        raise UnifiedSplitError("Calibration source is empty.")
    return unique


def _stable_order(values: Sequence[str], seed: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )


def _item_category(item: Mapping[str, Any]) -> str:
    return str(item.get("category") or item.get("kind") or "uncategorized")


def _grouped_balanced_items(
    values: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> list[list[dict[str, Any]]]:
    prompt_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in values:
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise UnifiedSplitError("Evaluation item has no nonempty prompt.")
        prompt_groups[prompt].append(dict(item))
    categories: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for prompt, items in prompt_groups.items():
        category = min(_item_category(item) for item in items)
        categories[category].append((prompt, items))
    queues: dict[str, deque[tuple[str, list[dict[str, Any]]]]] = {}
    for category, groups in categories.items():
        ordered = sorted(
            groups,
            key=lambda pair: (
                hashlib.sha256(
                    f"{seed}\0{category}\0{pair[0]}".encode("utf-8")
                ).hexdigest(),
                pair[0],
            ),
        )
        queues[category] = deque(ordered)
    result: list[list[dict[str, Any]]] = []
    while queues:
        for category in sorted(tuple(queues)):
            _prompt, items = queues[category].popleft()
            result.append(items)
            if not queues[category]:
                del queues[category]
    return result


def _take_distinct_prompt_groups(
    groups: Sequence[Sequence[dict[str, Any]]],
    count: int,
    *,
    forbidden: set[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    if count <= 0:
        raise UnifiedSplitError("Every requested evaluation count must be positive.")
    selected: list[dict[str, Any]] = []
    prompts: set[str] = set()
    for group in groups:
        prompt = str(group[0]["prompt"])
        if prompt in forbidden or prompt in prompts:
            continue
        remaining = count - len(selected)
        if remaining <= 0:
            break
        selected.extend(dict(item) for item in group[:remaining])
        prompts.add(prompt)
    if len(selected) != count:
        raise UnifiedSplitError(
            f"Could select only {len(selected)} of {count} requested prompt-disjoint rows."
        )
    return selected, prompts


def _take_texts(
    values: Sequence[str],
    count: int,
    *,
    seed: str,
    forbidden: set[str],
) -> tuple[list[str], set[str]]:
    selected = [
        value for value in _stable_order(list(dict.fromkeys(values)), seed) if value not in forbidden
    ][:count]
    if len(selected) != count:
        raise UnifiedSplitError(
            f"Could select only {len(selected)} of {count} requested text rows."
        )
    return selected, set(selected)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration_source = _load_calibration(args.calibration_source)
    ordered_calibration = _stable_order(calibration_source, args.seed)
    needed = args.calibration_count + args.sensitivity_gate_count
    if len(ordered_calibration) < needed:
        raise UnifiedSplitError(
            f"Need {needed} unique calibration prompts; found {len(ordered_calibration)}."
        )
    calibration = ordered_calibration[: args.calibration_count]
    sensitivity = ordered_calibration[args.calibration_count : needed]

    normalized = _load_eval_suite(args.eval_suite_source)
    forbidden = set(calibration) | set(sensitivity)
    candidate: dict[str, Any] = {}
    transaction: dict[str, Any] = {}
    candidate_prompts: set[str] = set()
    transaction_prompts: set[str] = set()

    candidate["text_evals"], prompts = _take_texts(
        normalized["text_evals"],
        args.candidate_text_count,
        seed=f"{args.seed}:candidate:text",
        forbidden=forbidden,
    )
    candidate_prompts.update(prompts)
    transaction["text_evals"], prompts = _take_texts(
        normalized["text_evals"],
        args.transaction_text_count,
        seed=f"{args.seed}:transaction:text",
        forbidden=forbidden | candidate_prompts,
    )
    transaction_prompts.update(prompts)

    specifications = (
        ("choice_probes", args.candidate_choice_count, args.transaction_choice_count),
        (
            "generation_prompts",
            args.candidate_generation_count,
            args.transaction_generation_count,
        ),
        (
            "behavior_probes",
            args.candidate_behavior_count,
            args.transaction_behavior_count,
        ),
    )
    for collection, candidate_count, transaction_count in specifications:
        groups = _grouped_balanced_items(
            normalized[collection], seed=f"{args.seed}:{collection}"
        )
        candidate[collection], prompts = _take_distinct_prompt_groups(
            groups,
            candidate_count,
            forbidden=forbidden | transaction_prompts,
        )
        candidate_prompts.update(prompts)
        transaction[collection], prompts = _take_distinct_prompt_groups(
            groups,
            transaction_count,
            forbidden=forbidden | candidate_prompts,
        )
        transaction_prompts.update(prompts)

    if candidate_prompts & transaction_prompts:
        raise UnifiedSplitError("Candidate and transaction prompt sets overlap.")
    if forbidden & (candidate_prompts | transaction_prompts):
        raise UnifiedSplitError("Calibration and evaluation prompt sets overlap.")

    calibration_path = output / "calibration.jsonl"
    sensitivity_path = output / "sensitivity_gate.jsonl"
    calibration_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in calibration),
        encoding="utf-8",
    )
    sensitivity_path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in sensitivity),
        encoding="utf-8",
    )
    candidate_path = output / "candidate_validation_suite.json"
    transaction_path = output / "transaction_gate_suite.json"
    _write_json(candidate_path, candidate)
    _write_json(transaction_path, transaction)

    files = (calibration_path, sensitivity_path, candidate_path, transaction_path)
    report = {
        "schema": "neural-forge-qwen3-unified-surgery-input-splits-v1",
        "seed": args.seed,
        "sources": {
            "calibration": str(args.calibration_source.resolve()),
            "evaluation": str(args.eval_suite_source.resolve()),
        },
        "counts": {
            "calibration": len(calibration),
            "sensitivity_gate": len(sensitivity),
            "candidate_validation": {
                key: len(value) for key, value in candidate.items()
            },
            "transaction_gate": {
                key: len(value) for key, value in transaction.items()
            },
        },
        "prompt_overlap_count": 0,
        "files": {
            path.name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        },
    }
    report["report_sha256"] = _sha256(report)
    _write_json(output / "split_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-source", type=Path, required=True)
    parser.add_argument("--eval-suite-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="unified-surgery-20260710")
    parser.add_argument("--calibration-count", type=int, default=8)
    parser.add_argument("--sensitivity-gate-count", type=int, default=8)
    parser.add_argument("--candidate-text-count", type=int, default=2)
    parser.add_argument("--candidate-choice-count", type=int, default=16)
    parser.add_argument("--candidate-generation-count", type=int, default=32)
    parser.add_argument("--candidate-behavior-count", type=int, default=32)
    parser.add_argument("--transaction-text-count", type=int, default=2)
    parser.add_argument("--transaction-choice-count", type=int, default=48)
    parser.add_argument("--transaction-generation-count", type=int, default=64)
    parser.add_argument("--transaction-behavior-count", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
