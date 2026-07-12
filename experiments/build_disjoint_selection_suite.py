#!/usr/bin/env python3
"""Build a deterministic eval suite with a token-disjoint prompt namespace."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from neural_forge.hf_tools import build_eval_suite


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _namespace_item(item: dict[str, Any], namespace: str, marker: str) -> dict[str, Any]:
    result = dict(item)
    if result.get("id") is not None:
        result["id"] = f"{namespace}:{result['id']}"
    result["prompt"] = marker + str(result["prompt"])
    return result


def build_selection_suite(
    output: Path,
    *,
    namespace: str,
    seed: int,
    text_count: int,
    choice_count: int,
    generation_count: int,
    behavior_count: int,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]+", namespace):
        raise ValueError("namespace must use only letters, digits, '_', '.', ':', or '-'.")
    marker = f"[NEURAL_FORGE_SPLIT:{namespace}]\n"
    build_eval_suite(
        output,
        behavior_count=behavior_count,
        generation_count=generation_count,
        choice_count=choice_count,
        text_count=text_count,
        seed=seed,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    source = payload["suite"]
    suite = {
        "text_evals": [marker + str(value) for value in source["text_evals"]],
        "choice_probes": [
            _namespace_item(dict(value), namespace, marker)
            for value in source["choice_probes"]
        ],
        "generation_prompts": [
            _namespace_item(dict(value), namespace, marker)
            for value in source["generation_prompts"]
        ],
        "behavior_probes": [
            _namespace_item(dict(value), namespace, marker)
            for value in source["behavior_probes"]
        ],
    }
    result = {
        "operation": "build_disjoint_selection_suite",
        "output_file": str(output),
        "namespace": namespace,
        "prompt_marker": marker,
        "seed": int(seed),
        "counts": {name: len(values) for name, values in suite.items()},
        "suite_sha256": _canonical_sha256(suite),
        "suite": suite,
    }
    _write_json(output, result)
    return {key: value for key, value in result.items() if key != "suite"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--text-count", type=int, default=32)
    parser.add_argument("--choice-count", type=int, default=128)
    parser.add_argument("--generation-count", type=int, default=256)
    parser.add_argument("--behavior-count", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_selection_suite(
        args.output,
        namespace=args.namespace,
        seed=args.seed,
        text_count=args.text_count,
        choice_count=args.choice_count,
        generation_count=args.generation_count,
        behavior_count=args.behavior_count,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
