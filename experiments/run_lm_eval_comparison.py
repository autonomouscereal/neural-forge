"""Run an lm-eval benchmark slice for several model artifacts.

This is intentionally a thin subprocess wrapper around EleutherAI
lm-evaluation-harness. It keeps Neural Forge benchmark comparisons repeatable
without adding lm-eval as a required package dependency.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default="arc_easy,hellaswag,piqa,winogrande")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lm-eval-command", default="python -m lm_eval")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable LABEL=MODEL_OR_PATH entry.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [_parse_model_entry(item) for item in args.model]
    summary: dict[str, Any] = {
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "models": [],
    }

    for label, model in entries:
        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *shlex.split(args.lm_eval_command),
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype={args.dtype},device={args.device}",
            "--tasks",
            args.tasks,
            "--batch_size",
            args.batch_size,
            "--output_path",
            str(model_dir),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        result = subprocess.run(command, text=True, capture_output=True)
        (model_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (model_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            summary["models"].append(
                {
                    "label": label,
                    "model": model,
                    "returncode": result.returncode,
                    "results": {},
                    "error": str(model_dir / "stderr.txt"),
                }
            )
            _write_json(output_dir / "summary.json", summary)
            return result.returncode
        summary["models"].append(
            {
                "label": label,
                "model": model,
                "returncode": result.returncode,
                "results": _extract_lm_eval_metrics(model_dir),
                "stdout": str(model_dir / "stdout.txt"),
                "stderr": str(model_dir / "stderr.txt"),
            }
        )

    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_model_entry(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise ValueError("--model entries must use LABEL=MODEL_OR_PATH.")
    label, model = text.split("=", 1)
    label = label.strip()
    model = model.strip()
    if not label or not model:
        raise ValueError("--model label and value must both be non-empty.")
    return label, model


def _extract_lm_eval_metrics(output_dir: Path) -> dict[str, dict[str, float]]:
    for path in sorted(output_dir.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        results = data.get("results")
        if not isinstance(results, dict):
            continue
        extracted: dict[str, dict[str, float]] = {}
        for task, values in results.items():
            if not isinstance(values, dict):
                continue
            extracted[str(task)] = {
                key: float(value)
                for key, value in values.items()
                if isinstance(value, (int, float)) and not key.endswith("_stderr")
            }
        if extracted:
            return extracted
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
