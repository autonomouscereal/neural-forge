"""Run an lm-eval benchmark slice for several model artifacts.

This is intentionally a thin subprocess wrapper around EleutherAI
lm-evaluation-harness. It keeps Neural Forge benchmark comparisons repeatable
without adding lm-eval as a required package dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
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
        "--cache-dir",
        help=(
            "Optional directory for reusable per-model lm-eval results. "
            "The cache key includes model, task set, limit, batch size, dtype, and command."
        ),
    )
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cached benchmark results.")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable LABEL=MODEL_OR_PATH entry.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        (label, _resolve_model_reference(model))
        for label, model in [_parse_model_entry(item) for item in args.model]
    ]
    summary: dict[str, Any] = {
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "cache_dir": args.cache_dir,
        "models": [],
    }
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for label, model in entries:
        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _cache_key(args, model)
        run_dir = cache_dir / cache_key if cache_dir is not None else model_dir
        cache_meta = run_dir / "neural_forge_lm_eval_cache.json"
        cached_results = {} if args.force_rerun else _cached_results(run_dir, cache_meta, cache_key)
        if cached_results:
            summary["models"].append(
                _summary_entry(
                    label=label,
                    model=model,
                    run_dir=run_dir,
                    results=cached_results,
                    cache_key=cache_key,
                    cache_hit=True,
                    cache_dir=cache_dir,
                )
            )
            _write_json(output_dir / "summary.json", summary)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            *shlex.split(args.lm_eval_command),
            "--model",
            "hf",
            "--model_args",
            f"pretrained={model},dtype={args.dtype}",
            "--device",
            args.device,
            "--tasks",
            args.tasks,
            "--batch_size",
            args.batch_size,
            "--output_path",
            str(run_dir),
        ]
        if args.limit > 0:
            command.extend(["--limit", str(args.limit)])
        result = subprocess.run(command, text=True, capture_output=True)
        (run_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            summary["models"].append(
                {
                    "label": label,
                    "model": model,
                    "returncode": result.returncode,
                    "results": {},
                    "cache_key": cache_key if cache_dir is not None else None,
                    "cache_hit": False,
                    "cache_dir": str(cache_dir) if cache_dir is not None else None,
                    "error": str(run_dir / "stderr.txt"),
                }
            )
            _write_json(output_dir / "summary.json", summary)
            return result.returncode
        metrics = _extract_lm_eval_metrics(run_dir)
        if cache_dir is not None:
            _write_json(
                cache_meta,
                {
                    "cache_key": cache_key,
                    "model": model,
                    "tasks": args.tasks,
                    "limit": args.limit,
                    "batch_size": args.batch_size,
                    "dtype": args.dtype,
                    "lm_eval_command": args.lm_eval_command,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "results": metrics,
                },
            )
        summary["models"].append(
            _summary_entry(
                label=label,
                model=model,
                run_dir=run_dir,
                results=metrics,
                cache_key=cache_key,
                cache_hit=False,
                cache_dir=cache_dir,
            )
        )
        _write_json(output_dir / "summary.json", summary)

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


def _resolve_model_reference(model: str) -> str:
    path = Path(model)
    if not path.exists():
        return model
    if _has_hf_model_config(path):
        return str(path)
    summary = path / "reports" / "chain_anchor_summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return model
        for key in ("final_model_dir", "best_model_dir"):
            value = data.get(key)
            if not isinstance(value, str) or not value:
                continue
            candidate = Path(value)
            if _has_hf_model_config(candidate):
                return str(candidate)
    return model


def _has_hf_model_config(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.exists():
        return False
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and bool(data.get("model_type"))


def _cache_key(args: argparse.Namespace, model: str) -> str:
    payload = {
        "model": model,
        "tasks": args.tasks,
        "limit": args.limit,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "lm_eval_command": args.lm_eval_command,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _cached_results(run_dir: Path, metadata_path: Path, cache_key: str) -> dict[str, dict[str, float]]:
    if not metadata_path.exists():
        return {}
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if metadata.get("cache_key") != cache_key:
        return {}
    results = _extract_lm_eval_metrics(run_dir)
    return results


def _summary_entry(
    *,
    label: str,
    model: str,
    run_dir: Path,
    results: dict[str, dict[str, float]],
    cache_key: str,
    cache_hit: bool,
    cache_dir: Path | None,
) -> dict[str, Any]:
    return {
        "label": label,
        "model": model,
        "returncode": 0,
        "results": results,
        "stdout": str(run_dir / "stdout.txt"),
        "stderr": str(run_dir / "stderr.txt"),
        "cache_key": cache_key if cache_dir is not None else None,
        "cache_hit": bool(cache_hit) if cache_dir is not None else False,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "output_path": str(run_dir),
    }


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
