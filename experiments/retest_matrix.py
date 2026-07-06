#!/usr/bin/env python3
"""Build, queue, and summarize Neural Forge retest matrices.

The matrix intentionally treats exact behavior, generation health, choice
accuracy, retention, and manual samples as first-class evidence. Perplexity is
kept as supporting signal only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RUN_ROOT = Path(
    "/home/cereal/model-surgery-lab/neural-forge-src/runs/"
    "qwen4b_to_smollm135m_transport_random_20260627"
)
DEFAULT_REPO_ROOT = Path("/home/cereal/model-surgery-lab/neural-forge-src")
DEFAULT_GPUS = (
    "GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561",
    "GPU-2ce6a95c-7b1f-644a-3808-d781732a0582",
    "GPU-f5a09535-fc85-a027-50f7-734b344747dd",
)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
SKIP_PATH_PARTS = {
    "evals",
    "logs",
    "lm_eval_cache_20260702",
    "multi_donor_stages",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Scan a run root and write a candidate manifest.")
    build.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    build.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--include-layer-checkpoints", action="store_true")
    build.add_argument("--max-candidates", type=int, default=0)

    queue = sub.add_parser("queue", help="Queue exact-gen/manual/retention retests from a manifest.")
    queue.add_argument("--manifest", type=Path, required=True)
    queue.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    queue.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    queue.add_argument("--eval-dir", type=Path)
    queue.add_argument("--queue-state", type=Path)
    queue.add_argument("--gpu-uuid", action="append", dest="gpu_uuids")
    queue.add_argument("--baseline-wait-pids", default="")
    queue.add_argument("--limit", type=int, default=0)
    queue.add_argument("--category", action="append", dest="categories")
    queue.add_argument("--dry-run", action="store_true")

    summarize = sub.add_parser("summarize", help="Summarize queued retest reports.")
    summarize.add_argument("--manifest", type=Path)
    summarize.add_argument("--queue-state", type=Path)
    summarize.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    summarize.add_argument("--eval-dir", type=Path)
    summarize.add_argument("--out-json", type=Path, required=True)
    summarize.add_argument("--out-md", type=Path)

    args = parser.parse_args()
    if args.command == "build":
        return _cmd_build(args)
    if args.command == "queue":
        return _cmd_queue(args)
    if args.command == "summarize":
        return _cmd_summarize(args)
    raise AssertionError(args.command)


def _cmd_build(args: argparse.Namespace) -> int:
    manifest = build_manifest(
        run_root=args.run_root,
        repo_root=args.repo_root,
        include_layer_checkpoints=args.include_layer_checkpoints,
        max_candidates=args.max_candidates,
    )
    _write_json(args.out, manifest)
    print(json.dumps({"manifest": str(args.out), "candidates": len(manifest["candidates"])}, indent=2))
    return 0


def _cmd_queue(args: argparse.Namespace) -> int:
    manifest = _load_json(args.manifest)
    run_root = args.run_root
    repo_root = args.repo_root
    eval_dir = args.eval_dir or run_root / "evals" / f"retest_matrix_{_stamp()}"
    queue_state = args.queue_state or eval_dir / "queue_state.json"
    gpu_uuids = tuple(args.gpu_uuids or DEFAULT_GPUS)
    category_filter = set(args.categories or [])
    candidates = list(manifest.get("candidates", []))
    if category_filter:
        candidates = [c for c in candidates if c.get("category") in category_filter]
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]
    eval_dir.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)

    queued: list[dict[str, Any]] = []
    last_pid_by_gpu: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        label = _safe_label(str(candidate["label"]))
        gpu_uuid = gpu_uuids[index % len(gpu_uuids)]
        queue_pid = last_pid_by_gpu.get(gpu_uuid, 0)
        report = eval_dir / f"eval_{label}_exactgen_20260702_1000x1000x250.json"
        manual = eval_dir / f"manual_{label}_exactgen_20260702.json"
        out_dir = eval_dir / f"retention_{label}_20260702"
        log = run_root / "logs" / f"retest_matrix_{label}_{_stamp()}.log"
        parent_candidates = _parent_report_candidates(candidate, run_root)
        baseline_reports = _baseline_reports(run_root)
        env = os.environ.copy()
        env.update(
            {
                "REPO_ROOT": str(repo_root),
                "RUN_ROOT": str(run_root),
                "EVAL_DIR": str(eval_dir),
                "TAG": f"retest_{label}",
                "MODEL_DIR": str(candidate["model_dir"]),
                "QUEUE_PID": str(queue_pid),
                "GPU_UUID": gpu_uuid,
                "REPORT": str(report),
                "MANUAL_JSON": str(manual),
                "OUT_DIR": str(out_dir),
                "PARENT_REPORT_CANDIDATES": parent_candidates,
                "BASELINE_REPORTS": baseline_reports,
                "PARENT_WAIT_PIDS": args.baseline_wait_pids,
                "MAX_BEHAVIOR_PROBES": "1000",
                "MAX_GENERATION_PROMPTS": "1000",
                "MAX_CHOICE_PROBES": "250",
                "MAX_TEXT_EVALS": "128",
                "MAX_NEW_TOKENS": "96",
            }
        )
        entry = {
            "label": label,
            "source_label": candidate["label"],
            "category": candidate.get("category"),
            "model_dir": candidate["model_dir"],
            "gpu_uuid": gpu_uuid,
            "wait_queue_pid": queue_pid,
            "report": str(report),
            "manual_json": str(manual),
            "retention_dir": str(out_dir),
            "log": str(log),
            "parent_report_candidates": parent_candidates,
            "baseline_reports": baseline_reports,
        }
        if args.dry_run:
            entry["pid"] = None
        else:
            with log.open("w", encoding="utf-8") as handle:
                proc = subprocess.Popen(
                    ["bash", "experiments/queue_eval_retention_compare.sh"],
                    cwd=repo_root,
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            entry["pid"] = proc.pid
            last_pid_by_gpu[gpu_uuid] = int(proc.pid)
        queued.append(entry)

    state = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "eval_dir": str(eval_dir),
        "baseline_wait_pids": args.baseline_wait_pids,
        "dry_run": args.dry_run,
        "queued": queued,
    }
    _write_json(queue_state, state)
    print(json.dumps({"queue_state": str(queue_state), "queued": len(queued), "dry_run": args.dry_run}, indent=2))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    if args.queue_state:
        state = _load_json(args.queue_state)
        rows = _rows_from_queue_state(state)
    else:
        if not args.manifest:
            raise SystemExit("--manifest is required when --queue-state is not provided")
        manifest = _load_json(args.manifest)
        eval_dir = args.eval_dir or args.run_root / "evals"
        rows = _rows_from_manifest(manifest, eval_dir)
    rows.sort(key=_rank_key)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "complete_count": sum(1 for row in rows if row.get("status") == "complete"),
        "rows": rows,
    }
    _write_json(args.out_json, summary)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "rows": len(rows)}, indent=2))
    return 0


def build_manifest(
    *,
    run_root: Path,
    repo_root: Path,
    include_layer_checkpoints: bool,
    max_candidates: int,
) -> dict[str, Any]:
    eval_refs = _eval_model_refs(run_root)
    ref_paths = {Path(ref["model_dir"]) for ref in eval_refs if ref.get("is_local_path")}
    candidates: dict[Path, dict[str, Any]] = {}
    for config in sorted(run_root.rglob("config.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        model_dir = config.parent
        if _skip_model_dir(model_dir, run_root, include_layer_checkpoints, ref_paths):
            continue
        if not _has_tokenizer_files(model_dir):
            continue
        label = _candidate_label(model_dir, run_root)
        candidates[model_dir] = {
            "label": label,
            "model_dir": str(model_dir),
            "category": _candidate_category(model_dir),
            "param_count": _config_param_hint(config),
            "mtime": model_dir.stat().st_mtime,
            "existing_eval_refs": [],
        }
    for ref in eval_refs:
        if not ref.get("is_local_path"):
            continue
        model_dir = Path(ref["model_dir"])
        if not model_dir.exists() or not (model_dir / "config.json").exists():
            continue
        if _skip_model_dir(model_dir, run_root, include_layer_checkpoints, ref_paths):
            continue
        record = candidates.setdefault(
            model_dir,
            {
                "label": _candidate_label(model_dir, run_root),
                "model_dir": str(model_dir),
                "category": _candidate_category(model_dir),
                "param_count": None,
                "mtime": model_dir.stat().st_mtime,
                "existing_eval_refs": [],
            },
        )
        record["existing_eval_refs"].append(ref)
    ordered = sorted(candidates.values(), key=lambda item: (item["category"] != "recovered", -item["mtime"]))
    if max_candidates and max_candidates > 0:
        ordered = ordered[:max_candidates]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "eval_suite": str(run_root / "behavior_suite_exactgen_20260702.json"),
        "baselines": _baseline_manifest(run_root),
        "candidates": ordered,
    }


def _eval_model_refs(run_root: Path) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for path in run_root.rglob("*.json"):
        if any(part.startswith("lm_eval_cache") for part in path.parts):
            continue
        try:
            data = _load_json(path)
        except Exception:
            continue
        if not isinstance(data, dict) or "model" not in data:
            continue
        model = data.get("model")
        if not isinstance(model, str) or not model:
            continue
        refs.append(
            {
                "eval_report": str(path),
                "model_dir": model,
                "is_local_path": model.startswith("/"),
                "behavior_correct": (data.get("behavior_suite") or {}).get("correct"),
                "behavior_total": (data.get("behavior_suite") or {}).get("total"),
                "generation_correct": (data.get("generation_health") or {}).get("correct"),
                "generation_total": (data.get("generation_health") or {}).get("total"),
                "choice_correct": (data.get("choice_accuracy") or {}).get("correct"),
                "choice_total": (data.get("choice_accuracy") or {}).get("total"),
            }
        )
    return refs


def _skip_model_dir(model_dir: Path, run_root: Path, include_layer_checkpoints: bool, ref_paths: set[Path]) -> bool:
    try:
        rel_parts = model_dir.relative_to(run_root).parts
    except ValueError:
        return True
    if any(part in SKIP_PATH_PARTS for part in rel_parts):
        return True
    if "layer_checkpoints" in rel_parts and not include_layer_checkpoints and model_dir not in ref_paths:
        return True
    if "multi_donor_stages" in rel_parts:
        return True
    return False


def _has_tokenizer_files(model_dir: Path) -> bool:
    return any((model_dir / name).exists() for name in TOKENIZER_FILES)


def _candidate_label(model_dir: Path, run_root: Path) -> str:
    try:
        rel = model_dir.relative_to(run_root)
    except ValueError:
        return _safe_label(model_dir.name)
    parts = rel.parts
    if len(parts) >= 3 and parts[-2] == "models":
        return _safe_label(f"{parts[0]}__{parts[-1]}")
    if len(parts) >= 3 and parts[-2] == "layer_checkpoints":
        return _safe_label(f"{parts[0]}__{parts[-1]}")
    return _safe_label("__".join(parts))


def _candidate_category(model_dir: Path) -> str:
    parts = model_dir.parts
    if "01_init" in parts:
        return "pure_fold"
    if "01_block" in parts:
        return "block_recovery"
    if "02_alm" in parts or "03_alm" in parts:
        return "recovered"
    if model_dir.name == "text_model":
        return "vision_text_model"
    if (model_dir / "neural_forge_recovery_report.json").exists() or (
        model_dir / "neural_forge_xfamily_recovery_report.json"
    ).exists():
        return "recovered"
    if (model_dir / "neural_forge_donor_projected_init_report.json").exists():
        return "pure_fold"
    return "candidate"


def _config_param_hint(config: Path) -> int | None:
    try:
        data = _load_json(config)
    except Exception:
        return None
    hidden = data.get("hidden_size") or data.get("n_embd")
    layers = data.get("num_hidden_layers") or data.get("n_layer")
    vocab = data.get("vocab_size")
    if not all(isinstance(item, int) for item in (hidden, layers, vocab)):
        return None
    return int(vocab * hidden + layers * hidden * hidden * 12)


def _baseline_manifest(run_root: Path) -> list[dict[str, str]]:
    return [
        {
            "label": "qwen_parent_strict",
            "model": "Qwen/Qwen3-4B-Base",
            "report": str(run_root / "evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json"),
        },
        {
            "label": "qwen_parent_exactgen_legacy",
            "model": "Qwen/Qwen3-4B-Base",
            "report": str(run_root / "evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json"),
        },
        {
            "label": "qwen_parent_legacy",
            "model": "Qwen/Qwen3-4B-Base",
            "report": str(run_root / "evals/eval_parent_qwen3_4b_base_1000g_20260701_1000x1000x250.json"),
        },
        {
            "label": "gpt2_donor_strict",
            "model": "openai-community/gpt2",
            "report": str(run_root / "evals/eval_donor_gpt2_strict_20260705_1000x1000x250.json"),
        },
        {
            "label": "gpt2_donor_exactgen_legacy",
            "model": "openai-community/gpt2",
            "report": str(run_root / "evals/eval_donor_gpt2_exactgen_20260702_1000x1000x250.json"),
        },
        {
            "label": "lfm230m_donor_strict",
            "model": "LiquidAI/LFM2.5-230M",
            "report": str(run_root / "evals/eval_donor_lfm230m_strict_20260705_1000x1000x250.json"),
        },
        {
            "label": "lfm230m_donor_exactgen_legacy",
            "model": "LiquidAI/LFM2.5-230M",
            "report": str(run_root / "evals/eval_donor_lfm230m_exactgen_20260702_1000x1000x250.json"),
        },
        {
            "label": "blank_smollm_strict",
            "model": str(run_root / "blank_smollm2_135m_seed0"),
            "report": str(run_root / "evals/eval_blank_smollm135m_random_strict_20260705_1000x1000x250.json"),
        },
        {
            "label": "blank_smollm_exactgen_legacy",
            "model": str(run_root / "blank_smollm2_135m_seed0"),
            "report": str(run_root / "evals/eval_blank_smollm135m_random_exactgen_20260702_1000x1000x250.json"),
        },
        {
            "label": "blank_smollm_legacy",
            "model": str(run_root / "blank_smollm2_135m_seed0"),
            "report": str(run_root / "evals/eval_blank_smollm135m_random_1000g_20260701_1000x1000x250.json"),
        },
    ]


def _parent_report_candidates(candidate: dict[str, Any], run_root: Path) -> str:
    label = str(candidate.get("label", "")).lower()
    if "lfm" in label and "qwen4b_to_lfm" not in label:
        paths = [
            run_root / "evals/eval_donor_lfm230m_strict_20260705_1000x1000x250.json",
            run_root / "evals/eval_donor_lfm230m_exactgen_20260702_1000x1000x250.json",
        ]
    else:
        paths = [
            run_root / "evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json",
            run_root / "evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json",
            run_root / "evals/eval_parent_qwen3_4b_base_1000g_20260701_1000x1000x250.json",
        ]
    return ";".join(str(path) for path in paths)


def _baseline_reports(run_root: Path) -> str:
    entries = {
        "qwen_parent_strict": run_root / "evals/eval_parent_qwen3_4b_base_strict_20260705_1000x1000x250.json",
        "qwen_parent_exactgen_legacy": run_root
        / "evals/eval_parent_qwen3_4b_base_exactgen_20260702_1000x1000x250.json",
        "qwen_parent_legacy": run_root / "evals/eval_parent_qwen3_4b_base_1000g_20260701_1000x1000x250.json",
        "gpt2_donor_strict": run_root / "evals/eval_donor_gpt2_strict_20260705_1000x1000x250.json",
        "gpt2_donor_legacy": run_root / "evals/eval_donor_gpt2_exactgen_20260702_1000x1000x250.json",
        "lfm230m_donor_strict": run_root / "evals/eval_donor_lfm230m_strict_20260705_1000x1000x250.json",
        "lfm230m_donor_legacy": run_root / "evals/eval_donor_lfm230m_exactgen_20260702_1000x1000x250.json",
        "blank_random_strict": run_root / "evals/eval_blank_smollm135m_random_strict_20260705_1000x1000x250.json",
        "blank_random_exactgen_legacy": run_root
        / "evals/eval_blank_smollm135m_random_exactgen_20260702_1000x1000x250.json",
        "blank_random_legacy": run_root / "evals/eval_blank_smollm135m_random_1000g_20260701_1000x1000x250.json",
        "cleanopen_full": run_root / "evals/eval_cleanopen_full_02_alm_exactgen_20260702_1000x1000x250.json",
        "mathdrill_full": run_root / "evals/eval_mathdrill_eora_geom_full_fix_best_20260701_1000x1000x250.json",
        "topneg_margin": run_root
        / "evals/eval_qwen4b_cleanopen_full02_topneg_margin_recover_80s_nogate_20260702_exactgen_1000x1000x250.json",
    }
    return ";".join(f"{key}={value}" for key, value in entries.items())


def _rows_from_queue_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [_row_from_entry(entry) for entry in state.get("queued", [])]


def _rows_from_manifest(manifest: dict[str, Any], eval_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for candidate in manifest.get("candidates", []):
        label = _safe_label(str(candidate["label"]))
        rows.append(
            _row_from_entry(
                {
                    "label": label,
                    "source_label": candidate["label"],
                    "category": candidate.get("category"),
                    "model_dir": candidate["model_dir"],
                    "report": str(eval_dir / f"eval_{label}_exactgen_20260702_1000x1000x250.json"),
                    "manual_json": str(eval_dir / f"manual_{label}_exactgen_20260702.json"),
                    "retention_dir": str(eval_dir / f"retention_{label}_20260702"),
                }
            )
        )
    return rows


def _row_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    report_path = Path(entry["report"])
    row = {
        "label": entry.get("label"),
        "source_label": entry.get("source_label", entry.get("label")),
        "category": entry.get("category"),
        "model_dir": entry.get("model_dir"),
        "report": str(report_path),
        "manual_json": entry.get("manual_json"),
        "retention_dir": entry.get("retention_dir"),
        "status": "missing",
    }
    if report_path.exists() and report_path.stat().st_size > 0:
        try:
            data = _load_json(report_path)
        except Exception as exc:
            row["status"] = "invalid"
            row["error"] = str(exc)
            return row
        row.update(_aggregate_eval(data))
        row["status"] = "complete"
        row["behavior_by_category"] = _category_counts((data.get("behavior_suite") or {}).get("items"), "passed")
        row["generation_by_category"] = _category_counts(data.get("generation_diagnostics"), "passed")
        row["manual_samples"] = _manual_samples(Path(str(entry.get("manual_json", ""))))
        row["retention"] = _load_retention_summary(Path(str(entry.get("retention_dir", ""))))
        row.update(_aggregate_retention(row["retention"]))
    return row


def _aggregate_eval(data: dict[str, Any]) -> dict[str, Any]:
    behavior = data.get("behavior_suite") or {}
    generation = data.get("generation_health") or {}
    choice = data.get("choice_accuracy") or {}
    perplexity = data.get("perplexity") or {}
    return {
        "model": data.get("model"),
        "param_count": data.get("param_count"),
        "behavior_correct": behavior.get("correct"),
        "behavior_total": behavior.get("total"),
        "generation_correct": generation.get("correct"),
        "generation_total": generation.get("total"),
        "choice_correct": choice.get("correct"),
        "choice_total": choice.get("total"),
        "perplexity_mean": perplexity.get("mean"),
    }


def _category_counts(items: Any, pass_key: str) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    if not isinstance(items, list):
        return counts
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "unknown")
        bucket = counts.setdefault(category, {"correct": 0, "total": 0})
        bucket["total"] += 1
        if bool(item.get(pass_key)):
            bucket["correct"] += 1
    return counts


def _manual_samples(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        data = _load_json(path)
    except Exception:
        return []
    samples = []
    for item in data.get("manual_generations", [])[:7]:
        if isinstance(item, dict):
            output = str(item.get("output", ""))
            samples.append({"prompt": str(item.get("prompt", "")), "output": output[:500]})
    return samples


def _load_retention_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        return _load_json(summary_path)
    except Exception:
        return {}


def _aggregate_retention(retention: dict[str, Any]) -> dict[str, Any]:
    behavior = _retention_overall(retention, "behavior")
    generation = _retention_overall(retention, "generation")
    choice = _retention_overall(retention, "choice")
    return {
        "behavior_parent_retention": behavior.get("shared_parent_pass_retention_rate"),
        "generation_parent_retention": generation.get("shared_parent_pass_retention_rate"),
        "choice_parent_retention": choice.get("shared_parent_pass_retention_rate"),
        "behavior_parent_pass": behavior.get("parent_pass"),
        "generation_parent_pass": generation.get("parent_pass"),
        "choice_parent_pass": choice.get("parent_pass"),
    }


def _retention_overall(retention: dict[str, Any], section: str) -> dict[str, Any]:
    value = retention.get(section)
    if not isinstance(value, dict):
        return {}
    overall = value.get("overall")
    if isinstance(overall, dict):
        return overall
    return {}


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("status") != "complete",
        -_ratio(row.get("behavior_correct"), row.get("behavior_total")),
        -_ratio(row.get("generation_correct"), row.get("generation_total")),
        -_retention_rate(row.get("behavior_parent_retention")),
        -_retention_rate(row.get("generation_parent_retention")),
        -_ratio(row.get("choice_correct"), row.get("choice_total")),
        -_retention_rate(row.get("choice_parent_retention")),
        row.get("perplexity_mean") if row.get("perplexity_mean") is not None else float("inf"),
        str(row.get("label")),
    )


def _retention_rate(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _ratio(correct: Any, total: Any) -> float:
    if not isinstance(correct, (int, float)) or not isinstance(total, (int, float)) or total == 0:
        return 0.0
    return float(correct) / float(total)


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Neural Forge Retest Matrix",
        "",
        f"Created: `{summary['created_at']}`",
        f"Rows: `{summary['row_count']}` complete: `{summary['complete_count']}`",
        "",
        "| Rank | Label | Category | Behavior | Generation | Choice | Retain B/G/C | PPL |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for index, row in enumerate(summary["rows"], start=1):
        lines.append(
            "| {rank} | `{label}` | {category} | {behavior} | {generation} | {choice} | {retention} | {ppl} |".format(
                rank=index,
                label=row.get("label"),
                category=row.get("category") or "",
                behavior=_metric(row, "behavior"),
                generation=_metric(row, "generation"),
                choice=_metric(row, "choice"),
                retention=_retention_metric(row),
                ppl=_fmt_number(row.get("perplexity_mean")),
            )
        )
    lines.append("")
    lines.append("## Manual Samples")
    for row in summary["rows"][:12]:
        lines.append("")
        lines.append(f"### `{row.get('label')}`")
        if row.get("status") != "complete":
            lines.append(f"Status: `{row.get('status')}`")
            continue
        for sample in row.get("manual_samples", [])[:3]:
            prompt = sample.get("prompt", "").replace("\n", "\\n")
            output = sample.get("output", "").replace("\n", "\\n")
            lines.append(f"- `{prompt}` -> `{output[:240]}`")
    return "\n".join(lines) + "\n"


def _metric(row: dict[str, Any], prefix: str) -> str:
    return f"{row.get(prefix + '_correct')}/{row.get(prefix + '_total')}"


def _retention_metric(row: dict[str, Any]) -> str:
    return "/".join(
        [
            _fmt_percent(row.get("behavior_parent_retention")),
            _fmt_percent(row.get("generation_parent_retention")),
            _fmt_percent(row.get("choice_parent_retention")),
        ]
    )


def _fmt_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _fmt_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return ""


def _safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return label[:180] or "candidate"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
