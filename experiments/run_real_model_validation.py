"""Run real-checkpoint validation for Neural Forge.

This intentionally uses small public 2025-era checkpoints so the run can finish
on a single workstation GPU. It measures behavior before and after surgery; a
shape-valid checkpoint is not considered validated by this script unless it can
load, answer probes, and generate text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_forge.hf_tools import evaluate_text_model, resize_gated_ffn, same_shape_merge, vision_probe


DEFAULT_BASE = "Qwen/Qwen3-0.6B-Base"
DEFAULT_TUNED = "Qwen/Qwen3-0.6B"
DEFAULT_VLM = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    reports = run_dir / "reports"
    models = run_dir / "models"
    reports.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "base_model": args.base_model,
        "tuned_model": args.tuned_model,
        "vision_model": args.vision_model,
        "steps": [],
    }

    if not args.skip_text:
        summary["steps"].append({"step": "eval_base"})
        base_report = evaluate_text_model(
            args.base_model,
            reports / "text_base.json",
            torch_dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
        )

        summary["steps"].append({"step": "eval_tuned"})
        tuned_report = evaluate_text_model(
            args.tuned_model,
            reports / "text_tuned.json",
            torch_dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
        )

        merged_dir = models / "text_merged_alpha_050"
        summary["steps"].append({"step": "merge", "output": str(merged_dir)})
        merge_report = same_shape_merge(
            [args.base_model, args.tuned_model],
            merged_dir,
            weights=[0.5, 0.5],
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )

        summary["steps"].append({"step": "eval_merged"})
        merged_report = evaluate_text_model(
            str(merged_dir),
            reports / "text_merged_alpha_050.json",
            torch_dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
        )

        shrunk_dir = models / f"text_merged_ffn_{int(args.shrink_ratio * 100):03d}"
        summary["steps"].append({"step": "resize_ffn", "output": str(shrunk_dir), "ratio": args.shrink_ratio})
        resize_report = resize_gated_ffn(
            str(merged_dir),
            shrunk_dir,
            ratio=args.shrink_ratio,
            torch_dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )

        summary["steps"].append({"step": "eval_shrunk"})
        shrunk_report = evaluate_text_model(
            str(shrunk_dir),
            reports / f"text_merged_ffn_{int(args.shrink_ratio * 100):03d}.json",
            torch_dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.max_new_tokens,
        )

        summary["text_metrics"] = {
            "base": _compact_text_metrics(base_report),
            "tuned": _compact_text_metrics(tuned_report),
            "merged_alpha_050": _compact_text_metrics(merged_report),
            f"merged_ffn_{int(args.shrink_ratio * 100):03d}": _compact_text_metrics(shrunk_report),
            "merge_report": merge_report,
            "resize_report": resize_report,
        }

    if not args.skip_vision:
        summary["steps"].append({"step": "vision_probe"})
        vision_report = vision_probe(
            args.vision_model,
            reports / "vision_probe.json",
            image_dir=run_dir / "vision_images",
            torch_dtype=args.dtype,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_new_tokens=args.vision_max_new_tokens,
        )
        summary["vision_metrics"] = {
            "param_count": vision_report["param_count"],
            "outputs": vision_report["outputs"],
        }

    summary_path = reports / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/real-model-validation")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--tuned-model", default=DEFAULT_TUNED)
    parser.add_argument("--vision-model", default=DEFAULT_VLM)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shrink-ratio", type=float, default=0.995)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--vision-max-new-tokens", type=int, default=80)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-text", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    return parser.parse_args()


def _compact_text_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "param_count": report["param_count"],
        "perplexity_mean": report["perplexity"]["mean"],
        "choice_accuracy": report["choice_accuracy"]["accuracy"],
        "choice_correct": report["choice_accuracy"]["correct"],
        "choice_total": report["choice_accuracy"]["total"],
        "generation_previews": [
            {
                "prompt": item["prompt"],
                "output": item["output"][:600],
            }
            for item in report["generations"]
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
