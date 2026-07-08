#!/usr/bin/env python3
"""Run a small before/after cross-family Neural Forge proof.

The proof is intentionally CPU-friendly and evidence-oriented:

1. Build a deterministic behavior suite.
2. Evaluate the donor model before folding.
3. Create and evaluate a random blank target architecture.
4. Fold the donor into the blank target shape with the current merge-first
   transport defaults.
5. Compare donor-vs-child item retention, then rank donor, blank, folded, and
   optional recovered children.

Use this for local smoke proof while the larger GPU matrix runs on the AI
server.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_DONOR_MODEL = "sshleifer/tiny-gpt2"
DEFAULT_TARGET_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
DEFAULT_BEHAVIOR_COUNT = 24
DEFAULT_GENERATION_COUNT = 12
DEFAULT_CHOICE_COUNT = 12
DEFAULT_TEXT_COUNT = 8


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/local_tiny_cross_family_proof"))
    parser.add_argument("--donor-model", default=DEFAULT_DONOR_MODEL)
    parser.add_argument("--target-config-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--target-tokenizer-model", default=DEFAULT_TARGET_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--behavior-count", type=int, default=DEFAULT_BEHAVIOR_COUNT)
    parser.add_argument("--generation-count", type=int, default=DEFAULT_GENERATION_COUNT)
    parser.add_argument("--choice-count", type=int, default=DEFAULT_CHOICE_COUNT)
    parser.add_argument("--text-count", type=int, default=DEFAULT_TEXT_COUNT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--activation-max-calibration-texts", type=int, default=16)
    parser.add_argument("--activation-max-length", type=int, default=96)
    parser.add_argument("--recover-steps", type=int, default=0)
    parser.add_argument("--recover-calibration-texts", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    proof = TinyCrossFamilyProof(args)
    return proof.run()


class TinyCrossFamilyProof:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.output_dir
        self.suite = self.root / "suite.json"
        self.calibration = self.root / "calibration.jsonl"
        self.blank_model = self.root / "blank_target"
        self.folded_model = self.root / "folded_child"
        self.recovered_model = self.root / "recovered_child"
        self.reports = self.root / "reports"
        self.retention = self.root / "retention"
        self.commands: list[list[str]] = []

    def run(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.retention.mkdir(parents=True, exist_ok=True)

        self._build_eval_suite()
        self._build_calibration()
        donor_eval = self._eval_model(self.args.donor_model, "donor")
        self._random_init_blank()
        blank_eval = self._eval_model(str(self.blank_model), "blank")
        self._fold_donor_into_blank()
        folded_eval = self._eval_model(str(self.folded_model), "folded")
        evals = [donor_eval, blank_eval, folded_eval]
        retentions = {
            "donor_vs_blank": self._compare_retention(donor_eval, blank_eval, "donor_vs_blank"),
            "donor_vs_folded": self._compare_retention(donor_eval, folded_eval, "donor_vs_folded"),
        }

        if int(self.args.recover_steps) > 0:
            self._recover_folded_child()
            recovered_eval = self._eval_model(str(self.recovered_model), "recovered")
            evals.append(recovered_eval)
            retentions["donor_vs_recovered"] = self._compare_retention(
                donor_eval,
                recovered_eval,
                "donor_vs_recovered",
            )
            retentions["folded_vs_recovered"] = self._compare_retention(
                folded_eval,
                recovered_eval,
                "folded_vs_recovered",
            )

        rank = self.reports / "rank.json"
        self._run(self._cli("hf-rank-evals", str(rank), *(str(path) for path in evals)), output=rank)
        summary = self._write_summary(evals, rank, retentions)
        if self.args.dry_run:
            print(json.dumps({"dry_run": True, "commands": self.commands, "summary": str(summary)}, indent=2))
        else:
            print(json.dumps({"summary": str(summary), "rank": str(rank)}, indent=2))
        return 0

    def _build_eval_suite(self) -> None:
        cmd = self._cli(
            "hf-build-eval-suite",
            str(self.suite),
            "--behavior-count",
            str(int(self.args.behavior_count)),
            "--generation-count",
            str(int(self.args.generation_count)),
            "--choice-count",
            str(int(self.args.choice_count)),
            "--text-count",
            str(int(self.args.text_count)),
            "--seed",
            str(int(self.args.seed)),
        )
        self._run(cmd, output=self.suite)

    def _build_calibration(self) -> None:
        cmd = self._cli(
            "hf-build-recovery-calibration",
            str(self.calibration),
            "--math-count",
            "16",
            "--code-count",
            "8",
            "--tokenizer-count",
            "8",
            "--model-surgery-count",
            "8",
            "--commonsense-count",
            "8",
            "--seed",
            str(int(self.args.seed)),
            "--exclude-eval-suite",
            str(self.suite),
        )
        self._run(cmd, output=self.calibration)

    def _random_init_blank(self) -> None:
        cmd = self._cli(
            "hf-random-init",
            str(self.blank_model),
            "--config-model",
            self.args.target_config_model,
            "--tokenizer-model",
            self.args.target_tokenizer_model,
            "--seed",
            str(int(self.args.seed)),
            "--dtype",
            self.args.dtype,
        )
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        self._run(cmd, output=self.blank_model)

    def _fold_donor_into_blank(self) -> None:
        cmd = self._cli(
            "hf-donor-project-init",
            str(self.folded_model),
            "--target-config-model",
            self.args.target_config_model,
            "--target-base-model",
            str(self.blank_model),
            "--target-tokenizer-model",
            self.args.target_tokenizer_model,
            "--donor-model",
            self.args.donor_model,
            "--seed",
            str(int(self.args.seed)),
            "--dtype",
            self.args.dtype,
            "--fold-mode",
            "transport",
            "--hidden-basis-mode",
            "activation-pca",
            "--activation-fold-assignment",
            "balanced-signed-similarity",
            "--fold-stat-repair",
            "rms",
            "--transport-backend",
            "sinkhorn",
            "--transport-support",
            "domain-diverse-medoid",
            "--donor-layer-fold-mode",
            "fold",
            "--donor-layer-output-scale",
            "group-count",
            "--donor-layer-group-strategy",
            "activation-cka-weighted",
            "--attention-fold-mode",
            "activation-coupled",
            "--token-init-mode",
            "donor-tokenized-weighted",
            "--token-fallback-mode",
            "donor-mean",
            "--projection-scale",
            "geomean-rms",
            "--projection-blend-alpha",
            "1.0",
            "--activation-feature-mode",
            "covariance",
            "--activation-subspace-rank",
            "8",
            "--activation-domain-balance",
            "keyword",
            "--activation-calibration-file",
            str(self.calibration),
            "--activation-max-calibration-texts",
            str(int(self.args.activation_max_calibration_texts)),
            "--activation-max-length",
            str(int(self.args.activation_max_length)),
            "--activation-device",
            self.args.device,
        )
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        self._run(cmd, output=self.folded_model)

    def _recover_folded_child(self) -> None:
        cmd = self._cli(
            "hf-xfamily-recover",
            self.args.donor_model,
            str(self.folded_model),
            str(self.recovered_model),
            "--steps",
            str(int(self.args.recover_steps)),
            "--batch-size",
            "1",
            "--max-length",
            "96",
            "--lr",
            "5e-5",
            "--anchor-kd-weight",
            "0.4",
            "--lm-weight",
            "0.2",
            "--projected-vocab-kd-weight",
            "0.05",
            "--token-kd-weight",
            "0.05",
            "--token-focus",
            "numbers",
            "--answer-first-token-margin-weight",
            "0.05",
            "--answer-first-token-top-negatives",
            "8",
            "--train-scope",
            "all",
            "--response-loss-mode",
            "auto",
            "--calibration-file",
            str(self.calibration),
            "--max-calibration-texts",
            str(int(self.args.recover_calibration_texts)),
            "--dtype",
            self.args.dtype,
            "--device",
            self.args.device,
        )
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        self._run(cmd, output=self.recovered_model)

    def _eval_model(self, model: str, label: str) -> Path:
        output = self.reports / f"{label}_eval.json"
        cmd = self._cli(
            "hf-eval",
            model,
            str(output),
            "--dtype",
            self.args.dtype,
            "--device",
            self.args.device,
            "--eval-suite",
            str(self.suite),
            "--max-new-tokens",
            str(int(self.args.max_new_tokens)),
        )
        if self.args.trust_remote_code:
            cmd.append("--trust-remote-code")
        self._run(cmd, output=output)
        return output

    def _compare_retention(self, parent: Path, child: Path, label: str) -> Path:
        output = self.retention / f"{label}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "compare_eval_retention.py"),
            str(parent),
            str(child),
            "--out",
            str(output),
            "--max-samples",
            "12",
        ]
        self._run(cmd, output=output)
        return output

    def _write_summary(self, evals: list[Path], rank: Path, retentions: dict[str, Path]) -> Path:
        summary_path = self.root / "proof_summary.json"
        if self.args.dry_run:
            summary = {
                "operation": "tiny_cross_family_proof",
                "dry_run": True,
                "donor_model": self.args.donor_model,
                "target_config_model": self.args.target_config_model,
                "target_tokenizer_model": self.args.target_tokenizer_model,
                "retention_reports": {label: str(path) for label, path in retentions.items()},
                "commands": self.commands,
            }
        else:
            reports = {path.stem.replace("_eval", ""): _compact_report(path) for path in evals}
            rank_data = _load_json(rank) if rank.exists() else {}
            retention_reports = {
                label: _compact_retention(path)
                for label, path in retentions.items()
                if path.exists()
            }
            summary = {
                "operation": "tiny_cross_family_proof",
                "dry_run": False,
                "donor_model": self.args.donor_model,
                "target_config_model": self.args.target_config_model,
                "target_tokenizer_model": self.args.target_tokenizer_model,
                "reports": reports,
                "retention_reports": retention_reports,
                "rank": str(rank),
                "best_model": (rank_data.get("best") or {}).get("model") if isinstance(rank_data, dict) else None,
                "folded_vs_blank": _delta(reports.get("folded"), reports.get("blank")),
                "recovered_vs_folded": _delta(reports.get("recovered"), reports.get("folded")),
            }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary_path

    def _run(self, cmd: list[str], *, output: Path) -> None:
        self.commands.append(cmd)
        if self.args.dry_run:
            return
        if output.exists() and not self.args.force:
            return
        subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    @staticmethod
    def _cli(*args: str) -> list[str]:
        return [sys.executable, "-m", "neural_forge.cli", *args]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_report(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    behavior = data.get("behavior_suite") or {}
    generation = data.get("generation_health") or {}
    choice = data.get("choice_accuracy") or {}
    perplexity = data.get("perplexity") or {}
    return {
        "path": str(path),
        "model": data.get("model"),
        "param_count": data.get("param_count"),
        "behavior": _metric(behavior),
        "generation": _metric(generation),
        "choice": _choice_metric(choice),
        "ppl_mean": perplexity.get("mean"),
        "critical_behavior": _critical_counts((behavior.get("items") or []), "passed"),
        "sample_generations": _sample_generations(data),
    }


def _compact_retention(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    return {
        "path": str(path),
        "parent": data.get("parent"),
        "child": data.get("child"),
        "param_compression_ratio": data.get("param_compression_ratio"),
        "behavior": _overall_retention(data.get("behavior")),
        "generation": _overall_retention(data.get("generation")),
        "choice": _overall_retention(data.get("choice")),
        "manual_samples": data.get("manual_samples", {}),
    }


def _overall_retention(section: Any) -> dict[str, Any]:
    if not isinstance(section, dict):
        return {}
    overall = section.get("overall")
    if isinstance(overall, dict):
        return overall
    return {}


def _metric(section: dict[str, Any]) -> dict[str, Any]:
    correct = int(section.get("correct") or 0)
    total = int(section.get("total") or 0)
    return {"correct": correct, "total": total, "rate": correct / max(1, total)}


def _choice_metric(section: dict[str, Any]) -> dict[str, Any]:
    out = _metric(section)
    out["accuracy"] = section.get("accuracy", out["rate"])
    return out


def _critical_counts(items: list[Any], pass_key: str) -> dict[str, dict[str, int]]:
    buckets = {
        "math": {"correct": 0, "total": 0},
        "code": {"correct": 0, "total": 0},
        "tokenizer": {"correct": 0, "total": 0},
        "model_surgery": {"correct": 0, "total": 0},
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or item.get("id") or "").lower().replace("-", "_")
        name = None
        if any(token in category for token in ("math", "addition", "multiply", "numeric")):
            name = "math"
        elif any(token in category for token in ("code", "python", "function")):
            name = "code"
        elif "token" in category or "vocab" in category:
            name = "tokenizer"
        elif any(token in category for token in ("surgery", "merge", "pruning", "neuron")):
            name = "model_surgery"
        if name is None:
            continue
        buckets[name]["total"] += 1
        if bool(item.get(pass_key)):
            buckets[name]["correct"] += 1
    return {key: value for key, value in buckets.items() if value["total"]}


def _sample_generations(data: dict[str, Any]) -> list[dict[str, str]]:
    samples = []
    for item in (data.get("generations") or [])[:5]:
        if isinstance(item, dict):
            samples.append(
                {
                    "prompt": str(item.get("prompt", "")),
                    "output": str(item.get("output", ""))[:400],
                }
            )
    return samples


def _delta(after: dict[str, Any] | None, before: dict[str, Any] | None) -> dict[str, Any]:
    if not after or not before:
        return {}
    return {
        "behavior_rate_delta": after["behavior"]["rate"] - before["behavior"]["rate"],
        "generation_rate_delta": after["generation"]["rate"] - before["generation"]["rate"],
        "choice_rate_delta": after["choice"]["rate"] - before["choice"]["rate"],
        "ppl_delta": (
            None
            if after.get("ppl_mean") is None or before.get("ppl_mean") is None
            else after["ppl_mean"] - before["ppl_mean"]
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
