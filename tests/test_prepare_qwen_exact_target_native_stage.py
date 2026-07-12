from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

import prepare_qwen_exact_target_native_stage as builder  # noqa: E402
import run_qwen_exact_target_native_stage as runner  # noqa: E402


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    padding_side = "left"

    def __call__(self, text, **_kwargs):
        pieces = re.findall(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]", text)
        return {
            "input_ids": [
                3 + int(hashlib.sha256(piece.encode()).hexdigest()[:7], 16) % 100_000
                for piece in pieces
            ]
        }


def _artifact(path: Path, files: dict[str, bytes]) -> Path:
    path.mkdir()
    for name, payload in files.items():
        (path / name).write_bytes(payload)
    return path


def _corpus(path: Path, count: int = 100) -> tuple[Path, str, int]:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(
                json.dumps(
                    {
                        "prompt": (
                            f"Natural corpus prompt identifier {index:04d} requests "
                            "a concise and context-sensitive answer."
                        )
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), count


def test_builder_seals_real_curated_corpus_without_model_creation(tmp_path) -> None:
    parent = _artifact(
        tmp_path / "parent",
        {"config.json": b"parent-config", "model.safetensors": b"parent"},
    )
    target = _artifact(
        tmp_path / "target",
        {"config.json": b"target-config", "model.safetensors": b"target"},
    )
    target_weight_sha = hashlib.sha256(b"target").hexdigest()
    corpus, corpus_sha, corpus_count = _corpus(tmp_path / "corpus.jsonl")
    output = tmp_path / "sealed"
    report = builder.prepare_stage(
        output_dir=output,
        parent=parent,
        target=target,
        corpus_path=corpus,
        corpus_sha256=corpus_sha,
        corpus_line_count=corpus_count,
        tokenizer_loader=lambda _path: _Tokenizer(),
        target_required_files={"model.safetensors": target_weight_sha},
    )
    assert report["model_weights_loaded"] is False
    assert report["model_checkpoint_created"] is False
    assert report["training_launched"] is False
    assert {path.name for path in output.iterdir()} == {
        "fit.json",
        "candidate.json",
        "sealed.json",
        "preregistration.json",
        "build_report.json",
    }
    counts = {
        name: len(json.loads((output / f"{name}.json").read_text())["prompts"])
        for name in ("fit", "candidate", "sealed")
    }
    assert counts == {"fit": 24, "candidate": 12, "sealed": 16}
    selected = [
        item
        for name in ("fit", "candidate", "sealed")
        for item in json.loads((output / f"{name}.json").read_text())["prompts"]
    ]
    assert len({item["source_line_number"] for item in selected}) == 52
    assert len({item["source_prompt_sha256"] for item in selected}) == 52
    preregistration = json.loads(
        (output / "preregistration.json").read_text(encoding="utf-8")
    )
    assert preregistration["training"] == {
        "steps": 24,
        "learning_rate": 0.001,
        "checkpoint_steps": [2, 4, 8, 12, 16, 20],
        "fit_horizon": 8,
        "evaluation_horizon": 8,
        "fit_prompts_per_step": 3,
        "temperature": 1.0,
        "action_loss_weight": 1.0,
        "distribution_loss_weight": 0.1,
        "top_k": 5,
        "gradient_clip_norm": 1.0,
        "max_prompt_tokens": 128,
        "max_train_prefixes": 1152,
        "max_evaluation_prefixes": 5000,
        "max_checkpoints": 8,
        "selection_policy": "rollout-first-acquisition",
    }
    assert preregistration["compiler"]["bootstrap_kwargs"] == {
        "heldout_gate_policy": "output_only",
        "trace_pooling": "last_valid",
    }
    assert preregistration["corpus_provenance"]["synthetic_replicas"] == 0
    assert preregistration["corpus_provenance"]["prompt_text_modified"] is False
    assert preregistration["thresholds"]["candidate"] == {
        "minimum_action_accuracy": 0.5,
        "minimum_exact_rollout_fraction": 1 / 12,
        "minimum_token_match_fraction": 0.2,
        "minimum_lcp_fraction": 0.125,
        "minimum_action_count": 96,
        "minimum_rollout_count": 12,
    }
    authenticated = runner.authenticate_preregistration(
        output / "preregistration.json",
        report["preregistration_sha256"],
    )
    assert authenticated.preflight_report()["models_loaded"] is False
    assert (
        authenticated.preflight_report()["canonical_parent_evidence"][
            "candidate_split_used"
        ]
        is False
    )
    with pytest.raises(builder.StagePreparationError, match="must not already exist"):
        builder.prepare_stage(
            output_dir=output,
            parent=parent,
            target=target,
            corpus_path=corpus,
            corpus_sha256=corpus_sha,
            corpus_line_count=corpus_count,
            tokenizer_loader=lambda _path: _Tokenizer(),
            target_required_files={"model.safetensors": target_weight_sha},
        )


def test_remote_wrapper_defaults_preflight_and_requires_explicit_execute() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "remote_run_qwen_exact_target_native_stage.sh"
    ).read_text(encoding="utf-8")
    assert "MODE=preflight" in script
    assert '[[ ${1:-} == "--execute" ]]' in script
    assert "--preflight" in script
    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,6}" in script
    assert "CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" in script
    assert "parent logical cuda:0, child logical cuda:1" in script
    assert 'ROOT=$(realpath -e "$ROOT")' in script
    assert 'test "$ROOT" = "/home/cereal/model-surgery-lab/neural-forge-src"' in script
    assert "nohup" not in script
    assert not any(line.rstrip().endswith("&") for line in script.splitlines())
