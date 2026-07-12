from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

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
        "max_operator_delta_ratio": 0.1,
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
    assert '[[ ${1:-} == "--lexical-continuation" ]]' in script
    assert "LEXICAL_CONTINUATION_SOURCE_REPORT" in script
    assert "--lexical-continuation-source-report" in script
    assert "--source-split-dir" in script


def test_lexical_builder_seals_nonrestarting_schedule_and_required_cache_hit(
    tmp_path, monkeypatch
) -> None:
    parent = _artifact(
        tmp_path / "parent",
        {"config.json": b"parent-config", "tokenizer.json": b"tokenizer"},
    )
    target_config = json.dumps(
        {
            "model_type": "qwen3",
            "vocab_size": runner.LEXICAL_VOCAB_SIZE,
            "hidden_size": runner.LEXICAL_HIDDEN_SIZE,
            "tie_word_embeddings": False,
        }
    ).encode()
    target = _artifact(
        tmp_path / "target",
        {"config.json": target_config, "model.safetensors": b"target"},
    )
    original_splits = tmp_path / "original"
    original_splits.mkdir()
    compiler_fit = original_splits / "fit.json"
    compiler_fit.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(
        builder.continuation_splits.PINNED_ORIGINAL_SPLIT_SHA256S,
        "fit",
        hashlib.sha256(compiler_fit.read_bytes()).hexdigest(),
    )
    source_splits = tmp_path / "source-splits"
    source_splits.mkdir()
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    identity_sha = "1" * 64
    cache_path = cache_root / identity_sha
    cache_path.mkdir()
    factor_path = tmp_path / "source_factor_pack.pt"
    factor_path.write_bytes(b"authenticated-factor-pack")
    factor_sha = hashlib.sha256(factor_path.read_bytes()).hexdigest()
    factor_state_sha = "2" * 64
    compiler_report = {"bootstrap_accepted": True, "target_budget_exact": True}
    compiler_report_sha = runner._canonical_sha256(compiler_report)
    source_report = {
        "status": "rejected",
        "rollback_exact": True,
        "rollback_state_sha256": "3" * 64,
        "splits": {
            role: {
                "sha256": builder.continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[
                    role
                ],
                "row_count": builder.continuation_splits.RESIDUAL_ATTENTION_COUNTS[
                    role
                ],
            }
            for role in ("fit", "candidate", "sealed")
        },
        "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
        "trainer_evidence": {
            "sealed_evaluation_unlocked": False,
            "sealed_metrics": None,
            "factor_pack_manifest": {
                "schema": runner.TRAINER_FACTOR_PACK_SCHEMA,
                "factor_state_sha256": factor_state_sha,
            },
        },
        "compiler_report": compiler_report,
        "compiled_child_cache": {
            "disposition": "published",
            "path": str(cache_path),
            "compiler_identity_sha256": identity_sha,
            "receipt_sha256": "4" * 64,
            "compiler_report_sha256": compiler_report_sha,
            "state_dict_sha256": "5" * 64,
            "artifact_tree_sha256": "6" * 64,
            "verified_before_factor_install": True,
        },
    }
    source_report_path = tmp_path / "source_run_report.json"
    source_report_path.write_text(
        json.dumps(source_report, sort_keys=True) + "\n", encoding="utf-8"
    )

    def fake_split_builder(**kwargs):
        output = Path(kwargs["output_dir"])
        output.mkdir()
        for role in ("fit", "candidate"):
            (output / f"{role}.json").write_text(
                json.dumps({"schema": "test", "split": role}) + "\n",
                encoding="utf-8",
            )
        return {"build_content_sha256": "7" * 64}

    monkeypatch.setattr(
        builder.continuation_splits,
        "prepare_lexical_continuation_splits",
        fake_split_builder,
    )
    monkeypatch.setattr(builder, "_default_target_required_files", lambda: {})
    monkeypatch.setattr(
        builder.stage,
        "authenticate_preregistration",
        lambda _path, _sha: SimpleNamespace(evaluation_contract_sha256="8" * 64),
    )
    output = tmp_path / "lexical-prereg"

    report = builder.prepare_lexical_continuation_stage(
        output_dir=output,
        parent=parent,
        target=target,
        source_run_report=source_report_path,
        source_split_dir=source_splits,
        original_split_dir=original_splits,
        compiler_cache_root=cache_root,
        corpus_path=tmp_path / "unused-corpus.jsonl",
    )

    preregistration = json.loads((output / "preregistration.json").read_text())
    assert report["mode"] == "authenticated-lexical-attention-continuation"
    assert preregistration["lexical_residual"] == {
        "rank": 32,
        "vocab_size": 151936,
        "hidden_size": 1536,
        "untied_required": True,
    }
    assert preregistration["splits"]["fit"]["expected_count"] == 120
    assert preregistration["splits"]["candidate"]["expected_count"] == 24
    assert preregistration["splits"]["sealed"] == {
        "path": str(source_splits / "sealed.json"),
        "sha256": builder.continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[
            "sealed"
        ],
        "expected_count": 32,
    }
    assert preregistration["compiler_cache"]["require_hit"] is True
    assert (
        preregistration["compiler_cache"]["expected_compiler_identity_sha256"]
        == identity_sha
    )
    schedule = preregistration["training"]
    assert schedule["steps"] == 48
    assert schedule["fit_prompts_per_step"] == 12
    assert schedule["qkvo_frozen_steps"] == 8
    assert schedule["gud_frozen_steps"] == 24
    assert schedule["lexical_embedding_frozen_steps"] == 24
    assert schedule["lexical_head_frozen_steps"] == 0
    assert schedule["learning_rate_decay_step_3"] == 40
    assert schedule["learning_rate_decay_scale_3"] == 0.5
    assert schedule["gud_learning_rate_scale"] == 0.25
    assert schedule["lexical_embedding_learning_rate_scale"] == 0.25
    assert schedule["max_checkpoints"] == 10
    assert preregistration["thresholds"]["fit"]["minimum_action_count"] == 960
    assert (
        preregistration["thresholds"]["candidate"]["minimum_action_count"]
        == 192
    )
    assert preregistration["thresholds"]["sealed"]["minimum_action_count"] == 256
