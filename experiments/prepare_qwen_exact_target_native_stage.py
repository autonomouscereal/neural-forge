#!/usr/bin/env python3
"""Seal authenticated exact target-native Qwen stage contracts.

The base mode derives the original 52-prompt contract from the existing
2,906-row canonical Neural Forge corpus without synthesizing prompts.  The
continuation modes reuse and authenticate terminal run evidence, retained
factor packs, immutable behavior splits, and the compiled-child cache.  The
output-Jacobian mode prepares a diagnostic-only contract and never loads model
weights or starts training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

import run_qwen_exact_target_native_stage as stage
import prepare_qwen_exact_target_native_continuation_splits as continuation_splits
from neural_forge.hf_compiled_child_cache import (
    CACHE_RECEIPT_FILENAME,
    CompiledChildCacheError,
    CompilerIdentity,
    verify_compiled_child_cache,
)
from neural_forge.hf_target_native_behavior_distillation import BehaviorPrompt


BUILD_SCHEMA = "neural-forge-qwen-exact-target-native-stage-build-v1"
SELECTION_SCHEMA = "neural-forge-hash-ordered-prefix-disjoint-corpus-v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = (
    REPO_ROOT
    / "runs"
    / "qwen4b_to_smollm135m_transport_random_20260627"
    / "calibration_mix_plus_recovery_drills_20260702.jsonl"
)
PINNED_CORPUS_SHA256 = (
    "ba2dd1202752e2dd2b9b981c52501060ee367e220e5fb73ce7f8c320e2bbdd90"
)
PINNED_CORPUS_LINE_COUNT = 2_906
DOMAIN_COUNTS = {"fit": 24, "candidate": 12, "sealed": 16}

DEFAULT_PARENT = Path(
    "/home/cereal/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-4B-Base/snapshots/"
    "906bfd4b4dc7f14ee4320094d8b41684abff8539"
)
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/prereg/qwen_exact_target_native_stage1_20260712"
)
DEFAULT_COMPILER_CACHE_ROOT = Path("/mnt/ai_data/neural-forge/compiled_child_cache")
OUTPUT_JACOBIAN_AUDIT_COUNTS = {"fit": 120, "candidate": 24, "sealed": 32}
OUTPUT_JACOBIAN_AUDIT_MODE = "authenticated-output-jacobian-audit"


class StagePreparationError(RuntimeError):
    """The corpus, artifact, or sealed stage contract is not reproducible."""


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _messages_to_prompt(messages: Any) -> str | None:
    if not isinstance(messages, list):
        return None
    pieces: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = message.get("role")
        prefix = (
            f"{str(role).strip()}: " if isinstance(role, str) and role.strip() else ""
        )
        pieces.append(prefix + content.strip())
    return "\n".join(pieces) or None


def _row_prompt(item: Any) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if not isinstance(item, Mapping):
        return None
    messages = _messages_to_prompt(item.get("messages"))
    if messages:
        return messages
    input_json = item.get("input_json")
    if isinstance(input_json, Mapping):
        messages = _messages_to_prompt(input_json.get("messages"))
        if messages:
            return messages
    for key in ("text", "prompt", "content"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_corpus(
    path: Path,
    *,
    expected_sha256: str,
    expected_line_count: int,
) -> tuple[dict[str, Any], ...]:
    if stage._file_sha256(path) != expected_sha256:
        raise StagePreparationError("Pinned canonical corpus SHA-256 changed.")
    rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    nonempty_lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            nonempty_lines += 1
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as error:
                raise StagePreparationError(
                    f"Invalid canonical JSONL at line {line_number}."
                ) from error
            text = _row_prompt(item)
            if text is None:
                raise StagePreparationError(
                    f"Canonical corpus row {line_number} has no prompt text."
                )
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append(
                {
                    "source_line_number": line_number,
                    "source_prompt_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "text": text,
                }
            )
    if nonempty_lines != expected_line_count:
        raise StagePreparationError(
            f"Canonical corpus has {nonempty_lines}, not {expected_line_count}, rows."
        )
    if len(rows) < sum(DOMAIN_COUNTS.values()):
        raise StagePreparationError("Canonical corpus has too few unique prompt texts.")
    return tuple(
        sorted(
            rows,
            key=lambda value: (
                value["source_prompt_sha256"],
                value["source_line_number"],
            ),
        )
    )


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise RuntimeError("Transformers is required to seal token splits.") from error
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise StagePreparationError("Pinned tokenizer has no pad/eos token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    values = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
    ):
        raise StagePreparationError("Pinned tokenizer returned invalid token IDs.")
    return tuple(values)


def _select_domains(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_prompt_tokens: int,
) -> dict[str, list[dict[str, Any]]]:
    selected = {role: [] for role in ("fit", "candidate", "sealed")}
    used_prompt_ids: set[str] = set()
    used_prefixes: set[tuple[int, ...]] = set()
    roles = iter(("fit", "candidate", "sealed"))
    role = next(roles)
    for record in records:
        tokens = _tokenize(tokenizer, str(record["text"]))
        if len(tokens) > max_prompt_tokens:
            continue
        prompt = BehaviorPrompt(tokens)
        prefixes = {tokens[:length] for length in range(8, len(tokens) + 1)}
        if prompt.prompt_id in used_prompt_ids or prefixes & used_prefixes:
            continue
        selected[role].append(
            {
                "token_ids": list(tokens),
                "prompt_id": prompt.prompt_id,
                "token_ids_sha256": prompt.token_ids_sha256,
                "source_line_number": int(record["source_line_number"]),
                "source_prompt_sha256": str(record["source_prompt_sha256"]),
            }
        )
        used_prompt_ids.add(prompt.prompt_id)
        used_prefixes.update(prefixes)
        if len(selected[role]) == DOMAIN_COUNTS[role]:
            try:
                role = next(roles)
            except StopIteration:
                break
    if {name: len(values) for name, values in selected.items()} != DOMAIN_COUNTS:
        raise StagePreparationError(
            "Could not select 52 natural, hash-ordered, prefix-disjoint prompts."
        )
    return selected


def _split_payload(
    role: str,
    prompts: Sequence[Mapping[str, Any]],
    *,
    corpus_path: Path,
    corpus_sha256: str,
    corpus_line_count: int,
) -> dict[str, Any]:
    return {
        "schema": stage.SPLIT_SCHEMA,
        "split": role,
        "sealed_before_model_access": role == "sealed",
        "prompts": [dict(value) for value in prompts],
        "source": {
            "path": str(corpus_path),
            "sha256": corpus_sha256,
            "line_count": corpus_line_count,
        },
        "selection": {
            "schema": SELECTION_SCHEMA,
            "global_order": "source_prompt_sha256, then source_line_number",
            "allocation_order": "first 24 fit, next 12 candidate, next 16 sealed",
            "exact_text_deduplicated": True,
            "cross_split_minimum_prefix_tokens": 8,
            "colliding_rows_skipped": True,
            "prompt_text_modified": False,
            "synthetic_replicas": 0,
        },
        "raw_prompt_text_retained": False,
    }


def _default_target_required_files() -> dict[str, str]:
    return {
        stage.DEFAULT_BLANK_WEIGHT_NAME: stage.DEFAULT_BLANK_WEIGHT_SHA256,
        stage.DEFAULT_BLANK_REPORT_NAME: (
            stage.DEFAULT_BLANK_MATERIALIZATION_REPORT_SHA256
        ),
        "config.json": stage.DEFAULT_BLANK_CONFIG_SHA256,
        "tokenizer.json": stage.DEFAULT_BLANK_TOKENIZER_SHA256,
        "tokenizer_config.json": stage.DEFAULT_BLANK_TOKENIZER_CONFIG_SHA256,
    }


def prepare_stage(
    *,
    output_dir: Path,
    parent: Path,
    target: Path,
    corpus_path: Path = DEFAULT_CORPUS,
    corpus_sha256: str = PINNED_CORPUS_SHA256,
    corpus_line_count: int = PINNED_CORPUS_LINE_COUNT,
    tokenizer_loader: Callable[[Path], Any] = _load_tokenizer,
    artifact_tree: Callable[[Path], Mapping[str, Any]] = stage._artifact_tree,
    target_required_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Write one deterministic source-, code-, and artifact-pinned contract."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise StagePreparationError("output_dir must not already exist.")
    parent = parent.expanduser().resolve()
    target = target.expanduser().resolve()
    corpus = corpus_path.expanduser().resolve()
    records = _load_corpus(
        corpus,
        expected_sha256=corpus_sha256,
        expected_line_count=corpus_line_count,
    )
    tokenizer = tokenizer_loader(parent)
    selected = _select_domains(records, tokenizer, max_prompt_tokens=128)
    payloads = {
        role: _split_payload(
            role,
            selected[role],
            corpus_path=corpus,
            corpus_sha256=corpus_sha256,
            corpus_line_count=corpus_line_count,
        )
        for role in ("fit", "candidate", "sealed")
    }
    parent_tree = dict(artifact_tree(parent))
    target_tree = dict(artifact_tree(target))
    for name, tree in (("parent", parent_tree), ("target", target_tree)):
        if not stage._SHA256.fullmatch(str(tree.get("tree_sha256", ""))):
            raise StagePreparationError(f"{name} artifact tree is not authenticated.")
    required = dict(target_required_files or _default_target_required_files())
    for relative, expected in required.items():
        if stage._file_sha256(target / relative) != expected:
            raise StagePreparationError(
                f"Canonical target required-file hash changed: {relative}"
            )
    output.mkdir(parents=True)
    split_bindings: dict[str, Any] = {}
    for role in ("fit", "candidate", "sealed"):
        path = output / f"{role}.json"
        _write_json_exclusive(path, payloads[role])
        split_bindings[role] = {
            "path": str(path),
            "sha256": stage._file_sha256(path),
            "expected_count": DOMAIN_COUNTS[role],
        }
    trainer_spec = importlib.util.find_spec(stage.CANONICAL_TRAINER_MODULE)
    if trainer_spec is None or trainer_spec.origin is None:
        raise StagePreparationError("Canonical trainer source is unavailable.")
    builder_path = Path(__file__).resolve()
    wrapper_path = (
        REPO_ROOT / "experiments" / "remote_run_qwen_exact_target_native_stage.sh"
    )
    source_hashes = {
        "corpus": corpus_sha256,
        "builder": stage._file_sha256(builder_path),
        "remote_wrapper": stage._file_sha256(wrapper_path),
    }
    thresholds = {
        "fit": {
            "minimum_action_accuracy": 0.65,
            "minimum_exact_rollout_fraction": 0.25,
            "minimum_token_match_fraction": 0.40,
            "minimum_lcp_fraction": 0.30,
            "minimum_action_count": 192,
            "minimum_rollout_count": 24,
        },
        "candidate": {
            "minimum_action_accuracy": 0.50,
            "minimum_exact_rollout_fraction": 1 / 12,
            "minimum_token_match_fraction": 0.20,
            "minimum_lcp_fraction": 0.125,
            "minimum_action_count": 96,
            "minimum_rollout_count": 12,
        },
        "sealed": {
            "minimum_action_accuracy": 0.60,
            "minimum_exact_rollout_fraction": 0.1875,
            "minimum_token_match_fraction": 0.35,
            "minimum_lcp_fraction": 0.25,
            "minimum_action_count": 128,
            "minimum_rollout_count": 16,
        },
    }
    preregistration = {
        "schema": stage.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "seed": 0,
        "implementation": {
            "runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
            "trainer_module": stage.CANONICAL_TRAINER_MODULE,
            "trainer_sha256": stage._file_sha256(Path(trainer_spec.origin)),
        },
        "parent": {
            "path": str(parent),
            "exact_unique_parameters": stage.QWEN3_4B_EXACT_PARAMETERS,
            "tree_sha256": parent_tree["tree_sha256"],
        },
        "blank_target": {
            "path": str(target),
            "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
            "tree_sha256": target_tree["tree_sha256"],
            "required_files": required,
        },
        "splits": split_bindings,
        "thresholds": thresholds,
        "compiler": {
            "capture_batch_size": 1,
            "max_prompt_tokens": 128,
            "canonical_validation_count": 4,
            "bootstrap_kwargs": {
                "heldout_gate_policy": "output_only",
                "max_operator_delta_ratio": 0.1,
                "trace_pooling": "last_valid",
            },
            "candidate_split_used_during_bootstrap": False,
            "sealed_split_used_during_bootstrap": False,
        },
        "training": {
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
        },
        "corpus_provenance": {
            "source_sha256s": source_hashes,
            "source_line_count": corpus_line_count,
            "unique_text_count": len(records),
            "allocation_counts": DOMAIN_COUNTS,
            "selection_schema": SELECTION_SCHEMA,
            "source_indices_and_hashes_stored_per_prompt": True,
            "prompt_text_modified": False,
            "synthetic_replicas": 0,
            "cross_split_exact_prompt_overlap": 0,
            "cross_split_meaningful_token_prefix_overlap": 0,
            "sealed_used_for_training_or_candidate_selection": False,
        },
        "runtime": {
            "dtype": "float16",
            "default_mode": "preflight",
            "explicit_execute_required": True,
        },
    }
    prereg_path = output / "preregistration.json"
    _write_json_exclusive(prereg_path, preregistration)
    prereg_sha = stage._file_sha256(prereg_path)
    authenticated = stage.authenticate_preregistration(prereg_path, prereg_sha)
    report = {
        "schema": BUILD_SCHEMA,
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": authenticated.evaluation_contract_sha256,
        "split_sha256s": {
            role: split_bindings[role]["sha256"]
            for role in ("fit", "candidate", "sealed")
        },
        "parent_tree_sha256": parent_tree["tree_sha256"],
        "target_tree_sha256": target_tree["tree_sha256"],
        "corpus_sha256": corpus_sha256,
        "corpus_line_count": corpus_line_count,
        "model_weights_loaded": False,
        "model_checkpoint_created": False,
        "ready_for_explicit_preflight": True,
        "training_launched": False,
    }
    _write_json_exclusive(output / "build_report.json", report)
    return report


def prepare_continuation_stage(
    *,
    output_dir: Path,
    parent: Path,
    target: Path,
    source_run_report: Path,
    original_split_dir: Path = continuation_splits.DEFAULT_ORIGINAL_SPLIT_DIR,
    compiler_cache_root: Path = DEFAULT_COMPILER_CACHE_ROOT,
    corpus_path: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    """Seal a factor-resuming, cache-backed residual-attention continuation."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise StagePreparationError("output_dir must not already exist.")
    parent = parent.expanduser().resolve()
    target = target.expanduser().resolve()
    source_report_path = source_run_report.expanduser().resolve()
    original_split_root = original_split_dir.expanduser().resolve()
    cache_root = compiler_cache_root.expanduser().resolve()
    source_report = stage._strict_json(source_report_path)
    if not isinstance(source_report, Mapping):
        raise StagePreparationError("Continuation source run report is malformed.")
    trainer_evidence = source_report.get("trainer_evidence")
    if (
        not isinstance(trainer_evidence, Mapping)
        or trainer_evidence.get("sealed_evaluation_unlocked") is not False
        or trainer_evidence.get("sealed_metrics") is not None
    ):
        raise StagePreparationError(
            "Original sealed prompts were exposed by the source run and cannot be "
            "reused as sealed continuation evidence."
        )
    factor_record = source_report.get("factor_pack")
    if not isinstance(factor_record, Mapping):
        raise StagePreparationError("Source report omits its retained factor pack.")
    factor_path = Path(str(factor_record.get("path", ""))).expanduser().resolve()
    factor_sha = stage._require_sha256(
        factor_record.get("sha256"), "source_run_report.factor_pack.sha256"
    )
    if not factor_path.is_file() or stage._file_sha256(factor_path) != factor_sha:
        raise StagePreparationError("Source factor pack is absent or changed.")
    source_report_sha = stage._file_sha256(source_report_path)
    source_compiler_report = source_report.get("compiler_report")
    if not isinstance(source_compiler_report, Mapping):
        raise StagePreparationError("Source report omits compiler evidence.")
    source_factor_manifest = trainer_evidence.get("factor_pack_manifest")
    if not isinstance(source_factor_manifest, Mapping):
        raise StagePreparationError("Source report omits selected factor evidence.")
    factor_state_sha = stage._require_sha256(
        source_factor_manifest.get("factor_state_sha256"),
        "source_run_report.factor_state_sha256",
    )
    compiled_state_sha = stage._require_sha256(
        source_report.get("rollback_state_sha256"),
        "source_run_report.rollback_state_sha256",
    )

    output.mkdir(parents=True)
    split_output = output / "splits"
    split_report = continuation_splits.prepare_continuation_splits(
        output_dir=split_output,
        original_split_dir=original_split_root,
        corpus_path=corpus_path,
        tokenizer_path=parent,
    )
    parent_tree = dict(stage._artifact_tree(parent))
    target_tree = dict(stage._artifact_tree(target))
    required = _default_target_required_files()
    for relative, expected in required.items():
        if stage._file_sha256(target / relative) != expected:
            raise StagePreparationError(
                f"Canonical target required-file hash changed: {relative}"
            )
    split_bindings = {
        role: {
            "path": str(split_output / f"{role}.json"),
            "sha256": stage._file_sha256(split_output / f"{role}.json"),
            "expected_count": continuation_splits.OUTPUT_COUNTS[role],
        }
        for role in ("fit", "candidate", "sealed")
    }
    compiler_split_path = original_split_root / "fit.json"
    compiler_split_sha = continuation_splits.PINNED_ORIGINAL_SPLIT_SHA256S["fit"]
    if stage._file_sha256(compiler_split_path) != compiler_split_sha:
        raise StagePreparationError("Original exact compiler fit split changed.")
    trainer_spec = importlib.util.find_spec(stage.CANONICAL_TRAINER_MODULE)
    cache_spec = importlib.util.find_spec("neural_forge.hf_compiled_child_cache")
    if (
        trainer_spec is None
        or trainer_spec.origin is None
        or cache_spec is None
        or cache_spec.origin is None
    ):
        raise StagePreparationError(
            "Canonical trainer/cache implementation sources are unavailable."
        )
    thresholds = {
        "fit": {
            "minimum_action_accuracy": 0.65,
            "minimum_exact_rollout_fraction": 0.25,
            "minimum_token_match_fraction": 0.40,
            "minimum_lcp_fraction": 0.30,
            "minimum_action_count": 768,
            "minimum_rollout_count": 96,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
        "candidate": {
            "minimum_action_accuracy": 0.50,
            "minimum_exact_rollout_fraction": 0.25,
            "minimum_token_match_fraction": 0.50,
            "minimum_lcp_fraction": 0.35,
            "minimum_action_count": 192,
            "minimum_rollout_count": 24,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
        "sealed": {
            "minimum_action_accuracy": 0.60,
            "minimum_exact_rollout_fraction": 0.1875,
            "minimum_token_match_fraction": 0.35,
            "minimum_lcp_fraction": 0.30,
            "minimum_action_count": 256,
            "minimum_rollout_count": 32,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
    }
    training = {
        "steps": 32,
        "learning_rate": 3e-4,
        "checkpoint_steps": [2, 4, 8, 12, 16, 24],
        "fit_horizon": 8,
        "evaluation_horizon": 8,
        "fit_prompts_per_step": 8,
        "temperature": 1.0,
        "action_loss_weight": 1.0,
        "distribution_loss_weight": 0.05,
        "action_margin_loss_weight": 0.25,
        "action_margin": 1.0,
        "repetition_excess_loss_weight": 0.20,
        "repetition_window": 8,
        "parent_lane_loss_weight": 3.0,
        "child_lane_loss_weight": 1.0,
        "parent_only_steps": 8,
        "warmup_fit_horizon": 4,
        "late_position_loss_multiplier": 1.0,
        "solved_prefix_loss_multiplier": 0.5,
        "divergence_loss_multiplier": 4.0,
        "post_divergence_loss_multiplier": 2.0,
        "post_divergence_positions": 1,
        "child_lane_starts_at_divergence": True,
        "max_child_lane_prefixes": 3,
        "child_repeat_stop_run_length": 3,
        "qk_learning_rate_scale": 0.5,
        "vo_learning_rate_scale": 1.0,
        "gud_learning_rate_scale": 0.5,
        "right_factor_learning_rate_scale": 0.25,
        "learning_rate_decay_step_1": 8,
        "learning_rate_decay_scale_1": 2.0 / 3.0,
        "learning_rate_decay_step_2": 20,
        "learning_rate_decay_scale_2": 0.5,
        "gud_frozen_steps": 8,
        "top_k": 5,
        "gradient_clip_norm": 0.5,
        "max_prompt_tokens": 128,
        "max_train_prefixes": 4096,
        "max_evaluation_prefixes": 12000,
        "max_checkpoints": 8,
        "selection_policy": "rollout-first-acquisition",
    }
    preregistration = {
        "schema": stage.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "seed": 0,
        "implementation": {
            "runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
            "trainer_module": stage.CANONICAL_TRAINER_MODULE,
            "trainer_sha256": stage._file_sha256(Path(trainer_spec.origin)),
            "compiler_cache_sha256": stage._file_sha256(Path(cache_spec.origin)),
        },
        "parent": {
            "path": str(parent),
            "exact_unique_parameters": stage.QWEN3_4B_EXACT_PARAMETERS,
            "tree_sha256": parent_tree["tree_sha256"],
        },
        "blank_target": {
            "path": str(target),
            "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
            "tree_sha256": target_tree["tree_sha256"],
            "required_files": required,
        },
        "compiler_split": {
            "path": str(compiler_split_path),
            "sha256": compiler_split_sha,
            "expected_count": continuation_splits.ORIGINAL_COUNTS["fit"],
        },
        "splits": split_bindings,
        "thresholds": thresholds,
        "compiler": {
            "capture_batch_size": 1,
            "max_prompt_tokens": 128,
            "canonical_validation_count": 4,
            "compiler_seed": 0,
            "bootstrap_kwargs": {
                "heldout_gate_policy": "output_only",
                "max_operator_delta_ratio": 0.1,
                "trace_pooling": "last_valid",
            },
            "candidate_split_used_during_bootstrap": False,
            "sealed_split_used_during_bootstrap": False,
        },
        "training": training,
        "compiler_cache": {"root": str(cache_root)},
        "continuation": {
            "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
            "source_run_report": {
                "path": str(source_report_path),
                "sha256": source_report_sha,
            },
            "expected_factor_state_sha256": factor_state_sha,
            "expected_compiler_report_sha256": stage._canonical_sha256(
                source_compiler_report
            ),
            "expected_compiled_state_sha256": compiled_state_sha,
        },
        "corpus_provenance": {
            "source_sha256": continuation_splits.PINNED_CORPUS_SHA256,
            "source_line_count": continuation_splits.PINNED_CORPUS_LINE_COUNT,
            "continuation_split_build_content_sha256": split_report[
                "build_content_sha256"
            ],
            "original_fit_replayed": 24,
            "fresh_fit": 72,
            "fresh_candidate": 24,
            "original_sealed_unexposed_and_reused": 16,
            "fresh_sealed": 16,
            "old_candidate_diagnostic_only": True,
        },
        "runtime": {
            "dtype": "float16",
            "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
            "explicit_execute_required": True,
        },
    }
    prereg_path = output / "preregistration.json"
    _write_json_exclusive(prereg_path, preregistration)
    prereg_sha = stage._file_sha256(prereg_path)
    authenticated = stage.authenticate_preregistration(prereg_path, prereg_sha)
    report = {
        "schema": BUILD_SCHEMA,
        "mode": "authenticated-factor-continuation",
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": authenticated.evaluation_contract_sha256,
        "compiler_split_sha256": compiler_split_sha,
        "behavior_split_sha256s": {
            role: split_bindings[role]["sha256"]
            for role in ("fit", "candidate", "sealed")
        },
        "source_run_report_sha256": source_report_sha,
        "source_factor_pack_sha256": factor_sha,
        "source_factor_state_sha256": factor_state_sha,
        "shared_compiler_cache_root": str(cache_root),
        "model_weights_loaded": False,
        "model_checkpoint_created": False,
        "training_launched": False,
        "ready_for_explicit_preflight": True,
    }
    _write_json_exclusive(output / "build_report.json", report)
    return report


def prepare_lexical_continuation_stage(
    *,
    output_dir: Path,
    parent: Path,
    target: Path,
    source_run_report: Path,
    source_split_dir: Path = continuation_splits.DEFAULT_RESIDUAL_ATTENTION_SPLIT_DIR,
    original_split_dir: Path = continuation_splits.DEFAULT_ORIGINAL_SPLIT_DIR,
    compiler_cache_root: Path = DEFAULT_COMPILER_CACHE_ROOT,
    corpus_path: Path = DEFAULT_CORPUS,
) -> dict[str, Any]:
    """Seal the cache-only lexical + residual-attention continuation."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise StagePreparationError("output_dir must not already exist.")
    parent = parent.expanduser().resolve()
    target = target.expanduser().resolve()
    source_report_path = source_run_report.expanduser().resolve()
    source_split_root = source_split_dir.expanduser().resolve()
    original_split_root = original_split_dir.expanduser().resolve()
    cache_root = compiler_cache_root.expanduser().resolve()
    source_report = stage._strict_json(source_report_path)
    if not isinstance(source_report, Mapping):
        raise StagePreparationError("Lexical continuation source report is malformed.")
    trainer_evidence = source_report.get("trainer_evidence")
    if (
        source_report.get("status") != "rejected"
        or source_report.get("rollback_exact") is not True
        or not isinstance(trainer_evidence, Mapping)
        or trainer_evidence.get("sealed_evaluation_unlocked") is not False
        or trainer_evidence.get("sealed_metrics") is not None
    ):
        raise StagePreparationError(
            "Lexical continuation requires an exactly rolled-back source whose "
            "sealed domain is still locked."
        )
    source_splits = source_report.get("splits")
    if not isinstance(source_splits, Mapping) or set(source_splits) != {
        "fit",
        "candidate",
        "sealed",
    }:
        raise StagePreparationError("Source report omits its behavior split bindings.")
    for role in ("fit", "candidate", "sealed"):
        binding = source_splits.get(role)
        if (
            not isinstance(binding, Mapping)
            or binding.get("sha256")
            != continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[role]
            or binding.get("row_count")
            != continuation_splits.RESIDUAL_ATTENTION_COUNTS[role]
        ):
            raise StagePreparationError(
                f"Source report {role} split is not the pinned residual-attention "
                "artifact."
            )
    factor_record = source_report.get("factor_pack")
    if not isinstance(factor_record, Mapping):
        raise StagePreparationError("Source report omits its retained factor pack.")
    factor_path = Path(str(factor_record.get("path", ""))).expanduser().resolve()
    factor_sha = stage._require_sha256(
        factor_record.get("sha256"), "source_run_report.factor_pack.sha256"
    )
    if not factor_path.is_file() or stage._file_sha256(factor_path) != factor_sha:
        raise StagePreparationError("Source factor pack is absent or changed.")
    source_factor_manifest = trainer_evidence.get("factor_pack_manifest")
    if not isinstance(source_factor_manifest, Mapping):
        raise StagePreparationError("Source report omits selected factor evidence.")
    factor_state_sha = stage._require_sha256(
        source_factor_manifest.get("factor_state_sha256"),
        "source_run_report.factor_state_sha256",
    )
    if source_factor_manifest.get("schema") not in {
        None,
        stage.TRAINER_FACTOR_PACK_SCHEMA,
    }:
        raise StagePreparationError(
            "The first lexical continuation must start from a v1 QKVO/GUD pack."
        )
    source_compiler_report = source_report.get("compiler_report")
    if not isinstance(source_compiler_report, Mapping):
        raise StagePreparationError("Source report omits compiler evidence.")
    compiler_report_sha = stage._canonical_sha256(source_compiler_report)
    compiled_state_sha = stage._require_sha256(
        source_report.get("rollback_state_sha256"),
        "source_run_report.rollback_state_sha256",
    )
    cache_evidence = source_report.get("compiled_child_cache")
    required_cache_fields = {
        "compiler_identity_sha256",
        "receipt_sha256",
        "compiler_report_sha256",
        "state_dict_sha256",
        "artifact_tree_sha256",
    }
    if (
        not isinstance(cache_evidence, Mapping)
        or cache_evidence.get("disposition") not in {"published", "hit"}
        or cache_evidence.get("verified_before_factor_install") is not True
        or not required_cache_fields.issubset(cache_evidence)
    ):
        raise StagePreparationError(
            "Lexical continuation requires the source's verified immutable cache hit."
        )
    cache_hashes = {
        name: stage._require_sha256(
            cache_evidence.get(name), f"source_run_report.compiled_child_cache.{name}"
        )
        for name in required_cache_fields
    }
    if cache_hashes["compiler_report_sha256"] != compiler_report_sha:
        raise StagePreparationError(
            "Source compiler report and immutable cache report hash disagree."
        )
    cache_path = Path(str(cache_evidence.get("path", ""))).expanduser().resolve()
    if (
        not cache_path.is_dir()
        or cache_path.parent != cache_root
        or cache_path.name != cache_hashes["compiler_identity_sha256"]
    ):
        raise StagePreparationError(
            "Source compiler cache path does not match the requested shared root."
        )

    target_config = stage._strict_json(target / "config.json")
    if not isinstance(target_config, Mapping) or (
        target_config.get("vocab_size") != stage.LEXICAL_VOCAB_SIZE
        or target_config.get("hidden_size") != stage.LEXICAL_HIDDEN_SIZE
        or target_config.get("tie_word_embeddings") is not False
    ):
        raise StagePreparationError(
            "Canonical target config is not the pinned untied lexical boundary."
        )
    parent_tree = dict(stage._artifact_tree(parent))
    target_tree = dict(stage._artifact_tree(target))
    required = _default_target_required_files()
    for relative, expected in required.items():
        if stage._file_sha256(target / relative) != expected:
            raise StagePreparationError(
                f"Canonical target required-file hash changed: {relative}"
            )

    output.mkdir(parents=True)
    split_output = output / "splits"
    split_report = continuation_splits.prepare_lexical_continuation_splits(
        output_dir=split_output,
        source_split_dir=source_split_root,
        corpus_path=corpus_path,
        tokenizer_path=parent,
    )
    split_bindings = {
        "fit": {
            "path": str(split_output / "fit.json"),
            "sha256": stage._file_sha256(split_output / "fit.json"),
            "expected_count": continuation_splits.LEXICAL_CONTINUATION_COUNTS["fit"],
        },
        "candidate": {
            "path": str(split_output / "candidate.json"),
            "sha256": stage._file_sha256(split_output / "candidate.json"),
            "expected_count": continuation_splits.LEXICAL_CONTINUATION_COUNTS[
                "candidate"
            ],
        },
        "sealed": {
            "path": str(source_split_root / "sealed.json"),
            "sha256": continuation_splits.PINNED_RESIDUAL_ATTENTION_SPLIT_SHA256S[
                "sealed"
            ],
            "expected_count": continuation_splits.LEXICAL_CONTINUATION_COUNTS[
                "sealed"
            ],
        },
    }
    compiler_split_path = original_split_root / "fit.json"
    compiler_split_sha = continuation_splits.PINNED_ORIGINAL_SPLIT_SHA256S["fit"]
    if stage._file_sha256(compiler_split_path) != compiler_split_sha:
        raise StagePreparationError("Original exact compiler fit split changed.")

    module_specs = {
        "trainer_sha256": stage.CANONICAL_TRAINER_MODULE,
        "compiler_cache_sha256": "neural_forge.hf_compiled_child_cache",
        "lexical_adapter_sha256": (
            "neural_forge.hf_dense_lexical_residual_adapter"
        ),
        "progressive_bridge_sha256": "neural_forge.hf_progressive_target_bridge",
        "qkvo_adapter_sha256": (
            "neural_forge.hf_dense_gqa_qkvo_residual_adapter"
        ),
        "swiglu_adapter_sha256": (
            "neural_forge.hf_dense_swiglu_projection_residual_adapter"
        ),
    }
    implementation_hashes: dict[str, str] = {}
    for field_name, module_name in module_specs.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise StagePreparationError(
                f"Required lexical continuation module {module_name!r} is unavailable."
            )
        implementation_hashes[field_name] = stage._file_sha256(Path(spec.origin))

    thresholds = {
        "fit": {
            "minimum_action_accuracy": 0.65,
            "minimum_exact_rollout_fraction": 0.25,
            "minimum_token_match_fraction": 0.40,
            "minimum_lcp_fraction": 0.30,
            "minimum_action_count": 960,
            "minimum_rollout_count": 120,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
        "candidate": {
            "minimum_action_accuracy": 0.50,
            "minimum_exact_rollout_fraction": 0.25,
            "minimum_token_match_fraction": 0.50,
            "minimum_lcp_fraction": 0.35,
            "minimum_action_count": 192,
            "minimum_rollout_count": 24,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
        "sealed": {
            "minimum_action_accuracy": 0.60,
            "minimum_exact_rollout_fraction": 0.1875,
            "minimum_token_match_fraction": 0.35,
            "minimum_lcp_fraction": 0.30,
            "minimum_action_count": 256,
            "minimum_rollout_count": 32,
            "maximum_repeat_fraction_excess": 0.20,
            "maximum_adjacent_repeat_fraction_excess": 0.25,
            "maximum_longest_run_excess": 2,
        },
    }
    training = {
        "steps": 48,
        "learning_rate": 1e-4,
        "checkpoint_steps": [2, 4, 8, 12, 16, 24, 32, 40, 48],
        "fit_horizon": 8,
        "evaluation_horizon": 8,
        "fit_prompts_per_step": 12,
        "temperature": 1.0,
        "action_loss_weight": 1.0,
        "distribution_loss_weight": 0.05,
        "action_margin_loss_weight": 0.25,
        "action_margin": 1.0,
        "repetition_excess_loss_weight": 0.40,
        "repetition_window": 8,
        "parent_lane_loss_weight": 3.0,
        "child_lane_loss_weight": 1.0,
        "parent_only_steps": 8,
        "warmup_fit_horizon": 4,
        "late_position_loss_multiplier": 1.0,
        "solved_prefix_loss_multiplier": 0.25,
        "divergence_loss_multiplier": 6.0,
        "post_divergence_loss_multiplier": 3.0,
        "post_divergence_positions": 2,
        "child_lane_starts_at_divergence": True,
        "max_child_lane_prefixes": 3,
        "child_repeat_stop_run_length": 3,
        "qk_learning_rate_scale": 0.25,
        "vo_learning_rate_scale": 0.50,
        "gud_learning_rate_scale": 0.25,
        "lexical_embedding_learning_rate_scale": 0.25,
        "lexical_head_learning_rate_scale": 1.0,
        "right_factor_learning_rate_scale": 0.25,
        "lexical_right_factor_learning_rate_scale": 0.25,
        "learning_rate_decay_step_1": 8,
        "learning_rate_decay_scale_1": 0.75,
        "learning_rate_decay_step_2": 24,
        "learning_rate_decay_scale_2": 2.0 / 3.0,
        "learning_rate_decay_step_3": 40,
        "learning_rate_decay_scale_3": 0.5,
        "qkvo_frozen_steps": 8,
        "gud_frozen_steps": 24,
        "lexical_embedding_frozen_steps": 24,
        "lexical_head_frozen_steps": 0,
        "top_k": 5,
        "gradient_clip_norm": 0.5,
        "max_prompt_tokens": 128,
        "max_train_prefixes": 9216,
        "max_evaluation_prefixes": 16000,
        "max_checkpoints": 10,
        "selection_policy": "rollout-first-acquisition",
    }
    preregistration = {
        "schema": stage.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "seed": 0,
        "implementation": {
            "runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
            "trainer_module": stage.CANONICAL_TRAINER_MODULE,
            **implementation_hashes,
        },
        "parent": {
            "path": str(parent),
            "exact_unique_parameters": stage.QWEN3_4B_EXACT_PARAMETERS,
            "tree_sha256": parent_tree["tree_sha256"],
        },
        "blank_target": {
            "path": str(target),
            "exact_unique_parameters": stage.QWEN3_1P4B_EXACT_PARAMETERS,
            "tree_sha256": target_tree["tree_sha256"],
            "required_files": required,
        },
        "compiler_split": {
            "path": str(compiler_split_path),
            "sha256": compiler_split_sha,
            "expected_count": continuation_splits.ORIGINAL_COUNTS["fit"],
        },
        "splits": split_bindings,
        "thresholds": thresholds,
        "compiler": {
            "capture_batch_size": 1,
            "max_prompt_tokens": 128,
            "canonical_validation_count": 4,
            "compiler_seed": 0,
            "bootstrap_kwargs": {
                "heldout_gate_policy": "output_only",
                "max_operator_delta_ratio": 0.1,
                "trace_pooling": "last_valid",
            },
            "candidate_split_used_during_bootstrap": False,
            "sealed_split_used_during_bootstrap": False,
        },
        "training": training,
        "lexical_residual": {
            "rank": stage.LEXICAL_RANK,
            "vocab_size": stage.LEXICAL_VOCAB_SIZE,
            "hidden_size": stage.LEXICAL_HIDDEN_SIZE,
            "untied_required": True,
        },
        "compiler_cache": {
            "root": str(cache_root),
            "require_hit": True,
            "expected_compiler_identity_sha256": cache_hashes[
                "compiler_identity_sha256"
            ],
            "expected_receipt_sha256": cache_hashes["receipt_sha256"],
            "expected_compiler_report_sha256": cache_hashes[
                "compiler_report_sha256"
            ],
            "expected_state_dict_sha256": cache_hashes["state_dict_sha256"],
            "expected_artifact_tree_sha256": cache_hashes[
                "artifact_tree_sha256"
            ],
        },
        "continuation": {
            "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
            "source_run_report": {
                "path": str(source_report_path),
                "sha256": stage._file_sha256(source_report_path),
            },
            "expected_factor_state_sha256": factor_state_sha,
            "expected_compiler_report_sha256": compiler_report_sha,
            "expected_compiled_state_sha256": compiled_state_sha,
        },
        "corpus_provenance": {
            "source_sha256": continuation_splits.PINNED_CORPUS_SHA256,
            "source_line_count": continuation_splits.PINNED_CORPUS_LINE_COUNT,
            "lexical_continuation_split_build_content_sha256": split_report[
                "build_content_sha256"
            ],
            "prior_fit_training_replay": 96,
            "exposed_prior_candidate_training_only": 24,
            "fresh_candidate": 24,
            "sealed_reused_by_exact_hash": 32,
        },
        "runtime": {
            "dtype": "float16",
            "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
            "explicit_execute_required": True,
        },
    }
    prereg_path = output / "preregistration.json"
    _write_json_exclusive(prereg_path, preregistration)
    prereg_sha = stage._file_sha256(prereg_path)
    authenticated = stage.authenticate_preregistration(prereg_path, prereg_sha)
    report = {
        "schema": BUILD_SCHEMA,
        "mode": "authenticated-lexical-attention-continuation",
        "preregistration": str(prereg_path),
        "preregistration_sha256": prereg_sha,
        "evaluation_contract_sha256": authenticated.evaluation_contract_sha256,
        "compiler_split_sha256": compiler_split_sha,
        "behavior_split_sha256s": {
            role: split_bindings[role]["sha256"]
            for role in ("fit", "candidate", "sealed")
        },
        "source_run_report_sha256": stage._file_sha256(source_report_path),
        "source_factor_pack_sha256": factor_sha,
        "source_factor_state_sha256": factor_state_sha,
        "lexical_rank": stage.LEXICAL_RANK,
        "shared_compiler_cache_root": str(cache_root),
        "required_compiler_cache_identity_sha256": cache_hashes[
            "compiler_identity_sha256"
        ],
        "model_weights_loaded": False,
        "model_checkpoint_created": False,
        "training_launched": False,
        "ready_for_explicit_preflight": True,
    }
    _write_json_exclusive(output / "build_report.json", report)
    return report


def _json_object_copy(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise StagePreparationError(f"{name} must be a JSON object.")
    try:
        copied = json.loads(
            json.dumps(
                dict(value),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as error:
        raise StagePreparationError(f"{name} is not canonical JSON.") from error
    if not isinstance(copied, dict):  # pragma: no cover - mapping invariant
        raise StagePreparationError(f"{name} must remain a JSON object.")
    return copied


def _reported_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise StagePreparationError(f"{name} must be a nonempty path string.")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _historical_implementation_authentication(
    source_preregistration: Mapping[str, Any],
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the persisted preflight proof without equating old and live code."""

    implementation = source_preregistration.get("implementation")
    authentication = source_report.get("implementation_authentication")
    source_fields = {
        "runner_sha256": "runner",
        "trainer_sha256": "trainer",
        "compiler_cache_sha256": "compiler_cache",
        "lexical_adapter_sha256": "lexical_adapter",
        "progressive_bridge_sha256": "progressive_bridge",
        "qkvo_adapter_sha256": "qkvo_adapter",
        "swiglu_adapter_sha256": "swiglu_adapter",
    }
    expected_implementation_fields = {
        "trainer_module",
        *source_fields,
    }
    expected_authentication_fields = {
        "authenticated_before_model_access",
        *source_fields.values(),
    }
    if (
        not isinstance(implementation, Mapping)
        or set(implementation) != expected_implementation_fields
        or implementation.get("trainer_module") != stage.CANONICAL_TRAINER_MODULE
        or not isinstance(authentication, Mapping)
        or set(authentication) != expected_authentication_fields
        or authentication.get("authenticated_before_model_access") is not True
    ):
        raise StagePreparationError(
            "Source preregistration does not contain its exact historical lexical "
            "implementation proof."
        )
    for source_field, evidence_field in source_fields.items():
        expected_sha = stage._require_sha256(
            implementation.get(source_field),
            f"source_preregistration.implementation.{source_field}",
        )
        evidence = authentication.get(evidence_field)
        evidence_path = (
            None
            if not isinstance(evidence, Mapping)
            else Path(str(evidence.get("path", ""))).expanduser()
        )
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("sha256") != expected_sha
            or evidence_path is None
            or not evidence_path.is_absolute()
            or evidence_path.is_symlink()
            or not evidence_path.is_file()
        ):
            raise StagePreparationError(
                f"Source implementation evidence {evidence_field!r} does not "
                "authenticate its preregistered SHA-256."
            )
    trainer_evidence = authentication["trainer"]
    if trainer_evidence.get("module") != stage.CANONICAL_TRAINER_MODULE:
        raise StagePreparationError(
            "Source trainer implementation evidence names the wrong module."
        )
    return _json_object_copy(
        authentication, "source_report.implementation_authentication"
    )


def prepare_output_jacobian_audit_stage(
    *,
    output_dir: Path,
    source_run_report: Path,
) -> dict[str, Any]:
    """Seal one model-free audit over the terminal lexical-v2 factor state."""

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise StagePreparationError("output_dir must not already exist.")
    source_report_path = source_run_report.expanduser().resolve()
    if not source_report_path.is_file():
        raise StagePreparationError("Output-Jacobian source report does not exist.")
    source_report_sha = stage._file_sha256(source_report_path)
    source_report = stage._strict_json(source_report_path)
    if not isinstance(source_report, Mapping):
        raise StagePreparationError("Output-Jacobian source report is malformed.")

    trainer_evidence = source_report.get("trainer_evidence")
    pre_bake_decisions = source_report.get("pre_bake_behavior_decisions")
    if (
        source_report.get("schema") != stage.RUN_SCHEMA
        or source_report.get("status") != "rejected"
        or source_report.get("reason") != "fit-gate-rejected"
        or source_report.get("rollback_exact") is not True
        or source_report.get("model_checkpoint_written") is not False
        or source_report.get("standalone_compaction_passed") is not False
        or source_report.get("publication") is not None
        or source_report.get("post_bake_behavior_decisions") is not None
        or source_report.get("native_factor_bake_performed") not in {None, False}
        or source_report.get("fresh_reload_certificate") is not None
    ):
        raise StagePreparationError(
            "Output-Jacobian audit requires the terminal pre-bake fit-gate "
            "rejection with exact rollback and no model checkpoint."
        )
    if (
        not isinstance(trainer_evidence, Mapping)
        or not isinstance(trainer_evidence.get("candidate_metrics"), Mapping)
        or trainer_evidence.get("candidate_gate_passed") is not False
        or trainer_evidence.get("sealed_evaluation_unlocked") is not False
        or trainer_evidence.get("sealed_metrics") is not None
        or not isinstance(pre_bake_decisions, Mapping)
        or not isinstance(pre_bake_decisions.get("candidate"), Mapping)
        or pre_bake_decisions["candidate"].get("passed") is not False
        or pre_bake_decisions.get("sealed") is not None
    ):
        raise StagePreparationError(
            "Source candidate must be evaluated and exposed while the sealed "
            "domain remains locked."
        )
    trainer_evidence_sha = stage._require_sha256(
        source_report.get("trainer_evidence_sha256"),
        "source_run_report.trainer_evidence_sha256",
    )
    if stage._canonical_sha256(trainer_evidence) != trainer_evidence_sha:
        raise StagePreparationError("Source trainer evidence hash does not verify.")

    factor_record = source_report.get("factor_pack")
    if not isinstance(factor_record, Mapping) or set(factor_record) != {
        "path",
        "sha256",
    }:
        raise StagePreparationError("Source report factor-pack binding is malformed.")
    factor_path = _reported_path(
        factor_record.get("path"),
        base=source_report_path.parent,
        name="source_run_report.factor_pack.path",
    )
    factor_sha = stage._require_sha256(
        factor_record.get("sha256"), "source_run_report.factor_pack.sha256"
    )
    if not factor_path.is_file() or stage._file_sha256(factor_path) != factor_sha:
        raise StagePreparationError("Source lexical-v2 factor pack is absent or changed.")
    try:
        envelope = torch.load(factor_path, map_location="cpu", weights_only=True)
        (
            factor_payload,
            factor_state_sha,
            envelope_provenance,
            factor_envelope_schema,
            trainer_factor_schema,
            factor_lexical_rank,
        ) = stage._authenticate_continuation_factor_payload(
            envelope,
            lexical_rank=stage.LEXICAL_RANK,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise StagePreparationError(
            "Source factor pack is not an authenticated exact lexical-v2 payload."
        ) from error
    manifest = trainer_evidence.get("factor_pack_manifest")
    expected_manifest_fields = {
        "schema",
        "factor_count",
        "factors",
        "factor_state_sha256",
        "parent_state_included",
        "native_child_state_included",
        "optimizer_state_included",
        "lexical_rank",
    }
    payload_manifest = {
        key: value for key, value in factor_payload.items() if key != "factor_tensors"
    }
    if (
        factor_envelope_schema != stage.FACTOR_PACK_V2_SCHEMA
        or trainer_factor_schema != stage.TRAINER_FACTOR_PACK_V2_SCHEMA
        or factor_lexical_rank != stage.LEXICAL_RANK
        or not isinstance(manifest, Mapping)
        or set(manifest) != expected_manifest_fields
        or manifest.get("schema") != stage.TRAINER_FACTOR_PACK_V2_SCHEMA
        or manifest.get("factor_count") != stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT
        or manifest.get("lexical_rank") != stage.LEXICAL_RANK
        or manifest.get("factor_state_sha256") != factor_state_sha
        or any(
            manifest.get(name) is not False
            for name in (
                "parent_state_included",
                "native_child_state_included",
                "optimizer_state_included",
            )
        )
        or not isinstance(manifest.get("factors"), list)
        or len(manifest["factors"]) != stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT
        or dict(manifest) != payload_manifest
    ):
        raise StagePreparationError(
            "Source trainer evidence and retained pack do not prove the exact "
            "rank-32 18-factor v2 state."
        )
    factor_acquisition = source_report.get("factor_acquisition")
    if (
        not isinstance(factor_acquisition, Mapping)
        or factor_acquisition.get("selected_factor_state_sha256")
        != factor_state_sha
        or factor_acquisition.get("target_native_factors_acquired_behavior") is not True
    ):
        raise StagePreparationError(
            "Source report does not bind the retained v2 factors to behavior "
            "acquisition."
        )

    source_prereg_path = _reported_path(
        source_report.get("preregistration"),
        base=source_report_path.parent,
        name="source_run_report.preregistration",
    )
    source_prereg_sha = stage._require_sha256(
        source_report.get("preregistration_sha256"),
        "source_run_report.preregistration_sha256",
    )
    source_evaluation_sha = stage._require_sha256(
        source_report.get("evaluation_contract_sha256"),
        "source_run_report.evaluation_contract_sha256",
    )
    if (
        not source_prereg_path.is_file()
        or stage._file_sha256(source_prereg_path) != source_prereg_sha
    ):
        raise StagePreparationError(
            "Terminal report's source preregistration path/SHA-256 does not verify."
        )
    source_prereg = stage._strict_json(source_prereg_path)
    if not isinstance(source_prereg, Mapping):
        raise StagePreparationError("Source preregistration is malformed.")
    if (
        source_prereg.get("schema") != stage.PREREGISTRATION_SCHEMA
        or source_prereg.get("sealed_before_model_access") is not True
        or "output_jacobian_audit" in source_prereg
        or source_prereg.get("lexical_residual")
        != {
            "rank": stage.LEXICAL_RANK,
            "vocab_size": stage.LEXICAL_VOCAB_SIZE,
            "hidden_size": stage.LEXICAL_HIDDEN_SIZE,
            "untied_required": True,
        }
    ):
        raise StagePreparationError(
            "Terminal source is not the exact lexical stage preregistration."
        )

    source_base = source_prereg_path.parent
    try:
        parent_binding = stage._artifact_binding(
            source_prereg.get("parent"),
            name="parent",
            base=source_base,
            expected_parameters=stage.QWEN3_4B_EXACT_PARAMETERS,
        )
        target_binding = stage._artifact_binding(
            source_prereg.get("blank_target"),
            name="blank_target",
            base=source_base,
            expected_parameters=stage.QWEN3_1P4B_EXACT_PARAMETERS,
        )
        parent_authentication = stage._authenticate_artifact(parent_binding)
        target_authentication = stage._authenticate_artifact(target_binding)
    except stage.ExactTargetNativeStageError as error:
        raise StagePreparationError(
            "Source parent/target artifact authentication failed."
        ) from error
    reported_parent = source_report.get("parent")
    reported_target = source_report.get("target")
    if (
        not isinstance(reported_parent, Mapping)
        or reported_parent.get("path") != str(parent_binding.path)
        or reported_parent.get("exact_unique_parameters")
        != stage.QWEN3_4B_EXACT_PARAMETERS
        or not isinstance(reported_target, Mapping)
        or reported_target.get("path") != str(target_binding.path)
        or reported_target.get("exact_unique_parameters")
        != stage.QWEN3_1P4B_EXACT_PARAMETERS
    ):
        raise StagePreparationError(
            "Source report parent/target bindings disagree with its preregistration."
        )
    target_config = stage._strict_json(target_binding.path / "config.json")
    if not isinstance(target_config, Mapping) or (
        target_config.get("vocab_size") != stage.LEXICAL_VOCAB_SIZE
        or target_config.get("hidden_size") != stage.LEXICAL_HIDDEN_SIZE
        or target_config.get("tie_word_embeddings") is not False
    ):
        raise StagePreparationError(
            "Source target is not the pinned untied lexical boundary."
        )

    raw_splits = source_prereg.get("splits")
    reported_splits = source_report.get("splits")
    if (
        not isinstance(raw_splits, Mapping)
        or set(raw_splits) != set(OUTPUT_JACOBIAN_AUDIT_COUNTS)
        or not isinstance(reported_splits, Mapping)
        or set(reported_splits) != set(OUTPUT_JACOBIAN_AUDIT_COUNTS)
    ):
        raise StagePreparationError("Source behavior split bindings are incomplete.")
    authenticated_splits: dict[str, Any] = {}
    split_bindings: dict[str, dict[str, Any]] = {}
    for role, expected_count in OUTPUT_JACOBIAN_AUDIT_COUNTS.items():
        try:
            split = stage._load_split(raw_splits[role], name=role, base=source_base)
        except stage.ExactTargetNativeStageError as error:
            raise StagePreparationError(
                f"Source {role} split failed content authentication."
            ) from error
        reported = reported_splits[role]
        if (
            len(split.row_ids) != expected_count
            or not isinstance(reported, Mapping)
            or set(reported) != {"path", "sha256", "row_count"}
            or reported.get("path") != str(split.path)
            or reported.get("sha256") != split.sha256
            or reported.get("row_count") != expected_count
        ):
            raise StagePreparationError(
                f"Source {role} split does not match its terminal {expected_count}-row "
                "binding."
            )
        authenticated_splits[role] = split
        split_bindings[role] = {
            "path": str(split.path),
            "sha256": split.sha256,
            "expected_count": expected_count,
        }
    for left_index, left in enumerate(("fit", "candidate", "sealed")):
        for right in ("fit", "candidate", "sealed")[left_index + 1 :]:
            if (
                set(authenticated_splits[left].row_ids)
                & set(authenticated_splits[right].row_ids)
                or set(authenticated_splits[left].prompt_sha256s)
                & set(authenticated_splits[right].prompt_sha256s)
            ):
                raise StagePreparationError(
                    f"Source split identities overlap between {left} and {right}."
                )

    raw_compiler_split = source_prereg.get("compiler_split")
    if not isinstance(raw_compiler_split, Mapping):
        raise StagePreparationError("Source preregistration omits compiler_split.")
    try:
        compiler_split = stage._load_split(
            raw_compiler_split, name="fit", base=source_base
        )
    except stage.ExactTargetNativeStageError as error:
        raise StagePreparationError(
            "Source immutable compiler split failed authentication."
        ) from error
    reported_compiler_split = source_report.get("compiler_split")
    if (
        not isinstance(reported_compiler_split, Mapping)
        or set(reported_compiler_split)
        != {"path", "sha256", "row_count", "reuses_behavior_fit"}
        or reported_compiler_split.get("path") != str(compiler_split.path)
        or reported_compiler_split.get("sha256") != compiler_split.sha256
        or reported_compiler_split.get("row_count") != len(compiler_split.row_ids)
        or reported_compiler_split.get("reuses_behavior_fit") is not False
    ):
        raise StagePreparationError(
            "Source compiler split disagrees with its terminal immutable binding."
        )
    for heldout_role in ("candidate", "sealed"):
        heldout = authenticated_splits[heldout_role]
        if (
            set(compiler_split.row_ids) & set(heldout.row_ids)
            or set(compiler_split.prompt_sha256s) & set(heldout.prompt_sha256s)
        ):
            raise StagePreparationError(
                f"Compiler split overlaps behavior {heldout_role}."
            )

    thresholds = source_prereg.get("thresholds")
    reported_thresholds = source_report.get("thresholds")
    if (
        not isinstance(thresholds, Mapping)
        or set(thresholds) != {"fit", "candidate", "sealed"}
        or not isinstance(reported_thresholds, Mapping)
    ):
        raise StagePreparationError("Source threshold bindings are incomplete.")
    normalized_thresholds: dict[str, Any] = {}
    for role in ("fit", "candidate", "sealed"):
        try:
            normalized_thresholds[role] = stage.BehaviorThresholds.from_mapping(
                thresholds[role], f"source_preregistration.thresholds.{role}"
            ).as_dict()
        except stage.ExactTargetNativeStageError as error:
            raise StagePreparationError(
                f"Source {role} thresholds are malformed."
            ) from error
    if normalized_thresholds != dict(reported_thresholds):
        raise StagePreparationError(
            "Source report thresholds disagree with its preregistration."
        )
    compiler = _json_object_copy(source_prereg.get("compiler"), "source compiler")
    training = _json_object_copy(source_prereg.get("training"), "source training")
    if training.get("selection_policy") != "rollout-first-acquisition":
        raise StagePreparationError("Source training selection policy changed.")
    required_trainer_evidence = {
        "checkpoints",
        "selected_step",
        "selection_policy",
        "fit_metrics",
        "candidate_metrics",
        "sealed_metrics",
        "training_audit",
        "factor_pack_manifest",
        "domain_binding_sha256",
        "schedule_sha256",
        "candidate_thresholds",
        "candidate_gate_passed",
        "sealed_evaluation_unlocked",
    }
    training_audit = trainer_evidence.get("training_audit")
    gradient_audit = (
        training_audit.get("gradient_activation")
        if isinstance(training_audit, Mapping)
        else None
    )
    manifest_factor_names = tuple(
        str(row.get("name"))
        for row in manifest["factors"]
        if isinstance(row, Mapping)
    )
    expected_factor_name_set = set(manifest_factor_names)
    output_only_false_fields = {
        "weight_targets_used",
        "activation_targets_used",
        "hidden_state_targets_used",
        "attention_map_targets_used",
        "cache_value_targets_used",
    }
    if (
        not required_trainer_evidence.issubset(trainer_evidence)
        or trainer_evidence.get("internal_numeric_matching_used") is not False
        or trainer_evidence.get("selection_uses_internal_numeric_targets") is not False
        or trainer_evidence.get("selection_policy") != training["selection_policy"]
        or trainer_evidence.get("candidate_thresholds")
        != normalized_thresholds["candidate"]
        or not isinstance(training_audit, Mapping)
        or training_audit.get("optimizer_steps") != training.get("steps")
        or not isinstance(training_audit.get("optimizer_class"), str)
        or not str(training_audit.get("optimizer_class", "")).strip()
        or any(
            not isinstance(training_audit.get(name), int)
            or isinstance(training_audit.get(name), bool)
            or int(training_audit[name]) <= 0
            for name in (
                "parent_greedy_prefix_count",
                "child_on_policy_prefix_count",
                "parent_output_forward_count",
                "child_output_forward_count",
                "parent_greedy_cache_prompt_count",
                "parent_greedy_cache_forward_count",
            )
        )
        or training_audit.get("invariant_parent_greedy_rows_cached_once") is not True
        or training_audit.get("teacher_signal")
        != "parent-causal-output-logits-and-actions"
        or training_audit.get("parent_greedy_prefix_training") is not True
        or training_audit.get("child_greedy_on_policy_prefix_training") is not True
        or any(training_audit.get(name) is not False for name in output_only_false_fields)
        or not isinstance(gradient_audit, Mapping)
        or set(gradient_audit.get("factor_names", ())) != expected_factor_name_set
        or set(gradient_audit.get("finite_gradient_factor_names", ()))
        != expected_factor_name_set
        or set(gradient_audit.get("nonzero_gradient_factor_names", ()))
        != expected_factor_name_set
        or gradient_audit.get("backward_observation_count", 0) <= 0
        or gradient_audit.get("every_qkvo_gud_factor_gradient_activated") is not True
        or gradient_audit.get("every_lexical_factor_gradient_activated") is not True
        or gradient_audit.get("every_combined_factor_gradient_activated") is not True
        or gradient_audit.get("combined_factor_count")
        != stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT
    ):
        raise StagePreparationError(
            "Source trainer evidence does not prove complete output-only lexical-v2 "
            "learning."
        )
    try:
        fit_passed, fit_decision = stage._behavior_gate(
            trainer_evidence.get("fit_metrics"),
            stage.BehaviorThresholds.from_mapping(
                thresholds["fit"], "source_preregistration.thresholds.fit"
            ),
        )
        candidate_passed, candidate_decision = stage._behavior_gate(
            trainer_evidence.get("candidate_metrics"),
            stage.BehaviorThresholds.from_mapping(
                thresholds["candidate"],
                "source_preregistration.thresholds.candidate",
            ),
        )
        recomputed_acquisition = stage._factor_acquisition_claim(trainer_evidence)
    except stage.ExactTargetNativeStageError as error:
        raise StagePreparationError(
            "Source trainer behavior/acquisition evidence does not authenticate."
        ) from error
    expected_pre_bake_decisions = {
        "fit": fit_decision,
        "candidate": candidate_decision,
        "sealed": None,
    }
    if (
        fit_passed
        or candidate_passed
        or trainer_evidence.get("candidate_gate_passed") is not candidate_passed
        or trainer_evidence.get("sealed_evaluation_unlocked") is not False
        or trainer_evidence.get("sealed_metrics") is not None
        or source_report.get("pre_bake_behavior_decisions")
        != expected_pre_bake_decisions
        or source_report.get("behavior_decisions") != expected_pre_bake_decisions
        or source_report.get("factor_acquisition") != recomputed_acquisition
        or recomputed_acquisition.get("selected_factor_state_sha256")
        != factor_state_sha
        or recomputed_acquisition.get("target_native_factors_acquired_behavior")
        is not True
    ):
        raise StagePreparationError(
            "Source fit/candidate decisions or selected checkpoint acquisition were "
            "not reproduced exactly."
        )
    validation_count = compiler.get(
        "canonical_validation_count", max(1, len(compiler_split.row_ids) // 5)
    )
    if (
        isinstance(validation_count, bool)
        or not isinstance(validation_count, int)
        or validation_count <= 0
        or validation_count >= len(compiler_split.row_ids)
    ):
        raise StagePreparationError(
            "Source compiler validation partition is not reproducible."
        )
    boundary = len(compiler_split.row_ids) - validation_count
    calibration_row_ids = list(compiler_split.row_ids[:boundary])
    validation_row_ids = list(compiler_split.row_ids[boundary:])
    expected_report_parent_evidence = {
        "source_split": "compiler_split",
        "source_fit_artifact_sha256": compiler_split.sha256,
        "calibration_row_count": len(calibration_row_ids),
        "calibration_row_ids_sha256": stage._canonical_sha256(calibration_row_ids),
        "validation_row_count": len(validation_row_ids),
        "validation_row_ids_sha256": stage._canonical_sha256(validation_row_ids),
        "candidate_split_used": False,
        "sealed_split_used": False,
        "partition_fixed_before_model_access": True,
    }
    if source_report.get("canonical_parent_evidence") != (
        expected_report_parent_evidence
    ):
        raise StagePreparationError(
            "Source run report compiler partition evidence does not reproduce."
        )
    historical_evaluation_parent_evidence = {
        "source_split": "authenticated-compiler-split-deterministic-partition",
        "candidate_split_used": False,
        "sealed_split_used": False,
        "compiler_options": compiler,
    }

    historical_implementation = _historical_implementation_authentication(
        source_prereg, source_report
    )
    computed_artifact_authentication = {
        "parent": parent_authentication,
        "blank_target": target_authentication,
        "split_sha256s": {
            role: authenticated_splits[role].sha256
            for role in ("fit", "candidate", "sealed")
        },
        "compiler_split_sha256": compiler_split.sha256,
    }
    if source_report.get("artifact_authentication") != computed_artifact_authentication:
        raise StagePreparationError(
            "Source artifact authentication evidence does not reproduce."
        )
    lexical_residual = _json_object_copy(
        source_prereg.get("lexical_residual"), "source lexical_residual"
    )
    source_evaluation_preimage = {
        "split_sha256s": computed_artifact_authentication["split_sha256s"],
        "compiler_split_sha256": compiler_split.sha256,
        "thresholds": normalized_thresholds,
        "selection_domain": "causal-output-actions-and-free-rollouts-only",
        "qkvo_sites": stage.QKVO_SITE_COUNT,
        "qkvo_rank": stage.QKVO_RANK,
        "swiglu_layers": stage.SWIGLU_LAYER_COUNT,
        "swiglu_rank": stage.SWIGLU_RANK,
        "lexical_residual": lexical_residual,
        "implementation": historical_implementation,
        "runtime_dtype": "float16",
        "runtime_attention_implementation": (
            stage.CANONICAL_ATTENTION_IMPLEMENTATION
        ),
        "canonical_parent_evidence": historical_evaluation_parent_evidence,
    }
    if (
        source_report.get("selection_domain")
        != "causal-output-actions-and-free-rollouts-only"
        or source_report.get("runtime_dtype") != "float16"
        or source_report.get("runtime_attention_implementation")
        != stage.CANONICAL_ATTENTION_IMPLEMENTATION
        or source_report.get("lexical_residual") != lexical_residual
        or stage._canonical_sha256(source_evaluation_preimage)
        != source_evaluation_sha
    ):
        raise StagePreparationError(
            "Source evaluation-contract SHA-256 does not reproduce from its "
            "persisted preregistration evidence."
        )

    source_cache = source_prereg.get("compiler_cache")
    report_cache_binding = source_report.get("compiler_cache")
    cache_evidence = source_report.get("compiled_child_cache")
    cache_fields = {
        "compiler_identity_sha256": "expected_compiler_identity_sha256",
        "receipt_sha256": "expected_receipt_sha256",
        "compiler_report_sha256": "expected_compiler_report_sha256",
        "state_dict_sha256": "expected_state_dict_sha256",
        "artifact_tree_sha256": "expected_artifact_tree_sha256",
    }
    expected_cache_keys = {"root", "require_hit", *cache_fields.values()}
    if (
        not isinstance(source_cache, Mapping)
        or set(source_cache) != expected_cache_keys
        or source_cache.get("require_hit") is not True
        or not isinstance(report_cache_binding, Mapping)
        or set(report_cache_binding)
        != {
            *expected_cache_keys,
            "content_addressed",
            "copied_into_run_output",
        }
        or report_cache_binding.get("require_hit") is not True
        or report_cache_binding.get("content_addressed") is not True
        or report_cache_binding.get("copied_into_run_output") is not False
        or not isinstance(cache_evidence, Mapping)
        or cache_evidence.get("disposition") != "hit"
        or cache_evidence.get("verified_before_factor_install") is not True
    ):
        raise StagePreparationError(
            "Output-Jacobian source requires a verified immutable compiler-cache hit."
        )
    cache_root = _reported_path(
        source_cache.get("root"), base=source_base, name="compiler_cache.root"
    )
    cache_path = _reported_path(
        cache_evidence.get("path"),
        base=source_report_path.parent,
        name="compiled_child_cache.path",
    )
    if _reported_path(
        report_cache_binding.get("root"),
        base=source_report_path.parent,
        name="source_run_report.compiler_cache.root",
    ) != cache_root:
        raise StagePreparationError(
            "Source report and preregistration name different compiler-cache roots."
        )
    cache_hashes: dict[str, str] = {}
    for evidence_name, prereg_name in cache_fields.items():
        digest = stage._require_sha256(
            cache_evidence.get(evidence_name),
            f"source_run_report.compiled_child_cache.{evidence_name}",
        )
        if (
            source_cache.get(prereg_name) != digest
            or report_cache_binding.get(prereg_name) != digest
        ):
            raise StagePreparationError(
                f"Source cache hash {evidence_name!r} disagrees with its "
                "preregistration."
            )
        cache_hashes[evidence_name] = digest
    source_compiler_report = source_report.get("compiler_report")
    if not isinstance(source_compiler_report, Mapping):
        raise StagePreparationError("Source compiler report is malformed.")
    receipt_path = cache_path / CACHE_RECEIPT_FILENAME
    try:
        receipt_payload = stage._strict_json(receipt_path)
        if not isinstance(receipt_payload, Mapping):
            raise StagePreparationError("Cache receipt is not a JSON object.")
        compiler_identity = CompilerIdentity.from_dict(
            receipt_payload.get("compiler_identity")
        )
        verified_cache = verify_compiled_child_cache(
            cache_path,
            expected_identity=compiler_identity,
            expected_receipt_sha256=cache_hashes["receipt_sha256"],
            expected_compiler_report_sha256=cache_hashes[
                "compiler_report_sha256"
            ],
            expected_state_dict_sha256=cache_hashes["state_dict_sha256"],
        )
    except (CompiledChildCacheError, stage.ExactTargetNativeStageError) as error:
        raise StagePreparationError(
            "Source compiler cache receipt/model artifacts failed public verification."
        ) from error
    verified_receipt = verified_cache.receipt.as_dict()
    if (
        cache_path.is_symlink()
        or cache_root.is_symlink()
        or not cache_path.is_dir()
        or cache_path.parent != cache_root
        or cache_path.name != cache_hashes["compiler_identity_sha256"]
        or compiler_identity.compiler_identity_sha256
        != cache_hashes["compiler_identity_sha256"]
        or verified_cache.compiler_identity.as_dict() != compiler_identity.as_dict()
        or verified_cache.receipt.receipt_sha256 != cache_hashes["receipt_sha256"]
        or verified_receipt["compiler_report"]["canonical_sha256"]
        != cache_hashes["compiler_report_sha256"]
        or verified_receipt["model_state"]["state_dict_sha256"]
        != cache_hashes["state_dict_sha256"]
        or verified_receipt["artifact_tree"]["tree_sha256"]
        != cache_hashes["artifact_tree_sha256"]
        or dict(verified_cache.compiler_report) != dict(source_compiler_report)
        or stage._canonical_sha256(source_compiler_report)
        != cache_hashes["compiler_report_sha256"]
    ):
        raise StagePreparationError(
            "Source compiler identity/receipt/report/state/tree claims disagree."
        )

    # This is the in-memory compiled tensor digest used by the runner after cache
    # load.  It is intentionally a different hash domain from the serialized
    # cache receipt's state_dict_sha256 and must remain a separate binding.
    compiled_state_sha = stage._require_sha256(
        source_report.get("rollback_state_sha256"),
        "source_run_report.rollback_state_sha256",
    )
    if envelope_provenance != {
        "reason": "fit-gate-rejected",
        "preregistration_sha256": source_prereg_sha,
        "evaluation_contract_sha256": source_evaluation_sha,
    }:
        raise StagePreparationError(
            "Source factor envelope provenance disagrees with the terminal report."
        )

    module_specs = {
        "trainer_sha256": stage.CANONICAL_TRAINER_MODULE,
        "compiler_cache_sha256": "neural_forge.hf_compiled_child_cache",
        "lexical_adapter_sha256": "neural_forge.hf_dense_lexical_residual_adapter",
        "progressive_bridge_sha256": "neural_forge.hf_progressive_target_bridge",
        "qkvo_adapter_sha256": (
            "neural_forge.hf_dense_gqa_qkvo_residual_adapter"
        ),
        "swiglu_adapter_sha256": (
            "neural_forge.hf_dense_swiglu_projection_residual_adapter"
        ),
        "output_jacobian_basis_sha256": stage.OUTPUT_JACOBIAN_BASIS_MODULE,
        "output_jacobian_sketch_sha256": stage.OUTPUT_JACOBIAN_SKETCH_MODULE,
        "qkvo_output_jacobian_audit_sha256": (
            stage.QKVO_OUTPUT_JACOBIAN_AUDIT_MODULE
        ),
    }
    implementation_hashes: dict[str, str] = {}
    for field_name, module_name in module_specs.items():
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise StagePreparationError(
                f"Required output-Jacobian module {module_name!r} is unavailable."
            )
        implementation_hashes[field_name] = stage._file_sha256(Path(spec.origin))

    runtime = _json_object_copy(source_prereg.get("runtime"), "source runtime")
    if runtime != {
        "dtype": "float16",
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "explicit_execute_required": True,
    }:
        raise StagePreparationError("Source runtime contract is not the pinned stage.")
    seed = source_prereg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StagePreparationError("Source seed must be a nonnegative integer.")
    normalized_parent = _json_object_copy(source_prereg["parent"], "source parent")
    normalized_parent["path"] = str(parent_binding.path)
    normalized_target = _json_object_copy(
        source_prereg["blank_target"], "source blank_target"
    )
    normalized_target["path"] = str(target_binding.path)
    normalized_cache = _json_object_copy(source_cache, "source compiler_cache")
    normalized_cache["root"] = str(cache_root)
    preregistration = {
        "schema": stage.PREREGISTRATION_SCHEMA,
        "sealed_before_model_access": True,
        "seed": seed,
        "implementation": {
            "runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
            "trainer_module": stage.CANONICAL_TRAINER_MODULE,
            **implementation_hashes,
        },
        "parent": normalized_parent,
        "blank_target": normalized_target,
        "compiler_split": {
            "path": str(compiler_split.path),
            "sha256": compiler_split.sha256,
            "expected_count": len(compiler_split.row_ids),
        },
        "splits": split_bindings,
        "thresholds": _json_object_copy(thresholds, "source thresholds"),
        "compiler": compiler,
        "training": training,
        "lexical_residual": lexical_residual,
        "compiler_cache": normalized_cache,
        "continuation": {
            "factor_pack": {"path": str(factor_path), "sha256": factor_sha},
            "source_run_report": {
                "path": str(source_report_path),
                "sha256": source_report_sha,
            },
            "expected_factor_state_sha256": factor_state_sha,
            "expected_compiler_report_sha256": cache_hashes[
                "compiler_report_sha256"
            ],
            "expected_compiled_state_sha256": compiled_state_sha,
        },
        "output_jacobian_audit": {
            "schema": stage.OUTPUT_JACOBIAN_AUDIT_SCHEMA,
            "source_exposed_candidate_as_fit": True,
            "basis_split_role": "candidate",
            "prompt_count": 24,
            "horizon": 8,
            "lanes": ["parent-greedy", "child-greedy"],
            "rank": 8,
            "sketch_width": 16,
            "min_normal_fraction": 0.20,
            "min_guided_over_random_ratio": 1.50,
            "max_bytes": 268_435_456,
        },
        **(
            {}
            if "corpus_provenance" not in source_prereg
            else {
                "corpus_provenance": _json_object_copy(
                    source_prereg["corpus_provenance"],
                    "source corpus_provenance",
                )
            }
        ),
        "runtime": runtime,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    )
    try:
        staged_prereg_path = staging / "preregistration.json"
        _write_json_exclusive(staged_prereg_path, preregistration)
        prereg_sha = stage._file_sha256(staged_prereg_path)
        authenticated = stage.authenticate_preregistration(
            staged_prereg_path, prereg_sha
        )
        if authenticated.output_jacobian_audit is None:
            raise StagePreparationError(
                "Live model-free preflight did not authenticate the "
                "output-Jacobian audit."
            )
        final_prereg_path = output / "preregistration.json"
        report = {
            "schema": BUILD_SCHEMA,
            "mode": OUTPUT_JACOBIAN_AUDIT_MODE,
            "preregistration": str(final_prereg_path),
            "preregistration_sha256": prereg_sha,
            "evaluation_contract_sha256": authenticated.evaluation_contract_sha256,
            "source_run_report": str(source_report_path),
            "source_run_report_sha256": source_report_sha,
            "source_preregistration": str(source_prereg_path),
            "source_preregistration_sha256": source_prereg_sha,
            "source_evaluation_contract_sha256": source_evaluation_sha,
            "source_factor_pack_sha256": factor_sha,
            "source_factor_state_sha256": factor_state_sha,
            "compiler_split_sha256": compiler_split.sha256,
            "behavior_split_sha256s": {
                role: split_bindings[role]["sha256"]
                for role in ("fit", "candidate", "sealed")
            },
            "required_compiler_cache_identity_sha256": cache_hashes[
                "compiler_identity_sha256"
            ],
            "source_candidate_exposed_as_fit": True,
            "sealed_evaluation_remains_locked": True,
            "audit_only": True,
            "model_weights_loaded": False,
            "model_checkpoint_created": False,
            "training_launched": False,
            "ready_for_explicit_preflight": True,
        }
        _write_json_exclusive(staging / "build_report.json", report)
        if output.exists():
            raise StagePreparationError("output_dir appeared during preparation.")
        os.rename(staging, output)
        return report
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--target", type=Path, default=stage.DEFAULT_BLANK_TARGET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--continuation-source-report", type=Path)
    parser.add_argument("--lexical-continuation-source-report", type=Path)
    parser.add_argument("--output-jacobian-audit-source-report", type=Path)
    parser.add_argument(
        "--original-split-dir",
        type=Path,
        default=continuation_splits.DEFAULT_ORIGINAL_SPLIT_DIR,
    )
    parser.add_argument(
        "--compiler-cache-root", type=Path, default=DEFAULT_COMPILER_CACHE_ROOT
    )
    parser.add_argument(
        "--source-split-dir",
        type=Path,
        default=continuation_splits.DEFAULT_RESIDUAL_ATTENTION_SPLIT_DIR,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    continuation_modes = (
        args.continuation_source_report,
        args.lexical_continuation_source_report,
        args.output_jacobian_audit_source_report,
    )
    if sum(value is not None for value in continuation_modes) > 1:
        raise SystemExit(
            "Choose at most one of --continuation-source-report, "
            "--lexical-continuation-source-report, or "
            "--output-jacobian-audit-source-report."
        )
    if args.output_jacobian_audit_source_report is not None:
        report = prepare_output_jacobian_audit_stage(
            output_dir=args.output_dir,
            source_run_report=args.output_jacobian_audit_source_report,
        )
    elif args.lexical_continuation_source_report is not None:
        report = prepare_lexical_continuation_stage(
            output_dir=args.output_dir,
            parent=args.parent,
            target=args.target,
            source_run_report=args.lexical_continuation_source_report,
            source_split_dir=args.source_split_dir,
            original_split_dir=args.original_split_dir,
            compiler_cache_root=args.compiler_cache_root,
            corpus_path=args.corpus,
        )
    elif args.continuation_source_report is None:
        report = prepare_stage(
            output_dir=args.output_dir,
            parent=args.parent,
            target=args.target,
            corpus_path=args.corpus,
        )
    else:
        report = prepare_continuation_stage(
            output_dir=args.output_dir,
            parent=args.parent,
            target=args.target,
            source_run_report=args.continuation_source_report,
            original_split_dir=args.original_split_dir,
            compiler_cache_root=args.compiler_cache_root,
            corpus_path=args.corpus,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
