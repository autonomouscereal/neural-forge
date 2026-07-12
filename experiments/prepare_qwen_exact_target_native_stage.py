#!/usr/bin/env python3
"""Seal the real 52-prompt contract for exact target-native Qwen stage 1.

The source is the existing 2,906-row canonical Neural Forge JSONL corpus.  No
prompt text is synthesized or replicated.  Unique texts are placed in stable
content-hash order, tokenized with the pinned Qwen tokenizer, and greedily
selected only when their complete >=8-token prefix inventory is disjoint from
all previously selected rows.  The first 24 become fit, the next 12 candidate,
and the next 16 sealed.  Model weights are content-addressed but never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import run_qwen_exact_target_native_stage as stage
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--target", type=Path, default=stage.DEFAULT_BLANK_TARGET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_stage(
        output_dir=args.output_dir,
        parent=args.parent,
        target=args.target,
        corpus_path=args.corpus,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
