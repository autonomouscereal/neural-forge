#!/usr/bin/env python3
"""Plan and optionally run a bounded progressive zero-recovery Qwen fold."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "neural-forge-qwen-progressive-fold-run-v1"
STAGE_MANIFEST_SCHEMA = "neural-forge-qwen-progressive-stages-v1"
IMPLEMENTATION_VERSION = "progressive-zero-recovery-v1"
MAX_STAGES = 8
EXACT_BEHAVIOR_PROBES = 1000
EXACT_GENERATION_PROMPTS = 1000
EXACT_CHOICE_PROBES = 250
MANUAL_PROMPTS = (
    "Q: What is 6+4? A:",
    "Q: If a box has 3 red balls and 2 blue balls, how many balls are there? A:",
    "Write a Python function that adds two numbers.",
    "Briefly explain tokenizer mismatch in cross-family model merging.",
    "What should happen before pruning neurons in a merge-first compression pipeline?",
    "A person hears thunder after seeing lightning. What should they do?",
    "Explain one risk when merging two neural models.",
)
_STAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_HEX_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_FORBIDDEN_FLAG_PARTS = (
    "recover",
    "train",
    "optimizer",
    "learning-rate",
    "epochs",
    "steps",
)
_ALLOWED_NEURAL_FORGE_COMMANDS = {
    "hf-qwen-build-target",
    "hf-qwen-pair-compose",
    "hf-eval",
}
_IDENTITY_SOURCE_FILES = (
    "src/neural_forge/cli.py",
    "src/neural_forge/hf_qwen_target.py",
    "src/neural_forge/hf_qwen_compose.py",
    "src/neural_forge/hf_qwen_span_costs.py",
    "src/neural_forge/hf_span_planner.py",
    "src/neural_forge/hf_qwen_rollout.py",
    "src/neural_forge/hf_qwen_rollout_refold.py",
    "src/neural_forge/hf_rollout_gate.py",
    "experiments/launch_qwen_progressive_fold_run.py",
)


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One exact random-init Qwen3 child shape in the progressive chain."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    seed: int = 0
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    target_params: int | None = None
    rollout_refold_rounds: int | None = None

    def validate(self) -> None:
        if not _STAGE_NAME.fullmatch(self.name):
            raise ValueError(
                "stage name must be one safe path component containing only "
                "letters, numbers, dot, underscore, or hyphen."
            )
        for field in (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"stage {self.name!r} has non-positive {field}.")
        if self.seed < 0:
            raise ValueError(f"stage {self.name!r} seed must be non-negative.")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                f"stage {self.name!r} query heads must be divisible by KV heads."
            )
        if self.target_params is not None and self.target_params <= 0:
            raise ValueError(
                f"stage {self.name!r} target_params must be positive when supplied."
            )
        if self.rollout_refold_rounds is not None and self.rollout_refold_rounds < 0:
            raise ValueError(
                f"stage {self.name!r} rollout_refold_rounds must be non-negative."
            )


@dataclass(frozen=True, slots=True)
class ComposeSettings:
    """Closed-form pair composition controls shared by all stages."""

    ridge: float = 1e-3
    max_length: int = 160
    batch_size: int = 2
    max_tokens_per_layer: int = 4096
    max_anchors: int = 4096
    max_calibration_texts: int = 256
    basis_mode: str = "activation-pca"
    basis_preprocess: str = "uncentered-rowrms"
    activation_domain_balance: str = "keyword"
    observable_coordinates: bool = True
    observable_coordinate_ridge: float = 1e-6
    path_decomposition: bool = True
    path_decomposition_max_rows: int = 1024
    path_decomposition_ridge: float = 1e-6
    path_decomposition_sketch_size: int = 64
    path_decomposition_interpolation_mode: str = "sparse-signature"
    adaptive_depth_critical_quantile: float = 0.8
    adaptive_depth_causal_magnitude_weight: float = 1.0
    adaptive_depth_incompatibility_weight: float = 1.0
    adaptive_depth_reconstruction_weight: float = 1.0
    rollout_refold_rounds: int = 0
    rollout_refold_max_prompts: int = 64
    rollout_refold_max_new_tokens: int = 24
    rollout_refold_batch_size: int = 4
    rollout_refold_candidate_alphas: str = "1,0.5,0.25"
    rollout_gate_top_k: int = 5
    rollout_gate_min_top1_gain: float = 0.0
    rollout_gate_min_action_logprob_gain: float = 0.0
    rollout_gate_max_repetition_regression: float = 0.0
    rollout_gate_max_longest_run_regression: int = 0
    rollout_gate_min_action_count_ratio: float = 0.9
    fit_gate_validation_fraction: float = 0.25
    fit_gate_min_validation_rows: int = 128
    fit_gate_min_relative_improvement: float = 0.02
    fit_gate_max_cosine_drop: float = 0.0
    fit_gate_max_relative_rmse: float = 0.75
    head_validation_vocab_rows: int = 4096
    dtype: str = "float16"
    candidate_dtype: str = "float32"
    device: str = "cuda:0"

    def validate(self) -> None:
        finite_non_negative = (
            ("ridge", self.ridge),
            ("observable_coordinate_ridge", self.observable_coordinate_ridge),
            ("path_decomposition_ridge", self.path_decomposition_ridge),
            (
                "adaptive_depth_causal_magnitude_weight",
                self.adaptive_depth_causal_magnitude_weight,
            ),
            (
                "adaptive_depth_incompatibility_weight",
                self.adaptive_depth_incompatibility_weight,
            ),
            (
                "adaptive_depth_reconstruction_weight",
                self.adaptive_depth_reconstruction_weight,
            ),
            (
                "fit_gate_min_relative_improvement",
                self.fit_gate_min_relative_improvement,
            ),
            ("fit_gate_max_cosine_drop", self.fit_gate_max_cosine_drop),
        )
        for name, value in finite_non_negative:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        for name in (
            "max_length",
            "batch_size",
            "max_tokens_per_layer",
            "max_anchors",
            "max_calibration_texts",
            "path_decomposition_max_rows",
            "path_decomposition_sketch_size",
            "rollout_refold_max_prompts",
            "rollout_refold_max_new_tokens",
            "rollout_refold_batch_size",
            "rollout_gate_top_k",
            "fit_gate_min_validation_rows",
            "head_validation_vocab_rows",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.rollout_refold_rounds < 0:
            raise ValueError("rollout_refold_rounds must be non-negative.")
        if not 0.0 <= self.adaptive_depth_critical_quantile <= 1.0:
            raise ValueError("adaptive_depth_critical_quantile must be in [0, 1].")
        if not 0.0 < self.fit_gate_validation_fraction < 1.0:
            raise ValueError("fit_gate_validation_fraction must be in (0, 1).")
        if self.fit_gate_max_relative_rmse <= 0:
            raise ValueError("fit_gate_max_relative_rmse must be positive.")
        if not 0.0 < self.rollout_gate_min_action_count_ratio <= 1.0:
            raise ValueError("rollout_gate_min_action_count_ratio must be in (0, 1].")
        for name in (
            "rollout_gate_min_top1_gain",
            "rollout_gate_min_action_logprob_gain",
            "rollout_gate_max_repetition_regression",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if self.rollout_gate_max_longest_run_regression < 0:
            raise ValueError(
                "rollout_gate_max_longest_run_regression must be non-negative."
            )
        alphas = [
            float(part.strip())
            for part in self.rollout_refold_candidate_alphas.split(",")
            if part.strip()
        ]
        if not alphas or any(
            not math.isfinite(alpha) or not 0.0 < alpha <= 1.0 for alpha in alphas
        ):
            raise ValueError("rollout refold alphas must all be in (0, 1].")
        if self.basis_mode not in {"activation-pca", "embedding-grouped"}:
            raise ValueError("unsupported basis_mode.")
        if self.basis_preprocess not in {
            "centered-rowrms",
            "uncentered-rowrms",
            "uncentered",
        }:
            raise ValueError("unsupported basis_preprocess.")
        if self.activation_domain_balance not in {"none", "keyword"}:
            raise ValueError("unsupported activation_domain_balance.")
        if self.path_decomposition_interpolation_mode not in {
            "dense",
            "sparse-signature",
        }:
            raise ValueError("unsupported path decomposition interpolation mode.")
        if not self.dtype or not self.candidate_dtype or not self.device:
            raise ValueError("dtype, candidate_dtype, and device cannot be empty.")


def default_stage_specs(seed: int = 0) -> tuple[StageSpec, ...]:
    """Return the documented 4B -> 1.4B -> 600M -> 158M schedule."""

    return (
        StageSpec("blank_1p4b", 1536, 5120, 30, 16, 4, 128, seed=seed),
        StageSpec("blank_600m", 1024, 3072, 24, 8, 2, 128, seed=seed),
        StageSpec("blank_158m", 384, 1152, 18, 8, 2, 128, seed=seed),
    )


def qwen3_parameter_count(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    tie_word_embeddings: bool = False,
    attention_bias: bool = False,
) -> int:
    """Count unique dense Transformers Qwen3 parameters exactly."""

    query_width = num_attention_heads * head_dim
    key_value_width = num_key_value_heads * head_dim
    attention_weights = 2 * hidden_size * (query_width + key_value_width)
    attention_biases = (
        query_width + 2 * key_value_width + hidden_size if attention_bias else 0
    )
    per_layer = (
        attention_weights
        + attention_biases
        + 2 * head_dim
        + 3 * hidden_size * intermediate_size
        + 2 * hidden_size
    )
    embedding_and_head = vocab_size * hidden_size * (1 if tie_word_embeddings else 2)
    return embedding_and_head + num_hidden_layers * per_layer + hidden_size


def _stage_parameter_count(stage: StageSpec, vocab_size: int) -> int:
    return qwen3_parameter_count(
        vocab_size=vocab_size,
        hidden_size=stage.hidden_size,
        intermediate_size=stage.intermediate_size,
        num_hidden_layers=stage.num_hidden_layers,
        num_attention_heads=stage.num_attention_heads,
        num_key_value_heads=stage.num_key_value_heads,
        head_dim=stage.head_dim,
        tie_word_embeddings=stage.tie_word_embeddings,
        attention_bias=stage.attention_bias,
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_cached_parent_report(
    path: Path | None, *, kind: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"cached parent {kind} report must be a non-empty file.")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cached parent {kind} report must be valid JSON.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"cached parent {kind} report must contain a JSON object.")
    if kind == "eval":
        behavior = payload.get("behavior_suite")
        generation = payload.get("generation_health")
        choice = payload.get("choice_accuracy")
        totals = (
            behavior.get("total") if isinstance(behavior, Mapping) else None,
            generation.get("total") if isinstance(generation, Mapping) else None,
            choice.get("total") if isinstance(choice, Mapping) else None,
        )
        expected = (
            EXACT_BEHAVIOR_PROBES,
            EXACT_GENERATION_PROMPTS,
            EXACT_CHOICE_PROBES,
        )
        if totals != expected:
            raise ValueError(
                "cached parent eval must contain exactly 1000 behavior, 1000 "
                "generation, and 250 choice results."
            )
    elif kind == "manual":
        generations = payload.get("manual_generations")
        if not isinstance(generations, list) or not generations:
            raise ValueError(
                "cached parent manual report must contain manual_generations."
            )
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"unsupported cached parent report kind {kind!r}.")
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
        "kind": kind,
        "validated": True,
    }


def checkpoint_content_sha256(checkpoint: Path) -> str:
    """Return a path-independent Merkle-style digest of checkpoint contents."""

    root = checkpoint.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source model must be a local checkpoint directory.")
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("source model checkpoint is empty.")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def implementation_content_sha256(repo_root: Path = REPO_ROOT) -> str:
    records = []
    for relative in _IDENTITY_SOURCE_FILES:
        path = repo_root / relative
        if path.is_file():
            records.append(
                {
                    "path": relative,
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not records:
        raise ValueError("no progressive-fold implementation files were found.")
    return _canonical_sha256(records)


def _read_source_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source model must contain a valid config.json.") from exc
    if not isinstance(config, dict):
        raise ValueError("source config.json must contain an object.")
    required = (
        "vocab_size",
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"source Qwen config is missing: {', '.join(missing)}.")
    return config


def _source_parameter_count(config: Mapping[str, Any]) -> int:
    heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // heads)
    return qwen3_parameter_count(
        vocab_size=int(config["vocab_size"]),
        hidden_size=hidden,
        intermediate_size=int(config["intermediate_size"]),
        num_hidden_layers=int(config["num_hidden_layers"]),
        num_attention_heads=heads,
        num_key_value_heads=int(config.get("num_key_value_heads") or heads),
        head_dim=head_dim,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        attention_bias=bool(config.get("attention_bias", False)),
    )


def _bool_value(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean.")


_STAGE_ALIASES = {
    "h": "hidden_size",
    "hidden": "hidden_size",
    "i": "intermediate_size",
    "intermediate": "intermediate_size",
    "l": "num_hidden_layers",
    "layers": "num_hidden_layers",
    "q": "num_attention_heads",
    "q_heads": "num_attention_heads",
    "kv": "num_key_value_heads",
    "kv_heads": "num_key_value_heads",
    "hd": "head_dim",
    "params": "target_params",
    "rollout": "rollout_refold_rounds",
    "tie": "tie_word_embeddings",
    "bias": "attention_bias",
}
_STAGE_FIELDS = {field.name for field in StageSpec.__dataclass_fields__.values()}


def stage_spec_from_mapping(
    value: Mapping[str, Any], *, default_seed: int = 0
) -> StageSpec:
    normalized: dict[str, Any] = {}
    for raw_name, raw_value in value.items():
        name = _STAGE_ALIASES.get(str(raw_name), str(raw_name))
        if name not in _STAGE_FIELDS:
            raise ValueError(f"unknown stage field {raw_name!r}.")
        if name in normalized:
            raise ValueError(f"duplicate stage field {name!r}.")
        normalized[name] = raw_value
    normalized.setdefault("seed", default_seed)
    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "seed",
        "target_params",
        "rollout_refold_rounds",
    ):
        if name in normalized and normalized[name] is not None:
            normalized[name] = int(normalized[name])
    for name in ("tie_word_embeddings", "attention_bias"):
        if name in normalized:
            normalized[name] = _bool_value(normalized[name], name)
    try:
        stage = StageSpec(**normalized)
    except TypeError as exc:
        raise ValueError(f"invalid stage definition: {exc}") from exc
    stage.validate()
    return stage


def parse_stage_override(value: str, *, default_seed: int = 0) -> StageSpec:
    text = value.strip()
    if not text:
        raise ValueError("--stage cannot be empty.")
    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("--stage JSON is invalid.") from exc
        if not isinstance(payload, dict):
            raise ValueError("--stage JSON must contain one object.")
    else:
        payload = {}
        for item in text.split(","):
            if "=" not in item:
                raise ValueError("compact --stage values must use key=value pairs.")
            name, raw = item.split("=", 1)
            name = name.strip()
            if not name or name in payload:
                raise ValueError("compact --stage contains an empty or duplicate key.")
            payload[name] = raw.strip()
    return stage_spec_from_mapping(payload, default_seed=default_seed)


def load_stage_specs(
    *,
    stage_manifest: Path | None = None,
    stage_overrides: Sequence[str] = (),
    default_seed: int = 0,
) -> tuple[StageSpec, ...]:
    if stage_manifest is not None and stage_overrides:
        raise ValueError("use either --stage-manifest or --stage, not both.")
    if stage_manifest is not None:
        try:
            payload = json.loads(stage_manifest.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("stage manifest must be valid JSON.") from exc
        if isinstance(payload, dict):
            schema = payload.get("schema")
            if schema is not None and schema != STAGE_MANIFEST_SCHEMA:
                raise ValueError(f"unsupported stage manifest schema {schema!r}.")
            payload = payload.get("stages")
        if not isinstance(payload, list):
            raise ValueError("stage manifest must contain a stages list.")
        stages = tuple(
            stage_spec_from_mapping(item, default_seed=default_seed)
            if isinstance(item, dict)
            else (_raise_stage_type(index))
            for index, item in enumerate(payload)
        )
    elif stage_overrides:
        stages = tuple(
            parse_stage_override(item, default_seed=default_seed)
            for item in stage_overrides
        )
    else:
        stages = default_stage_specs(default_seed)
    _validate_stage_list(stages)
    return stages


def _raise_stage_type(index: int) -> StageSpec:
    raise ValueError(f"stage manifest entry {index} must be an object.")


def _validate_stage_list(stages: Sequence[StageSpec]) -> None:
    if not stages:
        raise ValueError("at least one progressive stage is required.")
    if len(stages) > MAX_STAGES:
        raise ValueError(f"at most {MAX_STAGES} progressive stages are allowed.")
    names = []
    for stage in stages:
        stage.validate()
        names.append(stage.name)
    if len(set(names)) != len(names):
        raise ValueError("progressive stage names must be unique.")


def build_target_command(
    *,
    python: str | Path,
    tokenizer_model: Path,
    source_model: Path,
    output: Path,
    source_parameter_count: int,
    parameter_count: int,
    stage: StageSpec,
    dtype: str,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "neural_forge.cli",
        "hf-qwen-build-target",
        str(tokenizer_model),
        str(output),
        "--target-params",
        str(parameter_count),
        "--source-param-count",
        str(source_parameter_count),
        "--source-config",
        str(source_model),
        "--source-name",
        f"progressive-donor-for-{stage.name}",
        "--hidden-sizes",
        str(stage.hidden_size),
        "--intermediate-sizes",
        str(stage.intermediate_size),
        "--layer-counts",
        str(stage.num_hidden_layers),
        "--query-head-counts",
        str(stage.num_attention_heads),
        "--key-value-head-counts",
        str(stage.num_key_value_heads),
        "--head-dims",
        str(stage.head_dim),
        "--seed",
        str(stage.seed),
        "--dtype",
        dtype,
    ]
    if stage.tie_word_embeddings:
        command.append("--tie-word-embeddings")
    if stage.attention_bias:
        command.append("--attention-bias")
    _assert_zero_recovery_commands([command])
    return command


def build_compose_command(
    *,
    python: str | Path,
    source_model: Path,
    blank_target: Path,
    output: Path,
    calibration_file: Path,
    settings: ComposeSettings,
    rollout_refold_rounds: int | None = None,
) -> list[str]:
    settings.validate()
    rounds = (
        settings.rollout_refold_rounds
        if rollout_refold_rounds is None
        else rollout_refold_rounds
    )
    if rounds < 0:
        raise ValueError("stage rollout refold rounds must be non-negative.")
    command = [
        str(python),
        "-m",
        "neural_forge.cli",
        "hf-qwen-pair-compose",
        str(source_model),
        str(blank_target),
        str(output),
        "--adaptive-depth-spans",
        "--adaptive-depth-critical-quantile",
        str(settings.adaptive_depth_critical_quantile),
        "--adaptive-depth-causal-magnitude-weight",
        str(settings.adaptive_depth_causal_magnitude_weight),
        "--adaptive-depth-incompatibility-weight",
        str(settings.adaptive_depth_incompatibility_weight),
        "--adaptive-depth-reconstruction-weight",
        str(settings.adaptive_depth_reconstruction_weight),
        "--ridge",
        str(settings.ridge),
        "--max-length",
        str(settings.max_length),
        "--batch-size",
        str(settings.batch_size),
        "--max-tokens-per-layer",
        str(settings.max_tokens_per_layer),
        "--max-anchors",
        str(settings.max_anchors),
        "--basis-mode",
        settings.basis_mode,
        "--basis-preprocess",
        settings.basis_preprocess,
        "--activation-domain-balance",
        settings.activation_domain_balance,
        "--observable-coordinate-ridge",
        str(settings.observable_coordinate_ridge),
        "--path-decomposition-max-rows",
        str(settings.path_decomposition_max_rows),
        "--path-decomposition-ridge",
        str(settings.path_decomposition_ridge),
        "--path-decomposition-sketch-size",
        str(settings.path_decomposition_sketch_size),
        "--path-decomposition-interpolation-mode",
        settings.path_decomposition_interpolation_mode,
        "--rollout-refold-rounds",
        str(rounds),
        "--rollout-refold-max-prompts",
        str(settings.rollout_refold_max_prompts),
        "--rollout-refold-max-new-tokens",
        str(settings.rollout_refold_max_new_tokens),
        "--rollout-refold-batch-size",
        str(settings.rollout_refold_batch_size),
        "--rollout-refold-candidate-alphas",
        settings.rollout_refold_candidate_alphas,
        "--rollout-gate-top-k",
        str(settings.rollout_gate_top_k),
        "--rollout-gate-min-top1-gain",
        str(settings.rollout_gate_min_top1_gain),
        "--rollout-gate-min-action-logprob-gain",
        str(settings.rollout_gate_min_action_logprob_gain),
        "--rollout-gate-max-repetition-regression",
        str(settings.rollout_gate_max_repetition_regression),
        "--rollout-gate-max-longest-run-regression",
        str(settings.rollout_gate_max_longest_run_regression),
        "--rollout-gate-min-action-count-ratio",
        str(settings.rollout_gate_min_action_count_ratio),
        "--head-validation-vocab-rows",
        str(settings.head_validation_vocab_rows),
        "--calibration-file",
        str(calibration_file),
        "--max-calibration-texts",
        str(settings.max_calibration_texts),
        "--fit-gate-validation-fraction",
        str(settings.fit_gate_validation_fraction),
        "--fit-gate-min-validation-rows",
        str(settings.fit_gate_min_validation_rows),
        "--fit-gate-min-relative-improvement",
        str(settings.fit_gate_min_relative_improvement),
        "--fit-gate-max-cosine-drop",
        str(settings.fit_gate_max_cosine_drop),
        "--fit-gate-max-relative-rmse",
        str(settings.fit_gate_max_relative_rmse),
        "--dtype",
        settings.dtype,
        "--candidate-dtype",
        settings.candidate_dtype,
        "--device",
        settings.device,
    ]
    if not settings.observable_coordinates:
        command.append("--no-observable-coordinates")
    if not settings.path_decomposition:
        command.append("--no-path-decomposition")
    _assert_zero_recovery_commands([command])
    return command


def build_eval_command(
    *,
    python: str | Path,
    model: Path,
    report: Path,
    eval_suite: Path,
    device: str,
    dtype: str,
) -> list[str]:
    command = [
        str(python),
        "-m",
        "neural_forge.cli",
        "hf-eval",
        str(model),
        str(report),
        "--eval-suite",
        str(eval_suite),
        "--max-behavior-probes",
        str(EXACT_BEHAVIOR_PROBES),
        "--max-generation-prompts",
        str(EXACT_GENERATION_PROMPTS),
        "--max-choice-probes",
        str(EXACT_CHOICE_PROBES),
        "--max-text-evals",
        "128",
        "--max-new-tokens",
        "96",
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    _assert_zero_recovery_commands([command])
    return command


def build_manual_command(
    *,
    python: str | Path,
    launcher: Path,
    model: Path,
    report: Path,
    device: str,
    dtype: str,
    seed: int,
) -> list[str]:
    command = [
        str(python),
        str(launcher),
        "_manual-eval",
        str(model),
        str(report),
        "--device",
        device,
        "--dtype",
        dtype,
        "--seed",
        str(seed),
        "--max-new-tokens",
        "96",
    ]
    _assert_zero_recovery_commands([command])
    return command


def _assert_zero_recovery_commands(commands: Iterable[Sequence[str]]) -> None:
    for command in commands:
        values = [str(token) for token in command]
        if "neural_forge.cli" in values:
            module_index = values.index("neural_forge.cli")
            if module_index + 1 >= len(values):
                raise ValueError("neural_forge CLI command is incomplete.")
            subcommand = values[module_index + 1]
            if subcommand not in _ALLOWED_NEURAL_FORGE_COMMANDS:
                raise ValueError(
                    f"progressive zero-recovery plan rejects command {subcommand!r}."
                )
        elif "_manual-eval" not in values:
            raise ValueError("progressive plan contains an unknown command.")
        for token in values:
            if token.startswith("--") and any(
                part in token.lower() for part in _FORBIDDEN_FLAG_PARTS
            ):
                raise ValueError(
                    f"progressive zero-recovery plan rejects flag {token!r}."
                )


def _cache_record(
    *,
    cache_dir: Path,
    model_identity: str,
    eval_suite_sha256: str,
    implementation_sha256: str,
    eval_dtype: str,
    device: str,
    manual_seed: int,
) -> dict[str, Any]:
    eval_settings = {
        "behavior_probes": EXACT_BEHAVIOR_PROBES,
        "generation_prompts": EXACT_GENERATION_PROMPTS,
        "choice_probes": EXACT_CHOICE_PROBES,
        "max_text_evals": 128,
        "max_new_tokens": 96,
        "dtype": eval_dtype,
        "device": device,
    }
    eval_key = _canonical_sha256(
        {
            "kind": "hf-eval",
            "model_identity": model_identity,
            "eval_suite_sha256": eval_suite_sha256,
            "settings": eval_settings,
            "implementation_sha256": implementation_sha256,
        }
    )
    manual_settings = {
        "prompts": MANUAL_PROMPTS,
        "max_new_tokens": 96,
        "do_sample": False,
        "seed": manual_seed,
        "dtype": eval_dtype,
        "device": device,
    }
    manual_key = _canonical_sha256(
        {
            "kind": "manual-generation",
            "model_identity": model_identity,
            "settings": manual_settings,
            "implementation_sha256": implementation_sha256,
        }
    )
    return {
        "policy": "content-and-settings-addressed-reuse",
        "eval_key": eval_key,
        "eval_report": str(cache_dir / "eval" / f"{eval_key}.json"),
        "manual_key": manual_key,
        "manual_report": str(cache_dir / "manual" / f"{manual_key}.json"),
        "eval_settings": eval_settings,
        "manual_settings": manual_settings,
    }


def _command_record(command: Sequence[str]) -> dict[str, Any]:
    values = [str(token) for token in command]
    return {"argv": values, "argv_sha256": _canonical_sha256(values)}


def _validate_progressive_shapes(
    *,
    source_layers: int,
    source_parameter_count: int,
    stages: Sequence[StageSpec],
    vocab_size: int,
) -> list[int]:
    previous_layers = source_layers
    previous_count = source_parameter_count
    counts = []
    for stage in stages:
        count = _stage_parameter_count(stage, vocab_size)
        if stage.target_params is not None and stage.target_params != count:
            raise ValueError(
                f"stage {stage.name!r} target_params assertion {stage.target_params} "
                f"does not match exact Qwen3 count {count}."
            )
        if stage.num_hidden_layers > previous_layers:
            raise ValueError(
                f"stage {stage.name!r} increases depth; progressive fold stages "
                "must not exceed donor depth."
            )
        if previous_layers > 2 * stage.num_hidden_layers:
            raise ValueError(
                f"stage {stage.name!r} would require donor spans longer than two; "
                "insert an intermediate stage."
            )
        if count >= previous_count:
            raise ValueError(
                f"stage {stage.name!r} must reduce exact parameter count below its donor."
            )
        counts.append(count)
        previous_layers = stage.num_hidden_layers
        previous_count = count
    return counts


def build_progressive_plan(
    *,
    source_model: Path,
    output_dir: Path,
    cache_dir: Path,
    calibration_file: Path,
    eval_suite: Path,
    python: str | Path,
    stages: Sequence[StageSpec],
    compose_settings: ComposeSettings,
    blank_dtype: str = "float32",
    eval_dtype: str | None = None,
    manual_seed: int = 0,
    cuda_visible_devices: str | None = None,
    source_content_sha256: str | None = None,
    implementation_sha256: str | None = None,
    cached_parent_eval: Path | None = None,
    cached_parent_manual: Path | None = None,
) -> dict[str, Any]:
    """Build a complete progressive run plan without loading a model or GPU."""

    _validate_stage_list(stages)
    compose_settings.validate()
    source = source_model.resolve(strict=True)
    calibration = calibration_file.resolve(strict=True)
    suite = eval_suite.resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source model must be a local checkpoint directory.")
    if not calibration.is_file() or calibration.stat().st_size <= 0:
        raise ValueError("calibration file must be non-empty.")
    if not suite.is_file() or suite.stat().st_size <= 0:
        raise ValueError("evaluation suite must be non-empty.")
    if manual_seed < 0:
        raise ValueError("manual seed must be non-negative.")
    if not blank_dtype:
        raise ValueError("blank dtype cannot be empty.")
    if source_content_sha256 is not None and not _HEX_SHA256.fullmatch(
        source_content_sha256
    ):
        raise ValueError("source_content_sha256 must contain 64 hexadecimal digits.")
    if implementation_sha256 is not None and not _HEX_SHA256.fullmatch(
        implementation_sha256
    ):
        raise ValueError("implementation_sha256 must contain 64 hexadecimal digits.")

    config = _read_source_config(source)
    vocab_size = int(config["vocab_size"])
    parent_parameter_count = _source_parameter_count(config)
    stage_counts = _validate_progressive_shapes(
        source_layers=int(config["num_hidden_layers"]),
        source_parameter_count=parent_parameter_count,
        stages=stages,
        vocab_size=vocab_size,
    )
    source_digest = source_content_sha256 or checkpoint_content_sha256(source)
    implementation_digest = implementation_sha256 or implementation_content_sha256()
    calibration_digest = _file_sha256(calibration)
    suite_digest = _file_sha256(suite)
    eval_dtype_value = eval_dtype or compose_settings.dtype
    output = Path(os.path.abspath(os.fspath(output_dir)))
    cache = Path(os.path.abspath(os.fspath(cache_dir)))
    python_value = str(python)
    launcher = Path(__file__).resolve()

    parent_model_identity = _canonical_sha256(
        {
            "kind": "local-checkpoint-content",
            "checkpoint_sha256": source_digest,
        }
    )
    parent_cache = _cache_record(
        cache_dir=cache,
        model_identity=parent_model_identity,
        eval_suite_sha256=suite_digest,
        implementation_sha256=implementation_digest,
        eval_dtype=eval_dtype_value,
        device=compose_settings.device,
        manual_seed=manual_seed,
    )
    supplied_parent_eval = _validated_cached_parent_report(
        cached_parent_eval, kind="eval"
    )
    supplied_parent_manual = _validated_cached_parent_report(
        cached_parent_manual, kind="manual"
    )
    if supplied_parent_eval is not None:
        parent_cache["generated_eval_report"] = parent_cache["eval_report"]
        parent_cache["eval_report"] = supplied_parent_eval["path"]
        parent_cache["eval_source"] = "supplied-validated-cache"
    else:
        parent_cache["eval_source"] = "content-addressed-cache"
    if supplied_parent_manual is not None:
        parent_cache["generated_manual_report"] = parent_cache["manual_report"]
        parent_cache["manual_report"] = supplied_parent_manual["path"]
        parent_cache["manual_source"] = "supplied-validated-cache"
    else:
        parent_cache["manual_source"] = "content-addressed-cache"
    parent_eval_command = build_eval_command(
        python=python_value,
        model=source,
        report=Path(parent_cache["eval_report"]),
        eval_suite=suite,
        device=compose_settings.device,
        dtype=eval_dtype_value,
    )
    parent_manual_command = build_manual_command(
        python=python_value,
        launcher=launcher,
        model=source,
        report=Path(parent_cache["manual_report"]),
        device=compose_settings.device,
        dtype=eval_dtype_value,
        seed=manual_seed,
    )
    parent_record = {
        "role": "original_parent",
        "model_path": str(source),
        "model_identity": parent_model_identity,
        "checkpoint_content_sha256": source_digest,
        "supplied_eval": supplied_parent_eval,
        "supplied_manual": supplied_parent_manual,
        "parameter_count": parent_parameter_count,
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "cache": parent_cache,
        "commands": {
            "eval": _command_record(parent_eval_command),
            "manual": _command_record(parent_manual_command),
        },
    }

    stage_records = []
    current_source = source
    current_model_identity = parent_model_identity
    current_parameter_count = parent_parameter_count
    for index, (stage, parameter_count) in enumerate(
        zip(stages, stage_counts, strict=True), start=1
    ):
        stage_root = output / "stages" / f"{index:02d}_{stage.name}"
        blank_target = stage_root / "blank_target"
        folded_model = stage_root / "model"
        target_identity = _canonical_sha256(
            {
                "kind": "seeded-config-only-qwen3-target",
                "implementation_version": IMPLEMENTATION_VERSION,
                "implementation_sha256": implementation_digest,
                "tokenizer_identity": parent_model_identity,
                "source_model_identity": current_model_identity,
                "stage": asdict(stage),
                "parameter_count": parameter_count,
                "blank_dtype": blank_dtype,
            }
        )
        rounds = (
            compose_settings.rollout_refold_rounds
            if stage.rollout_refold_rounds is None
            else stage.rollout_refold_rounds
        )
        model_identity = _canonical_sha256(
            {
                "kind": "progressive-zero-recovery-qwen-pair-compose",
                "implementation_version": IMPLEMENTATION_VERSION,
                "implementation_sha256": implementation_digest,
                "donor_model_identity": current_model_identity,
                "blank_target_identity": target_identity,
                "calibration_sha256": calibration_digest,
                "compose_settings": asdict(compose_settings),
                "stage_rollout_refold_rounds": rounds,
            }
        )
        stage_cache = _cache_record(
            cache_dir=cache,
            model_identity=model_identity,
            eval_suite_sha256=suite_digest,
            implementation_sha256=implementation_digest,
            eval_dtype=eval_dtype_value,
            device=compose_settings.device,
            manual_seed=manual_seed,
        )
        target_command = build_target_command(
            python=python_value,
            tokenizer_model=source,
            source_model=current_source,
            output=blank_target,
            source_parameter_count=current_parameter_count,
            parameter_count=parameter_count,
            stage=stage,
            dtype=blank_dtype,
        )
        compose_command = build_compose_command(
            python=python_value,
            source_model=current_source,
            blank_target=blank_target,
            output=folded_model,
            calibration_file=calibration,
            settings=compose_settings,
            rollout_refold_rounds=rounds,
        )
        eval_command = build_eval_command(
            python=python_value,
            model=folded_model,
            report=Path(stage_cache["eval_report"]),
            eval_suite=suite,
            device=compose_settings.device,
            dtype=eval_dtype_value,
        )
        manual_command = build_manual_command(
            python=python_value,
            launcher=launcher,
            model=folded_model,
            report=Path(stage_cache["manual_report"]),
            device=compose_settings.device,
            dtype=eval_dtype_value,
            seed=manual_seed,
        )
        commands = {
            "build_blank_target": _command_record(target_command),
            "compose": _command_record(compose_command),
            "eval": _command_record(eval_command),
            "manual": _command_record(manual_command),
        }
        _assert_zero_recovery_commands(record["argv"] for record in commands.values())
        stage_records.append(
            {
                "index": index,
                "name": stage.name,
                "shape": {
                    "vocab_size": vocab_size,
                    "hidden_size": stage.hidden_size,
                    "intermediate_size": stage.intermediate_size,
                    "num_hidden_layers": stage.num_hidden_layers,
                    "num_attention_heads": stage.num_attention_heads,
                    "num_key_value_heads": stage.num_key_value_heads,
                    "head_dim": stage.head_dim,
                    "tie_word_embeddings": stage.tie_word_embeddings,
                    "attention_bias": stage.attention_bias,
                },
                "seed": stage.seed,
                "parameter_count": parameter_count,
                "donor_parameter_count": current_parameter_count,
                "stage_compression_ratio": current_parameter_count / parameter_count,
                "cumulative_compression_ratio": parent_parameter_count
                / parameter_count,
                "donor_model_path": str(current_source),
                "blank_target_path": str(blank_target),
                "model_path": str(folded_model),
                "blank_target_identity": target_identity,
                "model_identity": model_identity,
                "blank_target_origin": {
                    "type": "config-and-seeded-random-initialization-only",
                    "pretrained_child_weights_loaded": False,
                    "builder": "hf-qwen-build-target",
                },
                "depth_plan": {
                    "adaptive": True,
                    "method": "activation-costed-contiguous-dynamic-programming",
                    "maximum_donor_span_length": 2,
                },
                "rollout_refold_rounds": rounds,
                "cache": stage_cache,
                "artifacts": {
                    "blank_target": str(blank_target),
                    "folded_model": str(folded_model),
                    "eval_report": stage_cache["eval_report"],
                    "manual_report": stage_cache["manual_report"],
                },
                "commands": commands,
            }
        )
        current_source = folded_model
        current_model_identity = model_identity
        current_parameter_count = parameter_count

    all_commands = [
        parent_record["commands"]["eval"]["argv"],
        parent_record["commands"]["manual"]["argv"],
    ]
    for record in stage_records:
        all_commands.extend(command["argv"] for command in record["commands"].values())
    _assert_zero_recovery_commands(all_commands)
    final_stage = stage_records[-1]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "implementation_version": IMPLEMENTATION_VERSION,
        "implementation_sha256": implementation_digest,
        "inputs": {
            "source_model": str(source),
            "source_checkpoint_sha256": source_digest,
            "calibration_file": str(calibration),
            "calibration_sha256": calibration_digest,
            "eval_suite": str(suite),
            "eval_suite_sha256": suite_digest,
        },
        "output_dir": str(output),
        "cache_dir": str(cache),
        "environment": {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "settings": {
            "compose": asdict(compose_settings),
            "blank_dtype": blank_dtype,
            "eval_dtype": eval_dtype_value,
            "manual_seed": manual_seed,
            "exact_eval": {
                "behavior_probes": EXACT_BEHAVIOR_PROBES,
                "generation_prompts": EXACT_GENERATION_PROMPTS,
                "choice_probes": EXACT_CHOICE_PROBES,
            },
        },
        "parent": parent_record,
        "stages": stage_records,
        "final": {
            "stage_index": final_stage["index"],
            "stage_name": final_stage["name"],
            "model_path": final_stage["model_path"],
            "parameter_count": final_stage["parameter_count"],
            "cumulative_compression_ratio": final_stage["cumulative_compression_ratio"],
            "eval_report": final_stage["artifacts"]["eval_report"],
            "manual_report": final_stage["artifacts"]["manual_report"],
            "eval_command": final_stage["commands"]["eval"],
            "manual_command": final_stage["commands"]["manual"],
        },
        "cache_policy": {
            "identity_basis": (
                "source content, derived model identity, implementation content, "
                "suite content, and evaluation settings"
            ),
            "parent_eval_recomputed_only_on_cache_miss": True,
            "stage_evals_recomputed_only_on_cache_miss": True,
            "manual_outputs_recomputed_only_on_cache_miss": True,
            "validated_supplied_parent_eval_reused": supplied_parent_eval is not None,
            "validated_supplied_parent_manual_reused": (
                supplied_parent_manual is not None
            ),
        },
        "zero_recovery_assertion": {
            "asserted": True,
            "gradient_updates": 0,
            "recovery_commands": 0,
            "training_commands": 0,
            "blank_children_use_pretrained_weights": False,
            "allowed_model_mutations": (
                "seeded config-only target construction and closed-form pair composition"
            ),
            "command_count_checked": len(all_commands),
        },
    }


def _append_cached_command(
    lines: list[str], *, artifact: str, command: Sequence[str], label: str
) -> None:
    quoted_artifact = shlex.quote(artifact)
    lines.extend(
        [
            f"mkdir -p {shlex.quote(str(Path(artifact).parent))}",
            f"if [[ -s {quoted_artifact} ]]; then",
            f'  echo "cache_hit={label} artifact={quoted_artifact}"',
            "else",
            f'  echo "cache_miss={label} start=$(date -Is)"',
            f"  {shlex.join([str(token) for token in command])}",
            f"  [[ -s {quoted_artifact} ]] || {{ echo {shlex.quote('missing ' + label)} >&2; exit 3; }}",
            "fi",
        ]
    )


def _append_model_command(
    lines: list[str],
    *,
    model_dir: str,
    command: Sequence[str],
    label: str,
    identity: str,
) -> None:
    config = str(Path(model_dir) / "config.json")
    marker = str(Path(model_dir) / ".neural_forge_progressive_identity")
    quoted_model = shlex.quote(model_dir)
    quoted_config = shlex.quote(config)
    quoted_marker = shlex.quote(marker)
    quoted_identity = shlex.quote(identity)
    lines.extend(
        [
            f"mkdir -p {shlex.quote(str(Path(model_dir).parent))}",
            (
                f"if [[ -s {quoted_config} && -s {quoted_marker} ]] && "
                f"grep -Fqx -- {quoted_identity} {quoted_marker}; then"
            ),
            f'  echo "model_cache_hit={label} model={quoted_model}"',
            "else",
            f"  if [[ -e {quoted_model} ]]; then",
            f"    echo {shlex.quote('incomplete model artifact for ' + label)} >&2",
            "    exit 3",
            "  fi",
            f'  echo "model_build={label} start=$(date -Is)"',
            f"  {shlex.join([str(token) for token in command])}",
            f"  [[ -s {quoted_config} ]] || {{ echo {shlex.quote('missing model config for ' + label)} >&2; exit 3; }}",
            f"  printf '%s\\n' {quoted_identity} > {quoted_marker}",
            "fi",
        ]
    )


def render_run_script(plan: Mapping[str, Any], manifest_path: Path) -> str:
    """Render a sequential local sidecar; it never submits or detaches jobs."""

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "umask 077",
        f"RUN_MANIFEST={shlex.quote(str(manifest_path))}",
        f"mkdir -p {shlex.quote(str(plan['output_dir']))}",
        'echo "progressive_zero_recovery_start=$(date -Is) manifest=$RUN_MANIFEST"',
        "export HF_HUB_OFFLINE=1",
        "export TRANSFORMERS_OFFLINE=1",
        "export TOKENIZERS_PARALLELISM=false",
    ]
    cuda_devices = plan["environment"].get("CUDA_VISIBLE_DEVICES")
    if cuda_devices:
        lines.append(f"export CUDA_VISIBLE_DEVICES={shlex.quote(str(cuda_devices))}")
    parent = plan["parent"]
    _append_cached_command(
        lines,
        artifact=parent["cache"]["eval_report"],
        command=parent["commands"]["eval"]["argv"],
        label="parent_eval",
    )
    _append_cached_command(
        lines,
        artifact=parent["cache"]["manual_report"],
        command=parent["commands"]["manual"]["argv"],
        label="parent_manual",
    )
    for stage in plan["stages"]:
        prefix = f"stage_{stage['index']:02d}_{stage['name']}"
        _append_model_command(
            lines,
            model_dir=stage["blank_target_path"],
            command=stage["commands"]["build_blank_target"]["argv"],
            label=f"{prefix}_blank",
            identity=stage["blank_target_identity"],
        )
        _append_model_command(
            lines,
            model_dir=stage["model_path"],
            command=stage["commands"]["compose"]["argv"],
            label=f"{prefix}_fold",
            identity=stage["model_identity"],
        )
        _append_cached_command(
            lines,
            artifact=stage["cache"]["eval_report"],
            command=stage["commands"]["eval"]["argv"],
            label=f"{prefix}_eval",
        )
        _append_cached_command(
            lines,
            artifact=stage["cache"]["manual_report"],
            command=stage["commands"]["manual"]["argv"],
            label=f"{prefix}_manual",
        )
    success_path = Path(str(plan["output_dir"])) / "SUCCESS"
    lines.extend(
        [
            f"date -Is > {shlex.quote(str(success_path))}",
            'echo "progressive_zero_recovery_done=$(date -Is)"',
            "",
        ]
    )
    script = "\n".join(lines)
    structured_commands = [
        parent["commands"]["eval"]["argv"],
        parent["commands"]["manual"]["argv"],
    ]
    for stage in plan["stages"]:
        structured_commands.extend(
            value["argv"] for value in stage["commands"].values()
        )
    _assert_zero_recovery_commands(structured_commands)
    return script


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_manifest_and_digest(path: Path, manifest: Mapping[str, Any]) -> None:
    _write_json(path, manifest)
    digest_path = path.with_suffix(path.suffix + ".sha256")
    digest_path.write_text(
        f"{_file_sha256(path)}  {path.name}\n", encoding="ascii", newline="\n"
    )


def _manual_eval(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Internal deterministic manual eval.")
    parser.add_argument("model", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(list(argv))
    if args.seed < 0 or args.max_new_tokens <= 0:
        raise ValueError("manual eval seed/length is invalid.")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        local_files_only=True,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    generations = []
    for prompt in MANUAL_PROMPTS:
        encoded = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generations.append(
            {
                "prompt": prompt,
                "output": tokenizer.decode(generated[0], skip_special_tokens=True),
            }
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.report,
        {
            "schema": "neural-forge-progressive-manual-generations-v1",
            "model": str(args.model),
            "seed": args.seed,
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "manual_generations": generations,
        },
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--calibration-file", type=Path, required=True)
    parser.add_argument("--eval-suite", type=Path, required=True)
    parser.add_argument("--cached-parent-eval", type=Path)
    parser.add_argument("--cached-parent-manual", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--stage-manifest", type=Path)
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help="Replace defaults with repeated JSON or compact key=value stage specs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--manual-seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--source-content-sha256")
    parser.add_argument("--blank-dtype", default="float32")
    parser.add_argument("--eval-dtype")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-tokens-per-layer", type=int, default=4096)
    parser.add_argument("--max-anchors", type=int, default=4096)
    parser.add_argument("--max-calibration-texts", type=int, default=256)
    parser.add_argument("--basis-mode", default="activation-pca")
    parser.add_argument("--basis-preprocess", default="uncentered-rowrms")
    parser.add_argument("--activation-domain-balance", default="keyword")
    parser.add_argument(
        "--no-observable-coordinates",
        action="store_false",
        dest="observable_coordinates",
    )
    parser.add_argument("--observable-coordinate-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--no-path-decomposition",
        action="store_false",
        dest="path_decomposition",
    )
    parser.add_argument("--path-decomposition-max-rows", type=int, default=1024)
    parser.add_argument("--path-decomposition-ridge", type=float, default=1e-6)
    parser.add_argument("--path-decomposition-sketch-size", type=int, default=64)
    parser.add_argument(
        "--path-decomposition-interpolation-mode", default="sparse-signature"
    )
    parser.add_argument("--adaptive-depth-critical-quantile", type=float, default=0.8)
    parser.add_argument(
        "--adaptive-depth-causal-magnitude-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--adaptive-depth-incompatibility-weight", type=float, default=1.0
    )
    parser.add_argument(
        "--adaptive-depth-reconstruction-weight", type=float, default=1.0
    )
    parser.add_argument("--rollout-refold-rounds", type=int, default=0)
    parser.add_argument("--rollout-refold-max-prompts", type=int, default=64)
    parser.add_argument("--rollout-refold-max-new-tokens", type=int, default=24)
    parser.add_argument("--rollout-refold-batch-size", type=int, default=4)
    parser.add_argument("--rollout-refold-candidate-alphas", default="1,0.5,0.25")
    parser.add_argument("--rollout-gate-top-k", type=int, default=5)
    parser.add_argument("--rollout-gate-min-top1-gain", type=float, default=0.0)
    parser.add_argument(
        "--rollout-gate-min-action-logprob-gain", type=float, default=0.0
    )
    parser.add_argument(
        "--rollout-gate-max-repetition-regression", type=float, default=0.0
    )
    parser.add_argument(
        "--rollout-gate-max-longest-run-regression", type=int, default=0
    )
    parser.add_argument(
        "--rollout-gate-min-action-count-ratio", type=float, default=0.9
    )
    parser.add_argument("--fit-gate-validation-fraction", type=float, default=0.25)
    parser.add_argument("--fit-gate-min-validation-rows", type=int, default=128)
    parser.add_argument("--fit-gate-min-relative-improvement", type=float, default=0.02)
    parser.add_argument("--fit-gate-max-cosine-drop", type=float, default=0.0)
    parser.add_argument("--fit-gate-max-relative-rmse", type=float, default=0.75)
    parser.add_argument("--head-validation-vocab-rows", type=int, default=4096)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--candidate-dtype", default="float32")
    parser.add_argument("--device", default="cuda:0")
    return parser


def _compose_settings_from_args(args: argparse.Namespace) -> ComposeSettings:
    return ComposeSettings(
        ridge=args.ridge,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_tokens_per_layer=args.max_tokens_per_layer,
        max_anchors=args.max_anchors,
        max_calibration_texts=args.max_calibration_texts,
        basis_mode=args.basis_mode,
        basis_preprocess=args.basis_preprocess,
        activation_domain_balance=args.activation_domain_balance,
        observable_coordinates=args.observable_coordinates,
        observable_coordinate_ridge=args.observable_coordinate_ridge,
        path_decomposition=args.path_decomposition,
        path_decomposition_max_rows=args.path_decomposition_max_rows,
        path_decomposition_ridge=args.path_decomposition_ridge,
        path_decomposition_sketch_size=args.path_decomposition_sketch_size,
        path_decomposition_interpolation_mode=(
            args.path_decomposition_interpolation_mode
        ),
        adaptive_depth_critical_quantile=args.adaptive_depth_critical_quantile,
        adaptive_depth_causal_magnitude_weight=(
            args.adaptive_depth_causal_magnitude_weight
        ),
        adaptive_depth_incompatibility_weight=(
            args.adaptive_depth_incompatibility_weight
        ),
        adaptive_depth_reconstruction_weight=(
            args.adaptive_depth_reconstruction_weight
        ),
        rollout_refold_rounds=args.rollout_refold_rounds,
        rollout_refold_max_prompts=args.rollout_refold_max_prompts,
        rollout_refold_max_new_tokens=args.rollout_refold_max_new_tokens,
        rollout_refold_batch_size=args.rollout_refold_batch_size,
        rollout_refold_candidate_alphas=args.rollout_refold_candidate_alphas,
        rollout_gate_top_k=args.rollout_gate_top_k,
        rollout_gate_min_top1_gain=args.rollout_gate_min_top1_gain,
        rollout_gate_min_action_logprob_gain=(
            args.rollout_gate_min_action_logprob_gain
        ),
        rollout_gate_max_repetition_regression=(
            args.rollout_gate_max_repetition_regression
        ),
        rollout_gate_max_longest_run_regression=(
            args.rollout_gate_max_longest_run_regression
        ),
        rollout_gate_min_action_count_ratio=(args.rollout_gate_min_action_count_ratio),
        fit_gate_validation_fraction=args.fit_gate_validation_fraction,
        fit_gate_min_validation_rows=args.fit_gate_min_validation_rows,
        fit_gate_min_relative_improvement=args.fit_gate_min_relative_improvement,
        fit_gate_max_cosine_drop=args.fit_gate_max_cosine_drop,
        fit_gate_max_relative_rmse=args.fit_gate_max_relative_rmse,
        head_validation_vocab_rows=args.head_validation_vocab_rows,
        dtype=args.dtype,
        candidate_dtype=args.candidate_dtype,
        device=args.device,
    )


def _validate_tag(tag: str) -> str:
    if not _STAGE_NAME.fullmatch(tag):
        raise ValueError("tag must be one safe path component.")
    return tag


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if values and values[0] == "_manual-eval":
        return _manual_eval(values[1:])
    args = build_parser().parse_args(values)
    if args.seed < 0:
        raise ValueError("seed must be non-negative.")
    tag = _validate_tag(args.tag)
    stages = load_stage_specs(
        stage_manifest=args.stage_manifest,
        stage_overrides=args.stage,
        default_seed=args.seed,
    )
    settings = _compose_settings_from_args(args)
    settings.validate()
    run_root = args.run_root.resolve(strict=True)
    if not run_root.is_dir():
        raise ValueError("run-root must be an existing directory.")
    output = run_root / tag
    output.mkdir(exist_ok=False)
    cache = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else run_root / ".qwen_progressive_cache"
    )
    cache.mkdir(parents=True, exist_ok=True)
    try:
        plan = build_progressive_plan(
            source_model=args.source_model,
            output_dir=output,
            cache_dir=cache,
            calibration_file=args.calibration_file,
            eval_suite=args.eval_suite,
            python=args.python,
            stages=stages,
            compose_settings=settings,
            blank_dtype=args.blank_dtype,
            eval_dtype=args.eval_dtype,
            manual_seed=args.manual_seed,
            cuda_visible_devices=args.cuda_visible_devices,
            source_content_sha256=args.source_content_sha256,
            cached_parent_eval=args.cached_parent_eval,
            cached_parent_manual=args.cached_parent_manual,
        )
        manifest_path = output / "run_manifest.json"
        run_script = output / "run_progressive_fold.sh"
        script = render_run_script(plan, manifest_path)
        run_script.write_text(script, encoding="utf-8", newline="\n")
        os.chmod(run_script, 0o500)
        plan["orchestration"] = {
            "mode": "bounded-sequential-local-sidecar",
            "remote_submission": False,
            "detached_execution": False,
            "execute_requested": bool(args.execute),
            "status": "planned",
            "run_script": str(run_script),
            "run_script_sha256": _file_sha256(run_script),
        }
        _write_manifest_and_digest(manifest_path, plan)
        if args.execute:
            plan["orchestration"]["status"] = "running"
            plan["orchestration"]["started_at"] = datetime.now(timezone.utc).isoformat()
            _write_manifest_and_digest(manifest_path, plan)
            try:
                subprocess.run(["bash", str(run_script)], cwd=output, check=True)
            except (OSError, subprocess.CalledProcessError):
                plan["orchestration"]["status"] = "failed"
                plan["orchestration"]["finished_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                _write_manifest_and_digest(manifest_path, plan)
                raise
            plan["orchestration"]["status"] = "completed"
            plan["orchestration"]["finished_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            _write_manifest_and_digest(manifest_path, plan)
    except Exception:
        if not any(output.iterdir()):
            output.rmdir()
        raise
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "run_script": str(run_script),
                "execute_requested": bool(args.execute),
                "stage_count": len(stages),
                "final_parameter_count": plan["final"]["parameter_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
