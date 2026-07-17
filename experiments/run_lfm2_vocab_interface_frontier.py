#!/usr/bin/env python3
"""Run a large tied-vocabulary-interface folding frontier on local LFM2.

One deterministic CPU float64 SVD of the exact tied embedding/LM-head matrix
is reused for every raw-SVD rank prefix.  The primary allocation starts from
the authenticated strong 10%-compressed child and evaluates 15/20/25/35%
total parent-relative targets.  An optional direct-parent allocation uses the
same decomposition for 15/20% targets, isolating whether distributed
width-plus-vocabulary compression composes better than vocabulary-only
compression at the same total size.

There is no training, optimizer, backward pass, network access, hub loader, or
new parent trace/response generation.  Every candidate is physically installed
through the LFM2 embedding/output setters, evaluated against immutable caches,
passed through the current strict post-hoc intelligence gate, and rolled back
exactly.  Checkpoint publication is forbidden unless both strict intelligence
and an authenticated factorized reload pass; this raw frontier does not publish
diagnostic candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import evaluate_lfm2_intelligence_retention as strict_runner
    from experiments import run_lfm2_partial_neuron_bank_fold as bank_runner
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
    from experiments import (
        run_lfm2_residual_parameter_sharing_fold as shared_runner,
    )
except ImportError:  # Direct execution from experiments/.
    import evaluate_lfm2_intelligence_retention as strict_runner
    import run_lfm2_partial_neuron_bank_fold as bank_runner
    import run_lfm2_residual_attention_fold as fold_runtime
    import run_lfm2_residual_parameter_sharing_fold as shared_runner

from neural_forge.hf_factorized_vocab import (
    TiedVocabularySVDAnalysis,
    analyze_tied_vocabulary_modules,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_vocab_interface_fold import (
    InstalledLfm2VocabularyInterface,
    exact_model_parameter_counts,
    install_lfm2_tied_vocabulary_pair,
    solve_tied_vocabulary_target,
)


RUN_SCHEMA = "neural-forge-lfm2-tied-vocab-interface-frontier-run-v1"
ATTEMPT_SCHEMA = "neural-forge-lfm2-tied-vocab-interface-attempt-v1"
EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS = 206_726_912
EXPECTED_VOCAB_SIZE = 65_536
EXPECTED_HIDDEN_SIZE = 1_024
EXPECTED_WHOLE_OUTPUT_CASE_COUNT = 28
STRONG_COMPRESSION_FRACTIONS = (0.15, 0.20, 0.25, 0.35)
DIRECT_PARENT_COMPRESSION_FRACTIONS = (0.15, 0.20)


class Lfm2VocabularyFrontierError(RuntimeError):
    """An authenticated input, physical audit, evaluation, or rollback failed."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2VocabularyFrontierError(
            "Frontier evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while piece := handle.read(1024 * 1024):
            digest.update(piece)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2VocabularyFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2VocabularyFrontierError(
            f"{name} must be a positive integer."
        )
    return value


def _fraction_tuple(
    values: Any,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise Lfm2VocabularyFrontierError(
            f"{name} must be a nonempty sequence of fractions."
        )
    result = tuple(float(value) for value in values)
    if (
        not result
        or any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in result)
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
    ):
        raise Lfm2VocabularyFrontierError(
            f"{name} must contain unique ascending fractions in (0, 1)."
        )
    return result


def _local_directory(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2VocabularyFrontierError(
            f"{name} must be an explicit local directory."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2VocabularyFrontierError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _local_file(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2VocabularyFrontierError(
            f"{name} must be an explicit local file."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2VocabularyFrontierError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


@dataclass(frozen=True, slots=True)
class VocabularyFrontierConfig:
    strong_source_model: Path
    strong_source_cache: Path
    original_parent_model: Path
    parent_trajectory_cache: Path
    whole_output_parent_cache: Path
    parent_output_envelope: Path
    expected_parent_output_envelope_sha256: str
    expected_parent_cache_key: str
    expected_parent_state_fingerprint: str
    tokenizer_directory: Path
    expected_tokenizer_fingerprint: str
    expected_strong_source_state_sha256: str
    expected_strong_source_cache_key: str
    work_root: Path
    checkpoint_root: Path
    report_path: Path
    device: str = "cuda:6"
    dtype: str = "bfloat16"
    strong_compression_fractions: tuple[float, ...] = (
        STRONG_COMPRESSION_FRACTIONS
    )
    direct_parent_compression_fractions: tuple[float, ...] = (
        DIRECT_PARENT_COMPRESSION_FRACTIONS
    )
    expected_original_parent_unique_parameters: int = (
        EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
    )
    expected_strong_source_unique_parameters: int = (
        EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS
    )
    expected_vocab_size: int = EXPECTED_VOCAB_SIZE
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE
    expected_whole_output_case_count: int = EXPECTED_WHOLE_OUTPUT_CASE_COUNT
    minimum_parent_residual_cosine: float = 0.80
    minimum_incumbent_residual_cosine: float = 0.95
    minimum_parent_trajectory_score: float = 0.80
    minimum_incumbent_trajectory_score: float = 0.80
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 2
    whole_output_maximum_new_fragment_responses: int = 2
    whole_output_minimum_parent_nonempty_retention: float = 0.90
    whole_output_minimum_task_score_retention: float = 0.90
    whole_output_maximum_parent_task_score_regressions: int = 2
    whole_output_maximum_new_generation_limit_saturations: int = 2
    whole_output_maximum_repeated_ngram_share_regressions: int = 2
    whole_output_maximum_repeated_bigram_share_delta: float = 0.10

    @property
    def cached_parent_outputs(self) -> Path:
        """Compatibility alias for the shared whole-output evaluator."""

        return self.parent_output_envelope

    @property
    def cached_parent_outputs_sha256(self) -> str:
        """Compatibility alias for the authenticated envelope file hash."""

        return self.expected_parent_output_envelope_sha256

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "strong_source_model",
            _local_directory(self.strong_source_model, "strong_source_model"),
        )
        object.__setattr__(
            self,
            "strong_source_cache",
            _local_directory(self.strong_source_cache, "strong_source_cache"),
        )
        object.__setattr__(
            self,
            "original_parent_model",
            _local_directory(self.original_parent_model, "original_parent_model"),
        )
        object.__setattr__(
            self,
            "parent_trajectory_cache",
            _local_directory(
                self.parent_trajectory_cache,
                "parent_trajectory_cache",
            ),
        )
        object.__setattr__(
            self,
            "whole_output_parent_cache",
            _local_directory(
                self.whole_output_parent_cache,
                "whole_output_parent_cache",
            ),
        )
        object.__setattr__(
            self,
            "parent_output_envelope",
            _local_file(self.parent_output_envelope, "parent_output_envelope"),
        )
        object.__setattr__(
            self,
            "tokenizer_directory",
            _local_directory(self.tokenizer_directory, "tokenizer_directory"),
        )
        for name in (
            "expected_parent_output_envelope_sha256",
            "expected_parent_cache_key",
            "expected_parent_state_fingerprint",
            "expected_tokenizer_fingerprint",
            "expected_strong_source_state_sha256",
            "expected_strong_source_cache_key",
        ):
            _digest(getattr(self, name), name)
        object.__setattr__(
            self,
            "strong_compression_fractions",
            _fraction_tuple(
                self.strong_compression_fractions,
                "strong_compression_fractions",
            ),
        )
        object.__setattr__(
            self,
            "direct_parent_compression_fractions",
            _fraction_tuple(
                self.direct_parent_compression_fractions,
                "direct_parent_compression_fractions",
            ),
        )
        for name in (
            "expected_original_parent_unique_parameters",
            "expected_strong_source_unique_parameters",
            "expected_vocab_size",
            "expected_hidden_size",
            "expected_whole_output_case_count",
        ):
            _positive_int(getattr(self, name), name)
        if self.expected_strong_source_unique_parameters >= (
            self.expected_original_parent_unique_parameters
        ):
            raise Lfm2VocabularyFrontierError(
                "The strong source must be smaller than the original parent."
            )
        if self.checkpoint_root.exists():
            raise Lfm2VocabularyFrontierError(
                "checkpoint_root must not exist before the strict frontier."
            )
        if self.report_path.exists():
            raise Lfm2VocabularyFrontierError(
                "report_path must not already exist."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "strong_source_model": str(self.strong_source_model),
            "strong_source_cache": str(self.strong_source_cache),
            "original_parent_model": str(self.original_parent_model),
            "parent_trajectory_cache": str(self.parent_trajectory_cache),
            "whole_output_parent_cache": str(self.whole_output_parent_cache),
            "parent_output_envelope": str(self.parent_output_envelope),
            "expected_parent_output_envelope_sha256": (
                self.expected_parent_output_envelope_sha256
            ),
            "expected_parent_cache_key": self.expected_parent_cache_key,
            "expected_parent_state_fingerprint": (
                self.expected_parent_state_fingerprint
            ),
            "tokenizer_directory": str(self.tokenizer_directory),
            "expected_tokenizer_fingerprint": (
                self.expected_tokenizer_fingerprint
            ),
            "expected_strong_source_state_sha256": (
                self.expected_strong_source_state_sha256
            ),
            "expected_strong_source_cache_key": (
                self.expected_strong_source_cache_key
            ),
            "work_root": str(self.work_root),
            "checkpoint_root": str(self.checkpoint_root),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "strong_compression_fractions": list(
                self.strong_compression_fractions
            ),
            "direct_parent_compression_fractions": list(
                self.direct_parent_compression_fractions
            ),
            "expected_original_parent_unique_parameters": (
                self.expected_original_parent_unique_parameters
            ),
            "expected_strong_source_unique_parameters": (
                self.expected_strong_source_unique_parameters
            ),
            "expected_vocab_size": self.expected_vocab_size,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_whole_output_case_count": (
                self.expected_whole_output_case_count
            ),
            "factorization": "raw_weight_svd_float64_single_decomposition",
            "training_allowed": False,
            "new_parent_trace_generation_allowed": False,
            "new_parent_response_generation_allowed": False,
            "checkpoint_publication_requires_strict_and_loader_pass": True,
        }


def _checkpoint_absence(config: VocabularyFrontierConfig) -> dict[str, Any]:
    if config.checkpoint_root.exists():
        raise Lfm2VocabularyFrontierError(
            "A checkpoint appeared without strict and loader acceptance."
        )
    return {
        "path": str(config.checkpoint_root),
        "exists": False,
        "files": 0,
        "strict_pass_required": True,
        "authenticated_factorized_reload_required": True,
        "published": False,
    }


def _source_authentication(
    model: torch.nn.Module,
    *,
    source_cache: Lfm2CircuitTraceCache,
    expected_unique_parameters: int,
    expected_state_sha256: str,
    expected_cache_key: str,
) -> dict[str, Any]:
    counts = exact_model_parameter_counts(model)
    state = fold_runtime.model_state_sha256(model)
    identity = fingerprint_lfm2_parent(model).identity_sha256
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if (
        counts["unique"] != expected_unique_parameters
        or state != expected_state_sha256
        or source_cache.cache_key != expected_cache_key
        or source_cache.parent_fingerprint != identity
        or not isinstance(embedding, torch.nn.Embedding)
        or not isinstance(head, torch.nn.Linear)
        or embedding.weight is not head.weight
    ):
        raise Lfm2VocabularyFrontierError(
            "Source state, count, cache, identity, or dense tie authentication failed."
        )
    if (
        embedding.num_embeddings != EXPECTED_VOCAB_SIZE
        or embedding.embedding_dim != EXPECTED_HIDDEN_SIZE
    ):
        raise Lfm2VocabularyFrontierError(
            "Source tied vocabulary dimensions differ from the LFM2 contract."
        )
    gradients = shared_runner.gradient_audit(model)
    if not gradients["zero_gradient"]:
        raise Lfm2VocabularyFrontierError(
            "Source model contains pre-existing gradients."
        )
    return {
        "state_sha256": state,
        "identity_sha256": identity,
        "parameter_counts": counts,
        "cache_key": source_cache.cache_key,
        "cache_parent_fingerprint": source_cache.parent_fingerprint,
        "dense_tied_weight_parameter_identity": True,
        "vocab_size": embedding.num_embeddings,
        "hidden_size": embedding.embedding_dim,
        "vocab_weight_sha256": fold_runtime._tensor_sha256(embedding.weight),
        "gradient_audit": gradients,
    }


def _strict_posthoc(
    *,
    config: VocabularyFrontierConfig,
    behavior: Mapping[str, Any],
    phase: str,
    attempt_root: Path,
) -> dict[str, Any]:
    candidate_path = attempt_root / "candidate-whole-output.json"
    strict_path = attempt_root / "strict-intelligence.json"
    manual_path = attempt_root / "strict-manual-review-packet.json"
    shared_runner._atomic_write_new_json(
        candidate_path,
        {"phase": phase, "outputs": behavior["event"]["outputs"]},
    )
    candidate_sha = _file_sha256(candidate_path)
    report = strict_runner.evaluate_posthoc(
        strict_runner.EvaluationConfig(
            parent_envelope=config.parent_output_envelope,
            expected_parent_envelope_sha256=(
                config.expected_parent_output_envelope_sha256
            ),
            expected_parent_cache_key=config.expected_parent_cache_key,
            expected_parent_state_fingerprint=(
                config.expected_parent_state_fingerprint
            ),
            tokenizer_directory=config.tokenizer_directory,
            expected_tokenizer_fingerprint=(
                config.expected_tokenizer_fingerprint
            ),
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_sha,
            candidate_phase=phase,
            report_path=strict_path,
            manual_review_packet_path=manual_path,
            expected_case_count=config.expected_whole_output_case_count,
        )
    )
    return {
        "report": report,
        "report_path": str(strict_path),
        "report_file_sha256": _file_sha256(strict_path),
        "manual_review_packet_path": str(manual_path),
        "manual_review_packet_file_sha256": _file_sha256(manual_path),
        "candidate_event_path": str(candidate_path),
        "candidate_event_file_sha256": candidate_sha,
        "strict_promoted": bool(report["promoted"]),
    }


def _authenticated_rollback(
    installation: InstalledLfm2VocabularyInterface,
    model: torch.nn.Module,
    *,
    source_state_sha256: str,
    source_counts: Mapping[str, int],
) -> dict[str, Any]:
    rolled_back = installation.rollback()
    state = fold_runtime.model_state_sha256(model)
    counts = exact_model_parameter_counts(model)
    gradients = shared_runner.gradient_audit(model)
    exact = bool(
        rolled_back
        and state == source_state_sha256
        and counts == dict(source_counts)
        and gradients["zero_gradient"]
    )
    if not exact:
        raise Lfm2VocabularyFrontierError(
            "Candidate rollback did not restore the exact source state."
        )
    return {
        "performed": True,
        "exact": True,
        "source_state_sha256_restored": state,
        "source_parameter_counts_restored": counts,
        "gradient_audit": gradients,
    }


def _evaluate_attempt(
    *,
    config: VocabularyFrontierConfig,
    source_label: str,
    allocation: str,
    model: torch.nn.Module,
    source_cache: Lfm2CircuitTraceCache,
    source_rows: Any,
    source_authentication: Mapping[str, Any],
    analysis: TiedVocabularySVDAnalysis,
    target_fraction: float,
    parent_cache: Lfm2CircuitTraceCache,
    parent_rows: Any,
    parent_capture_config: Any,
    source_capture_config: Any,
    behavior_evaluator: Any,
) -> dict[str, Any]:
    target = solve_tied_vocabulary_target(
        original_parent_unique_parameters=(
            config.expected_original_parent_unique_parameters
        ),
        source_unique_parameters=source_authentication["parameter_counts"]["unique"],
        vocab_size=config.expected_vocab_size,
        hidden_size=config.expected_hidden_size,
        total_compression_fraction=target_fraction,
    )
    phase = (
        f"raw_tied_vocab_{source_label}_rank_{target.rank}_"
        f"total_{target_fraction:.3f}"
    )
    attempt_root = config.work_root / source_label / f"rank-{target.rank}"
    attempt_root.mkdir(parents=True, exist_ok=False)
    installation: InstalledLfm2VocabularyInterface | None = None
    attempt: dict[str, Any] = {
        "schema": ATTEMPT_SCHEMA,
        "source_label": source_label,
        "allocation": allocation,
        "phase": phase,
        "target": target.as_dict(),
        "status": "factorizing",
        "checkpoint_published": False,
    }
    try:
        pair = analysis.factorize(target.rank)
        installation = install_lfm2_tied_vocabulary_pair(model, pair)
        candidate_counts = exact_model_parameter_counts(model)
        if candidate_counts["unique"] != target.physical_target_unique_parameters:
            raise Lfm2VocabularyFrontierError(
                "Physical candidate count differs from the solved exact target."
            )
        parent_capture = bank_runner.capture_live_candidate_trajectory(
            model,
            token_cache=parent_cache,
            capture_config=parent_capture_config,
        )
        parent_trajectory = bank_runner.evaluate_live_trajectory(
            reference_name="original_parent",
            reference_cache=parent_cache,
            reference_rows=parent_rows,
            candidate=parent_capture,
            minimum_residual_cosine=config.minimum_parent_residual_cosine,
            minimum_score=config.minimum_parent_trajectory_score,
        )
        if source_cache.cache_key == parent_cache.cache_key:
            source_capture = parent_capture
            source_capture_reused = True
        else:
            source_capture = bank_runner.capture_live_candidate_trajectory(
                model,
                token_cache=source_cache,
                capture_config=source_capture_config,
            )
            source_capture_reused = False
        incumbent_trajectory = bank_runner.evaluate_live_trajectory(
            reference_name=source_label,
            reference_cache=source_cache,
            reference_rows=source_rows,
            candidate=source_capture,
            minimum_residual_cosine=config.minimum_incumbent_residual_cosine,
            minimum_score=config.minimum_incumbent_trajectory_score,
        )
        behavior = shared_runner._behavior_event(
            behavior_evaluator,
            model,
            phase=phase,
        )
        strict = _strict_posthoc(
            config=config,
            behavior=behavior,
            phase=phase,
            attempt_root=attempt_root,
        )
        behavioral_pass = bool(
            parent_trajectory["decision"]["accepted"]
            and incumbent_trajectory["decision"]["accepted"]
            and behavior["decision"]["accepted"]
        )
        strict_pass = bool(strict["strict_promoted"])
        loader_pass = False
        accepted = bool(behavioral_pass and strict_pass and loader_pass)
        attempt.update(
            {
                "status": "evaluated",
                "accepted": accepted,
                "candidate": {
                    "parameter_counts": candidate_counts,
                    "physical_unique_parameters": candidate_counts["unique"],
                    "parameters_removed_from_source": (
                        source_authentication["parameter_counts"]["unique"]
                        - candidate_counts["unique"]
                    ),
                    "parameters_removed_from_original_parent": (
                        config.expected_original_parent_unique_parameters
                        - candidate_counts["unique"]
                    ),
                    "total_compression_fraction": (
                        (
                            config.expected_original_parent_unique_parameters
                            - candidate_counts["unique"]
                        )
                        / config.expected_original_parent_unique_parameters
                    ),
                    "rank": pair.embedding.rank,
                    "retained_spectral_energy_fraction": (
                        pair.plan.retained_energy_fraction
                    ),
                    "embedding_and_head_factor_objects_shared": bool(
                        pair.embedding.vocab_factor
                        is pair.lm_head.vocab_factor
                        and pair.embedding.hidden_factor
                        is pair.lm_head.hidden_factor
                    ),
                    "dense_vocab_matrix_retained": False,
                    "installation": dict(installation.installation_audit),
                },
                "behavioral_evaluation": {
                    "original_parent_trajectory": parent_trajectory,
                    "incumbent_source_trajectory": incumbent_trajectory,
                    "cached_parent_whole_output": behavior,
                    "strict_parent_relative_intelligence": strict,
                    "trajectory_capture": {
                        "parent_capture_sha256": parent_capture.capture_sha256,
                        "source_capture_sha256": source_capture.capture_sha256,
                        "source_capture_reused_parent_capture": (
                            source_capture_reused
                        ),
                        "parent_candidate_forward_calls": (
                            parent_capture.forward_calls
                        ),
                        "source_candidate_forward_calls": (
                            0
                            if source_capture_reused
                            else source_capture.forward_calls
                        ),
                    },
                    "all_28_whole_output_cases_are_weight_fit_heldout": True,
                    "weight_fit_prompt_count": 0,
                    "new_parent_forward_calls": 0,
                    "new_parent_response_generation_calls": 0,
                    "behavioral_pass_before_manual_strict_gate": behavioral_pass,
                    "strict_pass": strict_pass,
                },
                "publication_gate": {
                    "strict_intelligence_passed": strict_pass,
                    "authenticated_factorized_loader_passed": loader_pass,
                    "checkpoint_publication_authorized": accepted,
                    "checkpoint_published": False,
                    "reason": (
                        "Strict intelligence did not promote the candidate."
                        if not strict_pass
                        else "Strict passed, but authenticated reload remains required."
                    ),
                },
            }
        )
        attempt["rollback"] = _authenticated_rollback(
            installation,
            model,
            source_state_sha256=source_authentication["state_sha256"],
            source_counts=source_authentication["parameter_counts"],
        )
        installation = None
        return attempt
    except BaseException as exc:
        rollback = None
        rollback_error = None
        if installation is not None and installation.active:
            try:
                rollback = _authenticated_rollback(
                    installation,
                    model,
                    source_state_sha256=source_authentication["state_sha256"],
                    source_counts=source_authentication["parameter_counts"],
                )
            except BaseException as rollback_exc:
                rollback_error = (
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        attempt.update(
            {
                "status": "error",
                "accepted": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
                "rollback_error": rollback_error,
            }
        )
        if rollback_error is not None:
            raise Lfm2VocabularyFrontierError(
                f"Attempt and exact rollback failed: {rollback_error}"
            ) from exc
        return attempt


def _behavior_evaluator(
    config: VocabularyFrontierConfig,
    *,
    cache: Lfm2CircuitTraceCache,
    tokenizer: Any,
    tokenizer_fingerprint: str,
) -> Any:
    return shared_runner._behavior_evaluator(
        config,
        cache=cache,
        tokenizer=tokenizer,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )


def _sealed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("run_sha256", None)
    return {**body, "run_sha256": _json_sha256(body)}


def run_experiment(config: VocabularyFrontierConfig) -> dict[str, Any]:
    """Execute all raw rank-prefix allocations with one shared decomposition."""

    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "initializing",
        "config": config.as_dict(),
        "training": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "optimizer_constructed": False,
            "backward_calls": 0,
            "gradients_used": False,
        },
        "network_access": False,
        "hub_loader_used": False,
        "new_parent_trace_generation_calls": 0,
        "new_parent_response_generation_calls": 0,
        "checkpoint_published": False,
        "checkpoint_absence": _checkpoint_absence(config),
    }
    try:
        if _file_sha256(config.parent_output_envelope) != (
            config.expected_parent_output_envelope_sha256
        ):
            raise Lfm2VocabularyFrontierError(
                "Parent output envelope file hash differs from expectation."
            )
        parent_cache = Lfm2CircuitTraceCache.open(
            config.parent_trajectory_cache,
            verify=True,
        )
        whole_output_cache = Lfm2CircuitTraceCache.open(
            config.whole_output_parent_cache,
            verify=True,
        )
        strong_cache = Lfm2CircuitTraceCache.open(
            config.strong_source_cache,
            verify=True,
        )
        if (
            whole_output_cache.cache_key != config.expected_parent_cache_key
            or whole_output_cache.parent_fingerprint
            != config.expected_parent_state_fingerprint
            or parent_cache.parent_fingerprint
            != whole_output_cache.parent_fingerprint
        ):
            raise Lfm2VocabularyFrontierError(
                "Parent trajectory, whole-output cache, and envelope bindings differ."
            )
        tokenizer_fingerprint = shared_runner._tokenizer_fingerprint(
            config.tokenizer_directory
        )
        if tokenizer_fingerprint != config.expected_tokenizer_fingerprint:
            raise Lfm2VocabularyFrontierError(
                "Local tokenizer fingerprint differs from expectation."
            )
        tokenizer = fold_runtime.load_local_tokenizer(config.tokenizer_directory)
        parent_capture_config = shared_runner.capture_config_from_parent_cache(
            parent_cache
        )
        strong_capture_config = shared_runner.capture_config_from_parent_cache(
            strong_cache
        )
        parent_rows = fold_runtime.extract_trace_rows(parent_cache)
        strong_rows = fold_runtime.extract_trace_rows(strong_cache)
        dtype = fold_runtime._torch_dtype(config.dtype)
        strong_model = fold_runtime.load_local_lfm2_model(
            config.strong_source_model,
            device=config.device,
            dtype=dtype,
        )
        strong_model.eval()
        strong_auth = _source_authentication(
            strong_model,
            source_cache=strong_cache,
            expected_unique_parameters=(
                config.expected_strong_source_unique_parameters
            ),
            expected_state_sha256=config.expected_strong_source_state_sha256,
            expected_cache_key=config.expected_strong_source_cache_key,
        )
        analysis = analyze_tied_vocabulary_modules(
            strong_model.get_input_embeddings(),
            strong_model.get_output_embeddings(),
        )
        singular_values = analysis.singular_values
        spectral_evidence = {
            "method": "deterministic_cpu_float64_raw_weight_svd",
            "decomposition_invocations": 1,
            "reused_for_every_rank_and_source_allocation": True,
            "rank": analysis.max_rank,
            "vocab_size": analysis.vocab_size,
            "hidden_size": analysis.hidden_size,
            "singular_values_sha256": _json_sha256(list(singular_values)),
            "largest_singular_value": singular_values[0],
            "smallest_singular_value": singular_values[-1],
            "source_vocab_weight_sha256": strong_auth["vocab_weight_sha256"],
            "activation_or_prompt_fit_used": False,
            "whole_output_heldout_case_count": (
                config.expected_whole_output_case_count
            ),
        }
        behavior_evaluator = _behavior_evaluator(
            config,
            cache=whole_output_cache,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        attempts: list[dict[str, Any]] = []
        for fraction in config.strong_compression_fractions:
            attempts.append(
                _evaluate_attempt(
                    config=config,
                    source_label="strong_10pct_child",
                    allocation="distributed_width_then_tied_vocab",
                    model=strong_model,
                    source_cache=strong_cache,
                    source_rows=strong_rows,
                    source_authentication=strong_auth,
                    analysis=analysis,
                    target_fraction=fraction,
                    parent_cache=parent_cache,
                    parent_rows=parent_rows,
                    parent_capture_config=parent_capture_config,
                    source_capture_config=strong_capture_config,
                    behavior_evaluator=behavior_evaluator,
                )
            )
        parent_model = fold_runtime.load_local_lfm2_model(
            config.original_parent_model,
            device=config.device,
            dtype=dtype,
        )
        parent_model.eval()
        parent_state = fold_runtime.model_state_sha256(parent_model)
        parent_auth = _source_authentication(
            parent_model,
            source_cache=parent_cache,
            expected_unique_parameters=(
                config.expected_original_parent_unique_parameters
            ),
            expected_state_sha256=parent_state,
            expected_cache_key=parent_cache.cache_key,
        )
        if (
            parent_auth["identity_sha256"]
            != config.expected_parent_state_fingerprint
            or parent_auth["vocab_weight_sha256"]
            != strong_auth["vocab_weight_sha256"]
            or not torch.equal(
                parent_model.get_input_embeddings().weight.detach(),
                strong_model.get_input_embeddings().weight.detach(),
            )
        ):
            raise Lfm2VocabularyFrontierError(
                "Original parent and strong child do not share the exact tied matrix."
            )
        for fraction in config.direct_parent_compression_fractions:
            attempts.append(
                _evaluate_attempt(
                    config=config,
                    source_label="original_parent",
                    allocation="direct_tied_vocab_only",
                    model=parent_model,
                    source_cache=parent_cache,
                    source_rows=parent_rows,
                    source_authentication=parent_auth,
                    analysis=analysis,
                    target_fraction=fraction,
                    parent_cache=parent_cache,
                    parent_rows=parent_rows,
                    parent_capture_config=parent_capture_config,
                    source_capture_config=parent_capture_config,
                    behavior_evaluator=behavior_evaluator,
                )
            )
        report.update(
            {
                "status": "completed",
                "source_authentication": {
                    "strong_10pct_child": strong_auth,
                    "original_parent": parent_auth,
                    "exact_same_tied_vocab_matrix": True,
                },
                "cache_authentication": {
                    "parent_trajectory_cache_key": parent_cache.cache_key,
                    "whole_output_parent_cache_key": whole_output_cache.cache_key,
                    "strong_source_cache_key": strong_cache.cache_key,
                    "parent_fingerprint": parent_cache.parent_fingerprint,
                    "tokenizer_fingerprint": tokenizer_fingerprint,
                    "parent_output_envelope_file_sha256": (
                        config.expected_parent_output_envelope_sha256
                    ),
                    "new_parent_trace_generation_calls": 0,
                    "new_parent_response_generation_calls": 0,
                },
                "spectral_evidence": spectral_evidence,
                "attempts": attempts,
                "accepted_attempt_count": sum(
                    bool(attempt.get("accepted")) for attempt in attempts
                ),
                "error_attempt_count": sum(
                    attempt.get("status") == "error" for attempt in attempts
                ),
                "checkpoint_published": False,
                "checkpoint_absence": _checkpoint_absence(config),
                "conclusion": {
                    "raw_svd_control_completed": True,
                    "allocation_comparison_completed": True,
                    "exact_output_matching_required": False,
                    "promotion_requires_coherent_parent_relative_intelligence": True,
                },
            }
        )
    except BaseException as exc:
        report.update(
            {
                "status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "checkpoint_published": False,
                "checkpoint_absence": _checkpoint_absence(config),
            }
        )
    sealed = _sealed_report(report)
    shared_runner._atomic_write_new_json(config.report_path, sealed)
    return sealed


def _parse_fractions(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-SVD raw tied-vocabulary LFM2 frontiers at "
            "15/20/25/35% total compaction."
        )
    )
    parser.add_argument("--strong-source-model", type=Path, required=True)
    parser.add_argument("--strong-source-cache", type=Path, required=True)
    parser.add_argument("--original-parent-model", type=Path, required=True)
    parser.add_argument("--parent-trajectory-cache", type=Path, required=True)
    parser.add_argument("--whole-output-parent-cache", type=Path, required=True)
    parser.add_argument("--parent-output-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-output-envelope-sha256",
        required=True,
    )
    parser.add_argument("--expected-parent-cache-key", required=True)
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--expected-strong-source-state-sha256", required=True)
    parser.add_argument("--expected-strong-source-cache-key", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--strong-compression-fractions",
        type=_parse_fractions,
        default=STRONG_COMPRESSION_FRACTIONS,
    )
    parser.add_argument(
        "--direct-parent-compression-fractions",
        type=_parse_fractions,
        default=DIRECT_PARENT_COMPRESSION_FRACTIONS,
    )
    parser.add_argument(
        "--expected-original-parent-unique-parameters",
        type=int,
        default=EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--expected-strong-source-unique-parameters",
        type=int,
        default=EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS,
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> VocabularyFrontierConfig:
    return VocabularyFrontierConfig(**vars(args))


def main(argv: Sequence[str] | None = None) -> int:
    config = _config_from_args(build_parser().parse_args(argv))
    report = run_experiment(config)
    print(
        json.dumps(
            {
                "status": report["status"],
                "run_sha256": report["run_sha256"],
                "report_path": str(config.report_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
