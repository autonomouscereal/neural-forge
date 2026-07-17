#!/usr/bin/env python3
"""Production local-only LFM2 runner for the automatic folding catalog.

This entrypoint configures the model-agnostic target synthesizer and folding
stack once, then submits one neutral exact-budget request.  It never downloads
a model, constructs an optimizer, calls backward, or regenerates parent traces
after the immutable parent cache has been built.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import run_lfm2_residual_attention_fold as residual
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_residual_attention_fold as residual

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_registry import (
    DurableExperimentLedger,
    ExperimentLedger,
)
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
)
from neural_forge.dynamic_lfm2_adapter import (
    HYBRID_MODEL_METHOD,
    PRODUCER_FIRST_MODEL_METHOD,
    Lfm2EvidenceBuildRequest,
    Lfm2ModelEvidenceBuildRequest,
    Lfm2PreparedEvidence,
    Lfm2PreparedHybridProducerEvidence,
    Lfm2PreparedModelEvidence,
    Lfm2PreparedProducerFirstEvidence,
    Lfm2ProducerFirstEvidenceBuildRequest,
    Lfm2TargetSynthesisAdapter,
    decode_lfm2_layer_types,
    decode_lfm2_source_spans,
    encode_lfm2_layer_types,
    encode_lfm2_source_spans,
)
from neural_forge.dynamic_lfm2_behavior import (
    CombinedLfm2FinalEvaluator,
    Lfm2WholeOutputBehaviorEvaluator,
    load_lfm2_cached_whole_output_baseline,
)
from neural_forge.dynamic_lfm2_system import (
    Lfm2DynamicFoldStackAdapter,
    Lfm2DynamicStackResources,
)
from neural_forge.dynamic_lfm2_targets import Lfm2Topology
from neural_forge.dynamic_system import (
    AutomaticDynamicFoldingSystem,
    DynamicFoldStackAdapterRegistry,
)
from neural_forge.dynamic_target_synthesis import (
    DynamicTargetAdapterRegistry,
    DynamicTargetRequest,
)
from neural_forge.hf_lfm2_dense_budget_model_fold import (
    DENSE_BUDGET_MODEL_METHOD,
    Lfm2DenseBudgetFoldConfig,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    _layer_topology as _lfm2_layer_topology,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
    open_lfm2_circuit_trace_cache,
)
from neural_forge.hf_lfm2_closed_loop_fold import (
    Lfm2AttentionFoldEvidence,
    Lfm2ClosedLoopFoldConfig,
    Lfm2ClosedLoopFoldEvidence,
    Lfm2MLPFoldEvidence,
)
from neural_forge.hf_lfm2_producer_first_model_fold import (
    Lfm2ProducerFirstCapture,
    Lfm2ProducerFirstEvidence,
    Lfm2ProducerFirstFoldConfig,
)


RECEIPT_SCHEMA = "neural-forge-production-lfm2-catalog-receipt-v1"
DIAGNOSTICS_SCHEMA = "neural-forge-production-lfm2-gate-diagnostics-v1"
SEALED_PARENT_CACHE_SOURCE_MODE = "sealed_existing"
CALIBRATION_PARENT_CACHE_SOURCE_MODE = "calibration_build_or_reuse"
SEALED_CACHE_SPLIT_SCHEMA = (
    "neural-forge-production-lfm2-sealed-cache-evidence-splits-v1"
)

DEFAULT_TARGET_QUERY_HEADS = 16
DEFAULT_TARGET_KEY_VALUE_HEADS = 8
DEFAULT_TARGET_INTERMEDIATE_SIZE = 2552
DEFAULT_TARGET_UNIQUE_PARAMETERS = 229_349_120
DEFAULT_CAPTURE_DECODE_STEPS = 8
DEFAULT_CAPTURE_BYTES = 128 * 1024 * 1024
DEFAULT_CAPTURE_LOGIT_RTOL = 0.2
DEFAULT_CAPTURE_LOGIT_ATOL = 1.0

_PREFIX_MISMATCH = re.compile(
    r"(?P<phase>cached prefill|decode\.\d{3}) cached logits differ from "
    r"use_cache=False full-prefix logits \(maximum error "
    r"(?P<error>[0-9.eE+-]+)\)"
)
_TOKENIZER_ARTIFACTS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "added_tokens.json",
)


class ProductionLfm2CatalogError(RuntimeError):
    """The local production catalog could not prove or publish its child."""


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProductionLfm2CatalogError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionLfm2CatalogError(f"{name} must be a non-negative integer.")
    return value


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionLfm2CatalogError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _finite_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProductionLfm2CatalogError(f"{name} must be a finite float.") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ProductionLfm2CatalogError(f"{name} must be a finite float.")
    return result


def _file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json(dict(value)) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_ledger_atomic(path: Path, ledger: ExperimentLedger) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ledger.write_jsonl(temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_ledger(path: Path) -> DurableExperimentLedger:
    return DurableExperimentLedger(path)


def _directory_keys(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.is_symlink()
    }


def _expected_capture_forwards(
    batches: Sequence[Mapping[str, torch.Tensor]],
) -> int:
    total = 0
    for batch in batches:
        forced = batch["forced_decode_ids"]
        total += 2 + 2 * int(forced.shape[1])
    return total


def fingerprint_local_tokenizer(
    tokenizer_directory: str | os.PathLike[str],
) -> str:
    """Validate and hash exactly the local tokenizer artifacts we package."""

    directory = Path(tokenizer_directory).expanduser().resolve(strict=True)
    if not directory.is_dir() or directory.is_symlink():
        raise ProductionLfm2CatalogError(
            f"Tokenizer directory is missing or unsafe: {directory}."
        )
    residual.load_local_tokenizer(directory)
    files = []
    for name in _TOKENIZER_ARTIFACTS:
        path = directory / name
        if path.is_file() and not path.is_symlink():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not files or files[0]["path"] != "tokenizer.json":
        raise ProductionLfm2CatalogError(
            "The local tokenizer package lacks tokenizer.json."
        )
    return canonical_sha256(
        {
            "schema": "neural-forge-local-tokenizer-artifacts-v1",
            "files": files,
        }
    )


@dataclass(frozen=True, slots=True)
class ProductionLfm2CatalogConfig:
    parent_model: Path
    calibration: Path | None
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    cache_root: Path
    snapshot_root: Path
    publish_destination: Path
    ledger_path: Path
    receipt_path: Path
    tokenizer_directory: Path | None = None
    diagnostics_path: Path | None = None
    sealed_parent_cache_directory: Path | None = None
    parent_name: str = "parent"
    child_name: str = "lfm2-ffn-2552"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    target_query_heads: int = DEFAULT_TARGET_QUERY_HEADS
    target_key_value_heads: int = DEFAULT_TARGET_KEY_VALUE_HEADS
    target_intermediate_size: int = DEFAULT_TARGET_INTERMEDIATE_SIZE
    exact_target_unique_parameters: int = DEFAULT_TARGET_UNIQUE_PARAMETERS
    compression_ratio: float | None = None
    dense_maximum_candidates: int = 64
    dense_residual_basis_mode: str = "circuit_balanced"
    dense_run_closed_loop: bool = False
    dense_vocabulary_margin_observability: bool = False
    dense_vocabulary_margin_top_k: int = 8
    dense_vocabulary_margin_floor_fraction: float = 0.05
    dense_vocabulary_margin_candidate_gate_enabled: bool = False
    dense_vocabulary_margin_candidate_replay_mode: str = (
        CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
    )
    dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction: (
        float
    ) = 0.9
    dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction: (
        float
    ) = 0.9
    dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction: (
        float
    ) = 0.9
    dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction: (
        float
    ) = 0.9
    hybrid_model_fold: bool = False
    producer_first_model_fold: bool = False
    target_layer_types: tuple[str, ...] | None = None
    ordered_source_spans: tuple[tuple[int, ...], ...] | None = None
    blank_seed: int = 17
    initial_mlp_step: int = 1
    maximum_mlp_step: int = 64
    sample_last_tokens: int = 8
    max_decode_steps: int = DEFAULT_CAPTURE_DECODE_STEPS
    max_capture_bytes: int = DEFAULT_CAPTURE_BYTES
    capture_logit_rtol: float = DEFAULT_CAPTURE_LOGIT_RTOL
    capture_logit_atol: float = DEFAULT_CAPTURE_LOGIT_ATOL
    require_capture_top1_agreement: bool = False
    child_decode_trace_mode: str = FULL_PREFIX_CANONICAL_TRACE_MODE
    max_selection_rows: int = 256
    max_fit_rows: int = 512
    max_validation_rows: int = 128
    validation_fraction: float = 0.20
    endpoint_ridge: float = 1e-6
    max_endpoint_relative_rmse: float = 1e6
    min_endpoint_cosine_similarity: float = -1.0
    endpoint_max_prior_delta_ratio: float = 0.0
    min_teacher_top1: float = 0.05
    min_free_run_match: float = 0.05
    min_residual_cosine: float = 0.0
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 0
    whole_output_maximum_new_fragment_responses: int = 0
    whole_output_minimum_parent_nonempty_retention: float = 1.0
    whole_output_minimum_task_score_retention: float = 1.0
    whole_output_maximum_parent_task_score_regressions: int = 0
    whole_output_maximum_new_generation_limit_saturations: int = 0
    whole_output_maximum_repeated_ngram_share_regressions: int = 0
    whole_output_maximum_repeated_bigram_share_delta: float = 0.05
    snapshot_chunk_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "parent_model",
            "cached_parent_outputs",
            "cache_root",
            "snapshot_root",
            "publish_destination",
            "ledger_path",
            "receipt_path",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "calibration",
            "tokenizer_directory",
            "diagnostics_path",
            "sealed_parent_cache_directory",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        if (
            self.calibration is None
            and self.sealed_parent_cache_directory is None
        ):
            raise ProductionLfm2CatalogError(
                "calibration is required unless sealed_parent_cache_directory "
                "selects an existing immutable parent cache."
            )
        for name in (
            "parent_name",
            "child_name",
            "device",
            "dtype",
            "child_decode_trace_mode",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ProductionLfm2CatalogError(f"{name} must be a nonempty string.")
            object.__setattr__(self, name, value.strip())
        if self.child_decode_trace_mode not in {
            CACHED_VERIFIED_TRACE_MODE,
            FULL_PREFIX_CANONICAL_TRACE_MODE,
        }:
            raise ProductionLfm2CatalogError(
                "child_decode_trace_mode must be 'cached_verified' or "
                "'full_prefix_canonical'."
            )
        for name in (
            "target_query_heads",
            "target_key_value_heads",
            "target_intermediate_size",
            "exact_target_unique_parameters",
            "dense_maximum_candidates",
            "dense_vocabulary_margin_top_k",
            "initial_mlp_step",
            "maximum_mlp_step",
            "sample_last_tokens",
            "max_decode_steps",
            "max_capture_bytes",
            "max_selection_rows",
            "max_fit_rows",
            "max_validation_rows",
            "whole_output_max_new_tokens",
            "whole_output_maximum_prompt_tokens",
            "whole_output_fragment_minimum_characters",
            "snapshot_chunk_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        for name in (
            "whole_output_maximum_new_empty_responses",
            "whole_output_maximum_new_fragment_responses",
            "whole_output_maximum_parent_task_score_regressions",
            "whole_output_maximum_new_generation_limit_saturations",
            "whole_output_maximum_repeated_ngram_share_regressions",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "blank_seed",
            _nonnegative_int(self.blank_seed, "blank_seed"),
        )
        if self.compression_ratio is not None:
            compression_ratio = _finite_float(
                self.compression_ratio,
                "compression_ratio",
            )
            if compression_ratio <= 1.0:
                raise ProductionLfm2CatalogError(
                    "compression_ratio must be greater than one."
                )
            object.__setattr__(
                self,
                "compression_ratio",
                compression_ratio,
            )
        basis_mode = str(self.dense_residual_basis_mode).strip()
        if basis_mode not in {"weight_complete", "circuit_balanced"}:
            raise ProductionLfm2CatalogError(
                "dense_residual_basis_mode must be weight_complete or "
                "circuit_balanced."
            )
        object.__setattr__(
            self,
            "dense_residual_basis_mode",
            basis_mode,
        )
        margin_floor = _finite_float(
            self.dense_vocabulary_margin_floor_fraction,
            "dense_vocabulary_margin_floor_fraction",
            minimum=0.0,
        )
        if margin_floor <= 0.0 or margin_floor > 1.0:
            raise ProductionLfm2CatalogError(
                "dense_vocabulary_margin_floor_fraction must be within "
                "(0, 1]."
            )
        object.__setattr__(
            self,
            "dense_vocabulary_margin_floor_fraction",
            margin_floor,
        )
        candidate_replay_mode = str(
            self.dense_vocabulary_margin_candidate_replay_mode
        ).strip()
        if candidate_replay_mode not in {
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        }:
            raise ProductionLfm2CatalogError(
                "dense_vocabulary_margin_candidate_replay_mode must be "
                "'prefix_by_prefix_canonical' or 'single_pass_causal'."
            )
        object.__setattr__(
            self,
            "dense_vocabulary_margin_candidate_replay_mode",
            candidate_replay_mode,
        )
        for name in (
            (
                "dense_vocabulary_margin_candidate_minimum_"
                "sign_agreement_fraction"
            ),
            (
                "dense_vocabulary_margin_candidate_minimum_"
                "top_competitor_nonnegative_fraction"
            ),
            (
                "dense_vocabulary_margin_candidate_minimum_"
                "all_top_k_nonnegative_fraction"
            ),
            (
                "dense_vocabulary_margin_candidate_minimum_"
                "exact_top1_agreement_fraction"
            ),
        ):
            value = _finite_float(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ProductionLfm2CatalogError(
                    f"{name} must be at most one."
                )
            object.__setattr__(self, name, value)
        if self.target_query_heads % self.target_key_value_heads:
            raise ProductionLfm2CatalogError(
                "Target query heads must be divisible by key/value heads."
            )
        if self.maximum_mlp_step < self.initial_mlp_step:
            raise ProductionLfm2CatalogError(
                "maximum_mlp_step cannot be smaller than initial_mlp_step."
            )
        for name in (
            "capture_logit_rtol",
            "capture_logit_atol",
            "endpoint_ridge",
            "endpoint_max_prior_delta_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name, minimum=0.0),
            )
        validation_fraction = _finite_float(
            self.validation_fraction,
            "validation_fraction",
            minimum=0.0,
        )
        if validation_fraction >= 1.0:
            raise ProductionLfm2CatalogError(
                "validation_fraction must be smaller than one."
            )
        object.__setattr__(
            self,
            "validation_fraction",
            validation_fraction,
        )
        for name in ("min_teacher_top1", "min_free_run_match"):
            value = _finite_float(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ProductionLfm2CatalogError(f"{name} must be at most one.")
            object.__setattr__(self, name, value)
        residual_cosine = _finite_float(
            self.min_residual_cosine,
            "min_residual_cosine",
        )
        endpoint_cosine = _finite_float(
            self.min_endpoint_cosine_similarity,
            "min_endpoint_cosine_similarity",
        )
        if not -1.0 <= residual_cosine <= 1.0:
            raise ProductionLfm2CatalogError("min_residual_cosine must be in [-1, 1].")
        if not -1.0 <= endpoint_cosine <= 1.0:
            raise ProductionLfm2CatalogError(
                "min_endpoint_cosine_similarity must be in [-1, 1]."
            )
        object.__setattr__(self, "min_residual_cosine", residual_cosine)
        object.__setattr__(
            self,
            "min_endpoint_cosine_similarity",
            endpoint_cosine,
        )
        object.__setattr__(
            self,
            "max_endpoint_relative_rmse",
            _finite_float(
                self.max_endpoint_relative_rmse,
                "max_endpoint_relative_rmse",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "cached_parent_outputs_sha256",
            _digest(
                self.cached_parent_outputs_sha256,
                "cached_parent_outputs_sha256",
            ),
        )
        repeated_delta = _finite_float(
            self.whole_output_maximum_repeated_bigram_share_delta,
            "whole_output_maximum_repeated_bigram_share_delta",
            minimum=0.0,
        )
        if repeated_delta > 1.0:
            raise ProductionLfm2CatalogError(
                "whole_output_maximum_repeated_bigram_share_delta must be at most one."
            )
        object.__setattr__(
            self,
            "whole_output_maximum_repeated_bigram_share_delta",
            repeated_delta,
        )
        for name in (
            "whole_output_minimum_parent_nonempty_retention",
            "whole_output_minimum_task_score_retention",
        ):
            value = _finite_float(getattr(self, name), name, minimum=0.0)
            if value > 1.0:
                raise ProductionLfm2CatalogError(f"{name} must be at most one.")
            object.__setattr__(self, name, value)
        if not isinstance(self.require_capture_top1_agreement, bool):
            raise ProductionLfm2CatalogError(
                "require_capture_top1_agreement must be boolean."
            )
        if not isinstance(self.hybrid_model_fold, bool):
            raise ProductionLfm2CatalogError("hybrid_model_fold must be boolean.")
        if not isinstance(self.producer_first_model_fold, bool):
            raise ProductionLfm2CatalogError(
                "producer_first_model_fold must be boolean."
            )
        for name in (
            "dense_run_closed_loop",
            "dense_vocabulary_margin_observability",
            "dense_vocabulary_margin_candidate_gate_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProductionLfm2CatalogError(
                    f"{name} must be boolean."
                )
        if self.hybrid_model_fold and self.producer_first_model_fold:
            raise ProductionLfm2CatalogError(
                "Hybrid depth folding and full-depth producer-first folding "
                "are mutually exclusive."
            )
        if (
            self.compression_ratio is not None
            and (self.hybrid_model_fold or self.producer_first_model_fold)
        ):
            raise ProductionLfm2CatalogError(
                "Automatic dense compression-ratio folding is mutually "
                "exclusive with explicit hybrid and producer-first modes."
            )
        if (
            self.dense_vocabulary_margin_observability
            and self.compression_ratio is None
        ):
            raise ProductionLfm2CatalogError(
                "dense_vocabulary_margin_observability requires an automatic "
                "compression_ratio request."
            )
        if (
            self.dense_vocabulary_margin_observability
            and self.dense_residual_basis_mode != "circuit_balanced"
        ):
            raise ProductionLfm2CatalogError(
                "dense vocabulary-margin observability requires the "
                "circuit_balanced residual basis."
            )
        if (
            self.dense_vocabulary_margin_candidate_gate_enabled
            and not self.dense_vocabulary_margin_observability
        ):
            raise ProductionLfm2CatalogError(
                "dense vocabulary-margin candidate gating requires "
                "dense_vocabulary_margin_observability."
            )
        if self.hybrid_model_fold:
            if self.target_layer_types is None or self.ordered_source_spans is None:
                raise ProductionLfm2CatalogError(
                    "Hybrid model folding requires target_layer_types and "
                    "ordered_source_spans."
                )
            layer_types = decode_lfm2_layer_types(
                encode_lfm2_layer_types(self.target_layer_types)
            )
            spans = decode_lfm2_source_spans(
                encode_lfm2_source_spans(self.ordered_source_spans)
            )
            if len(layer_types) != len(spans):
                raise ProductionLfm2CatalogError(
                    "Hybrid target layer count and source-span count differ."
                )
            object.__setattr__(self, "target_layer_types", layer_types)
            object.__setattr__(self, "ordered_source_spans", spans)
        elif self.target_layer_types is not None or self.ordered_source_spans is not None:
            raise ProductionLfm2CatalogError(
                "Hybrid layer topology flags require hybrid_model_fold=True."
            )

    @property
    def tokenizer_source(self) -> Path:
        return (
            self.parent_model
            if self.tokenizer_directory is None
            else self.tokenizer_directory
        )

    @property
    def target_topology(self) -> Lfm2Topology:
        return Lfm2Topology(
            self.target_query_heads,
            self.target_key_value_heads,
            self.target_intermediate_size,
        )

    @property
    def target_depth(self) -> int | None:
        return (
            None
            if self.target_layer_types is None
            else len(self.target_layer_types)
        )

    @property
    def parent_cache_root(self) -> Path:
        return self.cache_root / "parent"

    @property
    def parent_cache_source_mode(self) -> str:
        return (
            SEALED_PARENT_CACHE_SOURCE_MODE
            if self.sealed_parent_cache_directory is not None
            else CALIBRATION_PARENT_CACHE_SOURCE_MODE
        )

    @property
    def evidence_cache_root(self) -> Path:
        return self.cache_root / "child-evidence"

    @property
    def final_cache_root(self) -> Path:
        return self.cache_root / "child-final"

    @property
    def reload_cache_root(self) -> Path:
        return self.cache_root / "child-reload"

    def identity(self) -> dict[str, Any]:
        return {
            "parent_model": str(self.parent_model),
            "tokenizer_directory": str(self.tokenizer_source),
            "calibration": (
                None if self.calibration is None else str(self.calibration)
            ),
            "sealed_parent_cache_directory": (
                None
                if self.sealed_parent_cache_directory is None
                else str(self.sealed_parent_cache_directory)
            ),
            "parent_cache_source_mode": self.parent_cache_source_mode,
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": (self.cached_parent_outputs_sha256),
            "cache_root": str(self.cache_root),
            "snapshot_root": str(self.snapshot_root),
            "publish_destination": str(self.publish_destination),
            "ledger_path": str(self.ledger_path),
            "receipt_path": str(self.receipt_path),
            "diagnostics_path": (
                None if self.diagnostics_path is None else str(self.diagnostics_path)
            ),
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "device": self.device,
            "dtype": self.dtype,
            "target": {
                "query_heads": self.target_query_heads,
                "key_value_heads": self.target_key_value_heads,
                "intermediate_size": self.target_intermediate_size,
                "exact_unique_parameters": (self.exact_target_unique_parameters),
                "compression_ratio": self.compression_ratio,
                "dense_maximum_candidates": self.dense_maximum_candidates,
                "dense_residual_basis_mode": (
                    self.dense_residual_basis_mode
                ),
                "dense_run_closed_loop": self.dense_run_closed_loop,
                "dense_vocabulary_margin_observability": (
                    self.dense_vocabulary_margin_observability
                ),
                "dense_vocabulary_margin_top_k": (
                    self.dense_vocabulary_margin_top_k
                ),
                "dense_vocabulary_margin_floor_fraction": (
                    self.dense_vocabulary_margin_floor_fraction
                ),
                "dense_vocabulary_margin_candidate_gate_enabled": (
                    self.dense_vocabulary_margin_candidate_gate_enabled
                ),
                "dense_vocabulary_margin_candidate_replay_mode": (
                    self.dense_vocabulary_margin_candidate_replay_mode
                ),
                (
                    "dense_vocabulary_margin_candidate_minimum_"
                    "sign_agreement_fraction"
                ): (
                    self
                    .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
                ),
                (
                    "dense_vocabulary_margin_candidate_minimum_"
                    "top_competitor_nonnegative_fraction"
                ): (
                    self
                    .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
                ),
                (
                    "dense_vocabulary_margin_candidate_minimum_"
                    "all_top_k_nonnegative_fraction"
                ): (
                    self
                    .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
                ),
                (
                    "dense_vocabulary_margin_candidate_minimum_"
                    "exact_top1_agreement_fraction"
                ): (
                    self
                    .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
                ),
                "fold_method_id": (
                    DENSE_BUDGET_MODEL_METHOD
                    if self.compression_ratio is not None
                    else (
                        HYBRID_MODEL_METHOD
                        if self.hybrid_model_fold
                        else (
                            PRODUCER_FIRST_MODEL_METHOD
                            if self.producer_first_model_fold
                            else None
                        )
                    )
                ),
                "target_depth": self.target_depth,
                "target_layer_types": (
                    None
                    if self.target_layer_types is None
                    else list(self.target_layer_types)
                ),
                "ordered_source_spans": (
                    None
                    if self.ordered_source_spans is None
                    else [list(span) for span in self.ordered_source_spans]
                ),
            },
            "blank_seed": self.blank_seed,
            "capture": {
                "sample_last_tokens": self.sample_last_tokens,
                "max_decode_steps": self.max_decode_steps,
                "max_capture_bytes": self.max_capture_bytes,
                "logit_rtol": self.capture_logit_rtol,
                "logit_atol": self.capture_logit_atol,
                "require_top1_agreement": (self.require_capture_top1_agreement),
                "parent_decode_trace_mode": CACHED_VERIFIED_TRACE_MODE,
                "child_decode_trace_mode": self.child_decode_trace_mode,
            },
            "evidence": {
                "max_selection_rows": self.max_selection_rows,
                "max_fit_rows": self.max_fit_rows,
                "max_validation_rows": self.max_validation_rows,
                "validation_fraction": self.validation_fraction,
            },
            "fold": {
                "endpoint_ridge": self.endpoint_ridge,
                "max_endpoint_relative_rmse": (self.max_endpoint_relative_rmse),
                "min_endpoint_cosine_similarity": (self.min_endpoint_cosine_similarity),
                "endpoint_max_prior_delta_ratio": (self.endpoint_max_prior_delta_ratio),
                "endpoint_numeric_fit_is_diagnostic_only": True,
            },
            "whole_output_gate": {
                "max_new_tokens": self.whole_output_max_new_tokens,
                "maximum_prompt_tokens": (self.whole_output_maximum_prompt_tokens),
                "fragment_minimum_characters": (
                    self.whole_output_fragment_minimum_characters
                ),
                "maximum_new_empty_responses": (
                    self.whole_output_maximum_new_empty_responses
                ),
                "maximum_new_fragment_responses": (
                    self.whole_output_maximum_new_fragment_responses
                ),
                "minimum_parent_nonempty_retention": (
                    self.whole_output_minimum_parent_nonempty_retention
                ),
                "minimum_task_score_retention": (
                    self.whole_output_minimum_task_score_retention
                ),
                "maximum_parent_task_score_regressions": (
                    self.whole_output_maximum_parent_task_score_regressions
                ),
                "maximum_new_generation_limit_saturations": (
                    self.whole_output_maximum_new_generation_limit_saturations
                ),
                "maximum_repeated_ngram_share_regressions": (
                    self.whole_output_maximum_repeated_ngram_share_regressions
                ),
                "maximum_repeated_bigram_share_delta": (
                    self.whole_output_maximum_repeated_bigram_share_delta
                ),
                "exact_text_required": False,
                "parent_outputs_cached": True,
            },
            "snapshot_chunk_bytes": self.snapshot_chunk_bytes,
            "zero_gradient": True,
            "hub_loader_used": False,
            "training_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class ResolvedProductionParentCache:
    """Authenticated parent-cache inputs used by one catalog invocation."""

    cache: Lfm2CircuitTraceCache
    calibration_batches: tuple[Mapping[str, torch.Tensor], ...]
    calibration_splits: tuple[str, ...]
    calibration_source_sha256: str | None
    evidence_split_sha256: str
    supplied_calibration_sha256: str | None
    supplied_calibration_matches_cache: bool | None
    source_mode: str
    evidence_split_source: str
    parent_cache_reused: bool
    parent_cache_build_called: bool
    parent_cache_opened_with_full_verification: bool
    parent_trace_regeneration_forward_count: int

    def runtime_receipt_fields(self) -> dict[str, Any]:
        return {
            "parent_cache_source_mode": self.source_mode,
            "sealed_parent_cache_directory": (
                str(self.cache.path)
                if self.source_mode == SEALED_PARENT_CACHE_SOURCE_MODE
                else None
            ),
            "parent_cache_opened_with_full_verification": (
                self.parent_cache_opened_with_full_verification
            ),
            "parent_cache_build_called": self.parent_cache_build_called,
            "parent_cache_reused": self.parent_cache_reused,
            "parent_cache_build_forward_count": (
                self.parent_trace_regeneration_forward_count
            ),
            "parent_trace_regeneration_forward_count": (
                self.parent_trace_regeneration_forward_count
            ),
            "parent_trace_regeneration_after_cache": 0,
            "evidence_split_source": self.evidence_split_source,
            "evidence_split_sha256": self.evidence_split_sha256,
            "supplied_calibration_sha256": (
                self.supplied_calibration_sha256
            ),
            "supplied_calibration_matches_parent_cache": (
                self.supplied_calibration_matches_cache
            ),
        }


def _calibration_tensor(
    batch: Mapping[str, torch.Tensor],
    field: str,
) -> torch.Tensor | None:
    value = batch.get(field)
    if value is None and field == "attention_mask":
        input_ids = batch.get("input_ids")
        if torch.is_tensor(input_ids):
            value = torch.ones_like(input_ids)
    if value is None:
        return None
    if not torch.is_tensor(value):
        return None
    return value.detach().to(device="cpu", dtype=torch.long).contiguous()


def _calibration_matches_cache(
    supplied: Sequence[Mapping[str, torch.Tensor]],
    cached: Sequence[Mapping[str, torch.Tensor]],
) -> bool:
    if len(supplied) != len(cached):
        return False
    for supplied_batch, cached_batch in zip(supplied, cached):
        for field in (
            "input_ids",
            "attention_mask",
            "position_ids",
            "forced_decode_ids",
        ):
            left = _calibration_tensor(supplied_batch, field)
            right = _calibration_tensor(cached_batch, field)
            if (left is None) != (right is None):
                return False
            if left is not None and not torch.equal(left, right):
                return False
    return True


def _derive_sealed_cache_splits(
    cache: Lfm2CircuitTraceCache,
) -> tuple[tuple[str, ...], str]:
    manifest = cache.manifest
    ordered_inputs = tuple(
        str(value)
        for value in manifest["corpus"]["ordered_batch_input_sha256"]
    )
    if len(ordered_inputs) != len(cache):
        raise ProductionLfm2CatalogError(
            "Sealed parent cache corpus inventory is inconsistent."
        )
    if len(ordered_inputs) < 2:
        raise ProductionLfm2CatalogError(
            "A sealed parent cache needs at least two batches to derive "
            "disjoint selection and fit evidence."
        )
    ranked_indices = sorted(
        range(len(ordered_inputs)),
        key=lambda index: (ordered_inputs[index], index),
    )
    splits = ["fit"] * len(ordered_inputs)
    splits[ranked_indices[0]] = "selection"
    if len(ordered_inputs) >= 3:
        splits[ranked_indices[-1]] = "validation"
    result = tuple(splits)
    identity = {
        "schema": SEALED_CACHE_SPLIT_SCHEMA,
        "parent_cache_key": cache.cache_key,
        "corpus_fingerprint": cache.corpus_fingerprint,
        "ordered_batch_input_sha256": list(ordered_inputs),
        "ordered_splits": list(result),
        "assignment": "content_digest_rank_extrema_v1",
    }
    return result, canonical_sha256(identity)


def _validate_tokenizer_facing_parent(
    parent: Any,
    tokenizer: Any,
    calibration_batches: Sequence[Mapping[str, torch.Tensor]],
) -> None:
    try:
        tokenizer_vocab_size = int(len(tokenizer))
    except Exception as exc:
        raise ProductionLfm2CatalogError(
            "The tokenizer does not expose a stable vocabulary size."
        ) from exc
    config_vocab_size = int(getattr(getattr(parent, "config", None), "vocab_size", 0))
    get_input_embeddings = getattr(parent, "get_input_embeddings", None)
    get_output_embeddings = getattr(parent, "get_output_embeddings", None)
    if not callable(get_input_embeddings) or not callable(get_output_embeddings):
        raise ProductionLfm2CatalogError(
            "The sealed-cache parent does not expose tokenizer-facing embeddings."
        )
    input_weight = getattr(get_input_embeddings(), "weight", None)
    output_weight = getattr(get_output_embeddings(), "weight", None)
    if not torch.is_tensor(input_weight) or not torch.is_tensor(output_weight):
        raise ProductionLfm2CatalogError(
            "The sealed-cache parent tokenizer-facing embeddings are invalid."
        )
    vocabulary_sizes = {
        tokenizer_vocab_size,
        config_vocab_size,
        int(input_weight.shape[0]),
        int(output_weight.shape[0]),
    }
    if tokenizer_vocab_size <= 0 or len(vocabulary_sizes) != 1:
        raise ProductionLfm2CatalogError(
            "Tokenizer vocabulary, parent config, input embeddings, and output "
            "embeddings do not describe the same tokenizer-facing parent."
        )
    for batch_index, batch in enumerate(calibration_batches):
        for field in ("input_ids", "forced_decode_ids"):
            values = _calibration_tensor(batch, field)
            if values is None or values.numel() == 0:
                raise ProductionLfm2CatalogError(
                    f"Sealed parent cache batch {batch_index} lacks {field}."
                )
            if int(values.min()) < 0 or int(values.max()) >= tokenizer_vocab_size:
                raise ProductionLfm2CatalogError(
                    f"Sealed parent cache batch {batch_index} {field} is outside "
                    "the tokenizer-facing parent vocabulary."
                )


def _validate_sealed_parent_cache_binding(
    cache: Lfm2CircuitTraceCache,
    parent: Any,
    tokenizer: Any,
    calibration_batches: Sequence[Mapping[str, torch.Tensor]],
) -> None:
    live_parent = fingerprint_lfm2_parent(parent)
    manifest = cache.manifest
    if (
        live_parent.identity_sha256 != cache.parent_fingerprint
        or live_parent.as_dict() != manifest["parent"]
    ):
        raise ProductionLfm2CatalogError(
            "Sealed parent cache fingerprint does not match the loaded parent."
        )
    _layers, _contracts, live_topology = _lfm2_layer_topology(parent)
    if live_topology != manifest["layer_topology"]:
        raise ProductionLfm2CatalogError(
            "Sealed parent cache topology does not match the loaded parent."
        )
    _validate_tokenizer_facing_parent(
        parent,
        tokenizer,
        calibration_batches,
    )


def resolve_production_parent_cache(
    config: ProductionLfm2CatalogConfig,
    parent: Any,
    tokenizer: Any,
    parent_capture_config: Lfm2CircuitTraceCaptureConfig,
    *,
    input_device: Any,
) -> ResolvedProductionParentCache:
    """Open a sealed cache with no parent forwards, or retain legacy build mode."""

    if config.sealed_parent_cache_directory is not None:
        sealed_directory = (
            config.sealed_parent_cache_directory.expanduser().resolve(strict=True)
        )
        try:
            parent_cache = open_lfm2_circuit_trace_cache(
                sealed_directory,
                verify=True,
            )
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                f"Sealed parent cache failed full verification: {exc}"
            ) from exc
        calibration_batches = residual.calibration_batches_from_cache(
            parent_cache
        )
        _validate_sealed_parent_cache_binding(
            parent_cache,
            parent,
            tokenizer,
            calibration_batches,
        )
        supplied_sha256 = None
        supplied_matches = None
        calibration = None
        if config.calibration is not None:
            calibration_path = config.calibration.expanduser().resolve(strict=True)
            calibration = residual.load_calibration_batches(calibration_path)
            supplied_sha256 = calibration.source_sha256
            supplied_matches = _calibration_matches_cache(
                calibration.batches,
                calibration_batches,
            )
        if calibration is not None and supplied_matches:
            calibration_splits = tuple(calibration.splits)
            calibration_source_sha256 = calibration.source_sha256
            evidence_split_source = "supplied_matching_calibration"
            evidence_split_sha256 = canonical_sha256(
                {
                    "schema": SEALED_CACHE_SPLIT_SCHEMA,
                    "parent_cache_key": parent_cache.cache_key,
                    "corpus_fingerprint": parent_cache.corpus_fingerprint,
                    "ordered_splits": list(calibration_splits),
                    "assignment": "supplied_matching_calibration",
                    "calibration_sha256": calibration.source_sha256,
                }
            )
        else:
            (
                calibration_splits,
                evidence_split_sha256,
            ) = _derive_sealed_cache_splits(parent_cache)
            calibration_source_sha256 = None
            evidence_split_source = "sealed_cache_deterministic"
        return ResolvedProductionParentCache(
            cache=parent_cache,
            calibration_batches=tuple(calibration_batches),
            calibration_splits=calibration_splits,
            calibration_source_sha256=calibration_source_sha256,
            evidence_split_sha256=evidence_split_sha256,
            supplied_calibration_sha256=supplied_sha256,
            supplied_calibration_matches_cache=supplied_matches,
            source_mode=SEALED_PARENT_CACHE_SOURCE_MODE,
            evidence_split_source=evidence_split_source,
            parent_cache_reused=True,
            parent_cache_build_called=False,
            parent_cache_opened_with_full_verification=True,
            parent_trace_regeneration_forward_count=0,
        )

    if config.calibration is None:
        raise ProductionLfm2CatalogError(
            "Legacy parent-cache build mode requires calibration."
        )
    calibration_path = config.calibration.expanduser().resolve(strict=True)
    calibration = residual.load_calibration_batches(calibration_path)
    parent_cache_keys = _directory_keys(config.parent_cache_root)
    parent_cache = build_lfm2_circuit_trace_cache(
        config.parent_cache_root,
        parent,
        calibration.batches,
        config=parent_capture_config,
        input_device=input_device,
    )
    parent_cache_reused = parent_cache.cache_key in parent_cache_keys
    return ResolvedProductionParentCache(
        cache=parent_cache,
        calibration_batches=residual.calibration_batches_from_cache(
            parent_cache
        ),
        calibration_splits=tuple(calibration.splits),
        calibration_source_sha256=calibration.source_sha256,
        evidence_split_sha256=canonical_sha256(
            {
                "schema": SEALED_CACHE_SPLIT_SCHEMA,
                "parent_cache_key": parent_cache.cache_key,
                "corpus_fingerprint": parent_cache.corpus_fingerprint,
                "ordered_splits": list(calibration.splits),
                "assignment": "legacy_calibration",
                "calibration_sha256": calibration.source_sha256,
            }
        ),
        supplied_calibration_sha256=calibration.source_sha256,
        supplied_calibration_matches_cache=True,
        source_mode=CALIBRATION_PARENT_CACHE_SOURCE_MODE,
        evidence_split_source="legacy_calibration",
        parent_cache_reused=parent_cache_reused,
        parent_cache_build_called=True,
        parent_cache_opened_with_full_verification=False,
        parent_trace_regeneration_forward_count=(
            0
            if parent_cache_reused
            else _expected_capture_forwards(calibration.batches)
        ),
    )


@dataclass(frozen=True, slots=True)
class ProductionParentCachePayload:
    cache: Any
    rows: Any
    calibration_batches: tuple[Mapping[str, torch.Tensor], ...]
    calibration_splits: tuple[str, ...]
    partition: Any

    @property
    def cache_key(self) -> str:
        return str(self.cache.cache_key)

    def verify(self) -> None:
        self.cache.verify()
        if (
            self.rows.cache_key != self.cache.cache_key
            or self.rows.corpus_fingerprint != self.cache.corpus_fingerprint
            or len(self.calibration_batches) != len(self.cache)
            or len(self.calibration_splits) != len(self.cache)
        ):
            raise ProductionLfm2CatalogError(
                "Parent evidence payload differs from its immutable cache."
            )


class CachedProductionEvidenceBuilder:
    """Recollect child traces on cached prefixes and build real LFM evidence."""

    def __init__(
        self,
        *,
        cache_root: Path,
        capture_config: Lfm2CircuitTraceCaptureConfig,
    ) -> None:
        self.cache_root = cache_root
        self.capture_config = capture_config
        self.records: list[dict[str, Any]] = []

    def __call__(
        self,
        request: Lfm2EvidenceBuildRequest,
    ) -> Lfm2PreparedEvidence:
        payload = request.parent_cache_payload
        if not isinstance(payload, ProductionParentCachePayload):
            raise ProductionLfm2CatalogError(
                "LFM2 evidence callback received an unknown cache payload."
            )
        payload.verify()
        if request.parent_cache_fingerprint != payload.cache_key:
            raise ProductionLfm2CatalogError(
                "Evidence request cache fingerprint differs from the payload."
            )
        existing = _directory_keys(self.cache_root)
        child_cache, child_rows = residual.build_child_trace_cache(
            self.cache_root,
            request.child_model,
            payload.calibration_batches,
            capture_config=self.capture_config,
            batch_splits=payload.calibration_splits,
        )
        cache_reused = child_cache.cache_key in existing
        partition = payload.partition
        evidence, split = residual.build_cached_layer_evidence(
            payload.rows,
            child_rows,
            layer_index=request.layer_index,
            selection_indices=partition.selection_indices,
            fit_indices=partition.fit_indices,
            validation_indices=partition.validation_indices,
        )
        child_forwards = (
            0
            if cache_reused
            else _expected_capture_forwards(payload.calibration_batches)
        )
        self.records.append(
            {
                "layer_index": request.layer_index,
                "child_state_fingerprint": request.child_state_fingerprint,
                "child_cache_key": child_cache.cache_key,
                "child_trace_rows_sha256": child_rows.trace_rows_sha256,
                "cache_reused": cache_reused,
                "child_forward_count": child_forwards,
                "fit_rows": len(split.fit_indices),
                "validation_rows": len(split.validation_indices),
                "parent_forward_count": 0,
            }
        )
        return Lfm2PreparedEvidence(
            evidence=evidence,
            child_forward_count=child_forwards,
        )

    def build_producer_first(
        self,
        request: Lfm2ProducerFirstEvidenceBuildRequest,
    ) -> Lfm2PreparedProducerFirstEvidence:
        """Bind sealed parent rows without executing the blank child."""

        payload = request.parent_cache_payload
        if not isinstance(payload, ProductionParentCachePayload):
            raise ProductionLfm2CatalogError(
                "Producer-first callback received an unknown cache payload."
            )
        payload.verify()
        if request.parent_cache_fingerprint != payload.cache_key:
            raise ProductionLfm2CatalogError(
                "Producer-first request cache fingerprint differs from payload."
            )
        if fingerprint_model_state(request.child_model) != (
            request.blank_state_fingerprint
        ):
            raise ProductionLfm2CatalogError(
                "Producer-first evidence was not requested from the "
                "authenticated compiler-zero state."
            )

        def capture_child(model: Any) -> Lfm2ProducerFirstCapture:
            existing = _directory_keys(self.cache_root)
            child_cache, child_rows = residual.build_child_trace_cache(
                self.cache_root,
                model,
                payload.calibration_batches,
                capture_config=self.capture_config,
                batch_splits=payload.calibration_splits,
            )
            cache_reused = child_cache.cache_key in existing
            child_forwards = (
                0
                if cache_reused
                else _expected_capture_forwards(payload.calibration_batches)
            )
            self.records.append(
                {
                    "kind": "producer_first_capture",
                    "child_state_fingerprint": fingerprint_model_state(model),
                    "child_cache_key": child_cache.cache_key,
                    "child_trace_rows_sha256": child_rows.trace_rows_sha256,
                    "cache_reused": cache_reused,
                    "child_forward_count": child_forwards,
                    "parent_forward_count": 0,
                    "blank_child_features_used": False,
                }
            )
            return Lfm2ProducerFirstCapture(
                rows=child_rows,
                cache_key=child_cache.cache_key,
                child_forward_count=child_forwards,
                cache_reused=cache_reused,
            )

        partition = payload.partition
        evidence = Lfm2ProducerFirstEvidence(
            parent_rows=payload.rows,
            selection_indices=tuple(partition.selection_indices),
            fit_indices=tuple(partition.fit_indices),
            validation_indices=tuple(partition.validation_indices),
            capture_child=capture_child,
            parent_cache_fingerprint=payload.cache_key,
            parent_identity_sha256=payload.cache.parent_fingerprint,
            capture_builder_id=(
                "run_lfm2_dynamic_catalog.cached_child_trace_capture.v1"
            ),
        )
        self.records.append(
            {
                "kind": "producer_first_parent_only_evidence",
                "blank_state_fingerprint": request.blank_state_fingerprint,
                "evidence_fingerprint": evidence.fingerprint,
                "child_forward_count": 0,
                "parent_forward_count": 0,
                "blank_child_features_used": False,
            }
        )
        return Lfm2PreparedProducerFirstEvidence(
            evidence=evidence,
            child_forward_count=0,
        )

    @staticmethod
    def _index_rows(value: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        if not indices:
            raise ProductionLfm2CatalogError(
                "Hybrid LFM2 evidence requires nonempty row indices."
            )
        index = torch.tensor(tuple(indices), dtype=torch.long, device=value.device)
        return value.index_select(0, index).detach().to(device="cpu").contiguous()

    @classmethod
    def _optional_rows(
        cls,
        value: torch.Tensor,
        indices: Sequence[int],
    ) -> torch.Tensor | None:
        return None if not indices else cls._index_rows(value, indices)

    def build_model(
        self,
        request: Lfm2ModelEvidenceBuildRequest,
    ) -> Lfm2PreparedHybridProducerEvidence:
        """Bind parent rows and a callback without executing the blank child."""

        payload = request.parent_cache_payload
        if not isinstance(payload, ProductionParentCachePayload):
            raise ProductionLfm2CatalogError(
                "Hybrid producer callback received an unknown cache payload."
            )
        payload.verify()
        if request.parent_cache_fingerprint != payload.cache_key:
            raise ProductionLfm2CatalogError(
                "Hybrid producer request cache fingerprint differs from payload."
            )
        if fingerprint_model_state(request.child_model) != (
            request.child_state_fingerprint
        ):
            raise ProductionLfm2CatalogError(
                "Hybrid producer evidence was not requested from the "
                "authenticated compiler-zero state."
            )

        def capture_child(model: Any) -> Lfm2ProducerFirstCapture:
            producer_state = fingerprint_model_state(model)
            if producer_state == request.child_state_fingerprint:
                raise ProductionLfm2CatalogError(
                    "Refusing to capture hybrid features from the compiler-zero "
                    "blank child."
                )
            existing = _directory_keys(self.cache_root)
            child_cache, child_rows = residual.build_child_trace_cache(
                self.cache_root,
                model,
                payload.calibration_batches,
                capture_config=self.capture_config,
                batch_splits=payload.calibration_splits,
            )
            cache_reused = child_cache.cache_key in existing
            child_forwards = (
                0
                if cache_reused
                else _expected_capture_forwards(payload.calibration_batches)
            )
            if fingerprint_model_state(model) != producer_state:
                raise ProductionLfm2CatalogError(
                    "Hybrid producer child changed during trace capture."
                )
            self.records.append(
                {
                    "kind": "hybrid_producer_first_capture",
                    "child_state_fingerprint": producer_state,
                    "child_cache_key": child_cache.cache_key,
                    "child_trace_rows_sha256": child_rows.trace_rows_sha256,
                    "cache_reused": cache_reused,
                    "child_forward_count": child_forwards,
                    "parent_forward_count": 0,
                    "blank_child_features_used": False,
                }
            )
            return Lfm2ProducerFirstCapture(
                rows=child_rows,
                cache_key=child_cache.cache_key,
                child_forward_count=child_forwards,
                cache_reused=cache_reused,
            )

        partition = payload.partition
        evidence = Lfm2ProducerFirstEvidence(
            parent_rows=payload.rows,
            selection_indices=tuple(partition.selection_indices),
            fit_indices=tuple(partition.fit_indices),
            validation_indices=tuple(partition.validation_indices),
            capture_child=capture_child,
            parent_cache_fingerprint=payload.cache_key,
            parent_identity_sha256=payload.cache.parent_fingerprint,
            capture_builder_id=(
                "run_lfm2_dynamic_catalog.hybrid_producer_capture.v1"
            ),
        )
        self.records.append(
            {
                "kind": "hybrid_parent_only_evidence",
                "blank_state_fingerprint": request.child_state_fingerprint,
                "ordered_source_spans": encode_lfm2_source_spans(
                    request.selected_target.ordered_source_spans
                ),
                "target_layer_types": encode_lfm2_layer_types(
                    request.selected_target.target_layer_types
                ),
                "evidence_fingerprint": evidence.fingerprint,
                "child_forward_count": 0,
                "parent_forward_count": 0,
                "blank_child_features_used": False,
            }
        )
        return Lfm2PreparedHybridProducerEvidence(
            evidence=evidence,
            child_forward_count=0,
        )

    def build_model_from_installed_producer(
        self,
        request: Lfm2ModelEvidenceBuildRequest,
    ) -> Lfm2PreparedModelEvidence:
        """Assemble one model-level payload from the existing verified cache."""

        from neural_forge.hf_lfm2_model_fold import (
            Lfm2ModelFoldSpan,
            assemble_lfm2_model_fold_evidence,
            fingerprint_lfm2_model_feature_state,
        )

        payload = request.parent_cache_payload
        if not isinstance(payload, ProductionParentCachePayload):
            raise ProductionLfm2CatalogError(
                "LFM2 model evidence callback received an unknown cache payload."
            )
        payload.verify()
        if request.parent_cache_fingerprint != payload.cache_key:
            raise ProductionLfm2CatalogError(
                "Model evidence request cache fingerprint differs from payload."
            )
        producer_plan_fingerprint = getattr(
            request.child_model,
            "_neural_forge_producer_plan_fingerprint",
            None,
        )
        if (
            not isinstance(producer_plan_fingerprint, str)
            or len(producer_plan_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in producer_plan_fingerprint
            )
        ):
            raise ProductionLfm2CatalogError(
                "Hybrid child lacks an authenticated provisional producer plan."
            )
        feature_state_before = fingerprint_model_state(request.child_model)
        feature_tensor_state_before = fingerprint_lfm2_model_feature_state(
            request.child_model
        )
        if feature_state_before != request.child_state_fingerprint:
            raise ProductionLfm2CatalogError(
                "Hybrid child changed before feature capture."
            )
        existing = _directory_keys(self.cache_root)
        child_cache, child_rows = residual.build_child_trace_cache(
            self.cache_root,
            request.child_model,
            payload.calibration_batches,
            capture_config=self.capture_config,
            batch_splits=payload.calibration_splits,
        )
        cache_reused = child_cache.cache_key in existing
        feature_state_after = fingerprint_model_state(request.child_model)
        feature_tensor_state_after = fingerprint_lfm2_model_feature_state(
            request.child_model
        )
        if (
            feature_state_after != feature_state_before
            or feature_tensor_state_after != feature_tensor_state_before
        ):
            raise ProductionLfm2CatalogError(
                "Hybrid child changed while its features were captured."
            )
        parent_rows = payload.rows
        if (
            parent_rows.paths != child_rows.paths
            or parent_rows.path_splits != child_rows.path_splits
            or parent_rows.corpus_fingerprint != child_rows.corpus_fingerprint
        ):
            raise ProductionLfm2CatalogError(
                "Parent and hybrid-child trace paths are not exactly aligned."
            )
        selected = request.selected_target
        if tuple(child_rows.layer_kinds) != selected.target_layer_types:
            raise ProductionLfm2CatalogError(
                "Hybrid child trace topology differs from the selected target."
            )
        spans = tuple(
            Lfm2ModelFoldSpan(
                source_layer_indices=span,
                target_layer_index=target_index,
            )
            for target_index, span in enumerate(selected.ordered_source_spans)
        )
        partition = payload.partition
        selection = partition.selection_indices
        fit = partition.fit_indices
        validation = partition.validation_indices
        source_to_target = {
            source_index: span.target_layer_index
            for span in spans
            for source_index in span.source_layer_indices
        }
        layer_evidence: dict[int, Lfm2ClosedLoopFoldEvidence] = {}
        parent_pre: dict[int, torch.Tensor] = {}
        parent_post: dict[int, torch.Tensor] = {}
        validation_pre: dict[int, torch.Tensor] = {}
        validation_post: dict[int, torch.Tensor] = {}
        for source_index, target_index in sorted(source_to_target.items()):
            parent_pre[source_index] = self._index_rows(
                parent_rows.field(source_index, "residual.pre"),
                fit,
            )
            parent_post[source_index] = self._index_rows(
                parent_rows.field(source_index, "residual.post_layer"),
                fit,
            )
            if validation:
                validation_pre[source_index] = self._index_rows(
                    parent_rows.field(source_index, "residual.pre"),
                    validation,
                )
                validation_post[source_index] = self._index_rows(
                    parent_rows.field(source_index, "residual.post_layer"),
                    validation,
                )
            mlp = Lfm2MLPFoldEvidence(
                parent_gate_rows=self._index_rows(
                    parent_rows.field(source_index, "mlp.gate"),
                    selection,
                ),
                parent_up_rows=self._index_rows(
                    parent_rows.field(source_index, "mlp.up"),
                    selection,
                ),
                parent_post_swiglu_rows=self._index_rows(
                    parent_rows.field(source_index, "mlp.post_swiglu"),
                    selection,
                ),
                parent_contribution_rows=None,
                child_post_swiglu_rows=self._index_rows(
                    child_rows.field(target_index, "mlp.post_swiglu"),
                    fit,
                ),
                parent_branch_output_rows=parent_post[source_index],
                child_pre_branch_residual_rows=self._index_rows(
                    child_rows.field(target_index, "residual.post_operator"),
                    fit,
                ),
                validation_child_post_swiglu_rows=self._optional_rows(
                    child_rows.field(target_index, "mlp.post_swiglu"),
                    validation,
                ),
                validation_parent_branch_output_rows=(
                    None
                    if not validation
                    else validation_post[source_index]
                ),
                validation_child_pre_branch_residual_rows=self._optional_rows(
                    child_rows.field(target_index, "residual.post_operator"),
                    validation,
                ),
            )
            attention = None
            if (
                parent_rows.layer_kinds[source_index] == "full_attention"
                and child_rows.layer_kinds[target_index] == "full_attention"
            ):
                attention = Lfm2AttentionFoldEvidence(
                    parent_query_rows=self._index_rows(
                        parent_rows.field(source_index, "attention.q"),
                        selection,
                    ),
                    parent_key_rows=self._index_rows(
                        parent_rows.field(source_index, "attention.k"),
                        selection,
                    ),
                    parent_value_rows=self._index_rows(
                        parent_rows.field(source_index, "attention.v"),
                        selection,
                    ),
                    parent_context_rows=self._index_rows(
                        parent_rows.field(source_index, "attention.context"),
                        selection,
                    ),
                    parent_contribution_rows=None,
                    child_context_rows=self._index_rows(
                        child_rows.field(target_index, "attention.context"),
                        fit,
                    ),
                    parent_branch_output_rows=self._index_rows(
                        parent_rows.field(source_index, "residual.post_operator"),
                        fit,
                    ),
                    child_pre_branch_residual_rows=self._index_rows(
                        child_rows.field(target_index, "residual.pre"),
                        fit,
                    ),
                    validation_child_context_rows=self._optional_rows(
                        child_rows.field(target_index, "attention.context"),
                        validation,
                    ),
                    validation_parent_branch_output_rows=self._optional_rows(
                        parent_rows.field(
                            source_index,
                            "residual.post_operator",
                        ),
                        validation,
                    ),
                    validation_child_pre_branch_residual_rows=self._optional_rows(
                        child_rows.field(target_index, "residual.pre"),
                        validation,
                    ),
                )
            layer_evidence[source_index] = Lfm2ClosedLoopFoldEvidence(
                mlp=mlp,
                attention=attention,
            )
        evidence = assemble_lfm2_model_fold_evidence(
            cache_fingerprint=payload.cache_key,
            child_feature_state_sha256=feature_tensor_state_after,
            spans=spans,
            layer_fold_evidence=layer_evidence,
            parent_pre_residual_rows=parent_pre,
            parent_post_residual_rows=parent_post,
            validation_parent_pre_residual_rows=(
                validation_pre if validation else None
            ),
            validation_parent_post_residual_rows=(
                validation_post if validation else None
            ),
        )
        child_forwards = (
            0
            if cache_reused
            else _expected_capture_forwards(payload.calibration_batches)
        )
        self.records.append(
            {
                "kind": "hybrid_model",
                "ordered_source_spans": encode_lfm2_source_spans(
                    selected.ordered_source_spans
                ),
                "target_layer_types": encode_lfm2_layer_types(
                    selected.target_layer_types
                ),
                "child_state_fingerprint": request.child_state_fingerprint,
                "child_cache_key": child_cache.cache_key,
                "child_trace_rows_sha256": child_rows.trace_rows_sha256,
                "cache_reused": cache_reused,
                "child_forward_count": child_forwards,
                "fit_rows": len(fit),
                "validation_rows": len(validation),
                "parent_forward_count": 0,
            }
        )
        return Lfm2PreparedModelEvidence(
            evidence=evidence,
            producer_state_fingerprint=request.child_state_fingerprint,
            child_feature_state_fingerprint=feature_state_after,
            producer_plan_fingerprint=producer_plan_fingerprint,
            child_forward_count=child_forwards,
        )


def _gate_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    teacher = report["teacher_forced"]
    free_run = report["free_run"]
    return {
        "passed": bool(report["passed"]),
        "reasons": list(report["reasons"]),
        "score": float(report["score"]),
        "teacher_top1": float(teacher["top1_agreement"]),
        "free_run_token_match": float(free_run["token_match_fraction"]),
        "free_run_lcp": float(free_run["lcp_fraction"]),
        "mean_residual_cosine": float(report["mean_residual_cosine"]),
        "mean_residual_relative_rmse": float(report["mean_residual_relative_rmse"]),
        "cached_full_self_consistent": bool(report["cached_full_self_consistent"]),
        "cached_full_top1_diagnostic_fraction": float(
            report["cached_full_top1_diagnostic_fraction"]
        ),
    }


def _canonical_trace_layer_kind(value: str) -> str:
    normalized = str(value).strip()
    if normalized == "conv":
        return "short_conv"
    if normalized in {"short_conv", "full_attention"}:
        return normalized
    raise ProductionLfm2CatalogError(
        f"Unsupported LFM2 trajectory layer kind {value!r}."
    )


def build_span_mapped_parent_trajectory_rows(
    parent_rows: residual.TraceRowIndex,
    child_rows: residual.TraceRowIndex,
    *,
    ordered_source_spans: Sequence[Sequence[int]],
    target_layer_types: Sequence[str],
) -> tuple[residual.TraceRowIndex, Mapping[str, Any]]:
    """Project parent residual endpoints onto a depth-compacted child topology.

    Logits and free-running generations remain the original parent/child cache
    comparison.  Only the residual-layer view is projected: compact child
    layer ``i`` is compared with ``residual.post_layer`` at the final parent
    layer in its authenticated ordered source span.
    """

    if not isinstance(parent_rows, residual.TraceRowIndex) or not isinstance(
        child_rows,
        residual.TraceRowIndex,
    ):
        raise ProductionLfm2CatalogError(
            "Span-mapped trajectory evaluation requires typed trace rows."
        )
    try:
        spans = tuple(tuple(value) for value in ordered_source_spans)
        declared_target_types = tuple(str(value) for value in target_layer_types)
    except TypeError as exc:
        raise ProductionLfm2CatalogError(
            "Hybrid trajectory spans and target types must be sequences."
        ) from exc
    parent_layer_count = len(parent_rows.layer_kinds)
    child_layer_count = len(child_rows.layer_kinds)
    if (
        len(spans) != child_layer_count
        or len(declared_target_types) != child_layer_count
        or any(not span for span in spans)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for span in spans
            for index in span
        )
        or tuple(index for span in spans for index in span)
        != tuple(range(parent_layer_count))
    ):
        raise ProductionLfm2CatalogError(
            "Hybrid trajectory spans must partition every parent layer in "
            "order and bind exactly one nonempty span to every child layer."
        )
    expected_child_kinds = tuple(
        _canonical_trace_layer_kind(value) for value in declared_target_types
    )
    observed_child_kinds = tuple(
        _canonical_trace_layer_kind(value) for value in child_rows.layer_kinds
    )
    if observed_child_kinds != expected_child_kinds:
        raise ProductionLfm2CatalogError(
            "Hybrid trajectory child kinds differ from the selected target."
        )
    if (
        parent_rows.paths != child_rows.paths
        or parent_rows.path_splits != child_rows.path_splits
        or parent_rows.corpus_fingerprint != child_rows.corpus_fingerprint
    ):
        raise ProductionLfm2CatalogError(
            "Parent and hybrid-child trajectory paths, splits, or corpus differ."
        )

    projected_fields: dict[int, Mapping[str, torch.Tensor]] = {}
    entries: list[Mapping[str, Any]] = []
    for target_index, span in enumerate(spans):
        endpoint = span[-1]
        parent_residual = parent_rows.field(endpoint, "residual.post_layer")
        child_residual = child_rows.field(target_index, "residual.post_layer")
        if tuple(parent_residual.shape) != tuple(child_residual.shape):
            raise ProductionLfm2CatalogError(
                "A mapped parent span endpoint and child residual have "
                f"different shapes at target layer {target_index}."
            )
        projected_fields[target_index] = MappingProxyType(
            {"residual.post_layer": parent_residual}
        )
        entries.append(
            MappingProxyType(
                {
                    "target_layer_index": target_index,
                    "target_layer_kind": observed_child_kinds[target_index],
                    "source_layer_indices": list(span),
                    "source_span_endpoint_layer_index": endpoint,
                    "source_span_endpoint_layer_kind": (
                        _canonical_trace_layer_kind(
                            parent_rows.layer_kinds[endpoint]
                        )
                    ),
                    "parent_residual_field": "residual.post_layer",
                    "child_residual_field": "residual.post_layer",
                    "row_shape": [int(size) for size in parent_residual.shape],
                }
            )
        )

    mapping_identity = {
        "schema": "neural-forge-lfm2-span-mapped-trajectory-v1",
        "parent_trace_rows_sha256": parent_rows.trace_rows_sha256,
        "child_trace_rows_sha256": child_rows.trace_rows_sha256,
        "parent_layer_kinds": list(parent_rows.layer_kinds),
        "child_layer_kinds": list(observed_child_kinds),
        "ordered_source_spans": [list(span) for span in spans],
        "entries": [dict(value) for value in entries],
    }
    projected = residual.TraceRowIndex(
        paths=parent_rows.paths,
        path_splits=parent_rows.path_splits,
        layer_kinds=observed_child_kinds,
        layer_fields=MappingProxyType(projected_fields),
        cache_key=parent_rows.cache_key,
        corpus_fingerprint=parent_rows.corpus_fingerprint,
        trace_rows_sha256=canonical_sha256(mapping_identity),
    )
    # Reuse the legacy gate's exact metadata contract after projection.  This
    # authenticates paths, split labels, corpus, and the target layer kinds.
    projected.assert_aligned(child_rows)
    report = MappingProxyType(
        {
            **mapping_identity,
            "projected_parent_trace_rows_sha256": (
                projected.trace_rows_sha256
            ),
            "mapping_mode": "parent_source_span_endpoint_to_child_layer",
            "paths_authenticated": True,
            "path_splits_authenticated": True,
            "corpus_authenticated": True,
            "target_layer_kinds_authenticated": True,
            "complete_parent_layer_partition": True,
            "complete_child_layer_coverage": True,
        }
    )
    return projected, report


class CachedTrajectoryEvaluator:
    """Whole-model gate using immutable parent outputs and fresh child traces."""

    def __init__(
        self,
        *,
        payload: ProductionParentCachePayload,
        capture_config: Lfm2CircuitTraceCaptureConfig,
        gate_config: residual.TrajectoryGateConfig,
        final_cache_root: Path,
        reload_cache_root: Path,
        ordered_source_spans: Sequence[Sequence[int]] | None = None,
        target_layer_types: Sequence[str] | None = None,
    ) -> None:
        self.payload = payload
        self.capture_config = capture_config
        self.gate_config = gate_config
        self.final_cache_root = final_cache_root
        self.reload_cache_root = reload_cache_root
        if (ordered_source_spans is None) != (target_layer_types is None):
            raise ProductionLfm2CatalogError(
                "Hybrid trajectory evaluation requires both ordered source "
                "spans and target layer types."
            )
        self.ordered_source_spans = (
            None
            if ordered_source_spans is None
            else tuple(tuple(value) for value in ordered_source_spans)
        )
        self.target_layer_types = (
            None
            if target_layer_types is None
            else tuple(str(value) for value in target_layer_types)
        )
        self.events: list[dict[str, Any]] = []

    def _evaluate_model(
        self,
        model: Any,
        *,
        phase: str,
        cache_root: Path,
    ) -> FoldDecision:
        existing = _directory_keys(cache_root)
        cache, rows = residual.build_child_trace_cache(
            cache_root,
            model,
            self.payload.calibration_batches,
            capture_config=self.capture_config,
            batch_splits=self.payload.calibration_splits,
        )
        cache_reused = cache.cache_key in existing
        trajectory_parent_rows = self.payload.rows
        trajectory_mapping = None
        if self.ordered_source_spans is not None:
            assert self.target_layer_types is not None
            trajectory_parent_rows, trajectory_mapping = (
                build_span_mapped_parent_trajectory_rows(
                    self.payload.rows,
                    rows,
                    ordered_source_spans=self.ordered_source_spans,
                    target_layer_types=self.target_layer_types,
                )
            )
        report = residual.evaluate_trajectory_gate(
            self.payload.cache,
            cache,
            model,
            trajectory_parent_rows,
            rows,
            config=self.gate_config,
        )
        if trajectory_mapping is not None:
            report = {
                **report,
                "residual_trajectory_mapping": dict(trajectory_mapping),
            }
        summary = _gate_summary(report)
        child_forward_count = (
            0
            if cache_reused
            else _expected_capture_forwards(self.payload.calibration_batches)
        )
        self.events.append(
            {
                "phase": phase,
                "child_cache_key": cache.cache_key,
                "child_trace_rows_sha256": rows.trace_rows_sha256,
                "cache_reused": cache_reused,
                "child_forward_count": child_forward_count,
                "parent_forward_count": 0,
                "summary": summary,
                "residual_trajectory_mapping": (
                    None
                    if trajectory_mapping is None
                    else dict(trajectory_mapping)
                ),
                "report": report,
            }
        )
        metrics = {
            "trajectory_score": summary["score"],
            "teacher_top1": summary["teacher_top1"],
            "free_run_token_match": summary["free_run_token_match"],
            "mean_residual_cosine": summary["mean_residual_cosine"],
            "mean_residual_relative_rmse": (summary["mean_residual_relative_rmse"]),
            "cached_full_self_consistent": (summary["cached_full_self_consistent"]),
            "parent_forward_count": 0,
            "child_forward_count": child_forward_count,
            "span_mapped_residual_layers": trajectory_mapping is not None,
        }
        if summary["passed"]:
            return FoldDecision.accept(
                f"{phase} cached-parent trajectory gate accepted.",
                metrics,
            )
        return FoldDecision.reject(
            f"{phase} cached-parent trajectory gate rejected: "
            + "; ".join(summary["reasons"]),
            metrics,
        )

    def evaluate(self, context: Any) -> FoldDecision:
        return self._evaluate_model(
            context.execution_context.child_model,
            phase="accepted_in_memory_child",
            cache_root=self.final_cache_root,
        )

    def evaluate_fresh_reload(self, model: Any) -> FoldDecision:
        return self._evaluate_model(
            model,
            phase="freshly_reloaded_child",
            cache_root=self.reload_cache_root,
        )


class DirectLocalArtifactIO:
    """Package and reload JSON+safetensors plus the exact local tokenizer."""

    def __init__(
        self,
        *,
        tokenizer_directory: Path,
        tokenizer_fingerprint: str,
        device: str,
        dtype: torch.dtype,
    ) -> None:
        self.tokenizer_directory = tokenizer_directory
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.device = device
        self.dtype = dtype

    def save(self, model: Any, destination: Path) -> None:
        residual.atomic_save_local_checkpoint(
            model,
            destination,
            output_root=destination.parent,
        )
        for name in _TOKENIZER_ARTIFACTS:
            source = self.tokenizer_directory / name
            if source.is_file() and not source.is_symlink():
                shutil.copy2(source, destination / name)
        observed = fingerprint_local_tokenizer(destination)
        if observed != self.tokenizer_fingerprint:
            raise ProductionLfm2CatalogError(
                "Packaged tokenizer differs from the selected local interface."
            )

    def load(self, source: Path) -> Any:
        observed = fingerprint_local_tokenizer(source)
        if observed != self.tokenizer_fingerprint:
            raise ProductionLfm2CatalogError(
                "Reloaded tokenizer differs from the selected local interface."
            )
        model = residual.load_local_lfm2_model(
            source,
            device=self.device,
            dtype=self.dtype,
        )
        if getattr(model.config, "block_ff_dim", None) is None:
            model.config.block_ff_dim = int(model.config.intermediate_size)
        return model


@dataclass(frozen=True, slots=True)
class ProductionCatalogWiring:
    system: AutomaticDynamicFoldingSystem
    target_request: DynamicTargetRequest
    target_adapter: Lfm2TargetSynthesisAdapter
    stack_adapter: Lfm2DynamicFoldStackAdapter
    budget_plan: Any | None = None


def _build_model_fold_config(config: ProductionLfm2CatalogConfig) -> Any:
    from neural_forge.hf_lfm2_model_fold import Lfm2ModelFoldConfig

    return Lfm2ModelFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=config.endpoint_ridge,
            max_endpoint_relative_rmse=config.max_endpoint_relative_rmse,
            min_endpoint_cosine_similarity=config.min_endpoint_cosine_similarity,
            max_endpoint_prior_delta_ratio=(
                config.endpoint_max_prior_delta_ratio
            ),
        )
    )


def _build_producer_first_fold_config(
    config: ProductionLfm2CatalogConfig,
) -> Lfm2ProducerFirstFoldConfig:
    return Lfm2ProducerFirstFoldConfig(
        layer_config=Lfm2ClosedLoopFoldConfig(
            endpoint_ridge=config.endpoint_ridge,
            max_endpoint_relative_rmse=config.max_endpoint_relative_rmse,
            min_endpoint_cosine_similarity=config.min_endpoint_cosine_similarity,
            max_endpoint_prior_delta_ratio=(
                config.endpoint_max_prior_delta_ratio
            ),
        ),
        endpoint_refinement_passes=2,
    )


def build_production_catalog_wiring(
    config: ProductionLfm2CatalogConfig,
    *,
    parent_cache_payload: ProductionParentCachePayload,
    parent_graph: Any | None = None,
    tokenizer_fingerprint: str,
    evidence_builder: Any,
    final_evaluator: Any,
    model_saver: Any,
    model_loader: Any,
    fresh_reload_evaluator: Any,
    ledger: ExperimentLedger,
    device: torch.device | str,
    dtype: torch.dtype,
) -> ProductionCatalogWiring:
    """Configure the production target and stack catalogs exactly once."""

    snapshot = DynamicFoldSnapshotConfig.disk(
        config.snapshot_root,
        chunk_bytes=config.snapshot_chunk_bytes,
    )
    target_adapter = Lfm2TargetSynthesisAdapter(
        terminal_topology=(
            Lfm2Topology(1, 1, 32)
            if config.compression_ratio is not None
            else config.target_topology
        ),
        evidence_fingerprint=parent_cache_payload.cache_key,
        ledger=ledger,
        initial_mlp_step=config.initial_mlp_step,
        maximum_mlp_step=config.maximum_mlp_step,
        device=device,
        dtype=dtype,
    )
    stack_adapter = Lfm2DynamicFoldStackAdapter(
        Lfm2DynamicStackResources(
            parent_cache_fingerprint=parent_cache_payload.cache_key,
            parent_cache_payload=parent_cache_payload,
            evidence_builder=evidence_builder,
            final_evaluator=final_evaluator,
            model_saver=model_saver,
            model_loader=model_loader,
            fresh_reload_evaluator=fresh_reload_evaluator,
            saver_id="direct-local-lfm2-safetensors-tokenizer-v1",
            loader_id="direct-local-lfm2-safetensors-tokenizer-v1",
            ledger=ledger,
            fold_config=Lfm2ClosedLoopFoldConfig(
                endpoint_ridge=config.endpoint_ridge,
                max_endpoint_relative_rmse=(config.max_endpoint_relative_rmse),
                min_endpoint_cosine_similarity=(config.min_endpoint_cosine_similarity),
                max_endpoint_prior_delta_ratio=(config.endpoint_max_prior_delta_ratio),
            ),
            model_evidence_builder=(
                evidence_builder.build_model
                if config.hybrid_model_fold
                else None
            ),
            model_fold_config=(
                None
                if not config.hybrid_model_fold
                else _build_model_fold_config(config)
            ),
            producer_first_evidence_builder=(
                evidence_builder.build_producer_first
                if config.producer_first_model_fold
                else None
            ),
            producer_first_fold_config=(
                _build_producer_first_fold_config(config)
                if config.producer_first_model_fold
                else None
            ),
            dense_fold_config=Lfm2DenseBudgetFoldConfig(
                residual_basis_mode=config.dense_residual_basis_mode,
                run_closed_loop=config.dense_run_closed_loop,
                vocabulary_margin_observability=(
                    config.dense_vocabulary_margin_observability
                ),
                vocabulary_margin_top_k=(
                    config.dense_vocabulary_margin_top_k
                ),
                vocabulary_margin_floor_fraction=(
                    config.dense_vocabulary_margin_floor_fraction
                ),
                vocabulary_margin_candidate_gate_enabled=(
                    config.dense_vocabulary_margin_candidate_gate_enabled
                ),
                vocabulary_margin_candidate_replay_mode=(
                    config.dense_vocabulary_margin_candidate_replay_mode
                ),
                vocabulary_margin_candidate_minimum_sign_agreement_fraction=(
                    config
                    .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
                ),
                vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=(
                    config
                    .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
                ),
                vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=(
                    config
                    .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
                ),
                vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=(
                    config
                    .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
                ),
            ),
            snapshot_config=snapshot,
        )
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        ledger=ledger,
        snapshot_config=snapshot,
    )
    budget_plan = None
    if config.compression_ratio is not None:
        if parent_graph is None:
            raise ProductionLfm2CatalogError(
                "Automatic compression-ratio planning requires the inspected "
                "parent circuit graph."
            )
        budget_plan = system.plan_budgeted_target(
            parent_graph=parent_graph,
            budget_request=DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                compression_ratio=config.compression_ratio,
                maximum_candidates=config.dense_maximum_candidates,
            ),
            child_name=config.child_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
            blank_seed=config.blank_seed,
        )
        request = budget_plan.target_request
    else:
        request_configuration: list[tuple[str, str | int]] = [
            ("source_parent", config.parent_name)
        ]
        if config.hybrid_model_fold:
            assert config.target_layer_types is not None
            assert config.ordered_source_spans is not None
            request_configuration.extend(
                (
                    (
                        "target_layer_types",
                        encode_lfm2_layer_types(config.target_layer_types),
                    ),
                    (
                        "ordered_source_spans",
                        encode_lfm2_source_spans(config.ordered_source_spans),
                    ),
                    ("target_depth", len(config.target_layer_types)),
                )
            )
        request = DynamicTargetRequest(
            child_name=config.child_name,
            target_family="lfm2",
            target_modalities=("text",),
            exact_unique_parameters=config.exact_target_unique_parameters,
            tokenizer_fingerprint=tokenizer_fingerprint,
            adapter_id=target_adapter.adapter_id,
            fold_method_id=(
                HYBRID_MODEL_METHOD
                if config.hybrid_model_fold
                else (
                    PRODUCER_FIRST_MODEL_METHOD
                    if config.producer_first_model_fold
                    else None
                )
            ),
            blank_seed=config.blank_seed,
            configuration=tuple(request_configuration),
        )
    return ProductionCatalogWiring(
        system=system,
        target_request=request,
        target_adapter=target_adapter,
        stack_adapter=stack_adapter,
        budget_plan=budget_plan,
    )


def diagnose_failure(error: BaseException) -> dict[str, Any]:
    """Return the exact fail-closed cause and the next repair surface."""

    chain = []
    current: BaseException | None = error
    while current is not None and len(chain) < 8:
        chain.append(
            {
                "type": type(current).__name__,
                "message": str(current),
            }
        )
        current = current.__cause__ or current.__context__
    joined = "\n".join(row["message"] for row in chain)
    match = _PREFIX_MISMATCH.search(joined)
    if match is not None:
        return {
            "category": "cached_vs_full_prefix_logit_mismatch",
            "phase": match.group("phase"),
            "maximum_logit_error": float(match.group("error")),
            "detector": ("neural_forge.hf_lfm2_circuit_trace_cache._compare_logits"),
            "next_function_to_fix": (
                "neural_forge.hf_lfm2_circuit_trace_cache._capture_batch"
            ),
            "next_repair_scope": (
                "cached decode versus no-cache full-prefix attention mask, "
                "position IDs, and recurrent-cache construction"
            ),
            "tolerance_weakened": False,
            "fail_closed": True,
            "exception_chain": chain,
        }
    return {
        "category": "fold_or_infrastructure_failure",
        "fail_closed": True,
        "exception_chain": chain,
    }


def _sealed_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(body)
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _dense_vocabulary_margin_candidate_gate_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    return {
        "enabled": config.dense_vocabulary_margin_candidate_gate_enabled,
        "replay_mode": (
            config.dense_vocabulary_margin_candidate_replay_mode
        ),
        "minimum_sign_agreement_fraction": (
            config
            .dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction
        ),
        "minimum_top_competitor_nonnegative_fraction": (
            config
            .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction
        ),
        "minimum_all_top_k_nonnegative_fraction": (
            config
            .dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction
        ),
        "minimum_exact_top1_agreement_fraction": (
            config
            .dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction
        ),
    }


def _new_ledger_attempt_evidence(
    ledger: ExperimentLedger | None,
    initial_record_count: int,
) -> list[dict[str, Any]]:
    """Expose this invocation's decisions without confusing older records."""

    if ledger is None:
        return []
    if (
        isinstance(initial_record_count, bool)
        or not isinstance(initial_record_count, int)
        or initial_record_count < 0
        or initial_record_count > len(ledger.records)
    ):
        raise ProductionLfm2CatalogError(
            "initial_record_count is invalid for the current ledger."
        )
    return [
        {
            "record_sha256": record.record_sha256,
            "status": record.status,
            "reason_class": record.reason_class,
            "reason": record.reason,
            "method_id": record.signature.method_id,
            "candidate_sha256": record.signature.candidate_sha256,
            "metrics": dict(record.metrics),
        }
        for record in ledger.records[initial_record_count:]
    ]


def _diagnostics_document(
    evaluator: CombinedLfm2FinalEvaluator | None,
    evidence_builder: CachedProductionEvidenceBuilder | None,
) -> dict[str, Any]:
    body = {
        "schema": DIAGNOSTICS_SCHEMA,
        "behavior_gate_events": ([] if evaluator is None else evaluator.events),
        "whole_output_events": (
            [] if evaluator is None else evaluator.behavior_evaluator.events
        ),
        "evidence_builds": (
            [] if evidence_builder is None else evidence_builder.records
        ),
        "parent_traces_regenerated_after_cache": False,
        "zero_gradient": True,
        "training_steps": 0,
    }
    return {**body, "diagnostics_sha256": canonical_sha256(body)}


def run_production_catalog(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    """Run one automatic local fold and always emit a sealed compact receipt."""

    if not isinstance(config, ProductionLfm2CatalogConfig):
        raise ProductionLfm2CatalogError("config must be ProductionLfm2CatalogConfig.")
    ledger: ExperimentLedger | None = None
    evaluator: CombinedLfm2FinalEvaluator | None = None
    evidence_builder: CachedProductionEvidenceBuilder | None = None
    runtime: dict[str, Any] = {
        "parent_cache_source_mode": config.parent_cache_source_mode,
        "sealed_parent_cache_directory": (
            None
            if config.sealed_parent_cache_directory is None
            else str(config.sealed_parent_cache_directory)
        ),
        "parent_cache_opened_with_full_verification": False,
        "parent_cache_build_called": False,
        "parent_cache_build_forward_count": 0,
        "parent_trace_regeneration_forward_count": 0,
        "parent_trace_regeneration_after_cache": 0,
    }
    initial_ledger_record_count = 0
    receipt = _sealed_receipt(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "failed_closed",
            "configuration_sha256": canonical_sha256(config.identity()),
            "vocabulary_margin_candidate_gate_policy": (
                _dense_vocabulary_margin_candidate_gate_policy(config)
            ),
            "parent": runtime,
            "failure": {
                "category": "runner_interrupted_before_diagnostic",
                "fail_closed": True,
            },
            "folding_only": True,
            "training_steps": 0,
            "tolerance_weakened": False,
        }
    )
    try:
        ledger = _load_ledger(config.ledger_path)
        initial_ledger_record_count = len(ledger.records)
        for path in (
            config.cache_root,
            config.snapshot_root,
            config.publish_destination.parent,
            config.receipt_path.parent,
            config.ledger_path.parent,
        ):
            path.expanduser().absolute().mkdir(parents=True, exist_ok=True)
        if config.diagnostics_path is not None:
            config.diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

        parent_directory = config.parent_model.expanduser().resolve(strict=True)
        tokenizer_directory = config.tokenizer_source.expanduser().resolve(strict=True)
        cached_parent_outputs_path = config.cached_parent_outputs.expanduser().resolve(
            strict=True
        )
        publish_destination = config.publish_destination.expanduser().absolute()
        local_io = residual.direct_local_io_audit(parent_directory)
        tokenizer_fingerprint = fingerprint_local_tokenizer(tokenizer_directory)
        tokenizer = residual.load_local_tokenizer(tokenizer_directory)
        dtype = residual._torch_dtype(config.dtype)
        device = torch.device(config.device)
        parent = residual.load_local_lfm2_model(
            parent_directory,
            device=device,
            dtype=dtype,
        )
        parent_graph = inspect_model_circuits(
            parent,
            model_name=config.parent_name,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        parent_capture_config = Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=config.sample_last_tokens,
            max_decode_steps=config.max_decode_steps,
            max_capture_bytes=config.max_capture_bytes,
            logit_rtol=config.capture_logit_rtol,
            logit_atol=config.capture_logit_atol,
            require_top1_agreement=(config.require_capture_top1_agreement),
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
        )
        capture_config = Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=config.sample_last_tokens,
            max_decode_steps=config.max_decode_steps,
            max_capture_bytes=config.max_capture_bytes,
            logit_rtol=config.capture_logit_rtol,
            logit_atol=config.capture_logit_atol,
            require_top1_agreement=(config.require_capture_top1_agreement),
            decode_trace_mode=config.child_decode_trace_mode,
        )
        resolved_parent_cache = resolve_production_parent_cache(
            config,
            parent,
            tokenizer,
            parent_capture_config,
            input_device=device,
        )
        parent_cache = resolved_parent_cache.cache
        runtime.update(resolved_parent_cache.runtime_receipt_fields())
        runtime.update(
            {
                "parent_cache_key": parent_cache.cache_key,
                "parent_unique_parameters": (
                    parent_graph.unique_parameter_count
                ),
            }
        )
        whole_output_baseline = load_lfm2_cached_whole_output_baseline(
            cached_parent_outputs_path,
            expected_sha256=config.cached_parent_outputs_sha256,
            expected_parent_cache_key=parent_cache.cache_key,
            expected_parent_state_fingerprint=(parent_cache.parent_fingerprint),
            expected_tokenizer_fingerprint=tokenizer_fingerprint,
            expected_generation_limit=(config.whole_output_max_new_tokens),
            fragment_minimum_characters=(
                config.whole_output_fragment_minimum_characters
            ),
        )
        parent_rows = residual.extract_trace_rows(
            parent_cache,
            batch_splits=resolved_parent_cache.calibration_splits,
        )
        calibration_batches = resolved_parent_cache.calibration_batches
        partition = residual.deterministic_path_partition(
            parent_rows.paths,
            path_splits=parent_rows.path_splits,
            max_selection_rows=config.max_selection_rows,
            max_fit_rows=config.max_fit_rows,
            max_validation_rows=config.max_validation_rows,
            validation_fraction=config.validation_fraction,
        )
        payload = ProductionParentCachePayload(
            cache=parent_cache,
            rows=parent_rows,
            calibration_batches=calibration_batches,
            calibration_splits=resolved_parent_cache.calibration_splits,
            partition=partition,
        )
        payload.verify()
        gate_config = residual.TrajectoryGateConfig(
            min_teacher_forced_top1=config.min_teacher_top1,
            min_free_run_token_match=config.min_free_run_match,
            min_mean_residual_cosine=config.min_residual_cosine,
            logit_rtol=config.capture_logit_rtol,
            logit_atol=config.capture_logit_atol,
        )
        evidence_builder = CachedProductionEvidenceBuilder(
            cache_root=config.evidence_cache_root,
            capture_config=capture_config,
        )
        trajectory_evaluator = CachedTrajectoryEvaluator(
            payload=payload,
            capture_config=capture_config,
            gate_config=gate_config,
            final_cache_root=config.final_cache_root,
            reload_cache_root=config.reload_cache_root,
            ordered_source_spans=(
                config.ordered_source_spans
                if config.hybrid_model_fold
                else None
            ),
            target_layer_types=(
                config.target_layer_types
                if config.hybrid_model_fold
                else None
            ),
        )
        behavior_evaluator = Lfm2WholeOutputBehaviorEvaluator(
            baseline=whole_output_baseline,
            tokenizer=tokenizer,
            maximum_prompt_tokens=(config.whole_output_maximum_prompt_tokens),
            fragment_minimum_characters=(
                config.whole_output_fragment_minimum_characters
            ),
            maximum_new_empty_responses=(
                config.whole_output_maximum_new_empty_responses
            ),
            maximum_new_fragment_responses=(
                config.whole_output_maximum_new_fragment_responses
            ),
            minimum_parent_nonempty_retention=(
                config.whole_output_minimum_parent_nonempty_retention
            ),
            minimum_task_score_retention=(
                config.whole_output_minimum_task_score_retention
            ),
            maximum_parent_task_score_regressions=(
                config.whole_output_maximum_parent_task_score_regressions
            ),
            maximum_new_generation_limit_saturations=(
                config.whole_output_maximum_new_generation_limit_saturations
            ),
            maximum_repeated_ngram_share_regressions=(
                config.whole_output_maximum_repeated_ngram_share_regressions
            ),
            maximum_repeated_bigram_share_delta=(
                config.whole_output_maximum_repeated_bigram_share_delta
            ),
        )
        evaluator = CombinedLfm2FinalEvaluator(
            trajectory_evaluator=trajectory_evaluator,
            behavior_evaluator=behavior_evaluator,
        )
        artifact_io = DirectLocalArtifactIO(
            tokenizer_directory=tokenizer_directory,
            tokenizer_fingerprint=tokenizer_fingerprint,
            device=config.device,
            dtype=dtype,
        )
        wiring = build_production_catalog_wiring(
            config,
            parent_cache_payload=payload,
            parent_graph=parent_graph,
            tokenizer_fingerprint=tokenizer_fingerprint,
            evidence_builder=evidence_builder,
            final_evaluator=evaluator,
            model_saver=artifact_io.save,
            model_loader=artifact_io.load,
            fresh_reload_evaluator=evaluator.evaluate_fresh_reload,
            ledger=ledger,
            device=device,
            dtype=dtype,
        )
        runtime.update(
            {
                "local_io": local_io,
                "tokenizer_fingerprint": tokenizer_fingerprint,
                "calibration_sha256": (
                    resolved_parent_cache.calibration_source_sha256
                ),
                "calibration_batches": len(calibration_batches),
                "cached_parent_outputs_sha256": (whole_output_baseline.source_sha256),
                "whole_output_baseline_fingerprint": (
                    whole_output_baseline.fingerprint
                ),
                "whole_output_case_count": len(whole_output_baseline.cases),
                "whole_output_envelope_binding": {
                    "validated": True,
                    "parent_cache_key": (
                        whole_output_baseline.parent_cache_key
                    ),
                    "parent_state_fingerprint": (
                        whole_output_baseline.parent_state_fingerprint
                    ),
                    "tokenizer_fingerprint": (
                        whole_output_baseline.tokenizer_fingerprint
                    ),
                },
                "parent_trace_rows_sha256": (parent_rows.trace_rows_sha256),
                "parent_forward_count_after_cache": 0,
            }
        )
        result = wiring.system.run(
            parent_models={config.parent_name: parent},
            target_request=wiring.target_request,
            evidence_fingerprint=parent_cache.cache_key,
            destination=publish_destination,
            parent_tokenizer_fingerprints={config.parent_name: tokenizer_fingerprint},
        )
        run = result.published.compilation.run
        artifact = result.published.artifact
        selected_budget_candidate = None
        if wiring.budget_plan is not None:
            selected_budget_candidate = wiring.budget_plan.frontier.candidates[
                wiring.budget_plan.target_certificate.selected_candidate_rank
            ]
        target_runtime_config = result.target.model.config
        source_unique_parameters = result.published.compilation.parent_graphs[
            0
        ].unique_parameter_count
        dense_candidate_gate_report = None
        for attempt in run.attempts:
            transaction_report = dict(attempt.transaction_report)
            candidate_gate = transaction_report.get(
                "vocabulary_margin_candidate_gate"
            )
            if isinstance(candidate_gate, Mapping):
                dense_candidate_gate_report = dict(candidate_gate)
                break
        gate_summaries = {
            event["phase"]: event["summary"] for event in evaluator.events
        }
        receipt = _sealed_receipt(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "complete",
                "configuration_sha256": canonical_sha256(config.identity()),
                "parent": runtime,
                "target": {
                    "family": result.target.graph.family,
                    "modalities": list(result.target.resource_profile.modalities),
                    "query_heads": int(
                        target_runtime_config.num_attention_heads
                    ),
                    "key_value_heads": int(
                        target_runtime_config.num_key_value_heads
                    ),
                    "head_dim": int(target_runtime_config.head_dim),
                    "hidden_size": int(target_runtime_config.hidden_size),
                    "intermediate_size": int(
                        target_runtime_config.intermediate_size
                    ),
                    "fold_method_id": (
                        DENSE_BUDGET_MODEL_METHOD
                        if config.compression_ratio is not None
                        else (
                            HYBRID_MODEL_METHOD
                            if config.hybrid_model_fold
                            else (
                                PRODUCER_FIRST_MODEL_METHOD
                                if config.producer_first_model_fold
                                else "lfm2_closed_loop_layer"
                            )
                        )
                    ),
                    "requested_compression_ratio": (
                        config.compression_ratio
                    ),
                    "realized_compression_ratio": (
                        source_unique_parameters
                        / result.target.resource_profile.unique_parameters
                    ),
                    "selection_policy": (
                        None
                        if wiring.budget_plan is None
                        else wiring.budget_plan.target_certificate.selection_policy
                    ),
                    "required_compiler_capabilities": (
                        None
                        if selected_budget_candidate is None
                        else list(
                            selected_budget_candidate.required_compiler_capabilities
                        )
                    ),
                    "missing_compiler_capabilities": (
                        None
                        if selected_budget_candidate is None
                        else list(
                            selected_budget_candidate.missing_compiler_capabilities
                        )
                    ),
                    "target_depth": result.target.resource_profile.layer_count,
                    "target_layer_types": (
                        None
                        if config.target_layer_types is None
                        else list(config.target_layer_types)
                    ),
                    "ordered_source_spans": (
                        None
                        if config.ordered_source_spans is None
                        else [list(span) for span in config.ordered_source_spans]
                    ),
                    "exact_unique_parameters": (
                        result.target.resource_profile.unique_parameters
                    ),
                    "parameters_removed": (
                        source_unique_parameters
                        - result.target.resource_profile.unique_parameters
                    ),
                    "adapter_id": result.target.adapter_id,
                    "hub_loader_used": result.target.adapter_report["hub_loader_used"],
                },
                "fold": {
                    "plan_fingerprint": (result.published.compilation.plan.fingerprint),
                    "run_fingerprint": run.fingerprint,
                    "accepted_action_count": len(run.accepted_action_ids),
                    "snapshot_backend": (run.whole_model_snapshot_backend),
                    "optimizer_constructions": (run.optimizer_constructions),
                    "optimizer_steps": run.optimizer_steps,
                    "backward_calls": run.backward_calls,
                    "autograd_grad_calls": run.autograd_grad_calls,
                    "parent_traces_regenerated_after_cache": False,
                    "vocabulary_margin_candidate_gate_policy": (
                        _dense_vocabulary_margin_candidate_gate_policy(config)
                    ),
                    "vocabulary_margin_candidate_gate": (
                        dense_candidate_gate_report
                    ),
                },
                "behavior": gate_summaries,
                "artifact": {
                    "path": artifact.path,
                    "certificate_sha256": (artifact.certificate_sha256),
                    "artifact_tree_sha256": (artifact.artifact_tree_sha256),
                    "child_state_fingerprint": (artifact.child_state_fingerprint),
                    "fresh_reload_accepted": (artifact.fresh_reload_decision.accepted),
                    "tokenizer_packaged": True,
                    "atomic_publish": True,
                },
                "ledger": {
                    "path": str(config.ledger_path),
                    "record_count": (None if ledger is None else len(ledger.records)),
                    "fingerprint": (None if ledger is None else ledger.fingerprint),
                    "new_attempts": _new_ledger_attempt_evidence(
                        ledger,
                        initial_ledger_record_count,
                    ),
                },
                "folding_only": True,
                "training_steps": 0,
                "parameter_gradients_written": False,
            }
        )
    except Exception as error:
        receipt = _sealed_receipt(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "failed_closed",
                "configuration_sha256": canonical_sha256(config.identity()),
                "vocabulary_margin_candidate_gate_policy": (
                    _dense_vocabulary_margin_candidate_gate_policy(config)
                ),
                "parent": runtime,
                "failure": diagnose_failure(error),
                "publication_destination": str(config.publish_destination),
                "publication_present": (config.publish_destination.exists()),
                "ledger": {
                    "path": str(config.ledger_path),
                    "record_count": (None if ledger is None else len(ledger.records)),
                    "fingerprint": (None if ledger is None else ledger.fingerprint),
                    "new_attempts": _new_ledger_attempt_evidence(
                        ledger,
                        initial_ledger_record_count,
                    ),
                },
                "folding_only": True,
                "training_steps": 0,
                "tolerance_weakened": False,
            }
        )
    finally:
        if ledger is not None and not isinstance(ledger, DurableExperimentLedger):
            _write_ledger_atomic(config.ledger_path, ledger)
        if config.diagnostics_path is not None:
            _write_atomic_json(
                config.diagnostics_path,
                _diagnostics_document(evaluator, evidence_builder),
            )
        _write_atomic_json(config.receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local-only, zero-gradient automatic LFM2 folding catalog. "
            "The default target is the accepted 229,349,120-parameter "
            "16Q/8KV, FFN-2552 child."
        )
    )
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Calibration corpus used by legacy cache build/reuse mode, or "
            "optional split labels when it exactly matches a sealed cache."
        ),
    )
    parser.add_argument(
        "--sealed-parent-cache-directory",
        type=Path,
        help=(
            "Open and fully verify this exact immutable parent cache without "
            "calling the cache builder or executing parent forwards."
        ),
    )
    parser.add_argument(
        "--cached-parent-outputs",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--cached-parent-outputs-sha256",
        required=True,
    )
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument(
        "--publish-destination",
        required=True,
        type=Path,
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--parent-name", default="parent")
    parser.add_argument("--child-name", default="lfm2-ffn-2552")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--target-query-heads",
        type=int,
        default=DEFAULT_TARGET_QUERY_HEADS,
    )
    parser.add_argument(
        "--target-key-value-heads",
        type=int,
        default=DEFAULT_TARGET_KEY_VALUE_HEADS,
    )
    parser.add_argument(
        "--target-intermediate-size",
        type=int,
        default=DEFAULT_TARGET_INTERMEDIATE_SIZE,
    )
    parser.add_argument(
        "--exact-target-parameters",
        type=int,
        default=DEFAULT_TARGET_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--compression-ratio",
        type=float,
        help=(
            "Automatically synthesize and fold the densest supported blank "
            "LFM2 child that meets this parameter-count compression ratio."
        ),
    )
    parser.add_argument(
        "--dense-maximum-candidates",
        type=int,
        default=64,
        help=(
            "Maximum number of automatically synthesized dense LFM2 topology "
            "candidates to retain in the auditable parameter frontier."
        ),
    )
    parser.add_argument(
        "--dense-residual-basis-mode",
        choices=("weight_complete", "circuit_balanced"),
        default="circuit_balanced",
        help=(
            "Residual-basis compiler used by automatic dense budget folding; "
            "circuit_balanced can also consume cached vocabulary margins."
        ),
    )
    parser.add_argument(
        "--dense-run-closed-loop",
        action="store_true",
        help=(
            "Run the existing cached child closed-loop refinement after the "
            "dense structural basis has been installed."
        ),
    )
    parser.add_argument(
        "--dense-vocabulary-margin-observability",
        action="store_true",
        help=(
            "Make the circuit-balanced basis preserve cached parent top-token "
            "decision margins while compiling the blank child."
        ),
    )
    parser.add_argument(
        "--dense-vocabulary-margin-top-k",
        type=int,
        default=8,
        help="Cached parent vocabulary candidates retained per trace row.",
    )
    parser.add_argument(
        "--dense-vocabulary-margin-floor-fraction",
        type=float,
        default=0.05,
        help=(
            "Minimum retained fraction of each cached parent top-token margin "
            "used by the circuit-balanced residual basis."
        ),
    )
    parser.add_argument(
        "--dense-vocabulary-margin-candidate-gate",
        action="store_true",
        help=(
            "Fail closed before publication unless one child-only replay on "
            "the sealed held-out cache preserves the configured vocabulary "
            "decision margins."
        ),
    )
    parser.add_argument(
        "--dense-vocabulary-margin-candidate-replay-mode",
        choices=(
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        ),
        default=CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        help=(
            "Authenticated child-only replay contract used by the candidate "
            "margin gate."
        ),
    )
    parser.add_argument(
        "--dense-vocabulary-margin-candidate-minimum-sign-agreement",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        (
            "--dense-vocabulary-margin-candidate-minimum-"
            "top-competitor-nonnegative"
        ),
        type=float,
        default=0.9,
    )
    parser.add_argument(
        (
            "--dense-vocabulary-margin-candidate-minimum-"
            "all-top-k-nonnegative"
        ),
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dense-vocabulary-margin-candidate-minimum-exact-top1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--hybrid-model-fold",
        action="store_true",
        help=(
            "Opt into one standalone depth+attention+FFN model fold using "
            "the explicit target layer topology and ordered source spans."
        ),
    )
    parser.add_argument(
        "--producer-first-model-fold",
        action="store_true",
        help=(
            "Use one full-depth model transaction that installs complete "
            "parent medoid producers before capturing and sequentially "
            "refining the live child."
        ),
    )
    parser.add_argument(
        "--target-layer-types",
        help=(
            "Comma-separated ordered child layer types (conv or "
            "full_attention); required with --hybrid-model-fold."
        ),
    )
    parser.add_argument(
        "--ordered-source-spans",
        help=(
            "Semicolon-separated target-ordered parent spans, for example "
            "'0,1;2;3;4'; required with --hybrid-model-fold."
        ),
    )
    parser.add_argument("--blank-seed", type=int, default=17)
    parser.add_argument("--initial-mlp-step", type=int, default=1)
    parser.add_argument("--maximum-mlp-step", type=int, default=64)
    parser.add_argument("--sample-last-tokens", type=int, default=8)
    parser.add_argument(
        "--max-decode-steps",
        type=int,
        default=DEFAULT_CAPTURE_DECODE_STEPS,
    )
    parser.add_argument(
        "--max-capture-bytes",
        type=int,
        default=DEFAULT_CAPTURE_BYTES,
    )
    parser.add_argument(
        "--capture-logit-rtol",
        type=float,
        default=DEFAULT_CAPTURE_LOGIT_RTOL,
    )
    parser.add_argument(
        "--capture-logit-atol",
        type=float,
        default=DEFAULT_CAPTURE_LOGIT_ATOL,
    )
    parser.add_argument(
        "--require-capture-top1-agreement",
        action="store_true",
    )
    parser.add_argument(
        "--child-decode-trace-mode",
        choices=(
            CACHED_VERIFIED_TRACE_MODE,
            FULL_PREFIX_CANONICAL_TRACE_MODE,
        ),
        default=FULL_PREFIX_CANONICAL_TRACE_MODE,
        help=(
            "Use full-prefix-only child traces by default so an architecture-"
            "mutated child's cache implementation cannot contaminate folding "
            "evidence. Parent traces remain on the reusable cached-verified "
            "contract."
        ),
    )
    parser.add_argument("--max-selection-rows", type=int, default=256)
    parser.add_argument("--max-fit-rows", type=int, default=512)
    parser.add_argument("--max-validation-rows", type=int, default=128)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--endpoint-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--max-endpoint-relative-rmse",
        type=float,
        default=1e6,
    )
    parser.add_argument(
        "--min-endpoint-cosine-similarity",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--endpoint-max-prior-delta-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument("--min-teacher-top1", type=float, default=0.05)
    parser.add_argument("--min-free-run-match", type=float, default=0.05)
    parser.add_argument("--min-residual-cosine", type=float, default=0.0)
    parser.add_argument(
        "--whole-output-max-new-tokens",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--whole-output-maximum-prompt-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--whole-output-fragment-minimum-characters",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--whole-output-maximum-new-empty-responses",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whole-output-maximum-new-fragment-responses",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whole-output-minimum-parent-nonempty-retention",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--whole-output-minimum-task-score-retention",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--whole-output-maximum-parent-task-score-regressions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whole-output-maximum-new-generation-limit-saturations",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-ngram-share-regressions",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-bigram-share-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--snapshot-chunk-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ProductionLfm2CatalogConfig:
    return ProductionLfm2CatalogConfig(
        parent_model=args.parent_model,
        tokenizer_directory=args.tokenizer_directory,
        calibration=args.calibration,
        sealed_parent_cache_directory=args.sealed_parent_cache_directory,
        cached_parent_outputs=args.cached_parent_outputs,
        cached_parent_outputs_sha256=(args.cached_parent_outputs_sha256),
        cache_root=args.cache_root,
        snapshot_root=args.snapshot_root,
        publish_destination=args.publish_destination,
        ledger_path=args.ledger,
        receipt_path=args.receipt,
        diagnostics_path=args.diagnostics,
        parent_name=args.parent_name,
        child_name=args.child_name,
        device=args.device,
        dtype=args.dtype,
        target_query_heads=args.target_query_heads,
        target_key_value_heads=args.target_key_value_heads,
        target_intermediate_size=args.target_intermediate_size,
        exact_target_unique_parameters=args.exact_target_parameters,
        compression_ratio=args.compression_ratio,
        dense_maximum_candidates=args.dense_maximum_candidates,
        dense_residual_basis_mode=args.dense_residual_basis_mode,
        dense_run_closed_loop=args.dense_run_closed_loop,
        dense_vocabulary_margin_observability=(
            args.dense_vocabulary_margin_observability
        ),
        dense_vocabulary_margin_top_k=(
            args.dense_vocabulary_margin_top_k
        ),
        dense_vocabulary_margin_floor_fraction=(
            args.dense_vocabulary_margin_floor_fraction
        ),
        dense_vocabulary_margin_candidate_gate_enabled=(
            args.dense_vocabulary_margin_candidate_gate
        ),
        dense_vocabulary_margin_candidate_replay_mode=(
            args.dense_vocabulary_margin_candidate_replay_mode
        ),
        dense_vocabulary_margin_candidate_minimum_sign_agreement_fraction=(
            args.dense_vocabulary_margin_candidate_minimum_sign_agreement
        ),
        dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative_fraction=(
            args
            .dense_vocabulary_margin_candidate_minimum_top_competitor_nonnegative
        ),
        dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative_fraction=(
            args.dense_vocabulary_margin_candidate_minimum_all_top_k_nonnegative
        ),
        dense_vocabulary_margin_candidate_minimum_exact_top1_agreement_fraction=(
            args.dense_vocabulary_margin_candidate_minimum_exact_top1
        ),
        hybrid_model_fold=args.hybrid_model_fold,
        producer_first_model_fold=args.producer_first_model_fold,
        target_layer_types=(
            None
            if args.target_layer_types is None
            else decode_lfm2_layer_types(args.target_layer_types)
        ),
        ordered_source_spans=(
            None
            if args.ordered_source_spans is None
            else decode_lfm2_source_spans(args.ordered_source_spans)
        ),
        blank_seed=args.blank_seed,
        initial_mlp_step=args.initial_mlp_step,
        maximum_mlp_step=args.maximum_mlp_step,
        sample_last_tokens=args.sample_last_tokens,
        max_decode_steps=args.max_decode_steps,
        max_capture_bytes=args.max_capture_bytes,
        capture_logit_rtol=args.capture_logit_rtol,
        capture_logit_atol=args.capture_logit_atol,
        require_capture_top1_agreement=(args.require_capture_top1_agreement),
        child_decode_trace_mode=args.child_decode_trace_mode,
        max_selection_rows=args.max_selection_rows,
        max_fit_rows=args.max_fit_rows,
        max_validation_rows=args.max_validation_rows,
        validation_fraction=args.validation_fraction,
        endpoint_ridge=args.endpoint_ridge,
        max_endpoint_relative_rmse=args.max_endpoint_relative_rmse,
        min_endpoint_cosine_similarity=(args.min_endpoint_cosine_similarity),
        endpoint_max_prior_delta_ratio=(args.endpoint_max_prior_delta_ratio),
        min_teacher_top1=args.min_teacher_top1,
        min_free_run_match=args.min_free_run_match,
        min_residual_cosine=args.min_residual_cosine,
        whole_output_max_new_tokens=args.whole_output_max_new_tokens,
        whole_output_maximum_prompt_tokens=(args.whole_output_maximum_prompt_tokens),
        whole_output_fragment_minimum_characters=(
            args.whole_output_fragment_minimum_characters
        ),
        whole_output_maximum_new_empty_responses=(
            args.whole_output_maximum_new_empty_responses
        ),
        whole_output_maximum_new_fragment_responses=(
            args.whole_output_maximum_new_fragment_responses
        ),
        whole_output_minimum_parent_nonempty_retention=(
            args.whole_output_minimum_parent_nonempty_retention
        ),
        whole_output_minimum_task_score_retention=(
            args.whole_output_minimum_task_score_retention
        ),
        whole_output_maximum_parent_task_score_regressions=(
            args.whole_output_maximum_parent_task_score_regressions
        ),
        whole_output_maximum_new_generation_limit_saturations=(
            args.whole_output_maximum_new_generation_limit_saturations
        ),
        whole_output_maximum_repeated_ngram_share_regressions=(
            args.whole_output_maximum_repeated_ngram_share_regressions
        ),
        whole_output_maximum_repeated_bigram_share_delta=(
            args.whole_output_maximum_repeated_bigram_share_delta
        ),
        snapshot_chunk_bytes=args.snapshot_chunk_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    receipt = run_production_catalog(config)
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
