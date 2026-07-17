#!/usr/bin/env python3
"""Run a large, zero-training partial-neuron-bank fold on a local LFM2 child.

The source is an already-compacted *child*, not the original parent.  The
original parent's sealed circuit cache supplies only immutable token batches
and parent trajectory references; this runner never loads or executes the
parent model.  It first builds or reuses a second sealed trace cache by running
the live source child over those exact cached batches.  Post-SwiGLU rows from
that child cache are then sealed together with hashes of the exact child
weights before partial coupled-neuron sharing is planned.

The default is intentionally a real, aggressive target:

* original parent: 229,693,184 unique learned scalars;
* incumbent child: 206,726,912 unique learned scalars;
* 7,477 coupled-unit pairs x 3,072 scalars = 22,969,344 extra savings;
* candidate: 183,757,568 unique learned scalars, or about 20% below the
  original parent.

Only the indexed partial-bank candidate executes after installation.  It must
pass trajectories relative to both the original parent and the incumbent
child, plus the sealed 28-case whole-output gate.  A custom standalone
serializer is deliberately deferred until behavior passes, so every candidate
is rolled back exactly after evaluation and no checkpoint is published.
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
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
    from experiments import (
        run_lfm2_residual_parameter_sharing_fold as shared_runner,
    )
except ImportError:  # Direct execution from the experiments directory.
    import run_lfm2_residual_attention_fold as fold_runtime
    import run_lfm2_residual_parameter_sharing_fold as shared_runner

from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    build_lfm2_circuit_trace_cache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_partial_neuron_bank import (
    InstalledLfm2PartialNeuronBank,
    Lfm2PartialNeuronBankMLP,
    Lfm2PartialNeuronBankPlan,
    install_lfm2_partial_neuron_bank_plan,
    partial_neuron_bank_pair_savings,
    plan_lfm2_partial_neuron_bank,
    seal_lfm2_partial_neuron_bank_evidence,
)


RUN_SCHEMA = "neural-forge-lfm2-partial-neuron-bank-fold-run-v1"
PUBLICATION_BLOCK_STATUS = "behavior_passed_checkpoint_serializer_deferred"
EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_SOURCE_CHILD_UNIQUE_PARAMETERS = 206_726_912
DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS = 183_757_568
EXPECTED_LAYER_COUNT = 14
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_SOURCE_INTERMEDIATE_SIZE = 2026
EXPECTED_WHOLE_OUTPUT_CASE_COUNT = 28
DEFAULT_MAX_EVIDENCE_ROWS = 256
DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION = 0.1499
DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE = 0.80
DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE = 0.95
DEFAULT_MINIMUM_TRAJECTORY_SCORE = 0.80
DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION = 0.90
DEFAULT_MAXIMUM_CASE_REGRESSIONS = 2


class Lfm2PartialNeuronBankRunnerError(RuntimeError):
    """A local input, cache binding, gate, or rollback failed closed."""


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
        raise Lfm2PartialNeuronBankRunnerError(
            "Run evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2PartialNeuronBankRunnerError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2PartialNeuronBankRunnerError(
            f"{name} must be a non-negative integer."
        )
    return value


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool):
        raise Lfm2PartialNeuronBankRunnerError(f"{name} must be a finite float.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Lfm2PartialNeuronBankRunnerError(
            f"{name} must be a finite float."
        ) from exc
    if (
        not math.isfinite(result)
        or (minimum is not None and result < minimum)
        or (maximum is not None and result > maximum)
    ):
        raise Lfm2PartialNeuronBankRunnerError(f"{name} is outside its valid range.")
    return result


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2PartialNeuronBankRunnerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _local_directory(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2PartialNeuronBankRunnerError(
            f"{name} must be an explicit local directory, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2PartialNeuronBankRunnerError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _local_file(value: str | os.PathLike[str], name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2PartialNeuronBankRunnerError(
            f"{name} must be an explicit local file, not a URL."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2PartialNeuronBankRunnerError(f"{name} is missing or unsafe: {path}.")
    return path.resolve(strict=True)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _default_layer_pairs(layer_count: int) -> tuple[tuple[int, int], ...]:
    if layer_count < 2 or layer_count % 2:
        raise Lfm2PartialNeuronBankRunnerError(
            "The balanced default requires a positive even decoder depth."
        )
    return tuple((index, index + 1) for index in range(0, layer_count, 2))


def resolve_target_accounting(
    *,
    original_parent_unique_parameters: int,
    source_child_unique_parameters: int,
    hidden_size: int,
    target_total_unique_parameters: int | None,
    additional_parameters_saved: int | None,
) -> dict[str, Any]:
    """Resolve one exact coupled-unit target, including budget rounding."""

    parent = _positive_int(
        original_parent_unique_parameters,
        "original_parent_unique_parameters",
    )
    source = _positive_int(
        source_child_unique_parameters,
        "source_child_unique_parameters",
    )
    hidden = _positive_int(hidden_size, "hidden_size")
    if source >= parent:
        raise Lfm2PartialNeuronBankRunnerError(
            "The source child must already be smaller than the original parent."
        )
    if target_total_unique_parameters is None and additional_parameters_saved is None:
        target_total_unique_parameters = DEFAULT_TARGET_TOTAL_UNIQUE_PARAMETERS
    if (target_total_unique_parameters is None) == (
        additional_parameters_saved is None
    ):
        raise Lfm2PartialNeuronBankRunnerError(
            "Specify exactly one total unique-parameter target or additional savings."
        )
    unit_storage = partial_neuron_bank_pair_savings(hidden, 1)
    requested_budget: int | None
    if target_total_unique_parameters is not None:
        requested_budget = _positive_int(
            target_total_unique_parameters,
            "target_total_unique_parameters",
        )
        if requested_budget >= source:
            raise Lfm2PartialNeuronBankRunnerError(
                "The total target must be below the incumbent child."
            )
        required = source - requested_budget
        pair_count = (required + unit_storage - 1) // unit_storage
        exact_additional_savings = pair_count * unit_storage
    else:
        requested_budget = None
        exact_additional_savings = _positive_int(
            additional_parameters_saved,
            "additional_parameters_saved",
        )
        if exact_additional_savings % unit_storage:
            raise Lfm2PartialNeuronBankRunnerError(
                "Additional savings must be an exact multiple of one coupled unit."
            )
        pair_count = exact_additional_savings // unit_storage
    actual_total = source - exact_additional_savings
    if actual_total <= 0:
        raise Lfm2PartialNeuronBankRunnerError(
            "The requested bank target consumes the entire source child."
        )
    total_removed = parent - actual_total
    if total_removed <= 0:
        raise Lfm2PartialNeuronBankRunnerError(
            "The resolved candidate is not smaller than the original parent."
        )
    return {
        "original_parent_unique_parameters": parent,
        "source_child_unique_parameters": source,
        "requested_total_unique_parameter_budget": requested_budget,
        "unit_pair_storage_parameters": unit_storage,
        "unit_pair_count": pair_count,
        "additional_parameters_saved": exact_additional_savings,
        "actual_total_unique_parameters": actual_total,
        "total_parameters_removed_from_original_parent": total_removed,
        "additional_compression_fraction_of_incumbent": (
            exact_additional_savings / source
        ),
        "total_compression_fraction_of_original_parent": (total_removed / parent),
        "budget_rounding_parameters": (
            0 if requested_budget is None else requested_budget - actual_total
        ),
    }


@dataclass(frozen=True, slots=True)
class PartialNeuronBankRunConfig:
    """Authenticated local inputs and one meaningful physical target."""

    source_child_model: Path
    parent_cache: Path
    cached_parent_outputs: Path
    cached_parent_outputs_sha256: str
    work_root: Path
    checkpoint_destination: Path
    report_path: Path
    tokenizer_directory: Path | None = None
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    target_total_unique_parameters: int | None = None
    additional_parameters_saved: int | None = None
    layer_pairs: tuple[tuple[int, int], ...] | None = None
    max_shared_units_per_layer: int | None = None
    max_evidence_rows: int = DEFAULT_MAX_EVIDENCE_ROWS
    similarity_weights: tuple[float, float] = (0.5, 0.5)
    exact_assignment_limit: int = 256
    correlation_chunk_size: int = 256
    variance_floor: float = 0.0
    expected_original_parent_unique_parameters: int = (
        EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
    )
    expected_source_child_unique_parameters: int = (
        EXPECTED_SOURCE_CHILD_UNIQUE_PARAMETERS
    )
    expected_layer_count: int = EXPECTED_LAYER_COUNT
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE
    expected_source_intermediate_size: int = EXPECTED_SOURCE_INTERMEDIATE_SIZE
    expected_whole_output_case_count: int = EXPECTED_WHOLE_OUTPUT_CASE_COUNT
    minimum_total_compression_fraction: float = (
        DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION
    )
    minimum_parent_residual_cosine: float = DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE
    minimum_incumbent_residual_cosine: float = DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE
    minimum_parent_trajectory_score: float = DEFAULT_MINIMUM_TRAJECTORY_SCORE
    minimum_incumbent_trajectory_score: float = DEFAULT_MINIMUM_TRAJECTORY_SCORE
    whole_output_max_new_tokens: int = 64
    whole_output_maximum_prompt_tokens: int = 512
    whole_output_fragment_minimum_characters: int = 12
    whole_output_maximum_new_empty_responses: int = 2
    whole_output_maximum_new_fragment_responses: int = 2
    whole_output_minimum_parent_nonempty_retention: float = (
        DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION
    )
    whole_output_minimum_task_score_retention: float = (
        DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION
    )
    whole_output_maximum_parent_task_score_regressions: int = (
        DEFAULT_MAXIMUM_CASE_REGRESSIONS
    )
    whole_output_maximum_new_generation_limit_saturations: int = 2
    whole_output_maximum_repeated_ngram_share_regressions: int = 2
    whole_output_maximum_repeated_bigram_share_delta: float = 0.10

    def __post_init__(self) -> None:
        source = _local_directory(
            self.source_child_model,
            "source_child_model",
        )
        parent_cache = _local_directory(self.parent_cache, "parent_cache")
        outputs = _local_file(
            self.cached_parent_outputs,
            "cached_parent_outputs",
        )
        tokenizer = (
            source
            if self.tokenizer_directory is None
            else _local_directory(
                self.tokenizer_directory,
                "tokenizer_directory",
            )
        )
        object.__setattr__(self, "source_child_model", source)
        object.__setattr__(self, "parent_cache", parent_cache)
        object.__setattr__(self, "cached_parent_outputs", outputs)
        object.__setattr__(self, "tokenizer_directory", tokenizer)
        for name in ("work_root", "checkpoint_destination", "report_path"):
            path = Path(getattr(self, name)).expanduser().absolute()
            object.__setattr__(self, name, path)
        for protected in (source, parent_cache, tokenizer):
            for output in (
                self.work_root,
                self.checkpoint_destination,
                self.report_path,
            ):
                if _is_within(output, protected):
                    raise Lfm2PartialNeuronBankRunnerError(
                        "Work, checkpoint, and report paths must stay outside "
                        "source model/cache directories."
                    )
        if self.checkpoint_destination == self.report_path:
            raise Lfm2PartialNeuronBankRunnerError(
                "checkpoint_destination and report_path must differ."
            )
        if self.checkpoint_destination.exists():
            raise Lfm2PartialNeuronBankRunnerError(
                "checkpoint_destination already exists; refusing to replace it."
            )
        if self.report_path.exists():
            raise Lfm2PartialNeuronBankRunnerError(
                "report_path already exists; refusing to replace it."
            )
        for name in (
            "max_evidence_rows",
            "correlation_chunk_size",
            "expected_original_parent_unique_parameters",
            "expected_source_child_unique_parameters",
            "expected_layer_count",
            "expected_hidden_size",
            "expected_source_intermediate_size",
            "expected_whole_output_case_count",
            "whole_output_max_new_tokens",
            "whole_output_maximum_prompt_tokens",
            "whole_output_fragment_minimum_characters",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        for name in (
            "exact_assignment_limit",
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
        if self.exact_assignment_limit > 512:
            raise Lfm2PartialNeuronBankRunnerError(
                "exact_assignment_limit cannot exceed 512."
            )
        target = self.target_accounting
        if target["total_compression_fraction_of_original_parent"] < _finite_float(
            self.minimum_total_compression_fraction,
            "minimum_total_compression_fraction",
            minimum=0.0,
            maximum=1.0,
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "The requested target is a tiny experiment; this runner requires "
                "roughly 15% or greater total compression from the original parent."
            )
        object.__setattr__(
            self,
            "minimum_total_compression_fraction",
            float(self.minimum_total_compression_fraction),
        )
        pairs = (
            _default_layer_pairs(self.expected_layer_count)
            if self.layer_pairs is None
            else self._validated_layer_pairs(self.layer_pairs)
        )
        object.__setattr__(self, "layer_pairs", pairs)
        maximum = self.max_shared_units_per_layer
        if maximum is None:
            maximum = math.ceil(
                2 * target["unit_pair_count"] / self.expected_layer_count
            )
        maximum = _positive_int(maximum, "max_shared_units_per_layer")
        if maximum > self.expected_source_intermediate_size:
            raise Lfm2PartialNeuronBankRunnerError(
                "max_shared_units_per_layer exceeds source FFN width."
            )
        if len(pairs) * maximum < target["unit_pair_count"]:
            raise Lfm2PartialNeuronBankRunnerError(
                "Layer-pair capacity cannot meet the exact aggressive target."
            )
        object.__setattr__(self, "max_shared_units_per_layer", maximum)
        weights = tuple(float(value) for value in self.similarity_weights)
        if (
            len(weights) != 2
            or any(not math.isfinite(value) or value < 0.0 for value in weights)
            or sum(weights) <= 0.0
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "similarity_weights must be two finite non-negative values "
                "with a positive total."
            )
        object.__setattr__(self, "similarity_weights", weights)
        for name in (
            "variance_floor",
            "minimum_parent_residual_cosine",
            "minimum_incumbent_residual_cosine",
            "minimum_parent_trajectory_score",
            "minimum_incumbent_trajectory_score",
            "whole_output_minimum_parent_nonempty_retention",
            "whole_output_minimum_task_score_retention",
            "whole_output_maximum_repeated_bigram_share_delta",
        ):
            minimum = -1.0 if "residual_cosine" in name else 0.0
            maximum_float = None if name == "variance_floor" else 1.0
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name,
                    minimum=minimum,
                    maximum=maximum_float,
                ),
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise Lfm2PartialNeuronBankRunnerError("device must be nonempty.")
        if not isinstance(self.dtype, str) or not self.dtype.strip():
            raise Lfm2PartialNeuronBankRunnerError("dtype must be nonempty.")
        fold_runtime._torch_dtype(self.dtype)
        object.__setattr__(
            self,
            "cached_parent_outputs_sha256",
            _digest(
                self.cached_parent_outputs_sha256,
                "cached_parent_outputs_sha256",
            ),
        )

    def _validated_layer_pairs(
        self,
        values: Sequence[Sequence[int]],
    ) -> tuple[tuple[int, int], ...]:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise Lfm2PartialNeuronBankRunnerError(
                "layer_pairs must be a sequence of pairs."
            )
        normalized: list[tuple[int, int]] = []
        used_layers: set[int] = set()
        for value in values:
            if (
                isinstance(value, (str, bytes))
                or not isinstance(value, Sequence)
                or len(value) != 2
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in value
                )
            ):
                raise Lfm2PartialNeuronBankRunnerError(
                    "Every layer pair must contain two integer indices."
                )
            left, right = sorted((int(value[0]), int(value[1])))
            if (
                left == right
                or left < 0
                or right >= self.expected_layer_count
                or left in used_layers
                or right in used_layers
            ):
                raise Lfm2PartialNeuronBankRunnerError(
                    "Layer pairs must be disjoint, distinct, and in range."
                )
            normalized.append((left, right))
            used_layers.update((left, right))
        if not normalized:
            raise Lfm2PartialNeuronBankRunnerError(
                "At least one layer pair is required."
            )
        return tuple(sorted(normalized))

    @property
    def target_accounting(self) -> dict[str, Any]:
        return resolve_target_accounting(
            original_parent_unique_parameters=(
                self.expected_original_parent_unique_parameters
            ),
            source_child_unique_parameters=(
                self.expected_source_child_unique_parameters
            ),
            hidden_size=self.expected_hidden_size,
            target_total_unique_parameters=(self.target_total_unique_parameters),
            additional_parameters_saved=self.additional_parameters_saved,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_child_model": str(self.source_child_model),
            "parent_cache": str(self.parent_cache),
            "cached_parent_outputs": str(self.cached_parent_outputs),
            "cached_parent_outputs_sha256": (self.cached_parent_outputs_sha256),
            "tokenizer_directory": str(self.tokenizer_directory),
            "work_root": str(self.work_root),
            "checkpoint_destination": str(self.checkpoint_destination),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "target": self.target_accounting,
            "layer_pairs": [list(value) for value in self.layer_pairs],
            "max_shared_units_per_layer": (self.max_shared_units_per_layer),
            "max_evidence_rows": self.max_evidence_rows,
            "similarity_weights": list(self.similarity_weights),
            "exact_assignment_limit": self.exact_assignment_limit,
            "correlation_chunk_size": self.correlation_chunk_size,
            "variance_floor": self.variance_floor,
            "expected_layer_count": self.expected_layer_count,
            "expected_hidden_size": self.expected_hidden_size,
            "expected_source_intermediate_size": (
                self.expected_source_intermediate_size
            ),
            "expected_whole_output_case_count": (self.expected_whole_output_case_count),
            "trajectory_gates": {
                "minimum_parent_residual_cosine": (self.minimum_parent_residual_cosine),
                "minimum_incumbent_residual_cosine": (
                    self.minimum_incumbent_residual_cosine
                ),
                "minimum_parent_score": (self.minimum_parent_trajectory_score),
                "minimum_incumbent_score": (self.minimum_incumbent_trajectory_score),
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
                "exact_parent_text_required": False,
                "exact_token_match_required": False,
            },
            "offline": True,
            "original_parent_model_loaded": False,
            "new_parent_forwards_allowed": False,
            "training_allowed": False,
            "standalone_checkpoint_available": False,
        }


def authenticate_source_child_trace(
    *,
    source_child_identity_sha256: str,
    source_child_cache: Lfm2CircuitTraceCache,
    parent_cache: Lfm2CircuitTraceCache,
) -> dict[str, Any]:
    """Bind the separate child trace to the exact child and parent token corpus."""

    identity = _digest(
        source_child_identity_sha256,
        "source_child_identity_sha256",
    )
    if source_child_cache.parent_fingerprint != identity:
        raise Lfm2PartialNeuronBankRunnerError(
            "The source-child trace cache does not belong to the exact live child."
        )
    if source_child_cache.corpus_fingerprint != parent_cache.corpus_fingerprint:
        raise Lfm2PartialNeuronBankRunnerError(
            "The source-child trace did not use the parent cache's exact token batches."
        )
    if source_child_cache.batch_count != parent_cache.batch_count:
        raise Lfm2PartialNeuronBankRunnerError(
            "Parent and source-child trace batch counts differ."
        )
    return {
        "exact_source_child_bound": True,
        "source_child_identity_sha256": identity,
        "source_child_cache_key": source_child_cache.cache_key,
        "parent_cache_key": parent_cache.cache_key,
        "shared_corpus_fingerprint": parent_cache.corpus_fingerprint,
        "batch_count": parent_cache.batch_count,
        "parent_model_loaded": False,
        "new_parent_forward_calls": 0,
    }


def _directory_keys(path: Path) -> set[str]:
    if not path.is_dir():
        return set()
    return {
        child.name
        for child in path.iterdir()
        if child.is_dir() and not child.is_symlink()
    }


def build_source_child_trace_cache(
    *,
    parent_cache: Lfm2CircuitTraceCache,
    source_child: torch.nn.Module,
    cache_root: Path,
    capture_config: Lfm2CircuitTraceCaptureConfig,
) -> tuple[Lfm2CircuitTraceCache, Any, dict[str, Any]]:
    """Build/reuse the sole new persistent trace cache, using child forwards."""

    batches = fold_runtime.calibration_batches_from_cache(parent_cache)
    existing = _directory_keys(cache_root)
    child_cache = build_lfm2_circuit_trace_cache(
        cache_root,
        source_child,
        batches,
        config=capture_config,
        input_device=fold_runtime._model_device(source_child),
    )
    rows = fold_runtime.extract_trace_rows(child_cache)
    reused = child_cache.cache_key in existing
    audit = {
        "cache_reused": reused,
        "source_child_forward_calls": (
            0 if reused else shared_runner._expected_capture_forwards(parent_cache)
        ),
        "parent_forward_calls": 0,
        "trace_rows_sha256": rows.trace_rows_sha256,
    }
    return child_cache, rows, audit


class _LayerResidualCapture:
    """Capture decoder outputs from the custom indexed FFN runtime."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.layers = shared_runner._model_layers(model)
        self.handles: list[Any] = []
        self.active_sample_tokens: int | None = None
        self.values: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_hook(
                    lambda module, inputs, output, i=index: self._hook(
                        i,
                        output,
                    )
                )
            )

    def _hook(self, index: int, output: Any) -> None:
        if self.active_sample_tokens is None:
            return
        value = output
        if isinstance(value, (tuple, list)) and value:
            value = value[0]
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 3
            or int(value.shape[0]) != 1
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "A decoder layer did not return [1, sequence, hidden]."
            )
        if index in self.values:
            raise Lfm2PartialNeuronBankRunnerError(
                "A decoder layer executed twice in one capture phase."
            )
        self.values[index] = (
            value[:, -self.active_sample_tokens :, :]
            .detach()
            .to(device="cpu")
            .contiguous()
        )

    def begin(self, sample_tokens: int) -> None:
        if self.active_sample_tokens is not None:
            raise Lfm2PartialNeuronBankRunnerError(
                "A layer residual capture is already active."
            )
        self.active_sample_tokens = _positive_int(
            sample_tokens,
            "sample_tokens",
        )
        self.values = {}

    def finish(self) -> tuple[torch.Tensor, ...]:
        if self.active_sample_tokens is None:
            raise Lfm2PartialNeuronBankRunnerError(
                "No layer residual capture is active."
            )
        try:
            if set(self.values) != set(range(len(self.layers))):
                raise Lfm2PartialNeuronBankRunnerError(
                    "The live candidate did not execute every decoder layer."
                )
            return tuple(self.values[index] for index in range(len(self.layers)))
        finally:
            self.active_sample_tokens = None
            self.values = {}

    def remove(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.active_sample_tokens = None
        self.values = {}


def _output_value(output: Any, name: str) -> Any:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _output_logits(output: Any) -> torch.Tensor:
    value = _output_value(output, "logits")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or not bool(torch.isfinite(value).all())
    ):
        raise Lfm2PartialNeuronBankRunnerError(
            "The live candidate returned invalid logits."
        )
    return value


def _forward(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    use_cache: bool,
    past_key_values: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": use_cache,
        "past_key_values": past_key_values,
    }
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    return model(**kwargs)


@dataclass(frozen=True, slots=True)
class LiveCandidateTrajectory:
    """One live capture reused for both parent and incumbent comparisons."""

    logits: Mapping[tuple[int, str], torch.Tensor]
    residual_rows: tuple[torch.Tensor, ...]
    free_reports: tuple[Mapping[str, Any], ...]
    forward_calls: int
    capture_sha256: str


def capture_live_candidate_trajectory(
    model: torch.nn.Module,
    *,
    token_cache: Lfm2CircuitTraceCache,
    capture_config: Lfm2CircuitTraceCaptureConfig,
) -> LiveCandidateTrajectory:
    """Capture a custom partial-bank candidate without canonical dense-MLP hooks."""

    device = fold_runtime._model_device(model)
    layer_count = len(shared_runner._model_layers(model))
    residual_pieces: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    logits: dict[tuple[int, str], torch.Tensor] = {}
    free_reports: list[dict[str, Any]] = []
    forward_calls = 0
    modes = [(module, bool(module.training)) for module in model.modules()]
    hooks = _LayerResidualCapture(model)
    try:
        model.eval()
        with torch.inference_mode():
            for batch in token_cache.iter_batches():
                input_ids = batch.input_ids.to(device=device)
                attention_mask = batch.attention_mask.to(device=device)
                forced_ids = batch.forced_decode_ids.to(device=device)
                position_value = batch.tensors.get("input.position_ids")
                position_ids = (
                    None if position_value is None else position_value.to(device=device)
                )

                hooks.begin(capture_config.sample_last_tokens)
                _forward(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    past_key_values=None,
                )
                forward_calls += 1
                phase_rows = hooks.finish()
                for layer_index, value in enumerate(phase_rows):
                    residual_pieces[layer_index].append(
                        value.reshape(-1, value.shape[-1])
                    )

                cached_prefill = _forward(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=None,
                )
                forward_calls += 1
                past = _output_value(cached_prefill, "past_key_values")
                if past is None:
                    raise Lfm2PartialNeuronBankRunnerError(
                        "The live candidate cached prefill returned no cache."
                    )
                prefill_logits = _output_logits(cached_prefill)
                logits[(batch.index, "cached_prefill")] = (
                    prefill_logits[
                        :,
                        -capture_config.sample_last_tokens :,
                        :,
                    ]
                    .detach()
                    .to(device="cpu")
                    .contiguous()
                )

                prefix_mask = attention_mask
                prefix_positions = position_ids
                for step in range(int(forced_ids.shape[1])):
                    token = forced_ids[:, step : step + 1]
                    prefix_mask = torch.cat(
                        (
                            prefix_mask,
                            torch.ones(
                                (int(prefix_mask.shape[0]), 1),
                                dtype=prefix_mask.dtype,
                                device=device,
                            ),
                        ),
                        dim=1,
                    )
                    step_positions = None
                    if prefix_positions is not None:
                        step_positions = prefix_positions[:, -1:] + 1
                        prefix_positions = torch.cat(
                            (prefix_positions, step_positions),
                            dim=1,
                        )
                    hooks.begin(1)
                    step_output = _forward(
                        model,
                        input_ids=token,
                        attention_mask=prefix_mask,
                        position_ids=step_positions,
                        use_cache=True,
                        past_key_values=past,
                    )
                    forward_calls += 1
                    step_rows = hooks.finish()
                    for layer_index, value in enumerate(step_rows):
                        residual_pieces[layer_index].append(
                            value.reshape(-1, value.shape[-1])
                        )
                    past = _output_value(step_output, "past_key_values")
                    if past is None:
                        raise Lfm2PartialNeuronBankRunnerError(
                            "The live candidate lost its decode cache."
                        )
                    logits[(batch.index, f"decode.{step:03d}")] = (
                        _output_logits(step_output)[:, -1:, :]
                        .detach()
                        .to(device="cpu")
                        .contiguous()
                    )

                reference = [
                    int(value) for value in batch.forced_decode_ids.reshape(-1).tolist()
                ]
                generated: list[int] = []
                if reference:
                    free_mask = attention_mask
                    free_positions = position_ids
                    free_prefill = _forward(
                        model,
                        input_ids=input_ids,
                        attention_mask=free_mask,
                        position_ids=free_positions,
                        use_cache=True,
                        past_key_values=None,
                    )
                    forward_calls += 1
                    free_past = _output_value(
                        free_prefill,
                        "past_key_values",
                    )
                    if free_past is None:
                        raise Lfm2PartialNeuronBankRunnerError(
                            "Free-run cached prefill returned no cache."
                        )
                    generated.append(
                        int(
                            _output_logits(free_prefill)[:, -1:, :]
                            .argmax(dim=-1)
                            .item()
                        )
                    )
                    for _step in range(1, len(reference)):
                        token = torch.tensor(
                            [[generated[-1]]],
                            dtype=input_ids.dtype,
                            device=device,
                        )
                        free_mask = torch.cat(
                            (
                                free_mask,
                                torch.ones(
                                    (int(free_mask.shape[0]), 1),
                                    dtype=free_mask.dtype,
                                    device=device,
                                ),
                            ),
                            dim=1,
                        )
                        step_positions = None
                        if free_positions is not None:
                            step_positions = free_positions[:, -1:] + 1
                            free_positions = torch.cat(
                                (free_positions, step_positions),
                                dim=1,
                            )
                        free_step = _forward(
                            model,
                            input_ids=token,
                            attention_mask=free_mask,
                            position_ids=step_positions,
                            use_cache=True,
                            past_key_values=free_past,
                        )
                        forward_calls += 1
                        free_past = _output_value(
                            free_step,
                            "past_key_values",
                        )
                        if free_past is None:
                            raise Lfm2PartialNeuronBankRunnerError(
                                "Free-run decode lost its cache."
                            )
                        generated.append(
                            int(
                                _output_logits(free_step)[:, -1:, :]
                                .argmax(dim=-1)
                                .item()
                            )
                        )
                matches = sum(
                    left == right for left, right in zip(generated, reference)
                )
                lcp = 0
                for left, right in zip(generated, reference):
                    if left != right:
                        break
                    lcp += 1
                free_reports.append(
                    {
                        "batch_index": batch.index,
                        "reference_tokens": reference,
                        "candidate_generated_tokens": generated,
                        "token_matches": matches,
                        "longest_common_prefix": lcp,
                    }
                )
    finally:
        hooks.remove()
        for module, was_training in modes:
            module.training = was_training
    materialized = tuple(
        torch.cat(values, dim=0).contiguous() for values in residual_pieces
    )
    descriptors = {
        "logits": [
            {
                "batch": key[0],
                "phase": key[1],
                "shape": list(value.shape),
                "sha256": fold_runtime._tensor_sha256(value),
            }
            for key, value in sorted(logits.items())
        ],
        "residual_rows": [
            {
                "layer": index,
                "shape": list(value.shape),
                "sha256": fold_runtime._tensor_sha256(value),
            }
            for index, value in enumerate(materialized)
        ],
        "free_reports": free_reports,
        "forward_calls": forward_calls,
    }
    return LiveCandidateTrajectory(
        logits=logits,
        residual_rows=materialized,
        free_reports=tuple(free_reports),
        forward_calls=forward_calls,
        capture_sha256=_json_sha256(descriptors),
    )


def evaluate_live_trajectory(
    *,
    reference_name: str,
    reference_cache: Lfm2CircuitTraceCache,
    reference_rows: Any,
    candidate: LiveCandidateTrajectory,
    minimum_residual_cosine: float,
    minimum_score: float,
) -> dict[str, Any]:
    """Compare one live candidate capture to one immutable trajectory."""

    reference_logits: list[torch.Tensor] = []
    candidate_logits: list[torch.Tensor] = []
    for batch in reference_cache.iter_batches():
        phases = ["cached_prefill"]
        phases.extend(
            f"decode.{step:03d}"
            for step in range(int(batch.forced_decode_ids.shape[1]))
        )
        for phase in phases:
            reference_key = (
                "cached_prefill.logits"
                if phase == "cached_prefill"
                else f"{phase}.logits"
            )
            candidate_key = (batch.index, phase)
            if (
                reference_key not in batch.tensors
                or candidate_key not in candidate.logits
            ):
                raise Lfm2PartialNeuronBankRunnerError(
                    f"Missing {reference_name} trajectory phase {phase}."
                )
            left = batch.tensors[reference_key]
            right = candidate.logits[candidate_key]
            if left.shape != right.shape:
                raise Lfm2PartialNeuronBankRunnerError(
                    f"{reference_name} and candidate logit shapes differ."
                )
            reference_logits.append(left.reshape(-1, left.shape[-1]))
            candidate_logits.append(right.reshape(-1, right.shape[-1]))
    logits = fold_runtime._logit_metrics(
        torch.cat(reference_logits, dim=0),
        torch.cat(candidate_logits, dim=0),
    )
    if len(candidate.residual_rows) != len(reference_rows.layer_kinds):
        raise Lfm2PartialNeuronBankRunnerError(
            f"{reference_name} and candidate decoder depths differ."
        )
    layer_metrics = []
    for layer_index, candidate_rows in enumerate(candidate.residual_rows):
        expected_rows = reference_rows.field(
            layer_index,
            "residual.post_layer",
        )
        if expected_rows.shape != candidate_rows.shape:
            raise Lfm2PartialNeuronBankRunnerError(
                f"{reference_name} and candidate residual rows differ."
            )
        layer_metrics.append(
            {
                "layer_index": layer_index,
                **fold_runtime._residual_metrics(
                    expected_rows,
                    candidate_rows,
                ),
            }
        )
    mean_residual_cosine = sum(value["mean_cosine"] for value in layer_metrics) / len(
        layer_metrics
    )
    mean_relative_rmse = sum(value["relative_rmse"] for value in layer_metrics) / len(
        layer_metrics
    )
    free_total = sum(len(value["reference_tokens"]) for value in candidate.free_reports)
    free_matches = sum(int(value["token_matches"]) for value in candidate.free_reports)
    free_lcp = sum(
        int(value["longest_common_prefix"]) for value in candidate.free_reports
    )
    free_match_fraction = 1.0 if free_total == 0 else free_matches / free_total
    free_lcp_fraction = 1.0 if free_total == 0 else free_lcp / free_total
    score = (
        0.35 * logits["top1_agreement"]
        + 0.25 * free_match_fraction
        + 0.15 * free_lcp_fraction
        + 0.10 * ((logits["mean_cosine"] + 1.0) / 2.0)
        + 0.15 * ((mean_residual_cosine + 1.0) / 2.0)
    )
    finite_values = (
        logits["top1_agreement"],
        logits["mean_cosine"],
        logits["kl"],
        mean_residual_cosine,
        mean_relative_rmse,
        free_match_fraction,
        free_lcp_fraction,
        score,
    )
    reasons = []
    if not all(math.isfinite(float(value)) for value in finite_values):
        reasons.append("trajectory diagnostics contain non-finite values")
    if mean_residual_cosine < minimum_residual_cosine:
        reasons.append("mean residual-path cosine is below threshold")
    if score < minimum_score:
        reasons.append("combined trajectory score is below threshold")
    report = {
        "schema": "neural-forge-lfm2-live-partial-bank-trajectory-gate-v1",
        "reference": reference_name,
        "reference_cache_key": reference_cache.cache_key,
        "reference_trace_rows_sha256": reference_rows.trace_rows_sha256,
        "candidate_capture_sha256": candidate.capture_sha256,
        "passed": not reasons,
        "reasons": reasons,
        "teacher_forced": logits,
        "free_run": {
            "token_match_fraction": free_match_fraction,
            "lcp_fraction": free_lcp_fraction,
            "reports": [dict(value) for value in candidate.free_reports],
            "candidate_generated_prefixes": True,
            "parent_forward_passes": 0,
        },
        "residual_layers": layer_metrics,
        "mean_residual_cosine": mean_residual_cosine,
        "mean_relative_rmse": mean_relative_rmse,
        "score": score,
        "thresholds": {
            "minimum_residual_cosine": minimum_residual_cosine,
            "minimum_score": minimum_score,
        },
        "exact_token_match_is_diagnostic_only": True,
    }
    return {
        "decision": {
            "accepted": not reasons,
            "reasons": reasons,
        },
        "report": report,
        "report_sha256": _json_sha256(report),
        "candidate_forward_calls": candidate.forward_calls,
        "parent_forward_calls": 0,
    }


def audit_partial_bank_installation(
    model: torch.nn.Module,
    plan: Lfm2PartialNeuronBankPlan,
    installation: InstalledLfm2PartialNeuronBank,
) -> dict[str, Any]:
    """Independently count real bank Parameters and integer buffer bytes."""

    layers = shared_runner._model_layers(model)
    planned_layers = tuple(value.layer_index for value in plan.layer_plans)
    actual_layers = tuple(
        index
        for index, layer in enumerate(layers)
        if isinstance(layer.feed_forward, Lfm2PartialNeuronBankMLP)
    )
    if actual_layers != planned_layers:
        raise Lfm2PartialNeuronBankRunnerError(
            "Installed partial-bank layer surface differs from the sealed plan."
        )
    modules = tuple(layers[index].feed_forward for index in actual_layers)
    bank_ids = {id(module.bank) for module in modules}
    if len(bank_ids) != 1:
        raise Lfm2PartialNeuronBankRunnerError(
            "Installed partial-bank layers do not share one physical bank."
        )
    bank = modules[0].bank
    bank_parameters = sum(int(parameter.numel()) for parameter in bank.parameters())
    private_parameters = sum(
        int(parameter.numel())
        for module in modules
        for name, parameter in module.named_parameters(recurse=False)
        if name in {"private_gate", "private_up", "private_down"}
    )
    index_count = sum(int(module.bank_indices.numel()) for module in modules)
    index_bytes = sum(
        int(module.bank_indices.numel() * module.bank_indices.element_size())
        for module in modules
    )
    report = installation.report
    if (
        bank_parameters != report.bank_parameter_count
        or private_parameters != report.private_parameter_count
        or index_count != report.integer_index_buffer_count
        or index_bytes != report.integer_index_buffer_bytes
        or report.compensation_parameter_count != 0
    ):
        raise Lfm2PartialNeuronBankRunnerError(
            "Physical bank or index-buffer accounting differs from installation."
        )
    return {
        "exact": True,
        "replaced_layers": list(actual_layers),
        "one_shared_bank_object": True,
        "bank_unique_learned_parameters": bank_parameters,
        "private_unique_learned_parameters": private_parameters,
        "integer_index_buffer_count": index_count,
        "integer_index_buffer_bytes": index_bytes,
        "integer_index_buffers_are_parameters": False,
        "compensation_parameter_count": 0,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradients_used": False,
    }


def candidate_accepted(
    parent_trajectory: Mapping[str, Any],
    incumbent_trajectory: Mapping[str, Any],
    whole_output: Mapping[str, Any],
) -> bool:
    """Require all three behavioral gates; token identity remains diagnostic."""

    return bool(
        parent_trajectory["decision"]["accepted"]
        and incumbent_trajectory["decision"]["accepted"]
        and whole_output["decision"]["accepted"]
    )


def _checkpoint_absence_audit(
    config: PartialNeuronBankRunConfig,
) -> dict[str, Any]:
    exists = config.checkpoint_destination.exists()
    if exists:
        raise Lfm2PartialNeuronBankRunnerError(
            "A checkpoint appeared before a custom serializer was authorized."
        )
    return {
        "requested_destination": str(config.checkpoint_destination),
        "exists": False,
        "files_written": 0,
        "python_object_serialization_used": False,
        "dense_fallback_used": False,
        "checkpoint_serializer_deferred_until_behavior_passes": True,
    }


def _authenticated_rollback(
    installation: InstalledLfm2PartialNeuronBank,
    model: torch.nn.Module,
    *,
    source_state_sha256: str,
    source_counts: Mapping[str, int],
    source_topology_sha256: str,
) -> dict[str, Any]:
    """Use the shared runner's independent state/count/topology audit."""

    try:
        result = shared_runner._authenticated_rollback(
            installation,
            model,
            parent_state_sha256=source_state_sha256,
            parent_counts=source_counts,
            parent_topology_sha256=source_topology_sha256,
        )
    except BaseException as exc:
        raise Lfm2PartialNeuronBankRunnerError(
            "Partial-bank rollback could not restore the exact incumbent child."
        ) from exc
    return {
        **result,
        "restored_role": "incumbent_source_child",
    }


def _sealed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("run_sha256", None)
    return {**body, "run_sha256": _json_sha256(body)}


def _publish_report(
    config: PartialNeuronBankRunConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    sealed = _sealed_report(report)
    try:
        shared_runner._atomic_write_new_json(config.report_path, sealed)
    except BaseException as exc:
        raise Lfm2PartialNeuronBankRunnerError(
            "Could not atomically publish the sealed run report."
        ) from exc
    return sealed


def run_experiment(
    config: PartialNeuronBankRunConfig,
) -> dict[str, Any]:
    """Execute one aggressive in-memory candidate and always preserve source."""

    if not isinstance(config, PartialNeuronBankRunConfig):
        raise TypeError("config must be PartialNeuronBankRunConfig.")
    config.work_root.mkdir(parents=True, exist_ok=True)
    config.checkpoint_destination.parent.mkdir(parents=True, exist_ok=True)
    installation: InstalledLfm2PartialNeuronBank | None = None
    model: torch.nn.Module | None = None
    source_state_sha256: str | None = None
    source_counts: dict[str, int] | None = None
    source_topology: dict[str, Any] | None = None
    report: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "status": "initializing",
        "publication_status": "not_evaluated",
        "config": config.as_dict(),
        "training": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "optimizer_constructed": False,
            "gradients_used": False,
        },
        "original_parent": {
            "model_loaded": False,
            "forward_calls": 0,
            "role": "sealed_cache_reference_only",
        },
        "checkpoint_published": False,
        "checkpoint_absence": _checkpoint_absence_audit(config),
        "promotion_policy": {
            "original_parent_trajectory_required": True,
            "incumbent_child_trajectory_required": True,
            "sealed_28_case_whole_output_required": True,
            "exact_token_match_required": False,
            "exact_generated_text_required": False,
            "standalone_checkpoint_deferred_until_behavior_passes": True,
        },
    }
    try:
        parent_cache = Lfm2CircuitTraceCache.open(
            config.parent_cache,
            verify=True,
        )
        capture_config = shared_runner.capture_config_from_parent_cache(parent_cache)
        parent_rows = fold_runtime.extract_trace_rows(parent_cache)
        dtype = fold_runtime._torch_dtype(config.dtype)
        model = fold_runtime.load_local_lfm2_model(
            config.source_child_model,
            device=config.device,
            dtype=dtype,
        )
        model.eval()
        source_counts = shared_runner.parameter_counts(model)
        source_topology = shared_runner.topology_signature(model)
        source_state_sha256 = fold_runtime.model_state_sha256(model)
        source_identity = fingerprint_lfm2_parent(model).identity_sha256
        gradients_before = shared_runner.gradient_audit(model)
        if not gradients_before["zero_gradient"]:
            raise Lfm2PartialNeuronBankRunnerError(
                "The incumbent source child already contains gradients."
            )
        if (
            source_counts["unique"] != config.expected_source_child_unique_parameters
            or source_topology["layer_count"] != config.expected_layer_count
            or any(
                row["contract"]["hidden_size"] != config.expected_hidden_size
                or row["contract"]["ffn_intermediate_size"]
                != config.expected_source_intermediate_size
                for row in source_topology["layers"]
            )
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "The local source child differs from the configured LFM2 contract."
            )
        operator_objects = tuple(
            (layer.self_attn if layer.is_attention_layer else layer.conv)
            for layer in shared_runner._model_layers(model)
        )
        source_cache, source_rows, source_cache_audit = build_source_child_trace_cache(
            parent_cache=parent_cache,
            source_child=model,
            cache_root=config.work_root / "source-child-trace",
            capture_config=capture_config,
        )
        cache_binding = authenticate_source_child_trace(
            source_child_identity_sha256=source_identity,
            source_child_cache=source_cache,
            parent_cache=parent_cache,
        )
        if parent_rows.paths != source_rows.paths:
            raise Lfm2PartialNeuronBankRunnerError(
                "Parent and incumbent trace paths are not exactly aligned."
            )
        selected_rows, selected_indices = shared_runner.select_post_swiglu_rows(
            source_rows,
            maximum_rows=config.max_evidence_rows,
        )
        tokenizer_fingerprint = shared_runner._tokenizer_fingerprint(
            config.tokenizer_directory
        )
        tokenizer = fold_runtime.load_local_tokenizer(config.tokenizer_directory)
        behavior_evaluator = shared_runner._behavior_evaluator(
            config,
            cache=parent_cache,
            tokenizer=tokenizer,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )
        evidence = seal_lfm2_partial_neuron_bank_evidence(
            model,
            selected_rows,
            cache_key=source_cache.cache_key,
            heldout_corpus_sha256=source_cache.corpus_fingerprint,
        )
        if (
            evidence.parent_identity_sha256 != source_identity
            or evidence.cache_key != source_cache.cache_key
            or evidence.heldout_corpus_sha256 != source_cache.corpus_fingerprint
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "Sealed partial-bank evidence is not bound to the exact child cache."
            )
        target = config.target_accounting
        plan = plan_lfm2_partial_neuron_bank(
            model,
            evidence,
            target_parameters_saved=target["additional_parameters_saved"],
            layer_pairs=config.layer_pairs,
            activation_weight=config.similarity_weights[0],
            coupled_weight_weight=config.similarity_weights[1],
            max_shared_units_per_layer=(config.max_shared_units_per_layer),
            exact_assignment_limit=config.exact_assignment_limit,
            correlation_chunk_size=config.correlation_chunk_size,
            variance_floor=config.variance_floor,
        )
        if (
            plan.unit_pair_count != target["unit_pair_count"]
            or plan.target_parameters_saved != target["additional_parameters_saved"]
            or plan.projected_unique_parameters_after
            != target["actual_total_unique_parameters"]
            or plan.parent_identity_sha256 != source_identity
            or plan.evidence_sha256 != evidence.evidence_sha256
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "The sealed partial-bank plan missed the exact large target."
            )
        installation = install_lfm2_partial_neuron_bank_plan(
            model,
            evidence,
            plan,
        )
        candidate_counts = shared_runner.parameter_counts(model)
        candidate_state_sha256 = fold_runtime.model_state_sha256(model)
        candidate_gradients = shared_runner.gradient_audit(model)
        physical_audit = audit_partial_bank_installation(
            model,
            plan,
            installation,
        )
        observed_operators = tuple(
            (layer.self_attn if layer.is_attention_layer else layer.conv)
            for layer in shared_runner._model_layers(model)
        )
        if (
            candidate_counts["unique"] != target["actual_total_unique_parameters"]
            or source_counts["unique"] - candidate_counts["unique"]
            != target["additional_parameters_saved"]
            or not candidate_gradients["zero_gradient"]
            or any(
                before is not after
                for before, after in zip(
                    operator_objects,
                    observed_operators,
                )
            )
        ):
            raise Lfm2PartialNeuronBankRunnerError(
                "Installed candidate failed physical count, gradient, or "
                "attention/recurrent-operator identity checks."
            )
        candidate_capture = capture_live_candidate_trajectory(
            model,
            token_cache=parent_cache,
            capture_config=capture_config,
        )
        parent_trajectory = evaluate_live_trajectory(
            reference_name="original_parent",
            reference_cache=parent_cache,
            reference_rows=parent_rows,
            candidate=candidate_capture,
            minimum_residual_cosine=(config.minimum_parent_residual_cosine),
            minimum_score=config.minimum_parent_trajectory_score,
        )
        incumbent_trajectory = evaluate_live_trajectory(
            reference_name="incumbent_source_child",
            reference_cache=source_cache,
            reference_rows=source_rows,
            candidate=candidate_capture,
            minimum_residual_cosine=(config.minimum_incumbent_residual_cosine),
            minimum_score=config.minimum_incumbent_trajectory_score,
        )
        whole_output = shared_runner._behavior_event(
            behavior_evaluator,
            model,
            phase="partial_neuron_bank_in_memory_candidate",
        )
        accepted = candidate_accepted(
            parent_trajectory,
            incumbent_trajectory,
            whole_output,
        )
        report.update(
            {
                "status": "candidate_evaluated",
                "local_io": fold_runtime.direct_local_io_audit(
                    config.source_child_model
                ),
                "parent_cache": {
                    "path": str(parent_cache.path),
                    "cache_key": parent_cache.cache_key,
                    "parent_fingerprint": (parent_cache.parent_fingerprint),
                    "corpus_fingerprint": (parent_cache.corpus_fingerprint),
                    "verified": True,
                    "parent_model_loaded": False,
                    "new_parent_forward_calls": 0,
                },
                "source_child_cache": {
                    "path": str(source_cache.path),
                    "cache_key": source_cache.cache_key,
                    "source_child_fingerprint": (source_cache.parent_fingerprint),
                    "corpus_fingerprint": (source_cache.corpus_fingerprint),
                    "verified": True,
                    **source_cache_audit,
                    "binding": cache_binding,
                },
                "activation_evidence": {
                    **evidence.as_dict(),
                    "selected_path_indices": list(selected_indices),
                    "field": "mlp.post_swiglu",
                    "source": "separate_live_source_child_trace_cache",
                    "new_parent_forward_calls": 0,
                },
                "source_child": {
                    "identity_sha256": source_identity,
                    "state_sha256": source_state_sha256,
                    "parameter_counts": source_counts,
                    "topology": source_topology,
                    "gradient_audit": gradients_before,
                },
                "plan": plan.as_dict(),
                "installation": installation.report.as_dict(),
                "candidate": {
                    "state_sha256": candidate_state_sha256,
                    "parameter_counts": candidate_counts,
                    "target_accounting": target,
                    "real_unique_parameters": candidate_counts["unique"],
                    "integer_index_buffer_bytes": (
                        installation.report.integer_index_buffer_bytes
                    ),
                    "index_buffers_counted_separately_from_parameters": True,
                    "physical_audit": physical_audit,
                    "gradient_audit": candidate_gradients,
                    "attention_and_recurrent_operator_objects_retained": True,
                    "training_steps": 0,
                    "optimizer_steps": 0,
                    "backward_calls": 0,
                },
                "evaluation": {
                    "live_capture": {
                        "capture_sha256": (candidate_capture.capture_sha256),
                        "candidate_forward_calls": (candidate_capture.forward_calls),
                        "parent_forward_calls": 0,
                        "custom_partial_bank_compatible_capture": True,
                    },
                    "original_parent_trajectory": parent_trajectory,
                    "incumbent_child_trajectory": incumbent_trajectory,
                    "cached_28_case_whole_output": whole_output,
                    "accepted": accepted,
                    "exact_token_match_is_diagnostic_only": True,
                    "exact_generated_text_is_diagnostic_only": True,
                },
                "tokenizer_fingerprint": tokenizer_fingerprint,
            }
        )
        report["rollback"] = _authenticated_rollback(
            installation,
            model,
            source_state_sha256=source_state_sha256,
            source_counts=source_counts,
            source_topology_sha256=source_topology["topology_sha256"],
        )
        report["checkpoint_absence"] = _checkpoint_absence_audit(config)
        if not accepted:
            report.update(
                {
                    "status": "rejected",
                    "publication_status": "rejected_by_conservative_gates",
                    "checkpoint_published": False,
                    "candidate_gate_accepted": False,
                    "diagnostics_only": True,
                }
            )
            return _publish_report(config, report)
        report.update(
            {
                "status": PUBLICATION_BLOCK_STATUS,
                "publication_status": PUBLICATION_BLOCK_STATUS,
                "checkpoint_published": False,
                "candidate_gate_accepted": True,
                "diagnostics_only": False,
                "publication_blocker": {
                    "code": PUBLICATION_BLOCK_STATUS,
                    "reason": (
                        "Behavior passed, but an authenticated standalone "
                        "partial-bank serializer/reloader is intentionally the "
                        "next component."
                    ),
                    "dense_fallback_permitted": False,
                    "python_object_serialization_permitted": False,
                },
            }
        )
        return _publish_report(config, report)
    except BaseException as exc:
        cleanup_errors: list[str] = []
        rollback = None
        if (
            installation is not None
            and installation.active
            and model is not None
            and source_state_sha256 is not None
            and source_counts is not None
            and source_topology is not None
        ):
            try:
                rollback = _authenticated_rollback(
                    installation,
                    model,
                    source_state_sha256=source_state_sha256,
                    source_counts=source_counts,
                    source_topology_sha256=(source_topology["topology_sha256"]),
                )
            except BaseException as rollback_exc:  # catastrophic evidence
                cleanup_errors.append(f"{type(rollback_exc).__name__}: {rollback_exc}")
        try:
            checkpoint_absence = _checkpoint_absence_audit(config)
        except BaseException as checkpoint_exc:  # catastrophic evidence
            checkpoint_absence = {
                "audit_error": (f"{type(checkpoint_exc).__name__}: {checkpoint_exc}")
            }
            cleanup_errors.append(checkpoint_absence["audit_error"])
        report.update(
            {
                "status": "error",
                "publication_status": "error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
                "cleanup_errors": cleanup_errors,
                "checkpoint_published": False,
                "checkpoint_absence": checkpoint_absence,
                "diagnostics_only": True,
            }
        )
        return _publish_report(config, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compress a local dense LFM2 child to a real >=15% total target "
            "with a zero-training indexed coupled-neuron bank."
        )
    )
    parser.add_argument("--source-child-model", type=Path, required=True)
    parser.add_argument("--parent-cache", type=Path, required=True)
    parser.add_argument("--cached-parent-outputs", type=Path, required=True)
    parser.add_argument(
        "--cached-parent-outputs-sha256",
        required=True,
    )
    parser.add_argument("--tokenizer-directory", type=Path)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-destination",
        type=Path,
        required=True,
        help="Reserved path; no checkpoint is written by this experiment.",
    )
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--target-total-unique-parameters",
        type=int,
        help=(
            "Total candidate budget. Defaults to 183757568, exactly 7477 "
            "coupled pairs below the 206726912-parameter incumbent."
        ),
    )
    target.add_argument("--additional-parameters-saved", type=int)
    parser.add_argument(
        "--layer-pairs",
        type=shared_runner.parse_layer_pairs,
        help="Disjoint earlier:later pairs; default is all seven adjacent pairs.",
    )
    parser.add_argument("--max-shared-units-per-layer", type=int)
    parser.add_argument(
        "--max-evidence-rows",
        type=int,
        default=DEFAULT_MAX_EVIDENCE_ROWS,
    )
    parser.add_argument(
        "--similarity-weights",
        type=shared_runner.parse_weight_pair,
        default=(0.5, 0.5),
    )
    parser.add_argument("--exact-assignment-limit", type=int, default=256)
    parser.add_argument("--correlation-chunk-size", type=int, default=256)
    parser.add_argument("--variance-floor", type=float, default=0.0)
    parser.add_argument(
        "--expected-original-parent-unique-parameters",
        type=int,
        default=EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--expected-source-child-unique-parameters",
        type=int,
        default=EXPECTED_SOURCE_CHILD_UNIQUE_PARAMETERS,
    )
    parser.add_argument(
        "--expected-layer-count",
        type=int,
        default=EXPECTED_LAYER_COUNT,
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=EXPECTED_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--expected-source-intermediate-size",
        type=int,
        default=EXPECTED_SOURCE_INTERMEDIATE_SIZE,
    )
    parser.add_argument(
        "--expected-whole-output-case-count",
        type=int,
        default=EXPECTED_WHOLE_OUTPUT_CASE_COUNT,
    )
    parser.add_argument(
        "--minimum-total-compression-fraction",
        type=float,
        default=DEFAULT_MINIMUM_TOTAL_COMPRESSION_FRACTION,
    )
    parser.add_argument(
        "--minimum-parent-residual-cosine",
        type=float,
        default=DEFAULT_MINIMUM_PARENT_RESIDUAL_COSINE,
    )
    parser.add_argument(
        "--minimum-incumbent-residual-cosine",
        type=float,
        default=DEFAULT_MINIMUM_INCUMBENT_RESIDUAL_COSINE,
    )
    parser.add_argument(
        "--minimum-parent-trajectory-score",
        type=float,
        default=DEFAULT_MINIMUM_TRAJECTORY_SCORE,
    )
    parser.add_argument(
        "--minimum-incumbent-trajectory-score",
        type=float,
        default=DEFAULT_MINIMUM_TRAJECTORY_SCORE,
    )
    parser.add_argument("--whole-output-max-new-tokens", type=int, default=64)
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
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-new-fragment-responses",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-minimum-parent-nonempty-retention",
        type=float,
        default=DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION,
    )
    parser.add_argument(
        "--whole-output-minimum-task-score-retention",
        type=float,
        default=DEFAULT_MINIMUM_WHOLE_OUTPUT_RETENTION,
    )
    parser.add_argument(
        "--whole-output-maximum-parent-task-score-regressions",
        type=int,
        default=DEFAULT_MAXIMUM_CASE_REGRESSIONS,
    )
    parser.add_argument(
        "--whole-output-maximum-new-generation-limit-saturations",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-ngram-share-regressions",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--whole-output-maximum-repeated-bigram-share-delta",
        type=float,
        default=0.10,
    )
    return parser


def _config_from_namespace(
    arguments: argparse.Namespace,
) -> PartialNeuronBankRunConfig:
    return PartialNeuronBankRunConfig(**vars(arguments))


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    report = run_experiment(_config_from_namespace(arguments))
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == PUBLICATION_BLOCK_STATUS:
        return 0
    return 2 if report["status"] == "rejected" else 1


if __name__ == "__main__":
    raise SystemExit(main())
