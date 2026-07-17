#!/usr/bin/env python3
"""Screen and evaluate sparse-residual tied-vocabulary LFM2 folds.

This is the justified successor to the falsified global raw-SVD control.  One
reusable SVD supplies a low-rank tied vocabulary base.  A fit-only selector
then spends the exact remaining parameter budget on shared exact token-row
residuals, prioritizing generated tokens, parent top-k/margin competitors,
occurrence, and actual residual logit energy.

Exactly 16 authenticated recovery cases are used.  A deterministic 12/4
case-disjoint split selects residual rows and ranks allocations by held-back
logit top-1/margin/worst-case metrics.  The 12 old manual cases are never used
for selection.  A selected allocation is only a frontier probe until it is
installed transactionally and evaluated on the separate 64-case intelligence
holdout.  Exact parent wording or token identity is not a promotion criterion.

The runner has no training, optimizer, backward pass, network access, hub
loader, new parent trace generation, or new parent response generation.  It
also refuses to execute before an external authorization binds the exact GPU,
target grid, and independent holdout file.  Building or importing this module
does not launch an experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch

try:
    from experiments import (
        build_lfm2_intelligence_holdout_parent_cache as holdout_generation,
    )
    from experiments import evaluate_lfm2_intelligence_holdout as holdout_evaluator
    from experiments import evaluate_lfm2_intelligence_retention as posthoc
    from experiments import run_lfm2_long_horizon_repair as horizon_runtime
    from experiments import run_lfm2_residual_attention_fold as fold_runtime
    from experiments import (
        run_lfm2_residual_parameter_sharing_fold as shared_runtime,
    )
    from experiments import run_lfm2_vocab_interface_frontier as raw_control
except ImportError:  # Direct execution from experiments/.
    import build_lfm2_intelligence_holdout_parent_cache as holdout_generation
    import evaluate_lfm2_intelligence_holdout as holdout_evaluator
    import evaluate_lfm2_intelligence_retention as posthoc
    import run_lfm2_long_horizon_repair as horizon_runtime
    import run_lfm2_residual_attention_fold as fold_runtime
    import run_lfm2_residual_parameter_sharing_fold as shared_runtime
    import run_lfm2_vocab_interface_frontier as raw_control

from neural_forge.hf_factorized_vocab import (
    TiedVocabularySVDAnalysis,
    analyze_tied_vocabulary_modules,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    Lfm2CircuitTraceCache,
    fingerprint_lfm2_parent,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from neural_forge.hf_lfm2_vocab_interface_fold import (
    InstalledLfm2VocabularyInterface,
    exact_model_parameter_counts,
    install_lfm2_sparse_residual_vocabulary_pair,
)
from neural_forge.hf_sparse_residual_vocab import (
    SparseResidualTarget,
    SparseResidualVocabularyError,
    build_sparse_residual_vocabulary_pair,
    solve_sparse_residual_target,
)
from neural_forge.hf_sparse_residual_vocab_selection import (
    VocabularyCalibrationCase,
    evaluate_sparse_residual_logit_validation,
    logit_validation_sort_key,
    residual_output_energy,
    select_sparse_residual_token_rows,
    split_recovery_calibration_cases,
)


RUN_SCHEMA = "neural-forge-lfm2-sparse-residual-vocab-frontier-run-v1"
AUTHORIZATION_SCHEMA = (
    "neural-forge-lfm2-sparse-residual-vocab-execution-authorization-v1"
)
EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS = 229_693_184
EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS = 206_726_912
EXPECTED_VOCAB_SIZE = 65_536
EXPECTED_HIDDEN_SIZE = 1_024
EXPECTED_RECOVERY_CASE_COUNT = 16
EXPECTED_INDEPENDENT_HOLDOUT_CASE_COUNT = 64

STRONG_COMPRESSION_FRACTIONS = (0.15, 0.20, 0.25, 0.35, 0.50)
DIRECT_PARENT_COMPRESSION_FRACTIONS = (0.15,)
STRONG_RANK_GRID: Mapping[float, tuple[int, ...]] = {
    0.15: (768, 640, 512),
    0.20: (640, 576, 512),
    0.25: (448, 384, 320),
    0.35: (128, 96, 64),
    0.50: (64, 32, 16),
}
DIRECT_PARENT_RANK_GRID: Mapping[float, tuple[int, ...]] = {
    0.15: (448, 384, 320),
}


class Lfm2SparseResidualFrontierError(RuntimeError):
    """An input, authorization, allocation, evidence, or rollback failed."""


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
        raise Lfm2SparseResidualFrontierError(
            "Frontier evidence must be finite canonical JSON."
        ) from exc


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2SparseResidualFrontierError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2SparseResidualFrontierError(
            f"{name} must be a positive integer."
        )
    return value


def _local_directory(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2SparseResidualFrontierError(
            f"{name} must be an explicit local directory."
        )
    path = Path(raw).expanduser()
    if not path.is_dir() or path.is_symlink():
        raise Lfm2SparseResidualFrontierError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _local_file(value: Path, name: str) -> Path:
    raw = os.fspath(value)
    if "://" in raw:
        raise Lfm2SparseResidualFrontierError(
            f"{name} must be an explicit local file."
        )
    path = Path(raw).expanduser()
    if not path.is_file() or path.is_symlink():
        raise Lfm2SparseResidualFrontierError(
            f"{name} is missing or unsafe: {path}."
        )
    return path.resolve(strict=True)


def _fraction_tuple(value: Any, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Lfm2SparseResidualFrontierError(
            f"{name} must be a fraction sequence."
        )
    result = tuple(float(item) for item in value)
    if (
        not result
        or any(not math.isfinite(item) or not 0.0 < item < 1.0 for item in result)
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
    ):
        raise Lfm2SparseResidualFrontierError(
            f"{name} must contain unique ascending fractions in (0, 1)."
        )
    return result


def _rank_grid_document(
    fractions: Sequence[float],
    grid: Mapping[float, Sequence[int]],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for fraction in fractions:
        ranks = grid.get(float(fraction))
        if (
            ranks is None
            or isinstance(ranks, (str, bytes))
            or not isinstance(ranks, Sequence)
            or not ranks
            or any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0
                for rank in ranks
            )
            or len(set(ranks)) != len(ranks)
        ):
            raise Lfm2SparseResidualFrontierError(
                f"No valid rank grid exists for target {fraction:.2f}."
            )
        result[f"{float(fraction):.2f}"] = list(ranks)
    return result


@dataclass(frozen=True, slots=True)
class SparseResidualFrontierConfig:
    strong_source_model: Path
    strong_source_cache: Path
    original_parent_model: Path
    parent_trajectory_cache: Path
    recovery_horizon_cache: Path
    recovery_parent_envelope: Path
    expected_recovery_parent_envelope_file_sha256: str
    expected_recovery_horizon_cache_key: str
    expected_parent_state_fingerprint: str
    tokenizer_directory: Path
    expected_tokenizer_fingerprint: str
    expected_strong_source_state_sha256: str
    expected_strong_source_cache_key: str
    expected_original_parent_state_sha256: str
    expected_parent_trajectory_cache_key: str
    independent_holdout_parent_envelope: Path
    expected_holdout_parent_envelope_file_sha256: str
    expected_holdout_parent_envelope_sha256: str
    expected_holdout_manifest_sha256: str
    expected_holdout_parent_state_fingerprint: str
    expected_holdout_parent_model_fingerprint: str
    execution_authorization_file: Path
    expected_execution_authorization_file_sha256: str
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
    run_direct_parent_control: bool = True
    expected_original_parent_unique_parameters: int = (
        EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
    )
    expected_strong_source_unique_parameters: int = (
        EXPECTED_STRONG_SOURCE_UNIQUE_PARAMETERS
    )
    expected_vocab_size: int = EXPECTED_VOCAB_SIZE
    expected_hidden_size: int = EXPECTED_HIDDEN_SIZE

    def __post_init__(self) -> None:
        for name in (
            "strong_source_model",
            "strong_source_cache",
            "original_parent_model",
            "parent_trajectory_cache",
            "recovery_horizon_cache",
            "tokenizer_directory",
        ):
            object.__setattr__(
                self,
                name,
                _local_directory(getattr(self, name), name),
            )
        for name in (
            "recovery_parent_envelope",
            "independent_holdout_parent_envelope",
            "execution_authorization_file",
        ):
            object.__setattr__(
                self,
                name,
                _local_file(getattr(self, name), name),
            )
        for name in (
            "expected_recovery_parent_envelope_file_sha256",
            "expected_recovery_horizon_cache_key",
            "expected_parent_state_fingerprint",
            "expected_tokenizer_fingerprint",
            "expected_strong_source_state_sha256",
            "expected_strong_source_cache_key",
            "expected_original_parent_state_sha256",
            "expected_parent_trajectory_cache_key",
            "expected_holdout_parent_envelope_file_sha256",
            "expected_holdout_parent_envelope_sha256",
            "expected_holdout_manifest_sha256",
            "expected_holdout_parent_state_fingerprint",
            "expected_holdout_parent_model_fingerprint",
            "expected_execution_authorization_file_sha256",
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
        _rank_grid_document(
            self.strong_compression_fractions,
            STRONG_RANK_GRID,
        )
        _rank_grid_document(
            self.direct_parent_compression_fractions,
            DIRECT_PARENT_RANK_GRID,
        )
        if not isinstance(self.run_direct_parent_control, bool):
            raise Lfm2SparseResidualFrontierError(
                "run_direct_parent_control must be a boolean."
            )
        for name in (
            "expected_original_parent_unique_parameters",
            "expected_strong_source_unique_parameters",
            "expected_vocab_size",
            "expected_hidden_size",
        ):
            _positive_int(getattr(self, name), name)
        if self.expected_strong_source_unique_parameters >= (
            self.expected_original_parent_unique_parameters
        ):
            raise Lfm2SparseResidualFrontierError(
                "The strong source must be smaller than the original parent."
            )
        if (
            not isinstance(self.device, str)
            or not self.device.startswith("cuda:")
            or not self.device[5:].isdigit()
        ):
            raise Lfm2SparseResidualFrontierError(
                "device must be an explicit cuda:<index> target."
            )
        fold_runtime._torch_dtype(self.dtype)
        if self.checkpoint_root.exists():
            raise Lfm2SparseResidualFrontierError(
                "checkpoint_root must not exist before the probe."
            )
        if self.report_path.exists():
            raise Lfm2SparseResidualFrontierError(
                "report_path must not already exist."
            )

    def authorization_binding(self) -> dict[str, Any]:
        return {
            "purpose": "lfm2_sparse_residual_vocab_frontier",
            "device": self.device,
            "independent_holdout_parent_envelope_file_sha256": (
                self.expected_holdout_parent_envelope_file_sha256
            ),
            "strong_compression_fractions": list(
                self.strong_compression_fractions
            ),
            "direct_parent_compression_fractions": list(
                self.direct_parent_compression_fractions
            ),
            "strong_rank_grid": _rank_grid_document(
                self.strong_compression_fractions,
                STRONG_RANK_GRID,
            ),
            "direct_parent_rank_grid": _rank_grid_document(
                self.direct_parent_compression_fractions,
                DIRECT_PARENT_RANK_GRID,
            ),
            "run_direct_parent_control": self.run_direct_parent_control,
            "training_allowed": False,
            "new_parent_trace_generation_allowed": False,
            "new_parent_response_generation_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "strong_source_model": str(self.strong_source_model),
            "strong_source_cache": str(self.strong_source_cache),
            "original_parent_model": str(self.original_parent_model),
            "parent_trajectory_cache": str(self.parent_trajectory_cache),
            "recovery_horizon_cache": str(self.recovery_horizon_cache),
            "recovery_parent_envelope": str(self.recovery_parent_envelope),
            "expected_recovery_parent_envelope_file_sha256": (
                self.expected_recovery_parent_envelope_file_sha256
            ),
            "expected_recovery_horizon_cache_key": (
                self.expected_recovery_horizon_cache_key
            ),
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
            "expected_original_parent_state_sha256": (
                self.expected_original_parent_state_sha256
            ),
            "expected_parent_trajectory_cache_key": (
                self.expected_parent_trajectory_cache_key
            ),
            "independent_holdout_parent_envelope": str(
                self.independent_holdout_parent_envelope
            ),
            "expected_holdout_parent_envelope_file_sha256": (
                self.expected_holdout_parent_envelope_file_sha256
            ),
            "expected_holdout_parent_envelope_sha256": (
                self.expected_holdout_parent_envelope_sha256
            ),
            "expected_holdout_manifest_sha256": (
                self.expected_holdout_manifest_sha256
            ),
            "expected_holdout_parent_state_fingerprint": (
                self.expected_holdout_parent_state_fingerprint
            ),
            "expected_holdout_parent_model_fingerprint": (
                self.expected_holdout_parent_model_fingerprint
            ),
            "execution_authorization_file": str(
                self.execution_authorization_file
            ),
            "expected_execution_authorization_file_sha256": (
                self.expected_execution_authorization_file_sha256
            ),
            "work_root": str(self.work_root),
            "checkpoint_root": str(self.checkpoint_root),
            "report_path": str(self.report_path),
            "device": self.device,
            "dtype": self.dtype,
            "expected_original_parent_unique_parameters": (
                self.expected_original_parent_unique_parameters
            ),
            "expected_strong_source_unique_parameters": (
                self.expected_strong_source_unique_parameters
            ),
            "expected_vocab_size": self.expected_vocab_size,
            "expected_hidden_size": self.expected_hidden_size,
            **self.authorization_binding(),
            "calibration_case_count": EXPECTED_RECOVERY_CASE_COUNT,
            "calibration_fit_case_count": 12,
            "calibration_validation_case_count": 4,
            "old_manual_case_count_used_for_selection": 0,
            "independent_holdout_case_count": (
                EXPECTED_INDEPENDENT_HOLDOUT_CASE_COUNT
            ),
            "factorization": "one_svd_low_rank_plus_shared_sparse_exact_rows",
            "promotion_metric": "coherent_parent_relative_intelligence",
            "exact_token_match_is_promotion_metric": False,
        }


def _observed_gpu_uuid(device: str) -> str:
    index = int(device.split(":", 1)[1])
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )
    if len(values) != 1 or not values[0].startswith("GPU-"):
        raise Lfm2SparseResidualFrontierError(
            "Could not bind the selected CUDA device to one GPU UUID."
        )
    return values[0]


def authenticate_execution_authorization(
    config: SparseResidualFrontierConfig,
    *,
    observed_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Fail before model loading unless the exact run and holdout are approved."""

    if _file_sha256(config.execution_authorization_file) != (
        config.expected_execution_authorization_file_sha256
    ):
        raise Lfm2SparseResidualFrontierError(
            "Execution authorization file hash differs from its binding."
        )
    if _file_sha256(config.independent_holdout_parent_envelope) != (
        config.expected_holdout_parent_envelope_file_sha256
    ):
        raise Lfm2SparseResidualFrontierError(
            "Independent holdout file hash differs from its binding."
        )
    try:
        raw = config.execution_authorization_file.read_bytes()
        document = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2SparseResidualFrontierError(
            "Execution authorization is not ASCII JSON."
        ) from exc
    if (
        not isinstance(document, Mapping)
        or _canonical_json(document) + b"\n" != raw
    ):
        raise Lfm2SparseResidualFrontierError(
            "Execution authorization is not canonical JSON."
        )
    body = dict(document)
    seal = body.pop("authorization_sha256", None)
    gpu_uuid = (
        _observed_gpu_uuid(config.device)
        if observed_gpu_uuid is None
        else observed_gpu_uuid
    )
    expected = {
        "schema": AUTHORIZATION_SCHEMA,
        "authorized": True,
        "gpu_uuid": gpu_uuid,
        **config.authorization_binding(),
    }
    if (
        not isinstance(gpu_uuid, str)
        or not gpu_uuid.startswith("GPU-")
        or body != expected
        or seal != _json_sha256(body)
    ):
        raise Lfm2SparseResidualFrontierError(
            "Execution authorization does not bind the exact GPU, target grid, "
            "holdout, and zero-training/no-parent-call contract."
        )
    return {
        **body,
        "authorization_sha256": seal,
        "authorization_file_sha256": (
            config.expected_execution_authorization_file_sha256
        ),
        "authenticated_before_model_load": True,
    }


def plan_sparse_residual_allocations(
    *,
    source_label: str,
    source_unique_parameters: int,
    fractions: Sequence[float],
    rank_grid: Mapping[float, Sequence[int]],
    original_parent_unique_parameters: int = (
        EXPECTED_ORIGINAL_PARENT_UNIQUE_PARAMETERS
    ),
    vocab_size: int = EXPECTED_VOCAB_SIZE,
    hidden_size: int = EXPECTED_HIDDEN_SIZE,
    minimum_selected_rows: int = 256,
) -> tuple[dict[str, Any], ...]:
    """Solve every rank allocation and preserve impossible targets as evidence."""

    if not isinstance(source_label, str) or not source_label:
        raise Lfm2SparseResidualFrontierError(
            "source_label must be nonempty."
        )
    results = []
    grid_document = _rank_grid_document(fractions, rank_grid)
    for fraction in fractions:
        attempts = []
        for rank in grid_document[f"{float(fraction):.2f}"]:
            try:
                target = solve_sparse_residual_target(
                    original_parent_unique_parameters=(
                        original_parent_unique_parameters
                    ),
                    source_unique_parameters=source_unique_parameters,
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    total_compression_fraction=float(fraction),
                    rank=rank,
                    minimum_selected_rows=minimum_selected_rows,
                )
            except SparseResidualVocabularyError as exc:
                attempts.append(
                    {
                        "rank": rank,
                        "status": "impossible",
                        "error": str(exc),
                    }
                )
            else:
                attempts.append(
                    {
                        "rank": rank,
                        "status": "solved",
                        "target": target.as_dict(),
                    }
                )
        solved = [item for item in attempts if item["status"] == "solved"]
        target_parameters = math.floor(
            original_parent_unique_parameters * (1.0 - float(fraction))
        )
        item = {
            "source_label": source_label,
            "requested_total_compression_fraction": float(fraction),
            "requested_target_unique_parameters": target_parameters,
            "status": "solved" if solved else "impossible",
            "attempts": attempts,
            "solved_allocation_count": len(solved),
            "impossible_allocation_count": len(attempts) - len(solved),
        }
        results.append({**item, "plan_sha256": _json_sha256(item)})
    return tuple(results)


def _special_token_ids(
    tokenizer_directory: Path,
    tokenizer_fingerprint: str,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    identity = posthoc.load_bound_special_token_ids(
        tokenizer_directory,
        expected_tokenizer_fingerprint=tokenizer_fingerprint,
    )
    mapping = identity["special_token_ids"]
    missing = [
        name for name in ("bos", "eos", "pad") if not mapping.get(name)
    ]
    if missing:
        raise Lfm2SparseResidualFrontierError(
            "BOS/EOS/PAD residual anchors are mandatory; missing "
            + ",".join(missing)
            + "."
        )
    values = tuple(
        sorted(
            {
                int(token)
                for name in ("bos", "eos", "pad")
                for token in mapping[name]
            }
        )
    )
    return values, identity


def _last_layer_post_residual(
    tensors: Mapping[str, torch.Tensor],
    phase: str,
) -> torch.Tensor:
    suffix = ".residual.post_layer"
    names = sorted(
        name
        for name in tensors
        if name.startswith(f"{phase}.layer.") and name.endswith(suffix)
    )
    if not names:
        raise Lfm2SparseResidualFrontierError(
            f"Cached phase {phase} has no final residual."
        )
    return tensors[names[-1]]


def _prediction_row(
    value: torch.Tensor,
    name: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or int(value.shape[0]) != 1
        or int(value.shape[1]) <= 0
    ):
        raise Lfm2SparseResidualFrontierError(
            f"Cached {name} is not a one-example sequence tensor."
        )
    return value[:, -1, :]


def _resolve_lfm2_final_norm(parent_model: torch.nn.Module) -> torch.nn.Module:
    """Resolve the native LFM2 final norm without assuming a generic name."""

    backbone = getattr(parent_model, "model", None)
    # Transformers' native LFM2 backbone names its final RMS norm
    # ``embedding_norm``.  Synthetic/older fixtures may expose the generic
    # ``norm`` spelling, which remains a compatibility fallback.
    final_norm = getattr(backbone, "embedding_norm", None)
    if final_norm is None:
        final_norm = getattr(backbone, "norm", None)
    if not isinstance(final_norm, torch.nn.Module):
        raise Lfm2SparseResidualFrontierError(
            "Parent final norm is unavailable."
        )
    return final_norm


def build_recovery_calibration_cases(
    parent_model: torch.nn.Module,
    cache: Lfm2CircuitTraceCache,
    envelope: horizon_runtime.SealedEnvelopeAudit,
    *,
    device: str,
) -> tuple[tuple[VocabularyCalibrationCase, ...], dict[str, Any]]:
    """Derive exact LM-head inputs from cached residuals; call no parent."""

    if (
        cache.batch_count != EXPECTED_RECOVERY_CASE_COUNT
        or len(envelope.recovery_rows) != EXPECTED_RECOVERY_CASE_COUNT
        or not envelope.ready
        or cache.cache_key is None
    ):
        raise Lfm2SparseResidualFrontierError(
            "Recovery cache/envelope is not the exact authenticated 16-case set."
        )
    if cache.parent_fingerprint != envelope.parent_state_fingerprint:
        raise Lfm2SparseResidualFrontierError(
            "Recovery cache and envelope parent identities differ."
        )
    final_norm = _resolve_lfm2_final_norm(parent_model)
    head = parent_model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise Lfm2SparseResidualFrontierError(
            "Parent dense LM head is unavailable."
        )
    model_device = torch.device(device)
    cases = []
    row_descriptors = []
    with torch.inference_mode():
        for batch_index, envelope_row in enumerate(envelope.recovery_rows):
            batch = cache.load_batch(batch_index)
            case_id = str(envelope_row["id"])
            category = (
                "math" if case_id.startswith("recovery_math_") else "code"
            )
            prompt_ids = tuple(
                int(value) for value in batch.input_ids.reshape(-1).tolist()
            )
            attention = tuple(
                int(value) for value in batch.attention_mask.reshape(-1).tolist()
            )
            output_ids = tuple(
                int(value)
                for value in batch.forced_decode_ids.reshape(-1).tolist()
            )
            if (
                prompt_ids != tuple(envelope_row["input_ids"])
                or attention != tuple(envelope_row["attention_mask"])
                or output_ids != tuple(envelope_row["parent_output_ids"])
                or len(output_ids) <= 0
            ):
                raise Lfm2SparseResidualFrontierError(
                    f"Recovery cache batch {batch_index} differs from {case_id}."
                )
            phases = ("prefill",) + tuple(
                f"decode.{index:03d}"
                for index in range(len(output_ids) - 1)
            )
            hidden_rows = []
            parent_logits = []
            for output_index, phase in enumerate(phases):
                residual = _prediction_row(
                    _last_layer_post_residual(batch.tensors, phase),
                    f"{phase} final residual",
                ).to(device=model_device)
                cached_logits = _prediction_row(
                    batch.tensors[f"{phase}.logits"],
                    f"{phase} logits",
                )
                hidden = final_norm(residual)
                reproduced = head(hidden).to(
                    device="cpu",
                    dtype=torch.float32,
                )
                cached_float = cached_logits.to(dtype=torch.float32)
                expected_token = output_ids[output_index]
                cached_top1 = int(cached_float.argmax(dim=-1).item())
                reproduced_top1 = int(reproduced.argmax(dim=-1).item())
                numerically_close = bool(
                    torch.allclose(
                        reproduced,
                        cached_float,
                        rtol=0.20,
                        atol=1.0,
                    )
                )
                if (
                    reproduced_top1 != cached_top1
                    or not numerically_close
                ):
                    delta = (reproduced - cached_float).abs()
                    raise Lfm2SparseResidualFrontierError(
                        f"Cached LM-head evidence does not reproduce {case_id} "
                        f"output row {output_index}: expected={expected_token}, "
                        f"cached_top1={cached_top1}, "
                        f"reproduced_top1={reproduced_top1}, "
                        f"numerically_close={numerically_close}, "
                        f"maximum_absolute_error={float(delta.max().item()):.9g}, "
                        f"mean_absolute_error={float(delta.mean().item()):.9g}."
                    )
                hidden_rows.append(hidden.to(device="cpu"))
                parent_logits.append(cached_logits)
            case = VocabularyCalibrationCase.create(
                case_id=case_id,
                category=category,
                prompt_token_ids=prompt_ids,
                parent_output_token_ids=output_ids,
                hidden_rows=torch.cat(hidden_rows, dim=0),
                parent_logits=torch.cat(parent_logits, dim=0),
                vocab_size=int(head.out_features),
                hidden_size=int(head.in_features),
            )
            cases.append(case)
            row_descriptors.append(
                {
                    "batch_index": batch_index,
                    "input_sha256": batch.input_sha256,
                    "case": case.descriptor(),
                }
            )
    split = split_recovery_calibration_cases(cases)
    body = {
        "cache_key": cache.cache_key,
        "cache_parent_fingerprint": cache.parent_fingerprint,
        "envelope_sha256": envelope.envelope_sha256,
        "case_descriptors": row_descriptors,
        "split": split.as_dict(),
        "parent_forward_calls": 0,
        "parent_response_generation_calls": 0,
        "manual_case_count": 0,
    }
    return tuple(cases), {**body, "evidence_sha256": _json_sha256(body)}


def _source_authentication(
    model: torch.nn.Module,
    cache: Lfm2CircuitTraceCache,
    *,
    expected_unique_parameters: int,
    expected_state_sha256: str,
    expected_cache_key: str,
) -> dict[str, Any]:
    result = raw_control._source_authentication(
        model,
        source_cache=cache,
        expected_unique_parameters=expected_unique_parameters,
        expected_state_sha256=expected_state_sha256,
        expected_cache_key=expected_cache_key,
    )
    if (
        result["vocab_size"] != EXPECTED_VOCAB_SIZE
        or result["hidden_size"] != EXPECTED_HIDDEN_SIZE
    ):
        raise Lfm2SparseResidualFrontierError(
            "Source dimensions differ from the LFM2 sparse vocabulary contract."
        )
    return result


def _allocation_targets(
    plan: Sequence[Mapping[str, Any]],
) -> tuple[SparseResidualTarget, ...]:
    targets = []
    for fraction in plan:
        for attempt in fraction["attempts"]:
            if attempt["status"] != "solved":
                continue
            value = attempt["target"]
            targets.append(SparseResidualTarget(**value))
    return tuple(targets)


def _factor_seal(pair: Any) -> dict[str, Any]:
    core = pair.core
    body = {
        "vocab_factor_sha256": fold_runtime._tensor_sha256(
            core.vocab_factor
        ),
        "hidden_factor_sha256": fold_runtime._tensor_sha256(
            core.hidden_factor
        ),
        "residual_rows_sha256": fold_runtime._tensor_sha256(
            core.residual_rows
        ),
        "selected_token_ids_sha256": fold_runtime._tensor_sha256(
            core.selected_token_ids
        ),
        "vocab_factor_shape": list(core.vocab_factor.shape),
        "hidden_factor_shape": list(core.hidden_factor.shape),
        "residual_rows_shape": list(core.residual_rows.shape),
        "selected_token_ids_shape": list(core.selected_token_ids.shape),
        "one_shared_core": pair.embedding.core is pair.lm_head.core,
        "dense_vocabulary_matrix_retained_by_candidate_runtime": False,
    }
    return {**body, "factor_seal_sha256": _json_sha256(body)}


def screen_source_allocations(
    *,
    source_label: str,
    model: torch.nn.Module,
    analysis: TiedVocabularySVDAnalysis,
    plan: Sequence[Mapping[str, Any]],
    cases: Sequence[VocabularyCalibrationCase],
    special_token_ids: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]]]:
    """Select/evaluate every valid rank without installing or generating text."""

    split = split_recovery_calibration_cases(cases)
    fit_ids = set(split.fit_case_ids)
    fit_hidden = torch.cat(
        [case.hidden_rows for case in cases if case.case_id in fit_ids],
        dim=0,
    )
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    attempts = []
    best: dict[float, dict[str, Any]] = {}
    for target in _allocation_targets(plan):
        base = analysis.factorize(target.rank)
        energy = residual_output_energy(
            source_weight=embedding.weight,
            vocab_factor=base.embedding.vocab_factor,
            hidden_factor=base.embedding.hidden_factor,
            hidden_rows=fit_hidden,
        )
        selected = select_sparse_residual_token_rows(
            cases,
            split,
            selected_row_budget=target.selected_row_budget,
            special_token_ids=special_token_ids,
            residual_energy=energy,
            top_k=16,
        )
        pair = build_sparse_residual_vocabulary_pair(
            analysis,
            embedding,
            head,
            rank=target.rank,
            selected_token_ids=torch.tensor(
                selected.selected_token_ids,
                dtype=torch.long,
                device=embedding.weight.device,
            ),
        )
        validation = evaluate_sparse_residual_logit_validation(
            pair.core,
            cases,
            split,
            top_k=16,
        )
        original_interface = sum(
            int(parameter.numel())
            for parameter in {
                id(parameter): parameter
                for module in (embedding, head)
                for parameter in module.parameters()
            }.values()
        )
        physical = (
            exact_model_parameter_counts(model)["unique"]
            - original_interface
            + pair.unique_parameter_count
        )
        if physical != target.physical_target_unique_parameters:
            raise Lfm2SparseResidualFrontierError(
                "Built sparse allocation differs from its exact budget solution."
            )
        selection_document = selected.as_dict()
        attempt_body = {
            "source_label": source_label,
            "target": target.as_dict(),
            "pair_plan": pair.plan.as_dict(),
            "physical_model_unique_parameters": physical,
            "selection": selection_document,
            "selection_case_split": split.as_dict(),
            "validation": validation,
            "factor_seal": _factor_seal(pair),
            "training_steps": 0,
            "parent_forward_calls": 0,
            "parent_response_generation_calls": 0,
            "installed_for_screening": False,
        }
        attempt = {
            **attempt_body,
            "screen_sha256": _json_sha256(attempt_body),
        }
        attempts.append(attempt)
        fraction = target.requested_total_compression_fraction
        incumbent = best.get(fraction)
        candidate_key = (
            logit_validation_sort_key(validation),
            target.selected_row_budget,
            target.rank,
        )
        if incumbent is None or candidate_key > incumbent["_sort_key"]:
            best[fraction] = {
                **attempt,
                "_sort_key": candidate_key,
            }
        del pair, base, energy
    for value in best.values():
        value.pop("_sort_key", None)
    return attempts, best


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    shared_runtime._atomic_write_new_json(path, dict(value))


def _evaluate_independent_holdout(
    *,
    config: SparseResidualFrontierConfig,
    model: torch.nn.Module,
    tokenizer: Any,
    holdout: Any,
    attempt_root: Path,
    phase: str,
) -> dict[str, Any]:
    candidate_state = fold_runtime.model_state_sha256(model)
    generated = holdout_generation.generate_lfm2_holdout_rows_once(
        model,
        tokenizer,
        holdout.manifest,
    )
    candidate_document = seal_lfm2_intelligence_holdout_candidate_outputs(
        holdout.manifest,
        generated,
        candidate_state_fingerprint=candidate_state,
        tokenizer_fingerprint=config.expected_tokenizer_fingerprint,
        eos_token_ids=holdout.eos_token_ids,
    )
    candidate_path = attempt_root / "candidate-holdout.json"
    _write_new_json(candidate_path, candidate_document)
    candidate_file_sha = _file_sha256(candidate_path)
    report_path = attempt_root / "holdout-report.json"
    packet_path = attempt_root / "holdout-manual-review-packet.json"
    report = holdout_evaluator.evaluate_lfm2_intelligence_holdout(
        holdout_evaluator.HoldoutEvaluationConfig(
            parent_envelope=config.independent_holdout_parent_envelope,
            expected_parent_envelope_file_sha256=(
                config.expected_holdout_parent_envelope_file_sha256
            ),
            expected_parent_envelope_sha256=(
                config.expected_holdout_parent_envelope_sha256
            ),
            expected_manifest_sha256=config.expected_holdout_manifest_sha256,
            expected_parent_state_fingerprint=(
                config.expected_holdout_parent_state_fingerprint
            ),
            expected_parent_model_fingerprint=(
                config.expected_holdout_parent_model_fingerprint
            ),
            tokenizer_directory=config.tokenizer_directory,
            expected_tokenizer_fingerprint=(
                config.expected_tokenizer_fingerprint
            ),
            candidate_report=candidate_path,
            expected_candidate_report_sha256=candidate_file_sha,
            report_path=report_path,
            manual_review_packet_path=packet_path,
            expected_candidate_outputs_sha256=candidate_document[
                "candidate_outputs_sha256"
            ],
            expected_candidate_state_fingerprint=candidate_state,
        )
    )
    return {
        "phase": phase,
        "candidate_state_fingerprint": candidate_state,
        "candidate_output_path": str(candidate_path),
        "candidate_output_file_sha256": candidate_file_sha,
        "candidate_output_sha256": candidate_document[
            "candidate_outputs_sha256"
        ],
        "candidate_generate_invocations": len(generated),
        "parent_generate_invocations": 0,
        "parent_forward_calls": 0,
        "report": report,
        "report_path": str(report_path),
        "report_file_sha256": _file_sha256(report_path),
        "manual_review_packet_path": str(packet_path),
        "manual_review_packet_file_sha256": _file_sha256(packet_path),
        "strict_promoted": bool(report["promoted"]),
    }


def evaluate_selected_allocation(
    *,
    config: SparseResidualFrontierConfig,
    source_label: str,
    model: torch.nn.Module,
    analysis: TiedVocabularySVDAnalysis,
    source_authentication: Mapping[str, Any],
    screen: Mapping[str, Any],
    tokenizer: Any,
    holdout: Any,
) -> dict[str, Any]:
    """Install one screened candidate, run the 64-case holdout, and rollback."""

    target = screen["target"]
    selected_ids = screen["selection"]["selected_token_ids"]
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    pair = build_sparse_residual_vocabulary_pair(
        analysis,
        embedding,
        head,
        rank=int(target["rank"]),
        selected_token_ids=torch.tensor(
            selected_ids,
            dtype=torch.long,
            device=embedding.weight.device,
        ),
    )
    phase = (
        f"sparse_vocab_{source_label}_"
        f"{int(round(100 * float(target['requested_total_compression_fraction'])))}pct_"
        f"rank{int(target['rank'])}_rows{len(selected_ids)}"
    )
    attempt_root = config.work_root / phase
    if attempt_root.exists():
        raise Lfm2SparseResidualFrontierError(
            f"Attempt root already exists: {attempt_root}."
        )
    attempt_root.mkdir(parents=True)
    installation: InstalledLfm2VocabularyInterface | None = None
    result: dict[str, Any] = {
        "phase": phase,
        "source_label": source_label,
        "screen_sha256": screen["screen_sha256"],
        "status": "initializing",
        "accepted": False,
    }
    try:
        installation = install_lfm2_sparse_residual_vocabulary_pair(
            model,
            pair,
        )
        holdout_report = _evaluate_independent_holdout(
            config=config,
            model=model,
            tokenizer=tokenizer,
            holdout=holdout,
            attempt_root=attempt_root,
            phase=phase,
        )
        result.update(
            {
                "status": "evaluated",
                "accepted": bool(holdout_report["strict_promoted"]),
                "target": dict(target),
                "pair_plan": pair.plan.as_dict(),
                "factor_seal": _factor_seal(pair),
                "installation": dict(installation.installation_audit),
                "independent_holdout": holdout_report,
                "old_manual_12_usage": (
                    "not_used_for_selection_or_binding_evaluation"
                ),
                "checkpoint_published": False,
            }
        )
        result["rollback"] = raw_control._authenticated_rollback(
            installation,
            model,
            source_state_sha256=source_authentication["state_sha256"],
            source_counts=source_authentication["parameter_counts"],
        )
        installation = None
        return result
    except BaseException as exc:
        rollback = None
        rollback_error = None
        if installation is not None and installation.active:
            try:
                rollback = raw_control._authenticated_rollback(
                    installation,
                    model,
                    source_state_sha256=source_authentication["state_sha256"],
                    source_counts=source_authentication["parameter_counts"],
                )
            except BaseException as rollback_exc:
                rollback_error = (
                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                )
        result.update(
            {
                "status": "error",
                "accepted": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "rollback": rollback,
                "rollback_error": rollback_error,
                "checkpoint_published": False,
            }
        )
        if rollback_error is not None:
            raise Lfm2SparseResidualFrontierError(
                f"Attempt and exact rollback failed: {rollback_error}"
            ) from exc
        return result


def _checkpoint_absence(config: SparseResidualFrontierConfig) -> dict[str, Any]:
    if config.checkpoint_root.exists():
        raise Lfm2SparseResidualFrontierError(
            "A checkpoint appeared during a non-publication frontier probe."
        )
    return {
        "path": str(config.checkpoint_root),
        "exists": False,
        "published": False,
        "frontier_probes_are_not_promotion_claims": True,
    }


def _sealed_report(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("run_sha256", None)
    return {**body, "run_sha256": _json_sha256(body)}


def run_experiment(config: SparseResidualFrontierConfig) -> dict[str, Any]:
    """Run only after exact authorization; always publish a sealed report."""

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
        authorization = authenticate_execution_authorization(config)
        report["execution_authorization"] = authorization
        if _file_sha256(config.recovery_parent_envelope) != (
            config.expected_recovery_parent_envelope_file_sha256
        ):
            raise Lfm2SparseResidualFrontierError(
                "Recovery parent envelope file hash changed."
            )
        recovery_cache = Lfm2CircuitTraceCache.open(
            config.recovery_horizon_cache,
            verify=True,
        )
        strong_cache = Lfm2CircuitTraceCache.open(
            config.strong_source_cache,
            verify=True,
        )
        parent_cache = Lfm2CircuitTraceCache.open(
            config.parent_trajectory_cache,
            verify=True,
        )
        if (
            recovery_cache.cache_key
            != config.expected_recovery_horizon_cache_key
            or recovery_cache.parent_fingerprint
            != config.expected_parent_state_fingerprint
            or parent_cache.cache_key
            != config.expected_parent_trajectory_cache_key
        ):
            raise Lfm2SparseResidualFrontierError(
                "Recovery or parent cache identity differs from the CLI binding."
            )
        tokenizer_fingerprint, _files = (
            posthoc.fingerprint_bound_local_tokenizer(
                config.tokenizer_directory
            )
        )
        if tokenizer_fingerprint != config.expected_tokenizer_fingerprint:
            raise Lfm2SparseResidualFrontierError(
                "Local tokenizer fingerprint differs from its binding."
            )
        tokenizer = fold_runtime.load_local_tokenizer(
            config.tokenizer_directory
        )
        special_ids, special_identity = _special_token_ids(
            config.tokenizer_directory,
            tokenizer_fingerprint,
        )
        envelope = horizon_runtime.audit_sealed_horizon_envelope(
            config.recovery_parent_envelope,
            expected_sha256=(
                config.expected_recovery_parent_envelope_file_sha256
            ),
            tokenizer_eos_token_ids=special_identity["special_token_ids"]["eos"],
            require_token_texts=False,
        )
        if (
            not envelope.ready
            or envelope.parent_state_fingerprint
            != config.expected_parent_state_fingerprint
            or envelope.tokenizer_fingerprint
            != config.expected_tokenizer_fingerprint
        ):
            raise Lfm2SparseResidualFrontierError(
                "Recovery envelope is blocked or bound to another parent/tokenizer."
            )
        holdout_document = json.loads(
            config.independent_holdout_parent_envelope.read_text(
                encoding="utf-8"
            )
        )
        holdout = load_lfm2_intelligence_holdout_parent_envelope(
            holdout_document,
            expected_envelope_sha256=(
                config.expected_holdout_parent_envelope_sha256
            ),
            expected_manifest_sha256=config.expected_holdout_manifest_sha256,
            expected_parent_state_fingerprint=(
                config.expected_holdout_parent_state_fingerprint
            ),
            expected_parent_model_fingerprint=(
                config.expected_holdout_parent_model_fingerprint
            ),
            expected_tokenizer_fingerprint=(
                config.expected_tokenizer_fingerprint
            ),
        )
        if len(holdout.manifest.cases) != EXPECTED_INDEPENDENT_HOLDOUT_CASE_COUNT:
            raise Lfm2SparseResidualFrontierError(
                "Binding holdout is not the independent 64-case manifest."
            )
        dtype = fold_runtime._torch_dtype(config.dtype)
        strong_model = fold_runtime.load_local_lfm2_model(
            config.strong_source_model,
            device=config.device,
            dtype=dtype,
        )
        strong_model.eval()
        strong_auth = _source_authentication(
            strong_model,
            strong_cache,
            expected_unique_parameters=(
                config.expected_strong_source_unique_parameters
            ),
            expected_state_sha256=config.expected_strong_source_state_sha256,
            expected_cache_key=config.expected_strong_source_cache_key,
        )
        parent_model = fold_runtime.load_local_lfm2_model(
            config.original_parent_model,
            device=config.device,
            dtype=dtype,
        )
        parent_model.eval()
        parent_auth = _source_authentication(
            parent_model,
            parent_cache,
            expected_unique_parameters=(
                config.expected_original_parent_unique_parameters
            ),
            expected_state_sha256=config.expected_original_parent_state_sha256,
            expected_cache_key=config.expected_parent_trajectory_cache_key,
        )
        if (
            parent_auth["identity_sha256"]
            != config.expected_parent_state_fingerprint
            or fingerprint_lfm2_parent(parent_model).identity_sha256
            != config.expected_parent_state_fingerprint
            or not torch.equal(
                strong_model.get_input_embeddings().weight.detach(),
                parent_model.get_input_embeddings().weight.detach(),
            )
        ):
            raise Lfm2SparseResidualFrontierError(
                "Strong child and original parent do not share the exact tied matrix."
            )
        cases, calibration_evidence = build_recovery_calibration_cases(
            parent_model,
            recovery_cache,
            envelope,
            device=config.device,
        )
        split = split_recovery_calibration_cases(cases)
        analysis = analyze_tied_vocabulary_modules(
            strong_model.get_input_embeddings(),
            strong_model.get_output_embeddings(),
        )
        strong_plan = plan_sparse_residual_allocations(
            source_label="strong_10pct_child",
            source_unique_parameters=(
                config.expected_strong_source_unique_parameters
            ),
            fractions=config.strong_compression_fractions,
            rank_grid=STRONG_RANK_GRID,
            original_parent_unique_parameters=(
                config.expected_original_parent_unique_parameters
            ),
            vocab_size=config.expected_vocab_size,
            hidden_size=config.expected_hidden_size,
        )
        direct_plan = (
            plan_sparse_residual_allocations(
                source_label="original_parent",
                source_unique_parameters=(
                    config.expected_original_parent_unique_parameters
                ),
                fractions=config.direct_parent_compression_fractions,
                rank_grid=DIRECT_PARENT_RANK_GRID,
                original_parent_unique_parameters=(
                    config.expected_original_parent_unique_parameters
                ),
                vocab_size=config.expected_vocab_size,
                hidden_size=config.expected_hidden_size,
            )
            if config.run_direct_parent_control
            else ()
        )
        strong_screens, strong_best = screen_source_allocations(
            source_label="strong_10pct_child",
            model=strong_model,
            analysis=analysis,
            plan=strong_plan,
            cases=cases,
            special_token_ids=special_ids,
        )
        if config.run_direct_parent_control:
            direct_screens, direct_best = screen_source_allocations(
                source_label="original_parent",
                model=parent_model,
                analysis=analysis,
                plan=direct_plan,
                cases=cases,
                special_token_ids=special_ids,
            )
        else:
            direct_screens, direct_best = [], {}
        evaluations = []
        for fraction in sorted(strong_best):
            evaluations.append(
                evaluate_selected_allocation(
                    config=config,
                    source_label="strong_10pct_child",
                    model=strong_model,
                    analysis=analysis,
                    source_authentication=strong_auth,
                    screen=strong_best[fraction],
                    tokenizer=tokenizer,
                    holdout=holdout,
                )
            )
        for fraction in sorted(direct_best):
            evaluations.append(
                evaluate_selected_allocation(
                    config=config,
                    source_label="original_parent",
                    model=parent_model,
                    analysis=analysis,
                    source_authentication=parent_auth,
                    screen=direct_best[fraction],
                    tokenizer=tokenizer,
                    holdout=holdout,
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
                "calibration_evidence": calibration_evidence,
                "calibration_split": split.as_dict(),
                "special_token_anchors": {
                    "selected_ids": list(special_ids),
                    "identity": special_identity,
                    "bos_eos_pad_always_mandatory": True,
                },
                "one_svd_evidence": {
                    "decomposition_invocations": 1,
                    "reused_across_every_rank_and_exact_parent_matrix_copy": True,
                    "singular_values_sha256": _json_sha256(
                        list(analysis.singular_values)
                    ),
                    "source_vocab_weight_sha256": strong_auth[
                        "vocab_weight_sha256"
                    ],
                },
                "budget_plans": {
                    "strong_10pct_child": list(strong_plan),
                    "original_parent": list(direct_plan),
                    "direct_parent_control_enabled": (
                        config.run_direct_parent_control
                    ),
                },
                "screening": {
                    "strong_10pct_child": strong_screens,
                    "original_parent": direct_screens,
                    "selection_cases": 12,
                    "validation_cases": 4,
                    "case_disjoint": True,
                    "old_manual_12_cases_used": 0,
                    "candidate_model_forward_calls": 0,
                    "parent_model_forward_calls": 0,
                },
                "binding_independent_holdout_evaluations": evaluations,
                "accepted_evaluation_count": sum(
                    bool(item["accepted"]) for item in evaluations
                ),
                "error_evaluation_count": sum(
                    item["status"] == "error" for item in evaluations
                ),
                "new_parent_trace_generation_calls": 0,
                "new_parent_response_generation_calls": 0,
                "checkpoint_published": False,
                "checkpoint_absence": _checkpoint_absence(config),
                "conclusion": {
                    "frontier_probes_are_not_promotion_claims": True,
                    "exact_token_match_required": False,
                    "coherent_parent_relative_intelligence_required": True,
                    "fifty_percent_vocab_only_target_may_be_impossible_because_"
                    "nonvocab_parameters_alone_exceed_budget": True,
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
    _write_new_json(config.report_path, sealed)
    return sealed


def _parse_fractions(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run authorized low-rank plus exact sparse-residual LFM2 vocabulary "
            "frontiers at 15/20/25/35/50% total-parent targets."
        )
    )
    parser.add_argument("--strong-source-model", type=Path, required=True)
    parser.add_argument("--strong-source-cache", type=Path, required=True)
    parser.add_argument("--original-parent-model", type=Path, required=True)
    parser.add_argument("--parent-trajectory-cache", type=Path, required=True)
    parser.add_argument("--recovery-horizon-cache", type=Path, required=True)
    parser.add_argument("--recovery-parent-envelope", type=Path, required=True)
    parser.add_argument(
        "--expected-recovery-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-recovery-horizon-cache-key",
        required=True,
    )
    parser.add_argument("--expected-parent-state-fingerprint", required=True)
    parser.add_argument("--tokenizer-directory", type=Path, required=True)
    parser.add_argument("--expected-tokenizer-fingerprint", required=True)
    parser.add_argument("--expected-strong-source-state-sha256", required=True)
    parser.add_argument("--expected-strong-source-cache-key", required=True)
    parser.add_argument("--expected-original-parent-state-sha256", required=True)
    parser.add_argument(
        "--expected-parent-trajectory-cache-key",
        required=True,
    )
    parser.add_argument(
        "--independent-holdout-parent-envelope",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-holdout-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--expected-holdout-parent-envelope-sha256",
        required=True,
    )
    parser.add_argument("--expected-holdout-manifest-sha256", required=True)
    parser.add_argument(
        "--expected-holdout-parent-state-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-holdout-parent-model-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--execution-authorization-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--expected-execution-authorization-file-sha256",
        required=True,
    )
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
        "--skip-direct-parent-control",
        action="store_false",
        dest="run_direct_parent_control",
        help=(
            "Skip the preregistered direct-parent control screen and holdout "
            "evaluation while retaining the parent for cached calibration."
        ),
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
    parser.add_argument(
        "--expected-vocab-size",
        type=int,
        default=EXPECTED_VOCAB_SIZE,
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=EXPECTED_HIDDEN_SIZE,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    config = SparseResidualFrontierConfig(
        **vars(build_parser().parse_args(argv))
    )
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
