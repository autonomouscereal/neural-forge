#!/usr/bin/env python3
"""Plan and supervise the real LFM2 CPCF-v5 compression frontier.

This is an orchestration boundary, not another fold implementation.  It asks
the catalog for exact dense targets, assigns an isolated attempt namespace to
each ratio, and produces a no-clobber subprocess manifest.  A fixed internal
worker uses the ordinary runner's authenticated post-compile consumer seam to
call ``capture_quarantined_candidate`` with the established LFM2 saver/loader
and 64-case whole-output evaluator.  The ordinary v5 CLI itself remains a
diagnostic surface and is never treated as a capture command.

Planning is useful and safe today.  It loads the local parent only to inspect
its circuit graph and authenticate it against the sealed cache; it performs no
parent forward/generation, training, backward, optimizer, pruning,
quantization, download, or publication operation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# These are process invariants, including during plan-only discovery.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


SUPERVISOR_SCHEMA = "neural-forge-lfm2-cpcf-v5-frontier-supervisor-v1"
LANE_SCHEMA = "neural-forge-lfm2-cpcf-v5-frontier-supervisor-lane-v1"
COMMAND_SCHEMA = "neural-forge-lfm2-cpcf-v5-frontier-command-v1"
EXECUTION_SCHEMA = "neural-forge-lfm2-cpcf-v5-frontier-execution-v1"
FAILURE_EXECUTION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-frontier-execution-failure-v1"
)
DECLARATIVE_RATIOS = (4, 10, 25, 50, 100, 250)
DEFAULT_RTX3090_GPU_SELECTORS = (
    "GPU-4ead7f3a-cc79-fa5b-de42-ef024e27f561",
    "GPU-2ce6a95c-7b1f-644a-3808-d781732a0582",
)

_DIAGNOSTIC_RUNNER = (
    _REPOSITORY_ROOT
    / "experiments"
    / "run_lfm2_cpcf_v5_selective_blank_fold.py"
)
_BLANK_RUNNER = (
    _REPOSITORY_ROOT / "experiments" / "run_lfm2_cpcf_blank_fold.py"
)
_HOLDOUT_CAPTURE = (
    _REPOSITORY_ROOT
    / "experiments"
    / "capture_lfm2_cpcf_holdout_candidate.py"
)
_POSTHOC_EVALUATOR = (
    _REPOSITORY_ROOT
    / "experiments"
    / "evaluate_lfm2_intelligence_retention.py"
)
_GPU_PARITY_RECEIPT = (
    _REPOSITORY_ROOT / "reports" / "12061b4-two-rtx3090-bf16.json"
)
_RESIDUAL_RUNNER = (
    _REPOSITORY_ROOT
    / "experiments"
    / "run_lfm2_residual_attention_fold.py"
)
_FRONTIER_SOURCE = (
    _SOURCE_ROOT
    / "neural_forge"
    / "dynamic_lfm2_cpcf_v5_frontier.py"
)
_CAPTURE_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "dynamic_candidate_capture.py"
)
_LFM2_QUARANTINE_SOURCE = (
    _SOURCE_ROOT
    / "neural_forge"
    / "hf_lfm2_cpcf_v5_quarantine.py"
)
_COMPILER_SOURCE = (
    _SOURCE_ROOT
    / "neural_forge"
    / "hf_lfm2_cpcf_v5_model_compiler.py"
)
_FOLD_ONLY_RUNTIME_GUARD_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "dynamic_fold_executor.py"
)
_INTELLIGENCE_GATE_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "hf_lfm2_intelligence_gate.py"
)
_INTELLIGENCE_HOLDOUT_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "hf_lfm2_intelligence_holdout.py"
)
_SELECTIVE_EVIDENCE_SOURCE = (
    _SOURCE_ROOT
    / "neural_forge"
    / "hf_lfm2_cpcf_selective_evidence.py"
)
_CATALOG_ARTIFACT_SOURCE = (
    _SOURCE_ROOT
    / "neural_forge"
    / "dynamic_evidence_catalog_artifact.py"
)

_SHA256 = frozenset("0123456789abcdef")
_IDENTIFIER_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_.-"
)
_GIB = 1024**3


class Lfm2CpcfV5FrontierSupervisorError(RuntimeError):
    """The requested frontier plan or execution is not safe and complete."""


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
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Supervisor metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a positive integer."
        )
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a positive finite number."
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a positive finite number."
        ) from exc
    if not 0.0 < result < float("inf"):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a positive finite number."
        )
    return result


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not value[0].isalnum()
        or any(character not in _IDENTIFIER_CHARACTERS for character in value)
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must begin with an alphanumeric character and contain "
            "only alphanumerics, dot, underscore, or hyphen."
        )
    return value


def _path_text(value: str | os.PathLike[str], name: str) -> str:
    raw = os.fspath(value)
    if (
        not raw
        or "://" in raw
        or raw.startswith("\\\\")
        or raw.startswith("//")
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be an explicit local path."
        )
    return raw


def _existing_local_path(
    value: str | os.PathLike[str],
    *,
    name: str,
    directory: bool,
) -> Path:
    raw = _path_text(value, name)
    lexical = Path(raw).expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} cannot be a symbolic link."
        )
    try:
        result = lexical.resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} does not exist."
        ) from exc
    valid = result.is_dir() if directory else result.is_file()
    if not valid:
        expected = "directory" if directory else "file"
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be an existing local {expected}."
        )
    return result


def _new_local_root(
    value: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    raw = _path_text(value, name)
    lexical = Path(raw).expanduser().absolute()
    if lexical.exists() and lexical.is_symlink():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} cannot be a symbolic link."
        )
    ancestor = lexical
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} has an unsafe existing ancestor."
        )
    return lexical


def _local_file_target(
    value: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    raw = _path_text(value, name)
    lexical = Path(raw).expanduser().absolute()
    if lexical.is_symlink():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} cannot be a symbolic link."
        )
    if lexical.exists() and not lexical.is_file():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a regular file path."
        )
    ancestor = lexical.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} has an unsafe existing ancestor."
        )
    return lexical


def _paths_overlap(left: Path, right: Path) -> bool:
    left_value = left.expanduser().absolute().resolve(strict=False)
    right_value = right.expanduser().absolute().resolve(strict=False)
    return (
        left_value == right_value
        or left_value in right_value.parents
        or right_value in left_value.parents
    )


def _assert_catalog_store_disjoint(
    store: Path,
    *,
    roots: Mapping[str, Path],
) -> None:
    overlaps = [
        name
        for name, path in roots.items()
        if _paths_overlap(store, path)
    ]
    if overlaps:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "selective catalog store cannot equal, contain, or be contained "
            "by protected model, tokenizer, trace, work, resume, or "
            "quarantine roots: "
            + ", ".join(sorted(overlaps))
            + "."
        )


def _assert_catalog_receipt_disjoint(
    receipt: Path,
    *,
    roots: Mapping[str, Path],
) -> None:
    overlaps = [
        name
        for name, path in roots.items()
        if _paths_overlap(receipt, path)
    ]
    if overlaps:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "selective catalog receipt cannot equal, contain, or be contained "
            "by protected model, tokenizer, trace, corpus, envelope, work, "
            "resume, quarantine, catalog-store, or output paths: "
            + ", ".join(sorted(overlaps))
            + "."
        )


def _freeze_mapping(
    value: Mapping[str, Any] | None,
    *,
    name: str,
) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} must be a mapping."
        )
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class FrontierRuntimeLimits:
    """Resource limits for one isolated ratio process."""

    maximum_process_rss_gib: float = 64.0
    maximum_logical_resident_gib: float = 0.0
    maximum_cuda_gib_per_device: float = 20.0
    minimum_physical_rss_reduction_fraction: float = 0.10
    cpu_threads: int = 16
    gpu_selectors: tuple[str, str] = DEFAULT_RTX3090_GPU_SELECTORS

    def __post_init__(self) -> None:
        for name in (
            "maximum_process_rss_gib",
            "maximum_cuda_gib_per_device",
        ):
            object.__setattr__(
                self,
                name,
                _positive_float(getattr(self, name), name),
            )
        logical = float(self.maximum_logical_resident_gib)
        if logical < 0.0 or logical == float("inf"):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "maximum_logical_resident_gib must be finite and non-negative."
            )
        object.__setattr__(
            self,
            "maximum_logical_resident_gib",
            logical,
        )
        reduction = float(self.minimum_physical_rss_reduction_fraction)
        if not 0.0 < reduction < 1.0:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "minimum_physical_rss_reduction_fraction must be in (0, 1)."
            )
        object.__setattr__(
            self,
            "minimum_physical_rss_reduction_fraction",
            reduction,
        )
        object.__setattr__(
            self,
            "cpu_threads",
            _positive_int(self.cpu_threads, "cpu_threads"),
        )
        selectors = tuple(self.gpu_selectors)
        if (
            len(selectors) != 2
            or len(set(selectors)) != 2
            or any(
                not isinstance(value, str) or not value.strip()
                for value in selectors
            )
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Exactly two distinct RTX-3090 GPU selectors are required."
            )
        object.__setattr__(
            self,
            "gpu_selectors",
            (selectors[0].strip(), selectors[1].strip()),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "maximum_process_rss_gib": self.maximum_process_rss_gib,
            "maximum_process_rss_bytes": int(
                self.maximum_process_rss_gib * _GIB
            ),
            "maximum_logical_resident_gib": (
                self.maximum_logical_resident_gib
            ),
            "maximum_cuda_gib_per_device": (
                self.maximum_cuda_gib_per_device
            ),
            "maximum_cuda_bytes_per_device": int(
                self.maximum_cuda_gib_per_device * _GIB
            ),
            "minimum_physical_rss_reduction_fraction": (
                self.minimum_physical_rss_reduction_fraction
            ),
            "cpu_threads": self.cpu_threads,
            "physical_gpu_selectors": list(self.gpu_selectors),
            "logical_gpu_indices_inside_fold": [0, 1],
            "both_logical_gpus_required": True,
        }


@dataclass(frozen=True, slots=True)
class FrontierSupervisorConfig:
    """All local pins needed to plan one frontier or one isolated lane."""

    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    parent_envelope: Path
    work_directory: Path
    expected_parent_graph_fingerprint: str
    expected_bridge_artifact_sha256: str
    expected_cache_sha256: str
    expected_parent_envelope_file_sha256: str
    lifecycle: Any
    compression_ratios: tuple[int, ...] = DECLARATIVE_RATIOS
    approved_review_bindings: Mapping[str, Any] = field(
        default_factory=dict
    )
    resume_references: Mapping[str, Any] = field(default_factory=dict)
    runtime_limits: FrontierRuntimeLimits = field(
        default_factory=FrontierRuntimeLimits
    )
    parent_name: str = "LiquidAI-LFM2.5-230M"
    child_name_prefix: str = "lfm2-cpcf-v5"
    blank_seed: int = 0
    minimum_hidden_size: int = 32
    maximum_candidates_per_ratio: int = 8
    attempt_id: str = "attempt-0001"
    python_executable: str = sys.executable
    selective_catalog_store_directory: Path | None = None
    selective_catalog_receipt_json: Path | None = None
    expected_selective_catalog_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
            Lfm2CpcfV5FrontierLifecycle,
        )

        for name in (
            "parent_model_directory",
            "donor_tokenizer_directory",
            "target_tokenizer_directory",
            "bridge_directory",
            "trace_cache_directory",
        ):
            object.__setattr__(
                self,
                name,
                _existing_local_path(
                    getattr(self, name),
                    name=name,
                    directory=True,
                ),
            )
        for name in ("trajectory_corpus", "parent_envelope"):
            object.__setattr__(
                self,
                name,
                _existing_local_path(
                    getattr(self, name),
                    name=name,
                    directory=False,
                ),
            )
        object.__setattr__(
            self,
            "work_directory",
            _new_local_root(
                self.work_directory,
                name="work_directory",
            ),
        )
        catalog_store = (
            self.work_directory.parent / "selective-catalogs"
            if self.selective_catalog_store_directory is None
            else _new_local_root(
                self.selective_catalog_store_directory,
                name="selective_catalog_store_directory",
            )
        )
        catalog_receipt_candidate = (
            self.work_directory.parent
            / "selective-catalog-preparation.json"
            if self.selective_catalog_receipt_json is None
            else self.selective_catalog_receipt_json
        )
        catalog_receipt = _local_file_target(
            catalog_receipt_candidate,
            name="selective_catalog_receipt_json",
        )
        _assert_catalog_store_disjoint(
            catalog_store,
            roots={
                "parent_model_directory": self.parent_model_directory,
                "donor_tokenizer_directory": (
                    self.donor_tokenizer_directory
                ),
                "target_tokenizer_directory": (
                    self.target_tokenizer_directory
                ),
                "bridge_directory": self.bridge_directory,
                "trace_cache_directory": self.trace_cache_directory,
                "work_directory": self.work_directory,
                "selective_catalog_receipt_json": catalog_receipt,
            },
        )
        _assert_catalog_receipt_disjoint(
            catalog_receipt,
            roots={
                "parent_model_directory": self.parent_model_directory,
                "donor_tokenizer_directory": (
                    self.donor_tokenizer_directory
                ),
                "target_tokenizer_directory": (
                    self.target_tokenizer_directory
                ),
                "bridge_directory": self.bridge_directory,
                "trace_cache_directory": self.trace_cache_directory,
                "trajectory_corpus": self.trajectory_corpus,
                "trajectory_corpus_parent": (
                    self.trajectory_corpus.parent
                ),
                "parent_envelope": self.parent_envelope,
                "parent_envelope_parent": self.parent_envelope.parent,
                "work_directory": self.work_directory,
                "selective_catalog_store_directory": catalog_store,
            },
        )
        object.__setattr__(
            self,
            "selective_catalog_store_directory",
            catalog_store,
        )
        object.__setattr__(
            self,
            "selective_catalog_receipt_json",
            catalog_receipt,
        )
        for name in (
            "expected_parent_graph_fingerprint",
            "expected_bridge_artifact_sha256",
            "expected_cache_sha256",
            "expected_parent_envelope_file_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        if self.expected_selective_catalog_receipt_sha256 is not None:
            object.__setattr__(
                self,
                "expected_selective_catalog_receipt_sha256",
                _digest(
                    self.expected_selective_catalog_receipt_sha256,
                    "expected_selective_catalog_receipt_sha256",
                ),
            )
        if type(self.lifecycle) is not Lfm2CpcfV5FrontierLifecycle:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "lifecycle must be Lfm2CpcfV5FrontierLifecycle."
            )
        ratios = tuple(self.compression_ratios)
        if (
            not ratios
            or len(set(ratios)) != len(ratios)
            or any(value not in DECLARATIVE_RATIOS for value in ratios)
            or tuple(
                value for value in DECLARATIVE_RATIOS if value in ratios
            )
            != ratios
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "compression_ratios must be an ordered nonempty subset of "
                "the declarative (4, 10, 25, 50, 100, 250) frontier."
            )
        object.__setattr__(self, "compression_ratios", ratios)
        object.__setattr__(
            self,
            "approved_review_bindings",
            _freeze_mapping(
                self.approved_review_bindings,
                name="approved_review_bindings",
            ),
        )
        object.__setattr__(
            self,
            "resume_references",
            _freeze_mapping(
                self.resume_references,
                name="resume_references",
            ),
        )
        if not isinstance(self.runtime_limits, FrontierRuntimeLimits):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "runtime_limits must be FrontierRuntimeLimits."
            )
        for name in ("parent_name", "child_name_prefix"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise Lfm2CpcfV5FrontierSupervisorError(
                    f"{name} must be nonempty text."
                )
            object.__setattr__(self, name, value.strip())
        if (
            isinstance(self.blank_seed, bool)
            or not isinstance(self.blank_seed, int)
            or self.blank_seed < 0
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "blank_seed must be a non-negative integer."
            )
        object.__setattr__(
            self,
            "minimum_hidden_size",
            _positive_int(
                self.minimum_hidden_size,
                "minimum_hidden_size",
            ),
        )
        object.__setattr__(
            self,
            "maximum_candidates_per_ratio",
            _positive_int(
                self.maximum_candidates_per_ratio,
                "maximum_candidates_per_ratio",
            ),
        )
        object.__setattr__(
            self,
            "attempt_id",
            _identifier(self.attempt_id, "attempt_id"),
        )
        executable = _path_text(
            self.python_executable,
            "python_executable",
        )
        object.__setattr__(self, "python_executable", executable)

    def input_paths(self) -> dict[str, str]:
        return {
            "parent_model_directory": str(self.parent_model_directory),
            "donor_tokenizer_directory": str(
                self.donor_tokenizer_directory
            ),
            "target_tokenizer_directory": str(
                self.target_tokenizer_directory
            ),
            "bridge_directory": str(self.bridge_directory),
            "trace_cache_directory": str(self.trace_cache_directory),
            "trajectory_corpus": str(self.trajectory_corpus),
            "parent_envelope": str(self.parent_envelope),
            "selective_catalog_store_directory": str(
                self.selective_catalog_store_directory
            ),
            "selective_catalog_receipt_json": str(
                self.selective_catalog_receipt_json
            ),
        }


@dataclass(frozen=True, slots=True)
class FrontierEvidenceBindings:
    """Authenticated local identities discovered before target planning."""

    parent_model_identity_sha256: str
    donor_tokenizer_package_sha256: str
    cache_sha256: str
    corpus_sha256: str
    parent_envelope_file_sha256: str
    parent_envelope_sha256: str
    parent_envelope_document_sha256: str
    holdout_manifest_sha256: str
    holdout_manifest_document_sha256: str
    holdout_case_count: int
    selective_catalog_preparation: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        for name in (
            "parent_model_identity_sha256",
            "donor_tokenizer_package_sha256",
            "cache_sha256",
            "corpus_sha256",
            "parent_envelope_file_sha256",
            "parent_envelope_sha256",
            "parent_envelope_document_sha256",
            "holdout_manifest_sha256",
            "holdout_manifest_document_sha256",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "holdout_case_count",
            _positive_int(self.holdout_case_count, "holdout_case_count"),
        )
        if self.holdout_case_count != 64:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "The behavioral gate must be the exact 64-case holdout."
            )
        catalog = dict(self.selective_catalog_preparation)
        if catalog:
            required_catalog_digests = (
                "preparation_receipt_sha256",
                "binding_sha256",
                "artifact_manifest_sha256",
                "artifact_catalog_sha256",
                "selective_catalog_sha256",
                "selective_source_sha256",
                "selective_provider_sha256",
                "codec_sha256",
                "provider_implementation_sha256",
                "artifact_store_source_sha256",
                "selective_provider_source_sha256",
            )
            for name in required_catalog_digests:
                _digest(
                    catalog.get(name),
                    f"selective_catalog_preparation.{name}",
                )
            if (
                catalog.get("status")
                != "prepared_authenticated_tensor_free_catalog"
                or catalog.get("warm_hit_required_for_production_fold")
                is not True
                or catalog.get("cold_rebuild_permitted_in_production_fold")
                is not False
            ):
                raise Lfm2CpcfV5FrontierSupervisorError(
                    "Selective catalog preparation lifecycle is invalid."
                )
        object.__setattr__(
            self,
            "selective_catalog_preparation",
            MappingProxyType(catalog),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model_identity_sha256": (
                self.parent_model_identity_sha256
            ),
            "donor_tokenizer_package_sha256": (
                self.donor_tokenizer_package_sha256
            ),
            "cache_sha256": self.cache_sha256,
            "corpus_sha256": self.corpus_sha256,
            "parent_envelope_file_sha256": (
                self.parent_envelope_file_sha256
            ),
            "parent_envelope_sha256": self.parent_envelope_sha256,
            "parent_envelope_document_sha256": (
                self.parent_envelope_document_sha256
            ),
            "holdout_manifest_sha256": self.holdout_manifest_sha256,
            "holdout_manifest_document_sha256": (
                self.holdout_manifest_document_sha256
            ),
            "holdout_case_count": self.holdout_case_count,
            "selective_catalog_preparation": dict(
                self.selective_catalog_preparation
            ),
            "selective_catalog_prepared": bool(
                self.selective_catalog_preparation
            ),
            "parent_outputs_cached_once": True,
            "parent_trace_generation_permitted": False,
        }


@dataclass(frozen=True, slots=True)
class FrontierLanePaths:
    run_directory: Path
    diagnostic_json: Path
    resume_directory: Path
    quarantine_directory: Path
    process_manifest_json: Path
    execution_receipt_json: Path
    resume_reopen_allowed: bool = False

    def __post_init__(self) -> None:
        paths = (
            self.run_directory,
            self.diagnostic_json,
            self.resume_directory,
            self.quarantine_directory,
            self.process_manifest_json,
            self.execution_receipt_json,
        )
        if any(not isinstance(path, Path) for path in paths):
            raise TypeError("Frontier lane paths must be pathlib.Path values.")
        if len(set(paths)) != len(paths):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Diagnostic, resume, quarantine, and manifest paths must differ."
            )
        if not isinstance(self.resume_reopen_allowed, bool):
            raise TypeError("resume_reopen_allowed must be bool.")
        create_new_paths = (
            self.run_directory,
            self.diagnostic_json,
            self.quarantine_directory,
            self.process_manifest_json,
            self.execution_receipt_json,
        )
        if any(
            path.exists() or path.is_symlink()
            for path in create_new_paths
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "A planned lane path already exists; refusing overwrite."
            )
        resume_exists = (
            self.resume_directory.exists()
            or self.resume_directory.is_symlink()
        )
        if resume_exists and (
            not self.resume_reopen_allowed
            or self.resume_directory.is_symlink()
            or not self.resume_directory.is_dir()
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "The plan-scoped resume path may only reopen as an ordinary "
                "directory under an authenticated resume reference."
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_directory": str(self.run_directory),
            "diagnostic_json": str(self.diagnostic_json),
            "resume_directory": str(self.resume_directory),
            "quarantine_directory": str(self.quarantine_directory),
            "process_manifest_json": str(self.process_manifest_json),
            "execution_receipt_json": str(self.execution_receipt_json),
            "resume_reopen_allowed": self.resume_reopen_allowed,
        }


@dataclass(frozen=True, slots=True)
class FrontierSupervisorLane:
    """One exact catalog lane plus its fail-closed process manifest."""

    ratio: int
    catalog_lane: Any
    paths: FrontierLanePaths
    command_manifest: Mapping[str, Any]
    execution_blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_int(self.ratio, "ratio")
        if not isinstance(self.paths, FrontierLanePaths):
            raise TypeError("paths must be FrontierLanePaths.")
        manifest = dict(self.command_manifest)
        if manifest.get("schema") != COMMAND_SCHEMA:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "command_manifest has the wrong schema."
            )
        object.__setattr__(
            self,
            "command_manifest",
            MappingProxyType(manifest),
        )
        blockers = tuple(self.execution_blockers)
        object.__setattr__(self, "execution_blockers", blockers)

    @property
    def execution_ready(self) -> bool:
        return not self.execution_blockers

    def as_dict(self) -> dict[str, Any]:
        lane = self.catalog_lane
        body = {
            "schema": LANE_SCHEMA,
            "ratio": self.ratio,
            "target_plan_fingerprint": lane.target_plan_fingerprint,
            "target_unique_parameters": (
                lane.exact_compression.target_unique_parameters
            ),
            "realized_compression": lane.exact_compression.as_dict(),
            "selected_candidate_rank": lane.selected_candidate_rank,
            "selected_candidate_fingerprint": (
                lane.selected_candidate.fingerprint
            ),
            "selected_candidate_configuration": dict(
                lane.selected_candidate.configuration
            ),
            "lifecycle_contract": lane.lifecycle_contract.as_dict(),
            "paths": self.paths.as_dict(),
            "command_manifest": dict(self.command_manifest),
            "execution_ready": self.execution_ready,
            "execution_blockers": list(self.execution_blockers),
            "structural_diagnostics_can_authorize_promotion": False,
            "promotable": False,
            "publication_authorized": False,
            "standalone_reload_required": True,
            "whole_output_evaluation_required": True,
        }
        return {**body, "lane_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class FrontierSupervisorPlan:
    """A catalog schedule converted to isolated, non-executed process lanes."""

    schedule: Any
    config: FrontierSupervisorConfig
    evidence: FrontierEvidenceBindings
    source_identities: Mapping[str, str]
    lanes: tuple[FrontierSupervisorLane, ...]

    def __post_init__(self) -> None:
        identities = dict(self.source_identities)
        if not identities or any(
            _digest(value, f"source_identities[{name}]") != value
            for name, value in identities.items()
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "source_identities must contain exact file SHA-256 digests."
            )
        object.__setattr__(
            self,
            "source_identities",
            MappingProxyType(identities),
        )
        lanes = tuple(self.lanes)
        if (
            not lanes
            or any(
                not isinstance(lane, FrontierSupervisorLane)
                for lane in lanes
            )
            or tuple(lane.ratio for lane in lanes)
            != self.config.compression_ratios
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Supervisor lanes differ from the requested ratios."
            )
        object.__setattr__(self, "lanes", lanes)

    def lane_for_ratio(self, ratio: int) -> FrontierSupervisorLane:
        matches = tuple(lane for lane in self.lanes if lane.ratio == ratio)
        if len(matches) != 1:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "The requested ratio is not a unique supervisor lane."
            )
        return matches[0]

    def as_dict(self) -> dict[str, Any]:
        all_ready = all(lane.execution_ready for lane in self.lanes)
        body = {
            "schema": SUPERVISOR_SCHEMA,
            "status": (
                "capture_quarantine_processes_ready"
                if all_ready
                else "planned_with_fail_closed_blocked_lanes"
            ),
            "declarative_frontier_ratios": list(DECLARATIVE_RATIOS),
            "requested_ratios": list(self.config.compression_ratios),
            "catalog_schedule": self.schedule.as_dict(),
            "catalog_schedule_sha256": self.schedule.fingerprint,
            "local_inputs": self.config.input_paths(),
            "input_bindings": self.evidence.as_dict(),
            "source_identities": dict(self.source_identities),
            "runtime_limits": self.config.runtime_limits.as_dict(),
            "lanes": [lane.as_dict() for lane in self.lanes],
            "one_isolated_process_per_ratio": True,
            "maximum_parallel_ratio_processes": 1,
            "all_ratio_paths_disjoint": True,
            "all_outputs_no_clobber": True,
            "shared_selective_catalog_store_directory": str(
                self.config.selective_catalog_store_directory
            ),
            "shared_selective_catalog_receipt_json": str(
                self.config.selective_catalog_receipt_json
            ),
            "shared_selective_catalog_preparation_receipt_sha256": (
                self.evidence.selective_catalog_preparation.get(
                    "preparation_receipt_sha256"
                )
            ),
            "all_ratio_workers_require_exact_authenticated_warm_hit": True,
            "catalog_cold_rebuilds_permitted_in_ratio_workers": False,
            "catalog_post_run_reverification_required": True,
            "offline": True,
            "cached_parent_evidence_only": True,
            "parent_trace_generation_calls": 0,
            "parent_generation_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "structural_diagnostics_can_authorize_promotion": False,
            "standalone_reload_required": True,
            "whole_output_evaluation_required": True,
            "manual_review_required_before_reviewed_rerun": True,
            "publication_authorized": False,
        }
        return {**body, "plan_sha256": _sha(body)}


def _source_identities() -> dict[str, str]:
    sources = {
        "dynamic_evidence_catalog_artifact": _CATALOG_ARTIFACT_SOURCE,
        "frontier_supervisor": Path(__file__).resolve(),
        "frontier_catalog": _FRONTIER_SOURCE,
        "gpu_replay_parity_receipt": _GPU_PARITY_RECEIPT,
        "candidate_capture": _CAPTURE_SOURCE,
        "fold_only_runtime_guard": _FOLD_ONLY_RUNTIME_GUARD_SOURCE,
        "lfm2_v5_quarantine_backend": _LFM2_QUARANTINE_SOURCE,
        "v5_compiler": _COMPILER_SOURCE,
        "v5_diagnostic_runner": _DIAGNOSTIC_RUNNER,
        "blank_save_reload_runner": _BLANK_RUNNER,
        "holdout_capture": _HOLDOUT_CAPTURE,
        "tokenizer_package_fingerprint": _POSTHOC_EVALUATOR,
        "local_lfm2_loader": _RESIDUAL_RUNNER,
        "intelligence_gate": _INTELLIGENCE_GATE_SOURCE,
        "intelligence_holdout": _INTELLIGENCE_HOLDOUT_SOURCE,
        "lfm2_selective_evidence_provider": _SELECTIVE_EVIDENCE_SOURCE,
    }
    missing = tuple(name for name, path in sources.items() if not path.is_file())
    if missing:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Required supervisor sources are missing: "
            + ", ".join(missing)
            + "."
        )
    return {
        name: _file_sha256(path)
        for name, path in sorted(sources.items())
    }


def authenticate_frontier_catalog_preparation(
    config: FrontierSupervisorConfig,
) -> Mapping[str, Any]:
    """Authenticate the exact shared warm catalog before model allocation."""

    if not isinstance(config, FrontierSupervisorConfig):
        raise TypeError("config must be FrontierSupervisorConfig.")
    expected_receipt = config.expected_selective_catalog_receipt_sha256
    if expected_receipt is None:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Planning or executing a frontier requires the exact prepared "
            "selective catalog receipt SHA-256 pin."
        )
    from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner

    try:
        preparation = dict(
            runner.authenticate_prepared_selective_catalog(
                trace_cache_directory=config.trace_cache_directory,
                selective_catalog_store_directory=(
                    config.selective_catalog_store_directory
                ),
                selective_catalog_receipt_json=(
                    config.selective_catalog_receipt_json
                ),
                expected_selective_catalog_receipt_sha256=expected_receipt,
            )
        )
    except Exception as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "The shared selective catalog preparation failed exact "
            "authentication."
        ) from exc
    sources = _source_identities()
    expected_source_identities = {
        "dynamic_evidence_catalog_artifact": sources[
            "dynamic_evidence_catalog_artifact"
        ],
        "hf_lfm2_cpcf_selective_evidence": sources[
            "lfm2_selective_evidence_provider"
        ],
    }
    if (
        preparation.get("preparation_receipt_sha256")
        != expected_receipt
        or preparation.get("source_identities")
        != expected_source_identities
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "The prepared catalog receipt or its source implementation "
            "differs from the frontier source bundle."
        )
    return MappingProxyType(preparation)


def _authenticate_catalog_post_run_reverification(
    value: Any,
    *,
    catalog_preparation: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    """Authenticate the exact record emitted by the selective fold runner."""

    from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner

    try:
        return runner.authenticate_selective_catalog_post_run_reverification(
            value,
            catalog_preparation=catalog_preparation,
        )
    except Exception as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} differs from the exact runner-emitted selective catalog "
            "post-run reverification record."
        ) from exc


def _read_parent_envelope(
    config: FrontierSupervisorConfig,
    *,
    parent_model_identity_sha256: str,
) -> tuple[Any, Any]:
    from experiments.evaluate_lfm2_intelligence_retention import (
        Lfm2PosthocEvaluationError,
        fingerprint_bound_local_tokenizer,
    )
    from neural_forge.hf_lfm2_intelligence_gate import (
        load_lfm2_intelligence_holdout_manifest,
    )
    from neural_forge.hf_lfm2_intelligence_holdout import (
        load_lfm2_intelligence_holdout_parent_envelope,
    )

    observed_file_sha = _file_sha256(config.parent_envelope)
    if observed_file_sha != config.expected_parent_envelope_file_sha256:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Parent-envelope file differs from its independent SHA-256 pin."
        )
    try:
        document = json.loads(config.parent_envelope.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Parent envelope is not readable canonical JSON."
        ) from exc
    if not isinstance(document, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Parent envelope must be a JSON object."
        )
    try:
        donor_tokenizer_package_sha256, _donor_tokenizer_files = (
            fingerprint_bound_local_tokenizer(
                config.donor_tokenizer_directory
            )
        )
    except (Lfm2PosthocEvaluationError, OSError) as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "The bound donor tokenizer package is not safe or readable."
        ) from exc
    manifest = load_lfm2_intelligence_holdout_manifest()
    required = (
        "envelope_sha256",
        "parent_state_fingerprint",
        "parent_model_fingerprint",
        "tokenizer_fingerprint",
    )
    if any(not isinstance(document.get(name), str) for name in required):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Parent envelope is missing its typed identity fields."
        )
    envelope = load_lfm2_intelligence_holdout_parent_envelope(
        config.parent_envelope,
        expected_envelope_sha256=document["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=(
            document["parent_state_fingerprint"]
        ),
        expected_parent_model_fingerprint=(
            document["parent_model_fingerprint"]
        ),
        expected_tokenizer_fingerprint=document["tokenizer_fingerprint"],
    )
    if (
        envelope.parent_model_fingerprint
        != parent_model_identity_sha256
        or envelope.tokenizer_fingerprint
        != donor_tokenizer_package_sha256
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Parent holdout envelope and folding trace cache do not share "
            "the same parent model identity, or the envelope and live donor "
            "tokenizer package do not share the same package identity."
        )
    return manifest, envelope


def discover_frontier_schedule(
    config: FrontierSupervisorConfig,
) -> tuple[Any, FrontierEvidenceBindings]:
    """Inspect the local parent once, authenticate evidence, and plan targets."""

    catalog_preparation = authenticate_frontier_catalog_preparation(config)
    from experiments.run_lfm2_residual_attention_fold import (
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        plan_lfm2_cpcf_v5_frontier,
    )
    from neural_forge.dynamic_model_inspector import inspect_model_circuits
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )

    try:
        import torch
    except ImportError as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Local planning requires the repository's PyTorch runtime."
        ) from exc

    donor = load_local_tokenizer(config.donor_tokenizer_directory)
    target = load_local_tokenizer(config.target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        config.bridge_directory,
        donor_tokenizer=donor,
        target_tokenizer=target,
        expected_artifact_sha256=config.expected_bridge_artifact_sha256,
    )
    if (
        fingerprint_hf_tokenizer_semantics(donor)
        != bridge.donor_tokenizer_semantics_sha256
        or fingerprint_hf_tokenizer_semantics(target)
        != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Live local tokenizer semantics differ from the strict bridge."
        )
    cache = open_lfm2_circuit_trace_cache(
        config.trace_cache_directory,
        verify=False,
    )
    if cache.cache_key != config.expected_cache_sha256:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Trace cache differs from its independent cache-key pin."
        )
    parent = load_local_lfm2_model(
        config.parent_model_directory,
        device="cpu",
        dtype=torch.bfloat16,
    )
    try:
        parent.requires_grad_(False)
        parent.eval()
        parent_identity = fingerprint_lfm2_parent(parent).identity_sha256
        if parent_identity != cache.parent_fingerprint:
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Local parent state differs from the sealed trace cache."
            )
        parent_graph = inspect_model_circuits(
            parent,
            model_name=config.parent_name,
            tokenizer_fingerprint=(
                bridge.donor_tokenizer_semantics_sha256
            ),
        )
        if (
            parent_graph.fingerprint
            != config.expected_parent_graph_fingerprint
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Inspected parent graph differs from its independent pin."
            )
        schedule = plan_lfm2_cpcf_v5_frontier(
            parent_graph=parent_graph,
            expected_parent_graph_fingerprint=(
                config.expected_parent_graph_fingerprint
            ),
            tokenizer_bridge=bridge,
            expected_bridge_artifact_sha256=(
                config.expected_bridge_artifact_sha256
            ),
            compression_ratios=config.compression_ratios,
            lifecycle=config.lifecycle,
            approved_review_bindings=config.approved_review_bindings,
            resume_references=config.resume_references,
            child_name_prefix=config.child_name_prefix,
            blank_seed=config.blank_seed,
            maximum_candidates_per_ratio=(
                config.maximum_candidates_per_ratio
            ),
            maximum_parallel_lanes=1,
            minimum_hidden_size=config.minimum_hidden_size,
        )
    finally:
        del parent
        gc.collect()

    manifest, envelope = _read_parent_envelope(
        config,
        parent_model_identity_sha256=cache.parent_fingerprint,
    )
    corpus_sha = _file_sha256(config.trajectory_corpus)
    evidence = FrontierEvidenceBindings(
        parent_model_identity_sha256=cache.parent_fingerprint,
        donor_tokenizer_package_sha256=(
            envelope.tokenizer_fingerprint
        ),
        cache_sha256=cache.cache_key,
        corpus_sha256=corpus_sha,
        parent_envelope_file_sha256=(
            config.expected_parent_envelope_file_sha256
        ),
        parent_envelope_sha256=envelope.envelope_sha256,
        parent_envelope_document_sha256=envelope.document_sha256,
        holdout_manifest_sha256=manifest.manifest_sha256,
        holdout_manifest_document_sha256=manifest.document_sha256,
        holdout_case_count=len(manifest.cases),
        selective_catalog_preparation=catalog_preparation,
    )
    return schedule, evidence


def _lane_paths(
    config: FrontierSupervisorConfig,
    lane: Any,
    ratio: int,
) -> FrontierLanePaths:
    lifecycle = config.lifecycle.value
    plan_root = (
        config.work_directory
        / lifecycle
        / (
            f"{ratio:03d}x-"
            f"{lane.exact_compression.target_unique_parameters}-"
            f"{lane.target_plan_fingerprint[:12]}"
        )
    )
    run = plan_root / config.attempt_id
    return FrontierLanePaths(
        run_directory=run,
        diagnostic_json=run / "diagnostic.json",
        resume_directory=(
            plan_root
            / (
                "resume-"
                f"{lane.identity.resume_namespace_sha256[:16]}"
            )
        ),
        quarantine_directory=(
            run
            / (
                "quarantine-"
                f"{lane.identity.quarantine_namespace_sha256[:16]}"
            )
        ),
        process_manifest_json=run / "process-manifest.json",
        execution_receipt_json=run / "execution-receipt.json",
        resume_reopen_allowed=lane.resume_reference is not None,
    )


def _diagnostic_command(
    config: FrontierSupervisorConfig,
    lane: Any,
    paths: FrontierLanePaths,
    evidence: FrontierEvidenceBindings,
) -> tuple[str, ...]:
    requested = lane.exact_compression.requested_ratio
    if requested.denominator != 1:
        ratio_text = (
            f"{requested.numerator}/{requested.denominator}"
        )
    else:
        ratio_text = str(requested.numerator)
    command = [
        config.python_executable,
        str(_DIAGNOSTIC_RUNNER),
        "--parent-model-directory",
        str(config.parent_model_directory),
        "--donor-tokenizer-directory",
        str(config.donor_tokenizer_directory),
        "--target-tokenizer-directory",
        str(config.target_tokenizer_directory),
        "--bridge-directory",
        str(config.bridge_directory),
        "--trace-cache-directory",
        str(config.trace_cache_directory),
        "--trajectory-corpus",
        str(config.trajectory_corpus),
        "--diagnostic-json",
        str(paths.diagnostic_json),
        "--compression-ratio",
        ratio_text,
        "--maximum-process-rss-gib",
        str(config.runtime_limits.maximum_process_rss_gib),
        "--maximum-logical-resident-gib",
        str(config.runtime_limits.maximum_logical_resident_gib),
        "--maximum-cuda-gib-per-device",
        str(config.runtime_limits.maximum_cuda_gib_per_device),
        "--minimum-physical-rss-reduction-fraction",
        str(
            config.runtime_limits.minimum_physical_rss_reduction_fraction
        ),
        "--model-device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--parent-name",
        config.parent_name,
        "--child-name",
        lane.target_plan.target_request.child_name,
        "--blank-seed",
        str(config.blank_seed),
        "--minimum-hidden-size",
        str(config.minimum_hidden_size),
        "--maximum-candidates",
        str(config.maximum_candidates_per_ratio),
        "--expected-target-plan-fingerprint",
        lane.target_plan_fingerprint,
        "--resume-store-directory",
        str(paths.resume_directory),
        "--selective-catalog-store-directory",
        str(config.selective_catalog_store_directory),
        "--selective-catalog-receipt-json",
        str(config.selective_catalog_receipt_json),
        "--expected-selective-catalog-receipt-sha256",
        evidence.selective_catalog_preparation[
            "preparation_receipt_sha256"
        ],
    ]
    if lane.resume_reference is not None:
        command.extend(
            (
                "--resume-admission-sha256",
                lane.resume_reference.resume_admission_sha256,
            )
        )
    return tuple(command)


def _environment(config: FrontierSupervisorConfig) -> dict[str, str]:
    threads = str(config.runtime_limits.cpu_threads)
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_VISIBLE_DEVICES": ",".join(
            config.runtime_limits.gpu_selectors
        ),
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": str(config.blank_seed),
    }


def _worker_config_record(
    config: FrontierSupervisorConfig,
    *,
    ratio: int,
) -> dict[str, Any]:
    reviews = {
        key: value.as_dict()
        for key, value in config.approved_review_bindings.items()
    }
    resumes = {
        key: value.as_dict()
        for key, value in config.resume_references.items()
    }
    return {
        **config.input_paths(),
        "work_directory": str(config.work_directory),
        "expected_parent_graph_fingerprint": (
            config.expected_parent_graph_fingerprint
        ),
        "expected_bridge_artifact_sha256": (
            config.expected_bridge_artifact_sha256
        ),
        "expected_cache_sha256": config.expected_cache_sha256,
        "expected_parent_envelope_file_sha256": (
            config.expected_parent_envelope_file_sha256
        ),
        "expected_selective_catalog_receipt_sha256": (
            config.expected_selective_catalog_receipt_sha256
        ),
        "lifecycle": config.lifecycle.value,
        "compression_ratios": [ratio],
        "approved_review_bindings": reviews,
        "resume_references": resumes,
        "runtime_limits": config.runtime_limits.as_dict(),
        "parent_name": config.parent_name,
        "child_name_prefix": config.child_name_prefix,
        "blank_seed": config.blank_seed,
        "minimum_hidden_size": config.minimum_hidden_size,
        "maximum_candidates_per_ratio": (
            config.maximum_candidates_per_ratio
        ),
        "attempt_id": config.attempt_id,
        "python_executable": config.python_executable,
    }


def _worker_command(
    config: FrontierSupervisorConfig,
    paths: FrontierLanePaths,
) -> tuple[str, ...]:
    return (
        config.python_executable,
        str(Path(__file__).resolve()),
        "--internal-worker-manifest",
        str(paths.process_manifest_json),
    )


def _capture_template(
    *,
    config: FrontierSupervisorConfig,
    evidence: FrontierEvidenceBindings,
    lane: Any,
    source_identities: Mapping[str, str],
) -> dict[str, Any]:
    """Describe only bindings that can be known before live callbacks exist."""

    from neural_forge.dynamic_candidate_capture import (
        QUARANTINE_CAPTURE_SCHEMA,
        QUARANTINE_REQUEST_SCHEMA,
        capture_quarantined_candidate,
        fingerprint_quarantine_callable,
    )

    reviewed = lane.lifecycle_contract.approved_review_bindings
    return {
        "request_schema": QUARANTINE_REQUEST_SCHEMA,
        "capture_schema": QUARANTINE_CAPTURE_SCHEMA,
        "capture_api": (
            f"{capture_quarantined_candidate.__module__}."
            f"{capture_quarantined_candidate.__name__}"
        ),
        "callable_fingerprint_api": (
            f"{fingerprint_quarantine_callable.__module__}."
            f"{fingerprint_quarantine_callable.__name__}"
        ),
        "resolved_plan_and_input_bindings": {
            "plan_sha256": lane.target_plan_fingerprint,
            "parent_sha256": evidence.parent_model_identity_sha256,
            "cache_sha256": evidence.cache_sha256,
            "bridge_sha256": (
                config.expected_bridge_artifact_sha256
            ),
            "corpus_sha256": evidence.corpus_sha256,
            "expected_model_state_sha256": (
                None
                if reviewed is None
                else reviewed.approved_candidate_state_sha256
            ),
            "expected_output_sha256": (
                None
                if reviewed is None
                else reviewed.approved_candidate_outputs_sha256
            ),
        },
        "prelaunch_source_bundle_sha256": _sha(dict(source_identities)),
        "unresolved_live_callable_bindings": {
            "compiler_sha256": None,
            "source_sha256": None,
            "saver_sha256": None,
            "loader_sha256": None,
            "evaluator_sha256": None,
        },
        "quarantine_capture_request_constructed": False,
        "reason_request_not_constructed": (
            "QuarantineCaptureRequest is intentionally constructed only in "
            "the isolated worker after the compiler exposes its authenticated "
            "live-child context and the fixed backend instantiates the exact "
            "saver, loader, and evaluator callbacks."
        ),
        "required_reuse": {
            "saver_and_reload_alias_logic": (
                "experiments.run_lfm2_cpcf_blank_fold."
                "_save_reload_evaluate"
            ),
            "local_loader": (
                "experiments.run_lfm2_residual_attention_fold."
                "load_local_lfm2_model"
            ),
            "whole_output_generator": (
                "experiments.capture_lfm2_cpcf_holdout_candidate."
                "generate_candidate_rows_once"
            ),
            "semantic_evaluator": (
                "neural_forge.hf_lfm2_intelligence_gate."
                "evaluate_cached_intelligence"
            ),
            "parent_envelope": str(config.parent_envelope),
            "holdout_manifest_sha256": (
                evidence.holdout_manifest_sha256
            ),
            "per_case_manifest_generation_limits_required": True,
            "arbitrary_global_64_token_cap_forbidden": True,
            "exact_token_matching_is_not_acceptance": True,
        },
    }


def _execution_blockers(ratio: int, lifecycle: str) -> tuple[str, ...]:
    del ratio
    blockers: list[str] = []
    if lifecycle == "reviewed_rerun":
        blockers.append(
            "reviewed_rerun requires the separate reviewed behavior-gate "
            "reproduction backend; the quarantine worker is intentionally "
            "capture_quarantine-only."
        )
        return tuple(blockers)
    try:
        from experiments.run_lfm2_cpcf_v5_selective_blank_fold import (
            run_with_compiled_candidate_consumer,
        )
        from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
            Lfm2CpcfV5QuarantineConsumer,
        )
    except (ImportError, AttributeError) as exc:
        blockers.append(
            "The fixed live-child consumer or LFM2 quarantine backend is "
            f"unavailable: {type(exc).__name__}."
        )
    else:
        if not callable(run_with_compiled_candidate_consumer) or not callable(
            Lfm2CpcfV5QuarantineConsumer
        ):
            blockers.append(
                "The fixed live-child consumer or quarantine backend is not "
                "callable."
            )
    return tuple(blockers)


def build_frontier_supervisor_plan(
    schedule: Any,
    config: FrontierSupervisorConfig,
    evidence: FrontierEvidenceBindings,
) -> FrontierSupervisorPlan:
    """Turn one typed catalog schedule into disjoint process manifests."""

    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5FrontierSchedule,
    )

    if type(schedule) is not Lfm2CpcfV5FrontierSchedule:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "schedule must be Lfm2CpcfV5FrontierSchedule."
        )
    if not isinstance(config, FrontierSupervisorConfig):
        raise TypeError("config must be FrontierSupervisorConfig.")
    if not isinstance(evidence, FrontierEvidenceBindings):
        raise TypeError("evidence must be FrontierEvidenceBindings.")
    catalog_preparation = dict(
        evidence.selective_catalog_preparation
    )
    if (
        not catalog_preparation
        or config.expected_selective_catalog_receipt_sha256 is None
        or catalog_preparation.get("preparation_receipt_sha256")
        != config.expected_selective_catalog_receipt_sha256
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Every frontier plan requires the exact shared authenticated "
            "selective catalog preparation."
        )
    if (
        schedule.parent_graph_fingerprint
        != config.expected_parent_graph_fingerprint
        or schedule.bridge_artifact_sha256
        != config.expected_bridge_artifact_sha256
        or schedule.lifecycle is not config.lifecycle
        or schedule.maximum_parallel_lanes != 1
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Schedule differs from the supervisor's graph, bridge, lifecycle, "
            "or one-process-at-a-time policy."
        )
    ratios = tuple(
        int(lane.exact_compression.requested_ratio)
        for lane in schedule.lanes
    )
    if any(
        lane.exact_compression.requested_ratio.denominator != 1
        for lane in schedule.lanes
    ) or ratios != config.compression_ratios:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Schedule ratios differ from the declarative supervisor request."
        )

    sources = _source_identities()
    lanes: list[FrontierSupervisorLane] = []
    all_paths: list[Path] = []
    for ratio, catalog_lane in zip(
        ratios,
        schedule.lanes,
        strict=True,
    ):
        paths = _lane_paths(config, catalog_lane, ratio)
        _assert_catalog_store_disjoint(
            config.selective_catalog_store_directory,
            roots={
                "run_directory": paths.run_directory,
                "diagnostic_json": paths.diagnostic_json,
                "resume_directory": paths.resume_directory,
                "quarantine_directory": paths.quarantine_directory,
            },
        )
        _assert_catalog_receipt_disjoint(
            config.selective_catalog_receipt_json,
            roots={
                "run_directory": paths.run_directory,
                "diagnostic_json": paths.diagnostic_json,
                "resume_directory": paths.resume_directory,
                "quarantine_directory": paths.quarantine_directory,
                "process_manifest_json": paths.process_manifest_json,
                "execution_receipt_json": paths.execution_receipt_json,
                "selective_catalog_store_directory": (
                    config.selective_catalog_store_directory
                ),
            },
        )
        all_paths.extend(
            (
                paths.run_directory,
                paths.diagnostic_json,
                paths.resume_directory,
                paths.quarantine_directory,
                paths.process_manifest_json,
                paths.execution_receipt_json,
            )
        )
        diagnostic_command = _diagnostic_command(
            config,
            catalog_lane,
            paths,
            evidence,
        )
        command = _worker_command(config, paths)
        blockers = _execution_blockers(ratio, config.lifecycle.value)
        command_body = {
            "schema": COMMAND_SCHEMA,
            "kind": "fixed_quarantine_worker_subprocess_manifest",
            "ratio": ratio,
            "target_plan_fingerprint": (
                catalog_lane.target_plan_fingerprint
            ),
            "target_unique_parameters": (
                catalog_lane.exact_compression.target_unique_parameters
            ),
            "selected_candidate_fingerprint": (
                catalog_lane.selected_candidate.fingerprint
            ),
            "selected_candidate_rank": (
                catalog_lane.selected_candidate_rank
            ),
            "command": list(command),
            "diagnostic_runner_command": list(diagnostic_command),
            "environment_overrides": _environment(config),
            "working_directory": str(_REPOSITORY_ROOT),
            "paths": paths.as_dict(),
            "worker_config": _worker_config_record(
                config,
                ratio=ratio,
            ),
            "input_bindings": evidence.as_dict(),
            "selective_catalog_artifact": catalog_preparation,
            "selective_catalog_authenticated_warm_hit_required": True,
            "selective_catalog_cold_rebuild_permitted": False,
            "schedule_bindings": {
                "parent_graph_fingerprint": (
                    schedule.parent_graph_fingerprint
                ),
                "bridge_artifact_sha256": (
                    schedule.bridge_artifact_sha256
                ),
                "bridge_profile_sha256": schedule.bridge_profile_sha256,
                "donor_tokenizer_semantics_sha256": (
                    schedule.donor_tokenizer_semantics_sha256
                ),
                "target_tokenizer_semantics_sha256": (
                    schedule.target_tokenizer_semantics_sha256
                ),
            },
            "source_identities": dict(sources),
            "runtime_limits": config.runtime_limits.as_dict(),
            "quarantine_capture_template": _capture_template(
                config=config,
                evidence=evidence,
                lane=catalog_lane,
                source_identities=sources,
            ),
            "one_ratio_only": True,
            "new_process_required": True,
            "ordinary_diagnostic_runner_cli_used_for_capture": False,
            "fixed_programmatic_consumer_seam_used": True,
            "safe_to_execute_as_capture_run": not blockers,
            "execution_blockers": list(blockers),
            "offline": True,
            "parent_trace_generation_permitted": False,
            "training_permitted": False,
            "backward_permitted": False,
            "optimizer_permitted": False,
            "pruning_permitted": False,
            "quantization_permitted": False,
            "structural_diagnostic_may_promote": False,
        }
        lanes.append(
            FrontierSupervisorLane(
                ratio=ratio,
                catalog_lane=catalog_lane,
                paths=paths,
                command_manifest={
                    **command_body,
                    "command_sha256": _sha(command_body),
                },
                execution_blockers=blockers,
            )
        )
    if len(set(all_paths)) != len(all_paths):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Frontier lane paths overlap across ratios."
        )
    return FrontierSupervisorPlan(
        schedule=schedule,
        config=config,
        evidence=evidence,
        source_identities=sources,
        lanes=tuple(lanes),
    )


def _read_sealed_json(
    path: Path,
    *,
    schema: str,
    seal_field: str,
    name: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} is missing or unsafe."
        )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} is not canonical ASCII JSON."
        ) from exc
    if (
        not isinstance(payload, Mapping)
        or raw != _canonical_json(payload) + b"\n"
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} is not an exact canonical JSON object."
        )
    result = dict(payload)
    claimed = _digest(result.pop(seal_field, None), f"{name}.{seal_field}")
    if result.get("schema") != schema or _sha(result) != claimed:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} schema or self-hash is invalid."
        )
    return {**result, seal_field: claimed}


def _typed_inline_records(
    payload: Any,
    *,
    kind: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{kind} worker records must be a mapping."
        )
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5ApprovedReviewBindings,
        Lfm2CpcfV5ResumeReference,
    )

    value_type = (
        Lfm2CpcfV5ApprovedReviewBindings
        if kind == "approved_review_bindings"
        else Lfm2CpcfV5ResumeReference
    )
    result = {}
    for key, raw in payload.items():
        if not isinstance(raw, Mapping):
            raise Lfm2CpcfV5FrontierSupervisorError(
                f"{kind}[{key!r}] must be an object."
            )
        record = dict(raw)
        record.pop("schema", None)
        record.pop("kind", None)
        record.pop("approved_bindings_are_behavioral_evidence", None)
        instance = value_type(**record)
        if instance.target_plan_fingerprint != key:
            raise Lfm2CpcfV5FrontierSupervisorError(
                f"{kind} worker key differs from its plan fingerprint."
            )
        result[key] = instance
    return result


def _config_from_worker_record(
    value: Any,
) -> FrontierSupervisorConfig:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "worker_config must be an object."
        )
    record = dict(value)
    runtime = record.get("runtime_limits")
    if not isinstance(runtime, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "worker runtime_limits are missing."
        )
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5FrontierLifecycle,
    )

    return FrontierSupervisorConfig(
        parent_model_directory=Path(record["parent_model_directory"]),
        donor_tokenizer_directory=Path(
            record["donor_tokenizer_directory"]
        ),
        target_tokenizer_directory=Path(
            record["target_tokenizer_directory"]
        ),
        bridge_directory=Path(record["bridge_directory"]),
        trace_cache_directory=Path(record["trace_cache_directory"]),
        trajectory_corpus=Path(record["trajectory_corpus"]),
        parent_envelope=Path(record["parent_envelope"]),
        work_directory=Path(record["work_directory"]),
        expected_parent_graph_fingerprint=record[
            "expected_parent_graph_fingerprint"
        ],
        expected_bridge_artifact_sha256=record[
            "expected_bridge_artifact_sha256"
        ],
        expected_cache_sha256=record["expected_cache_sha256"],
        expected_parent_envelope_file_sha256=record[
            "expected_parent_envelope_file_sha256"
        ],
        lifecycle=Lfm2CpcfV5FrontierLifecycle(record["lifecycle"]),
        compression_ratios=tuple(record["compression_ratios"]),
        approved_review_bindings=_typed_inline_records(
            record["approved_review_bindings"],
            kind="approved_review_bindings",
        ),
        resume_references=_typed_inline_records(
            record["resume_references"],
            kind="resume_references",
        ),
        runtime_limits=FrontierRuntimeLimits(
            maximum_process_rss_gib=runtime[
                "maximum_process_rss_gib"
            ],
            maximum_logical_resident_gib=runtime[
                "maximum_logical_resident_gib"
            ],
            maximum_cuda_gib_per_device=runtime[
                "maximum_cuda_gib_per_device"
            ],
            minimum_physical_rss_reduction_fraction=runtime[
                "minimum_physical_rss_reduction_fraction"
            ],
            cpu_threads=runtime["cpu_threads"],
            gpu_selectors=tuple(runtime["physical_gpu_selectors"]),
        ),
        parent_name=record["parent_name"],
        child_name_prefix=record["child_name_prefix"],
        blank_seed=record["blank_seed"],
        minimum_hidden_size=record["minimum_hidden_size"],
        maximum_candidates_per_ratio=record[
            "maximum_candidates_per_ratio"
        ],
        attempt_id=record["attempt_id"],
        python_executable=record["python_executable"],
        selective_catalog_store_directory=Path(
            record["selective_catalog_store_directory"]
        ),
        selective_catalog_receipt_json=Path(
            record["selective_catalog_receipt_json"]
        ),
        expected_selective_catalog_receipt_sha256=record[
            "expected_selective_catalog_receipt_sha256"
        ],
    )


def _validate_worker_environment(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("environment_overrides")
    if not isinstance(expected, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Worker environment overrides are missing."
        )
    changed = {
        name: (os.environ.get(str(name)), value)
        for name, value in expected.items()
        if os.environ.get(str(name)) != value
    }
    if changed:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Worker environment differs from the sealed process manifest."
        )


def _worker_failure_receipt(
    manifest: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    body = {
        "schema": FAILURE_EXECUTION_SCHEMA,
        "status": "failed_closed",
        "ratio": manifest.get("ratio"),
        "target_plan_fingerprint": manifest.get(
            "target_plan_fingerprint"
        ),
        "error_type": (
            f"{type(error).__module__}.{type(error).__qualname__}"
        ),
        "error_message": str(error),
        "publication_authorized": False,
        "promotable": False,
        "zero_operation_audit_authenticated": False,
        "operation_counts_claimed": False,
        "training_steps": None,
        "parameters_pruned": None,
        "parameters_quantized": None,
    }
    return {**body, "receipt_sha256": _sha(body)}


def _authenticate_runtime_guard_receipt(
    value: Any,
    *,
    schema: str,
    guard_source_sha256: str,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} is not a mapping."
        )
    report = dict(value)
    claimed = _digest(
        report.pop("report_sha256", None),
        f"{name} report SHA",
    )
    if report.get("schema") != schema or _sha(report) != claimed:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} is not an authenticated sealed report."
        )
    implementation = report.get("guard_implementation")
    counter_names = (
        "optimizer_constructions",
        "optimizer_steps",
        "backward_calls",
        "autograd_grad_calls",
        "training_steps",
        "pruning_operations",
        "quantization_operations",
    )
    if (
        not isinstance(implementation, Mapping)
        or implementation.get("source_sha256") != guard_source_sha256
        or implementation.get("training_api_firewall") is not True
        or implementation.get("common_pruning_api_firewall") is not True
        or implementation.get("common_quantization_api_firewall") is not True
        or report.get("clean") is not True
        or any(
            type(report.get(field)) is not int or report.get(field) != 0
            for field in counter_names
        )
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{name} does not prove a clean source-bound zero-operation guard."
        )
    return MappingProxyType({**report, "report_sha256": claimed})


def _run_internal_worker(manifest_path: Path) -> int:
    """Execute one fixed capture consumer from one sealed process manifest."""

    manifest: dict[str, Any] | None = None
    runner_config: Any | None = None
    parsed_runner_args: argparse.Namespace | None = None
    try:
        manifest = _read_sealed_json(
            manifest_path,
            schema=COMMAND_SCHEMA,
            seal_field="command_sha256",
            name="worker process manifest",
        )
        if (
            manifest.get("kind")
            != "fixed_quarantine_worker_subprocess_manifest"
            or manifest.get("one_ratio_only") is not True
            or manifest.get("new_process_required") is not True
            or manifest.get("ordinary_diagnostic_runner_cli_used_for_capture")
            is not False
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker manifest is not the fixed one-ratio quarantine contract."
            )
        if manifest.get("source_identities") != _source_identities():
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker source files differ from the sealed launch manifest."
            )
        _validate_worker_environment(manifest)
        config = _config_from_worker_record(manifest["worker_config"])
        if (
            config.lifecycle.value != "capture_quarantine"
            or config.compression_ratios != (manifest["ratio"],)
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker is not one capture_quarantine ratio."
            )
        evidence = manifest.get("input_bindings")
        schedule = manifest.get("schedule_bindings")
        paths = manifest.get("paths")
        if not all(
            isinstance(value, Mapping)
            for value in (evidence, schedule, paths)
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker input, schedule, or path bindings are missing."
            )
        if (
            evidence.get("cache_sha256")
            != config.expected_cache_sha256
            or schedule.get("parent_graph_fingerprint")
            != config.expected_parent_graph_fingerprint
            or schedule.get("bridge_artifact_sha256")
            != config.expected_bridge_artifact_sha256
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker config differs from its schedule or evidence pins."
            )
        catalog_preparation = authenticate_frontier_catalog_preparation(
            config
        )
        if (
            evidence.get("selective_catalog_preparation")
            != dict(catalog_preparation)
            or manifest.get("selective_catalog_artifact")
            != dict(catalog_preparation)
            or manifest.get(
                "selective_catalog_authenticated_warm_hit_required"
            )
            is not True
            or manifest.get("selective_catalog_cold_rebuild_permitted")
            is not False
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker does not bind the exact authenticated shared warm "
                "selective catalog."
            )

        diagnostic_command = manifest.get("diagnostic_runner_command")
        if (
            not isinstance(diagnostic_command, list)
            or len(diagnostic_command) < 3
            or Path(diagnostic_command[1]).resolve() != (
                _DIAGNOSTIC_RUNNER.resolve()
            )
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker diagnostic-runner command is invalid."
            )
        from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner

        parsed_runner_args = runner.build_parser().parse_args(
            diagnostic_command[2:]
        )
        runner_config = runner.config_from_args(parsed_runner_args)
        if (
            runner_config.expected_target_plan_fingerprint
            != manifest["target_plan_fingerprint"]
            or runner_config.selected_candidate_rank is not None
            or runner_config.maximum_cuda_gib_per_device
            != config.runtime_limits.maximum_cuda_gib_per_device
            or Path(runner_config.diagnostic_json)
            != Path(paths["diagnostic_json"])
            or runner_config.selective_catalog_store_directory
            != config.selective_catalog_store_directory
            or runner_config.selective_catalog_receipt_json
            != config.selective_catalog_receipt_json
            or runner_config.expected_selective_catalog_receipt_sha256
            != catalog_preparation["preparation_receipt_sha256"]
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker runner config lost its automatic plan, CUDA, or "
                "diagnostic binding."
            )

        _manifest, parent_envelope = _read_parent_envelope(
            config,
            parent_model_identity_sha256=evidence[
                "parent_model_identity_sha256"
            ],
        )
        if (
            parent_envelope.tokenizer_fingerprint
            != evidence.get("donor_tokenizer_package_sha256")
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker donor tokenizer package differs from the planned "
                "evidence binding."
            )
        from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
            Lfm2CpcfV5QuarantineConfig,
            Lfm2CpcfV5QuarantineConsumer,
        )

        quarantine_config = Lfm2CpcfV5QuarantineConfig(
            destination=Path(paths["quarantine_directory"]),
            parent_envelope=parent_envelope,
            target_tokenizer_semantics_sha256=schedule[
                "target_tokenizer_semantics_sha256"
            ],
            device=runner_config.model_device,
            dtype=runner_config.dtype,
        )
        consumer = Lfm2CpcfV5QuarantineConsumer(quarantine_config)
        result = runner.run_with_compiled_candidate_consumer(
            runner_config,
            consumer,
        )
        candidate = result.candidate
        if (
            result.target_plan_fingerprint
            != manifest["target_plan_fingerprint"]
            or Path(candidate.path).resolve()
            != Path(paths["quarantine_directory"]).resolve()
            or result.compiler_child_checkpoint_saved is not False
            or result.compiler_publication_attempted is not False
            or result.quarantine_candidate_saved is not True
            or result.quarantine_publication_nonpromotable is not True
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker result differs from its plan, quarantine path, or "
                "non-publication lifecycle."
            )
        compiled_catalog = result.compile_diagnostic.get(
            "selective_catalog_artifact"
        )
        compiled_catalog_reverification = result.compile_diagnostic.get(
            "selective_catalog_post_run_reverification"
        )
        authenticated_catalog_reverification = (
            _authenticate_catalog_post_run_reverification(
                compiled_catalog_reverification,
                catalog_preparation=catalog_preparation,
                name="Compiler diagnostic",
            )
        )
        if (
            not isinstance(compiled_catalog, Mapping)
            or dict(compiled_catalog) != dict(catalog_preparation)
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Compiler diagnostic did not retain and freshly reverify the "
                "exact prepared selective catalog."
            )
        blank = result.compile_diagnostic.get("blank_target")
        if (
            not isinstance(blank, Mapping)
            or blank.get("target_plan_fingerprint")
            != manifest["target_plan_fingerprint"]
            or blank.get("target_unique_parameters")
            != manifest["target_unique_parameters"]
            or blank.get("selected_candidate_fingerprint")
            != manifest["selected_candidate_fingerprint"]
            or blank.get("selected_candidate_rank")
            != manifest["selected_candidate_rank"]
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Compiled blank differs from the exact catalog lane."
            )
        resume_reference = config.resume_references.get(
            manifest["target_plan_fingerprint"]
        )
        resume_progress = result.compile_diagnostic.get(
            "resume_progress"
        )
        if not isinstance(resume_progress, Mapping):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Compile diagnostic lacks typed resume progress."
            )
        invocation_head = resume_progress.get("invocation_head")
        if resume_reference is None:
            if (
                resume_progress.get("resume_requested") is not False
                or invocation_head is not None
            ):
                raise Lfm2CpcfV5FrontierSupervisorError(
                    "Cold lane unexpectedly opened resume state."
                )
        else:
            checkpoint_reference = (
                invocation_head.get("checkpoint_reference")
                if isinstance(invocation_head, Mapping)
                else None
            )
            if (
                resume_progress.get("resume_requested") is not True
                or resume_progress.get("requested_admission_sha256")
                != resume_reference.resume_admission_sha256
                or not isinstance(invocation_head, Mapping)
                or invocation_head.get("admission_sha256")
                != resume_reference.resume_admission_sha256
                or not isinstance(checkpoint_reference, Mapping)
                or checkpoint_reference.get("manifest_sha256")
                != resume_reference.resume_artifact_sha256
            ):
                raise Lfm2CpcfV5FrontierSupervisorError(
                    "Typed resume did not open the exact admission and "
                    "checkpoint artifact bound by the catalog lane."
                )
        post_capture_source_identities = _source_identities()
        post_capture_catalog_preparation = (
            authenticate_frontier_catalog_preparation(config)
        )
        if (
            manifest.get("source_identities")
            != post_capture_source_identities
            or dict(post_capture_catalog_preparation)
            != dict(catalog_preparation)
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Worker source files or prepared catalog changed during "
                "compilation or capture."
            )
        runtime_guard = result.compile_diagnostic.get(
            "compiler_runtime_fold_only_guard"
        )
        guard_implementation = (
            runtime_guard.get("guard_implementation")
            if isinstance(runtime_guard, Mapping)
            else None
        )
        if (
            not isinstance(guard_implementation, Mapping)
            or guard_implementation.get("source_sha256")
            != post_capture_source_identities["fold_only_runtime_guard"]
            or guard_implementation.get("training_api_firewall") is not True
            or guard_implementation.get("common_pruning_api_firewall")
            is not True
            or guard_implementation.get("common_quantization_api_firewall")
            is not True
            or runtime_guard.get("clean") is not True
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Runtime fold-only guard identity or zero-operation proof "
                "differs from the post-capture source binding."
            )
        compiler_guard = _authenticate_runtime_guard_receipt(
            runtime_guard,
            schema=(
                "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1"
            ),
            guard_source_sha256=post_capture_source_identities[
                "fold_only_runtime_guard"
            ],
            name="compiler runtime fold-only guard",
        )
        final_reload_verification = dict(
            candidate.verification_receipt
        )
        if (
            final_reload_verification.get("capture_receipt_sha256")
            != result.capture_receipt_sha256
            or final_reload_verification.get(
                "fresh_final_destination_reload_performed"
            )
            is not True
            or final_reload_verification.get("model_state_sha256")
            != result.target_state_sha256
            or final_reload_verification.get("promotable") is not False
            or final_reload_verification.get("publication_authorized")
            is not False
            or final_reload_verification.get("requires_reviewed_rerun")
            is not True
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Final-destination reload verification is absent or differs "
                "from the captured child."
            )
        consumer_guard = _authenticate_runtime_guard_receipt(
            result.consumer_runtime_guard_receipt,
            schema=(
                "neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"
            ),
            guard_source_sha256=post_capture_source_identities[
                "fold_only_runtime_guard"
            ],
            name="consumer runtime fold-only guard",
        )
        if (
            consumer_guard.get("compiler_runtime_guard_sha256")
            != compiler_guard["report_sha256"]
            or consumer_guard.get("capture_verification_sha256")
            != final_reload_verification.get("verification_sha256")
        ):
            raise Lfm2CpcfV5FrontierSupervisorError(
                "Consumer runtime guard differs from the compiler guard or "
                "final retained-destination reload."
            )
        runner._write_diagnostic(
            Path(paths["diagnostic_json"]),
            result.compile_diagnostic,
        )
        body = {
            "schema": EXECUTION_SCHEMA,
            "status": "captured_quarantine_requires_reviewed_rerun",
            "ratio": manifest["ratio"],
            "target_plan_fingerprint": result.target_plan_fingerprint,
            "target_unique_parameters": manifest[
                "target_unique_parameters"
            ],
            "selected_candidate_fingerprint": manifest[
                "selected_candidate_fingerprint"
            ],
            "selected_candidate_rank": manifest[
                "selected_candidate_rank"
            ],
            "compiler_receipt_sha256": result.compiler_receipt_sha256,
            "capture_receipt_sha256": result.capture_receipt_sha256,
            "target_state_sha256": result.target_state_sha256,
            "parent_sha256": result.parent_sha256,
            "cache_sha256": result.cache_sha256,
            "bridge_sha256": result.bridge_sha256,
            "corpus_sha256": result.corpus_sha256,
            "selective_source_sha256": result.source_sha256,
            "compiler_implementation_sha256": result.compiler_sha256,
            "runner_source_sha256": result.runner_source_sha256,
            "source_binding_sha256": result.source_binding_sha256,
            "selective_catalog_artifact": dict(catalog_preparation),
            "selective_catalog_post_run_reverification": dict(
                authenticated_catalog_reverification
            ),
            "post_capture_source_identities": (
                post_capture_source_identities
            ),
            "final_reload_verification": final_reload_verification,
            "consumer_runtime_guard_receipt": dict(consumer_guard),
            "diagnostic_json": paths["diagnostic_json"],
            "quarantine_directory": paths["quarantine_directory"],
            "standalone_candidate": True,
            "standalone_reload_verified": True,
            "whole_output_evaluation_captured": True,
            "manual_review_status": "pending",
            "promotable": False,
            "publication_authorized": False,
            "requires_reviewed_rerun": True,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        receipt = {**body, "receipt_sha256": _sha(body)}
        _atomic_new_json(Path(paths["execution_receipt_json"]), receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    except BaseException as exc:
        if (
            runner_config is not None
            and parsed_runner_args is not None
            and not Path(runner_config.diagnostic_json).exists()
        ):
            try:
                from experiments import (
                    run_lfm2_cpcf_v5_selective_blank_fold as runner,
                )

                failure = runner.build_failure_diagnostic(
                    exc,
                    config=runner_config,
                    parsed_args=parsed_runner_args,
                )
                runner._write_diagnostic(
                    Path(runner_config.diagnostic_json),
                    failure,
                )
            except BaseException:
                pass
        if manifest is not None:
            try:
                paths = manifest.get("paths")
                if isinstance(paths, Mapping):
                    failure_path = Path(paths["execution_receipt_json"])
                    if not failure_path.exists():
                        _atomic_new_json(
                            failure_path,
                            _worker_failure_receipt(manifest, exc),
                        )
            except BaseException:
                pass
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _retained_inventory_matches(
    quarantine_directory: Path,
    inventory: Mapping[str, Any],
) -> bool:
    """Recompute the retained tree's bytes and filesystem identities."""

    from neural_forge.dynamic_candidate_capture import (
        CANDIDATE_DIRECTORY_NAME,
        QUARANTINE_RECEIPT_FILENAME,
    )

    try:
        entries = {entry.name for entry in quarantine_directory.iterdir()}
        if entries != {
            CANDIDATE_DIRECTORY_NAME,
            QUARANTINE_RECEIPT_FILENAME,
        }:
            return False
        candidate = quarantine_directory / CANDIDATE_DIRECTORY_NAME
        receipt = quarantine_directory / QUARANTINE_RECEIPT_FILENAME
        if (
            candidate.is_symlink()
            or receipt.is_symlink()
            or not candidate.is_dir()
            or not receipt.is_file()
        ):
            return False

        descendants = sorted(
            candidate.rglob("*"),
            key=lambda path: path.relative_to(candidate).as_posix(),
        )
        if any(path.is_symlink() for path in descendants):
            return False
        actual_directories = [
            path.relative_to(candidate).as_posix()
            for path in descendants
            if path.is_dir()
        ]
        actual_files = [
            path.relative_to(candidate).as_posix()
            for path in descendants
            if path.is_file()
        ]
        if actual_directories != inventory.get("directories"):
            return False
        declared_files = inventory.get("files")
        if (
            not isinstance(declared_files, list)
            or actual_files
            != [record.get("path") for record in declared_files]
        ):
            return False

        root_stat = candidate.stat()
        root_identity = inventory.get("root_identity")
        if not isinstance(root_identity, Mapping) or root_identity != {
            "device": int(root_stat.st_dev),
            "inode": int(root_stat.st_ino),
            "link_count": int(root_stat.st_nlink),
            "mode": int(stat.S_IFMT(root_stat.st_mode)),
        }:
            return False
        directory_identities = inventory.get("directory_identities")
        if (
            not isinstance(directory_identities, list)
            or len(directory_identities) != len(actual_directories)
        ):
            return False
        for relative, expected in zip(
            actual_directories,
            directory_identities,
            strict=True,
        ):
            value = (candidate / relative).stat()
            if expected != {
                "path": relative,
                "device": int(value.st_dev),
                "inode": int(value.st_ino),
                "mode": int(stat.S_IFMT(value.st_mode)),
            }:
                return False
        for relative, expected in zip(
            actual_files,
            declared_files,
            strict=True,
        ):
            value = (candidate / relative).stat()
            if expected != {
                "path": relative,
                "size_bytes": int(value.st_size),
                "sha256": _file_sha256(candidate / relative),
                "device": int(value.st_dev),
                "inode": int(value.st_ino),
                "link_count": int(value.st_nlink),
                "mode": int(stat.S_IFMT(value.st_mode)),
            }:
                return False
    except (OSError, TypeError, ValueError):
        return False
    return True


def _authenticate_quarantine_receipt(
    plan: FrontierSupervisorPlan,
    lane: FrontierSupervisorLane,
    execution_receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate the generic capture receipt and its retained candidate."""

    from neural_forge.dynamic_candidate_capture import (
        QUARANTINE_CAPTURE_SCHEMA,
        QUARANTINE_RECEIPT_FILENAME,
        QUARANTINE_VERIFICATION_SCHEMA,
        QuarantineCaptureReceipt,
    )
    from neural_forge.hf_lfm2_cpcf_v5_quarantine import (
        _read_artifact_metadata,
    )

    path = (
        lane.paths.quarantine_directory
        / QUARANTINE_RECEIPT_FILENAME
    )
    payload = _read_sealed_json(
        path,
        schema=QUARANTINE_CAPTURE_SCHEMA,
        seal_field="receipt_sha256",
        name="generic quarantine receipt",
    )
    try:
        authenticated = QuarantineCaptureReceipt.from_dict(
            payload
        ).as_dict()
    except Exception as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Generic quarantine receipt failed its typed authentication."
        ) from exc

    bindings = authenticated.get("bindings")
    request = authenticated.get("capture_request")
    artifact = authenticated.get("candidate_artifact")
    model_state = authenticated.get("model_state")
    evaluations = authenticated.get("evaluations")
    reload_receipt = authenticated.get("standalone_reload_receipt")
    audit = authenticated.get("audit")
    final_reload = execution_receipt.get(
        "final_reload_verification"
    )
    expected_bindings = {
        "plan_sha256": lane.catalog_lane.target_plan_fingerprint,
        "parent_sha256": plan.evidence.parent_model_identity_sha256,
        "cache_sha256": plan.evidence.cache_sha256,
        "bridge_sha256": plan.config.expected_bridge_artifact_sha256,
        "corpus_sha256": plan.evidence.corpus_sha256,
        "compiler_sha256": execution_receipt.get(
            "compiler_receipt_sha256"
        ),
        "source_sha256": execution_receipt.get(
            "source_binding_sha256"
        ),
        "model_state_sha256": execution_receipt.get(
            "target_state_sha256"
        ),
    }
    lifecycle_false = (
        "published_compilation",
        "accepted_compilation",
        "behavior_capability_proven",
        "promotable",
        "publication_authorized",
    )
    lifecycle_true = (
        "complete_artifact_inventory",
        "standalone_candidate",
        "requires_reviewed_rerun",
        "no_parent_sidecar",
        "no_cache_sidecar",
        "no_coordinator_sidecar",
        "no_resume_sidecar",
        "receipt_outside_candidate_artifact",
    )
    inventory = (
        artifact.get("inventory")
        if isinstance(artifact, Mapping)
        else None
    )
    try:
        metadata = _read_artifact_metadata(
            lane.paths.quarantine_directory / "candidate"
        )
    except Exception as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Retained LFM2 candidate metadata failed authentication."
        ) from exc
    parameter_inventory = metadata.get("parameter_inventory")
    final_reload_body = (
        dict(final_reload)
        if isinstance(final_reload, Mapping)
        else {}
    )
    final_reload_sha = final_reload_body.pop(
        "verification_sha256",
        None,
    )
    if (
        authenticated.get("receipt_sha256")
        != execution_receipt.get("capture_receipt_sha256")
        or authenticated.get("status")
        != "quarantined-unreviewed-candidate"
        or authenticated.get("lifecycle") != "immutable-quarantine"
        or not isinstance(bindings, Mapping)
        or any(
            bindings.get(name) != value
            for name, value in expected_bindings.items()
        )
        or not isinstance(request, Mapping)
        or any(
            request.get(name) != value
            for name, value in expected_bindings.items()
            if name != "model_state_sha256"
        )
        or request.get("expected_model_state_sha256")
        != execution_receipt.get("target_state_sha256")
        or any(
            authenticated.get(name) is not False
            for name in lifecycle_false
        )
        or any(
            authenticated.get(name) is not True
            for name in lifecycle_true
        )
        or not isinstance(artifact, Mapping)
        or artifact.get("standalone") is not True
        or artifact.get("strict_sidecar_scan_passed") is not True
        or artifact.get("foreign_entries_present") is not False
        or not isinstance(inventory, Mapping)
        or artifact.get("inventory_sha256")
        != inventory.get("inventory_sha256")
        or not isinstance(model_state, Mapping)
        or model_state.get("in_memory_sha256")
        != execution_receipt.get("target_state_sha256")
        or model_state.get("fresh_reload_sha256")
        != execution_receipt.get("target_state_sha256")
        or model_state.get("byte_identical_after_reload") is not True
        or model_state.get("dense_materialized") is not True
        or model_state.get("quantized") is not False
        or model_state.get("pruned") is not False
        or not isinstance(evaluations, Mapping)
        or evaluations.get("output_reproduced_after_reload") is not True
        or evaluations.get("evaluation_reproduced_after_reload") is not True
        or evaluations.get("corpus_sha256") != plan.evidence.corpus_sha256
        or evaluations.get("evaluator_sha256")
        != request.get("evaluator_sha256")
        or not isinstance(reload_receipt, Mapping)
        or reload_receipt.get("state_reproduced_after_reload") is not True
        or reload_receipt.get("output_reproduced_after_reload") is not True
        or reload_receipt.get("evaluation_reproduced_after_reload") is not True
        or reload_receipt.get("standalone") is not True
        or reload_receipt.get("promotable") is not False
        or reload_receipt.get("publication_authorized") is not False
        or not isinstance(audit, Mapping)
        or audit.get("combined_clean") is not True
        or metadata.get("model_state_sha256")
        != execution_receipt.get("target_state_sha256")
        or not isinstance(parameter_inventory, Mapping)
        or parameter_inventory.get("unique_parameters")
        != lane.catalog_lane.exact_compression.target_unique_parameters
        or parameter_inventory.get("independent_methods_agree") is not True
        or parameter_inventory.get("quantization_adjustment_used") is not False
        or parameter_inventory.get("pruning_mask_adjustment_used") is not False
        or parameter_inventory.get("inventory_sha256")
        != metadata.get("parameter_inventory_sha256")
        or final_reload_body.get("schema")
        != QUARANTINE_VERIFICATION_SCHEMA
        or _digest(
            final_reload_sha,
            "final reload verification SHA",
        )
        != _sha(final_reload_body)
        or final_reload_body.get("capture_receipt_sha256")
        != authenticated.get("receipt_sha256")
        or final_reload_body.get("model_state_sha256")
        != execution_receipt.get("target_state_sha256")
        or final_reload_body.get("output_sha256")
        != bindings.get("output_sha256")
        or final_reload_body.get("evaluation_sha256")
        != evaluations.get("fresh_reload_evaluation_sha256")
        or final_reload_body.get("artifact_inventory_sha256")
        != artifact.get("inventory_sha256")
        or final_reload_body.get("loader_sha256")
        != request.get("loader_sha256")
        or final_reload_body.get("evaluator_sha256")
        != request.get("evaluator_sha256")
        or final_reload_body.get(
            "fresh_final_destination_reload_performed"
        )
        is not True
        or final_reload_body.get("promotable") is not False
        or final_reload_body.get("publication_authorized") is not False
        or final_reload_body.get("requires_reviewed_rerun") is not True
        or not _retained_inventory_matches(
            lane.paths.quarantine_directory,
            inventory,
        )
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Generic quarantine receipt, lifecycle, bindings, or retained "
            "inventory differs from the exact worker execution."
        )
    return MappingProxyType(authenticated)


def execute_single_ratio(
    plan: FrontierSupervisorPlan,
    ratio: int,
    *,
    subprocess_runner: Callable[..., Any] = subprocess.run,
) -> Mapping[str, Any]:
    """Run exactly one fixed capture worker and authenticate its receipt."""

    if not isinstance(plan, FrontierSupervisorPlan):
        raise TypeError("plan must be FrontierSupervisorPlan.")
    lane = plan.lane_for_ratio(ratio)
    if lane.execution_blockers or not lane.execution_ready:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Refusing to execute a blocked frontier lane: "
            + " | ".join(lane.execution_blockers)
        )
    catalog_preparation = authenticate_frontier_catalog_preparation(
        plan.config
    )
    if (
        dict(catalog_preparation)
        != dict(plan.evidence.selective_catalog_preparation)
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "The shared selective catalog differs from the planned binding."
        )
    lane.paths.run_directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        lane.paths.run_directory.mkdir()
    except FileExistsError as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Ratio run directory appeared after planning; refusing overwrite."
        ) from exc
    _atomic_new_json(
        lane.paths.process_manifest_json,
        lane.command_manifest,
    )
    environment = dict(os.environ)
    environment.update(
        lane.command_manifest["environment_overrides"]
    )
    completed = subprocess_runner(
        lane.command_manifest["command"],
        cwd=lane.command_manifest["working_directory"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if getattr(completed, "returncode", None) != 0:
        stderr = str(getattr(completed, "stderr", "")).strip()
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Capture worker failed closed"
            + (f": {stderr[-2000:]}" if stderr else ".")
        )
    if dict(plan.source_identities) != _source_identities():
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Supervisor source files changed across the worker boundary."
        )
    post_run_catalog_preparation = (
        authenticate_frontier_catalog_preparation(plan.config)
    )
    if (
        dict(post_run_catalog_preparation)
        != dict(catalog_preparation)
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "The shared selective catalog changed across the worker "
            "boundary."
        )
    receipt = _read_sealed_json(
        lane.paths.execution_receipt_json,
        schema=EXECUTION_SCHEMA,
        seal_field="receipt_sha256",
        name="worker execution receipt",
    )
    from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner

    diagnostic = _read_sealed_json(
        lane.paths.diagnostic_json,
        schema=runner.RUN_SCHEMA,
        seal_field="receipt_sha256",
        name="worker compile diagnostic",
    )
    _authenticate_quarantine_receipt(plan, lane, receipt)
    inputs = diagnostic.get("inputs")
    compiled = diagnostic.get("compiled_model_fold")
    compiler_implementation = (
        compiled.get("compiler_implementation")
        if isinstance(compiled, Mapping)
        else None
    )
    runtime_guard = diagnostic.get("compiler_runtime_fold_only_guard")
    guard_implementation = (
        runtime_guard.get("guard_implementation")
        if isinstance(runtime_guard, Mapping)
        else None
    )
    compiler_guard = _authenticate_runtime_guard_receipt(
        runtime_guard,
        schema=(
            "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1"
        ),
        guard_source_sha256=plan.source_identities[
            "fold_only_runtime_guard"
        ],
        name="retained compiler runtime fold-only guard",
    )
    consumer_guard = _authenticate_runtime_guard_receipt(
        receipt.get("consumer_runtime_guard_receipt"),
        schema=(
            "neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"
        ),
        guard_source_sha256=plan.source_identities[
            "fold_only_runtime_guard"
        ],
        name="retained consumer runtime fold-only guard",
    )
    final_reload_verification = receipt.get("final_reload_verification")
    resume_progress = diagnostic.get("resume_progress")
    resume_reference = lane.catalog_lane.resume_reference
    invocation_head = (
        resume_progress.get("invocation_head")
        if isinstance(resume_progress, Mapping)
        else None
    )
    checkpoint_reference = (
        invocation_head.get("checkpoint_reference")
        if isinstance(invocation_head, Mapping)
        else None
    )
    if resume_reference is None:
        resume_binding_valid = (
            isinstance(resume_progress, Mapping)
            and resume_progress.get("resume_requested") is False
            and invocation_head is None
        )
    else:
        resume_binding_valid = (
            isinstance(resume_progress, Mapping)
            and resume_progress.get("resume_requested") is True
            and resume_progress.get("requested_admission_sha256")
            == resume_reference.resume_admission_sha256
            and isinstance(invocation_head, Mapping)
            and invocation_head.get("admission_sha256")
            == resume_reference.resume_admission_sha256
            and isinstance(checkpoint_reference, Mapping)
            and checkpoint_reference.get("manifest_sha256")
            == resume_reference.resume_artifact_sha256
        )
    expected_source_binding = runner._source_binding_sha256(
        selective_source_sha256=receipt.get(
            "selective_source_sha256"
        ),
        compiler_implementation_sha256=receipt.get(
            "compiler_implementation_sha256"
        ),
        runner_source_sha256=receipt.get("runner_source_sha256"),
        fold_only_runtime_guard_identity_sha256=(
            guard_implementation.get("identity_sha256")
            if isinstance(guard_implementation, Mapping)
            else None
        ),
        catalog_preparation=catalog_preparation,
    )
    diagnostic_catalog = diagnostic.get("selective_catalog_artifact")
    diagnostic_catalog_reverification = diagnostic.get(
        "selective_catalog_post_run_reverification"
    )
    receipt_catalog_reverification = receipt.get(
        "selective_catalog_post_run_reverification"
    )
    _authenticate_catalog_post_run_reverification(
        diagnostic_catalog_reverification,
        catalog_preparation=catalog_preparation,
        name="Retained compiler diagnostic",
    )
    _authenticate_catalog_post_run_reverification(
        receipt_catalog_reverification,
        catalog_preparation=catalog_preparation,
        name="Worker execution receipt",
    )
    if (
        receipt.get("status")
        != "captured_quarantine_requires_reviewed_rerun"
        or receipt.get("target_plan_fingerprint")
        != lane.catalog_lane.target_plan_fingerprint
        or receipt.get("target_unique_parameters")
        != lane.catalog_lane.exact_compression.target_unique_parameters
        or receipt.get("selected_candidate_fingerprint")
        != lane.catalog_lane.selected_candidate.fingerprint
        or receipt.get("selected_candidate_rank")
        != lane.catalog_lane.selected_candidate_rank
        or receipt.get("parent_sha256")
        != plan.evidence.parent_model_identity_sha256
        or receipt.get("cache_sha256") != plan.evidence.cache_sha256
        or receipt.get("bridge_sha256")
        != plan.config.expected_bridge_artifact_sha256
        or receipt.get("corpus_sha256") != plan.evidence.corpus_sha256
        or receipt.get("source_binding_sha256")
        != expected_source_binding
        or receipt.get("selective_catalog_artifact")
        != dict(catalog_preparation)
        or receipt.get("runner_source_sha256")
        != plan.source_identities["v5_diagnostic_runner"]
        or receipt.get("post_capture_source_identities")
        != dict(plan.source_identities)
        or receipt.get("diagnostic_json")
        != str(lane.paths.diagnostic_json)
        or receipt.get("quarantine_directory")
        != str(lane.paths.quarantine_directory)
        or receipt.get("standalone_reload_verified") is not True
        or receipt.get("whole_output_evaluation_captured") is not True
        or receipt.get("manual_review_status") != "pending"
        or receipt.get("training_steps") != 0
        or receipt.get("backward_calls") != 0
        or receipt.get("optimizer_steps") != 0
        or receipt.get("parameters_pruned") != 0
        or receipt.get("parameters_quantized") != 0
        or not resume_binding_valid
        or not isinstance(inputs, Mapping)
        or inputs.get("parent_fingerprint")
        != receipt.get("parent_sha256")
        or inputs.get("cache_key") != receipt.get("cache_sha256")
        or inputs.get("bridge_artifact_sha256")
        != receipt.get("bridge_sha256")
        or inputs.get("trajectory_corpus_sha256")
        != receipt.get("corpus_sha256")
        or not isinstance(diagnostic_catalog, Mapping)
        or dict(diagnostic_catalog) != dict(catalog_preparation)
        or not isinstance(compiled, Mapping)
        or compiled.get("result_sha256")
        != receipt.get("compiler_receipt_sha256")
        or compiled.get("selective_source_sha256")
        != receipt.get("selective_source_sha256")
        or not isinstance(compiler_implementation, Mapping)
        or compiler_implementation.get("implementation_sha256")
        != receipt.get("compiler_implementation_sha256")
        or not isinstance(guard_implementation, Mapping)
        or guard_implementation.get("source_sha256")
        != plan.source_identities.get("fold_only_runtime_guard")
        or guard_implementation.get("training_api_firewall") is not True
        or guard_implementation.get("common_pruning_api_firewall") is not True
        or guard_implementation.get("common_quantization_api_firewall")
        is not True
        or runtime_guard.get("clean") is not True
        or consumer_guard.get("compiler_runtime_guard_sha256")
        != compiler_guard["report_sha256"]
        or not isinstance(final_reload_verification, Mapping)
        or consumer_guard.get("capture_verification_sha256")
        != final_reload_verification.get("verification_sha256")
        or receipt.get("promotable") is not False
        or receipt.get("publication_authorized") is not False
        or receipt.get("requires_reviewed_rerun") is not True
        or receipt.get("standalone_candidate") is not True
        or not lane.paths.quarantine_directory.is_dir()
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Capture worker receipt or retained artifacts differ from the "
            "exact non-promotable lane."
        )
    return receipt


def _atomic_new_json(path: Path, value: Mapping[str, Any]) -> None:
    lexical = Path(
        _path_text(path, "plan_output")
    ).expanduser().absolute()
    if lexical.exists() or lexical.is_symlink():
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Plan output already exists; refusing overwrite."
        )
    destination = _new_local_root(lexical, name="plan_output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Plan output already exists; refusing overwrite."
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(dict(value)) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        published = True
    except FileExistsError as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            "Plan output appeared concurrently; refusing overwrite."
        ) from exc
    finally:
        if temporary.exists():
            temporary.unlink()
        if not published and destination.exists():
            # A destination can only exist here if this process linked it.
            # Never remove a concurrent or pre-existing path.
            pass


def _load_typed_mapping(
    path: Path | None,
    *,
    kind: str,
) -> Mapping[str, Any]:
    if path is None:
        return {}
    source = _existing_local_path(path, name=f"{kind}_json", directory=False)
    try:
        payload = json.loads(source.read_text("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{kind} JSON is unreadable."
        ) from exc
    if not isinstance(payload, Mapping):
        raise Lfm2CpcfV5FrontierSupervisorError(
            f"{kind} JSON must be an object keyed by target-plan fingerprint."
        )
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5ApprovedReviewBindings,
        Lfm2CpcfV5ResumeReference,
    )

    value_type = (
        Lfm2CpcfV5ApprovedReviewBindings
        if kind == "approved_review_bindings"
        else Lfm2CpcfV5ResumeReference
    )
    result = {}
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            raise Lfm2CpcfV5FrontierSupervisorError(
                f"{kind}[{key!r}] must be an object."
            )
        record = dict(value)
        record.pop("schema", None)
        record.pop("kind", None)
        record.pop("approved_bindings_are_behavioral_evidence", None)
        instance = value_type(**record)
        if instance.target_plan_fingerprint != key:
            raise Lfm2CpcfV5FrontierSupervisorError(
                f"{kind} key differs from its typed plan fingerprint."
            )
        result[key] = instance
    return result


def build_parser() -> argparse.ArgumentParser:
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5FrontierLifecycle,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--plan-only",
        action="store_true",
        help="Plan all six ratios without starting a fold subprocess.",
    )
    mode.add_argument(
        "--execute-ratio",
        type=int,
        choices=DECLARATIVE_RATIOS,
        help=(
            "Capture exactly one ratio through the fixed isolated consumer "
            "worker as a non-promotable standalone quarantine."
        ),
    )
    mode.add_argument(
        "--prepare-selective-catalog-only",
        action="store_true",
        help=(
            "Build and authenticate the one shared tensor-free selective "
            "catalog, then exit before parent, blank-target, or CUDA "
            "allocation."
        ),
    )
    parser.add_argument("--parent-model-directory", type=Path, required=True)
    parser.add_argument("--donor-tokenizer-directory", type=Path, required=True)
    parser.add_argument("--target-tokenizer-directory", type=Path, required=True)
    parser.add_argument("--bridge-directory", type=Path, required=True)
    parser.add_argument("--trace-cache-directory", type=Path, required=True)
    parser.add_argument("--trajectory-corpus", type=Path, required=True)
    parser.add_argument("--parent-envelope", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument(
        "--selective-catalog-store-directory",
        type=Path,
        help=(
            "Shared immutable selective catalog store. Defaults to a sibling "
            "of the work directory."
        ),
    )
    parser.add_argument(
        "--selective-catalog-receipt-json",
        type=Path,
        help=(
            "Create-new preparation receipt or exact existing receipt for "
            "planning and folding."
        ),
    )
    parser.add_argument(
        "--expected-selective-catalog-receipt-sha256",
        help=(
            "Exact prepared receipt SHA-256 required for every planning and "
            "fold worker process."
        ),
    )
    parser.add_argument(
        "--expected-parent-graph-fingerprint",
        required=True,
    )
    parser.add_argument(
        "--expected-bridge-artifact-sha256",
        required=True,
    )
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument(
        "--expected-parent-envelope-file-sha256",
        required=True,
    )
    parser.add_argument(
        "--lifecycle",
        choices=tuple(value.value for value in Lfm2CpcfV5FrontierLifecycle),
        default=Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE.value,
    )
    parser.add_argument("--approved-review-bindings-json", type=Path)
    parser.add_argument("--resume-references-json", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--maximum-process-rss-gib", type=float, default=64.0)
    parser.add_argument(
        "--maximum-logical-resident-gib",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--maximum-cuda-gib-per-device",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--minimum-physical-rss-reduction-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument(
        "--gpu-device",
        action="append",
        dest="gpu_devices",
        help=(
            "Physical GPU UUID or index. Supply exactly twice; the fold sees "
            "them as logical CUDA devices 0 and 1."
        ),
    )
    parser.add_argument(
        "--parent-name",
        default="LiquidAI-LFM2.5-230M",
    )
    parser.add_argument("--child-name-prefix", default="lfm2-cpcf-v5")
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    parser.add_argument("--maximum-candidates-per-ratio", type=int, default=8)
    parser.add_argument(
        "--attempt-id",
        default="attempt-0001",
        help=(
            "Create-new attempt namespace. Choose a new value to retry an "
            "immutable failed lane; authenticated resume state remains "
            "plan-scoped outside this namespace."
        ),
    )
    parser.add_argument("--python-executable", default=sys.executable)
    return parser


def config_from_args(args: argparse.Namespace) -> FrontierSupervisorConfig:
    from neural_forge.dynamic_lfm2_cpcf_v5_frontier import (
        Lfm2CpcfV5FrontierLifecycle,
    )

    lifecycle = Lfm2CpcfV5FrontierLifecycle(args.lifecycle)
    reviews = _load_typed_mapping(
        args.approved_review_bindings_json,
        kind="approved_review_bindings",
    )
    resumes = _load_typed_mapping(
        args.resume_references_json,
        kind="resume_references",
    )
    if (
        lifecycle
        is Lfm2CpcfV5FrontierLifecycle.CAPTURE_QUARANTINE
        and reviews
    ):
        raise Lfm2CpcfV5FrontierSupervisorError(
            "capture_quarantine accepts no approved review bindings."
        )
    ratios = (
        DECLARATIVE_RATIOS
        if args.execute_ratio is None
        else (args.execute_ratio,)
    )
    devices = (
        DEFAULT_RTX3090_GPU_SELECTORS
        if args.gpu_devices is None
        else tuple(args.gpu_devices)
    )
    return FrontierSupervisorConfig(
        parent_model_directory=args.parent_model_directory,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        trace_cache_directory=args.trace_cache_directory,
        trajectory_corpus=args.trajectory_corpus,
        parent_envelope=args.parent_envelope,
        work_directory=args.work_directory,
        expected_parent_graph_fingerprint=(
            args.expected_parent_graph_fingerprint
        ),
        expected_bridge_artifact_sha256=(
            args.expected_bridge_artifact_sha256
        ),
        expected_cache_sha256=args.expected_cache_sha256,
        expected_parent_envelope_file_sha256=(
            args.expected_parent_envelope_file_sha256
        ),
        lifecycle=lifecycle,
        compression_ratios=ratios,
        approved_review_bindings=reviews,
        resume_references=resumes,
        runtime_limits=FrontierRuntimeLimits(
            maximum_process_rss_gib=args.maximum_process_rss_gib,
            maximum_logical_resident_gib=(
                args.maximum_logical_resident_gib
            ),
            maximum_cuda_gib_per_device=(
                args.maximum_cuda_gib_per_device
            ),
            minimum_physical_rss_reduction_fraction=(
                args.minimum_physical_rss_reduction_fraction
            ),
            cpu_threads=args.cpu_threads,
            gpu_selectors=devices,
        ),
        parent_name=args.parent_name,
        child_name_prefix=args.child_name_prefix,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
        maximum_candidates_per_ratio=args.maximum_candidates_per_ratio,
        attempt_id=args.attempt_id,
        python_executable=args.python_executable,
        selective_catalog_store_directory=(
            args.selective_catalog_store_directory
        ),
        selective_catalog_receipt_json=(
            args.selective_catalog_receipt_json
        ),
        expected_selective_catalog_receipt_sha256=(
            args.expected_selective_catalog_receipt_sha256
        ),
    )


def prepare_frontier_selective_catalog(
    config: FrontierSupervisorConfig,
) -> Mapping[str, Any]:
    """Run the shared preparation phase without frontier/model discovery."""

    if not isinstance(config, FrontierSupervisorConfig):
        raise TypeError("config must be FrontierSupervisorConfig.")
    from experiments import run_lfm2_cpcf_v5_selective_blank_fold as runner

    preparation_args = (
        "--parent-model-directory",
        str(config.parent_model_directory),
        "--donor-tokenizer-directory",
        str(config.donor_tokenizer_directory),
        "--target-tokenizer-directory",
        str(config.target_tokenizer_directory),
        "--bridge-directory",
        str(config.bridge_directory),
        "--trace-cache-directory",
        str(config.trace_cache_directory),
        "--trajectory-corpus",
        str(config.trajectory_corpus),
        "--diagnostic-json",
        str(
            config.work_directory
            / "catalog-preparation-failure-diagnostic.json"
        ),
        "--compression-ratio",
        "4",
        "--maximum-process-rss-gib",
        str(config.runtime_limits.maximum_process_rss_gib),
        "--selective-catalog-store-directory",
        str(config.selective_catalog_store_directory),
        "--selective-catalog-receipt-json",
        str(config.selective_catalog_receipt_json),
        "--prepare-selective-catalog-only",
    )
    runner_config = runner.config_from_args(
        runner.build_parser().parse_args(preparation_args)
    )
    return MappingProxyType(
        dict(runner.prepare_selective_catalog(runner_config))
    )


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--internal-worker-manifest" in raw:
        worker = argparse.ArgumentParser(add_help=False)
        worker.add_argument(
            "--internal-worker-manifest",
            type=Path,
            required=True,
        )
        worker_args = worker.parse_args(raw)
        return _run_internal_worker(worker_args.internal_worker_manifest)
    args = build_parser().parse_args(raw)
    try:
        config = config_from_args(args)
        if args.plan_output is not None:
            _assert_catalog_receipt_disjoint(
                config.selective_catalog_receipt_json,
                roots={
                    "plan_output": _new_local_root(
                        args.plan_output,
                        name="plan_output",
                    )
                },
            )
        if args.prepare_selective_catalog_only:
            preparation = dict(
                prepare_frontier_selective_catalog(config)
            )
            print(json.dumps(preparation, indent=2, sort_keys=True))
            return 0
        schedule, evidence = discover_frontier_schedule(config)
        plan = build_frontier_supervisor_plan(
            schedule,
            config,
            evidence,
        )
        document = plan.as_dict()
        if args.plan_output is not None:
            _atomic_new_json(args.plan_output, document)
        if args.execute_ratio is not None:
            execute_single_ratio(plan, args.execute_ratio)
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
