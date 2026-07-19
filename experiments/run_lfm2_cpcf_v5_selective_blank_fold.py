#!/usr/bin/env python3
"""Run the opt-in LFM2 CPCF v5 selective fold on one exact dense blank.

This entry point is deliberately narrower than the legacy production runner.
It reuses that runner's strict local input, tokenizer-bridge, trajectory, and
blank-target helpers, but routes the fold through the bounded-residency v5
compiler and the authenticated deterministic two-RTX-3090 replay backend.

The process opens an already sealed local trace cache.  It never regenerates
traces, executes the parent, trains, prunes, quantizes, downloads, or publishes
a child checkpoint.  An explicit local resume store may retain authenticated
v5 child-prefix checkpoints, and continuation accepts only one exact admitted
v5 digest.  Its final output remains a diagnostic JSON file.  Successful
compilation therefore proves an in-memory structural fold, not behavioral
intelligence retention or a publishable model.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
_SOURCE_ROOT = _REPOSITORY_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

# These must be set before any Transformers or CUDA-backed module is imported.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


RUN_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-blank-fold-run-v1"
FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-blank-fold-failure-v1"
RESUME_PROGRESS_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-runner-resume-progress-v1"
CUDA_ALLOCATOR_POLICY_SCHEMA = "neural-forge-lfm2-cpcf-v5-cuda-allocator-policy-v1"
COMPILED_CANDIDATE_CONTEXT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-compiled-candidate-consumer-context-v1"
)
COMPILED_CANDIDATE_RESULT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-compiled-candidate-consumer-result-v1"
)
COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-compiled-candidate-zero-operation-audit-v1"
)
SELECTIVE_CATALOG_PREPARATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-selective-catalog-preparation-v1"
)
GPU_REPLAY_SCHEMA = "neural-forge-lfm2-cpcf-v5-authenticated-two-gpu-replay-v1"
PROVEN_GPU_PARITY_RECEIPT = (
    _REPOSITORY_ROOT / "reports" / "2a80198-two-rtx3090-bf16.json"
)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3090"
GPU_LOGICAL_DEVICES = (0, 1)
GPU_PINNED_BYTES_PER_SLOT = 256 << 20
GPU_PARITY_ABSOLUTE_TOLERANCE = 0.125
GPU_PARITY_RELATIVE_TOLERANCE = 0.02
GPU_PARITY_MAXIMUM_RELATIVE_RMSE = 0.02
GPU_PARITY_MINIMUM_COSINE_SIMILARITY = 0.999
PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE = 0.0
PREREGISTERED_PER_LAG_COMMUTATOR_GATE = 1.0
_GIB = 1024**3
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONSUMER_ZERO_COUNTER_FIELDS = (
    "optimizer_constructions",
    "optimizer_steps",
    "backward_calls",
    "autograd_grad_calls",
    "training_steps",
    "pruning_operations",
    "quantization_operations",
    "parent_forward_calls",
    "parent_inference_calls",
    "target_forward_calls",
    "cache_generation_calls",
)
_FORBIDDEN_REQUEST_FIELDS = (
    "resume",
    "resume_state",
    "resume_checkpoint_reference",
    "publish",
    "output_directory",
    "publication_directory",
    "save_child",
    "regenerate_traces",
    "train",
    "prune",
    "quantize",
    "download",
)
_FORBIDDEN_FOLD_MUTATION_MODULES = (
    "torch.nn.utils.prune",
    "torch.ao.quantization",
    "torch.quantization",
    "torchao.quantization",
)
_FORBIDDEN_FOLD_MUTATION_SYMBOLS = frozenset(
    {
        "torch.quantize_per_tensor",
        "torch.quantize_per_channel",
        "torch.fake_quantize_per_tensor_affine",
        "torch.fake_quantize_per_channel_affine",
    }
)
_SELECTIVE_EVIDENCE_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "hf_lfm2_cpcf_selective_evidence.py"
)
_CATALOG_ARTIFACT_SOURCE = (
    _SOURCE_ROOT / "neural_forge" / "dynamic_evidence_catalog_artifact.py"
)


class Lfm2CpcfV5SelectiveBlankFoldRunnerError(RuntimeError):
    """The strict local v5 selective experiment failed closed."""


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
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Diagnostic metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _plain_json(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _base_runner() -> Any:
    """Import the established strict-local helper module without importing torch."""

    from experiments import run_lfm2_cpcf_blank_fold

    return run_lfm2_cpcf_blank_fold


def _build_preregistered_real_v5_numerical_adapter() -> Any:
    """Construct the explicit source-independent v5 measurement policy."""

    from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
        Lfm2CpcfV5FirBasisPolicy,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
        Lfm2CpcfShortConvV5NumericalFrontierAdapter,
    )

    return Lfm2CpcfShortConvV5NumericalFrontierAdapter(
        basis_policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=(
                PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
            ),
            maximum_per_lag_commutator_relative_frobenius=(
                PREREGISTERED_PER_LAG_COMMUTATOR_GATE
            ),
        )
    )


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{name} must be finite.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be finite."
        ) from exc
    if not math.isfinite(result):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{name} must be finite.")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be a positive integer."
        )
    return value


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be one exact lowercase SHA-256 digest."
        )
    return value


def _local_lexical_path(value: Path, name: str) -> Path:
    raw = os.fspath(value.expanduser())
    if not raw or "://" in raw or raw.startswith("\\\\") or raw.startswith("//"):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{name} must be an explicit local path."
        )
    return value.expanduser().absolute().resolve(strict=False)


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _assert_catalog_store_disjoint(
    store: Path,
    *,
    roots: Mapping[str, Path],
) -> None:
    normalized_store = _local_lexical_path(
        store,
        "selective_catalog_store_directory",
    )
    overlaps = []
    for name, value in roots.items():
        normalized = _local_lexical_path(value, name)
        if _paths_overlap(normalized_store, normalized):
            overlaps.append(name)
    if overlaps:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "selective_catalog_store_directory cannot equal, contain, or "
            "be contained by protected roots: " + ", ".join(sorted(overlaps)) + "."
        )


def _assert_catalog_receipt_disjoint(
    receipt: Path,
    *,
    roots: Mapping[str, Path],
) -> None:
    normalized_receipt = _local_lexical_path(
        receipt,
        "selective_catalog_receipt_json",
    )
    overlaps = []
    for name, value in roots.items():
        normalized = _local_lexical_path(value, name)
        if _paths_overlap(normalized_receipt, normalized):
            overlaps.append(name)
    if overlaps:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "selective_catalog_receipt_json cannot equal, contain, or be "
            "contained by protected inputs, outputs, work, resume, or catalog "
            "store paths: " + ", ".join(sorted(overlaps)) + "."
        )


def _sealed_mapping(
    value: Mapping[str, Any],
    *,
    seal_field: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{label} must be a mapping.")
    report = dict(value)
    claimed = _digest(report.pop(seal_field, None), f"{label}.{seal_field}")
    if _sha(report) != claimed:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{label} is not self-sealed.")
    return MappingProxyType({**report, seal_field: claimed})


def _immutable_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(f"{name} must be a mapping.")
    # Canonicalization rejects non-finite or non-JSON evidence before the
    # mapping is exposed across the consumer boundary.
    normalized = dict(value)
    _canonical_json(normalized)
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class V5CompiledCandidateConsumerContext:
    """Authenticated live post-compile state exposed to one capture backend."""

    target_model: Any
    target_tokenizer: Any
    forbidden_parent_model: Any
    target_plan_fingerprint: str
    blank_target_report: Mapping[str, Any]
    compiled_model_fold_report: Mapping[str, Any]
    compiler_receipt_sha256: str
    compile_diagnostic: Mapping[str, Any]
    parent_sha256: str
    cache_sha256: str
    bridge_sha256: str
    corpus_sha256: str
    source_sha256: str
    compiler_sha256: str
    runner_source_sha256: str
    source_binding_sha256: str
    parent_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    target_tokenizer_package_sha256: str | None
    parent_parameter_inventory: Mapping[str, Any]
    blank_parameter_inventory: Mapping[str, Any]
    compiled_parameter_inventory: Mapping[str, Any]
    zero_operation_audit: Mapping[str, Any]
    target_state_sha256: str
    target_runtime_sha256: str
    parent_state_sha256: str
    parent_runtime_sha256: str

    def __post_init__(self) -> None:
        if self.target_model is None or self.target_tokenizer is None:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate context requires a live target and tokenizer."
            )
        if (
            self.forbidden_parent_model is None
            or self.forbidden_parent_model is self.target_model
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate context requires a distinct forbidden parent."
            )
        for name in (
            "target_plan_fingerprint",
            "compiler_receipt_sha256",
            "parent_sha256",
            "cache_sha256",
            "bridge_sha256",
            "corpus_sha256",
            "source_sha256",
            "compiler_sha256",
            "runner_source_sha256",
            "source_binding_sha256",
            "parent_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
            "target_state_sha256",
            "target_runtime_sha256",
            "parent_state_sha256",
            "parent_runtime_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.target_tokenizer_package_sha256 is not None:
            object.__setattr__(
                self,
                "target_tokenizer_package_sha256",
                _digest(
                    self.target_tokenizer_package_sha256,
                    "target_tokenizer_package_sha256",
                ),
            )

        blank = _sealed_mapping(
            self.blank_target_report,
            seal_field="report_sha256",
            label="blank_target_report",
        )
        compiled = _sealed_mapping(
            self.compiled_model_fold_report,
            seal_field="result_sha256",
            label="compiled_model_fold_report",
        )
        diagnostic = _sealed_mapping(
            self.compile_diagnostic,
            seal_field="receipt_sha256",
            label="compile_diagnostic",
        )
        if (
            blank.get("target_plan_fingerprint") != self.target_plan_fingerprint
            or compiled.get("blank_target_report_sha256") != blank["report_sha256"]
            or compiled["result_sha256"] != self.compiler_receipt_sha256
            or diagnostic.get("compiled_model_fold") != dict(compiled)
            or diagnostic.get("blank_target") != dict(blank)
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate plan, blank, compiler, or diagnostic "
                "receipts do not share one exact provenance chain."
            )
        inputs = diagnostic.get("inputs")
        compiler_implementation = compiled.get("compiler_implementation")
        seam = diagnostic.get("compiled_candidate_consumer_seam")
        catalog_preparation = diagnostic.get("selective_catalog_artifact")
        if (
            not isinstance(inputs, Mapping)
            or not isinstance(
                compiler_implementation,
                Mapping,
            )
            or not isinstance(seam, Mapping)
            or not isinstance(
                catalog_preparation,
                Mapping,
            )
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate diagnostic lacks authenticated identities."
            )
        from neural_forge import dynamic_fold_executor

        expected_guard_identity = _fold_only_guard_implementation_identity(
            dynamic_fold_executor
        )
        if (
            inputs.get("parent_fingerprint") != self.parent_sha256
            or inputs.get("cache_key") != self.cache_sha256
            or inputs.get("bridge_artifact_sha256") != self.bridge_sha256
            or inputs.get("trajectory_corpus_sha256") != self.corpus_sha256
            or inputs.get("donor_tokenizer_semantics_sha256")
            != self.parent_tokenizer_semantics_sha256
            or inputs.get("target_tokenizer_semantics_sha256")
            != self.target_tokenizer_semantics_sha256
            or compiled.get("selective_source_sha256") != self.source_sha256
            or compiler_implementation.get("implementation_sha256")
            != self.compiler_sha256
            or self.source_binding_sha256
            != _source_binding_sha256(
                selective_source_sha256=self.source_sha256,
                compiler_implementation_sha256=self.compiler_sha256,
                runner_source_sha256=self.runner_source_sha256,
                fold_only_runtime_guard_identity_sha256=(
                    expected_guard_identity["identity_sha256"]
                ),
                catalog_preparation=catalog_preparation,
            )
            or self.runner_source_sha256 != _file_sha256(Path(__file__).resolve())
            or any(
                seam.get(name) != value
                for name, value in {
                    "target_plan_fingerprint": self.target_plan_fingerprint,
                    "blank_target_report_sha256": blank["report_sha256"],
                    "compiler_receipt_sha256": (self.compiler_receipt_sha256),
                    "parent_sha256": self.parent_sha256,
                    "cache_sha256": self.cache_sha256,
                    "bridge_sha256": self.bridge_sha256,
                    "corpus_sha256": self.corpus_sha256,
                    "source_sha256": self.source_sha256,
                    "compiler_sha256": self.compiler_sha256,
                    "runner_source_sha256": self.runner_source_sha256,
                    "source_binding_sha256": self.source_binding_sha256,
                    **{
                        "selective_catalog_preparation_receipt_sha256": (
                            catalog_preparation["preparation_receipt_sha256"]
                        ),
                        "selective_catalog_binding_sha256": (
                            catalog_preparation["binding_sha256"]
                        ),
                        "selective_catalog_artifact_manifest_sha256": (
                            catalog_preparation["artifact_manifest_sha256"]
                        ),
                        "selective_catalog_artifact_sha256": (
                            catalog_preparation["artifact_catalog_sha256"]
                        ),
                        "selective_catalog_codec_sha256": (
                            catalog_preparation["codec_sha256"]
                        ),
                        "selective_catalog_provider_implementation_sha256": (
                            catalog_preparation["provider_implementation_sha256"]
                        ),
                    },
                    "fold_only_runtime_guard_identity_sha256": (
                        expected_guard_identity["identity_sha256"]
                    ),
                    "parent_tokenizer_semantics_sha256": (
                        self.parent_tokenizer_semantics_sha256
                    ),
                    "target_tokenizer_semantics_sha256": (
                        self.target_tokenizer_semantics_sha256
                    ),
                    "target_tokenizer_package_sha256": (
                        self.target_tokenizer_package_sha256
                    ),
                    "target_state_sha256": self.target_state_sha256,
                    "target_runtime_sha256": self.target_runtime_sha256,
                    "parent_state_sha256": self.parent_state_sha256,
                    "parent_runtime_sha256": self.parent_runtime_sha256,
                }.items()
            )
            or diagnostic.get("parent_parameter_inventory")
            != dict(self.parent_parameter_inventory)
            or diagnostic.get("blank_parameter_inventory")
            != dict(self.blank_parameter_inventory)
            or diagnostic.get("compiled_parameter_inventory")
            != dict(self.compiled_parameter_inventory)
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate context identity or inventory binding changed."
            )
        claims = diagnostic.get("claims")
        if (
            not isinstance(claims, Mapping)
            or claims.get("child_checkpoint_saved") is not False
            or claims.get("publication_attempted") is not False
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "The compiler diagnostic must remain non-publishing."
            )
        audit = _immutable_mapping(
            self.zero_operation_audit,
            "zero_operation_audit",
        )
        if (
            audit.get("schema") != COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA
            or audit.get("clean") is not True
            or any(
                isinstance(audit.get(name), bool) or audit.get(name) != 0
                for name in _CONSUMER_ZERO_COUNTER_FIELDS
            )
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiled candidate context does not prove a zero-operation fold."
            )
        if seam.get("zero_operation_audit") != dict(audit):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Zero-operation audit differs from the sealed diagnostic."
            )
        runtime_guard = _sealed_mapping(
            audit.get("compiler_runtime_guard"),
            seal_field="report_sha256",
            label="compiler_runtime_guard",
        )
        static_source_audit = _sealed_mapping(
            runtime_guard.get("static_pruning_quantization_source_audit"),
            seal_field="report_sha256",
            label="static_pruning_quantization_source_audit",
        )
        if (
            diagnostic.get("compiler_runtime_fold_only_guard") != dict(runtime_guard)
            or runtime_guard.get("guard_implementation")
            != dict(expected_guard_identity)
            or static_source_audit.get("compiler_implementation_sha256")
            != self.compiler_sha256
            or static_source_audit.get("authenticated_compiler_source_files")
            != compiler_implementation.get("ordered_source_files")
            or static_source_audit.get("runner_source_sha256")
            != self.runner_source_sha256
            or static_source_audit.get("clean") is not True
            or static_source_audit.get("forbidden_finding_count") != 0
            or static_source_audit.get("forbidden_findings") != []
            or runtime_guard.get("clean") is not True
            or any(
                type(runtime_guard.get(name)) is not int or runtime_guard.get(name) != 0
                for name in (
                    "optimizer_constructions",
                    "optimizer_steps",
                    "backward_calls",
                    "autograd_grad_calls",
                    "pruning_operations",
                    "quantization_operations",
                    "parent_module_forward_calls",
                    "parent_root_forward_calls",
                    "parent_inference_entrypoint_calls",
                    "target_root_forward_calls",
                    "target_inference_entrypoint_calls",
                )
            )
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Runtime fold-only guard evidence or source identity changed."
            )
        object.__setattr__(self, "blank_target_report", blank)
        object.__setattr__(self, "compiled_model_fold_report", compiled)
        object.__setattr__(self, "compile_diagnostic", diagnostic)
        object.__setattr__(
            self,
            "parent_parameter_inventory",
            _immutable_mapping(
                self.parent_parameter_inventory,
                "parent_parameter_inventory",
            ),
        )
        object.__setattr__(
            self,
            "blank_parameter_inventory",
            _immutable_mapping(
                self.blank_parameter_inventory,
                "blank_parameter_inventory",
            ),
        )
        object.__setattr__(
            self,
            "compiled_parameter_inventory",
            _immutable_mapping(
                self.compiled_parameter_inventory,
                "compiled_parameter_inventory",
            ),
        )
        object.__setattr__(self, "zero_operation_audit", audit)

    @property
    def schema(self) -> str:
        return COMPILED_CANDIDATE_CONTEXT_SCHEMA


@dataclass(frozen=True, slots=True)
class V5CompiledCandidateConsumerResult:
    """One separately captured candidate bound to its compiler receipt."""

    candidate: Any
    capture_receipt: Mapping[str, Any]
    capture_receipt_sha256: str
    compiler_receipt_sha256: str
    target_plan_fingerprint: str
    target_state_sha256: str
    parent_sha256: str
    cache_sha256: str
    bridge_sha256: str
    corpus_sha256: str
    source_sha256: str
    compiler_sha256: str
    runner_source_sha256: str
    source_binding_sha256: str
    parent_tokenizer_semantics_sha256: str
    target_tokenizer_semantics_sha256: str
    compile_diagnostic: Mapping[str, Any]
    verification_receipt: Mapping[str, Any] | None = None
    consumer_runtime_guard_receipt: Mapping[str, Any] | None = None
    compiler_child_checkpoint_saved: bool = False
    compiler_publication_attempted: bool = False
    quarantine_candidate_saved: bool = True
    quarantine_publication_nonpromotable: bool = True

    def __post_init__(self) -> None:
        from neural_forge.dynamic_candidate_capture import (
            CANDIDATE_DIRECTORY_NAME,
            QUARANTINE_AUDIT_SCHEMA,
            QUARANTINE_RECEIPT_FILENAME,
            QUARANTINE_VERIFICATION_SCHEMA,
            QuarantineCaptureReceipt,
            QuarantinedDynamicFoldCandidate,
            _candidate_inventory,
        )
        from neural_forge.dynamic_circuit_ir import canonical_sha256

        if type(self.candidate) is not QuarantinedDynamicFoldCandidate:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer result requires the exact generic quarantined candidate type."
            )
        for name in (
            "capture_receipt_sha256",
            "compiler_receipt_sha256",
            "target_plan_fingerprint",
            "target_state_sha256",
            "parent_sha256",
            "cache_sha256",
            "bridge_sha256",
            "corpus_sha256",
            "source_sha256",
            "compiler_sha256",
            "runner_source_sha256",
            "source_binding_sha256",
            "parent_tokenizer_semantics_sha256",
            "target_tokenizer_semantics_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.capture_receipt_sha256 == self.compiler_receipt_sha256:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Capture and compiler receipts must remain distinct."
            )
        diagnostic = _sealed_mapping(
            self.compile_diagnostic,
            seal_field="receipt_sha256",
            label="compile_diagnostic",
        )
        runtime_guard = diagnostic.get("compiler_runtime_fold_only_guard")
        catalog_preparation = diagnostic.get("selective_catalog_artifact")
        guard_implementation = (
            runtime_guard.get("guard_implementation")
            if isinstance(runtime_guard, Mapping)
            else None
        )
        from neural_forge import dynamic_fold_executor

        expected_guard_identity = _fold_only_guard_implementation_identity(
            dynamic_fold_executor
        )
        if guard_implementation != dict(expected_guard_identity):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer result runtime-guard source identity changed."
            )
        if not isinstance(
            catalog_preparation,
            Mapping,
        ) or self.source_binding_sha256 != _source_binding_sha256(
            selective_source_sha256=self.source_sha256,
            compiler_implementation_sha256=self.compiler_sha256,
            runner_source_sha256=self.runner_source_sha256,
            fold_only_runtime_guard_identity_sha256=(
                expected_guard_identity["identity_sha256"]
            ),
            catalog_preparation=catalog_preparation,
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer result source binding is invalid."
            )
        capture = _sealed_mapping(
            self.capture_receipt,
            seal_field="receipt_sha256",
            label="capture_receipt",
        )
        if capture["receipt_sha256"] != self.capture_receipt_sha256:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Capture receipt digest differs from the consumer result."
            )
        candidate_receipt = self.candidate.receipt
        candidate_model = self.candidate.model
        candidate_path = self.candidate.path
        candidate_artifact_path = self.candidate.artifact_path
        if (
            type(candidate_receipt) is not QuarantineCaptureReceipt
            or candidate_receipt.receipt_sha256 != self.capture_receipt_sha256
            or candidate_receipt.as_dict() != dict(capture)
            or not isinstance(candidate_path, Path)
            or not isinstance(candidate_artifact_path, Path)
            or candidate_path != candidate_path.resolve()
            or candidate_artifact_path != candidate_artifact_path.resolve()
            or candidate_artifact_path != candidate_path / CANDIDATE_DIRECTORY_NAME
            or not candidate_path.is_dir()
            or not candidate_artifact_path.is_dir()
            or not (candidate_path / QUARANTINE_RECEIPT_FILENAME).is_file()
            or candidate_model is None
            or self.candidate.promotable is not False
            or self.candidate.publication_authorized is not False
            or self.candidate.requires_reviewed_rerun is not True
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer candidate object differs from its quarantine receipt "
                "or non-promotable lifecycle."
            )
        try:
            persisted_receipt = json.loads(
                (candidate_path / QUARANTINE_RECEIPT_FILENAME).read_text(
                    encoding="utf-8"
                )
            )
            actual_inventory = _candidate_inventory(candidate_artifact_path)
        except Exception as exc:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Could not reauthenticate retained quarantine bytes."
            ) from exc
        candidate_artifact = capture.get("candidate_artifact")
        if (
            persisted_receipt != dict(capture)
            or not isinstance(candidate_artifact, Mapping)
            or candidate_artifact.get("inventory") != actual_inventory
            or candidate_artifact.get("inventory_sha256")
            != actual_inventory.get("inventory_sha256")
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Retained quarantine receipt or artifact inventory changed."
            )
        try:
            from neural_forge.dynamic_fold_executor import (
                fingerprint_model_state,
            )

            candidate_state_sha256 = fingerprint_model_state(candidate_model)
        except Exception as exc:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Could not authenticate the consumer candidate model."
            ) from exc
        if candidate_state_sha256 != self.target_state_sha256:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer candidate model state differs from the captured "
                "compiler target."
            )
        candidate_verification = dict(self.candidate.verification_receipt)
        supplied_verification = (
            candidate_verification
            if self.verification_receipt is None
            else dict(
                _immutable_mapping(
                    self.verification_receipt,
                    "verification_receipt",
                )
            )
        )
        verification_body = dict(supplied_verification)
        verification_sha256 = verification_body.pop(
            "verification_sha256",
            None,
        )
        evaluations = capture.get("evaluations")
        capture_request = capture.get("capture_request")
        filesystem_identity = capture.get("filesystem_identity")
        fresh_execution = supplied_verification.get("fresh_reload_execution")
        observed_audit = supplied_verification.get("capture_observed_audit")
        if (
            supplied_verification != candidate_verification
            or supplied_verification.get("schema") != QUARANTINE_VERIFICATION_SCHEMA
            or _digest(
                verification_sha256,
                "verification_receipt.verification_sha256",
            )
            != canonical_sha256(verification_body)
            or supplied_verification.get("capture_receipt_sha256")
            != self.capture_receipt_sha256
            or supplied_verification.get("destination_path_sha256")
            != canonical_sha256(os.fspath(candidate_path))
            or not isinstance(filesystem_identity, Mapping)
            or filesystem_identity.get("destination_path_sha256")
            != supplied_verification.get("destination_path_sha256")
            or supplied_verification.get("model_state_sha256")
            != self.target_state_sha256
            or supplied_verification.get("artifact_inventory_sha256")
            != actual_inventory.get("inventory_sha256")
            or not isinstance(evaluations, Mapping)
            or supplied_verification.get("output_sha256")
            != evaluations.get("fresh_reload_output_sha256")
            or supplied_verification.get("evaluation_sha256")
            != evaluations.get("fresh_reload_evaluation_sha256")
            or not isinstance(capture_request, Mapping)
            or supplied_verification.get("loader_sha256")
            != capture_request.get("loader_sha256")
            or supplied_verification.get("evaluator_sha256")
            != capture_request.get("evaluator_sha256")
            or supplied_verification.get("fresh_final_destination_reload_performed")
            is not True
            or supplied_verification.get("promotable") is not False
            or supplied_verification.get("publication_authorized") is not False
            or supplied_verification.get("requires_reviewed_rerun") is not True
            or not isinstance(fresh_execution, Mapping)
            or any(
                type(fresh_execution.get(name)) is not int
                or fresh_execution.get(name) <= 0
                for name in (
                    "module_forward_calls",
                    "root_forward_calls",
                    "inference_entrypoint_calls",
                )
            )
            or not isinstance(observed_audit, Mapping)
            or observed_audit.get("schema") != QUARANTINE_AUDIT_SCHEMA
            or observed_audit.get("clean") is not True
            or any(
                type(observed_audit.get(name)) is not int
                or observed_audit.get(name) != 0
                for name in (
                    "optimizer_constructions",
                    "optimizer_steps",
                    "backward_calls",
                    "autograd_grad_calls",
                    "training_steps",
                    "pruning_operations",
                    "quantization_operations",
                    "parent_forward_calls",
                    "parent_inference_calls",
                )
            )
            or observed_audit.get("training_used") is not False
            or observed_audit.get("pruning_used") is not False
            or observed_audit.get("quantization_used") is not False
            or observed_audit.get("live_parent_calls_used") is not False
            or observed_audit.get("pruning_operations_runtime_observed") is not True
            or observed_audit.get("quantization_operations_runtime_observed")
            is not True
            or observed_audit.get(
                "dense_unpruned_unquantized_retained_state_checked_separately"
            )
            is not True
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Final retained-destination reload verification is incomplete "
                "or differs from the quarantine receipt."
            )
        bindings = capture.get("bindings")
        expected_bindings = {
            "plan_sha256": self.target_plan_fingerprint,
            "parent_sha256": self.parent_sha256,
            "cache_sha256": self.cache_sha256,
            "bridge_sha256": self.bridge_sha256,
            "corpus_sha256": self.corpus_sha256,
            "compiler_sha256": self.compiler_receipt_sha256,
            "source_sha256": self.source_binding_sha256,
            "model_state_sha256": self.target_state_sha256,
        }
        if not isinstance(bindings, Mapping) or any(
            bindings.get(name) != value for name, value in expected_bindings.items()
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Capture receipt bindings differ from the compiled candidate."
            )
        if (
            diagnostic.get("compiled_model_fold", {}).get("result_sha256")
            != self.compiler_receipt_sha256
            or self.compiler_child_checkpoint_saved is not False
            or self.compiler_publication_attempted is not False
            or self.quarantine_candidate_saved is not True
            or self.quarantine_publication_nonpromotable is not True
            or capture.get("standalone_candidate") is not True
            or capture.get("promotable") is not False
            or capture.get("publication_authorized") is not False
            or capture.get("requires_reviewed_rerun") is not True
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Consumer result confuses compiler publication with a "
                "non-promotable standalone quarantine capture."
            )
        object.__setattr__(self, "capture_receipt", capture)
        object.__setattr__(self, "compile_diagnostic", diagnostic)
        object.__setattr__(
            self,
            "verification_receipt",
            MappingProxyType(supplied_verification),
        )
        if self.consumer_runtime_guard_receipt is not None:
            consumer_guard = _sealed_mapping(
                self.consumer_runtime_guard_receipt,
                seal_field="report_sha256",
                label="consumer_runtime_guard_receipt",
            )
            if (
                consumer_guard.get("guard_implementation")
                != dict(expected_guard_identity)
                or consumer_guard.get("compiler_runtime_guard_sha256")
                != runtime_guard.get("report_sha256")
                or consumer_guard.get("capture_verification_sha256")
                != supplied_verification.get("verification_sha256")
                or consumer_guard.get("clean") is not True
                or any(
                    type(consumer_guard.get(name)) is not int
                    or consumer_guard.get(name) != 0
                    for name in (
                        "optimizer_constructions",
                        "optimizer_steps",
                        "backward_calls",
                        "autograd_grad_calls",
                        "pruning_operations",
                        "quantization_operations",
                        "parent_module_forward_calls",
                        "parent_root_forward_calls",
                        "parent_inference_entrypoint_calls",
                        "direct_parent_forward_calls",
                    )
                )
            ):
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "Consumer runtime fold-only guard receipt is invalid."
                )
            object.__setattr__(
                self,
                "consumer_runtime_guard_receipt",
                consumer_guard,
            )

    @property
    def schema(self) -> str:
        return COMPILED_CANDIDATE_RESULT_SCHEMA


@dataclass(frozen=True, slots=True)
class V5SelectiveBlankFoldConfig:
    """One local-only, optionally resumable, non-publishing v5 fold request."""

    parent_model_directory: Path
    donor_tokenizer_directory: Path
    target_tokenizer_directory: Path
    bridge_directory: Path
    trace_cache_directory: Path
    trajectory_corpus: Path
    diagnostic_json: Path
    compression: Any
    maximum_process_rss_gib: float
    maximum_logical_resident_gib: float
    minimum_physical_rss_reduction_fraction: float
    model_device: str
    dtype: str
    parent_name: str
    child_name: str
    blank_seed: int
    minimum_hidden_size: int
    maximum_candidates: int
    selected_candidate_rank: int | None
    ridge: float
    boundary_state_reconstruction_weight: float
    boundary_predictive_energy_floor: float
    boundary_validation_retention_floor: float
    short_conv_v4_ridge_alphas: tuple[float, ...]
    short_conv_v4_variable_projection_passes: int
    short_conv_v4_writer_trust_radius: float
    short_conv_v4_fir_trust_radius: float
    short_conv_v4_bcx_trust_radius: float
    short_conv_v4_correction_rms_floor: float
    short_conv_v4_context_objective_weight: float
    maximum_cuda_gib_per_device: float = 20.0
    expected_target_plan_fingerprint: str | None = None
    resume_store_directory: Path | None = None
    resume_admission_sha256: str | None = None
    selective_catalog_store_directory: Path | None = None
    selective_catalog_receipt_json: Path | None = None
    expected_selective_catalog_receipt_sha256: str | None = None
    prepare_selective_catalog_only: bool = False
    forbidden_requests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "parent_model_directory",
            "donor_tokenizer_directory",
            "target_tokenizer_directory",
            "bridge_directory",
            "trace_cache_directory",
            "trajectory_corpus",
            "diagnostic_json",
        ):
            if not isinstance(getattr(self, name), Path):
                raise TypeError(f"{name} must be pathlib.Path.")
        if self.resume_store_directory is not None:
            if not isinstance(self.resume_store_directory, Path):
                raise TypeError("resume_store_directory must be pathlib.Path.")
            raw_resume_path = os.fspath(self.resume_store_directory.expanduser())
            if (
                "://" in raw_resume_path
                or raw_resume_path.startswith("\\\\")
                or raw_resume_path.startswith("//")
            ):
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "resume_store_directory must be an explicit local path."
                )
        catalog_paths = (
            self.selective_catalog_store_directory,
            self.selective_catalog_receipt_json,
        )
        if any(value is None for value in catalog_paths) and any(
            value is not None for value in catalog_paths
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "selective catalog store and receipt paths must be supplied together."
            )
        if self.selective_catalog_store_directory is not None:
            if not isinstance(
                self.selective_catalog_store_directory,
                Path,
            ) or not isinstance(self.selective_catalog_receipt_json, Path):
                raise TypeError(
                    "selective catalog store and receipt must be pathlib.Path."
                )
            _assert_catalog_store_disjoint(
                self.selective_catalog_store_directory,
                roots={
                    "parent_model_directory": self.parent_model_directory,
                    "donor_tokenizer_directory": (self.donor_tokenizer_directory),
                    "target_tokenizer_directory": (self.target_tokenizer_directory),
                    "bridge_directory": self.bridge_directory,
                    "trace_cache_directory": self.trace_cache_directory,
                    "resume_store_directory": (
                        self.resume_store_directory
                        if self.resume_store_directory is not None
                        else self.diagnostic_json.parent / ".no-resume"
                    ),
                    "diagnostic_work_root": self.diagnostic_json.parent,
                    "catalog_receipt_json": (self.selective_catalog_receipt_json),
                },
            )
            receipt_protected_paths = {
                "parent_model_directory": self.parent_model_directory,
                "donor_tokenizer_directory": (self.donor_tokenizer_directory),
                "target_tokenizer_directory": (self.target_tokenizer_directory),
                "bridge_directory": self.bridge_directory,
                "trace_cache_directory": self.trace_cache_directory,
                "trajectory_corpus": self.trajectory_corpus,
                "trajectory_corpus_parent": self.trajectory_corpus.parent,
                "diagnostic_json": self.diagnostic_json,
                "diagnostic_work_root": self.diagnostic_json.parent,
                "selective_catalog_store_directory": (
                    self.selective_catalog_store_directory
                ),
            }
            if self.resume_store_directory is not None:
                receipt_protected_paths["resume_store_directory"] = (
                    self.resume_store_directory
                )
            _assert_catalog_receipt_disjoint(
                self.selective_catalog_receipt_json,
                roots=receipt_protected_paths,
            )
        if not isinstance(self.prepare_selective_catalog_only, bool):
            raise TypeError("prepare_selective_catalog_only must be bool.")
        if self.expected_selective_catalog_receipt_sha256 is not None:
            _digest(
                self.expected_selective_catalog_receipt_sha256,
                "expected_selective_catalog_receipt_sha256",
            )
            if self.selective_catalog_store_directory is None:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "Expected selective catalog receipt requires explicit "
                    "store and receipt paths."
                )
        if (
            self.prepare_selective_catalog_only
            and self.selective_catalog_store_directory is None
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Catalog preparation requires explicit store and receipt paths."
            )
        if self.expected_target_plan_fingerprint is not None and (
            not isinstance(self.expected_target_plan_fingerprint, str)
            or _SHA256.fullmatch(self.expected_target_plan_fingerprint) is None
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "expected_target_plan_fingerprint must be one exact lowercase "
                "SHA-256 digest."
            )
        if self.resume_admission_sha256 is not None:
            if (
                not isinstance(self.resume_admission_sha256, str)
                or _SHA256.fullmatch(self.resume_admission_sha256) is None
            ):
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "resume_admission_sha256 must be one exact lowercase "
                    "SHA-256 digest."
                )
            if self.resume_store_directory is None:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "resume_admission_sha256 requires an explicit local "
                    "resume_store_directory."
                )
        if not hasattr(self.compression, "as_kwargs") or not callable(
            self.compression.as_kwargs
        ):
            raise TypeError("compression must be the shared CompressionRequest.")
        process_limit = _finite_float(
            self.maximum_process_rss_gib,
            "maximum_process_rss_gib",
        )
        if process_limit <= 0.0 or int(process_limit * _GIB) <= 0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "maximum_process_rss_gib must set a positive absolute bound."
            )
        logical_limit = _finite_float(
            self.maximum_logical_resident_gib,
            "maximum_logical_resident_gib",
        )
        if logical_limit < 0.0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "maximum_logical_resident_gib must be nonnegative."
            )
        cuda_limit = _finite_float(
            self.maximum_cuda_gib_per_device,
            "maximum_cuda_gib_per_device",
        )
        if cuda_limit <= 0.0 or int(cuda_limit * _GIB) <= 0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "maximum_cuda_gib_per_device must set a positive bound."
            )
        reduction = _finite_float(
            self.minimum_physical_rss_reduction_fraction,
            "minimum_physical_rss_reduction_fraction",
        )
        if not 0.0 <= reduction <= 1.0:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "minimum_physical_rss_reduction_fraction must be in [0, 1]."
            )
        object.__setattr__(self, "maximum_process_rss_gib", process_limit)
        object.__setattr__(
            self,
            "maximum_logical_resident_gib",
            logical_limit,
        )
        object.__setattr__(
            self,
            "minimum_physical_rss_reduction_fraction",
            reduction,
        )
        object.__setattr__(
            self,
            "maximum_cuda_gib_per_device",
            cuda_limit,
        )
        if self.model_device not in {"cpu", "cuda:0"}:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "model_device must be cpu or cuda:0; replay devices are fixed "
                "to logical CUDA 0 and 1."
            )
        if self.dtype != "bfloat16":
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "The sealed two-RTX-3090 replay policy requires bfloat16."
            )
        if not self.parent_name.strip() or not self.child_name.strip():
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "parent_name and child_name must be nonempty."
            )
        if isinstance(self.blank_seed, bool) or not isinstance(
            self.blank_seed,
            int,
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "blank_seed must be an integer."
            )
        for value, name in (
            (self.minimum_hidden_size, "minimum_hidden_size"),
            (self.maximum_candidates, "maximum_candidates"),
        ):
            _positive_int(value, name)
        if self.selected_candidate_rank is not None and (
            isinstance(self.selected_candidate_rank, bool)
            or not isinstance(self.selected_candidate_rank, int)
            or self.selected_candidate_rank < 0
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "selected_candidate_rank must be nonnegative."
            )
        for value, name in (
            (self.ridge, "ridge"),
            (
                self.boundary_state_reconstruction_weight,
                "boundary_state_reconstruction_weight",
            ),
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            normalized = _finite_float(value, name)
            if normalized < 0.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be nonnegative."
                )
        for value, name in (
            (
                self.boundary_predictive_energy_floor,
                "boundary_predictive_energy_floor",
            ),
            (
                self.boundary_validation_retention_floor,
                "boundary_validation_retention_floor",
            ),
        ):
            if float(value) > 1.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be within [0, 1]."
                )
        alphas = tuple(
            _finite_float(value, "short_conv_v4_ridge_alphas")
            for value in self.short_conv_v4_ridge_alphas
        )
        if (
            not alphas
            or any(value < 0.0 for value in alphas)
            or tuple(sorted(set(alphas))) != alphas
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_ridge_alphas must be unique, nonnegative, "
                "and increasing."
            )
        object.__setattr__(self, "short_conv_v4_ridge_alphas", alphas)
        if self.short_conv_v4_variable_projection_passes not in {2, 3}:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_variable_projection_passes must be 2 or 3."
            )
        for value, name in (
            (
                self.short_conv_v4_writer_trust_radius,
                "short_conv_v4_writer_trust_radius",
            ),
            (
                self.short_conv_v4_fir_trust_radius,
                "short_conv_v4_fir_trust_radius",
            ),
            (
                self.short_conv_v4_bcx_trust_radius,
                "short_conv_v4_bcx_trust_radius",
            ),
            (
                self.short_conv_v4_correction_rms_floor,
                "short_conv_v4_correction_rms_floor",
            ),
        ):
            if _finite_float(value, name) <= 0.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"{name} must be positive."
                )
        context_weight = _finite_float(
            self.short_conv_v4_context_objective_weight,
            "short_conv_v4_context_objective_weight",
        )
        if not 0.0 <= context_weight <= 0.1:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "short_conv_v4_context_objective_weight must be in [0, 0.1]."
            )
        forbidden = tuple(sorted(set(self.forbidden_requests)))
        if forbidden:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "V5 selective folding forbids requested resume, publication, "
                "trace regeneration, training, pruning, quantization, or "
                f"download actions: {', '.join(forbidden)}."
            )
        object.__setattr__(self, "forbidden_requests", forbidden)

    @property
    def maximum_process_rss_bytes(self) -> int:
        return int(self.maximum_process_rss_gib * _GIB)

    @property
    def maximum_cuda_bytes_per_device(self) -> int:
        return int(self.maximum_cuda_gib_per_device * _GIB)

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_model_directory": str(self.parent_model_directory),
            "donor_tokenizer_directory": str(self.donor_tokenizer_directory),
            "target_tokenizer_directory": str(self.target_tokenizer_directory),
            "bridge_directory": str(self.bridge_directory),
            "trace_cache_directory": str(self.trace_cache_directory),
            "trajectory_corpus": str(self.trajectory_corpus),
            "diagnostic_json": str(self.diagnostic_json),
            "compression": self.compression.as_kwargs(),
            "maximum_process_rss_gib": self.maximum_process_rss_gib,
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "maximum_logical_resident_gib": (self.maximum_logical_resident_gib),
            "minimum_physical_rss_reduction_fraction": (
                self.minimum_physical_rss_reduction_fraction
            ),
            "maximum_cuda_gib_per_device": self.maximum_cuda_gib_per_device,
            "maximum_cuda_bytes_per_device": (self.maximum_cuda_bytes_per_device),
            "model_device": self.model_device,
            "dtype": self.dtype,
            "parent_name": self.parent_name,
            "child_name": self.child_name,
            "blank_seed": self.blank_seed,
            "minimum_hidden_size": self.minimum_hidden_size,
            "maximum_candidates": self.maximum_candidates,
            "selected_candidate_rank": self.selected_candidate_rank,
            "ridge": self.ridge,
            "boundary_state_reconstruction_weight": (
                self.boundary_state_reconstruction_weight
            ),
            "boundary_predictive_energy_floor": (self.boundary_predictive_energy_floor),
            "boundary_validation_retention_floor": (
                self.boundary_validation_retention_floor
            ),
            "short_conv_v4_ridge_alphas": list(self.short_conv_v4_ridge_alphas),
            "short_conv_v4_variable_projection_passes": (
                self.short_conv_v4_variable_projection_passes
            ),
            "short_conv_v4_writer_trust_radius": (
                self.short_conv_v4_writer_trust_radius
            ),
            "short_conv_v4_fir_trust_radius": (self.short_conv_v4_fir_trust_radius),
            "short_conv_v4_bcx_trust_radius": (self.short_conv_v4_bcx_trust_radius),
            "short_conv_v4_correction_rms_floor": (
                self.short_conv_v4_correction_rms_floor
            ),
            "short_conv_v4_context_objective_weight": (
                self.short_conv_v4_context_objective_weight
            ),
            "expected_target_plan_fingerprint": (self.expected_target_plan_fingerprint),
            "resume_store_directory": (
                None
                if self.resume_store_directory is None
                else str(self.resume_store_directory)
            ),
            "resume_admission_sha256": self.resume_admission_sha256,
            "resume_checkpointing_requested": (self.resume_store_directory is not None),
            "resume_requested": self.resume_admission_sha256 is not None,
            "selective_catalog_store_directory": (
                None
                if self.selective_catalog_store_directory is None
                else str(self.selective_catalog_store_directory)
            ),
            "selective_catalog_receipt_json": (
                None
                if self.selective_catalog_receipt_json is None
                else str(self.selective_catalog_receipt_json)
            ),
            "expected_selective_catalog_receipt_sha256": (
                self.expected_selective_catalog_receipt_sha256
            ),
            "prepare_selective_catalog_only": (self.prepare_selective_catalog_only),
            "production_catalog_warm_hit_required": (
                not self.prepare_selective_catalog_only
            ),
            "publication_requested": False,
        }


def _request_derived_structural_compression_floor(compression: Any) -> float:
    """Use the catalog request as a structural check, never as a success claim."""

    ratio = getattr(compression, "compression_ratio", None)
    return 1.0 if ratio is None else float(ratio)


@dataclass(frozen=True, slots=True)
class _GpuReplayBundle:
    evaluator: Any
    report: Mapping[str, Any]


@dataclass(slots=True)
class _ResumeProgressTracker:
    """Retain exact typed v5 progress references for run diagnostics."""

    store_directory: Path | None
    requested_admission_sha256: str | None
    catalog_preparation: Mapping[str, Any] | None = None
    invocation_head: dict[str, Any] | None = None
    published_pending_prefixes: list[dict[str, Any]] = field(default_factory=list)
    admitted_prefixes: list[dict[str, Any]] = field(default_factory=list)
    rejected_prefixes: list[dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def _pending_record(pending: Any) -> dict[str, Any]:
        return {
            "next_layer_index": int(pending.next_layer_index),
            "frontier_reference": pending.frontier_reference.as_dict(),
            "checkpoint_reference": pending.checkpoint.reference.as_dict(),
            "frontier_sha256": str(pending.frontier_sha256),
            "event_report_sha256": str(pending.event_report_sha256),
        }

    def record_invocation_head(self, state: Any) -> None:
        self.invocation_head = {
            "admission_sha256": str(state.admission["admission_sha256"]),
            "next_layer_index": int(state.next_layer_index),
            "frontier_reference": state.frontier_reference.as_dict(),
            "checkpoint_reference": state.checkpoint_reference.as_dict(),
            "frontier_sha256": str(state.frontier.frontier_sha256),
            "opened_by_exact_admission_digest": True,
            "typed_v5_state_opened": True,
        }

    def record_pending(self, pending: Any) -> None:
        self.published_pending_prefixes.append(self._pending_record(pending))

    def record_admitted(
        self,
        pending: Any,
        admission: Mapping[str, Any],
    ) -> None:
        record = self._pending_record(pending)
        record.pop("event_report_sha256")
        record["admission_sha256"] = str(admission["admission_sha256"])
        self.admitted_prefixes.append(record)

    def record_rejected(
        self,
        pending: Any,
        rejection: Mapping[str, Any],
    ) -> None:
        record = self._pending_record(pending)
        record["rejection_sha256"] = str(rejection["rejection_sha256"])
        record["promotable"] = False
        self.rejected_prefixes.append(record)

    @staticmethod
    def _ordered(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            dict(record)
            for record in sorted(
                records,
                key=lambda item: (
                    int(item["next_layer_index"]),
                    str(item["checkpoint_reference"]["artifact_key"]),
                ),
            )
        ]

    def as_dict(self) -> dict[str, Any]:
        published = self._ordered(self.published_pending_prefixes)
        admitted = self._ordered(self.admitted_prefixes)
        rejected = self._ordered(self.rejected_prefixes)
        terminal_keys = {
            str(record["checkpoint_reference"]["artifact_key"])
            for record in (*admitted, *rejected)
        }
        outstanding = [
            dict(record)
            for record in published
            if str(record["checkpoint_reference"]["artifact_key"]) not in terminal_keys
        ]
        catalog_authentication = (
            None
            if self.catalog_preparation is None
            else dict(_catalog_authentication_summary(self.catalog_preparation))
        )
        body = {
            "schema": RESUME_PROGRESS_SCHEMA,
            "resume_store_directory": (
                None if self.store_directory is None else str(self.store_directory)
            ),
            "checkpointing_enabled": self.store_directory is not None,
            "requested_admission_sha256": (self.requested_admission_sha256),
            "resume_requested": (self.requested_admission_sha256 is not None),
            "selective_catalog_artifact_bound": (catalog_authentication is not None),
            "selective_catalog_artifact": catalog_authentication,
            "invocation_head": self.invocation_head,
            "published_pending_prefixes": published,
            "outstanding_pending_prefixes": outstanding,
            "admitted_prefixes": admitted,
            "rejected_prefixes": rejected,
            "published_pending_prefix_count": len(published),
            "outstanding_pending_prefix_count": len(outstanding),
            "admitted_prefix_count": len(admitted),
            "rejected_prefix_count": len(rejected),
            "only_child_prefix_tensors_archived": True,
            "parent_or_cache_tensor_sidecars": 0,
            "child_publication_attempted": False,
            "operation_audit": {
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "generation_calls": 0,
                "parent_cache_generation_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            },
        }
        return {**body, "report_sha256": _sha(body)}


@dataclass(frozen=True, slots=True)
class _ResumeRuntime:
    coordinator: Any | None
    state: Any | None
    progress: _ResumeProgressTracker


def _exact_resume_coordinator(
    coordinator_type: type,
    store: Any,
) -> Any:
    """Construct the production coordinator without altering its class identity."""

    coordinator = coordinator_type(store)
    if type(coordinator) is not coordinator_type:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The resume runtime did not construct the exact production "
            "coordinator type."
        )
    return coordinator


def _observe_compiled_resume_receipt(
    compiled: Any,
    progress: _ResumeProgressTracker,
) -> None:
    """Copy diagnostic progress from the compiler's authenticated proof chain."""

    report = getattr(compiled, "report", None)
    if not isinstance(report, Mapping):
        return
    records = report.get("resume_admitted_prefixes")
    if records is None:
        return
    if not isinstance(records, Sequence) or isinstance(
        records,
        (str, bytes, bytearray),
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The compiler resume-admission receipt is not an ordered sequence."
        )
    published: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "A compiler resume-admission record is not a mapping."
            )
        proof = record.get("admission_chain_proof")
        if not isinstance(proof, Mapping):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "A compiler resume-admission record lacks its authenticated "
                "admission-chain proof."
            )
        pending = proof.get("pending_document")
        if not isinstance(pending, Mapping):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "A compiler admission-chain proof lacks its pending document."
            )
        publication = {
            "next_layer_index": int(pending["next_layer_index"]),
            "frontier_reference": _plain_json(pending["frontier_reference"]),
            "checkpoint_reference": _plain_json(pending["checkpoint_reference"]),
            "frontier_sha256": str(pending["frontier_sha256"]),
            "event_report_sha256": str(pending["event_report_sha256"]),
        }
        if (
            publication["next_layer_index"] != record.get("next_layer_index")
            or publication["frontier_reference"] != record.get("frontier_reference")
            or publication["checkpoint_reference"] != record.get("checkpoint_reference")
            or publication["frontier_sha256"] != record.get("frontier_sha256")
            or proof.get("proof_sha256") != record.get("admission_chain_proof_sha256")
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "The compiler resume-admission summary differs from its "
                "authenticated admission-chain proof."
            )
        published.append(publication)
        admitted.append(_plain_json(record))
    progress.published_pending_prefixes = published
    progress.admitted_prefixes = admitted
    progress.rejected_prefixes = []


def _attach_resume_progress(
    error: Exception,
    progress: _ResumeProgressTracker,
) -> None:
    try:
        error.resume_progress_report = progress.as_dict()
    except Exception as attachment_error:
        error.add_note(
            "Authenticated resume progress could not be attached to the "
            f"exception: {attachment_error}"
        )


def _build_resume_runtime(
    config: V5SelectiveBlankFoldConfig,
    *,
    catalog_preparation: Mapping[str, Any] | None = None,
) -> _ResumeRuntime:
    progress = _ResumeProgressTracker(
        store_directory=(
            None
            if config.resume_store_directory is None
            else config.resume_store_directory.expanduser().absolute()
        ),
        requested_admission_sha256=config.resume_admission_sha256,
        catalog_preparation=catalog_preparation,
    )
    if config.resume_store_directory is None:
        return _ResumeRuntime(
            coordinator=None,
            state=None,
            progress=progress,
        )

    from neural_forge.hf_lfm2_cpcf_resume_artifact import (
        Lfm2CpcfResumeArtifactStore,
    )
    from neural_forge.hf_lfm2_cpcf_v5_resume import (
        Lfm2CpcfV5ResumeCoordinator,
        Lfm2CpcfV5ResumeState,
    )

    try:
        store = Lfm2CpcfResumeArtifactStore(config.resume_store_directory)
        progress.store_directory = store.root
        coordinator = _exact_resume_coordinator(
            Lfm2CpcfV5ResumeCoordinator,
            store,
        )
        state = None
        if config.resume_admission_sha256 is not None:
            state = coordinator.open_admitted(config.resume_admission_sha256)
            if type(state) is not Lfm2CpcfV5ResumeState:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "The resume admission did not open an exact typed v5 resume state."
                )
            if catalog_preparation is not None:
                catalog_authentication = _catalog_authentication_summary(
                    catalog_preparation
                )
                context = state.frontier.context
                if (
                    context.get("selective_catalog_sha256")
                    != catalog_authentication["selective_catalog_sha256"]
                    or context.get("selective_source_sha256")
                    != catalog_authentication["selective_source_sha256"]
                    or context.get("selective_provider_sha256")
                    != catalog_authentication["selective_provider_sha256"]
                ):
                    raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                        "Resume state selective identities differ from the "
                        "prepared catalog artifact."
                    )
            progress.record_invocation_head(state)
    except Exception as error:
        _attach_resume_progress(error, progress)
        raise
    return _ResumeRuntime(
        coordinator=coordinator,
        state=state,
        progress=progress,
    )


def _compile_with_authenticated_resume(
    compile_function: Any,
    source_model: Any,
    target_model: Any,
    *,
    resume_runtime: _ResumeRuntime,
    compile_kwargs: Mapping[str, Any],
) -> Any:
    """Invoke v5 with only the typed coordinator/state continuation seam."""

    if "resume_state" in compile_kwargs or "resume_coordinator" in compile_kwargs:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Resume inputs must come only from the authenticated runner seam."
        )
    try:
        compiled = compile_function(
            source_model,
            target_model,
            **dict(compile_kwargs),
            resume_state=resume_runtime.state,
            resume_coordinator=resume_runtime.coordinator,
        )
        _observe_compiled_resume_receipt(
            compiled,
            resume_runtime.progress,
        )
        return compiled
    except Exception as error:
        _attach_resume_progress(error, resume_runtime.progress)
        raise


def _configure_torch() -> Any:
    """Pin the deterministic backend before CUDA is initialized."""

    import torch

    if torch.cuda.is_initialized():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CUDA initialized before the deterministic replay contract."
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    return torch


def _enforce_cuda_allocator_caps(
    torch: Any,
    *,
    maximum_cuda_bytes_per_device: int,
) -> Mapping[str, Any]:
    """Install one hard PyTorch allocator fraction on both replay devices."""

    requested = _positive_int(
        maximum_cuda_bytes_per_device,
        "maximum_cuda_bytes_per_device",
    )
    cuda = getattr(torch, "cuda", None)
    set_fraction = getattr(cuda, "set_per_process_memory_fraction", None)
    get_properties = getattr(cuda, "get_device_properties", None)
    if (
        cuda is None
        or not callable(getattr(cuda, "is_available", None))
        or not cuda.is_available()
        or not callable(getattr(cuda, "device_count", None))
        or int(cuda.device_count()) < len(GPU_LOGICAL_DEVICES)
        or not callable(set_fraction)
        or not callable(get_properties)
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The per-device CUDA allocator cap cannot be enforced."
        )
    devices = []
    for index in GPU_LOGICAL_DEVICES:
        try:
            properties = get_properties(index)
            device_name = str(getattr(properties, "name", ""))
            if device_name != EXPECTED_GPU_NAME:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"Logical CUDA device {index} is not the authenticated "
                    f"{EXPECTED_GPU_NAME}."
                )
            total = _positive_int(
                int(properties.total_memory),
                f"cuda:{index}.total_memory",
            )
            allowed = min(requested, total)
            fraction = min(1.0, float(allowed) / float(total))
            while int(math.floor(float(total) * fraction)) > allowed:
                fraction = math.nextafter(fraction, 0.0)
            if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    f"cuda:{index} allocator fraction is not enforceable."
                )
            set_fraction(fraction, device=index)
        except Lfm2CpcfV5SelectiveBlankFoldRunnerError:
            raise
        except Exception as exc:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Could not enforce the CUDA allocator cap on logical device {index}."
            ) from exc
        effective = int(math.floor(float(total) * fraction))
        if effective <= 0 or effective > requested or effective > total:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Logical CUDA device {index} allocator cap is invalid."
            )
        devices.append(
            {
                "logical_device_index": index,
                "device_name": device_name,
                "observed_total_memory_bytes": total,
                "requested_maximum_bytes": requested,
                "allowed_bytes": allowed,
                "allocator_fraction": fraction,
                "effective_allocator_maximum_bytes": effective,
                "set_per_process_memory_fraction_called": True,
                "enforced_before_substantive_cuda_allocations": True,
            }
        )
    body = {
        "schema": CUDA_ALLOCATOR_POLICY_SCHEMA,
        "logical_devices": list(GPU_LOGICAL_DEVICES),
        "requested_maximum_cuda_bytes_per_device": requested,
        "device_receipts": devices,
        "all_devices_enforced": len(devices) == len(GPU_LOGICAL_DEVICES),
        "enforced_before_substantive_cuda_allocations": True,
    }
    return MappingProxyType({**body, "receipt_sha256": _sha(body)})


def _expected_gpu_scheduler_policy() -> dict[str, Any]:
    return {
        "expected_device_count": 2,
        "expected_device_name": EXPECTED_GPU_NAME,
        "assignment": "canonical_round_robin_by_first_shard_key",
        "pinned_buffer_slots_per_device": 2,
        "maximum_pinned_bytes_per_slot": GPU_PINNED_BYTES_PER_SLOT,
        "parity_absolute_tolerance": GPU_PARITY_ABSOLUTE_TOLERANCE,
        "parity_relative_tolerance": GPU_PARITY_RELATIVE_TOLERANCE,
        "parity_maximum_relative_rmse": (GPU_PARITY_MAXIMUM_RELATIVE_RMSE),
        "parity_minimum_cosine_similarity": (GPU_PARITY_MINIMUM_COSINE_SIMILARITY),
        "require_deterministic_debug_mode": True,
        "cpu_cuda_parity_required_before_scheduled_work": True,
        "runtime_reauthenticated_after_parity_and_work": True,
        "installed_state_reauthenticated_after_parity_and_work": True,
        "folding_math_changed": False,
    }


def _load_proven_gpu_parity_receipt(
    path: Path = PROVEN_GPU_PARITY_RECEIPT,
) -> Mapping[str, Any]:
    """Authenticate the prior real two-3090 candidate-replay proof."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The proven local two-RTX-3090 parity receipt is unavailable."
        ) from exc
    if not isinstance(value, dict):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The proven GPU parity receipt is not an object."
        )
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    execution = value.get("backend_report", {}).get(
        "execution_report",
        {},
    )
    policy = execution.get("plan", {}).get("policy")
    devices = value.get("devices")
    if (
        value.get("schema") != "neural-forge-lfm2-cpcf-candidate-gpu-parity-v1"
        or value.get("status") != "passed"
        or claimed != _sha(body)
        or value.get("target_dtype") != "bfloat16"
        or value.get("candidate_state_sha256_equal") is not True
        or value.get("backend_report", {}).get("representative_cpu_cuda_parity_passed")
        is not True
        or execution.get("backend_receipts_reauthenticated") is not True
        or execution.get("installed_state_reauthenticated") is not True
        or policy != _expected_gpu_scheduler_policy()
        or not isinstance(devices, list)
        or [row.get("device_index") for row in devices] != [0, 1]
        or any(row.get("device_name") != EXPECTED_GPU_NAME for row in devices)
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The local two-RTX-3090 parity proof is incomplete or changed."
        )
    source_paths = {
        "dynamic_replay_scheduler.py": (
            _REPOSITORY_ROOT / "src" / "neural_forge" / "dynamic_replay_scheduler.py"
        ),
        "hf_lfm2_cpcf_candidate_replay.py": (
            _REPOSITORY_ROOT
            / "src"
            / "neural_forge"
            / "hf_lfm2_cpcf_candidate_replay.py"
        ),
        "verify_lfm2_cpcf_candidate_replay_gpu_parity.py": (
            _REPOSITORY_ROOT
            / "experiments"
            / "verify_lfm2_cpcf_candidate_replay_gpu_parity.py"
        ),
    }
    expected_sources = value.get("source_sha256")
    observed_sources = {
        name: _file_sha256(source) for name, source in source_paths.items()
    }
    if expected_sources != observed_sources:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "GPU replay source changed after the proven parity receipt."
        )
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "receipt_sha256": claimed,
        "source_sha256": observed_sources,
        "scheduler_policy": dict(policy),
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }


def _build_authenticated_gpu_replay_evaluator(torch: Any) -> _GpuReplayBundle:
    """Build the fixed logical-CUDA-0/1 scheduler used by the proven lane."""

    from neural_forge.dynamic_replay_scheduler import (
        AuthenticatedPinnedHostBufferFactory,
        AuthenticatedReplayStateProbes,
        DeterministicReplayScheduler,
        ReplayReplicaReceipt,
        ReplaySchedulerPolicy,
        capture_cuda_replay_device_receipt,
        fingerprint_installed_replay_state,
    )
    from neural_forge.hf_lfm2_cpcf_candidate_replay import (
        AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
        lfm2_cpcf_cpu_parity_worker,
        lfm2_cpcf_cuda_replay_worker,
    )

    parity_proof = dict(_load_proven_gpu_parity_receipt())
    if not torch.cuda.is_available() or int(torch.cuda.device_count()) < 2:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Logical CUDA devices 0 and 1 must both be available."
        )
    contract_cpu = {
        "candidate_replay_contract": torch.arange(
            64,
            dtype=torch.float32,
        )
    }
    contract_gpu = {
        index: {
            name: value.to(device=f"cuda:{index}")
            for name, value in contract_cpu.items()
        }
        for index in GPU_LOGICAL_DEVICES
    }
    cpu_fingerprint = fingerprint_installed_replay_state(contract_cpu)
    device_receipts = {
        index: capture_cuda_replay_device_receipt(index)
        for index in GPU_LOGICAL_DEVICES
    }
    if (
        tuple(sorted(device_receipts)) != GPU_LOGICAL_DEVICES
        or any(
            receipt.device_index != index or receipt.device_name != EXPECTED_GPU_NAME
            for index, receipt in device_receipts.items()
        )
        or len(
            {receipt.hardware_profile_sha256 for receipt in device_receipts.values()}
        )
        != 1
        or len({receipt.runtime_profile_sha256 for receipt in device_receipts.values()})
        != 1
        or len({receipt.device_uuid for receipt in device_receipts.values()}) != 2
        or len(
            {
                (
                    receipt.pci_domain_id,
                    receipt.pci_bus_id,
                    receipt.pci_device_id,
                )
                for receipt in device_receipts.values()
            }
        )
        != 2
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Logical CUDA 0/1 are not two distinct homogeneous RTX 3090s."
        )
    gpu_fingerprints = {
        index: fingerprint_installed_replay_state(contract_gpu[index])
        for index in GPU_LOGICAL_DEVICES
    }
    if any(fingerprint != cpu_fingerprint for fingerprint in gpu_fingerprints.values()):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CPU and CUDA replay-contract fingerprints differ."
        )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=device_receipts[index],
            installed_state=gpu_fingerprints[index],
        )
        for index in GPU_LOGICAL_DEVICES
    )
    scheduler_policy = ReplaySchedulerPolicy(
        expected_device_count=2,
        expected_device_name=EXPECTED_GPU_NAME,
        pinned_buffer_slots_per_device=2,
        maximum_pinned_bytes_per_slot=GPU_PINNED_BYTES_PER_SLOT,
        parity_absolute_tolerance=GPU_PARITY_ABSOLUTE_TOLERANCE,
        parity_relative_tolerance=GPU_PARITY_RELATIVE_TOLERANCE,
        parity_maximum_relative_rmse=(GPU_PARITY_MAXIMUM_RELATIVE_RMSE),
        parity_minimum_cosine_similarity=(GPU_PARITY_MINIMUM_COSINE_SIMILARITY),
        require_deterministic_debug_mode=True,
    )
    if scheduler_policy.as_dict() != _expected_gpu_scheduler_policy():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Live GPU scheduler policy differs from the proven receipt."
        )
    state_probes = AuthenticatedReplayStateProbes(
        cpu_tensors=contract_cpu,
        gpu_tensors=contract_gpu,
    )
    scheduler = DeterministicReplayScheduler(
        replicas=replicas,
        policy=scheduler_policy,
        gpu_worker=lfm2_cpcf_cuda_replay_worker,
        cpu_parity_worker=lfm2_cpcf_cpu_parity_worker,
        device_receipt_probe=capture_cuda_replay_device_receipt,
        installed_state_probe=state_probes.installed_state_probe,
        cpu_installed_state_probe=state_probes.cpu_installed_state_probe,
        host_buffer_factory=AuthenticatedPinnedHostBufferFactory(
            scheduler_policy.maximum_pinned_bytes_per_slot
        ),
    )
    runtime_authentication = dict(scheduler.authenticate_runtime())
    evaluator = AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator(scheduler)
    body = {
        "schema": GPU_REPLAY_SCHEMA,
        "logical_device_indices": list(GPU_LOGICAL_DEVICES),
        "devices": [device_receipts[index].as_dict() for index in GPU_LOGICAL_DEVICES],
        "replicas": [replica.as_dict() for replica in replicas],
        "cpu_installed_state": cpu_fingerprint.as_dict(),
        "scheduler_policy": scheduler_policy.as_dict(),
        "runtime_authentication": runtime_authentication,
        "runtime_authentication_sha256": (runtime_authentication["report_sha256"]),
        "proven_parity_receipt": parity_proof,
        "contract_tensor_names": sorted(contract_cpu),
        "contract_cpu_tensors_retained_for_runtime_reauthentication": True,
        "contract_gpu_tensors_retained_for_runtime_reauthentication": True,
        "homogeneous_authenticated_receipts": True,
        "candidate_payload_staging": (
            "bounded_double_buffered_pinned_system_memory_to_cuda"
        ),
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return _GpuReplayBundle(
        evaluator=evaluator,
        report={**body, "report_sha256": _sha(body)},
    )


def _maximum_current_plus_next_bytes(references: Sequence[Any]) -> int:
    ordered = tuple(references)
    if not ordered:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog contains no layer references."
        )
    return max(
        int(reference.logical_bytes)
        + (0 if index + 1 == len(ordered) else int(ordered[index + 1].logical_bytes))
        for index, reference in enumerate(ordered)
    )


def _resolved_logical_resident_bytes(
    *,
    references: Sequence[Any],
    final_norm_logical_bytes: int,
    requested_gib: float,
) -> tuple[int, int]:
    required = max(
        _maximum_current_plus_next_bytes(references),
        _positive_int(
            int(final_norm_logical_bytes),
            "final_norm_logical_bytes",
        ),
    )
    resolved = required if requested_gib == 0.0 else int(requested_gib * _GIB)
    if resolved < required:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The logical evidence budget is below the largest authenticated "
            "current-plus-next layer pair or final-norm payload."
        )
    return resolved, required


def _derive_effective_rss_reduction_fraction(
    *,
    full_phase_tree_logical_bytes: int,
    compiler_baseline_rss_bytes: int,
    maximum_process_rss_bytes: int,
    requested_minimum_reduction_fraction: float,
) -> tuple[float, int]:
    """Translate the absolute process cap into the compiler's delta policy."""

    full = _positive_int(
        full_phase_tree_logical_bytes,
        "full_phase_tree_logical_bytes",
    )
    baseline = _positive_int(
        compiler_baseline_rss_bytes,
        "compiler_baseline_rss_bytes",
    )
    limit = _positive_int(
        maximum_process_rss_bytes,
        "maximum_process_rss_bytes",
    )
    requested = _finite_float(
        requested_minimum_reduction_fraction,
        "requested_minimum_reduction_fraction",
    )
    if not 0.0 <= requested <= 1.0:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "requested_minimum_reduction_fraction must be in [0, 1]."
        )
    if baseline > limit:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler baseline RSS already exceeds --maximum-process-rss-gib."
        )
    absolute_headroom = limit - baseline
    reduction_headroom = int(math.floor(full * (1.0 - requested)))
    allowed_increase = min(full, absolute_headroom, reduction_headroom)
    effective = max(requested, 1.0 - (allowed_increase / full))
    # Float conversion must never relax the integer byte cap.
    while int(math.floor(full * (1.0 - effective))) > allowed_increase:
        adjusted = math.nextafter(effective, 1.0)
        if adjusted == effective:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Cannot encode the absolute RSS cap in the v5 runtime policy."
            )
        effective = adjusted
    return effective, allowed_increase


def _memory_sample(
    v5_compiler: Any,
    *,
    label: str,
    maximum_process_rss_bytes: int,
) -> dict[str, Any]:
    current = int(v5_compiler._process_rss_bytes())
    peak = int(v5_compiler._process_peak_rss_bytes())
    sample = {
        "label": label,
        "current_rss_bytes": current,
        "process_high_water_bytes": peak,
        "maximum_process_rss_bytes": maximum_process_rss_bytes,
        "passed": max(current, peak) <= maximum_process_rss_bytes,
    }
    if sample["passed"] is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"Process RSS exceeded the absolute cap at {label!r}: "
            f"{max(current, peak)} > {maximum_process_rss_bytes}."
        )
    return sample


def _walk_mappings(value: Any) -> Sequence[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    pending = [value]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(current, Mapping):
            result.append(current)
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return result


def _gpu_execution_audit(
    layer_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    executions = [
        mapping
        for report in layer_reports
        for mapping in _walk_mappings(report)
        if mapping.get("schema") == "neural-forge-dynamic-replay-execution-v1"
    ]
    if any(
        execution.get("backend_receipts_reauthenticated") is not True
        or execution.get("installed_state_reauthenticated") is not True
        or execution.get("parity_passed_before_scheduled_work") is not True
        or {
            int(report["device_index"])
            for report in execution.get("parity_reports", ())
        }
        != set(GPU_LOGICAL_DEVICES)
        for execution in executions
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "A candidate replay execution lacks two-GPU parity or "
            "reauthentication evidence."
        )
    return {
        "candidate_gpu_replay_invoked": bool(executions),
        "execution_count": len(executions),
        "execution_report_sha256": [
            execution.get("report_sha256") for execution in executions
        ],
        "both_logical_devices_reauthenticated_per_execution": bool(executions),
        "no_gpu_work_claimed_when_no_candidate_batch_was_scheduled": True,
    }


def _live_model_authentication(model: Any) -> Mapping[str, Any]:
    """Fingerprint state, topology/runtime bindings, and parameter inventory."""

    from neural_forge.dynamic_fold_executor import (
        fingerprint_model_runtime_state,
        fingerprint_model_state,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        independent_parameter_inventory,
    )

    try:
        return MappingProxyType(
            {
                "state_sha256": fingerprint_model_state(model),
                "runtime_sha256": fingerprint_model_runtime_state(model),
                "parameter_inventory": dict(independent_parameter_inventory(model)),
            }
        )
    except Exception as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Could not authenticate live candidate-consumer model state."
        ) from exc


def _dense_retained_state_facts(
    model: Any,
    *,
    label: str,
) -> Mapping[str, Any]:
    """Prove retained dense state without claiming complete operation history."""

    from neural_forge.dynamic_candidate_capture import (
        _assert_dense_unpruned_model,
    )

    try:
        _assert_dense_unpruned_model(model)
        parameters = tuple(model.named_parameters())
        buffers = tuple(model.named_buffers())
    except Exception as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"{label} is not retained dense, unpruned, unquantized state."
        ) from exc
    return MappingProxyType(
        {
            "label": label,
            "parameter_tensor_count": len(parameters),
            "buffer_tensor_count": len(buffers),
            "dense_materialized_retained_state": True,
            "pruning_reparameterizations_absent": True,
            "quantized_parameters_or_module_wrappers_absent": True,
            "evidence_scope": ("retained-state-fact-not-complete-operation-history"),
        }
    )


def _resolved_ast_symbol(
    node: ast.AST,
    bindings: Mapping[str, str],
) -> str:
    if isinstance(node, ast.Name):
        return bindings.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        owner = _resolved_ast_symbol(node.value, bindings)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _is_forbidden_fold_mutation_symbol(symbol: str) -> bool:
    lowered = symbol.lower()
    return (
        symbol in _FORBIDDEN_FOLD_MUTATION_SYMBOLS
        or any(
            symbol == module or symbol.startswith(f"{module}.")
            for module in _FORBIDDEN_FOLD_MUTATION_MODULES
        )
        or symbol.startswith("torch.ops.quantized.")
        or (
            symbol.startswith("torch._C.")
            and any(
                marker in lowered
                for marker in (
                    ".quantize",
                    ".quantized",
                    ".fake_quant",
                )
            )
        )
    )


def _fold_only_static_source_findings(
    source: str,
    *,
    source_label: str,
) -> tuple[Mapping[str, Any], ...]:
    """Find source-visible aliases that could evade process-local API patches."""

    try:
        tree = ast.parse(source, filename=source_label)
    except SyntaxError as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            f"Authenticated compiler source {source_label} is not parseable."
        ) from exc
    bindings: dict[str, str] = {}
    findings: set[tuple[str, str, int, int]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                qualified = alias.name if alias.asname else local_name
                bindings[local_name] = qualified
                if _is_forbidden_fold_mutation_symbol(alias.name):
                    findings.add(
                        (
                            "forbidden-import-alias",
                            alias.name,
                            node.lineno,
                            node.col_offset,
                        )
                    )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                qualified = f"{module}.{alias.name}" if module else alias.name
                bindings[alias.asname or alias.name] = qualified
                if _is_forbidden_fold_mutation_symbol(qualified):
                    findings.add(
                        (
                            "forbidden-import-alias",
                            qualified,
                            node.lineno,
                            node.col_offset,
                        )
                    )

    assignment_nodes = (
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    )
    for node in assignment_nodes:
        value = getattr(node, "value", None)
        qualified = (
            _resolved_ast_symbol(value, bindings) if isinstance(value, ast.AST) else ""
        )
        if not _is_forbidden_fold_mutation_symbol(qualified):
            continue
        if isinstance(node, ast.Assign):
            targets = node.targets
        else:
            targets = (node.target,)
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = qualified
                findings.add(
                    (
                        "forbidden-prebound-alias",
                        qualified,
                        node.lineno,
                        node.col_offset,
                    )
                )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            qualified = _resolved_ast_symbol(node.func, bindings)
            if _is_forbidden_fold_mutation_symbol(qualified):
                findings.add(
                    (
                        "forbidden-call",
                        qualified,
                        node.lineno,
                        node.col_offset,
                    )
                )
            if (
                qualified == "importlib.import_module"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and _is_forbidden_fold_mutation_symbol(node.args[0].value)
            ):
                findings.add(
                    (
                        "forbidden-dynamic-import",
                        node.args[0].value,
                        node.lineno,
                        node.col_offset,
                    )
                )
        elif isinstance(node, ast.Attribute) and isinstance(
            node.ctx,
            ast.Load,
        ):
            qualified = _resolved_ast_symbol(node, bindings)
            if _is_forbidden_fold_mutation_symbol(qualified):
                findings.add(
                    (
                        "forbidden-reference",
                        qualified,
                        node.lineno,
                        node.col_offset,
                    )
                )
    return tuple(
        MappingProxyType(
            {
                "kind": kind,
                "symbol": symbol,
                "line": line,
                "column": column,
            }
        )
        for kind, symbol, line, column in sorted(findings)
    )


def _audit_authenticated_compiler_sources(
    compiler_implementation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(compiler_implementation, Mapping):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "compiler_implementation must be a mapping."
        )
    try:
        implementation = json.loads(
            _canonical_json(dict(compiler_implementation)).decode("ascii")
        )
    except json.JSONDecodeError as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "compiler_implementation is not canonical JSON."
        ) from exc
    if not isinstance(implementation, dict):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "compiler_implementation must be a JSON object."
        )
    implementation_sha256 = _digest(
        implementation.pop("implementation_sha256", None),
        "compiler_implementation.implementation_sha256",
    )
    if _sha(implementation) != implementation_sha256:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler implementation source manifest is not self-sealed."
        )
    ordered_sources = implementation.get("ordered_source_files")
    if not isinstance(ordered_sources, list) or not ordered_sources:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler implementation has no authenticated source manifest."
        )

    source_specs: list[dict[str, Any]] = []
    seen_modules: set[str] = set()
    for raw_row in ordered_sources:
        if not isinstance(raw_row, Mapping):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiler source manifest row is not a mapping."
            )
        row = dict(raw_row)
        module_name = row.get("module")
        filename = row.get("filename")
        expected_sha256 = row.get("sha256")
        if (
            not isinstance(module_name, str)
            or not module_name
            or module_name in seen_modules
            or not isinstance(filename, str)
            or Path(filename).name != filename
        ):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Compiler source manifest identities are invalid or repeated."
            )
        expected_sha256 = _digest(
            expected_sha256,
            f"compiler source {module_name}",
        )
        module = sys.modules.get(module_name)
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Authenticated compiler module {module_name} is not loaded."
            )
        path = Path(raw_path).resolve()
        if path.name != filename or _file_sha256(path) != expected_sha256:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Authenticated compiler source {module_name} changed."
            )
        seen_modules.add(module_name)
        source_specs.append(
            {
                "role": "compiler-implementation",
                "module": module_name,
                "filename": filename,
                "sha256": expected_sha256,
                "path": path,
            }
        )

    runner_path = Path(__file__).resolve()
    source_specs.append(
        {
            "role": "compiler-runner",
            "module": __name__,
            "filename": runner_path.name,
            "sha256": _file_sha256(runner_path),
            "path": runner_path,
        }
    )
    audits = []
    all_findings = []
    for spec in source_specs:
        path = spec["path"]
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Could not read authenticated compiler source {path.name}."
            ) from exc
        findings = _fold_only_static_source_findings(
            source,
            source_label=f"{spec['module']}:{path.name}",
        )
        all_findings.extend(
            {
                "module": spec["module"],
                **dict(finding),
            }
            for finding in findings
        )
        audits.append(
            {
                "role": spec["role"],
                "module": spec["module"],
                "filename": spec["filename"],
                "sha256": spec["sha256"],
                "forbidden_finding_count": len(findings),
                "clean": not findings,
            }
        )
    body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-static-fold-mutation-source-audit-v1"),
        "compiler_implementation_sha256": implementation_sha256,
        "authenticated_compiler_source_files": [dict(row) for row in ordered_sources],
        "runner_source_sha256": source_specs[-1]["sha256"],
        "audited_source_count": len(audits),
        "source_audits": audits,
        "forbidden_findings": all_findings,
        "forbidden_finding_count": len(all_findings),
        "audit_scope": (
            "authenticated Python compiler and runner sources; imports, "
            "prebound assignments, calls, references, and constant dynamic "
            "imports of public PyTorch pruning/quantization and low-level "
            "quantized entrypoints"
        ),
        "clean": not all_findings,
    }
    report = {**body, "report_sha256": _sha(body)}
    if report["clean"] is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Authenticated compiler source contains a pruning or "
            "quantization alias that could evade the runtime firewall."
        )
    return MappingProxyType(report)


def _catalog_preparation_source_identities() -> Mapping[str, str]:
    sources = {
        "dynamic_evidence_catalog_artifact": _CATALOG_ARTIFACT_SOURCE,
        "hf_lfm2_cpcf_selective_evidence": _SELECTIVE_EVIDENCE_SOURCE,
    }
    if any(not path.is_file() for path in sources.values()):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation source files are missing."
        )
    return MappingProxyType(
        {name: _file_sha256(path) for name, path in sorted(sources.items())}
    )


def _catalog_preparation_static_audit() -> Mapping[str, Any]:
    """AST-audit preparation sources for training and mutation entrypoints."""

    findings = []
    source_audits = []
    forbidden_calls = {
        "backward",
        "grad",
        "step",
        "train",
        "generate",
        "quantize",
        "quantize_",
        "prune",
    }
    for module_name, path in (
        (
            "neural_forge.dynamic_evidence_catalog_artifact",
            _CATALOG_ARTIFACT_SOURCE,
        ),
        (
            "neural_forge.hf_lfm2_cpcf_selective_evidence",
            _SELECTIVE_EVIDENCE_SOURCE,
        ),
    ):
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                f"Could not AST-audit catalog source {path.name}."
            ) from exc
        mutation_findings = [
            dict(value)
            for value in _fold_only_static_source_findings(
                source,
                source_label=f"{module_name}:{path.name}",
            )
        ]
        operation_findings = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if called in forbidden_calls:
                operation_findings.append(
                    {
                        "kind": "forbidden-operation-call",
                        "symbol": called,
                        "line": node.lineno,
                        "column": node.col_offset,
                    }
                )
        module_findings = mutation_findings + operation_findings
        findings.extend({"module": module_name, **value} for value in module_findings)
        source_audits.append(
            {
                "module": module_name,
                "filename": path.name,
                "sha256": _file_sha256(path),
                "forbidden_finding_count": len(module_findings),
                "clean": not module_findings,
            }
        )
    body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-catalog-preparation-static-audit-v1"),
        "source_audits": source_audits,
        "forbidden_findings": findings,
        "forbidden_finding_count": len(findings),
        "audited_source_count": len(source_audits),
        "training_or_mutation_calls_permitted": False,
        "clean": not findings,
    }
    report = {**body, "report_sha256": _sha(body)}
    if report["clean"] is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation source contains a forbidden "
            "training, generation, pruning, or quantization call."
        )
    return MappingProxyType(report)


def _catalog_authentication_summary(
    preparation: Mapping[str, Any],
) -> Mapping[str, Any]:
    fields = (
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
    result = {
        name: _digest(preparation.get(name), f"catalog {name}") for name in fields
    }
    return MappingProxyType(result)


def selective_catalog_post_run_reverification_record(
    catalog_preparation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Build the one exact post-provider-close catalog verification record."""

    authentication = _catalog_authentication_summary(catalog_preparation)
    return MappingProxyType(
        {
            "preparation_receipt_sha256": authentication["preparation_receipt_sha256"],
            "binding_sha256": authentication["binding_sha256"],
            "artifact_manifest_sha256": authentication["artifact_manifest_sha256"],
            "artifact_catalog_sha256": authentication["artifact_catalog_sha256"],
            "reverified_after_provider_close": True,
            "unchanged_across_fold": True,
        }
    )


def authenticate_selective_catalog_post_run_reverification(
    value: Any,
    *,
    catalog_preparation: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Authenticate an exact runner-emitted catalog reverification record."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog post-run reverification is not a mapping."
        )
    expected = dict(
        selective_catalog_post_run_reverification_record(catalog_preparation)
    )
    observed = dict(value)
    if observed != expected:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog post-run reverification differs from the "
            "exact authenticated runner record."
        )
    return MappingProxyType(observed)


def _source_binding_sha256(
    *,
    selective_source_sha256: str,
    compiler_implementation_sha256: str,
    runner_source_sha256: str,
    fold_only_runtime_guard_identity_sha256: str,
    catalog_preparation: Mapping[str, Any],
) -> str:
    return _sha(
        {
            "selective_source_sha256": _digest(
                selective_source_sha256,
                "selective_source_sha256",
            ),
            "compiler_implementation_sha256": _digest(
                compiler_implementation_sha256,
                "compiler_implementation_sha256",
            ),
            "runner_source_sha256": _digest(
                runner_source_sha256,
                "runner_source_sha256",
            ),
            "fold_only_runtime_guard_identity_sha256": _digest(
                fold_only_runtime_guard_identity_sha256,
                "fold_only_runtime_guard_identity_sha256",
            ),
            "selective_catalog_artifact": dict(
                _catalog_authentication_summary(catalog_preparation)
            ),
        }
    )


def _read_selective_catalog_preparation(
    config: V5SelectiveBlankFoldConfig,
    *,
    expected_receipt_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Authenticate the preparation receipt and immutable artifact metadata."""

    if (
        config.selective_catalog_store_directory is None
        or config.selective_catalog_receipt_json is None
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Production folding requires a prepared selective catalog store "
            "and receipt."
        )
    receipt_path = _local_lexical_path(
        config.selective_catalog_receipt_json,
        "selective_catalog_receipt_json",
    )
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation receipt is missing or unsafe."
        )
    try:
        raw = receipt_path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation receipt is unreadable."
        ) from exc
    if not isinstance(value, Mapping) or raw != _canonical_json(value) + b"\n":
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation receipt is not canonical JSON."
        )
    receipt = dict(value)
    claimed = _digest(
        receipt.pop("preparation_receipt_sha256", None),
        "preparation_receipt_sha256",
    )
    required = (
        expected_receipt_sha256
        if expected_receipt_sha256 is not None
        else config.expected_selective_catalog_receipt_sha256
    )
    if (
        receipt.get("schema") != SELECTIVE_CATALOG_PREPARATION_SCHEMA
        or _sha(receipt) != claimed
        or (required is not None and claimed != required)
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation receipt schema, seal, or pin is invalid."
        )
    complete = {
        **receipt,
        "preparation_receipt_sha256": claimed,
    }
    summary = _catalog_authentication_summary(complete)
    if (
        receipt.get("status") != "prepared_authenticated_tensor_free_catalog"
        or receipt.get("catalog_store_directory")
        != str(
            _local_lexical_path(
                config.selective_catalog_store_directory,
                "selective_catalog_store_directory",
            )
        )
        or receipt.get("catalog_receipt_json") != str(receipt_path)
        or receipt.get("trace_cache_directory")
        != str(
            _local_lexical_path(
                config.trace_cache_directory,
                "trace_cache_directory",
            )
        )
        or receipt.get("warm_hit_required_for_production_fold") is not True
        or receipt.get("cold_rebuild_permitted_in_production_fold") is not False
        or receipt.get("tensor_payloads_resident") is not False
        or receipt.get("parent_model_loaded") is not False
        or receipt.get("blank_target_allocated") is not False
        or receipt.get("cuda_initialized") is not False
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation lifecycle or path binding changed."
        )
    current_sources = _catalog_preparation_source_identities()
    if (
        receipt.get("source_identities") != dict(current_sources)
        or summary["artifact_store_source_sha256"]
        != current_sources["dynamic_evidence_catalog_artifact"]
        or summary["selective_provider_source_sha256"]
        != current_sources["hf_lfm2_cpcf_selective_evidence"]
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation source identities changed."
        )
    from neural_forge.dynamic_evidence_catalog_artifact import (
        DynamicEvidenceCatalogArtifactStore,
        DynamicEvidenceCatalogBinding,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        lfm2_cpcf_selective_catalog_from_artifact_payload,
        lfm2_cpcf_selective_provider_implementation,
    )

    binding_value = receipt.get("binding")
    if not isinstance(binding_value, Mapping):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation has no typed binding."
        )
    binding = DynamicEvidenceCatalogBinding.from_dict(binding_value)
    provider_implementation = dict(lfm2_cpcf_selective_provider_implementation())
    if (
        binding.binding_sha256 != summary["binding_sha256"]
        or binding.codec_sha256 != summary["codec_sha256"]
        or binding.provider_sha256 != summary["provider_implementation_sha256"]
        or provider_implementation.get("implementation_sha256")
        != summary["provider_implementation_sha256"]
        or receipt.get("provider_implementation") != provider_implementation
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog binding, codec, or provider implementation changed."
        )
    store = DynamicEvidenceCatalogArtifactStore(
        config.selective_catalog_store_directory
    )
    artifact = store.load(binding)
    if (
        artifact is None
        or artifact.disposition != "loaded"
        or artifact.manifest_sha256 != summary["artifact_manifest_sha256"]
        or artifact.catalog_sha256 != summary["artifact_catalog_sha256"]
        or str(artifact.path) != receipt.get("artifact_path")
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Prepared selective catalog artifact is absent or changed."
        )
    catalog = lfm2_cpcf_selective_catalog_from_artifact_payload(artifact.payload)
    if (
        catalog.catalog_sha256 != summary["selective_catalog_sha256"]
        or catalog.source_sha256 != summary["selective_source_sha256"]
        or catalog.provider_sha256 != summary["selective_provider_sha256"]
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Prepared typed selective catalog identity changed."
        )
    artifact.verify()
    return MappingProxyType(complete)


def authenticate_prepared_selective_catalog(
    *,
    trace_cache_directory: Path,
    selective_catalog_store_directory: Path,
    selective_catalog_receipt_json: Path,
    expected_selective_catalog_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Public metadata-only authentication seam used by the supervisor."""

    @dataclass(frozen=True, slots=True)
    class _PreparationRequest:
        trace_cache_directory: Path
        selective_catalog_store_directory: Path
        selective_catalog_receipt_json: Path
        expected_selective_catalog_receipt_sha256: str

    request = _PreparationRequest(
        trace_cache_directory=trace_cache_directory,
        selective_catalog_store_directory=(selective_catalog_store_directory),
        selective_catalog_receipt_json=selective_catalog_receipt_json,
        expected_selective_catalog_receipt_sha256=(
            expected_selective_catalog_receipt_sha256
        ),
    )
    return _read_selective_catalog_preparation(request)  # type: ignore[arg-type]


def prepare_selective_catalog(
    config: V5SelectiveBlankFoldConfig,
) -> Mapping[str, Any]:
    """Build/authenticate one shared catalog before model or CUDA allocation."""

    if not isinstance(config, V5SelectiveBlankFoldConfig):
        raise TypeError("config must be V5SelectiveBlankFoldConfig.")
    if (
        not config.prepare_selective_catalog_only
        or config.selective_catalog_store_directory is None
        or config.selective_catalog_receipt_json is None
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "prepare_selective_catalog requires the explicit prepare-only "
            "configuration."
        )
    receipt_path = _local_lexical_path(
        config.selective_catalog_receipt_json,
        "selective_catalog_receipt_json",
    )
    if receipt_path.exists() or receipt_path.is_symlink():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation receipt already exists."
        )
    cache_directory = _base_runner()._local_directory(
        config.trace_cache_directory,
        "trace_cache_directory",
    )
    from neural_forge import dynamic_fold_executor
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        lfm2_cpcf_selective_provider_implementation,
        open_or_build_lfm2_cpcf_selective_evidence_provider,
    )

    try:
        import torch
    except ImportError as exc:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation requires local PyTorch."
        ) from exc
    if torch.cuda.is_initialized():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Catalog preparation must begin before CUDA initialization."
        )
    static_audit = _catalog_preparation_static_audit()
    guard = dynamic_fold_executor._FoldOnlyTorchGuard()
    cache = open_lfm2_circuit_trace_cache(
        cache_directory,
        verify=False,
    )
    provider = None
    with guard:
        try:
            provider = open_or_build_lfm2_cpcf_selective_evidence_provider(
                cache,
                artifact_store=(config.selective_catalog_store_directory),
            )
            artifact = provider.catalog_artifact
            if artifact is None:
                raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                    "Catalog preparation produced no immutable artifact."
                )
            catalog = provider.catalog
            artifact.verify()
        finally:
            if provider is not None:
                provider.close()
    if torch.cuda.is_initialized():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Catalog preparation initialized CUDA."
        )
    artifact.verify()
    provider_implementation = dict(lfm2_cpcf_selective_provider_implementation())
    source_identities = _catalog_preparation_source_identities()
    operation_audit = {
        "optimizer_constructions": int(guard.optimizer_constructions),
        "optimizer_steps": int(guard.optimizer_steps),
        "backward_calls": int(guard.backward_calls),
        "autograd_grad_calls": int(guard.autograd_grad_calls),
        "pruning_operations": int(guard.pruning_operations),
        "quantization_operations": int(guard.quantization_operations),
        "training_steps": 0,
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_model_loads": 0,
        "blank_target_allocations": 0,
        "cuda_initializations": 0,
    }
    if any(value != 0 for value in operation_audit.values()):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Catalog preparation attempted a forbidden model or training operation."
        )
    body = {
        "schema": SELECTIVE_CATALOG_PREPARATION_SCHEMA,
        "status": "prepared_authenticated_tensor_free_catalog",
        "catalog_store_directory": str(
            _local_lexical_path(
                config.selective_catalog_store_directory,
                "selective_catalog_store_directory",
            )
        ),
        "catalog_receipt_json": str(receipt_path),
        "trace_cache_directory": str(cache_directory),
        "cache_key": cache.cache_key,
        "artifact_path": str(artifact.path),
        "binding": artifact.binding.as_dict(),
        "binding_sha256": artifact.binding.binding_sha256,
        "artifact_manifest_sha256": artifact.manifest_sha256,
        "artifact_catalog_sha256": artifact.catalog_sha256,
        "selective_catalog_sha256": catalog.catalog_sha256,
        "selective_source_sha256": catalog.source_sha256,
        "selective_provider_sha256": catalog.provider_sha256,
        "codec_sha256": artifact.binding.codec_sha256,
        "provider_implementation": provider_implementation,
        "provider_implementation_sha256": provider_implementation[
            "implementation_sha256"
        ],
        "source_identities": dict(source_identities),
        "artifact_store_source_sha256": source_identities[
            "dynamic_evidence_catalog_artifact"
        ],
        "selective_provider_source_sha256": source_identities[
            "hf_lfm2_cpcf_selective_evidence"
        ],
        "static_source_audit": dict(static_audit),
        "operation_audit": operation_audit,
        "tensor_payloads_resident": False,
        "parent_model_loaded": False,
        "blank_target_allocated": False,
        "cuda_initialized": False,
        "warm_hit_required_for_production_fold": True,
        "cold_rebuild_permitted_in_production_fold": False,
        "post_prepare_artifact_reverified": True,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
    }
    receipt = {
        **body,
        "preparation_receipt_sha256": _sha(body),
    }
    _write_diagnostic(receipt_path, receipt)
    return _read_selective_catalog_preparation(
        config,
        expected_receipt_sha256=receipt["preparation_receipt_sha256"],
    )


def _fold_only_guard_implementation_identity(module: Any) -> Mapping[str, Any]:
    raw_path = getattr(module, "__file__", None)
    guard = getattr(module, "_FoldOnlyTorchGuard", None)
    if not isinstance(raw_path, str) or not isinstance(guard, type):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The runtime fold-only guard has no exact source identity."
        )
    path = Path(raw_path).resolve()
    body = {
        "schema": "neural-forge-fold-only-runtime-guard-identity-v1",
        "module": str(getattr(module, "__name__", "")),
        "class": f"{guard.__module__}.{guard.__qualname__}",
        "source_filename": path.name,
        "source_sha256": _file_sha256(path),
        "training_api_firewall": True,
        "common_pruning_api_firewall": True,
        "common_quantization_api_firewall": True,
        "prebound_aliases_runtime_observed": False,
        "authenticated_source_alias_audit_required": True,
        "ordinary_bfloat16_and_float32_casts_blocked": False,
    }
    return MappingProxyType({**body, "identity_sha256": _sha(body)})


def _authenticate_live_consumer_context(
    context: V5CompiledCandidateConsumerContext,
) -> None:
    target = _live_model_authentication(context.target_model)
    parent = _live_model_authentication(context.forbidden_parent_model)
    if (
        target["state_sha256"] != context.target_state_sha256
        or target["runtime_sha256"] != context.target_runtime_sha256
        or target["parameter_inventory"] != dict(context.compiled_parameter_inventory)
        or parent["state_sha256"] != context.parent_state_sha256
        or parent["runtime_sha256"] != context.parent_runtime_sha256
        or parent["parameter_inventory"] != dict(context.parent_parameter_inventory)
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Live parent or target state/topology differs from the sealed "
            "compiled-candidate context."
        )


def _invoke_compiled_candidate_consumer(
    context: V5CompiledCandidateConsumerContext,
    consumer: Callable[
        [V5CompiledCandidateConsumerContext],
        V5CompiledCandidateConsumerResult,
    ],
) -> V5CompiledCandidateConsumerResult:
    """Invoke exactly one capture consumer under live parent/target guards."""

    if type(context) is not V5CompiledCandidateConsumerContext:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer requires an exact typed context."
        )
    if not callable(consumer):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer must be callable."
        )
    from neural_forge.dynamic_behavior_execution import (
        observe_model_execution,
    )
    from neural_forge import dynamic_fold_executor

    _authenticate_live_consumer_context(context)
    consumer_guard = dynamic_fold_executor._FoldOnlyTorchGuard()
    observed_parent: Mapping[str, int] = {}
    direct_parent_forward_calls = 0
    result: Any = None
    consumer_error: BaseException | None = None
    parent = context.forbidden_parent_model
    parent_forward = getattr(parent, "forward", None)
    if not callable(parent_forward):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Forbidden parent has no observable forward method."
        )
    parent_attributes = getattr(parent, "__dict__", {})
    forward_had_instance_value = (
        isinstance(parent_attributes, Mapping) and "forward" in parent_attributes
    )
    forward_instance_value = (
        parent_attributes.get("forward") if forward_had_instance_value else None
    )

    def observed_direct_parent_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal direct_parent_forward_calls
        direct_parent_forward_calls += 1
        return parent_forward(*args, **kwargs)

    object.__setattr__(parent, "forward", observed_direct_parent_forward)
    try:
        with consumer_guard:
            with observe_model_execution(parent) as observed_parent:
                result = consumer(context)
    except BaseException as exc:
        consumer_error = exc
    finally:
        if forward_had_instance_value:
            object.__setattr__(parent, "forward", forward_instance_value)
        else:
            object.__delattr__(parent, "forward")

    authentication_error: BaseException | None = None
    try:
        _authenticate_live_consumer_context(context)
    except BaseException as exc:
        authentication_error = exc
    parent_calls = (
        sum(
            int(observed_parent.get(name, 0))
            for name in (
                "module_forward_calls",
                "root_forward_calls",
                "inference_entrypoint_calls",
            )
        )
        + direct_parent_forward_calls
    )
    consumer_guard_body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-consumer-runtime-fold-only-guard-v1"),
        "guard_implementation": dict(
            _fold_only_guard_implementation_identity(dynamic_fold_executor)
        ),
        "compiler_runtime_guard_sha256": context.zero_operation_audit[
            "compiler_runtime_guard"
        ]["report_sha256"],
        "optimizer_constructions": int(consumer_guard.optimizer_constructions),
        "optimizer_steps": int(consumer_guard.optimizer_steps),
        "backward_calls": int(consumer_guard.backward_calls),
        "autograd_grad_calls": int(consumer_guard.autograd_grad_calls),
        "pruning_operations": int(consumer_guard.pruning_operations),
        "quantization_operations": int(consumer_guard.quantization_operations),
        "parent_module_forward_calls": int(
            observed_parent.get("module_forward_calls", 0)
        ),
        "parent_root_forward_calls": int(observed_parent.get("root_forward_calls", 0)),
        "parent_inference_entrypoint_calls": int(
            observed_parent.get("inference_entrypoint_calls", 0)
        ),
        "direct_parent_forward_calls": int(direct_parent_forward_calls),
        "training_steps": 0,
        "training_steps_runtime_observed_directly": False,
        "training_steps_zero_derivation": (
            "zero optimizer/backward/autograd API attempts under no-grad"
        ),
        "pruning_observation_scope": (
            "process-local public PyTorch pruning module functions and "
            "exported BasePruningMethod class apply/prune/apply_mask/remove "
            "bindings installed during the guard"
        ),
        "quantization_observation_scope": (
            "process-local common public PyTorch/torchao quantization and "
            "low-level torch quantize/fake-quant bindings installed during "
            "the guard"
        ),
        "prebound_alias_scope": (
            "compiler aliases rejected by sealed authenticated-source AST "
            "audit; aliases captured before this guard inside independently "
            "source-bound capture callables are outside runtime interception, "
            "while retained candidate state is independently reauthenticated"
        ),
        "ordinary_bfloat16_and_float32_casts_blocked": False,
    }
    consumer_guard_body["clean"] = all(
        consumer_guard_body[name] == 0
        for name in (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
            "pruning_operations",
            "quantization_operations",
            "parent_module_forward_calls",
            "parent_root_forward_calls",
            "parent_inference_entrypoint_calls",
            "direct_parent_forward_calls",
        )
    )
    if parent_calls != 0:
        error = Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer executed forbidden live parent inference."
        )
        if consumer_error is not None:
            error.add_note(f"Consumer also failed: {consumer_error}")
        raise error from consumer_error
    if authentication_error is not None:
        error = Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer mutated parent/target state, topology, "
            "runtime mode, gradients, or hooks."
        )
        if consumer_error is not None:
            error.add_note(f"Consumer also failed: {consumer_error}")
        raise error from authentication_error
    if consumer_error is not None:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer failed before returning a capture receipt."
        ) from consumer_error
    if type(result) is not V5CompiledCandidateConsumerResult:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer returned the wrong exact result type."
        )
    expected = {
        "compiler_receipt_sha256": context.compiler_receipt_sha256,
        "target_plan_fingerprint": context.target_plan_fingerprint,
        "target_state_sha256": context.target_state_sha256,
        "parent_sha256": context.parent_sha256,
        "cache_sha256": context.cache_sha256,
        "bridge_sha256": context.bridge_sha256,
        "corpus_sha256": context.corpus_sha256,
        "source_sha256": context.source_sha256,
        "compiler_sha256": context.compiler_sha256,
        "runner_source_sha256": context.runner_source_sha256,
        "source_binding_sha256": context.source_binding_sha256,
        "parent_tokenizer_semantics_sha256": (
            context.parent_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            context.target_tokenizer_semantics_sha256
        ),
    }
    if any(getattr(result, name) != value for name, value in expected.items()):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer result differs from the exact compiler, "
            "plan, state, or source bindings."
        )
    if dict(result.compile_diagnostic) != dict(context.compile_diagnostic):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer returned a different sealed compile diagnostic."
        )
    if result.consumer_runtime_guard_receipt is not None:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Candidate consumer cannot predeclare its outer runtime guard."
        )
    consumer_guard_body["capture_verification_sha256"] = result.verification_receipt[
        "verification_sha256"
    ]
    consumer_guard_report = {
        **consumer_guard_body,
        "report_sha256": _sha(consumer_guard_body),
    }
    return replace(
        result,
        consumer_runtime_guard_receipt=consumer_guard_report,
    )


def _run(
    config: V5SelectiveBlankFoldConfig,
    *,
    candidate_consumer: Callable[
        [V5CompiledCandidateConsumerContext],
        V5CompiledCandidateConsumerResult,
    ]
    | None = None,
) -> dict[str, Any] | V5CompiledCandidateConsumerResult:
    base = _base_runner()
    parent_directory = base._local_directory(
        config.parent_model_directory,
        "parent_model_directory",
    )
    donor_tokenizer_directory = base._local_directory(
        config.donor_tokenizer_directory,
        "donor_tokenizer_directory",
    )
    target_tokenizer_directory = base._local_directory(
        config.target_tokenizer_directory,
        "target_tokenizer_directory",
    )
    bridge_directory = base._local_directory(
        config.bridge_directory,
        "bridge_directory",
    )
    cache_directory = base._local_directory(
        config.trace_cache_directory,
        "trace_cache_directory",
    )
    trajectory_path = base._local_file(
        config.trajectory_corpus,
        "trajectory_corpus",
    )
    if config.expected_selective_catalog_receipt_sha256 is None:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Production folding requires an exact selective catalog "
            "preparation receipt digest."
        )
    catalog_preparation = _read_selective_catalog_preparation(config)
    catalog_authentication = _catalog_authentication_summary(catalog_preparation)
    torch = _configure_torch()
    cuda_allocator_policy = _enforce_cuda_allocator_caps(
        torch,
        maximum_cuda_bytes_per_device=(config.maximum_cuda_bytes_per_device),
    )

    from experiments.run_lfm2_residual_attention_fold import (
        direct_local_io_audit,
        load_local_lfm2_model,
        load_local_tokenizer,
    )
    from neural_forge.hf_lfm2_circuit_trace_cache import (
        fingerprint_lfm2_parent,
        open_lfm2_circuit_trace_cache,
    )
    from neural_forge.hf_lfm2_cpcf_blank_target import (
        plan_and_synthesize_lfm2_cpcf_blank,
    )
    from neural_forge.hf_lfm2_cpcf_model_fold import (
        Lfm2CpcfModelFoldPolicy,
        audit_lfm2_cpcf_parameter_compaction,
        independent_parameter_inventory,
    )
    from neural_forge.hf_lfm2_cpcf_selective_evidence import (
        open_lfm2_cpcf_selective_evidence_provider_from_catalog,
    )
    from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
        Lfm2CpcfShortConvV4SolverPolicy,
    )
    from neural_forge import dynamic_fold_executor
    from neural_forge import hf_lfm2_cpcf_v5_model_compiler as v5_compiler
    from neural_forge.dynamic_behavior_execution import (
        observe_model_execution,
    )
    from neural_forge.hf_tokenizer_bridge import (
        fingerprint_hf_tokenizer_semantics,
        load_strict_tokenizer_bridge_artifact,
    )

    authenticated_compiler_implementation = dict(
        v5_compiler.lfm2_cpcf_v5_compiler_implementation()
    )
    compiler_static_source_audit = _audit_authenticated_compiler_sources(
        authenticated_compiler_implementation
    )
    memory_samples = [
        _memory_sample(
            v5_compiler,
            label="deterministic_torch_configured",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    ]
    gpu = _build_authenticated_gpu_replay_evaluator(torch)
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="authenticated_two_gpu_replay_ready",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )
    dtype = base._torch_dtype(torch, config.dtype)
    device = torch.device(config.model_device)
    if not torch.cuda.is_available():
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "CUDA is required for authenticated candidate replay."
        )

    donor_tokenizer = load_local_tokenizer(donor_tokenizer_directory)
    target_tokenizer = load_local_tokenizer(target_tokenizer_directory)
    bridge = load_strict_tokenizer_bridge_artifact(
        bridge_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    bridge.verify_files()
    donor_fingerprint = fingerprint_hf_tokenizer_semantics(donor_tokenizer)
    target_fingerprint = fingerprint_hf_tokenizer_semantics(target_tokenizer)
    if (
        donor_fingerprint != bridge.donor_tokenizer_semantics_sha256
        or target_fingerprint != bridge.target_tokenizer_semantics_sha256
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Live local tokenizers differ from the sealed token bridge."
        )
    rows = base._read_trajectory_rows(trajectory_path)
    cache = open_lfm2_circuit_trace_cache(cache_directory, verify=False)
    trace_adapter = base._trace_adapter(
        rows=rows,
        target_tokenizer=target_tokenizer,
        bridge=bridge,
        cache_manifest=cache.manifest,
    )
    parent = load_local_lfm2_model(
        parent_directory,
        device=device,
        dtype=dtype,
    )
    parent.requires_grad_(False)
    parent.eval()
    parent_fingerprint_before = fingerprint_lfm2_parent(parent)
    if parent_fingerprint_before.identity_sha256 != cache.parent_fingerprint:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The explicit local parent differs from the sealed trace cache."
        )
    parent_inventory = independent_parameter_inventory(parent)
    blank = plan_and_synthesize_lfm2_cpcf_blank(
        parent,
        parent_name=config.parent_name,
        parent_tokenizer_fingerprint=donor_fingerprint,
        evidence_fingerprint=cache.cache_key,
        child_name=config.child_name,
        target_vocabulary_size=(bridge.transport_profile.target_row_capacity),
        child_tokenizer_fingerprint=target_fingerprint,
        tokenizer_bridge_artifact=bridge,
        minimum_hidden_size=config.minimum_hidden_size,
        maximum_candidates=config.maximum_candidates,
        selected_candidate_rank=config.selected_candidate_rank,
        blank_seed=config.blank_seed,
        device=device,
        dtype=dtype,
        **config.compression.as_kwargs(),
    )
    target = blank.model
    target.requires_grad_(False)
    target.eval()
    blank_inventory = independent_parameter_inventory(target)
    precompile_dense_state = _dense_retained_state_facts(
        target,
        label="precompile exact dense blank",
    )
    if (
        config.expected_target_plan_fingerprint is not None
        and blank.plan.fingerprint != config.expected_target_plan_fingerprint
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The runtime blank target plan differs from the catalog-planned "
            "frontier lane."
        )
    if config.compression.target_unique_parameters is not None and (
        blank.unique_parameters != config.compression.target_unique_parameters
        or int(blank_inventory["unique_parameters"])
        != config.compression.target_unique_parameters
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The synthesized dense blank misses the exact parameter target."
        )
    structural_compression_floor = _request_derived_structural_compression_floor(
        config.compression
    )
    structural_compression = audit_lfm2_cpcf_parameter_compaction(
        parent,
        target,
        planned_target_unique_parameters=blank.unique_parameters,
        promotion_compression_ratio=structural_compression_floor,
    )
    if structural_compression.get("promotion_ratio_met") is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The synthesized blank misses the exact request-derived structural "
            "compression floor."
        )
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="local_parent_and_exact_dense_blank_ready",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )

    provider = None
    try:
        provider = open_lfm2_cpcf_selective_evidence_provider_from_catalog(
            cache,
            artifact_store=config.selective_catalog_store_directory,
            expected_binding_sha256=(catalog_authentication["binding_sha256"]),
            expected_manifest_sha256=(
                catalog_authentication["artifact_manifest_sha256"]
            ),
            expected_catalog_sha256=(catalog_authentication["artifact_catalog_sha256"]),
        )
        catalog = provider.catalog
        references = catalog.dynamic_catalog.references
        logical_budget, required_logical_budget = _resolved_logical_resident_bytes(
            references=references,
            final_norm_logical_bytes=(catalog.final_norm_logical_bytes),
            requested_gib=config.maximum_logical_resident_gib,
        )
        full_phase_tree_bytes = int(catalog.final_norm_logical_bytes) + sum(
            int(reference.logical_bytes) for reference in references
        )
        memory_samples.append(
            _memory_sample(
                v5_compiler,
                label="selective_catalog_authenticated",
                maximum_process_rss_bytes=(config.maximum_process_rss_bytes),
            )
        )
        compiler_baseline = int(v5_compiler._process_rss_bytes())
        estimated_effective_reduction, estimated_maximum_compiler_increase = (
            _derive_effective_rss_reduction_fraction(
                full_phase_tree_logical_bytes=full_phase_tree_bytes,
                compiler_baseline_rss_bytes=compiler_baseline,
                maximum_process_rss_bytes=(config.maximum_process_rss_bytes),
                requested_minimum_reduction_fraction=(
                    config.minimum_physical_rss_reduction_fraction
                ),
            )
        )
        short_conv_policy = Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=config.short_conv_v4_ridge_alphas,
            variable_projection_passes=(
                config.short_conv_v4_variable_projection_passes
            ),
            writer_trust_radius=(config.short_conv_v4_writer_trust_radius),
            fir_trust_radius=config.short_conv_v4_fir_trust_radius,
            bcx_trust_radius=config.short_conv_v4_bcx_trust_radius,
            correction_rms_floor=(config.short_conv_v4_correction_rms_floor),
            context_objective_weight=(config.short_conv_v4_context_objective_weight),
            target_dtype=config.dtype,
        )
        model_policy = Lfm2CpcfModelFoldPolicy(
            ridge=config.ridge,
            state_reconstruction_weight=(config.boundary_state_reconstruction_weight),
            minimum_predictive_energy_fraction=(
                config.boundary_predictive_energy_floor
            ),
            minimum_validation_block_retention=(
                config.boundary_validation_retention_floor
            ),
            enforce_validation_block_floor=True,
            short_conv_v4_solver_policy=short_conv_policy,
            promotion_compression_ratio=structural_compression_floor,
            require_promotion_ratio=True,
        )
        runtime_policy = v5_compiler.Lfm2CpcfV5RuntimePolicy(
            model_fold_policy=model_policy,
            maximum_resident_evidence_bytes=logical_budget,
            full_phase_tree_logical_bytes=full_phase_tree_bytes,
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
            minimum_physical_rss_reduction_fraction=(
                config.minimum_physical_rss_reduction_fraction
            ),
            rss_sample_interval_seconds=0.05,
            require_short_conv_v5_endpoint_first_selector=True,
            require_short_conv_v5_authenticated_whole_macrocell_selector=True,
            require_short_conv_v5_real_numerical_frontier_adapter=True,
        )
        source_tied_table = base._mapped_source_tied_table(parent, bridge)
        selector = v5_compiler.Lfm2CpcfV5EndpointFirstCandidateSelector()
        numerical_adapter = _build_preregistered_real_v5_numerical_adapter()
        resume_runtime = _build_resume_runtime(
            config,
            catalog_preparation=catalog_preparation,
        )
        compiler_guard = dynamic_fold_executor._FoldOnlyTorchGuard()
        compiler_parent_execution: Mapping[str, int] = {}
        compiler_target_execution: Mapping[str, int] = {}
        with compiler_guard:
            with ExitStack() as compiler_observers:
                compiler_parent_execution = compiler_observers.enter_context(
                    observe_model_execution(parent)
                )
                compiler_target_execution = compiler_observers.enter_context(
                    observe_model_execution(target)
                )
                compiled = _compile_with_authenticated_resume(
                    v5_compiler.compile_lfm2_cpcf_v5_selective_model_fold,
                    parent,
                    target,
                    resume_runtime=resume_runtime,
                    compile_kwargs={
                        "trace_adapter": trace_adapter,
                        "evidence_provider": provider,
                        "source_tied_table": source_tied_table,
                        "blank_target_report": blank.as_dict(),
                        "planned_target_unique_parameters": (blank.unique_parameters),
                        "runtime_policy": runtime_policy,
                        "candidate_replay_evaluator": gpu.evaluator,
                        "candidate_selector": selector,
                        "numerical_frontier_adapter": numerical_adapter,
                    },
                )
    finally:
        if provider is not None:
            provider.close()
    post_run_catalog_preparation = _read_selective_catalog_preparation(config)
    if dict(post_run_catalog_preparation) != dict(catalog_preparation):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Selective catalog preparation changed across the fold."
        )
    guard_identity = _fold_only_guard_implementation_identity(dynamic_fold_executor)
    compiler_runtime_guard = {
        "schema": "neural-forge-lfm2-cpcf-v5-compiler-runtime-guard-v1",
        "guard_implementation": dict(guard_identity),
        "static_pruning_quantization_source_audit": dict(compiler_static_source_audit),
        "optimizer_constructions": int(compiler_guard.optimizer_constructions),
        "optimizer_steps": int(compiler_guard.optimizer_steps),
        "backward_calls": int(compiler_guard.backward_calls),
        "autograd_grad_calls": int(compiler_guard.autograd_grad_calls),
        "pruning_operations": int(compiler_guard.pruning_operations),
        "quantization_operations": int(compiler_guard.quantization_operations),
        "parent_module_forward_calls": int(
            compiler_parent_execution.get("module_forward_calls", 0)
        ),
        "parent_root_forward_calls": int(
            compiler_parent_execution.get("root_forward_calls", 0)
        ),
        "parent_inference_entrypoint_calls": int(
            compiler_parent_execution.get(
                "inference_entrypoint_calls",
                0,
            )
        ),
        "target_module_forward_calls": int(
            compiler_target_execution.get("module_forward_calls", 0)
        ),
        "target_root_forward_calls": int(
            compiler_target_execution.get("root_forward_calls", 0)
        ),
        "target_inference_entrypoint_calls": int(
            compiler_target_execution.get(
                "inference_entrypoint_calls",
                0,
            )
        ),
        "training_steps": 0,
        "training_steps_runtime_observed_directly": False,
        "training_steps_zero_derivation": (
            "zero optimizer/backward/autograd API attempts under no-grad"
        ),
        "pruning_observation_scope": (
            "process-local public PyTorch pruning module functions and "
            "exported BasePruningMethod class apply/prune/apply_mask/remove "
            "bindings installed during the guard"
        ),
        "quantization_observation_scope": (
            "process-local common public PyTorch/torchao quantization and "
            "low-level torch quantize/fake-quant bindings installed during "
            "the guard"
        ),
        "prebound_alias_static_audit_sha256": (
            compiler_static_source_audit["report_sha256"]
        ),
        "ordinary_bfloat16_and_float32_casts_blocked": False,
    }
    compiler_runtime_guard["clean"] = all(
        compiler_runtime_guard[name] == 0
        for name in (
            "optimizer_constructions",
            "optimizer_steps",
            "backward_calls",
            "autograd_grad_calls",
            "pruning_operations",
            "quantization_operations",
            "parent_module_forward_calls",
            "parent_root_forward_calls",
            "parent_inference_entrypoint_calls",
            "target_root_forward_calls",
            "target_inference_entrypoint_calls",
        )
    )
    if compiler_runtime_guard["clean"] is not True:
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The compiler attempted a forbidden training, pruning, "
            "quantization, parent-execution, or target-root API."
        )
    compiler_runtime_guard = {
        **compiler_runtime_guard,
        "report_sha256": _sha(compiler_runtime_guard),
    }
    memory_samples.append(
        _memory_sample(
            v5_compiler,
            label="v5_compiler_complete_provider_closed",
            maximum_process_rss_bytes=config.maximum_process_rss_bytes,
        )
    )
    parent_fingerprint_after = fingerprint_lfm2_parent(parent)
    if (
        parent_fingerprint_after.identity_sha256
        != parent_fingerprint_before.identity_sha256
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Parent state changed during the selective fold."
        )
    compiled_inventory = independent_parameter_inventory(target)
    postcompile_dense_state = _dense_retained_state_facts(
        target,
        label="postcompile folded dense target",
    )
    if int(compiled_inventory["unique_parameters"]) != int(
        blank_inventory["unique_parameters"]
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Folding changed the dense child parameter count."
        )
    compiled_report = compiled.as_dict()
    layer_reports = [result.as_dict() for result in compiled.layer_results]
    gpu_execution = _gpu_execution_audit(layer_reports)
    resume_progress = resume_runtime.progress.as_dict()
    checkpointing_enabled = resume_runtime.coordinator is not None
    resume_used = resume_runtime.state is not None
    if (
        compiled_report.get(
            "short_conv_v5_authenticated_whole_macrocell_selection_integrated"
        )
        is not True
        or compiled_report.get("short_conv_v5_macrocell_selector_final_authority")
        is not True
        or compiled_report.get("short_conv_v5_endpoint_first_selector_final_authority")
        is not False
        or compiled_report.get("short_conv_v5_endpoint_prefilter_control_integrated")
        is not True
        or compiled_report.get("macrocell_candidate_selector_layer_indices")
        != compiled_report.get("short_conv_layer_indices")
        or compiled_report.get(
            "short_conv_v5_numerical_candidate_generation_integrated"
        )
        is not True
        or compiled_report.get("short_conv_v5_numerical_policy_integrated") is not True
        or compiled_report.get("short_conv_v5_real_numerical_adapter_required")
        is not True
        or compiled_report.get("candidate_replay_backend_injected") is not True
        or compiled_report.get("resume_supported") is not True
        or compiled_report.get("resume_fail_closed") is not True
        or compiled_report.get("resume_used") is not resume_used
        or compiled_report.get("resume_publication_enabled")
        is not checkpointing_enabled
        or compiled_report.get("resume_parent_or_cache_tensor_sidecars") != 0
        or compiled_report.get("resume_admitted_prefixes")
        != resume_progress["admitted_prefixes"]
        or resume_progress["outstanding_pending_prefix_count"] != 0
        or resume_progress["rejected_prefix_count"] != 0
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The completed fold lacks the required real v5 numerical "
            "provenance, authenticated whole-macrocell selection authority, "
            "endpoint-prefilter control, replay injection, or authenticated "
            "typed resume receipt."
        )
    compiler_operation_audit = compiled_report.get("operation_audit")
    if not isinstance(compiler_operation_audit, Mapping) or any(
        isinstance(compiler_operation_audit.get(name), bool)
        or compiler_operation_audit.get(name) != 0
        for name in (
            "parent_model_forward_calls",
            "target_model_forward_calls",
            "generation_calls",
            "parent_cache_generation_calls",
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
            "fallback_paths",
        )
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The compiler receipt does not prove zero-operation folding."
        )
    observed_compiler_crosscheck = {
        "parent_model_forward_calls": compiler_runtime_guard[
            "parent_root_forward_calls"
        ],
        "target_model_forward_calls": compiler_runtime_guard[
            "target_root_forward_calls"
        ],
        "generation_calls": compiler_runtime_guard["parent_inference_entrypoint_calls"],
        "training_steps": compiler_runtime_guard["training_steps"],
        "backward_calls": compiler_runtime_guard["backward_calls"],
        "optimizer_steps": compiler_runtime_guard["optimizer_steps"],
        "parameters_pruned": compiler_runtime_guard["pruning_operations"],
        "parameters_quantized": compiler_runtime_guard["quantization_operations"],
    }
    if any(
        compiler_operation_audit.get(name) != value
        for name, value in observed_compiler_crosscheck.items()
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler-emitted operation claims differ from runtime guard observations."
        )
    compiler_implementation = compiled_report.get("compiler_implementation")
    if (
        not isinstance(compiler_implementation, Mapping)
        or dict(compiler_implementation) != authenticated_compiler_implementation
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The compiler receipt lacks or changed its preauthenticated "
            "implementation identity."
        )
    compiler_receipt_sha256 = _digest(
        compiled_report.get("result_sha256"),
        "compiled_model_fold.result_sha256",
    )
    compiler_sha256 = _digest(
        compiler_implementation.get("implementation_sha256"),
        "compiler_implementation.implementation_sha256",
    )
    source_sha256 = _digest(
        compiled_report.get("selective_source_sha256"),
        "compiled_model_fold.selective_source_sha256",
    )
    if (
        compiled_report.get("selective_catalog_sha256")
        != catalog_authentication["selective_catalog_sha256"]
        or source_sha256 != catalog_authentication["selective_source_sha256"]
        or compiled_report.get("selective_provider_sha256")
        != catalog_authentication["selective_provider_sha256"]
    ):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Compiler selective identities differ from the prepared catalog."
        )
    runner_source_sha256 = _file_sha256(Path(__file__).resolve())
    source_binding_sha256 = _source_binding_sha256(
        selective_source_sha256=source_sha256,
        compiler_implementation_sha256=compiler_sha256,
        runner_source_sha256=runner_source_sha256,
        fold_only_runtime_guard_identity_sha256=guard_identity["identity_sha256"],
        catalog_preparation=catalog_preparation,
    )
    corpus_sha256 = _file_sha256(trajectory_path)
    target_consumer_auth = _live_model_authentication(target)
    parent_consumer_auth = _live_model_authentication(parent)
    if target_consumer_auth["parameter_inventory"] != dict(
        compiled_inventory
    ) or parent_consumer_auth["parameter_inventory"] != dict(parent_inventory):
        raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "Post-compile consumer authentication changed parameter inventory."
        )
    zero_operation_audit = {
        "schema": COMPILED_CANDIDATE_ZERO_AUDIT_SCHEMA,
        "optimizer_constructions": compiler_runtime_guard["optimizer_constructions"],
        "optimizer_steps": compiler_runtime_guard["optimizer_steps"],
        "backward_calls": compiler_runtime_guard["backward_calls"],
        "autograd_grad_calls": compiler_runtime_guard["autograd_grad_calls"],
        "training_steps": compiler_runtime_guard["training_steps"],
        "pruning_operations": compiler_runtime_guard["pruning_operations"],
        "quantization_operations": compiler_runtime_guard["quantization_operations"],
        "parent_forward_calls": compiler_runtime_guard["parent_module_forward_calls"],
        "parent_inference_calls": compiler_runtime_guard[
            "parent_inference_entrypoint_calls"
        ],
        "target_forward_calls": compiler_runtime_guard["target_root_forward_calls"],
        "cache_generation_calls": compiler_operation_audit[
            "parent_cache_generation_calls"
        ],
        "compiler_operation_audit": dict(compiler_operation_audit),
        "compiler_runtime_guard": dict(compiler_runtime_guard),
        "training_steps_evidence_scope": (
            "derived-from-runtime-zero-optimizer-backward-autograd-attempts"
        ),
        "cache_generation_calls_evidence_scope": (
            "compiler-receipt-and-sealed-cache-only-source-contract"
        ),
        "pruning_operations_runtime_observed": True,
        "pruning_observation_scope": compiler_runtime_guard[
            "pruning_observation_scope"
        ],
        "quantization_operations_runtime_observed": True,
        "quantization_observation_scope": compiler_runtime_guard[
            "quantization_observation_scope"
        ],
        "retained_state_facts": {
            "before_compile": dict(precompile_dense_state),
            "after_compile": dict(postcompile_dense_state),
        },
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "live_parent_calls_used": False,
        "clean": True,
    }
    body = {
        "schema": RUN_SCHEMA,
        "status": "compiled_in_memory_diagnostic_not_published",
        "inputs": {
            "parent_model_directory": str(parent_directory),
            "donor_tokenizer_directory": str(donor_tokenizer_directory),
            "target_tokenizer_directory": str(target_tokenizer_directory),
            "bridge_directory": str(bridge_directory),
            "trace_cache_directory": str(cache_directory),
            "trajectory_corpus": str(trajectory_path),
            "trajectory_corpus_sha256": corpus_sha256,
            "cache_key": cache.cache_key,
            "parent_fingerprint": cache.parent_fingerprint,
            "bridge_artifact_sha256": bridge.artifact_sha256,
            "donor_tokenizer_semantics_sha256": donor_fingerprint,
            "target_tokenizer_semantics_sha256": target_fingerprint,
            "trace_adapter_sha256": trace_adapter.evidence_sha256,
            "selective_catalog_preparation_receipt_sha256": (
                catalog_authentication["preparation_receipt_sha256"]
            ),
            "selective_catalog_binding_sha256": (
                catalog_authentication["binding_sha256"]
            ),
            "selective_catalog_artifact_manifest_sha256": (
                catalog_authentication["artifact_manifest_sha256"]
            ),
            "selective_catalog_artifact_sha256": (
                catalog_authentication["artifact_catalog_sha256"]
            ),
            "selective_catalog_codec_sha256": (catalog_authentication["codec_sha256"]),
            "selective_catalog_provider_implementation_sha256": (
                catalog_authentication["provider_implementation_sha256"]
            ),
        },
        "request": config.as_dict(),
        "local_io_audit": direct_local_io_audit(parent_directory),
        "blank_target": blank.as_dict(),
        "parent_parameter_inventory": dict(parent_inventory),
        "blank_parameter_inventory": dict(blank_inventory),
        "compiled_parameter_inventory": dict(compiled_inventory),
        "structural_compression": dict(structural_compression),
        "request_derived_structural_compression_floor": (structural_compression_floor),
        "compiled_model_fold": compiled_report,
        "compiler_runtime_fold_only_guard": dict(compiler_runtime_guard),
        "retained_dense_state_facts": {
            "before_compile": dict(precompile_dense_state),
            "after_compile": dict(postcompile_dense_state),
        },
        "resume_progress": resume_progress,
        "layer_results": layer_reports,
        "boundary_bundle": compiled.boundary_bundle.as_dict(),
        "interface_install": dict(compiled.interface_install_report),
        "neutral_norm_install": dict(compiled.neutral_norm_install_report),
        "final_norm": dict(compiled.final_norm_report),
        "selective_catalog": catalog.as_dict(),
        "selective_catalog_artifact": dict(catalog_preparation),
        "selective_catalog_post_run_reverification": dict(
            selective_catalog_post_run_reverification_record(catalog_preparation)
        ),
        "selective_residency_policy": {
            "full_phase_tree_logical_bytes": full_phase_tree_bytes,
            "largest_required_current_plus_next_or_final_norm_bytes": (
                required_logical_budget
            ),
            "maximum_resident_evidence_bytes": logical_budget,
            "bounded_current_plus_next_system_memory": True,
            "full_phase_tree_materialized": False,
        },
        "absolute_process_rss_policy": {
            "maximum_process_rss_bytes": (config.maximum_process_rss_bytes),
            "runner_estimate_baseline_rss_bytes": compiler_baseline,
            "runner_estimate_maximum_compiler_rss_increase_bytes": (
                estimated_maximum_compiler_increase
            ),
            "runner_estimate_effective_minimum_reduction_fraction": (
                estimated_effective_reduction
            ),
            "guard_start_rss_bytes": compiled_report["physical_rss_receipt"][
                "start_rss_bytes"
            ],
            "guard_start_lifetime_peak_bytes": compiled_report["physical_rss_receipt"][
                "start_lifetime_peak_bytes"
            ],
            "guard_absolute_headroom_bytes": compiled_report["physical_rss_receipt"][
                "absolute_headroom_bytes"
            ],
            "guard_relative_maximum_increase_bytes": compiled_report[
                "physical_rss_receipt"
            ]["relative_maximum_increase_bytes"],
            "guard_effective_maximum_increase_bytes": compiled_report[
                "physical_rss_receipt"
            ]["maximum_increase_bytes"],
            "samples": memory_samples,
            "absolute_cap_passed": True,
        },
        "cuda_allocator_policy": dict(cuda_allocator_policy),
        "gpu_candidate_replay": dict(gpu.report),
        "gpu_execution_audit": gpu_execution,
        "preregistered_short_conv_v5_measurement_policy": {
            "projected_fir_offdiagonal_gate": (
                PREREGISTERED_PROJECTED_FIR_OFFDIAGONAL_GATE
            ),
            "per_lag_commutator_gate": (PREREGISTERED_PER_LAG_COMMUTATOR_GATE),
            "basis_policy": (numerical_adapter.basis_policy.authentication_body()),
            "basis_policy_sha256": (
                numerical_adapter.basis_policy.authentication_sha256
            ),
            "ridge_policy": (numerical_adapter.ridge_policy.authentication_body()),
            "ridge_policy_sha256": (
                numerical_adapter.ridge_policy.authentication_sha256
            ),
            "source_independent_preregistered_before_fit": True,
            "real_v5_adapter_required": True,
        },
        "dynamic_system_contract": {
            "automatic_blank_target_registry_used": True,
            "automatic_parameter_frontier_used": True,
            "automatic_target_topology_selection_used": True,
            "automatic_selective_layer_iteration_used": True,
            "manual_layer_surgery_in_runner": False,
            "current_executable_lane": "lfm2_text_parent_to_lfm2_text_blank",
            "any_family_any_modality_proven_by_this_run": False,
            "architecture_and_modality_adapters_remain_registry_driven": True,
        },
        "compiled_candidate_consumer_seam": {
            "schema": COMPILED_CANDIDATE_CONTEXT_SCHEMA,
            "available": True,
            "consumer_invoked_before_compile_diagnostic_sealed": False,
            "consumer_invocation_requires_sealed_compile_diagnostic": True,
            "target_plan_fingerprint": blank.plan.fingerprint,
            "blank_target_report_sha256": blank.as_dict()["report_sha256"],
            "compiler_receipt_sha256": compiler_receipt_sha256,
            "parent_sha256": cache.parent_fingerprint,
            "cache_sha256": cache.cache_key,
            "bridge_sha256": bridge.artifact_sha256,
            "corpus_sha256": corpus_sha256,
            "source_sha256": source_sha256,
            "compiler_sha256": compiler_sha256,
            "runner_source_sha256": runner_source_sha256,
            "source_binding_sha256": source_binding_sha256,
            "selective_catalog_preparation_receipt_sha256": (
                catalog_authentication["preparation_receipt_sha256"]
            ),
            "selective_catalog_binding_sha256": (
                catalog_authentication["binding_sha256"]
            ),
            "selective_catalog_artifact_manifest_sha256": (
                catalog_authentication["artifact_manifest_sha256"]
            ),
            "selective_catalog_artifact_sha256": (
                catalog_authentication["artifact_catalog_sha256"]
            ),
            "selective_catalog_codec_sha256": (catalog_authentication["codec_sha256"]),
            "selective_catalog_provider_implementation_sha256": (
                catalog_authentication["provider_implementation_sha256"]
            ),
            "fold_only_runtime_guard_identity_sha256": guard_identity[
                "identity_sha256"
            ],
            "parent_tokenizer_semantics_sha256": donor_fingerprint,
            "target_tokenizer_semantics_sha256": target_fingerprint,
            "target_tokenizer_package_sha256": None,
            "target_state_sha256": target_consumer_auth["state_sha256"],
            "target_runtime_sha256": target_consumer_auth["runtime_sha256"],
            "parent_state_sha256": parent_consumer_auth["state_sha256"],
            "parent_runtime_sha256": parent_consumer_auth["runtime_sha256"],
            "zero_operation_audit": zero_operation_audit,
            "live_target_exposed_only_to_typed_consumer": True,
            "live_parent_forbidden_and_observed": True,
            "target_mutation_or_gradients_rejected": True,
            "capture_receipt_must_remain_distinct": True,
        },
        "operation_audit": {
            "local_parent_model_loads": 1,
            "parent_forward_calls": compiler_runtime_guard[
                "parent_module_forward_calls"
            ],
            "parent_generation_calls": compiler_runtime_guard[
                "parent_inference_entrypoint_calls"
            ],
            "parent_cache_generation_calls": compiler_operation_audit[
                "parent_cache_generation_calls"
            ],
            "parent_cache_regeneration_calls": 0,
            "sealed_cache_open_count": 1,
            "selective_catalog_cold_build_count": 0,
            "selective_catalog_authenticated_warm_hit_count": 1,
            "selective_catalog_post_run_reverification_count": 1,
            "local_resume_store_open_count": int(checkpointing_enabled),
            "authenticated_resume_state_open_count": int(resume_used),
            "training_steps": compiler_runtime_guard["training_steps"],
            "training_steps_runtime_observed_directly": False,
            "training_steps_evidence_scope": (
                compiler_runtime_guard["training_steps_zero_derivation"]
            ),
            "backward_calls": compiler_runtime_guard["backward_calls"],
            "autograd_grad_calls": compiler_runtime_guard["autograd_grad_calls"],
            "optimizer_constructions": compiler_runtime_guard[
                "optimizer_constructions"
            ],
            "optimizer_steps": compiler_runtime_guard["optimizer_steps"],
            "pruning_operations": compiler_runtime_guard["pruning_operations"],
            "quantization_operations": compiler_runtime_guard[
                "quantization_operations"
            ],
            "pruning_quantization_api_observation_scope": (
                "common-public-pytorch-mutation-entrypoints"
            ),
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "parameter_pruning_quantization_evidence_scope": (
                "retained-dense-state-fact-not-complete-operation-history"
            ),
            "network_access": False,
            "hub_loader_used": False,
        },
        "claims": {
            "folding_only": True,
            "fresh_exact_dense_zero_blank_used": True,
            "tokenizer_bridge_used": True,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "trace_regeneration_used": False,
            "download_used": False,
            "resume_checkpointing_used": checkpointing_enabled,
            "resume_used": resume_used,
            "resume_state_was_exact_typed_v5": (
                not resume_used
                or resume_progress["invocation_head"]["typed_v5_state_opened"]
            ),
            "resume_parent_or_cache_tensor_sidecars": 0,
            "child_checkpoint_saved": False,
            "publication_attempted": False,
            "publication_claimed": False,
            "behavioral_intelligence_retention_proven": False,
            "manual_intelligence_review_performed": False,
        },
    }
    compile_diagnostic = {**body, "receipt_sha256": _sha(body)}
    if candidate_consumer is None:
        return compile_diagnostic
    context = V5CompiledCandidateConsumerContext(
        target_model=target,
        target_tokenizer=target_tokenizer,
        forbidden_parent_model=parent,
        target_plan_fingerprint=blank.plan.fingerprint,
        blank_target_report=blank.as_dict(),
        compiled_model_fold_report=compiled_report,
        compiler_receipt_sha256=compiler_receipt_sha256,
        compile_diagnostic=compile_diagnostic,
        parent_sha256=cache.parent_fingerprint,
        cache_sha256=cache.cache_key,
        bridge_sha256=bridge.artifact_sha256,
        corpus_sha256=corpus_sha256,
        source_sha256=source_sha256,
        compiler_sha256=compiler_sha256,
        runner_source_sha256=runner_source_sha256,
        source_binding_sha256=source_binding_sha256,
        parent_tokenizer_semantics_sha256=donor_fingerprint,
        target_tokenizer_semantics_sha256=target_fingerprint,
        target_tokenizer_package_sha256=None,
        parent_parameter_inventory=parent_inventory,
        blank_parameter_inventory=blank_inventory,
        compiled_parameter_inventory=compiled_inventory,
        zero_operation_audit=zero_operation_audit,
        target_state_sha256=target_consumer_auth["state_sha256"],
        target_runtime_sha256=target_consumer_auth["runtime_sha256"],
        parent_state_sha256=parent_consumer_auth["state_sha256"],
        parent_runtime_sha256=parent_consumer_auth["runtime_sha256"],
    )
    return _invoke_compiled_candidate_consumer(context, candidate_consumer)


def run(config: V5SelectiveBlankFoldConfig) -> dict[str, Any]:
    """Execute one strict run while retaining pre-compiler failure context."""

    try:
        result = _run(config)
        if not isinstance(result, dict):
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Ordinary run unexpectedly invoked a candidate consumer."
            )
        return result
    except Exception as exc:
        wrapped = Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The v5 selective blank fold failed closed."
        )
        wrapped.diagnostic_report = {
            "schema": FAILURE_SCHEMA,
            "request": config.as_dict(),
            "stage": "run",
            "publication_claimed": False,
            "resume_claimed": False,
            "resume_requested": (config.resume_admission_sha256 is not None),
            "resume_checkpointing_requested": (
                config.resume_store_directory is not None
            ),
            "failure_evidence_only": True,
        }
        raise wrapped from exc


def run_with_compiled_candidate_consumer(
    config: V5SelectiveBlankFoldConfig,
    consumer: Callable[
        [V5CompiledCandidateConsumerContext],
        V5CompiledCandidateConsumerResult,
    ],
) -> V5CompiledCandidateConsumerResult:
    """Compile, seal the diagnostic, then capture the still-live target."""

    try:
        result = _run(config, candidate_consumer=consumer)
        if type(result) is not V5CompiledCandidateConsumerResult:
            raise Lfm2CpcfV5SelectiveBlankFoldRunnerError(
                "Candidate-consumer run returned the wrong exact result type."
            )
        return result
    except Exception as exc:
        wrapped = Lfm2CpcfV5SelectiveBlankFoldRunnerError(
            "The v5 selective blank fold/candidate capture failed closed."
        )
        wrapped.diagnostic_report = {
            "schema": FAILURE_SCHEMA,
            "request": config.as_dict(),
            "stage": "run_with_compiled_candidate_consumer",
            "publication_claimed": False,
            "compiler_child_checkpoint_saved": False,
            "quarantine_capture_requested": True,
            "resume_requested": (config.resume_admission_sha256 is not None),
            "resume_checkpointing_requested": (
                config.resume_store_directory is not None
            ),
            "failure_evidence_only": True,
        }
        raise wrapped from exc


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 32:
        return "<maximum diagnostic depth reached>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_json_safe(item, depth=depth + 1) for item in value]
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _exception_chain(error: BaseException) -> list[dict[str, Any]]:
    chain = []
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        reports = {}
        for name in (
            "diagnostic_report",
            "failure_report",
            "report",
            "layer_failure_report",
            "diagnostics",
            "resume_progress_report",
        ):
            try:
                value = getattr(current, name)
            except Exception:
                continue
            if value is not None:
                reports[name] = _json_safe(value)
        chain.append(
            {
                "type": (f"{type(current).__module__}.{type(current).__qualname__}"),
                "message": str(current),
                "notes": [str(note) for note in getattr(current, "__notes__", ())],
                "structured_reports": reports,
            }
        )
        current = (
            current.__cause__ if current.__cause__ is not None else current.__context__
        )
    return chain


def build_failure_diagnostic(
    error: BaseException,
    *,
    config: V5SelectiveBlankFoldConfig | None = None,
    parsed_args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    chain = _exception_chain(error)
    preserved = [
        {
            "exception_index": index,
            "name": name,
            "report": report,
        }
        for index, record in enumerate(chain)
        for name, report in record["structured_reports"].items()
    ]
    resume_progress_reports = [
        item["report"] for item in preserved if item["name"] == "resume_progress_report"
    ]
    resume_requested = bool(
        (config is not None and config.resume_admission_sha256 is not None)
        or (
            parsed_args is not None
            and getattr(parsed_args, "resume_admission_sha256", None) is not None
        )
    )
    resume_checkpointing_requested = bool(
        (config is not None and config.resume_store_directory is not None)
        or (
            parsed_args is not None
            and getattr(parsed_args, "resume_store_directory", None) is not None
        )
    )
    body = {
        "schema": FAILURE_SCHEMA,
        "status": "failed",
        "request": (
            config.as_dict()
            if config is not None
            else _json_safe(vars(parsed_args) if parsed_args is not None else {})
        ),
        "exception_chain": chain,
        "preserved_structured_failure_reports": preserved,
        "structured_failure_report_count": len(preserved),
        "failure_evidence_only": True,
        "compiled_child_success_receipt_issued": False,
        "child_checkpoint_saved": False,
        "publication_attempted": False,
        "publication_claimed": False,
        "resume_attempted": resume_requested,
        "resume_checkpointing_requested": (resume_checkpointing_requested),
        "resume_progress_reports": resume_progress_reports,
        "resume_progress_report_count": len(resume_progress_reports),
        "resume_artifacts_preserved_for_diagnosis": any(
            report.get("published_pending_prefix_count", 0) > 0
            for report in resume_progress_reports
            if isinstance(report, Mapping)
        ),
        "resume_parent_or_cache_tensor_sidecars": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "network_access": False,
    }
    return {**body, "report_sha256": _sha(body)}


def _write_diagnostic(path: Path, value: Mapping[str, Any]) -> None:
    # Reuse the established local runner's create-new atomic JSON contract.
    _base_runner()._atomic_json(path, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model-directory", required=True, type=Path)
    parser.add_argument(
        "--donor-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--target-tokenizer-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--bridge-directory", required=True, type=Path)
    parser.add_argument(
        "--trace-cache-directory",
        required=True,
        type=Path,
    )
    parser.add_argument("--trajectory-corpus", required=True, type=Path)
    parser.add_argument("--diagnostic-json", required=True, type=Path)
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument("--compression-ratio")
    budget.add_argument("--target-unique-parameters", type=int)
    parser.add_argument(
        "--maximum-process-rss-gib",
        type=float,
        required=True,
        help="Hard absolute process high-water/RSS limit.",
    )
    parser.add_argument(
        "--maximum-logical-resident-gib",
        type=float,
        default=0.0,
        help=(
            "Bound for authenticated CPU evidence. Zero selects the exact "
            "largest current-plus-next layer pair or final-norm payload."
        ),
    )
    parser.add_argument(
        "--minimum-physical-rss-reduction-fraction",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--maximum-cuda-gib-per-device",
        type=float,
        default=20.0,
        help=(
            "Hard per-process PyTorch allocator cap enforced independently "
            "on each authenticated logical replay device."
        ),
    )
    parser.add_argument(
        "--model-device",
        choices=("cpu", "cuda:0"),
        default="cuda:0",
    )
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--parent-name", default="lfm2-parent")
    parser.add_argument(
        "--child-name",
        default="lfm2-cpcf-v5-selective-folded-child",
    )
    parser.add_argument("--blank-seed", type=int, default=0)
    parser.add_argument("--minimum-hidden-size", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=8)
    parser.add_argument("--selected-candidate-rank", type=int)
    parser.add_argument("--expected-target-plan-fingerprint")
    parser.add_argument("--ridge", type=float, default=1e-8)
    parser.add_argument(
        "--boundary-state-reconstruction-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-predictive-energy-floor",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--boundary-validation-retention-floor",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--short-conv-v4-ridge-alpha",
        action="append",
        type=float,
        dest="short_conv_v4_ridge_alphas",
    )
    parser.add_argument(
        "--short-conv-v4-variable-projection-passes",
        choices=(2, 3),
        type=int,
        default=2,
    )
    parser.add_argument(
        "--short-conv-v4-writer-trust-radius",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--short-conv-v4-fir-trust-radius",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--short-conv-v4-bcx-trust-radius",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--short-conv-v4-correction-rms-floor",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--short-conv-v4-context-objective-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--resume-store-directory",
        type=Path,
        help=(
            "Explicit local immutable v5 prefix-checkpoint store. Supplying "
            "only this path enables cold-run checkpointing."
        ),
    )
    parser.add_argument(
        "--resume-admission-sha256",
        help=(
            "Exact admitted typed-v5 prefix digest to continue from within "
            "the explicit local resume store."
        ),
    )
    parser.add_argument(
        "--selective-catalog-store-directory",
        type=Path,
        help=(
            "Shared immutable tensor-free selective catalog store. Production "
            "folds require an exact prepared warm hit."
        ),
    )
    parser.add_argument(
        "--selective-catalog-receipt-json",
        type=Path,
        help=(
            "Create-new preparation receipt in prepare-only mode, or the "
            "existing authenticated receipt required by a fold worker."
        ),
    )
    parser.add_argument(
        "--expected-selective-catalog-receipt-sha256",
        help=("Exact preparation receipt digest required by a production fold worker."),
    )
    parser.add_argument(
        "--prepare-selective-catalog-only",
        action="store_true",
        help=(
            "Build/authenticate only the shared tensor-free catalog and exit "
            "before parent, blank-target, or CUDA allocation."
        ),
    )
    # Parse legacy/untyped resume and all publication or learning requests so
    # they fail with an explicit safety reason rather than being ignored.
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume-state", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume-checkpoint-reference",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--publish", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--publication-directory",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--save-child",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    for name in (
        "regenerate-traces",
        "train",
        "prune",
        "quantize",
        "download",
    ):
        parser.add_argument(
            f"--{name}",
            action="store_true",
            help=argparse.SUPPRESS,
        )
    return parser


def config_from_args(args: argparse.Namespace) -> V5SelectiveBlankFoldConfig:
    base = _base_runner()
    forbidden = tuple(
        name
        for name in _FORBIDDEN_REQUEST_FIELDS
        if getattr(args, name, None) not in (None, False)
    )
    return V5SelectiveBlankFoldConfig(
        parent_model_directory=args.parent_model_directory,
        donor_tokenizer_directory=args.donor_tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        bridge_directory=args.bridge_directory,
        trace_cache_directory=args.trace_cache_directory,
        trajectory_corpus=args.trajectory_corpus,
        diagnostic_json=args.diagnostic_json,
        compression=base.CompressionRequest(
            compression_ratio=args.compression_ratio,
            target_unique_parameters=args.target_unique_parameters,
        ),
        maximum_process_rss_gib=args.maximum_process_rss_gib,
        maximum_logical_resident_gib=(args.maximum_logical_resident_gib),
        minimum_physical_rss_reduction_fraction=(
            args.minimum_physical_rss_reduction_fraction
        ),
        maximum_cuda_gib_per_device=args.maximum_cuda_gib_per_device,
        model_device=args.model_device,
        dtype=args.dtype,
        parent_name=args.parent_name,
        child_name=args.child_name,
        blank_seed=args.blank_seed,
        minimum_hidden_size=args.minimum_hidden_size,
        maximum_candidates=args.maximum_candidates,
        selected_candidate_rank=args.selected_candidate_rank,
        expected_target_plan_fingerprint=(args.expected_target_plan_fingerprint),
        ridge=args.ridge,
        boundary_state_reconstruction_weight=(
            args.boundary_state_reconstruction_weight
        ),
        boundary_predictive_energy_floor=(args.boundary_predictive_energy_floor),
        boundary_validation_retention_floor=(args.boundary_validation_retention_floor),
        short_conv_v4_ridge_alphas=(
            (1e-8, 1e-4)
            if args.short_conv_v4_ridge_alphas is None
            else tuple(args.short_conv_v4_ridge_alphas)
        ),
        short_conv_v4_variable_projection_passes=(
            args.short_conv_v4_variable_projection_passes
        ),
        short_conv_v4_writer_trust_radius=(args.short_conv_v4_writer_trust_radius),
        short_conv_v4_fir_trust_radius=(args.short_conv_v4_fir_trust_radius),
        short_conv_v4_bcx_trust_radius=(args.short_conv_v4_bcx_trust_radius),
        short_conv_v4_correction_rms_floor=(args.short_conv_v4_correction_rms_floor),
        short_conv_v4_context_objective_weight=(
            args.short_conv_v4_context_objective_weight
        ),
        resume_store_directory=args.resume_store_directory,
        resume_admission_sha256=args.resume_admission_sha256,
        selective_catalog_store_directory=(args.selective_catalog_store_directory),
        selective_catalog_receipt_json=(args.selective_catalog_receipt_json),
        expected_selective_catalog_receipt_sha256=(
            args.expected_selective_catalog_receipt_sha256
        ),
        prepare_selective_catalog_only=(args.prepare_selective_catalog_only),
        forbidden_requests=forbidden,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config: V5SelectiveBlankFoldConfig | None = None
    try:
        config = config_from_args(args)
        receipt = (
            dict(prepare_selective_catalog(config))
            if config.prepare_selective_catalog_only
            else run(config)
        )
    except Exception as exc:
        failure = build_failure_diagnostic(
            exc,
            config=config,
            parsed_args=args,
        )
        try:
            _write_diagnostic(args.diagnostic_json, failure)
        except Exception as write_error:
            print(
                "error: v5 fold failed and diagnostic persistence also "
                f"failed: {write_error}",
                file=sys.stderr,
            )
        print(f"error: {exc}", file=sys.stderr)
        for record in _exception_chain(exc):
            print(
                f"cause: {record['type']}: {record['message']}",
                file=sys.stderr,
            )
        return 2
    if not config.prepare_selective_catalog_only:
        try:
            _write_diagnostic(config.diagnostic_json, receipt)
        except Exception as exc:
            print(
                "error: the in-memory fold completed but its diagnostic could "
                f"not be persisted: {exc}",
                file=sys.stderr,
            )
            return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt.get(
                    "receipt_sha256",
                    receipt.get("preparation_receipt_sha256"),
                ),
                "diagnostic_json": (
                    None
                    if config.prepare_selective_catalog_only
                    else str(config.diagnostic_json.expanduser().absolute())
                ),
                "selective_catalog_receipt_json": (
                    None
                    if config.selective_catalog_receipt_json is None
                    else str(
                        config.selective_catalog_receipt_json.expanduser().absolute()
                    )
                ),
                "child_checkpoint_saved": False,
                "publication_claimed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
