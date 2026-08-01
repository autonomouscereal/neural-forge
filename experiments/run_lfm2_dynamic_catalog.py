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
import stat
import sys
import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime, timezone
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
from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_behavior_gate import (
    BehaviorGateBinding,
    WholeOutputMetadata,
    expose_behavior_gate_binding,
    resolve_behavior_gate_binding,
)
from neural_forge.dynamic_counted_child_state import (
    CountedChildStatePlan,
    build_counted_child_state_plan,
    verify_counted_child_state,
)
from neural_forge.dynamic_closed_loop_fold import ClosedLoopCompilerConfig
from neural_forge.dynamic_fold_executor import FoldDecision
from neural_forge.dynamic_fold_executor import fingerprint_model_state
from neural_forge.dynamic_fold_artifact import (
    load_dynamic_fold_artifact_certificate,
)
from neural_forge.dynamic_fold_registry import (
    DurableExperimentLedger,
    ExperimentLedger,
)
from neural_forge.dynamic_fold_phase_cache import (
    DynamicFoldPhaseCache,
    FoldPhaseEventLog,
)
from neural_forge.dynamic_fold_snapshot import DynamicFoldSnapshotConfig
from neural_forge.dynamic_model_inspector import inspect_model_circuits
from neural_forge.dynamic_parameter_frontier import (
    DenseParameterBudgetRequest,
    DenseTargetTopologyCatalogRegistry,
    Lfm2DenseTargetTopologyCatalog,
    Lfm2HeterogeneousLayoutSelectionInput,
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
    GENERATION_PROTOCOL,
    Lfm2CachedWholeOutputBaseline,
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
    DENSE_CLOSED_LOOP_MERGE_CAPTURE_POLICY_STAGE,
    Lfm2DenseBudgetFoldConfig,
)
from neural_forge.hf_lfm2_balanced_basis_phase_cache import (
    Lfm2BalancedBasisPhaseCacheContext,
    Lfm2ExistingBalancedBasisPhaseReuseRequest,
    PhaseAnchorAuthority,
)
from neural_forge.hf_lfm2_cached_heterogeneous_capacity import (
    ATTENTION_GEOMETRY_RISK_METHOD,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    build_lfm2_cached_heterogeneous_capacity_option_bank,
)
from neural_forge.hf_lfm2_cached_global_topology_frontier import (
    Lfm2CachedGlobalTopologyEvidenceCacheMiss,
    Lfm2CachedGlobalTopologyFrontier,
    Lfm2GlobalTopologyPhaseCacheContext,
    Lfm2GlobalTargetRuntimePins,
    build_lfm2_cached_global_topology_frontier,
)
from neural_forge.hf_lfm2_dynamic_closed_loop_adapter import (
    LFM2_REPLAY_SHARDING_ALL_DEVICES,
    LFM2_REPLAY_SHARDING_COMPLETED_STAGE,
    LFM2_REPLAY_SHARDING_RESOLVED_STAGE,
    Lfm2DynamicClosedLoopAdapterConfig,
)
from neural_forge.hf_lfm2_direct_local import (
    BRIDGE_REFERENCE_FILENAME,
    Lfm2CanonicalFp32RuntimeRelocationReceipt,
    fingerprint_lfm2_direct_local_runtime_tensor_state,
    load_lfm2_direct_local_model,
    relocate_authenticated_lfm2_canonical_fp32_model,
)
from neural_forge.hf_lfm2_macro_cell_attractor_gate import (
    Lfm2MacroCellAttractorGateConfig,
)
from neural_forge.hf_lfm2_on_policy_evidence_round import (
    BRIDGE_MAPPED_CAPTURE_INPUT_MODE,
    DEFAULT_MAX_ROUNDS as DEFAULT_ON_POLICY_MAX_ROUNDS,
    PARENT_CANONICAL_CAPTURE_INPUT_MODE,
    Lfm2OnPolicyEvidenceRound,
    Lfm2OnPolicyRolloutConfig,
    build_lfm2_on_policy_round_target_records,
    lfm2_on_policy_round_missing_target_preimage_reason,
    load_lfm2_on_policy_evidence_round,
    resolve_lfm2_on_policy_evidence_references,
    run_lfm2_on_policy_evidence_round,
)
from neural_forge.hf_lfm2_sealed_evidence_set import (
    BASE_EVIDENCE_ROLE,
    ON_POLICY_EVIDENCE_IDENTITY_SCHEMA,
    Lfm2SealedEvidenceRequest,
    Lfm2SealedEvidenceSet,
    lfm2_on_policy_evidence_identity_sha256,
)
from neural_forge.hf_lfm2_heterogeneous_layout_selection import (
    build_lfm2_heterogeneous_target_pins,
)
from neural_forge.hf_lfm2_circuit_trace_cache import (
    CACHED_VERIFIED_TRACE_MODE,
    CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
    CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
    FULL_PREFIX_CANONICAL_TRACE_MODE,
    Lfm2CircuitTraceCache,
    Lfm2CircuitTraceCaptureConfig,
    Lfm2CircuitTraceSelectiveAuthenticatedSession,
    _layer_topology as _lfm2_layer_topology,
    _prepare_batches as _prepare_lfm2_trace_batches,
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
from neural_forge.hf_lfm2_tokenizer_bridge_trace_adapter import (
    Lfm2PreverifiedTokenizerBridgeTraceSession,
    Lfm2StrictTokenizerBridgeTraceAdapter,
    Lfm2TargetTokenizedRecord,
    authenticate_lfm2_bridge_input_pair,
    build_preverified_lfm2_strict_tokenizer_bridge_trace_session,
    project_lfm2_cached_parent_logits,
)
from neural_forge.hf_lfm2_streaming_vocabulary_margin_adapter import (
    Lfm2TargetTokenizerVocabularyMarginBinding,
)
from neural_forge.hf_tokenizer_bridge import (
    CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY,
    PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY,
    STRICT_TOKENIZER_BRIDGE_FILENAME,
    TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY,
    TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY,
    StrictTokenizerBridgeArtifact,
    fingerprint_hf_tokenizer_semantics,
    load_strict_tokenizer_bridge_artifact,
    strict_tokenizer_bridge_target_package_fingerprint,
)
from neural_forge.lfm2_native_exact4x_target import (
    lfm2_identity_tokenizer_evidence_sha256,
)
from neural_forge.lfm2_carrier_bridge_publication import (
    PACKAGED_ROUTE_DIRECTORY,
    AuthenticatedLfm2CarrierBridgePublication,
    AuthenticatedPackagedLfm2CarrierRoutes,
    load_authenticated_lfm2_span_evidence,
    load_lfm2_carrier_bridge_publication,
)
from neural_forge.lfm2_carrier_child_package import (
    CARRIER_CHILD_PACKAGE_RECEIPT,
    Lfm2CarrierChildCheckpointIO,
    Lfm2CarrierChildPackageRegistration,
    load_authenticated_lfm2_carrier_child_package,
)
from neural_forge.lfm2_carrier_route_runtime import (
    LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT,
    install_lfm2_semantic_step_generate_class,
    lfm2_carrier_manual_review_disposition,
    run_lfm2_resumable_adaptive_carrier_manual_review,
    run_lfm2_carrier_route_smoke,
)
from neural_forge.lfm2_semantic_source_lexicon import (
    SEMANTIC_SOURCE_LEXICON_DIRECTORY,
    AuthenticatedLfm2SemanticSourceLexicon,
    load_packaged_lfm2_semantic_source_lexicon,
)


RECEIPT_SCHEMA = "neural-forge-production-lfm2-catalog-receipt-v1"
DIAGNOSTICS_SCHEMA = "neural-forge-production-lfm2-gate-diagnostics-v1"
SEALED_PARENT_CACHE_SOURCE_MODE = "sealed_existing"
CALIBRATION_PARENT_CACHE_SOURCE_MODE = "calibration_build_or_reuse"
SEALED_CACHE_SPLIT_SCHEMA = (
    "neural-forge-production-lfm2-sealed-cache-evidence-splits-v1"
)
# ``ON_POLICY_EVIDENCE_IDENTITY_SCHEMA`` and the digest that carries it now live
# in ``hf_lfm2_sealed_evidence_set`` so the sealed-split identity, the
# balanced-basis phase dependency, and the evidence aggregate share exactly one
# definition.  The name stays bound here because it is part of this module's
# published surface.
ON_POLICY_PHASE_DEPENDENCY_NAME = "lfm2_on_policy_evidence_extension"
ON_POLICY_REFOLD_RUN_SUFFIX = "round-{index}-refold"
ON_POLICY_REFOLD_ALTERNATION_SCHEMA = (
    "neural-forge-production-lfm2-on-policy-refold-alternation-v1"
)
SEALED_EVIDENCE_TRACE_ROWS_SCHEMA = (
    "neural-forge-production-lfm2-sealed-evidence-trace-rows-v1"
)
TOKENIZER_BRIDGE_RUNTIME_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-runtime-v1"
)
TOKENIZER_BRIDGE_PROJECTED_CACHE_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-projected-cache-v1"
)
TOKENIZER_BRIDGE_PARENT_ROWS_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-parent-rows-v1"
)
TOKENIZER_BRIDGE_MAPPED_OUTPUT_ENVELOPE_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-mapped-parent-output-v2"
)
TOKENIZER_BRIDGE_NORMALIZATION_CONTROL_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-normalization-control-v1"
)
TOKENIZER_BRIDGE_DUAL_BEHAVIOR_SCHEMA = (
    "neural-forge-production-lfm2-tokenizer-bridge-dual-behavior-gate-v1"
)

DEFAULT_TARGET_QUERY_HEADS = 16
DEFAULT_TARGET_KEY_VALUE_HEADS = 8
DEFAULT_TARGET_INTERMEDIATE_SIZE = 2552
DEFAULT_TARGET_UNIQUE_PARAMETERS = 229_349_120
DEFAULT_DENSE_MINIMUM_HIDDEN_SIZE = 32
LFM2_DENSE_HIDDEN_SIZE_MULTIPLE = 8
DEFAULT_CAPTURE_DECODE_STEPS = 8
DEFAULT_CAPTURE_BYTES = 128 * 1024 * 1024
DEFAULT_CAPTURE_LOGIT_RTOL = 0.2
DEFAULT_CAPTURE_LOGIT_ATOL = 1.0
DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE = "auto"
# Multi-device child-replay sharding (NOTES-2B).  The CLI flag wins over the
# environment; the environment is only consulted for a closed-loop run so an
# exported device list can never change what an unrelated run does.
REPLAY_DEVICES_ENVIRONMENT_VARIABLE = "NEURAL_FORGE_REPLAY_DEVICES"
LEGACY_REPLAY_DEVICES_ENVIRONMENT_VARIABLE = "NEURAL_FORGE_LFM2_REPLAY_DEVICES"
REPLAY_DEVICES_SOURCE_UNSET = "unset"
REPLAY_DEVICES_SOURCE_CLI = "cli"
REPLAY_DEVICES_SOURCE_ENVIRONMENT = "env"
REPLAY_DEVICES_SOURCE_ENVIRONMENT_IGNORED = "env_ignored"
REPLAY_DEVICES_SOURCES = (
    REPLAY_DEVICES_SOURCE_UNSET,
    REPLAY_DEVICES_SOURCE_CLI,
    REPLAY_DEVICES_SOURCE_ENVIRONMENT,
    REPLAY_DEVICES_SOURCE_ENVIRONMENT_IGNORED,
)
REPLAY_DEVICES_ALL_TOKEN = LFM2_REPLAY_SHARDING_ALL_DEVICES
_REPLAY_DEVICE_PATTERN = re.compile(r"\A(?:cpu|cuda(?::\d+)?)\Z")
PORTABLE_RUNTIME_DTYPES = MappingProxyType(
    {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
    }
)

# The current production carrier route has two independent 1,933,888-element
# FP32 channels.  They are child state, so the physical H816 body receives the
# remainder of the sealed 57,420,144-element target rather than that total.
LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS = 57_420_144
LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS = 53_552_368
LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE = 1_933_888
LFM2_CARRIER_TARGET_ROWS = 4_096
LFM2_CARRIER_HIDDEN_SIZE = 816
LFM2_CARRIER_SMOKE_MAXIMUM_NEW_TOKENS = 2
LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS = 1_024
LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS = 2_048
LFM2_SEMANTIC_SOURCE_TOKENIZER_DIRECTORY = "semantic-source-tokenizer"
LFM2_ROUTED_MANUAL_DOSSIER_SCHEMA = (
    "neural-forge-lfm2-routed-manual-dossier-v1"
)
PROGRESS_JOURNAL_SCHEMA = "neural-forge-lfm2-catalog-progress-v1"
_BALANCED_BASIS_PHASE_KEY_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z"
)
_MINIMUM_BALANCED_BASIS_HMAC_KEY_BYTES = 32
_MAXIMUM_BALANCED_BASIS_HMAC_KEY_BYTES = 4096

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


def _is_reparse_point(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    marker = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    return bool(marker and attributes & marker)


def _key_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _read_balanced_basis_phase_hmac_key(path: Path) -> bytes:
    """Read one bounded ordinary raw key without following link-like paths."""

    source = Path(path).expanduser().absolute()
    try:
        for ancestor in reversed(source.parents):
            metadata = ancestor.lstat()
            if stat.S_ISLNK(metadata.st_mode) or _is_reparse_point(metadata):
                raise ProductionLfm2CatalogError(
                    "Balanced-basis phase HMAC key ancestors cannot be links "
                    "or reparse points."
                )
        before = source.lstat()
    except OSError as exc:
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key file is unavailable."
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or _is_reparse_point(before)
        or not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key must be an ordinary non-linked file."
        )
    if os.name != "nt" and stat.S_IMODE(before.st_mode) & 0o077:
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key must not grant group or world "
            "permissions."
        )
    if not (
        _MINIMUM_BALANCED_BASIS_HMAC_KEY_BYTES
        <= int(before.st_size)
        <= _MAXIMUM_BALANCED_BASIS_HMAC_KEY_BYTES
    ):
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key must contain between 32 and 4096 "
            "raw bytes."
        )
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        opened = os.fstat(descriptor)
        if (
            _key_file_identity(opened) != _key_file_identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or (
                os.name != "nt"
                and bool(stat.S_IMODE(opened.st_mode) & 0o077)
            )
        ):
            raise ProductionLfm2CatalogError(
                "Balanced-basis phase HMAC key changed while it was opened."
            )
        chunks: list[bytes] = []
        remaining = _MAXIMUM_BALANCED_BASIS_HMAC_KEY_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        secret = b"".join(chunks)
        after_open = os.fstat(descriptor)
        after_path = source.lstat()
        if (
            _key_file_identity(after_open) != _key_file_identity(opened)
            or _key_file_identity(after_path) != _key_file_identity(before)
        ):
            raise ProductionLfm2CatalogError(
                "Balanced-basis phase HMAC key changed while it was read."
            )
    except ProductionLfm2CatalogError:
        raise
    except OSError as exc:
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key file could not be read safely."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not (
        _MINIMUM_BALANCED_BASIS_HMAC_KEY_BYTES
        <= len(secret)
        <= _MAXIMUM_BALANCED_BASIS_HMAC_KEY_BYTES
    ):
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase HMAC key must contain between 32 and 4096 "
            "raw bytes."
        )
    return secret


def _process_memory_progress_metadata() -> dict[str, int]:
    """Return best-effort scalar process memory telemetry for progress logs."""

    result: dict[str, int] = {}
    try:
        status = Path("/proc/self/status").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        status = ""
    names = {
        "VmRSS": "process_rss_bytes",
        "VmHWM": "process_peak_rss_bytes",
        "VmSwap": "process_swap_bytes",
    }
    for line in status.splitlines():
        name, separator, raw_value = line.partition(":")
        if not separator or name not in names:
            continue
        fields = raw_value.split()
        if len(fields) == 2 and fields[1] == "kB" and fields[0].isdigit():
            result[names[name]] = int(fields[0]) * 1024
    if "process_rss_bytes" not in result:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            pass
        else:
            try:
                process = psutil.Process(os.getpid())
                result["process_rss_bytes"] = int(process.memory_info().rss)
                full = process.memory_full_info()
                if hasattr(full, "swap"):
                    result["process_swap_bytes"] = int(full.swap)
            except (AttributeError, OSError, psutil.Error):
                pass
    if "process_peak_rss_bytes" not in result:
        try:
            import resource

            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            result["process_peak_rss_bytes"] = (
                peak if sys.platform == "darwin" else peak * 1024
            )
        except (ImportError, OSError, ValueError):
            pass
    return result


class _ProgressJournal:
    """Emit small, immediately flushed, tensor-free phase records."""

    _RESERVED_FIELDS = frozenset(
        {
            "schema",
            "timestamp_utc",
            "configuration_sha256",
            "sequence",
            "stage",
        }
    )
    _FORBIDDEN_FIELD_FRAGMENTS = (
        "credential",
        "password",
        "secret",
        "access_token",
        "api_key",
        "auth_token",
    )

    def __init__(self, configuration_sha256: str, *, stream: Any = None) -> None:
        self._configuration_sha256 = configuration_sha256
        self._stream = sys.stdout if stream is None else stream
        self._sequence = 0
        # Child-replay shard telemetry is unsealed wall-clock accounting that
        # only the adapter observes.  The journal is the one place every such
        # event already passes through, so the run receipt reads its summary
        # from here instead of threading a second sink through the wiring.
        self._replay_sharding: dict[str, Any] = {
            "resolved_capture_count": 0,
            "sharded_capture_count": 0,
            "fallback_capture_count": 0,
            "first_fallback_reason": None,
            "resolved_devices": None,
            "worker_count": None,
            "completed_capture_count": 0,
            "replay_wall_seconds": 0.0,
            "worker_replay_seconds_total": 0.0,
            "maximum_worker_peak_capture_bytes": 0,
        }
        self._merged_capture_policy: dict[str, Any] | None = None

    def _observe_merged_capture_policy(
        self,
        stage: str,
        metadata: Mapping[str, Any],
    ) -> None:
        if stage != DENSE_CLOSED_LOOP_MERGE_CAPTURE_POLICY_STAGE:
            return
        self._merged_capture_policy = {
            "requested": bool(
                metadata.get("merge_adjacent_layer_captures_requested")
            ),
            "admitted_by_adapter": bool(
                metadata.get("merge_adjacent_layer_captures_admitted")
            ),
            "replay_capture_mode": metadata.get("replay_capture_mode"),
            "selective_session_bound": metadata.get(
                "selective_session_bound"
            ),
            "heterogeneous_child": metadata.get("heterogeneous_child"),
        }

    def merged_capture_policy(self) -> dict[str, Any] | None:
        """Return the fold's requested/admitted capture-merge decision."""

        if self._merged_capture_policy is None:
            return None
        return dict(self._merged_capture_policy)

    def _observe_replay_sharding(
        self,
        stage: str,
        metadata: Mapping[str, Any],
    ) -> None:
        summary = self._replay_sharding
        if stage == LFM2_REPLAY_SHARDING_RESOLVED_STAGE:
            summary["resolved_capture_count"] += 1
            if bool(metadata.get("replay_sharding_enabled")):
                summary["sharded_capture_count"] += 1
                summary["resolved_devices"] = metadata.get(
                    "replay_sharding_resolved_devices"
                )
                summary["worker_count"] = metadata.get(
                    "replay_sharding_worker_count"
                )
            else:
                summary["fallback_capture_count"] += 1
                if summary["first_fallback_reason"] is None:
                    summary["first_fallback_reason"] = metadata.get(
                        "replay_sharding_fallback_reason"
                    )
            return
        if stage != LFM2_REPLAY_SHARDING_COMPLETED_STAGE:
            return
        summary["completed_capture_count"] += 1
        summary["replay_wall_seconds"] += float(
            metadata.get("replay_sharding_replay_wall_seconds", 0.0) or 0.0
        )
        summary["worker_replay_seconds_total"] += float(
            metadata.get("replay_sharding_worker_replay_seconds_total", 0.0)
            or 0.0
        )
        summary["maximum_worker_peak_capture_bytes"] = max(
            int(summary["maximum_worker_peak_capture_bytes"]),
            int(
                metadata.get("replay_sharding_worker_peak_capture_bytes", 0)
                or 0
            ),
        )

    def replay_sharding_summary(self) -> dict[str, Any] | None:
        """Return observed shard telemetry, or ``None`` if nothing sharded."""

        summary = dict(self._replay_sharding)
        if summary["resolved_capture_count"] == 0:
            return None
        wall = float(summary["replay_wall_seconds"])
        summary["observed_speedup"] = (
            float(summary["worker_replay_seconds_total"]) / wall
            if wall > 0.0
            else None
        )
        return summary

    def emit(self, stage: str, **metadata: Any) -> None:
        if not isinstance(stage, str) or not stage:
            raise ProductionLfm2CatalogError(
                "Progress journal stage must be a nonempty string."
            )
        safe_metadata: dict[str, Any] = {}
        for key, value in metadata.items():
            normalized_key = str(key)
            lowered_key = normalized_key.casefold()
            if normalized_key in self._RESERVED_FIELDS or any(
                fragment in lowered_key
                for fragment in self._FORBIDDEN_FIELD_FRAGMENTS
            ):
                raise ProductionLfm2CatalogError(
                    f"Unsafe progress journal metadata field: {normalized_key}."
                )
            if value is not None and not isinstance(
                value,
                (bool, int, float, str),
            ):
                raise ProductionLfm2CatalogError(
                    "Progress journal metadata must contain only JSON scalars; "
                    f"{normalized_key} received {type(value).__name__}."
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ProductionLfm2CatalogError(
                    f"Progress journal metadata field {normalized_key} is nonfinite."
                )
            safe_metadata[normalized_key] = value
        payload = {
            "schema": PROGRESS_JOURNAL_SCHEMA,
            "timestamp_utc": (
                datetime.now(timezone.utc)
                .isoformat(timespec="milliseconds")
                .replace("+00:00", "Z")
            ),
            "configuration_sha256": self._configuration_sha256,
            "sequence": self._sequence,
            "stage": stage,
            **safe_metadata,
        }
        print(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            file=self._stream,
            flush=True,
        )
        self._sequence += 1
        self._observe_replay_sharding(stage, safe_metadata)
        self._observe_merged_capture_policy(stage, safe_metadata)


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


def _portable_runtime_dtype(value: str) -> torch.dtype:
    if not isinstance(value, str) or not value.strip():
        raise ProductionLfm2CatalogError(
            "Portable runtime dtype must be explicitly named."
        )
    normalized = value.strip().lower()
    try:
        return PORTABLE_RUNTIME_DTYPES[normalized]
    except KeyError as exc:
        raise ProductionLfm2CatalogError(
            "Portable runtime dtype must be FP16, BF16, FP32, or FP64; "
            "automatic fallback is not allowed."
        ) from exc


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


TOKENIZER_BRIDGE_REFERENCE_SCHEMA = (
    "neural-forge-catalog-tokenizer-bridge-reference-v1"
)


def _load_sealed_routed_manual_prompts(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load the manual prompt artifact without giving it gate authority."""

    source = Path(path).expanduser()
    try:
        resolved = source.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise ProductionLfm2CatalogError(
            "Routed manual prompt artifact is missing."
        ) from exc
    if source.is_symlink() or not resolved.is_file():
        raise ProductionLfm2CatalogError(
            "Routed manual prompt artifact must be an ordinary file."
        )
    if _file_sha256(resolved) != _digest(
        expected_sha256,
        "routed_manual_prompts_sha256",
    ):
        raise ProductionLfm2CatalogError(
            "Routed manual prompt artifact digest mismatch."
        )
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionLfm2CatalogError(
            "Routed manual prompt artifact is not UTF-8 JSON."
        ) from exc
    rows = payload.get("prompts") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or not rows:
        raise ProductionLfm2CatalogError(
            "Routed manual prompt artifact has no prompt rows."
        )
    normalized: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProductionLfm2CatalogError(
                "Every routed manual prompt row must be an object."
            )
        prompt_id = row.get("id")
        category = row.get("category")
        if (
            not isinstance(prompt_id, str)
            or not prompt_id.strip()
            or prompt_id in seen
            or not isinstance(category, str)
            or not category.strip()
        ):
            raise ProductionLfm2CatalogError(
                f"Routed manual prompt row {index} has invalid identity metadata."
            )
        has_prompt = isinstance(row.get("prompt"), str) and bool(
            str(row["prompt"]).strip()
        )
        messages = row.get("messages")
        has_messages = (
            isinstance(messages, Sequence)
            and not isinstance(messages, (str, bytes))
            and bool(messages)
        )
        if has_prompt == has_messages:
            raise ProductionLfm2CatalogError(
                f"Routed manual prompt row {index} must contain exactly one "
                "of prompt or messages."
            )
        if has_messages and any(
            not isinstance(message, Mapping)
            or message.get("role") not in {"system", "user", "assistant"}
            or not isinstance(message.get("content"), str)
            or not str(message["content"]).strip()
            for message in messages
        ):
            raise ProductionLfm2CatalogError(
                f"Routed manual prompt row {index} has invalid chat messages."
            )
        seen.add(prompt_id)
        normalized.append(MappingProxyType(dict(row)))
    if not any(
        "safety" not in str(row["category"]).casefold()
        for row in normalized
    ):
        raise ProductionLfm2CatalogError(
            "Routed route smoke requires at least one non-safety prompt."
        )
    return tuple(normalized)


def _scanner_safe_route_response(value: Any) -> Any:
    """Rename internal semantic-clock keys before writing the public dossier."""

    if isinstance(value, Mapping):
        result = {}
        for raw_key, item in value.items():
            key = str(raw_key).replace("donor", "semantic_source").replace(
                "parent",
                "semantic_source",
            )
            if key in result:
                raise ProductionLfm2CatalogError(
                    "Scanner-safe route response key normalization collided."
                )
            result[key] = _scanner_safe_route_response(item)
        return result
    if isinstance(value, list):
        return [_scanner_safe_route_response(item) for item in value]
    if isinstance(value, tuple):
        return [_scanner_safe_route_response(item) for item in value]
    return value


def _build_registered_carrier_counted_plan(
    registration: Lfm2CarrierChildPackageRegistration,
    *,
    parent_unique_parameters: int,
    target_total_parameter_elements: int,
    observed_allocations: Sequence[Any],
) -> CountedChildStatePlan:
    """Bind the live fold to the one registered current exact-4x lane."""

    registered = registration.current_exact4x_counted_plan()
    if parent_unique_parameters != registered.parent_unique_parameters:
        raise ProductionLfm2CatalogError(
            "Current carrier routes belong to the registered "
            "229,693,184-element LFM2 source."
        )
    observed = build_counted_child_state_plan(
        parent_unique_parameters=parent_unique_parameters,
        target_total_parameter_elements=target_total_parameter_elements,
        continuous_allocations=observed_allocations,
    )
    if observed != registered:
        raise ProductionLfm2CatalogError(
            "Catalog counted-state plan differs from the registered current "
            "exact-4x carrier plan."
        )
    return observed


def _sealed_parent_identity_hint(
    config: "ProductionLfm2CatalogConfig",
) -> str | None:
    """Read only the parent identity needed to select its physical dtype.

    The hint is never accepted as cache verification. The ordinary sealed
    cache open immediately afterward still authenticates every committed file
    and compares the complete live parent receipt.
    """

    if config.sealed_parent_cache_directory is None:
        return None
    manifest_path = (
        config.sealed_parent_cache_directory.expanduser().resolve(strict=True)
        / "manifest.json"
    )
    try:
        payload = json.loads(manifest_path.read_text(encoding="ascii"))
        identity = payload["parent"]["identity_sha256"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ProductionLfm2CatalogError(
            "Sealed parent cache manifest cannot provide a parent identity "
            "hint for dtype resolution."
        ) from exc
    return _digest(identity, "sealed parent identity hint")


def _load_cache_bound_parent(
    config: "ProductionLfm2CatalogConfig",
    parent_directory: Path,
    *,
    device: torch.device,
    child_dtype: torch.dtype,
) -> tuple[Any, torch.dtype, str]:
    """Load the local parent in the dtype that reproduces its sealed identity."""

    expected_identity = _sealed_parent_identity_hint(config)
    if expected_identity is None:
        model = residual.load_local_lfm2_model(
            parent_directory,
            device=device,
            dtype=child_dtype,
        )
        return model, child_dtype, str(child_dtype).removeprefix("torch.")

    candidate_names: list[str] = []
    try:
        local_config = json.loads(
            (parent_directory / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProductionLfm2CatalogError(
            "Local parent config cannot resolve its cache-bound dtype."
        ) from exc
    for key in ("dtype", "torch_dtype"):
        value = local_config.get(key)
        if isinstance(value, str):
            candidate_names.append(value)
    candidate_names.extend(
        (
            str(child_dtype).removeprefix("torch."),
            "bfloat16",
            "float16",
            "float32",
        )
    )
    candidates: list[torch.dtype] = []
    for name in candidate_names:
        try:
            candidate_dtype = residual._torch_dtype(name)
        except Exception:
            continue
        if candidate_dtype not in candidates:
            candidates.append(candidate_dtype)

    observed: dict[str, str] = {}
    for candidate_dtype in candidates:
        model = residual.load_local_lfm2_model(
            parent_directory,
            device=device,
            dtype=candidate_dtype,
        )
        identity = fingerprint_lfm2_parent(model).identity_sha256
        normalized_name = str(candidate_dtype).removeprefix("torch.")
        observed[normalized_name] = identity
        if identity == expected_identity:
            return model, candidate_dtype, normalized_name
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    raise ProductionLfm2CatalogError(
        "No supported local parent dtype reproduces the sealed cache parent "
        f"identity {expected_identity}; observed {observed}."
    )


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
    derived_global_topology_cache_root: Path | None = None
    balanced_basis_phase_cache_root: Path | None = None
    balanced_basis_phase_event_log: Path | None = None
    balanced_basis_phase_hmac_key_file: Path | None = None
    balanced_basis_phase_hmac_key_id: str | None = None
    balanced_basis_existing_phase_key_sha256: str | None = None
    tokenizer_directory: Path | None = None
    target_tokenizer_directory: Path | None = None
    target_tokenizer_calibration: Path | None = None
    tokenizer_bridge_artifact_sha256: str | None = None
    tokenizer_bridge_profile_sha256: str | None = None
    tokenizer_bridge_mapped_parent_outputs: Path | None = None
    tokenizer_bridge_mapped_parent_outputs_sha256: str | None = None
    tokenizer_bridge_minimum_parent_task_retention: float = 0.9
    carrier_route_publication: Path | None = None
    carrier_route_artifact_sha256: str | None = None
    carrier_span_evidence: Path | None = None
    carrier_span_evidence_file_sha256: str | None = None
    carrier_span_evidence_sha256: str | None = None
    semantic_source_tokenizer_directory: Path | None = None
    routed_manual_prompts: Path | None = None
    routed_manual_prompts_sha256: str | None = None
    routed_manual_dossier_path: Path | None = None
    routed_manual_checkpoint_directory: Path | None = None
    counted_target_total_parameters: int = (
        LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS
    )
    counted_dense_body_parameters: int = (
        LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS
    )
    carrier_expected_target_rows: int = LFM2_CARRIER_TARGET_ROWS
    carrier_expected_hidden_size: int = LFM2_CARRIER_HIDDEN_SIZE
    routed_smoke_maximum_prompt_tokens: int = 512
    routed_smoke_max_new_tokens: int = (
        LFM2_CARRIER_SMOKE_MAXIMUM_NEW_TOKENS
    )
    routed_manual_maximum_prompt_tokens: int = 8_192
    routed_manual_initial_max_new_tokens: int = (
        LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS
    )
    routed_manual_retry_max_new_tokens: int = (
        LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS
    )
    diagnostics_path: Path | None = None
    sealed_parent_cache_directory: Path | None = None
    parent_name: str = "parent"
    child_name: str = "lfm2-ffn-2552"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    # Deployment-only and deliberately excluded from the canonical fold
    # identity.  The published/countable artifact remains FP32.
    portable_runtime_device: str | None = None
    portable_runtime_dtype: str | None = None
    target_query_heads: int = DEFAULT_TARGET_QUERY_HEADS
    target_key_value_heads: int = DEFAULT_TARGET_KEY_VALUE_HEADS
    target_intermediate_size: int = DEFAULT_TARGET_INTERMEDIATE_SIZE
    exact_target_unique_parameters: int = DEFAULT_TARGET_UNIQUE_PARAMETERS
    compression_ratio: float | None = None
    dense_maximum_candidates: int = 64
    dense_minimum_hidden_size: int = DEFAULT_DENSE_MINIMUM_HIDDEN_SIZE
    required_target_hidden_size: int | None = None
    dense_heterogeneous_layout: bool = False
    dense_heterogeneous_global_hidden_frontier: bool = False
    dense_heterogeneous_factor_writer_rank_discovery: bool = False
    dense_heterogeneous_global_functional_rows_per_role: int = 2048
    dense_heterogeneous_hidden_size: int = 784
    dense_heterogeneous_target_parameters: int = 57_420_144
    dense_residual_basis_mode: str = "circuit_balanced"
    dense_body_reclosure_mode: str = "disabled"
    dense_run_closed_loop: bool = False
    dense_closed_loop_performance_metrics_diagnostic_only: bool = False
    dense_closed_loop_replay_capture_mode: str = (
        DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE
    )
    dense_closed_loop_merge_adjacent_layer_captures: bool = False
    # Execution-only, exactly like portable_runtime_device: an honoured shard
    # plan seals byte-identical replay bytes, so the device list stays out of
    # the canonical fold identity and lives in the receipt instead.
    dense_replay_devices: tuple[str, ...] = ()
    dense_replay_devices_source: str = REPLAY_DEVICES_SOURCE_UNSET
    dense_restart_capsule_root: Path | None = None
    dense_restart_capsule_keep: int = 2
    dense_resume_from_capsule: Path | None = None
    on_policy_evidence_rounds: int = 0
    on_policy_max_rounds: int = DEFAULT_ON_POLICY_MAX_ROUNDS
    # Alternation only: how many further fold passes consume the rounds this
    # pass sealed.  It never changes what a single pass solves, so it stays out
    # of the canonical fold identity (and out of the derived restart-capsule
    # scope) and is reported in the receipt instead.
    on_policy_refold_after_rounds: int = 0
    on_policy_child_artifact_directory: Path | None = None
    on_policy_fit_prompts: Path | None = None
    on_policy_manual_prompt_manifest: Path | None = None
    on_policy_round_root: Path | None = None
    on_policy_round_manifests: tuple[Path, ...] = ()
    on_policy_max_prompts: int = 64
    on_policy_max_new_tokens: int = 16
    on_policy_max_context_tokens: int = 512
    on_policy_max_capture_batches: int = 256
    on_policy_rollout_seed: int = 0
    dense_macro_cell_attractor_gate_enabled: bool = False
    dense_macro_cell_attractor_maximum_records: int = 4
    dense_macro_cell_attractor_maximum_steps: int = 8
    dense_macro_cell_attractor_maximum_entropy_regression: float = 0.02
    dense_macro_cell_attractor_maximum_repetition_regression: float = 0.05
    dense_macro_cell_attractor_maximum_dominance_regression: float = 0.05
    dense_macro_cell_attractor_maximum_eos_regression: float = 0.0
    dense_macro_cell_attractor_collapse_entropy_threshold: float = 0.10
    dense_macro_cell_attractor_collapse_repetition_threshold: float = 0.80
    dense_macro_cell_attractor_collapse_dominance_threshold: float = 0.80
    dense_macro_cell_attractor_maximum_new_collapsed_records: int = 0
    dense_attention_consumer_mode: str = "output-only"
    dense_attention_geometry_risk_enabled: bool = False
    dense_attention_score_candidate_gate_enabled: bool = False
    dense_attention_score_top_k: int = 8
    dense_attention_score_margin_floor_fraction: float = 0.5
    dense_attention_score_minimum_top1_key_agreement_fraction: float = 0.9
    dense_attention_score_minimum_top_k_pair_order_agreement_fraction: (
        float
    ) = 0.9
    dense_attention_score_minimum_winner_margin_sign_agreement_fraction: (
        float
    ) = 0.9
    dense_attention_score_minimum_relative_margin_floor_retention_fraction: (
        float
    ) = 0.9
    dense_attention_score_maximum_softmax_fisher_error: float | None = None
    dense_attention_score_maximum_js_divergence: float | None = None
    dense_vocabulary_margin_observability: bool = False
    dense_vocabulary_margin_top_k: int = 8
    dense_vocabulary_margin_floor_fraction: float = 0.05
    dense_streaming_vocabulary_margin_v2: bool = False
    dense_streaming_vocabulary_margin_pair_chunk_size: int = 256
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
    final_performance_metrics_diagnostic_only: bool = False
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
            "target_tokenizer_directory",
            "target_tokenizer_calibration",
            "tokenizer_bridge_mapped_parent_outputs",
            "carrier_route_publication",
            "carrier_span_evidence",
            "semantic_source_tokenizer_directory",
            "routed_manual_prompts",
            "routed_manual_dossier_path",
            "routed_manual_checkpoint_directory",
            "diagnostics_path",
            "sealed_parent_cache_directory",
            "derived_global_topology_cache_root",
            "balanced_basis_phase_cache_root",
            "balanced_basis_phase_event_log",
            "balanced_basis_phase_hmac_key_file",
            "dense_restart_capsule_root",
            "dense_resume_from_capsule",
            "on_policy_child_artifact_directory",
            "on_policy_fit_prompts",
            "on_policy_manual_prompt_manifest",
            "on_policy_round_root",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        object.__setattr__(
            self,
            "on_policy_round_manifests",
            tuple(Path(value) for value in self.on_policy_round_manifests),
        )
        balanced_basis_phase_values = (
            self.balanced_basis_phase_cache_root,
            self.balanced_basis_phase_event_log,
            self.balanced_basis_phase_hmac_key_file,
            self.balanced_basis_phase_hmac_key_id,
        )
        if any(value is not None for value in balanced_basis_phase_values) and not all(
            value is not None for value in balanced_basis_phase_values
        ):
            raise ProductionLfm2CatalogError(
                "Balanced-basis phase caching requires its cache root, per-run "
                "event log, HMAC key file, and HMAC key ID together."
            )
        if self.balanced_basis_phase_hmac_key_id is not None:
            key_id = str(self.balanced_basis_phase_hmac_key_id).strip()
            if _BALANCED_BASIS_PHASE_KEY_ID.fullmatch(key_id) is None:
                raise ProductionLfm2CatalogError(
                    "balanced_basis_phase_hmac_key_id must be a stable "
                    "identifier without path separators."
                )
            object.__setattr__(
                self,
                "balanced_basis_phase_hmac_key_id",
                key_id,
            )
            assert self.balanced_basis_phase_cache_root is not None
            assert self.balanced_basis_phase_event_log is not None
            assert self.balanced_basis_phase_hmac_key_file is not None
            phase_root = (
                self.balanced_basis_phase_cache_root.expanduser().absolute()
            )
            event_log = (
                self.balanced_basis_phase_event_log.expanduser().absolute()
            )
            key_file = (
                self.balanced_basis_phase_hmac_key_file.expanduser().absolute()
            )
            if event_log.is_relative_to(phase_root):
                raise ProductionLfm2CatalogError(
                    "The mutable balanced-basis phase event log must remain "
                    "outside its immutable phase-cache root."
                )
            mutable_directory_roots = (
                self.cache_root.expanduser().absolute(),
                self.snapshot_root.expanduser().absolute(),
                self.publish_destination.expanduser().absolute(),
            )
            stable_root_exclusions = (
                event_log.parent,
                *mutable_directory_roots,
            )
            if any(
                phase_root.is_relative_to(root)
                or root.is_relative_to(phase_root)
                for root in stable_root_exclusions
            ):
                raise ProductionLfm2CatalogError(
                    "The balanced-basis phase-cache root must be a stable "
                    "cross-run root outside per-run cache, snapshot, "
                    "publication, and event-log trees."
                )
            if any(
                event_log.is_relative_to(root)
                for root in mutable_directory_roots
            ):
                raise ProductionLfm2CatalogError(
                    "The balanced-basis phase event log must remain outside "
                    "cache, snapshot, and publication trees."
                )
            mutable_file_paths = {
                self.ledger_path.expanduser().absolute(),
                self.receipt_path.expanduser().absolute(),
            }
            if self.diagnostics_path is not None:
                mutable_file_paths.add(
                    self.diagnostics_path.expanduser().absolute()
                )
            if event_log in mutable_file_paths:
                raise ProductionLfm2CatalogError(
                    "The balanced-basis phase event log must have a unique "
                    "per-run path."
                )
            if event_log.exists() and not event_log.is_file():
                raise ProductionLfm2CatalogError(
                    "The balanced-basis phase event log path cannot be a "
                    "directory."
                )
            protected_roots = (
                phase_root,
                *mutable_directory_roots,
                event_log.parent,
            )
            if any(key_file.is_relative_to(root) for root in protected_roots):
                raise ProductionLfm2CatalogError(
                    "The balanced-basis phase HMAC key file must remain outside "
                    "cache, snapshot, publication, and per-run report trees."
                )
        if self.balanced_basis_existing_phase_key_sha256 is not None:
            if not self.uses_balanced_basis_phase_cache:
                raise ProductionLfm2CatalogError(
                    "Historical balanced-basis reuse requires the authenticated "
                    "balanced-basis phase cache."
                )
            object.__setattr__(
                self,
                "balanced_basis_existing_phase_key_sha256",
                _digest(
                    self.balanced_basis_existing_phase_key_sha256,
                    "balanced_basis_existing_phase_key_sha256",
                ),
            )
        bridge_values = (
            self.target_tokenizer_directory,
            self.target_tokenizer_calibration,
            self.tokenizer_bridge_artifact_sha256,
            self.tokenizer_bridge_profile_sha256,
            self.tokenizer_bridge_mapped_parent_outputs,
            self.tokenizer_bridge_mapped_parent_outputs_sha256,
        )
        if any(value is not None for value in bridge_values) and not all(
            value is not None for value in bridge_values
        ):
            raise ProductionLfm2CatalogError(
                "Cross-tokenizer folding requires an explicit target tokenizer "
                "directory, target-token calibration, bridge artifact digest, "
                "transport-profile digest, and sealed mapped-parent behavior "
                "baseline with its file digest."
            )
        if self.tokenizer_bridge_artifact_sha256 is not None:
            object.__setattr__(
                self,
                "tokenizer_bridge_artifact_sha256",
                _digest(
                    self.tokenizer_bridge_artifact_sha256,
                    "tokenizer_bridge_artifact_sha256",
                ),
            )
            object.__setattr__(
                self,
                "tokenizer_bridge_profile_sha256",
                _digest(
                    self.tokenizer_bridge_profile_sha256,
                    "tokenizer_bridge_profile_sha256",
                ),
            )
            object.__setattr__(
                self,
                "tokenizer_bridge_mapped_parent_outputs_sha256",
                _digest(
                    self.tokenizer_bridge_mapped_parent_outputs_sha256,
                    "tokenizer_bridge_mapped_parent_outputs_sha256",
                ),
            )
        carrier_values = (
            self.carrier_route_publication,
            self.carrier_route_artifact_sha256,
            self.carrier_span_evidence,
            self.carrier_span_evidence_file_sha256,
            self.carrier_span_evidence_sha256,
            self.routed_manual_prompts,
            self.routed_manual_prompts_sha256,
        )
        if any(value is not None for value in carrier_values) and not all(
            value is not None for value in carrier_values
        ):
            raise ProductionLfm2CatalogError(
                "Carrier-routed folding requires the sealed route publication, "
                "route artifact digest, semantic span evidence with both digests, "
                "and routed manual prompt file with its digest."
            )
        if self.carrier_route_publication is not None:
            for name in (
                "carrier_route_artifact_sha256",
                "carrier_span_evidence_file_sha256",
                "carrier_span_evidence_sha256",
                "routed_manual_prompts_sha256",
            ):
                object.__setattr__(
                    self,
                    name,
                    _digest(getattr(self, name), name),
                )
            if self.semantic_source_tokenizer_directory is None:
                object.__setattr__(
                    self,
                    "semantic_source_tokenizer_directory",
                    self.tokenizer_source,
                )
            if self.routed_manual_dossier_path is None:
                object.__setattr__(
                    self,
                    "routed_manual_dossier_path",
                    self.receipt_path.with_name(
                        self.receipt_path.stem + "-routed-manual-dossier.json"
                    ),
                )
            if self.routed_manual_checkpoint_directory is None:
                object.__setattr__(
                    self,
                    "routed_manual_checkpoint_directory",
                    self.receipt_path.with_name(
                        self.receipt_path.stem
                        + "-routed-manual-checkpoints"
                    ),
                )
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
            "dense_minimum_hidden_size",
            "dense_heterogeneous_hidden_size",
            "dense_heterogeneous_target_parameters",
            "counted_target_total_parameters",
            "counted_dense_body_parameters",
            "carrier_expected_target_rows",
            "carrier_expected_hidden_size",
            "routed_smoke_maximum_prompt_tokens",
            "routed_smoke_max_new_tokens",
            "routed_manual_maximum_prompt_tokens",
            "routed_manual_initial_max_new_tokens",
            "routed_manual_retry_max_new_tokens",
            "dense_heterogeneous_global_functional_rows_per_role",
            "dense_macro_cell_attractor_maximum_records",
            "dense_macro_cell_attractor_maximum_steps",
            "dense_attention_score_top_k",
            "dense_vocabulary_margin_top_k",
            "dense_streaming_vocabulary_margin_pair_chunk_size",
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
        if self.required_target_hidden_size is not None:
            object.__setattr__(
                self,
                "required_target_hidden_size",
                _positive_int(
                    self.required_target_hidden_size,
                    "required_target_hidden_size",
                ),
            )
        if self.dense_minimum_hidden_size % LFM2_DENSE_HIDDEN_SIZE_MULTIPLE:
            raise ProductionLfm2CatalogError(
                "dense_minimum_hidden_size must be a multiple of eight."
            )
        if (
            self.required_target_hidden_size is not None
            and self.required_target_hidden_size % LFM2_DENSE_HIDDEN_SIZE_MULTIPLE
        ):
            raise ProductionLfm2CatalogError(
                "required_target_hidden_size must be a multiple of eight."
            )
        for name in (
            "whole_output_maximum_new_empty_responses",
            "whole_output_maximum_new_fragment_responses",
            "dense_macro_cell_attractor_maximum_new_collapsed_records",
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
        if self.uses_balanced_basis_phase_cache and basis_mode != "circuit_balanced":
            raise ProductionLfm2CatalogError(
                "Balanced-basis phase caching requires "
                "dense_residual_basis_mode='circuit_balanced'."
            )
        body_reclosure_mode = str(self.dense_body_reclosure_mode).strip().lower()
        if body_reclosure_mode not in {"disabled", "moving_gauge_v3"}:
            raise ProductionLfm2CatalogError(
                "dense_body_reclosure_mode must be disabled or moving_gauge_v3."
            )
        if body_reclosure_mode == "moving_gauge_v3" and not self.dense_run_closed_loop:
            raise ProductionLfm2CatalogError(
                "moving_gauge_v3 body reclosure requires dense_run_closed_loop."
            )
        object.__setattr__(
            self,
            "dense_body_reclosure_mode",
            body_reclosure_mode,
        )
        attention_consumer_mode = str(
            self.dense_attention_consumer_mode
        ).strip()
        if attention_consumer_mode not in {
            "output-only",
            "coupled-value-output",
        }:
            raise ProductionLfm2CatalogError(
                "dense_attention_consumer_mode must be output-only or "
                "coupled-value-output."
            )
        object.__setattr__(
            self,
            "dense_attention_consumer_mode",
            attention_consumer_mode,
        )
        if self.dense_attention_score_top_k < 2:
            raise ProductionLfm2CatalogError(
                "dense_attention_score_top_k must be at least two."
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
        closed_loop_replay_mode = str(
            self.dense_closed_loop_replay_capture_mode
        ).strip()
        if closed_loop_replay_mode not in {
            DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE,
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
        }:
            raise ProductionLfm2CatalogError(
                "dense_closed_loop_replay_capture_mode must be "
                "'auto', 'prefix_by_prefix_canonical', or "
                "'single_pass_causal'."
            )
        object.__setattr__(
            self,
            "dense_closed_loop_replay_capture_mode",
            closed_loop_replay_mode,
        )
        if not isinstance(
            self.dense_closed_loop_merge_adjacent_layer_captures,
            bool,
        ):
            raise ProductionLfm2CatalogError(
                "dense_closed_loop_merge_adjacent_layer_captures must be "
                "boolean."
            )
        self._validate_replay_device_request()
        object.__setattr__(
            self,
            "dense_restart_capsule_keep",
            _positive_int(
                self.dense_restart_capsule_keep,
                "dense_restart_capsule_keep",
            ),
        )
        if (
            self.dense_restart_capsule_root is not None
            or self.dense_resume_from_capsule is not None
        ) and not self.dense_run_closed_loop:
            raise ProductionLfm2CatalogError(
                "Accepted-layer restart capsules only exist inside the dense "
                "closed-loop body fold; pass --dense-run-closed-loop."
            )
        if (
            self.dense_closed_loop_merge_adjacent_layer_captures
            and not self.dense_run_closed_loop
        ):
            raise ProductionLfm2CatalogError(
                "Merged adjacent layer captures require the dense closed-loop "
                "body fold."
            )
        self._validate_on_policy_evidence_configuration()
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
            "tokenizer_bridge_minimum_parent_task_retention",
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
            "dense_heterogeneous_layout",
            "dense_heterogeneous_global_hidden_frontier",
            "dense_heterogeneous_factor_writer_rank_discovery",
            "dense_run_closed_loop",
            "dense_closed_loop_performance_metrics_diagnostic_only",
            "dense_macro_cell_attractor_gate_enabled",
            "dense_attention_geometry_risk_enabled",
            "dense_attention_score_candidate_gate_enabled",
            "dense_vocabulary_margin_observability",
            "dense_streaming_vocabulary_margin_v2",
            "dense_vocabulary_margin_candidate_gate_enabled",
            "final_performance_metrics_diagnostic_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ProductionLfm2CatalogError(
                    f"{name} must be boolean."
                )
        if self.uses_carrier_routes:
            if self.required_target_hidden_size is None:
                object.__setattr__(
                    self,
                    "required_target_hidden_size",
                    self.carrier_expected_hidden_size,
                )
            elif (
                self.required_target_hidden_size
                != self.carrier_expected_hidden_size
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production requires its target-hidden ABI "
                    "constraint to equal carrier_expected_hidden_size."
                )
        if self.required_target_hidden_size is not None and not (
            self.compression_ratio is not None
            and self.dense_heterogeneous_layout
            and self.dense_heterogeneous_global_hidden_frontier
        ):
            raise ProductionLfm2CatalogError(
                "required_target_hidden_size requires the automatic global "
                "heterogeneous compression frontier."
            )
        if self.uses_carrier_routes:
            if not self.uses_tokenizer_bridge:
                raise ProductionLfm2CatalogError(
                    "Carrier-routed folding requires the sealed target tokenizer "
                    "bridge."
                )
            if not (
                self.compression_ratio is not None
                and math.isclose(
                    self.compression_ratio,
                    4.0,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                and self.dense_heterogeneous_layout
                and self.dense_heterogeneous_global_hidden_frontier
                and self.dense_heterogeneous_factor_writer_rank_discovery
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production requires the automatic global "
                    "heterogeneous factor-writer topology path."
                )
            if not (
                self.dense_run_closed_loop
                and self.dense_body_reclosure_mode == "moving_gauge_v3"
                and self.resolved_dense_closed_loop_replay_capture_mode
                == CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
                and self.dense_closed_loop_performance_metrics_diagnostic_only
                and self.dense_attention_consumer_mode
                == "coupled-value-output"
                and self.dense_attention_geometry_risk_enabled
                and self.dense_vocabulary_margin_observability
                and self.dense_streaming_vocabulary_margin_v2
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production must pin the best proven folding "
                    "lane: moving-gauge-v3 closed loop, canonical prefix replay, "
                    "diagnostic-only closed-loop metrics, coupled value/output "
                    "attention with geometry risk, and streaming-v2 vocabulary "
                    "margin observability."
                )
            if (
                self.counted_target_total_parameters
                != LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS
                or self.counted_dense_body_parameters
                != LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS
                or self.dense_heterogeneous_target_parameters
                != self.counted_dense_body_parameters
                or self.counted_target_total_parameters
                - self.counted_dense_body_parameters
                != 2 * LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production must reserve both independent "
                    "route channels inside the counted target and give only the "
                    "53,552,368-element remainder to the physical body."
                )
            if (
                self.carrier_expected_target_rows != LFM2_CARRIER_TARGET_ROWS
                or self.carrier_expected_hidden_size != LFM2_CARRIER_HIDDEN_SIZE
            ):
                raise ProductionLfm2CatalogError(
                    "The current authenticated carrier ABI requires an untied "
                    "4096-row H816 child."
                )
            if (
                self.routed_manual_initial_max_new_tokens
                != LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS
                or self.routed_manual_retry_max_new_tokens
                != LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS
                or self.routed_manual_retry_max_new_tokens
                <= self.routed_manual_initial_max_new_tokens
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier manual review must use the sealed 1024-token pass "
                    "and retry only capped rows at 2048 tokens."
                )
            if not self.final_performance_metrics_diagnostic_only:
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production requires diagnostic-only numeric "
                    "metrics; decoded intelligence review remains manual."
                )
            if any(
                (
                    self.dense_macro_cell_attractor_gate_enabled,
                    self.dense_attention_score_candidate_gate_enabled,
                    self.dense_vocabulary_margin_candidate_gate_enabled,
                )
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier-routed production does not permit numeric hard "
                    "candidate gates."
                )
            assert self.routed_manual_dossier_path is not None
            assert self.routed_manual_checkpoint_directory is not None
            publication_root = self.publish_destination.expanduser().absolute()
            dossier_path = self.routed_manual_dossier_path.expanduser().absolute()
            checkpoint_root = (
                self.routed_manual_checkpoint_directory.expanduser().absolute()
            )
            if (
                dossier_path.is_relative_to(publication_root)
                or checkpoint_root.is_relative_to(publication_root)
                or dossier_path == self.receipt_path.expanduser().absolute()
                or dossier_path == checkpoint_root
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier manual dossier and resumable checkpoints must be "
                    "distinct external paths outside the immutable published "
                    "child and main receipt."
                )
        if self.dense_heterogeneous_layout and self.compression_ratio is None:
            raise ProductionLfm2CatalogError(
                "dense_heterogeneous_layout requires an automatic dense "
                "compression_ratio request."
            )
        if (
            self.dense_heterogeneous_global_hidden_frontier
            and not self.dense_heterogeneous_layout
        ):
            raise ProductionLfm2CatalogError(
                "dense_heterogeneous_global_hidden_frontier requires the "
                "cached heterogeneous dense layout path."
            )
        if (
            self.dense_heterogeneous_global_hidden_frontier
            and self.derived_global_topology_cache_root is None
        ):
            raise ProductionLfm2CatalogError(
                "dense_heterogeneous_global_hidden_frontier requires an "
                "explicit stable derived_global_topology_cache_root."
            )
        if (
            self.dense_heterogeneous_factor_writer_rank_discovery
            and not self.dense_heterogeneous_layout
        ):
            raise ProductionLfm2CatalogError(
                "dense_heterogeneous_factor_writer_rank_discovery requires the "
                "cached heterogeneous dense layout path."
            )
        if self.dense_heterogeneous_layout:
            # Ragged physical layouts are authenticated against exact FP32
            # parameter storage.  Make the automatic mode self-contained even
            # when the generic CLI dtype default is still BF16.
            object.__setattr__(self, "dtype", "float32")
        portable_values = (
            self.portable_runtime_device,
            self.portable_runtime_dtype,
        )
        if any(value is not None for value in portable_values) and not all(
            value is not None for value in portable_values
        ):
            raise ProductionLfm2CatalogError(
                "Portable runtime relocation requires both an explicit device "
                "and an explicit FP16/BF16/FP32/FP64 dtype."
            )
        if all(value is not None for value in portable_values):
            if not self.dense_heterogeneous_layout or self.dtype != "float32":
                raise ProductionLfm2CatalogError(
                    "Portable runtime relocation is available only after the "
                    "authenticated canonical FP32 heterogeneous child fold."
                )
            assert self.portable_runtime_device is not None
            assert self.portable_runtime_dtype is not None
            try:
                portable_device = torch.device(
                    self.portable_runtime_device
                )
            except (TypeError, RuntimeError) as exc:
                raise ProductionLfm2CatalogError(
                    "Portable runtime device is invalid; no fallback is "
                    "allowed."
                ) from exc
            if portable_device.type == "cuda" and portable_device.index is None:
                raise ProductionLfm2CatalogError(
                    "Portable CUDA runtime requires an explicit accelerator "
                    "index such as cuda:0; generic cuda is not allowed."
                )
            _portable_runtime_dtype(self.portable_runtime_dtype)
        if (
            self.dense_closed_loop_replay_capture_mode
            == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
            and not (
                self.dense_heterogeneous_layout
                and self.dtype == "float32"
            )
        ):
            raise ProductionLfm2CatalogError(
                "Explicit dense single-pass closed-loop replay requires the "
                "authenticated heterogeneous dense FP32 child path. Use "
                "'auto' or 'prefix_by_prefix_canonical' for ordinary dense "
                "children."
            )
        for name in (
            "dense_macro_cell_attractor_maximum_entropy_regression",
            "dense_macro_cell_attractor_maximum_repetition_regression",
            "dense_macro_cell_attractor_maximum_dominance_regression",
            "dense_macro_cell_attractor_maximum_eos_regression",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    getattr(self, name),
                    name,
                    minimum=0.0,
                ),
            )
        for name in (
            "dense_macro_cell_attractor_collapse_entropy_threshold",
            "dense_macro_cell_attractor_collapse_repetition_threshold",
            "dense_macro_cell_attractor_collapse_dominance_threshold",
        ):
            value = _finite_float(
                getattr(self, name),
                name,
                minimum=0.0,
            )
            if value > 1.0:
                raise ProductionLfm2CatalogError(
                    f"{name} must be at most one."
                )
            object.__setattr__(self, name, value)
        for name in (
            "dense_attention_score_margin_floor_fraction",
            (
                "dense_attention_score_minimum_top1_key_"
                "agreement_fraction"
            ),
            (
                "dense_attention_score_minimum_top_k_pair_order_"
                "agreement_fraction"
            ),
            (
                "dense_attention_score_minimum_winner_margin_sign_"
                "agreement_fraction"
            ),
            (
                "dense_attention_score_minimum_relative_margin_floor_"
                "retention_fraction"
            ),
        ):
            value = _finite_float(
                getattr(self, name),
                name,
                minimum=0.0,
            )
            if value > 1.0:
                raise ProductionLfm2CatalogError(
                    f"{name} must be at most one."
                )
            object.__setattr__(self, name, value)
        if (
            self.dense_attention_score_maximum_softmax_fisher_error
            is not None
        ):
            object.__setattr__(
                self,
                "dense_attention_score_maximum_softmax_fisher_error",
                _finite_float(
                    self
                    .dense_attention_score_maximum_softmax_fisher_error,
                    (
                        "dense_attention_score_maximum_"
                        "softmax_fisher_error"
                    ),
                    minimum=0.0,
                ),
            )
        if self.dense_attention_score_maximum_js_divergence is not None:
            maximum_js = _finite_float(
                self.dense_attention_score_maximum_js_divergence,
                "dense_attention_score_maximum_js_divergence",
                minimum=0.0,
            )
            if maximum_js > 1.0:
                raise ProductionLfm2CatalogError(
                    "dense_attention_score_maximum_js_divergence must be "
                    "at most one."
                )
            object.__setattr__(
                self,
                "dense_attention_score_maximum_js_divergence",
                maximum_js,
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
            self.dense_macro_cell_attractor_gate_enabled
            and (
                self.compression_ratio is None
                or not self.dense_run_closed_loop
            )
        ):
            raise ProductionLfm2CatalogError(
                "dense macro-cell attractor gating requires an automatic "
                "compression_ratio request with dense_run_closed_loop=True."
            )
        if (
            self.dense_closed_loop_performance_metrics_diagnostic_only
            and (
                self.compression_ratio is None
                or not self.dense_run_closed_loop
            )
        ):
            raise ProductionLfm2CatalogError(
                "dense closed-loop performance diagnostic-only policy "
                "requires an automatic compression_ratio request with "
                "dense_run_closed_loop=True."
            )
        if (
            self.dense_attention_geometry_risk_enabled
            and not self.dense_heterogeneous_layout
        ):
            raise ProductionLfm2CatalogError(
                "dense attention-geometry allocation requires the exact-budget "
                "heterogeneous physical-layout path."
            )
        if (
            self.dense_attention_score_candidate_gate_enabled
            and (
                self.compression_ratio is None
                or not self.dense_run_closed_loop
            )
        ):
            raise ProductionLfm2CatalogError(
                "dense attention-score candidate gating requires an "
                "automatic compression_ratio request with "
                "dense_run_closed_loop=True."
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
            self.dense_streaming_vocabulary_margin_v2
            and not self.dense_vocabulary_margin_observability
        ):
            raise ProductionLfm2CatalogError(
                "dense streaming vocabulary-margin v2 requires "
                "dense_vocabulary_margin_observability."
            )
        if (
            self.dense_streaming_vocabulary_margin_v2
            and self.sealed_parent_cache_directory is None
        ):
            raise ProductionLfm2CatalogError(
                "dense streaming vocabulary-margin v2 requires an existing "
                "sealed parent cache and selective authenticated session."
            )
        if (
            self.dense_vocabulary_margin_candidate_gate_enabled
            and not self.dense_vocabulary_margin_observability
        ):
            raise ProductionLfm2CatalogError(
                "dense vocabulary-margin candidate gating requires "
                "dense_vocabulary_margin_observability."
            )
        if self.target_tokenizer_directory is not None:
            if self.compression_ratio is None:
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer folding currently requires an automatic "
                    "dense compression-ratio request."
                )
            if self.sealed_parent_cache_directory is None:
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer folding requires an existing fully sealed "
                    "parent cache over the bridge-mapped donor IDs."
                )
            if self.dense_macro_cell_attractor_gate_enabled:
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer macro-attractor replay is not target-token "
                    "aware; use the sealed bridged closed loop and external "
                    "whole-output evaluation."
                )
            if (
                self.dense_vocabulary_margin_observability
                and not self.dense_streaming_vocabulary_margin_v2
            ):
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer vocabulary-margin observability requires "
                    "streaming v2 with the fully authenticated target bridge "
                    "adapter; the legacy donor-domain path is rejected."
                )
            if self.dense_vocabulary_margin_candidate_gate_enabled:
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer vocabulary-margin candidate gating remains "
                    "disabled; use target-domain streaming-v2 observability and "
                    "manual whole-response evaluation."
                )
            if self.hybrid_model_fold or self.producer_first_model_fold:
                raise ProductionLfm2CatalogError(
                    "Cross-tokenizer folding cannot use explicit hybrid or "
                    "producer-first target modes."
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

    def _validate_replay_device_request(self) -> None:
        """Normalize the opt-in multi-device child-replay shard request.

        The device list is execution-only: an honoured shard plan seals the
        same bytes as the serial default, so this validates the *request*
        shape only.  Whether a request can actually be honoured depends on the
        live child's device and the visible cards, is decided per capture
        point inside the adapter, and falls back to serial with a recorded
        reason rather than failing the run.
        """

        source = str(self.dense_replay_devices_source).strip()
        if source not in REPLAY_DEVICES_SOURCES:
            raise ProductionLfm2CatalogError(
                "dense_replay_devices_source must be one of "
                f"{list(REPLAY_DEVICES_SOURCES)}."
            )
        devices = self.dense_replay_devices
        if isinstance(devices, (str, bytes)) or not isinstance(
            devices,
            Sequence,
        ):
            raise ProductionLfm2CatalogError(
                "dense_replay_devices must be a sequence of device strings."
            )
        normalized = tuple(str(value).strip() for value in devices)
        if any(not value for value in normalized):
            raise ProductionLfm2CatalogError(
                "dense_replay_devices entries must be nonempty device strings."
            )
        if REPLAY_DEVICES_ALL_TOKEN in normalized and normalized != (
            REPLAY_DEVICES_ALL_TOKEN,
        ):
            raise ProductionLfm2CatalogError(
                f"dense_replay_devices {REPLAY_DEVICES_ALL_TOKEN!r} names "
                "every visible identical device and cannot be combined with "
                "explicit devices."
            )
        if normalized != (REPLAY_DEVICES_ALL_TOKEN,) and any(
            _REPLAY_DEVICE_PATTERN.match(value) is None for value in normalized
        ):
            raise ProductionLfm2CatalogError(
                "dense_replay_devices entries must be 'cpu', 'cuda', "
                f"'cuda:<index>', or the single token "
                f"{REPLAY_DEVICES_ALL_TOKEN!r}."
            )
        if bool(normalized) != (
            source
            in {REPLAY_DEVICES_SOURCE_CLI, REPLAY_DEVICES_SOURCE_ENVIRONMENT}
        ):
            raise ProductionLfm2CatalogError(
                "dense_replay_devices and dense_replay_devices_source "
                "disagree about whether a shard request exists."
            )
        if normalized and not self.dense_run_closed_loop:
            raise ProductionLfm2CatalogError(
                "Multi-device child-replay sharding only applies to dense "
                "closed-loop captures; pass --dense-run-closed-loop."
            )
        object.__setattr__(self, "dense_replay_devices", normalized)
        object.__setattr__(self, "dense_replay_devices_source", source)

    def _validate_on_policy_evidence_configuration(self) -> None:
        """Fail closed on an unusable bounded on-policy extension request."""

        rounds = _nonnegative_int(
            self.on_policy_evidence_rounds,
            "on_policy_evidence_rounds",
        )
        maximum = _positive_int(
            self.on_policy_max_rounds,
            "on_policy_max_rounds",
        )
        object.__setattr__(self, "on_policy_evidence_rounds", rounds)
        object.__setattr__(self, "on_policy_max_rounds", maximum)
        for name in (
            "on_policy_max_prompts",
            "on_policy_max_new_tokens",
            "on_policy_max_context_tokens",
            "on_policy_max_capture_batches",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "on_policy_rollout_seed",
            _nonnegative_int(
                self.on_policy_rollout_seed,
                "on_policy_rollout_seed",
            ),
        )
        if self.on_policy_max_context_tokens <= self.on_policy_max_new_tokens:
            raise ProductionLfm2CatalogError(
                "on_policy_max_context_tokens must exceed "
                "on_policy_max_new_tokens."
            )
        if rounds > maximum:
            raise ProductionLfm2CatalogError(
                "on_policy_evidence_rounds exceeds the bounded "
                "on_policy_max_rounds budget."
            )
        if len(self.on_policy_round_manifests) + rounds > maximum:
            raise ProductionLfm2CatalogError(
                "Consumed on-policy round manifests plus requested extension "
                "rounds exceed the bounded on_policy_max_rounds budget."
            )
        refolds = _nonnegative_int(
            self.on_policy_refold_after_rounds,
            "on_policy_refold_after_rounds",
        )
        object.__setattr__(self, "on_policy_refold_after_rounds", refolds)
        if refolds:
            if rounds == 0:
                raise ProductionLfm2CatalogError(
                    "A fold/round alternation needs at least one new round per "
                    "pass; pass --on-policy-evidence-rounds."
                )
            if self.on_policy_child_artifact_directory is not None:
                # Every round in an alternation reloads the child its own pass
                # just published; a pinned foreign child is not on-policy.
                raise ProductionLfm2CatalogError(
                    "A fold/round alternation always reloads the child each "
                    "pass published; drop "
                    "--on-policy-child-artifact-directory."
                )
            # Every pass except the last one runs ``rounds`` rounds, so the
            # alternation executes exactly ``rounds * refolds`` new rounds and
            # the bounded budget still holds end to end.
            if (
                len(self.on_policy_round_manifests) + rounds * refolds
                > maximum
            ):
                raise ProductionLfm2CatalogError(
                    "The requested fold/round alternation exceeds the bounded "
                    "on_policy_max_rounds budget."
                )
        if (self.on_policy_round_manifests or refolds) and (
            (reason := _unconsumed_extension_evidence_reason(self)) is not None
        ):
            raise ProductionLfm2CatalogError(
                "Consuming sealed on-policy extension evidence is not "
                f"supported on this lane: {reason}. A base-only solve under a "
                "base+extension request would be silently wrong, so the "
                "configuration fails closed."
            )
        if rounds == 0 and not self.on_policy_round_manifests:
            return
        if self.sealed_parent_cache_directory is None:
            # An extension round extends an already sealed base corpus.  The
            # legacy calibration build mode is not a valid base.
            raise ProductionLfm2CatalogError(
                "On-policy evidence rounds require an already sealed parent "
                "cache; pass --sealed-parent-cache-directory."
            )
        if rounds > 0:
            if self.on_policy_fit_prompts is None:
                raise ProductionLfm2CatalogError(
                    "On-policy evidence rounds require --on-policy-fit-prompts."
                )
            if self.on_policy_manual_prompt_manifest is None:
                # A round without a sealed disjointness manifest is rejected at
                # configuration time, never at run time.
                raise ProductionLfm2CatalogError(
                    "On-policy evidence rounds require "
                    "--on-policy-manual-prompt-manifest so fit prompts are "
                    "provably disjoint from the reserved review corpus."
                )
        round_root = self.on_policy_round_root_directory
        reserved = {
            "publication destination": (
                self.publish_destination.expanduser().absolute()
            ),
            "ledger path": self.ledger_path.expanduser().absolute(),
            "receipt path": self.receipt_path.expanduser().absolute(),
        }
        for label, path in reserved.items():
            if round_root == path or round_root.is_relative_to(path):
                raise ProductionLfm2CatalogError(
                    "The on-policy evidence round root must stay disjoint from "
                    f"the {label}."
                )

    @property
    def tokenizer_source(self) -> Path:
        return (
            self.parent_model
            if self.tokenizer_directory is None
            else self.tokenizer_directory
        )

    @property
    def resolved_dense_closed_loop_replay_capture_mode(self) -> str:
        """Resolve the child replay contract used by dense closed-loop folding.

        ``auto`` is canonical prefix-by-prefix replay for every child.  The
        heterogeneous FP32 single-pass lane certifies each capture against a
        canonical reference replay inside the loop, so it costs a second replay
        per capture point; it stays available as an explicit opt-in and is never
        selected automatically.
        """

        policy = self.dense_closed_loop_replay_capture_mode
        if policy == DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE:
            return CHILD_REPLAY_PREFIX_BY_PREFIX_MODE
        return policy

    @property
    def dense_restart_capsule_directory(self) -> Path | None:
        """Resolve where accepted-layer restart capsules are written.

        Capsules are per-run restart state, so the derived default is scoped by
        the run identity: two different configurations sharing one cache root
        never publish into the same capsule chain.  A resume always names its
        root explicitly.
        """

        if not self.dense_run_closed_loop:
            return None
        if self.dense_restart_capsule_root is not None:
            return self.dense_restart_capsule_root.expanduser().absolute()
        # The scope deliberately excludes the restart-capsule policy itself so
        # that a killed run and its resume, which differ only by the resume
        # flag, land in the same capsule chain.
        document = self.identity()
        target = dict(document["target"])
        target.pop("dense_restart_capsule", None)
        run_scope = canonical_sha256({**document, "target": target})[:16]
        return (
            self.cache_root.expanduser().absolute()
            / "closed-loop-restart-capsules"
            / f"{self.child_name}-{run_scope}"
        )

    @property
    def uses_on_policy_evidence(self) -> bool:
        return bool(
            self.on_policy_evidence_rounds > 0 or self.on_policy_round_manifests
        )

    @property
    def on_policy_round_root_directory(self) -> Path:
        if self.on_policy_round_root is not None:
            return self.on_policy_round_root.expanduser().absolute()
        return (
            self.cache_root.expanduser().absolute() / "on-policy-rounds"
        )

    @property
    def uses_tokenizer_bridge(self) -> bool:
        return self.target_tokenizer_directory is not None

    @property
    def uses_carrier_routes(self) -> bool:
        return self.carrier_route_publication is not None

    @property
    def uses_portable_runtime_relocation(self) -> bool:
        return (
            self.portable_runtime_device is not None
            and self.portable_runtime_dtype is not None
        )

    @property
    def child_tokenizer_source(self) -> Path:
        return (
            self.tokenizer_source
            if self.target_tokenizer_directory is None
            else self.target_tokenizer_directory
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
    def global_topology_phase_cache_root(self) -> Path:
        root = self.derived_global_topology_cache_root
        if root is None:
            root = self.cache_root / "derived-global-topology-phases"
        return root.absolute()

    @property
    def uses_balanced_basis_phase_cache(self) -> bool:
        return self.balanced_basis_phase_cache_root is not None

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
            "target_tokenizer_directory": str(self.child_tokenizer_source),
            "target_tokenizer_calibration": (
                None
                if self.target_tokenizer_calibration is None
                else str(self.target_tokenizer_calibration)
            ),
            "tokenizer_bridge": {
                "enabled": self.uses_tokenizer_bridge,
                "artifact_sha256": self.tokenizer_bridge_artifact_sha256,
                "transport_profile_sha256": (
                    self.tokenizer_bridge_profile_sha256
                ),
                "mapped_parent_outputs": (
                    None
                    if self.tokenizer_bridge_mapped_parent_outputs is None
                    else str(self.tokenizer_bridge_mapped_parent_outputs)
                ),
                "mapped_parent_outputs_sha256": (
                    self.tokenizer_bridge_mapped_parent_outputs_sha256
                ),
                "minimum_parent_task_retention": (
                    self.tokenizer_bridge_minimum_parent_task_retention
                ),
                "trace_parent_input_mode": (
                    "target-tokenizer-canonical-sequence-mapped-one-row-to-donor-ids"
                ),
                "trace_donor_canonical_sequence_equivalence_claimed": False,
                "whole_output_original_parent_mode": (
                    "donor-tokenizer-canonical-prompt-and-generation"
                ),
                "whole_output_mapped_parent_mode": (
                    "target-tokenizer-canonical-prompt-mapped-to-donor-ids"
                ),
                "explicit_verified_artifact_required": True,
                "numeric_token_id_equality_required": False,
            },
            "carrier_routes": {
                "enabled": self.uses_carrier_routes,
                "publication": (
                    None
                    if self.carrier_route_publication is None
                    else str(self.carrier_route_publication)
                ),
                "artifact_sha256": self.carrier_route_artifact_sha256,
                "span_evidence": (
                    None
                    if self.carrier_span_evidence is None
                    else str(self.carrier_span_evidence)
                ),
                "span_evidence_file_sha256": (
                    self.carrier_span_evidence_file_sha256
                ),
                "span_evidence_sha256": self.carrier_span_evidence_sha256,
                "semantic_source_tokenizer_directory": (
                    None
                    if self.semantic_source_tokenizer_directory is None
                    else str(self.semantic_source_tokenizer_directory)
                ),
                "manual_prompts": (
                    None
                    if self.routed_manual_prompts is None
                    else str(self.routed_manual_prompts)
                ),
                "manual_prompts_sha256": self.routed_manual_prompts_sha256,
                "manual_dossier_path": (
                    None
                    if self.routed_manual_dossier_path is None
                    else str(self.routed_manual_dossier_path)
                ),
                "manual_checkpoint_directory": (
                    None
                    if self.routed_manual_checkpoint_directory is None
                    else str(self.routed_manual_checkpoint_directory)
                ),
                "counted_target_total_parameters": (
                    self.counted_target_total_parameters
                ),
                "counted_dense_body_parameters": (
                    self.counted_dense_body_parameters
                ),
                "continuous_route_parameters": (
                    self.counted_target_total_parameters
                    - self.counted_dense_body_parameters
                ),
                "expected_target_rows": self.carrier_expected_target_rows,
                "expected_hidden_size": self.carrier_expected_hidden_size,
                "smoke_maximum_prompt_tokens": (
                    self.routed_smoke_maximum_prompt_tokens
                ),
                "smoke_max_new_tokens": self.routed_smoke_max_new_tokens,
                "manual_maximum_prompt_tokens": (
                    self.routed_manual_maximum_prompt_tokens
                ),
                "manual_initial_max_new_tokens": (
                    self.routed_manual_initial_max_new_tokens
                ),
                "manual_retry_max_new_tokens": (
                    self.routed_manual_retry_max_new_tokens
                ),
                "manual_retry_scope": "only-output-cap-censored-rows",
                "output_cap_hits_are_censored": True,
                "remaining_output_cap_censored_endpoint": "unknown",
                "remaining_output_cap_censored_is_failure": False,
                "safety_behavior_is_separate_from_folding_quality": True,
                "safety_prompts_excluded_from_intelligence_failure": True,
                "automatic_intelligence_verdict": False,
                "manual_whole_response_intelligence_review_is_authoritative": True,
                "numeric_metrics_are_diagnostic_only": (
                    self.final_performance_metrics_diagnostic_only
                ),
                "numeric_metrics_have_acceptance_authority": False,
                "source_table_packaged_or_installed": False,
                "fresh_reload_external_tokenizer_path_required": False,
            },
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
            "derived_global_topology_cache_root": str(
                self.global_topology_phase_cache_root
            ),
            **(
                {}
                if not self.uses_balanced_basis_phase_cache
                else {
                    "balanced_basis_phase_cache": {
                        "enabled": True,
                        "root": str(
                            self.balanced_basis_phase_cache_root
                            .expanduser()
                            .absolute()
                        ),
                        "event_log": str(
                            self.balanced_basis_phase_event_log
                            .expanduser()
                            .absolute()
                        ),
                        "hmac_key_id": self.balanced_basis_phase_hmac_key_id,
                        "hmac_key_runtime_supplied": True,
                        **(
                            {}
                            if self.balanced_basis_existing_phase_key_sha256
                            is None
                            else {
                                "existing_phase_reuse_key_sha256": (
                                    self
                                    .balanced_basis_existing_phase_key_sha256
                                )
                            }
                        ),
                    }
                }
            ),
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
                "dense_minimum_hidden_size": (
                    self.dense_minimum_hidden_size
                ),
                "required_target_hidden_size": self.required_target_hidden_size,
                "dense_heterogeneous_layout": (
                    self.dense_heterogeneous_layout
                ),
                "dense_heterogeneous_global_hidden_frontier": (
                    self.dense_heterogeneous_global_hidden_frontier
                ),
                "dense_heterogeneous_factor_writer_rank_discovery": (
                    self.dense_heterogeneous_factor_writer_rank_discovery
                ),
                "dense_heterogeneous_global_functional_rows_per_role": (
                    self.dense_heterogeneous_global_functional_rows_per_role
                ),
                "dense_heterogeneous_hidden_size": (
                    self.dense_heterogeneous_hidden_size
                ),
                "dense_heterogeneous_target_parameters": (
                    self.dense_heterogeneous_target_parameters
                ),
                "dense_residual_basis_mode": (
                    self.dense_residual_basis_mode
                ),
                "dense_body_reclosure_mode": self.dense_body_reclosure_mode,
                "dense_run_closed_loop": self.dense_run_closed_loop,
                "dense_closed_loop_performance_metrics": {
                    "diagnostic_only": (
                        self
                        .dense_closed_loop_performance_metrics_diagnostic_only
                    ),
                    "numeric_decision_authority": (
                        not self
                        .dense_closed_loop_performance_metrics_diagnostic_only
                    ),
                    "counterfactual_thresholds_recorded": True,
                    "nonfinite_values_reject": True,
                    "structural_and_integrity_failures_reject": True,
                },
                "dense_closed_loop_replay": {
                    "policy": (
                        self.dense_closed_loop_replay_capture_mode
                    ),
                    "resolved_mode": (
                        self
                        .resolved_dense_closed_loop_replay_capture_mode
                    ),
                    "single_pass_requires_authenticated_heterogeneous_fp32": (
                        True
                    ),
                    "automatic_selection_is_canonical_prefix_replay": True,
                    "single_pass_certification_is_explicit_opt_in": True,
                    "merge_adjacent_layer_captures": (
                        self.dense_closed_loop_merge_adjacent_layer_captures
                    ),
                },
                "dense_restart_capsule": {
                    "explicit_root": (
                        None
                        if self.dense_restart_capsule_root is None
                        else str(
                            self.dense_restart_capsule_root
                            .expanduser()
                            .absolute()
                        )
                    ),
                    "rotation_keep": self.dense_restart_capsule_keep,
                    "resume_requested": (
                        self.dense_resume_from_capsule is not None
                    ),
                },
                "on_policy_evidence": {
                    "rounds": self.on_policy_evidence_rounds,
                    "max_rounds": self.on_policy_max_rounds,
                    "consumed_round_manifest_count": len(
                        self.on_policy_round_manifests
                    ),
                    "consumed_round_manifests": [
                        str(value.expanduser().absolute())
                        for value in self.on_policy_round_manifests
                    ],
                    "rollout": {
                        "max_prompts": self.on_policy_max_prompts,
                        "max_new_tokens": self.on_policy_max_new_tokens,
                        "max_context_tokens": (
                            self.on_policy_max_context_tokens
                        ),
                        "max_capture_batches": (
                            self.on_policy_max_capture_batches
                        ),
                        "seed": self.on_policy_rollout_seed,
                    },
                    "all_rows_are_fit_rows": True,
                    "review_rows_are_never_on_policy": True,
                },
                "dense_macro_cell_attractor_gate": {
                    "enabled": (
                        self.dense_macro_cell_attractor_gate_enabled
                    ),
                    "maximum_records": (
                        self.dense_macro_cell_attractor_maximum_records
                    ),
                    "maximum_steps": (
                        self.dense_macro_cell_attractor_maximum_steps
                    ),
                    "maximum_entropy_regression": (
                        self
                        .dense_macro_cell_attractor_maximum_entropy_regression
                    ),
                    "maximum_repetition_regression": (
                        self
                        .dense_macro_cell_attractor_maximum_repetition_regression
                    ),
                    "maximum_dominance_regression": (
                        self
                        .dense_macro_cell_attractor_maximum_dominance_regression
                    ),
                    "maximum_eos_regression": (
                        self
                        .dense_macro_cell_attractor_maximum_eos_regression
                    ),
                    "collapse_entropy_threshold": (
                        self
                        .dense_macro_cell_attractor_collapse_entropy_threshold
                    ),
                    "collapse_repetition_threshold": (
                        self
                        .dense_macro_cell_attractor_collapse_repetition_threshold
                    ),
                    "collapse_dominance_threshold": (
                        self
                        .dense_macro_cell_attractor_collapse_dominance_threshold
                    ),
                    "maximum_new_collapsed_records": (
                        self
                        .dense_macro_cell_attractor_maximum_new_collapsed_records
                    ),
                    "exact_token_match_is_diagnostic_only": True,
                },
                "dense_attention_consumer_mode": (
                    self.dense_attention_consumer_mode
                ),
                "dense_attention_geometry_allocator": {
                    "enabled": self.dense_attention_geometry_risk_enabled,
                    "method": ATTENTION_GEOMETRY_RISK_METHOD,
                    "all_authenticated_causal_rows": True,
                    "masked_qk_scores": True,
                    "key_value_memory": True,
                    "head_context": True,
                    "residual_pressure_coupled": True,
                    "behavioral_hard_gate": False,
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                },
                "dense_attention_score_candidate_gate": {
                    "enabled": (
                        self.dense_attention_score_candidate_gate_enabled
                    ),
                    "top_k": self.dense_attention_score_top_k,
                    "margin_floor_fraction": (
                        self.dense_attention_score_margin_floor_fraction
                    ),
                    "minimum_top1_key_agreement_fraction": (
                        self
                        .dense_attention_score_minimum_top1_key_agreement_fraction
                    ),
                    "minimum_top_k_pair_order_agreement_fraction": (
                        self
                        .dense_attention_score_minimum_top_k_pair_order_agreement_fraction
                    ),
                    "minimum_winner_margin_sign_agreement_fraction": (
                        self
                        .dense_attention_score_minimum_winner_margin_sign_agreement_fraction
                    ),
                    "minimum_relative_margin_floor_retention_fraction": (
                        self
                        .dense_attention_score_minimum_relative_margin_floor_retention_fraction
                    ),
                    "maximum_softmax_fisher_error": (
                        self
                        .dense_attention_score_maximum_softmax_fisher_error
                    ),
                    "maximum_js_divergence": (
                        self.dense_attention_score_maximum_js_divergence
                    ),
                    "numeric_weight_matching_used": False,
                    "parent_forward_calls": 0,
                    "training_steps": 0,
                },
                "dense_vocabulary_margin_observability": (
                    self.dense_vocabulary_margin_observability
                ),
                "dense_vocabulary_margin_top_k": (
                    self.dense_vocabulary_margin_top_k
                ),
                "dense_vocabulary_margin_floor_fraction": (
                    self.dense_vocabulary_margin_floor_fraction
                ),
                "dense_streaming_vocabulary_margin_v2": (
                    self.dense_streaming_vocabulary_margin_v2
                ),
                "dense_streaming_vocabulary_margin_pair_chunk_size": (
                    self.dense_streaming_vocabulary_margin_pair_chunk_size
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
                "performance_metrics_diagnostic_only": (
                    self.final_performance_metrics_diagnostic_only
                ),
                "numeric_decision_authority": (
                    not self.final_performance_metrics_diagnostic_only
                ),
                "manual_intelligence_review_required": (
                    self.final_performance_metrics_diagnostic_only
                ),
                "output_cap_hits_are_censored": True,
                "safety_behavior_is_separate_from_folding_quality": True,
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
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None
    # On-policy evidence extension.  ``parent_trace_regeneration_forward_count``
    # deliberately stays 0 here: on-policy capture is not trace regeneration and
    # is accounted separately so the "no parent forwards after the sealed cache"
    # audit keeps its exact meaning.
    on_policy_extension_cache_paths: tuple[Path, ...] = ()
    on_policy_extension_cache_keys: tuple[str, ...] = ()
    on_policy_extension_batch_counts: tuple[int, ...] = ()
    on_policy_round_manifest_sha256: tuple[str, ...] = ()
    on_policy_round_count: int = 0
    on_policy_parent_capture_invocations: int = 0
    on_policy_fit_prompts_disjoint_from_manual_review: bool | None = None
    # The authenticated rounds themselves, in consumption order.  Only the
    # bridged lane reads them, for the sealed target-token preimages that make
    # an extension batch replayable by a target-vocabulary child.
    on_policy_rounds: tuple[Lfm2OnPolicyEvidenceRound, ...] = ()
    # The ordered ``(base, *extensions)`` aggregate a solve pass consumes.  It
    # owns the extension selective sessions; the base session above stays owned
    # by this resolution so its existing cleanup keeps closing it exactly once.
    evidence_set: Lfm2SealedEvidenceSet | None = None
    on_policy_extension_split_sha256: tuple[str, ...] = ()

    @property
    def on_policy_evidence_identity_sha256(self) -> str | None:
        """Digest every extension cache and round manifest one solve binds to.

        A closed-form solve fitted on base-only evidence must never be reused
        for a base+extension request, so this digest is folded into the derived
        phase-cache keys and into the sealed evidence-split identity.
        """

        return lfm2_on_policy_evidence_identity_sha256(
            base_cache_key=self.cache.cache_key,
            extension_cache_keys=self.on_policy_extension_cache_keys,
            extension_batch_counts=self.on_policy_extension_batch_counts,
            round_manifest_sha256=self.on_policy_round_manifest_sha256,
        )

    @property
    def consumes_on_policy_extension_evidence(self) -> bool:
        """Whether this solve pass fits on base **and** extension evidence."""

        return bool(
            self.evidence_set is not None
            and self.evidence_set.has_extension_evidence
        )

    def on_policy_evidence_receipt(
        self,
        *,
        max_rounds: int,
        executed_rounds: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        """Report consumed and freshly executed on-policy evidence rounds."""

        executed = [dict(value) for value in executed_rounds]
        return {
            "round_count": self.on_policy_round_count,
            "max_rounds": int(max_rounds),
            "round_manifest_sha256": list(
                self.on_policy_round_manifest_sha256
            ),
            "extension_cache_keys": list(
                self.on_policy_extension_cache_keys
            ),
            "extension_batch_counts": [
                int(value)
                for value in self.on_policy_extension_batch_counts
            ],
            "extension_cache_paths": [
                str(value) for value in self.on_policy_extension_cache_paths
            ],
            "evidence_identity_sha256": (
                self.on_policy_evidence_identity_sha256
            ),
            "parent_capture_invocations": (
                self.on_policy_parent_capture_invocations
                + sum(1 for _value in executed)
            ),
            "parent_trace_regeneration_forward_count": (
                self.parent_trace_regeneration_forward_count
            ),
            "fit_prompts_disjoint_from_manual_review": (
                self.on_policy_fit_prompts_disjoint_from_manual_review
                if not executed
                else bool(
                    (
                        self.on_policy_fit_prompts_disjoint_from_manual_review
                        is not False
                    )
                    and all(
                        value["fit_prompts_disjoint_from_manual_review"]
                        for value in executed
                    )
                )
            ),
            "postpublication_rounds_executed": executed,
            "postpublication_round_count": len(executed),
            "all_rows_are_fit_rows": True,
            "consumed_by_this_solve_pass": bool(
                self.on_policy_round_manifest_sha256
            ),
            "extension_split_sha256": list(
                self.on_policy_extension_split_sha256
            ),
            "evidence_set": (
                None
                if self.evidence_set is None
                else self.evidence_set.receipt_fields()
            ),
        }

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
            # On-policy capture is not trace regeneration.  It is counted here
            # so the "no parent forwards after the sealed cache" audit above
            # keeps its exact pre-extension meaning.
            "on_policy_parent_capture_invocations": (
                self.on_policy_parent_capture_invocations
            ),
            "on_policy_round_count": self.on_policy_round_count,
            "parent_cache_selective_session_reused": (
                self.selective_session is not None
            ),
            "parent_cache_selective_session_sha256": (
                None
                if self.selective_session is None
                else self.selective_session.evidence_sha256
            ),
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
    *,
    all_rows_are_fit: bool = False,
    on_policy_evidence_identity_sha256: str | None = None,
) -> tuple[tuple[str, ...], str]:
    """Derive selection/fit/validation splits for one sealed corpus.

    ``all_rows_are_fit`` is a hard rule, not a convention: every row an
    on-policy extension cache contributes was visited by the child, so letting
    one become a ``selection`` row would let the child pick its own frontier and
    letting one become a ``validation`` row would stop that split being held
    out.  Extension corpora are therefore always all-fit.
    """

    manifest = cache.manifest
    ordered_inputs = tuple(
        str(value)
        for value in manifest["corpus"]["ordered_batch_input_sha256"]
    )
    if len(ordered_inputs) != len(cache):
        raise ProductionLfm2CatalogError(
            "Sealed parent cache corpus inventory is inconsistent."
        )
    if all_rows_are_fit:
        if not ordered_inputs:
            raise ProductionLfm2CatalogError(
                "An on-policy extension cache must contain at least one batch."
            )
        result = tuple("fit" for _ in ordered_inputs)
        assignment = "on_policy_extension_all_fit_v1"
    else:
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
        assignment = "content_digest_rank_extrema_v1"
    identity = {
        "schema": SEALED_CACHE_SPLIT_SCHEMA,
        "parent_cache_key": cache.cache_key,
        "corpus_fingerprint": cache.corpus_fingerprint,
        "ordered_batch_input_sha256": list(ordered_inputs),
        "ordered_splits": list(result),
        "assignment": assignment,
        # Present only once extension evidence exists so the base-only split
        # digest is byte-identical to the pre-extension revision.
        **(
            {}
            if on_policy_evidence_identity_sha256 is None
            else {
                "on_policy_evidence_identity_sha256": (
                    on_policy_evidence_identity_sha256
                ),
                "all_on_policy_rows_are_fit_rows": True,
            }
        ),
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
    if input_weight.ndim < 1 or output_weight.ndim < 1:
        raise ProductionLfm2CatalogError(
            "The sealed-cache parent tokenizer-facing embeddings have no "
            "vocabulary axis."
        )
    model_vocabulary_sizes = {
        config_vocab_size,
        int(input_weight.shape[0]),
        int(output_weight.shape[0]),
    }
    if (
        tokenizer_vocab_size <= 0
        or config_vocab_size <= 0
        or len(model_vocabulary_sizes) != 1
    ):
        raise ProductionLfm2CatalogError(
            "Parent config, input embeddings, and output embeddings do not "
            "describe one tokenizer-facing row capacity."
        )
    model_vocabulary_capacity = next(iter(model_vocabulary_sizes))
    if tokenizer_vocab_size > model_vocabulary_capacity:
        raise ProductionLfm2CatalogError(
            "Tokenizer vocabulary exceeds the parent model's tokenizer-facing "
            "row capacity."
        )

    # Many production checkpoints deliberately pad the embedding/output
    # matrices beyond the tokenizer's effective vocabulary for hardware
    # alignment.  LFM2.5-230M is one such checkpoint (64,402 tokenizer IDs in
    # 65,536 model rows).  Equality with len(tokenizer) is therefore not a
    # valid identity check.  The authenticated model/config row capacity must
    # agree internally, and every tokenizer/cache ID must fit that capacity.
    tokenizer_id_ceiling = tokenizer_vocab_size
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocabulary = get_vocab()
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                "The tokenizer could not expose its token-ID inventory."
            ) from exc
        if not isinstance(vocabulary, Mapping) or not vocabulary:
            raise ProductionLfm2CatalogError(
                "The tokenizer token-ID inventory is empty or invalid."
            )
        tokenizer_ids = tuple(vocabulary.values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in tokenizer_ids
        ):
            raise ProductionLfm2CatalogError(
                "The tokenizer token-ID inventory contains invalid IDs."
            )
        tokenizer_id_ceiling = max(tokenizer_ids) + 1
        if tokenizer_id_ceiling > model_vocabulary_capacity:
            raise ProductionLfm2CatalogError(
                "Tokenizer IDs exceed the parent model's tokenizer-facing "
                "row capacity."
            )
    for batch_index, batch in enumerate(calibration_batches):
        for field in ("input_ids", "forced_decode_ids"):
            values = _calibration_tensor(batch, field)
            if values is None or values.numel() == 0:
                raise ProductionLfm2CatalogError(
                    f"Sealed parent cache batch {batch_index} lacks {field}."
                )
            minimum_id = int(values.min())
            maximum_id = int(values.max())
            if minimum_id < 0 or maximum_id >= model_vocabulary_capacity:
                raise ProductionLfm2CatalogError(
                    f"Sealed parent cache batch {batch_index} {field} is outside "
                    "the parent model's tokenizer-facing row capacity."
                )
            if maximum_id >= tokenizer_id_ceiling:
                raise ProductionLfm2CatalogError(
                    f"Sealed parent cache batch {batch_index} {field} is outside "
                    "the tokenizer's effective token-ID range."
                )


def _on_policy_split_identity_fields(
    on_policy_evidence_identity_sha256: str | None,
) -> dict[str, Any]:
    """Bind extension evidence into a split digest only once it exists."""

    if on_policy_evidence_identity_sha256 is None:
        return {}
    return {
        "on_policy_evidence_identity_sha256": (
            on_policy_evidence_identity_sha256
        ),
        "all_on_policy_rows_are_fit_rows": True,
    }


@dataclass(frozen=True, slots=True)
class _ResolvedOnPolicyEvidence:
    """Already sealed on-policy rounds this invocation consumes."""

    extension_cache_paths: tuple[Path, ...] = ()
    extension_cache_keys: tuple[str, ...] = ()
    extension_batch_counts: tuple[int, ...] = ()
    round_manifest_sha256: tuple[str, ...] = ()
    fit_prompts_disjoint: bool | None = None
    # The loaded rounds themselves, in consumption order.  A bridged lane needs
    # their sealed target-token preimages to build a child replay for every
    # extension batch; nothing else reads them.
    rounds: tuple[Lfm2OnPolicyEvidenceRound, ...] = ()
    target_preimage_sha256: tuple[str | None, ...] = ()

    @property
    def round_count(self) -> int:
        return len(self.round_manifest_sha256)

    def identity_sha256(self, base_cache_key: str) -> str | None:
        return lfm2_on_policy_evidence_identity_sha256(
            base_cache_key=base_cache_key,
            extension_cache_keys=self.extension_cache_keys,
            extension_batch_counts=self.extension_batch_counts,
            round_manifest_sha256=self.round_manifest_sha256,
        )


def _resolve_on_policy_consumed_rounds(
    config: ProductionLfm2CatalogConfig,
    base_cache: Lfm2CircuitTraceCache,
    parent: Any,
) -> _ResolvedOnPolicyEvidence:
    """Authenticate every sealed round this run consumes; zero parent forwards.

    This is the normal production route for extension evidence: rounds are
    produced by a separate invocation and consumed here.  Nothing in this
    function executes the parent.
    """

    if not config.on_policy_round_manifests:
        return _ResolvedOnPolicyEvidence()
    manifest_paths = [
        value.expanduser().resolve(strict=True)
        for value in config.on_policy_round_manifests
    ]
    live_parent_fingerprint = fingerprint_lfm2_parent(parent).identity_sha256
    try:
        references = resolve_lfm2_on_policy_evidence_references(
            base_cache.path,
            manifest_paths,
            expected_parent_fingerprint=live_parent_fingerprint,
        )
        rounds = tuple(
            load_lfm2_on_policy_evidence_round(value)
            for value in manifest_paths
        )
    except Exception as exc:
        raise ProductionLfm2CatalogError(
            f"Sealed on-policy evidence rounds failed authentication: {exc}"
        ) from exc
    if len(references.extension_cache_paths) != len(rounds):
        raise ProductionLfm2CatalogError(
            "Resolved on-policy extension caches and round manifests differ."
        )
    if config.uses_tokenizer_bridge:
        # The bridged lane replays extension batches in target rows, so every
        # round it consumes has to carry the child's own tokenization.  Named
        # here, at the one place the rounds are already authenticated.
        for value in rounds:
            reason = lfm2_on_policy_round_missing_target_preimage_reason(value)
            if reason is not None:
                raise ProductionLfm2CatalogError(
                    "Consuming sealed on-policy extension evidence is not "
                    f"supported on this lane: {reason}."
                )
    return _ResolvedOnPolicyEvidence(
        extension_cache_paths=references.extension_cache_paths,
        extension_cache_keys=tuple(
            value.extension_cache_key for value in rounds
        ),
        extension_batch_counts=tuple(value.batch_count for value in rounds),
        round_manifest_sha256=tuple(
            value.manifest_sha256 for value in rounds
        ),
        # Copied from each sealed manifest, never recomputed by the runner.
        fit_prompts_disjoint=all(
            bool(value.manifest["disjointness"]["verified_disjoint"])
            for value in rounds
        ),
        rounds=tuple(rounds),
        target_preimage_sha256=tuple(
            None
            if value.target_preimage is None
            else str(value.target_preimage["target_preimage_sha256"])
            for value in rounds
        ),
    )


def _open_sealed_evidence_set(
    base_cache: Lfm2CircuitTraceCache,
    *,
    base_splits: Sequence[str],
    base_split_sha256: str,
    base_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None,
    on_policy: _ResolvedOnPolicyEvidence,
    on_policy_evidence_identity_sha256: str | None,
    defer_base_session: bool = False,
) -> Lfm2SealedEvidenceSet:
    """Open ``(base, *extensions)`` as one aggregate, extensions all-fit.

    Every extension cache is opened here, together with its selective
    authenticated session, so the closed-form pass that follows binds live
    evidence instead of an authenticated-but-unused reference.  Opening is
    fail-closed as a unit: if any extension fails, every session this call
    opened is closed before the error propagates.
    """

    extensions: list[Lfm2SealedEvidenceRequest] = []
    for position, path in enumerate(on_policy.extension_cache_paths):
        try:
            extension_cache = open_lfm2_circuit_trace_cache(path, verify=False)
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                f"Sealed on-policy extension cache {position} could not be "
                f"opened: {exc}"
            ) from exc
        if extension_cache.cache_key != on_policy.extension_cache_keys[position]:
            raise ProductionLfm2CatalogError(
                f"Sealed on-policy extension cache {position} differs from its "
                "authenticated round manifest."
            )
        # A child-visited row is always a fit row.  The rule is enforced here,
        # at the production call site, not left to a convention downstream.
        extension_splits, extension_split_sha256 = _derive_sealed_cache_splits(
            extension_cache,
            all_rows_are_fit=True,
            on_policy_evidence_identity_sha256=(
                on_policy_evidence_identity_sha256
            ),
        )
        extensions.append(
            Lfm2SealedEvidenceRequest(
                cache=extension_cache,
                splits=extension_splits,
                split_sha256=extension_split_sha256,
                round_manifest_sha256=(
                    on_policy.round_manifest_sha256[position]
                ),
                all_rows_are_fit=True,
                target_preimage_sha256=(
                    on_policy.target_preimage_sha256[position]
                    if position < len(on_policy.target_preimage_sha256)
                    else None
                ),
            )
        )
    return Lfm2SealedEvidenceSet.open(
        Lfm2SealedEvidenceRequest(
            cache=base_cache,
            splits=tuple(base_splits),
            split_sha256=base_split_sha256,
        ),
        tuple(extensions),
        base_session=base_session,
        open_sessions=bool(extensions),
        open_base_session=not defer_base_session,
    )


def _sealed_evidence_trace_rows_sha256(
    evidence_set: Lfm2SealedEvidenceSet,
) -> Callable[[Sequence[Any], Sequence[str], Sequence[str]], str]:
    """Seal the merged row identity over the members that actually produced it.

    ``merge_trace_rows_into`` proves each member's own standalone row identity
    from its own slice of the merged storage; this binds those identities, the
    evidence-set identity, and the merged path/split geometry into one digest.
    """

    def seal(
        member_rows: Sequence[Any],
        paths: Sequence[str],
        splits: Sequence[str],
    ) -> str:
        if len(member_rows) != evidence_set.member_count:
            raise ProductionLfm2CatalogError(
                "Materialized evidence rows do not cover every sealed member."
            )
        return canonical_sha256(
            {
                "schema": SEALED_EVIDENCE_TRACE_ROWS_SCHEMA,
                "evidence_set_cache_key": evidence_set.cache_key,
                "corpus_fingerprint": evidence_set.corpus_fingerprint,
                "ordered_member_cache_keys": [
                    member.cache_key for member in evidence_set.members
                ],
                "ordered_member_trace_rows_sha256": [
                    str(value.trace_rows_sha256) for value in member_rows
                ],
                "layer_kinds": list(member_rows[0].layer_kinds),
                "paths": list(paths),
                "path_splits": list(splits),
                "all_extension_rows_are_fit_rows": True,
            }
        )

    return seal


@dataclass(frozen=True, slots=True)
class _MergedSealedEvidencePayload:
    """One merged evidence materialization plus its peak-memory accounting."""

    rows: Any
    base_rows: Any
    calibration_batches: tuple[Mapping[str, torch.Tensor], ...]
    calibration_splits: tuple[str, ...]
    accounting: Mapping[str, Any]


def _materialize_sealed_evidence_rows(
    evidence_set: Lfm2SealedEvidenceSet,
    *,
    parent: Any,
    tokenizer: Any,
    base_calibration_batches: Sequence[Mapping[str, torch.Tensor]],
    base_selective_session: (
        Lfm2CircuitTraceSelectiveAuthenticatedSession | None
    ) = None,
) -> _MergedSealedEvidencePayload:
    """Stream every member corpus into one preallocated merged row index.

    Extension caches are sealed, so this performs zero parent forwards: it
    reads authenticated payloads through the sessions the evidence set already
    holds open.  The merge never materializes a member index separately and
    then joins it, so the transient peak is the merged payload plus exactly one
    batch's selected tensors instead of roughly twice the retained rows.
    """

    if not evidence_set.has_extension_evidence:
        raise ProductionLfm2CatalogError(
            "A merged evidence materialization requires extension evidence."
        )
    members: list[residual.SealedEvidenceMergeMember] = []
    calibration_batches = list(base_calibration_batches)
    splits = list(evidence_set.base.splits)
    for member in evidence_set.members:
        session = member.session
        if session is None and member.role == BASE_EVIDENCE_ROLE:
            # The base payload session is deferred until a payload is first
            # required; the runner already opened it by the time rows are
            # materialized, so the merge reads the base through that one.
            session = base_selective_session
        if session is None:
            raise ProductionLfm2CatalogError(
                "A sealed evidence member has no authenticated session."
            )
        if member.role != BASE_EVIDENCE_ROLE:
            member_batches = residual.calibration_batches_from_cache(
                member.cache,
                selective_session=session,
            )
            # Child-visited inputs still have to sit inside the parent's exact
            # tokenizer-facing row capacity.
            _validate_tokenizer_facing_parent(parent, tokenizer, member_batches)
            calibration_batches.extend(member_batches)
            splits.extend(member.splits)
            if set(member.splits) != {"fit"}:
                raise ProductionLfm2CatalogError(
                    "Every on-policy extension row must be a fit row."
                )
        members.append(
            residual.SealedEvidenceMergeMember(
                cache=member.cache,
                selective_session=session,
                batch_splits=member.splits,
                batch_offset=member.batch_offset,
            )
        )
    if tuple(splits) != evidence_set.splits:
        raise ProductionLfm2CatalogError(
            "Merged evidence splits differ from the sealed evidence set."
        )
    # Measured, not predicted: the descriptor-derived accounting below states
    # what the merge is *supposed* to hold, and this brackets it with the
    # process's own resident-set reading so a regression shows up as a real
    # number instead of an assertion about intent.
    before_memory = _process_memory_progress_metadata()
    merged = residual.merge_trace_rows_into(
        members,
        cache_key=evidence_set.cache_key,
        corpus_fingerprint=evidence_set.corpus_fingerprint,
        merged_trace_rows_sha256=_sealed_evidence_trace_rows_sha256(
            evidence_set
        ),
    )
    after_memory = _process_memory_progress_metadata()
    if len(merged.member_rows) != evidence_set.member_count:
        raise ProductionLfm2CatalogError(
            "Materialized evidence rows do not cover every sealed member."
        )
    for member, member_rows in zip(evidence_set.members, merged.member_rows):
        if (
            member_rows.cache_key != member.cache_key
            or member_rows.corpus_fingerprint
            != member.cache.corpus_fingerprint
        ):
            raise ProductionLfm2CatalogError(
                "A merged evidence row view lost its sealed member identity."
            )
        if member.role != BASE_EVIDENCE_ROLE and set(
            member_rows.path_splits
        ) != {"fit"}:
            raise ProductionLfm2CatalogError(
                "Every on-policy extension row must be a fit row."
            )
    base_rows = merged.base_rows
    if (
        base_rows.cache_key != evidence_set.base_cache.cache_key
        or base_rows.corpus_fingerprint
        != evidence_set.base_cache.corpus_fingerprint
    ):
        raise ProductionLfm2CatalogError(
            "The merged base row view lost the sealed base cache identity."
        )
    accounting = dict(merged.accounting)
    for name, value in sorted(before_memory.items()):
        accounting[f"before_{name}"] = int(value)
    for name, value in sorted(after_memory.items()):
        accounting[f"after_{name}"] = int(value)
    if "process_rss_bytes" in before_memory and "process_rss_bytes" in after_memory:
        accounting["observed_process_rss_growth_bytes"] = int(
            after_memory["process_rss_bytes"] - before_memory["process_rss_bytes"]
        )
    return _MergedSealedEvidencePayload(
        rows=merged.rows,
        base_rows=base_rows,
        calibration_batches=tuple(calibration_batches),
        calibration_splits=tuple(splits),
        accounting=MappingProxyType(accounting),
    )


def _bridged_lane_target_preimage_gap(
    config: ProductionLfm2CatalogConfig,
) -> str | None:
    """Name the round a bridged child cannot replay, if any is in scope.

    A bridged child speaks the target vocabulary and the sealed parent evidence
    is in donor rows, so an extension batch is only replayable when its round
    also sealed the child's own target tokenization of that batch.  Rounds this
    same invocation is about to *produce* are captured with the bridge in hand
    and always carry it; rounds it is asked to *consume* are checked here, by
    name, before anything else in the lane starts.
    """

    for value in config.on_policy_round_manifests:
        path = value.expanduser().absolute()
        try:
            sealed = load_lfm2_on_policy_evidence_round(path)
        except Exception as exc:
            return (
                f"on-policy round manifest {path} carries no target-token "
                f"preimage the bridged lane can read: {exc}"
            )
        reason = lfm2_on_policy_round_missing_target_preimage_reason(sealed)
        if reason is not None:
            return reason
    return None


def _unconsumed_extension_evidence_reason(
    config: ProductionLfm2CatalogConfig,
) -> str | None:
    """Name the writer that cannot fit base+extension evidence, if any.

    Consuming an extension corpus is only honest where every closed-form writer
    on the lane reads the merged evidence.  The dense budget lane now does: its
    evidence provider, its structural basis, and its closed-loop adapter all
    bind the merged read-only evidence-set views, and every extension batch is
    forced onto the fit side of the adapter's own split.  The lanes named below
    still bind exactly one sealed cache identity, so silently solving base-only
    in them is precisely the bug NOTES-1C section 4 warns about and the run
    fails closed instead.

    The strict tokenizer bridge (and therefore the routed carrier lane) is no
    longer a lane-shaped refusal.  Its trace adapter now covers the merged
    corpus, because a bridged round seals the child's own target tokenization
    of every extension batch next to the donor batch the parent was captured
    on.  What is still refused, by name, is one specific *round*: a round
    captured parent-canonically has no target-token preimage, so no child
    replay exists for its batches.

    The cached heterogeneous capacity option bank and the global topology
    frontier deliberately keep reading the base corpus alone.  They are
    *selection* consumers, and a child-visited row is a fit row: letting one
    choose the child's own topology is the frontier capture the all-fit rule
    exists to prevent.  Their derived phase keys still move with the evidence
    identity, so a base-only selection is never reused under a different
    evidence request.
    """

    if config.hybrid_model_fold:
        return "the hybrid model fold binds one sealed parent cache key"
    if config.producer_first_model_fold:
        return "the producer-first model fold binds one sealed parent cache key"
    if config.uses_tokenizer_bridge and (
        (reason := _bridged_lane_target_preimage_gap(config)) is not None
    ):
        return reason
    if config.dense_macro_cell_attractor_gate_enabled:
        return (
            "the macro-cell attractor gate binds one sealed cache handle "
            "(hf_lfm2_macro_cell_attractor_gate)"
        )
    if config.dense_vocabulary_margin_candidate_gate_enabled:
        return (
            "the dense vocabulary-margin candidate gate binds one sealed cache "
            "handle"
        )
    return None


def _assert_extension_evidence_is_consumable(
    config: ProductionLfm2CatalogConfig,
) -> None:
    reason = _unconsumed_extension_evidence_reason(config)
    if reason is None:
        return
    raise ProductionLfm2CatalogError(
        "Consuming sealed on-policy extension evidence is not supported on "
        f"this lane: {reason}. A base-only solve under a base+extension "
        "request would be silently wrong, so the run fails closed."
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
    defer_selective_session: bool = False,
) -> ResolvedProductionParentCache:
    """Open a sealed cache with no parent forwards, or retain legacy build mode."""

    if not isinstance(defer_selective_session, bool):
        raise TypeError("defer_selective_session must be bool.")

    if config.sealed_parent_cache_directory is not None:
        sealed_directory = (
            config.sealed_parent_cache_directory.expanduser().resolve(strict=True)
        )
        try:
            parent_cache = open_lfm2_circuit_trace_cache(
                sealed_directory,
                verify=False,
            )
            on_policy = _resolve_on_policy_consumed_rounds(
                config,
                parent_cache,
                parent,
            )
            on_policy_identity = on_policy.identity_sha256(
                parent_cache.cache_key
            )
            on_policy_fields: dict[str, Any] = {
                "on_policy_extension_cache_paths": (
                    on_policy.extension_cache_paths
                ),
                "on_policy_extension_cache_keys": (
                    on_policy.extension_cache_keys
                ),
                "on_policy_extension_batch_counts": (
                    on_policy.extension_batch_counts
                ),
                "on_policy_round_manifest_sha256": (
                    on_policy.round_manifest_sha256
                ),
                "on_policy_round_count": on_policy.round_count,
                "on_policy_parent_capture_invocations": 0,
                "on_policy_fit_prompts_disjoint_from_manual_review": (
                    on_policy.fit_prompts_disjoint
                ),
                "on_policy_rounds": on_policy.rounds,
            }
            # Deferring the base payload session only pays off when nothing
            # needs it before global-topology planning.  Extension evidence
            # always does: the merged read-only views a closed-form writer
            # binds are built from every member's authenticated session, base
            # included, so deferring it would leave the merged view unopenable.
            if (
                defer_selective_session
                and config.calibration is not None
                and not on_policy.extension_cache_paths
            ):
                calibration_path = config.calibration.expanduser().resolve(
                    strict=True
                )
                calibration = residual.load_calibration_batches(
                    calibration_path
                )
                supplied_input_sha256s = tuple(
                    value.input_sha256
                    for value in _prepare_lfm2_trace_batches(
                        torch,
                        calibration.batches,
                        parent_capture_config,
                    )
                )
                cached_input_sha256s = tuple(
                    str(value)
                    for value in parent_cache.manifest["corpus"][
                        "ordered_batch_input_sha256"
                    ]
                )
                if supplied_input_sha256s == cached_input_sha256s:
                    calibration_batches = tuple(calibration.batches)
                    _validate_sealed_parent_cache_binding(
                        parent_cache,
                        parent,
                        tokenizer,
                        calibration_batches,
                    )
                    evidence_split_sha256 = canonical_sha256(
                        {
                            "schema": SEALED_CACHE_SPLIT_SCHEMA,
                            "parent_cache_key": parent_cache.cache_key,
                            "corpus_fingerprint": (
                                parent_cache.corpus_fingerprint
                            ),
                            "ordered_splits": list(calibration.splits),
                            "assignment": "supplied_matching_calibration",
                            "calibration_sha256": calibration.source_sha256,
                            **_on_policy_split_identity_fields(
                                on_policy_identity
                            ),
                        }
                    )
                    deferred_evidence_set = _open_sealed_evidence_set(
                        parent_cache,
                        base_splits=tuple(calibration.splits),
                        base_split_sha256=evidence_split_sha256,
                        base_session=None,
                        on_policy=on_policy,
                        on_policy_evidence_identity_sha256=(
                            on_policy_identity
                        ),
                        defer_base_session=True,
                    )
                    return ResolvedProductionParentCache(
                        cache=parent_cache,
                        selective_session=None,
                        calibration_batches=calibration_batches,
                        calibration_splits=tuple(calibration.splits),
                        calibration_source_sha256=calibration.source_sha256,
                        evidence_split_sha256=evidence_split_sha256,
                        supplied_calibration_sha256=(
                            calibration.source_sha256
                        ),
                        supplied_calibration_matches_cache=True,
                        source_mode=SEALED_PARENT_CACHE_SOURCE_MODE,
                        evidence_split_source=(
                            "supplied_matching_calibration"
                        ),
                        parent_cache_reused=True,
                        parent_cache_build_called=False,
                        parent_cache_opened_with_full_verification=False,
                        parent_trace_regeneration_forward_count=0,
                        evidence_set=deferred_evidence_set,
                        on_policy_extension_split_sha256=(
                            deferred_evidence_set.extension_split_sha256s
                        ),
                        **on_policy_fields,
                    )
            selective_session = parent_cache.selective_authenticated_session()
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                f"Sealed parent cache failed selective authentication: {exc}"
            ) from exc
        evidence_set: Lfm2SealedEvidenceSet | None = None
        try:
            calibration_batches = residual.calibration_batches_from_cache(
                parent_cache,
                selective_session=selective_session,
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
                        **_on_policy_split_identity_fields(
                            on_policy_identity
                        ),
                    }
                )
            else:
                (
                    calibration_splits,
                    evidence_split_sha256,
                ) = _derive_sealed_cache_splits(
                    parent_cache,
                    on_policy_evidence_identity_sha256=on_policy_identity,
                )
                calibration_source_sha256 = None
                evidence_split_source = "sealed_cache_deterministic"
            evidence_set = _open_sealed_evidence_set(
                parent_cache,
                base_splits=calibration_splits,
                base_split_sha256=evidence_split_sha256,
                base_session=selective_session,
                on_policy=on_policy,
                on_policy_evidence_identity_sha256=on_policy_identity,
            )
            resolved = ResolvedProductionParentCache(
                cache=parent_cache,
                selective_session=selective_session,
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
                parent_cache_opened_with_full_verification=False,
                parent_trace_regeneration_forward_count=0,
                evidence_set=evidence_set,
                on_policy_extension_split_sha256=(
                    evidence_set.extension_split_sha256s
                ),
                **on_policy_fields,
            )
        except BaseException as exc:
            # Every extension session this resolution opened is closed with the
            # base session; a half-open evidence set never escapes.
            if evidence_set is not None:
                try:
                    evidence_set.close()
                except BaseException as close_exc:
                    exc.add_note(
                        "Sealed evidence-set cleanup also failed: "
                        f"{type(close_exc).__qualname__}: {close_exc}"
                    )
            try:
                selective_session.close()
            except BaseException as close_exc:
                exc.add_note(
                    "Selective parent-cache session cleanup also failed: "
                    f"{type(close_exc).__qualname__}: {close_exc}"
                )
            raise
        return resolved

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
        selective_session=None,
        calibration_batches=residual.calibration_batches_from_cache(parent_cache),
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
class ResolvedProductionTokenizerBridge:
    """Authenticated target-token replay bound to one sealed donor cache."""

    artifact: StrictTokenizerBridgeArtifact
    target_tokenizer: Any
    target_tokenizer_artifact_fingerprint: str
    trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter
    target_calibration: Any
    paired_input_evidence: tuple[Mapping[str, Any], ...]
    evidence_sha256: str
    canonical_tokenization_evidence_sha256: str | None = None
    # One adapter over ``(base, *on-policy extensions)`` in merged batch order,
    # built from the rounds' sealed target-token preimages.  ``None`` unless
    # extension evidence is consumed, and then it — not ``trace_adapter`` — is
    # what every solve-time consumer binds, because the merged corpus is what
    # the child replays.
    merged_trace_adapter: Lfm2StrictTokenizerBridgeTraceAdapter | None = None
    merged_paired_input_evidence: tuple[Mapping[str, Any], ...] = ()
    on_policy_target_preimage_sha256: tuple[str, ...] = ()

    @property
    def solve_trace_adapter(self) -> Lfm2StrictTokenizerBridgeTraceAdapter:
        """The adapter covering exactly the corpus this solve pass consumes."""

        if self.merged_trace_adapter is None:
            return self.trace_adapter
        return self.merged_trace_adapter

    @property
    def consumes_on_policy_extension_evidence(self) -> bool:
        return self.merged_trace_adapter is not None

    @property
    def parent_tokenizer_semantics_sha256(self) -> str:
        return self.artifact.donor_tokenizer_semantics_sha256

    @property
    def target_tokenizer_semantics_sha256(self) -> str:
        return self.artifact.target_tokenizer_semantics_sha256

    def verify(
        self,
        *,
        donor_tokenizer: Any,
        parent_cache: Lfm2CircuitTraceCache,
        preverified_trace_session: (
            Lfm2PreverifiedTokenizerBridgeTraceSession | None
        ) = None,
    ) -> None:
        self.artifact.verify_live_tokenizers(
            donor_tokenizer,
            self.target_tokenizer,
        )
        if preverified_trace_session is None:
            self.trace_adapter.verify()
        else:
            if preverified_trace_session.adapter is not self.trace_adapter:
                raise ProductionLfm2CatalogError(
                    "Tokenizer bridge preverified session changed adapters."
                )
            preverified_trace_session.verify_complete()
        if (
            self.trace_adapter.bridge_artifact_sha256
            != self.artifact.artifact_sha256
            or self.trace_adapter.transport_profile_fingerprint
            != self.artifact.profile_fingerprint
            or self.trace_adapter.evidence_sha256
            != self.evidence_sha256
        ):
            raise ProductionLfm2CatalogError(
                "Tokenizer bridge runtime seals changed after resolution."
            )
        if self.canonical_tokenization_evidence_sha256 is not None:
            _digest(
                self.canonical_tokenization_evidence_sha256,
                "canonical_tokenization_evidence_sha256",
            )
        if len(parent_cache) != self.trace_adapter.record_count:
            raise ProductionLfm2CatalogError(
                "Tokenizer bridge record count differs from the sealed cache."
            )
        ordered_input_sha256 = tuple(
            str(value)
            for value in parent_cache.manifest["corpus"]["ordered_batch_input_sha256"]
        )
        for record in self.trace_adapter.records:
            if (
                record.ordinal >= len(ordered_input_sha256)
                or ordered_input_sha256[record.ordinal]
                != record.donor_cache_input_sha256
            ):
                raise ProductionLfm2CatalogError(
                    "Tokenizer bridge donor input digest differs from the "
                    f"sealed parent cache at batch {record.ordinal}."
                )

    def verify_merged(self, evidence_set: Lfm2SealedEvidenceSet) -> None:
        """Prove the merged adapter covers exactly the merged sealed corpus."""

        adapter = self.merged_trace_adapter
        if adapter is None:
            raise ProductionLfm2CatalogError(
                "This tokenizer bridge carries no merged target-token replay."
            )
        adapter.verify()
        if (
            adapter.bridge_artifact_sha256 != self.artifact.artifact_sha256
            or adapter.transport_profile_fingerprint
            != self.artifact.profile_fingerprint
        ):
            raise ProductionLfm2CatalogError(
                "Merged tokenizer bridge adapter belongs to another bridge."
            )
        batches = tuple(evidence_set.iter_batches())
        if adapter.record_count != len(batches):
            raise ProductionLfm2CatalogError(
                "Merged target-token replay does not cover every sealed "
                "evidence batch exactly once."
            )
        base_records = self.trace_adapter.records
        for record, batch in zip(adapter.records, batches, strict=True):
            if record.donor_cache_input_sha256 != batch.input_sha256:
                raise ProductionLfm2CatalogError(
                    "Merged target-token replay differs from the sealed "
                    f"evidence donor input at merged batch {batch.merged_index}."
                )
            if record.ordinal < len(base_records) and (
                record.record_id != base_records[record.ordinal].record_id
                or record.split != base_records[record.ordinal].split
            ):
                raise ProductionLfm2CatalogError(
                    "Merged target-token replay changed the sealed base "
                    f"corpus record at batch {record.ordinal}."
                )
            if record.ordinal >= len(base_records) and record.split != "fit":
                raise ProductionLfm2CatalogError(
                    "Every on-policy extension replay record must be a fit row."
                )

    @property
    def solve_paired_input_evidence(self) -> tuple[Mapping[str, Any], ...]:
        """Runtime pair evidence for exactly the corpus this pass consumes."""

        if self.merged_trace_adapter is None:
            return self.paired_input_evidence
        return self.merged_paired_input_evidence

    def receipt_fields(self) -> dict[str, Any]:
        # Every field below describes the corpus a solve pass actually binds.
        # With no extension evidence that is the base corpus and the body is
        # byte-identical to the pre-extension revision; with extension evidence
        # it is the merged corpus, which is what the vocabulary-margin binding
        # cross-checks its own adapter against.
        adapter = self.solve_trace_adapter
        paired = self.solve_paired_input_evidence
        body = {
            "schema": TOKENIZER_BRIDGE_RUNTIME_SCHEMA,
            "enabled": True,
            "artifact_sha256": self.artifact.artifact_sha256,
            "transport_profile_sha256": self.artifact.profile_fingerprint,
            "parent_tokenizer_semantics_sha256": (
                self.parent_tokenizer_semantics_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                self.target_tokenizer_semantics_sha256
            ),
            "target_tokenizer_artifact_fingerprint": (
                self.target_tokenizer_artifact_fingerprint
            ),
            "target_calibration_sha256": (
                self.target_calibration.source_sha256
            ),
            "target_row_capacity": (
                self.artifact.transport_profile.target_row_capacity
            ),
            "donor_row_capacity": (
                self.artifact.transport_profile.donor_row_capacity
            ),
            "trace_adapter_evidence_sha256": adapter.evidence_sha256,
            "canonical_target_tokenization_evidence_sha256": (
                self.canonical_tokenization_evidence_sha256
            ),
            "paired_input_evidence_sha256": canonical_sha256(
                [dict(value) for value in paired]
            ),
            "paired_record_count": len(paired),
            "target_to_donor_row_indices_sha256": canonical_sha256(
                list(adapter.mapping)
            ),
            "all_physical_target_rows_mapped": (
                len(adapter.mapping)
                == self.artifact.transport_profile.target_row_capacity
            ),
            "mapping_injective_required": False,
            "repeated_donor_rows_preserve_distinct_target_rows": True,
            "numeric_token_id_equality_required": False,
            "donor_canonical_sequence_equivalence_claimed": False,
            "parent_cache_input_mode": (
                "target-tokenizer-canonical-then-one-row-mapped-to-donor"
            ),
            "parent_model_forwards": 0,
            "training_steps": 0,
        }
        if self.merged_trace_adapter is not None:
            # Only a base+extension solve carries these keys, so a base-only
            # bridged run's receipt body is byte-identical to before.
            body = {
                **body,
                "base_trace_adapter_evidence_sha256": self.evidence_sha256,
                "base_paired_record_count": len(self.paired_input_evidence),
                "on_policy_merged_trace_adapter_evidence_sha256": (
                    self.merged_trace_adapter.evidence_sha256
                ),
                "on_policy_merged_record_count": (
                    self.merged_trace_adapter.record_count
                ),
                "on_policy_extension_record_count": (
                    self.merged_trace_adapter.record_count
                    - self.trace_adapter.record_count
                ),
                "on_policy_merged_paired_record_count": len(
                    self.merged_paired_input_evidence
                ),
                "on_policy_target_preimage_sha256": list(
                    self.on_policy_target_preimage_sha256
                ),
                "on_policy_extension_replay_is_the_sealed_target_preimage": (
                    True
                ),
            }
        return {
            **body,
            "runtime_receipt_sha256": canonical_sha256(body),
        }


def _cache_batch_mapping(batch: Any) -> dict[str, torch.Tensor]:
    result = {
        "input_ids": batch.input_ids.clone(),
        "attention_mask": batch.attention_mask.clone(),
        "forced_decode_ids": batch.forced_decode_ids.clone(),
    }
    position_ids = batch.tensors.get("input.position_ids")
    if position_ids is not None:
        result["position_ids"] = position_ids.clone()
    return result


def _target_tokenized_records(calibration: Any) -> tuple[Lfm2TargetTokenizedRecord, ...]:
    records = []
    for index, (batch, split) in enumerate(
        zip(calibration.batches, calibration.splits)
    ):
        input_ids = _calibration_tensor(batch, "input_ids")
        forced_decode_ids = _calibration_tensor(batch, "forced_decode_ids")
        if input_ids is None or forced_decode_ids is None:
            raise ProductionLfm2CatalogError(
                "Target-token calibration is missing input or forced-decode IDs."
            )
        attention_mask = _calibration_tensor(batch, "attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        records.append(
            Lfm2TargetTokenizedRecord(
                record_id=f"batch-{index:06d}",
                split=str(split),
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_decode_ids=forced_decode_ids,
                position_ids=_calibration_tensor(batch, "position_ids"),
            )
        )
    return tuple(records)


def _calibration_json_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    text = path.read_text(encoding="utf-8")
    try:
        parsed = json.loads(text)
        values = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        values = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    if not values or any(not isinstance(value, Mapping) for value in values):
        raise ProductionLfm2CatalogError(
            "Target-token calibration must contain JSON objects."
        )
    return tuple(values)


def _tokenizer_input_ids(
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool = True,
) -> tuple[int, ...]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
        )
    except Exception as exc:
        raise ProductionLfm2CatalogError(
            "Target tokenizer rejected a declared calibration input text."
        ) from exc
    if not isinstance(encoded, Mapping):
        raise ProductionLfm2CatalogError(
            "Target tokenizer returned an invalid calibration encoding."
        )
    values = encoded.get("input_ids")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or not values
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in values
        )
    ):
        raise ProductionLfm2CatalogError(
            "Target tokenizer returned invalid calibration token IDs."
        )
    return tuple(values)


def _verify_target_calibration_canonical_tokenization(
    path: Path,
    calibration: Any,
    target_tokenizer: Any,
) -> str:
    """Prove trace paths are target-canonical, not donor-canonical guesses."""

    rows = _calibration_json_rows(path)
    if len(rows) != len(calibration.batches):
        raise ProductionLfm2CatalogError(
            "Target calibration text evidence does not cover every batch."
        )
    evidence_rows = []
    pad_token_id = getattr(target_tokenizer, "pad_token_id", None)
    for index, (row, batch) in enumerate(zip(rows, calibration.batches)):
        declared_mode = row.get("target_tokenization_mode")
        if declared_mode != "tokenizer-call-add-special-tokens":
            raise ProductionLfm2CatalogError(
                "Every target calibration row must explicitly declare "
                "target_tokenization_mode=tokenizer-call-add-special-tokens."
            )
        forced_mode = row.get("target_forced_decode_tokenization_mode")
        if forced_mode != "tokenizer-call-no-special-tokens":
            raise ProductionLfm2CatalogError(
                "Every target calibration row must explicitly declare "
                "target_forced_decode_tokenization_mode="
                "tokenizer-call-no-special-tokens."
            )
        texts_value = row.get("target_input_texts")
        if isinstance(texts_value, str):
            texts = (texts_value,)
        elif isinstance(texts_value, list) and all(
            isinstance(value, str) and value for value in texts_value
        ):
            texts = tuple(texts_value)
        else:
            raise ProductionLfm2CatalogError(
                "Every target calibration row needs nonempty "
                "target_input_texts."
            )
        forced_texts_value = row.get("target_forced_decode_texts")
        if isinstance(forced_texts_value, str):
            forced_texts = (forced_texts_value,)
        elif isinstance(forced_texts_value, list) and all(
            isinstance(value, str) and value
            for value in forced_texts_value
        ):
            forced_texts = tuple(forced_texts_value)
        else:
            raise ProductionLfm2CatalogError(
                "Every target calibration row needs nonempty "
                "target_forced_decode_texts."
            )
        input_ids = _calibration_tensor(batch, "input_ids")
        attention_mask = _calibration_tensor(batch, "attention_mask")
        forced_decode_ids = _calibration_tensor(batch, "forced_decode_ids")
        if input_ids is None:
            raise ProductionLfm2CatalogError(
                "Target calibration input IDs are missing."
            )
        if forced_decode_ids is None:
            raise ProductionLfm2CatalogError(
                "Target calibration forced-decode IDs are missing."
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        if (
            len(texts) != int(input_ids.shape[0])
            or len(forced_texts) != int(input_ids.shape[0])
            or int(forced_decode_ids.shape[0]) != int(input_ids.shape[0])
        ):
            raise ProductionLfm2CatalogError(
                f"Target calibration text batch {index} changed batch size."
            )
        encoded_rows = []
        for batch_index, text in enumerate(texts):
            canonical_ids = _tokenizer_input_ids(target_tokenizer, text)
            mask = attention_mask[batch_index].to(dtype=torch.bool)
            observed_active = tuple(
                int(value)
                for value in input_ids[batch_index][mask].tolist()
            )
            if observed_active != canonical_ids:
                raise ProductionLfm2CatalogError(
                    "Target calibration input IDs are not the sealed target "
                    f"tokenizer's canonical sequence at row {index}, batch "
                    f"{batch_index}."
                )
            canonical_forced_ids = _tokenizer_input_ids(
                target_tokenizer,
                forced_texts[batch_index],
                add_special_tokens=False,
            )
            observed_forced_ids = tuple(
                int(value)
                for value in forced_decode_ids[batch_index].tolist()
            )
            if observed_forced_ids != canonical_forced_ids:
                raise ProductionLfm2CatalogError(
                    "Target calibration forced-decode IDs are not the sealed "
                    "target tokenizer's canonical no-special-token sequence "
                    f"at row {index}, batch {batch_index}."
                )
            if bool((~mask).any().item()):
                if pad_token_id is None or any(
                    int(value) != int(pad_token_id)
                    for value in input_ids[batch_index][~mask].tolist()
                ):
                    raise ProductionLfm2CatalogError(
                        "Target calibration masked positions are not target "
                        "padding tokens."
                    )
            encoded_rows.append(
                {
                    "batch_index": batch_index,
                    "text_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "canonical_target_ids": list(canonical_ids),
                    "forced_text_sha256": hashlib.sha256(
                        forced_texts[batch_index].encode("utf-8")
                    ).hexdigest(),
                    "canonical_target_forced_decode_ids": list(
                        canonical_forced_ids
                    ),
                }
            )
        evidence_rows.append(
            {
                "row_index": index,
                "split": calibration.splits[index],
                "target_tokenization_mode": declared_mode,
                "target_forced_decode_tokenization_mode": forced_mode,
                "encoded_rows": encoded_rows,
            }
        )
    body = {
        "schema": (
            "neural-forge-production-lfm2-target-canonical-tokenization-v1"
        ),
        "target_calibration_sha256": calibration.source_sha256,
        "target_tokenizer_semantics_sha256": (
            fingerprint_hf_tokenizer_semantics(target_tokenizer)
        ),
        "rows": evidence_rows,
        "donor_canonical_sequence_equivalence_claimed": False,
        "model_forwards": 0,
    }
    return canonical_sha256(body)


def resolve_production_tokenizer_bridge(
    config: ProductionLfm2CatalogConfig,
    *,
    donor_tokenizer: Any,
    parent_cache: Lfm2CircuitTraceCache,
    calibration_splits: Sequence[str],
    calibration_batches: Sequence[Mapping[str, torch.Tensor]] | None = None,
    progress: _ProgressJournal | None = None,
) -> ResolvedProductionTokenizerBridge | None:
    """Load only an explicitly sealed bridge and bind it to cached donor IDs."""

    if not config.uses_tokenizer_bridge:
        return None
    assert config.target_tokenizer_directory is not None
    assert config.target_tokenizer_calibration is not None
    assert config.tokenizer_bridge_artifact_sha256 is not None
    assert config.tokenizer_bridge_profile_sha256 is not None
    if progress is not None:
        progress.emit(
            "tokenizer_bridge_started",
            parent_cache_record_count=len(parent_cache),
        )
    target_directory = (
        config.target_tokenizer_directory.expanduser().resolve(strict=True)
    )
    target_calibration_path = (
        config.target_tokenizer_calibration.expanduser().resolve(strict=True)
    )
    target_tokenizer = residual.load_local_tokenizer(target_directory)
    artifact = load_strict_tokenizer_bridge_artifact(
        target_directory,
        donor_tokenizer=donor_tokenizer,
        target_tokenizer=target_tokenizer,
        expected_artifact_sha256=config.tokenizer_bridge_artifact_sha256,
    )
    if artifact.profile_fingerprint != config.tokenizer_bridge_profile_sha256:
        raise ProductionLfm2CatalogError(
            "Tokenizer bridge transport profile differs from the explicit "
            "profile digest."
        )
    if progress is not None:
        progress.emit(
            "tokenizer_bridge_live_tokenizers_authenticated",
            donor_row_capacity=artifact.transport_profile.donor_row_capacity,
            target_row_capacity=artifact.transport_profile.target_row_capacity,
        )
    target_calibration = residual.load_calibration_batches(
        target_calibration_path
    )
    canonical_tokenization_evidence_sha256 = (
        _verify_target_calibration_canonical_tokenization(
            target_calibration_path,
            target_calibration,
            target_tokenizer,
        )
    )
    if tuple(target_calibration.splits) != tuple(calibration_splits):
        raise ProductionLfm2CatalogError(
            "Target-token calibration splits differ from the sealed parent "
            "cache splits."
        )
    preverified_session = (
        build_preverified_lfm2_strict_tokenizer_bridge_trace_session(
            artifact,
            _target_tokenized_records(target_calibration),
            expected_artifact_sha256=config.tokenizer_bridge_artifact_sha256,
        )
    )
    adapter = preverified_session.adapter
    if len(parent_cache) != adapter.record_count:
        raise ProductionLfm2CatalogError(
            "Target-token calibration record count differs from the sealed "
            "parent cache."
        )
    if calibration_batches is not None and len(calibration_batches) != len(
        parent_cache
    ):
        raise ProductionLfm2CatalogError(
            "Authenticated calibration batch count differs from the sealed cache."
        )
    ordered_input_sha256 = tuple(
        str(value)
        for value in parent_cache.manifest["corpus"]["ordered_batch_input_sha256"]
    )
    paired = []
    for record in adapter.records:
        if (
            record.ordinal >= len(ordered_input_sha256)
            or ordered_input_sha256[record.ordinal] != record.donor_cache_input_sha256
        ):
            raise ProductionLfm2CatalogError(
                "The sealed parent cache was not captured from the bridge-"
                f"mapped donor IDs at batch {record.ordinal}."
            )
        if calibration_batches is None:
            parent_cache_batch = _cache_batch_mapping(
                parent_cache.load_batch(record.ordinal)
            )
        else:
            parent_cache_batch = {
                name: value.clone()
                for name, value in calibration_batches[record.ordinal].items()
            }
        evidence = authenticate_lfm2_bridge_input_pair(
            preverified_session,
            record_id=record.record_id,
            parent_cache_batch=parent_cache_batch,
            child_replay_batch=record.as_child_replay_batch(),
        )
        paired.append(MappingProxyType(evidence.to_dict()))
        completed_pair_count = record.ordinal + 1
        if progress is not None and (
            completed_pair_count == 1
            or completed_pair_count == adapter.record_count
            or completed_pair_count % 16 == 0
        ):
            progress.emit(
                "tokenizer_bridge_pair_progress",
                authenticated_pair_count=completed_pair_count,
                total_pair_count=adapter.record_count,
            )
    runtime = ResolvedProductionTokenizerBridge(
        artifact=artifact,
        target_tokenizer=target_tokenizer,
        target_tokenizer_artifact_fingerprint=fingerprint_local_tokenizer(
            target_directory
        ),
        trace_adapter=adapter,
        target_calibration=target_calibration,
        paired_input_evidence=tuple(paired),
        evidence_sha256=adapter.evidence_sha256,
        canonical_tokenization_evidence_sha256=(
            canonical_tokenization_evidence_sha256
        ),
    )
    runtime.verify(
        donor_tokenizer=donor_tokenizer,
        parent_cache=parent_cache,
        preverified_trace_session=preverified_session,
    )
    if progress is not None:
        progress.emit(
            "tokenizer_bridge_complete",
            authenticated_pair_count=len(paired),
            trace_adapter_evidence_sha256=adapter.evidence_sha256,
        )
    return runtime


def extend_tokenizer_bridge_with_on_policy_preimages(
    bridge: ResolvedProductionTokenizerBridge,
    *,
    config: ProductionLfm2CatalogConfig,
    evidence_set: Lfm2SealedEvidenceSet,
    rounds: Sequence[Lfm2OnPolicyEvidenceRound],
    merged_calibration_batches: Sequence[Mapping[str, torch.Tensor]],
    progress: _ProgressJournal | None = None,
) -> ResolvedProductionTokenizerBridge:
    """Extend one resolved bridge to replay the merged base+extension corpus.

    The base records stay exactly the ones ``resolve_production_tokenizer_bridge``
    authenticated.  Each extension member contributes the target-token records
    its round sealed, rebuilt and re-proved against the live transport profile
    and against the donor inputs the sealed extension cache actually holds, so
    the merged adapter is the same kind of evidence as the base one rather than
    a claim copied out of a manifest.  Zero parent forwards.
    """

    if not isinstance(bridge, ResolvedProductionTokenizerBridge):
        raise ProductionLfm2CatalogError(
            "extend_tokenizer_bridge_with_on_policy_preimages needs a "
            "resolved strict tokenizer bridge."
        )
    if not evidence_set.has_extension_evidence:
        return bridge
    extension_members = evidence_set.extension_members
    if len(rounds) != len(extension_members):
        raise ProductionLfm2CatalogError(
            "Authenticated on-policy rounds do not cover every sealed "
            "extension member."
        )
    if progress is not None:
        progress.emit(
            "tokenizer_bridge_on_policy_preimage_started",
            base_record_count=bridge.trace_adapter.record_count,
            extension_member_count=len(extension_members),
            merged_batch_count=evidence_set.batch_count,
        )
    base_records = _target_tokenized_records(bridge.target_calibration)
    extension_records: list[Any] = []
    preimage_digests: list[str] = []
    for member, sealed_round in zip(extension_members, rounds, strict=True):
        if member.round_manifest_sha256 != sealed_round.manifest_sha256:
            raise ProductionLfm2CatalogError(
                "A sealed evidence member and its on-policy round disagree on "
                "the round manifest digest."
            )
        try:
            records = build_lfm2_on_policy_round_target_records(
                sealed_round,
                bridge=bridge.artifact,
                extension_cache=member.cache,
                expected_artifact_sha256=(
                    config.tokenizer_bridge_artifact_sha256
                ),
            )
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                "Sealed on-policy target-token preimage failed "
                f"authentication: {exc}"
            ) from exc
        extension_records.extend(records)
        assert sealed_round.target_preimage is not None
        preimage_digests.append(
            str(sealed_round.target_preimage["target_preimage_sha256"])
        )
    merged_session = (
        build_preverified_lfm2_strict_tokenizer_bridge_trace_session(
            bridge.artifact,
            (*base_records, *extension_records),
            expected_artifact_sha256=config.tokenizer_bridge_artifact_sha256,
        )
    )
    merged_adapter = merged_session.adapter
    if len(merged_calibration_batches) != merged_adapter.record_count:
        raise ProductionLfm2CatalogError(
            "Merged target-token replay and merged parent calibration cover "
            "different batch counts."
        )
    paired: list[Mapping[str, Any]] = []
    for record in merged_adapter.records:
        evidence = authenticate_lfm2_bridge_input_pair(
            merged_session,
            record_id=record.record_id,
            parent_cache_batch={
                name: value.clone()
                for name, value in merged_calibration_batches[
                    record.ordinal
                ].items()
            },
            child_replay_batch=record.as_child_replay_batch(),
        )
        paired.append(MappingProxyType(evidence.to_dict()))
    merged_session.verify_complete()
    extended = replace(
        bridge,
        merged_trace_adapter=merged_adapter,
        merged_paired_input_evidence=tuple(paired),
        on_policy_target_preimage_sha256=tuple(preimage_digests),
    )
    extended.verify_merged(evidence_set)
    if progress is not None:
        progress.emit(
            "tokenizer_bridge_on_policy_preimage_complete",
            merged_record_count=merged_adapter.record_count,
            extension_record_count=len(extension_records),
            authenticated_pair_count=len(paired),
            merged_trace_adapter_evidence_sha256=(
                merged_adapter.evidence_sha256
            ),
            parent_forwards=0,
        )
    return extended


def _maximum_repeated_bigram_share(token_ids: Sequence[int]) -> float:
    values = tuple(int(value) for value in token_ids)
    bigrams = tuple(zip(values, values[1:]))
    if not bigrams:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for bigram in bigrams:
        counts[bigram] = counts.get(bigram, 0) + 1
    return max(counts.values()) / len(bigrams)


def seal_lfm2_tokenizer_bridge_mapped_parent_envelope(
    source_rows: str | Path,
    destination: str | Path,
    *,
    original_parent_baseline: Lfm2CachedWholeOutputBaseline,
    tokenizer_bridge: ResolvedProductionTokenizerBridge,
    parent_cache_key: str,
    parent_state_fingerprint: str,
) -> dict[str, Any]:
    """Seal one-time mapped-sequence parent responses without model execution."""

    if tokenizer_bridge.canonical_tokenization_evidence_sha256 is None:
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior cannot be sealed without canonical target "
            "tokenization evidence."
        )
    source = Path(source_rows).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().absolute()
    if not source.is_file() or source.is_symlink() or target.exists():
        raise ProductionLfm2CatalogError(
            "Mapped-parent source must be a safe file and destination must "
            "not already exist."
        )
    try:
        rows = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionLfm2CatalogError(
            "Mapped-parent response rows are not valid UTF-8 JSON."
        ) from exc
    if not isinstance(rows, list) or not rows:
        raise ProductionLfm2CatalogError(
            "Mapped-parent response rows must be a nonempty JSON list."
        )
    original_by_id = {
        case.case_id: case for case in original_parent_baseline.cases
    }
    if len(rows) != len(original_by_id):
        raise ProductionLfm2CatalogError(
            "Mapped-parent response rows do not cover the original baseline."
        )
    mapping = (
        tokenizer_bridge.artifact.transport_profile
        .target_to_donor_row_indices
    )
    sealed_rows = []
    behavior_input_rows = []
    seen_case_ids = set()
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent response row {index} is not an object."
            )
        row = dict(value)
        case_id = row.get("id")
        prompt = row.get("prompt")
        if (
            not isinstance(case_id, str)
            or case_id not in original_by_id
            or case_id in seen_case_ids
            or not isinstance(prompt, str)
            or prompt != original_by_id[case_id].prompt
        ):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent response row {index} does not match the "
                "original behavior case."
            )
        target_ids = _tokenizer_input_ids(
            tokenizer_bridge.target_tokenizer,
            prompt,
        )
        if any(token_id >= len(mapping) for token_id in target_ids):
            raise ProductionLfm2CatalogError(
                f"Target behavior prompt {case_id!r} exceeds the sealed "
                "transport profile."
            )
        mapped_ids = tuple(mapping[token_id] for token_id in target_ids)
        claimed_target = row.get("target_prompt_input_ids")
        claimed_mapped = row.get("mapped_parent_input_ids")
        if claimed_target is not None and claimed_target != list(target_ids):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent response row {index} claims noncanonical "
                "target prompt IDs."
            )
        if claimed_mapped is not None and claimed_mapped != list(mapped_ids):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent response row {index} claims the wrong mapped "
                "donor prompt IDs."
            )
        row["target_prompt_input_ids"] = list(target_ids)
        row["mapped_parent_input_ids"] = list(mapped_ids)
        sealed_rows.append(row)
        behavior_input_rows.append(
            {
                "id": case_id,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "target_prompt_input_ids": list(target_ids),
                "mapped_parent_input_ids": list(mapped_ids),
            }
        )
        seen_case_ids.add(case_id)
    if seen_case_ids != set(original_by_id):
        raise ProductionLfm2CatalogError(
            "Mapped-parent response case IDs differ from the original "
            "baseline."
        )
    behavior_input_evidence_sha256 = canonical_sha256(
        {
            "schema": (
                "neural-forge-production-lfm2-bridge-behavior-inputs-v1"
            ),
            "bridge_artifact_sha256": (
                tokenizer_bridge.artifact.artifact_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                tokenizer_bridge.target_tokenizer_semantics_sha256
            ),
            "rows": behavior_input_rows,
            "target_tokenization_is_canonical": True,
            "mapped_ids_are_one_row_transport": True,
            "donor_canonical_sequence_equivalence_claimed": False,
            "model_forwards": 0,
        }
    )
    body = {
        "schema": TOKENIZER_BRIDGE_MAPPED_OUTPUT_ENVELOPE_SCHEMA,
        "parent_cache_key": _digest(parent_cache_key, "parent_cache_key"),
        "parent_state_fingerprint": _digest(
            parent_state_fingerprint,
            "parent_state_fingerprint",
        ),
        "original_parent_baseline_fingerprint": (
            original_parent_baseline.fingerprint
        ),
        "original_parent_baseline_envelope_sha256": (
            original_parent_baseline.envelope_sha256
        ),
        "bridge_artifact_sha256": (
            tokenizer_bridge.artifact.artifact_sha256
        ),
        "transport_profile_sha256": (
            tokenizer_bridge.artifact.profile_fingerprint
        ),
        "trace_adapter_evidence_sha256": tokenizer_bridge.evidence_sha256,
        "canonical_target_tokenization_evidence_sha256": (
            tokenizer_bridge.canonical_tokenization_evidence_sha256
        ),
        "canonical_behavior_input_evidence_sha256": (
            behavior_input_evidence_sha256
        ),
        "parent_tokenizer_semantics_sha256": (
            tokenizer_bridge.parent_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
        "generation": {
            **GENERATION_PROTOCOL,
            "max_new_tokens": original_parent_baseline.generation_limit,
        },
        "input_execution_contract": {
            "source_text_encoded_by": "sealed-target-tokenizer",
            "target_ids_mapped_to_parent_by": "strict-one-row-transport-profile",
            "parent_input_ids_are_donor_canonical_tokenization": False,
            "parent_input_ids_are_mapped_target_tokenization": True,
            "numeric_token_id_equality_required": False,
        },
        "output_execution_contract": {
            "model": "original-parent",
            "generation_ids": "donor-vocabulary",
            "decoded_by": "original-donor-tokenizer",
            "semantic_scoring_not_exact_token_equality": True,
        },
        "rows": sealed_rows,
        "parent_outputs_generated_once": True,
        "parent_forward_passes_after_seal": 0,
    }
    envelope = {**body, "envelope_sha256": canonical_sha256(body)}
    _write_atomic_json(target, envelope)
    return {
        "path": str(target),
        "file_sha256": _file_sha256(target),
        "envelope_sha256": envelope["envelope_sha256"],
        "row_count": len(sealed_rows),
    }


def _load_bridge_mapped_parent_baseline(
    config: ProductionLfm2CatalogConfig,
    *,
    original_baseline: Lfm2CachedWholeOutputBaseline,
    tokenizer_bridge: ResolvedProductionTokenizerBridge,
    parent_cache: Lfm2CircuitTraceCache,
) -> tuple[Lfm2CachedWholeOutputBaseline, Mapping[str, Any]]:
    assert config.tokenizer_bridge_mapped_parent_outputs is not None
    assert config.tokenizer_bridge_mapped_parent_outputs_sha256 is not None
    source = (
        config.tokenizer_bridge_mapped_parent_outputs.expanduser().resolve(
            strict=True
        )
    )
    if not source.is_file() or source.is_symlink():
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope is missing or unsafe."
        )
    raw = source.read_bytes()
    if _file_sha256(source) != config.tokenizer_bridge_mapped_parent_outputs_sha256:
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope differs from its explicit digest."
        )
    try:
        envelope = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope is not canonical JSON."
        ) from exc
    if not isinstance(envelope, Mapping):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope must be an object."
        )
    body = dict(envelope)
    seal = body.pop("envelope_sha256", None)
    if (
        body.get("schema")
        != TOKENIZER_BRIDGE_MAPPED_OUTPUT_ENVELOPE_SCHEMA
        or seal != canonical_sha256(body)
        or _canonical_json(envelope) + b"\n" != raw
    ):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope seal or encoding is invalid."
        )
    expected_bindings = {
        "parent_cache_key": parent_cache.cache_key,
        "parent_state_fingerprint": parent_cache.parent_fingerprint,
        "original_parent_baseline_fingerprint": original_baseline.fingerprint,
        "original_parent_baseline_envelope_sha256": (
            original_baseline.envelope_sha256
        ),
        "bridge_artifact_sha256": (
            tokenizer_bridge.artifact.artifact_sha256
        ),
        "transport_profile_sha256": (
            tokenizer_bridge.artifact.profile_fingerprint
        ),
        "trace_adapter_evidence_sha256": tokenizer_bridge.evidence_sha256,
        "canonical_target_tokenization_evidence_sha256": (
            tokenizer_bridge.canonical_tokenization_evidence_sha256
        ),
        "parent_tokenizer_semantics_sha256": (
            tokenizer_bridge.parent_tokenizer_semantics_sha256
        ),
        "target_tokenizer_semantics_sha256": (
            tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
    }
    if any(body.get(name) != value for name, value in expected_bindings.items()):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior envelope differs from the live parent, "
            "bridge, trace evidence, or original baseline."
        )
    expected_generation = {
        **GENERATION_PROTOCOL,
        "max_new_tokens": original_baseline.generation_limit,
    }
    expected_input_contract = {
        "source_text_encoded_by": "sealed-target-tokenizer",
        "target_ids_mapped_to_parent_by": "strict-one-row-transport-profile",
        "parent_input_ids_are_donor_canonical_tokenization": False,
        "parent_input_ids_are_mapped_target_tokenization": True,
        "numeric_token_id_equality_required": False,
    }
    expected_output_contract = {
        "model": "original-parent",
        "generation_ids": "donor-vocabulary",
        "decoded_by": "original-donor-tokenizer",
        "semantic_scoring_not_exact_token_equality": True,
    }
    if (
        body.get("generation") != expected_generation
        or body.get("input_execution_contract") != expected_input_contract
        or body.get("output_execution_contract") != expected_output_contract
        or body.get("parent_outputs_generated_once") is not True
        or body.get("parent_forward_passes_after_seal") != 0
    ):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior execution contract is incomplete or "
            "ambiguous."
        )
    rows = body.get("rows")
    if not isinstance(rows, list) or len(rows) != len(original_baseline.cases):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior rows do not cover the original baseline."
        )
    by_id = {}
    behavior_input_rows = []
    mapping = (
        tokenizer_bridge.artifact.transport_profile
        .target_to_donor_row_indices
    )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent behavior row {index} is not an object."
            )
        case_id = row.get("id")
        prompt = row.get("prompt")
        output = row.get("mapped_parent_output_text")
        token_ids = row.get("mapped_parent_output_ids")
        target_prompt_input_ids = row.get("target_prompt_input_ids")
        mapped_parent_input_ids = row.get("mapped_parent_input_ids")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in by_id
            or not isinstance(prompt, str)
            or not isinstance(output, str)
            or not isinstance(token_ids, list)
            or not token_ids
            or len(token_ids) > original_baseline.generation_limit
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value
                >= tokenizer_bridge.artifact.transport_profile.donor_row_capacity
                for value in token_ids
            )
            or not isinstance(target_prompt_input_ids, list)
            or not target_prompt_input_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value >= len(mapping)
                for value in target_prompt_input_ids
            )
            or not isinstance(mapped_parent_input_ids, list)
            or not mapped_parent_input_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                or value
                >= tokenizer_bridge.artifact.transport_profile.donor_row_capacity
                for value in mapped_parent_input_ids
            )
        ):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent behavior row {index} is malformed."
            )
        canonical_target_ids = _tokenizer_input_ids(
            tokenizer_bridge.target_tokenizer,
            prompt,
        )
        canonical_mapped_ids = tuple(
            mapping[token_id] for token_id in canonical_target_ids
        )
        if (
            tuple(target_prompt_input_ids) != canonical_target_ids
            or tuple(mapped_parent_input_ids) != canonical_mapped_ids
        ):
            raise ProductionLfm2CatalogError(
                f"Mapped-parent behavior row {index} does not prove the "
                "canonical target-tokenized one-row mapped parent input."
            )
        behavior_input_rows.append(
            {
                "id": case_id,
                "prompt_sha256": hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest(),
                "target_prompt_input_ids": list(canonical_target_ids),
                "mapped_parent_input_ids": list(canonical_mapped_ids),
            }
        )
        by_id[case_id] = {
            "prompt": prompt,
            "output": output,
            "token_ids": tuple(token_ids),
        }
    if set(by_id) != {case.case_id for case in original_baseline.cases}:
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior case IDs differ from the original baseline."
        )
    behavior_input_evidence_sha256 = canonical_sha256(
        {
            "schema": (
                "neural-forge-production-lfm2-bridge-behavior-inputs-v1"
            ),
            "bridge_artifact_sha256": (
                tokenizer_bridge.artifact.artifact_sha256
            ),
            "target_tokenizer_semantics_sha256": (
                tokenizer_bridge.target_tokenizer_semantics_sha256
            ),
            "rows": behavior_input_rows,
            "target_tokenization_is_canonical": True,
            "mapped_ids_are_one_row_transport": True,
            "donor_canonical_sequence_equivalence_claimed": False,
            "model_forwards": 0,
        }
    )
    if body.get("canonical_behavior_input_evidence_sha256") != (
        behavior_input_evidence_sha256
    ):
        raise ProductionLfm2CatalogError(
            "Mapped-parent behavior input evidence differs from the sealed "
            "target tokenizer or transport profile."
        )
    mapped_cases = []
    semantic_regressions = []
    original_relevant = 0
    mapped_relevant = 0
    retained_relevant = 0
    original_nonempty = 0
    mapped_nonempty = 0
    retained_nonempty = 0
    original_correct = 0
    mapped_correct = 0
    retained_correct = 0
    correctness_scored = 0
    new_saturations = 0
    repetition_regressions = 0
    for case in original_baseline.cases:
        mapped = by_id[case.case_id]
        if mapped["prompt"] != case.prompt:
            raise ProductionLfm2CatalogError(
                f"Mapped-parent prompt differs for {case.case_id!r}."
            )
        output = str(mapped["output"])
        token_ids = tuple(mapped["token_ids"])
        original_task = original_baseline.task_score(
            case,
            case.parent_output,
        )
        mapped_task = original_baseline.task_score(case, output)
        original_relevant += int(original_task is True)
        mapped_relevant += int(mapped_task is True)
        retained_relevant += int(
            original_task is True and mapped_task is True
        )
        if original_task is True and mapped_task is not True:
            semantic_regressions.append(case.case_id)
        original_nonempty += int(bool(case.parent_output.strip()))
        mapped_nonempty += int(bool(output.strip()))
        retained_nonempty += int(
            bool(case.parent_output.strip()) and bool(output.strip())
        )
        original_correctness = original_baseline.correctness_score(
            case,
            case.parent_output,
        )
        mapped_correctness = original_baseline.correctness_score(case, output)
        if (
            original_correctness is not None
            and mapped_correctness is not None
        ):
            correctness_scored += 1
            original_correct += int(original_correctness)
            mapped_correct += int(mapped_correctness)
            retained_correct += int(
                original_correctness is True
                and mapped_correctness is True
            )
        repeated = _maximum_repeated_bigram_share(token_ids)
        if (
            repeated
            > case.parent_metadata.maximum_repeated_ngram_share
            + config.whole_output_maximum_repeated_bigram_share_delta
        ):
            repetition_regressions += 1
        if (
            len(token_ids) >= original_baseline.generation_limit
            and case.parent_metadata.generated_token_count
            < original_baseline.generation_limit
        ):
            new_saturations += 1
        mapped_cases.append(
            replace(
                case,
                parent_output=output,
                parent_fragment_like=(
                    len(output.strip())
                    < config.whole_output_fragment_minimum_characters
                    and mapped_task is not True
                ),
                parent_metadata=WholeOutputMetadata(
                    generated_token_count=len(token_ids),
                    generation_limit=original_baseline.generation_limit,
                    maximum_repeated_ngram_share=repeated,
                ),
            )
        )
    task_retention = (
        1.0
        if original_relevant == 0
        else retained_relevant / original_relevant
    )
    nonempty_retention = (
        1.0
        if original_nonempty == 0
        else retained_nonempty / original_nonempty
    )
    correctness_retention = (
        1.0
        if original_correct == 0
        else retained_correct / original_correct
    )
    counterfactual_numeric_passed = (
        task_retention
        >= config.tokenizer_bridge_minimum_parent_task_retention
        and nonempty_retention
        >= config.tokenizer_bridge_minimum_parent_task_retention
        and correctness_retention
        >= config.tokenizer_bridge_minimum_parent_task_retention
        and repetition_regressions
        <= config.whole_output_maximum_repeated_ngram_share_regressions
    )
    passed = (
        counterfactual_numeric_passed
        or config.final_performance_metrics_diagnostic_only
    )
    report_body = {
        "schema": TOKENIZER_BRIDGE_NORMALIZATION_CONTROL_SCHEMA,
        "passed": passed,
        "counterfactual_numeric_passed": counterfactual_numeric_passed,
        "performance_metrics_diagnostic_only": (
            config.final_performance_metrics_diagnostic_only
        ),
        "numeric_decision_authority": (
            not config.final_performance_metrics_diagnostic_only
        ),
        "manual_intelligence_review_required": (
            config.final_performance_metrics_diagnostic_only
        ),
        "minimum_parent_task_retention": (
            config.tokenizer_bridge_minimum_parent_task_retention
        ),
        "original_parent_baseline_fingerprint": (
            original_baseline.fingerprint
        ),
        "mapped_parent_envelope_sha256": seal,
        "mapped_parent_file_sha256": _file_sha256(source),
        "bridge_artifact_sha256": (
            tokenizer_bridge.artifact.artifact_sha256
        ),
        "transport_profile_sha256": (
            tokenizer_bridge.artifact.profile_fingerprint
        ),
        "trace_adapter_evidence_sha256": tokenizer_bridge.evidence_sha256,
        "canonical_target_tokenization_evidence_sha256": (
            tokenizer_bridge.canonical_tokenization_evidence_sha256
        ),
        "canonical_behavior_input_evidence_sha256": (
            behavior_input_evidence_sha256
        ),
        "task_score_retention": task_retention,
        "parent_nonempty_retention": nonempty_retention,
        "correctness_retention": correctness_retention,
        "original_task_relevant_count": original_relevant,
        "mapped_task_relevant_count": mapped_relevant,
        "retained_parent_task_relevant_count": retained_relevant,
        "correctness_scored_case_count": correctness_scored,
        "original_correct_case_count": original_correct,
        "mapped_correct_case_count": mapped_correct,
        "retained_parent_correct_case_count": retained_correct,
        "original_nonempty_case_count": original_nonempty,
        "mapped_nonempty_case_count": mapped_nonempty,
        "retained_parent_nonempty_case_count": retained_nonempty,
        "semantic_regression_case_ids": semantic_regressions,
        "new_generation_limit_saturations": new_saturations,
        "generation_limit_saturations_are_censored": True,
        "generation_limit_saturations_have_acceptance_authority": False,
        "repeated_bigram_share_regressions": repetition_regressions,
        "original_parent_input_mode": "donor-tokenizer-canonical",
        "mapped_parent_input_mode": (
            "target-tokenizer-canonical-then-one-row-mapped-to-donor"
        ),
        "donor_canonical_sequence_equivalence_claimed": False,
        "semantic_scoring_not_exact_token_equality": True,
        "parent_model_forwards_after_seal": 0,
    }
    report = MappingProxyType(
        {
            **report_body,
            "control_sha256": canonical_sha256(report_body),
        }
    )
    if not passed:
        raise ProductionLfm2CatalogError(
            "Tokenizer bridge normalization control rejected before folding: "
            f"task_retention={task_retention:.6f}, "
            f"nonempty_retention={nonempty_retention:.6f}, "
            f"correctness_retention={correctness_retention:.6f}, "
            f"new_saturations={new_saturations}, "
            f"repetition_regressions={repetition_regressions}."
        )
    mapped_baseline = Lfm2CachedWholeOutputBaseline(
        source_path=str(source),
        source_sha256=_file_sha256(source),
        envelope_sha256=str(seal),
        parent_cache_key=parent_cache.cache_key,
        parent_state_fingerprint=parent_cache.parent_fingerprint,
        tokenizer_fingerprint=(
            tokenizer_bridge.target_tokenizer_semantics_sha256
        ),
        cases=tuple(mapped_cases),
        contracts=original_baseline.contracts,
        generation_limit=original_baseline.generation_limit,
    )
    return mapped_baseline, report


class BridgeDualWholeOutputBehaviorEvaluator:
    """Require mapped-parent fidelity and original-parent capability retention."""

    def __init__(
        self,
        *,
        original_evaluator: Lfm2WholeOutputBehaviorEvaluator,
        mapped_evaluator: Lfm2WholeOutputBehaviorEvaluator,
        normalization_control: Mapping[str, Any],
    ) -> None:
        self.original_evaluator = original_evaluator
        self.mapped_evaluator = mapped_evaluator
        self.normalization_control = MappingProxyType(
            dict(normalization_control)
        )
        self.events: list[dict[str, Any]] = []
        original_binding = resolve_behavior_gate_binding(original_evaluator)
        mapped_binding = resolve_behavior_gate_binding(mapped_evaluator)
        baseline_fingerprint = canonical_sha256(
            {
                "original_parent": original_binding.baseline_fingerprint,
                "mapped_parent": mapped_binding.baseline_fingerprint,
                "normalization_control": self.normalization_control[
                    "control_sha256"
                ],
            }
        )
        manifest_body = {
            "schema": TOKENIZER_BRIDGE_DUAL_BEHAVIOR_SCHEMA,
            "gate_id": (
                "neural_forge.lfm2.tokenizer_bridge_dual_whole_output"
            ),
            "gate_version": 1,
            "baseline_fingerprint": baseline_fingerprint,
            "components": [
                original_binding.fields(),
                mapped_binding.fields(),
            ],
            "normalization_control": dict(self.normalization_control),
            "acceptance": {
                "mapped_parent_fidelity_required": True,
                "original_parent_retention_required": True,
                "exact_token_equality_required": False,
            },
        }
        manifest_fingerprint = canonical_sha256(manifest_body)
        expose_behavior_gate_binding(
            self,
            BehaviorGateBinding(
                gate_id=manifest_body["gate_id"],
                gate_version=1,
                baseline_fingerprint=baseline_fingerprint,
                manifest_fingerprint=manifest_fingerprint,
                manifest={
                    **manifest_body,
                    "manifest_fingerprint": manifest_fingerprint,
                },
            ),
        )

    def evaluate_model(self, model: Any, *, phase: str) -> FoldDecision:
        mapped = self.mapped_evaluator.evaluate_model(
            model,
            phase=f"{phase}:mapped_parent_fidelity",
        )
        original = self.original_evaluator.evaluate_model(
            model,
            phase=f"{phase}:original_parent_retention",
        )
        metrics = {
            **{
                f"mapped_parent_{name}": value
                for name, value in dict(mapped.metrics).items()
            },
            **{
                f"original_parent_{name}": value
                for name, value in dict(original.metrics).items()
            },
        }
        if mapped.accepted and original.accepted:
            decision = FoldDecision.accept(
                "Mapped-parent fidelity and original-parent retention passed.",
                metrics,
            )
        else:
            decision = FoldDecision.reject(
                "Tokenizer-bridge dual behavior gate rejected: "
                f"mapped={mapped.reason}; original={original.reason}",
                metrics,
            )
        self.events.append(
            {
                "phase": phase,
                "decision": decision.as_dict(),
                "mapped_parent_fidelity": mapped.as_dict(),
                "original_parent_retention": original.as_dict(),
                "normalization_control_sha256": (
                    self.normalization_control["control_sha256"]
                ),
                "exact_token_equality_required": False,
                "parent_forward_count": 0,
            }
        )
        return decision

    def evaluate(self, context: Any) -> FoldDecision:
        return self.evaluate_model(
            context.execution_context.child_model,
            phase="accepted_in_memory_child",
        )

    def evaluate_fresh_reload(self, model: Any) -> FoldDecision:
        return self.evaluate_model(
            model,
            phase="freshly_reloaded_child",
        )


@dataclass(frozen=True, slots=True)
class ProductionParentCachePayload:
    cache: Any
    rows: Any
    calibration_batches: tuple[Mapping[str, torch.Tensor], ...]
    calibration_splits: tuple[str, ...]
    partition: Any
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None
    # ``(base, *on-policy extensions)``.  With no extension evidence this is a
    # pass-through whose cache key, corpus fingerprint, and batch inventory are
    # the base cache's own, so every identity below is unchanged.
    evidence_set: Lfm2SealedEvidenceSet | None = None
    # The base corpus rows alone, for the *selection* consumers (the cached
    # heterogeneous capacity option bank and the global topology frontier).  A
    # child-visited row is a fit row: letting it choose the child's own
    # topology is exactly the frontier capture the all-fit rule forbids.  With
    # no extension evidence this is ``rows`` itself.
    selection_rows: Any = None

    @property
    def base_rows(self) -> Any:
        return self.rows if self.selection_rows is None else self.selection_rows

    @property
    def cache_key(self) -> str:
        if self.evidence_set is not None:
            return str(self.evidence_set.cache_key)
        return str(self.cache.cache_key)

    def verify(self) -> None:
        if self.selective_session is None:
            self.cache.verify()
        else:
            self.selective_session.verify()
            if (
                self.selective_session.cache_key != self.cache.cache_key
                or self.selective_session.path != self.cache.path
                or len(self.selective_session) != len(self.cache)
            ):
                raise ProductionLfm2CatalogError(
                    "Parent cache selective session differs from its payload."
                )
        expected_cache_key = self.cache.cache_key
        expected_corpus_fingerprint = self.cache.corpus_fingerprint
        expected_batch_count = len(self.cache)
        if self.evidence_set is not None and self.evidence_set.has_extension_evidence:
            if self.evidence_set.base_cache is not self.cache:
                raise ProductionLfm2CatalogError(
                    "Sealed evidence set is bound to a different base cache."
                )
            self.evidence_set.verify()
            expected_cache_key = self.evidence_set.cache_key
            expected_corpus_fingerprint = self.evidence_set.corpus_fingerprint
            expected_batch_count = self.evidence_set.batch_count
        if (
            self.rows.cache_key != expected_cache_key
            or self.rows.corpus_fingerprint != expected_corpus_fingerprint
            or len(self.calibration_batches) != expected_batch_count
            or len(self.calibration_splits) != expected_batch_count
        ):
            raise ProductionLfm2CatalogError(
                "Parent evidence payload differs from its immutable cache."
            )
        base_rows = self.base_rows
        if (
            base_rows.cache_key != self.cache.cache_key
            or base_rows.corpus_fingerprint != self.cache.corpus_fingerprint
            or len(base_rows.paths) > len(self.rows.paths)
            or base_rows.paths != self.rows.paths[: len(base_rows.paths)]
        ):
            raise ProductionLfm2CatalogError(
                "The base selection rows are not the sealed base cache prefix "
                "of the merged evidence rows."
            )
        if self.tokenizer_bridge is not None:
            adapter = self.tokenizer_bridge.solve_trace_adapter
            adapter.verify()
            if (
                adapter.record_count != len(self.calibration_batches)
                or tuple(record.split for record in adapter.records)
                != self.calibration_splits
            ):
                raise ProductionLfm2CatalogError(
                    "Tokenizer bridge replay evidence differs from the parent "
                    "cache payload."
                )
            if (
                self.evidence_set is not None
                and self.evidence_set.has_extension_evidence
            ):
                # Merged evidence needs a merged replay: a base-only adapter
                # would leave every extension batch without a target-token
                # child replay, which is exactly the gap this lane fails on.
                self.tokenizer_bridge.verify_merged(self.evidence_set)

    @property
    def child_replay_batches(
        self,
    ) -> tuple[Mapping[str, torch.Tensor], ...]:
        if self.tokenizer_bridge is None:
            return self.calibration_batches
        return self.tokenizer_bridge.solve_trace_adapter.child_replay_batches()


def _production_parent_cache_integrity_check(payload: Any) -> str:
    if not isinstance(payload, ProductionParentCachePayload):
        raise ProductionLfm2CatalogError(
            "Parent cache integrity check received an unknown payload."
        )
    payload.verify()
    return payload.cache_key


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
            payload.child_replay_batches,
            capture_config=self.capture_config,
            batch_splits=payload.calibration_splits,
        )
        cache_reused = child_cache.cache_key in existing
        partition = payload.partition
        evidence_parent_rows = payload.rows
        bridge_alignment = None
        if payload.tokenizer_bridge is not None:
            evidence_parent_rows, bridge_alignment = (
                _bridge_aligned_parent_rows(
                    payload.rows,
                    child_rows,
                    payload.tokenizer_bridge,
                )
            )
        evidence, split = residual.build_cached_layer_evidence(
            evidence_parent_rows,
            child_rows,
            layer_index=request.layer_index,
            selection_indices=partition.selection_indices,
            fit_indices=partition.fit_indices,
            validation_indices=partition.validation_indices,
        )
        child_forwards = (
            0
            if cache_reused
            else _expected_capture_forwards(payload.child_replay_batches)
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
                "tokenizer_bridge_row_alignment": (
                    None
                    if bridge_alignment is None
                    else dict(bridge_alignment)
                ),
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
                payload.child_replay_batches,
                capture_config=self.capture_config,
                batch_splits=payload.calibration_splits,
            )
            cache_reused = child_cache.cache_key in existing
            child_forwards = (
                0
                if cache_reused
                else _expected_capture_forwards(payload.child_replay_batches)
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
                payload.child_replay_batches,
                capture_config=self.capture_config,
                batch_splits=payload.calibration_splits,
            )
            cache_reused = child_cache.cache_key in existing
            child_forwards = (
                0
                if cache_reused
                else _expected_capture_forwards(payload.child_replay_batches)
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
            payload.child_replay_batches,
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
            else _expected_capture_forwards(payload.child_replay_batches)
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
        if (
            int(parent_residual.shape[0])
            != int(child_residual.shape[0])
        ):
            raise ProductionLfm2CatalogError(
                "A mapped parent span endpoint and child residual have "
                f"different row counts at target layer {target_index}."
            )
        parent_shape = [int(size) for size in parent_residual.shape]
        child_shape = [int(size) for size in child_residual.shape]
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
                    "parent_row_shape": parent_shape,
                    "child_row_shape": child_shape,
                    "cross_width": parent_shape != child_shape,
                    "representation_comparison": (
                        "same_width_row_cosine_rmse"
                        if parent_shape == child_shape
                        else "cross_width_linear_cka"
                    ),
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


@dataclass(frozen=True, slots=True)
class _BridgeProjectedParentBatch:
    index: int
    input_sha256: str
    _input_ids: torch.Tensor
    _attention_mask: torch.Tensor
    _forced_decode_ids: torch.Tensor
    _tensors: Mapping[str, torch.Tensor]

    @property
    def input_ids(self) -> torch.Tensor:
        return self._input_ids

    @property
    def attention_mask(self) -> torch.Tensor:
        return self._attention_mask

    @property
    def forced_decode_ids(self) -> torch.Tensor:
        return self._forced_decode_ids

    @property
    def tensors(self) -> Mapping[str, torch.Tensor]:
        return self._tensors


@dataclass(frozen=True, slots=True)
class _BridgeProjectedParentCache:
    cache_key: str
    parent_fingerprint: str
    corpus_fingerprint: str
    _batches: tuple[_BridgeProjectedParentBatch, ...]
    projection_evidence: tuple[Mapping[str, Any], ...]
    projection_evidence_sha256: str

    def __len__(self) -> int:
        return len(self._batches)

    def load_batch(self, index: int) -> _BridgeProjectedParentBatch:
        return self._batches[index]


def _project_bridge_parent_cache(
    parent_cache: Any,
    bridge: ResolvedProductionTokenizerBridge,
    *,
    top_k: int,
    selective_session: Any | None = None,
) -> _BridgeProjectedParentCache:
    # With extension evidence this is the merged cache and the merged replay,
    # so the projected parent batches cover exactly the corpus the child
    # replays.  Without it both are the single sealed handle as before.
    trace_adapter = bridge.solve_trace_adapter
    trace_adapter.verify()
    if selective_session is not None and (
        selective_session.cache_key != parent_cache.cache_key
        or selective_session.path != parent_cache.path
        or len(selective_session) != len(parent_cache)
    ):
        raise ProductionLfm2CatalogError(
            "Projected-cache selective session differs from the parent cache."
        )
    batches = []
    evidence = []
    for record in trace_adapter.records:
        phase_keys = ("cached_prefill.logits",) + tuple(
            f"decode.{step:03d}.logits" for step in range(record.forced_decode_steps)
        )
        selected_payload = None
        if selective_session is None:
            source = parent_cache.load_batch(record.ordinal)
            source_input_sha256 = source.input_sha256
            source_tensors = source.tensors
        else:
            reference = selective_session.batch_references[record.ordinal]
            source_input_sha256 = reference.input_sha256
            selected_payload = selective_session.load_selected_tensors(
                record.ordinal,
                tuple(sorted(phase_keys)),
            )
            source_tensors = selected_payload.tensors
        if source_input_sha256 != record.donor_cache_input_sha256:
            if selected_payload is not None:
                selective_session.release_selected_tensors(selected_payload)
            raise ProductionLfm2CatalogError(
                "Projected parent cache input differs from sealed bridge "
                f"record {record.record_id!r}."
            )
        projected_tensors: dict[str, torch.Tensor] = {
            "input.input_ids": record.target_input_ids,
            "input.attention_mask": record.attention_mask,
            "input.forced_decode_ids": record.target_forced_decode_ids,
        }
        position_ids = record.position_ids
        if position_ids is not None:
            projected_tensors["input.position_ids"] = position_ids
        try:
            for key in phase_keys:
                donor_logits = source_tensors.get(key)
                if donor_logits is None:
                    raise ProductionLfm2CatalogError(
                        "Sealed parent cache lacks full donor-vocabulary logits "
                        f"for {record.record_id!r} phase {key!r}."
                    )
                projection = project_lfm2_cached_parent_logits(
                    trace_adapter,
                    record_id=record.record_id,
                    parent_cache_input_sha256=source_input_sha256,
                    cache_key=parent_cache.cache_key,
                    cache_batch_index=record.ordinal,
                    phase=key.removesuffix(".logits"),
                    donor_logits=donor_logits,
                    top_k=top_k,
                )
                projection.verify(trace_adapter)
                projected_tensors[key] = projection.target_logits
                evidence.append(MappingProxyType(projection.to_evidence_dict()))
        finally:
            if selected_payload is not None:
                selective_session.release_selected_tensors(selected_payload)
        batches.append(
            _BridgeProjectedParentBatch(
                index=record.ordinal,
                input_sha256=record.target_cache_input_sha256,
                _input_ids=record.target_input_ids,
                _attention_mask=record.attention_mask,
                _forced_decode_ids=record.target_forced_decode_ids,
                _tensors=MappingProxyType(projected_tensors),
            )
        )
    evidence_body = {
        "schema": TOKENIZER_BRIDGE_PROJECTED_CACHE_SCHEMA,
        "parent_cache_key": parent_cache.cache_key,
        "bridge_artifact_sha256": bridge.artifact.artifact_sha256,
        "transport_profile_sha256": bridge.artifact.profile_fingerprint,
        "trace_adapter_evidence_sha256": trace_adapter.evidence_sha256,
        "ordered_projections": [dict(value) for value in evidence],
        "projection_count": len(evidence),
        "numeric_token_id_equality_required": False,
        "parent_model_forwards": 0,
        "training_steps": 0,
    }
    return _BridgeProjectedParentCache(
        cache_key=parent_cache.cache_key,
        parent_fingerprint=parent_cache.parent_fingerprint,
        corpus_fingerprint=canonical_sha256(evidence_body),
        _batches=tuple(batches),
        projection_evidence=tuple(evidence),
        projection_evidence_sha256=canonical_sha256(evidence_body),
    )


def _bridge_aligned_parent_rows(
    parent_rows: residual.TraceRowIndex,
    child_rows: residual.TraceRowIndex,
    bridge: ResolvedProductionTokenizerBridge,
) -> tuple[residual.TraceRowIndex, Mapping[str, Any]]:
    trace_adapter = bridge.solve_trace_adapter
    trace_adapter.verify()
    if (
        parent_rows.paths != child_rows.paths
        or parent_rows.path_splits != child_rows.path_splits
        or parent_rows.layer_kinds != child_rows.layer_kinds
    ):
        raise ProductionLfm2CatalogError(
            "Bridge-mapped parent and target-token child traces changed path "
            "geometry, splits, or layer topology."
        )
    for layer_index in range(len(parent_rows.layer_kinds)):
        if int(
            parent_rows.field(layer_index, "residual.post_layer").shape[0]
        ) != int(
            child_rows.field(layer_index, "residual.post_layer").shape[0]
        ):
            raise ProductionLfm2CatalogError(
                "Bridge-mapped parent and child traces changed residual row "
                f"count at layer {layer_index}."
            )
    body = {
        "schema": TOKENIZER_BRIDGE_PARENT_ROWS_SCHEMA,
        "parent_trace_rows_sha256": parent_rows.trace_rows_sha256,
        "child_trace_rows_sha256": child_rows.trace_rows_sha256,
        "bridge_artifact_sha256": bridge.artifact.artifact_sha256,
        "transport_profile_sha256": bridge.artifact.profile_fingerprint,
        "trace_adapter_evidence_sha256": trace_adapter.evidence_sha256,
        "paths_sha256": canonical_sha256(list(child_rows.paths)),
        "splits_sha256": canonical_sha256(list(child_rows.path_splits)),
        "target_corpus_fingerprint": child_rows.corpus_fingerprint,
        "numeric_token_id_equality_required": False,
        "sequence_geometry_authenticated": True,
        "parent_model_forwards": 0,
    }
    aligned = residual.TraceRowIndex(
        paths=child_rows.paths,
        path_splits=child_rows.path_splits,
        layer_kinds=parent_rows.layer_kinds,
        layer_fields=parent_rows.layer_fields,
        cache_key=parent_rows.cache_key,
        corpus_fingerprint=child_rows.corpus_fingerprint,
        trace_rows_sha256=canonical_sha256(body),
    )
    aligned.assert_aligned(child_rows)
    return aligned, MappingProxyType(
        {**body, "aligned_parent_trace_rows_sha256": aligned.trace_rows_sha256}
    )


@dataclass(frozen=True, slots=True)
class _SealedEvidenceParentCacheView:
    """One merged batch view across an ordered sealed evidence set.

    ``evaluate_trajectory_gate`` walks the parent's sealed forced-decode logits
    batch by batch against a freshly captured child cache.  Once extension
    evidence is live the child replays the merged corpus, so the parent side
    has to present the same merged batch order.  Every batch returned here is
    read from an already sealed member cache, so this view performs zero parent
    forwards.
    """

    evidence_set: Lfm2SealedEvidenceSet

    @property
    def cache_key(self) -> str:
        return self.evidence_set.cache_key

    @property
    def parent_fingerprint(self) -> str:
        return self.evidence_set.parent_fingerprint

    @property
    def corpus_fingerprint(self) -> str:
        return self.evidence_set.corpus_fingerprint

    def __len__(self) -> int:
        return self.evidence_set.batch_count

    def load_batch(self, index: int) -> Any:
        provenance = self.evidence_set.batch(index)
        member = self.evidence_set.members[provenance.member_ordinal]
        batch = member.cache.load_batch(provenance.member_batch_index)
        if batch.input_sha256 != provenance.input_sha256:
            raise ProductionLfm2CatalogError(
                f"Sealed evidence batch {index} content differs from its "
                f"member inventory ({provenance.member_role})."
            )
        return batch


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

    def _model_trajectory_mapping(
        self,
        model: Any,
    ) -> tuple[
        tuple[tuple[int, ...], ...] | None,
        tuple[str, ...] | None,
        str | None,
    ]:
        if self.ordered_source_spans is not None:
            assert self.target_layer_types is not None
            return (
                self.ordered_source_spans,
                self.target_layer_types,
                "configured_hybrid_mapping",
            )
        config = getattr(model, "config", None)
        encoded_spans = getattr(
            config,
            "_neural_forge_dense_ordered_source_spans",
            None,
        )
        encoded_types = getattr(
            config,
            "_neural_forge_dense_target_layer_types",
            None,
        )
        if encoded_spans is None and encoded_types is None:
            return None, None, None
        if not isinstance(encoded_spans, str) or not isinstance(
            encoded_types,
            str,
        ):
            raise ProductionLfm2CatalogError(
                "Dense child trajectory mapping markers are incomplete."
            )
        try:
            spans = decode_lfm2_source_spans(encoded_spans)
            layer_types = decode_lfm2_layer_types(encoded_types)
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                "Dense child trajectory mapping markers are invalid."
            ) from exc
        return spans, layer_types, "authenticated_dense_child_config"

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
            self.payload.child_replay_batches,
            capture_config=self.capture_config,
            batch_splits=self.payload.calibration_splits,
        )
        cache_reused = cache.cache_key in existing
        trajectory_parent_rows = self.payload.rows
        trajectory_parent_cache: Any = self.payload.cache
        evidence_set = self.payload.evidence_set
        if evidence_set is not None and evidence_set.has_extension_evidence:
            trajectory_parent_cache = _SealedEvidenceParentCacheView(
                evidence_set
            )
        trajectory_mapping = None
        tokenizer_bridge_projection = None
        tokenizer_bridge_row_alignment = None
        if self.payload.tokenizer_bridge is not None:
            # The child replayed the merged corpus, so the projected parent
            # side has to be the merged corpus too; both handles come from the
            # evidence set's own authenticated sessions.
            bridge_cache: Any = self.payload.cache
            bridge_session: Any = self.payload.selective_session
            if evidence_set is not None and evidence_set.has_extension_evidence:
                bridge_cache = evidence_set.merged_cache()
                bridge_session = evidence_set.merged_session()
            trajectory_parent_cache = _project_bridge_parent_cache(
                bridge_cache,
                self.payload.tokenizer_bridge,
                top_k=8,
                selective_session=bridge_session,
            )
            (
                trajectory_parent_rows,
                tokenizer_bridge_row_alignment,
            ) = _bridge_aligned_parent_rows(
                self.payload.rows,
                rows,
                self.payload.tokenizer_bridge,
            )
            tokenizer_bridge_projection = {
                "projection_evidence_sha256": (
                    trajectory_parent_cache.projection_evidence_sha256
                ),
                "projection_count": len(
                    trajectory_parent_cache.projection_evidence
                ),
                "row_alignment": dict(tokenizer_bridge_row_alignment),
                "numeric_token_id_equality_required": False,
            }
        (
            ordered_source_spans,
            target_layer_types,
            mapping_binding,
        ) = self._model_trajectory_mapping(model)
        if ordered_source_spans is not None:
            assert target_layer_types is not None
            trajectory_parent_rows, trajectory_mapping = (
                build_span_mapped_parent_trajectory_rows(
                    trajectory_parent_rows,
                    rows,
                    ordered_source_spans=ordered_source_spans,
                    target_layer_types=target_layer_types,
                )
            )
            trajectory_mapping = MappingProxyType(
                {
                    **dict(trajectory_mapping),
                    "mapping_binding": mapping_binding,
                }
            )
        report = residual.evaluate_trajectory_gate(
            trajectory_parent_cache,
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
        if tokenizer_bridge_projection is not None:
            report = {
                **report,
                "tokenizer_bridge_projection": tokenizer_bridge_projection,
            }
        summary = _gate_summary(report)
        child_forward_count = (
            0
            if cache_reused
            else _expected_capture_forwards(self.payload.child_replay_batches)
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
                "tokenizer_bridge_projection": tokenizer_bridge_projection,
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
        tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None,
        donor_tokenizer: Any | None = None,
        packaged_semantic_source_tokenizer_semantics_sha256: str | None = None,
    ) -> None:
        self.tokenizer_directory = tokenizer_directory
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.device = device
        self.dtype = dtype
        self.tokenizer_bridge = tokenizer_bridge
        self.donor_tokenizer = donor_tokenizer
        self.packaged_semantic_source_tokenizer_semantics_sha256 = (
            None
            if packaged_semantic_source_tokenizer_semantics_sha256 is None
            else _digest(
                packaged_semantic_source_tokenizer_semantics_sha256,
                "packaged_semantic_source_tokenizer_semantics_sha256",
            )
        )
        if (tokenizer_bridge is None) != (donor_tokenizer is None):
            raise ProductionLfm2CatalogError(
                "Bridge-aware artifact IO requires both the resolved bridge "
                "and its live donor tokenizer."
            )
        if (
            self.packaged_semantic_source_tokenizer_semantics_sha256 is not None
            and tokenizer_bridge is None
        ):
            raise ProductionLfm2CatalogError(
                "Packaged semantic-source tokenizer verification requires the "
                "strict target tokenizer bridge."
            )

    def save(self, model: Any, destination: Path) -> None:
        residual.atomic_save_local_checkpoint(
            model,
            destination,
            output_root=destination.parent,
        )
        if self.tokenizer_bridge is None:
            relative_paths = tuple(
                name
                for name in _TOKENIZER_ARTIFACTS
                if (self.tokenizer_directory / name).is_file()
            )
        else:
            self.tokenizer_bridge.artifact.verify_files()
            # The sealed bridge manifest stays external provenance: its
            # donor-named seal keys are forbidden inside a standalone child
            # package, so only the sealed tokenizer files travel with the
            # child plus a sanitized bridge reference (the pattern the one
            # successfully published bridge child already used).
            relative_paths = tuple(
                item.relative_path
                for item in self.tokenizer_bridge.artifact.target_files
            )
        for relative_path in relative_paths:
            source = self.tokenizer_directory / relative_path
            if not source.is_file() or source.is_symlink():
                raise ProductionLfm2CatalogError(
                    "Selected tokenizer artifact is missing or unsafe: "
                    f"{relative_path}."
                )
            target = destination / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        observed = fingerprint_local_tokenizer(destination)
        if observed != self.tokenizer_fingerprint:
            raise ProductionLfm2CatalogError(
                "Packaged tokenizer differs from the selected local interface."
            )
        if self.tokenizer_bridge is not None:
            artifact = self.tokenizer_bridge.artifact
            self._verify_packaged_bridge_files(
                destination,
                artifact,
                context="Packaged",
            )
            loaded_tokenizer = residual.load_local_tokenizer(destination)
            artifact.verify_live_tokenizers(
                self.donor_tokenizer,
                loaded_tokenizer,
            )
            _write_atomic_json(
                destination / BRIDGE_REFERENCE_FILENAME,
                {
                    "schema": TOKENIZER_BRIDGE_REFERENCE_SCHEMA,
                    "bridge_artifact_sha256": artifact.artifact_sha256,
                    "bridge_profile_sha256": artifact.profile_fingerprint,
                    "target_tokenizer_semantics_sha256": (
                        artifact.target_tokenizer_semantics_sha256
                    ),
                    "target_tokenizer_package_fingerprint": (
                        strict_tokenizer_bridge_target_package_fingerprint(
                            artifact
                        )
                    ),
                    "strict_bridge_manifest_is_external_provenance": True,
                },
            )

    @staticmethod
    def _verify_packaged_bridge_files(
        root: Path,
        artifact: StrictTokenizerBridgeArtifact,
        *,
        context: str,
    ) -> None:
        for record in artifact.target_files:
            packaged = root / record.relative_path
            if (
                not packaged.is_file()
                or packaged.is_symlink()
                or packaged.stat().st_size != record.size_bytes
                or _file_sha256(packaged) != record.sha256
            ):
                raise ProductionLfm2CatalogError(
                    f"{context} tokenizer file differs from the sealed "
                    f"bridge: {record.relative_path}."
                )

    def load(self, source: Path) -> Any:
        observed = fingerprint_local_tokenizer(source)
        if observed != self.tokenizer_fingerprint:
            raise ProductionLfm2CatalogError(
                "Reloaded tokenizer differs from the selected local interface."
            )
        if self.tokenizer_bridge is not None:
            semantic_source_tokenizer = self.donor_tokenizer
            if (
                self.packaged_semantic_source_tokenizer_semantics_sha256
                is not None
            ):
                semantic_source_tokenizer = (
                    load_packaged_lfm2_semantic_source_lexicon(
                        source,
                        expected_tokenizer_semantics_sha256=(
                            self
                            .packaged_semantic_source_tokenizer_semantics_sha256
                        ),
                    ).tokenizer
                )
            artifact = self.tokenizer_bridge.artifact
            self._verify_packaged_bridge_files(
                source,
                artifact,
                context="Reloaded",
            )
            reference_path = source / BRIDGE_REFERENCE_FILENAME
            try:
                reference = json.loads(reference_path.read_bytes())
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProductionLfm2CatalogError(
                    "Packaged tokenizer bridge reference is missing or "
                    "invalid."
                ) from exc
            if (
                not isinstance(reference, Mapping)
                or reference.get("schema") != TOKENIZER_BRIDGE_REFERENCE_SCHEMA
                or reference.get("bridge_artifact_sha256")
                != artifact.artifact_sha256
                or reference.get("bridge_profile_sha256")
                != artifact.profile_fingerprint
                or reference.get("target_tokenizer_semantics_sha256")
                != artifact.target_tokenizer_semantics_sha256
            ):
                raise ProductionLfm2CatalogError(
                    "Packaged tokenizer bridge reference differs from the "
                    "sealed artifact."
                )
            reloaded_tokenizer = residual.load_local_tokenizer(source)
            artifact.verify_live_tokenizers(
                semantic_source_tokenizer,
                reloaded_tokenizer,
            )
        model = load_lfm2_direct_local_model(
            source,
            device=self.device,
            dtype=self.dtype,
        )
        for parameter in model.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        # Restore the lossy serialized block_ff_dim alias only when the
        # published config.json actually declared it: an unconditional
        # injection fabricates a config key the artifact never published and
        # moves the reloaded config fingerprint away from the compiled child.
        if getattr(model.config, "block_ff_dim", None) is None:
            try:
                declared_config = json.loads(
                    (source / "config.json").read_bytes()
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ProductionLfm2CatalogError(
                    "Reloaded child config.json is missing or invalid."
                ) from exc
            if (
                isinstance(declared_config, Mapping)
                and "block_ff_dim" in declared_config
            ):
                model.config.block_ff_dim = int(
                    model.config.intermediate_size
                )
        return model

    def load_authenticated_canonical_fp32_portable(
        self,
        source: Path,
        *,
        runtime_device: Any,
        runtime_dtype: Any,
    ) -> tuple[Any, Lfm2CanonicalFp32RuntimeRelocationReceipt]:
        """Strictly reload canonical FP32, then relocate only in memory."""

        if self.dtype is not torch.float32:
            raise ProductionLfm2CatalogError(
                "Portable direct-local runtime requires canonical FP32 "
                "artifact provenance."
            )
        canonical_loader = self
        if torch.device(self.device).type != "cpu":
            canonical_loader = DirectLocalArtifactIO(
                tokenizer_directory=self.tokenizer_directory,
                tokenizer_fingerprint=self.tokenizer_fingerprint,
                device="cpu",
                dtype=torch.float32,
                tokenizer_bridge=self.tokenizer_bridge,
                donor_tokenizer=self.donor_tokenizer,
                packaged_semantic_source_tokenizer_semantics_sha256=(
                    self
                    .packaged_semantic_source_tokenizer_semantics_sha256
                ),
            )
        canonical_model = canonical_loader.load(source)
        provenance_fingerprint = canonical_sha256(
            {
                "schema": (
                    "neural-forge-production-lfm2-direct-local-canonical-"
                    "fp32-provenance-v1"
                ),
                "tokenizer_fingerprint": self.tokenizer_fingerprint,
                "canonical_device": "cpu",
                "canonical_dtype": str(self.dtype),
                "tokenizer_bridge_artifact_sha256": (
                    None
                    if self.tokenizer_bridge is None
                    else self.tokenizer_bridge.artifact.artifact_sha256
                ),
                "source": str(Path(source).expanduser().resolve(strict=True)),
            }
        )
        try:
            return relocate_authenticated_lfm2_canonical_fp32_model(
                canonical_model,
                provenance_resource_fingerprint=provenance_fingerprint,
                runtime_device=runtime_device,
                runtime_dtype=runtime_dtype,
                fresh_loaded_runtime_copy=True,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ProductionLfm2CatalogError(
                "Authenticated canonical FP32 direct-local child could not be "
                "relocated to the exact requested runtime policy."
            ) from exc


def _load_verified_portable_direct_runtime(
    artifact_io: DirectLocalArtifactIO,
    source: Path,
    *,
    runtime_device: Any,
    runtime_dtype: torch.dtype,
    expected_unique_parameters: int,
    progress: _ProgressJournal,
) -> tuple[Any, Mapping[str, Any]]:
    """Return and journal only an executable authenticated runtime copy."""

    portable_child, relocation = (
        artifact_io.load_authenticated_canonical_fp32_portable(
            source,
            runtime_device=runtime_device,
            runtime_dtype=runtime_dtype,
        )
    )
    body = MappingProxyType(relocation.as_dict())
    if (
        body["canonical_unique_parameter_elements"]
        != expected_unique_parameters
        or body["runtime_unique_parameter_elements"]
        != expected_unique_parameters
        or body["runtime_dtype_parameter_compaction_credit"] != 0
        or body["runtime_device_parameter_compaction_credit"] != 0
        or body["deterministic_post_cast_state_match"] is not True
        or body["state_names_shapes_and_alias_abi_preserved"] is not True
        or body["executable_forward_smoke"]["logits_finite"] is not True
        or body["executable_forward_smoke"]["use_cache"] is not False
        or body["canonical_disk_artifact_mutated"] is not False
        or body["caller_owned_canonical_copy_mutated"] is not False
    ):
        raise ProductionLfm2CatalogError(
            "Portable direct-local runtime failed executable, state, ABI, "
            "count, or canonical-artifact verification."
        )
    progress.emit(
        "portable_runtime_relocation_completed",
        runtime_device=body["runtime_device"],
        runtime_dtype=body["runtime_dtype"],
        canonical_dense_parameter_elements=expected_unique_parameters,
        counted_total_parameter_elements=expected_unique_parameters,
        parameter_compaction_credit=0,
        carrier_coefficients_remain_fp32=None,
        executable_forward_smoke_sha256=body[
            "executable_forward_smoke"
        ]["smoke_sha256"],
    )
    return portable_child, body


class CarrierAwareDirectLocalArtifactIO:
    """Catalog adapter over the audited standalone carrier child package."""

    def __init__(
        self,
        *,
        base: DirectLocalArtifactIO,
        registration: Lfm2CarrierChildPackageRegistration,
        semantic_source_tokenizer_directory: Path,
        semantic_source_tokenizer_semantics_sha256: str,
        semantic_source_tokenizer: Any,
        target_tokenizer: Any,
    ) -> None:
        self.base = base
        self.registration = registration
        self.delegate = Lfm2CarrierChildCheckpointIO(
            base=base,
            registration=registration,
            semantic_source_tokenizer_directory=(
                semantic_source_tokenizer_directory
            ),
            semantic_source_tokenizer_sha256=(
                semantic_source_tokenizer_semantics_sha256
            ),
            in_memory_semantic_source_tokenizer=(
                semantic_source_tokenizer
            ),
            in_memory_target_tokenizer=target_tokenizer,
            target_tokenizer_loader=residual.load_local_tokenizer,
        )
        self.saver_id = self.delegate.saver_id
        self.loader_id = self.delegate.loader_id
        self.resource_fingerprint = self.delegate.resource_fingerprint
        self._packaged_models: weakref.WeakSet[Any] = weakref.WeakSet()
        self.last_outer_inventory: Mapping[str, Any] | None = None
        self.last_route_package: AuthenticatedPackagedLfm2CarrierRoutes | None = None
        self.last_lexicon_package: (
            AuthenticatedLfm2SemanticSourceLexicon | None
        ) = None

    @property
    def route_artifact_sha256(self) -> str:
        return self.registration.expected_artifact_sha256

    @property
    def expected_target_rows(self) -> int:
        return self.registration.expected_target_rows

    @property
    def expected_hidden_size(self) -> int:
        return self.registration.expected_hidden_size

    @staticmethod
    def _ordinary_root_entries(
        child_directory: Path,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        files: list[str] = []
        directories: list[str] = []
        for path in child_directory.iterdir():
            if path.is_symlink():
                raise ProductionLfm2CatalogError(
                    "Carrier child root contains a symbolic link."
                )
            if path.is_file():
                if getattr(path.stat(), "st_nlink", 1) != 1:
                    raise ProductionLfm2CatalogError(
                        "Carrier child root contains a hard-linked file."
                    )
                files.append(path.name)
            elif path.is_dir():
                directories.append(path.name)
            else:
                raise ProductionLfm2CatalogError(
                    "Carrier child root contains an unsupported entry."
                )
        return tuple(sorted(files)), tuple(sorted(directories))

    def verify_outer_inventory(self, child_directory: Path) -> Mapping[str, Any]:
        child = Path(child_directory).expanduser().resolve(strict=True)
        files, directories = self._ordinary_root_entries(child)
        package = load_authenticated_lfm2_carrier_child_package(child)
        base_records = package.receipt["base_root_file_records"]
        expected_files = tuple(
            sorted(
                {
                    *(str(value["relative_path"]) for value in base_records),
                    CARRIER_CHILD_PACKAGE_RECEIPT,
                }
            )
        )
        expected_directories = tuple(
            sorted(
                (
                    PACKAGED_ROUTE_DIRECTORY,
                    SEMANTIC_SOURCE_LEXICON_DIRECTORY,
                )
            )
        )
        if files != expected_files or directories != expected_directories:
            raise ProductionLfm2CatalogError(
                "Standalone carrier child contains an uncounted outer artifact."
            )
        file_rows = [
            {
                "name": name,
                "size_bytes": int((child / name).stat().st_size),
                "sha256": _file_sha256(child / name),
            }
            for name in files
        ]
        body = {
            "schema": "neural-forge-lfm2-carrier-outer-inventory-v1",
            "root_files": file_rows,
            "resource_directories": list(directories),
            "carrier_child_package_sha256": package.package_sha256,
            "authenticated_child_file_records": list(
                package.receipt["child_file_records"]
            ),
            "inventory_derived_from_authenticated_child_local_receipt": True,
            "process_local_save_history_required": False,
            "only_base_checkpoint_tokenizer_metadata_and_counted_resources": True,
            "only_base_checkpoint_tokenizer_and_counted_resources": True,
            "unregistered_continuous_sidecars": 0,
        }
        sealed = MappingProxyType(
            {**body, "inventory_sha256": canonical_sha256(body)}
        )
        self.last_outer_inventory = sealed
        return sealed

    def save(self, model: Any, destination: Path) -> None:
        self.delegate.save(model, destination)
        package = load_authenticated_lfm2_carrier_child_package(
            destination,
            child_model=model,
        )
        self.last_route_package = package.routes
        self.last_lexicon_package = package.semantic_source_lexicon
        self.verify_outer_inventory(destination)

    def load(self, source: Path) -> Any:
        child = self.delegate.load(source)
        package = load_authenticated_lfm2_carrier_child_package(
            source,
            child_model=child,
        )
        self.verify_outer_inventory(source)
        self._packaged_models.add(child)
        self.last_route_package = package.routes
        self.last_lexicon_package = package.semantic_source_lexicon
        return child

    def has_packaged_runtime(self, model: Any) -> bool:
        return model in self._packaged_models

    def register_in_memory(self, model: Any) -> Any:
        return self.delegate.register_in_memory(model)

    def authenticate_counted_resources(
        self,
        child_directory: Path,
        child_model: Any,
    ) -> tuple[
        AuthenticatedPackagedLfm2CarrierRoutes,
        AuthenticatedLfm2SemanticSourceLexicon,
        Mapping[str, Any],
    ]:
        package = load_authenticated_lfm2_carrier_child_package(
            child_directory,
            child_model=child_model,
        )
        inventory = self.verify_outer_inventory(child_directory)
        return (
            package.routes,
            package.semantic_source_lexicon,
            inventory,
        )


class CarrierRouteSmokeBehaviorEvaluator:
    """Cheap routed ABI proof used by in-memory and both reload checks."""

    gate_id = "neural_forge.lfm2.carrier_route_smoke"
    gate_version = 1

    def __init__(
        self,
        *,
        artifact_io: CarrierAwareDirectLocalArtifactIO,
        semantic_source_tokenizer: Any,
        target_tokenizer: Any,
        prompts: Sequence[Mapping[str, Any]],
        maximum_prompt_tokens: int,
        maximum_new_tokens: int,
        baseline_fingerprint: str,
    ) -> None:
        self.artifact_io = artifact_io
        self.semantic_source_tokenizer = semantic_source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.maximum_prompt_tokens = _positive_int(
            maximum_prompt_tokens,
            "routed_smoke_maximum_prompt_tokens",
        )
        self.maximum_new_tokens = _positive_int(
            maximum_new_tokens,
            "routed_smoke_max_new_tokens",
        )
        smoke = next(
            (
                dict(value)
                for value in prompts
                if "safety" not in str(value.get("category", "")).casefold()
            ),
            None,
        )
        if smoke is None:
            raise ProductionLfm2CatalogError(
                "Carrier route smoke requires a non-safety prompt."
            )
        self.prompts = (MappingProxyType(smoke),)
        self.events: list[dict[str, Any]] = []
        contract = BehaviorInferenceExecutionContract(
            entrypoint=LFM2_SEMANTIC_STEP_GENERATE_ENTRYPOINT,
            required_module_path="model",
        )
        manifest_body = {
            "schema": "neural-forge-lfm2-carrier-route-smoke-gate-v1",
            "gate_id": self.gate_id,
            "gate_version": self.gate_version,
            "baseline_fingerprint": _digest(
                baseline_fingerprint,
                "carrier route smoke baseline fingerprint",
            ),
            "route_artifact_sha256": artifact_io.route_artifact_sha256,
            "prompt_sha256": canonical_sha256(smoke),
            "maximum_prompt_tokens": self.maximum_prompt_tokens,
            "maximum_new_tokens": self.maximum_new_tokens,
            "structural_abi_smoke_only": True,
            "numeric_intelligence_gate": False,
            "safety_behavior_is_separate": True,
            "output_cap_hits_are_censored": True,
            "inference_execution": contract.as_dict(),
        }
        manifest_fingerprint = canonical_sha256(manifest_body)
        expose_behavior_gate_binding(
            self,
            BehaviorGateBinding(
                gate_id=self.gate_id,
                gate_version=self.gate_version,
                baseline_fingerprint=manifest_body["baseline_fingerprint"],
                manifest_fingerprint=manifest_fingerprint,
                manifest={
                    **manifest_body,
                    "manifest_fingerprint": manifest_fingerprint,
                },
            ),
        )

    def _register_in_memory_runtime(self, model: Any) -> None:
        if self.artifact_io.has_packaged_runtime(model):
            return
        self.artifact_io.register_in_memory(model)

    def evaluate_model(self, model: Any, *, phase: str) -> FoldDecision:
        self._register_in_memory_runtime(model)
        report = run_lfm2_carrier_route_smoke(
            model,
            self.prompts,
            maximum_prompt_tokens=self.maximum_prompt_tokens,
            maximum_new_tokens=self.maximum_new_tokens,
        )
        audit = report["operation_audit"]
        observed_counts = (
            report.get("child_unique_parameter_count"),
            report.get("routing_input_fixed_continuous_state_count"),
            report.get("routing_output_fixed_continuous_state_count"),
            report.get("effective_child_parameter_count"),
        )
        expected_counts = (
            LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS,
            LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE,
            LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE,
            LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS,
        )
        if observed_counts != expected_counts:
            raise ProductionLfm2CatalogError(
                "Carrier ABI smoke observed the wrong body, C_in, C_out, or "
                "effective counted total."
            )
        response = report["responses"][0]
        diagnostics = response["diagnostics"]
        summary = {
            "child_body_forward_calls": int(audit["child_body_forward_calls"]),
            "physical_body_parameter_count": int(observed_counts[0]),
            "input_route_state_count": int(
                observed_counts[1]
            ),
            "output_route_state_count": int(
                observed_counts[2]
            ),
            "effective_counted_parameter_count": int(observed_counts[3]),
            "ordinary_generate_calls": int(audit["child_generate_calls"]),
            "ordinary_lm_head_calls": int(audit["child_lm_head_forward_calls"]),
            "output_cap_censored": bool(
                diagnostics["censored_by_output_cap"]
            ),
            "output_cap_censoring_is_failure": False,
            "safety_behavior_evaluated": False,
            "runtime_sha256": report["runtime_sha256"],
        }
        self.events.append(
            {
                "phase": phase,
                "summary": summary,
                "route_smoke_accepted": True,
                "numeric_intelligence_gate": False,
            }
        )
        return FoldDecision.accept(
            "Carrier C_in/C_out and child-body public ABI smoke completed.",
            {
                "child_body_forward_calls": summary["child_body_forward_calls"],
                "physical_body_parameter_count": summary[
                    "physical_body_parameter_count"
                ],
                "input_route_state_count": summary["input_route_state_count"],
                "output_route_state_count": summary["output_route_state_count"],
                "effective_counted_parameter_count": summary[
                    "effective_counted_parameter_count"
                ],
                "output_cap_censored": float(summary["output_cap_censored"]),
                "numeric_intelligence_gate": 0.0,
            },
        )


def _verify_carrier_route_smoke_phase_coverage(
    events: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    phases = tuple(str(value.get("phase")) for value in events)
    in_memory = phases.count("accepted_in_memory_child")
    fresh_reload = phases.count("freshly_reloaded_child")
    if in_memory < 1 or fresh_reload < 2:
        raise ProductionLfm2CatalogError(
            "Carrier route smoke did not execute on the in-memory child, "
            "stage reload, and final reload."
        )
    body = {
        "schema": "neural-forge-lfm2-carrier-smoke-phase-coverage-v1",
        "accepted_in_memory_child_smokes": in_memory,
        "freshly_reloaded_child_smokes": fresh_reload,
        "in_memory_stage_and_final_reload_covered": True,
    }
    return MappingProxyType(
        {**body, "coverage_sha256": canonical_sha256(body)}
    )


def _run_verified_portable_carrier_execution_smoke(
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    *,
    maximum_prompt_tokens: int,
    portable_runtime_receipt: Mapping[str, Any],
    canonical_child_artifact_sha256: str,
    route_state_observer: Callable[[Any], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], str]:
    """Execute one routed child step and derive its runtime review identity."""

    direct = portable_runtime_receipt.get("direct_local_relocation")
    if not isinstance(direct, Mapping):
        raise ProductionLfm2CatalogError(
            "Portable carrier execution has no direct relocation receipt."
        )
    runtime_device = direct.get("runtime_device")
    runtime_dtype = direct.get("runtime_dtype")
    if (
        not isinstance(runtime_device, str)
        or not runtime_device
        or runtime_dtype
        not in {
            "torch.float16",
            "torch.bfloat16",
            "torch.float32",
            "torch.float64",
        }
    ):
        raise ProductionLfm2CatalogError(
            "Portable carrier execution has no exact runtime device/dtype."
        )
    if not callable(route_state_observer):
        raise ProductionLfm2CatalogError(
            "Portable carrier execution requires a fixed route-state observer."
        )
    model_state_before = MappingProxyType(
        fingerprint_lfm2_direct_local_runtime_tensor_state(model)
    )
    route_state_before = route_state_observer(model)
    if (
        not isinstance(route_state_before, Mapping)
        or route_state_before.get("continuous_coefficients_remain_fp32")
        is not True
        or model_state_before.get("state_sha256")
        != direct.get("runtime_state_sha256")
    ):
        raise ProductionLfm2CatalogError(
            "Portable carrier execution began from different model or route "
            "state than its authenticated relocation receipt."
        )
    report = run_lfm2_carrier_route_smoke(
        model,
        prompts,
        maximum_prompt_tokens=maximum_prompt_tokens,
        maximum_new_tokens=1,
    )
    model_state_after = MappingProxyType(
        fingerprint_lfm2_direct_local_runtime_tensor_state(model)
    )
    route_state_after = route_state_observer(model)
    if (
        model_state_after != model_state_before
        or route_state_after != route_state_before
    ):
        raise ProductionLfm2CatalogError(
            "Portable carrier C_in/body/C_out smoke mutated full model or "
            "fixed owned route tensor state."
        )
    audit = report.get("operation_audit")
    sessions = report.get("routing_session_receipt")
    input_session = sessions.get("input") if isinstance(sessions, Mapping) else None
    output_session = (
        sessions.get("output") if isinstance(sessions, Mapping) else None
    )
    if (
        not isinstance(audit, Mapping)
        or not isinstance(input_session, Mapping)
        or not isinstance(output_session, Mapping)
    ):
        raise ProductionLfm2CatalogError(
            "Portable carrier execution smoke omitted body or route evidence."
        )
    body = {
        "schema": "neural-forge-lfm2-portable-carrier-execution-smoke-v1",
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "runtime_state_sha256": _digest(
            direct.get("runtime_state_sha256"),
            "portable runtime state SHA-256",
        ),
        "carrier_runtime_sha256": _digest(
            portable_runtime_receipt.get("carrier_runtime_sha256"),
            "portable carrier runtime SHA-256",
        ),
        "child_body_forward_calls": _positive_int(
            audit.get("child_body_forward_calls"),
            "portable child body forward calls",
        ),
        "input_route_use_count": _positive_int(
            input_session.get("routing_cache_use_count"),
            "portable input route use count",
        ),
        "output_route_use_count": _positive_int(
            output_session.get("routing_cache_use_count"),
            "portable output route use count",
        ),
        "maximum_new_tokens": 1,
        "body_input_and_output_routes_executed": True,
        "model_runtime_state_sha256": model_state_after["state_sha256"],
        "owned_route_tensor_state_sha256": route_state_after["state_sha256"],
        "model_parameters_and_all_buffers_unchanged": True,
        "owned_route_tensors_unchanged": True,
    }
    smoke = MappingProxyType(
        {**body, "smoke_sha256": canonical_sha256(body)}
    )
    review_identity_sha256 = canonical_sha256(
        {
            "schema": (
                "neural-forge-lfm2-portable-runtime-review-identity-v1"
            ),
            "canonical_child_artifact_sha256": _digest(
                canonical_child_artifact_sha256,
                "canonical child artifact SHA-256",
            ),
            "runtime_device": body["runtime_device"],
            "runtime_dtype": body["runtime_dtype"],
            "runtime_state_sha256": body["runtime_state_sha256"],
            "carrier_runtime_sha256": body["carrier_runtime_sha256"],
            "execution_smoke_sha256": smoke["smoke_sha256"],
        }
    )
    return smoke, review_identity_sha256


def _write_routed_manual_dossier(
    *,
    model: Any,
    prompts: Sequence[Mapping[str, Any]],
    config: ProductionLfm2CatalogConfig,
    counted_verification: Mapping[str, Any],
    outer_inventory: Mapping[str, Any],
    route_package: AuthenticatedPackagedLfm2CarrierRoutes,
    lexicon_package: AuthenticatedLfm2SemanticSourceLexicon,
    published_child_artifact_sha256: str,
    portable_runtime_review_identity_sha256: str | None = None,
    progress: _ProgressJournal | None = None,
) -> Mapping[str, Any]:
    if config.routed_manual_dossier_path is None:
        raise ProductionLfm2CatalogError(
            "Carrier-routed review has no dossier destination."
        )
    if config.routed_manual_checkpoint_directory is None:
        raise ProductionLfm2CatalogError(
            "Carrier-routed review has no external checkpoint directory."
        )
    review_progress_stages = frozenset(
        {
            "adaptive-review-initial-row-resolved",
            "adaptive-review-retry-row-resolved",
            "adaptive-review-row-finalized",
            "adaptive-review-complete",
        }
    )

    def relay_review_progress(event: Mapping[str, Any]) -> None:
        if progress is None:
            return
        stage = event.get("stage")
        if stage == "semantic-step-complete":
            step_index = event.get("step_index")
            if (
                isinstance(step_index, int)
                and (
                    (step_index + 1) % 128 == 0
                    or event.get("eos_selected") is True
                    or event.get("generation_limit_selected") is True
                )
            ):
                progress.emit(
                    "manual_review_generation_progress",
                    prompt_id=str(event.get("prompt_id")),
                    generated_semantic_token_count=step_index + 1,
                    ended_with_eos=event.get("eos_selected") is True,
                    generation_limit_reached=(
                        event.get("generation_limit_selected") is True
                    ),
                )
            return
        if stage not in review_progress_stages:
            return
        metadata = {
            str(key): value
            for key, value in event.items()
            if key != "stage"
            and (
                value is None
                or isinstance(value, (bool, int, float, str))
            )
        }
        progress.emit(str(stage).replace("-", "_"), **metadata)

    canonical_published_artifact_sha256 = _digest(
        published_child_artifact_sha256,
        "published_child_artifact_sha256",
    )
    review_checkpoint_directory = config.routed_manual_checkpoint_directory
    if portable_runtime_review_identity_sha256 is not None:
        runtime_identity = _digest(
            portable_runtime_review_identity_sha256,
            "portable_runtime_review_identity_sha256",
        )
        review_checkpoint_directory = (
            review_checkpoint_directory
            / f"portable-runtime-{runtime_identity}"
        )
    routed = run_lfm2_resumable_adaptive_carrier_manual_review(
        model,
        prompts,
        checkpoint_directory=review_checkpoint_directory,
        published_child_artifact_sha256=(
            canonical_published_artifact_sha256
        ),
        maximum_prompt_tokens=config.routed_manual_maximum_prompt_tokens,
        initial_maximum_new_tokens=(
            config.routed_manual_initial_max_new_tokens
        ),
        retry_maximum_new_tokens=config.routed_manual_retry_max_new_tokens,
        progress_callback=(
            None if progress is None else relay_review_progress
        ),
    )
    responses = routed.get("responses")
    if not isinstance(responses, list) or len(responses) != len(prompts):
        raise ProductionLfm2CatalogError(
            "Adaptive routed review did not preserve every manual prompt."
        )
    if (
        routed.get("initial_maximum_new_tokens")
        != LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS
        or routed.get("retry_maximum_new_tokens")
        != LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS
        or routed.get("adaptive_retry_response_count")
        != routed.get("initial_cap_censored_response_count")
    ):
        raise ProductionLfm2CatalogError(
            "Exact-4x manual review did not apply one 2048-token retry to "
            "every and only 1024-token cap hit."
        )
    initial_capped_ids = routed.get("initial_cap_censored_response_ids")
    retry_ids = routed.get("adaptive_retry_response_ids")
    if (
        initial_capped_ids is not None
        or retry_ids is not None
    ) and (
        not isinstance(initial_capped_ids, list)
        or not isinstance(retry_ids, list)
        or initial_capped_ids != retry_ids
        or len(set(str(value) for value in retry_ids)) != len(retry_ids)
    ):
        raise ProductionLfm2CatalogError(
            "Exact-4x manual review retry IDs differ from the 1024-token "
            "cap-censored IDs."
        )
    intelligence_rows = []
    safety_rows = []
    dispositions = []
    for prompt, response in zip(prompts, responses, strict=True):
        if not isinstance(response, Mapping):
            raise ProductionLfm2CatalogError(
                "Adaptive routed review returned a non-object response."
            )
        disposition = lfm2_carrier_manual_review_disposition(prompt, response)
        safe_response = _scanner_safe_route_response(response)
        safe_response["evaluation_disposition"] = dict(disposition)
        dispositions.append(dict(disposition))
        if disposition["safety_prompt"]:
            safety_rows.append(safe_response)
        else:
            intelligence_rows.append(safe_response)
    remaining_capped_ids = [
        str(value["prompt_id"])
        for value in dispositions
        if value["output_cap_censored"]
    ]
    unknown_censored_intelligence_ids = [
        str(value["prompt_id"])
        for value in dispositions
        if value["censored_intelligence_endpoint_is_unknown"]
    ]
    if len(remaining_capped_ids) != routed[
        "remaining_cap_censored_response_count"
    ]:
        raise ProductionLfm2CatalogError(
            "Exact-4x manual dossier cap-censor count changed during policy "
            "classification."
        )
    routed_remaining_ids = routed.get("remaining_cap_censored_response_ids")
    if routed_remaining_ids is not None and routed_remaining_ids != (
        remaining_capped_ids
    ):
        raise ProductionLfm2CatalogError(
            "Exact-4x manual dossier changed the remaining capped prompt IDs."
        )
    body = {
        "schema": LFM2_ROUTED_MANUAL_DOSSIER_SCHEMA,
        "status": "complete-ready-for-manual-intelligence-review",
        "intelligence_review_responses": intelligence_rows,
        "safety_observations": safety_rows,
        "response_evaluation_dispositions": dispositions,
        "intelligence_response_count": len(intelligence_rows),
        "safety_observation_count": len(safety_rows),
        "safety_observations_counted_as_folding_failures": False,
        "safety_observations_counted_as_intelligence_failures": False,
        "initial_output_cap": routed["initial_maximum_new_tokens"],
        "retry_output_cap": routed["retry_maximum_new_tokens"],
        "initial_cap_censored_response_count": routed[
            "initial_cap_censored_response_count"
        ],
        "adaptive_retry_response_count": routed[
            "adaptive_retry_response_count"
        ],
        "remaining_cap_censored_response_count": routed[
            "remaining_cap_censored_response_count"
        ],
        "remaining_cap_censored_response_ids": remaining_capped_ids,
        "unknown_censored_intelligence_response_count": (
            len(unknown_censored_intelligence_ids)
        ),
        "unknown_censored_intelligence_response_ids": (
            unknown_censored_intelligence_ids
        ),
        "routed_review_sha256": routed["review_sha256"],
        "published_child_artifact_sha256": routed[
            "published_child_artifact_sha256"
        ],
        **(
            {}
            if portable_runtime_review_identity_sha256 is None
            else {
                "canonical_published_child_artifact_sha256": (
                    canonical_published_artifact_sha256
                ),
                "portable_runtime_review_identity_sha256": (
                    portable_runtime_review_identity_sha256
                ),
                "portable_runtime_checkpoint_namespace": str(
                    review_checkpoint_directory
                ),
                (
                    "review_checkpoints_keyed_by_runtime_device_dtype_and_state"
                ): True,
            }
        ),
        "checkpoint_directory": str(review_checkpoint_directory),
        "checkpoint_root": str(config.routed_manual_checkpoint_directory),
        "checkpoint_row_receipts": routed["checkpoint_row_receipts"],
        "checkpoint_row_count": routed["checkpoint_row_count"],
        "checkpoint_rows_reused": routed["checkpoint_rows_reused"],
        "checkpoint_rows_executed": routed["checkpoint_rows_executed"],
        "runtime_sha256s": routed["runtime_sha256s"],
        "checkpoint_rows_live_outside_child": True,
        "cap_censored_responses_counted_as_folding_failures": False,
        "adaptive_retry_scope": "only-output-cap-censored-rows",
        "route_package_sha256": route_package.package_sha256,
        "semantic_lexicon_package_sha256": lexicon_package.package_sha256,
        "counted_child_state_verification_sha256": counted_verification[
            "verification_sha256"
        ],
        "outer_inventory_sha256": outer_inventory["inventory_sha256"],
        "ordinary_generate_used": False,
        "automatic_intelligence_verdict_performed": False,
        "manual_whole_response_intelligence_review_is_authoritative": True,
        "numeric_metrics_are_diagnostic_only": True,
        "numeric_metrics_have_acceptance_authority": False,
        "evaluation_policy": {
            "safety_prompts_excluded_from_folding_failure": True,
            "safety_prompts_excluded_from_intelligence_failure": True,
            "initial_output_cap": (
                LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS
            ),
            "retry_output_cap": (
                LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS
            ),
            "retry_scope": "only-initial-output-cap-censored-responses",
            "remaining_output_cap_censored_endpoint": "unknown",
            "remaining_output_cap_censored_is_failure": False,
            "manual_whole_response_intelligence_review_is_authoritative": True,
            "numeric_metrics_are_diagnostic_only": True,
            "numeric_metrics_have_acceptance_authority": False,
            "automatic_intelligence_verdict_performed": False,
        },
    }
    dossier = MappingProxyType(
        {**body, "dossier_sha256": canonical_sha256(body)}
    )
    _write_atomic_json(config.routed_manual_dossier_path, dossier)
    return dossier


@dataclass(frozen=True, slots=True)
class ProductionCatalogWiring:
    system: AutomaticDynamicFoldingSystem
    target_request: DynamicTargetRequest
    target_adapter: Lfm2TargetSynthesisAdapter
    stack_adapter: Lfm2DynamicFoldStackAdapter
    budget_plan: Any | None = None
    heterogeneous_capacity_option_bank: (
        Lfm2CachedHeterogeneousCapacityOptionBank | None
    ) = None
    heterogeneous_global_topology_frontier: (
        Lfm2CachedGlobalTopologyFrontier | None
    ) = None
    planned_target_abi: Mapping[str, Any] | None = None


def _verify_planned_target_abi(
    config: ProductionLfm2CatalogConfig,
    request: DynamicTargetRequest,
    frontier: Lfm2CachedGlobalTopologyFrontier | None,
) -> Mapping[str, Any] | None:
    """Fail before synthesis when a constrained H/V/untied plan drifts."""

    required_hidden_size = config.required_target_hidden_size
    if required_hidden_size is None:
        return None
    if frontier is None:
        raise ProductionLfm2CatalogError(
            "A required target-hidden ABI needs an authenticated global frontier."
        )
    pins = frontier.selected_input.target_pins
    configuration = dict(request.configuration)
    expected_vocab_size = (
        config.carrier_expected_target_rows
        if config.uses_carrier_routes
        else pins.vocab_size
    )
    observed_hidden_size = configuration.get("hidden_size")
    observed_vocab_size = configuration.get("vocab_size")
    tie_word_embeddings = configuration.get("tie_word_embeddings")
    if (
        frontier.required_target_hidden_size != required_hidden_size
        or frontier.candidate_hidden_sizes != (required_hidden_size,)
        or pins.hidden_size != required_hidden_size
        or pins.vocab_size != expected_vocab_size
        or observed_hidden_size != required_hidden_size
        or observed_vocab_size != expected_vocab_size
        or tie_word_embeddings is not False
    ):
        raise ProductionLfm2CatalogError(
            "The planned blank target violated its required H/V/untied ABI "
            "before model materialization."
        )
    body = {
        "schema": "neural-forge-production-lfm2-planned-target-abi-v1",
        "required_hidden_size": required_hidden_size,
        "expected_vocab_size": expected_vocab_size,
        "planned_hidden_size": observed_hidden_size,
        "planned_vocab_size": observed_vocab_size,
        "tie_word_embeddings": tie_word_embeddings,
        "target_pins_sha256": pins.pins_sha256,
        "frontier_sha256": frontier.frontier_sha256,
        "checked_before_model_materialization": True,
        "checked_before_system_fold_started": True,
    }
    return MappingProxyType(
        {**body, "evidence_sha256": canonical_sha256(body)}
    )


def _verify_materialized_target_abi(
    config: ProductionLfm2CatalogConfig,
    model: Any,
    planned_target_abi: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Authenticate the concrete constrained child configuration and tables."""

    required_hidden_size = config.required_target_hidden_size
    if required_hidden_size is None:
        return None
    if planned_target_abi is None:
        raise ProductionLfm2CatalogError(
            "A constrained materialized child requires planned ABI evidence."
        )
    expected_vocab_size = planned_target_abi.get("expected_vocab_size")
    if type(expected_vocab_size) is not int or expected_vocab_size <= 0:
        raise ProductionLfm2CatalogError(
            "Planned target ABI evidence has no valid vocabulary size."
        )
    runtime_config = getattr(model, "config", None)
    get_input = getattr(model, "get_input_embeddings", None)
    get_output = getattr(model, "get_output_embeddings", None)
    if (
        runtime_config is None
        or not callable(get_input)
        or not callable(get_output)
    ):
        raise ProductionLfm2CatalogError(
            "The materialized constrained child lacks its runtime config or tables."
        )
    input_weight = getattr(get_input(), "weight", None)
    output_weight = getattr(get_output(), "weight", None)
    expected_shape = (expected_vocab_size, required_hidden_size)
    if (
        not isinstance(input_weight, torch.Tensor)
        or not isinstance(output_weight, torch.Tensor)
        or tuple(input_weight.shape) != expected_shape
        or tuple(output_weight.shape) != expected_shape
        or input_weight is output_weight
        or input_weight.untyped_storage().data_ptr()
        == output_weight.untyped_storage().data_ptr()
        or getattr(runtime_config, "hidden_size", None) != required_hidden_size
        or getattr(runtime_config, "vocab_size", None) != expected_vocab_size
        or getattr(runtime_config, "tie_word_embeddings", None) is not False
    ):
        raise ProductionLfm2CatalogError(
            "The materialized constrained child violated its required H/V/untied "
            "table ABI."
        )
    body = {
        "schema": "neural-forge-production-lfm2-materialized-target-abi-v1",
        "required_hidden_size": required_hidden_size,
        "expected_vocab_size": expected_vocab_size,
        "runtime_hidden_size": int(runtime_config.hidden_size),
        "runtime_vocab_size": int(runtime_config.vocab_size),
        "tie_word_embeddings": bool(runtime_config.tie_word_embeddings),
        "input_table_shape": list(input_weight.shape),
        "output_table_shape": list(output_weight.shape),
        "input_output_tables_untied": True,
        "planned_target_abi_sha256": planned_target_abi["evidence_sha256"],
    }
    return MappingProxyType(
        {**body, "evidence_sha256": canonical_sha256(body)}
    )


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


def _runtime_config_value(
    runtime_config: Any,
    names: Sequence[str],
    default: Any,
) -> Any:
    for name in names:
        value = getattr(runtime_config, name, None)
        if value is not None:
            return value
    return default


def _build_cached_heterogeneous_selection_input(
    config: ProductionLfm2CatalogConfig,
    *,
    parent_model: Any,
    parent_graph: Any,
    parent_cache: Lfm2CircuitTraceCache,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    tokenizer_fingerprint: str,
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None,
) -> tuple[
    Lfm2HeterogeneousLayoutSelectionInput,
    Lfm2CachedHeterogeneousCapacityOptionBank,
]:
    """Build exact-budget options from cached writes, never layer indices."""

    runtime_config = getattr(parent_model, "config", None)
    if runtime_config is None:
        raise ProductionLfm2CatalogError(
            "Heterogeneous capacity discovery requires the local parent config."
        )
    if tokenizer_bridge is None:
        target_vocab = int(runtime_config.vocab_size)
        target_vocab_evidence_sha256 = tokenizer_fingerprint
        token_bridge_evidence_sha256 = (
            lfm2_identity_tokenizer_evidence_sha256(
                tokenizer_semantics_sha256=tokenizer_fingerprint,
                vocabulary_size=target_vocab,
            )
        )
        special_source = runtime_config
    else:
        target_vocab = (
            tokenizer_bridge.artifact.transport_profile.target_row_capacity
        )
        target_vocab_evidence_sha256 = (
            tokenizer_bridge.target_tokenizer_artifact_fingerprint
        )
        token_bridge_evidence_sha256 = tokenizer_bridge.artifact.artifact_sha256
        special_source = tokenizer_bridge.target_tokenizer
    if config.uses_carrier_routes and (
        tokenizer_bridge is None
        or target_vocab != config.carrier_expected_target_rows
    ):
        raise ProductionLfm2CatalogError(
            "Carrier planning requires the tokenizer bridge target-row capacity "
            f"to equal {config.carrier_expected_target_rows}."
        )
    conv_kernel_size = int(
        _runtime_config_value(runtime_config, ("conv_L_cache",), 3)
    )
    option_bank = build_lfm2_cached_heterogeneous_capacity_option_bank(
        parent_graph,
        parent_cache,
        selective_session=selective_session,
        target_hidden_size=config.dense_heterogeneous_hidden_size,
        target_vocab_size=target_vocab,
        exact_target_unique_parameters=(
            config.dense_heterogeneous_target_parameters
        ),
        conv_kernel_size=conv_kernel_size,
        attention_geometry_risk_enabled=(
            config.dense_attention_geometry_risk_enabled
        ),
        discover_exact_short_conv_writer_ranks=(
            config.dense_heterogeneous_factor_writer_rank_discovery
        ),
    )
    rope_parameters = getattr(runtime_config, "rope_parameters", None)
    rope_theta = _runtime_config_value(
        runtime_config,
        ("rope_theta",),
        (
            rope_parameters.get("rope_theta", 1_000_000.0)
            if isinstance(rope_parameters, Mapping)
            else 1_000_000.0
        ),
    )
    runtime_body = {
        "schema": "neural-forge-lfm2-heterogeneous-runtime-pins-v1",
        "parent_graph_fingerprint": parent_graph.fingerprint,
        "parent_cache_key": parent_cache.cache_key,
        "capacity_option_bank_sha256": option_bank.evidence_sha256,
        "hidden_size": config.dense_heterogeneous_hidden_size,
        "vocab_size": target_vocab,
        "conv_kernel_size": conv_kernel_size,
        "rope_theta": float(rope_theta),
        "norm_eps": float(
            _runtime_config_value(
                runtime_config,
                ("norm_eps", "rms_norm_eps"),
                1e-5,
            )
        ),
        "max_position_embeddings": int(
            _runtime_config_value(
                runtime_config,
                ("max_position_embeddings",),
                128_000,
            )
        ),
        "use_cache": bool(
            _runtime_config_value(runtime_config, ("use_cache",), True)
        ),
        "attention_implementation": "sdpa",
        "block_auto_adjust_ff_dim": False,
        "cached_parent_traces_only": True,
        "hardcoded_layer_indices": False,
        "parent_model_calls": 0,
        "training_steps": 0,
    }
    pins = build_lfm2_heterogeneous_target_pins(
        hidden_size=config.dense_heterogeneous_hidden_size,
        vocab_size=target_vocab,
        conv_kernel_size=runtime_body["conv_kernel_size"],
        rope_theta=runtime_body["rope_theta"],
        norm_eps=runtime_body["norm_eps"],
        max_position_embeddings=runtime_body["max_position_embeddings"],
        use_cache=runtime_body["use_cache"],
        attention_implementation="sdpa",
        bos_token_id=int(
            _runtime_config_value(special_source, ("bos_token_id",), 1)
        ),
        eos_token_id=int(
            _runtime_config_value(special_source, ("eos_token_id",), 7)
        ),
        pad_token_id=int(
            _runtime_config_value(special_source, ("pad_token_id",), 0)
        ),
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=target_vocab_evidence_sha256,
        runtime_evidence_sha256=canonical_sha256(runtime_body),
        token_bridge_evidence_sha256=token_bridge_evidence_sha256,
    )
    return (
        Lfm2HeterogeneousLayoutSelectionInput(
            target_pins=pins,
            layer_options=option_bank.layer_options,
        ),
        option_bank,
    )


def _build_cached_global_heterogeneous_selection_input(
    config: ProductionLfm2CatalogConfig,
    *,
    parent_model: Any,
    parent_graph: Any,
    parent_cache: Lfm2CircuitTraceCache,
    selective_session: Lfm2CircuitTraceSelectiveAuthenticatedSession | None = None,
    retained_parent_rows: Any | None = None,
    defer_evidence_cache_miss: bool = False,
    tokenizer_fingerprint: str,
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None,
    phase_cache_context: Lfm2GlobalTopologyPhaseCacheContext | None = None,
) -> tuple[
    Lfm2HeterogeneousLayoutSelectionInput,
    Lfm2CachedHeterogeneousCapacityOptionBank,
    Lfm2CachedGlobalTopologyFrontier,
]:
    """Select one complete exact child over every catalog-legal hidden width."""

    runtime_config = getattr(parent_model, "config", None)
    if runtime_config is None:
        raise ProductionLfm2CatalogError(
            "Global heterogeneous planning requires the local parent config."
        )
    if tokenizer_bridge is None:
        target_vocab = int(runtime_config.vocab_size)
        target_vocab_evidence_sha256 = tokenizer_fingerprint
        token_bridge_evidence_sha256 = (
            lfm2_identity_tokenizer_evidence_sha256(
                tokenizer_semantics_sha256=tokenizer_fingerprint,
                vocabulary_size=target_vocab,
            )
        )
        special_source = runtime_config
    else:
        target_vocab = (
            tokenizer_bridge.artifact.transport_profile.target_row_capacity
        )
        target_vocab_evidence_sha256 = (
            tokenizer_bridge.target_tokenizer_artifact_fingerprint
        )
        token_bridge_evidence_sha256 = tokenizer_bridge.artifact.artifact_sha256
        special_source = tokenizer_bridge.target_tokenizer
    if config.uses_carrier_routes and (
        tokenizer_bridge is None
        or target_vocab != config.carrier_expected_target_rows
    ):
        raise ProductionLfm2CatalogError(
            "Carrier planning requires the tokenizer bridge target-row capacity "
            f"to equal {config.carrier_expected_target_rows}."
        )
    conv_kernel_size = int(
        _runtime_config_value(runtime_config, ("conv_L_cache",), 3)
    )
    rope_parameters = getattr(runtime_config, "rope_parameters", None)
    rope_theta = _runtime_config_value(
        runtime_config,
        ("rope_theta",),
        (
            rope_parameters.get("rope_theta", 1_000_000.0)
            if isinstance(rope_parameters, Mapping)
            else 1_000_000.0
        ),
    )
    runtime = Lfm2GlobalTargetRuntimePins(
        vocab_size=target_vocab,
        conv_kernel_size=conv_kernel_size,
        rope_theta=float(rope_theta),
        norm_eps=float(
            _runtime_config_value(
                runtime_config,
                ("norm_eps", "rms_norm_eps"),
                1e-5,
            )
        ),
        max_position_embeddings=int(
            _runtime_config_value(
                runtime_config,
                ("max_position_embeddings",),
                128_000,
            )
        ),
        use_cache=bool(
            _runtime_config_value(runtime_config, ("use_cache",), True)
        ),
        attention_implementation="sdpa",
        bos_token_id=int(
            _runtime_config_value(special_source, ("bos_token_id",), 1)
        ),
        eos_token_id=int(
            _runtime_config_value(special_source, ("eos_token_id",), 7)
        ),
        pad_token_id=int(
            _runtime_config_value(special_source, ("pad_token_id",), 0)
        ),
        block_auto_adjust_ff_dim=False,
        target_vocab_evidence_sha256=target_vocab_evidence_sha256,
        token_bridge_evidence_sha256=token_bridge_evidence_sha256,
    )
    frontier_kwargs: dict[str, Any] = {
        "runtime": runtime,
        "exact_target_unique_parameters": (
            config.dense_heterogeneous_target_parameters
        ),
        "required_target_hidden_size": config.required_target_hidden_size,
        "minimum_hidden_size": config.dense_minimum_hidden_size,
        "maximum_functional_rows_per_role": (
            config.dense_heterogeneous_global_functional_rows_per_role
        ),
        "discover_exact_short_conv_writer_ranks": (
            config.dense_heterogeneous_factor_writer_rank_discovery
        ),
    }
    if selective_session is not None:
        frontier_kwargs["selective_session"] = selective_session
    if retained_parent_rows is not None:
        frontier_kwargs["retained_parent_rows"] = retained_parent_rows
    if defer_evidence_cache_miss:
        frontier_kwargs["defer_evidence_cache_miss"] = True
    if phase_cache_context is not None:
        frontier_kwargs["phase_cache_context"] = phase_cache_context
    frontier = build_lfm2_cached_global_topology_frontier(
        parent_graph,
        parent_cache,
        **frontier_kwargs,
    )
    required_hidden_size = config.required_target_hidden_size
    if required_hidden_size is not None:
        selected_pins = frontier.selected_input.target_pins
        if (
            frontier.required_target_hidden_size != required_hidden_size
            or frontier.candidate_hidden_sizes != (required_hidden_size,)
            or selected_pins.hidden_size != required_hidden_size
            or (
                config.uses_carrier_routes
                and selected_pins.vocab_size != config.carrier_expected_target_rows
            )
        ):
            raise ProductionLfm2CatalogError(
                "The authenticated global frontier violated the required "
                "target-hidden ABI before target synthesis: required "
                f"H{required_hidden_size}, selected H{selected_pins.hidden_size}, "
                "exact body budget "
                f"{config.dense_heterogeneous_target_parameters}."
            )
    return (
        frontier.selected_input,
        frontier.selected_capacity_option_bank,
        frontier,
    )


def _global_topology_phase_cache_context(
    config: ProductionLfm2CatalogConfig,
    *,
    evidence_split_sha256: str,
    progress: _ProgressJournal | None = None,
    selective_session_factory: Any | None = None,
) -> Lfm2GlobalTopologyPhaseCacheContext:
    """Build the durable, lazily authenticated production topology cache."""

    def relay(stage: str, details: Mapping[str, Any]) -> None:
        if progress is not None:
            progress.emit(stage, **dict(details))

    return Lfm2GlobalTopologyPhaseCacheContext(
        cache=DynamicFoldPhaseCache(
            config.global_topology_phase_cache_root
        ),
        evidence_split_sha256=_digest(
            evidence_split_sha256,
            "global topology evidence_split_sha256",
        ),
        progress=None if progress is None else relay,
        selective_session_factory=selective_session_factory,
    )


def _balanced_basis_phase_cache_receipt_fields(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    if not config.uses_balanced_basis_phase_cache:
        return {}
    assert config.balanced_basis_phase_cache_root is not None
    assert config.balanced_basis_phase_event_log is not None
    assert config.balanced_basis_phase_hmac_key_id is not None
    return {
        "balanced_basis_phase_cache": {
            "enabled": True,
            "root": str(
                config.balanced_basis_phase_cache_root.expanduser().absolute()
            ),
            "event_log": str(
                config.balanced_basis_phase_event_log.expanduser().absolute()
            ),
            "source_anchor_algorithm": "hmac-sha256",
            "hmac_key_id": config.balanced_basis_phase_hmac_key_id,
            "hmac_key_runtime_supplied": True,
            "hmac_secret_serialized": False,
            **(
                {}
                if config.balanced_basis_existing_phase_key_sha256 is None
                else {
                    "existing_phase_reuse_key_sha256": (
                        config.balanced_basis_existing_phase_key_sha256
                    )
                }
            ),
        }
    }


def _balanced_basis_phase_anchor_authority(
    config: ProductionLfm2CatalogConfig,
) -> PhaseAnchorAuthority | None:
    if not config.uses_balanced_basis_phase_cache:
        return None
    assert config.balanced_basis_phase_hmac_key_file is not None
    assert config.balanced_basis_phase_hmac_key_id is not None
    secret = _read_balanced_basis_phase_hmac_key(
        config.balanced_basis_phase_hmac_key_file
    )
    try:
        return PhaseAnchorAuthority(
            key_id=config.balanced_basis_phase_hmac_key_id,
            secret=secret,
        )
    finally:
        del secret


def _balanced_basis_phase_cache_context(
    config: ProductionLfm2CatalogConfig,
    *,
    configuration_sha256: str,
    anchor_authority: PhaseAnchorAuthority,
    progress: _ProgressJournal | None = None,
    dependency_provider: Callable[[], Mapping[str, str]] | None = None,
) -> Lfm2BalancedBasisPhaseCacheContext:
    """Build one source-authenticated, per-run observable basis cache."""

    if not config.uses_balanced_basis_phase_cache:
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase cache context requires the opt-in config."
        )
    assert config.balanced_basis_phase_cache_root is not None
    assert config.balanced_basis_phase_event_log is not None
    assert config.balanced_basis_phase_hmac_key_id is not None
    if anchor_authority.key_id != config.balanced_basis_phase_hmac_key_id:
        raise ProductionLfm2CatalogError(
            "Balanced-basis phase anchor authority differs from its config."
        )

    def relay(stage: str, details: Mapping[str, Any]) -> None:
        if progress is None:
            return
        metadata = dict(details)
        reserved = sorted(
            set(metadata).intersection(_ProgressJournal._RESERVED_FIELDS)
        )
        if reserved:
            raise ProductionLfm2CatalogError(
                "Balanced-basis phase progress contained reserved runner "
                f"fields: {', '.join(reserved)}."
            )
        if stage == "balanced_basis_row_selection_progress":
            for name, value in _process_memory_progress_metadata().items():
                metadata[name] = value
        progress.emit(stage, **metadata)

    event_path = (
        config.balanced_basis_phase_event_log.expanduser().absolute()
    )
    event_path.parent.mkdir(parents=True, exist_ok=True)
    return Lfm2BalancedBasisPhaseCacheContext(
        cache=DynamicFoldPhaseCache(
            config.balanced_basis_phase_cache_root.expanduser().absolute()
        ),
        anchor_authority=anchor_authority,
        events=FoldPhaseEventLog(
            event_path,
            run_id=f"lfm2-balanced-basis-{_digest(configuration_sha256, 'configuration_sha256')}",
        ),
        progress=None if progress is None else relay,
        dependency_provider=dependency_provider,
        existing_phase_reuse_request=(
            None
            if config.balanced_basis_existing_phase_key_sha256 is None
            else Lfm2ExistingBalancedBasisPhaseReuseRequest(
                phase_key_sha256=(
                    config.balanced_basis_existing_phase_key_sha256
                )
            )
        ),
    )


def _compose_on_policy_phase_dependencies(
    inner: Callable[[], Mapping[str, str]] | None,
    on_policy_evidence_identity_sha256: str | None,
) -> Callable[[], Mapping[str, str]] | None:
    """Bind extension evidence into the balanced-basis phase-cache key.

    A closed-form solve fitted on base-only evidence must never be reused for a
    base+extension request, so every extension cache key and round-manifest
    digest enters the derived phase key through this external dependency.  With
    no extension evidence the provider is returned unchanged and the phase key
    is byte-identical to the pre-extension revision.
    """

    if on_policy_evidence_identity_sha256 is None:
        return inner
    identity = _digest(
        on_policy_evidence_identity_sha256,
        "on_policy_evidence_identity_sha256",
    )

    def provider() -> Mapping[str, str]:
        dependencies = dict({} if inner is None else inner())
        if ON_POLICY_PHASE_DEPENDENCY_NAME in dependencies:
            raise ProductionLfm2CatalogError(
                "On-policy phase dependency name collides with an existing "
                "external phase dependency."
            )
        dependencies[ON_POLICY_PHASE_DEPENDENCY_NAME] = identity
        return dependencies

    return provider


def _authenticated_carrier_phase_dependencies(
    registration: Lfm2CarrierChildPackageRegistration,
    *,
    expected_registration_fingerprint: str,
    expected_route_artifact_sha256: str,
    expected_span_evidence_sha256: str,
    expected_counted_plan_fingerprint: str,
) -> Mapping[str, str]:
    """Freshly authenticate every external carrier dependency boundary."""

    verified = registration.verify()
    current_counted_plan = verified.current_exact4x_counted_plan()
    if (
        verified.fingerprint != expected_registration_fingerprint
        or verified.expected_artifact_sha256 != expected_route_artifact_sha256
        or verified.expected_span_evidence_sha256
        != expected_span_evidence_sha256
        or current_counted_plan.fingerprint
        != expected_counted_plan_fingerprint
    ):
        raise ProductionLfm2CatalogError(
            "Carrier dependencies changed during balanced-basis phase "
            "authentication."
        )
    return MappingProxyType(
        {
            "carrier_registration": expected_registration_fingerprint,
            "carrier_route_artifact": expected_route_artifact_sha256,
            "carrier_span_evidence": expected_span_evidence_sha256,
            "counted_child_state_plan": expected_counted_plan_fingerprint,
        }
    )


def build_production_catalog_wiring(
    config: ProductionLfm2CatalogConfig,
    *,
    parent_cache_payload: ProductionParentCachePayload,
    parent_model: Any | None = None,
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
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None,
    global_topology_phase_cache_context: (
        Lfm2GlobalTopologyPhaseCacheContext | None
    ) = None,
    balanced_basis_phase_cache_context: (
        Lfm2BalancedBasisPhaseCacheContext | None
    ) = None,
    progress: _ProgressJournal | None = None,
) -> ProductionCatalogWiring:
    """Configure the production target and stack catalogs exactly once."""

    target_vocabulary_margin_binding = None
    if config.uses_tokenizer_bridge and config.dense_vocabulary_margin_observability:
        if tokenizer_bridge is None:
            raise ProductionLfm2CatalogError(
                "Cross-tokenizer streaming-v2 vocabulary margins require the "
                "fully resolved target bridge runtime."
            )
        if not config.dense_streaming_vocabulary_margin_v2:
            raise ProductionLfm2CatalogError(
                "Cross-tokenizer vocabulary margins cannot use the legacy "
                "donor-domain path."
            )
        if (
            tokenizer_bridge.artifact.artifact_sha256
            != config.tokenizer_bridge_artifact_sha256
            or tokenizer_bridge.artifact.profile_fingerprint
            != config.tokenizer_bridge_profile_sha256
        ):
            raise ProductionLfm2CatalogError(
                "Resolved tokenizer bridge differs from the explicitly sealed "
                "target vocabulary-margin configuration."
            )
        target_vocabulary_margin_binding = (
            Lfm2TargetTokenizerVocabularyMarginBinding(
                trace_adapter=tokenizer_bridge.solve_trace_adapter,
                runtime_receipt=tokenizer_bridge.receipt_fields(),
            )
        )

    heterogeneous_selection_input = None
    heterogeneous_capacity_option_bank = None
    heterogeneous_global_topology_frontier = None
    if config.dense_heterogeneous_layout:
        if parent_model is None or parent_graph is None:
            raise ProductionLfm2CatalogError(
                "Cached heterogeneous dense planning requires the local parent "
                "model and inspected parent circuit graph."
            )
        selection_kwargs = {
            "parent_model": parent_model,
            "parent_graph": parent_graph,
            "parent_cache": parent_cache_payload.cache,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "tokenizer_bridge": tokenizer_bridge,
        }
        if parent_cache_payload.selective_session is not None:
            selection_kwargs["selective_session"] = (
                parent_cache_payload.selective_session
            )
        if config.dense_heterogeneous_global_hidden_frontier:
            # Selection reads the base corpus only.  Extension rows are fit
            # rows; letting them into the capacity/frontier search would let
            # the child pick its own topology from its own visited states.
            selection_kwargs["retained_parent_rows"] = (
                parent_cache_payload.base_rows
            )
            phase_context = global_topology_phase_cache_context
            if phase_context is None:
                split_sha256 = canonical_sha256(
                    {
                        "schema": (
                            "neural-forge-lfm2-global-topology-splits-v1"
                        ),
                        "parent_cache_key": parent_cache_payload.cache_key,
                        "ordered_splits": list(
                            parent_cache_payload.calibration_splits
                        ),
                    }
                )
                phase_context = _global_topology_phase_cache_context(
                    config,
                    evidence_split_sha256=split_sha256,
                    progress=progress,
                )
            selection_kwargs["phase_cache_context"] = phase_context
            (
                heterogeneous_selection_input,
                heterogeneous_capacity_option_bank,
                heterogeneous_global_topology_frontier,
            ) = _build_cached_global_heterogeneous_selection_input(
                config,
                **selection_kwargs,
            )
        else:
            (
                heterogeneous_selection_input,
                heterogeneous_capacity_option_bank,
            ) = _build_cached_heterogeneous_selection_input(
                config,
                **selection_kwargs,
            )

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
        tokenizer_bridge_artifact=(
            None if tokenizer_bridge is None else tokenizer_bridge.artifact
        ),
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
            saver_id=getattr(
                getattr(model_saver, "__self__", None),
                "saver_id",
                "direct-local-lfm2-safetensors-tokenizer-v1",
            ),
            loader_id=getattr(
                getattr(model_loader, "__self__", None),
                "loader_id",
                "direct-local-lfm2-safetensors-tokenizer-v1",
            ),
            ledger=ledger,
            parent_cache_integrity_check=(_production_parent_cache_integrity_check),
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
                body_reclosure_mode=config.dense_body_reclosure_mode,
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
                streaming_vocabulary_margin_v2=(
                    config.dense_streaming_vocabulary_margin_v2
                ),
                streaming_vocabulary_margin_pair_chunk_size=(
                    config.dense_streaming_vocabulary_margin_pair_chunk_size
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
                closed_loop_adapter_config=(
                    Lfm2DynamicClosedLoopAdapterConfig(
                        performance_metrics_diagnostic_only=(
                            config
                            .dense_closed_loop_performance_metrics_diagnostic_only
                        ),
                        replay_capture_mode=(
                            config
                            .resolved_dense_closed_loop_replay_capture_mode
                        ),
                        attention_consumer_mode=(
                            config.dense_attention_consumer_mode
                        ),
                        attention_score_candidate_gate_enabled=(
                            config
                            .dense_attention_score_candidate_gate_enabled
                        ),
                        attention_score_top_k=(
                            config.dense_attention_score_top_k
                        ),
                        attention_score_margin_floor_fraction=(
                            config
                            .dense_attention_score_margin_floor_fraction
                        ),
                        attention_score_minimum_top1_key_agreement_fraction=(
                            config
                            .dense_attention_score_minimum_top1_key_agreement_fraction
                        ),
                        attention_score_minimum_top_k_pair_order_agreement_fraction=(
                            config
                            .dense_attention_score_minimum_top_k_pair_order_agreement_fraction
                        ),
                        attention_score_minimum_winner_margin_sign_agreement_fraction=(
                            config
                            .dense_attention_score_minimum_winner_margin_sign_agreement_fraction
                        ),
                        attention_score_minimum_relative_margin_floor_retention_fraction=(
                            config
                            .dense_attention_score_minimum_relative_margin_floor_retention_fraction
                        ),
                        attention_score_maximum_softmax_fisher_error=(
                            config
                            .dense_attention_score_maximum_softmax_fisher_error
                        ),
                        attention_score_maximum_js_divergence=(
                            config
                            .dense_attention_score_maximum_js_divergence
                        ),
                    )
                ),
                macro_cell_attractor_gate_config=(
                    None
                    if not config.dense_macro_cell_attractor_gate_enabled
                    else Lfm2MacroCellAttractorGateConfig(
                        maximum_records=(
                            config
                            .dense_macro_cell_attractor_maximum_records
                        ),
                        maximum_steps=(
                            config
                            .dense_macro_cell_attractor_maximum_steps
                        ),
                        maximum_entropy_parent_distance_regression=(
                            config
                            .dense_macro_cell_attractor_maximum_entropy_regression
                        ),
                        maximum_repetition_excess_regression=(
                            config
                            .dense_macro_cell_attractor_maximum_repetition_regression
                        ),
                        maximum_dominant_token_excess_regression=(
                            config
                            .dense_macro_cell_attractor_maximum_dominance_regression
                        ),
                        maximum_eos_position_error_regression=(
                            config
                            .dense_macro_cell_attractor_maximum_eos_regression
                        ),
                        collapse_entropy_threshold=(
                            config
                            .dense_macro_cell_attractor_collapse_entropy_threshold
                        ),
                        collapse_repetition_share_threshold=(
                            config
                            .dense_macro_cell_attractor_collapse_repetition_threshold
                        ),
                        collapse_dominant_token_share_threshold=(
                            config
                            .dense_macro_cell_attractor_collapse_dominance_threshold
                        ),
                        maximum_new_collapsed_records=(
                            config
                            .dense_macro_cell_attractor_maximum_new_collapsed_records
                        ),
                    )
                ),
                closed_loop_compiler_config=ClosedLoopCompilerConfig(
                    merge_adjacent_layer_captures=(
                        config
                        .dense_closed_loop_merge_adjacent_layer_captures
                    ),
                ),
                closed_loop_replay_devices=(
                    config.dense_replay_devices or None
                ),
                restart_capsule_directory=(
                    config.dense_restart_capsule_directory
                ),
                restart_capsule_rotation_keep=(
                    config.dense_restart_capsule_keep
                ),
                resume_from_restart_capsule=(
                    config.dense_resume_from_capsule
                ),
            ),
            tokenizer_bridge_artifact=(
                None if tokenizer_bridge is None else tokenizer_bridge.artifact
            ),
            tokenizer_bridge_trace_adapter=(
                None
                if tokenizer_bridge is None
                else tokenizer_bridge.solve_trace_adapter
            ),
            target_tokenizer_vocabulary_margin_binding=(
                target_vocabulary_margin_binding
            ),
            residual_basis_phase_cache_context=(
                balanced_basis_phase_cache_context
            ),
            dense_fold_progress_sink=(
                None
                if progress is None
                else lambda stage, metadata: progress.emit(
                    stage,
                    **dict(metadata),
                )
            ),
            snapshot_config=snapshot,
        )
    )
    topology_catalog_kwargs: dict[str, Any] = {
        "minimum_hidden_size": config.dense_minimum_hidden_size,
        "tokenizer_bridge_artifact": (
            None if tokenizer_bridge is None else tokenizer_bridge.artifact
        ),
    }
    if config.dense_heterogeneous_layout:
        topology_catalog_kwargs.update(
            {
                "target_token_interface_mode": "untied",
                "heterogeneous_layout_selection_input": (
                    heterogeneous_selection_input
                ),
            }
        )
    topology_catalog = Lfm2DenseTargetTopologyCatalog(
        **topology_catalog_kwargs
    )
    system = AutomaticDynamicFoldingSystem(
        target_adapters=DynamicTargetAdapterRegistry((target_adapter,)),
        stack_adapters=DynamicFoldStackAdapterRegistry((stack_adapter,)),
        topology_catalogs=DenseTargetTopologyCatalogRegistry(
            (topology_catalog,)
        ),
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
        budget_request_kwargs: dict[str, Any]
        if config.dense_heterogeneous_layout:
            budget_request_kwargs = {
                "target_unique_parameters": (
                    config.dense_heterogeneous_target_parameters
                ),
                "match_policy": "exact",
            }
        else:
            budget_request_kwargs = {
                "compression_ratio": config.compression_ratio,
            }
        budget_plan = system.plan_budgeted_target(
            parent_graph=parent_graph,
            budget_request=DenseParameterBudgetRequest(
                target_family="lfm2",
                target_modalities=("text",),
                target_vocabulary_size=(
                    None
                    if tokenizer_bridge is None
                    else tokenizer_bridge.artifact.transport_profile.target_row_capacity
                ),
                maximum_candidates=config.dense_maximum_candidates,
                **budget_request_kwargs,
            ),
            child_name=config.child_name,
            prefer_component_risk=config.dense_heterogeneous_layout,
            tokenizer_fingerprint=(
                tokenizer_fingerprint
                if tokenizer_bridge is None
                else tokenizer_bridge.target_tokenizer_semantics_sha256
            ),
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
    planned_target_abi = _verify_planned_target_abi(
        config,
        request,
        heterogeneous_global_topology_frontier,
    )
    return ProductionCatalogWiring(
        system=system,
        target_request=request,
        target_adapter=target_adapter,
        stack_adapter=stack_adapter,
        budget_plan=budget_plan,
        heterogeneous_capacity_option_bank=(
            heterogeneous_capacity_option_bank
        ),
        heterogeneous_global_topology_frontier=(
            heterogeneous_global_topology_frontier
        ),
        planned_target_abi=planned_target_abi,
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


def _dense_closed_loop_performance_metrics_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    diagnostic_only = (
        config.dense_closed_loop_performance_metrics_diagnostic_only
    )
    return {
        "diagnostic_only": diagnostic_only,
        "numeric_decision_authority": not diagnostic_only,
        "counterfactual_thresholds_recorded": True,
        "nonfinite_values_reject": True,
        "structural_and_integrity_failures_reject": True,
        "external_whole_output_behavior_remains_promotion_authority": True,
        "training_steps": 0,
    }


def _dense_closed_loop_replay_policy(
    config: ProductionLfm2CatalogConfig,
    progress: "_ProgressJournal | None" = None,
) -> dict[str, Any]:
    """Record the replay contract the dense closed loop actually resolved to."""

    resolved = config.resolved_dense_closed_loop_replay_capture_mode
    return {
        "policy": config.dense_closed_loop_replay_capture_mode,
        "resolved_mode": resolved,
        "automatic_mode_is_canonical_prefix_replay": True,
        "single_pass_certification_is_explicit_opt_in": True,
        "in_loop_single_pass_certification_replay": (
            resolved == CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE
        ),
        "single_pass_requires_authenticated_heterogeneous_fp32": True,
        "merge_adjacent_layer_captures": (
            config.dense_closed_loop_merge_adjacent_layer_captures
        ),
        # Requested is configuration; admitted is the adapter's decision for
        # this exact child, replay mode, and parent evidence.  Recording only
        # the request would make an inapplicable opt-in look effective.
        "merge_adjacent_layer_captures_observed": (
            None if progress is None else progress.merged_capture_policy()
        ),
        "replay_sharding": _dense_replay_sharding_policy(config, progress),
    }


def _dense_replay_sharding_policy(
    config: ProductionLfm2CatalogConfig,
    progress: "_ProgressJournal | None" = None,
) -> dict[str, Any]:
    """Record the shard request, and what the captures actually did with it.

    The request is a configuration fact; the resolution is a runtime fact that
    only the adapter knows, so it is read back out of the progress journal's
    observed sharding events.  Both halves are recorded because an honoured
    plan and a fallback seal exactly the same bytes and are otherwise
    indistinguishable in the receipt.
    """

    devices = tuple(config.dense_replay_devices)
    policy: dict[str, Any] = {
        "requested_devices": ",".join(devices) if devices else None,
        "requested_device_count": len(devices),
        "request_source": config.dense_replay_devices_source,
        "environment_variable": REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        "serial_default_seals_identical_bytes": True,
        "primary_device_is_worker_zero": True,
        "identical_device_class_required": True,
        "unusable_request_falls_back_to_serial": True,
    }
    observed = None if progress is None else progress.replay_sharding_summary()
    policy["runtime"] = observed
    return policy


def _dense_restart_capsule_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    """Record where accepted-layer restart capsules live for this run."""

    directory = config.dense_restart_capsule_directory
    return {
        "enabled": directory is not None,
        "root": None if directory is None else str(directory),
        "root_explicitly_configured": (
            config.dense_restart_capsule_root is not None
        ),
        "rotation_keep": config.dense_restart_capsule_keep,
        "resume_requested": config.dense_resume_from_capsule is not None,
        "resume_root": (
            None
            if config.dense_resume_from_capsule is None
            else str(
                config.dense_resume_from_capsule.expanduser().absolute()
            )
        ),
        "mid_body_resume_requires_disabled_body_reclosure": True,
        "restart_capsule_is_not_a_promotable_child": True,
    }


def _on_policy_evidence_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    """Record the bounded on-policy round budget, even when it is zero."""

    return {
        "requested_rounds": config.on_policy_evidence_rounds,
        "max_rounds": config.on_policy_max_rounds,
        "consumed_round_manifest_count": len(config.on_policy_round_manifests),
        "refold_after_rounds": config.on_policy_refold_after_rounds,
        "consumes_extension_evidence": bool(
            config.on_policy_round_manifests
        ),
        "extension_evidence_consumer_gap": (
            _unconsumed_extension_evidence_reason(config)
        ),
        "round_root": str(config.on_policy_round_root_directory),
        "fit_prompts": (
            None
            if config.on_policy_fit_prompts is None
            else str(config.on_policy_fit_prompts.expanduser().absolute())
        ),
        "manual_prompt_manifest": (
            None
            if config.on_policy_manual_prompt_manifest is None
            else str(
                config.on_policy_manual_prompt_manifest
                .expanduser()
                .absolute()
            )
        ),
        "rollout": {
            "max_prompts": config.on_policy_max_prompts,
            "max_new_tokens": config.on_policy_max_new_tokens,
            "max_context_tokens": config.on_policy_max_context_tokens,
            "max_capture_batches": config.on_policy_max_capture_batches,
            "seed": config.on_policy_rollout_seed,
            "greedy": True,
        },
        "runs_after_successful_publication": True,
        "all_rows_are_fit_rows": True,
        "review_rows_are_never_on_policy": True,
    }


def _dense_macro_cell_attractor_gate_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    return {
        "enabled": config.dense_macro_cell_attractor_gate_enabled,
        "maximum_records": (
            config.dense_macro_cell_attractor_maximum_records
        ),
        "maximum_steps": config.dense_macro_cell_attractor_maximum_steps,
        "maximum_entropy_parent_distance_regression": (
            config.dense_macro_cell_attractor_maximum_entropy_regression
        ),
        "maximum_repetition_excess_regression": (
            config.dense_macro_cell_attractor_maximum_repetition_regression
        ),
        "maximum_dominant_token_excess_regression": (
            config.dense_macro_cell_attractor_maximum_dominance_regression
        ),
        "maximum_eos_position_error_regression": (
            config.dense_macro_cell_attractor_maximum_eos_regression
        ),
        "collapse_entropy_threshold": (
            config.dense_macro_cell_attractor_collapse_entropy_threshold
        ),
        "collapse_repetition_share_threshold": (
            config.dense_macro_cell_attractor_collapse_repetition_threshold
        ),
        "collapse_dominant_token_share_threshold": (
            config.dense_macro_cell_attractor_collapse_dominance_threshold
        ),
        "maximum_new_collapsed_records": (
            config.dense_macro_cell_attractor_maximum_new_collapsed_records
        ),
        "parent_outputs_cached": True,
        "child_only_free_rollout": True,
        "exact_token_match_is_diagnostic_only": True,
        "training_steps": 0,
    }


def _dense_attention_score_candidate_gate_policy(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    return {
        "enabled": config.dense_attention_score_candidate_gate_enabled,
        "consumer_mode": config.dense_attention_consumer_mode,
        "top_k": config.dense_attention_score_top_k,
        "relative_margin_floor": (
            config.dense_attention_score_margin_floor_fraction
        ),
        "minimum_top1_key_agreement_fraction": (
            config
            .dense_attention_score_minimum_top1_key_agreement_fraction
        ),
        "minimum_top_k_pair_order_agreement_fraction": (
            config
            .dense_attention_score_minimum_top_k_pair_order_agreement_fraction
        ),
        "minimum_winner_margin_sign_agreement_fraction": (
            config
            .dense_attention_score_minimum_winner_margin_sign_agreement_fraction
        ),
        "minimum_relative_margin_floor_retention_fraction": (
            config
            .dense_attention_score_minimum_relative_margin_floor_retention_fraction
        ),
        "maximum_softmax_fisher_error": (
            config.dense_attention_score_maximum_softmax_fisher_error
        ),
        "maximum_js_divergence": (
            config.dense_attention_score_maximum_js_divergence
        ),
        "masked_keys_excluded": True,
        "numeric_weight_matching_used": False,
        "parent_forward_calls": 0,
        "training_steps": 0,
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


def _planned_dense_target_evidence(
    wiring: ProductionCatalogWiring | None,
) -> dict[str, Any] | None:
    if wiring is None or wiring.budget_plan is None:
        return None
    plan = wiring.budget_plan
    rank = plan.target_certificate.selected_candidate_rank
    candidate = plan.frontier.candidates[rank]
    configuration = dict(candidate.configuration)
    option_bank = wiring.heterogeneous_capacity_option_bank
    global_topology_frontier = wiring.heterogeneous_global_topology_frontier
    return {
        "schema": "neural-forge-production-lfm2-planned-dense-target-v1",
        "budget_plan_sha256": plan.fingerprint,
        "frontier_sha256": plan.frontier.fingerprint,
        "topology_certificate_sha256": plan.target_certificate.fingerprint,
        "selected_candidate_rank": rank,
        "selected_candidate_sha256": candidate.fingerprint,
        "selection_policy": plan.target_certificate.selection_policy,
        "source_unique_parameters": candidate.source_unique_parameters,
        "requested_unique_parameters": candidate.requested_unique_parameters,
        "planned_unique_parameters": candidate.unique_parameters,
        "planned_logical_parameters": candidate.logical_parameters,
        "planned_native_tied_parameter_count": (
            candidate.native_tied_parameter_count
        ),
        "planned_parameters_removed": (
            candidate.source_unique_parameters - candidate.unique_parameters
        ),
        "requested_compression_ratio": (
            plan.frontier.request.compression_ratio
        ),
        "planned_compression_ratio": candidate.compression_ratio,
        "budget_delta": candidate.budget_delta,
        "at_or_below_budget": candidate.at_or_below_budget,
        "family": candidate.family,
        "modalities": list(candidate.modalities),
        "target_depth": len(candidate.ordered_source_spans),
        "ordered_source_spans": [
            list(span) for span in candidate.ordered_source_spans
        ],
        "configuration": configuration,
        "target_vocabulary_size": candidate.target_vocabulary_size,
        "required_target_hidden_size": (
            None
            if global_topology_frontier is None
            else global_topology_frontier.required_target_hidden_size
        ),
        "planned_target_abi": (
            None
            if wiring.planned_target_abi is None
            else dict(wiring.planned_target_abi)
        ),
        "tokenizer_bridge_required": candidate.tokenizer_bridge_required,
        "tokenizer_bridge_artifact_sha256": configuration.get(
            TOKENIZER_BRIDGE_ARTIFACT_CONFIG_KEY
        ),
        "tokenizer_transport_profile_sha256": configuration.get(
            TOKENIZER_BRIDGE_PROFILE_CONFIG_KEY
        ),
        "parent_tokenizer_semantics_sha256": configuration.get(
            PARENT_TOKENIZER_SEMANTICS_CONFIG_KEY
        ),
        "target_tokenizer_semantics_sha256": configuration.get(
            CHILD_TOKENIZER_SEMANTICS_CONFIG_KEY
        ),
        "required_compiler_capabilities": list(
            candidate.required_compiler_capabilities
        ),
        "declared_compiler_capabilities": list(
            candidate.declared_compiler_capabilities
        ),
        "missing_compiler_capabilities": list(
            candidate.missing_compiler_capabilities
        ),
        "compiler_executable": candidate.compiler_executable,
        "heterogeneous_capacity_option_bank": (
            None if option_bank is None else dict(option_bank.as_dict())
        ),
        "heterogeneous_global_topology_frontier": (
            None
            if global_topology_frontier is None
            else dict(global_topology_frontier.as_dict())
        ),
        "blank_target_only": True,
        "model_weights_materialized_at_planning": False,
    }


def _prepublication_gate_evidence(
    evaluator: CombinedLfm2FinalEvaluator | None,
    evidence_builder: CachedProductionEvidenceBuilder | None,
) -> dict[str, Any]:
    behavior = None if evaluator is None else evaluator.behavior_evaluator
    return {
        "trajectory_and_combined_events": (
            [] if evaluator is None else evaluator.events
        ),
        "whole_output_events": (
            [] if behavior is None else behavior.events
        ),
        "original_parent_whole_output_events": (
            []
            if behavior is None
            else getattr(
                getattr(behavior, "original_evaluator", None),
                "events",
                [],
            )
        ),
        "mapped_parent_whole_output_events": (
            []
            if behavior is None
            else getattr(
                getattr(behavior, "mapped_evaluator", None),
                "events",
                [],
            )
        ),
        "evidence_builds": (
            [] if evidence_builder is None else evidence_builder.records
        ),
    }


def _load_on_policy_fit_prompts(path: Path) -> tuple[str, ...]:
    """Load the explicit JSON fit-prompt corpus one rollout round replays."""

    resolved = path.expanduser().resolve(strict=True)
    try:
        document: Any = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ProductionLfm2CatalogError(
            f"On-policy fit prompts are not readable JSON: {resolved}."
        ) from exc
    if isinstance(document, Mapping):
        document = document.get("prompts")
    if not isinstance(document, list) or not document:
        raise ProductionLfm2CatalogError(
            "On-policy fit prompts must be a nonempty JSON list of prompt "
            "strings or rows carrying a nonempty 'prompt' string."
        )
    prompts: list[str] = []
    for row in document:
        value = row.get("prompt") if isinstance(row, Mapping) else row
        if not isinstance(value, str) or not value.strip():
            raise ProductionLfm2CatalogError(
                "Every on-policy fit prompt must be a nonempty string."
            )
        prompts.append(value)
    return tuple(prompts)


def _on_policy_round_receipt(round_value: Lfm2OnPolicyEvidenceRound) -> dict[str, Any]:
    """Copy one sealed round's identities; nothing here is recomputed."""

    manifest = round_value.manifest
    return {
        "round_index": round_value.round_index,
        "max_rounds": round_value.max_rounds,
        "manifest_path": str(round_value.manifest_path),
        "manifest_sha256": round_value.manifest_sha256,
        "request_sha256": round_value.request_sha256,
        "extension_cache_path": str(round_value.extension_cache_path),
        "extension_cache_key": round_value.extension_cache_key,
        "extension_batch_count": round_value.batch_count,
        "parent_identity_sha256": round_value.parent_fingerprint,
        "fit_prompts_disjoint_from_manual_review": bool(
            manifest["disjointness"]["verified_disjoint"]
        ),
        "parent_capture_invocations": int(
            manifest["counters"]["parent_capture_invocations"]
        ),
        "optimizer_steps": int(manifest["counters"]["optimizer_steps"]),
        "backward_calls": int(manifest["counters"]["backward_calls"]),
        "training_steps": int(manifest["counters"]["training_steps"]),
        "all_rows_are_fit_rows": True,
        "consumed_by_this_solve_pass": False,
        "parent_cache_input_mode": round_value.parent_cache_input_mode,
        "carries_target_token_preimage": (
            round_value.carries_target_preimage
        ),
        "target_preimage_sha256": (
            None
            if round_value.target_preimage is None
            else str(round_value.target_preimage["target_preimage_sha256"])
        ),
    }


def _run_postpublication_on_policy_rounds(
    config: ProductionLfm2CatalogConfig,
    *,
    parent: Any,
    parent_tokenizer: Any,
    parent_fingerprint: str,
    base_cache_directory: Path,
    capture_config: Lfm2CircuitTraceCaptureConfig,
    published_child_directory: Path,
    consumed_round_manifests: Sequence[Path],
    input_device: Any,
    progress: _ProgressJournal,
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None,
) -> tuple[dict[str, Any], ...]:
    """Run the bounded on-policy extension rounds this publication earned.

    The child was just published, so the round reloads that standalone artifact
    and is the only model executed while rollouts are collected.  The parent is
    executed exactly once per surviving capture batch, inside the single
    sanctioned capture invocation that seals the round's extension cache.  That
    is accounted separately from ``parent_forward_count_after_cache``.
    """

    if config.on_policy_evidence_rounds <= 0:
        return ()
    assert config.on_policy_fit_prompts is not None
    assert config.on_policy_manual_prompt_manifest is not None
    child_directory = (
        config.on_policy_child_artifact_directory.expanduser().resolve(
            strict=True
        )
        if config.on_policy_child_artifact_directory is not None
        else published_child_directory
    )
    prompts = _load_on_policy_fit_prompts(config.on_policy_fit_prompts)
    manual_manifest = (
        config.on_policy_manual_prompt_manifest.expanduser().resolve(
            strict=True
        )
    )
    output_root = config.on_policy_round_root_directory
    output_root.mkdir(parents=True, exist_ok=True)
    rollout = Lfm2OnPolicyRolloutConfig(
        max_prompts=config.on_policy_max_prompts,
        max_new_tokens=config.on_policy_max_new_tokens,
        max_context_tokens=config.on_policy_max_context_tokens,
        seed=config.on_policy_rollout_seed,
        max_capture_batches=config.on_policy_max_capture_batches,
    )
    previous: list[Path] = [
        value.expanduser().resolve(strict=True)
        for value in consumed_round_manifests
    ]
    receipts: list[dict[str, Any]] = []
    first_index = len(previous)
    for offset in range(config.on_policy_evidence_rounds):
        round_index = first_index + offset
        progress.emit(
            "on_policy_evidence_round_started",
            round_index=round_index,
            max_rounds=config.on_policy_max_rounds,
            previous_round_count=len(previous),
            fit_prompt_count=len(prompts),
            child_artifact_directory=str(child_directory),
            round_root=str(output_root),
            tokenizer_bridge_enabled=tokenizer_bridge is not None,
        )
        try:
            executed = run_lfm2_on_policy_evidence_round(
                child_artifact_directory=child_directory,
                parent_model=parent,
                parent_tokenizer=parent_tokenizer,
                parent_fingerprint=parent_fingerprint,
                fit_prompts=prompts,
                manual_prompt_manifest_path=manual_manifest,
                base_cache_directory=base_cache_directory,
                output_root=output_root,
                capture_config=capture_config,
                round_index=round_index,
                max_rounds=config.on_policy_max_rounds,
                rollout=rollout,
                previous_round_manifest_paths=tuple(previous),
                child_device=config.device,
                child_dtype=config.dtype,
                input_device=input_device,
                parent_tokenizer_name=config.parent_name,
                # A bridged lane's rounds are captured on the bridge image of
                # the child's own tokenization and seal that preimage; a later
                # pass cannot replay them otherwise.
                tokenizer_bridge=(
                    None if tokenizer_bridge is None else tokenizer_bridge.artifact
                ),
                expected_tokenizer_bridge_artifact_sha256=(
                    None
                    if tokenizer_bridge is None
                    else config.tokenizer_bridge_artifact_sha256
                ),
            )
        except Exception as exc:
            raise ProductionLfm2CatalogError(
                f"On-policy evidence round {round_index} failed closed: {exc}"
            ) from exc
        receipt_body = _on_policy_round_receipt(executed)
        receipts.append(receipt_body)
        previous.append(executed.manifest_path)
        progress.emit(
            "on_policy_evidence_round_sealed",
            round_index=executed.round_index,
            manifest_sha256=executed.manifest_sha256,
            extension_cache_key=executed.extension_cache_key,
            extension_batch_count=executed.batch_count,
            parent_capture_invocations=(
                receipt_body["parent_capture_invocations"]
            ),
            fit_prompts_disjoint_from_manual_review=(
                receipt_body["fit_prompts_disjoint_from_manual_review"]
            ),
            all_rows_are_fit_rows=True,
            consumed_by_this_solve_pass=False,
            parent_cache_input_mode=executed.parent_cache_input_mode,
            carries_target_token_preimage=(
                executed.carries_target_preimage
            ),
        )
    return tuple(receipts)


def _diagnostics_document(
    evaluator: CombinedLfm2FinalEvaluator | None,
    evidence_builder: CachedProductionEvidenceBuilder | None,
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None,
) -> dict[str, Any]:
    behavior = None if evaluator is None else evaluator.behavior_evaluator
    body = {
        "schema": DIAGNOSTICS_SCHEMA,
        "behavior_gate_events": ([] if evaluator is None else evaluator.events),
        "whole_output_events": (
            [] if behavior is None else behavior.events
        ),
        "original_parent_whole_output_events": (
            []
            if behavior is None
            else getattr(
                getattr(behavior, "original_evaluator", None),
                "events",
                [],
            )
        ),
        "mapped_parent_whole_output_events": (
            []
            if behavior is None
            else getattr(
                getattr(behavior, "mapped_evaluator", None),
                "events",
                [],
            )
        ),
        "evidence_builds": (
            [] if evidence_builder is None else evidence_builder.records
        ),
        "tokenizer_bridge": (
            None
            if tokenizer_bridge is None
            else {
                **tokenizer_bridge.receipt_fields(),
                "paired_input_evidence": [
                    dict(value)
                    for value in tokenizer_bridge.solve_paired_input_evidence
                ],
            }
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
    if config.on_policy_refold_after_rounds:
        # One invocation is one fold pass.  A requested alternation must be
        # driven by the orchestrator that seals each pass, or the flag would be
        # a silent no-op.
        raise ProductionLfm2CatalogError(
            "A fold/round alternation must be driven by "
            "run_production_catalog_on_policy_alternation; "
            "run_production_catalog executes exactly one fold pass."
        )
    configuration_sha256 = canonical_sha256(config.identity())
    progress = _ProgressJournal(configuration_sha256)
    progress.emit(
        "run_started",
        carrier_routes_enabled=config.uses_carrier_routes,
        device=config.device,
        dtype=config.dtype,
        required_target_hidden_size=config.required_target_hidden_size,
        derived_global_topology_cache_root=(
            str(config.global_topology_phase_cache_root)
            if config.dense_heterogeneous_global_hidden_frontier
            else None
        ),
        **(
            {}
            if not config.uses_balanced_basis_phase_cache
            else {
                "balanced_basis_phase_cache_enabled": True,
                "balanced_basis_phase_cache_root": str(
                    config.balanced_basis_phase_cache_root
                    .expanduser()
                    .absolute()
                ),
                "balanced_basis_phase_event_log": str(
                    config.balanced_basis_phase_event_log
                    .expanduser()
                    .absolute()
                ),
                "balanced_basis_phase_hmac_key_id": (
                    config.balanced_basis_phase_hmac_key_id
                ),
            }
        ),
        folding_only=True,
    )
    if config.dense_replay_devices or config.dense_replay_devices_source != (
        REPLAY_DEVICES_SOURCE_UNSET
    ):
        progress.emit(
            "dense_replay_sharding_requested",
            requested_devices=(
                ",".join(config.dense_replay_devices)
                if config.dense_replay_devices
                else None
            ),
            requested_device_count=len(config.dense_replay_devices),
            request_source=config.dense_replay_devices_source,
            serial_default_seals_identical_bytes=True,
        )
    ledger: ExperimentLedger | None = None
    evaluator: CombinedLfm2FinalEvaluator | None = None
    evidence_builder: CachedProductionEvidenceBuilder | None = None
    wiring: ProductionCatalogWiring | None = None
    resolved_parent_cache: ResolvedProductionParentCache | None = None
    active_selective_session: (
        Lfm2CircuitTraceSelectiveAuthenticatedSession | None
    ) = None
    global_topology_phase_context: (
        Lfm2GlobalTopologyPhaseCacheContext | None
    ) = None
    balanced_basis_phase_anchor_authority: PhaseAnchorAuthority | None = None
    balanced_basis_phase_context: (
        Lfm2BalancedBasisPhaseCacheContext | None
    ) = None
    preplanned_global_topology_frontier: (
        Lfm2CachedGlobalTopologyFrontier | None
    ) = None
    global_topology_evidence_deferred = False
    tokenizer_bridge: ResolvedProductionTokenizerBridge | None = None
    tokenizer_bridge_normalization_control: Mapping[str, Any] | None = None
    carrier_source: AuthenticatedLfm2CarrierBridgePublication | None = None
    carrier_registration: Lfm2CarrierChildPackageRegistration | None = None
    carrier_spans: Mapping[str, Any] | None = None
    carrier_prompts: tuple[Mapping[str, Any], ...] | None = None
    carrier_counted_plan: CountedChildStatePlan | None = None
    carrier_artifact_io: CarrierAwareDirectLocalArtifactIO | None = None
    carrier_behavior_evaluator: CarrierRouteSmokeBehaviorEvaluator | None = None
    carrier_smoke_coverage: Mapping[str, Any] | None = None
    carrier_counted_verification: Mapping[str, Any] | None = None
    carrier_outer_inventory: Mapping[str, Any] | None = None
    carrier_route_package: AuthenticatedPackagedLfm2CarrierRoutes | None = None
    carrier_lexicon_package: AuthenticatedLfm2SemanticSourceLexicon | None = None
    carrier_manual_dossier: Mapping[str, Any] | None = None
    portable_runtime_receipt: Mapping[str, Any] | None = None
    portable_runtime_execution_smoke: Mapping[str, Any] | None = None
    portable_runtime_review_identity_sha256: str | None = None
    materialized_target_abi: Mapping[str, Any] | None = None
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
            "configuration_sha256": configuration_sha256,
            "global_topology_phase_cache": {
                "enabled": config.dense_heterogeneous_global_hidden_frontier,
                "root": (
                    str(config.global_topology_phase_cache_root)
                    if config.dense_heterogeneous_global_hidden_frontier
                    else None
                ),
                "required_target_hidden_size": config.required_target_hidden_size,
            },
            **_balanced_basis_phase_cache_receipt_fields(config),
            "closed_loop_performance_metrics_policy": (
                _dense_closed_loop_performance_metrics_policy(config)
            ),
            "closed_loop_replay_policy": (
                _dense_closed_loop_replay_policy(config, progress)
            ),
            "restart_capsule_policy": _dense_restart_capsule_policy(config),
            "on_policy_evidence_policy": _on_policy_evidence_policy(config),
            "vocabulary_margin_candidate_gate_policy": (
                _dense_vocabulary_margin_candidate_gate_policy(config)
            ),
            "macro_cell_attractor_gate_policy": (
                _dense_macro_cell_attractor_gate_policy(config)
            ),
            "attention_score_candidate_gate_policy": (
                _dense_attention_score_candidate_gate_policy(config)
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
        progress.emit(
            "directories_ready",
            diagnostics_enabled=config.diagnostics_path is not None,
            directory_count=(
                6 if config.diagnostics_path is not None else 5
            ),
        )
        balanced_basis_phase_anchor_authority = (
            _balanced_basis_phase_anchor_authority(config)
        )
        if balanced_basis_phase_anchor_authority is not None:
            progress.emit(
                "balanced_basis_phase_cache_runtime_verified",
                cache_root=str(
                    config.balanced_basis_phase_cache_root
                    .expanduser()
                    .absolute()
                ),
                event_log=str(
                    config.balanced_basis_phase_event_log
                    .expanduser()
                    .absolute()
                ),
                hmac_key_id=balanced_basis_phase_anchor_authority.key_id,
            )

        parent_directory = config.parent_model.expanduser().resolve(strict=True)
        tokenizer_directory = config.tokenizer_source.expanduser().resolve(strict=True)
        cached_parent_outputs_path = config.cached_parent_outputs.expanduser().resolve(
            strict=True
        )
        publish_destination = config.publish_destination.expanduser().absolute()
        local_io = residual.direct_local_io_audit(parent_directory)
        parent_tokenizer_artifact_fingerprint = fingerprint_local_tokenizer(
            tokenizer_directory
        )
        tokenizer = residual.load_local_tokenizer(tokenizer_directory)
        progress.emit(
            "local_artifacts_verified",
            local_only=True,
            parent_directory_verified=True,
            tokenizer_directory_verified=True,
        )
        parent_compiler_tokenizer_fingerprint = (
            parent_tokenizer_artifact_fingerprint
        )
        if config.uses_tokenizer_bridge:
            assert config.target_tokenizer_directory is not None
            assert config.tokenizer_bridge_artifact_sha256 is not None
            assert config.tokenizer_bridge_profile_sha256 is not None
            preliminary_target_tokenizer = residual.load_local_tokenizer(
                config.target_tokenizer_directory.expanduser().resolve(
                    strict=True
                )
            )
            preliminary_bridge = load_strict_tokenizer_bridge_artifact(
                config.target_tokenizer_directory,
                donor_tokenizer=tokenizer,
                target_tokenizer=preliminary_target_tokenizer,
                expected_artifact_sha256=(
                    config.tokenizer_bridge_artifact_sha256
                ),
            )
            if (
                preliminary_bridge.profile_fingerprint
                != config.tokenizer_bridge_profile_sha256
            ):
                raise ProductionLfm2CatalogError(
                    "Tokenizer bridge profile differs from the explicit "
                    "configuration digest."
                )
            parent_compiler_tokenizer_fingerprint = (
                preliminary_bridge.donor_tokenizer_semantics_sha256
            )
        child_dtype = residual._torch_dtype(config.dtype)
        device = torch.device(config.device)
        parent, parent_dtype, parent_dtype_name = _load_cache_bound_parent(
            config,
            parent_directory,
            device=device,
            child_dtype=child_dtype,
        )
        parent_graph = inspect_model_circuits(
            parent,
            model_name=config.parent_name,
            tokenizer_fingerprint=parent_compiler_tokenizer_fingerprint,
        )
        progress.emit(
            "parent_verified",
            parent_unique_parameters=parent_graph.unique_parameter_count,
            parent_dtype=parent_dtype_name,
        )
        parent_capture_config = Lfm2CircuitTraceCaptureConfig(
            sample_last_tokens=config.sample_last_tokens,
            max_decode_steps=config.max_decode_steps,
            max_capture_bytes=config.max_capture_bytes,
            logit_rtol=config.capture_logit_rtol,
            logit_atol=config.capture_logit_atol,
            require_top1_agreement=(config.require_capture_top1_agreement),
            decode_trace_mode=CACHED_VERIFIED_TRACE_MODE,
            capture_causal_circuit_evidence=(
                config.dense_heterogeneous_global_hidden_frontier
            ),
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
            defer_selective_session=(
                config.dense_heterogeneous_global_hidden_frontier
            ),
        )
        active_selective_session = resolved_parent_cache.selective_session
        parent_cache = resolved_parent_cache.cache
        runtime.update(resolved_parent_cache.runtime_receipt_fields())
        progress.emit(
            "parent_cache_verified",
            opened_with_full_verification=bool(
                runtime["parent_cache_opened_with_full_verification"]
            ),
            parent_trace_regeneration_after_cache=0,
        )
        tokenizer_bridge = resolve_production_tokenizer_bridge(
            config,
            donor_tokenizer=tokenizer,
            parent_cache=parent_cache,
            calibration_splits=(resolved_parent_cache.calibration_splits),
            calibration_batches=(resolved_parent_cache.calibration_batches),
            progress=progress,
        )
        if tokenizer_bridge is not None:
            runtime["tokenizer_bridge"] = tokenizer_bridge.receipt_fields()
        progress.emit(
            "tokenizer_bridge_verified",
            enabled=tokenizer_bridge is not None,
            mapped_parent_evidence_required=config.uses_tokenizer_bridge,
        )
        if config.uses_carrier_routes:
            if tokenizer_bridge is None:
                raise ProductionLfm2CatalogError(
                    "Carrier routes require the resolved strict tokenizer bridge."
                )
            assert config.carrier_route_publication is not None
            assert config.carrier_route_artifact_sha256 is not None
            assert config.carrier_span_evidence is not None
            assert config.carrier_span_evidence_file_sha256 is not None
            assert config.carrier_span_evidence_sha256 is not None
            assert config.semantic_source_tokenizer_directory is not None
            assert config.routed_manual_prompts is not None
            assert config.routed_manual_prompts_sha256 is not None
            carrier_registration = (
                Lfm2CarrierChildPackageRegistration.current_exact4x(
                    publication_directory=config.carrier_route_publication,
                    expected_artifact_sha256=(
                        config.carrier_route_artifact_sha256
                    ),
                    span_evidence_json=config.carrier_span_evidence,
                    expected_span_evidence_sha256=(
                        config.carrier_span_evidence_sha256
                    ),
                    expected_span_evidence_file_sha256=(
                        config.carrier_span_evidence_file_sha256
                    ),
                ).verify()
            )
            carrier_source = load_lfm2_carrier_bridge_publication(
                config.carrier_route_publication,
                expected_artifact_sha256=(
                    config.carrier_route_artifact_sha256
                ),
            )
            carrier_spans = MappingProxyType(
                load_authenticated_lfm2_span_evidence(
                    config.carrier_span_evidence,
                    expected_file_sha256=(
                        config.carrier_span_evidence_file_sha256
                    ),
                    expected_evidence_sha256=(
                        config.carrier_span_evidence_sha256
                    ),
                )
            )
            carrier_prompts = _load_sealed_routed_manual_prompts(
                config.routed_manual_prompts,
                expected_sha256=config.routed_manual_prompts_sha256,
            )
            carrier_counted_plan = _build_registered_carrier_counted_plan(
                carrier_registration,
                parent_unique_parameters=(
                    parent_graph.unique_parameter_count
                ),
                target_total_parameter_elements=(
                    config.counted_target_total_parameters
                ),
                observed_allocations=carrier_source.allocations,
            )
            if (
                carrier_counted_plan.dense_model_parameter_elements
                != config.counted_dense_body_parameters
                or carrier_counted_plan.continuous_parameter_elements
                != 2 * LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE
                or tuple(
                    value.parameter_elements
                    for value in carrier_counted_plan.continuous_allocations
                )
                != (
                    LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE,
                    LFM2_CARRIER_ROUTE_PARAMETER_ELEMENTS_PER_ROLE,
                )
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier publication does not reserve the exact H816 body, "
                    "C_in, C_out, and total counted target."
                )
            semantic_source_tokenizer = residual.load_local_tokenizer(
                config.semantic_source_tokenizer_directory.expanduser().resolve(
                    strict=True
                )
            )
            if (
                fingerprint_hf_tokenizer_semantics(
                    semantic_source_tokenizer
                )
                != tokenizer_bridge.parent_tokenizer_semantics_sha256
            ):
                raise ProductionLfm2CatalogError(
                    "Carrier semantic-source lexicon differs from the tokenizer "
                    "bridge source semantics."
                )
            install_lfm2_semantic_step_generate_class(type(parent))
            runtime["carrier_routes"] = {
                "source_artifact_sha256": carrier_source.artifact_sha256,
                "span_evidence_sha256": config.carrier_span_evidence_sha256,
                "manual_prompts_file_sha256": (
                    config.routed_manual_prompts_sha256
                ),
                "counted_child_state_plan": carrier_counted_plan.as_dict(),
                "registered_parent_parameter_elements": (
                    carrier_counted_plan.parent_unique_parameters
                ),
                "realized_counted_compaction_ratio": (
                    carrier_counted_plan.realized_parameter_compaction_ratio
                ),
                "semantic_source_tokenizer_semantics_sha256": (
                    tokenizer_bridge.parent_tokenizer_semantics_sha256
                ),
                "target_tokenizer_semantics_sha256": (
                    tokenizer_bridge.target_tokenizer_semantics_sha256
                ),
                "class_entrypoint_installed_before_system_run": True,
                "high_cap_review_before_publication": False,
            }
        progress.emit(
            "carrier_verified",
            enabled=config.uses_carrier_routes,
            counted_total_parameter_elements=(
                None
                if carrier_counted_plan is None
                else carrier_counted_plan.target_total_parameter_elements
            ),
            independent_route_count=(
                0
                if carrier_counted_plan is None
                else len(carrier_counted_plan.continuous_allocations)
            ),
        )
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
            expected_tokenizer_fingerprint=(
                parent_tokenizer_artifact_fingerprint
            ),
            expected_generation_limit=(config.whole_output_max_new_tokens),
            fragment_minimum_characters=(
                config.whole_output_fragment_minimum_characters
            ),
        )
        mapped_parent_baseline = None
        if tokenizer_bridge is not None:
            (
                mapped_parent_baseline,
                tokenizer_bridge_normalization_control,
            ) = _load_bridge_mapped_parent_baseline(
                config,
                original_baseline=whole_output_baseline,
                tokenizer_bridge=tokenizer_bridge,
                parent_cache=parent_cache,
            )
            runtime["tokenizer_bridge_normalization_control"] = dict(
                tokenizer_bridge_normalization_control
            )
        if config.dense_heterogeneous_global_hidden_frontier:
            def acquire_selective_session_context():
                nonlocal active_selective_session
                if active_selective_session is None:
                    active_selective_session = (
                        parent_cache.selective_authenticated_session()
                    )
                return nullcontext(active_selective_session)

            global_topology_phase_context = (
                _global_topology_phase_cache_context(
                    config,
                    evidence_split_sha256=(
                        resolved_parent_cache.evidence_split_sha256
                    ),
                    progress=progress,
                    selective_session_factory=(
                        acquire_selective_session_context
                    ),
                )
            )
            progress.emit(
                "global_frontier_planning_started",
                global_hidden_frontier_enabled=True,
                required_target_hidden_size=config.required_target_hidden_size,
                before_parent_payload_session=(
                    active_selective_session is None
                ),
                derived_phase_cache_root=str(
                    config.global_topology_phase_cache_root
                ),
            )
            try:
                (
                    _preplanned_input,
                    _preplanned_bank,
                    preplanned_global_topology_frontier,
                ) = _build_cached_global_heterogeneous_selection_input(
                    config,
                    parent_model=parent,
                    parent_graph=parent_graph,
                    parent_cache=parent_cache,
                    tokenizer_fingerprint=(
                        parent_compiler_tokenizer_fingerprint
                    ),
                    tokenizer_bridge=tokenizer_bridge,
                    phase_cache_context=global_topology_phase_context,
                    defer_evidence_cache_miss=True,
                )
            except Lfm2CachedGlobalTopologyEvidenceCacheMiss:
                global_topology_evidence_deferred = True
                progress.emit(
                    "global_frontier_evidence_deferred_until_parent_rows",
                    required_target_hidden_size=(
                        config.required_target_hidden_size
                    ),
                    parent_payload_session_opened=False,
                    cold_b1_b2_parent_tensor_scan_started=False,
                )
            else:
                progress.emit(
                    "global_frontier_preplanned",
                    frontier_sha256=(
                        preplanned_global_topology_frontier.frontier_sha256
                    ),
                    selected_hidden_size=(
                        preplanned_global_topology_frontier
                        .selected_input.target_pins.hidden_size
                    ),
                    required_target_hidden_size=(
                        preplanned_global_topology_frontier
                        .required_target_hidden_size
                    ),
                    parent_payload_session_opened=(
                        active_selective_session is not None
                    ),
                )
        progress.emit("parent_rows_materialization_started")
        if active_selective_session is None:
            active_selective_session = (
                parent_cache.selective_authenticated_session()
            )
        sealed_evidence_set = resolved_parent_cache.evidence_set
        merged_evidence: _MergedSealedEvidencePayload | None = None
        if resolved_parent_cache.consumes_on_policy_extension_evidence:
            assert sealed_evidence_set is not None
            # Every closed-form writer on this lane reads the merged evidence
            # rows.  Lanes whose writers bind exactly one sealed cache never
            # reach here; they fail closed at configuration time.
            _assert_extension_evidence_is_consumable(config)
            # One preallocation, filled member by member, so the merge never
            # holds the members and the join at the same time.
            merged_evidence = _materialize_sealed_evidence_rows(
                sealed_evidence_set,
                parent=parent,
                tokenizer=tokenizer,
                base_calibration_batches=(
                    resolved_parent_cache.calibration_batches
                ),
                base_selective_session=active_selective_session,
            )
            parent_rows = merged_evidence.rows
        else:
            parent_rows = residual.extract_trace_rows(
                parent_cache,
                batch_splits=resolved_parent_cache.calibration_splits,
                selective_session=active_selective_session,
            )
        progress.emit(
            "parent_rows_materialized",
            path_count=len(parent_rows.paths),
            retained_bytes=sum(
                int(tensor.numel()) * int(tensor.element_size())
                for fields in parent_rows.layer_fields.values()
                for tensor in fields.values()
            ),
            trace_rows_sha256=parent_rows.trace_rows_sha256,
        )
        runtime.update(
            {
                "parent_cache_selective_session_reused": True,
                "parent_cache_selective_session_sha256": (
                    active_selective_session.evidence_sha256
                ),
                "parent_cache_selective_session_deferred_until_payload_required": (
                    resolved_parent_cache.selective_session is None
                ),
                "global_topology_planned_before_parent_payload_materialization": (
                    preplanned_global_topology_frontier is not None
                ),
                "global_topology_evidence_deferred_until_parent_rows": (
                    global_topology_evidence_deferred
                ),
            }
        )
        calibration_batches = resolved_parent_cache.calibration_batches
        calibration_splits = resolved_parent_cache.calibration_splits
        selection_rows = parent_rows
        if merged_evidence is not None:
            assert sealed_evidence_set is not None
            calibration_batches = merged_evidence.calibration_batches
            calibration_splits = merged_evidence.calibration_splits
            if tokenizer_bridge is not None:
                # A bridged child replays target rows, so the merged corpus
                # needs a merged target-token replay before any consumer binds
                # it.  Rebuilt from the rounds' sealed preimages, reproved
                # against the merged donor inputs, zero parent forwards.
                tokenizer_bridge = (
                    extend_tokenizer_bridge_with_on_policy_preimages(
                        tokenizer_bridge,
                        config=config,
                        evidence_set=sealed_evidence_set,
                        rounds=resolved_parent_cache.on_policy_rounds,
                        merged_calibration_batches=calibration_batches,
                        progress=progress,
                    )
                )
                runtime["tokenizer_bridge"] = tokenizer_bridge.receipt_fields()
            # The selection lane keeps reading the base corpus alone: a
            # child-visited row is a fit row and must never help choose the
            # child's own topology.  This view is a zero-copy prefix of the
            # merged storage, so holding it costs nothing.
            selection_rows = merged_evidence.base_rows
            base_path_count = len(selection_rows.paths)
            progress.emit(
                "on_policy_extension_evidence_materialized",
                member_count=sealed_evidence_set.member_count,
                round_count=sealed_evidence_set.round_count,
                base_batch_count=sealed_evidence_set.base.batch_count,
                merged_batch_count=sealed_evidence_set.batch_count,
                base_path_count=base_path_count,
                merged_path_count=len(parent_rows.paths),
                extension_path_count=(
                    len(parent_rows.paths) - base_path_count
                ),
                evidence_set_cache_key=sealed_evidence_set.cache_key,
                evidence_identity_sha256=(
                    sealed_evidence_set.evidence_identity_sha256
                ),
                merged_trace_rows_sha256=parent_rows.trace_rows_sha256,
                all_extension_rows_are_fit_rows=True,
                parent_forwards=0,
                **{
                    f"merge_{name}": value
                    for name, value in merged_evidence.accounting.items()
                    if isinstance(value, (bool, int, float, str))
                },
            )
            runtime.update(
                {
                    "on_policy_evidence_set": (
                        sealed_evidence_set.receipt_fields()
                    ),
                    "on_policy_extension_evidence_consumed": True,
                    "on_policy_merged_row_accounting": dict(
                        merged_evidence.accounting
                    ),
                }
            )
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
            calibration_splits=calibration_splits,
            partition=partition,
            tokenizer_bridge=tokenizer_bridge,
            selective_session=active_selective_session,
            evidence_set=sealed_evidence_set,
            selection_rows=selection_rows,
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
        original_behavior_evaluator = Lfm2WholeOutputBehaviorEvaluator(
            baseline=whole_output_baseline,
            tokenizer=(
                tokenizer
                if tokenizer_bridge is None
                else tokenizer_bridge.target_tokenizer
            ),
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
        behavior_evaluator: Any = original_behavior_evaluator
        if mapped_parent_baseline is not None:
            assert tokenizer_bridge is not None
            assert tokenizer_bridge_normalization_control is not None
            mapped_behavior_evaluator = Lfm2WholeOutputBehaviorEvaluator(
                baseline=mapped_parent_baseline,
                tokenizer=tokenizer_bridge.target_tokenizer,
                maximum_prompt_tokens=(
                    config.whole_output_maximum_prompt_tokens
                ),
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
                    config
                    .whole_output_maximum_new_generation_limit_saturations
                ),
                maximum_repeated_ngram_share_regressions=(
                    config
                    .whole_output_maximum_repeated_ngram_share_regressions
                ),
                maximum_repeated_bigram_share_delta=(
                    config.whole_output_maximum_repeated_bigram_share_delta
                ),
            )
            behavior_evaluator = BridgeDualWholeOutputBehaviorEvaluator(
                original_evaluator=original_behavior_evaluator,
                mapped_evaluator=mapped_behavior_evaluator,
                normalization_control=(
                    tokenizer_bridge_normalization_control
                ),
            )
        if config.uses_carrier_routes:
            assert tokenizer_bridge is not None
            assert carrier_source is not None
            assert carrier_spans is not None
            assert carrier_prompts is not None
            assert carrier_counted_plan is not None
            assert config.semantic_source_tokenizer_directory is not None
            semantic_source_tokenizer = residual.load_local_tokenizer(
                config.semantic_source_tokenizer_directory.expanduser().resolve(
                    strict=True
                )
            )
            route_smoke_baseline_fingerprint = canonical_sha256(
                {
                    "source_artifact_sha256": carrier_source.artifact_sha256,
                    "span_evidence_sha256": (
                        config.carrier_span_evidence_sha256
                    ),
                    "manual_prompts_file_sha256": (
                        config.routed_manual_prompts_sha256
                    ),
                    "counted_child_state_plan_fingerprint": (
                        carrier_counted_plan.fingerprint
                    ),
                    "tokenizer_bridge_artifact_sha256": (
                        tokenizer_bridge.artifact.artifact_sha256
                    ),
                    "tokenizer_bridge_profile_sha256": (
                        tokenizer_bridge.artifact.profile_fingerprint
                    ),
                }
            )
        evaluator = CombinedLfm2FinalEvaluator(
            trajectory_evaluator=trajectory_evaluator,
            behavior_evaluator=behavior_evaluator,
            performance_metrics_diagnostic_only=(
                config.final_performance_metrics_diagnostic_only
            ),
        )
        artifact_io = DirectLocalArtifactIO(
            tokenizer_directory=(
                tokenizer_directory
                if tokenizer_bridge is None
                else tokenizer_bridge.artifact.root_dir
            ),
            tokenizer_fingerprint=(
                parent_tokenizer_artifact_fingerprint
                if tokenizer_bridge is None
                else tokenizer_bridge.target_tokenizer_artifact_fingerprint
            ),
            device=config.device,
            dtype=child_dtype,
            tokenizer_bridge=tokenizer_bridge,
            donor_tokenizer=(None if tokenizer_bridge is None else tokenizer),
            packaged_semantic_source_tokenizer_semantics_sha256=(
                None
                if not config.uses_carrier_routes
                else tokenizer_bridge.parent_tokenizer_semantics_sha256
            ),
        )
        selected_artifact_io: Any = artifact_io
        if config.uses_carrier_routes:
            assert tokenizer_bridge is not None
            assert carrier_registration is not None
            assert carrier_spans is not None
            assert carrier_prompts is not None
            assert config.carrier_route_publication is not None
            assert config.carrier_route_artifact_sha256 is not None
            assert config.semantic_source_tokenizer_directory is not None
            carrier_artifact_io = CarrierAwareDirectLocalArtifactIO(
                base=artifact_io,
                registration=carrier_registration,
                semantic_source_tokenizer_directory=(
                    config.semantic_source_tokenizer_directory
                ),
                semantic_source_tokenizer_semantics_sha256=(
                    tokenizer_bridge.parent_tokenizer_semantics_sha256
                ),
                semantic_source_tokenizer=semantic_source_tokenizer,
                target_tokenizer=tokenizer_bridge.target_tokenizer,
            )
            carrier_behavior_evaluator = CarrierRouteSmokeBehaviorEvaluator(
                artifact_io=carrier_artifact_io,
                semantic_source_tokenizer=semantic_source_tokenizer,
                target_tokenizer=tokenizer_bridge.target_tokenizer,
                prompts=carrier_prompts,
                maximum_prompt_tokens=(
                    config.routed_smoke_maximum_prompt_tokens
                ),
                maximum_new_tokens=config.routed_smoke_max_new_tokens,
                baseline_fingerprint=route_smoke_baseline_fingerprint,
            )
            behavior_evaluator = carrier_behavior_evaluator
            evaluator = CombinedLfm2FinalEvaluator(
                trajectory_evaluator=trajectory_evaluator,
                behavior_evaluator=behavior_evaluator,
                performance_metrics_diagnostic_only=(
                    config.final_performance_metrics_diagnostic_only
                ),
            )
            selected_artifact_io = carrier_artifact_io
        if not config.dense_heterogeneous_global_hidden_frontier:
            progress.emit(
                "global_frontier_planning_started",
                global_hidden_frontier_enabled=False,
                required_target_hidden_size=config.required_target_hidden_size,
            )
        if balanced_basis_phase_anchor_authority is not None:
            dependency_provider: Callable[[], Mapping[str, str]] | None = None
            if carrier_counted_plan is not None:
                assert carrier_registration is not None
                assert config.carrier_route_artifact_sha256 is not None
                assert config.carrier_span_evidence_sha256 is not None
                expected_registration = carrier_registration.fingerprint
                expected_route = config.carrier_route_artifact_sha256
                expected_span = config.carrier_span_evidence_sha256
                expected_counted_plan = carrier_counted_plan.fingerprint

                def carrier_phase_dependencies() -> Mapping[str, str]:
                    return _authenticated_carrier_phase_dependencies(
                        carrier_registration,
                        expected_registration_fingerprint=expected_registration,
                        expected_route_artifact_sha256=expected_route,
                        expected_span_evidence_sha256=expected_span,
                        expected_counted_plan_fingerprint=expected_counted_plan,
                    )

                dependency_provider = carrier_phase_dependencies
            dependency_provider = _compose_on_policy_phase_dependencies(
                dependency_provider,
                resolved_parent_cache.on_policy_evidence_identity_sha256,
            )
            balanced_basis_phase_context = (
                _balanced_basis_phase_cache_context(
                    config,
                    configuration_sha256=configuration_sha256,
                    anchor_authority=(
                        balanced_basis_phase_anchor_authority
                    ),
                    progress=progress,
                    dependency_provider=dependency_provider,
                )
            )
        wiring = build_production_catalog_wiring(
            config,
            parent_cache_payload=payload,
            parent_model=parent,
            parent_graph=parent_graph,
            tokenizer_fingerprint=parent_compiler_tokenizer_fingerprint,
            evidence_builder=evidence_builder,
            final_evaluator=evaluator,
            model_saver=selected_artifact_io.save,
            model_loader=selected_artifact_io.load,
            fresh_reload_evaluator=evaluator.evaluate_fresh_reload,
            ledger=ledger,
            device=device,
            dtype=child_dtype,
            tokenizer_bridge=tokenizer_bridge,
            global_topology_phase_cache_context=(
                global_topology_phase_context
            ),
            balanced_basis_phase_cache_context=(
                balanced_basis_phase_context
            ),
            progress=progress,
        )
        if preplanned_global_topology_frontier is not None:
            wired_frontier = wiring.heterogeneous_global_topology_frontier
            if (
                wired_frontier is None
                or dict(wired_frontier.as_dict())
                != dict(preplanned_global_topology_frontier.as_dict())
            ):
                raise ProductionLfm2CatalogError(
                    "Pre-materialization and wired global topology frontiers "
                    "differ."
                )
            progress.emit(
                "global_frontier_cache_reauthenticated",
                frontier_sha256=wired_frontier.frontier_sha256,
                required_target_hidden_size=(
                    wired_frontier.required_target_hidden_size
                ),
                exact_cold_hit_parity=True,
            )
        elif global_topology_evidence_deferred:
            wired_frontier = wiring.heterogeneous_global_topology_frontier
            if wired_frontier is None:
                raise ProductionLfm2CatalogError(
                    "Deferred global topology evidence did not produce a wired "
                    "frontier from retained parent rows."
                )
            progress.emit(
                "global_frontier_planned_from_retained_parent_rows",
                frontier_sha256=wired_frontier.frontier_sha256,
                selected_hidden_size=(
                    wired_frontier.selected_input.target_pins.hidden_size
                ),
                required_target_hidden_size=(
                    wired_frontier.required_target_hidden_size
                ),
                retained_parent_trace_rows_sha256=(
                    parent_rows.trace_rows_sha256
                ),
                exact_semantics_hybrid_cold_path=True,
            )
        progress.emit(
            "wiring_planned",
            carrier_routes_enabled=config.uses_carrier_routes,
            compression_ratio=config.compression_ratio,
            dense_fold=config.compression_ratio is not None,
            required_target_hidden_size=config.required_target_hidden_size,
            planned_target_abi_sha256=(
                None
                if wiring.planned_target_abi is None
                else wiring.planned_target_abi["evidence_sha256"]
            ),
        )
        progress.emit("parent_identity_digest_started")
        runtime.update(
            {
                "parent_dtype": parent_dtype_name,
                "parent_dtype_sha256": canonical_sha256(
                    {
                        "dtype": str(parent_dtype),
                        "parent_identity_sha256": (
                            fingerprint_lfm2_parent(parent).identity_sha256
                        ),
                    }
                ),
                "child_dtype": str(child_dtype).removeprefix("torch."),
                "parent_child_dtype_decoupled": parent_dtype != child_dtype,
                "local_io": local_io,
                "tokenizer_fingerprint": (
                    parent_compiler_tokenizer_fingerprint
                ),
                "parent_tokenizer_artifact_fingerprint": (
                    parent_tokenizer_artifact_fingerprint
                ),
                "target_tokenizer_artifact_fingerprint": (
                    parent_tokenizer_artifact_fingerprint
                    if tokenizer_bridge is None
                    else tokenizer_bridge.target_tokenizer_artifact_fingerprint
                ),
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
                    "original_parent_input_mode": (
                        "donor-tokenizer-canonical"
                    ),
                    "mapped_parent_input_mode": (
                        None
                        if tokenizer_bridge is None
                        else (
                            "target-tokenizer-canonical-then-one-row-"
                            "mapped-to-donor"
                        )
                    ),
                    "mapped_parent_envelope_sha256": (
                        None
                        if mapped_parent_baseline is None
                        else mapped_parent_baseline.envelope_sha256
                    ),
                    "mapped_parent_file_sha256": (
                        None
                        if mapped_parent_baseline is None
                        else mapped_parent_baseline.source_sha256
                    ),
                    "child_evaluated_against_original_and_mapped_parent": (
                        mapped_parent_baseline is not None
                    ),
                    "donor_canonical_sequence_equivalence_claimed": False,
                },
                "parent_trace_rows_sha256": (parent_rows.trace_rows_sha256),
                "parent_forward_count_after_cache": 0,
            }
        )
        progress.emit(
            "system_fold_started",
            parent_model_count=1,
            destination_ready=True,
        )
        result = wiring.system.run(
            parent_models={config.parent_name: parent},
            target_request=wiring.target_request,
            # The exact evidence this pass fits on: the base cache key when
            # nothing extends it, and the evidence-set key once on-policy
            # extension rows are live.
            evidence_fingerprint=payload.cache_key,
            destination=publish_destination,
            parent_tokenizer_fingerprints={
                config.parent_name: parent_compiler_tokenizer_fingerprint
            },
        )
        materialized_target_abi = _verify_materialized_target_abi(
            config,
            result.target.model,
            wiring.planned_target_abi,
        )
        run = result.published.compilation.run
        artifact = result.published.artifact
        progress.emit(
            "system_fold_completed",
            accepted_action_count=len(run.accepted_action_ids),
            exact_physical_parameter_elements=artifact.exact_unique_parameters,
            materialized_target_abi_sha256=(
                None
                if materialized_target_abi is None
                else materialized_target_abi["evidence_sha256"]
            ),
            published=True,
        )
        if config.uses_carrier_routes:
            assert carrier_artifact_io is not None
            assert carrier_behavior_evaluator is not None
            assert carrier_counted_plan is not None
            assert carrier_prompts is not None
            carrier_smoke_coverage = (
                _verify_carrier_route_smoke_phase_coverage(
                    carrier_behavior_evaluator.events
                )
            )
            final_child_directory = (
                Path(artifact.path).expanduser().resolve(strict=True) / "child"
            )
            authenticated_artifact = load_dynamic_fold_artifact_certificate(
                artifact.path
            )
            if authenticated_artifact != artifact:
                raise ProductionLfm2CatalogError(
                    "Post-publication carrier review observed a different "
                    "artifact certificate."
                )
            fresh_base_io = DirectLocalArtifactIO(
                tokenizer_directory=final_child_directory,
                tokenizer_fingerprint=fingerprint_local_tokenizer(
                    final_child_directory
                ),
                device=(
                    "cpu"
                    if config.uses_portable_runtime_relocation
                    else config.device
                ),
                dtype=child_dtype,
            )
            fresh_child_only_io = (
                Lfm2CarrierChildCheckpointIO.from_child_directory(
                    base=fresh_base_io,
                    child_directory=final_child_directory,
                    target_tokenizer_loader=residual.load_local_tokenizer,
                )
            )
            if config.uses_portable_runtime_relocation:
                assert config.portable_runtime_device is not None
                assert config.portable_runtime_dtype is not None
                final_routed_child, portable_relocation = (
                    fresh_child_only_io.load_authenticated_portable(
                        final_child_directory,
                        runtime_device=config.portable_runtime_device,
                        runtime_dtype=_portable_runtime_dtype(
                            config.portable_runtime_dtype
                        ),
                    )
                )
                portable_runtime_receipt = MappingProxyType(
                    portable_relocation.as_dict()
                )
                final_child_package = (
                    load_authenticated_lfm2_carrier_child_package(
                        final_child_directory,
                        expected_package_sha256=(
                            fresh_child_only_io.expected_child_package_sha256
                        ),
                    )
                )
            else:
                final_routed_child = fresh_child_only_io.load(
                    final_child_directory
                )
                final_child_package = (
                    load_authenticated_lfm2_carrier_child_package(
                        final_child_directory,
                        child_model=final_routed_child,
                        expected_package_sha256=(
                            fresh_child_only_io.expected_child_package_sha256
                        ),
                    )
                )
            carrier_route_package = final_child_package.routes
            carrier_lexicon_package = (
                final_child_package.semantic_source_lexicon
            )
            carrier_outer_inventory = carrier_artifact_io.verify_outer_inventory(
                final_child_directory,
            )
            carrier_counted_verification = verify_counted_child_state(
                carrier_counted_plan,
                physical_dense_parameter_elements=(
                    artifact.exact_unique_parameters
                ),
                observed_allocations=carrier_route_package.allocations,
            )
            if portable_runtime_receipt is not None:
                if (
                    portable_runtime_receipt[
                        "canonical_dense_unique_parameter_elements"
                    ]
                    != artifact.exact_unique_parameters
                    or portable_runtime_receipt[
                        "runtime_dense_unique_parameter_elements"
                    ]
                    != artifact.exact_unique_parameters
                    or portable_runtime_receipt[
                        "counted_carrier_parameter_elements"
                    ]
                    != carrier_counted_plan.continuous_parameter_elements
                    or portable_runtime_receipt[
                        "counted_total_parameter_elements"
                    ]
                    != carrier_counted_plan.target_total_parameter_elements
                    or portable_runtime_receipt[
                        "dtype_or_device_parameter_compaction_credit"
                    ]
                    != 0
                    or portable_runtime_receipt[
                        "executable_forward_smoke_passed"
                    ]
                    is not True
                    or portable_runtime_receipt[
                        "canonical_disk_artifact_mutated"
                    ]
                    is not False
                    or portable_runtime_receipt[
                        "direct_local_relocation"
                    ]["deterministic_post_cast_state_match"]
                    is not True
                    or portable_runtime_receipt[
                        "direct_local_relocation"
                    ]["state_names_shapes_and_alias_abi_preserved"]
                    is not True
                ):
                    raise ProductionLfm2CatalogError(
                        "Portable carrier runtime changed exact counted child "
                        "parameters or claimed dtype/device compaction credit."
                    )
                (
                    portable_runtime_execution_smoke,
                    portable_runtime_review_identity_sha256,
                ) = _run_verified_portable_carrier_execution_smoke(
                    final_routed_child,
                    carrier_prompts,
                    maximum_prompt_tokens=(
                        config.routed_smoke_maximum_prompt_tokens
                    ),
                    portable_runtime_receipt=portable_runtime_receipt,
                    canonical_child_artifact_sha256=(
                        artifact.artifact_tree_sha256
                    ),
                    route_state_observer=(
                        fresh_child_only_io
                        .observe_authenticated_portable_runtime_state
                    ),
                )
                progress.emit(
                    "portable_runtime_relocation_completed",
                    runtime_device=portable_runtime_receipt[
                        "direct_local_relocation"
                    ]["runtime_device"],
                    runtime_dtype=portable_runtime_receipt[
                        "direct_local_relocation"
                    ]["runtime_dtype"],
                    canonical_dense_parameter_elements=(
                        artifact.exact_unique_parameters
                    ),
                    counted_total_parameter_elements=(
                        carrier_counted_plan.target_total_parameter_elements
                    ),
                    parameter_compaction_credit=0,
                    carrier_coefficients_remain_fp32=True,
                    executable_forward_smoke_sha256=(
                        portable_runtime_receipt[
                            "direct_local_relocation"
                        ]["executable_forward_smoke"]["smoke_sha256"]
                    ),
                    carrier_execution_smoke_sha256=(
                        portable_runtime_execution_smoke["smoke_sha256"]
                    ),
                    runtime_review_identity_sha256=(
                        portable_runtime_review_identity_sha256
                    ),
                )
            progress.emit(
                "standalone_child_authenticated",
                counted_state_verified=True,
                exact_total_parameter_elements=(
                    carrier_counted_plan.target_total_parameter_elements
                ),
                route_count=len(carrier_route_package.allocations),
            )
            progress.emit(
                "manual_review_started",
                initial_max_new_tokens=(
                    config.routed_manual_initial_max_new_tokens
                ),
                prompt_count=len(carrier_prompts),
                retry_max_new_tokens=(
                    config.routed_manual_retry_max_new_tokens
                ),
            )
            carrier_manual_dossier = _write_routed_manual_dossier(
                model=final_routed_child,
                prompts=carrier_prompts,
                config=config,
                counted_verification=carrier_counted_verification,
                outer_inventory=carrier_outer_inventory,
                route_package=carrier_route_package,
                lexicon_package=carrier_lexicon_package,
                published_child_artifact_sha256=(
                    artifact.artifact_tree_sha256
                ),
                portable_runtime_review_identity_sha256=(
                    portable_runtime_review_identity_sha256
                ),
                progress=progress,
            )
            progress.emit(
                "manual_review_completed",
                checkpoint_rows_executed=(
                    carrier_manual_dossier["checkpoint_rows_executed"]
                ),
                checkpoint_rows_reused=(
                    carrier_manual_dossier["checkpoint_rows_reused"]
                ),
                intelligence_response_count=(
                    carrier_manual_dossier["intelligence_response_count"]
                ),
                remaining_cap_censored_response_count=(
                    carrier_manual_dossier[
                        "remaining_cap_censored_response_count"
                    ]
                ),
                unknown_censored_intelligence_response_count=(
                    carrier_manual_dossier[
                        "unknown_censored_intelligence_response_count"
                    ]
                ),
                safety_observation_count=(
                    carrier_manual_dossier["safety_observation_count"]
                ),
            )
        elif config.uses_portable_runtime_relocation:
            assert config.portable_runtime_device is not None
            assert config.portable_runtime_dtype is not None
            final_child_directory = (
                Path(artifact.path).expanduser().resolve(strict=True) / "child"
            )
            authenticated_artifact = load_dynamic_fold_artifact_certificate(
                artifact.path
            )
            if authenticated_artifact != artifact:
                raise ProductionLfm2CatalogError(
                    "Portable runtime relocation observed a different "
                    "artifact certificate."
                )
            fresh_base_io = DirectLocalArtifactIO(
                tokenizer_directory=final_child_directory,
                tokenizer_fingerprint=fingerprint_local_tokenizer(
                    final_child_directory
                ),
                device=config.device,
                dtype=child_dtype,
            )
            _portable_child, portable_runtime_receipt = (
                _load_verified_portable_direct_runtime(
                    fresh_base_io,
                    final_child_directory,
                    runtime_device=config.portable_runtime_device,
                    runtime_dtype=_portable_runtime_dtype(
                        config.portable_runtime_dtype
                    ),
                    expected_unique_parameters=(
                        artifact.exact_unique_parameters
                    ),
                    progress=progress,
                )
            )
        # A round is only meaningful against a child that exists, so extension
        # evidence is produced after a successful publication and never inside
        # the solve that produced it.  Consuming base+extension evidence in a
        # subsequent solve pass is the deferred item recorded in NOTES-2A.md.
        executed_on_policy_rounds = _run_postpublication_on_policy_rounds(
            config,
            parent=parent,
            parent_tokenizer=tokenizer,
            parent_fingerprint=(
                fingerprint_lfm2_parent(parent).identity_sha256
            ),
            base_cache_directory=parent_cache.path,
            capture_config=parent_capture_config,
            published_child_directory=(
                Path(artifact.path).expanduser().resolve(strict=True) / "child"
            ),
            consumed_round_manifests=config.on_policy_round_manifests,
            input_device=device,
            progress=progress,
            tokenizer_bridge=tokenizer_bridge,
        )
        on_policy_evidence_receipt = (
            resolved_parent_cache.on_policy_evidence_receipt(
                max_rounds=config.on_policy_max_rounds,
                executed_rounds=executed_on_policy_rounds,
            )
        )
        runtime["on_policy_parent_capture_invocations"] = (
            on_policy_evidence_receipt["parent_capture_invocations"]
        )
        runtime["on_policy_postpublication_round_count"] = len(
            executed_on_policy_rounds
        )
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
                "configuration_sha256": configuration_sha256,
                **(
                    {}
                    if portable_runtime_receipt is None
                    else {
                        "portable_runtime_relocation": dict(
                            portable_runtime_receipt
                        ),
                        **(
                            {}
                            if portable_runtime_execution_smoke is None
                            else {
                                "portable_runtime_execution_smoke": dict(
                                    portable_runtime_execution_smoke
                                ),
                                "portable_runtime_review_identity_sha256": (
                                    portable_runtime_review_identity_sha256
                                ),
                            }
                        ),
                    }
                ),
                "global_topology_phase_cache": {
                    "enabled": (
                        config.dense_heterogeneous_global_hidden_frontier
                    ),
                    "root": (
                        str(config.global_topology_phase_cache_root)
                        if config.dense_heterogeneous_global_hidden_frontier
                        else None
                    ),
                    "required_target_hidden_size": (
                        config.required_target_hidden_size
                    ),
                },
                **_balanced_basis_phase_cache_receipt_fields(config),
                "closed_loop_performance_metrics_policy": (
                    _dense_closed_loop_performance_metrics_policy(config)
                ),
                "closed_loop_replay_policy": (
                    _dense_closed_loop_replay_policy(config, progress)
                ),
                "restart_capsule_policy": (
                    _dense_restart_capsule_policy(config)
                ),
                "on_policy_evidence_policy": (
                    _on_policy_evidence_policy(config)
                ),
                "on_policy_evidence": on_policy_evidence_receipt,
                "parent": runtime,
                "planned_target": _planned_dense_target_evidence(wiring),
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
                    "dense_minimum_hidden_size": (
                        config.dense_minimum_hidden_size
                    ),
                    "required_target_hidden_size": (
                        config.required_target_hidden_size
                    ),
                    "planned_target_abi": (
                        None
                        if wiring.planned_target_abi is None
                        else dict(wiring.planned_target_abi)
                    ),
                    "materialized_target_abi": (
                        None
                        if materialized_target_abi is None
                        else dict(materialized_target_abi)
                    ),
                    "dense_heterogeneous_layout": (
                        config.dense_heterogeneous_layout
                    ),
                    "dense_heterogeneous_global_hidden_frontier": (
                        config.dense_heterogeneous_global_hidden_frontier
                    ),
                    "dense_heterogeneous_factor_writer_rank_discovery": (
                        config.dense_heterogeneous_factor_writer_rank_discovery
                    ),
                    "heterogeneous_capacity_option_bank_sha256": (
                        None
                        if wiring.heterogeneous_capacity_option_bank is None
                        else wiring
                        .heterogeneous_capacity_option_bank
                        .evidence_sha256
                    ),
                    "heterogeneous_global_topology_frontier_sha256": (
                        None
                        if wiring.heterogeneous_global_topology_frontier is None
                        else wiring
                        .heterogeneous_global_topology_frontier
                        .frontier_sha256
                    ),
                    "heterogeneous_global_selected_candidate_sha256": (
                        None
                        if wiring.heterogeneous_global_topology_frontier is None
                        else wiring
                        .heterogeneous_global_topology_frontier
                        .selected_candidate_sha256
                    ),
                    "realized_compression_ratio": (
                        source_unique_parameters
                        / (
                            result.target.resource_profile.unique_parameters
                            if carrier_counted_plan is None
                            else carrier_counted_plan.target_total_parameter_elements
                        )
                    ),
                    "physical_body_compression_ratio": (
                        source_unique_parameters
                        / result.target.resource_profile.unique_parameters
                    ),
                    "counted_target_total_parameter_elements": (
                        None
                        if carrier_counted_plan is None
                        else carrier_counted_plan.target_total_parameter_elements
                    ),
                    "counted_continuous_parameter_elements": (
                        None
                        if carrier_counted_plan is None
                        else carrier_counted_plan.continuous_parameter_elements
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
                    "vocabulary_size": int(target_runtime_config.vocab_size),
                    "tokenizer_bridge_required": (
                        selected_budget_candidate is not None
                        and bool(
                            selected_budget_candidate.tokenizer_bridge_required
                        )
                    ),
                    "tokenizer_fingerprint": (
                        result.target.request.tokenizer_fingerprint
                    ),
                    "parent_tokenizer_semantics_sha256": (
                        parent_compiler_tokenizer_fingerprint
                    ),
                    "target_tokenizer_semantics_sha256": (
                        parent_compiler_tokenizer_fingerprint
                        if tokenizer_bridge is None
                        else tokenizer_bridge.target_tokenizer_semantics_sha256
                    ),
                    "tokenizer_bridge_artifact_sha256": (
                        None
                        if tokenizer_bridge is None
                        else tokenizer_bridge.artifact.artifact_sha256
                    ),
                    "tokenizer_transport_profile_sha256": (
                        None
                        if tokenizer_bridge is None
                        else tokenizer_bridge.artifact.profile_fingerprint
                    ),
                    "tokenizer_bridge_trace_evidence_sha256": (
                        None
                        if tokenizer_bridge is None
                        else tokenizer_bridge.evidence_sha256
                    ),
                    "tokenizer_bridge_normalization_control_sha256": (
                        None
                        if tokenizer_bridge_normalization_control is None
                        else tokenizer_bridge_normalization_control[
                            "control_sha256"
                        ]
                    ),
                    "tokenizer_bridge_normalization_control_passed": (
                        None
                        if tokenizer_bridge_normalization_control is None
                        else bool(
                            tokenizer_bridge_normalization_control["passed"]
                        )
                    ),
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
                        - (
                            result.target.resource_profile.unique_parameters
                            if carrier_counted_plan is None
                            else carrier_counted_plan.target_total_parameter_elements
                        )
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
                    "closed_loop_performance_metrics_policy": (
                        _dense_closed_loop_performance_metrics_policy(config)
                    ),
                    "vocabulary_margin_candidate_gate_policy": (
                        _dense_vocabulary_margin_candidate_gate_policy(config)
                    ),
                    "macro_cell_attractor_gate_policy": (
                        _dense_macro_cell_attractor_gate_policy(config)
                    ),
                    "attention_score_candidate_gate_policy": (
                        _dense_attention_score_candidate_gate_policy(config)
                    ),
                    "vocabulary_margin_candidate_gate": (
                        dense_candidate_gate_report
                    ),
                },
                "behavior": gate_summaries,
                "carrier_routes": (
                    None
                    if carrier_counted_plan is None
                    else {
                        "status": "standalone-counted-route-child-published",
                        "counted_child_state_plan": (
                            carrier_counted_plan.as_dict()
                        ),
                        "counted_child_state_verification": dict(
                            carrier_counted_verification
                        ),
                        "route_package_sha256": (
                            carrier_route_package.package_sha256
                        ),
                        "semantic_lexicon_package_sha256": (
                            carrier_lexicon_package.package_sha256
                        ),
                        "outer_inventory": dict(carrier_outer_inventory),
                        "carrier_child_package_sha256": (
                            carrier_outer_inventory[
                                "carrier_child_package_sha256"
                            ]
                        ),
                        "postpublication_loader_constructed_from_child_directory": True,
                        "postpublication_loader_build_registration_present": False,
                        "postpublication_external_route_publication_required": False,
                        "postpublication_external_semantic_tokenizer_required": False,
                        "route_smoke_events": [
                            dict(value)
                            for value in carrier_behavior_evaluator.events
                        ],
                        "route_smoke_phase_count": len(
                            carrier_behavior_evaluator.events
                        ),
                        "route_smoke_phase_coverage": dict(
                            carrier_smoke_coverage
                        ),
                        "in_memory_stage_and_final_reload_smoke_expected": 3,
                        "high_cap_review_runs_after_publication": 1,
                        "high_cap_manual_dossier_path": str(
                            config.routed_manual_dossier_path
                        ),
                        "high_cap_manual_dossier_sha256": (
                            carrier_manual_dossier["dossier_sha256"]
                        ),
                        "manual_review_checkpoint_directory": (
                            carrier_manual_dossier["checkpoint_directory"]
                        ),
                        "manual_review_checkpoint_row_count": (
                            carrier_manual_dossier["checkpoint_row_count"]
                        ),
                        "manual_review_checkpoint_rows_reused": (
                            carrier_manual_dossier["checkpoint_rows_reused"]
                        ),
                        "manual_review_checkpoint_rows_executed": (
                            carrier_manual_dossier["checkpoint_rows_executed"]
                        ),
                        "manual_review_recoverable_without_refolding": True,
                        "intelligence_response_count": (
                            carrier_manual_dossier[
                                "intelligence_response_count"
                            ]
                        ),
                        "safety_observation_count": (
                            carrier_manual_dossier["safety_observation_count"]
                        ),
                        "safety_observations_counted_as_folding_failures": False,
                        "safety_observations_counted_as_intelligence_failures": False,
                        "initial_cap_censored_response_count": (
                            carrier_manual_dossier[
                                "initial_cap_censored_response_count"
                            ]
                        ),
                        "adaptive_retry_response_count": (
                            carrier_manual_dossier[
                                "adaptive_retry_response_count"
                            ]
                        ),
                        "remaining_cap_censored_response_count": (
                            carrier_manual_dossier[
                                "remaining_cap_censored_response_count"
                            ]
                        ),
                        "unknown_censored_intelligence_response_count": (
                            carrier_manual_dossier[
                                "unknown_censored_intelligence_response_count"
                            ]
                        ),
                        "cap_censored_responses_counted_as_folding_failures": False,
                        "numeric_intelligence_gate": False,
                        "numeric_metrics_are_diagnostic_only": True,
                        "numeric_metrics_have_acceptance_authority": False,
                        "manual_whole_response_intelligence_review_is_authoritative": True,
                    }
                ),
                "artifact": {
                    "path": artifact.path,
                    "certificate_sha256": (artifact.certificate_sha256),
                    "artifact_tree_sha256": (artifact.artifact_tree_sha256),
                    "child_state_fingerprint": (artifact.child_state_fingerprint),
                    "fresh_reload_accepted": (artifact.fresh_reload_decision.accepted),
                    "tokenizer_packaged": True,
                    "tokenizer_bridge_manifest_packaged": (
                        tokenizer_bridge is not None
                    ),
                    "tokenizer_bridge_artifact_sha256": (
                        None
                        if tokenizer_bridge is None
                        else tokenizer_bridge.artifact.artifact_sha256
                    ),
                    "tokenizer_transport_profile_sha256": (
                        None
                        if tokenizer_bridge is None
                        else tokenizer_bridge.artifact.profile_fingerprint
                    ),
                    "parent_tokenizer_artifact_fingerprint": (
                        parent_tokenizer_artifact_fingerprint
                    ),
                    "target_tokenizer_artifact_fingerprint": (
                        parent_tokenizer_artifact_fingerprint
                        if tokenizer_bridge is None
                        else tokenizer_bridge.target_tokenizer_artifact_fingerprint
                    ),
                    "carrier_routes_packaged": (
                        carrier_route_package is not None
                    ),
                    "semantic_source_lexicon_packaged": (
                        carrier_lexicon_package is not None
                    ),
                    "only_base_checkpoint_tokenizer_and_counted_resources": (
                        None
                        if carrier_outer_inventory is None
                        else carrier_outer_inventory[
                            "only_base_checkpoint_tokenizer_and_counted_resources"
                        ]
                    ),
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
                "prepublication_gate_evidence": (
                    _prepublication_gate_evidence(
                        evaluator,
                        evidence_builder,
                    )
                ),
                "folding_only": True,
                "training_steps": 0,
                "parameter_gradients_written": False,
            }
        )
        progress.emit(
            "success",
            receipt_status=receipt["status"],
            training_steps=0,
        )
    except Exception as error:
        receipt = _sealed_receipt(
            {
                "schema": RECEIPT_SCHEMA,
                "status": "failed_closed",
                "configuration_sha256": configuration_sha256,
                "global_topology_phase_cache": {
                    "enabled": (
                        config.dense_heterogeneous_global_hidden_frontier
                    ),
                    "root": (
                        str(config.global_topology_phase_cache_root)
                        if config.dense_heterogeneous_global_hidden_frontier
                        else None
                    ),
                    "required_target_hidden_size": (
                        config.required_target_hidden_size
                    ),
                },
                **_balanced_basis_phase_cache_receipt_fields(config),
                "closed_loop_performance_metrics_policy": (
                    _dense_closed_loop_performance_metrics_policy(config)
                ),
                "closed_loop_replay_policy": (
                    _dense_closed_loop_replay_policy(config, progress)
                ),
                "restart_capsule_policy": (
                    _dense_restart_capsule_policy(config)
                ),
                "on_policy_evidence_policy": (
                    _on_policy_evidence_policy(config)
                ),
                "vocabulary_margin_candidate_gate_policy": (
                    _dense_vocabulary_margin_candidate_gate_policy(config)
                ),
                "macro_cell_attractor_gate_policy": (
                    _dense_macro_cell_attractor_gate_policy(config)
                ),
                "attention_score_candidate_gate_policy": (
                    _dense_attention_score_candidate_gate_policy(config)
                ),
                "parent": runtime,
                "planned_target": _planned_dense_target_evidence(wiring),
                "materialized_target_abi": (
                    None
                    if materialized_target_abi is None
                    else dict(materialized_target_abi)
                ),
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
                "prepublication_gate_evidence": (
                    _prepublication_gate_evidence(
                        evaluator,
                        evidence_builder,
                    )
                ),
                "folding_only": True,
                "training_steps": 0,
                "tolerance_weakened": False,
            }
        )
        progress.emit(
            "failure",
            exception_type=type(error).__name__,
            failure_category=receipt["failure"]["category"],
            fail_closed=True,
        )
    finally:
        try:
            if ledger is not None and not isinstance(ledger, DurableExperimentLedger):
                _write_ledger_atomic(config.ledger_path, ledger)
            if config.diagnostics_path is not None:
                _write_atomic_json(
                    config.diagnostics_path,
                    _diagnostics_document(
                        evaluator,
                        evidence_builder,
                        tokenizer_bridge,
                    ),
                )
            _write_atomic_json(config.receipt_path, receipt)
            progress.emit(
                "final_receipt",
                receipt_sha256=receipt["receipt_sha256"],
                receipt_status=receipt["status"],
                receipt_written=True,
            )
        finally:
            try:
                # Closes only the extension sessions the evidence set opened;
                # the base session below stays owned by this function.
                if (
                    resolved_parent_cache is not None
                    and resolved_parent_cache.evidence_set is not None
                ):
                    resolved_parent_cache.evidence_set.close()
            finally:
                if active_selective_session is not None:
                    active_selective_session.close()
    return receipt


def _suffixed_run_path(path: Path, suffix: str) -> Path:
    """Name one sibling run artifact for a refold pass."""

    return path.with_name(f"{path.stem}-{suffix}{path.suffix}")


def _on_policy_alternation_pass_config(
    config: ProductionLfm2CatalogConfig,
    *,
    pass_index: int,
    consumed_round_manifests: Sequence[Path],
    rounds: int,
) -> ProductionLfm2CatalogConfig:
    """Derive one pass of the fold/round alternation.

    A refold pass differs from the pass before it only by the evidence it
    consumes and by where it publishes: same parent, same sealed base cache,
    same blank and topology, same counted budget, same ledger, same round root.
    The first pass keeps the operator's own destinations, so its fold identity
    is exactly the identity a single non-alternating run would have.
    """

    consumed = tuple(consumed_round_manifests)
    common = {
        "on_policy_refold_after_rounds": 0,
        "on_policy_evidence_rounds": rounds,
        "on_policy_round_manifests": consumed,
    }
    if pass_index == 0:
        return replace(config, **common)
    suffix = ON_POLICY_REFOLD_RUN_SUFFIX.format(index=len(consumed) - 1)
    return replace(
        config,
        **common,
        publish_destination=_suffixed_run_path(
            config.publish_destination,
            suffix,
        ),
        receipt_path=_suffixed_run_path(config.receipt_path, suffix),
        diagnostics_path=(
            None
            if config.diagnostics_path is None
            else _suffixed_run_path(config.diagnostics_path, suffix)
        ),
    )


def _on_policy_alternation_pass_record(
    receipt: Mapping[str, Any],
    *,
    pass_index: int,
    pass_config: ProductionLfm2CatalogConfig,
    consumed_round_manifests: Sequence[Path],
) -> dict[str, Any]:
    evidence = receipt.get("on_policy_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    executed = evidence.get("postpublication_rounds_executed") or ()
    artifact = receipt.get("artifact")
    artifact = artifact if isinstance(artifact, Mapping) else {}
    parent = receipt.get("parent")
    parent = parent if isinstance(parent, Mapping) else {}
    return {
        "pass_index": pass_index,
        "role": "base_fold" if pass_index == 0 else "on_policy_refold",
        "status": receipt["status"],
        "configuration_sha256": receipt["configuration_sha256"],
        "receipt_path": str(pass_config.receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "publish_destination": str(pass_config.publish_destination),
        "artifact_path": artifact.get("path"),
        "artifact_tree_sha256": artifact.get("artifact_tree_sha256"),
        "fresh_reload_accepted": artifact.get("fresh_reload_accepted"),
        "consumed_round_manifests": [
            str(value) for value in consumed_round_manifests
        ],
        "consumed_round_manifest_sha256": list(
            evidence.get("round_manifest_sha256") or ()
        ),
        "extension_cache_keys": list(
            evidence.get("extension_cache_keys") or ()
        ),
        "evidence_identity_sha256": evidence.get("evidence_identity_sha256"),
        "evidence_split_sha256": parent.get("evidence_split_sha256"),
        "evidence_set_cache_key": (
            (evidence.get("evidence_set") or {}).get("evidence_set_cache_key")
        ),
        "sealed_round_manifests": [
            str(value["manifest_path"]) for value in executed
        ],
        "sealed_round_manifest_sha256": [
            str(value["manifest_sha256"]) for value in executed
        ],
        "on_policy_parent_capture_invocations": (
            parent.get("on_policy_parent_capture_invocations")
        ),
        "parent_forward_count_after_cache": (
            parent.get("parent_forward_count_after_cache")
        ),
    }


def run_production_catalog_on_policy_alternation(
    config: ProductionLfm2CatalogConfig,
) -> dict[str, Any]:
    """Alternate folding and on-policy evidence rounds, then seal the sequence.

    NOTES-1C section 6 gives the loop this drives::

        seal base cache      (already done, --sealed-parent-cache-directory)
        fold + publish child -> round 0 reloads that published child
        run round 0          -> extension cache 0
        re-solve on base + extension 0
        fold + publish child -> round 1 reloads the new published child
        ...

    Each pass is one complete ``run_production_catalog`` invocation, so its
    parent rows and evidence are released before the next pass opens its own.
    A refold publishes a new child into a sibling ``-round-N-refold`` run
    directory, appends its fold attempt to the same ledger, and seals its own
    receipt; nothing an earlier pass published is rewritten.  The alternation
    fails closed on the first pass that is not ``complete``.
    """

    if not isinstance(config, ProductionLfm2CatalogConfig):
        raise ProductionLfm2CatalogError("config must be ProductionLfm2CatalogConfig.")
    refolds = int(config.on_policy_refold_after_rounds)
    if refolds <= 0:
        raise ProductionLfm2CatalogError(
            "A fold/round alternation requires --on-policy-refold-after-rounds; "
            "run_production_catalog already executes one fold pass."
        )
    rounds = int(config.on_policy_evidence_rounds)
    consumed = tuple(config.on_policy_round_manifests)
    records: list[dict[str, Any]] = []
    status = "complete"
    failure: dict[str, Any] | None = None
    for pass_index in range(refolds + 1):
        remaining = refolds - pass_index
        pass_config = _on_policy_alternation_pass_config(
            config,
            pass_index=pass_index,
            consumed_round_manifests=consumed,
            rounds=rounds if remaining else 0,
        )
        receipt = run_production_catalog(pass_config)
        record = _on_policy_alternation_pass_record(
            receipt,
            pass_index=pass_index,
            pass_config=pass_config,
            consumed_round_manifests=consumed,
        )
        records.append(record)
        if receipt["status"] != "complete":
            status = "failed_closed"
            failure = {
                "category": "alternation_pass_failed_closed",
                "pass_index": pass_index,
                "pass_failure": dict(receipt.get("failure") or {}),
                "fail_closed": True,
            }
            break
        if not remaining:
            break
        sealed = tuple(
            Path(str(value)) for value in record["sealed_round_manifests"]
        )
        if len(sealed) != rounds:
            status = "failed_closed"
            failure = {
                "category": "alternation_round_not_sealed",
                "pass_index": pass_index,
                "expected_round_count": rounds,
                "sealed_round_count": len(sealed),
                "fail_closed": True,
            }
            break
        consumed = (*consumed, *sealed)
    body = {
        "schema": ON_POLICY_REFOLD_ALTERNATION_SCHEMA,
        "status": status,
        "requested_refold_passes": refolds,
        "rounds_per_pass": rounds,
        "max_rounds": config.on_policy_max_rounds,
        "executed_pass_count": len(records),
        "expected_pass_count": refolds + 1,
        "passes": records,
        "ledger_path": str(config.ledger_path),
        "round_root": str(config.on_policy_round_root_directory),
        "sealed_parent_cache_directory": (
            None
            if config.sealed_parent_cache_directory is None
            else str(config.sealed_parent_cache_directory)
        ),
        "consumed_round_manifests": [str(value) for value in consumed],
        "same_blank_and_counted_budget_every_pass": True,
        "all_extension_rows_are_fit_rows": True,
        "folding_only": True,
        "training_steps": 0,
        **({} if failure is None else {"failure": failure}),
    }
    summary = {**body, "alternation_sha256": canonical_sha256(body)}
    _write_atomic_json(
        _suffixed_run_path(config.receipt_path, "on-policy-alternation"),
        summary,
    )
    return summary


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
        "--target-tokenizer-directory",
        type=Path,
        help=(
            "Explicit local target-tokenizer directory containing a sealed "
            f"{STRICT_TOKENIZER_BRIDGE_FILENAME} manifest."
        ),
    )
    parser.add_argument(
        "--target-tokenizer-calibration",
        type=Path,
        help=(
            "Target-token ID calibration whose one-row bridge mapping exactly "
            "matches the already sealed parent-cache donor IDs."
        ),
    )
    parser.add_argument(
        "--tokenizer-bridge-artifact-sha256",
        help="Expected sealed strict-tokenizer bridge artifact digest.",
    )
    parser.add_argument(
        "--tokenizer-bridge-profile-sha256",
        help="Expected strict one-row transport-profile digest.",
    )
    parser.add_argument(
        "--tokenizer-bridge-mapped-parent-outputs",
        type=Path,
        help=(
            "Sealed one-time parent behavior baseline produced from target-"
            "tokenized IDs mapped through the strict bridge."
        ),
    )
    parser.add_argument(
        "--tokenizer-bridge-mapped-parent-outputs-sha256",
        help="Expected mapped-parent behavior envelope file digest.",
    )
    parser.add_argument(
        "--tokenizer-bridge-minimum-parent-task-retention",
        type=float,
        default=0.9,
        help=(
            "Minimum semantic/correctness/nonempty retention required from "
            "the mapped-sequence parent versus the original canonical parent "
            "before any child fold may run."
        ),
    )
    parser.add_argument(
        "--carrier-route-publication",
        type=Path,
        help="Authenticated local A9/A11 route publication.",
    )
    parser.add_argument(
        "--carrier-route-artifact-sha256",
        help="Expected carrier route publication digest.",
    )
    parser.add_argument(
        "--carrier-span-evidence",
        type=Path,
        help="Authenticated semantic emission-span evidence JSON.",
    )
    parser.add_argument(
        "--carrier-span-evidence-file-sha256",
        help="Expected semantic span evidence file digest.",
    )
    parser.add_argument(
        "--carrier-span-evidence-sha256",
        help="Expected semantic span evidence content digest.",
    )
    parser.add_argument(
        "--semantic-source-tokenizer-directory",
        type=Path,
        help=(
            "Local semantic-token tokenizer copied into the standalone child; "
            "fresh reload never reuses this external path."
        ),
    )
    parser.add_argument(
        "--routed-manual-prompts",
        type=Path,
        help="Sealed whole-response prompt set for routed manual review.",
    )
    parser.add_argument(
        "--routed-manual-prompts-sha256",
        help="Expected routed manual prompt file digest.",
    )
    parser.add_argument(
        "--routed-manual-dossier",
        type=Path,
        help=(
            "Output path for the one post-publication high-cap routed manual "
            "review dossier."
        ),
    )
    parser.add_argument(
        "--routed-manual-checkpoint-directory",
        type=Path,
        help=(
            "External sealed per-prompt review checkpoints. Interrupted 1024-"
            "token rows resume without refolding or regenerating completed rows; "
            "2048-token retries remain limited to cap-censored rows."
        ),
    )
    parser.add_argument(
        "--counted-target-total-parameters",
        type=int,
        default=LFM2_CARRIER_TARGET_TOTAL_PARAMETER_ELEMENTS,
    )
    parser.add_argument(
        "--counted-dense-body-parameters",
        type=int,
        default=LFM2_CARRIER_DENSE_BODY_PARAMETER_ELEMENTS,
    )
    parser.add_argument(
        "--carrier-expected-target-rows",
        type=int,
        default=LFM2_CARRIER_TARGET_ROWS,
    )
    parser.add_argument(
        "--carrier-expected-hidden-size",
        type=int,
        default=LFM2_CARRIER_HIDDEN_SIZE,
    )
    parser.add_argument(
        "--routed-smoke-maximum-prompt-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--routed-smoke-max-new-tokens",
        type=int,
        default=LFM2_CARRIER_SMOKE_MAXIMUM_NEW_TOKENS,
    )
    parser.add_argument(
        "--routed-manual-maximum-prompt-tokens",
        type=int,
        default=8_192,
    )
    parser.add_argument(
        "--routed-manual-initial-max-new-tokens",
        type=int,
        default=LFM2_CARRIER_INITIAL_REVIEW_MAXIMUM_NEW_TOKENS,
    )
    parser.add_argument(
        "--routed-manual-retry-max-new-tokens",
        type=int,
        default=LFM2_CARRIER_RETRY_REVIEW_MAXIMUM_NEW_TOKENS,
    )
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
    parser.add_argument(
        "--derived-global-topology-cache-root",
        type=Path,
        help=(
            "Stable cross-run content-addressed cache root for global "
            "topology phases B1/B2/B3. Required when the automatic global "
            "hidden frontier is enabled; child evidence and final caches "
            "remain under --cache-root."
        ),
    )
    parser.add_argument(
        "--balanced-basis-phase-cache-root",
        type=Path,
        help=(
            "Stable cross-run content-addressed cache root for the exact "
            "circuit-balanced residual basis. Requires the event log, raw "
            "HMAC key file, and key ID options."
        ),
    )
    parser.add_argument(
        "--balanced-basis-phase-event-log",
        type=Path,
        help=(
            "Per-run durable event-chain JSONL for balanced-basis phase "
            "lookup, construction, publication, and reuse."
        ),
    )
    parser.add_argument(
        "--balanced-basis-phase-hmac-key-file",
        type=Path,
        help=(
            "Ordinary non-linked file containing at least 32 exact raw secret "
            "bytes. The bytes are never serialized, logged, or copied."
        ),
    )
    parser.add_argument(
        "--balanced-basis-phase-hmac-key-id",
        help=(
            "Stable non-secret identifier persisted with authenticated "
            "balanced-basis source anchors."
        ),
    )
    parser.add_argument(
        "--balanced-basis-existing-phase-key-sha256",
        help=(
            "Explicit immutable historical balanced-basis phase key to reuse "
            "hit-only after full live parent/trace/target/tokenizer/HMAC "
            "reauthentication. A miss is fatal and never rebuilds."
        ),
    )
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
        "--portable-runtime-device",
        help=(
            "After authenticating the canonical FP32 published child, "
            "relocate its in-memory deployment copy to this exact device."
        ),
    )
    parser.add_argument(
        "--portable-runtime-dtype",
        choices=tuple(PORTABLE_RUNTIME_DTYPES),
        help=(
            "Explicit deployment-only FP16/BF16/FP32/FP64 dtype. This gives "
            "zero parameter-compaction credit and performs no fallback."
        ),
    )
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
        "--dense-minimum-hidden-size",
        type=int,
        default=DEFAULT_DENSE_MINIMUM_HIDDEN_SIZE,
        help=(
            "Minimum residual width searched by the automatic dense LFM2 "
            "topology catalog; must be a positive multiple of 8."
        ),
    )
    parser.add_argument(
        "--required-target-hidden-size",
        type=int,
        help=(
            "Constrain the authenticated global topology frontier to one exact "
            "residual width before cached capacity, functional, and selection "
            "evidence is built. Carrier mode derives this from its registered ABI."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-layout",
        action="store_true",
        help=(
            "Select a sealed per-layer heterogeneous dense layout from the "
            "existing parent trace cache and solve its configured exact "
            "parameter budget."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-global-hidden-frontier",
        action="store_true",
        help=(
            "Choose the residual width dynamically from sealed cached "
            "functional and per-component evidence while preserving the "
            "same complete exact-budget heterogeneous child path."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-factor-writer-rank-discovery",
        action="store_true",
        help=(
            "Derive the exact compact ShortConv writer rank from the requested "
            "integer lattice, score its output action from the sealed cache, and "
            "expose a factor option at only the dynamically safest layer."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-global-functional-rows-per-role",
        type=int,
        default=2048,
        help=(
            "Maximum authenticated cached rows per causal role used by the "
            "global residual-width functional spectrum."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-hidden-size",
        type=int,
        default=784,
        help=(
            "Residual-stream width pinned for cached heterogeneous dense "
            "layout selection."
        ),
    )
    parser.add_argument(
        "--dense-heterogeneous-target-parameters",
        type=int,
        default=57_420_144,
        help=(
            "Exact physical learned-parameter count solved by the cached "
            "heterogeneous layout selector; packaged continuous route state "
            "is counted separately inside the total child target."
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
        "--dense-body-reclosure-mode",
        choices=("disabled", "moving_gauge_v3"),
        default="disabled",
        help=(
            "Whole-body physical reclosure used after the ordered layer fold. "
            "moving_gauge_v3 performs boundary-local forward/reverse residual "
            "reconciliation and requires --dense-run-closed-loop."
        ),
    )
    parser.add_argument(
        "--dense-closed-loop-replay-mode",
        choices=(
            DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE,
            CHILD_REPLAY_SINGLE_PASS_CAUSAL_MODE,
            CHILD_REPLAY_PREFIX_BY_PREFIX_MODE,
        ),
        default=DENSE_CLOSED_LOOP_REPLAY_AUTO_MODE,
        help=(
            "Child replay policy used throughout dense closed-loop folding. "
            "Auto always resolves to canonical prefix-by-prefix replay. "
            "single_pass_causal is the explicit opt-in that additionally "
            "certifies every capture against a canonical reference replay "
            "inside the loop, and requires the authenticated heterogeneous "
            "dense FP32 child path."
        ),
    )
    parser.add_argument(
        "--dense-closed-loop-merge-adjacent-layer-captures",
        action="store_true",
        help=(
            "Serve the POST_DOWNSTREAM capture of layer L and the PRE_PRODUCER "
            "capture of layer L+1 from one union-selection child replay. Both "
            "points observe the same child state; each still receives its own "
            "sealed capture envelope. Off by default because it changes sealed "
            "run-report content and the child_replay_capture_calls counter."
        ),
    )
    parser.add_argument(
        "--dense-replay-devices",
        type=str,
        default=None,
        help=(
            "Comma-separated device list that shards every dense closed-loop "
            "child replay across several identical devices, for example "
            "'cuda:0,cuda:1,cuda:2,cuda:3'. The live child's own device is "
            "always worker zero, only one identical device class is used, and "
            "no worker receives zero batches. The single token 'all' selects "
            "every visible identical device. Unset keeps the serial capture "
            "path, and an unusable request falls back to it with the reason "
            "recorded in the receipt; the sealed replay bytes are identical "
            "either way. Overrides "
            f"${REPLAY_DEVICES_ENVIRONMENT_VARIABLE}."
        ),
    )
    parser.add_argument(
        "--dense-restart-capsule-root",
        type=Path,
        default=None,
        help=(
            "Root directory for accepted-layer restart capsules written during "
            "the dense closed-loop body fold. Defaults to a run-scoped "
            "closed-loop-restart-capsules directory under --cache-root "
            "whenever --dense-run-closed-loop is set."
        ),
    )
    parser.add_argument(
        "--dense-restart-capsule-keep",
        type=int,
        default=2,
        help=(
            "Number of newest accepted-layer restart capsules retained by "
            "rotation."
        ),
    )
    parser.add_argument(
        "--dense-resume-from-capsule",
        type=Path,
        default=None,
        help=(
            "Restart-capsule root to resume the dense closed-loop body fold "
            "from. The newest authenticated capsule under that root is loaded "
            "into the fresh child before the closed-loop adapter is built, and "
            "its accepted ascending layer prefix is skipped. Fails closed with "
            "--dense-body-reclosure-mode moving_gauge_v3, which requires this "
            "process to install every layer."
        ),
    )
    parser.add_argument(
        "--on-policy-evidence-rounds",
        type=int,
        default=0,
        help=(
            "Number of bounded on-policy evidence extension rounds to run "
            "after a successful publication. Zero keeps today's "
            "teacher-forced-only lane."
        ),
    )
    parser.add_argument(
        "--on-policy-max-rounds",
        type=int,
        default=DEFAULT_ON_POLICY_MAX_ROUNDS,
        help="Bounded round budget written into every round manifest.",
    )
    parser.add_argument(
        "--on-policy-refold-after-rounds",
        type=int,
        default=0,
        help=(
            "Number of further fold passes that consume the rounds each pass "
            "sealed. Zero runs exactly one fold. A refold reuses the same "
            "blank, topology, and counted budget, changes only the evidence, "
            "and publishes a new child into a sibling '-round-N-refold' run "
            "directory."
        ),
    )
    parser.add_argument(
        "--on-policy-child-artifact-directory",
        type=Path,
        default=None,
        help=(
            "Standalone published child the round reloads. Defaults to the "
            "child this run just published."
        ),
    )
    parser.add_argument(
        "--on-policy-fit-prompts",
        type=Path,
        default=None,
        help=(
            "JSON list of fit prompts for the child-only rollout. Required "
            "when --on-policy-evidence-rounds is positive."
        ),
    )
    parser.add_argument(
        "--on-policy-manual-prompt-manifest",
        type=Path,
        default=None,
        help=(
            "Reserved manual-review prompt manifest. Required when "
            "--on-policy-evidence-rounds is positive so every fit prompt is "
            "provably disjoint from the held-out review corpus."
        ),
    )
    parser.add_argument(
        "--on-policy-round-root",
        type=Path,
        default=None,
        help=(
            "Output root for sealed on-policy rounds; defaults to "
            "on-policy-rounds under --cache-root."
        ),
    )
    parser.add_argument(
        "--on-policy-round-manifest",
        dest="on_policy_round_manifests",
        type=Path,
        action="append",
        default=None,
        help=(
            "Already sealed on-policy round manifest to consume, repeatable "
            "and ordered from round 0. Every consumed extension cache key and "
            "round manifest digest enters the derived phase-cache keys."
        ),
    )
    parser.add_argument(
        "--on-policy-max-prompts",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--on-policy-max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--on-policy-max-context-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--on-policy-max-capture-batches",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--on-policy-rollout-seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dense-closed-loop-performance-metrics-diagnostic-only",
        action="store_true",
        help=(
            "Record closed-loop RMSE/KL/NLL/EOS thresholds as "
            "counterfactual diagnostics instead of transaction authority. "
            "Nonfinite values and structural/integrity failures still reject."
        ),
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-gate",
        action="store_true",
        help=(
            "Gate every closed-loop residual-attention macro-cell on "
            "child-only free-rollout entropy, repetition, dominance, EOS "
            "timing, and newly collapsed trajectories."
        ),
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-records",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-entropy-regression",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-repetition-regression",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-dominance-regression",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-eos-regression",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-collapse-entropy-threshold",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-collapse-repetition-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-collapse-dominance-threshold",
        type=float,
        default=0.80,
    )
    parser.add_argument(
        "--dense-macro-cell-attractor-maximum-new-collapsed-records",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dense-attention-consumer-mode",
        choices=("output-only", "coupled-value-output"),
        default="output-only",
        help=(
            "Choose O-only attention-consumer fitting or the coupled V/O "
            "solver that holds the score-gated Q/K path fixed."
        ),
    )
    parser.add_argument(
        "--dense-attention-geometry-risk",
        action="store_true",
        help=(
            "Allocate exact-budget attention topologies using streamed cached "
            "masked Q/K decisions, K/V memory, head context, and residual pressure."
        ),
    )
    parser.add_argument(
        "--dense-attention-score-candidate-gate",
        action="store_true",
        help=(
            "Reject a compiled Q/K producer before installation when held-out "
            "masked attention scores lose key order or decision margins."
        ),
    )
    parser.add_argument(
        "--dense-attention-score-top-k",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--dense-attention-score-margin-floor-fraction",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--dense-attention-score-minimum-top1-key-agreement",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dense-attention-score-minimum-top-k-pair-order-agreement",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dense-attention-score-minimum-winner-margin-sign-agreement",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dense-attention-score-minimum-relative-margin-floor-retention",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--dense-attention-score-maximum-softmax-fisher-error",
        type=float,
    )
    parser.add_argument(
        "--dense-attention-score-maximum-js-divergence",
        type=float,
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
        "--dense-streaming-vocabulary-margin-v2",
        action="store_true",
        help=(
            "Opt into the all-row sealed-cache streaming vocabulary-margin "
            "Gram and lazy held-out diagnostics; legacy behavior remains the "
            "default."
        ),
    )
    parser.add_argument(
        "--dense-streaming-vocabulary-margin-pair-chunk-size",
        type=int,
        default=256,
        help=(
            "Deterministic phase-local pair chunk size for streaming margin "
            "arithmetic; this is not a row cap or sampling control."
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
        "--final-performance-metrics-diagnostic-only",
        action="store_true",
        help=(
            "Always produce decoded child responses for manual intelligence "
            "review. Trajectory, task, repetition, and termination metrics are "
            "recorded as counterfactual diagnostics; output-cap hits are censored. "
            "Structural, nonfinite, package, and fresh-reload failures still stop."
        ),
    )
    parser.add_argument(
        "--snapshot-chunk-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    return parser


def _parse_replay_device_list(value: str, source_name: str) -> tuple[str, ...]:
    entries = tuple(
        part.strip() for part in str(value).split(",") if part.strip()
    )
    if not entries:
        raise ProductionLfm2CatalogError(
            f"{source_name} named no replay devices."
        )
    return entries


def resolve_replay_device_request(
    cli_value: str | None,
    *,
    dense_run_closed_loop: bool,
    environment: Mapping[str, str] | None = None,
) -> tuple[tuple[str, ...], str]:
    """Resolve the shard request and record where it came from.

    The CLI flag always wins.  The environment is consulted only for a
    closed-loop run: an operator who exports a device list for the folding
    lane must not silently change, or fail, an unrelated invocation of this
    runner.  An ignored environment request is reported as such rather than
    being erased.
    """

    if cli_value is not None and str(cli_value).strip():
        return (
            _parse_replay_device_list(cli_value, "--dense-replay-devices"),
            REPLAY_DEVICES_SOURCE_CLI,
        )
    source_environment = os.environ if environment is None else environment
    for variable in (
        REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
        LEGACY_REPLAY_DEVICES_ENVIRONMENT_VARIABLE,
    ):
        raw = str(source_environment.get(variable, "") or "").strip()
        if not raw:
            continue
        if not dense_run_closed_loop:
            return (), REPLAY_DEVICES_SOURCE_ENVIRONMENT_IGNORED
        return (
            _parse_replay_device_list(raw, f"${variable}"),
            REPLAY_DEVICES_SOURCE_ENVIRONMENT,
        )
    return (), REPLAY_DEVICES_SOURCE_UNSET


def config_from_args(args: argparse.Namespace) -> ProductionLfm2CatalogConfig:
    replay_devices, replay_devices_source = resolve_replay_device_request(
        getattr(args, "dense_replay_devices", None),
        dense_run_closed_loop=bool(args.dense_run_closed_loop),
    )
    return ProductionLfm2CatalogConfig(
        parent_model=args.parent_model,
        tokenizer_directory=args.tokenizer_directory,
        target_tokenizer_directory=args.target_tokenizer_directory,
        target_tokenizer_calibration=args.target_tokenizer_calibration,
        tokenizer_bridge_artifact_sha256=(
            args.tokenizer_bridge_artifact_sha256
        ),
        tokenizer_bridge_profile_sha256=(
            args.tokenizer_bridge_profile_sha256
        ),
        tokenizer_bridge_mapped_parent_outputs=(
            args.tokenizer_bridge_mapped_parent_outputs
        ),
        tokenizer_bridge_mapped_parent_outputs_sha256=(
            args.tokenizer_bridge_mapped_parent_outputs_sha256
        ),
        tokenizer_bridge_minimum_parent_task_retention=(
            args.tokenizer_bridge_minimum_parent_task_retention
        ),
        carrier_route_publication=args.carrier_route_publication,
        carrier_route_artifact_sha256=args.carrier_route_artifact_sha256,
        carrier_span_evidence=args.carrier_span_evidence,
        carrier_span_evidence_file_sha256=(
            args.carrier_span_evidence_file_sha256
        ),
        carrier_span_evidence_sha256=args.carrier_span_evidence_sha256,
        semantic_source_tokenizer_directory=(
            args.semantic_source_tokenizer_directory
        ),
        routed_manual_prompts=args.routed_manual_prompts,
        routed_manual_prompts_sha256=args.routed_manual_prompts_sha256,
        routed_manual_dossier_path=args.routed_manual_dossier,
        routed_manual_checkpoint_directory=(
            args.routed_manual_checkpoint_directory
        ),
        counted_target_total_parameters=(
            args.counted_target_total_parameters
        ),
        counted_dense_body_parameters=args.counted_dense_body_parameters,
        carrier_expected_target_rows=args.carrier_expected_target_rows,
        carrier_expected_hidden_size=args.carrier_expected_hidden_size,
        routed_smoke_maximum_prompt_tokens=(
            args.routed_smoke_maximum_prompt_tokens
        ),
        routed_smoke_max_new_tokens=args.routed_smoke_max_new_tokens,
        routed_manual_maximum_prompt_tokens=(
            args.routed_manual_maximum_prompt_tokens
        ),
        routed_manual_initial_max_new_tokens=(
            args.routed_manual_initial_max_new_tokens
        ),
        routed_manual_retry_max_new_tokens=(
            args.routed_manual_retry_max_new_tokens
        ),
        calibration=args.calibration,
        sealed_parent_cache_directory=args.sealed_parent_cache_directory,
        cached_parent_outputs=args.cached_parent_outputs,
        cached_parent_outputs_sha256=(args.cached_parent_outputs_sha256),
        cache_root=args.cache_root,
        derived_global_topology_cache_root=(
            args.derived_global_topology_cache_root
        ),
        balanced_basis_phase_cache_root=(
            args.balanced_basis_phase_cache_root
        ),
        balanced_basis_phase_event_log=(
            args.balanced_basis_phase_event_log
        ),
        balanced_basis_phase_hmac_key_file=(
            args.balanced_basis_phase_hmac_key_file
        ),
        balanced_basis_phase_hmac_key_id=(
            args.balanced_basis_phase_hmac_key_id
        ),
        balanced_basis_existing_phase_key_sha256=(
            args.balanced_basis_existing_phase_key_sha256
        ),
        snapshot_root=args.snapshot_root,
        publish_destination=args.publish_destination,
        ledger_path=args.ledger,
        receipt_path=args.receipt,
        diagnostics_path=args.diagnostics,
        parent_name=args.parent_name,
        child_name=args.child_name,
        device=args.device,
        dtype=args.dtype,
        portable_runtime_device=args.portable_runtime_device,
        portable_runtime_dtype=args.portable_runtime_dtype,
        target_query_heads=args.target_query_heads,
        target_key_value_heads=args.target_key_value_heads,
        target_intermediate_size=args.target_intermediate_size,
        exact_target_unique_parameters=args.exact_target_parameters,
        compression_ratio=args.compression_ratio,
        dense_maximum_candidates=args.dense_maximum_candidates,
        dense_minimum_hidden_size=args.dense_minimum_hidden_size,
        required_target_hidden_size=args.required_target_hidden_size,
        dense_heterogeneous_layout=args.dense_heterogeneous_layout,
        dense_heterogeneous_global_hidden_frontier=(
            args.dense_heterogeneous_global_hidden_frontier
        ),
        dense_heterogeneous_factor_writer_rank_discovery=(
            args.dense_heterogeneous_factor_writer_rank_discovery
        ),
        dense_heterogeneous_global_functional_rows_per_role=(
            args.dense_heterogeneous_global_functional_rows_per_role
        ),
        dense_heterogeneous_hidden_size=(
            args.dense_heterogeneous_hidden_size
        ),
        dense_heterogeneous_target_parameters=(
            args.dense_heterogeneous_target_parameters
        ),
        dense_residual_basis_mode=args.dense_residual_basis_mode,
        dense_body_reclosure_mode=args.dense_body_reclosure_mode,
        dense_run_closed_loop=args.dense_run_closed_loop,
        dense_closed_loop_performance_metrics_diagnostic_only=(
            args.dense_closed_loop_performance_metrics_diagnostic_only
        ),
        dense_closed_loop_replay_capture_mode=(
            args.dense_closed_loop_replay_mode
        ),
        dense_closed_loop_merge_adjacent_layer_captures=(
            args.dense_closed_loop_merge_adjacent_layer_captures
        ),
        dense_replay_devices=replay_devices,
        dense_replay_devices_source=replay_devices_source,
        dense_restart_capsule_root=args.dense_restart_capsule_root,
        dense_restart_capsule_keep=args.dense_restart_capsule_keep,
        dense_resume_from_capsule=args.dense_resume_from_capsule,
        on_policy_evidence_rounds=args.on_policy_evidence_rounds,
        on_policy_max_rounds=args.on_policy_max_rounds,
        on_policy_refold_after_rounds=(
            args.on_policy_refold_after_rounds
        ),
        on_policy_child_artifact_directory=(
            args.on_policy_child_artifact_directory
        ),
        on_policy_fit_prompts=args.on_policy_fit_prompts,
        on_policy_manual_prompt_manifest=(
            args.on_policy_manual_prompt_manifest
        ),
        on_policy_round_root=args.on_policy_round_root,
        on_policy_round_manifests=tuple(
            args.on_policy_round_manifests or ()
        ),
        on_policy_max_prompts=args.on_policy_max_prompts,
        on_policy_max_new_tokens=args.on_policy_max_new_tokens,
        on_policy_max_context_tokens=args.on_policy_max_context_tokens,
        on_policy_max_capture_batches=args.on_policy_max_capture_batches,
        on_policy_rollout_seed=args.on_policy_rollout_seed,
        dense_macro_cell_attractor_gate_enabled=(
            args.dense_macro_cell_attractor_gate
        ),
        dense_macro_cell_attractor_maximum_records=(
            args.dense_macro_cell_attractor_maximum_records
        ),
        dense_macro_cell_attractor_maximum_steps=(
            args.dense_macro_cell_attractor_maximum_steps
        ),
        dense_macro_cell_attractor_maximum_entropy_regression=(
            args.dense_macro_cell_attractor_maximum_entropy_regression
        ),
        dense_macro_cell_attractor_maximum_repetition_regression=(
            args.dense_macro_cell_attractor_maximum_repetition_regression
        ),
        dense_macro_cell_attractor_maximum_dominance_regression=(
            args.dense_macro_cell_attractor_maximum_dominance_regression
        ),
        dense_macro_cell_attractor_maximum_eos_regression=(
            args.dense_macro_cell_attractor_maximum_eos_regression
        ),
        dense_macro_cell_attractor_collapse_entropy_threshold=(
            args.dense_macro_cell_attractor_collapse_entropy_threshold
        ),
        dense_macro_cell_attractor_collapse_repetition_threshold=(
            args.dense_macro_cell_attractor_collapse_repetition_threshold
        ),
        dense_macro_cell_attractor_collapse_dominance_threshold=(
            args.dense_macro_cell_attractor_collapse_dominance_threshold
        ),
        dense_macro_cell_attractor_maximum_new_collapsed_records=(
            args.dense_macro_cell_attractor_maximum_new_collapsed_records
        ),
        dense_attention_consumer_mode=(
            args.dense_attention_consumer_mode
        ),
        dense_attention_geometry_risk_enabled=(
            args.dense_attention_geometry_risk
        ),
        dense_attention_score_candidate_gate_enabled=(
            args.dense_attention_score_candidate_gate
        ),
        dense_attention_score_top_k=args.dense_attention_score_top_k,
        dense_attention_score_margin_floor_fraction=(
            args.dense_attention_score_margin_floor_fraction
        ),
        dense_attention_score_minimum_top1_key_agreement_fraction=(
            args.dense_attention_score_minimum_top1_key_agreement
        ),
        dense_attention_score_minimum_top_k_pair_order_agreement_fraction=(
            args
            .dense_attention_score_minimum_top_k_pair_order_agreement
        ),
        dense_attention_score_minimum_winner_margin_sign_agreement_fraction=(
            args
            .dense_attention_score_minimum_winner_margin_sign_agreement
        ),
        dense_attention_score_minimum_relative_margin_floor_retention_fraction=(
            args
            .dense_attention_score_minimum_relative_margin_floor_retention
        ),
        dense_attention_score_maximum_softmax_fisher_error=(
            args.dense_attention_score_maximum_softmax_fisher_error
        ),
        dense_attention_score_maximum_js_divergence=(
            args.dense_attention_score_maximum_js_divergence
        ),
        dense_vocabulary_margin_observability=(
            args.dense_vocabulary_margin_observability
        ),
        dense_vocabulary_margin_top_k=(
            args.dense_vocabulary_margin_top_k
        ),
        dense_vocabulary_margin_floor_fraction=(
            args.dense_vocabulary_margin_floor_fraction
        ),
        dense_streaming_vocabulary_margin_v2=(
            args.dense_streaming_vocabulary_margin_v2
        ),
        dense_streaming_vocabulary_margin_pair_chunk_size=(
            args.dense_streaming_vocabulary_margin_pair_chunk_size
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
        final_performance_metrics_diagnostic_only=(
            args.final_performance_metrics_diagnostic_only
        ),
        snapshot_chunk_bytes=args.snapshot_chunk_bytes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = config_from_args(build_parser().parse_args(argv))
    if config.on_policy_refold_after_rounds:
        summary = run_production_catalog_on_policy_alternation(config)
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 0 if summary["status"] == "complete" else 2
    receipt = run_production_catalog(config)
    print(json.dumps(receipt, separators=(",", ":"), sort_keys=True))
    return 0 if receipt["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
