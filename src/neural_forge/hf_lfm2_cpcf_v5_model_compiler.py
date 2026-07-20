"""Opt-in bounded-residency LFM2 CPCF v5 model compiler.

The legacy v4 compiler intentionally remains untouched and is still the
default production lane.  This module is a separately fingerprinted v5 lane
which:

* keeps canonical target-phase metadata in tensor-free evidence shells;
* loads only one authenticated source layer plus its prefetched successor;
* fits each boundary chart immediately before compiling that same layer;
* drops tensor-owning replay results before advancing the residency window;
* optionally routes short-conv candidate replay through an injected
  authenticated evaluator; and
* continuously samples physical process RSS in addition to reporting logical
  evidence bytes.

It consumes an already committed local trace cache through
``Lfm2CpcfSelectiveEvidenceProvider``.  It never runs a parent, trains,
prunes, quantizes, or changes the v4 byte/provenance contract.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import secrets
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, MappingProxyType, ModuleType
from typing import Any

from neural_forge.dynamic_circuit_ir import canonical_sha256
from neural_forge.dynamic_evidence_residency import (
    DynamicEvidenceLayerAuthentication,
    DynamicEvidenceResidencyWindow,
)
from neural_forge.hf_cpcf_boundary_chart import fit_cpcf_boundary_chart
from neural_forge import hf_lfm2_cpcf_boundary_bundle as boundary
from neural_forge import hf_lfm2_cpcf_model_compiler as model_compiler
from neural_forge import hf_lfm2_cpcf_model_fold as legacy_model_fold
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    Lfm2CpcfCandidateReplayEvaluator,
)
from neural_forge.hf_lfm2_cpcf_installability import (
    Lfm2CpcfInstallableLayerEvidence,
    Lfm2CpcfInstallablePhaseEvidence,
)
from neural_forge.hf_lfm2_cpcf_layer_compiler import (
    LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS,
    LFM2_CPCF_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS,
    Lfm2CpcfEndpointSequentialTransportPolicy,
    compile_and_install_lfm2_cpcf_layer,
)
from neural_forge.hf_lfm2_cpcf_residual_error_controller import (
    Lfm2CpcfPostOperatorTransportPolicy,
    fit_lfm2_cpcf_post_operator_chart_transport_for_profile,
)
from neural_forge.hf_lfm2_composed_qk_attention_fold import (
    ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
    LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA,
    LFM2_ENDPOINT_QK_START_SCHEMA,
    SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    Lfm2ComposedEndpointResidualAttentionFoldPolicy,
    Lfm2ComposedQkAttentionFoldPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    Lfm2CpcfShortConvNumericalFrontierAdapter,
    Lfm2CpcfShortConvV5CandidateSelector,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    rescore_sealed_v4_short_conv_selection_frontier,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    LFM2_CPCF_TARGET_PHASE_SCHEMA,
    Lfm2CpcfTargetPhase,
    install_lfm2_cpcf_neutral_norms,
    install_lfm2_cpcf_tied_interface,
    rebind_lfm2_cpcf_phase_state_bank,
    seed_lfm2_cpcf_phase_states,
)
from neural_forge.hf_lfm2_cpcf_selective_evidence import (
    Lfm2CpcfSelectiveEvidenceCatalog,
    Lfm2CpcfSelectiveEvidenceProvider,
    Lfm2CpcfSelectiveFinalNormPayload,
    Lfm2CpcfSelectiveLayerPayload,
    lfm2_cpcf_selective_provider_implementation,
)
from neural_forge.hf_lfm2_cpcf_v5_resume import (
    Lfm2CpcfV5BoundaryFrontier,
    Lfm2CpcfV5PhysicalRssCheckpointExpectation,
    Lfm2CpcfV5PhysicalRssCheckpointReceipt,
    Lfm2CpcfV5ProgressEvent,
    Lfm2CpcfV5ResumeCoordinator,
    Lfm2CpcfV5ResumeState,
    build_lfm2_cpcf_v5_physical_rss_policy_sha256,
    build_lfm2_cpcf_v5_resume_context,
    lfm2_cpcf_v5_resume_implementation,
)
from neural_forge.hf_lfm2_cpcf_v5_admission_proof import (
    build_lfm2_cpcf_v5_admission_chain_proof,
)
from neural_forge.hf_lfm2_cpcf_terminal_action_residual import (
    Lfm2CpcfTerminalActionPolicy,
)
from neural_forge.hf_lfm2_cpcf_v5_terminal_action_integration import (
    Lfm2CpcfV5TerminalActionCapture,
    capture_lfm2_cpcf_v5_terminal_action_inputs,
    compile_install_lfm2_cpcf_v5_terminal_action,
)
from neural_forge.hf_lfm2_cpcf_resume_artifact import (
    build_lfm2_cpcf_target_topology_snapshot,
)
from neural_forge.hf_tied_causal_interface_anchor_fold import (
    fit_tied_causal_interface_anchor_residual,
)
from neural_forge.hf_tied_causal_interface_fold import (
    fit_tied_causal_interface,
)


LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-compiler-implementation-v1"
)
LFM2_CPCF_V5_MODEL_FOLD_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-model-fold-v1"
LFM2_CPCF_V5_RESUME_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-resume-v1"
LFM2_CPCF_V5_ARTIFACT_SCHEMA = "neural-forge-lfm2-cpcf-v5-selective-artifact-v1"
LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA = "neural-forge-lfm2-cpcf-v5-physical-rss-v1"
LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-checkpoint-v1"
)
LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-physical-rss-expectation-v1"
)
LFM2_CPCF_V5_COMPOSED_ATTENTION_COVERAGE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-composed-attention-coverage-v1"
)
LFM2_CPCF_V5_ENDPOINT_RESIDUAL_ATTENTION_COVERAGE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-endpoint-residual-attention-coverage-v1"
)
LFM2_CPCF_V5_ENDPOINT_SEQUENTIAL_TRANSPORT_COVERAGE_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-endpoint-sequential-transport-coverage-v1"
)
LFM2_CPCF_V5_TERMINAL_RESUME_PUBLICATION_SCHEMA = (
    "neural-forge-lfm2-cpcf-v5-terminal-resume-publication-v1"
)
LFM2_CPCF_V5_COMPOSED_ATTENTION_LAYER_INDICES = (2, 4, 6, 8, 10, 12)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DECODE_PHASE = re.compile(r"decode\.(\d{3,})\Z")
_SPLITS = ("selection", "fit", "validation")
_BOUNDARIES = ("residual.post_operator", "residual.post_layer")
_RSS_SAMPLE_INTERVAL_SECONDS = 0.05


class Lfm2CpcfV5ModelCompilerError(RuntimeError):
    """Selective evidence, physical residency, or v5 compilation failed."""


class Lfm2CpcfV5PhysicalRssError(Lfm2CpcfV5ModelCompilerError):
    """A sealed physical-RSS receipt rejected a pending prefix."""

    def __init__(
        self,
        message: str,
        report: Lfm2CpcfV5PhysicalRssCheckpointReceipt,
    ) -> None:
        super().__init__(message)
        if type(report) is not Lfm2CpcfV5PhysicalRssCheckpointReceipt:
            raise TypeError("report must be the exact typed RSS receipt.")
        self.report = report


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5EndpointFirstCandidateSelector:
    """Apply the source-bound endpoint-first v5 policy before installation."""

    def select(
        self,
        request: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("request must be an authenticated mapping.")
        reports = request.get("candidate_selection_reports")
        if not isinstance(reports, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                "Candidate-selector request has no sealed selection frontier."
            )
        return rescore_sealed_v4_short_conv_selection_frontier(reports)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for LFM2 CPCF v5 model compilation."
        ) from exc
    return torch


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
        raise Lfm2CpcfV5ModelCompilerError(
            "V5 compiler metadata must be finite canonical JSON."
        ) from exc


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_metadata(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(child) for child in value]
    return value


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfV5ModelCompilerError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be a positive integer.")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be within [0, 1].")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be within [0, 1].") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be within [0, 1].")
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Cannot fingerprint compiler source {path}."
        ) from exc
    return digest.hexdigest()


def lfm2_cpcf_v5_compiler_implementation() -> Mapping[str, Any]:
    """Return the source-bound v5 compiler and resume/artifact contract."""

    from neural_forge import dynamic_evidence_residency
    from neural_forge import dynamic_replay_scheduler
    from neural_forge import hf_cpcf_boundary_chart
    from neural_forge import hf_cpcf_sequential_boundary_transport
    from neural_forge import hf_lfm2_cached_qk_functional_refit
    from neural_forge import hf_lfm2_cached_qk_score_residual_fold
    from neural_forge import hf_lfm2_cached_endpoint_qk_residual_fold
    from neural_forge import hf_lfm2_composed_qk_attention_fold
    from neural_forge import hf_lfm2_cpcf_attention_dimension_fold
    from neural_forge import hf_lfm2_cpcf_candidate_replay
    from neural_forge import hf_lfm2_cpcf_layer_compiler
    from neural_forge import hf_lfm2_cpcf_macrocell_replay
    from neural_forge import hf_lfm2_cpcf_model_compiler
    from neural_forge import hf_lfm2_cpcf_residual_error_controller
    from neural_forge import hf_lfm2_cpcf_short_conv_v5
    from neural_forge import hf_lfm2_cpcf_short_conv_v5_fhvr_mf
    from neural_forge import hf_lfm2_cpcf_short_conv_v5_numerical_adapter
    from neural_forge import hf_lfm2_cpcf_short_conv_v4_fold
    from neural_forge import hf_lfm2_cpcf_selective_evidence
    from neural_forge import hf_lfm2_qk_global_gauge_sync
    from neural_forge import hf_lfm2_sequence_effective_attention_fold
    from neural_forge import hf_lfm2_structural_residual_attention_fold
    from neural_forge import hf_lfm2_cpcf_terminal_action_residual
    from neural_forge import hf_lfm2_cpcf_v5_terminal_action_integration

    modules = (
        sys.modules[__name__],
        boundary,
        dynamic_evidence_residency,
        dynamic_replay_scheduler,
        hf_cpcf_boundary_chart,
        hf_cpcf_sequential_boundary_transport,
        hf_lfm2_cached_qk_functional_refit,
        hf_lfm2_cached_qk_score_residual_fold,
        hf_lfm2_cached_endpoint_qk_residual_fold,
        hf_lfm2_composed_qk_attention_fold,
        hf_lfm2_cpcf_attention_dimension_fold,
        hf_lfm2_cpcf_candidate_replay,
        hf_lfm2_cpcf_layer_compiler,
        hf_lfm2_cpcf_macrocell_replay,
        hf_lfm2_cpcf_model_compiler,
        hf_lfm2_cpcf_residual_error_controller,
        legacy_model_fold,
        hf_lfm2_cpcf_short_conv_v5,
        hf_lfm2_cpcf_short_conv_v5_fhvr_mf,
        hf_lfm2_cpcf_short_conv_v5_numerical_adapter,
        hf_lfm2_cpcf_short_conv_v4_fold,
        hf_lfm2_cpcf_selective_evidence,
        hf_lfm2_qk_global_gauge_sync,
        hf_lfm2_sequence_effective_attention_fold,
        hf_lfm2_structural_residual_attention_fold,
        hf_lfm2_cpcf_terminal_action_residual,
        hf_lfm2_cpcf_v5_terminal_action_integration,
        sys.modules["neural_forge.hf_lfm2_cpcf_v5_resume"],
    )
    files = []
    for module in modules:
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not raw:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Compiler module {module.__name__!r} has no source file."
            )
        path = Path(raw).resolve()
        files.append(
            {
                "module": str(module.__name__),
                "filename": path.name,
                "sha256": _file_sha256(path),
            }
        )
    files.sort(key=lambda row: row["module"])
    short_conv_v5_source = next(
        row
        for row in files
        if row["module"] == "neural_forge.hf_lfm2_cpcf_short_conv_v5"
    )
    provider = dict(lfm2_cpcf_selective_provider_implementation())
    body = {
        "schema": LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA,
        "model_fold_schema": LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
        "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
        "artifact_schema": LFM2_CPCF_V5_ARTIFACT_SCHEMA,
        "ordered_source_files": files,
        "selective_provider_implementation_sha256": (provider["implementation_sha256"]),
        "short_conv_v5_numerical_policy_schema": (
            hf_lfm2_cpcf_short_conv_v5.LFM2_CPCF_SHORT_CONV_V5_SCHEMA
        ),
        "short_conv_v5_numerical_policy_source_sha256": (
            short_conv_v5_source["sha256"]
        ),
        "legacy_v4_default_unchanged": True,
        "selective_evidence_required": True,
        "current_plus_next_residency_required": True,
        "physical_rss_gate_required": True,
        "short_conv_v5_numerical_policy_integration_status": (
            "real_v5_numerical_frontier_integrated_with_per_layer_provenance_gate"
        ),
        "resume_implementation_status": (
            "authenticated_partial_boundary_frontier_integrated"
        ),
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "implementation_sha256": canonical_sha256(body)})


@dataclass(frozen=True, slots=True)
class Lfm2CpcfV5RuntimePolicy:
    """Bounded evidence and physical-RSS controls for the opt-in v5 lane."""

    model_fold_policy: legacy_model_fold.Lfm2CpcfModelFoldPolicy
    maximum_resident_evidence_bytes: int
    full_phase_tree_logical_bytes: int
    maximum_process_rss_bytes: int = (1 << 63) - 1
    minimum_physical_rss_reduction_fraction: float = 0.10
    rss_sample_interval_seconds: float = _RSS_SAMPLE_INTERVAL_SECONDS
    require_short_conv_v5_endpoint_first_selector: bool = False
    require_short_conv_v5_authenticated_whole_macrocell_selector: bool = False
    require_short_conv_v5_real_numerical_frontier_adapter: bool = False
    maximum_physical_rss_increase_cap_bytes: int | None = None
    composed_qk_attention_policy: Lfm2ComposedQkAttentionFoldPolicy | None = None
    endpoint_residual_attention_policy: (
        Lfm2ComposedEndpointResidualAttentionFoldPolicy | None
    ) = None
    terminal_action_residual_policy: Lfm2CpcfTerminalActionPolicy | None = None
    endpoint_sequential_transport_policy: (
        Lfm2CpcfEndpointSequentialTransportPolicy | None
    ) = None
    post_operator_transport_policy: Lfm2CpcfPostOperatorTransportPolicy | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.model_fold_policy,
            legacy_model_fold.Lfm2CpcfModelFoldPolicy,
        ):
            raise TypeError("model_fold_policy must be Lfm2CpcfModelFoldPolicy.")
        _positive_int(
            self.maximum_resident_evidence_bytes,
            "maximum_resident_evidence_bytes",
        )
        _positive_int(
            self.full_phase_tree_logical_bytes,
            "full_phase_tree_logical_bytes",
        )
        _positive_int(
            self.maximum_process_rss_bytes,
            "maximum_process_rss_bytes",
        )
        object.__setattr__(
            self,
            "minimum_physical_rss_reduction_fraction",
            _fraction(
                self.minimum_physical_rss_reduction_fraction,
                "minimum_physical_rss_reduction_fraction",
            ),
        )
        interval = float(self.rss_sample_interval_seconds)
        if not math.isfinite(interval) or interval <= 0.0 or interval > 1.0:
            raise Lfm2CpcfV5ModelCompilerError(
                "rss_sample_interval_seconds must be within (0, 1]."
            )
        object.__setattr__(self, "rss_sample_interval_seconds", interval)
        if not isinstance(
            self.require_short_conv_v5_endpoint_first_selector,
            bool,
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "require_short_conv_v5_endpoint_first_selector must be boolean."
            )
        if not isinstance(
            self.require_short_conv_v5_authenticated_whole_macrocell_selector,
            bool,
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "require_short_conv_v5_authenticated_whole_macrocell_selector "
                "must be boolean."
            )
        if not isinstance(
            self.require_short_conv_v5_real_numerical_frontier_adapter,
            bool,
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "require_short_conv_v5_real_numerical_frontier_adapter must be boolean."
            )
        cap = self.maximum_physical_rss_increase_cap_bytes
        if cap is not None and (
            isinstance(cap, bool) or not isinstance(cap, int) or cap < 0
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "maximum_physical_rss_increase_cap_bytes must be a "
                "nonnegative integer or None."
            )
        composed = self.composed_qk_attention_policy
        if composed is not None and (
            type(composed) is not Lfm2ComposedQkAttentionFoldPolicy
            or not composed.verify_authentication()
        ):
            raise TypeError(
                "composed_qk_attention_policy must be an exact authenticated "
                "Lfm2ComposedQkAttentionFoldPolicy."
            )
        endpoint = self.endpoint_residual_attention_policy
        if endpoint is not None and (
            type(endpoint) is not Lfm2ComposedEndpointResidualAttentionFoldPolicy
            or not endpoint.verify_authentication()
        ):
            raise TypeError(
                "endpoint_residual_attention_policy must be an exact "
                "authenticated Lfm2ComposedEndpointResidualAttentionFoldPolicy."
            )
        if endpoint is not None and composed is None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Endpoint-residual attention requires the composed Q/K policy."
            )
        terminal = self.terminal_action_residual_policy
        if terminal is not None and (
            type(terminal) is not Lfm2CpcfTerminalActionPolicy
            or not terminal.verify_authentication()
        ):
            raise TypeError(
                "terminal_action_residual_policy must be an exact authenticated "
                "Lfm2CpcfTerminalActionPolicy."
            )
        sequential = self.model_fold_policy.attention_sequential_transport_policy
        combination = self.endpoint_sequential_transport_policy
        post_operator = self.post_operator_transport_policy
        if post_operator is not None and (
            type(post_operator) is not Lfm2CpcfPostOperatorTransportPolicy
            or not post_operator.verify_authentication()
        ):
            raise TypeError(
                "post_operator_transport_policy must be the exact "
                "authenticated Lfm2CpcfPostOperatorTransportPolicy."
            )
        if combination is not None and (
            type(combination) is not Lfm2CpcfEndpointSequentialTransportPolicy
            or not combination.verify_authentication()
        ):
            raise TypeError(
                "endpoint_sequential_transport_policy must be the exact "
                "authenticated Lfm2CpcfEndpointSequentialTransportPolicy."
            )
        combined_components_valid = (
            combination is not None
            and composed is not None
            and endpoint is not None
            and terminal is not None
            and sequential is not None
            and sequential.enabled
            and post_operator is not None
            and combination.authenticates_components(
                composed_qk_attention_policy=composed,
                endpoint_residual_attention_policy=endpoint,
                attention_sequential_transport_policy=sequential,
            )
            and combination.post_operator_transport_policy_sha256
            == post_operator.authentication_sha256
            and combination.terminal_action_residual_policy_sha256
            == terminal.authentication_sha256
        )
        if combination is not None and not combined_components_valid:
            raise Lfm2CpcfV5ModelCompilerError(
                "The endpoint sequential-transport runtime policy requires "
                "exact bound composed, endpoint, enabled sequential, and "
                "terminal-W2 policies."
            )
        if post_operator is not None and combination is None:
            raise Lfm2CpcfV5ModelCompilerError(
                "The production post-operator transport policy is authorized "
                "only by the exact endpoint sequential-terminal profile."
            )
        if (
            (composed is not None or endpoint is not None)
            and sequential is not None
            and sequential.enabled
            and not combined_components_valid
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Composed Q/K attention folding and attention sequential "
                "transport remain mutually exclusive outside the exact "
                "authenticated endpoint sequential-terminal profile."
            )

    @property
    def maximum_physical_rss_increase_bytes(self) -> int:
        retained = 1.0 - self.minimum_physical_rss_reduction_fraction
        logical_limit = int(math.floor(self.full_phase_tree_logical_bytes * retained))
        cap = self.maximum_physical_rss_increase_cap_bytes
        return logical_limit if cap is None else min(logical_limit, cap)

    def resolved_composed_qk_attention_policy(
        self,
        operator_kind: str,
    ) -> Lfm2ComposedQkAttentionFoldPolicy | None:
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfV5ModelCompilerError(
                "Composed Q/K attention folding requires one canonical operator kind."
            )
        if operator_kind == "full_attention":
            return self.composed_qk_attention_policy
        return None

    def resolved_endpoint_residual_attention_policy(
        self,
        operator_kind: str,
    ) -> Lfm2ComposedEndpointResidualAttentionFoldPolicy | None:
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfV5ModelCompilerError(
                "Endpoint-residual attention folding requires one canonical "
                "operator kind."
            )
        if operator_kind == "full_attention":
            return self.endpoint_residual_attention_policy
        return None

    def resolved_endpoint_sequential_transport_policy(
        self,
        operator_kind: str,
    ) -> Lfm2CpcfEndpointSequentialTransportPolicy | None:
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfV5ModelCompilerError(
                "Endpoint sequential transport requires one canonical operator kind."
            )
        if operator_kind == "full_attention":
            return self.endpoint_sequential_transport_policy
        return None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "model_fold_policy": self.model_fold_policy.as_dict(),
            "maximum_resident_evidence_bytes": (self.maximum_resident_evidence_bytes),
            "full_phase_tree_logical_bytes": (self.full_phase_tree_logical_bytes),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "minimum_physical_rss_reduction_fraction": (
                self.minimum_physical_rss_reduction_fraction
            ),
            "maximum_physical_rss_increase_bytes": (
                self.maximum_physical_rss_increase_bytes
            ),
            "maximum_physical_rss_increase_cap_bytes": (
                self.maximum_physical_rss_increase_cap_bytes
            ),
            "rss_sample_interval_seconds": (self.rss_sample_interval_seconds),
            "require_short_conv_v5_endpoint_first_selector": (
                self.require_short_conv_v5_endpoint_first_selector
            ),
            "require_short_conv_v5_authenticated_whole_macrocell_selector": (
                self.require_short_conv_v5_authenticated_whole_macrocell_selector
            ),
            "require_short_conv_v5_real_numerical_frontier_adapter": (
                self.require_short_conv_v5_real_numerical_frontier_adapter
            ),
        }
        if self.composed_qk_attention_policy is not None:
            result["composed_qk_attention_policy"] = (
                self.composed_qk_attention_policy.as_dict()
            )
        if self.endpoint_residual_attention_policy is not None:
            result["endpoint_residual_attention_policy"] = (
                self.endpoint_residual_attention_policy.as_dict()
            )
        if self.terminal_action_residual_policy is not None:
            result["terminal_action_residual_policy"] = (
                self.terminal_action_residual_policy.as_dict()
            )
        if self.endpoint_sequential_transport_policy is not None:
            result["endpoint_sequential_transport_policy"] = (
                self.endpoint_sequential_transport_policy.as_dict()
            )
        if self.post_operator_transport_policy is not None:
            result["post_operator_transport_policy"] = (
                self.post_operator_transport_policy.as_dict()
            )
        return result


def _terminal_resume_prefix_publication_decision(
    *,
    next_layer_index: int,
    expected_layer_count: int,
    terminal_action_enabled: bool,
) -> Mapping[str, Any]:
    """Seal whether a pre-terminal layer prefix is safe to resume from."""

    if (
        isinstance(next_layer_index, bool)
        or not isinstance(next_layer_index, int)
        or isinstance(expected_layer_count, bool)
        or not isinstance(expected_layer_count, int)
        or expected_layer_count <= 0
        or not 0 <= next_layer_index <= expected_layer_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Resume publication decision requires a valid layer boundary."
        )
    if type(terminal_action_enabled) is not bool:
        raise TypeError("terminal_action_enabled must be bool.")
    preintegration_final = bool(
        terminal_action_enabled and next_layer_index == expected_layer_count
    )
    body = {
        "schema": LFM2_CPCF_V5_TERMINAL_RESUME_PUBLICATION_SCHEMA,
        "next_layer_index": next_layer_index,
        "expected_layer_count": expected_layer_count,
        "terminal_action_enabled": terminal_action_enabled,
        "preintegration_final_layer_prefix": preintegration_final,
        "publish_resume_prefix": not preintegration_final,
        "latest_safe_recovery_boundary": (
            "preceding_admitted_prefix_then_reexecute_final_layer_and_terminal"
            if preintegration_final
            else "this_prefix"
        ),
        "postintegration_checkpoint_published": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _sealed_mapping_sha256(
    value: Any,
    *,
    seal_field: str,
    name: str,
) -> str:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be a mapping.")
    body = _plain_metadata(value)
    if not isinstance(body, dict):
        raise Lfm2CpcfV5ModelCompilerError(f"{name} must be canonical metadata.")
    seal = _digest(body.pop(seal_field, None), f"{name}.{seal_field}")
    if _sha(body) != seal:
        raise Lfm2CpcfV5ModelCompilerError(f"{name} has an invalid seal.")
    return seal


def _authenticate_endpoint_attention_candidate_inventory(
    solver: Mapping[str, Any],
    *,
    layer_index: int,
) -> tuple[tuple[str, ...], bool, str]:
    """Authenticate the exact per-layer endpoint tournament inventory."""

    selection = solver.get("attention_candidate_selection")
    selection_sha = _sealed_mapping_sha256(
        selection,
        seal_field="report_sha256",
        name=f"layer[{layer_index}].attention_candidate_selection",
    )
    evaluations = solver.get("attention_candidate_evaluations")
    composed_candidate_eligible = solver.get("composed_qk_candidate_eligible")
    if type(composed_candidate_eligible) is not bool:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Attention layer {layer_index} did not state exact composed "
            "candidate eligibility."
        )
    expected_candidate_ids = (
        LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS
        if composed_candidate_eligible
        else LFM2_CPCF_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS
    )
    candidate_ids = (
        selection.get("candidate_ids") if isinstance(selection, Mapping) else None
    )
    if (
        not isinstance(evaluations, Mapping)
        or not isinstance(candidate_ids, list)
        or candidate_ids != list(expected_candidate_ids)
        or list(evaluations) != candidate_ids
        or selection.get("candidate_count") != len(expected_candidate_ids)
        or selection.get("candidate_local_mlp_preparation_count")
        != len(expected_candidate_ids)
        or selection.get("cross_candidate_prepared_mlp_basis_reused") is not False
        or ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID not in evaluations
        or (
            (
                SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
                in evaluations
            )
            is not composed_candidate_eligible
        )
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            f"Attention layer {layer_index} did not seal the exact "
            f"{len(expected_candidate_ids)}-arm candidate-local MLP tournament "
            "for its authenticated composed-candidate eligibility."
        )
    return expected_candidate_ids, composed_candidate_eligible, selection_sha


def _build_composed_attention_coverage_receipt(
    completed_receipts: Sequence[legacy_model_fold.Lfm2CpcfCompletedLayerReceipt],
    policy: Lfm2ComposedQkAttentionFoldPolicy,
    *,
    allow_endpoint_fallback: bool = False,
) -> Mapping[str, Any]:
    if type(allow_endpoint_fallback) is not bool:
        raise TypeError("allow_endpoint_fallback must be bool.")
    if type(policy) is not Lfm2ComposedQkAttentionFoldPolicy or not (
        policy.verify_authentication()
    ):
        raise TypeError("policy must be an exact authenticated composed Q/K policy.")
    receipts = tuple(completed_receipts)
    expected = LFM2_CPCF_V5_COMPOSED_ATTENTION_LAYER_INDICES
    observed = tuple(
        receipt.layer_index
        for receipt in receipts
        if receipt.operator_kind == "full_attention"
    )
    if observed != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Composed Q/K attention coverage requires the exact ordered "
            f"LFM2 attention layer inventory {list(expected)}, got "
            f"{list(observed)}."
        )
    expected_policy = policy.as_dict()
    rows = []
    covered = []
    for receipt in receipts:
        solver = receipt.report.get("solver_evidence_policy")
        if not isinstance(solver, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Layer {receipt.layer_index} has no solver evidence policy."
            )
        if receipt.operator_kind != "full_attention":
            if any(
                name in solver
                for name in (
                    "composed_qk_attention_policy",
                    "composed_qk_producer_artifact",
                    "composed_qk_producer_artifact_sha256",
                )
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "A short-conv layer received a composed attention policy."
                )
            continue
        if solver.get("composed_qk_attention_policy") != expected_policy:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not bind the exact "
                "runtime composed Q/K policy."
            )
        evaluations = solver.get("attention_candidate_evaluations")
        if not isinstance(evaluations, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} has no candidate inventory."
            )
        candidate_ids: tuple[str, ...] | None = None
        candidate_selection_sha: str | None = None
        if allow_endpoint_fallback:
            (
                candidate_ids,
                composed_candidate_eligible,
                candidate_selection_sha,
            ) = _authenticate_endpoint_attention_candidate_inventory(
                solver,
                layer_index=receipt.layer_index,
            )
        else:
            composed_candidate_eligible = True
            if solver.get("composed_qk_candidate_eligible", True) is not True:
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} rejected the composed "
                    "candidate in a composed-only runtime."
                )
        selection = solver.get("attention_candidate_selection")
        selected_candidate_id = (
            None
            if not isinstance(selection, Mapping)
            else selection.get("selected_candidate_id")
        )
        producer = solver.get("composed_qk_producer_artifact")
        if not composed_candidate_eligible:
            if (
                not allow_endpoint_fallback
                or producer is not None
                or solver.get("composed_qk_producer_artifact_sha256") is not None
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} relabeled a rejected "
                    "composed candidate."
                )
            attempt = solver.get("composed_qk_stage_one_attempt")
            attempt_sha = _sealed_mapping_sha256(
                attempt,
                seal_field="attempt_sha256",
                name=f"layer[{receipt.layer_index}].composed_qk_stage_one_attempt",
            )
            endpoint_start = solver.get("endpoint_qk_start_artifact")
            endpoint_start_sha = _sealed_mapping_sha256(
                endpoint_start,
                seal_field="artifact_sha256",
                name=f"layer[{receipt.layer_index}].endpoint_qk_start_artifact",
            )
            global_sync = solver.get("composed_qk_global_sync")
            global_sync_sha = _sealed_mapping_sha256(
                global_sync,
                seal_field="report_sha256",
                name=f"layer[{receipt.layer_index}].composed_qk_global_sync",
            )
            score_result = solver.get("composed_qk_score_result")
            score_result_sha = _sealed_mapping_sha256(
                score_result,
                seal_field="report_sha256",
                name=f"layer[{receipt.layer_index}].composed_qk_score_result",
            )
            attempt_operation_audit = (
                attempt.get("operation_audit") if isinstance(attempt, Mapping) else None
            )
            if (
                candidate_ids is None
                or not isinstance(attempt, Mapping)
                or not isinstance(endpoint_start, Mapping)
                or not isinstance(score_result, Mapping)
                or not isinstance(attempt_operation_audit, Mapping)
                or attempt.get("schema") != LFM2_COMPOSED_QK_STAGE_ONE_ATTEMPT_SCHEMA
                or attempt.get("composed_candidate_eligible") is not False
                or attempt.get("score_result_accepted") is not False
                or attempt.get("failed_composed_candidate_omitted") is not True
                or attempt.get("endpoint_start_issued_independently") is not True
                or attempt.get("endpoint_start_source")
                != "structural_global_gauge_dense_qkv"
                or attempt.get("composed_candidate_artifact_sha256") is not None
                or attempt.get("endpoint_start_artifact_sha256") != endpoint_start_sha
                or attempt.get("global_gauge_fallback_start_artifact_sha256")
                != endpoint_start_sha
                or attempt.get("score_result_report_sha256") != score_result_sha
                or solver.get("endpoint_qk_start_source")
                != attempt.get("endpoint_start_source")
                or endpoint_start.get("schema") != LFM2_ENDPOINT_QK_START_SCHEMA
                or endpoint_start.get("global_sync_report_sha256") != global_sync_sha
                or endpoint_start.get("score_result_report_sha256") != score_result_sha
                or score_result.get("accepted") is not False
                or any(value != 0 for value in attempt_operation_audit.values())
                or ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID not in evaluations
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} has an invalid "
                    "authenticated composed-rejection endpoint fallback."
                )
            covered.append(receipt.layer_index)
            rows.append(
                {
                    "layer_index": receipt.layer_index,
                    "layer_result_sha256": receipt.result_sha256,
                    "composed_candidate_eligible": False,
                    "candidate_count": len(candidate_ids),
                    "candidate_ids": list(candidate_ids),
                    "candidate_selection_report_sha256": candidate_selection_sha,
                    "composed_qk_stage_one_attempt_sha256": attempt_sha,
                    "endpoint_qk_start_artifact_sha256": endpoint_start_sha,
                    "global_sync_report_sha256": global_sync_sha,
                    "score_result_report_sha256": score_result_sha,
                    "selected_candidate_id": selected_candidate_id,
                }
            )
            continue
        producer_sha = _sealed_mapping_sha256(
            producer,
            seal_field="artifact_sha256",
            name=(f"layer[{receipt.layer_index}].composed_qk_producer_artifact"),
        )
        if solver.get("composed_qk_producer_artifact_sha256") != producer_sha:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} changed its composed "
                "producer artifact reference."
            )
        evaluation = evaluations.get(
            SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
        )
        evaluation_sha = _sealed_mapping_sha256(
            evaluation,
            seal_field="report_sha256",
            name=f"layer[{receipt.layer_index}].composed_candidate_evaluation",
        )
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("candidate_id")
            != SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} has the wrong composed "
                "candidate identity."
            )
        outer = evaluation.get("operator_artifact")
        outer_sha = _sealed_mapping_sha256(
            outer,
            seal_field="artifact_sha256",
            name=f"layer[{receipt.layer_index}].composed_stage_two_artifact",
        )
        actual_child_context_sha = _digest(
            outer.get("fit_value_output_actual_child_context_sha256")
            if isinstance(outer, Mapping)
            else None,
            (
                f"layer[{receipt.layer_index}]."
                "fit_value_output_actual_child_context_sha256"
            ),
        )
        reconstructed_live_context_sha = _digest(
            outer.get("fit_value_output_reconstructed_live_context_sha256")
            if isinstance(outer, Mapping)
            else None,
            (
                f"layer[{receipt.layer_index}]."
                "fit_value_output_reconstructed_live_context_sha256"
            ),
        )
        if (
            not isinstance(outer, Mapping)
            or outer.get("candidate_id")
            != SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            or outer.get("composed_qk_artifact_sha256") != producer_sha
            or evaluation.get("operator_artifact_sha256")
            != _sha(_plain_metadata(outer))
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not cross-bind "
                "the composed producer to the residual-seeded sequence-effective "
                "stage-two artifact."
            )
        covered.append(receipt.layer_index)
        row = {
            "layer_index": receipt.layer_index,
            "layer_result_sha256": receipt.result_sha256,
            "composed_candidate_eligible": True,
            "composed_qk_producer_artifact_sha256": producer_sha,
            "sequence_effective_stage_two_artifact_sha256": outer_sha,
            "actual_child_context_sha256": actual_child_context_sha,
            "reconstructed_live_context_sha256": reconstructed_live_context_sha,
            "candidate_evaluation_report_sha256": evaluation_sha,
            "selected_candidate_id": selected_candidate_id,
        }
        if candidate_ids is not None:
            row.update(
                {
                    "candidate_count": len(candidate_ids),
                    "candidate_ids": list(candidate_ids),
                    "candidate_selection_report_sha256": candidate_selection_sha,
                }
            )
        rows.append(row)
    if tuple(covered) != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Composed Q/K attention did not cover every required attention layer."
        )
    body = {
        "schema": LFM2_CPCF_V5_COMPOSED_ATTENTION_COVERAGE_SCHEMA,
        "policy_authentication_sha256": policy.authentication_sha256,
        "expected_attention_layer_indices": list(expected),
        "observed_attention_layer_indices": list(observed),
        "covered_attention_layer_indices": covered,
        "attention_layer_count": len(expected),
        "ordered_layer_coverage": rows,
        "endpoint_fallback_permitted": allow_endpoint_fallback,
        "composed_candidate_eligible_per_attention_layer": [
            row["composed_candidate_eligible"] for row in rows
        ],
        "composed_candidate_admitted_layer_indices": [
            row["layer_index"] for row in rows if row["composed_candidate_eligible"]
        ],
        "authenticated_endpoint_fallback_layer_indices": [
            row["layer_index"] for row in rows if not row["composed_candidate_eligible"]
        ],
        "exact_ordered_attention_layer_coverage": True,
        "full_attention_only_forwarding": True,
        "actual_upstream_child_contexts_consumed": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _build_endpoint_residual_attention_coverage_receipt(
    completed_receipts: Sequence[legacy_model_fold.Lfm2CpcfCompletedLayerReceipt],
    policy: Lfm2ComposedEndpointResidualAttentionFoldPolicy,
) -> Mapping[str, Any]:
    """Authenticate exact all-layer coverage of the opt-in fifth arm."""

    if type(policy) is not Lfm2ComposedEndpointResidualAttentionFoldPolicy or not (
        policy.verify_authentication()
    ):
        raise TypeError(
            "policy must be an exact authenticated endpoint-residual policy."
        )
    receipts = tuple(completed_receipts)
    expected = LFM2_CPCF_V5_COMPOSED_ATTENTION_LAYER_INDICES
    observed = tuple(
        receipt.layer_index
        for receipt in receipts
        if receipt.operator_kind == "full_attention"
    )
    if observed != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Endpoint-residual attention coverage requires the exact ordered "
            f"LFM2 attention layer inventory {list(expected)}, got "
            f"{list(observed)}."
        )
    expected_policy = policy.as_dict()
    rows = []
    covered = []
    for receipt in receipts:
        solver = receipt.report.get("solver_evidence_policy")
        if not isinstance(solver, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Layer {receipt.layer_index} has no solver evidence policy."
            )
        endpoint_fields = (
            "endpoint_residual_attention_policy",
            "endpoint_residual_attention_artifact",
            "endpoint_residual_attention_artifact_sha256",
            "endpoint_residual_block_reports",
            "endpoint_residual_qk_fit_proposal_reports",
            "endpoint_residual_output_prior_receipt",
            "endpoint_residual_blank_target_values_read",
        )
        if receipt.operator_kind != "full_attention":
            if any(name in solver for name in endpoint_fields):
                raise Lfm2CpcfV5ModelCompilerError(
                    "A short-conv layer received an endpoint-residual "
                    "attention policy or artifact."
                )
            continue
        if solver.get("endpoint_residual_attention_policy") != expected_policy:
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not bind the exact "
                "runtime endpoint-residual policy."
            )
        artifact = solver.get("endpoint_residual_attention_artifact")
        artifact_sha = _sealed_mapping_sha256(
            artifact,
            seal_field="artifact_sha256",
            name=(f"layer[{receipt.layer_index}].endpoint_residual_attention_artifact"),
        )
        if (
            not isinstance(artifact, Mapping)
            or solver.get("endpoint_residual_attention_artifact_sha256") != artifact_sha
            or artifact.get("candidate_id")
            != ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
            or artifact.get("accepted") is not True
            or artifact.get("policy") != expected_policy
            or artifact.get("policy_sha256") != policy.authentication_sha256
            or artifact.get("fit_check_endpoint_and_route_nonregressing") is not True
            or artifact.get("fit_check_first_metric_evaluation_after_all_outer_blocks")
            is not True
            or artifact.get("fit_check_triggered_refit") is not False
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} has an invalid "
                "endpoint-residual artifact."
            )
        operation_audit = artifact.get("operation_audit")
        if not isinstance(operation_audit, Mapping) or any(
            value != 0 for value in operation_audit.values()
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} endpoint artifact "
                "contains a forbidden execution path."
            )
        output_prior = solver.get("endpoint_residual_output_prior_receipt")
        output_prior_sha = _digest(
            output_prior.get("tensor_sha256")
            if isinstance(output_prior, Mapping)
            else None,
            f"layer[{receipt.layer_index}].endpoint_output_prior.tensor_sha256",
        )
        if (
            not isinstance(output_prior, Mapping)
            or output_prior.get("construction")
            != "zero_from_target_shape_and_dtype_metadata_only"
            or output_prior.get("blank_target_values_read") is not False
            or solver.get("endpoint_residual_blank_target_values_read") is not False
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not authenticate "
                "blank-value-independent fifth-arm initialization."
            )
        evaluations = solver.get("attention_candidate_evaluations")
        evaluation = (
            None
            if not isinstance(evaluations, Mapping)
            else evaluations.get(ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID)
        )
        evaluation_sha = _sealed_mapping_sha256(
            evaluation,
            seal_field="report_sha256",
            name=f"layer[{receipt.layer_index}].endpoint_candidate_evaluation",
        )
        outer = evaluation.get("operator_artifact")
        outer_sha = _sealed_mapping_sha256(
            outer,
            seal_field="artifact_sha256",
            name=f"layer[{receipt.layer_index}].endpoint_candidate_artifact",
        )
        if (
            not isinstance(evaluation, Mapping)
            or evaluation.get("candidate_id")
            != ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
            or evaluation.get("operator_artifact_sha256")
            != _sha(_plain_metadata(outer))
            or outer_sha != artifact_sha
            or _plain_metadata(outer) != _plain_metadata(artifact)
            or evaluation.get("mlp_compiled_from_splits") != ["fit"]
            or evaluation.get("validation_phase_count_consumed") != 0
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not evaluate the "
                "exact endpoint artifact with its own FIT-only downstream MLP."
            )
        (
            candidate_ids,
            composed_candidate_eligible,
            candidate_selection_sha,
        ) = _authenticate_endpoint_attention_candidate_inventory(
            solver,
            layer_index=receipt.layer_index,
        )
        selection = solver["attention_candidate_selection"]
        block_reports = solver.get("endpoint_residual_block_reports")
        proposal_reports = solver.get("endpoint_residual_qk_fit_proposal_reports")
        outer_block_count = artifact.get("outer_block_count")
        if (
            not isinstance(block_reports, Sequence)
            or isinstance(block_reports, (str, bytes, bytearray))
            or not isinstance(proposal_reports, Sequence)
            or isinstance(proposal_reports, (str, bytes, bytearray))
            or isinstance(outer_block_count, bool)
            or not isinstance(outer_block_count, int)
            or outer_block_count <= 0
            or len(block_reports) != outer_block_count
            or len(proposal_reports) != outer_block_count
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} has incomplete "
                "endpoint outer-block evidence."
            )
        block_sha256s = []
        proposal_sha256s = []
        execution_receipt_sha256s = []
        observed_compute_devices = []
        for block_index, (block, proposal) in enumerate(
            zip(block_reports, proposal_reports, strict=True)
        ):
            block_sha = _sealed_mapping_sha256(
                block,
                seal_field="block_sha256",
                name=(f"layer[{receipt.layer_index}].endpoint_block[{block_index}]"),
            )
            proposal_sha = _sealed_mapping_sha256(
                proposal,
                seal_field="report_sha256",
                name=(
                    f"layer[{receipt.layer_index}].endpoint_qk_proposal[{block_index}]"
                ),
            )
            execution = (
                proposal.get("observed_execution_receipt")
                if isinstance(proposal, Mapping)
                else None
            )
            execution_sha = _sealed_mapping_sha256(
                execution,
                seal_field="receipt_sha256",
                name=(
                    f"layer[{receipt.layer_index}]."
                    f"endpoint_execution_receipt[{block_index}]"
                ),
            )
            requested_devices = list(policy.endpoint_policy.compute_devices)
            concurrent_static_cuda_shards = (
                len(requested_devices) > 1
                and len(set(requested_devices)) == len(requested_devices)
                and all(device.startswith("cuda:") for device in requested_devices)
            )
            expected_concurrency = (
                "thread_barrier_concurrent_static_cuda_shards"
                if concurrent_static_cuda_shards
                else "serial_static_device_shards"
            )
            expected_transfer_execution = (
                "one_local_accumulation_per_concurrent_static_cuda_shard_then_"
                "canonical_requested_device_reduction"
                if concurrent_static_cuda_shards
                else "once_per_static_device_shard_serial_requested_device_order"
            )
            if (
                not isinstance(block, Mapping)
                or not isinstance(proposal, Mapping)
                or not isinstance(execution, Mapping)
                or block.get("block_index") != block_index
                or block.get("qk_fit_proposal_sha256") != proposal_sha
                or block.get("fit_check_accessible") is not False
                or block.get("fit_check_cases_consumed") != 0
                or proposal.get("policy") != policy.endpoint_policy.as_dict()
                or proposal.get("fit_check_accessible") is not False
                or proposal.get("fit_check_cases_consumed") != 0
                or execution.get("requested_solver_device")
                != policy.endpoint_policy.solver_device
                or execution.get("requested_compute_devices") != requested_devices
                or execution.get("requested_compute_device_count")
                != len(requested_devices)
                or execution.get("every_requested_compute_device_observed") is not True
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} endpoint block "
                    f"{block_index} changed its FIT-only policy or execution "
                    "receipt."
                )
            observations = execution.get("compute_observations")
            summaries = execution.get("compute_device_summaries")
            resource_observation = execution.get("resource_observation")
            solver_observation = execution.get("solver_observation")
            numerical_sweep_count = execution.get("numerical_sweep_count")
            numerical_normal_calls = execution.get("numerical_normal_operator_calls")
            if (
                not isinstance(observations, Sequence)
                or isinstance(observations, (str, bytes, bytearray))
                or len(observations) < len(requested_devices)
                or not isinstance(summaries, Sequence)
                or isinstance(summaries, (str, bytes, bytearray))
                or len(summaries) != len(requested_devices)
                or not isinstance(resource_observation, Mapping)
                or resource_observation.get("fit_case_count") != len(observations)
                or resource_observation.get("within_all_resource_ceilings") is not True
                or resource_observation.get("every_compute_device_within_byte_ceiling")
                is not True
                or execution.get("case_assignment")
                != "authenticated_bank_order_round_robin"
                or execution.get("compute_device_concurrency") != expected_concurrency
                or execution.get("qk_transfer_execution") != expected_transfer_execution
                or type(numerical_sweep_count) is not int
                or numerical_sweep_count < 1
                or type(numerical_normal_calls) is not int
                or numerical_normal_calls < 1
                or execution.get("total_semantic_probe_calls") != len(observations)
                or not isinstance(execution.get("operation_audit"), Mapping)
                or any(
                    type(value) is not int or value != 0
                    for value in execution["operation_audit"].values()
                )
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} endpoint block "
                    f"{block_index} lacks authenticated round-robin semantic work."
                )
            expected_observed_devices = [
                requested_devices[index % len(requested_devices)]
                for index in range(len(observations))
            ]
            observed_devices = []
            per_device_assignment_indices: dict[str, list[int]] = {
                device: [] for device in requested_devices
            }
            per_device_semantic_probe_calls: dict[str, int] = {
                device: 0 for device in requested_devices
            }
            per_device_physical_identity: dict[str, tuple[Any, Any]] = {}
            for assignment_index, observation in enumerate(observations):
                observation_sha = _sealed_mapping_sha256(
                    observation,
                    seal_field="case_execution_receipt_sha256",
                    name=(
                        f"layer[{receipt.layer_index}].endpoint_block"
                        f"[{block_index}].compute_observation"
                        f"[{assignment_index}]"
                    ),
                )
                expected_device = expected_observed_devices[assignment_index]
                scalar = (
                    observation.get("analytical_scalar")
                    if isinstance(observation, Mapping)
                    else None
                )
                if (
                    not observation_sha
                    or not isinstance(observation, Mapping)
                    or observation.get("assignment_index") != assignment_index
                    or observation.get("intended_logical_device") != expected_device
                    or observation.get("observed_logical_device") != expected_device
                    or observation.get("analytical_operation")
                    != "exact_endpoint_jvp_then_vjp_normal_probe"
                    or observation.get("semantic_probe_calls") != 1
                    or observation.get("forced_host_scalar_read") is not True
                    or isinstance(scalar, bool)
                    or not isinstance(scalar, (int, float))
                    or not math.isfinite(float(scalar))
                    or float(scalar) <= 0.0
                    or any(
                        not isinstance(observation.get(name), str)
                        or re.fullmatch(
                            r"[0-9a-f]{64}",
                            str(observation.get(name)),
                        )
                        is None
                        for name in (
                            "semantic_delta_probability_sha256",
                            "semantic_delta_endpoint_sha256",
                            "semantic_query_gradient_sha256",
                            "semantic_key_gradient_sha256",
                        )
                    )
                ):
                    raise Lfm2CpcfV5ModelCompilerError(
                        f"Attention layer {receipt.layer_index} endpoint block "
                        f"{block_index} has an unauthenticated semantic case "
                        f"observation at assignment {assignment_index}."
                    )
                observed_devices.append(expected_device)
                per_device_assignment_indices[expected_device].append(assignment_index)
                per_device_semantic_probe_calls[expected_device] += int(
                    observation["semantic_probe_calls"]
                )
                identity = (
                    observation.get("cuda_uuid"),
                    observation.get("cuda_name"),
                )
                prior_identity = per_device_physical_identity.setdefault(
                    expected_device,
                    identity,
                )
                if identity != prior_identity:
                    raise Lfm2CpcfV5ModelCompilerError(
                        f"Attention layer {receipt.layer_index} endpoint block "
                        f"{block_index} changed the physical identity of "
                        f"{expected_device}."
                    )
            if observed_devices != expected_observed_devices:
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} endpoint block "
                    f"{block_index} changed round-robin device assignment."
                )
            summary_devices = []
            for summary_index, summary in enumerate(summaries):
                expected_device = requested_devices[summary_index]
                if (
                    not isinstance(summary, Mapping)
                    or summary.get("logical_device") != expected_device
                    or summary.get("case_assignment_indices")
                    != per_device_assignment_indices[expected_device]
                    or summary.get("case_count")
                    != len(per_device_assignment_indices[expected_device])
                    or summary.get("semantic_probe_calls")
                    != per_device_semantic_probe_calls[expected_device]
                    or summary.get("cuda_uuid")
                    != per_device_physical_identity[expected_device][0]
                    or summary.get("cuda_name")
                    != per_device_physical_identity[expected_device][1]
                    or summary.get("within_compute_device_byte_ceiling") is not True
                    or summary.get("qk_transfer_execution")
                    != expected_transfer_execution
                    or summary.get("concurrent_rhs_shard_launches")
                    != (numerical_sweep_count if concurrent_static_cuda_shards else 0)
                    or summary.get("concurrent_normal_shard_launches")
                    != (numerical_normal_calls if concurrent_static_cuda_shards else 0)
                ):
                    raise Lfm2CpcfV5ModelCompilerError(
                        f"Attention layer {receipt.layer_index} endpoint block "
                        f"{block_index} lacks the exact summary for "
                        f"{expected_device}."
                    )
                summary_devices.append(expected_device)
            if summary_devices != requested_devices:
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} endpoint block "
                    f"{block_index} did not summarize every requested device "
                    "exactly once."
                )
            solver_sha = _sealed_mapping_sha256(
                solver_observation,
                seal_field="solver_execution_receipt_sha256",
                name=(
                    f"layer[{receipt.layer_index}].endpoint_block"
                    f"[{block_index}].solver_observation"
                ),
            )
            solver_scalar = (
                solver_observation.get("analytical_scalar")
                if isinstance(solver_observation, Mapping)
                else None
            )
            if (
                not solver_sha
                or not isinstance(solver_observation, Mapping)
                or solver_observation.get("intended_logical_device")
                != policy.endpoint_policy.solver_device
                or solver_observation.get("observed_logical_device")
                != policy.endpoint_policy.solver_device
                or solver_observation.get("forced_host_scalar_read") is not True
                or isinstance(solver_scalar, bool)
                or not isinstance(solver_scalar, (int, float))
                or not math.isfinite(float(solver_scalar))
                or float(solver_scalar) <= 0.0
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Attention layer {receipt.layer_index} endpoint block "
                    f"{block_index} lacks observed solver work."
                )
            block_sha256s.append(block_sha)
            proposal_sha256s.append(proposal_sha)
            execution_receipt_sha256s.append(execution_sha)
            observed_compute_devices.append(observed_devices)
        if block_sha256s != artifact.get("block_sha256s"):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} changed its ordered "
                "endpoint block lineage."
            )
        covered.append(receipt.layer_index)
        rows.append(
            {
                "layer_index": receipt.layer_index,
                "layer_result_sha256": receipt.result_sha256,
                "endpoint_artifact_sha256": artifact_sha,
                "candidate_evaluation_report_sha256": evaluation_sha,
                "candidate_selection_report_sha256": candidate_selection_sha,
                "candidate_count": len(candidate_ids),
                "candidate_ids": list(candidate_ids),
                "composed_candidate_eligible": composed_candidate_eligible,
                "outer_block_count": outer_block_count,
                "ordered_block_sha256s": block_sha256s,
                "ordered_qk_fit_proposal_sha256s": proposal_sha256s,
                "ordered_execution_receipt_sha256s": (execution_receipt_sha256s),
                "ordered_observed_compute_devices": observed_compute_devices,
                "source_inventory_sha256": artifact.get("source_inventory_sha256"),
                "blank_independent_output_prior_sha256": output_prior_sha,
                "evaluation_fit_evidence_sha256": artifact.get(
                    "evaluation_fit_evidence_sha256"
                ),
                "evaluation_fit_check_evidence_sha256": artifact.get(
                    "evaluation_fit_check_evidence_sha256"
                ),
                "selected_candidate_id": selection.get("selected_candidate_id"),
            }
        )
    if tuple(covered) != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Endpoint-residual attention did not cover every required layer."
        )
    body = {
        "schema": LFM2_CPCF_V5_ENDPOINT_RESIDUAL_ATTENTION_COVERAGE_SCHEMA,
        "policy_authentication_sha256": policy.authentication_sha256,
        "expected_attention_layer_indices": list(expected),
        "observed_attention_layer_indices": list(observed),
        "covered_attention_layer_indices": covered,
        "attention_layer_count": len(expected),
        "ordered_layer_coverage": rows,
        "exact_ordered_attention_layer_coverage": True,
        "full_attention_only_forwarding": True,
        "candidate_count_per_attention_layer": [row["candidate_count"] for row in rows],
        "candidate_ids_per_attention_layer": [row["candidate_ids"] for row in rows],
        "composed_candidate_eligible_per_attention_layer": [
            row["composed_candidate_eligible"] for row in rows
        ],
        "candidate_local_fit_only_downstream_mlp_evaluation": True,
        "fit_check_final_one_shot_nonregression_only": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "blank_target_values_read": False,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _build_endpoint_sequential_transport_coverage_receipt(
    completed_receipts: Sequence[legacy_model_fold.Lfm2CpcfCompletedLayerReceipt],
    policy: Lfm2CpcfEndpointSequentialTransportPolicy,
) -> Mapping[str, Any]:
    """Authenticate both gauge transports on every exact attention layer."""

    if (
        type(policy) is not Lfm2CpcfEndpointSequentialTransportPolicy
        or not policy.verify_authentication()
    ):
        raise TypeError(
            "policy must be the exact authenticated endpoint sequential-"
            "transport policy."
        )
    receipts = tuple(completed_receipts)
    expected = LFM2_CPCF_V5_COMPOSED_ATTENTION_LAYER_INDICES
    observed = tuple(
        receipt.layer_index
        for receipt in receipts
        if receipt.operator_kind == "full_attention"
    )
    if observed != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Endpoint sequential transport requires exact ordered attention "
            f"coverage {list(expected)}, got {list(observed)}."
        )
    rows = []
    expected_policy = policy.as_dict()
    for receipt in receipts:
        solver = receipt.report.get("solver_evidence_policy")
        if not isinstance(solver, Mapping):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Layer {receipt.layer_index} has no solver evidence policy."
            )
        combined_fields = (
            "endpoint_sequential_transport_policy",
            "fit_only_post_operator_transport_controller",
            "attention_candidate_local_post_layer_transport",
        )
        if receipt.operator_kind != "full_attention":
            if any(name in solver for name in combined_fields):
                raise Lfm2CpcfV5ModelCompilerError(
                    "A short-conv layer received a combined attention "
                    "transport policy or report."
                )
            continue
        if (
            solver.get("endpoint_sequential_transport_policy") != expected_policy
            or solver.get("endpoint_sequential_transport_policy_sha256")
            != policy.authentication_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not bind the "
                "exact combined transport policy."
            )
        upstream = solver.get("fit_only_post_operator_transport_controller")
        upstream_sha = _sealed_mapping_sha256(
            upstream,
            seal_field="report_sha256",
            name=f"layer[{receipt.layer_index}].post_operator_transport",
        )
        if (
            not isinstance(upstream, Mapping)
            or solver.get("fit_only_post_operator_transport_controller_sha256")
            != upstream_sha
            or upstream.get("layer_index") != receipt.layer_index
            or upstream.get("component") != "fit_only_post_operator_chart_transport"
            or upstream.get("production_profile_authorized") is not True
            or upstream.get("post_operator_transport_policy_sha256")
            != policy.post_operator_transport_policy_sha256
            or upstream.get("transported_layer_charts_sha256")
            != receipt.report.get("boundary_charts_sha256")
            or upstream.get("validation_rows_read") != 0
            or upstream.get("selection_used_by_solver") is not False
            or upstream.get("fit_only_closed_form_orthogonal_procrustes") is not True
            or upstream.get("source_bank_unchanged") is not True
            or upstream.get("source_charts_unchanged") is not True
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} has an invalid "
                "FIT-only post-operator transport receipt."
            )
        selection = solver.get("attention_candidate_selection")
        selection_sha = _sealed_mapping_sha256(
            selection,
            seal_field="report_sha256",
            name=f"layer[{receipt.layer_index}].attention_candidate_selection",
        )
        evaluations = solver.get("attention_candidate_evaluations")
        final_binding = solver.get("attention_candidate_local_post_layer_transport")
        if (
            not isinstance(selection, Mapping)
            or selection.get("candidate_count") not in {4, 5}
            or not isinstance(selection.get("candidate_ids"), list)
            or selection.get("candidate_local_post_layer_transport_enabled") is not True
            or selection.get("operator_transport_mlp_bundle_frozen_together")
            is not True
            or selection.get("selection_refits_or_mutates_transport") is not False
            or not isinstance(evaluations, Mapping)
            or not evaluations
            or len(evaluations) != selection.get("candidate_count")
            or set(evaluations) != set(selection.get("candidate_ids", ()))
            or any(
                not isinstance(evaluation, Mapping)
                or evaluation.get("candidate_id") != candidate_id
                or evaluation.get(
                    "candidate_local_post_layer_transport_frozen_before_mlp"
                )
                is not True
                or evaluation.get("operator_transport_and_mlp_ranked_as_one_bundle")
                is not True
                for candidate_id, evaluation in evaluations.items()
            )
            or not isinstance(final_binding, Mapping)
            or final_binding.get("selected_candidate_id")
            != selection.get("selected_candidate_id")
            or final_binding.get("source_boundary_charts_sha256")
            != receipt.report.get("boundary_charts_sha256")
            or final_binding.get("effective_boundary_charts_sha256")
            != selection.get("selected_effective_boundary_charts_sha256")
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"Attention layer {receipt.layer_index} did not freeze one "
                "candidate-local downstream transport and own closed-form MLP "
                "for every exact QKVO candidate before selection."
            )
        rows.append(
            {
                "layer_index": receipt.layer_index,
                "layer_result_sha256": receipt.result_sha256,
                "post_operator_transport_report_sha256": upstream_sha,
                "candidate_selection_report_sha256": selection_sha,
                "candidate_count": selection.get("candidate_count"),
                "selected_candidate_id": selection.get("selected_candidate_id"),
                "selected_effective_boundary_charts_sha256": selection.get(
                    "selected_effective_boundary_charts_sha256"
                ),
                "fit_required_write_rms_reduction_fraction": upstream.get(
                    "fit_required_write_rms_reduction_fraction"
                ),
                "selection_required_write_rms_reduction_fraction": upstream.get(
                    "selection_required_write_rms_reduction_fraction"
                ),
            }
        )
    if tuple(row["layer_index"] for row in rows) != expected:
        raise Lfm2CpcfV5ModelCompilerError(
            "Two-sided sequential transport did not cover every attention layer."
        )
    body = {
        "schema": LFM2_CPCF_V5_ENDPOINT_SEQUENTIAL_TRANSPORT_COVERAGE_SCHEMA,
        "policy_authentication_sha256": policy.authentication_sha256,
        "expected_attention_layer_indices": list(expected),
        "observed_attention_layer_indices": list(observed),
        "covered_attention_layer_indices": [row["layer_index"] for row in rows],
        "ordered_layer_coverage": rows,
        "fit_only_post_operator_transport_per_attention_layer": True,
        "candidate_local_post_layer_transport_per_qkvo_candidate": True,
        "candidate_specific_closed_form_mlp_per_qkvo_candidate": True,
        "winner_bundle_frozen_before_validation": True,
        "actual_child_post_layer_state_propagated": True,
        "short_conv_transport_enabled": False,
        "validation_rows_read_by_transport_solvers": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _windows_process_memory() -> tuple[int, int]:
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    )
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(),
        ctypes.byref(counters),
        counters.cb,
    ):
        raise Lfm2CpcfV5ModelCompilerError("GetProcessMemoryInfo failed.")
    return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)


def _process_rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except ImportError:
        pass
    if os.name == "posix" and os.path.exists("/proc/self/statm"):
        with open("/proc/self/statm", encoding="ascii") as handle:
            fields = handle.read().split()
        if len(fields) < 2:
            raise Lfm2CpcfV5ModelCompilerError("/proc/self/statm lacks an RSS field.")
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    if os.name == "nt":
        return _windows_process_memory()[0]
    raise Lfm2CpcfV5ModelCompilerError(
        "Current process RSS is unavailable; v5 fails closed."
    )


def _process_peak_rss_bytes() -> int:
    if os.name == "nt":
        return _windows_process_memory()[1]
    if os.name == "posix" and os.path.exists("/proc/self/status"):
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    fields = line.split()
                    if len(fields) >= 2:
                        return int(fields[1]) * 1024
    if os.name == "posix":
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if sys.platform == "darwin" else peak * 1024
    raise Lfm2CpcfV5ModelCompilerError(
        "Peak process RSS is unavailable; v5 fails closed."
    )


class _PhysicalRssGuard:
    """Continuously retain physical RSS peaks and enforce a hard delta cap."""

    def __init__(
        self,
        *,
        maximum_increase_bytes: int,
        maximum_process_rss_bytes: int,
        full_phase_tree_logical_bytes: int,
        minimum_reduction_fraction: float,
        sample_interval_seconds: float,
    ) -> None:
        if (
            isinstance(maximum_increase_bytes, bool)
            or not isinstance(maximum_increase_bytes, int)
            or maximum_increase_bytes < 0
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "maximum_increase_bytes must be a nonnegative integer."
            )
        self.relative_maximum_increase_bytes = maximum_increase_bytes
        self.maximum_process_rss_bytes = _positive_int(
            maximum_process_rss_bytes,
            "maximum_process_rss_bytes",
        )
        self.full_phase_tree_logical_bytes = _positive_int(
            full_phase_tree_logical_bytes,
            "full_phase_tree_logical_bytes",
        )
        self.minimum_reduction_fraction = _fraction(
            minimum_reduction_fraction,
            "minimum_reduction_fraction",
        )
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.start_rss_bytes = _process_rss_bytes()
        raw_start_lifetime_peak_bytes = _process_peak_rss_bytes()
        if (
            self.start_rss_bytes < 0
            or raw_start_lifetime_peak_bytes < 0
            or self.start_rss_bytes > self.maximum_process_rss_bytes
            or raw_start_lifetime_peak_bytes > self.maximum_process_rss_bytes
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS probes returned an invalid or over-cap "
                "current/HWM baseline."
            )
        # Current RSS and lifetime HWM are independent process snapshots.  On
        # Linux their accounting can momentarily disagree under allocation
        # pressure even though a true lifetime HWM cannot be below current
        # RSS.  Promoting HWM to current is conservative for both limits.
        self.start_lifetime_peak_bytes = max(
            self.start_rss_bytes,
            raw_start_lifetime_peak_bytes,
        )
        self.absolute_headroom_bytes = (
            self.maximum_process_rss_bytes - self.start_rss_bytes
        )
        self.maximum_increase_bytes = min(
            self.relative_maximum_increase_bytes,
            self.absolute_headroom_bytes,
        )
        self.current_rss_bytes = self.start_rss_bytes
        self.peak_rss_bytes = self.start_rss_bytes
        self.current_lifetime_peak_bytes = self.start_lifetime_peak_bytes
        self.peak_current_rss_increase_bytes = 0
        self.lifetime_peak_increase_bytes = 0
        self.checkpoints: list[dict[str, Any]] = []
        self._last_checkpoint_report_sha256: str | None = None
        self.rss_guard_id_sha256 = _sha(
            {
                "schema": LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA,
                "process_id": os.getpid(),
                "guard_nonce_sha256": hashlib.sha256(
                    secrets.token_bytes(32)
                ).hexdigest(),
                "start_rss_bytes": self.start_rss_bytes,
                "start_lifetime_peak_bytes": (self.start_lifetime_peak_bytes),
                "full_phase_tree_logical_bytes": (self.full_phase_tree_logical_bytes),
                "minimum_reduction_fraction": (self.minimum_reduction_fraction),
                "relative_maximum_increase_bytes": (
                    self.relative_maximum_increase_bytes
                ),
                "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
                "absolute_headroom_bytes": self.absolute_headroom_bytes,
                "maximum_increase_bytes": self.maximum_increase_bytes,
                "sample_interval_seconds": self.sample_interval_seconds,
            }
        )
        self._guard_authority = object()
        self._outstanding_checkpoint_authorization: dict[str, Any] | None = None
        self._receipt_construction_authorization: dict[str, Any] | None = None
        self._authorized_next_layer_index: int | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._violation: str | None = None
        self._stopped = False
        self.last_report: Mapping[str, Any] | None = None
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="lfm2-cpcf-v5-rss-sampler",
            daemon=True,
        )
        self._thread.start()
        self.checkpoint("rss_guard_start")

    @property
    def peak_increase_bytes(self) -> int:
        return max(
            self.peak_current_rss_increase_bytes,
            self.lifetime_peak_increase_bytes,
        )

    @property
    def current_increase_bytes(self) -> int:
        return max(0, self.current_rss_bytes - self.start_rss_bytes)

    def _sample(self) -> None:
        current = _process_rss_bytes()
        raw_lifetime_peak = _process_peak_rss_bytes()
        if current < 0 or raw_lifetime_peak < 0:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS probes returned an invalid current/HWM sample."
            )
        # Reconcile only the impossible ordering caused by non-atomic probes.
        # This never lowers either measurement, so hard delta and absolute RSS
        # enforcement remain fail-closed.
        lifetime_peak = max(current, raw_lifetime_peak)
        with self._lock:
            self.current_rss_bytes = current
            self.peak_rss_bytes = max(self.peak_rss_bytes, current)
            self.current_lifetime_peak_bytes = max(
                self.current_lifetime_peak_bytes,
                lifetime_peak,
            )
            self.peak_current_rss_increase_bytes = max(
                self.peak_current_rss_increase_bytes,
                max(0, current - self.start_rss_bytes),
            )
            self.lifetime_peak_increase_bytes = max(
                0,
                self.current_lifetime_peak_bytes - self.start_lifetime_peak_bytes,
            )
            increase = self.peak_increase_bytes
            if (
                increase > self.maximum_increase_bytes
                or current > self.maximum_process_rss_bytes
                or lifetime_peak > self.maximum_process_rss_bytes
            ) and self._violation is None:
                self._violation = (
                    f"physical RSS increase {increase} or absolute RSS/HWM "
                    f"{max(current, lifetime_peak)} exceeded delta "
                    f"{self.maximum_increase_bytes} or process cap "
                    f"{self.maximum_process_rss_bytes}"
                )

    def _sample_loop(self) -> None:
        try:
            while not self._stop.wait(self.sample_interval_seconds):
                self._sample()
        except BaseException as exc:  # pragma: no cover - platform failure
            with self._lock:
                self._error = exc

    def _checkpoint_receipt(
        self,
        label: str,
        *,
        passed: bool,
    ) -> Lfm2CpcfV5PhysicalRssCheckpointReceipt:
        outstanding = self._outstanding_checkpoint_authorization
        sequence_capability = (
            object() if outstanding is None else outstanding["sequence_capability"]
        )
        body = {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA,
            "label": str(label),
            "start_rss_bytes": self.start_rss_bytes,
            "current_rss_bytes": self.current_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "start_lifetime_peak_bytes": (self.start_lifetime_peak_bytes),
            "current_lifetime_peak_bytes": (self.current_lifetime_peak_bytes),
            "current_increase_bytes": self.current_increase_bytes,
            "peak_current_rss_increase_bytes": (self.peak_current_rss_increase_bytes),
            "lifetime_peak_increase_bytes": (self.lifetime_peak_increase_bytes),
            "peak_increase_bytes": self.peak_increase_bytes,
            "full_phase_tree_logical_bytes": (self.full_phase_tree_logical_bytes),
            "minimum_reduction_fraction": (self.minimum_reduction_fraction),
            "relative_maximum_increase_bytes": (self.relative_maximum_increase_bytes),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "absolute_headroom_bytes": self.absolute_headroom_bytes,
            "maximum_increase_bytes": self.maximum_increase_bytes,
            "rss_guard_id_sha256": self.rss_guard_id_sha256,
            "checkpoint_ordinal": len(self.checkpoints) - 1,
            "previous_checkpoint_report_sha256": (self._last_checkpoint_report_sha256),
            "physical_rss_gate_passed": passed,
        }
        sealed = {
            **body,
            "report_sha256": canonical_sha256(body),
        }
        self._receipt_construction_authorization = {
            "body": sealed,
            "sequence_capability": sequence_capability,
        }
        try:
            receipt = Lfm2CpcfV5PhysicalRssCheckpointReceipt(
                sealed,
                _guard=self,
                _sequence_capability=sequence_capability,
            )
        finally:
            self._receipt_construction_authorization = None
        if (
            outstanding is not None
            and outstanding["sequence_capability"] is sequence_capability
        ):
            self._outstanding_checkpoint_authorization = None
        return receipt

    def _authorize_rss_expectation_construction(
        self,
        expectation: Mapping[str, Any],
        *,
        sequence_capability: object,
    ) -> object:
        authorization = self._outstanding_checkpoint_authorization
        if (
            authorization is None
            or authorization["sequence_capability"] is not sequence_capability
            or dict(expectation) != authorization["expectation"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "RSS expectation construction lacks this live guard's exact "
                "next-checkpoint authority."
            )
        return self._guard_authority

    def _authorize_rss_receipt_construction(
        self,
        receipt: Mapping[str, Any],
        *,
        sequence_capability: object,
    ) -> object:
        authorization = self._receipt_construction_authorization
        if (
            authorization is None
            or authorization["sequence_capability"] is not sequence_capability
            or dict(receipt) != authorization["body"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "RSS receipt construction lacks this live guard's exact "
                "sampled checkpoint authority."
            )
        return self._guard_authority

    def _raise_failure(self, label: str) -> None:
        with self._lock:
            error = self._error
            violation = self._violation
        if error is not None:
            report = self._checkpoint_receipt(label, passed=False)
            self.last_report = report
            raise Lfm2CpcfV5PhysicalRssError(
                f"Physical RSS sampler failed at {label!r}.",
                report,
            ) from error
        if violation is not None:
            report = self._checkpoint_receipt(label, passed=False)
            self.last_report = report
            raise Lfm2CpcfV5PhysicalRssError(
                f"{violation} at checkpoint {label!r}.",
                report,
            )

    def checkpoint(self, label: str) -> Mapping[str, Any]:
        if self._stopped:
            raise Lfm2CpcfV5ModelCompilerError(
                "Cannot checkpoint a stopped physical RSS guard."
            )
        outstanding = self._outstanding_checkpoint_authorization
        if outstanding is not None and outstanding["label"] != str(label):
            raise Lfm2CpcfV5ModelCompilerError(
                "RSS checkpoint label differs from its outstanding one-shot "
                "authorization."
            )
        self._sample()
        record = {
            "label": str(label),
            "current_rss_bytes": self.current_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "current_increase_bytes": self.current_increase_bytes,
            "peak_increase_bytes": self.peak_increase_bytes,
        }
        self.checkpoints.append(record)
        self._raise_failure(label)
        receipt = self._checkpoint_receipt(label, passed=True)
        self._last_checkpoint_report_sha256 = receipt["report_sha256"]
        self.last_report = receipt
        return receipt

    def next_checkpoint_expectation(
        self,
        label: str,
    ) -> Lfm2CpcfV5PhysicalRssCheckpointExpectation:
        """Seal the exact next RSS receipt lineage before persistence."""

        if self._stopped:
            raise Lfm2CpcfV5ModelCompilerError(
                "Cannot authorize a checkpoint on a stopped RSS guard."
            )
        if self._outstanding_checkpoint_authorization is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS guard already has one outstanding checkpoint "
                "authorization."
            )
        label_value = str(label)
        if label_value == "prefix_000000_persisted_before_residency":
            next_layer_index = 0
        else:
            match = re.fullmatch(
                r"layer_([0-9]{3,})_compiled_checkpoint_"
                r"persisted_before_release",
                label_value,
            )
            if match is None:
                raise Lfm2CpcfV5ModelCompilerError(
                    "Resumable RSS expectation label is not a canonical "
                    "prefix boundary."
                )
            next_layer_index = int(match.group(1)) + 1
        body = {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA,
            "label": label_value,
            "rss_guard_id_sha256": self.rss_guard_id_sha256,
            "checkpoint_ordinal": len(self.checkpoints),
            "previous_checkpoint_report_sha256": (self._last_checkpoint_report_sha256),
            "start_rss_bytes": self.start_rss_bytes,
            "start_lifetime_peak_bytes": (self.start_lifetime_peak_bytes),
            "full_phase_tree_logical_bytes": (self.full_phase_tree_logical_bytes),
            "minimum_reduction_fraction": (self.minimum_reduction_fraction),
            "relative_maximum_increase_bytes": (self.relative_maximum_increase_bytes),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "absolute_headroom_bytes": self.absolute_headroom_bytes,
            "maximum_increase_bytes": self.maximum_increase_bytes,
        }
        expectation = {
            **body,
            "expectation_sha256": canonical_sha256(body),
        }
        sequence_capability = object()
        self._outstanding_checkpoint_authorization = {
            "label": label_value,
            "next_layer_index": next_layer_index,
            "expectation": expectation,
            "sequence_capability": sequence_capability,
        }
        self._authorized_next_layer_index = next_layer_index
        try:
            return Lfm2CpcfV5PhysicalRssCheckpointExpectation(
                expectation,
                _guard=self,
                _sequence_capability=sequence_capability,
            )
        except BaseException:
            self._outstanding_checkpoint_authorization = None
            raise
        finally:
            self._authorized_next_layer_index = None

    def close(self) -> Mapping[str, Any]:
        if self._stopped:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS guard can be closed exactly once."
            )
        if self._outstanding_checkpoint_authorization is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Physical RSS guard cannot close with an outstanding "
                "checkpoint authorization."
            )
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 4))
        if self._thread.is_alive():
            raise Lfm2CpcfV5ModelCompilerError("Physical RSS sampler did not stop.")
        self._sample()
        self._stopped = True
        self._raise_failure("rss_guard_close")
        passed = (
            self.peak_increase_bytes <= self.maximum_increase_bytes
            and self.current_increase_bytes <= self.maximum_increase_bytes
            and self.current_rss_bytes <= self.maximum_process_rss_bytes
            and self.current_lifetime_peak_bytes <= self.maximum_process_rss_bytes
        )
        body = {
            "schema": LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
            "start_rss_bytes": self.start_rss_bytes,
            "current_rss_bytes": self.current_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "start_lifetime_peak_bytes": self.start_lifetime_peak_bytes,
            "current_lifetime_peak_bytes": (self.current_lifetime_peak_bytes),
            "current_increase_bytes": self.current_increase_bytes,
            "peak_current_rss_increase_bytes": (self.peak_current_rss_increase_bytes),
            "lifetime_peak_increase_bytes": (self.lifetime_peak_increase_bytes),
            "peak_increase_bytes": self.peak_increase_bytes,
            "full_phase_tree_logical_bytes": (self.full_phase_tree_logical_bytes),
            "minimum_reduction_fraction": (self.minimum_reduction_fraction),
            "relative_maximum_increase_bytes": (self.relative_maximum_increase_bytes),
            "maximum_process_rss_bytes": self.maximum_process_rss_bytes,
            "absolute_headroom_bytes": self.absolute_headroom_bytes,
            "maximum_increase_bytes": self.maximum_increase_bytes,
            "sample_interval_seconds": self.sample_interval_seconds,
            "checkpoint_count": len(self.checkpoints),
            "checkpoints": list(self.checkpoints),
            "sampler_stopped": True,
            "physical_rss_gate_passed": passed,
            "comparison_basis": (
                "maximum_of_current_rss_growth_from_start_current_and_"
                "lifetime_hwm_growth_from_start_hwm_versus_authenticated_"
                "full_phase_tree_logical_bytes_and_absolute_process_cap"
            ),
        }
        if not passed:
            report = self._checkpoint_receipt(
                "rss_guard_close",
                passed=False,
            )
            self.last_report = report
            raise Lfm2CpcfV5PhysicalRssError(
                "Physical RSS did not prove material reduction from the full "
                "phase-tree baseline.",
                report,
            )
        report = MappingProxyType({**body, "report_sha256": canonical_sha256(body)})
        self.last_report = report
        return report

    def abort(self) -> Mapping[str, Any] | None:
        if self._stopped:
            return self.last_report
        self._stop.set()
        self._thread.join(timeout=max(2.0, self.sample_interval_seconds * 4))
        self._stopped = True
        self._outstanding_checkpoint_authorization = None
        self._receipt_construction_authorization = None
        if self.last_report is None and self._violation is not None:
            self.last_report = self._checkpoint_receipt(
                "rss_guard_abort",
                passed=False,
            )
        return self.last_report


def _stub_layers(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    phase_index: int,
    *,
    active_layer_index: int | None = None,
    active_evidence: Lfm2CpcfInstallableLayerEvidence | None = None,
) -> tuple[Lfm2CpcfInstallableLayerEvidence, ...]:
    reference = catalog.phase_references[phase_index]
    result = []
    for layer_index, operator_kind in enumerate(reference.layer_operator_kinds):
        if layer_index == active_layer_index:
            if (
                not isinstance(
                    active_evidence,
                    Lfm2CpcfInstallableLayerEvidence,
                )
                or active_evidence.layer_index != layer_index
                or active_evidence.operator_kind != operator_kind
                or active_evidence.source_width != catalog.source_width
                or active_evidence.evidence_sha256
                != reference.layer_evidence_sha256[layer_index]
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "Active layer evidence differs from its phase reference."
                )
            result.append(active_evidence)
        else:
            result.append(
                Lfm2CpcfInstallableLayerEvidence(
                    layer_index=layer_index,
                    operator_kind=operator_kind,
                    source_width=catalog.source_width,
                    tensors=MappingProxyType({}),
                    evidence_sha256=reference.layer_evidence_sha256[layer_index],
                )
            )
    return tuple(result)


def _phase_shell(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    phase_index: int,
    *,
    active_layer_index: int | None = None,
    active_evidence: Lfm2CpcfInstallableLayerEvidence | None = None,
    final_norm_tensors: Mapping[str, Any] | None = None,
) -> Lfm2CpcfInstallablePhaseEvidence:
    reference = catalog.phase_references[phase_index]
    return Lfm2CpcfInstallablePhaseEvidence(
        cache_key=catalog.cache_key,
        batch_index=reference.batch_index,
        input_sha256=reference.input_sha256,
        phase=reference.phase,
        layers=_stub_layers(
            catalog,
            phase_index,
            active_layer_index=active_layer_index,
            active_evidence=active_evidence,
        ),
        final_norm_tensors=(
            MappingProxyType({})
            if final_norm_tensors is None
            else MappingProxyType(dict(final_norm_tensors))
        ),
        evidence_sha256=reference.existing_phase_evidence_sha256,
    )


def _target_prefix(
    record: Any,
    *,
    phase_name: str,
) -> tuple[Any, Any, Any]:
    torch = _torch()
    prompt_ids = (
        legacy_model_fold._torch()
        .as_tensor(record.target_input_ids)
        .detach()
        .to(device="cpu", dtype=torch.long)
        .contiguous()
    )
    prompt_mask = (
        legacy_model_fold._torch()
        .as_tensor(record.attention_mask)
        .detach()
        .to(device="cpu", dtype=torch.long)
        .contiguous()
    )
    forced = (
        legacy_model_fold._torch()
        .as_tensor(record.target_forced_decode_ids)
        .detach()
        .to(device="cpu", dtype=torch.long)
        .contiguous()
    )
    if prompt_ids.ndim != 2 or prompt_mask.shape != prompt_ids.shape:
        raise Lfm2CpcfV5ModelCompilerError(
            "Target tokenizer record IDs and attention mask are invalid."
        )
    raw_positions = getattr(record, "position_ids", None)
    if raw_positions is None:
        prompt_positions = (
            torch.arange(int(prompt_ids.shape[1]), dtype=torch.long)
            .unsqueeze(0)
            .expand(int(prompt_ids.shape[0]), -1)
            .contiguous()
        )
    else:
        prompt_positions = (
            torch.as_tensor(raw_positions)
            .detach()
            .to(
                device="cpu",
                dtype=torch.long,
            )
            .contiguous()
        )
    if phase_name in {"prefill", "cached_prefill"}:
        return prompt_ids, prompt_mask, prompt_positions
    match = _DECODE_PHASE.fullmatch(phase_name)
    if match is None:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Unsupported selective phase {phase_name!r}."
        )
    count = int(match.group(1)) + 1
    if forced.ndim != 2 or count > int(forced.shape[1]):
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective decode phase exceeds target forced-decode IDs."
        )
    suffix_ids = forced[:, :count]
    ids = torch.cat((prompt_ids, suffix_ids), dim=1).contiguous()
    mask = torch.cat(
        (
            prompt_mask,
            torch.ones_like(suffix_ids, dtype=torch.long),
        ),
        dim=1,
    ).contiguous()
    start = prompt_positions[:, -1:] + 1
    offsets = torch.arange(count, dtype=torch.long).unsqueeze(0)
    positions = torch.cat(
        (prompt_positions, start + offsets),
        dim=1,
    ).contiguous()
    return ids, mask, positions


def build_lfm2_cpcf_v5_target_phase_shells(
    trace_adapter: Any,
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    bootstrap_payload: Lfm2CpcfSelectiveLayerPayload | None,
    *,
    final_norm_payload: Lfm2CpcfSelectiveFinalNormPayload | None = None,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Build canonical target phases with no resident parent tensor tree."""

    verify = getattr(trace_adapter, "verify", None)
    records = getattr(trace_adapter, "records", None)
    if not callable(verify) or not isinstance(records, tuple):
        raise Lfm2CpcfV5ModelCompilerError(
            "A verified strict tokenizer trace adapter is required."
        )
    verify()
    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise TypeError("catalog must be Lfm2CpcfSelectiveEvidenceCatalog.")
    layer_bootstrap = isinstance(
        bootstrap_payload,
        Lfm2CpcfSelectiveLayerPayload,
    )
    final_bootstrap = isinstance(
        final_norm_payload,
        Lfm2CpcfSelectiveFinalNormPayload,
    )
    if layer_bootstrap == final_bootstrap:
        raise TypeError(
            "Exactly one layer or final-normalization bootstrap payload is required."
        )
    if layer_bootstrap and (
        len(bootstrap_payload.phases) != catalog.phase_count
        or bootstrap_payload.layer_index >= catalog.layer_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Bootstrap layer payload differs from the selective catalog."
        )
    if final_bootstrap and len(final_norm_payload.phases) != catalog.phase_count:
        raise Lfm2CpcfV5ModelCompilerError(
            "Bootstrap final-normalization payload differs from the selective catalog."
        )
    by_ordinal = {int(record.ordinal): record for record in records}
    if len(by_ordinal) != len(records):
        raise Lfm2CpcfV5ModelCompilerError(
            "Tokenizer trace record ordinals are not unique."
        )
    result = []
    payload_phases = (
        bootstrap_payload.phases if layer_bootstrap else final_norm_payload.phases
    )
    for phase_index, (reference, phase_payload) in enumerate(
        zip(catalog.phase_references, payload_phases, strict=True)
    ):
        if phase_payload.phase_reference_sha256 != reference.reference_sha256:
            raise Lfm2CpcfV5ModelCompilerError(
                "Bootstrap phase order or evidence binding changed."
            )
        if layer_bootstrap:
            evidence = phase_payload.evidence
            if evidence.layer_index != bootstrap_payload.layer_index:
                raise Lfm2CpcfV5ModelCompilerError(
                    "Bootstrap layer evidence index changed."
                )
        record = by_ordinal.get(reference.batch_index)
        if record is None or record.donor_cache_input_sha256 != reference.input_sha256:
            raise Lfm2CpcfV5ModelCompilerError(
                "Target tokenizer record is not bound to the parent cache."
            )
        try:
            sampled_tensor = (
                phase_payload.evidence.tensors["residual.pre"]
                if layer_bootstrap
                else phase_payload.tensors["output.final_norm_output"]
            )
            sampled = int(sampled_tensor.shape[1])
        except (AttributeError, KeyError, TypeError) as exc:
            raise Lfm2CpcfV5ModelCompilerError(
                "Bootstrap evidence lacks sampled-suffix geometry."
            ) from exc
        _positive_int(sampled, "sampled_suffix_tokens")
        ids, mask, positions = _target_prefix(
            record,
            phase_name=reference.phase,
        )
        shell = _phase_shell(catalog, phase_index)
        body = {
            "schema": LFM2_CPCF_TARGET_PHASE_SCHEMA,
            "batch_index": reference.batch_index,
            "record_id": str(record.record_id),
            "split": str(record.split),
            "phase": reference.phase,
            "donor_input_sha256": reference.input_sha256,
            "target_alignment_sha256": str(record.alignment_sha256),
            "input_ids_sha256": model_compiler._tensor_sha256(ids),
            "attention_mask_sha256": model_compiler._tensor_sha256(mask),
            "position_ids_sha256": model_compiler._tensor_sha256(positions),
            "batch_size": int(ids.shape[0]),
            "full_prefix_tokens": int(ids.shape[1]),
            "sampled_suffix_tokens": sampled,
            "parent_evidence_sha256": shell.evidence_sha256,
            "numeric_token_id_equality_required": False,
            "actual_child_full_prefix_required": True,
        }
        result.append(
            Lfm2CpcfTargetPhase(
                batch_index=reference.batch_index,
                record_id=str(record.record_id),
                split=str(record.split),
                phase=reference.phase,
                donor_input_sha256=reference.input_sha256,
                target_alignment_sha256=str(record.alignment_sha256),
                input_ids=ids,
                attention_mask=mask,
                position_ids=positions,
                sampled_suffix_tokens=sampled,
                parent_evidence=shell,
                phase_sha256=model_compiler._sha(body),
            )
        )
    phases = tuple(result)
    if tuple(phase.identity for phase in phases) != tuple(
        (reference.batch_index, reference.phase)
        for reference in catalog.phase_references
    ) or any(
        any(layer.tensors for layer in phase.parent_evidence.layers)
        or bool(phase.parent_evidence.final_norm_tensors)
        for phase in phases
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Tensor-free target phase shells are incomplete or retain tensors."
        )
    verify()
    return phases


def _copy_phase_with_evidence(
    phase: Lfm2CpcfTargetPhase,
    evidence: Lfm2CpcfInstallablePhaseEvidence,
) -> Lfm2CpcfTargetPhase:
    result = Lfm2CpcfTargetPhase(
        batch_index=phase.batch_index,
        record_id=phase.record_id,
        split=phase.split,
        phase=phase.phase,
        donor_input_sha256=phase.donor_input_sha256,
        target_alignment_sha256=phase.target_alignment_sha256,
        input_ids=phase.input_ids,
        attention_mask=phase.attention_mask,
        position_ids=phase.position_ids,
        sampled_suffix_tokens=phase.sampled_suffix_tokens,
        parent_evidence=evidence,
        phase_sha256=phase.phase_sha256,
    )
    if result.as_dict() != phase.as_dict():
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective phase materialization changed target phase metadata."
        )
    return result


def materialize_lfm2_cpcf_v5_layer_phases(
    phases: Sequence[Lfm2CpcfTargetPhase],
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    payload: Lfm2CpcfSelectiveLayerPayload,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Bind exactly one resident source layer to tensor-free target phases."""

    ordered = tuple(phases)
    if (
        not ordered
        or not isinstance(payload, Lfm2CpcfSelectiveLayerPayload)
        or len(ordered) != catalog.phase_count
        or len(payload.phases) != catalog.phase_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Layer phase materialization inputs are incomplete."
        )
    result = []
    for phase_index, (phase, phase_payload, reference) in enumerate(
        zip(
            ordered,
            payload.phases,
            catalog.phase_references,
            strict=True,
        )
    ):
        if (
            phase.identity != (reference.batch_index, reference.phase)
            or phase_payload.phase_reference_sha256 != reference.reference_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Layer payload phase order differs from target phases."
            )
        evidence = _phase_shell(
            catalog,
            phase_index,
            active_layer_index=payload.layer_index,
            active_evidence=phase_payload.evidence,
        )
        result.append(_copy_phase_with_evidence(phase, evidence))
    materialized = tuple(result)
    for phase in materialized:
        tensor_layers = [
            layer.layer_index
            for layer in phase.parent_evidence.layers
            if bool(layer.tensors)
        ]
        if tensor_layers != [payload.layer_index]:
            raise Lfm2CpcfV5ModelCompilerError(
                "Materialized target phase owns more than its active layer."
            )
    return materialized


def materialize_lfm2_cpcf_v5_final_norm_phases(
    phases: Sequence[Lfm2CpcfTargetPhase],
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    payload: Lfm2CpcfSelectiveFinalNormPayload,
) -> tuple[Lfm2CpcfTargetPhase, ...]:
    """Bind only final-normalization evidence to target phase shells."""

    ordered = tuple(phases)
    if (
        not ordered
        or not isinstance(payload, Lfm2CpcfSelectiveFinalNormPayload)
        or len(ordered) != catalog.phase_count
        or len(payload.phases) != catalog.phase_count
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization phase materialization is incomplete."
        )
    result = []
    for phase_index, (phase, phase_payload, reference) in enumerate(
        zip(
            ordered,
            payload.phases,
            catalog.phase_references,
            strict=True,
        )
    ):
        if (
            phase.identity != (reference.batch_index, reference.phase)
            or phase_payload.phase_reference_sha256 != reference.reference_sha256
            or phase_payload.evidence_sha256 != reference.final_norm_evidence_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization payload phase binding changed."
            )
        evidence = _phase_shell(
            catalog,
            phase_index,
            final_norm_tensors=phase_payload.tensors,
        )
        result.append(_copy_phase_with_evidence(phase, evidence))
    materialized = tuple(result)
    if any(
        any(layer.tensors for layer in phase.parent_evidence.layers)
        or not phase.parent_evidence.final_norm_tensors
        for phase in materialized
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization phases retained layer evidence."
        )
    return materialized


class _SelectiveBoundaryCompiler:
    """Fit the exact v2 boundary bundle without an all-layer tensor tree."""

    def __init__(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        catalog: Lfm2CpcfSelectiveEvidenceCatalog,
        source_tied_table: Any,
        *,
        target_hidden_width: int,
        model_policy: legacy_model_fold.Lfm2CpcfModelFoldPolicy,
        preserve_vocabulary_mean: bool = True,
        metric_chunk_rows: int = 512,
    ) -> None:
        self.phases = tuple(phases)
        self.catalog = catalog
        self.table = boundary._matrix(
            source_tied_table,
            "source_tied_table",
        )
        self.source_width = int(self.table.shape[1])
        self.target_width = boundary._positive_int(
            target_hidden_width,
            "target_hidden_width",
        )
        if (
            self.target_width >= self.source_width
            or self.source_width != catalog.source_width
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary widths differ from source/catalog geometry."
            )
        if not isinstance(preserve_vocabulary_mean, bool):
            raise TypeError("preserve_vocabulary_mean must be boolean.")
        self.preserve_vocabulary_mean = preserve_vocabulary_mean
        self.metric_chunk_rows = boundary._positive_int(
            metric_chunk_rows,
            "metric_chunk_rows",
        )
        self.model_policy = model_policy
        self.layer_kinds = tuple(catalog.phase_references[0].layer_operator_kinds)
        if len(self.layer_kinds) != catalog.layer_count or any(
            tuple(reference.layer_operator_kinds) != self.layer_kinds
            for reference in catalog.phase_references
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective catalog layer topology changes across phases."
            )
        self.split_phases = MappingProxyType(
            {
                split: tuple(phase for phase in self.phases if phase.split == split)
                for split in _SPLITS
            }
        )
        if not self.split_phases["fit"] or not self.split_phases["validation"]:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary fit needs fit and validation phases."
            )
        self.split_phase_sha256 = MappingProxyType(
            {
                split: boundary._sha256(
                    [
                        {
                            "ordinal": ordinal,
                            "batch_index": phase.batch_index,
                            "record_id": phase.record_id,
                            "phase": phase.phase,
                            "phase_sha256": phase.phase_sha256,
                            "parent_evidence_sha256": (
                                phase.parent_evidence.evidence_sha256
                            ),
                        }
                        for ordinal, phase in enumerate(self.split_phases[split])
                    ]
                )
                for split in _SPLITS
            }
        )
        self.split_row_sha256 = MappingProxyType(
            {
                split: boundary._sha256(
                    [
                        row
                        for phase in self.split_phases[split]
                        for row in boundary._row_keys(phase)
                    ]
                )
                for split in _SPLITS
            }
        )
        self.split_rows = MappingProxyType(
            {
                split: sum(
                    phase.batch_size * phase.sampled_suffix_tokens
                    for phase in self.split_phases[split]
                )
                for split in _SPLITS
            }
        )
        if (
            self.split_phase_sha256["fit"] == self.split_phase_sha256["validation"]
            or self.split_row_sha256["fit"] == self.split_row_sha256["validation"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Fit and validation phase/row identities are not disjoint."
            )
        self.fit_weights: Any | None = None
        self.validation_weights: Any | None = None
        self.fit_weight_summary: list[dict[str, Any]] | None = None
        self.validation_weight_summary: list[dict[str, Any]] | None = None
        self.base_interface: Any | None = None
        self.anchor_fold: Any | None = None
        self.interface: Any | None = None
        self.anchor_split_reports: dict[str, Any] | None = None
        self.fit_future: Any | None = None
        self.validation_future: Any | None = None
        self.layer_results: list[Any] = []
        self._observation_sha: dict[tuple[int, str], str] = {}

    def _minimal_inventory(
        self,
        tensors: Mapping[tuple[tuple[int, str], str], Any],
    ) -> boundary._PhaseInventory:
        return boundary._PhaseInventory(
            phases=self.phases,
            layer_kinds=self.layer_kinds,
            source_width=self.source_width,
            tensors=MappingProxyType(dict(tensors)),
            split_phases=self.split_phases,
            split_phase_sha256=self.split_phase_sha256,
            split_row_sha256=self.split_row_sha256,
            split_rows=self.split_rows,
            source_observation_sha256="0" * 64,
        )

    def fit_interface(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        *,
        target_norm_epsilon: float,
    ) -> Any:
        if self.interface is not None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective interface can be fitted exactly once."
            )
        ordered = tuple(phases)
        if tuple(phase.phase_sha256 for phase in ordered) != tuple(
            phase.phase_sha256 for phase in self.phases
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization phase lineage changed."
            )
        final_field = "output.final_norm_output"
        tensors = {}
        for phase_index, phase in enumerate(ordered):
            try:
                value = phase.parent_evidence.final_norm_tensors[final_field]
            except KeyError as exc:
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Phase {phase.identity!r} lacks final norm evidence."
                ) from exc
            tensor = boundary._phase_tensor(
                value,
                f"{phase.identity}.{final_field}",
                phase=phase,
                source_width=self.source_width,
            )
            tensors[(phase.identity, final_field)] = tensor
            self._observation_sha[(phase_index, final_field)] = boundary._tensor_sha256(
                tensor
            )
        inventory = self._minimal_inventory(tensors)
        (
            self.fit_weights,
            self.fit_weight_summary,
        ) = boundary._balanced_phase_weights(inventory, "fit")
        (
            self.validation_weights,
            self.validation_weight_summary,
        ) = boundary._balanced_phase_weights(inventory, "validation")
        fit_final = boundary._rows(inventory, "fit", final_field)
        validation_final = boundary._rows(
            inventory,
            "validation",
            final_field,
        )
        self.base_interface = fit_tied_causal_interface(
            self.table,
            fit_final,
            target_width=self.target_width,
            validation_producer_rows=validation_final,
            fit_weights=self.fit_weights,
            validation_weights=self.validation_weights,
            preserve_vocabulary_mean=self.preserve_vocabulary_mean,
            target_norm_epsilon=target_norm_epsilon,
            minimum_validation_top1_agreement=0.0,
            metric_chunk_rows=self.metric_chunk_rows,
        )
        anchor_splits = boundary._causal_anchor_splits(inventory)
        self.anchor_fold = fit_tied_causal_interface_anchor_residual(
            self.base_interface,
            self.table,
            selection_split=anchor_splits["selection"],
            fit_split=anchor_splits["fit"],
            validation_split=anchor_splits["validation"],
            target_norm_epsilon=target_norm_epsilon,
            metric_chunk_rows=self.metric_chunk_rows,
        )
        self.interface = self.anchor_fold.interface
        self.fit_future = self.interface.encode_producer_rows(fit_final)
        self.validation_future = self.interface.encode_producer_rows(validation_final)
        self.anchor_split_reports = {
            split: anchor_splits[split].as_dict() for split in _SPLITS
        }
        del inventory, tensors, fit_final, validation_final, anchor_splits
        gc.collect()
        return self.interface

    def fit_layer(
        self,
        phases: Sequence[Lfm2CpcfTargetPhase],
        *,
        layer_index: int,
    ) -> Any:
        index = boundary._positive_int(layer_index + 1, "layer_index") - 1
        if (
            self.interface is None
            or self.fit_weights is None
            or self.validation_weights is None
            or index != len(self.layer_results)
            or index >= len(self.layer_kinds)
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective layer charts must be fitted in canonical order "
                "after the interface."
            )
        ordered = tuple(phases)
        if tuple(phase.phase_sha256 for phase in ordered) != tuple(
            phase.phase_sha256 for phase in self.phases
        ):
            raise Lfm2CpcfV5ModelCompilerError("Layer chart phase lineage changed.")
        values: dict[str, dict[tuple[int, str], Any]] = {
            name: {} for name in _BOUNDARIES
        }
        for phase_index, phase in enumerate(ordered):
            evidence = phase.parent_evidence.layers[index]
            if (
                evidence.layer_index != index
                or evidence.operator_kind != self.layer_kinds[index]
                or evidence.source_width != self.source_width
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    f"Phase {phase.identity!r} active layer changed."
                )
            for name in _BOUNDARIES:
                try:
                    raw = evidence.tensors[name]
                except KeyError as exc:
                    raise Lfm2CpcfV5ModelCompilerError(
                        f"Phase {phase.identity!r} layer {index} lacks {name!r}."
                    ) from exc
                tensor = boundary._phase_tensor(
                    raw,
                    f"{phase.identity}.layer.{index:03d}.{name}",
                    phase=phase,
                    source_width=self.source_width,
                )
                values[name][phase.identity] = tensor
                field = f"layer.{index:03d}.{name}"
                self._observation_sha[(phase_index, field)] = boundary._tensor_sha256(
                    tensor
                )

        def rows(
            split: str,
            name: str,
            source_values: Mapping[
                str,
                Mapping[tuple[int, str], Any],
            ] = values,
        ) -> Any:
            torch = _torch()
            return torch.cat(
                [
                    source_values[name][phase.identity].reshape(
                        -1,
                        self.source_width,
                    )
                    for phase in self.split_phases[split]
                ],
                dim=0,
            ).contiguous()

        charts = {}
        for name in _BOUNDARIES:
            field = f"layer.{index:03d}.{name}"
            charts[name] = fit_cpcf_boundary_chart(
                field,
                rows("fit", name),
                self.fit_future,
                target_width=self.target_width,
                validation_parent_states=rows("validation", name),
                validation_future_coordinates=self.validation_future,
                fit_weights=self.fit_weights,
                validation_weights=self.validation_weights,
                state_reconstruction_weight=(
                    self.model_policy.state_reconstruction_weight
                ),
                ridge=self.model_policy.ridge,
                minimum_predictive_energy_fraction=(
                    self.model_policy.minimum_predictive_energy_fraction
                ),
                minimum_validation_block_retention=(
                    self.model_policy.minimum_validation_block_retention
                ),
                enforce_validation_block_floor=(
                    self.model_policy.enforce_validation_block_floor
                ),
            )
        result = boundary._layer_charts(
            layer_index=index,
            operator_kind=self.layer_kinds[index],
            post_operator=charts["residual.post_operator"],
            post_layer=charts["residual.post_layer"],
        )
        self.layer_results.append(result)
        del values, charts
        gc.collect()
        return result

    def replace_latest_layer_charts(
        self,
        source: boundary.Lfm2CpcfLayerBoundaryCharts,
        effective: boundary.Lfm2CpcfLayerBoundaryCharts,
        *,
        controller_report: Mapping[str, Any],
    ) -> None:
        """Persist one authenticated production chart transport in the frontier."""

        if (
            not self.layer_results
            or self.layer_results[-1] is not source
            or not isinstance(effective, boundary.Lfm2CpcfLayerBoundaryCharts)
            or not isinstance(controller_report, Mapping)
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Only the just-fitted source chart may be replaced by one "
                "typed effective chart."
            )
        report = dict(controller_report)
        report_body = dict(report)
        report_seal = report_body.pop("report_sha256", None)
        if (
            report_seal != _sha(report_body)
            or report.get("component") != "fit_only_post_operator_chart_transport"
            or report.get("production_profile_authorized") is not True
            or report.get("source_layer_charts_sha256") != source.layer_charts_sha256
            or report.get("transported_layer_charts_sha256")
            != effective.layer_charts_sha256
            or source.layer_index != effective.layer_index
            or source.operator_kind != effective.operator_kind
            or source.source_width != effective.source_width
            or source.target_width != effective.target_width
            or source.post_layer is not effective.post_layer
            or source.post_operator is effective.post_operator
            or report.get("unchanged_post_layer_report_sha256")
            != source.post_layer.report["report_sha256"]
            or report.get("transported_post_operator_report_sha256")
            != effective.post_operator.report["report_sha256"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "The production chart replacement is not exactly bound to "
                "the just-fitted source and transported boundary charts."
            )
        self.layer_results[-1] = effective

    def finalize(self) -> boundary.Lfm2CpcfBoundaryBundle:
        if (
            self.interface is None
            or self.base_interface is None
            or self.anchor_fold is None
            or self.anchor_split_reports is None
            or len(self.layer_results) != len(self.layer_kinds)
            or self.fit_weight_summary is None
            or self.validation_weight_summary is None
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective boundary bundle is incomplete."
            )
        final_field = "output.final_norm_output"
        observations = []
        for phase_index, phase in enumerate(self.phases):
            fields = [final_field]
            for layer_index in range(len(self.layer_kinds)):
                fields.extend(f"layer.{layer_index:03d}.{name}" for name in _BOUNDARIES)
            for field in fields:
                try:
                    tensor_sha = self._observation_sha[(phase_index, field)]
                except KeyError as exc:
                    raise Lfm2CpcfV5ModelCompilerError(
                        "Selective source observation inventory is incomplete."
                    ) from exc
                observations.append(
                    {
                        "phase_sha256": phase.phase_sha256,
                        "field": field,
                        "tensor_sha256": tensor_sha,
                    }
                )
        layers = tuple(self.layer_results)
        anchor_residual_selected = bool(
            self.anchor_fold.report["active_anchor_residual_selected"]
        )
        phase_counts = {split: len(self.split_phases[split]) for split in _SPLITS}
        row_counts = dict(self.split_rows)
        split_phase_hashes = dict(self.split_phase_sha256)
        split_row_hashes = dict(self.split_row_sha256)
        expected_observations = len(self.phases) * (1 + 2 * len(self.layer_kinds))
        predictive_floor = self.model_policy.minimum_predictive_energy_fraction
        validation_floor = self.model_policy.minimum_validation_block_retention
        body = {
            "schema": boundary.LFM2_CPCF_BOUNDARY_BUNDLE_SCHEMA,
            "method": (
                "tied_causal_interface_then_behavior_preserving_anchor_selection_then_"
                "independent_predictive_boundary_charts"
            ),
            "source_width": self.source_width,
            "target_width": self.target_width,
            "vocabulary_size": int(self.table.shape[0]),
            "layer_count": len(layers),
            "layer_operator_kinds": list(self.layer_kinds),
            "interface_report_sha256": boundary._report_digest(
                self.interface.report,
                "interface",
            ),
            "base_interface_report_sha256": boundary._report_digest(
                self.base_interface.report,
                "base_interface",
            ),
            "causal_anchor_report_sha256": boundary._report_digest(
                self.anchor_fold.report,
                "causal_anchor",
            ),
            "interface_lineage": {
                "base_interface_report": _plain_metadata(self.base_interface.report),
                "causal_anchor_report": _plain_metadata(self.anchor_fold.report),
                "base_interface_report_sha256": boundary._report_digest(
                    self.base_interface.report,
                    "base_interface",
                ),
                "causal_anchor_report_sha256": boundary._report_digest(
                    self.anchor_fold.report,
                    "causal_anchor",
                ),
                "active_interface_report_sha256": boundary._report_digest(
                    self.interface.report,
                    "interface",
                ),
                "active_interface_is_anchor_repaired": anchor_residual_selected,
                "active_interface_is_base_control": not anchor_residual_selected,
                "producer_encoder_fixed_by_anchor_selection": True,
                "producer_encoder_fixed_by_anchor_repair": (anchor_residual_selected),
                "additional_runtime_parameter_count": 0,
            },
            "causal_anchor_extraction": {
                "method": (
                    "one_last_sampled_final_norm_endpoint_per_record_causal_step"
                ),
                "label_source": "next_phase_newly_appended_target_token",
                "phase_label_contract": {
                    "prefill": "forced_decode[0]",
                    "cached_prefill": "forced_decode[0]",
                    "decode.k": "forced_decode[k+1]",
                },
                "final_unlabeled_phase_excluded": True,
                "canonical_prefill_or_cached_prefill_decode_order_required": True,
                "one_unique_row_id_per_record_batch_row_causal_step": True,
                "row_id_binds_batch_index_batch_row_and_prefix_position": True,
                "no_sampled_prefix_row_duplication": True,
                "split_inventory": _plain_metadata(self.anchor_split_reports),
            },
            "layer_charts": [_plain_metadata(layer.as_dict()) for layer in layers],
            "split_phase_counts": phase_counts,
            "split_row_counts": row_counts,
            "split_phase_sha256": split_phase_hashes,
            "split_row_sha256": split_row_hashes,
            "fit_validation_disjoint_phase_identities": True,
            "fit_validation_disjoint_row_identities": True,
            "selection": {
                "phase_count": phase_counts["selection"],
                "row_count": row_counts["selection"],
                "phase_sha256": split_phase_hashes["selection"],
                "row_sha256": split_row_hashes["selection"],
                "used_to_fit_base_interface": False,
                "used_to_fit_causal_anchor_values": False,
                "used_to_select_causal_anchor_ridge_margin": True,
                "used_to_fit_boundary_charts": False,
                "used_for_validation_gates": False,
                "authenticated_and_receipted": True,
            },
            "phase_weighting": {
                "method": "equal_phase_mass_then_uniform_rows_within_phase",
                "balanced_phase_weights": True,
                "fit_total_weight": float(self.fit_weights.sum().item()),
                "validation_total_weight": float(self.validation_weights.sum().item()),
                "fit_weights_sha256": boundary._tensor_sha256(self.fit_weights),
                "validation_weights_sha256": boundary._tensor_sha256(
                    self.validation_weights
                ),
                "fit_phases": _plain_metadata(self.fit_weight_summary),
                "validation_phases": _plain_metadata(self.validation_weight_summary),
            },
            "interface_top1": {
                "principal_objective": False,
                "capacity_gate_used": False,
                "fit_rmsnorm_parent_top1_agreement": (
                    self.base_interface.report["fit"]["rmsnorm_parent_top1_agreement"]
                ),
                "validation_rmsnorm_parent_top1_agreement": (
                    self.base_interface.report["validation"][
                        "rmsnorm_parent_top1_agreement"
                    ]
                ),
                "diagnostic_only": True,
            },
            "predictive_heldout_gates": {
                "principal_boundary_objective": True,
                "minimum_predictive_energy_fraction": predictive_floor,
                "minimum_validation_block_retention": validation_floor,
                "validation_block_floor_enforced": (
                    self.model_policy.enforce_validation_block_floor
                ),
                "fit_rows_used_for_parameters": True,
                "validation_rows_used_for_parameters": False,
            },
            "source_coverage": {
                "phase_count": len(self.phases),
                "layer_count": len(self.layer_kinds),
                "boundary_fields_per_layer": list(_BOUNDARIES),
                "final_output_field": final_field,
                "expected_tensor_observations": expected_observations,
                "observed_tensor_observations": len(observations),
                "source_observation_sha256": boundary._sha256(observations),
                "all_phase_final_outputs_covered": True,
                "all_layer_post_operator_boundaries_covered": True,
                "all_layer_post_layer_boundaries_covered": True,
                "full_source_coverage": True,
            },
            "coordinate_contract": {
                "future_coordinates": (
                    "behavior_selected_tied_interface_encoded_parent_final_norm_output"
                ),
                "active_interface_is_anchor_repaired": anchor_residual_selected,
                "active_interface_is_base_control": not anchor_residual_selected,
                "one_global_residual_basis_used": False,
                "independent_post_operator_and_post_layer_charts": True,
                "residual_write_formula": (
                    "encoded_parent_endpoint_minus_actual_child_pre_state"
                ),
            },
            "operation_audit": {
                "parent_forward_count": 0,
                "target_forward_count": 0,
                "parent_generation_count": 0,
                "parent_cache_regeneration_count": 0,
                "model_mutation_count": 0,
                "training_step_count": 0,
                "backward_call_count": 0,
                "optimizer_step_count": 0,
                "quantization_used": False,
                "pruning_used": False,
                "blank_target_values_read": False,
            },
        }
        if len(observations) != expected_observations:
            raise Lfm2CpcfV5ModelCompilerError(
                "Selective source observation count is incomplete."
            )
        bundle_sha = boundary._sha256(body)
        return boundary.Lfm2CpcfBoundaryBundle(
            interface=self.interface,
            layers=layers,
            split_phase_sha256=split_phase_hashes,
            split_row_sha256=split_row_hashes,
            report={**body, "bundle_sha256": bundle_sha},
            bundle_sha256=bundle_sha,
        )


def _authenticated_bootstrap_payload(
    provider: Lfm2CpcfSelectiveEvidenceProvider,
    *,
    layer_index: int,
) -> tuple[
    Lfm2CpcfSelectiveLayerPayload,
    DynamicEvidenceLayerAuthentication,
]:
    reference = provider.dynamic_catalog.references[layer_index]
    payload = provider.load_layer_payload(reference)
    try:
        authentication = provider.authenticate_layer_payload(
            reference,
            payload,
        )
    except BaseException:
        provider.release_layer_payload(reference, payload)
        raise
    if (
        not isinstance(
            authentication,
            DynamicEvidenceLayerAuthentication,
        )
        or authentication.layer_index != layer_index
        or authentication.evidence_sha256 != reference.evidence_sha256
        or authentication.logical_bytes != reference.logical_bytes
    ):
        provider.release_layer_payload(reference, payload)
        raise Lfm2CpcfV5ModelCompilerError(
            "Bootstrap evidence authentication differs from the catalog."
        )
    return payload, authentication


_REAL_V5_NUMERICAL_ADAPTER_IMPLEMENTATION = (
    "neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter."
    "Lfm2CpcfShortConvV5NumericalFrontierAdapter.v6"
)
_REAL_V5_NUMERICAL_ZERO_EXECUTION_FIELDS = {
    "parent_model_forward_calls",
    "target_model_forward_calls",
    "generation_calls",
    "training_steps",
    "backward_calls",
    "optimizer_steps",
    "parameters_pruned",
    "parameters_quantized",
}


def _mapping_seal_is_valid(
    value: Mapping[str, Any],
    *,
    seal_field: str,
) -> bool:
    body = dict(value)
    supplied = body.pop(seal_field, None)
    return bool(
        isinstance(supplied, str)
        and _SHA256.fullmatch(supplied) is not None
        and supplied == _sha(body)
    )


def _real_v5_numerical_adapter_contract(
    adapter: Lfm2CpcfShortConvNumericalFrontierAdapter | None,
) -> Mapping[str, Any] | None:
    """Return the authenticated live policy contract for the concrete adapter."""

    if type(adapter) is not Lfm2CpcfShortConvV5NumericalFrontierAdapter:
        return None
    basis_policy = adapter.basis_policy
    ridge_policy = adapter.ridge_policy
    if (
        not basis_policy.verify_authentication()
        or not ridge_policy.verify_authentication()
    ):
        return None
    body = {
        "adapter_type": (f"{type(adapter).__module__}.{type(adapter).__qualname__}"),
        "adapter_id": adapter.adapter_id,
        "implementation": _REAL_V5_NUMERICAL_ADAPTER_IMPLEMENTATION,
        "basis_policy": basis_policy.authentication_body(),
        "basis_policy_sha256": basis_policy.authentication_sha256,
        "ridge_policy": ridge_policy.authentication_body(),
        "ridge_policy_sha256": ridge_policy.authentication_sha256,
    }
    return MappingProxyType({**body, "contract_sha256": _sha(body)})


def _prove_real_v5_numerical_layer(
    receipt: legacy_model_fold.Lfm2CpcfCompletedLayerReceipt,
    *,
    adapter_contract: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Verify one installed short-conv layer came from the concrete v5 adapter."""

    report = receipt.report
    if report.get("operator_kind") != "short_conv":
        return None
    artifact = report.get("operator_artifact")
    if not isinstance(artifact, Mapping):
        return None
    frontier = artifact.get("numerical_frontier")
    if not isinstance(frontier, Mapping):
        return None
    provenance = frontier.get("adapter_provenance")
    if not isinstance(provenance, Mapping):
        return None
    frontier_execution = frontier.get("execution")
    provenance_execution = provenance.get("execution")
    zero_frontier_execution = (
        isinstance(frontier_execution, Mapping)
        and set(frontier_execution) == _REAL_V5_NUMERICAL_ZERO_EXECUTION_FIELDS
        and all(
            type(frontier_execution[name]) is int and frontier_execution[name] == 0
            for name in _REAL_V5_NUMERICAL_ZERO_EXECUTION_FIELDS
        )
    )
    zero_provenance_execution = isinstance(
        provenance_execution,
        Mapping,
    ) and (
        set(provenance_execution) == _REAL_V5_NUMERICAL_ZERO_EXECUTION_FIELDS
        and all(
            type(provenance_execution[name]) is int and provenance_execution[name] == 0
            for name in _REAL_V5_NUMERICAL_ZERO_EXECUTION_FIELDS
        )
    )
    requested_commutator_gate = adapter_contract["basis_policy"][
        "maximum_per_lag_commutator_relative_frobenius"
    ]
    observed_commutator = provenance.get("basis_actual_commutator_residual")
    if (
        not _mapping_seal_is_valid(report, seal_field="result_sha256")
        or not _mapping_seal_is_valid(
            artifact,
            seal_field="report_sha256",
        )
        or not _mapping_seal_is_valid(
            frontier,
            seal_field="report_sha256",
        )
        or frontier.get("adapter_provenance_sha256") != _sha(provenance)
        or adapter_contract.get("contract_sha256")
        != _sha(
            {
                name: value
                for name, value in adapter_contract.items()
                if name != "contract_sha256"
            }
        )
        or artifact.get("numerical_frontier_adapter_active") is not True
        or artifact.get("fit_only_numerical_generation") is not True
        or artifact.get("legacy_v4_numerical_generation_claimed") is not False
        or frontier.get("kind") != "authenticated_fit_only_numerical_frontier"
        or frontier.get("adapter_id") != adapter_contract["adapter_id"]
        or frontier.get("full_source_numeric_participation") is not True
        or frontier.get("source_channels_dropped") != 0
        or frontier.get("fit_only") is not True
        or frontier.get("selection_cases_consumed") != 0
        or frontier.get("validation_cases_consumed") != 0
        or provenance.get("implementation") != adapter_contract["implementation"]
        or provenance.get("real_v5_numerical_algorithm") is not True
        or provenance.get("dynamic_source_target_topology") is not True
        or provenance.get("basis_policy_sha256")
        != adapter_contract["basis_policy_sha256"]
        or provenance.get("ridge_policy_sha256")
        != adapter_contract["ridge_policy_sha256"]
        or provenance.get("basis_actual_commutator_gate") != requested_commutator_gate
        or not isinstance(observed_commutator, (int, float))
        or isinstance(observed_commutator, bool)
        or not math.isfinite(float(observed_commutator))
        or float(observed_commutator) > float(requested_commutator_gate)
        or provenance.get("basis_actual_commutator_gate_passed") is not True
        or provenance.get("basis_full_source_numeric_participation") is not True
        or provenance.get("source_channels_dropped") != 0
        or provenance.get("source_channels_pruned") != 0
        or provenance.get("source_channels_quantized") != 0
        or provenance.get("source_fir_resized_before_basis") is not True
        or provenance.get("fir_solver_backend")
        != "authenticated_cpu_fp64_batched_all_channel_ridge"
        or provenance.get("fir_solver_receipt_scope")
        != "aggregate_all_target_fir_channels"
        or provenance.get("fir_scalar_oracle_policy")
        != ("deterministic_first_middle_last_plus_threshold_ambiguous_fallback")
        or provenance.get("fit_only") is not True
        or provenance.get("selection_cases_consumed") != 0
        or provenance.get("validation_cases_consumed") != 0
        or provenance.get("selection_performed_by_adapter") is not False
        or provenance.get("validation_performed_by_adapter") is not False
        or provenance.get("model_forward_calls") != 0
        or provenance.get("training") is not False
        or provenance.get("pruning") is not False
        or provenance.get("quantization") is not False
        or not zero_frontier_execution
        or not zero_provenance_execution
    ):
        return None
    body = {
        "layer_index": receipt.layer_index,
        "operator_artifact_report_sha256": artifact["report_sha256"],
        "numerical_frontier_report_sha256": frontier["report_sha256"],
        "adapter_provenance_sha256": frontier["adapter_provenance_sha256"],
        "adapter_contract_sha256": adapter_contract["contract_sha256"],
        "basis_policy_sha256": adapter_contract["basis_policy_sha256"],
        "ridge_policy_sha256": adapter_contract["ridge_policy_sha256"],
        "projected_fir_offdiagonal_gate": adapter_contract["basis_policy"][
            "maximum_projected_fir_offdiagonal_relative_frobenius"
        ],
        "per_lag_commutator_gate": requested_commutator_gate,
        "observed_per_lag_commutator_residual": observed_commutator,
        "full_source_numeric_participation": True,
        "fit_only_candidate_generation": True,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "concrete_real_v5_adapter_provenance_proven": True,
    }
    return MappingProxyType({**body, "proof_sha256": _sha(body)})


def _stable_python_code_sha256(code: CodeType) -> str:
    """Fingerprint Python semantics without CPython's adaptive cache state.

    ``marshal.dumps(code)`` is not a stable runtime identity on adaptive
    CPython builds: executing a function can specialize interpreter caches
    embedded in the marshalled code object.  The public ``co_code`` view is
    de-specialized, so bind it together with every execution-relevant constant,
    name, signature, flag, and exception-table field.
    """

    if not isinstance(code, CodeType):
        raise TypeError("code must be a Python code object.")

    def constant_manifest(value: Any) -> Any:
        if isinstance(value, CodeType):
            return {
                "kind": "nested_code",
                "manifest": code_manifest(value),
            }
        if value is None:
            return {"kind": "none"}
        if value is Ellipsis:
            return {"kind": "ellipsis"}
        if value is NotImplemented:
            return {"kind": "not_implemented"}
        if isinstance(value, bool):
            return {"kind": "bool", "value": value}
        if isinstance(value, int):
            return {"kind": "int", "value": str(value)}
        if isinstance(value, float):
            return {"kind": "float", "value": value.hex()}
        if isinstance(value, complex):
            return {
                "kind": "complex",
                "real": value.real.hex(),
                "imag": value.imag.hex(),
            }
        if isinstance(value, str):
            return {"kind": "str", "value": value}
        if isinstance(value, bytes):
            return {
                "kind": "bytes",
                "length": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            }
        if isinstance(value, tuple):
            return {
                "kind": "tuple",
                "values": [constant_manifest(child) for child in value],
            }
        if isinstance(value, frozenset):
            children = [constant_manifest(child) for child in value]
            return {
                "kind": "frozenset",
                "values": sorted(children, key=_canonical_json),
            }
        raise Lfm2CpcfV5ModelCompilerError(
            "Python code contains an unsupported constant type "
            f"{type(value).__module__}.{type(value).__qualname__}."
        )

    def code_manifest(value: CodeType) -> dict[str, Any]:
        return {
            "name": value.co_name,
            "qualname": getattr(value, "co_qualname", value.co_name),
            "argcount": value.co_argcount,
            "posonlyargcount": value.co_posonlyargcount,
            "kwonlyargcount": value.co_kwonlyargcount,
            "nlocals": value.co_nlocals,
            "stacksize": value.co_stacksize,
            "flags": value.co_flags,
            "bytecode_sha256": hashlib.sha256(value.co_code).hexdigest(),
            "exception_table_sha256": hashlib.sha256(
                getattr(value, "co_exceptiontable", b"")
            ).hexdigest(),
            "constants": [constant_manifest(child) for child in value.co_consts],
            "names": list(value.co_names),
            "varnames": list(value.co_varnames),
            "freevars": list(value.co_freevars),
            "cellvars": list(value.co_cellvars),
        }

    return _sha(
        {
            "schema": "neural-forge-stable-python-code-fingerprint-v1",
            "code": code_manifest(code),
        }
    )


def _optional_runtime_contract_sha256(
    value: Any,
    *,
    kind: str,
) -> str:
    if value is None:
        return _sha({"kind": kind, "provided": False})

    repository_root = Path(__file__).resolve().parents[2]
    first_party_source_roots = (
        repository_root / "src",
        repository_root / "experiments",
    )

    def first_party_source(raw_path: Any) -> bool:
        if not isinstance(raw_path, str):
            return False
        resolved = Path(raw_path).resolve()
        return any(resolved.is_relative_to(root) for root in first_party_source_roots)

    manifest_first_seen: dict[int, tuple[Any, str]] = {}

    def state_graph_reference(
        state_value: Any,
        *,
        name: str,
        active: frozenset[int],
    ) -> dict[str, Any] | None:
        """Bind shared/cyclic topology to a canonical, address-free path."""

        identity = id(state_value)
        existing = manifest_first_seen.get(identity)
        if existing is None:
            # Retaining the object prevents process-local id reuse during this
            # contract invocation.  The object itself is never serialized.
            manifest_first_seen[identity] = (state_value, name)
            return None
        existing_value, target_path = existing
        if existing_value is not state_value:
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} runtime graph identity changed during authentication."
            )
        return {
            "kind": "runtime_state_reference",
            "relation": ("cyclic_back_edge" if identity in active else "shared_alias"),
            "type": (
                f"{type(state_value).__module__}.{type(state_value).__qualname__}"
            ),
            "target_path": target_path,
        }

    def forbidden_prebound_alias(state_value: Any) -> tuple[str, str] | None:
        """Identify callable aliases which the process-local guard cannot patch."""

        if isinstance(state_value, ModuleType):
            module_name = getattr(state_value, "__name__", None)
            qualname = None
        elif isinstance(state_value, type) or callable(state_value):
            function = getattr(state_value, "__func__", state_value)
            module_name = getattr(function, "__module__", None)
            qualname = getattr(function, "__qualname__", None)
            if not isinstance(module_name, str):
                module_name = getattr(type(state_value), "__module__", None)
            if not isinstance(qualname, str):
                qualname = getattr(type(state_value), "__qualname__", None)
        else:
            return None
        if not isinstance(module_name, str):
            return None
        symbol = (
            module_name
            if not isinstance(qualname, str)
            else f"{module_name}.{qualname}"
        )
        forbidden_module_categories = (
            ("torch.optim", "training"),
            ("torch.nn.utils.prune", "pruning"),
            ("torch.ao.quantization", "quantization"),
            ("torch.quantization", "quantization"),
            ("torchao.quantization", "quantization"),
        )
        for module_prefix, category in forbidden_module_categories:
            if module_name == module_prefix or module_name.startswith(
                f"{module_prefix}."
            ):
                return category, symbol
        if symbol in {
            "torch.autograd.backward",
            "torch.autograd.grad",
            "torch._tensor.Tensor.backward",
            "torch.autograd.grad_mode.enable_grad",
            "torch.autograd.grad_mode.set_grad_enabled",
        }:
            return "training", symbol
        callable_name = (
            None if not isinstance(qualname, str) else qualname.rsplit(".", 1)[-1]
        )
        if module_name in {"torch", "torch._C"} and callable_name in {
            "quantize_per_tensor",
            "quantize_per_channel",
            "fake_quantize_per_tensor_affine",
            "fake_quantize_per_channel_affine",
        }:
            return "quantization", symbol
        return None

    alias_audit_seen: set[int] = set()
    alias_audit_module_attributes: set[tuple[int, str]] = set()

    def audit_reachable_prebound_aliases(
        state_value: Any,
        *,
        name: str,
        referenced_names: frozenset[str] = frozenset(),
    ) -> None:
        """Scan through a shallow-manifest boundary without losing alias checks."""

        forbidden_alias = forbidden_prebound_alias(state_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        if state_value is None or isinstance(
            state_value,
            (bool, int, float, str, bytes, os.PathLike),
        ):
            return
        identity = id(state_value)
        if isinstance(state_value, ModuleType):
            for attribute in sorted(referenced_names):
                # A module's meta-namespace contains unrelated global
                # registries.  It is source-bound above, but treating the
                # whole __dict__ as one callable dependency creates false
                # reachability through library dispatch infrastructure.
                if attribute.startswith("__") and attribute.endswith("__"):
                    continue
                binding = (identity, attribute)
                if binding in alias_audit_module_attributes:
                    continue
                alias_audit_module_attributes.add(binding)
                if hasattr(state_value, attribute):
                    audit_reachable_prebound_aliases(
                        getattr(state_value, attribute),
                        name=f"{name}.{attribute}",
                    )
            return
        if identity in alias_audit_seen:
            return
        alias_audit_seen.add(identity)
        if isinstance(state_value, type):
            return
        if isinstance(state_value, _torch().Tensor):
            return
        if isinstance(state_value, Mapping):
            for key, child in state_value.items():
                audit_reachable_prebound_aliases(
                    key,
                    name=f"{name}.<mapping-key>",
                )
                audit_reachable_prebound_aliases(
                    child,
                    name=f"{name}.<mapping-value>",
                )
            return
        if isinstance(state_value, (list, tuple, set, frozenset)):
            for child in state_value:
                audit_reachable_prebound_aliases(
                    child,
                    name=f"{name}[]",
                )
            return
        if callable(state_value):
            function = getattr(state_value, "__func__", state_value)
            owner = getattr(state_value, "__self__", None)
            if getattr(function, "__code__", None) is None:
                callable_impl = getattr(type(state_value), "__call__", None)
                if getattr(callable_impl, "__code__", None) is None:
                    return
                if owner is None:
                    owner = state_value
                function = callable_impl
            code = function.__code__
            for child in (
                getattr(function, "__defaults__", None),
                getattr(function, "__kwdefaults__", None),
            ):
                audit_reachable_prebound_aliases(
                    child,
                    name=f"{name}.<callable-state>",
                )
            closure = getattr(function, "__closure__", None)
            if closure is not None:
                for cell in closure:
                    audit_reachable_prebound_aliases(
                        cell.cell_contents,
                        name=f"{name}.<closure>",
                    )
            global_namespace = getattr(function, "__globals__", {})
            function_names = frozenset(str(value) for value in code.co_names)
            for global_name in sorted(function_names):
                if global_name not in global_namespace:
                    continue
                child = global_namespace[global_name]
                audit_reachable_prebound_aliases(
                    child,
                    name=f"{name}.__globals__.{global_name}",
                    referenced_names=(
                        function_names if isinstance(child, ModuleType) else frozenset()
                    ),
                )
            if owner is None or isinstance(owner, type):
                return
            state_value = owner
            identity = id(state_value)
            if identity in alias_audit_seen:
                return
            alias_audit_seen.add(identity)
        fields: dict[str, Any] = {}
        raw_dict = getattr(state_value, "__dict__", None)
        if isinstance(raw_dict, dict):
            fields.update(raw_dict)
        for owner_type in type(state_value).__mro__:
            slots = getattr(owner_type, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for field in slots:
                if (
                    field in {"__dict__", "__weakref__"}
                    or field in fields
                    or not hasattr(state_value, field)
                ):
                    continue
                fields[field] = getattr(state_value, field)
        for field in sorted(fields):
            audit_reachable_prebound_aliases(
                fields[field],
                name=f"{name}.{field}",
            )

    def instance_state_manifest(
        instance: Any,
        *,
        name: str,
        excluded: frozenset[str] = frozenset(),
        active: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        identity = id(instance)
        type_name = f"{type(instance).__module__}.{type(instance).__qualname__}"
        reference = state_graph_reference(
            instance,
            name=name,
            active=active,
        )
        if reference is not None:
            return reference
        fields: dict[str, Any] = {}
        raw_dict = getattr(instance, "__dict__", None)
        if isinstance(raw_dict, dict):
            fields.update(raw_dict)
        for owner in type(instance).__mro__:
            slots = getattr(owner, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for field in slots:
                if (
                    field in {"__dict__", "__weakref__"}
                    or field in fields
                    or not hasattr(instance, field)
                ):
                    continue
                fields[field] = getattr(instance, field)
        normalized = {
            field: state_value_manifest(
                fields[field],
                name=f"{name}.{field}",
                active=active | {identity},
            )
            for field in sorted(fields)
            if field not in excluded
        }
        return {
            "type": type_name,
            "fields": normalized,
        }

    def mapping_member_manifest(
        member: Any,
        *,
        name: str,
        active: frozenset[int],
    ) -> Any:
        """Bind registry members without expanding every dispatch target graph."""

        audit_reachable_prebound_aliases(member, name=name)
        if isinstance(member, ModuleType):
            return runtime_module_manifest(
                member,
                name=name,
                referenced_names=frozenset(),
                active=active,
            )
        if isinstance(member, type):
            return runtime_class_manifest(
                member,
                name=name,
                bind_python_methods=False,
                active=active,
            )
        if callable(member):
            function = getattr(member, "__func__", member)
            if getattr(function, "__code__", None) is None:
                callable_impl = getattr(type(member), "__call__", None)
                if getattr(callable_impl, "__code__", None) is not None:
                    function = callable_impl
            code = getattr(function, "__code__", None)
            if code is None:
                return {
                    "kind": "native_callable_reference",
                    "module": getattr(member, "__module__", None),
                    "qualname": getattr(member, "__qualname__", None),
                    "type": (f"{type(member).__module__}.{type(member).__qualname__}"),
                }
            return {
                "kind": "python_callable_reference",
                "manifest": callable_manifest(
                    member,
                    name=name,
                    bind_owner=True,
                    bind_referenced_globals=False,
                    active=active,
                ),
            }
        return state_value_manifest(
            member,
            name=name,
            active=active,
        )

    def state_value_manifest(
        state_value: Any,
        *,
        name: str,
        active: frozenset[int],
    ) -> Any:
        forbidden_alias = forbidden_prebound_alias(state_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        if state_value is None or isinstance(
            state_value,
            (bool, int, str),
        ):
            return state_value
        if isinstance(state_value, float):
            if math.isfinite(state_value):
                return state_value
            return {
                "kind": "nonfinite_float",
                "value": (
                    "nan"
                    if math.isnan(state_value)
                    else (
                        "positive_infinity" if state_value > 0 else "negative_infinity"
                    )
                ),
            }
        if isinstance(state_value, bytes):
            return {
                "kind": "bytes",
                "length": len(state_value),
                "sha256": hashlib.sha256(state_value).hexdigest(),
            }
        if isinstance(state_value, os.PathLike):
            return {
                "kind": "path",
                "value": os.fspath(state_value),
            }
        if isinstance(state_value, ModuleType):
            return runtime_module_manifest(
                state_value,
                name=name,
                referenced_names=frozenset(),
                active=active,
            )
        if isinstance(state_value, type):
            return runtime_class_manifest(
                state_value,
                name=name,
                active=active,
            )
        torch = _torch()
        if isinstance(state_value, torch.Tensor):
            return {
                "kind": "tensor",
                "sha256": legacy_model_fold._tensor_sha256(state_value),
            }
        if isinstance(state_value, Mapping):
            mapping_identity = id(state_value)
            mapping_type = (
                f"{type(state_value).__module__}.{type(state_value).__qualname__}"
            )
            reference = state_graph_reference(
                state_value,
                name=name,
                active=active,
            )
            if reference is not None:
                return reference
            mapping_active = active | {mapping_identity}
            if all(isinstance(key, str) for key in state_value):
                return {
                    key: mapping_member_manifest(
                        state_value[key],
                        name=f"{name}.{key}",
                        active=mapping_active,
                    )
                    for key in sorted(state_value)
                }
            entries = [
                {
                    "key": mapping_member_manifest(
                        key,
                        name=f"{name}.<mapping-key>",
                        active=mapping_active,
                    ),
                    "value": mapping_member_manifest(
                        child,
                        name=f"{name}.<mapping-value>",
                        active=mapping_active,
                    ),
                }
                for key, child in state_value.items()
            ]
            return {
                "kind": "mapping",
                "type": mapping_type,
                "entries": sorted(entries, key=_canonical_json),
            }
        if isinstance(state_value, (list, tuple)):
            sequence_identity = id(state_value)
            reference = state_graph_reference(
                state_value,
                name=name,
                active=active,
            )
            if reference is not None:
                return reference
            sequence_active = active | {sequence_identity}
            return [
                state_value_manifest(
                    child,
                    name=f"{name}[{index}]",
                    active=sequence_active,
                )
                for index, child in enumerate(state_value)
            ]
        if isinstance(state_value, (set, frozenset)):
            set_identity = id(state_value)
            reference = state_graph_reference(
                state_value,
                name=name,
                active=active,
            )
            if reference is not None:
                return reference
            set_active = active | {set_identity}
            children = [
                state_value_manifest(
                    child,
                    name=f"{name}[]",
                    active=set_active,
                )
                for child in state_value
            ]
            return {
                "kind": type(state_value).__name__,
                "values": sorted(children, key=_canonical_json),
            }
        if callable(state_value):
            function = getattr(state_value, "__func__", state_value)
            callable_impl = getattr(type(state_value), "__call__", None)
            if (
                getattr(function, "__code__", None) is None
                and getattr(callable_impl, "__code__", None) is None
            ):
                return {
                    "kind": "native_callable",
                    "module": getattr(state_value, "__module__", None),
                    "qualname": getattr(state_value, "__qualname__", None),
                    "type": (
                        f"{type(state_value).__module__}."
                        f"{type(state_value).__qualname__}"
                    ),
                }
            return {
                "kind": "callable",
                "manifest": callable_manifest(
                    state_value,
                    name=name,
                    bind_owner=True,
                    active=active,
                ),
            }
        as_dict = getattr(state_value, "as_dict", None)
        if callable(as_dict):
            return {
                "kind": "authenticated_object",
                "type": (
                    f"{type(state_value).__module__}.{type(state_value).__qualname__}"
                ),
                "value": state_value_manifest(
                    as_dict(),
                    name=f"{name}.as_dict",
                    active=active,
                ),
            }
        return {
            "kind": "instance",
            "state": instance_state_manifest(
                state_value,
                name=name,
                active=active,
            ),
        }

    def runtime_class_manifest(
        class_value: type,
        *,
        name: str,
        bind_python_methods: bool = True,
        bind_python_method_globals: bool = True,
        active: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        forbidden_alias = forbidden_prebound_alias(class_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        module_name = getattr(class_value, "__module__", None)
        qualname = getattr(class_value, "__qualname__", None)
        module = (
            None if not isinstance(module_name, str) else sys.modules.get(module_name)
        )
        raw_path = getattr(module, "__file__", None)
        if not isinstance(module_name, str) or not isinstance(qualname, str):
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} class has no exact runtime identity."
            )
        class_identity = id(class_value)
        base = {
            "kind": "class",
            "module": module_name,
            "qualname": qualname,
            "source_sha256": (
                None
                if not isinstance(raw_path, str)
                else _file_sha256(Path(raw_path).resolve())
            ),
        }
        if class_identity in active:
            return {**base, "recursive_reference": True}
        if not bind_python_methods:
            return {**base, "python_methods": {}, "reference_only": True}
        class_active = active | {class_identity}
        members = {}
        for member_name, raw_member in sorted(vars(class_value).items()):
            member = (
                raw_member.__func__
                if isinstance(raw_member, (staticmethod, classmethod))
                else raw_member
            )
            if getattr(member, "__code__", None) is not None:
                members[member_name] = callable_manifest(
                    member,
                    name=f"{name}.{member_name}",
                    bind_referenced_globals=bind_python_method_globals,
                    active=class_active,
                )
        manifest = {
            **base,
            "python_methods": members,
        }
        if not bind_python_method_globals:
            manifest["external_globals_bound_by_source"] = True
        return manifest

    def runtime_module_manifest(
        module_value: ModuleType,
        *,
        name: str,
        referenced_names: frozenset[str],
        bind_referenced_attributes: bool = True,
        active: frozenset[int],
    ) -> dict[str, Any]:
        forbidden_alias = forbidden_prebound_alias(module_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        module_name = getattr(module_value, "__name__", None)
        raw_path = getattr(module_value, "__file__", None)
        if not isinstance(module_name, str):
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} module has no exact runtime identity."
            )
        module_identity = id(module_value)
        base = {
            "kind": "module",
            "module": module_name,
            "source_sha256": (
                None
                if not isinstance(raw_path, str)
                else _file_sha256(Path(raw_path).resolve())
            ),
        }
        if module_identity in active:
            return {**base, "recursive_reference": True}
        module_active = active | {module_identity}
        attributes = (
            {
                attribute: runtime_global_manifest(
                    getattr(module_value, attribute),
                    name=f"{name}.{attribute}",
                    referenced_names=referenced_names,
                    active=module_active,
                )
                for attribute in sorted(referenced_names)
                if hasattr(module_value, attribute)
            }
            if bind_referenced_attributes
            else {}
        )
        manifest = {
            **base,
            "referenced_attributes": attributes,
        }
        if not bind_referenced_attributes:
            manifest["external_attributes_bound_by_source"] = True
        return manifest

    def runtime_global_manifest(
        global_value: Any,
        *,
        name: str,
        referenced_names: frozenset[str],
        active: frozenset[int],
    ) -> Any:
        forbidden_alias = forbidden_prebound_alias(global_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        if isinstance(global_value, ModuleType):
            raw_path = getattr(global_value, "__file__", None)
            external_source = not first_party_source(raw_path)
            return runtime_module_manifest(
                global_value,
                name=name,
                referenced_names=referenced_names,
                bind_referenced_attributes=not external_source,
                active=active,
            )
        if isinstance(global_value, type):
            module_name = getattr(global_value, "__module__", None)
            module = (
                None
                if not isinstance(module_name, str)
                else sys.modules.get(module_name)
            )
            external_source = not first_party_source(getattr(module, "__file__", None))
            return runtime_class_manifest(
                global_value,
                name=name,
                bind_python_method_globals=not external_source,
                active=active,
            )
        if callable(global_value):
            function = getattr(global_value, "__func__", global_value)
            if getattr(function, "__code__", None) is not None:
                module_name = getattr(function, "__module__", None)
                module = (
                    None
                    if not isinstance(module_name, str)
                    else sys.modules.get(module_name)
                )
                external_source = not first_party_source(
                    getattr(module, "__file__", None)
                )
                return {
                    "kind": "python_callable",
                    "manifest": callable_manifest(
                        global_value,
                        name=name,
                        bind_owner=True,
                        bind_referenced_globals=not external_source,
                        active=active,
                    ),
                }
            return {
                "kind": "native_callable",
                "module": getattr(global_value, "__module__", None),
                "qualname": getattr(global_value, "__qualname__", None),
                "type": (
                    f"{type(global_value).__module__}.{type(global_value).__qualname__}"
                ),
            }
        return state_value_manifest(
            global_value,
            name=name,
            active=active,
        )

    def callable_manifest(
        callable_value: Any,
        *,
        name: str,
        bind_owner: bool = False,
        bind_referenced_globals: bool = True,
        active: frozenset[int] = frozenset(),
    ) -> dict[str, Any]:
        forbidden_alias = forbidden_prebound_alias(callable_value)
        if forbidden_alias is not None:
            category, symbol = forbidden_alias
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} contains forbidden pre-bound {category} alias {symbol}."
            )
        function = getattr(callable_value, "__func__", callable_value)
        callable_object_owner = None
        if not callable(function) or getattr(function, "__code__", None) is None:
            callable_object_owner = callable_value
            function = getattr(type(callable_value), "__call__", None)
        code = getattr(function, "__code__", None)
        module_name = getattr(function, "__module__", None)
        qualname = getattr(function, "__qualname__", None)
        module = (
            None if not isinstance(module_name, str) else sys.modules.get(module_name)
        )
        raw_path = getattr(module, "__file__", None)
        if (
            code is None
            or not isinstance(module_name, str)
            or not isinstance(qualname, str)
            or not isinstance(raw_path, str)
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} has no exact Python callable/source identity."
            )
        path = Path(raw_path).resolve()
        function_identity = id(function)
        base_manifest = {
            "module": module_name,
            "qualname": qualname,
            "code_sha256": _stable_python_code_sha256(code),
            "source_filename": path.name,
            "source_sha256": _file_sha256(path),
        }
        if function_identity in active:
            return {**base_manifest, "recursive_reference": True}
        callable_active = active | {function_identity}
        defaults = getattr(function, "__defaults__", None)
        kwdefaults = getattr(function, "__kwdefaults__", None)
        closure = getattr(function, "__closure__", None)
        freevars = tuple(getattr(code, "co_freevars", ()))
        if closure is not None and len(closure) != len(freevars):
            raise Lfm2CpcfV5ModelCompilerError(
                f"{name} closure cannot be authenticated."
            )
        global_namespace = getattr(function, "__globals__", {})
        referenced_names = frozenset(str(value) for value in code.co_names)
        manifest = {
            **base_manifest,
            "defaults": state_value_manifest(
                defaults,
                name=f"{name}.__defaults__",
                active=callable_active,
            ),
            "kwdefaults": state_value_manifest(
                kwdefaults,
                name=f"{name}.__kwdefaults__",
                active=callable_active,
            ),
            "closure": (
                {}
                if closure is None
                else {
                    freevar: state_value_manifest(
                        cell.cell_contents,
                        name=f"{name}.__closure__.{freevar}",
                        active=callable_active,
                    )
                    for freevar, cell in zip(
                        freevars,
                        closure,
                        strict=True,
                    )
                }
            ),
            "referenced_globals": (
                {
                    global_name: runtime_global_manifest(
                        global_namespace[global_name],
                        name=f"{name}.__globals__.{global_name}",
                        referenced_names=referenced_names,
                        active=callable_active,
                    )
                    for global_name in sorted(referenced_names)
                    if global_name in global_namespace
                }
                if bind_referenced_globals
                else {}
            ),
        }
        if not bind_referenced_globals:
            manifest["external_globals_bound_by_source"] = True
        owner = getattr(callable_value, "__self__", None)
        if owner is None:
            owner = callable_object_owner
        if bind_owner and owner is not None and not isinstance(owner, type):
            manifest["bound_owner_state"] = instance_state_manifest(
                owner,
                name=f"{name}.__self__",
                active=callable_active,
            )
        return manifest

    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    body: dict[str, Any] = {
        "kind": kind,
        "provided": True,
        "type": type_name,
    }
    method_name = {
        "candidate_selector": "select",
        "candidate_replay": "evaluate",
        "numerical_frontier": "generate",
    }.get(kind)
    if method_name is None:
        raise Lfm2CpcfV5ModelCompilerError(
            f"Unknown optional runtime contract kind {kind!r}."
        )
    method = getattr(value, method_name, None)
    if not callable(method):
        raise Lfm2CpcfV5ModelCompilerError(
            f"{kind} has no exact {method_name} callable."
        )
    body[f"{method_name}_callable"] = callable_manifest(
        method,
        name=f"{kind}.{method_name}",
    )
    scheduler = getattr(value, "scheduler", None)
    body["instance_state"] = instance_state_manifest(
        value,
        name=kind,
        excluded=frozenset({"scheduler"}),
    )
    if scheduler is not None:
        policy = getattr(scheduler, "policy", None)
        replicas = getattr(scheduler, "replicas", None)
        if not callable(getattr(policy, "as_dict", None)) or not isinstance(
            replicas,
            tuple,
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Candidate replay scheduler has no canonical live contract."
            )
        body["scheduler_policy"] = policy.as_dict()
        body["scheduler_replicas"] = [replica.as_dict() for replica in replicas]
        body["scheduler_type"] = (
            f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
        )
        body["scheduler_execute_callable"] = callable_manifest(
            getattr(scheduler, "execute", None),
            name="candidate replay scheduler.execute",
        )
        body["scheduler_instance_state"] = instance_state_manifest(
            scheduler,
            name="candidate replay scheduler",
            excluded=frozenset(
                {
                    "policy",
                    "replicas",
                    "gpu_worker",
                    "cpu_parity_worker",
                    "device_receipt_probe",
                    "installed_state_probe",
                    "cpu_installed_state_probe",
                    "host_buffer_factory",
                }
            ),
        )
        scheduler_callables = {}
        for field in (
            "gpu_worker",
            "cpu_parity_worker",
            "device_receipt_probe",
            "installed_state_probe",
            "cpu_installed_state_probe",
            "host_buffer_factory",
        ):
            scheduler_callables[field] = callable_manifest(
                getattr(scheduler, field, None),
                name=f"candidate replay scheduler.{field}",
                bind_owner=True,
            )
        body["scheduler_runtime_callables"] = scheduler_callables
    return _sha(body)


def _trace_resume_bindings(trace_adapter: Any) -> tuple[str, str, str]:
    verify = getattr(trace_adapter, "verify", None)
    if not callable(verify):
        raise Lfm2CpcfV5ModelCompilerError(
            "Resume context requires a verified tokenizer trace adapter."
        )
    verify()
    trace_sha = _digest(
        getattr(trace_adapter, "evidence_sha256", None),
        "trace_adapter.evidence_sha256",
    )
    tokenizer_sha = _digest(
        getattr(
            trace_adapter,
            "target_tokenizer_package_fingerprint",
            None,
        ),
        "trace_adapter.target_tokenizer_package_fingerprint",
    )
    bridge_sha = _digest(
        getattr(trace_adapter, "bridge_artifact_sha256", None),
        "trace_adapter.bridge_artifact_sha256",
    )
    verify()
    return trace_sha, tokenizer_sha, bridge_sha


def _build_live_resume_context(
    *,
    source_before: Any,
    target_before: Any,
    target_topology_sha256: str,
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    trace_adapter: Any,
    source_tied_table: Any,
    blank_report_sha256: str,
    planned_target_unique_parameters: int,
    runtime_policy: Lfm2CpcfV5RuntimePolicy,
    compiler_implementation: Mapping[str, Any],
    candidate_replay_evaluator: Any,
    candidate_selector: Any,
    numerical_frontier_adapter: Any,
    bootstrap_authentication_sha256: str,
    final_norm_authentication_sha256: str,
) -> Mapping[str, Any]:
    trace_sha, tokenizer_sha, token_bridge_sha = _trace_resume_bindings(trace_adapter)
    selector_sha = _optional_runtime_contract_sha256(
        candidate_selector,
        kind="candidate_selector",
    )
    numerical_sha = _optional_runtime_contract_sha256(
        numerical_frontier_adapter,
        kind="numerical_frontier",
    )
    return build_lfm2_cpcf_v5_resume_context(
        source_model_state_sha256=source_before.state_fingerprint,
        source_model_inventory_sha256=source_before.full_inventory_sha256,
        cache_sha256=catalog.cache_key,
        selective_catalog_sha256=catalog.catalog_sha256,
        selective_source_sha256=catalog.source_sha256,
        selective_provider_sha256=catalog.provider_sha256,
        trace_sha256=trace_sha,
        tokenizer_sha256=tokenizer_sha,
        token_bridge_sha256=token_bridge_sha,
        source_tied_table_sha256=legacy_model_fold._tensor_sha256(source_tied_table),
        blank_target_report_sha256=blank_report_sha256,
        blank_target_state_sha256=target_before.state_fingerprint,
        target_topology_sha256=target_topology_sha256,
        planned_target_unique_parameters=planned_target_unique_parameters,
        runtime_policy_sha256=_sha(runtime_policy.as_dict()),
        physical_rss_policy_sha256=(
            build_lfm2_cpcf_v5_physical_rss_policy_sha256(
                full_phase_tree_logical_bytes=(
                    runtime_policy.full_phase_tree_logical_bytes
                ),
                minimum_reduction_fraction=(
                    runtime_policy.minimum_physical_rss_reduction_fraction
                ),
                maximum_increase_bytes=(
                    runtime_policy.maximum_physical_rss_increase_bytes
                ),
                maximum_process_rss_bytes=(runtime_policy.maximum_process_rss_bytes),
            )
        ),
        compiler_implementation_sha256=compiler_implementation["implementation_sha256"],
        resume_implementation_sha256=lfm2_cpcf_v5_resume_implementation()[
            "implementation_sha256"
        ],
        selector_contract_sha256=selector_sha,
        numerical_contract_sha256=numerical_sha,
        gpu_contract_sha256=_optional_runtime_contract_sha256(
            candidate_replay_evaluator,
            kind="candidate_replay",
        ),
        bootstrap_authentication_sha256=(bootstrap_authentication_sha256),
        final_norm_authentication_sha256=(final_norm_authentication_sha256),
    )


def _completed_resume_residency_receipt(
    catalog: Lfm2CpcfSelectiveEvidenceCatalog,
    runtime_policy: Lfm2CpcfV5RuntimePolicy,
) -> Mapping[str, Any]:
    body = {
        "schema": ("neural-forge-dynamic-evidence-residency-window-v1"),
        "catalog_sha256": catalog.dynamic_catalog.catalog_sha256,
        "source_sha256": catalog.source_sha256,
        "provider_sha256": catalog.provider_sha256,
        "state": "complete",
        "start_layer_index": catalog.layer_count,
        "current_layer_index": None,
        "occupied_layer_indices": [],
        "max_resident_layers": 2,
        "maximum_resident_bytes": (runtime_policy.maximum_resident_evidence_bytes),
        "resident_bytes": 0,
        "prefetch_distance_layers": 1,
        "peak_occupied_layers": 0,
        "peak_resident_bytes": 0,
        "requested_layer_indices": [],
        "promoted_layer_indices": [],
        "released_layer_indices": [],
        "canonical_layer_order_required": True,
        "parent_forward_calls": 0,
        "training_steps": 0,
        "compiler_fingerprint_binding": (
            "authenticated_by_lfm2_cpcf_v5_resume_context"
        ),
        "resume_schema_binding": LFM2_CPCF_V5_RESUME_SCHEMA,
        "standalone_orchestration_only": True,
    }
    return MappingProxyType({**body, "snapshot_sha256": canonical_sha256(body)})


def _has_exact_short_conv_layer_coverage(
    short_conv_layer_indices: Sequence[int],
    receipt_layer_indices: Sequence[int],
) -> bool:
    """Require one ordered receipt for every short-conv layer and no others."""

    return bool(short_conv_layer_indices) and tuple(receipt_layer_indices) == tuple(
        short_conv_layer_indices
    )


def compile_lfm2_cpcf_v5_selective_model_fold(
    source_model: Any,
    target_model: Any,
    *,
    trace_adapter: Any,
    evidence_provider: Lfm2CpcfSelectiveEvidenceProvider,
    source_tied_table: Any,
    blank_target_report: Mapping[str, Any],
    planned_target_unique_parameters: int,
    runtime_policy: Lfm2CpcfV5RuntimePolicy,
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None = None,
    candidate_selector: Lfm2CpcfShortConvV5CandidateSelector | None = None,
    numerical_frontier_adapter: (
        Lfm2CpcfShortConvNumericalFrontierAdapter | None
    ) = None,
    resume_state: Lfm2CpcfV5ResumeState | None = None,
    resume_coordinator: Lfm2CpcfV5ResumeCoordinator | None = None,
) -> legacy_model_fold.Lfm2CpcfCompiledModelFold:
    """Compile one dense blank using bounded selective evidence.

    An optional v5 coordinator persists k=0 and every accepted layer boundary.
    Resume accepts only an admitted v5 partial frontier restored into the
    caller-created exact fresh blank; v4 or untyped continuation objects fail
    closed.
    """

    if resume_state is not None and not isinstance(
        resume_state,
        Lfm2CpcfV5ResumeState,
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "V5 selective resume requires an admitted typed v5 resume state; "
            "refusing a v4 or untyped resume artifact."
        )
    if resume_coordinator is not None and not isinstance(
        resume_coordinator,
        Lfm2CpcfV5ResumeCoordinator,
    ):
        raise TypeError("resume_coordinator must be Lfm2CpcfV5ResumeCoordinator.")
    if resume_state is not None:
        if resume_coordinator is None:
            resume_coordinator = resume_state.coordinator
        elif resume_coordinator.store.root != resume_state.coordinator.store.root:
            raise Lfm2CpcfV5ModelCompilerError(
                "Resume state and coordinator use different stores."
            )
    if not isinstance(runtime_policy, Lfm2CpcfV5RuntimePolicy):
        raise TypeError("runtime_policy must be Lfm2CpcfV5RuntimePolicy.")
    if candidate_selector is not None and not callable(
        getattr(candidate_selector, "select", None)
    ):
        raise TypeError("candidate_selector must implement select(request).")
    if numerical_frontier_adapter is not None and not callable(
        getattr(numerical_frontier_adapter, "generate", None)
    ):
        raise TypeError("numerical_frontier_adapter must implement generate(request).")
    real_v5_adapter_contract = _real_v5_numerical_adapter_contract(
        numerical_frontier_adapter
    )
    if (
        runtime_policy.require_short_conv_v5_real_numerical_frontier_adapter
        and real_v5_adapter_contract is None
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "The runtime policy requires the concrete authenticated "
            "Lfm2CpcfShortConvV5NumericalFrontierAdapter; a missing, generic, "
            "toy, subclassed, or unauthenticated adapter is not admissible."
        )
    if runtime_policy.require_short_conv_v5_endpoint_first_selector and not isinstance(
        candidate_selector,
        Lfm2CpcfV5EndpointFirstCandidateSelector,
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "The required source-bound short-conv v5 endpoint-first selector "
            "needs Lfm2CpcfV5EndpointFirstCandidateSelector at the "
            "pre-install candidate frontier."
        )
    if not isinstance(
        evidence_provider,
        Lfm2CpcfSelectiveEvidenceProvider,
    ):
        raise TypeError("evidence_provider must be Lfm2CpcfSelectiveEvidenceProvider.")
    if candidate_replay_evaluator is not None and not callable(
        getattr(candidate_replay_evaluator, "evaluate", None)
    ):
        raise TypeError("candidate_replay_evaluator must implement evaluate().")
    compiler_implementation = dict(lfm2_cpcf_v5_compiler_implementation())
    catalog = evidence_provider.catalog
    if not isinstance(catalog, Lfm2CpcfSelectiveEvidenceCatalog):
        raise Lfm2CpcfV5ModelCompilerError(
            "Selective provider has no authenticated LFM2 catalog."
        )
    authenticated_full_phase_tree_bytes = catalog.final_norm_logical_bytes + sum(
        reference.logical_bytes for reference in catalog.dynamic_catalog.references
    )
    if (
        runtime_policy.full_phase_tree_logical_bytes
        != authenticated_full_phase_tree_bytes
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Configured full phase-tree bytes differ from the authenticated "
            "selective evidence catalog."
        )
    if (
        catalog.final_norm_logical_bytes
        > runtime_policy.maximum_resident_evidence_bytes
    ):
        raise Lfm2CpcfV5ModelCompilerError(
            "Final-normalization evidence exceeds the configured resident "
            "evidence budget."
        )
    rss_guard = _PhysicalRssGuard(
        maximum_increase_bytes=(runtime_policy.maximum_physical_rss_increase_bytes),
        maximum_process_rss_bytes=runtime_policy.maximum_process_rss_bytes,
        full_phase_tree_logical_bytes=(runtime_policy.full_phase_tree_logical_bytes),
        minimum_reduction_fraction=(
            runtime_policy.minimum_physical_rss_reduction_fraction
        ),
        sample_interval_seconds=runtime_policy.rss_sample_interval_seconds,
    )
    source_before = legacy_model_fold._snapshot_model_auth(source_model)
    target_before = legacy_model_fold._snapshot_model(target_model)
    provider_closed = False
    bootstrap_payload: Lfm2CpcfSelectiveLayerPayload | None = None
    final_payload: Lfm2CpcfSelectiveFinalNormPayload | None = None
    resume_final_phases: tuple[Lfm2CpcfTargetPhase, ...] | None = None
    terminal_action_capture: Lfm2CpcfV5TerminalActionCapture | None = None
    terminal_action_report: Mapping[str, Any] | None = None
    terminal_resume_publication_receipt: Mapping[str, Any] | None = None
    admitted_prefixes: list[dict[str, Any]] = []
    suppressed_terminal_preintegration_prefixes: list[int] = []
    resume_input_admission_chain_proof: Mapping[str, Any] | None = None
    executed_layer_indices: list[int] = []
    try:
        rss_guard.checkpoint("request_authenticated")
        target_inventory_before = legacy_model_fold.independent_parameter_inventory(
            target_model
        )
        target_count = int(target_inventory_before["unique_parameters"])
        planned = _positive_int(
            planned_target_unique_parameters,
            "planned_target_unique_parameters",
        )
        if target_count != planned:
            raise Lfm2CpcfV5ModelCompilerError(
                "Live target count differs from the planned dense blank."
            )
        blank = legacy_model_fold._blank_receipt(
            blank_target_report,
            target_unique_parameters=target_count,
        )
        legacy_model_fold._assert_target_is_zero_blank(target_model)
        source_layers = tuple(
            legacy_model_fold._model_core(
                source_model,
                "source_model",
            ).layers
        )
        target_layers = tuple(
            legacy_model_fold._model_core(
                target_model,
                "target_model",
            ).layers
        )
        if (
            not source_layers
            or len(source_layers) != len(target_layers)
            or len(source_layers) != catalog.layer_count
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Source, target, and selective catalog depth differ."
            )
        parameter_receipt = legacy_model_fold.audit_lfm2_cpcf_parameter_compaction(
            source_model,
            target_model,
            planned_target_unique_parameters=planned,
            promotion_compression_ratio=(
                runtime_policy.model_fold_policy.promotion_compression_ratio
            ),
        )
        if (
            runtime_policy.model_fold_policy.require_promotion_ratio
            and not parameter_receipt["promotion_ratio_met"]
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Target does not meet the required promotion ratio."
            )
        target_topology = build_lfm2_cpcf_target_topology_snapshot(
            prefix_model=target_before.tensor_values,
            layer_count=catalog.layer_count,
            snapshot_metadata=(
                legacy_model_fold._snapshot_metadata_plain(target_before)
            ),
        )
        resume_used = resume_state is not None
        if resume_used:
            final_payload = evidence_provider.load_final_norm_payload()
            final_norm_authentication_sha256 = (
                evidence_provider.authenticate_final_norm_payload(final_payload)
            )
            phases = build_lfm2_cpcf_v5_target_phase_shells(
                trace_adapter,
                catalog,
                None,
                final_norm_payload=final_payload,
            )
            resume_final_phases = materialize_lfm2_cpcf_v5_final_norm_phases(
                phases,
                catalog,
                final_payload,
            )
            evidence_provider.release_final_norm_payload(final_payload)
            final_payload = None
            bootstrap_authentication_sha256 = _digest(
                resume_state.frontier.context["bootstrap_authentication_sha256"],
                "resume bootstrap authentication SHA-256",
            )
        else:
            bootstrap_payload, bootstrap_authentication = (
                _authenticated_bootstrap_payload(
                    evidence_provider,
                    layer_index=0,
                )
            )
            bootstrap_authentication_sha256 = (
                bootstrap_authentication.authentication_sha256
            )
            phases = build_lfm2_cpcf_v5_target_phase_shells(
                trace_adapter,
                catalog,
                bootstrap_payload,
            )
            evidence_provider.release_layer_payload(
                catalog.dynamic_catalog.references[0],
                bootstrap_payload,
            )
            bootstrap_payload = None
        gc.collect()
        rss_guard.checkpoint("tensor_free_phase_shells_built")
        boundary_compiler = _SelectiveBoundaryCompiler(
            phases,
            catalog,
            source_tied_table,
            target_hidden_width=int(target_model.config.hidden_size),
            model_policy=runtime_policy.model_fold_policy,
        )
        if resume_used:
            if resume_final_phases is None:
                raise Lfm2CpcfV5ModelCompilerError(
                    "Resume lost its authenticated final-normalization "
                    "rehydration phases."
                )
            interface = boundary_compiler.fit_interface(
                resume_final_phases,
                target_norm_epsilon=(
                    legacy_model_fold._final_norm_epsilon(target_model)
                ),
            )
            del resume_final_phases
            resume_final_phases = None
            gc.collect()
            rss_guard.checkpoint("resume_interface_recomputed_from_final_norm")
        else:
            final_payload = evidence_provider.load_final_norm_payload()
            final_norm_authentication_sha256 = (
                evidence_provider.authenticate_final_norm_payload(final_payload)
            )
            final_phases = materialize_lfm2_cpcf_v5_final_norm_phases(
                phases,
                catalog,
                final_payload,
            )
            interface = boundary_compiler.fit_interface(
                final_phases,
                target_norm_epsilon=(
                    legacy_model_fold._final_norm_epsilon(target_model)
                ),
            )
            del final_phases
            evidence_provider.release_final_norm_payload(final_payload)
            final_payload = None
            gc.collect()
            rss_guard.checkpoint("interface_fitted_final_norm_released")
        resume_context = _build_live_resume_context(
            source_before=source_before,
            target_before=target_before,
            target_topology_sha256=target_topology["topology_sha256"],
            catalog=catalog,
            trace_adapter=trace_adapter,
            source_tied_table=source_tied_table,
            blank_report_sha256=blank["report_sha256"],
            planned_target_unique_parameters=planned,
            runtime_policy=runtime_policy,
            compiler_implementation=compiler_implementation,
            candidate_replay_evaluator=candidate_replay_evaluator,
            candidate_selector=candidate_selector,
            numerical_frontier_adapter=numerical_frontier_adapter,
            bootstrap_authentication_sha256=(bootstrap_authentication_sha256),
            final_norm_authentication_sha256=(final_norm_authentication_sha256),
        )

        if resume_used:
            restored = resume_coordinator.restore(
                resume_state,
                target_model=target_model,
                phases=phases,
                boundary_compiler=boundary_compiler,
                expected_context=resume_context,
                expected_blank_state_sha256=target_before.state_fingerprint,
            )
            state_bank = restored.state_bank
            completed_receipts = list(restored.completed_layer_receipts)
            start_layer_index = restored.frontier.next_layer_index
            interface = boundary_compiler.interface
            installed_state_sha256 = restored.prefix_install_report[
                "installed_state_sha256_after"
            ]
            resume_input_admission_chain_proof = (
                build_lfm2_cpcf_v5_admission_chain_proof(
                    resume_coordinator,
                    restored.admission["admission_sha256"],
                )
            )
        else:
            interface_install = install_lfm2_cpcf_tied_interface(
                target_model,
                interface,
            )
            neutral_norm_install = install_lfm2_cpcf_neutral_norms(
                target_model,
                attention_qk_norm_weights=(
                    legacy_model_fold._preserved_qk_norm_weights(
                        source_model,
                        target_model,
                    )
                ),
            )
            state_bank = seed_lfm2_cpcf_phase_states(
                target_model,
                phases,
            )
            completed_receipts = []
            start_layer_index = 0
            installed_state_sha256 = None
        if resume_used:
            interface_install = install_lfm2_cpcf_tied_interface(
                target_model,
                interface,
            )
            neutral_norm_install = install_lfm2_cpcf_neutral_norms(
                target_model,
                attention_qk_norm_weights=(
                    legacy_model_fold._preserved_qk_norm_weights(
                        source_model,
                        target_model,
                    )
                ),
            )
            if (
                legacy_model_fold.fingerprint_model_state(target_model)
                != installed_state_sha256
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "Reauthenticating the restored interface and neutral "
                    "norms changed the installed prefix."
                )

        def persist_prefix_before_rss(
            next_layer_index: int,
            *,
            label: str,
            publish_resume_prefix: bool = True,
        ) -> None:
            if not publish_resume_prefix:
                suppressed_terminal_preintegration_prefixes.append(next_layer_index)
            if resume_coordinator is None:
                rss_guard.checkpoint(label)
                return
            if (
                _optional_runtime_contract_sha256(
                    candidate_selector,
                    kind="candidate_selector",
                )
                != resume_context["selector_contract_sha256"]
                or _optional_runtime_contract_sha256(
                    candidate_replay_evaluator,
                    kind="candidate_replay",
                )
                != resume_context["gpu_contract_sha256"]
                or _optional_runtime_contract_sha256(
                    numerical_frontier_adapter,
                    kind="numerical_frontier",
                )
                != resume_context["numerical_contract_sha256"]
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "Selector, replay-evaluator, or numerical-frontier "
                    "runtime state changed before resumable prefix "
                    "publication."
                )
            if not publish_resume_prefix:
                rss_guard.checkpoint(label)
                return
            frontier = Lfm2CpcfV5BoundaryFrontier.capture(
                boundary_compiler,
                context=resume_context,
                next_layer_index=next_layer_index,
                expected_layer_count=catalog.layer_count,
            )
            event = Lfm2CpcfV5ProgressEvent.capture(
                frontier=frontier,
                incoming_state_bank=state_bank,
                completed_layer_receipts=completed_receipts,
                target_model=target_model,
            )
            pending = resume_coordinator.publish_pending(
                event,
                rss_checkpoint_expectation=(
                    rss_guard.next_checkpoint_expectation(label)
                ),
            )
            del event, frontier
            gc.collect()
            try:
                rss_receipt = rss_guard.checkpoint(label)
            except Lfm2CpcfV5PhysicalRssError as rss_error:
                resume_coordinator.record_rss_rejection(
                    pending,
                    rss_violation_report=rss_error.report,
                )
                raise
            admission = resume_coordinator.admit(
                pending,
                rss_checkpoint_receipt=rss_receipt,
            )
            admission_chain_proof = build_lfm2_cpcf_v5_admission_chain_proof(
                resume_coordinator,
                admission["admission_sha256"],
            )
            admitted_prefixes.append(
                {
                    "next_layer_index": next_layer_index,
                    "frontier_reference": (pending.frontier_reference.as_dict()),
                    "checkpoint_reference": (pending.checkpoint.reference.as_dict()),
                    "frontier_sha256": pending.frontier_sha256,
                    "admission_sha256": admission["admission_sha256"],
                    "admission_chain_proof": dict(admission_chain_proof),
                    "admission_chain_proof_sha256": (
                        admission_chain_proof["proof_sha256"]
                    ),
                }
            )

        if not resume_used:
            persist_prefix_before_rss(
                0,
                label="prefix_000000_persisted_before_residency",
            )
        residency_snapshot: Mapping[str, Any] | None = None
        if start_layer_index == catalog.layer_count:
            residency_snapshot = _completed_resume_residency_receipt(
                catalog,
                runtime_policy,
            )
        else:
            with DynamicEvidenceResidencyWindow(
                catalog.dynamic_catalog,
                evidence_provider,
                maximum_resident_bytes=(runtime_policy.maximum_resident_evidence_bytes),
            ) as residency:
                resident = residency.start(start_layer_index=start_layer_index)
                rss_guard.checkpoint("residency_started")
                for layer_index in range(
                    start_layer_index,
                    catalog.layer_count,
                ):
                    executed_layer_indices.append(layer_index)
                    if resident.layer_index != layer_index or not isinstance(
                        resident.payload,
                        Lfm2CpcfSelectiveLayerPayload,
                    ):
                        raise Lfm2CpcfV5ModelCompilerError(
                            "Residency window promoted the wrong LFM2 layer."
                        )
                    materialized_phases = materialize_lfm2_cpcf_v5_layer_phases(
                        phases,
                        catalog,
                        resident.payload,
                    )
                    layer_bank = rebind_lfm2_cpcf_phase_state_bank(
                        state_bank,
                        materialized_phases,
                    )
                    charts = boundary_compiler.fit_layer(
                        materialized_phases,
                        layer_index=layer_index,
                    )
                    endpoint_sequential_policy = (
                        runtime_policy.resolved_endpoint_sequential_transport_policy(
                            charts.operator_kind
                        )
                    )
                    controller = None
                    if endpoint_sequential_policy is not None:
                        controller = (
                            fit_lfm2_cpcf_post_operator_chart_transport_for_profile(
                                layer_bank,
                                charts,
                                policy=runtime_policy.post_operator_transport_policy,
                            )
                        )
                        if (
                            controller.report.get("validation_rows_read", 0) != 0
                            or controller.report.get("post_layer_chart_same_object")
                            is not True
                            or controller.report.get(
                                "fit_only_closed_form_orthogonal_procrustes"
                            )
                            is not True
                            or controller.report.get("component")
                            != "fit_only_post_operator_chart_transport"
                            or controller.report.get("production_profile_authorized")
                            is not True
                        ):
                            raise Lfm2CpcfV5ModelCompilerError(
                                "The combined profile's post-operator chart "
                                "controller did not prove one frozen FIT-only "
                                "orthogonal transport."
                            )
                        boundary_compiler.replace_latest_layer_charts(
                            charts,
                            controller.boundary_charts,
                            controller_report=controller.report,
                        )
                        charts = controller.boundary_charts
                    result = compile_and_install_lfm2_cpcf_layer(
                        source_model,
                        target_model,
                        layer_index=layer_index,
                        input_bank=layer_bank,
                        boundary_charts=charts,
                        ridge=runtime_policy.model_fold_policy.ridge,
                        short_conv_gate_policy=(
                            runtime_policy.model_fold_policy.resolved_short_conv_policy()
                        ),
                        short_conv_v4_solver_policy=(
                            runtime_policy.model_fold_policy.short_conv_v4_solver_policy
                        ),
                        candidate_replay_evaluator=(candidate_replay_evaluator),
                        candidate_selector=candidate_selector,
                        numerical_frontier_adapter=(numerical_frontier_adapter),
                        attention_sequential_transport_policy=(
                            runtime_policy.model_fold_policy.resolved_attention_sequential_transport_policy(
                                charts.operator_kind
                            )
                        ),
                        composed_qk_attention_policy=(
                            runtime_policy.resolved_composed_qk_attention_policy(
                                charts.operator_kind
                            )
                        ),
                        endpoint_residual_attention_policy=(
                            runtime_policy.resolved_endpoint_residual_attention_policy(
                                charts.operator_kind
                            )
                        ),
                        endpoint_sequential_transport_policy=(
                            endpoint_sequential_policy
                        ),
                        post_operator_transport_controller_report=(
                            None if controller is None else controller.report
                        ),
                    )
                    receipt = legacy_model_fold.Lfm2CpcfCompletedLayerReceipt(
                        result.report
                    )
                    next_bank = rebind_lfm2_cpcf_phase_state_bank(
                        result.next_bank,
                        phases,
                    )
                    if (
                        receipt.input_bank_sha256 != state_bank.state_bank_sha256
                        or receipt.next_bank_sha256 != next_bank.state_bank_sha256
                    ):
                        raise Lfm2CpcfV5ModelCompilerError(
                            "Layer receipt differs from tensor-free state-bank lineage."
                        )
                    if (
                        runtime_policy.terminal_action_residual_policy is not None
                        and layer_index + 1 == catalog.layer_count
                    ):
                        terminal_action_capture = (
                            capture_lfm2_cpcf_v5_terminal_action_inputs(
                                layer_index=layer_index,
                                input_bank=rebind_lfm2_cpcf_phase_state_bank(
                                    layer_bank,
                                    phases,
                                ),
                                final_replay=result.final_replay,
                            )
                        )
                    completed_receipts.append(receipt)
                    state_bank = next_bank
                    resume_publication_decision = (
                        _terminal_resume_prefix_publication_decision(
                            next_layer_index=layer_index + 1,
                            expected_layer_count=catalog.layer_count,
                            terminal_action_enabled=(
                                runtime_policy.terminal_action_residual_policy
                                is not None
                            ),
                        )
                    )
                    terminal_preintegration_final_prefix = (
                        resume_publication_decision["preintegration_final_layer_prefix"]
                        is True
                    )
                    if terminal_preintegration_final_prefix:
                        terminal_resume_publication_receipt = (
                            resume_publication_decision
                        )
                    persist_prefix_before_rss(
                        layer_index + 1,
                        label=(
                            (
                                f"layer_{layer_index:03d}_compiled_terminal_"
                                "pending_without_resume_publication"
                            )
                            if terminal_preintegration_final_prefix
                            else (
                                f"layer_{layer_index:03d}_compiled_"
                                "checkpoint_persisted_before_release"
                            )
                        ),
                        publish_resume_prefix=(
                            resume_publication_decision["publish_resume_prefix"]
                        ),
                    )
                    del (
                        result,
                        receipt,
                        layer_bank,
                        materialized_phases,
                        charts,
                    )
                    gc.collect()
                    following = residency.advance(completed_layer_index=layer_index)
                    rss_guard.checkpoint(f"layer_{layer_index:03d}_released")
                    if layer_index + 1 < catalog.layer_count:
                        if following is None:
                            raise Lfm2CpcfV5ModelCompilerError(
                                "Residency ended before the final layer."
                            )
                        resident = following
                    elif following is not None:
                        raise Lfm2CpcfV5ModelCompilerError(
                            "Residency retained evidence after the final layer."
                        )
                residency_snapshot = residency.snapshot()
        if residency_snapshot is None:
            raise Lfm2CpcfV5ModelCompilerError(
                "Residency window emitted no completion receipt."
            )
        selector_applied_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if "preinstall_candidate_selector_receipt"
            in receipt.report["operator_artifact"]
        ]
        macrocell_selector_applied_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if "macrocell_candidate_selector_receipt"
            in receipt.report["operator_artifact"]
        ]
        numerical_frontier_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if receipt.report["operator_artifact"].get(
                "numerical_frontier_adapter_active"
            )
            is True
        ]
        short_conv_layer_indices = [
            receipt.layer_index
            for receipt in completed_receipts
            if receipt.report.get("operator_kind") == "short_conv"
        ]
        real_v5_layer_proofs = []
        if real_v5_adapter_contract is not None:
            for receipt in completed_receipts:
                if receipt.report.get("operator_kind") != "short_conv":
                    continue
                proof = _prove_real_v5_numerical_layer(
                    receipt,
                    adapter_contract=real_v5_adapter_contract,
                )
                if proof is not None:
                    real_v5_layer_proofs.append(proof)
        real_v5_layer_indices = [
            int(proof["layer_index"]) for proof in real_v5_layer_proofs
        ]
        real_v5_numerical_policy_integrated = _has_exact_short_conv_layer_coverage(
            short_conv_layer_indices,
            real_v5_layer_indices,
        )
        authenticated_whole_macrocell_selection_integrated = (
            _has_exact_short_conv_layer_coverage(
                short_conv_layer_indices,
                macrocell_selector_applied_layer_indices,
            )
        )
        endpoint_prefilter_control_integrated = bool(
            authenticated_whole_macrocell_selection_integrated
            and isinstance(
                candidate_selector,
                Lfm2CpcfV5EndpointFirstCandidateSelector,
            )
        )
        composed_attention_coverage = None
        if runtime_policy.composed_qk_attention_policy is not None:
            composed_attention_coverage = _build_composed_attention_coverage_receipt(
                completed_receipts,
                runtime_policy.composed_qk_attention_policy,
                allow_endpoint_fallback=(
                    runtime_policy.endpoint_residual_attention_policy is not None
                ),
            )
        endpoint_residual_attention_coverage = None
        if runtime_policy.endpoint_residual_attention_policy is not None:
            endpoint_residual_attention_coverage = (
                _build_endpoint_residual_attention_coverage_receipt(
                    completed_receipts,
                    runtime_policy.endpoint_residual_attention_policy,
                )
            )
        endpoint_sequential_transport_coverage = None
        if runtime_policy.endpoint_sequential_transport_policy is not None:
            endpoint_sequential_transport_coverage = (
                _build_endpoint_sequential_transport_coverage_receipt(
                    completed_receipts,
                    runtime_policy.endpoint_sequential_transport_policy,
                )
            )
        if (
            runtime_policy.require_short_conv_v5_authenticated_whole_macrocell_selector
            and not authenticated_whole_macrocell_selection_integrated
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "The required authenticated whole-macrocell selector did not "
                "produce a macrocell candidate-selector receipt for every "
                "short-conv layer."
            )
        if (
            runtime_policy.require_short_conv_v5_real_numerical_frontier_adapter
            and not real_v5_numerical_policy_integrated
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "The required concrete real v5 numerical adapter did not "
                "produce authenticated provenance for every short-conv "
                "layer."
            )
        bundle = boundary_compiler.finalize()
        final_payload = evidence_provider.load_final_norm_payload()
        if (
            evidence_provider.authenticate_final_norm_payload(final_payload)
            != final_norm_authentication_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Final-normalization evidence changed between interface and "
                "final-norm fitting."
            )
        final_phases = materialize_lfm2_cpcf_v5_final_norm_phases(
            phases,
            catalog,
            final_payload,
        )
        final_norm_report = legacy_model_fold._fit_and_install_final_norm(
            target_model,
            phases=final_phases,
            final_state_bank=rebind_lfm2_cpcf_phase_state_bank(
                state_bank,
                final_phases,
            ),
            boundary_bundle=bundle,
            ridge=runtime_policy.model_fold_policy.ridge,
        )
        if runtime_policy.terminal_action_residual_policy is not None:
            if terminal_action_capture is None:
                raise Lfm2CpcfV5ModelCompilerError(
                    "Terminal-action folding requires the final layer to execute "
                    "in this invocation; a completed pre-terminal resume frontier "
                    "does not contain its input replay bank."
                )
            terminal_result = compile_install_lfm2_cpcf_v5_terminal_action(
                target_model=target_model,
                capture=terminal_action_capture,
                incumbent_final_state_bank=state_bank,
                final_phases=final_phases,
                interface=bundle.interface,
                source_tied_table=source_tied_table,
                norm_epsilon=legacy_model_fold._final_norm_epsilon(target_model),
                policy=runtime_policy.terminal_action_residual_policy,
            )
            state_bank = terminal_result.final_state_bank
            terminal_action_report = terminal_result.as_dict()
            del terminal_result, terminal_action_capture
            terminal_action_capture = None
        del final_phases
        evidence_provider.release_final_norm_payload(final_payload)
        final_payload = None
        gc.collect()
        rss_guard.checkpoint("final_norm_fitted_evidence_released")
        target_inventory_after = legacy_model_fold.independent_parameter_inventory(
            target_model
        )
        legacy_model_fold._authenticate_model(source_model, source_before)
        if (
            int(target_inventory_after["unique_parameters"]) != target_count
            or legacy_model_fold._sha(
                legacy_model_fold._model_inventory(
                    target_model,
                    include_tensor_bytes=False,
                )
            )
            != target_before.structure_inventory_sha256
        ):
            raise Lfm2CpcfV5ModelCompilerError(
                "Successful v5 fold changed source or target topology."
            )
        evidence_provider.close()
        provider_closed = True
        gc.collect()
        rss_guard.checkpoint("provider_closed")
        physical_rss_report = rss_guard.close()
        body = {
            "schema": LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
            "artifact_schema": LFM2_CPCF_V5_ARTIFACT_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "resume_supported": True,
            "resume_fail_closed": True,
            "resume_used": resume_used,
            "resume_start_layer_index": start_layer_index,
            "resume_completed_prefix_layer_indices": list(range(start_layer_index)),
            "resume_executed_layer_indices": executed_layer_indices,
            "resume_completed_layers_replayed": False,
            "resume_actual_upstream_child_state_restored": resume_used,
            "resume_prefix_install_report_sha256": (
                None
                if not resume_used
                else restored.prefix_install_report["report_sha256"]
            ),
            "resume_frontier_sha256": (
                None if not resume_used else restored.frontier.frontier_sha256
            ),
            "resume_admission_sha256": (
                None if not resume_used else restored.admission["admission_sha256"]
            ),
            "resume_input_admission_chain_proof": (
                None
                if resume_input_admission_chain_proof is None
                else dict(resume_input_admission_chain_proof)
            ),
            "resume_input_admission_chain_proof_sha256": (
                None
                if resume_input_admission_chain_proof is None
                else resume_input_admission_chain_proof["proof_sha256"]
            ),
            "resume_publication_enabled": resume_coordinator is not None,
            "resume_admitted_prefixes": admitted_prefixes,
            "resume_parent_or_cache_tensor_sidecars": 0,
            "resume_checkpoint_receipt_chain_authenticated": True,
            "compiler_implementation": compiler_implementation,
            "blank_target_report_sha256": blank["report_sha256"],
            "policy": runtime_policy.as_dict(),
            "phase_count": len(phases),
            "phase_sha256": [phase.phase_sha256 for phase in phases],
            "selective_catalog_sha256": catalog.catalog_sha256,
            "selective_source_sha256": catalog.source_sha256,
            "selective_provider_sha256": catalog.provider_sha256,
            "authenticated_full_phase_tree_logical_bytes": (
                authenticated_full_phase_tree_bytes
            ),
            "physical_rss_baseline_bound_to_authenticated_catalog": True,
            "bootstrap_authentication_sha256": (bootstrap_authentication_sha256),
            "final_norm_authentication_sha256": (final_norm_authentication_sha256),
            "residency_receipt": dict(residency_snapshot),
            "physical_rss_receipt": dict(physical_rss_report),
            "boundary_bundle_sha256": bundle.bundle_sha256,
            "legacy_boundary_bundle_schema_preserved": True,
            "interface_install_report_sha256": (interface_install["report_sha256"]),
            "neutral_norm_install_report_sha256": (
                neutral_norm_install["report_sha256"]
            ),
            "ordered_layer_result_sha256": [
                receipt.result_sha256 for receipt in completed_receipts
            ],
            "final_state_bank_sha256": state_bank.state_bank_sha256,
            "final_norm_report_sha256": (final_norm_report["report_sha256"]),
            "terminal_action_residual_enabled": (
                runtime_policy.terminal_action_residual_policy is not None
            ),
            "terminal_action_residual_integrated": (terminal_action_report is not None),
            "terminal_action_residual_report": (
                None if terminal_action_report is None else dict(terminal_action_report)
            ),
            "terminal_action_residual_report_sha256": (
                None
                if terminal_action_report is None
                else terminal_action_report["result_sha256"]
            ),
            "terminal_action_residual_requires_final_layer_execution_on_resume": (
                runtime_policy.terminal_action_residual_policy is not None
            ),
            "parameter_receipt": dict(parameter_receipt),
            "target_parameter_inventory_before": dict(target_inventory_before),
            "target_parameter_inventory_after": dict(target_inventory_after),
            "source_state_preserved": True,
            "target_parameter_count_preserved": True,
            "all_layers_compiled_in_order": True,
            "actual_folded_upstream_replay_used": True,
            "full_all_layer_phase_tree_materialized": False,
            "tensor_free_phase_shells_used": True,
            "active_layer_evidence_released_before_successor_compile": True,
            "final_norm_evidence_separately_bounded_and_reauthenticated": True,
            "candidate_replay_backend_injected": (
                candidate_replay_evaluator is not None
            ),
            "preinstall_candidate_selector_provided": (candidate_selector is not None),
            "preinstall_candidate_selector_active": bool(
                selector_applied_layer_indices
            ),
            "preinstall_candidate_selector_layer_indices": (
                selector_applied_layer_indices
            ),
            "short_conv_v5_endpoint_first_selector_integrated": (
                bool(selector_applied_layer_indices)
                and isinstance(
                    candidate_selector,
                    Lfm2CpcfV5EndpointFirstCandidateSelector,
                )
            ),
            "macrocell_candidate_selector_provided": (
                isinstance(
                    candidate_selector,
                    Lfm2CpcfV5EndpointFirstCandidateSelector,
                )
            ),
            "macrocell_candidate_selector_active": bool(
                macrocell_selector_applied_layer_indices
            ),
            "macrocell_candidate_selector_layer_indices": (
                macrocell_selector_applied_layer_indices
            ),
            "short_conv_v5_authenticated_whole_macrocell_selector_required": (
                runtime_policy.require_short_conv_v5_authenticated_whole_macrocell_selector
            ),
            "short_conv_v5_authenticated_whole_macrocell_selection_integrated": (
                authenticated_whole_macrocell_selection_integrated
            ),
            "short_conv_v5_macrocell_selector_final_authority": (
                authenticated_whole_macrocell_selection_integrated
            ),
            "short_conv_v5_endpoint_first_selector_final_authority": False,
            "short_conv_v5_endpoint_prefilter_control_integrated": (
                endpoint_prefilter_control_integrated
            ),
            "short_conv_numerical_frontier_adapter_provided": (
                numerical_frontier_adapter is not None
            ),
            "short_conv_numerical_frontier_adapter_active": bool(
                numerical_frontier_layer_indices
            ),
            "short_conv_numerical_frontier_adapter_layer_indices": (
                numerical_frontier_layer_indices
            ),
            "short_conv_layer_indices": short_conv_layer_indices,
            "short_conv_v5_real_numerical_adapter_required": (
                runtime_policy.require_short_conv_v5_real_numerical_frontier_adapter
            ),
            "short_conv_v5_real_numerical_adapter_concrete_instance_provided": (
                real_v5_adapter_contract is not None
            ),
            "short_conv_v5_real_numerical_adapter_contract": (
                None
                if real_v5_adapter_contract is None
                else dict(real_v5_adapter_contract)
            ),
            "short_conv_v5_real_numerical_provenance_layer_indices": (
                real_v5_layer_indices
            ),
            "short_conv_v5_real_numerical_layer_proofs": [
                dict(proof) for proof in real_v5_layer_proofs
            ],
            "short_conv_v5_numerical_candidate_generation_integrated": (
                real_v5_numerical_policy_integrated
            ),
            "short_conv_v5_numerical_policy_integrated": (
                real_v5_numerical_policy_integrated
            ),
            "short_conv_v5_post_install_rescoring_performed": False,
            "legacy_v4_default_or_fingerprint_changed": False,
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
                "fallback_paths": 0,
            },
        }
        if composed_attention_coverage is not None:
            body["composed_qk_attention_coverage"] = dict(composed_attention_coverage)
        if endpoint_residual_attention_coverage is not None:
            body["endpoint_residual_attention_coverage"] = dict(
                endpoint_residual_attention_coverage
            )
        if endpoint_sequential_transport_coverage is not None:
            body["endpoint_sequential_transport_coverage"] = dict(
                endpoint_sequential_transport_coverage
            )
        if terminal_action_report is not None:
            terminal_operation = terminal_action_report["operation_audit"]
            operation_audit = dict(body["operation_audit"])
            operation_audit.pop("target_model_forward_calls")
            operation_audit["target_full_model_forward_calls"] = 0
            operation_audit["target_final_layer_replay_calls"] = terminal_operation[
                "target_final_layer_replay_calls"
            ]
            body["operation_audit"] = operation_audit
        if runtime_policy.terminal_action_residual_policy is not None:
            if any(
                int(record["next_layer_index"]) >= catalog.layer_count
                for record in admitted_prefixes
            ):
                raise Lfm2CpcfV5ModelCompilerError(
                    "A pre-terminal final-layer prefix was incorrectly admitted "
                    "as resumable."
                )
            body[
                "terminal_action_preintegration_final_prefix_publication_suppressed"
            ] = True
            if terminal_resume_publication_receipt is None:
                raise Lfm2CpcfV5ModelCompilerError(
                    "Terminal-action compilation produced no sealed final-prefix "
                    "resume publication decision."
                )
            body["terminal_action_resume_publication_receipt"] = dict(
                terminal_resume_publication_receipt
            )
            body["terminal_action_suppressed_resume_prefix_indices"] = list(
                suppressed_terminal_preintegration_prefixes
            )
            body["terminal_action_resume_requires_final_layer_reexecution"] = True
            body["terminal_action_postintegration_resume_checkpoint_published"] = False
        result_sha = _sha(body)
        return legacy_model_fold.Lfm2CpcfCompiledModelFold(
            boundary_bundle=bundle,
            layer_results=tuple(completed_receipts),
            final_state_bank=state_bank,
            parameter_receipt=parameter_receipt,
            interface_install_report=interface_install,
            neutral_norm_install_report=neutral_norm_install,
            final_norm_report=final_norm_report,
            report={**body, "result_sha256": result_sha},
            result_sha256=result_sha,
        )
    except BaseException as error:
        if bootstrap_payload is not None:
            try:
                evidence_provider.release_layer_payload(
                    catalog.dynamic_catalog.references[bootstrap_payload.layer_index],
                    bootstrap_payload,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    f"Bootstrap evidence release also failed: {cleanup_error}"
                )
        if final_payload is not None:
            try:
                evidence_provider.release_final_norm_payload(final_payload)
            except BaseException as cleanup_error:
                error.add_note(
                    f"Final-normalization evidence release also failed: {cleanup_error}"
                )
        if not provider_closed:
            try:
                evidence_provider.close()
            except BaseException as cleanup_error:
                error.add_note(f"Selective provider close also failed: {cleanup_error}")
        rss_guard.abort()
        try:
            legacy_model_fold._restore_model(target_model, target_before)
            rollback = legacy_model_fold._prove_exact_whole_model_rollback(
                source_model,
                target_model,
                source_snapshot_before=source_before,
                target_snapshot_before=target_before,
            )
        except BaseException as rollback_error:
            raise Lfm2CpcfV5ModelCompilerError(
                "V5 selective compilation failed and exact whole-model "
                "rollback could not be proven."
            ) from rollback_error
        if rollback.get("exact_whole_model_rollback_proven") is not True:
            raise Lfm2CpcfV5ModelCompilerError(
                "V5 selective compilation rollback receipt is incomplete."
            ) from error
        if not isinstance(error, Exception):
            raise
        raise Lfm2CpcfV5ModelCompilerError(
            f"V5 selective compilation failed after exact whole-model rollback: {error}"
        ) from error


__all__ = [
    "LFM2_CPCF_V5_ARTIFACT_SCHEMA",
    "LFM2_CPCF_V5_COMPOSED_ATTENTION_COVERAGE_SCHEMA",
    "LFM2_CPCF_V5_COMPOSED_ATTENTION_LAYER_INDICES",
    "LFM2_CPCF_V5_ENDPOINT_RESIDUAL_ATTENTION_COVERAGE_SCHEMA",
    "LFM2_CPCF_V5_ENDPOINT_SEQUENTIAL_TRANSPORT_COVERAGE_SCHEMA",
    "LFM2_CPCF_V5_TERMINAL_RESUME_PUBLICATION_SCHEMA",
    "LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA",
    "LFM2_CPCF_V5_MODEL_FOLD_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_CHECKPOINT_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_EXPECTATION_SCHEMA",
    "LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA",
    "LFM2_CPCF_V5_RESUME_SCHEMA",
    "Lfm2CpcfV5EndpointFirstCandidateSelector",
    "Lfm2CpcfV5ModelCompilerError",
    "Lfm2CpcfV5PhysicalRssError",
    "Lfm2CpcfV5RuntimePolicy",
    "build_lfm2_cpcf_v5_target_phase_shells",
    "compile_lfm2_cpcf_v5_selective_model_fold",
    "lfm2_cpcf_v5_compiler_implementation",
    "materialize_lfm2_cpcf_v5_final_norm_phases",
    "materialize_lfm2_cpcf_v5_layer_phases",
]
