"""Reviewed, replay-authenticated promotion gate for severe LFM2 CPCF folds.

The first CPCF fold is a quarantined candidate.  Promotion is permitted only
after an independent 64-case holdout has been captured, mechanically scored,
and reviewed parent-relatively.  This module authenticates that complete
evidence chain and then proves that both the in-memory child and every fresh
reload reproduce the exact reviewed free-generation artifact.

The gate has no model or tokenizer loader, parent-model path, training path,
optimizer, pruning operation, quantizer, or network path.  Its tokenizer is an
already-loaded object whose semantics are bound to an authenticated local
tokenizer package.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
    execution_contract_satisfied,
    observe_model_execution,
)
from .dynamic_behavior_gate import (
    build_behavior_gate_binding,
    expose_behavior_gate_binding,
    identify_behavior_callable,
    resolve_behavior_gate_binding,
)
from .dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from .dynamic_fold_executor import FoldDecision, FoldEvaluationContext
from .dynamic_evidence_residency import EVIDENCE_RESIDENCY_WINDOW_SCHEMA
from .dynamic_replay_scheduler import (
    DYNAMIC_REPLAY_BUFFER_SCHEMA,
    DYNAMIC_REPLAY_EXECUTION_SCHEMA,
    DYNAMIC_REPLAY_PARITY_SCHEMA,
    CudaReplayDeviceReceipt,
    InstalledReplayStateFingerprint,
    ReplayAssignment,
    ReplayReplicaReceipt,
    ReplaySchedulePlan,
    ReplaySchedulerPolicy,
)
from .hf_lfm2_intelligence_gate import (
    HOLDOUT_CASE_COUNT,
    REPORT_SCHEMA as AUTOMATIC_GATE_REPORT_SCHEMA,
    IntelligenceHoldoutManifest,
    IntelligenceGatePolicy,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_lfm2_intelligence_holdout_manifest,
    load_manual_review_packet,
)
from .hf_lfm2_intelligence_holdout import (
    baseline_from_lfm2_intelligence_holdout_envelope,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
)
from .hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
    Lfm2CpcfCompletedLayerReceipt,
    Lfm2CpcfModelFoldPolicy,
)
from .hf_lfm2_cpcf_candidate_replay import (
    LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
    LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA,
)
from .hf_lfm2_cpcf_short_conv_fold import Lfm2CpcfShortConvGatePolicy
from .hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
)
from .hf_lfm2_cpcf_short_conv_v4_policy import (
    parse_lfm2_cpcf_short_conv_v4_solver_policy,
)
from .hf_lfm2_cpcf_short_conv_v5 import (
    Lfm2CpcfV5FirBasisPolicy,
    Lfm2CpcfV5RidgePolicy,
    rescore_sealed_v4_short_conv_selection_frontier,
)
from .hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from .hf_lfm2_cpcf_resume_artifact import Lfm2CpcfArtifactReference
from .hf_lfm2_cpcf_v5_admission_proof import (
    authenticate_lfm2_cpcf_v5_admission_chain_proof,
)
from .hf_lfm2_cpcf_v5_resume import (
    build_lfm2_cpcf_v5_physical_rss_policy_sha256,
    lfm2_cpcf_v5_resume_implementation,
)
from .hf_lfm2_cpcf_v5_model_compiler import (
    LFM2_CPCF_V5_ARTIFACT_SCHEMA,
    LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA,
    LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
    LFM2_CPCF_V5_RESUME_SCHEMA,
    Lfm2CpcfV5RuntimePolicy,
    _prove_real_v5_numerical_layer,
    _real_v5_numerical_adapter_contract,
    lfm2_cpcf_v5_compiler_implementation,
)
from .hf_lfm2_parent_relative_manual_gate import (
    REPORT_SCHEMA as PARENT_RELATIVE_REPORT_SCHEMA,
    RATINGS_SCHEMA as PARENT_RELATIVE_RATINGS_SCHEMA,
    build_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)
from .hf_tokenizer_bridge import (
    LOCAL_TOKENIZER_PACKAGE_FILENAMES,
    LOCAL_TOKENIZER_PACKAGE_SCHEMA,
    fingerprint_hf_tokenizer_semantics,
)


CPCF_RUN_SCHEMA = "neural-forge-lfm2-cpcf-blank-fold-run-v1"
CPCF_BLANK_TARGET_SCHEMA = "neural-forge-lfm2-cpcf-blank-target-v1"
CPCF_CAPTURE_SCHEMA = "neural-forge-lfm2-cpcf-holdout-candidate-capture-v1"
AUTOMATIC_RUNNER_REPORT_SCHEMA = "neural-forge-lfm2-intelligence-holdout-evaluation-v1"
COMPOSITE_PROMOTION_SCHEMA = "neural-forge-lfm2-cpcf-reviewed-promotion-report-v1"
REVIEWED_EVIDENCE_SCHEMA = "neural-forge-lfm2-cpcf-reviewed-promotion-evidence-v1"
BEHAVIOR_GATE_ID = "lfm2-cpcf-reviewed-whole-response-replay"
BEHAVIOR_GATE_VERSION = 1
IMPLEMENTATION_SEAL_SCHEMA = "neural-forge-lfm2-cpcf-reviewed-gate-implementation-v1"
ORDERED_PHASE_BINDING_SCHEMA = "neural-forge-lfm2-cpcf-ordered-phase-binding-v1"
V5_GPU_REPLAY_SCHEMA = "neural-forge-lfm2-cpcf-v5-authenticated-two-gpu-replay-v1"
V5_GPU_RUNTIME_AUTHENTICATION_SCHEMA = (
    "neural-forge-dynamic-replay-live-runtime-authentication-v1"
)
V5_GPU_STATE_PROBES_SCHEMA = "neural-forge-dynamic-replay-live-state-probes-v1"
SUPPORTED_COMPILED_MODEL_FOLD_SCHEMAS = frozenset(
    {
        LFM2_CPCF_MODEL_FOLD_SCHEMA,
        LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    }
)

_EVIDENCE_LOADER_TOKEN = object()


class Lfm2CpcfReviewedPromotionError(RuntimeError):
    """Reviewed evidence or live replay does not authorize promotion."""


def _digest(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} must be a lowercase SHA-256 digest."
        )
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Lfm2CpcfReviewedPromotionError(f"{name} must be a positive integer.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfReviewedPromotionError(f"{name} must be a nonnegative integer.")
    return value


def _canonical_copy(value: Any, name: str) -> Any:
    try:
        return json.loads(canonical_json_bytes(value).decode("ascii"))
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfReviewedPromotionError(f"{name} must be canonical JSON.") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_file_sha256(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(document)) + b"\n").hexdigest()


def _sealed_document(
    document: Mapping[str, Any],
    *,
    schema: str,
    seal_name: str,
    name: str,
) -> tuple[dict[str, Any], str]:
    body = dict(document)
    seal = body.pop(seal_name, None)
    if body.get("schema") != schema or seal != canonical_sha256(body):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} schema or canonical seal is invalid."
        )
    return body, _digest(seal, f"{name} {seal_name}")


@dataclass(frozen=True, slots=True)
class CanonicalJsonArtifact:
    """One trusted canonical-JSON artifact and its out-of-band document root."""

    source: Mapping[str, Any] | str | os.PathLike[str]
    document_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_sha256",
            _digest(self.document_sha256, "document_sha256"),
        )

    def load(self, name: str) -> dict[str, Any]:
        if isinstance(self.source, Mapping):
            document = _canonical_copy(dict(self.source), name)
        else:
            raw_path = os.fspath(self.source)
            if "://" in raw_path:
                raise Lfm2CpcfReviewedPromotionError(
                    f"{name} must be a local canonical JSON file."
                )
            try:
                path = Path(raw_path).expanduser().resolve(strict=True)
            except OSError as exc:
                raise Lfm2CpcfReviewedPromotionError(f"{name} does not exist.") from exc
            if not path.is_file() or path.is_symlink():
                raise Lfm2CpcfReviewedPromotionError(f"{name} is missing or unsafe.")
            raw = path.read_bytes()
            try:
                value = json.loads(raw.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise Lfm2CpcfReviewedPromotionError(
                    f"{name} is not canonical ASCII JSON."
                ) from exc
            if not isinstance(value, Mapping):
                raise Lfm2CpcfReviewedPromotionError(f"{name} must be a JSON object.")
            canonical = canonical_json_bytes(value)
            if raw not in (canonical, canonical + b"\n"):
                raise Lfm2CpcfReviewedPromotionError(
                    f"{name} is not canonically encoded."
                )
            document = dict(value)
        if canonical_sha256(document) != self.document_sha256:
            raise Lfm2CpcfReviewedPromotionError(
                f"{name} differs from its trusted document SHA-256."
            )
        return document


@dataclass(frozen=True, slots=True)
class Lfm2CpcfReviewedPromotionArtifacts:
    """All immutable artifacts required for reviewed CPCF promotion."""

    manifest: CanonicalJsonArtifact
    parent_envelope: CanonicalJsonArtifact
    cpcf_run_receipt: CanonicalJsonArtifact
    candidate_outputs: CanonicalJsonArtifact
    capture_receipt: CanonicalJsonArtifact
    automatic_report: CanonicalJsonArtifact
    manual_packet: CanonicalJsonArtifact
    parent_relative_manual_packet: CanonicalJsonArtifact
    parent_relative_ratings: CanonicalJsonArtifact
    parent_relative_report: CanonicalJsonArtifact
    composite_promotion_report: CanonicalJsonArtifact

    def __post_init__(self) -> None:
        for name in (
            "manifest",
            "parent_envelope",
            "cpcf_run_receipt",
            "candidate_outputs",
            "capture_receipt",
            "automatic_report",
            "manual_packet",
            "parent_relative_manual_packet",
            "parent_relative_ratings",
            "parent_relative_report",
            "composite_promotion_report",
        ):
            if not isinstance(getattr(self, name), CanonicalJsonArtifact):
                raise Lfm2CpcfReviewedPromotionError(
                    f"{name} must be a CanonicalJsonArtifact."
                )


def fingerprint_local_tokenizer_package(
    tokenizer_directory: str | os.PathLike[str],
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    """Hash the bounded local tokenizer files used by the CPCF capture."""

    raw = os.fspath(tokenizer_directory)
    if "://" in raw:
        raise Lfm2CpcfReviewedPromotionError(
            "Tokenizer package must be a local directory."
        )
    try:
        directory = Path(raw).expanduser().resolve(strict=True)
    except OSError as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "Local tokenizer directory does not exist."
        ) from exc
    if not directory.is_dir() or directory.is_symlink():
        raise Lfm2CpcfReviewedPromotionError(
            "Local tokenizer directory is missing or unsafe."
        )
    files = []
    for name in LOCAL_TOKENIZER_PACKAGE_FILENAMES:
        path = directory / name
        if path.is_symlink():
            raise Lfm2CpcfReviewedPromotionError(
                "Tokenizer artifacts cannot be symlinks."
            )
        if path.is_file():
            files.append(
                {
                    "path": name,
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not files or files[0]["path"] != "tokenizer.json":
        raise Lfm2CpcfReviewedPromotionError(
            "Local tokenizer package lacks tokenizer.json."
        )
    body = {
        "schema": LOCAL_TOKENIZER_PACKAGE_SCHEMA,
        "files": files,
    }
    return canonical_sha256(body), tuple(MappingProxyType(row) for row in files)


def fingerprint_cpcf_candidate_state(model: Any) -> str:
    """Reproduce the state identity sealed by the CPCF capture workflow."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise Lfm2CpcfReviewedPromotionError(
            "PyTorch is required for CPCF state authentication."
        ) from exc
    state_dict = getattr(model, "state_dict", None)
    if not callable(state_dict):
        raise Lfm2CpcfReviewedPromotionError("Candidate model has no state_dict().")
    rows = []
    for name, value in sorted(state_dict().items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise Lfm2CpcfReviewedPromotionError(
                "Candidate state entries must be named tensors."
            )
        if value.device.type == "meta" or value.layout != torch.strided:
            raise Lfm2CpcfReviewedPromotionError(
                "Candidate state must contain materialized dense tensors."
            )
        tensor = value.detach().contiguous()
        raw = tensor.view(torch.uint8).reshape(-1)
        tensor_digest = hashlib.sha256()
        tensor_digest.update(
            canonical_json_bytes(
                {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                }
            )
        )
        for start in range(0, int(raw.numel()), 8 * 1024**2):
            piece = raw[start : start + 8 * 1024**2]
            piece = piece.to(device="cpu").contiguous()
            tensor_digest.update(piece.numpy().tobytes())
        rows.append(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_digest.hexdigest(),
            }
        )
    if not rows:
        raise Lfm2CpcfReviewedPromotionError("Candidate model state is empty.")
    return canonical_sha256(rows)


def _verify_cpcf_run_receipt(
    document: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    body, seal = _sealed_document(
        document,
        schema=CPCF_RUN_SCHEMA,
        seal_name="receipt_sha256",
        name="CPCF run receipt",
    )
    status = body.get("status")
    claims = body.get("claims")
    audit = body.get("operation_audit")
    compression = body.get("compression")
    required_claims = {
        "folding_only": True,
        "blank_target_filled": True,
        "training_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "parent_backed_fallback_used": False,
        "exact_token_match_used_as_principal_objective": False,
        "marginal_compaction_reported_as_success": False,
    }
    required_audit = {
        "parent_forward_calls": 0,
        "parent_generation_calls": 0,
        "parent_cache_generation_calls": 0,
        "parent_cache_regeneration_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "quantized_parameters": 0,
        "pruned_parameters": 0,
        "fallback_model_calls": 0,
        "network_access": False,
    }
    if (
        status != "folded_child_requires_manual_intelligence_review"
        or body.get("fresh_reload_verified") is not True
        or body.get("structural_parameter_compaction_proven") is not True
        or body.get("parent_state_preserved") is not True
        or body.get("project_promotion_ratio_met") is not True
        or body.get("diagnostic_control") is not False
        or body.get("behavioral_intelligence_retention_proven") is not False
        or body.get("manual_intelligence_review_required") is not True
        or not isinstance(claims, Mapping)
        or any(claims.get(key) is not value for key, value in required_claims.items())
        or not isinstance(audit, Mapping)
        or any(audit.get(key) != value for key, value in required_audit.items())
        or not isinstance(compression, Mapping)
        or compression.get("parameter_count_compaction_only") is not True
        or compression.get("checkpoint_byte_ratio_used") is not False
        or compression.get("promotion_ratio_met") is not True
        or compression.get("quantization_used") is not False
        or compression.get("pruning_used") is not False
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF run receipt does not prove an accepted severe, folding-only "
            "blank-child candidate."
        )
    return seal, body


def _verify_parameter_inventory(
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Authenticate one independently counted learned-parameter inventory."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} must be a parameter-inventory object."
        )
    document = _canonical_copy(dict(value), name)
    body = dict(document)
    seal = body.pop("inventory_sha256", None)
    unique_parameters = _positive_int(
        body.get("unique_parameters"),
        f"{name} unique_parameters",
    )
    raw_occurrences = _positive_int(
        body.get("raw_named_parameter_occurrences"),
        f"{name} raw_named_parameter_occurrences",
    )
    raw_numel = _positive_int(
        body.get("raw_named_parameter_numel"),
        f"{name} raw_named_parameter_numel",
    )
    iterator_numel = _positive_int(
        body.get("parameter_iterator_unique_numel"),
        f"{name} parameter_iterator_unique_numel",
    )
    alias_count = _nonnegative_int(
        body.get("alias_group_count"),
        f"{name} alias_group_count",
    )
    alias_groups = body.get("alias_groups")
    if (
        body.get("method")
        != "unique_materialized_parameter_storage_offset_numel_identity"
        or body.get("independent_methods_agree") is not True
        or iterator_numel != unique_parameters
        or raw_numel < unique_parameters
        or raw_occurrences < 1
        or not isinstance(alias_groups, list)
        or len(alias_groups) != alias_count
        or body.get("checkpoint_byte_size_used") is not False
        or body.get("quantization_adjustment_used") is not False
        or body.get("pruning_mask_adjustment_used") is not False
        or seal != canonical_sha256(body)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} is not a sealed independent dense-parameter inventory."
        )
    return document, _digest(seal, f"{name} inventory_sha256")


def _verify_parameter_receipt(
    value: Any,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    """Authenticate one parameter-count-only CPCF compaction receipt."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} must be a parameter-compaction receipt."
        )
    document = _canonical_copy(dict(value), name)
    body, receipt_sha256 = _sealed_document(
        document,
        schema=LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
        seal_name="receipt_sha256",
        name=name,
    )
    source_inventory, _source_inventory_sha256 = _verify_parameter_inventory(
        body.get("source_inventory"),
        name=f"{name} source_inventory",
    )
    target_inventory, _target_inventory_sha256 = _verify_parameter_inventory(
        body.get("target_inventory"),
        name=f"{name} target_inventory",
    )
    source_parameters = _positive_int(
        body.get("source_unique_parameters"),
        f"{name} source_unique_parameters",
    )
    target_parameters = _positive_int(
        body.get("target_unique_parameters"),
        f"{name} target_unique_parameters",
    )
    planned_parameters = _positive_int(
        body.get("planned_target_unique_parameters"),
        f"{name} planned_target_unique_parameters",
    )
    realized_ratio = body.get("realized_compression_ratio")
    promotion_ratio = body.get("promotion_compression_ratio")
    if (
        isinstance(realized_ratio, bool)
        or not isinstance(realized_ratio, (int, float))
        or not math.isfinite(float(realized_ratio))
        or isinstance(promotion_ratio, bool)
        or not isinstance(promotion_ratio, (int, float))
        or not math.isfinite(float(promotion_ratio))
        or float(promotion_ratio) <= 0.0
    ):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} has invalid compression-ratio accounting."
        )
    expected_ratio = source_parameters / target_parameters
    promotion_met = expected_ratio >= float(promotion_ratio)
    if (
        source_inventory["unique_parameters"] != source_parameters
        or target_inventory["unique_parameters"] != target_parameters
        or not 0 < target_parameters < source_parameters
        or planned_parameters != target_parameters
        or body.get("parameters_removed_from_topology")
        != source_parameters - target_parameters
        or float(realized_ratio) != expected_ratio
        or body.get("promotion_ratio_met") is not promotion_met
        or body.get("valid_lower_ratio_diagnostic_fold") is promotion_met
        or body.get("parameter_count_compaction_only") is not True
        or body.get("checkpoint_byte_ratio_used") is not False
        or body.get("quantization_used") is not False
        or body.get("pruning_used") is not False
    ):
        raise Lfm2CpcfReviewedPromotionError(
            f"{name} does not prove one dense parameter-count compaction."
        )
    return document, receipt_sha256


def _normalize_model_fold_policy(value: Any) -> dict[str, Any]:
    """Round-trip the receipt policy through the compiler's typed policy."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model fold has no typed policy."
        )
    policy = _canonical_copy(dict(value), "CPCF compiled model-fold policy")
    short_conv = policy.get("short_conv_gate_policy")
    if not isinstance(short_conv, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold policy has no short-conv gate policy."
        )
    try:
        short_conv_policy = Lfm2CpcfShortConvGatePolicy(**dict(short_conv))
        constructor = dict(policy)
        constructor["short_conv_gate_policy"] = short_conv_policy
        serialized_v4 = constructor.pop(
            "short_conv_v4_solver_policy",
            None,
        )
        if serialized_v4 is not None:
            constructor["short_conv_v4_solver_policy"] = (
                parse_lfm2_cpcf_short_conv_v4_solver_policy(serialized_v4)
            )
        normalized = Lfm2CpcfModelFoldPolicy(**constructor).as_dict()
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold policy is invalid."
        ) from exc
    if canonical_json_bytes(policy) != canonical_json_bytes(normalized):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold policy is incomplete or noncanonical."
        )
    return normalized


def _normalize_v5_model_fold_policy(value: Any) -> dict[str, Any]:
    """Round-trip the selective-v5 receipt policy through its typed policy."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold has no typed runtime policy."
        )
    policy = _canonical_copy(
        dict(value),
        "CPCF v5 compiled model-fold runtime policy",
    )
    model_policy_value = policy.get("model_fold_policy")
    model_policy = _normalize_model_fold_policy(model_policy_value)
    try:
        model_constructor = dict(model_policy)
        short_conv = model_constructor.get("short_conv_gate_policy")
        if not isinstance(short_conv, Mapping):
            raise TypeError("short_conv_gate_policy is missing")
        model_constructor["short_conv_gate_policy"] = Lfm2CpcfShortConvGatePolicy(
            **dict(short_conv)
        )
        serialized_v4 = model_constructor.pop(
            "short_conv_v4_solver_policy",
            None,
        )
        if serialized_v4 is not None:
            model_constructor["short_conv_v4_solver_policy"] = (
                parse_lfm2_cpcf_short_conv_v4_solver_policy(serialized_v4)
            )
        typed_model_policy = Lfm2CpcfModelFoldPolicy(**model_constructor)

        constructor = dict(policy)
        constructor["model_fold_policy"] = typed_model_policy
        supplied_maximum_increase = constructor.pop(
            "maximum_physical_rss_increase_bytes",
            None,
        )
        typed = Lfm2CpcfV5RuntimePolicy(**constructor)
        normalized = typed.as_dict()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model-fold runtime policy is invalid."
        ) from exc
    if supplied_maximum_increase != normalized[
        "maximum_physical_rss_increase_bytes"
    ] or canonical_json_bytes(policy) != canonical_json_bytes(normalized):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model-fold runtime policy is incomplete or noncanonical."
        )
    return normalized


def _authenticate_v5_numerical_adapter_contract(
    value: Any,
) -> dict[str, Any]:
    """Reconstruct the exact concrete numerical adapter named by v5."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold has no numerical adapter contract."
        )
    contract = _canonical_copy(
        dict(value),
        "CPCF v5 numerical adapter contract",
    )
    contract_body = dict(contract)
    contract_sha256 = _digest(
        contract_body.pop("contract_sha256", None),
        "CPCF v5 numerical adapter contract SHA-256",
    )
    if canonical_sha256(contract_body) != contract_sha256:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 numerical adapter contract seal is invalid."
        )

    basis_value = contract.get("basis_policy")
    ridge_value = contract.get("ridge_policy")
    if not isinstance(basis_value, Mapping) or not isinstance(
        ridge_value,
        Mapping,
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 numerical adapter contract lacks typed policies."
        )
    basis_body = _canonical_copy(
        dict(basis_value),
        "CPCF v5 numerical basis policy",
    )
    ridge_body = _canonical_copy(
        dict(ridge_value),
        "CPCF v5 numerical ridge policy",
    )
    try:
        basis_policy = Lfm2CpcfV5FirBasisPolicy(
            fir_descriptor_weight=basis_body.get("fir_descriptor_weight"),
            activation_descriptor_weight=basis_body.get("activation_descriptor_weight"),
            endpoint_descriptor_weight=basis_body.get("endpoint_descriptor_weight"),
            energy_floor_relative=basis_body.get("energy_floor_relative"),
            maximum_projected_fir_offdiagonal_relative_frobenius=(
                basis_body.get("maximum_projected_fir_offdiagonal_relative_frobenius")
            ),
            maximum_per_lag_commutator_relative_frobenius=(
                basis_body.get("maximum_per_lag_commutator_relative_frobenius")
            ),
            authentication_sha256=str(contract.get("basis_policy_sha256", "")),
        )
        ridge_policy = Lfm2CpcfV5RidgePolicy(
            condition_caps=tuple(ridge_body.get("condition_caps", ())),
            rank_relative_tolerance=ridge_body.get("rank_relative_tolerance"),
            parameter_trust_radius=ridge_body.get("parameter_trust_radius"),
            effect_trust_radius=ridge_body.get("effect_trust_radius"),
            parameter_reference_floor=ridge_body.get("parameter_reference_floor"),
            effect_reference_floor=ridge_body.get("effect_reference_floor"),
            authentication_sha256=str(contract.get("ridge_policy_sha256", "")),
        )
        adapter = Lfm2CpcfShortConvV5NumericalFrontierAdapter(
            basis_policy=basis_policy,
            ridge_policy=ridge_policy,
            adapter_id=str(contract.get("adapter_id", "")),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 numerical adapter contract cannot reconstruct the "
            "concrete authenticated adapter."
        ) from exc
    recomputed = _real_v5_numerical_adapter_contract(adapter)
    if (
        not isinstance(recomputed, Mapping)
        or canonical_json_bytes(basis_body)
        != canonical_json_bytes(basis_policy.authentication_body())
        or canonical_json_bytes(ridge_body)
        != canonical_json_bytes(ridge_policy.authentication_body())
        or canonical_json_bytes(contract) != canonical_json_bytes(dict(recomputed))
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 numerical adapter contract differs from its fully "
            "reconstructed concrete adapter."
        )
    return contract


def _typed_v5_gpu_device_receipt(value: Any) -> CudaReplayDeviceReceipt:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError("CPCF v5 GPU device receipt is missing.")
    document = _canonical_copy(dict(value), "CPCF v5 GPU device receipt")
    expected_fields = {
        "schema",
        "backend",
        "device_index",
        "device_uuid",
        "device_name",
        "pci_domain_id",
        "pci_bus_id",
        "pci_device_id",
        "capability",
        "total_memory_bytes",
        "multiprocessor_count",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "deterministic_algorithms_enabled",
        "deterministic_debug_mode",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "cudnn_allow_tf32",
        "cuda_matmul_allow_tf32",
        "cuda_matmul_allow_bf16_reduced_precision_reduction",
        "cuda_matmul_allow_fp16_reduced_precision_reduction",
        "float32_matmul_precision",
        "cublas_workspace_config",
        "receipt_sha256",
    }
    try:
        device = CudaReplayDeviceReceipt(
            device_index=document["device_index"],
            device_uuid=document["device_uuid"],
            device_name=document["device_name"],
            pci_domain_id=document["pci_domain_id"],
            pci_bus_id=document["pci_bus_id"],
            pci_device_id=document["pci_device_id"],
            capability=tuple(document["capability"]),
            total_memory_bytes=document["total_memory_bytes"],
            multiprocessor_count=document["multiprocessor_count"],
            torch_version=document["torch_version"],
            cuda_runtime_version=document["cuda_runtime_version"],
            cudnn_version=document["cudnn_version"],
            deterministic_algorithms_enabled=(
                document["deterministic_algorithms_enabled"]
            ),
            deterministic_debug_mode=document["deterministic_debug_mode"],
            cudnn_deterministic=document["cudnn_deterministic"],
            cudnn_benchmark=document["cudnn_benchmark"],
            cudnn_allow_tf32=document["cudnn_allow_tf32"],
            cuda_matmul_allow_tf32=document["cuda_matmul_allow_tf32"],
            cuda_matmul_allow_bf16_reduced_precision_reduction=(
                document["cuda_matmul_allow_bf16_reduced_precision_reduction"]
            ),
            cuda_matmul_allow_fp16_reduced_precision_reduction=(
                document["cuda_matmul_allow_fp16_reduced_precision_reduction"]
            ),
            float32_matmul_precision=document["float32_matmul_precision"],
            cublas_workspace_config=document["cublas_workspace_config"],
            receipt_sha256=document["receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU device receipt cannot reconstruct its exact type."
        ) from exc
    if (
        set(document) != expected_fields
        or document != device.as_dict()
        or device.device_name != "NVIDIA GeForce RTX 3090"
        or "+cpu" in device.torch_version
        or device.deterministic_algorithms_enabled is not True
        or device.deterministic_debug_mode != 2
        or device.cudnn_deterministic is not True
        or device.cudnn_benchmark is not False
        or device.cudnn_allow_tf32 is not False
        or device.cuda_matmul_allow_tf32 is not False
        or device.cuda_matmul_allow_bf16_reduced_precision_reduction is not False
        or device.cuda_matmul_allow_fp16_reduced_precision_reduction is not False
        or device.float32_matmul_precision != "highest"
        or device.cublas_workspace_config != ":4096:8"
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU device receipt is noncanonical or not the exact "
            "deterministic RTX-3090 runtime."
        )
    return device


def _typed_v5_installed_state(
    value: Any,
) -> InstalledReplayStateFingerprint:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 installed replay state is missing."
        )
    document = _canonical_copy(
        dict(value),
        "CPCF v5 installed replay state",
    )
    try:
        state = InstalledReplayStateFingerprint(
            state_sha256=document["state_sha256"],
            tensor_count=document["tensor_count"],
            logical_bytes=document["logical_bytes"],
            fingerprint_sha256=document["fingerprint_sha256"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 installed replay state cannot reconstruct its exact type."
        ) from exc
    if (
        set(document)
        != {
            "schema",
            "state_sha256",
            "tensor_count",
            "logical_bytes",
            "fingerprint_sha256",
        }
        or document != state.as_dict()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 installed replay state is noncanonical."
        )
    return state


def _typed_v5_replica_receipt(value: Any) -> ReplayReplicaReceipt:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 replay replica receipt is missing."
        )
    document = _canonical_copy(
        dict(value),
        "CPCF v5 replay replica receipt",
    )
    device = _typed_v5_gpu_device_receipt(document.get("device"))
    installed = _typed_v5_installed_state(document.get("installed_state"))
    try:
        replica = ReplayReplicaReceipt(
            device=device,
            installed_state=installed,
            receipt_sha256=document["receipt_sha256"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 replay replica cannot reconstruct its exact type."
        ) from exc
    if (
        set(document)
        != {
            "device_receipt_sha256",
            "installed_state_fingerprint_sha256",
            "device",
            "installed_state",
            "receipt_sha256",
        }
        or document != replica.as_dict()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 replay replica receipt is noncanonical."
        )
    return replica


def _typed_v5_scheduler_policy(value: Any) -> ReplaySchedulerPolicy:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError("CPCF v5 scheduler policy is missing.")
    document = _canonical_copy(dict(value), "CPCF v5 scheduler policy")
    constructor = dict(document)
    fixed_fields = {
        "assignment": "canonical_round_robin_by_first_shard_key",
        "cpu_cuda_parity_required_before_scheduled_work": True,
        "runtime_reauthenticated_after_parity_and_work": True,
        "installed_state_reauthenticated_after_parity_and_work": True,
        "folding_math_changed": False,
    }
    if any(
        constructor.pop(name, None) != expected
        for name, expected in fixed_fields.items()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 scheduler policy lost a fixed execution invariant."
        )
    try:
        policy = ReplaySchedulerPolicy(**constructor)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 scheduler policy cannot reconstruct its exact type."
        ) from exc
    if (
        document != policy.as_dict()
        or policy.expected_device_name != "NVIDIA GeForce RTX 3090"
        or policy.maximum_pinned_bytes_per_slot != 256 << 20
        or policy.parity_absolute_tolerance != 0.125
        or policy.parity_relative_tolerance != 0.02
        or policy.parity_maximum_relative_rmse != 0.02
        or policy.parity_minimum_cosine_similarity != 0.999
        or policy.require_deterministic_debug_mode is not True
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 scheduler policy differs from the exact reviewed "
            "two-RTX-3090 BF16 replay policy."
        )
    return policy


def _authenticate_v5_gpu_candidate_runtime(
    value: Any,
) -> tuple[
    ReplaySchedulerPolicy,
    tuple[ReplayReplicaReceipt, ...],
    str,
]:
    """Reconstruct the exact run-level two-GPU scheduler authority."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run receipt has no GPU candidate-replay authority."
        )
    body, report_sha256 = _sealed_document(
        value,
        schema=V5_GPU_REPLAY_SCHEMA,
        seal_name="report_sha256",
        name="CPCF v5 run-level GPU candidate replay",
    )
    expected_fields = {
        "schema",
        "logical_device_indices",
        "devices",
        "replicas",
        "cpu_installed_state",
        "scheduler_policy",
        "runtime_authentication",
        "runtime_authentication_sha256",
        "proven_parity_receipt",
        "contract_tensor_names",
        "contract_cpu_tensors_retained_for_runtime_reauthentication",
        "contract_gpu_tensors_retained_for_runtime_reauthentication",
        "homogeneous_authenticated_receipts",
        "candidate_payload_staging",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
    }
    replicas_value = body.get("replicas")
    devices_value = body.get("devices")
    if (
        set(body) != expected_fields
        or not isinstance(replicas_value, list)
        or not isinstance(devices_value, list)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run-level GPU authority fields are incomplete or expanded."
        )
    replicas = tuple(_typed_v5_replica_receipt(row) for row in replicas_value)
    devices = tuple(_typed_v5_gpu_device_receipt(row) for row in devices_value)
    policy = _typed_v5_scheduler_policy(body.get("scheduler_policy"))
    cpu_state = _typed_v5_installed_state(body.get("cpu_installed_state"))
    if (
        len(replicas) != 2
        or tuple(replica.device.device_index for replica in replicas) != (0, 1)
        or tuple(replica.device for replica in replicas) != devices
        or len({device.device_uuid for device in devices}) != 2
        or len(
            {
                (
                    device.pci_domain_id,
                    device.pci_bus_id,
                    device.pci_device_id,
                )
                for device in devices
            }
        )
        != 2
        or any(replica.installed_state != cpu_state for replica in replicas)
        or body.get("logical_device_indices") != [0, 1]
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run-level GPU authority lacks two distinct exact "
            "replicas sharing the authenticated installed state."
        )

    runtime_value = body.get("runtime_authentication")
    if not isinstance(runtime_value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run-level GPU authority lacks runtime authentication."
        )
    runtime, runtime_sha256 = _sealed_document(
        runtime_value,
        schema=V5_GPU_RUNTIME_AUTHENTICATION_SCHEMA,
        seal_name="report_sha256",
        name="CPCF v5 live GPU runtime authentication",
    )
    state_contract = runtime.get("state_probe_contract")
    if not isinstance(state_contract, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 live runtime lacks its state-probe contract."
        )
    state_body = dict(state_contract)
    state_sha256 = _digest(
        state_body.pop("contract_sha256", None),
        "CPCF v5 state-probe contract SHA-256",
    )
    parity = body.get("proven_parity_receipt")
    if (
        set(runtime)
        != {
            "schema",
            "replicas",
            "policy",
            "state_probe_contract",
            "device_receipts_freshly_probed",
            "installed_states_freshly_rehashed",
            "bounded_pinned_host_factory_authenticated",
        }
        or runtime.get("replicas") != [row.as_dict() for row in replicas]
        or runtime.get("policy") != policy.as_dict()
        or runtime.get("device_receipts_freshly_probed") is not True
        or runtime.get("installed_states_freshly_rehashed") is not True
        or runtime.get("bounded_pinned_host_factory_authenticated") is not True
        or body.get("runtime_authentication_sha256") != runtime_sha256
        or canonical_sha256(state_body) != state_sha256
        or state_body
        != {
            "schema": V5_GPU_STATE_PROBES_SCHEMA,
            "logical_device_indices": [0, 1],
            "contract_tensor_names": body.get("contract_tensor_names"),
            "installed_state": cpu_state.as_dict(),
            "cpu_tensors_retained": True,
            "cuda_tensors_retained": True,
        }
        or not isinstance(parity, Mapping)
        or set(parity)
        != {
            "path",
            "file_sha256",
            "receipt_sha256",
            "source_sha256",
            "scheduler_policy",
            "target_dtype",
            "real_two_gpu_parity_passed",
        }
        or _digest(
            parity.get("file_sha256"),
            "CPCF v5 proven parity file SHA-256",
        )
        != parity.get("file_sha256")
        or _digest(
            parity.get("receipt_sha256"),
            "CPCF v5 proven parity receipt SHA-256",
        )
        != parity.get("receipt_sha256")
        or parity.get("scheduler_policy") != policy.as_dict()
        or parity.get("target_dtype") != "bfloat16"
        or parity.get("real_two_gpu_parity_passed") is not True
        or body.get("contract_cpu_tensors_retained_for_runtime_reauthentication")
        is not True
        or body.get("contract_gpu_tensors_retained_for_runtime_reauthentication")
        is not True
        or body.get("homogeneous_authenticated_receipts") is not True
        or body.get("candidate_payload_staging")
        != "bounded_double_buffered_pinned_system_memory_to_cuda"
        or any(
            body.get(name) != 0
            for name in (
                "training_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run-level GPU authority failed its typed runtime, "
            "state, parity, pinned-host, or zero-operation contract."
        )
    return policy, replicas, report_sha256


def _typed_v5_replay_plan(
    value: Any,
    *,
    policy: ReplaySchedulerPolicy,
    replicas: tuple[ReplayReplicaReceipt, ...],
) -> ReplaySchedulePlan:
    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU execution has no scheduler plan."
        )
    document = _canonical_copy(dict(value), "CPCF v5 GPU scheduler plan")
    assignments_value = document.get("assignments")
    if not isinstance(assignments_value, list):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU scheduler plan has no assignments."
        )
    try:
        assignments = tuple(
            ReplayAssignment(**dict(row))
            for row in assignments_value
            if isinstance(row, Mapping)
        )
        plan = ReplaySchedulePlan(
            assignments=assignments,
            policy=policy,
            replicas=replicas,
            plan_sha256=document["plan_sha256"],
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU scheduler plan cannot reconstruct its exact typed assignments."
        ) from exc
    if len(assignments) != len(assignments_value) or document != plan.as_dict():
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU scheduler plan differs from its typed deterministic "
            "reconstruction."
        )
    return plan


def _finite_float(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Lfm2CpcfReviewedPromotionError(f"{name} must be a finite number.")
    return float(value)


def _typed_v5_parity_structure(
    value: Any,
    *,
    expected_candidate_id: str,
    expected_candidate_state_sha256: str,
    expected_policy_sha256: str,
    expected_target_dtype: str,
    expected_split: str,
    expected_cases: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Reconstruct the exact candidate-replay output tree and tensor leaves."""

    descriptor = _canonical_copy(value, "CPCF v5 parity structure descriptor")
    tensor_leaves: list[Mapping[str, Any]] = []

    def walk(node_value: Any, *, expected_path: str) -> Mapping[str, Any]:
        if not isinstance(node_value, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity structure node is not a mapping."
            )
        node = _canonical_copy(
            dict(node_value),
            f"CPCF v5 parity structure node {expected_path}",
        )
        if node.get("path") != expected_path or not isinstance(
            node.get("kind"),
            str,
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity structure path or kind is invalid."
            )
        kind = node["kind"]
        if kind == "tensor":
            shape = node.get("shape")
            if (
                set(node) != {"path", "kind", "dtype", "shape"}
                or not isinstance(node.get("dtype"), str)
                or not isinstance(shape, list)
                or len(shape) != 3
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 parity tensor structure is not a nonempty "
                    "three-dimensional replay history."
                )
            tensor_leaves.append(MappingProxyType(node))
            return MappingProxyType(node)
        if kind in {"mapping", "sequence"}:
            children_value = node.get("children")
            if set(node) != {"path", "kind", "children"} or not isinstance(
                children_value, list
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 parity container structure is incomplete."
                )
            children = []
            for index, child_value in enumerate(children_value):
                if not isinstance(child_value, Mapping):
                    raise Lfm2CpcfReviewedPromotionError(
                        "CPCF v5 parity structure child is not a mapping."
                    )
                child_path = child_value.get("path")
                if kind == "sequence":
                    required_path = f"{expected_path}[{index}]"
                else:
                    prefix = f"{expected_path}."
                    if (
                        not isinstance(child_path, str)
                        or not child_path.startswith(prefix)
                        or any(
                            marker in child_path[len(prefix) :]
                            for marker in (".", "[", "]")
                        )
                    ):
                        raise Lfm2CpcfReviewedPromotionError(
                            "CPCF v5 parity mapping child path is noncanonical."
                        )
                    required_path = child_path
                children.append(walk(child_value, expected_path=required_path))
            child_paths = [child["path"] for child in children]
            if len(set(child_paths)) != len(child_paths) or (
                kind == "mapping" and child_paths != sorted(child_paths)
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 parity structure children are duplicated or unordered."
                )
            return MappingProxyType(node)
        if set(node) != {"path", "kind", "value"}:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity scalar structure is incomplete."
            )
        scalar = node.get("value")
        observed_kind = type(scalar).__name__
        if kind != observed_kind:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity scalar kind differs from its value."
            )
        return MappingProxyType(node)

    root = walk(descriptor, expected_path="$")
    if root.get("kind") != "mapping":
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 parity output root is not the candidate replay mapping."
        )

    def children_by_path(node: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        children = node.get("children")
        if not isinstance(children, list):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity structure has no typed children."
            )
        return {str(child["path"]): child for child in children}

    top = children_by_path(root)
    expected_top_paths = {
        "$.candidate_id",
        "$.candidate_state_sha256",
        "$.cases",
        "$.policy_sha256",
        "$.required_split",
        "$.schema",
        "$.target_dtype",
    }
    scalar_expectations = {
        "$.candidate_id": expected_candidate_id,
        "$.candidate_state_sha256": expected_candidate_state_sha256,
        "$.policy_sha256": expected_policy_sha256,
        "$.required_split": expected_split,
        "$.schema": LFM2_CPCF_CANDIDATE_REPLAY_OUTPUT_SCHEMA,
        "$.target_dtype": expected_target_dtype,
    }
    if set(top) != expected_top_paths or any(
        top[path].get("kind") != type(expected).__name__
        or top[path].get("value") != expected
        for path, expected in scalar_expectations.items()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 parity structure is not the exact representative "
            "candidate replay output contract."
        )
    cases_node = top["$.cases"]
    case_nodes = cases_node.get("children")
    ordered_cases = tuple(
        sorted(expected_cases, key=lambda row: row.get("case_id", ""))
    )
    if (
        cases_node.get("kind") != "sequence"
        or not isinstance(case_nodes, list)
        or not case_nodes
        or len(case_nodes) != len(ordered_cases)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 representative parity contains no exact replay cases."
        )
    replay_roles = (
        "b_history",
        "c_history",
        "context_history",
        "write_history",
        "x_history",
    )
    common_hidden_width: int | None = None
    for case_index, (case_node, expected_case) in enumerate(
        zip(case_nodes, ordered_cases, strict=True)
    ):
        case_path = f"$.cases[{case_index}]"
        if (
            not isinstance(case_node, Mapping)
            or case_node.get("path") != case_path
            or case_node.get("kind") != "mapping"
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 representative parity case structure is invalid."
            )
        case_children = children_by_path(case_node)
        required_paths = {
            *(f"{case_path}.{role}" for role in replay_roles),
            f"{case_path}.case_id",
            f"{case_path}.split",
            f"{case_path}.suffix_start",
            f"{case_path}.projection_case_authentication_sha256",
        }
        case_id = expected_case.get("case_id")
        projection_sha256 = expected_case.get("projection_case_authentication_sha256")
        if (
            set(case_children) != required_paths
            or not isinstance(case_id, str)
            or not case_id
            or _digest(
                projection_sha256,
                "CPCF v5 projection-case authentication SHA-256",
            )
            != projection_sha256
            or case_children[f"{case_path}.case_id"].get("value") != case_id
            or case_children[f"{case_path}.split"].get("value") != expected_split
            or case_children[f"{case_path}.projection_case_authentication_sha256"].get(
                "value"
            )
            != projection_sha256
            or _nonnegative_int(
                case_children[f"{case_path}.suffix_start"].get("value"),
                "CPCF v5 parity suffix_start",
            )
            < 0
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 representative parity case metadata differs from "
                "its selection evidence."
            )
        shapes = []
        for role in replay_roles:
            leaf = case_children[f"{case_path}.{role}"]
            if (
                leaf.get("kind") != "tensor"
                or leaf.get("dtype") != expected_target_dtype
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 representative parity tensor dtype or role is invalid."
                )
            shapes.append(tuple(leaf["shape"]))
        if len(set(shapes)) != 1:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 replay histories disagree on batch, sequence, or width."
            )
        hidden_width = shapes[0][-1]
        if common_hidden_width is None:
            common_hidden_width = hidden_width
        elif hidden_width != common_hidden_width:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 replay cases disagree on target hidden width."
            )
    return tuple(tensor_leaves)


def _authenticate_v5_parity_tensor_metrics(
    tensor_reports: Sequence[Any],
    *,
    tensor_leaves: Sequence[Mapping[str, Any]],
    policy: ReplaySchedulerPolicy,
) -> tuple[Mapping[str, Any], ...]:
    if not tensor_reports or len(tensor_reports) != len(tensor_leaves):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 representative parity requires nonempty tensor evidence."
        )
    leaf_by_path = {str(leaf["path"]): leaf for leaf in tensor_leaves}
    authenticated = []
    for index, value in enumerate(tensor_reports):
        if not isinstance(value, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity tensor metric row is not a mapping."
            )
        row = _canonical_copy(
            dict(value),
            f"CPCF v5 parity tensor metric[{index}]",
        )
        expected_fields = {
            "path",
            "dtype",
            "shape",
            "maximum_absolute_error",
            "allowed_absolute_error",
            "relative_rmse",
            "maximum_relative_rmse",
            "cosine_similarity",
            "minimum_cosine_similarity",
            "passed",
        }
        path = row.get("path")
        leaf = leaf_by_path.get(path)
        maximum_absolute_error = _finite_float(
            row.get("maximum_absolute_error"),
            "CPCF v5 parity maximum absolute error",
        )
        allowed_absolute_error = _finite_float(
            row.get("allowed_absolute_error"),
            "CPCF v5 parity allowed absolute error",
        )
        relative_rmse = _finite_float(
            row.get("relative_rmse"),
            "CPCF v5 parity relative RMSE",
        )
        maximum_relative_rmse = _finite_float(
            row.get("maximum_relative_rmse"),
            "CPCF v5 parity maximum relative RMSE",
        )
        cosine_similarity = _finite_float(
            row.get("cosine_similarity"),
            "CPCF v5 parity cosine similarity",
        )
        minimum_cosine_similarity = _finite_float(
            row.get("minimum_cosine_similarity"),
            "CPCF v5 parity minimum cosine similarity",
        )
        if (
            set(row) != expected_fields
            or leaf is None
            or row.get("dtype") != leaf.get("dtype")
            or row.get("shape") != leaf.get("shape")
            or maximum_absolute_error < 0.0
            or allowed_absolute_error < policy.parity_absolute_tolerance
            or maximum_absolute_error > allowed_absolute_error
            or relative_rmse < 0.0
            or maximum_relative_rmse != policy.parity_maximum_relative_rmse
            or relative_rmse > maximum_relative_rmse
            or minimum_cosine_similarity != policy.parity_minimum_cosine_similarity
            or cosine_similarity < minimum_cosine_similarity
            or cosine_similarity < -1.0000001
            or cosine_similarity > 1.0000001
            or row.get("passed") is not True
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 parity tensor metric differs from its typed leaf, "
                "scheduler tolerance, or passing numerical bound."
            )
        authenticated.append(MappingProxyType(row))
    if set(leaf_by_path) != {str(row["path"]) for row in authenticated}:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 parity tensor metrics do not cover every replay leaf exactly."
        )
    return tuple(authenticated)


def _authenticate_v5_canonical_assignments(
    plan: ReplaySchedulePlan,
    *,
    artifact: Mapping[str, Any],
) -> None:
    """Recompute shard-first round-robin from the exact ordered work manifest."""

    reports = artifact.get("candidate_selection_reports")
    if not isinstance(reports, Mapping) or not reports:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 candidate assignments have no selection work manifest."
        )
    candidate_ids = sorted(reports)
    assignments = tuple(plan.assignments)
    ordered_replicas = tuple(
        sorted(plan.replicas, key=lambda replica: replica.device.device_index)
    )
    if (
        len(assignments) != len(candidate_ids)
        or not ordered_replicas
        or tuple(assignment.ordinal for assignment in assignments)
        != tuple(range(len(assignments)))
        or len({assignment.input_payload_sha256 for assignment in assignments})
        != len(assignments)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 candidate work items are missing, duplicated, or unordered."
        )
    shard_devices: dict[str, ReplayReplicaReceipt] = {}
    for ordinal, (candidate_id, assignment) in enumerate(
        zip(candidate_ids, assignments, strict=True)
    ):
        report = reports.get(candidate_id)
        if not isinstance(report, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 candidate work item has no selection report."
            )
        expected_work_id = f"lfm2-cpcf-candidate::{candidate_id}"
        expected_replica = shard_devices.get(expected_work_id)
        if expected_replica is None:
            expected_replica = ordered_replicas[
                len(shard_devices) % len(ordered_replicas)
            ]
            shard_devices[expected_work_id] = expected_replica
        candidate_state_sha256 = report.get("candidate_state_sha256")
        if (
            assignment.ordinal != ordinal
            or assignment.work_id != expected_work_id
            or assignment.shard_key != expected_work_id
            or assignment.device_index != expected_replica.device.device_index
            or assignment.device_receipt_sha256
            != expected_replica.device.receipt_sha256
            or assignment.installed_state_sha256
            != expected_replica.installed_state.state_sha256
            or _digest(
                assignment.input_payload_sha256,
                "CPCF v5 candidate input payload SHA-256",
            )
            != assignment.input_payload_sha256
            or _digest(
                candidate_state_sha256,
                "CPCF v5 candidate state SHA-256",
            )
            != candidate_state_sha256
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 scheduler assignment differs from canonical "
                "shard-first deterministic round-robin."
            )


def _authenticate_v5_current_gpu_execution(
    short_conv_receipts: Sequence[Lfm2CpcfCompletedLayerReceipt],
    *,
    gpu_execution_audit: Any,
    gpu_candidate_replay: Any,
) -> tuple[str, ...]:
    """Prove serialized receipts contain current-run two-GPU replay."""

    policy, replicas, _runtime_sha256 = _authenticate_v5_gpu_candidate_runtime(
        gpu_candidate_replay
    )
    replica_by_device = {replica.device.device_index: replica for replica in replicas}
    execution_sha256: list[str] = []
    for receipt in short_conv_receipts:
        artifact = receipt.report["operator_artifact"]
        backend_value = artifact.get("candidate_replay_backend")
        if not isinstance(backend_value, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 short-conv receipt lacks injected GPU replay evidence."
            )
        backend, backend_sha256 = _sealed_document(
            backend_value,
            schema=LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
            seal_name="report_sha256",
            name=(f"CPCF v5 layer {receipt.layer_index} candidate replay backend"),
        )
        execution_value = backend.get("execution_report")
        if not isinstance(execution_value, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 candidate replay backend has no current execution."
            )
        execution, execution_digest = _sealed_document(
            execution_value,
            schema=DYNAMIC_REPLAY_EXECUTION_SCHEMA,
            seal_name="report_sha256",
            name=(f"CPCF v5 layer {receipt.layer_index} GPU execution"),
        )
        if set(backend) != {
            "schema",
            "kind",
            "canonical_output_work_ids",
            "canonical_input_payload_sha256",
            "canonical_output_sha256",
            "compute_device_by_work_id",
            "execution_report",
            "representative_cpu_cuda_parity_passed",
            "canonical_candidate_order_restored",
            "closed_form_solve_math_changed",
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
        } or set(execution) != {
            "schema",
            "plan_sha256",
            "plan",
            "parity_work_id",
            "parity_payload_sha256",
            "parity_reports",
            "parity_passed_before_scheduled_work",
            "scheduled_records",
            "canonical_output_work_ids",
            "canonical_output_sha256",
            "buffer_reports",
            "backend_receipts_reauthenticated",
            "installed_state_reauthenticated",
            "cpu_reference_state_reauthenticated",
            "dynamic_work_stealing",
            "canonical_result_order_restored",
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
            "folding_math_changed",
        }:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 GPU backend or execution fields are incomplete or expanded."
            )

        plan = _typed_v5_replay_plan(
            execution.get("plan"),
            policy=policy,
            replicas=replicas,
        )
        _authenticate_v5_canonical_assignments(
            plan,
            artifact=artifact,
        )
        assignments = tuple(plan.assignments)
        work_ids = [assignment.work_id for assignment in assignments]
        input_sha256 = [assignment.input_payload_sha256 for assignment in assignments]
        output_sha256 = execution.get("canonical_output_sha256")
        scheduled_records = execution.get("scheduled_records")
        parity_reports = execution.get("parity_reports")
        buffer_reports = execution.get("buffer_reports")
        compute_devices = backend.get("compute_device_by_work_id")
        if (
            not isinstance(parity_reports, list)
            or not isinstance(scheduled_records, list)
            or not isinstance(buffer_reports, list)
            or not isinstance(output_sha256, list)
            or not isinstance(compute_devices, Mapping)
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 GPU execution omits parity, assignments, buffers, "
                "outputs, or compute-device evidence."
            )

        selection_reports = artifact.get("candidate_selection_reports")
        if not isinstance(selection_reports, Mapping) or not selection_reports:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 representative parity has no candidate selection evidence."
            )
        try:
            representative_candidate_id = sorted(selection_reports)[0]
        except TypeError as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 candidate selection IDs are noncanonical."
            ) from exc
        representative_report = selection_reports.get(representative_candidate_id)
        if not isinstance(representative_report, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 representative candidate report is unavailable."
            )
        representative_cases = representative_report.get("cases")
        if not isinstance(representative_cases, list) or not representative_cases:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 representative candidate has no replay cases."
            )
        parity_devices = []
        common_cpu_output_sha256: str | None = None
        common_structure_sha256: str | None = None
        common_structure_descriptor: Any | None = None
        common_tensor_identity: Any | None = None
        for index, parity_value in enumerate(parity_reports):
            if not isinstance(parity_value, Mapping):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 GPU parity report is not a mapping."
                )
            parity, _parity_sha256 = _sealed_document(
                parity_value,
                schema=DYNAMIC_REPLAY_PARITY_SCHEMA,
                seal_name="report_sha256",
                name=f"CPCF v5 GPU parity report[{index}]",
            )
            tensor_reports = parity.get("tensor_reports")
            structure_descriptor = parity.get("structure_descriptor")
            if (
                set(parity)
                != {
                    "schema",
                    "device_index",
                    "cpu_output_sha256",
                    "cuda_output_sha256",
                    "tensor_reports",
                    "tensor_leaf_count",
                    "structure_descriptor",
                    "structure_sha256",
                    "non_tensor_output_exact_match",
                    "passed",
                }
                or not isinstance(tensor_reports, list)
                or parity.get("tensor_leaf_count") != len(tensor_reports)
                or not tensor_reports
                or parity.get("passed") is not True
                or parity.get("non_tensor_output_exact_match") is not True
                or _digest(
                    parity.get("cpu_output_sha256"),
                    "CPCF v5 CPU parity output SHA-256",
                )
                != parity.get("cpu_output_sha256")
                or _digest(
                    parity.get("cuda_output_sha256"),
                    "CPCF v5 CUDA parity output SHA-256",
                )
                != parity.get("cuda_output_sha256")
                or _digest(
                    parity.get("structure_sha256"),
                    "CPCF v5 parity structure SHA-256",
                )
                != parity.get("structure_sha256")
                or canonical_sha256(structure_descriptor)
                != parity.get("structure_sha256")
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 GPU parity report failed its exact schema, "
                    "seal, output, tensor, or pass contract."
                )
            tensor_leaves = _typed_v5_parity_structure(
                structure_descriptor,
                expected_candidate_id=representative_candidate_id,
                expected_candidate_state_sha256=representative_report.get(
                    "candidate_state_sha256"
                ),
                expected_policy_sha256=representative_report.get("policy_sha256"),
                expected_target_dtype=representative_report.get("target_dtype"),
                expected_split=representative_report.get("split"),
                expected_cases=tuple(representative_cases),
            )
            authenticated_metrics = _authenticate_v5_parity_tensor_metrics(
                tensor_reports,
                tensor_leaves=tensor_leaves,
                policy=policy,
            )
            tensor_identity = [
                {
                    "path": row["path"],
                    "dtype": row["dtype"],
                    "shape": row["shape"],
                    "maximum_relative_rmse": row["maximum_relative_rmse"],
                    "minimum_cosine_similarity": row["minimum_cosine_similarity"],
                }
                for row in authenticated_metrics
            ]
            if common_cpu_output_sha256 is None:
                common_cpu_output_sha256 = parity["cpu_output_sha256"]
                common_structure_sha256 = parity["structure_sha256"]
                common_structure_descriptor = structure_descriptor
                common_tensor_identity = tensor_identity
            elif (
                parity["cpu_output_sha256"] != common_cpu_output_sha256
                or parity["structure_sha256"] != common_structure_sha256
                or canonical_json_bytes(structure_descriptor)
                != canonical_json_bytes(common_structure_descriptor)
                or canonical_json_bytes(tensor_identity)
                != canonical_json_bytes(common_tensor_identity)
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 device parity rows do not share the one CPU "
                    "reference output, structure, and tensor-leaf identity."
                )
            parity_devices.append(parity.get("device_index"))

        if len(scheduled_records) != len(assignments) or len(output_sha256) != len(
            assignments
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 scheduled-record count differs from its plan."
            )
        device_record_counts = {0: 0, 1: 0}
        for index, (record_value, assignment) in enumerate(
            zip(scheduled_records, assignments, strict=True)
        ):
            if not isinstance(record_value, Mapping):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 scheduled record is not a mapping."
                )
            record = _canonical_copy(
                dict(record_value),
                f"CPCF v5 scheduled record[{index}]",
            )
            record_body = dict(record)
            record_sha256 = _digest(
                record_body.pop("record_sha256", None),
                f"CPCF v5 scheduled record[{index}] SHA-256",
            )
            device_count = device_record_counts.get(assignment.device_index)
            if device_count is None:
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 scheduled assignment names an unknown device."
                )
            expected_record = {
                "ordinal": assignment.ordinal,
                "work_id": assignment.work_id,
                "shard_key": assignment.shard_key,
                "device_index": assignment.device_index,
                "device_receipt_sha256": (assignment.device_receipt_sha256),
                "installed_state_sha256": (assignment.installed_state_sha256),
                "input_payload_sha256": (assignment.input_payload_sha256),
                "output_sha256": output_sha256[index],
                "slot_index": (device_count % policy.pinned_buffer_slots_per_device),
                "folding_math_changed": False,
            }
            device_record_counts[assignment.device_index] += 1
            if (
                canonical_sha256(record_body) != record_sha256
                or record_body != expected_record
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 scheduled record differs from its exact plan, "
                    "device, input, output, slot, or seal."
                )

        buffer_devices = []
        expected_stage_counts = {
            replica.device.device_index: (
                device_record_counts[replica.device.device_index] + 1
            )
            for replica in replicas
        }
        for index, buffer_value in enumerate(buffer_reports):
            if not isinstance(buffer_value, Mapping):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 pinned-host buffer report is not a mapping."
                )
            buffer, _buffer_sha256 = _sealed_document(
                buffer_value,
                schema=DYNAMIC_REPLAY_BUFFER_SCHEMA,
                seal_name="report_sha256",
                name=f"CPCF v5 pinned-host buffer report[{index}]",
            )
            device_index = buffer.get("device_index")
            replica = replica_by_device.get(device_index)
            if (
                set(buffer)
                != {
                    "schema",
                    "device_index",
                    "device_receipt_sha256",
                    "slot_count",
                    "maximum_bytes_per_slot",
                    "maximum_total_pinned_bytes",
                    "maximum_observed_slot_bytes",
                    "stage_count",
                    "pinned_host_memory",
                    "asynchronous_h2d",
                    "canonical_pinned_payload_hash_verified",
                }
                or replica is None
                or buffer.get("device_receipt_sha256") != replica.device.receipt_sha256
                or buffer.get("slot_count") != policy.pinned_buffer_slots_per_device
                or buffer.get("maximum_bytes_per_slot")
                != policy.maximum_pinned_bytes_per_slot
                or buffer.get("maximum_total_pinned_bytes")
                != (
                    policy.pinned_buffer_slots_per_device
                    * policy.maximum_pinned_bytes_per_slot
                )
                or _nonnegative_int(
                    buffer.get("maximum_observed_slot_bytes"),
                    "CPCF v5 maximum observed pinned bytes",
                )
                > policy.maximum_pinned_bytes_per_slot
                or _nonnegative_int(
                    buffer.get("stage_count"),
                    "CPCF v5 pinned-host stage count",
                )
                != expected_stage_counts[device_index]
                or buffer.get("pinned_host_memory") is not True
                or buffer.get("asynchronous_h2d") is not True
                or buffer.get("canonical_pinned_payload_hash_verified") is not True
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 pinned-host buffer report differs from its "
                    "replica, policy, bound, or authenticated staging "
                    "contract."
                )
            buffer_devices.append(device_index)

        required_zero_fields = (
            "training_steps",
            "backward_calls",
            "optimizer_steps",
            "parameters_pruned",
            "parameters_quantized",
        )
        if (
            artifact.get("candidate_replay_backend_report_sha256") != backend_sha256
            or artifact.get("selection_candidate_replay_backend")
            != "authenticated_injected_backend"
            or artifact.get("fit_and_validation_winner_replay_backend")
            != "cpu_reference"
            or execution.get("plan_sha256") != plan.plan_sha256
            or execution.get("canonical_output_work_ids") != work_ids
            or len(output_sha256) != len(work_ids)
            or any(
                _digest(value, "CPCF v5 canonical GPU output SHA-256") != value
                for value in output_sha256
            )
            or backend.get("kind") != "authenticated_two_cuda_scheduler"
            or backend.get("canonical_output_work_ids") != work_ids
            or backend.get("canonical_input_payload_sha256") != input_sha256
            or backend.get("canonical_output_sha256") != output_sha256
            or backend.get("representative_cpu_cuda_parity_passed") is not True
            or backend.get("canonical_candidate_order_restored") is not True
            or backend.get("closed_form_solve_math_changed") is not False
            or any(backend.get(name) != 0 for name in required_zero_fields)
            or dict(compute_devices)
            != {
                assignment.work_id: f"cuda:{assignment.device_index}"
                for assignment in assignments
            }
            or parity_devices != [0, 1]
            or buffer_devices != [0, 1]
            or not assignments
            or any(
                assignment.device_index not in replica_by_device
                or assignment.device_receipt_sha256
                != replica_by_device[assignment.device_index].device.receipt_sha256
                or assignment.installed_state_sha256
                != replica_by_device[
                    assignment.device_index
                ].installed_state.state_sha256
                for assignment in assignments
            )
            or execution.get("parity_passed_before_scheduled_work") is not True
            or execution.get("backend_receipts_reauthenticated") is not True
            or execution.get("installed_state_reauthenticated") is not True
            or execution.get("cpu_reference_state_reauthenticated") is not True
            or execution.get("canonical_result_order_restored") is not True
            or execution.get("dynamic_work_stealing") is not False
            or execution.get("folding_math_changed") is not False
            or any(execution.get(name) != 0 for name in required_zero_fields)
            or _digest(
                execution.get("parity_payload_sha256"),
                "CPCF v5 parity payload SHA-256",
            )
            != assignments[0].input_payload_sha256
            or execution.get("parity_work_id")
            != "lfm2-cpcf-candidate::representative-parity"
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 candidate replay lacks authenticated current-run "
                "two-GPU parity, scheduled CUDA work, or zero-training "
                "execution."
            )
        execution_sha256.append(execution_digest)

    if not execution_sha256 or not isinstance(gpu_execution_audit, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run receipt lacks current-run GPU execution evidence."
        )
    expected_audit = {
        "candidate_gpu_replay_invoked": True,
        "execution_count": len(execution_sha256),
        "execution_report_sha256": execution_sha256,
        "both_logical_devices_reauthenticated_per_execution": True,
        "no_gpu_work_claimed_when_no_candidate_batch_was_scheduled": True,
    }
    if canonical_json_bytes(gpu_execution_audit) != canonical_json_bytes(
        expected_audit
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 GPU execution audit differs from the exact serialized "
            "layer executions."
        )
    return tuple(execution_sha256)


def _verify_v5_compiled_model_fold_semantics(
    body: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    layer_results: Any,
    gpu_execution_audit: Any,
    gpu_candidate_replay: Any,
) -> None:
    """Recompute v5 numerical, selector, layer, and GPU semantics."""

    if not isinstance(layer_results, list) or not layer_results:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 run receipt has no serialized layer receipts."
        )
    try:
        receipts = tuple(
            Lfm2CpcfCompletedLayerReceipt(report) for report in layer_results
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 serialized layer receipts are not exact typed receipts."
        ) from exc
    if tuple(receipt.layer_index for receipt in receipts) != tuple(
        range(len(receipts))
    ) or body.get("ordered_layer_result_sha256") != [
        receipt.result_sha256 for receipt in receipts
    ]:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 top-level report and serialized ordered layers differ."
        )
    short_conv_receipts = tuple(
        receipt for receipt in receipts if receipt.operator_kind == "short_conv"
    )
    short_conv_indices = [receipt.layer_index for receipt in short_conv_receipts]
    if not short_conv_receipts:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 report proves no short-conv numerical folding."
        )

    contract = _authenticate_v5_numerical_adapter_contract(
        body.get("short_conv_v5_real_numerical_adapter_contract")
    )
    recomputed_proofs = []
    selector_indices = []
    for receipt in short_conv_receipts:
        proof = _prove_real_v5_numerical_layer(
            receipt,
            adapter_contract=contract,
        )
        if not isinstance(proof, Mapping):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 short-conv layer lacks concrete real numerical "
                "adapter provenance."
            )
        recomputed_proofs.append(dict(proof))

        artifact = receipt.report["operator_artifact"]
        selection_reports = artifact.get("candidate_selection_reports")
        selector_value = artifact.get("preinstall_candidate_selector_receipt")
        if not isinstance(selection_reports, Mapping) or not isinstance(
            selector_value,
            Mapping,
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 short-conv layer lacks an endpoint-first selector "
                "frontier and receipt."
            )
        try:
            selector = rescore_sealed_v4_short_conv_selection_frontier(
                selection_reports
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 endpoint-first selector frontier cannot be "
                "independently rescored."
            ) from exc
        if (
            canonical_json_bytes(selector_value) != canonical_json_bytes(dict(selector))
            or artifact.get("preinstall_candidate_selector_receipt_sha256")
            != selector["report_sha256"]
            or artifact.get("preinstall_candidate_selector_active") is not True
            or artifact.get("selected_candidate_id")
            != selector["selected_candidate_id"]
            or artifact.get("default_v4_selected_candidate_id") is None
            or artifact.get("post_install_rescoring_performed") is not False
            or artifact.get("legacy_v4_report_claimed") is not False
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 endpoint-first selector receipt differs from a "
                "fresh deterministic rescore."
            )
        selector_indices.append(receipt.layer_index)

    proofs = body.get("short_conv_v5_real_numerical_layer_proofs")
    if (
        policy.get("require_short_conv_v5_endpoint_first_selector") is not True
        or policy.get("require_short_conv_v5_real_numerical_frontier_adapter")
        is not True
        or body.get("candidate_replay_backend_injected") is not True
        or body.get("preinstall_candidate_selector_provided") is not True
        or body.get("preinstall_candidate_selector_active") is not True
        or body.get("preinstall_candidate_selector_layer_indices") != selector_indices
        or body.get("short_conv_v5_endpoint_first_selector_integrated") is not True
        or body.get("short_conv_numerical_frontier_adapter_provided") is not True
        or body.get("short_conv_numerical_frontier_adapter_active") is not True
        or body.get("short_conv_numerical_frontier_adapter_layer_indices")
        != short_conv_indices
        or body.get("short_conv_layer_indices") != short_conv_indices
        or body.get("short_conv_v5_real_numerical_adapter_required") is not True
        or body.get("short_conv_v5_real_numerical_adapter_concrete_instance_provided")
        is not True
        or body.get("short_conv_v5_real_numerical_provenance_layer_indices")
        != short_conv_indices
        or proofs != recomputed_proofs
        or body.get("short_conv_v5_numerical_candidate_generation_integrated")
        is not True
        or body.get("short_conv_v5_numerical_policy_integrated") is not True
        or body.get("short_conv_v5_post_install_rescoring_performed") is not False
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 report lacks its complete numerical policy, every "
            "recomputed per-layer proof, or endpoint-first selector "
            "integration."
        )
    _authenticate_v5_current_gpu_execution(
        short_conv_receipts,
        gpu_execution_audit=gpu_execution_audit,
        gpu_candidate_replay=gpu_candidate_replay,
    )


def _verify_v5_compiled_model_fold_metadata(
    body: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    layer_results: Any,
    gpu_execution_audit: Any,
    gpu_candidate_replay: Any,
) -> None:
    """Authenticate the source-bound selective-residency v5 envelope."""

    implementation_document = body.get("compiler_implementation")
    if not isinstance(implementation_document, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold has no compiler implementation seal."
        )
    _implementation_body, _implementation_sha256 = _sealed_document(
        implementation_document,
        schema=LFM2_CPCF_V5_COMPILER_IMPLEMENTATION_SCHEMA,
        seal_name="implementation_sha256",
        name="CPCF v5 compiler implementation",
    )
    live_implementation = dict(lfm2_cpcf_v5_compiler_implementation())
    if canonical_json_bytes(implementation_document) != canonical_json_bytes(
        live_implementation
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiler implementation differs from the live "
            "source-bound implementation."
        )

    residency_value = body.get("residency_receipt")
    physical_value = body.get("physical_rss_receipt")
    if not isinstance(residency_value, Mapping) or not isinstance(
        physical_value,
        Mapping,
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold lacks sealed residency evidence."
        )
    residency, _residency_sha256 = _sealed_document(
        residency_value,
        schema=EVIDENCE_RESIDENCY_WINDOW_SCHEMA,
        seal_name="snapshot_sha256",
        name="CPCF v5 residency receipt",
    )
    physical, _physical_sha256 = _sealed_document(
        physical_value,
        schema=LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
        seal_name="report_sha256",
        name="CPCF v5 physical RSS receipt",
    )

    catalog_sha256 = _digest(
        body.get("selective_catalog_sha256"),
        "CPCF v5 selective catalog SHA-256",
    )
    source_sha256 = _digest(
        body.get("selective_source_sha256"),
        "CPCF v5 selective source SHA-256",
    )
    provider_sha256 = _digest(
        body.get("selective_provider_sha256"),
        "CPCF v5 selective provider SHA-256",
    )
    logical_bytes = _positive_int(
        body.get("authenticated_full_phase_tree_logical_bytes"),
        "CPCF v5 authenticated full phase-tree logical bytes",
    )
    ordered_layer_results = body.get("ordered_layer_result_sha256")
    resume_publication = body.get("resume_publication_enabled")
    resume_used = body.get("resume_used")
    resume_start = body.get("resume_start_layer_index")
    admitted_prefixes = body.get("resume_admitted_prefixes")
    layer_count = (
        -1
        if not isinstance(ordered_layer_results, list)
        else len(ordered_layer_results)
    )

    def authenticate_admission_proof_compiler_binding(
        chain_proof: Mapping[str, Any],
    ) -> None:
        proof_frontier = chain_proof.get("boundary_reports")
        proof_context = (
            None
            if not isinstance(proof_frontier, Mapping)
            else proof_frontier.get("context")
        )
        if not isinstance(proof_frontier, Mapping) or not isinstance(
            proof_context,
            Mapping,
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 admission-chain proof lacks compiler-bound "
                "frontier context."
            )
        try:
            physical_rss_policy_sha256 = (
                build_lfm2_cpcf_v5_physical_rss_policy_sha256(
                    full_phase_tree_logical_bytes=policy[
                        "full_phase_tree_logical_bytes"
                    ],
                    minimum_reduction_fraction=policy[
                        "minimum_physical_rss_reduction_fraction"
                    ],
                    maximum_increase_bytes=policy[
                        "maximum_physical_rss_increase_bytes"
                    ],
                    maximum_process_rss_bytes=policy[
                        "maximum_process_rss_bytes"
                    ],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 policy cannot authenticate an admission-chain proof."
            ) from exc
        if (
            proof_frontier.get("expected_layer_count") != layer_count
            or proof_frontier.get("ordered_phase_sha256")
            != body.get("phase_sha256")
            or proof_context.get("selective_catalog_sha256")
            != body.get("selective_catalog_sha256")
            or proof_context.get("selective_source_sha256")
            != body.get("selective_source_sha256")
            or proof_context.get("selective_provider_sha256")
            != body.get("selective_provider_sha256")
            or proof_context.get("runtime_policy_sha256")
            != canonical_sha256(policy)
            or proof_context.get("physical_rss_policy_sha256")
            != physical_rss_policy_sha256
            or proof_context.get("compiler_implementation_sha256")
            != live_implementation["implementation_sha256"]
            or proof_context.get("resume_implementation_sha256")
            != lfm2_cpcf_v5_resume_implementation()["implementation_sha256"]
            or proof_context.get("bootstrap_authentication_sha256")
            != body.get("bootstrap_authentication_sha256")
            or proof_context.get("final_norm_authentication_sha256")
            != body.get("final_norm_authentication_sha256")
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 admission-chain proof belongs to a different "
                "compiler execution."
            )

    if (
        not isinstance(resume_publication, bool)
        or not isinstance(resume_used, bool)
        or (resume_used and not resume_publication)
        or isinstance(resume_start, bool)
        or not isinstance(resume_start, int)
        or resume_start < 0
        or layer_count < 0
        or resume_start > layer_count
        or body.get("resume_completed_prefix_layer_indices")
        != list(range(resume_start))
        or body.get("resume_executed_layer_indices")
        != list(range(resume_start, layer_count))
        or body.get("resume_completed_layers_replayed") is not False
        or body.get("resume_actual_upstream_child_state_restored") is not resume_used
        or body.get("resume_checkpoint_receipt_chain_authenticated") is not True
        or body.get("resume_parent_or_cache_tensor_sidecars") != 0
        or not isinstance(admitted_prefixes, list)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold has an invalid fail-closed resume "
            "execution state machine."
        )
    for digest_field in (
        "resume_prefix_install_report_sha256",
        "resume_frontier_sha256",
        "resume_admission_sha256",
    ):
        value = body.get(digest_field)
        if resume_used:
            _digest(value, f"CPCF v5 {digest_field}")
        elif value is not None:
            raise Lfm2CpcfReviewedPromotionError(
                f"Cold CPCF v5 report unexpectedly populated {digest_field}."
            )
    input_proof_value = body.get("resume_input_admission_chain_proof")
    input_proof_sha256_value = body.get(
        "resume_input_admission_chain_proof_sha256"
    )
    if resume_used:
        try:
            input_proof = dict(
                authenticate_lfm2_cpcf_v5_admission_chain_proof(
                    input_proof_value
                )
            )
        except Exception as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 resumed input has no valid standalone "
                "admission-chain proof."
            ) from exc
        input_proof_sha256 = _digest(
            input_proof_sha256_value,
            "CPCF v5 input admission-chain proof SHA-256",
        )
        input_pending = input_proof["pending_document"]
        input_admission = input_proof["admission_document"]
        authenticate_admission_proof_compiler_binding(input_proof)
        if (
            input_proof["proof_sha256"] != input_proof_sha256
            or input_pending.get("next_layer_index") != resume_start
            or input_pending.get("frontier_sha256")
            != body.get("resume_frontier_sha256")
            or input_admission.get("admission_sha256")
            != body.get("resume_admission_sha256")
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 resumed input differs from its standalone "
                "admission-chain proof."
            )
    elif (
        input_proof_value is not None
        or input_proof_sha256_value is not None
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Cold CPCF v5 report unexpectedly carries an input "
            "admission-chain proof."
        )
    expected_admitted_indices = (
        list(
            range(
                resume_start + (1 if resume_used else 0),
                layer_count + 1,
            )
        )
        if resume_publication
        else []
    )
    observed_admitted_indices = [
        record.get("next_layer_index")
        for record in admitted_prefixes
        if isinstance(record, Mapping)
    ]
    if observed_admitted_indices != expected_admitted_indices or len(
        admitted_prefixes
    ) != len(expected_admitted_indices):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 admitted resume prefixes are not the exact executed "
            "publication suffix."
        )
    for record in admitted_prefixes:
        base_record_fields = {
            "next_layer_index",
            "frontier_reference",
            "checkpoint_reference",
            "frontier_sha256",
            "admission_sha256",
        }
        proof_fields = {
            "admission_chain_proof",
            "admission_chain_proof_sha256",
        }
        proof_present = isinstance(record, Mapping) and bool(
            set(record).intersection(proof_fields)
        )
        if (
            not isinstance(record, Mapping)
            or not proof_present
            or set(record)
            != base_record_fields | (proof_fields if proof_present else set())
            or not isinstance(record["frontier_reference"], Mapping)
            or not isinstance(record["checkpoint_reference"], Mapping)
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 admitted resume-prefix record is incomplete."
            )
        try:
            frontier_reference = Lfm2CpcfArtifactReference(
                **dict(record["frontier_reference"])
            )
            checkpoint_reference = Lfm2CpcfArtifactReference(
                **dict(record["checkpoint_reference"])
            )
        except Exception as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 admitted resume-prefix record has invalid typed "
                "artifact references."
            ) from exc
        if (
            frontier_reference.artifact_kind != "boundary"
            or checkpoint_reference.artifact_kind != "checkpoint"
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "CPCF v5 admitted resume-prefix artifact reference kinds "
                "are invalid."
            )
        frontier_sha256 = _digest(
            record["frontier_sha256"],
            "CPCF v5 admitted frontier SHA-256",
        )
        admission_sha256 = _digest(
            record["admission_sha256"],
            "CPCF v5 admitted admission SHA-256",
        )
        if proof_present:
            try:
                chain_proof = dict(
                    authenticate_lfm2_cpcf_v5_admission_chain_proof(
                        record["admission_chain_proof"]
                    )
                )
            except Exception as exc:
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 admitted resume prefix has an invalid "
                    "standalone admission-chain proof."
                ) from exc
            proof_sha256 = _digest(
                record["admission_chain_proof_sha256"],
                "CPCF v5 admission-chain proof SHA-256",
            )
            proof_pending = chain_proof["pending_document"]
            proof_admission = chain_proof["admission_document"]
            authenticate_admission_proof_compiler_binding(chain_proof)
            if (
                chain_proof["proof_sha256"] != proof_sha256
                or proof_pending.get("next_layer_index")
                != record["next_layer_index"]
                or proof_pending.get("frontier_reference")
                != frontier_reference.as_dict()
                or proof_pending.get("checkpoint_reference")
                != checkpoint_reference.as_dict()
                or proof_pending.get("frontier_sha256") != frontier_sha256
                or proof_admission.get("admission_sha256")
                != admission_sha256
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    "CPCF v5 admitted resume-prefix record differs from "
                    "its standalone admission-chain proof."
                )
    if (
        body.get("artifact_schema") != LFM2_CPCF_V5_ARTIFACT_SCHEMA
        or body.get("resume_schema") != LFM2_CPCF_V5_RESUME_SCHEMA
        or body.get("resume_supported") is not True
        or body.get("resume_fail_closed") is not True
        or body.get("physical_rss_baseline_bound_to_authenticated_catalog") is not True
        or body.get("full_all_layer_phase_tree_materialized") is not False
        or body.get("tensor_free_phase_shells_used") is not True
        or body.get("active_layer_evidence_released_before_successor_compile")
        is not True
        or body.get("final_norm_evidence_separately_bounded_and_reauthenticated")
        is not True
        or body.get("legacy_v4_default_or_fingerprint_changed") is not False
        or residency.get("catalog_sha256") != catalog_sha256
        or residency.get("source_sha256") != source_sha256
        or residency.get("provider_sha256") != provider_sha256
        or residency.get("max_resident_layers") != 2
        or residency.get("prefetch_distance_layers") != 1
        or residency.get("resident_bytes") != 0
        or residency.get("peak_resident_bytes", 1)
        > residency.get("maximum_resident_bytes", 0)
        or residency.get("maximum_resident_bytes")
        != policy["maximum_resident_evidence_bytes"]
        or logical_bytes != policy["full_phase_tree_logical_bytes"]
        or physical.get("full_phase_tree_logical_bytes") != logical_bytes
        or physical.get("minimum_reduction_fraction")
        != policy["minimum_physical_rss_reduction_fraction"]
        or physical.get("maximum_increase_bytes")
        != policy["maximum_physical_rss_increase_bytes"]
        or physical.get("sample_interval_seconds")
        != policy["rss_sample_interval_seconds"]
        or physical.get("sampler_stopped") is not True
        or physical.get("physical_rss_gate_passed") is not True
        or physical.get("peak_increase_bytes", 1)
        > physical.get("maximum_increase_bytes", 0)
        or physical.get("current_increase_bytes", 1)
        > physical.get("maximum_increase_bytes", 0)
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF v5 compiled model fold lacks its exact compiler, bounded "
            "residency, physical-RSS, or fail-closed resume invariants."
        )
    _verify_v5_compiled_model_fold_semantics(
        body,
        policy=policy,
        layer_results=layer_results,
        gpu_execution_audit=gpu_execution_audit,
        gpu_candidate_replay=gpu_candidate_replay,
    )


def _ordered_phase_binding_body(
    *,
    phase_count: int,
    ordered_phase_sha256: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": ORDERED_PHASE_BINDING_SCHEMA,
        "phase_count": phase_count,
        "ordered_phase_sha256": list(ordered_phase_sha256),
    }


def _verify_blank_target_report(
    value: Any,
    *,
    source_unique_parameters: int,
    target_unique_parameters: int,
) -> tuple[dict[str, Any], str]:
    """Authenticate the exact compiler-created zero blank named by the run."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF run receipt has no blank-target report."
        )
    document = _canonical_copy(dict(value), "CPCF blank-target report")
    body, report_sha256 = _sealed_document(
        document,
        schema=CPCF_BLANK_TARGET_SCHEMA,
        seal_name="report_sha256",
        name="CPCF blank-target report",
    )
    ratio = body.get("realized_compression_ratio")
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or float(ratio) != source_unique_parameters / target_unique_parameters
        or body.get("parent_unique_parameters") != source_unique_parameters
        or body.get("target_unique_parameters") != target_unique_parameters
        or body.get("compiler_created_blank") is not True
        or body.get("all_parameters_zero") is not True
        or body.get("dense_materialized_parameters") is not True
        or body.get("topology_provenance_verified") is not True
        or body.get("parent_state_preserved") is not True
        or body.get("quantization_used") is not False
        or body.get("pruning_used") is not False
        or body.get("training_used") is not False
        or body.get("weights_installed") is not False
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF blank-target report does not prove the exact dense zero blank."
        )
    return body, report_sha256


def _verify_compiled_model_fold(
    value: Any,
    *,
    layer_results: Any,
    gpu_execution_audit: Any,
    gpu_candidate_replay: Any,
    blank_target_report_sha256: str,
    parent_inventory: Mapping[str, Any],
    blank_inventory: Mapping[str, Any],
    compiled_inventory: Mapping[str, Any],
    reloaded_inventory: Mapping[str, Any],
    compression_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and cross-bind the real model compiler's nested report."""

    if not isinstance(value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF run receipt has no compiled_model_fold report."
        )
    document = _canonical_copy(dict(value), "CPCF compiled model-fold report")
    compiled_model_fold_schema = document.get("schema")
    if compiled_model_fold_schema not in SUPPORTED_COMPILED_MODEL_FOLD_SCHEMAS:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold schema is unsupported."
        )
    body, result_sha256 = _sealed_document(
        document,
        schema=str(compiled_model_fold_schema),
        seal_name="result_sha256",
        name="CPCF compiled model-fold report",
    )
    if compiled_model_fold_schema == LFM2_CPCF_MODEL_FOLD_SCHEMA:
        policy = _normalize_model_fold_policy(body.get("policy"))
    else:
        policy = _normalize_v5_model_fold_policy(body.get("policy"))
        _verify_v5_compiled_model_fold_metadata(
            body,
            policy=policy,
            layer_results=layer_results,
            gpu_execution_audit=gpu_execution_audit,
            gpu_candidate_replay=gpu_candidate_replay,
        )
    policy_sha256 = canonical_sha256(policy)
    phase_count = _positive_int(
        body.get("phase_count"),
        "CPCF compiled model-fold phase_count",
    )
    phase_value = body.get("phase_sha256")
    if not isinstance(phase_value, list) or len(phase_value) != phase_count:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold phase seals do not cover every phase."
        )
    phase_sha256 = tuple(
        _digest(value, "CPCF compiled model-fold phase SHA-256")
        for value in phase_value
    )
    if len(set(phase_sha256)) != len(phase_sha256):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model-fold ordered phase seals repeat."
        )
    phase_binding_sha256 = canonical_sha256(
        _ordered_phase_binding_body(
            phase_count=phase_count,
            ordered_phase_sha256=phase_sha256,
        )
    )
    for field_name in (
        "boundary_bundle_sha256",
        "interface_install_report_sha256",
        "neutral_norm_install_report_sha256",
        "final_state_bank_sha256",
        "final_norm_report_sha256",
    ):
        _digest(
            body.get(field_name),
            f"CPCF compiled model-fold {field_name}",
        )
    layer_results = body.get("ordered_layer_result_sha256")
    if not isinstance(layer_results, list) or not layer_results:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model fold has no ordered layer results."
        )
    for value in layer_results:
        _digest(value, "CPCF compiled model-fold layer result SHA-256")
    operation_audit = body.get("operation_audit")
    required_audit = {
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
    }
    if (
        body.get("blank_target_report_sha256") != blank_target_report_sha256
        or body.get("source_state_preserved") is not True
        or body.get("target_parameter_count_preserved") is not True
        or body.get("all_layers_compiled_in_order") is not True
        or body.get("actual_folded_upstream_replay_used") is not True
        or not isinstance(operation_audit, Mapping)
        or any(
            operation_audit.get(name) != expected
            for name, expected in required_audit.items()
        )
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model fold lacks its blank, folding, or zero-operation "
            "invariants."
        )
    parameter_receipt, parameter_receipt_sha256 = _verify_parameter_receipt(
        body.get("parameter_receipt"),
        name="CPCF compiled model-fold parameter receipt",
    )
    before_inventory, _before_sha256 = _verify_parameter_inventory(
        body.get("target_parameter_inventory_before"),
        name="CPCF compiled model-fold target inventory before",
    )
    after_inventory, _after_sha256 = _verify_parameter_inventory(
        body.get("target_parameter_inventory_after"),
        name="CPCF compiled model-fold target inventory after",
    )
    receipt_source = parameter_receipt["source_inventory"]
    receipt_target = parameter_receipt["target_inventory"]
    compression_source = compression_receipt["source_inventory"]
    compression_target = compression_receipt["target_inventory"]
    exact_bindings = (
        (receipt_source, parent_inventory),
        (compression_source, parent_inventory),
        (receipt_target, blank_inventory),
        (before_inventory, blank_inventory),
        (after_inventory, compiled_inventory),
        (compression_target, reloaded_inventory),
        (parameter_receipt, compression_receipt),
    )
    if any(
        canonical_json_bytes(left) != canonical_json_bytes(right)
        for left, right in exact_bindings
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF compiled model fold, blank, compiled child, reload, and "
            "parameter receipts do not share exact inventories."
        )
    return {
        "compiled_model_fold_schema": compiled_model_fold_schema,
        "compiled_model_fold_result_sha256": result_sha256,
        "compiled_model_fold_policy": policy,
        "compiled_model_fold_policy_sha256": policy_sha256,
        "compiled_model_fold_phase_count": phase_count,
        "compiled_model_fold_phase_sha256": phase_sha256,
        "compiled_model_fold_phase_binding_sha256": phase_binding_sha256,
        "compiled_blank_target_report_sha256": blank_target_report_sha256,
        "compiled_parameter_receipt_sha256": parameter_receipt_sha256,
    }


def _target_plan_binding_body(
    *,
    parent_graph_fingerprint: str,
    exact_target_unique_parameters: int,
    requested_budget: Mapping[str, Any],
    frontier_fingerprint: str,
    target_certificate_sha256: str,
    selected_candidate_fingerprint: str,
    selected_candidate_rank: int,
    target_topology_fingerprint: str,
    blank_state_fingerprint: str,
) -> dict[str, Any]:
    """Canonical identity of the exact catalog choice that made the blank."""

    return {
        "schema": "neural-forge-lfm2-cpcf-target-plan-binding-v1",
        "parent_graph_fingerprint": parent_graph_fingerprint,
        "exact_target_unique_parameters": exact_target_unique_parameters,
        "requested_budget": dict(requested_budget),
        "frontier_fingerprint": frontier_fingerprint,
        "target_certificate_sha256": target_certificate_sha256,
        "selected_candidate_fingerprint": selected_candidate_fingerprint,
        "selected_candidate_rank": selected_candidate_rank,
        "target_topology_fingerprint": target_topology_fingerprint,
        "blank_state_fingerprint": blank_state_fingerprint,
    }


def _extract_cpcf_run_provenance(
    body: Mapping[str, Any],
    *,
    expected_parent_model_fingerprint: str,
) -> dict[str, Any]:
    """Validate and normalize the production-v3 run identity."""

    inputs = body.get("inputs")
    blank = body.get("blank_target")
    parent_inventory = body.get("parent_parameter_inventory")
    blank_inventory = body.get("blank_parameter_inventory")
    compiled_inventory = body.get("compiled_parameter_inventory")
    reloaded_inventory = body.get("reloaded_parameter_inventory")
    compression = body.get("compression")
    if not all(
        isinstance(value, Mapping)
        for value in (
            inputs,
            blank,
            parent_inventory,
            blank_inventory,
            compiled_inventory,
            reloaded_inventory,
            compression,
        )
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF run receipt lacks its production-v3 input, blank, "
            "inventory, or compression binding."
        )
    assert isinstance(inputs, Mapping)
    assert isinstance(blank, Mapping)
    assert isinstance(parent_inventory, Mapping)
    assert isinstance(blank_inventory, Mapping)
    assert isinstance(compiled_inventory, Mapping)
    assert isinstance(reloaded_inventory, Mapping)
    assert isinstance(compression, Mapping)

    parent_inventory_document, _parent_inventory_sha256 = _verify_parameter_inventory(
        parent_inventory,
        name="CPCF parent parameter inventory",
    )
    blank_inventory_document, _blank_inventory_sha256 = _verify_parameter_inventory(
        blank_inventory,
        name="CPCF blank parameter inventory",
    )
    compiled_inventory_document, _compiled_inventory_sha256 = (
        _verify_parameter_inventory(
            compiled_inventory,
            name="CPCF compiled parameter inventory",
        )
    )
    reloaded_inventory_document, _reloaded_inventory_sha256 = (
        _verify_parameter_inventory(
            reloaded_inventory,
            name="CPCF reloaded parameter inventory",
        )
    )
    compression_document, _compression_receipt_sha256 = _verify_parameter_receipt(
        compression,
        name="CPCF reload compression receipt",
    )

    cache_key = _digest(inputs.get("cache_key"), "CPCF parent cache key")
    cache_live_parent = _digest(
        inputs.get("parent_fingerprint"),
        "CPCF cache/live parent fingerprint",
    )
    if cache_live_parent != _digest(
        expected_parent_model_fingerprint,
        "holdout parent model fingerprint",
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF cache/live parent differs from the reviewed holdout parent."
        )
    trace_adapter = _digest(
        inputs.get("trace_adapter_sha256"),
        "CPCF trace adapter SHA-256",
    )
    trajectory_corpus = _digest(
        inputs.get("trajectory_corpus_sha256"),
        "CPCF trajectory corpus SHA-256",
    )
    live_parent_state = _digest(
        blank.get("parent_state_fingerprint"),
        "CPCF blank-plan live parent state fingerprint",
    )
    parent_graph = _digest(
        blank.get("parent_graph_fingerprint"),
        "CPCF blank target parent graph fingerprint",
    )
    exact_parameters = _positive_int(
        blank.get("target_unique_parameters"),
        "CPCF exact target unique parameters",
    )
    source_parameters = _positive_int(
        parent_inventory_document.get("unique_parameters"),
        "CPCF parent unique parameters",
    )
    blank_document, blank_target_report_sha256 = _verify_blank_target_report(
        blank,
        source_unique_parameters=source_parameters,
        target_unique_parameters=exact_parameters,
    )
    observed_target_counts = (
        _positive_int(
            blank_inventory_document.get("unique_parameters"),
            "CPCF blank inventory unique parameters",
        ),
        _positive_int(
            compiled_inventory_document.get("unique_parameters"),
            "CPCF compiled inventory unique parameters",
        ),
        _positive_int(
            reloaded_inventory_document.get("unique_parameters"),
            "CPCF reloaded inventory unique parameters",
        ),
        _positive_int(
            compression_document.get("target_unique_parameters"),
            "CPCF compression target unique parameters",
        ),
        _positive_int(
            compression_document.get("planned_target_unique_parameters"),
            "CPCF planned target unique parameters",
        ),
    )
    if (
        any(value != exact_parameters for value in observed_target_counts)
        or compression_document.get("source_unique_parameters") != source_parameters
        or not 0 < exact_parameters < source_parameters
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF run receipt does not bind one exact target parameter count "
            "across its blank, compiled child, reload, and compression audit."
        )
    compiled_provenance = _verify_compiled_model_fold(
        body.get("compiled_model_fold"),
        layer_results=body.get("layer_results"),
        gpu_execution_audit=body.get("gpu_execution_audit"),
        gpu_candidate_replay=body.get("gpu_candidate_replay"),
        blank_target_report_sha256=blank_target_report_sha256,
        parent_inventory=parent_inventory_document,
        blank_inventory=blank_inventory_document,
        compiled_inventory=compiled_inventory_document,
        reloaded_inventory=reloaded_inventory_document,
        compression_receipt=compression_document,
    )

    requested_budget_value = blank_document.get("requested_budget")
    if not isinstance(requested_budget_value, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF blank target lacks its exact requested budget."
        )
    requested_budget = _canonical_copy(
        dict(requested_budget_value),
        "CPCF requested target budget",
    )
    if not requested_budget:
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF requested target budget cannot be empty."
        )
    frontier = _digest(
        blank_document.get("frontier_fingerprint"),
        "CPCF target frontier fingerprint",
    )
    certificate = _digest(
        blank_document.get("target_certificate_sha256"),
        "CPCF target certificate SHA-256",
    )
    candidate = _digest(
        blank_document.get("selected_candidate_fingerprint"),
        "CPCF selected target candidate fingerprint",
    )
    candidate_rank = _nonnegative_int(
        blank_document.get("selected_candidate_rank"),
        "CPCF selected target candidate rank",
    )
    topology = _digest(
        blank_document.get("target_topology_fingerprint"),
        "CPCF target topology fingerprint",
    )
    blank_state = _digest(
        blank_document.get("blank_state_fingerprint"),
        "CPCF blank target state fingerprint",
    )
    plan_body = _target_plan_binding_body(
        parent_graph_fingerprint=parent_graph,
        exact_target_unique_parameters=exact_parameters,
        requested_budget=requested_budget,
        frontier_fingerprint=frontier,
        target_certificate_sha256=certificate,
        selected_candidate_fingerprint=candidate,
        selected_candidate_rank=candidate_rank,
        target_topology_fingerprint=topology,
        blank_state_fingerprint=blank_state,
    )
    return {
        "parent_cache_key": cache_key,
        "cache_live_parent_fingerprint": cache_live_parent,
        "live_parent_state_fingerprint": live_parent_state,
        "trace_adapter_sha256": trace_adapter,
        "trajectory_corpus_sha256": trajectory_corpus,
        "blank_target_parent_graph_fingerprint": parent_graph,
        "exact_target_unique_parameters": exact_parameters,
        "target_requested_budget": requested_budget,
        "target_frontier_fingerprint": frontier,
        "target_certificate_sha256": certificate,
        "selected_target_candidate_fingerprint": candidate,
        "selected_target_candidate_rank": candidate_rank,
        "target_topology_fingerprint": topology,
        "blank_target_state_fingerprint": blank_state,
        "target_plan_binding_sha256": canonical_sha256(plan_body),
        **compiled_provenance,
    }


def _verify_capture_receipt(
    document: Mapping[str, Any],
    *,
    manifest_sha256: str,
    manifest_document_sha256: str,
    cpcf_receipt: Mapping[str, Any],
    cpcf_receipt_sha256: str,
    cpcf_receipt_file_sha256: str,
    candidate_document: Mapping[str, Any],
    candidate_outputs_sha256: str,
    candidate_state_fingerprint: str,
    candidate_tokenizer_fingerprint: str,
) -> tuple[str, dict[str, Any]]:
    body, seal = _sealed_document(
        document,
        schema=CPCF_CAPTURE_SCHEMA,
        seal_name="capture_receipt_sha256",
        name="CPCF capture receipt",
    )
    manifest = body.get("manifest")
    provenance = body.get("cpcf_provenance")
    candidate = body.get("candidate")
    protocol = body.get("generation_protocol")
    audit = body.get("operation_audit")
    claims = body.get("claims")
    tokenizer = body.get("target_tokenizer")
    bridge = body.get("strict_tokenizer_bridge")
    required_audit = {
        "candidate_generate_invocations": HOLDOUT_CASE_COUNT,
        "parent_model_loaded": False,
        "parent_forward_passes": 0,
        "parent_generate_invocations": 0,
        "trace_cache_opened": False,
        "training_used": False,
        "training_steps": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradients_used": False,
        "quantization_used": False,
        "pruning_used": False,
        "network_access": False,
        "hub_loader_used": False,
    }
    required_claims = {
        "fresh_local_frozen_child_loaded": True,
        "candidate_state_preserved": True,
        "folding_receipt_authenticated": True,
        "strict_bridge_authenticated": True,
        "strict_bridge_artifact_and_target_domain_authenticated": True,
        "target_tokenizer_package_authenticated": True,
        "target_tokenizer_semantics_authenticated": True,
        "candidate_outputs_are_not_training": True,
        "candidate_outputs_are_not_exact_token_matching": True,
        "intelligence_retention_scored_here": False,
    }
    if (
        body.get("status") != "captured_frozen_child_holdout"
        or not isinstance(manifest, Mapping)
        or manifest.get("manifest_sha256") != manifest_sha256
        or manifest.get("manifest_document_sha256") != manifest_document_sha256
        or manifest.get("case_count") != HOLDOUT_CASE_COUNT
        or not isinstance(provenance, Mapping)
        or provenance.get("receipt_sha256") != cpcf_receipt_sha256
        or provenance.get("external_receipt_file_sha256") != cpcf_receipt_file_sha256
        or provenance.get("internal_receipt_file_sha256") != cpcf_receipt_file_sha256
        or provenance.get("external_internal_receipts_identical") is not True
        or provenance.get("saved_checkpoint_files_match_receipt") is not True
        or provenance.get("reloaded_parameter_inventory_matches_receipt") is not True
        or provenance.get("parameter_inventory_unchanged_after_generation") is not True
        or provenance.get("reloaded_parameter_inventory")
        != cpcf_receipt.get("reloaded_parameter_inventory")
        or not isinstance(candidate, Mapping)
        or candidate.get("output_file_sha256")
        != _canonical_file_sha256(candidate_document)
        or candidate.get("candidate_outputs_sha256") != candidate_outputs_sha256
        or candidate.get("candidate_state_fingerprint") != candidate_state_fingerprint
        or candidate.get("state_before_sha256") != candidate_state_fingerprint
        or candidate.get("state_after_sha256") != candidate_state_fingerprint
        or candidate.get("state_unchanged") is not True
        or candidate.get("case_count") != HOLDOUT_CASE_COUNT
        or candidate.get("all_output_token_ids_are_child_domain_only") is not True
        or candidate.get("decoded_text_sealed") is not True
        or not isinstance(protocol, Mapping)
        or protocol.get("mode") != "greedy_whole_response"
        or protocol.get("do_sample") is not False
        or protocol.get("use_cache") is not True
        or protocol.get("one_generate_invocation_per_case") is not True
        or protocol.get("per_case_manifest_generation_limits_used") is not True
        or protocol.get("global_64_token_cap_used") is not False
        or protocol.get("exact_parent_wording_used") is not False
        or protocol.get("exact_parent_token_matching_used") is not False
        or not isinstance(audit, Mapping)
        or any(audit.get(key) != value for key, value in required_audit.items())
        or not isinstance(claims, Mapping)
        or any(claims.get(key) is not value for key, value in required_claims.items())
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("checkpoint_package_fingerprint")
        != candidate_tokenizer_fingerprint
        or tokenizer.get("source_and_checkpoint_packages_identical") is not True
        or tokenizer.get("source_and_checkpoint_semantics_identical") is not True
        or not isinstance(bridge, Mapping)
        or bridge.get("files_verified") is not True
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF capture receipt lost its exact candidate, model-state, "
            "tokenizer, or zero-operation binding."
        )
    eos = protocol.get("eos_token_ids")
    if (
        not isinstance(eos, list)
        or not eos
        or len(set(eos)) != len(eos)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in eos
        )
        or tokenizer.get("eos_token_ids") != eos
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "CPCF capture receipt has no exact target EOS identity."
        )
    for name in (
        "semantics_sha256",
        "source_package_fingerprint",
        "checkpoint_package_fingerprint",
    ):
        _digest(tokenizer.get(name), f"capture target tokenizer {name}")
    for name in ("artifact_sha256", "profile_sha256"):
        _digest(bridge.get(name), f"capture bridge {name}")
    return seal, body


def _automatic_report_document(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    schema = document.get("schema")
    if schema == AUTOMATIC_GATE_REPORT_SCHEMA:
        _body, seal = _sealed_document(
            document,
            schema=AUTOMATIC_GATE_REPORT_SCHEMA,
            seal_name="report_sha256",
            name="automatic intelligence report",
        )
        return document, seal
    if schema != AUTOMATIC_RUNNER_REPORT_SCHEMA:
        raise Lfm2CpcfReviewedPromotionError("Automatic report schema is unsupported.")
    body, seal = _sealed_document(
        document,
        schema=AUTOMATIC_RUNNER_REPORT_SCHEMA,
        seal_name="report_sha256",
        name="automatic holdout runner report",
    )
    audit = body.get("execution_audit")
    semantics = body.get("semantic_evaluation")
    required_audit = {
        "parent_model_loaded": False,
        "candidate_model_loaded": False,
        "parent_forward_passes": 0,
        "parent_generate_invocations": 0,
        "candidate_forward_passes": 0,
        "candidate_generate_invocations": 0,
        "optimizer_constructed": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "gradients_used": False,
        "network_access": False,
        "hub_loader_used": False,
        "candidate_python_exec_used": False,
        "cached_outputs_only": True,
    }
    if (
        not isinstance(audit, Mapping)
        or any(audit.get(key) != value for key, value in required_audit.items())
        or not isinstance(semantics, Mapping)
        or semantics.get("exact_parent_wording_used") is not False
        or semantics.get("exact_parent_token_matching_used") is not False
        or semantics.get("manual_packet_required_for_open_ended_claim") is not True
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Automatic report permits a model call, forbidden operation, or "
            "exact-output promotion criterion."
        )
    gate = body.get("core_intelligence_gate")
    if not isinstance(gate, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "Automatic runner report has no core intelligence gate."
        )
    return gate, seal


def _automatic_gate_ready(report: Mapping[str, Any]) -> bool:
    components = report.get("components")
    manual = report.get("manual_review")
    reasons = report.get("reasons")
    if (
        not isinstance(components, list)
        or not components
        or any(
            not isinstance(value, Mapping) or value.get("passed") is not True
            for value in components
        )
        or not isinstance(manual, Mapping)
        or not isinstance(reasons, list)
    ):
        return False
    if report.get("promoted") is True:
        return manual.get("status") == "accepted" and not reasons
    return manual.get("status") == "pending" and set(reasons) == {
        "manual_review:pending",
        "headline_lower_bound:withheld_until_all_components",
    }


def _parent_relative_ratings_sha256(
    document: Mapping[str, Any],
    *,
    expected_packet_sha256: str,
) -> str:
    body, seal = _sealed_document(
        document,
        schema=PARENT_RELATIVE_RATINGS_SCHEMA,
        seal_name="ratings_sha256",
        name="parent-relative manual ratings",
    )
    if (
        body.get("packet_sha256") != expected_packet_sha256
        or body.get("exact_wording_comparison_used") is not False
        or not isinstance(body.get("reviewer_id"), str)
        or not body["reviewer_id"].strip()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Parent-relative ratings lost packet or semantic-rubric binding."
        )
    return seal


def build_cpcf_reviewed_promotion_report(
    *,
    reviewed_evidence: Mapping[str, Any],
    inference_execution_contract: BehaviorInferenceExecutionContract,
) -> dict[str, Any]:
    """Seal the only final report shape accepted by the replay gate."""

    if not isinstance(reviewed_evidence, Mapping):
        raise Lfm2CpcfReviewedPromotionError("reviewed_evidence must be a mapping.")
    evidence = _canonical_copy(dict(reviewed_evidence), "reviewed evidence")
    evidence_fingerprint = canonical_sha256(evidence)
    if not isinstance(
        inference_execution_contract,
        BehaviorInferenceExecutionContract,
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "inference_execution_contract has the wrong type."
        )
    if (
        inference_execution_contract
        != BehaviorInferenceExecutionContract.top_level_forward()
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "LFM2 CPCF promotion requires observed top-level model forward execution."
        )
    if (
        evidence.get("automatic_gate_ready_for_manual_completion") is not True
        or evidence.get("parent_relative_manual_report_passed") is not True
        or evidence.get("case_count") != HOLDOUT_CASE_COUNT
        or evidence.get("training_used") is not False
        or evidence.get("pruning_used") is not False
        or evidence.get("quantization_used") is not False
        or evidence.get("network_access") is not False
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed evidence is not eligible for composite promotion."
        )
    body = {
        "schema": COMPOSITE_PROMOTION_SCHEMA,
        "status": "promoted",
        "promoted": True,
        "reviewed_evidence_fingerprint": evidence_fingerprint,
        "reviewed_evidence": evidence,
        "inference_execution": inference_execution_contract.as_dict(),
        "promotion_contract": {
            "first_fold_quarantined": True,
            "exact_candidate_state_required": True,
            "exact_64_case_greedy_replay_required": True,
            "in_memory_and_fresh_reload_required": True,
            "sealed_reviewed_output_digest_required": True,
            "parent_relative_semantic_review_required": True,
            "exact_parent_wording_criterion": False,
            "exact_parent_token_criterion": False,
            "local_tokenizer_only": True,
            "parent_model_loaded": False,
            "training_used": False,
            "pruning_used": False,
            "quantization_used": False,
            "network_access": False,
        },
    }
    return {
        **body,
        "promotion_report_sha256": canonical_sha256(body),
    }


@dataclass(frozen=True, slots=True, init=False)
class Lfm2CpcfReviewedPromotionEvidence:
    """Loader-authenticated evidence complete before live model evaluation."""

    manifest: IntelligenceHoldoutManifest
    cpcf_run_receipt_sha256: str
    cpcf_run_receipt_document_sha256: str
    parent_cache_key: str
    cache_live_parent_fingerprint: str
    live_parent_state_fingerprint: str
    trace_adapter_sha256: str
    trajectory_corpus_sha256: str
    blank_target_parent_graph_fingerprint: str
    exact_target_unique_parameters: int
    target_requested_budget: Mapping[str, Any]
    target_frontier_fingerprint: str
    target_certificate_sha256: str
    selected_target_candidate_fingerprint: str
    selected_target_candidate_rank: int
    target_topology_fingerprint: str
    blank_target_state_fingerprint: str
    target_plan_binding_sha256: str
    compiled_model_fold_schema: str
    compiled_model_fold_result_sha256: str
    compiled_model_fold_policy: Mapping[str, Any]
    compiled_model_fold_policy_sha256: str
    compiled_model_fold_phase_count: int
    compiled_model_fold_phase_sha256: tuple[str, ...]
    compiled_model_fold_phase_binding_sha256: str
    compiled_blank_target_report_sha256: str
    compiled_parameter_receipt_sha256: str
    candidate_rows: tuple[Mapping[str, Any], ...]
    candidate_state_fingerprint: str
    candidate_outputs_sha256: str
    candidate_tokenizer_fingerprint: str
    target_tokenizer_semantics_sha256: str
    eos_token_ids: tuple[int, ...]
    reviewed_evidence: Mapping[str, Any]
    reviewed_evidence_fingerprint: str
    composite_report_sha256: str
    composite_report_document_sha256: str
    full_reviewed_evidence_fingerprint: str
    inference_execution_contract: BehaviorInferenceExecutionContract
    _loader_authority: object = field(init=False, repr=False, compare=False)
    _loader_seal_sha256: str = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        *,
        manifest: IntelligenceHoldoutManifest,
        cpcf_run_receipt_sha256: str,
        cpcf_run_receipt_document_sha256: str,
        parent_cache_key: str,
        cache_live_parent_fingerprint: str,
        live_parent_state_fingerprint: str,
        trace_adapter_sha256: str,
        trajectory_corpus_sha256: str,
        blank_target_parent_graph_fingerprint: str,
        exact_target_unique_parameters: int,
        target_requested_budget: Mapping[str, Any],
        target_frontier_fingerprint: str,
        target_certificate_sha256: str,
        selected_target_candidate_fingerprint: str,
        selected_target_candidate_rank: int,
        target_topology_fingerprint: str,
        blank_target_state_fingerprint: str,
        target_plan_binding_sha256: str,
        compiled_model_fold_schema: str,
        compiled_model_fold_result_sha256: str,
        compiled_model_fold_policy: Mapping[str, Any],
        compiled_model_fold_policy_sha256: str,
        compiled_model_fold_phase_count: int,
        compiled_model_fold_phase_sha256: tuple[str, ...],
        compiled_model_fold_phase_binding_sha256: str,
        compiled_blank_target_report_sha256: str,
        compiled_parameter_receipt_sha256: str,
        candidate_rows: tuple[Mapping[str, Any], ...],
        candidate_state_fingerprint: str,
        candidate_outputs_sha256: str,
        candidate_tokenizer_fingerprint: str,
        target_tokenizer_semantics_sha256: str,
        eos_token_ids: tuple[int, ...],
        reviewed_evidence: Mapping[str, Any],
        reviewed_evidence_fingerprint: str,
        composite_report_sha256: str,
        composite_report_document_sha256: str,
        full_reviewed_evidence_fingerprint: str,
        inference_execution_contract: BehaviorInferenceExecutionContract,
        _loader_token: object | None = None,
    ) -> None:
        if _loader_token is not _EVIDENCE_LOADER_TOKEN:
            raise TypeError(
                "Lfm2CpcfReviewedPromotionEvidence is loader-only; use "
                "load_lfm2_cpcf_reviewed_promotion_evidence()."
            )
        values = {
            "manifest": manifest,
            "cpcf_run_receipt_sha256": cpcf_run_receipt_sha256,
            "cpcf_run_receipt_document_sha256": (cpcf_run_receipt_document_sha256),
            "parent_cache_key": parent_cache_key,
            "cache_live_parent_fingerprint": cache_live_parent_fingerprint,
            "live_parent_state_fingerprint": live_parent_state_fingerprint,
            "trace_adapter_sha256": trace_adapter_sha256,
            "trajectory_corpus_sha256": trajectory_corpus_sha256,
            "blank_target_parent_graph_fingerprint": (
                blank_target_parent_graph_fingerprint
            ),
            "exact_target_unique_parameters": exact_target_unique_parameters,
            "target_requested_budget": target_requested_budget,
            "target_frontier_fingerprint": target_frontier_fingerprint,
            "target_certificate_sha256": target_certificate_sha256,
            "selected_target_candidate_fingerprint": (
                selected_target_candidate_fingerprint
            ),
            "selected_target_candidate_rank": selected_target_candidate_rank,
            "target_topology_fingerprint": target_topology_fingerprint,
            "blank_target_state_fingerprint": blank_target_state_fingerprint,
            "target_plan_binding_sha256": target_plan_binding_sha256,
            "compiled_model_fold_schema": compiled_model_fold_schema,
            "compiled_model_fold_result_sha256": (compiled_model_fold_result_sha256),
            "compiled_model_fold_policy": compiled_model_fold_policy,
            "compiled_model_fold_policy_sha256": (compiled_model_fold_policy_sha256),
            "compiled_model_fold_phase_count": (compiled_model_fold_phase_count),
            "compiled_model_fold_phase_sha256": (compiled_model_fold_phase_sha256),
            "compiled_model_fold_phase_binding_sha256": (
                compiled_model_fold_phase_binding_sha256
            ),
            "compiled_blank_target_report_sha256": (
                compiled_blank_target_report_sha256
            ),
            "compiled_parameter_receipt_sha256": (compiled_parameter_receipt_sha256),
            "candidate_rows": candidate_rows,
            "candidate_state_fingerprint": candidate_state_fingerprint,
            "candidate_outputs_sha256": candidate_outputs_sha256,
            "candidate_tokenizer_fingerprint": (candidate_tokenizer_fingerprint),
            "target_tokenizer_semantics_sha256": (target_tokenizer_semantics_sha256),
            "eos_token_ids": eos_token_ids,
            "reviewed_evidence": reviewed_evidence,
            "reviewed_evidence_fingerprint": (reviewed_evidence_fingerprint),
            "composite_report_sha256": composite_report_sha256,
            "composite_report_document_sha256": (composite_report_document_sha256),
            "full_reviewed_evidence_fingerprint": (full_reviewed_evidence_fingerprint),
            "inference_execution_contract": (inference_execution_contract),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            type(self.manifest) is not IntelligenceHoldoutManifest
            or len(self.manifest.cases) != HOLDOUT_CASE_COUNT
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence lacks the exact 64-case manifest."
            )
        rows = tuple(
            MappingProxyType(_canonical_copy(dict(value), "candidate row"))
            for value in self.candidate_rows
        )
        if len(rows) != HOLDOUT_CASE_COUNT:
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence lacks 64 candidate rows."
            )
        object.__setattr__(self, "candidate_rows", rows)
        for name in (
            "cpcf_run_receipt_sha256",
            "cpcf_run_receipt_document_sha256",
            "parent_cache_key",
            "cache_live_parent_fingerprint",
            "live_parent_state_fingerprint",
            "trace_adapter_sha256",
            "trajectory_corpus_sha256",
            "blank_target_parent_graph_fingerprint",
            "target_frontier_fingerprint",
            "target_certificate_sha256",
            "selected_target_candidate_fingerprint",
            "target_topology_fingerprint",
            "blank_target_state_fingerprint",
            "target_plan_binding_sha256",
            "compiled_model_fold_result_sha256",
            "compiled_model_fold_policy_sha256",
            "compiled_model_fold_phase_binding_sha256",
            "compiled_blank_target_report_sha256",
            "compiled_parameter_receipt_sha256",
            "candidate_state_fingerprint",
            "candidate_outputs_sha256",
            "candidate_tokenizer_fingerprint",
            "target_tokenizer_semantics_sha256",
            "reviewed_evidence_fingerprint",
            "composite_report_sha256",
            "composite_report_document_sha256",
            "full_reviewed_evidence_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                _digest(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "exact_target_unique_parameters",
            _positive_int(
                self.exact_target_unique_parameters,
                "exact_target_unique_parameters",
            ),
        )
        object.__setattr__(
            self,
            "selected_target_candidate_rank",
            _nonnegative_int(
                self.selected_target_candidate_rank,
                "selected_target_candidate_rank",
            ),
        )
        object.__setattr__(
            self,
            "compiled_model_fold_phase_count",
            _positive_int(
                self.compiled_model_fold_phase_count,
                "compiled_model_fold_phase_count",
            ),
        )
        if self.compiled_model_fold_schema not in (
            LFM2_CPCF_MODEL_FOLD_SCHEMA,
            LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Typed CPCF compiled model-fold schema is unsupported."
            )
        compiled_policy = (
            _normalize_model_fold_policy(self.compiled_model_fold_policy)
            if self.compiled_model_fold_schema == LFM2_CPCF_MODEL_FOLD_SCHEMA
            else _normalize_v5_model_fold_policy(self.compiled_model_fold_policy)
        )
        if canonical_sha256(compiled_policy) != self.compiled_model_fold_policy_sha256:
            raise Lfm2CpcfReviewedPromotionError(
                "Typed CPCF compiled model-fold policy seal is invalid."
            )
        object.__setattr__(
            self,
            "compiled_model_fold_policy",
            MappingProxyType(compiled_policy),
        )
        compiled_phase_sha256 = tuple(
            _digest(value, "compiled_model_fold_phase_sha256")
            for value in self.compiled_model_fold_phase_sha256
        )
        if len(compiled_phase_sha256) != self.compiled_model_fold_phase_count or len(
            set(compiled_phase_sha256)
        ) != len(compiled_phase_sha256):
            raise Lfm2CpcfReviewedPromotionError(
                "Typed CPCF compiled model-fold phase sequence is invalid."
            )
        expected_compiled_phase_binding = canonical_sha256(
            _ordered_phase_binding_body(
                phase_count=self.compiled_model_fold_phase_count,
                ordered_phase_sha256=compiled_phase_sha256,
            )
        )
        if (
            self.compiled_model_fold_phase_binding_sha256
            != expected_compiled_phase_binding
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Typed CPCF ordered model-fold phase binding is invalid."
            )
        object.__setattr__(
            self,
            "compiled_model_fold_phase_sha256",
            compiled_phase_sha256,
        )
        requested_budget = _canonical_copy(
            dict(self.target_requested_budget),
            "target_requested_budget",
        )
        if not requested_budget:
            raise Lfm2CpcfReviewedPromotionError(
                "target_requested_budget cannot be empty."
            )
        object.__setattr__(
            self,
            "target_requested_budget",
            MappingProxyType(requested_budget),
        )
        eos = tuple(self.eos_token_ids)
        if (
            not eos
            or len(set(eos)) != len(eos)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in eos
            )
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed target EOS identity is invalid."
            )
        object.__setattr__(self, "eos_token_ids", eos)
        normalized = _canonical_copy(
            dict(self.reviewed_evidence),
            "reviewed evidence",
        )
        if canonical_sha256(normalized) != self.reviewed_evidence_fingerprint:
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence fingerprint is invalid."
            )
        if normalized.get("schema") != REVIEWED_EVIDENCE_SCHEMA:
            raise Lfm2CpcfReviewedPromotionError("Reviewed evidence schema is invalid.")
        if (
            normalized.get("manifest_sha256") != self.manifest.manifest_sha256
            or normalized.get("manifest_document_sha256")
            != self.manifest.document_sha256
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence manifest identity differs from its typed manifest."
            )
        duplicated_fields = {
            "cpcf_run_receipt_sha256": self.cpcf_run_receipt_sha256,
            "cpcf_run_receipt_document_sha256": (self.cpcf_run_receipt_document_sha256),
            "parent_cache_key": self.parent_cache_key,
            "cache_live_parent_fingerprint": self.cache_live_parent_fingerprint,
            "live_parent_state_fingerprint": self.live_parent_state_fingerprint,
            "trace_adapter_sha256": self.trace_adapter_sha256,
            "trajectory_corpus_sha256": self.trajectory_corpus_sha256,
            "blank_target_parent_graph_fingerprint": (
                self.blank_target_parent_graph_fingerprint
            ),
            "target_frontier_fingerprint": self.target_frontier_fingerprint,
            "target_certificate_sha256": self.target_certificate_sha256,
            "selected_target_candidate_fingerprint": (
                self.selected_target_candidate_fingerprint
            ),
            "target_topology_fingerprint": self.target_topology_fingerprint,
            "blank_target_state_fingerprint": (self.blank_target_state_fingerprint),
            "target_plan_binding_sha256": self.target_plan_binding_sha256,
            "compiled_model_fold_result_sha256": (
                self.compiled_model_fold_result_sha256
            ),
            "compiled_model_fold_policy_sha256": (
                self.compiled_model_fold_policy_sha256
            ),
            "compiled_model_fold_phase_binding_sha256": (
                self.compiled_model_fold_phase_binding_sha256
            ),
            "compiled_blank_target_report_sha256": (
                self.compiled_blank_target_report_sha256
            ),
            "compiled_parameter_receipt_sha256": (
                self.compiled_parameter_receipt_sha256
            ),
            "candidate_state_fingerprint": self.candidate_state_fingerprint,
            "candidate_outputs_sha256": self.candidate_outputs_sha256,
            "candidate_tokenizer_fingerprint": (self.candidate_tokenizer_fingerprint),
            "target_tokenizer_semantics_sha256": (
                self.target_tokenizer_semantics_sha256
            ),
        }
        if (
            normalized.get("compiled_model_fold_schema")
            != self.compiled_model_fold_schema
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence compiled model-fold schema differs from "
                "its typed field."
            )
        for name, expected in duplicated_fields.items():
            observed = _digest(
                normalized.get(name),
                f"reviewed evidence {name}",
            )
            if observed != expected:
                raise Lfm2CpcfReviewedPromotionError(
                    f"Reviewed evidence {name} differs from its typed field."
                )
        if (
            normalized.get("exact_target_unique_parameters")
            != self.exact_target_unique_parameters
            or normalized.get("selected_target_candidate_rank")
            != self.selected_target_candidate_rank
            or canonical_json_bytes(normalized.get("target_requested_budget"))
            != canonical_json_bytes(dict(self.target_requested_budget))
            or normalized.get("compiled_model_fold_phase_count")
            != self.compiled_model_fold_phase_count
            or canonical_json_bytes(normalized.get("compiled_model_fold_policy"))
            != canonical_json_bytes(dict(self.compiled_model_fold_policy))
            or canonical_json_bytes(normalized.get("compiled_model_fold_phase_sha256"))
            != canonical_json_bytes(list(self.compiled_model_fold_phase_sha256))
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence target plan or compiled model-fold sequence "
                "differs from its typed run provenance."
            )
        expected_plan_binding = canonical_sha256(
            _target_plan_binding_body(
                parent_graph_fingerprint=(self.blank_target_parent_graph_fingerprint),
                exact_target_unique_parameters=(self.exact_target_unique_parameters),
                requested_budget=self.target_requested_budget,
                frontier_fingerprint=self.target_frontier_fingerprint,
                target_certificate_sha256=self.target_certificate_sha256,
                selected_candidate_fingerprint=(
                    self.selected_target_candidate_fingerprint
                ),
                selected_candidate_rank=self.selected_target_candidate_rank,
                target_topology_fingerprint=self.target_topology_fingerprint,
                blank_state_fingerprint=self.blank_target_state_fingerprint,
            )
        )
        if self.target_plan_binding_sha256 != expected_plan_binding:
            raise Lfm2CpcfReviewedPromotionError(
                "Typed CPCF target-plan binding is invalid."
            )
        reviewed_eos = normalized.get("eos_token_ids")
        if (
            not isinstance(reviewed_eos, list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in reviewed_eos
            )
            or tuple(reviewed_eos) != eos
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed evidence EOS identity differs from its typed field."
            )
        _digest(
            normalized.get("bridge_artifact_sha256"),
            "reviewed evidence bridge_artifact_sha256",
        )
        _digest(
            normalized.get("bridge_profile_sha256"),
            "reviewed evidence bridge_profile_sha256",
        )
        object.__setattr__(
            self,
            "reviewed_evidence",
            MappingProxyType(normalized),
        )
        if not isinstance(
            self.inference_execution_contract,
            BehaviorInferenceExecutionContract,
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed inference execution contract is invalid."
            )
        recomputed_candidate = seal_lfm2_intelligence_holdout_candidate_outputs(
            self.manifest,
            [dict(row) for row in self.candidate_rows],
            candidate_state_fingerprint=(self.candidate_state_fingerprint),
            tokenizer_fingerprint=self.candidate_tokenizer_fingerprint,
            eos_token_ids=self.eos_token_ids,
        )
        if (
            recomputed_candidate.get("candidate_outputs_sha256")
            != self.candidate_outputs_sha256
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed candidate rows differ from their typed output seal."
            )
        expected_full_fingerprint = canonical_sha256(
            {
                "reviewed_evidence_fingerprint": (self.reviewed_evidence_fingerprint),
                "composite_report_sha256": (self.composite_report_sha256),
                "composite_report_document_sha256": (
                    self.composite_report_document_sha256
                ),
            }
        )
        if self.full_reviewed_evidence_fingerprint != expected_full_fingerprint:
            raise Lfm2CpcfReviewedPromotionError(
                "Full reviewed-evidence fingerprint is invalid."
            )
        object.__setattr__(self, "_loader_authority", _EVIDENCE_LOADER_TOKEN)
        object.__setattr__(
            self,
            "_loader_seal_sha256",
            canonical_sha256(_reviewed_evidence_loader_seal_body(self)),
        )

    @property
    def bridge_artifact_sha256(self) -> str:
        """Authenticated strict-tokenizer-bridge artifact identity."""

        return _digest(
            self.reviewed_evidence.get("bridge_artifact_sha256"),
            "reviewed evidence bridge_artifact_sha256",
        )

    @property
    def bridge_profile_sha256(self) -> str:
        """Authenticated strict-tokenizer-bridge transport-profile identity."""

        return _digest(
            self.reviewed_evidence.get("bridge_profile_sha256"),
            "reviewed evidence bridge_profile_sha256",
        )


def _reviewed_evidence_loader_seal_body(
    evidence: Lfm2CpcfReviewedPromotionEvidence,
) -> dict[str, Any]:
    return {
        "schema": "neural-forge-lfm2-cpcf-reviewed-evidence-loader-seal-v1",
        "manifest_sha256": evidence.manifest.manifest_sha256,
        "manifest_document_sha256": evidence.manifest.document_sha256,
        "cpcf_run_receipt_sha256": evidence.cpcf_run_receipt_sha256,
        "cpcf_run_receipt_document_sha256": (evidence.cpcf_run_receipt_document_sha256),
        "parent_cache_key": evidence.parent_cache_key,
        "cache_live_parent_fingerprint": (evidence.cache_live_parent_fingerprint),
        "live_parent_state_fingerprint": evidence.live_parent_state_fingerprint,
        "trace_adapter_sha256": evidence.trace_adapter_sha256,
        "trajectory_corpus_sha256": evidence.trajectory_corpus_sha256,
        "blank_target_parent_graph_fingerprint": (
            evidence.blank_target_parent_graph_fingerprint
        ),
        "exact_target_unique_parameters": (evidence.exact_target_unique_parameters),
        "target_requested_budget": dict(evidence.target_requested_budget),
        "target_frontier_fingerprint": evidence.target_frontier_fingerprint,
        "target_certificate_sha256": evidence.target_certificate_sha256,
        "selected_target_candidate_fingerprint": (
            evidence.selected_target_candidate_fingerprint
        ),
        "selected_target_candidate_rank": (evidence.selected_target_candidate_rank),
        "target_topology_fingerprint": evidence.target_topology_fingerprint,
        "blank_target_state_fingerprint": (evidence.blank_target_state_fingerprint),
        "target_plan_binding_sha256": evidence.target_plan_binding_sha256,
        "compiled_model_fold_schema": evidence.compiled_model_fold_schema,
        "compiled_model_fold_result_sha256": (
            evidence.compiled_model_fold_result_sha256
        ),
        "compiled_model_fold_policy": dict(evidence.compiled_model_fold_policy),
        "compiled_model_fold_policy_sha256": (
            evidence.compiled_model_fold_policy_sha256
        ),
        "compiled_model_fold_phase_count": (evidence.compiled_model_fold_phase_count),
        "compiled_model_fold_phase_sha256": list(
            evidence.compiled_model_fold_phase_sha256
        ),
        "compiled_model_fold_phase_binding_sha256": (
            evidence.compiled_model_fold_phase_binding_sha256
        ),
        "compiled_blank_target_report_sha256": (
            evidence.compiled_blank_target_report_sha256
        ),
        "compiled_parameter_receipt_sha256": (
            evidence.compiled_parameter_receipt_sha256
        ),
        "candidate_rows": [dict(row) for row in evidence.candidate_rows],
        "candidate_state_fingerprint": (evidence.candidate_state_fingerprint),
        "candidate_outputs_sha256": evidence.candidate_outputs_sha256,
        "candidate_tokenizer_fingerprint": (evidence.candidate_tokenizer_fingerprint),
        "target_tokenizer_semantics_sha256": (
            evidence.target_tokenizer_semantics_sha256
        ),
        "eos_token_ids": list(evidence.eos_token_ids),
        "reviewed_evidence": dict(evidence.reviewed_evidence),
        "reviewed_evidence_fingerprint": (evidence.reviewed_evidence_fingerprint),
        "composite_report_sha256": evidence.composite_report_sha256,
        "composite_report_document_sha256": (evidence.composite_report_document_sha256),
        "full_reviewed_evidence_fingerprint": (
            evidence.full_reviewed_evidence_fingerprint
        ),
        "inference_execution_contract": (
            evidence.inference_execution_contract.as_dict()
        ),
    }


def authenticate_lfm2_cpcf_reviewed_promotion_evidence(
    value: Any,
) -> Lfm2CpcfReviewedPromotionEvidence:
    """Reject direct/self-signed or mutated reviewed-promotion evidence."""

    if type(value) is not Lfm2CpcfReviewedPromotionEvidence:
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF evidence must have the exact loader-owned type."
        )
    try:
        authority = object.__getattribute__(value, "_loader_authority")
        seal = object.__getattribute__(value, "_loader_seal_sha256")
        observed = canonical_sha256(_reviewed_evidence_loader_seal_body(value))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF evidence lacks loader authentication."
        ) from exc
    if (
        authority is not _EVIDENCE_LOADER_TOKEN
        or not isinstance(seal, str)
        or seal != observed
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF evidence is direct, self-signed, or mutated."
        )
    return value


def load_lfm2_cpcf_reviewed_promotion_evidence(
    artifacts: Lfm2CpcfReviewedPromotionArtifacts,
) -> Lfm2CpcfReviewedPromotionEvidence:
    """Authenticate and recompute every artifact in the review chain."""

    if not isinstance(artifacts, Lfm2CpcfReviewedPromotionArtifacts):
        raise TypeError("artifacts must be Lfm2CpcfReviewedPromotionArtifacts.")
    manifest_document = artifacts.manifest.load("holdout manifest")
    manifest = load_lfm2_intelligence_holdout_manifest(
        manifest_document,
        expected_manifest_sha256=str(manifest_document.get("manifest_sha256", "")),
    )

    parent_document = artifacts.parent_envelope.load("parent envelope")
    parent_envelope = load_lfm2_intelligence_holdout_parent_envelope(
        parent_document,
        expected_envelope_sha256=str(parent_document.get("envelope_sha256", "")),
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=str(
            parent_document.get("parent_state_fingerprint", "")
        ),
        expected_parent_model_fingerprint=str(
            parent_document.get("parent_model_fingerprint", "")
        ),
        expected_tokenizer_fingerprint=str(
            parent_document.get("tokenizer_fingerprint", "")
        ),
    )
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(parent_envelope)

    cpcf_document = artifacts.cpcf_run_receipt.load("CPCF run receipt")
    cpcf_receipt_sha256, cpcf_body = _verify_cpcf_run_receipt(cpcf_document)
    run_provenance = _extract_cpcf_run_provenance(
        cpcf_body,
        expected_parent_model_fingerprint=(parent_envelope.parent_model_fingerprint),
    )
    cpcf_inputs = cpcf_body.get("inputs")
    if not isinstance(cpcf_inputs, Mapping):
        raise Lfm2CpcfReviewedPromotionError("CPCF run receipt has no input binding.")
    bridge_artifact_sha256 = _digest(
        cpcf_inputs.get("bridge_artifact_sha256"),
        "CPCF bridge artifact SHA-256",
    )
    bridge_profile_sha256 = _digest(
        cpcf_inputs.get("bridge_profile_sha256"),
        "CPCF bridge profile SHA-256",
    )

    candidate_document = artifacts.candidate_outputs.load("candidate outputs")
    candidate_state = _digest(
        candidate_document.get("candidate_state_fingerprint"),
        "candidate state fingerprint",
    )
    candidate_tokenizer = _digest(
        candidate_document.get("tokenizer_fingerprint"),
        "candidate tokenizer fingerprint",
    )
    candidate_outputs_sha256 = _digest(
        candidate_document.get("candidate_outputs_sha256"),
        "candidate outputs SHA-256",
    )

    capture_document = artifacts.capture_receipt.load("capture receipt")
    capture_body = dict(capture_document)
    capture_protocol = capture_body.get("generation_protocol")
    if not isinstance(capture_protocol, Mapping):
        raise Lfm2CpcfReviewedPromotionError(
            "Capture receipt has no generation protocol."
        )
    eos_token_ids = tuple(capture_protocol.get("eos_token_ids", ()))
    candidate = load_lfm2_intelligence_holdout_candidate_outputs(
        candidate_document,
        expected_candidate_outputs_sha256=candidate_outputs_sha256,
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=candidate_state,
        expected_tokenizer_fingerprint=candidate_tokenizer,
        eos_token_ids=eos_token_ids,
    )
    capture_receipt_sha256, capture_body = _verify_capture_receipt(
        capture_document,
        manifest_sha256=manifest.manifest_sha256,
        manifest_document_sha256=manifest.document_sha256,
        cpcf_receipt=cpcf_body,
        cpcf_receipt_sha256=cpcf_receipt_sha256,
        cpcf_receipt_file_sha256=_canonical_file_sha256(cpcf_document),
        candidate_document=candidate_document,
        candidate_outputs_sha256=candidate_outputs_sha256,
        candidate_state_fingerprint=candidate_state,
        candidate_tokenizer_fingerprint=candidate_tokenizer,
    )
    capture_tokenizer = capture_body["target_tokenizer"]
    capture_bridge = capture_body["strict_tokenizer_bridge"]
    if (
        capture_bridge.get("artifact_sha256") != bridge_artifact_sha256
        or capture_bridge.get("profile_sha256") != bridge_profile_sha256
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Capture receipt and fold receipt name different token bridges."
        )
    tokenizer_semantics = _digest(
        capture_tokenizer.get("semantics_sha256"),
        "target tokenizer semantics SHA-256",
    )

    policy = IntelligenceGatePolicy()
    recomputed_automatic = evaluate_cached_intelligence(
        baseline,
        candidate.candidates,
        policy=policy,
        manual_ratings=None,
    ).as_dict()
    automatic_document = artifacts.automatic_report.load(
        "automatic intelligence report"
    )
    automatic_gate, automatic_outer_sha256 = _automatic_report_document(
        automatic_document
    )
    if canonical_json_bytes(automatic_gate) != canonical_json_bytes(
        recomputed_automatic
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Automatic 64-case report differs from a fresh typed recomputation."
        )
    if not _automatic_gate_ready(automatic_gate):
        raise Lfm2CpcfReviewedPromotionError(
            "Automatic report is rejected beyond its expected manual-review "
            "withholding state."
        )
    if automatic_document.get("schema") == AUTOMATIC_RUNNER_REPORT_SCHEMA:
        inputs = automatic_document.get("inputs")
        if (
            not isinstance(inputs, Mapping)
            or inputs.get("manifest_sha256") != manifest.manifest_sha256
            or inputs.get("case_count") != HOLDOUT_CASE_COUNT
            or inputs.get("parent_envelope_sha256") != parent_envelope.envelope_sha256
            or inputs.get("candidate_report_file_sha256")
            != _canonical_file_sha256(candidate_document)
            or inputs.get("sealed_candidate_outputs_sha256") != candidate_outputs_sha256
            or inputs.get("candidate_state_fingerprint") != candidate_state
            or inputs.get("child_tokenizer_fingerprint") != candidate_tokenizer
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Automatic runner report lost its exact holdout input binding."
            )

    manual_document = artifacts.manual_packet.load("manual packet")
    recomputed_manual = build_manual_review_packet(
        baseline,
        candidate.candidates,
        policy,
    )
    if canonical_json_bytes(manual_document) != canonical_json_bytes(recomputed_manual):
        raise Lfm2CpcfReviewedPromotionError(
            "Manual packet differs from the authenticated holdout."
        )
    manual_packet = load_manual_review_packet(
        manual_document,
        expected_packet_sha256=str(manual_document.get("packet_sha256", "")),
    )

    parent_packet_document = artifacts.parent_relative_manual_packet.load(
        "parent-relative manual packet"
    )
    recomputed_parent_packet = build_parent_relative_manual_packet(manual_packet)
    if canonical_json_bytes(parent_packet_document) != canonical_json_bytes(
        recomputed_parent_packet
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Parent-relative packet differs from the exact manual packet."
        )
    parent_packet_sha256 = _digest(
        parent_packet_document.get("packet_sha256"),
        "parent-relative packet SHA-256",
    )

    ratings_document = artifacts.parent_relative_ratings.load("parent-relative ratings")
    ratings_sha256 = _parent_relative_ratings_sha256(
        ratings_document,
        expected_packet_sha256=parent_packet_sha256,
    )
    recomputed_parent_report = evaluate_parent_relative_manual_retention(
        parent_packet_document,
        ratings_document,
    )
    parent_report_document = artifacts.parent_relative_report.load(
        "parent-relative report"
    )
    if canonical_json_bytes(parent_report_document) != canonical_json_bytes(
        recomputed_parent_report
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Parent-relative report differs from a fresh typed recomputation."
        )
    parent_report_body, parent_report_sha256 = _sealed_document(
        parent_report_document,
        schema=PARENT_RELATIVE_REPORT_SCHEMA,
        seal_name="report_sha256",
        name="parent-relative manual report",
    )
    if (
        parent_report_body.get("status") != "accepted"
        or parent_report_body.get("passed") is not True
        or parent_report_body.get("retention", 0.0) < 0.90
        or parent_report_body.get("clear_regression_count") != 0
        or parent_report_body.get("packet_sha256") != parent_packet_sha256
        or parent_report_body.get("ratings_sha256") != ratings_sha256
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Parent-relative manual report is pending, rejected, or regressed."
        )

    reviewed_evidence = {
        "schema": REVIEWED_EVIDENCE_SCHEMA,
        "case_count": HOLDOUT_CASE_COUNT,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_document_sha256": artifacts.manifest.document_sha256,
        "parent_envelope_sha256": parent_envelope.envelope_sha256,
        "parent_envelope_document_sha256": (artifacts.parent_envelope.document_sha256),
        "cpcf_run_receipt_sha256": cpcf_receipt_sha256,
        "cpcf_run_receipt_document_sha256": (
            artifacts.cpcf_run_receipt.document_sha256
        ),
        "parent_cache_key": run_provenance["parent_cache_key"],
        "cache_live_parent_fingerprint": (
            run_provenance["cache_live_parent_fingerprint"]
        ),
        "live_parent_state_fingerprint": (
            run_provenance["live_parent_state_fingerprint"]
        ),
        "trace_adapter_sha256": run_provenance["trace_adapter_sha256"],
        "trajectory_corpus_sha256": (run_provenance["trajectory_corpus_sha256"]),
        "blank_target_parent_graph_fingerprint": (
            run_provenance["blank_target_parent_graph_fingerprint"]
        ),
        "exact_target_unique_parameters": (
            run_provenance["exact_target_unique_parameters"]
        ),
        "target_requested_budget": run_provenance["target_requested_budget"],
        "target_frontier_fingerprint": (run_provenance["target_frontier_fingerprint"]),
        "target_certificate_sha256": (run_provenance["target_certificate_sha256"]),
        "selected_target_candidate_fingerprint": (
            run_provenance["selected_target_candidate_fingerprint"]
        ),
        "selected_target_candidate_rank": (
            run_provenance["selected_target_candidate_rank"]
        ),
        "target_topology_fingerprint": (run_provenance["target_topology_fingerprint"]),
        "blank_target_state_fingerprint": (
            run_provenance["blank_target_state_fingerprint"]
        ),
        "target_plan_binding_sha256": (run_provenance["target_plan_binding_sha256"]),
        "compiled_model_fold_schema": (run_provenance["compiled_model_fold_schema"]),
        "compiled_model_fold_result_sha256": (
            run_provenance["compiled_model_fold_result_sha256"]
        ),
        "compiled_model_fold_policy": (run_provenance["compiled_model_fold_policy"]),
        "compiled_model_fold_policy_sha256": (
            run_provenance["compiled_model_fold_policy_sha256"]
        ),
        "compiled_model_fold_phase_count": (
            run_provenance["compiled_model_fold_phase_count"]
        ),
        "compiled_model_fold_phase_sha256": list(
            run_provenance["compiled_model_fold_phase_sha256"]
        ),
        "compiled_model_fold_phase_binding_sha256": (
            run_provenance["compiled_model_fold_phase_binding_sha256"]
        ),
        "compiled_blank_target_report_sha256": (
            run_provenance["compiled_blank_target_report_sha256"]
        ),
        "compiled_parameter_receipt_sha256": (
            run_provenance["compiled_parameter_receipt_sha256"]
        ),
        "bridge_artifact_sha256": bridge_artifact_sha256,
        "bridge_profile_sha256": bridge_profile_sha256,
        "candidate_state_fingerprint": candidate_state,
        "candidate_outputs_sha256": candidate_outputs_sha256,
        "candidate_outputs_document_sha256": (
            artifacts.candidate_outputs.document_sha256
        ),
        "capture_receipt_sha256": capture_receipt_sha256,
        "capture_receipt_document_sha256": (artifacts.capture_receipt.document_sha256),
        "automatic_report_sha256": automatic_outer_sha256,
        "automatic_report_document_sha256": (
            artifacts.automatic_report.document_sha256
        ),
        "automatic_core_report_sha256": automatic_gate["report_sha256"],
        "automatic_gate_ready_for_manual_completion": True,
        "manual_packet_sha256": manual_packet["packet_sha256"],
        "manual_packet_document_sha256": (artifacts.manual_packet.document_sha256),
        "parent_relative_manual_packet_sha256": parent_packet_sha256,
        "parent_relative_manual_packet_document_sha256": (
            artifacts.parent_relative_manual_packet.document_sha256
        ),
        "parent_relative_ratings_sha256": ratings_sha256,
        "parent_relative_ratings_document_sha256": (
            artifacts.parent_relative_ratings.document_sha256
        ),
        "parent_relative_manual_report_sha256": parent_report_sha256,
        "parent_relative_manual_report_document_sha256": (
            artifacts.parent_relative_report.document_sha256
        ),
        "parent_relative_manual_report_passed": True,
        "parent_relative_manual_retention": parent_report_body["retention"],
        "candidate_tokenizer_fingerprint": candidate_tokenizer,
        "target_tokenizer_semantics_sha256": tokenizer_semantics,
        "eos_token_ids": list(eos_token_ids),
        "parent_model_loaded": False,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "network_access": False,
    }
    reviewed_fingerprint = canonical_sha256(reviewed_evidence)

    composite_document = artifacts.composite_promotion_report.load(
        "composite promotion report"
    )
    composite_body, composite_sha256 = _sealed_document(
        composite_document,
        schema=COMPOSITE_PROMOTION_SCHEMA,
        seal_name="promotion_report_sha256",
        name="composite promotion report",
    )
    inference_value = composite_body.get("inference_execution")
    try:
        inference_contract = BehaviorInferenceExecutionContract.from_dict(
            inference_value
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "Composite report has an invalid inference execution contract."
        ) from exc
    recomputed_composite = build_cpcf_reviewed_promotion_report(
        reviewed_evidence=reviewed_evidence,
        inference_execution_contract=inference_contract,
    )
    if (
        canonical_json_bytes(composite_document)
        != canonical_json_bytes(recomputed_composite)
        or composite_body.get("status") != "promoted"
        or composite_body.get("promoted") is not True
        or composite_body.get("reviewed_evidence_fingerprint") != reviewed_fingerprint
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Composite final report is pending, rejected, spoofed, or stale."
        )
    full_fingerprint = canonical_sha256(
        {
            "reviewed_evidence_fingerprint": reviewed_fingerprint,
            "composite_report_sha256": composite_sha256,
            "composite_report_document_sha256": (
                artifacts.composite_promotion_report.document_sha256
            ),
        }
    )
    evidence = Lfm2CpcfReviewedPromotionEvidence(
        manifest=manifest,
        cpcf_run_receipt_sha256=cpcf_receipt_sha256,
        cpcf_run_receipt_document_sha256=(artifacts.cpcf_run_receipt.document_sha256),
        parent_cache_key=run_provenance["parent_cache_key"],
        cache_live_parent_fingerprint=(run_provenance["cache_live_parent_fingerprint"]),
        live_parent_state_fingerprint=(run_provenance["live_parent_state_fingerprint"]),
        trace_adapter_sha256=run_provenance["trace_adapter_sha256"],
        trajectory_corpus_sha256=(run_provenance["trajectory_corpus_sha256"]),
        blank_target_parent_graph_fingerprint=(
            run_provenance["blank_target_parent_graph_fingerprint"]
        ),
        exact_target_unique_parameters=(
            run_provenance["exact_target_unique_parameters"]
        ),
        target_requested_budget=run_provenance["target_requested_budget"],
        target_frontier_fingerprint=(run_provenance["target_frontier_fingerprint"]),
        target_certificate_sha256=(run_provenance["target_certificate_sha256"]),
        selected_target_candidate_fingerprint=(
            run_provenance["selected_target_candidate_fingerprint"]
        ),
        selected_target_candidate_rank=(
            run_provenance["selected_target_candidate_rank"]
        ),
        target_topology_fingerprint=(run_provenance["target_topology_fingerprint"]),
        blank_target_state_fingerprint=(
            run_provenance["blank_target_state_fingerprint"]
        ),
        target_plan_binding_sha256=(run_provenance["target_plan_binding_sha256"]),
        compiled_model_fold_schema=(run_provenance["compiled_model_fold_schema"]),
        compiled_model_fold_result_sha256=(
            run_provenance["compiled_model_fold_result_sha256"]
        ),
        compiled_model_fold_policy=(run_provenance["compiled_model_fold_policy"]),
        compiled_model_fold_policy_sha256=(
            run_provenance["compiled_model_fold_policy_sha256"]
        ),
        compiled_model_fold_phase_count=(
            run_provenance["compiled_model_fold_phase_count"]
        ),
        compiled_model_fold_phase_sha256=(
            run_provenance["compiled_model_fold_phase_sha256"]
        ),
        compiled_model_fold_phase_binding_sha256=(
            run_provenance["compiled_model_fold_phase_binding_sha256"]
        ),
        compiled_blank_target_report_sha256=(
            run_provenance["compiled_blank_target_report_sha256"]
        ),
        compiled_parameter_receipt_sha256=(
            run_provenance["compiled_parameter_receipt_sha256"]
        ),
        candidate_rows=tuple(candidate.rows),
        candidate_state_fingerprint=candidate_state,
        candidate_outputs_sha256=candidate_outputs_sha256,
        candidate_tokenizer_fingerprint=candidate_tokenizer,
        target_tokenizer_semantics_sha256=tokenizer_semantics,
        eos_token_ids=eos_token_ids,
        reviewed_evidence=reviewed_evidence,
        reviewed_evidence_fingerprint=reviewed_fingerprint,
        composite_report_sha256=composite_sha256,
        composite_report_document_sha256=(
            artifacts.composite_promotion_report.document_sha256
        ),
        full_reviewed_evidence_fingerprint=full_fingerprint,
        inference_execution_contract=inference_contract,
        _loader_token=_EVIDENCE_LOADER_TOKEN,
    )
    authenticate_lfm2_cpcf_reviewed_gate_implementation()
    return authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "Candidate model has no materialized parameters."
        ) from exc


def _normalized_eos_token_id(
    tokenizer: Any,
    expected: tuple[int, ...],
) -> int | list[int]:
    value = getattr(tokenizer, "eos_token_id", None)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        observed = tuple(int(item) for item in value)
    elif value is None:
        observed = ()
    else:
        observed = (int(value),)
    if observed != expected:
        raise Lfm2CpcfReviewedPromotionError(
            "Live tokenizer EOS identity differs from reviewed capture."
        )
    return observed[0] if len(observed) == 1 else list(observed)


def _greedy_replay_rows(
    model: Any,
    tokenizer: Any,
    evidence: Lfm2CpcfReviewedPromotionEvidence,
) -> tuple[dict[str, Any], ...]:
    """Execute the exact typed suite once and return seal-compatible rows."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise Lfm2CpcfReviewedPromotionError(
            "PyTorch is required for CPCF behavioral replay."
        ) from exc
    if not isinstance(model, torch.nn.Module):
        raise Lfm2CpcfReviewedPromotionError("Candidate must be a torch.nn.Module.")
    generate = getattr(model, "generate", None)
    if not callable(generate):
        raise Lfm2CpcfReviewedPromotionError(
            "Candidate model has no generate() entrypoint."
        )
    device = _model_device(model)
    eos_token_id = _normalized_eos_token_id(
        tokenizer,
        evidence.eos_token_ids,
    )
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = (
            eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        )
    reviewed_by_id = {str(row["case_id"]): row for row in evidence.candidate_rows}
    rows = []
    with torch.inference_mode():
        for spec in evidence.manifest.cases:
            encoded = tokenizer(
                spec.prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            if (
                not isinstance(encoded, Mapping)
                or not isinstance(encoded.get("input_ids"), torch.Tensor)
                or encoded["input_ids"].ndim != 2
                or int(encoded["input_ids"].shape[0]) != 1
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    f"Tokenizer output is invalid for {spec.case_id}."
                )
            input_ids = encoded["input_ids"].to(
                device=device,
                dtype=torch.long,
            )
            attention_mask = encoded.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            elif not isinstance(attention_mask, torch.Tensor):
                raise Lfm2CpcfReviewedPromotionError(
                    f"Attention mask is invalid for {spec.case_id}."
                )
            attention_mask = attention_mask.to(
                device=device,
                dtype=torch.long,
            )
            generated = generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=spec.generation_limit,
                do_sample=False,
                use_cache=True,
                pad_token_id=int(pad_token_id),
                eos_token_id=eos_token_id,
            )
            if (
                not isinstance(generated, torch.Tensor)
                or generated.ndim != 2
                or int(generated.shape[0]) != 1
                or int(generated.shape[1]) < int(input_ids.shape[1])
                or not torch.equal(
                    generated[0, : int(input_ids.shape[1])],
                    input_ids[0],
                )
            ):
                raise Lfm2CpcfReviewedPromotionError(
                    f"Generation or prompt prefix is invalid for {spec.case_id}."
                )
            continuation = generated[0, int(input_ids.shape[1]) :]
            output_ids = [
                int(value) for value in continuation.detach().to("cpu").tolist()
            ]
            if len(output_ids) > spec.generation_limit:
                raise Lfm2CpcfReviewedPromotionError(
                    f"Generation exceeded the limit for {spec.case_id}."
                )
            output = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            )
            if not isinstance(output, str):
                raise Lfm2CpcfReviewedPromotionError(
                    f"Decoded output is invalid for {spec.case_id}."
                )
            reviewed = reviewed_by_id[spec.case_id]
            rows.append(
                {
                    "case_id": spec.case_id,
                    "output": output,
                    "output_token_ids": output_ids,
                    "generation_limit": spec.generation_limit,
                    "topical_relevance": reviewed["topical_relevance"],
                }
            )
    return tuple(rows)


class _ImmutableReviewedGateMeta(type):
    """Prevent ordinary runtime replacement of a reviewed gate class."""

    def __setattr__(cls, name: str, value: Any) -> None:
        if cls.__dict__.get("_reviewed_gate_class_sealed", False):
            raise TypeError(
                "The reviewed CPCF promotion gate class is implementation-sealed."
            )
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        if cls.__dict__.get("_reviewed_gate_class_sealed", False):
            raise TypeError(
                "The reviewed CPCF promotion gate class is implementation-sealed."
            )
        super().__delattr__(name)


class Lfm2CpcfReviewedPromotionGate(metaclass=_ImmutableReviewedGateMeta):
    """Final evaluator shared by in-memory and fresh-reload publication gates."""

    __slots__ = (
        "_behavior_gate_binding",
        "_construction_complete",
        "_implementation_seal_sha256",
        "baseline_fingerprint",
        "behavior_baseline_fingerprint",
        "behavior_gate_id",
        "behavior_gate_manifest",
        "behavior_gate_manifest_fingerprint",
        "behavior_gate_version",
        "behavior_inference_execution_contract",
        "evidence",
        "tokenizer",
        "tokenizer_directory",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            construction_complete = object.__getattribute__(
                self,
                "_construction_complete",
            )
        except AttributeError as exc:
            raise AttributeError(
                "Reviewed CPCF gate fields are constructor-managed."
            ) from exc
        if construction_complete:
            raise AttributeError(
                "Reviewed CPCF gate instances are immutable after construction."
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        del name
        raise AttributeError(
            "Reviewed CPCF gate instances are immutable after construction."
        )

    def __init__(
        self,
        evidence: Lfm2CpcfReviewedPromotionEvidence,
        *,
        tokenizer: Any,
        tokenizer_directory: str | os.PathLike[str],
    ) -> None:
        authenticate_lfm2_cpcf_reviewed_gate_implementation()
        try:
            object.__getattribute__(self, "_construction_complete")
        except AttributeError:
            pass
        else:
            raise Lfm2CpcfReviewedPromotionError(
                "Reviewed CPCF gate instances cannot be reinitialized."
            )
        object.__setattr__(self, "_construction_complete", False)
        if type(evidence) is not Lfm2CpcfReviewedPromotionEvidence:
            raise TypeError("evidence must be Lfm2CpcfReviewedPromotionEvidence.")
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "tokenizer", tokenizer)
        raw_directory = os.fspath(tokenizer_directory)
        if "://" in raw_directory:
            raise Lfm2CpcfReviewedPromotionError("Tokenizer directory must be local.")
        try:
            directory = Path(raw_directory).expanduser().resolve(strict=True)
        except OSError as exc:
            raise Lfm2CpcfReviewedPromotionError(
                "Tokenizer directory does not exist."
            ) from exc
        object.__setattr__(self, "tokenizer_directory", directory)
        self._verify_tokenizer()
        cases = [
            {
                "case_id": spec.case_id,
                "prompt_sha256": hashlib.sha256(
                    spec.prompt.encode("utf-8")
                ).hexdigest(),
                "generation_limit": spec.generation_limit,
                "split": spec.split,
                "category": spec.category,
            }
            for spec in evidence.manifest.cases
        ]
        generator_identity = identify_behavior_callable(
            _greedy_replay_rows,
            implementation_id=("neural-forge.lfm2-cpcf-reviewed-greedy-replay-v1"),
            version=1,
        )
        binding = build_behavior_gate_binding(
            gate_id=BEHAVIOR_GATE_ID,
            gate_version=BEHAVIOR_GATE_VERSION,
            baseline_fingerprint=(evidence.full_reviewed_evidence_fingerprint),
            threshold_config={
                "minimum_parent_relative_intelligence_retention": 0.90,
                "clear_manual_regressions_allowed": 0,
                "exact_reviewed_output_digest_required": (
                    evidence.candidate_outputs_sha256
                ),
                "exact_candidate_state_required": (
                    evidence.candidate_state_fingerprint
                ),
            },
            cases=cases,
            baseline={
                "full_reviewed_evidence_fingerprint": (
                    evidence.full_reviewed_evidence_fingerprint
                ),
                "reviewed_evidence": dict(evidence.reviewed_evidence),
                "composite_report_sha256": (evidence.composite_report_sha256),
                "composite_report_document_sha256": (
                    evidence.composite_report_document_sha256
                ),
            },
            generator_identity=generator_identity,
            scorer_identity=None,
            coverage_policy={
                "exact_case_count": HOLDOUT_CASE_COUNT,
                "whole_response_greedy_replay": True,
                "parent_relative_semantic_review": True,
                "fresh_reload_required": True,
            },
            components=(
                {
                    "schema": REVIEWED_EVIDENCE_SCHEMA,
                    "full_reviewed_evidence_fingerprint": (
                        evidence.full_reviewed_evidence_fingerprint
                    ),
                    "reviewed_evidence_fingerprint": (
                        evidence.reviewed_evidence_fingerprint
                    ),
                    "composite_report_sha256": (evidence.composite_report_sha256),
                    "cpcf_run_receipt_sha256": (evidence.cpcf_run_receipt_sha256),
                    "cpcf_run_receipt_document_sha256": (
                        evidence.cpcf_run_receipt_document_sha256
                    ),
                    "parent_cache_key": evidence.parent_cache_key,
                    "cache_live_parent_fingerprint": (
                        evidence.cache_live_parent_fingerprint
                    ),
                    "live_parent_state_fingerprint": (
                        evidence.live_parent_state_fingerprint
                    ),
                    "trace_adapter_sha256": evidence.trace_adapter_sha256,
                    "trajectory_corpus_sha256": (evidence.trajectory_corpus_sha256),
                    "blank_target_parent_graph_fingerprint": (
                        evidence.blank_target_parent_graph_fingerprint
                    ),
                    "exact_target_unique_parameters": (
                        evidence.exact_target_unique_parameters
                    ),
                    "target_plan_binding_sha256": (evidence.target_plan_binding_sha256),
                    "compiled_model_fold_schema": (evidence.compiled_model_fold_schema),
                    "compiled_model_fold_result_sha256": (
                        evidence.compiled_model_fold_result_sha256
                    ),
                    "compiled_model_fold_policy": dict(
                        evidence.compiled_model_fold_policy
                    ),
                    "compiled_model_fold_policy_sha256": (
                        evidence.compiled_model_fold_policy_sha256
                    ),
                    "compiled_model_fold_phase_count": (
                        evidence.compiled_model_fold_phase_count
                    ),
                    "compiled_model_fold_phase_sha256": list(
                        evidence.compiled_model_fold_phase_sha256
                    ),
                    "compiled_model_fold_phase_binding_sha256": (
                        evidence.compiled_model_fold_phase_binding_sha256
                    ),
                    "compiled_blank_target_report_sha256": (
                        evidence.compiled_blank_target_report_sha256
                    ),
                    "compiled_parameter_receipt_sha256": (
                        evidence.compiled_parameter_receipt_sha256
                    ),
                    "candidate_state_fingerprint": (
                        evidence.candidate_state_fingerprint
                    ),
                    "candidate_outputs_sha256": (evidence.candidate_outputs_sha256),
                    "training_used": False,
                    "pruning_used": False,
                    "quantization_used": False,
                    "network_access": False,
                    "hub_loader_used": False,
                },
            ),
            inference_execution_contract=(evidence.inference_execution_contract),
        )
        expose_behavior_gate_binding(self, binding)
        object.__setattr__(
            self,
            "_implementation_seal_sha256",
            _EXPECTED_GATE_IMPLEMENTATION_SEAL_SHA256,
        )
        object.__setattr__(self, "_construction_complete", True)
        authenticate_lfm2_cpcf_reviewed_promotion_gate(self)

    def _verify_tokenizer(self) -> None:
        authenticate_lfm2_cpcf_reviewed_gate_implementation()
        package, _files = fingerprint_local_tokenizer_package(self.tokenizer_directory)
        if package != self.evidence.candidate_tokenizer_fingerprint:
            raise Lfm2CpcfReviewedPromotionError(
                "Local tokenizer package differs from reviewed candidate."
            )
        semantics = fingerprint_hf_tokenizer_semantics(self.tokenizer)
        if semantics != self.evidence.target_tokenizer_semantics_sha256:
            raise Lfm2CpcfReviewedPromotionError(
                "Live tokenizer semantics differ from reviewed capture."
            )
        _normalized_eos_token_id(
            self.tokenizer,
            self.evidence.eos_token_ids,
        )

    def binding_fields(self) -> dict[str, Any]:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(self)
        return resolve_behavior_gate_binding(self).fields()

    def evaluate_model(self, model: Any) -> FoldDecision:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(self)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise Lfm2CpcfReviewedPromotionError(
                "PyTorch is required for CPCF reviewed promotion."
            ) from exc
        self._verify_tokenizer()
        if not isinstance(model, torch.nn.Module):
            raise Lfm2CpcfReviewedPromotionError("Candidate must be a torch.nn.Module.")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise Lfm2CpcfReviewedPromotionError(
                "Candidate contains gradients before reviewed replay."
            )
        state_before = fingerprint_cpcf_candidate_state(model)
        if state_before != self.evidence.candidate_state_fingerprint:
            raise Lfm2CpcfReviewedPromotionError(
                "Candidate model state differs from reviewed state."
            )
        was_training = bool(model.training)
        model.eval()
        try:
            with observe_model_execution(
                model,
                contract=self.evidence.inference_execution_contract,
            ) as execution:
                rows = _greedy_replay_rows(
                    model,
                    self.tokenizer,
                    self.evidence,
                )
        finally:
            model.train(was_training)
        if not execution_contract_satisfied(execution):
            raise Lfm2CpcfReviewedPromotionError(
                "Generation returned without executing the required model "
                "inference root."
            )
        state_after = fingerprint_cpcf_candidate_state(model)
        if state_after != state_before or any(
            parameter.grad is not None for parameter in model.parameters()
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Candidate state or gradients changed during reviewed replay."
            )
        self._verify_tokenizer()
        replay = seal_lfm2_intelligence_holdout_candidate_outputs(
            self.evidence.manifest,
            rows,
            candidate_state_fingerprint=state_before,
            tokenizer_fingerprint=(self.evidence.candidate_tokenizer_fingerprint),
            eos_token_ids=self.evidence.eos_token_ids,
        )
        if replay.get(
            "candidate_outputs_sha256"
        ) != self.evidence.candidate_outputs_sha256 or canonical_json_bytes(
            replay.get("rows")
        ) != canonical_json_bytes([dict(row) for row in self.evidence.candidate_rows]):
            raise Lfm2CpcfReviewedPromotionError(
                "Greedy 64-case replay differs from the reviewed candidate "
                "output digest."
            )
        metrics = {
            "case_count": HOLDOUT_CASE_COUNT,
            "candidate_state_authenticated": True,
            "candidate_outputs_replayed_exactly": True,
            "inference_contract_satisfied": True,
            "observed_root_forward_calls": execution["root_forward_calls"],
            "observed_inference_entrypoint_calls": execution[
                "inference_entrypoint_calls"
            ],
            "reviewed_evidence_fingerprint": (
                self.evidence.full_reviewed_evidence_fingerprint
            ),
            "composite_promotion_report_sha256": (
                self.evidence.composite_report_sha256
            ),
            "cpcf_run_receipt_sha256": (self.evidence.cpcf_run_receipt_sha256),
            "parent_cache_key": self.evidence.parent_cache_key,
            "target_plan_binding_sha256": (self.evidence.target_plan_binding_sha256),
            "exact_target_unique_parameters": (
                self.evidence.exact_target_unique_parameters
            ),
            "parent_model_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "pruned_parameters": 0,
            "quantized_parameters": 0,
            "network_access": False,
        }
        return FoldDecision.accept(
            "Reviewed CPCF candidate reproduced all 64 sealed greedy "
            "whole-response generations from its exact authenticated state.",
            metrics,
        )

    def evaluate(self, context: FoldEvaluationContext) -> FoldDecision:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(self)
        if not isinstance(context, FoldEvaluationContext):
            raise TypeError("context must be FoldEvaluationContext.")
        if (
            context.candidate_state_fingerprint
            != context.execution_context.child_state_fingerprint
        ):
            raise Lfm2CpcfReviewedPromotionError(
                "Fold evaluation context does not describe the live child state."
            )
        return self.evaluate_model(context.execution_context.child_model)

    def __call__(self, model: Any) -> FoldDecision:
        authenticate_lfm2_cpcf_reviewed_promotion_gate(self)
        return self.evaluate_model(model)


def _reviewed_gate_implementation_targets() -> tuple[tuple[str, Any], ...]:
    """Return every callable that can authorize or materially alter replay."""

    class_namespace = vars(Lfm2CpcfReviewedPromotionGate)
    evidence_namespace = vars(Lfm2CpcfReviewedPromotionEvidence)
    return (
        ("gate.__init__", class_namespace["__init__"]),
        ("gate.__setattr__", class_namespace["__setattr__"]),
        ("gate.__delattr__", class_namespace["__delattr__"]),
        ("gate._verify_tokenizer", class_namespace["_verify_tokenizer"]),
        ("gate.binding_fields", class_namespace["binding_fields"]),
        ("gate.evaluate_model", class_namespace["evaluate_model"]),
        ("gate.evaluate", class_namespace["evaluate"]),
        ("gate.__call__", class_namespace["__call__"]),
        (
            "evidence.authenticate",
            authenticate_lfm2_cpcf_reviewed_promotion_evidence,
        ),
        ("evidence.load", load_lfm2_cpcf_reviewed_promotion_evidence),
        ("evidence.__init__", evidence_namespace["__init__"]),
        ("evidence.__post_init__", evidence_namespace["__post_init__"]),
        (
            "evidence.bridge_artifact_sha256",
            evidence_namespace["bridge_artifact_sha256"].fget,
        ),
        (
            "evidence.bridge_profile_sha256",
            evidence_namespace["bridge_profile_sha256"].fget,
        ),
        (
            "evidence.loader_seal_body",
            _reviewed_evidence_loader_seal_body,
        ),
        ("evidence.verify_cpcf_run_receipt", _verify_cpcf_run_receipt),
        ("evidence.verify_parameter_inventory", _verify_parameter_inventory),
        ("evidence.verify_parameter_receipt", _verify_parameter_receipt),
        ("evidence.normalize_model_fold_policy", _normalize_model_fold_policy),
        (
            "evidence.normalize_v5_model_fold_policy",
            _normalize_v5_model_fold_policy,
        ),
        (
            "evidence.authenticate_v5_numerical_adapter_contract",
            _authenticate_v5_numerical_adapter_contract,
        ),
        (
            "evidence.authenticate_v5_current_gpu_execution",
            _authenticate_v5_current_gpu_execution,
        ),
        (
            "evidence.typed_v5_gpu_device_receipt",
            _typed_v5_gpu_device_receipt,
        ),
        (
            "evidence.typed_v5_installed_state",
            _typed_v5_installed_state,
        ),
        (
            "evidence.typed_v5_replica_receipt",
            _typed_v5_replica_receipt,
        ),
        (
            "evidence.typed_v5_scheduler_policy",
            _typed_v5_scheduler_policy,
        ),
        (
            "evidence.authenticate_v5_gpu_candidate_runtime",
            _authenticate_v5_gpu_candidate_runtime,
        ),
        ("evidence.typed_v5_replay_plan", _typed_v5_replay_plan),
        ("evidence.finite_float", _finite_float),
        (
            "evidence.typed_v5_parity_structure",
            _typed_v5_parity_structure,
        ),
        (
            "evidence.authenticate_v5_parity_tensor_metrics",
            _authenticate_v5_parity_tensor_metrics,
        ),
        (
            "evidence.authenticate_v5_canonical_assignments",
            _authenticate_v5_canonical_assignments,
        ),
        (
            "evidence.verify_v5_compiled_model_fold_semantics",
            _verify_v5_compiled_model_fold_semantics,
        ),
        (
            "evidence.verify_v5_compiled_model_fold_metadata",
            _verify_v5_compiled_model_fold_metadata,
        ),
        (
            "v5.compiler_implementation",
            lfm2_cpcf_v5_compiler_implementation,
        ),
        (
            "v5.runtime_policy.__post_init__",
            Lfm2CpcfV5RuntimePolicy.__post_init__,
        ),
        ("v5.runtime_policy.as_dict", Lfm2CpcfV5RuntimePolicy.as_dict),
        (
            "v5.replay_scheduler_policy.__post_init__",
            ReplaySchedulerPolicy.__post_init__,
        ),
        (
            "v5.replay_scheduler_policy.as_dict",
            ReplaySchedulerPolicy.as_dict,
        ),
        (
            "v5.cuda_device_receipt.__post_init__",
            CudaReplayDeviceReceipt.__post_init__,
        ),
        (
            "v5.cuda_device_receipt.as_dict",
            CudaReplayDeviceReceipt.as_dict,
        ),
        (
            "v5.installed_state.__post_init__",
            InstalledReplayStateFingerprint.__post_init__,
        ),
        (
            "v5.installed_state.as_dict",
            InstalledReplayStateFingerprint.as_dict,
        ),
        (
            "v5.replica_receipt.__post_init__",
            ReplayReplicaReceipt.__post_init__,
        ),
        (
            "v5.replica_receipt.as_dict",
            ReplayReplicaReceipt.as_dict,
        ),
        (
            "v5.replay_assignment.__post_init__",
            ReplayAssignment.__post_init__,
        ),
        ("v5.replay_assignment.as_dict", ReplayAssignment.as_dict),
        (
            "v5.replay_plan.__post_init__",
            ReplaySchedulePlan.__post_init__,
        ),
        ("v5.replay_plan.as_dict", ReplaySchedulePlan.as_dict),
        (
            "v5.numerical_adapter_contract",
            _real_v5_numerical_adapter_contract,
        ),
        ("v5.numerical_layer_proof", _prove_real_v5_numerical_layer),
        (
            "v5.endpoint_first_rescore",
            rescore_sealed_v4_short_conv_selection_frontier,
        ),
        (
            "v5.completed_layer_receipt.__post_init__",
            Lfm2CpcfCompletedLayerReceipt.__post_init__,
        ),
        (
            "v5.numerical_adapter.__post_init__",
            Lfm2CpcfShortConvV5NumericalFrontierAdapter.__post_init__,
        ),
        (
            "v5.basis_policy.__post_init__",
            Lfm2CpcfV5FirBasisPolicy.__post_init__,
        ),
        (
            "v5.basis_policy.authentication_body",
            Lfm2CpcfV5FirBasisPolicy.authentication_body,
        ),
        (
            "v5.ridge_policy.__post_init__",
            Lfm2CpcfV5RidgePolicy.__post_init__,
        ),
        (
            "v5.ridge_policy.authentication_body",
            Lfm2CpcfV5RidgePolicy.authentication_body,
        ),
        (
            "evidence.parse_short_conv_v4_solver_policy",
            parse_lfm2_cpcf_short_conv_v4_solver_policy,
        ),
        (
            "short_conv_v4_policy.__post_init__",
            Lfm2CpcfShortConvV4SolverPolicy.__post_init__,
        ),
        (
            "short_conv_v4_policy.authentication_body",
            Lfm2CpcfShortConvV4SolverPolicy.authentication_body,
        ),
        (
            "short_conv_v4_policy.verify_authentication",
            Lfm2CpcfShortConvV4SolverPolicy.verify_authentication,
        ),
        (
            "short_conv_v4_policy.as_dict",
            Lfm2CpcfShortConvV4SolverPolicy.as_dict,
        ),
        ("evidence.ordered_phase_binding_body", _ordered_phase_binding_body),
        ("evidence.verify_blank_target_report", _verify_blank_target_report),
        ("evidence.verify_compiled_model_fold", _verify_compiled_model_fold),
        ("evidence.extract_cpcf_run_provenance", _extract_cpcf_run_provenance),
        ("evidence.target_plan_binding_body", _target_plan_binding_body),
        ("replay._greedy_replay_rows", _greedy_replay_rows),
        ("replay._model_device", _model_device),
        ("replay._normalized_eos_token_id", _normalized_eos_token_id),
        ("state.fingerprint", fingerprint_cpcf_candidate_state),
        ("tokenizer.package_fingerprint", fingerprint_local_tokenizer_package),
        ("tokenizer.semantics_fingerprint", fingerprint_hf_tokenizer_semantics),
        ("execution.observe_model_execution", observe_model_execution),
        ("execution.contract_satisfied", execution_contract_satisfied),
        (
            "candidate.seal_outputs",
            seal_lfm2_intelligence_holdout_candidate_outputs,
        ),
        ("binding.build", build_behavior_gate_binding),
        ("binding.expose", expose_behavior_gate_binding),
        ("binding.resolve", resolve_behavior_gate_binding),
        ("binding.identify_callable", identify_behavior_callable),
        ("canonical.json_bytes", canonical_json_bytes),
    )


def _reviewed_gate_implementation_manifest(
    targets: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    rows = []
    for name, value in targets:
        try:
            identity = identify_behavior_callable(
                value,
                implementation_id=(f"neural-forge.lfm2-cpcf-reviewed-gate.{name}"),
                version=1,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise Lfm2CpcfReviewedPromotionError(
                f"Reviewed CPCF implementation callable {name} is invalid."
            ) from exc
        rows.append(
            {
                "name": name,
                "identity": identity.as_dict(),
            }
        )
    return {
        "schema": IMPLEMENTATION_SEAL_SCHEMA,
        "callables": rows,
    }


_EXPECTED_GATE_IMPLEMENTATION_TARGETS = _reviewed_gate_implementation_targets()
_EXPECTED_GATE_IMPLEMENTATION_MANIFEST = _reviewed_gate_implementation_manifest(
    _EXPECTED_GATE_IMPLEMENTATION_TARGETS
)
_EXPECTED_GATE_IMPLEMENTATION_SEAL_SHA256 = canonical_sha256(
    _EXPECTED_GATE_IMPLEMENTATION_MANIFEST
)


def authenticate_lfm2_cpcf_reviewed_gate_implementation() -> str:
    """Reauthenticate class code, replay code, and bound callable resources."""

    live_targets = _reviewed_gate_implementation_targets()
    if len(live_targets) != len(_EXPECTED_GATE_IMPLEMENTATION_TARGETS) or any(
        live_name != expected_name or live_value is not expected_value
        for (live_name, live_value), (expected_name, expected_value) in zip(
            live_targets,
            _EXPECTED_GATE_IMPLEMENTATION_TARGETS,
            strict=True,
        )
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF gate or replay implementation object was replaced."
        )
    live_manifest = _reviewed_gate_implementation_manifest(live_targets)
    live_seal = canonical_sha256(live_manifest)
    if live_seal != _EXPECTED_GATE_IMPLEMENTATION_SEAL_SHA256 or canonical_json_bytes(
        live_manifest
    ) != canonical_json_bytes(_EXPECTED_GATE_IMPLEMENTATION_MANIFEST):
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF gate or replay callable code/resources changed."
        )
    return live_seal


def authenticate_lfm2_cpcf_reviewed_promotion_gate(
    value: Any,
) -> Lfm2CpcfReviewedPromotionGate:
    """Authenticate a constructor-built immutable gate before external use."""

    implementation_seal = authenticate_lfm2_cpcf_reviewed_gate_implementation()
    if type(value) is not Lfm2CpcfReviewedPromotionGate:
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF evaluator must have the exact sealed gate type."
        )
    try:
        construction_complete = object.__getattribute__(
            value,
            "_construction_complete",
        )
        instance_seal = object.__getattribute__(
            value,
            "_implementation_seal_sha256",
        )
        evidence = object.__getattribute__(value, "evidence")
        binding = object.__getattribute__(value, "_behavior_gate_binding")
    except AttributeError as exc:
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF gate was not constructor-authenticated."
        ) from exc
    if (
        construction_complete is not True
        or instance_seal != implementation_seal
        or type(evidence) is not Lfm2CpcfReviewedPromotionEvidence
        or binding.baseline_fingerprint != evidence.full_reviewed_evidence_fingerprint
        or binding.gate_id != BEHAVIOR_GATE_ID
        or binding.gate_version != BEHAVIOR_GATE_VERSION
    ):
        raise Lfm2CpcfReviewedPromotionError(
            "Reviewed CPCF gate construction or evidence binding is invalid."
        )
    authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)
    return value


type.__setattr__(
    Lfm2CpcfReviewedPromotionGate,
    "_reviewed_gate_class_sealed",
    True,
)


__all__ = [
    "AUTOMATIC_RUNNER_REPORT_SCHEMA",
    "BEHAVIOR_GATE_ID",
    "BEHAVIOR_GATE_VERSION",
    "COMPOSITE_PROMOTION_SCHEMA",
    "CPCF_CAPTURE_SCHEMA",
    "CPCF_RUN_SCHEMA",
    "IMPLEMENTATION_SEAL_SCHEMA",
    "SUPPORTED_COMPILED_MODEL_FOLD_SCHEMAS",
    "CanonicalJsonArtifact",
    "Lfm2CpcfReviewedPromotionArtifacts",
    "Lfm2CpcfReviewedPromotionError",
    "Lfm2CpcfReviewedPromotionEvidence",
    "Lfm2CpcfReviewedPromotionGate",
    "authenticate_lfm2_cpcf_reviewed_gate_implementation",
    "authenticate_lfm2_cpcf_reviewed_promotion_evidence",
    "authenticate_lfm2_cpcf_reviewed_promotion_gate",
    "build_cpcf_reviewed_promotion_report",
    "fingerprint_cpcf_candidate_state",
    "fingerprint_local_tokenizer_package",
    "load_lfm2_cpcf_reviewed_promotion_evidence",
]
