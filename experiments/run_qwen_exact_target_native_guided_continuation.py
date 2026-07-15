#!/usr/bin/env python3
"""Run the authenticated guided-QKVO continuation of exact Qwen stage 1.

This runner consumes, rather than repeats, the terminal lexical continuation
and its passed output-Jacobian audit.  Its model-free preflight authenticates
the complete historical chain, the immutable guided basis, a new behavior
selection contract, and the exact compiler-cache identity before model access.

Only causal output actions and free-running rollouts may select a checkpoint.
The temporary guided product is installed as a separate zero-left tranche on
the existing QKVO transaction.  A rejection retains the selected factor-only
pack and proves byte-exact rollback.  An acceptance uses the guided-aware
composite publication handle, then freshly reloads the exact-size native child;
no parent, adapter, factor, or sidecar is part of inference.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import importlib.util
import json
import re
import subprocess
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_progressive_target_bridge import ExternalBehaviorGateReceipt


RUN_SCHEMA = "neural-forge-qwen-exact-target-native-guided-continuation-v1"
PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-exact-target-native-guided-continuation-preregistration-v1"
)
FACTOR_ENVELOPE_SCHEMA = "neural-forge-rejected-target-native-guided-factor-pack-v1"
TRAINER_FACTOR_PACK_V3_SCHEMA = "neural-forge-target-native-factor-pack-v3"
GUIDED_TRAINER_MODULE = stage.CANONICAL_TRAINER_MODULE
GUIDED_ADAPTER_MODULE = "neural_forge.hf_dense_gqa_qkvo_residual_adapter"
GUIDED_BUILDER_FILENAME = "prepare_qwen_exact_target_native_guided_continuation.py"
GUIDED_HISTORICAL_IMPLEMENTATION_PATHS = {
    "runner_sha256": (
        "experiments/run_qwen_exact_target_native_guided_continuation.py"
    ),
    "builder_sha256": (
        "experiments/prepare_qwen_exact_target_native_guided_continuation.py"
    ),
    "base_runner_sha256": "experiments/run_qwen_exact_target_native_stage.py",
    "trainer_sha256": (
        "src/neural_forge/hf_target_native_behavior_distillation.py"
    ),
    "guided_adapter_sha256": (
        "src/neural_forge/hf_dense_gqa_qkvo_residual_adapter.py"
    ),
    "progressive_bridge_sha256": (
        "src/neural_forge/hf_progressive_target_bridge.py"
    ),
    "compiled_child_cache_sha256": (
        "src/neural_forge/hf_compiled_child_cache.py"
    ),
}
GUIDED_FACTOR_COUNT = 8
GUIDED_FACTOR_PARAMETERS = 7_127_040
GUIDED_TOTAL_FACTOR_COUNT = (
    stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT + GUIDED_FACTOR_COUNT
)
GUIDED_TOTAL_FACTOR_PARAMETERS = (
    stage.LEXICAL_TARGET_NATIVE_FACTOR_PARAMETERS + GUIDED_FACTOR_PARAMETERS
)
GUIDED_ROLES = ("q", "k", "v", "o")
GUIDED_SPLIT_COUNTS = {"fit": 144, "candidate": 24, "sealed": 32}
SELECTION_DOMAIN = "causal-output-actions-and-free-rollouts-only"
SCHEDULE_POLICY = (
    "steps-1-8-guided-left-only;steps-9-24-guided-left-right;"
    "steps-25-64-joint-old-guided"
)
GUIDED_METHOD_GATE = {
    "minimum_selected_step": 1,
    "minimum_candidate_lcp_at_least_4_count": 1,
    "require_candidate_repeat_fraction_excess_nonregression_vs_step0": True,
    "require_candidate_adjacent_repeat_fraction_excess_nonregression_vs_step0": True,
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GuidedContinuationError(RuntimeError):
    """A guided provenance, behavior, rollback, or publication gate failed."""


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise GuidedContinuationError(f"{name} must be a lowercase SHA-256.")
    return value


def _strict_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GuidedContinuationError(
            f"{name} must contain exactly {sorted(expected)}; got {actual!r}."
        )
    return value


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise GuidedContinuationError(f"{name} must be a nonempty path string.")
    raw = Path(value).expanduser()
    path = (base / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if path.is_symlink():
        raise GuidedContinuationError(f"{name} cannot be a symlink.")
    return path


def _binding(
    value: Any,
    *,
    base: Path,
    name: str,
    expected_keys: set[str] | None = None,
) -> tuple[Path, str]:
    row = _strict_keys(
        value,
        {"path", "sha256"} if expected_keys is None else expected_keys,
        name,
    )
    path = _resolve(base, row.get("path"), f"{name}.path")
    digest = _require_sha256(row.get("sha256"), f"{name}.sha256")
    if not path.is_file() or stage._file_sha256(path) != digest:
        raise GuidedContinuationError(f"{name} is absent or its SHA-256 changed.")
    return path, digest


def _module_sha256(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise GuidedContinuationError(
            f"Required module {module_name!r} is unavailable."
        )
    return stage._file_sha256(Path(spec.origin).resolve())


@dataclass(frozen=True, slots=True)
class GuidedAuditArtifacts:
    audit_preregistration_path: Path
    audit_preregistration_sha256: str
    audit_run_report_path: Path
    audit_run_report_sha256: str
    guided_basis_pack_path: Path
    basis_envelope_file_sha256: str
    basis_domain_artifact_sha256: str
    basis_payload_sha256: str
    audit_report_self_sha256: str
    audit_report_canonical_sha256: str
    basis_manifest_sha256: str
    source_factor_state_sha256: str
    source_run_report_path: Path
    source_run_report_sha256: str
    source_factor_pack_path: Path
    source_factor_pack_sha256: str
    audit_report: Mapping[str, Any]
    basis_pack: Mapping[str, Any]
    basis_tensor_sha256s: Mapping[str, str]

    def as_report(self) -> dict[str, Any]:
        return {
            "audit_preregistration": str(self.audit_preregistration_path),
            "audit_preregistration_sha256": self.audit_preregistration_sha256,
            "audit_run_report": str(self.audit_run_report_path),
            "audit_run_report_sha256": self.audit_run_report_sha256,
            "guided_basis_pack": str(self.guided_basis_pack_path),
            "basis_envelope_file_sha256": self.basis_envelope_file_sha256,
            "basis_domain_artifact_sha256": self.basis_domain_artifact_sha256,
            "basis_payload_sha256": self.basis_payload_sha256,
            "audit_report_self_sha256": self.audit_report_self_sha256,
            "audit_report_canonical_sha256": self.audit_report_canonical_sha256,
            "basis_manifest_sha256": self.basis_manifest_sha256,
            "source_factor_state_sha256": self.source_factor_state_sha256,
            "source_run_report": str(self.source_run_report_path),
            "source_run_report_sha256": self.source_run_report_sha256,
            "source_factor_pack": str(self.source_factor_pack_path),
            "source_factor_pack_sha256": self.source_factor_pack_sha256,
            "authorized_roles": list(GUIDED_ROLES),
            "basis_tensor_sha256s": dict(self.basis_tensor_sha256s),
            "authenticated_before_model_access": True,
        }


@dataclass(frozen=True, slots=True)
class AuthenticatedGuidedContinuation:
    path: Path
    file_sha256: str
    source_contract: stage.AuthenticatedPreRegistration
    execution_contract: stage.AuthenticatedPreRegistration
    artifacts: GuidedAuditArtifacts
    evaluation_contract_sha256: str
    implementation_authentication: Mapping[str, Any]
    source_authentication: Mapping[str, Any]
    training: Mapping[str, Any]
    method_gate: Mapping[str, Any]

    def preflight_report(self) -> dict[str, Any]:
        base = self.execution_contract.preflight_report()
        return {
            "schema": RUN_SCHEMA,
            "status": "preflight-passed",
            "preregistration": str(self.path),
            "preregistration_sha256": self.file_sha256,
            "evaluation_contract_sha256": self.evaluation_contract_sha256,
            "parent": base["parent"],
            "target": base["target"],
            "compiler_split": base["compiler_split"],
            "splits": base["splits"],
            "thresholds": base["thresholds"],
            "guided_qkvo": {
                "roles": list(GUIDED_ROLES),
                "site_count": stage.QKVO_SITE_COUNT,
                "rank": stage.QKVO_RANK,
                "new_factor_count": GUIDED_FACTOR_COUNT,
                "new_factor_parameters": GUIDED_FACTOR_PARAMETERS,
                "combined_factor_count": GUIDED_TOTAL_FACTOR_COUNT,
                "combined_factor_parameters": GUIDED_TOTAL_FACTOR_PARAMETERS,
                "separate_product_no_regroup": True,
                "zero_left_install_required": True,
                "schedule_policy": SCHEDULE_POLICY,
            },
            "source": dict(self.source_authentication),
            "audit": self.artifacts.as_report(),
            "implementation_authentication": dict(self.implementation_authentication),
            "selection_domain": SELECTION_DOMAIN,
            "method_gate": dict(self.method_gate),
            "internal_numeric_matching_can_promote": False,
            "sealed_evaluation_locked_until_candidate_passes": True,
            "models_loaded": False,
            "gpu_work_launched": False,
        }


class _PersistedAuditResult:
    """Minimal adapter into the already exhaustive source audit verifier."""

    def __init__(
        self, report: Mapping[str, Any], basis_pack: Mapping[str, Any]
    ) -> None:
        self.report = SimpleNamespace(to_dict=lambda: dict(report))
        self._basis_pack = basis_pack

    def basis_pack_payload(self) -> Mapping[str, Any]:
        return self._basis_pack


def _load_torch_mapping(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise GuidedContinuationError(f"Could not load {name} safely.") from error
    if not isinstance(value, Mapping):
        raise GuidedContinuationError(f"{name} must contain a mapping.")
    return value


def _guided_basis_payload_sha256(basis_pack: Mapping[str, Any]) -> str:
    """Use the adapter's tensor-aware digest; never serialize a second pack."""

    module = importlib.import_module(GUIDED_ADAPTER_MODULE)
    helper = getattr(module, "guided_qkvo_basis_payload_sha256", None)
    if not callable(helper):
        # During model-free source checkout construction this fallback has the
        # same manifest/tensor binding domain as the authenticated audit.  The
        # production adapter helper is still source-pinned and mandatory at
        # execution time.
        factors = basis_pack.get("factors")
        if not isinstance(factors, list):
            raise GuidedContinuationError(
                "Guided adapter lacks its tensor-aware basis pack digest helper."
            )
        return stage._canonical_sha256(
            {
                "schema": basis_pack.get("schema"),
                "factor_state_sha256": basis_pack.get("factor_state_sha256"),
                "basis_manifest_sha256": basis_pack.get("basis_manifest_sha256"),
                "site_indices_sha256": basis_pack.get("site_indices_sha256"),
                "factors": factors,
                "authorized_roles": basis_pack.get("authorized_roles"),
            }
        )
    result = helper(basis_pack)
    return _require_sha256(result, "guided_basis_payload_sha256")


def _authenticate_persisted_audit(
    audit_value: Mapping[str, Any],
    source_value: Mapping[str, Any],
    *,
    base: Path,
) -> tuple[stage.AuthenticatedPreRegistration, GuidedAuditArtifacts]:
    audit_row = _strict_keys(
        audit_value,
        {
            "preregistration",
            "historical_git_revision",
            "run_report",
            "guided_basis_pack",
            "basis_envelope_file_sha256",
            "basis_domain_artifact_sha256",
            "basis_payload_sha256",
            "audit_report_self_sha256",
            "audit_report_canonical_sha256",
            "basis_manifest_sha256",
            "authorized_roles",
            "basis_tensor_sha256s",
        },
        "audit",
    )
    audit_prereg_path, audit_prereg_sha = _binding(
        audit_row["preregistration"], base=base, name="audit.preregistration"
    )
    historical_revision = audit_row.get("historical_git_revision")
    if (
        not isinstance(historical_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", historical_revision) is None
    ):
        raise GuidedContinuationError(
            "Audit historical revision must be a full commit."
        )
    run_path, run_sha = _binding(
        audit_row["run_report"], base=base, name="audit.run_report"
    )
    run_report = stage._strict_json(run_path)
    if not isinstance(run_report, Mapping):
        raise GuidedContinuationError("Audit run report must be a JSON object.")
    historical_implementation = run_report.get("implementation_authentication")
    try:
        audit_contract = stage.authenticate_preregistration(
            audit_prereg_path,
            audit_prereg_sha,
            historical_git_revision=historical_revision,
            historical_implementation_authentication=historical_implementation,
        )
    except stage.ExactTargetNativeStageError as error:
        raise GuidedContinuationError(
            "The source output-Jacobian preregistration no longer authenticates."
        ) from error
    if audit_contract.historical_implementation_revision != historical_revision:
        raise GuidedContinuationError(
            "The source output-Jacobian contract is not marked as exact historical "
            "replay evidence."
        )
    if (
        audit_contract.output_jacobian_audit is None
        or audit_contract.continuation is None
    ):
        raise GuidedContinuationError(
            "Guided continuation requires an authenticated continuation audit."
        )

    if audit_contract.implementation_authentication != historical_implementation:
        raise GuidedContinuationError(
            "Historical source authentication did not reproduce terminal evidence."
        )
    basis_path, basis_file_sha = _binding(
        audit_row["guided_basis_pack"],
        base=base,
        name="audit.guided_basis_pack",
    )
    reported_basis = run_report.get("guided_basis_pack")
    if not isinstance(reported_basis, Mapping):
        raise GuidedContinuationError("Audit run report omits its guided basis pack.")
    reported_basis_path = _resolve(
        run_path.parent,
        reported_basis.get("path"),
        "audit.run_report.guided_basis_pack.path",
    )
    report_audit = run_report.get("output_jacobian_audit_report")
    report_audit_self_sha = _require_sha256(
        audit_row.get("audit_report_self_sha256"),
        "audit.audit_report_self_sha256",
    )
    report_audit_canonical_sha = _require_sha256(
        audit_row.get("audit_report_canonical_sha256"),
        "audit.audit_report_canonical_sha256",
    )
    basis_manifest_sha = _require_sha256(
        audit_row.get("basis_manifest_sha256"), "audit.basis_manifest_sha256"
    )
    continuation = audit_contract.continuation
    expected_top_level = {
        "status": "diagnostic-passed",
        "reason": "output-jacobian-audit-passed",
        "preregistration_sha256": audit_prereg_sha,
        "evaluation_contract_sha256": audit_contract.evaluation_contract_sha256,
        "output_jacobian_audit_report_sha256": report_audit_canonical_sha,
        "sealed_not_accessed": True,
        "training_started": False,
        "optimizer_created": False,
        "optimizer_steps": 0,
        "training_checkpoints_written": 0,
        "native_factor_bake_performed": False,
        "model_checkpoint_written": False,
        "standalone_compaction_passed": False,
        "rollback_exact": True,
        "rollback_state_sha256": continuation.compiled_state_sha256,
    }
    if (
        run_report.get("schema") != stage.RUN_SCHEMA
        or any(
            run_report.get(key) != value for key, value in expected_top_level.items()
        )
        or not isinstance(report_audit, Mapping)
        or report_audit.get("report_sha256") != report_audit_self_sha
        or stage._canonical_sha256(report_audit) != report_audit_canonical_sha
        or reported_basis_path != basis_path
        or reported_basis.get("sha256") != basis_file_sha
        or reported_basis.get("size_bytes") != basis_path.stat().st_size
        or audit_row.get("basis_envelope_file_sha256") != basis_file_sha
    ):
        raise GuidedContinuationError(
            "Audit terminal status, rollback, report, or basis binding changed."
        )

    envelope = _load_torch_mapping(basis_path, "guided basis envelope")
    expected_envelope_keys = {
        "schema",
        "preregistration_sha256",
        "evaluation_contract_sha256",
        "output_jacobian_audit",
        "output_jacobian_audit_report_sha256",
        "source_factor_state_sha256",
        "source_factor_pack_schema",
        "basis_pack",
        "contains_model_checkpoint",
        "contains_parent_parameters",
        "contains_optimizer_state",
        "contains_activations_or_hidden_states",
    }
    _strict_keys(envelope, expected_envelope_keys, "guided basis envelope")
    basis_pack = envelope.get("basis_pack")
    if not isinstance(basis_pack, Mapping):
        raise GuidedContinuationError("Guided basis envelope omits its basis mapping.")
    restore = run_report.get("output_jacobian_restore")
    verification = run_report.get("output_jacobian_factor_verification")
    if not isinstance(restore, Mapping) or not isinstance(verification, Mapping):
        raise GuidedContinuationError("Audit restore/factor verification is absent.")
    execution = stage.OutputJacobianAuditExecution(
        result=_PersistedAuditResult(report_audit, basis_pack),
        restore_mode=restore.get("mode"),
        restored_factor_state_sha256=restore.get("factor_state_sha256"),
        restored_factor_schema=restore.get("factor_pack_schema"),
        restored_factor_lexical_rank=restore.get("lexical_rank"),
        restored_state_clone_tensor_count=restore.get(
            "restored_state_clone_tensor_count"
        ),
        restored_state_clone_cpu_bytes=restore.get("restored_state_clone_cpu_bytes"),
        restored_state_released_before_audit=restore.get(
            "restored_state_released_before_audit"
        ),
        live_target_native_factor_state_clone_tensor_count_during_audit=verification.get(
            "live_target_native_factor_state_clone_tensor_count_during_audit"
        ),
        live_target_native_factor_state_clone_cpu_bytes_during_audit=verification.get(
            "live_target_native_factor_state_clone_cpu_bytes_during_audit"
        ),
        post_audit_full_factor_capture_helper_called=verification.get(
            "post_audit_full_factor_capture_helper_called"
        ),
        post_audit_factor_verification_mode=verification.get("mode"),
    )
    try:
        canonical_report, canonical_basis = stage._output_jacobian_audit_artifacts(
            execution, audit_contract
        )
    except (stage.ExactTargetNativeStageError, TypeError, ValueError) as error:
        raise GuidedContinuationError(
            "Persisted output-Jacobian evidence failed canonical replay verification."
        ) from error

    expected_envelope = stage._guided_basis_envelope(
        canonical_basis, canonical_report, audit_contract
    )
    scalar_envelope = {
        key: value for key, value in envelope.items() if key != "basis_pack"
    }
    expected_scalar_envelope = {
        key: value for key, value in expected_envelope.items() if key != "basis_pack"
    }
    if stage._canonical_json(scalar_envelope) != stage._canonical_json(
        expected_scalar_envelope
    ):
        raise GuidedContinuationError(
            "Guided basis outer envelope disagrees with the authenticated audit."
        )
    if (
        canonical_report.get("passed") is not True
        or canonical_report.get("authorized_roles") != list(GUIDED_ROLES)
        or canonical_report.get("installable_roles") != list(GUIDED_ROLES)
        or canonical_basis.get("authorized_for_training") is not True
        or canonical_basis.get("authorized_roles") != list(GUIDED_ROLES)
        or canonical_basis.get("installable_roles") != list(GUIDED_ROLES)
        or canonical_basis.get("basis_manifest_sha256") != basis_manifest_sha
    ):
        raise GuidedContinuationError(
            "Every Q/K/V/O role must independently pass and be installable."
        )

    if canonical_report.get("report_sha256") != report_audit_self_sha:
        raise GuidedContinuationError("Nested audit self-hash domain changed.")
    basis_domain_sha = _require_sha256(
        canonical_report.get("basis_artifact_sha256"),
        "output_jacobian_audit_report.basis_artifact_sha256",
    )
    if (
        basis_domain_sha
        != _require_sha256(
            audit_row.get("basis_domain_artifact_sha256"),
            "audit.basis_domain_artifact_sha256",
        )
        or basis_domain_sha != audit_contract.splits["candidate"].sha256
    ):
        raise GuidedContinuationError("Audit basis-domain artifact hash changed.")
    payload_sha = _guided_basis_payload_sha256(canonical_basis)
    if payload_sha != _require_sha256(
        audit_row.get("basis_payload_sha256"),
        "audit.basis_payload_sha256",
    ):
        raise GuidedContinuationError("Guided basis content digest changed.")
    declared_roles = audit_row.get("authorized_roles")
    if declared_roles != list(GUIDED_ROLES):
        raise GuidedContinuationError("Outer audit role authorization changed.")
    declared_tensor_hashes = audit_row.get("basis_tensor_sha256s")
    rows = canonical_basis.get("factors")
    if not isinstance(rows, list):
        raise GuidedContinuationError("Guided basis manifest is malformed.")
    manifest_hashes = {
        str(row["name"]): _require_sha256(
            row.get("tensor_sha256"), f"audit.basis.{row.get('name')}.tensor_sha256"
        )
        for row in rows
        if isinstance(row, Mapping)
    }
    expected_basis_names = {
        "q": "q_orthogonal_right",
        "k": "k_orthogonal_right",
        "v": "v_orthogonal_right",
        "o": "o_orthogonal_right",
    }
    if set(manifest_hashes) != set(expected_basis_names.values()):
        raise GuidedContinuationError("Guided basis tensor names changed.")
    tensor_hashes = {
        role: manifest_hashes[name] for role, name in expected_basis_names.items()
    }
    if declared_tensor_hashes != tensor_hashes or len(tensor_hashes) != 4:
        raise GuidedContinuationError("Per-role guided tensor hashes changed.")

    source_row = _strict_keys(
        source_value,
        {
            "run_report",
            "factor_pack",
            "factor_state_sha256",
            "source_preregistration_sha256",
            "source_evaluation_contract_sha256",
            "compiled_state_sha256",
            "compiler_report_sha256",
            "parent_tree_sha256",
            "blank_target_tree_sha256",
            "compiler_cache",
        },
        "source",
    )
    source_report_path, source_report_sha = _binding(
        source_row["run_report"], base=base, name="source.run_report"
    )
    factor_path, factor_sha = _binding(
        source_row["factor_pack"], base=base, name="source.factor_pack"
    )
    if (
        source_report_path != continuation.source_run_report_path
        or source_report_sha != continuation.source_run_report_sha256
        or factor_path != continuation.factor_pack_path
        or factor_sha != continuation.factor_pack_sha256
        or source_row.get("factor_state_sha256") != continuation.factor_state_sha256
        or source_row.get("source_preregistration_sha256")
        != continuation.source_preregistration_sha256
        or source_row.get("source_evaluation_contract_sha256")
        != continuation.source_evaluation_contract_sha256
        or source_row.get("compiled_state_sha256") != continuation.compiled_state_sha256
        or source_row.get("compiler_report_sha256")
        != continuation.compiler_report_sha256
    ):
        raise GuidedContinuationError(
            "Lexical continuation source/factor/state provenance changed."
        )
    parent_tree = audit_contract.artifact_authentication["parent"].get("tree")
    target_tree = audit_contract.artifact_authentication["blank_target"].get("tree")
    if (
        not isinstance(parent_tree, Mapping)
        or not isinstance(target_tree, Mapping)
        or source_row.get("parent_tree_sha256") != parent_tree.get("tree_sha256")
        or source_row.get("blank_target_tree_sha256") != target_tree.get("tree_sha256")
    ):
        raise GuidedContinuationError("Parent or exact blank artifact tree changed.")
    cache = audit_contract.compiler_cache
    cache_row = _strict_keys(
        source_row.get("compiler_cache"),
        {
            "compiler_identity_sha256",
            "receipt_sha256",
            "compiler_report_sha256",
            "state_dict_sha256",
            "artifact_tree_sha256",
        },
        "source.compiler_cache",
    )
    expected_cache = (
        None
        if cache is None
        else {
            "compiler_identity_sha256": cache.expected_compiler_identity_sha256,
            "receipt_sha256": cache.expected_receipt_sha256,
            "compiler_report_sha256": cache.expected_compiler_report_sha256,
            "state_dict_sha256": cache.expected_state_dict_sha256,
            "artifact_tree_sha256": cache.expected_artifact_tree_sha256,
        }
    )
    if (
        cache is None
        or cache.require_hit is not True
        or dict(cache_row) != expected_cache
    ):
        raise GuidedContinuationError("Immutable compiled-child cache pins changed.")

    return audit_contract, GuidedAuditArtifacts(
        audit_preregistration_path=audit_prereg_path,
        audit_preregistration_sha256=audit_prereg_sha,
        audit_run_report_path=run_path,
        audit_run_report_sha256=run_sha,
        guided_basis_pack_path=basis_path,
        basis_envelope_file_sha256=basis_file_sha,
        basis_domain_artifact_sha256=basis_domain_sha,
        basis_payload_sha256=payload_sha,
        audit_report_self_sha256=report_audit_self_sha,
        audit_report_canonical_sha256=report_audit_canonical_sha,
        basis_manifest_sha256=basis_manifest_sha,
        source_factor_state_sha256=continuation.factor_state_sha256,
        source_run_report_path=source_report_path,
        source_run_report_sha256=source_report_sha,
        source_factor_pack_path=factor_path,
        source_factor_pack_sha256=factor_sha,
        audit_report=canonical_report,
        basis_pack=canonical_basis,
        basis_tensor_sha256s=tensor_hashes,
    )


def _authenticate_historical_git_blob_pins(
    implementation: Mapping[str, Any],
    revision: Any,
    *,
    paths: Mapping[str, str],
) -> None:
    """Authenticate declared source hashes against one ancestor commit tree."""

    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise GuidedContinuationError(
            "Historical implementation revision must be a full lowercase commit SHA."
        )
    if set(paths) != set(implementation) - {"trainer_module"}:
        raise GuidedContinuationError(
            "Historical implementation path map does not cover the exact source pins."
        )
    repo = Path(__file__).resolve().parents[1]
    environment = stage._historical_git_environment()
    try:
        resolved_repo = (
            subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repo),
                    "rev-parse",
                    "--show-toplevel",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            .stdout.decode("utf-8")
            .strip()
        )
        resolved_revision = (
            subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repo),
                    "rev-parse",
                    "--verify",
                    f"{revision}^{{commit}}",
                ],
                check=True,
                capture_output=True,
                env=environment,
            )
            .stdout.decode("ascii")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise GuidedContinuationError(
            "Historical implementation commit is unavailable in this repository."
        ) from error
    if Path(resolved_repo).resolve() != repo or resolved_revision != revision:
        raise GuidedContinuationError(
            "Historical implementation commit did not resolve exactly in this repository."
        )
    try:
        subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "-C",
                str(repo),
                "merge-base",
                "--is-ancestor",
                revision,
                "HEAD",
            ],
            check=True,
            capture_output=True,
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise GuidedContinuationError(
            "Historical implementation commit is not an ancestor of current HEAD."
        ) from error
    for field, relative_path in sorted(paths.items()):
        declared = _require_sha256(
            implementation.get(field), f"implementation.{field}"
        )
        try:
            blob = subprocess.run(
                [
                    "git",
                    "--no-replace-objects",
                    "-C",
                    str(repo),
                    "cat-file",
                    "blob",
                    f"{revision}:{relative_path}",
                ],
                check=True,
                capture_output=True,
                env=environment,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise GuidedContinuationError(
                f"Historical implementation path {relative_path!r} is unavailable."
            ) from error
        if hashlib.sha256(blob).hexdigest() != declared:
            raise GuidedContinuationError(
                f"Historical implementation path {relative_path!r} failed its SHA-256 pin."
            )


def _authenticate_implementation(
    value: Any,
    *,
    historical_git_revision: str | None = None,
) -> dict[str, Any]:
    row = _strict_keys(
        value,
        {
            "runner_sha256",
            "builder_sha256",
            "base_runner_sha256",
            "trainer_module",
            "trainer_sha256",
            "guided_adapter_sha256",
            "progressive_bridge_sha256",
            "compiled_child_cache_sha256",
        },
        "implementation",
    )
    if row.get("trainer_module") != GUIDED_TRAINER_MODULE:
        raise GuidedContinuationError("Guided trainer module changed.")
    if historical_git_revision is not None:
        _authenticate_historical_git_blob_pins(
            row,
            historical_git_revision,
            paths=GUIDED_HISTORICAL_IMPLEMENTATION_PATHS,
        )
        return dict(row)
    expected = {
        "runner_sha256": stage._file_sha256(Path(__file__).resolve()),
        "builder_sha256": stage._file_sha256(
            Path(__file__).resolve().with_name(GUIDED_BUILDER_FILENAME)
        ),
        "base_runner_sha256": stage._file_sha256(Path(stage.__file__).resolve()),
        "trainer_module": GUIDED_TRAINER_MODULE,
        "trainer_sha256": _module_sha256(GUIDED_TRAINER_MODULE),
        "guided_adapter_sha256": _module_sha256(GUIDED_ADAPTER_MODULE),
        "progressive_bridge_sha256": _module_sha256(
            "neural_forge.hf_progressive_target_bridge"
        ),
        "compiled_child_cache_sha256": _module_sha256(
            "neural_forge.hf_compiled_child_cache"
        ),
    }
    if dict(row) != expected:
        raise GuidedContinuationError("Guided implementation source pins changed.")
    return expected


def guided_training_schedule_from_source(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise GuidedContinuationError("Source training schedule must be a mapping.")
    training = dict(source)
    try:
        source_train_cap = int(source["max_train_prefixes"])
        source_eval_cap = int(source["max_evaluation_prefixes"])
    except (KeyError, TypeError, ValueError) as error:
        raise GuidedContinuationError("Source work ceilings are malformed.") from error
    training.update(
        {
            "steps": 64,
            "checkpoint_steps": [2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64],
            "fit_prompts_per_step": 12,
            "parent_only_steps": 8,
            "guided_continuation_enabled": True,
            "guided_qk_learning_rate_scale": 0.5,
            "guided_vo_learning_rate_scale": 1.0,
            "guided_right_factor_learning_rate_scale": 0.25,
            "guided_joint_inherited_learning_rate_scale": 0.25,
            "guided_gud_unlock_step": 0,
            "guided_lexical_embedding_unlock_step": 0,
            "guided_lexical_head_unlock_step": 41,
            "learning_rate_decay_step_1": 24,
            "learning_rate_decay_scale_1": 2.0 / 3.0,
            "learning_rate_decay_step_2": 40,
            "learning_rate_decay_scale_2": 0.5,
            "learning_rate_decay_step_3": 56,
            "learning_rate_decay_scale_3": 0.5,
            "max_train_prefixes": max(12_288, source_train_cap),
            "max_evaluation_prefixes": max(20_000, source_eval_cap),
            "max_checkpoints": 16,
            "selection_policy": "rollout-first-acquisition",
        }
    )
    module = importlib.import_module(GUIDED_TRAINER_MODULE)
    schedule_type = getattr(module, "BehaviorDistillationSchedule", None)
    if not callable(schedule_type):
        raise GuidedContinuationError("Canonical guided schedule type is unavailable.")
    try:
        normalized = schedule_type(**training).as_dict()
    except (TypeError, ValueError, RuntimeError) as error:
        raise GuidedContinuationError("Guided 64-step schedule is invalid.") from error
    return dict(normalized)


def _authenticate_training(
    value: Any, *, source_training: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise GuidedContinuationError("training must be a mapping.")
    result = dict(value)
    expected = guided_training_schedule_from_source(source_training)
    if stage._canonical_json(result) != stage._canonical_json(expected):
        raise GuidedContinuationError(
            "Training schedule differs from the exact source-derived 64-step "
            "guided curriculum."
        )
    return result


def _split_token_sequences(split: stage.SplitBinding) -> tuple[tuple[int, ...], ...]:
    prompts = split.payload.get("prompts")
    if not isinstance(prompts, list):
        raise GuidedContinuationError(f"{split.name} prompts are malformed.")
    return tuple(tuple(int(token) for token in row["token_ids"]) for row in prompts)


def _meaningful_prefixes(tokens: tuple[int, ...]) -> frozenset[tuple[int, ...]]:
    return frozenset(tokens[:length] for length in range(8, len(tokens) + 1))


def _authenticate_split_evolution(
    splits: Mapping[str, stage.SplitBinding],
    source_contract: stage.AuthenticatedPreRegistration,
) -> None:
    if {
        name: len(value.row_ids) for name, value in splits.items()
    } != GUIDED_SPLIT_COUNTS:
        raise GuidedContinuationError("Guided split counts must be exactly 144/24/32.")
    source_fit = _split_token_sequences(source_contract.splits["fit"])
    source_candidate = _split_token_sequences(source_contract.splits["candidate"])
    source_sealed = _split_token_sequences(source_contract.splits["sealed"])
    if (
        _split_token_sequences(splits["fit"]) != source_fit + source_candidate
        or splits["sealed"].path != source_contract.splits["sealed"].path
        or splits["sealed"].sha256 != source_contract.splits["sealed"].sha256
        or _split_token_sequences(splits["sealed"]) != source_sealed
    ):
        raise GuidedContinuationError(
            "Fit must be prior 120 + exposed candidate 24 and sealed must be the "
            "exact untouched prior artifact."
        )
    active = {name: _split_token_sequences(value) for name, value in splits.items()}
    seen_sequences: dict[tuple[int, ...], str] = {}
    seen_prefixes: dict[tuple[int, ...], str] = {}
    for role in ("fit", "candidate", "sealed"):
        for tokens in active[role]:
            if tokens in seen_sequences:
                raise GuidedContinuationError(
                    f"Exact prompt overlap between {seen_sequences[tokens]} and {role}."
                )
            seen_sequences[tokens] = role
            for prefix in _meaningful_prefixes(tokens):
                if prefix in seen_prefixes:
                    raise GuidedContinuationError(
                        "Meaningful token-prefix overlap leaks across guided domains."
                    )
                seen_prefixes[prefix] = role


def _guided_evaluation_preimage(
    *,
    artifacts: GuidedAuditArtifacts,
    splits: Mapping[str, stage.SplitBinding],
    thresholds: Mapping[str, stage.BehaviorThresholds],
    training: Mapping[str, Any],
    implementation: Mapping[str, Any],
    runtime: Mapping[str, Any],
    method_gate: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "source_run_report_sha256": artifacts.source_run_report_sha256,
        "source_factor_pack_sha256": artifacts.source_factor_pack_sha256,
        "source_factor_state_sha256": artifacts.source_factor_state_sha256,
        "audit_preregistration_sha256": artifacts.audit_preregistration_sha256,
        "audit_run_report_sha256": artifacts.audit_run_report_sha256,
        "audit_report_self_sha256": artifacts.audit_report_self_sha256,
        "audit_report_canonical_sha256": artifacts.audit_report_canonical_sha256,
        "basis_envelope_file_sha256": artifacts.basis_envelope_file_sha256,
        "basis_domain_artifact_sha256": artifacts.basis_domain_artifact_sha256,
        "basis_payload_sha256": artifacts.basis_payload_sha256,
        "basis_manifest_sha256": artifacts.basis_manifest_sha256,
        "basis_tensor_sha256s": dict(artifacts.basis_tensor_sha256s),
        "split_sha256s": {name: value.sha256 for name, value in splits.items()},
        "split_counts": dict(GUIDED_SPLIT_COUNTS),
        "thresholds": {name: value.as_dict() for name, value in thresholds.items()},
        "training": dict(training),
        "method_gate": dict(method_gate),
        "guided_topology": {
            "site_count": stage.QKVO_SITE_COUNT,
            "rank": stage.QKVO_RANK,
            "roles": list(GUIDED_ROLES),
            "new_factor_count": GUIDED_FACTOR_COUNT,
            "new_factor_parameters": GUIDED_FACTOR_PARAMETERS,
            "combined_factor_count": GUIDED_TOTAL_FACTOR_COUNT,
            "combined_factor_parameters": GUIDED_TOTAL_FACTOR_PARAMETERS,
            "separate_product_no_regroup": True,
            "zero_left_install_required": True,
        },
        "selection_domain": SELECTION_DOMAIN,
        "sealed_locked_until_candidate_passes": True,
        "implementation": dict(implementation),
        "runtime": dict(runtime),
    }


def authenticate_guided_preregistration(
    path: Path | str,
    expected_sha256: str,
    *,
    historical_git_revision: str | None = None,
) -> AuthenticatedGuidedContinuation:
    prereg_path = Path(path).expanduser().resolve()
    expected = _require_sha256(expected_sha256, "preregistration_sha256")
    if prereg_path.is_symlink() or not prereg_path.is_file():
        raise GuidedContinuationError("Guided preregistration is absent or a symlink.")
    if stage._file_sha256(prereg_path) != expected:
        raise GuidedContinuationError("Guided preregistration SHA-256 changed.")
    raw = stage._strict_json(prereg_path)
    prereg = _strict_keys(
        raw,
        {
            "schema",
            "sealed_before_model_access",
            "seed",
            "implementation",
            "source",
            "audit",
            "splits",
            "thresholds",
            "training",
            "method_gate",
            "runtime",
            "evaluation_contract_sha256",
        },
        "preregistration",
    )
    if (
        prereg.get("schema") != PREREGISTRATION_SCHEMA
        or prereg.get("sealed_before_model_access") is not True
    ):
        raise GuidedContinuationError("Guided preregistration schema/seal changed.")
    seed = prereg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise GuidedContinuationError("Guided seed must be a nonnegative integer.")
    runtime = _strict_keys(
        prereg.get("runtime"),
        {"dtype", "attention_implementation", "explicit_execute_required"},
        "runtime",
    )
    if dict(runtime) != {
        "dtype": "float16",
        "attention_implementation": stage.CANONICAL_ATTENTION_IMPLEMENTATION,
        "explicit_execute_required": True,
    }:
        raise GuidedContinuationError("Guided runtime contract changed.")
    implementation = (
        _authenticate_implementation(prereg.get("implementation"))
        if historical_git_revision is None
        else _authenticate_implementation(
            prereg.get("implementation"),
            historical_git_revision=historical_git_revision,
        )
    )
    source_contract, artifacts = _authenticate_persisted_audit(
        prereg.get("audit"), prereg.get("source"), base=prereg_path.parent
    )
    if seed != source_contract.seed:
        raise GuidedContinuationError(
            "Guided seed differs from the source transaction."
        )
    split_rows = _strict_keys(
        prereg.get("splits"), {"fit", "candidate", "sealed"}, "splits"
    )
    splits = {
        name: stage._load_split(split_rows[name], name=name, base=prereg_path.parent)
        for name in ("fit", "candidate", "sealed")
    }
    _authenticate_split_evolution(splits, source_contract)
    threshold_rows = _strict_keys(
        prereg.get("thresholds"), {"fit", "candidate", "sealed"}, "thresholds"
    )
    expected_thresholds = {
        name: source_contract.thresholds[name].as_dict()
        for name in ("fit", "candidate", "sealed")
    }
    expected_thresholds["fit"].update(
        {"minimum_action_count": 1_152, "minimum_rollout_count": 144}
    )
    if stage._canonical_json(dict(threshold_rows)) != stage._canonical_json(
        expected_thresholds
    ):
        raise GuidedContinuationError(
            "Guided thresholds must preserve the strict source coherence gates "
            "with the exact 144-prompt fit work counts."
        )
    thresholds = {
        name: stage.BehaviorThresholds.from_mapping(
            threshold_rows[name], f"thresholds.{name}"
        )
        for name in ("fit", "candidate", "sealed")
    }
    training = _authenticate_training(
        prereg.get("training"), source_training=source_contract.training
    )
    method_gate = _strict_keys(
        prereg.get("method_gate"), set(GUIDED_METHOD_GATE), "method_gate"
    )
    if dict(method_gate) != GUIDED_METHOD_GATE:
        raise GuidedContinuationError("Guided method-positive gate changed.")
    source_authentication = dict(prereg["source"])
    evaluation_preimage = _guided_evaluation_preimage(
        artifacts=artifacts,
        splits=splits,
        thresholds=thresholds,
        training=training,
        implementation=implementation,
        runtime=runtime,
        method_gate=method_gate,
    )
    evaluation_sha = stage._canonical_sha256(evaluation_preimage)
    if prereg.get("evaluation_contract_sha256") != evaluation_sha:
        raise GuidedContinuationError("Guided evaluation contract SHA-256 changed.")
    execution_contract = dataclasses.replace(
        source_contract,
        path=prereg_path,
        file_sha256=expected,
        splits=splits,
        thresholds=thresholds,
        training=training,
        evaluation_contract_sha256=evaluation_sha,
        implementation_authentication=implementation,
        output_jacobian_audit=None,
        # The historical contract authenticated persisted evidence only.  The
        # current guided source pins and its complete evaluation preimage have
        # both authenticated above, so the new execution contract is live.
        historical_implementation_revision=None,
    )
    return AuthenticatedGuidedContinuation(
        path=prereg_path,
        file_sha256=expected,
        source_contract=source_contract,
        execution_contract=execution_contract,
        artifacts=artifacts,
        evaluation_contract_sha256=evaluation_sha,
        implementation_authentication=implementation,
        source_authentication=source_authentication,
        training=training,
        method_gate=dict(method_gate),
    )


class ProductionGuidedRuntime(stage.ProductionStageRuntime):
    """Heavy boundary that installs and trains the separate guided tranche."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._guided_tranche: Any | None = None

    def restore_source_v2(
        self,
        transaction: Any,
        contract: AuthenticatedGuidedContinuation,
    ) -> Mapping[str, Any]:
        continuation = contract.execution_contract.continuation
        if (
            continuation is None
            or continuation.trainer_factor_pack_schema
            != stage.TRAINER_FACTOR_PACK_V2_SCHEMA
            or continuation.lexical_rank != stage.LEXICAL_RANK
        ):
            raise GuidedContinuationError(
                "Guided continuation requires a full v2 source."
            )
        trainer = importlib.import_module(GUIDED_TRAINER_MODULE)
        restore = getattr(trainer, "restore_target_native_factor_payload", None)
        if not callable(restore):
            raise GuidedContinuationError("Public v2 factor restore is unavailable.")
        state = restore(transaction, continuation.factor_payload)
        tensors = getattr(state, "tensors", None)
        if (
            getattr(state, "schema", None) != stage.TRAINER_FACTOR_PACK_V2_SCHEMA
            or getattr(state, "factor_state_sha256", None)
            != continuation.factor_state_sha256
            or not isinstance(tensors, tuple)
            or len(tensors) != stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT
        ):
            raise GuidedContinuationError(
                "Public restore did not recover exact v2 bytes."
            )
        clone_bytes = sum(
            int(value.numel() * value.element_size()) for value in tensors
        )
        references = tuple(weakref.ref(value) for value in tensors)
        del tensors, state
        gc.collect()
        if any(reference() is not None for reference in references):
            raise GuidedContinuationError(
                "Temporary v2 restore clone remained live before guided installation."
            )
        return {
            "schema": stage.TRAINER_FACTOR_PACK_V2_SCHEMA,
            "factor_state_sha256": continuation.factor_state_sha256,
            "factor_count": stage.LEXICAL_TARGET_NATIVE_FACTOR_COUNT,
            "temporary_restore_clone_cpu_bytes": clone_bytes,
            "temporary_restore_clone_released_before_guided_install": True,
            "full_tensor_or_model_clone_retained": False,
        }

    def install_guided_transaction(
        self,
        transaction: Any,
        contract: AuthenticatedGuidedContinuation,
    ) -> Mapping[str, Any]:
        inherited = getattr(transaction, "qkvo_handle", None)
        if inherited is None:
            raise GuidedContinuationError("Base transaction omits its QKVO handle.")
        adapter = importlib.import_module(GUIDED_ADAPTER_MODULE)
        install = getattr(
            adapter, "install_guided_dense_gqa_qkvo_residual_tranche", None
        )
        if not callable(install):
            raise GuidedContinuationError("Guided QKVO installer is unavailable.")
        composite = install(
            inherited,
            audit_report=contract.artifacts.audit_report,
            basis_pack=contract.artifacts.basis_pack,
            expected_audit_report_sha256=(contract.artifacts.audit_report_self_sha256),
            expected_basis_envelope_file_sha256=(
                contract.artifacts.basis_envelope_file_sha256
            ),
            expected_basis_payload_sha256=(contract.artifacts.basis_payload_sha256),
            expected_basis_manifest_sha256=contract.artifacts.basis_manifest_sha256,
        )
        report_object = getattr(composite, "report", None)
        report = (
            report_object.as_dict()
            if callable(getattr(report_object, "as_dict", None))
            else report_object
        )
        if not isinstance(report, Mapping):
            rollback_guided = getattr(composite, "rollback_guided_only", None)
            if callable(rollback_guided):
                rollback_guided()
            raise GuidedContinuationError("Guided installer returned no report.")
        checks = {
            "guided_adapter_parameter_count": GUIDED_FACTOR_PARAMETERS,
            "audit_report_sha256": contract.artifacts.audit_report_self_sha256,
            "basis_envelope_file_sha256": (
                contract.artifacts.basis_envelope_file_sha256
            ),
            "basis_payload_sha256": contract.artifacts.basis_payload_sha256,
            "basis_manifest_sha256": contract.artifacts.basis_manifest_sha256,
            "factor_concatenation_used": False,
            "factor_regrouping_used": False,
            "zero_guided_left_preserves_inherited_function_exactly": True,
        }
        if (
            any(report.get(name) != value for name, value in checks.items())
            or report.get("source_basis_tensor_sha256s")
            != dict(contract.artifacts.basis_tensor_sha256s)
            or report.get("model_parameter_count_installed")
            - report.get("model_parameter_count_before_guided_installation")
            != GUIDED_FACTOR_PARAMETERS
        ):
            rollback_guided = getattr(composite, "rollback_guided_only", None)
            if callable(rollback_guided):
                rollback_guided()
            raise GuidedContinuationError(
                "Guided installation did not prove exact zero-left 7,127,040 "
                "parameter installation."
            )
        if composite.inherited_handle is not transaction.qkvo_handle:
            composite.rollback_guided_only()
            raise GuidedContinuationError(
                "Guided tranche is bound to another transaction."
            )
        # The trainer authenticates that the transaction still owns the inherited
        # handle.  Handoff to the composite occurs only after all pre-bake gates.
        self._guided_tranche = composite
        return dict(report)

    def rollback_guided_installation(self, transaction: Any) -> None:
        composite = self._guided_tranche
        if composite is None:
            return
        if transaction.qkvo_handle is composite:
            # Once handed off, the containing transaction must own rollback.
            return
        if transaction.qkvo_handle is not composite.inherited_handle:
            raise GuidedContinuationError("Guided rollback ownership changed.")
        composite.rollback_guided_only()
        self._guided_tranche = None

    def handoff_guided_for_publication(
        self, transaction: Any, contract: AuthenticatedGuidedContinuation
    ) -> Mapping[str, Any]:
        del contract
        composite = self._guided_tranche
        if (
            composite is None
            or transaction.qkvo_handle is not composite.inherited_handle
        ):
            raise GuidedContinuationError("Guided publication handoff is not live.")
        composite.assert_live()
        transaction.qkvo_handle = composite
        if transaction.qkvo_handle is not composite:
            raise GuidedContinuationError("Guided publication handoff was not atomic.")
        return {
            "handoff_complete": True,
            "transaction_qkvo_handle_is_guided_composite": True,
            "old_only_publication_path_disabled": True,
        }

    def train_guided(
        self,
        transaction: Any,
        parent: nn.Module,
        contract: AuthenticatedGuidedContinuation,
    ) -> Any:
        module = importlib.import_module(GUIDED_TRAINER_MODULE)
        loader = getattr(module, "load_behavior_domain_suite")
        base_contract = contract.execution_contract
        domains = {
            name: loader(
                name,
                base_contract.splits[name].payload,
                artifact_sha256=base_contract.splits[name].sha256,
            )
            for name in ("fit", "candidate", "sealed")
        }
        schedule_type = getattr(module, "BehaviorDistillationSchedule")
        schedule = schedule_type(**dict(base_contract.training))
        threshold_type = getattr(module, "CandidateBehaviorThresholds")
        candidate_thresholds = threshold_type(
            **base_contract.thresholds["candidate"].as_dict()
        )
        trainer = getattr(module, "train_target_native_behavior_distillation")
        result = trainer(
            transaction=transaction,
            parent_model=parent,
            domains=domains,
            schedule=schedule,
            candidate_thresholds=candidate_thresholds,
            seed=base_contract.seed,
            initial_factor_payload=base_contract.continuation.factor_payload,
            guided_tranche=self._guided_tranche,
        )
        result_type = getattr(module, "BehaviorDistillationResult")
        if type(result) is not result_type:
            raise GuidedContinuationError(
                "Guided trainer returned a noncanonical type."
            )
        return result


def _guided_factor_envelope(
    payload: Mapping[str, Any],
    contract: AuthenticatedGuidedContinuation,
    reason: str,
) -> dict[str, Any]:
    forbidden = ("parent_state", "child_state", "activation", "hidden", "attention")
    if any(name in payload for name in forbidden):
        raise GuidedContinuationError("Guided factor pack contains forbidden state.")
    return {
        "schema": FACTOR_ENVELOPE_SCHEMA,
        "reason": reason,
        "preregistration_sha256": contract.file_sha256,
        "evaluation_contract_sha256": contract.evaluation_contract_sha256,
        "source_factor_state_sha256": contract.artifacts.source_factor_state_sha256,
        "audit_report_self_sha256": contract.artifacts.audit_report_self_sha256,
        "audit_report_canonical_sha256": (
            contract.artifacts.audit_report_canonical_sha256
        ),
        "basis_envelope_file_sha256": (contract.artifacts.basis_envelope_file_sha256),
        "basis_domain_artifact_sha256": contract.artifacts.basis_domain_artifact_sha256,
        "basis_payload_sha256": contract.artifacts.basis_payload_sha256,
        "factor_pack": dict(payload),
        "contains_model_checkpoint": False,
        "contains_parent_parameters": False,
        "contains_activations_or_hidden_states": False,
    }


def _validate_v3_evidence(
    result: Any,
    contract: AuthenticatedGuidedContinuation,
    guided_installation: Mapping[str, Any],
) -> tuple[str, dict[str, Any], Mapping[str, Any]]:
    try:
        evidence_sha, evidence = stage._trainer_evidence(result)
    except stage.ExactTargetNativeStageError as error:
        raise GuidedContinuationError("Guided trainer evidence failed.") from error
    manifest = evidence.get("factor_pack_manifest")
    source = contract.execution_contract.continuation
    if source is None:
        raise GuidedContinuationError("Guided run lost its source factor binding.")
    source_rows = source.factor_payload.get("factors")
    if not isinstance(source_rows, list):
        raise GuidedContinuationError("Authenticated v2 source manifest is absent.")
    source_tensor_hashes = {
        str(row.get("name")): _require_sha256(
            row.get("tensor_sha256"), f"source.factor.{row.get('name')}.sha256"
        )
        for row in source_rows
        if isinstance(row, Mapping)
    }
    expected_source_binding = {
        "source_v2_factor_state_sha256": source.factor_state_sha256,
        "source_v2_tensor_sha256s": source_tensor_hashes,
        "audit_report_sha256": contract.artifacts.audit_report_self_sha256,
        "basis_envelope_file_sha256": contract.artifacts.basis_envelope_file_sha256,
        "basis_payload_sha256": contract.artifacts.basis_payload_sha256,
        "basis_manifest_sha256": contract.artifacts.basis_manifest_sha256,
        "source_basis_tensor_sha256s": dict(contract.artifacts.basis_tensor_sha256s),
        "inherited_installation_sha256": guided_installation.get(
            "inherited_installation_sha256"
        ),
        "inherited_factor_state_sha256_at_install": guided_installation.get(
            "inherited_factor_state_sha256_at_install"
        ),
        "adapter_site_indices_sha256": guided_installation.get(
            "adapter_site_indices_sha256"
        ),
        "audit_site_indices_sha256": guided_installation.get(
            "audit_site_indices_sha256"
        ),
    }
    source_binding_sha = stage._canonical_sha256(expected_source_binding)
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != TRAINER_FACTOR_PACK_V3_SCHEMA
        or manifest.get("factor_count") != GUIDED_TOTAL_FACTOR_COUNT
        or manifest.get("lexical_rank") != stage.LEXICAL_RANK
        or manifest.get("guided_source_binding") != expected_source_binding
        or manifest.get("guided_source_binding_sha256") != source_binding_sha
        or manifest.get("inherited_product_regrouped") is not False
        or manifest.get("guided_product_regrouped") is not False
    ):
        raise GuidedContinuationError(
            "Trainer did not emit the exact authenticated 26-tensor v3 pack."
        )
    training_audit = evidence.get("training_audit")
    gradient = (
        training_audit.get("gradient_activation")
        if isinstance(training_audit, Mapping)
        else None
    )
    guided_left = {f"guided_qkvo.{role}_left" for role in GUIDED_ROLES}
    guided_all = {
        f"guided_qkvo.{role}_{side}"
        for role in GUIDED_ROLES
        for side in ("left", "right")
    }
    inherited_qkvo = {
        f"qkvo.{role}_{side}" for role in GUIDED_ROLES for side in ("left", "right")
    }
    lexical_head = {"lexical.lm_head_left", "lexical.lm_head_right"}
    expected_stages = {
        "guided-left-from-step-1": guided_left,
        "guided-from-step-9": guided_all,
        "joint-from-step-25": guided_all | inherited_qkvo,
        "joint-from-step-41": guided_all | inherited_qkvo | lexical_head,
    }
    raw_stages = (
        gradient.get("stage_authorized_factor_names")
        if isinstance(gradient, Mapping)
        else None
    )
    normalized_stages = (
        {
            str(name): set(values)
            for name, values in raw_stages.items()
            if isinstance(values, list)
        }
        if isinstance(raw_stages, Mapping)
        else None
    )
    if (
        evidence.get("continued_from_factor_pack") is not True
        or evidence.get("initial_factor_state_sha256") != source.factor_state_sha256
        or evidence.get("internal_numeric_matching_used") is not False
        or evidence.get("guided_source_binding") != expected_source_binding
        or evidence.get("guided_source_binding_sha256") != source_binding_sha
        or evidence.get("selected_factor_pack_retained_on_acceptance") is not True
        or not isinstance(gradient, Mapping)
        or gradient.get("every_authorized_factor_gradient_activated") is not True
        or normalized_stages != expected_stages
    ):
        raise GuidedContinuationError(
            "Trainer did not prove v2 restoration, staged guidance, and output-only "
            "selection."
        )
    payload_method = getattr(result, "factor_pack_payload", None)
    payload = payload_method() if callable(payload_method) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != TRAINER_FACTOR_PACK_V3_SCHEMA
        or payload.get("factor_count") != GUIDED_TOTAL_FACTOR_COUNT
        or not isinstance(payload.get("factor_tensors"), Mapping)
        or len(payload["factor_tensors"]) != GUIDED_TOTAL_FACTOR_COUNT
        or sum(int(value.numel()) for value in payload["factor_tensors"].values())
        != GUIDED_TOTAL_FACTOR_PARAMETERS
        or payload.get("guided_source_binding") != expected_source_binding
        or payload.get("guided_source_binding_sha256") != source_binding_sha
    ):
        raise GuidedContinuationError("Selected guided factor payload is malformed.")
    return evidence_sha, evidence, payload


def _guided_method_gate(evidence: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    checkpoints = evidence.get("checkpoints")
    selected_step = evidence.get("selected_step")
    candidate = evidence.get("candidate_metrics")
    by_step = (
        {
            row.get("step"): row.get("metrics")
            for row in checkpoints
            if isinstance(row, Mapping)
        }
        if isinstance(checkpoints, list)
        else {}
    )
    step0 = by_step.get(0)
    selected = by_step.get(selected_step)
    if (
        isinstance(selected_step, bool)
        or not isinstance(selected_step, int)
        or not isinstance(step0, Mapping)
        or not isinstance(selected, Mapping)
        or not isinstance(candidate, Mapping)
        or dict(selected) != dict(candidate)
    ):
        raise GuidedContinuationError(
            "Guided method gate cannot bind step zero, selected checkpoint, and "
            "candidate behavior."
        )
    required_numbers = (
        "lcp_at_least_4_count",
        "repeat_fraction_excess",
        "adjacent_repeat_fraction_excess",
    )
    for name in required_numbers:
        if (
            isinstance(candidate.get(name), bool)
            or not isinstance(candidate.get(name), (int, float))
            or isinstance(step0.get(name), bool)
            or not isinstance(step0.get(name), (int, float))
        ):
            raise GuidedContinuationError(
                f"Guided method metric {name!r} is malformed."
            )
    checks = {
        "selected_step_acquired": selected_step
        >= GUIDED_METHOD_GATE["minimum_selected_step"],
        "candidate_has_one_lcp_at_least_4_rollout": int(
            candidate["lcp_at_least_4_count"]
        )
        >= GUIDED_METHOD_GATE["minimum_candidate_lcp_at_least_4_count"],
        "candidate_repeat_fraction_nonregression_vs_step0": float(
            candidate["repeat_fraction_excess"]
        )
        <= float(step0["repeat_fraction_excess"]),
        "candidate_adjacent_repeat_fraction_nonregression_vs_step0": float(
            candidate["adjacent_repeat_fraction_excess"]
        )
        <= float(step0["adjacent_repeat_fraction_excess"]),
    }
    trainer_gate = evidence.get("guided_candidate_gate")
    expected_trainer_gate = {
        "selected_step": selected_step,
        "selected_step_is_promoted": checks["selected_step_acquired"],
        "minimum_lcp_at_least_4_count": GUIDED_METHOD_GATE[
            "minimum_candidate_lcp_at_least_4_count"
        ],
        "observed_lcp_at_least_4_count": int(candidate["lcp_at_least_4_count"]),
        "baseline_repeat_fraction_excess": float(step0["repeat_fraction_excess"]),
        "candidate_repeat_fraction_excess": float(candidate["repeat_fraction_excess"]),
        "repeat_fraction_nonregression": checks[
            "candidate_repeat_fraction_nonregression_vs_step0"
        ],
        "baseline_adjacent_repeat_fraction_excess": float(
            step0["adjacent_repeat_fraction_excess"]
        ),
        "candidate_adjacent_repeat_fraction_excess": float(
            candidate["adjacent_repeat_fraction_excess"]
        ),
        "adjacent_repeat_fraction_nonregression": checks[
            "candidate_adjacent_repeat_fraction_nonregression_vs_step0"
        ],
        "passed": all(checks.values()),
    }
    if trainer_gate != expected_trainer_gate:
        raise GuidedContinuationError(
            "Trainer guided-candidate gate disagrees with persisted behavior metrics."
        )
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(GUIDED_METHOD_GATE),
        "selected_step": selected_step,
        "step0_metrics": {name: step0[name] for name in required_numbers},
        "candidate_metrics": {name: candidate[name] for name in required_numbers},
        "four_isolated_one_token_lcps_cannot_substitute": True,
        "internal_numeric_matching_consulted": False,
        "trainer_gate": expected_trainer_gate,
    }
    decision["decision_sha256"] = stage._canonical_sha256(decision)
    return bool(decision["passed"]), decision


def _post_bake_guided_method_gate(
    metrics: Any, evidence: Mapping[str, Any]
) -> tuple[bool, dict[str, Any]]:
    payload = stage._metrics_dict(metrics)
    checkpoints = evidence.get("checkpoints")
    step0 = (
        next(
            (
                row.get("metrics")
                for row in checkpoints
                if isinstance(row, Mapping) and row.get("step") == 0
            ),
            None,
        )
        if isinstance(checkpoints, list)
        else None
    )
    selected_step = evidence.get("selected_step")
    if not isinstance(step0, Mapping) or not isinstance(selected_step, int):
        raise GuidedContinuationError("Post-bake method gate lacks step-zero binding.")
    checks = {
        "selected_step_acquired": selected_step > 0,
        "candidate_has_one_lcp_at_least_4_rollout": int(payload["lcp_at_least_4_count"])
        >= 1,
        "candidate_repeat_fraction_nonregression_vs_step0": float(
            payload["repeat_fraction_excess"]
        )
        <= float(step0["repeat_fraction_excess"]),
        "candidate_adjacent_repeat_fraction_nonregression_vs_step0": float(
            payload["adjacent_repeat_fraction_excess"]
        )
        <= float(step0["adjacent_repeat_fraction_excess"]),
    }
    decision = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": dict(GUIDED_METHOD_GATE),
        "selected_step": selected_step,
        "step0_repeat_fraction_excess": step0["repeat_fraction_excess"],
        "step0_adjacent_repeat_fraction_excess": step0[
            "adjacent_repeat_fraction_excess"
        ],
        "candidate_lcp_at_least_4_count": payload["lcp_at_least_4_count"],
        "candidate_repeat_fraction_excess": payload["repeat_fraction_excess"],
        "candidate_adjacent_repeat_fraction_excess": payload[
            "adjacent_repeat_fraction_excess"
        ],
        "internal_numeric_matching_consulted": False,
    }
    decision["decision_sha256"] = stage._canonical_sha256(decision)
    return bool(decision["passed"]), decision


def _refresh_guided_execution_contract(
    contract: AuthenticatedGuidedContinuation,
) -> AuthenticatedGuidedContinuation:
    """Reauthenticate every live guided source immediately before execution."""

    if type(contract) is not AuthenticatedGuidedContinuation:
        raise GuidedContinuationError(
            "Production guided execution requires its canonical authenticated contract."
        )
    refreshed = authenticate_guided_preregistration(contract.path, contract.file_sha256)
    if (
        refreshed.path != contract.path
        or refreshed.file_sha256 != contract.file_sha256
        or refreshed.evaluation_contract_sha256 != contract.evaluation_contract_sha256
        or refreshed.implementation_authentication
        != contract.implementation_authentication
    ):
        raise GuidedContinuationError(
            "Guided preregistration, implementation, or evaluation authentication "
            "changed before execution."
        )
    source_revision = refreshed.source_contract.historical_implementation_revision
    if (
        not isinstance(source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_revision) is None
        or refreshed.execution_contract.historical_implementation_revision is not None
    ):
        raise GuidedContinuationError(
            "Guided execution did not preserve replay-only source evidence while "
            "clearing the historical marker on the freshly authenticated live contract."
        )
    return refreshed


def execute_authenticated_guided_continuation(
    contract: AuthenticatedGuidedContinuation,
    output_root: Path | str,
    *,
    runtime: Any | None = None,
) -> dict[str, Any]:
    """Execute one guided transaction after an explicit model-free preflight."""

    runner = ProductionGuidedRuntime() if runtime is None else runtime
    if type(contract) is AuthenticatedGuidedContinuation or isinstance(
        runner, ProductionGuidedRuntime
    ):
        contract = _refresh_guided_execution_contract(contract)
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise GuidedContinuationError("output_root must not already exist.")
    output.mkdir(parents=True)
    base = contract.execution_contract
    parent: Any = None
    target: Any = None
    tokenizer: Any = None
    compiled: Any = None
    transaction: Any = None
    compiled_state_sha: str | None = None
    published = False
    report: dict[str, Any] = {
        **contract.preflight_report(),
        "status": "running",
        "output_root": str(output),
    }
    try:
        prepare_cache = getattr(runner, "prepare_compiler_cache", None)
        if callable(prepare_cache):
            cache_plan = prepare_cache(base)
            if cache_plan is not None:
                report["compiled_child_cache"] = dict(cache_plan)
        parent, target, tokenizer = runner.load_models(base)
        report.update({"models_loaded": True, "gpu_work_launched": True})
        compiled = runner.compile(parent, target, tokenizer, base)
        compiler_report = stage._compiled_report_dict(compiled)
        compiled_state_sha = runner.state_sha256(compiled.child_model)
        stage._validate_continuation_compilation(
            base, compiler_report, compiled_state_sha
        )
        finalize = getattr(runner, "finalize_compiled_base", None)
        if callable(finalize):
            finalized = finalize(compiled, base)
            if (
                stage._compiled_report_dict(finalized) != compiler_report
                or runner.state_sha256(finalized.child_model) != compiled_state_sha
            ):
                raise GuidedContinuationError(
                    "Compiler cache finalization changed the exact compiled base."
                )
            compiled = finalized
        transaction = runner.install_transaction(compiled)
        inherited_install = transaction.installation_report()
        if (
            inherited_install.get("zero_step_identity") is not True
            or inherited_install.get("temporary_lexical_factor_parameters")
            != stage.LEXICAL_FACTOR_PARAMETERS
        ):
            raise GuidedContinuationError(
                "Inherited v2 transaction is not exact zero-step."
            )
        source_v2_restore = runner.restore_source_v2(transaction, contract)
        guided_install = runner.install_guided_transaction(transaction, contract)
        report.update(
            {
                "compiler_report": compiler_report,
                "inherited_installation": inherited_install,
                "source_v2_restore": dict(source_v2_restore),
                "guided_installation": guided_install,
                "last_completed_phase": "guided-zero-left-installation",
            }
        )
        result = runner.train_guided(transaction, parent, contract)
        evidence_sha, evidence, factor_payload = _validate_v3_evidence(
            result, contract, guided_install
        )
        factor_path = output / "selected_guided_factor_pack.pt"
        factor_sha = stage._atomic_torch_save(
            factor_path, _guided_factor_envelope(factor_payload, contract, "selected")
        )
        factor_binding = {"path": str(factor_path), "sha256": factor_sha}
        fit_passed, fit_decision = stage._behavior_gate(
            stage._training_metric(result, "fit_metrics"), base.thresholds["fit"]
        )
        candidate_behavior_passed, candidate_decision = stage._behavior_gate(
            stage._training_metric(result, "candidate_metrics"),
            base.thresholds["candidate"],
        )
        method_passed, method_decision = _guided_method_gate(evidence)
        candidate_passed = candidate_behavior_passed and method_passed
        trainer_candidate_passed = getattr(result, "candidate_gate_passed", None)
        if trainer_candidate_passed is not candidate_passed:
            raise GuidedContinuationError(
                "Trainer and runner disagree on the combined candidate/method gate."
            )
        sealed_metrics = stage._training_metric(result, "sealed_metrics")
        sealed_unlocked = bool(
            getattr(result, "sealed_evaluation_unlocked", sealed_metrics is not None)
        )
        if candidate_passed != sealed_unlocked or (
            not candidate_passed and sealed_metrics is not None
        ):
            raise GuidedContinuationError(
                "Sealed behavior was not locked exactly to the candidate gate."
            )
        sealed_passed = False
        sealed_decision = None
        if candidate_passed:
            sealed_passed, sealed_decision = stage._behavior_gate(
                sealed_metrics, base.thresholds["sealed"]
            )
        decisions = {
            "fit": fit_decision,
            "candidate": candidate_decision,
            "sealed": sealed_decision,
        }
        accepted = fit_passed and candidate_passed and sealed_passed
        report.update(
            {
                "trainer_evidence_sha256": evidence_sha,
                "trainer_evidence": evidence,
                "factor_acquisition": stage._factor_acquisition_claim(evidence),
                "selected_factor_pack": factor_binding,
                "pre_bake_behavior_decisions": decisions,
                "guided_method_decision": method_decision,
                "selection_used_internal_numeric_matching": False,
            }
        )
        if not accepted:
            reason = (
                "fit-gate-rejected"
                if not fit_passed
                else (
                    "candidate-gate-rejected"
                    if not candidate_behavior_passed
                    else (
                        "guided-method-gate-rejected"
                        if not method_passed
                        else "sealed-gate-rejected"
                    )
                )
            )
            runner.rollback_guided_installation(transaction)
            transaction.rollback()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_state_sha:
                raise GuidedContinuationError(
                    "Rejected guided transaction did not restore exact child bytes."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": reason,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                    "retained_artifacts": [
                        "selected_guided_factor_pack.pt",
                        "run_report.json",
                    ],
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report

        sealed_payload = stage._metrics_dict(sealed_metrics)
        receipt = ExternalBehaviorGateReceipt(
            evidence_sha256=evidence_sha,
            evaluation_contract_sha256=contract.evaluation_contract_sha256,
            heldout_action_count=int(sealed_payload["action_count"]),
            free_running_rollout_count=int(sealed_payload["rollout_count"]),
            passed=True,
            internal_similarity_only=False,
        )
        # qkvo_handle is now the guided-aware composite.  The containing
        # transaction preserves lexical -> (old QKVO then guided QKVO) -> GUD
        # publication and reverse-order atomic restoration on any later gate.
        handoff = runner.handoff_guided_for_publication(transaction, contract)
        report["guided_publication_handoff"] = dict(handoff)
        publication = transaction.bake_for_publication(receipt)
        report.update(
            {
                "publication": publication,
                "last_completed_phase": "guided-native-sequential-bake",
            }
        )
        post: dict[str, dict[str, Any] | None] = {
            "fit": None,
            "candidate": None,
            "sealed": None,
        }
        rejection: str | None = None
        post_bake_method_decision: dict[str, Any] | None = None
        for name in ("fit", "candidate", "sealed"):
            if name == "sealed" and (
                post["candidate"] is None or post["candidate"]["passed"] is not True
            ):
                break
            metrics = runner.evaluate_baked_behavior(
                compiled.child_model, parent, base, name
            )
            passed, decision = stage._behavior_gate(metrics, base.thresholds[name])
            post[name] = decision
            if not passed:
                rejection = f"post-bake-{name}-gate-rejected"
                break
            if name == "candidate":
                post_method_passed, post_bake_method_decision = (
                    _post_bake_guided_method_gate(metrics, evidence)
                )
                if not post_method_passed:
                    rejection = "post-bake-guided-method-gate-rejected"
                    break
        report["post_bake_behavior_decisions"] = post
        report["post_bake_guided_method_decision"] = post_bake_method_decision
        if rejection is not None:
            transaction.restore_prepublication_native()
            rollback_sha = runner.state_sha256(compiled.child_model)
            if rollback_sha != compiled_state_sha:
                raise GuidedContinuationError(
                    "Post-bake guided rejection did not restore exact child bytes."
                )
            report.update(
                {
                    "status": "rejected",
                    "reason": rejection,
                    "rollback_exact": True,
                    "rollback_state_sha256": rollback_sha,
                    "prepublication_native_restored": True,
                    "standalone_compaction_passed": False,
                    "model_checkpoint_written": False,
                    "retained_artifacts": [
                        "selected_guided_factor_pack.pt",
                        "run_report.json",
                    ],
                }
            )
            stage._atomic_json(output / "run_report.json", report)
            return report
        certificate = runner.save_reload_certify(
            compiled.child_model, tokenizer, output
        )
        published = True
        if (
            post_bake_method_decision is None
            or post_bake_method_decision.get("passed") is not True
            or certificate.get("state_bytes_match_pre_save") is not True
            or certificate.get("fresh_reload_performed") is not True
            or certificate.get("published_adapter_parameter_count") != 0
            or certificate.get("published_parent_runtime_reference_count") != 0
            or certificate.get("published_source_sized_sidecar_parameter_count") != 0
        ):
            raise GuidedContinuationError(
                "Fresh-reloaded guided child retained an inference sidecar."
            )
        report.update(
            {
                "status": "accepted",
                "standalone_compaction_passed": True,
                "fresh_reload_certificate": certificate,
                "fresh_reload_guided_method_gate": {
                    "passed": True,
                    "basis": "exact-reloaded-state-bytes-match-post-bake-candidate",
                    "post_bake_decision_sha256": post_bake_method_decision[
                        "decision_sha256"
                    ],
                    "state_bytes_match_pre_save": certificate.get(
                        "state_bytes_match_pre_save"
                    ),
                    "internal_numeric_matching_consulted": False,
                },
                "rollback_exact": None,
                "model_checkpoint_written": True,
                "model_save_count": 1,
                "retained_artifacts": [
                    "selected_guided_factor_pack.pt",
                    "accepted_child",
                    "run_report.json",
                ],
            }
        )
        stage._atomic_json(output / "run_report.json", report)
        return report
    except BaseException as error:
        rollback_error: str | None = None
        restored = False
        if transaction is not None:
            restore = getattr(transaction, "restore_prepublication_native", None)
            rollback = getattr(transaction, "rollback", None)
            try:
                rollback_guided = getattr(runner, "rollback_guided_installation", None)
                if callable(rollback_guided):
                    rollback_guided(transaction)
                if callable(restore):
                    restore()
                    restored = True
                elif callable(rollback) and getattr(transaction, "is_live", False):
                    rollback()
                    restored = True
                if (
                    restored
                    and compiled is not None
                    and compiled_state_sha is not None
                    and runner.state_sha256(compiled.child_model) != compiled_state_sha
                ):
                    raise GuidedContinuationError(
                        "Failure rollback changed child bytes."
                    )
            except BaseException as rollback_failure:  # pragma: no cover
                rollback_error = repr(rollback_failure)
        if published or (output / "accepted_child").exists():
            try:
                stage._remove_published_child_tree(output / "accepted_child", output)
            except BaseException as cleanup_failure:  # pragma: no cover
                rollback_error = f"{rollback_error or ''}; {cleanup_failure!r}"
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "rollback_error": rollback_error,
            "prepublication_native_restored": restored,
            "model_checkpoint_promoted": False,
        }
        try:
            if not (output / "run_report.json").exists():
                stage._atomic_json(output / "run_report.json", failure)
        except BaseException:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", "--dry-run", dest="preflight", action="store_true")
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize model loading and guided execution.",
    )
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--parent-device-map", default="auto")
    parser.add_argument("--target-device-map", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_guided_preregistration(
        args.preregistration, args.preregistration_sha256
    )
    if args.preflight:
        if args.output_root is not None:
            raise SystemExit("--output-root is only valid with --execute.")
        print(json.dumps(contract.preflight_report(), indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required with --execute.")
    runtime = ProductionGuidedRuntime(
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    result = execute_authenticated_guided_continuation(
        contract, args.output_root, runtime=runtime
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
