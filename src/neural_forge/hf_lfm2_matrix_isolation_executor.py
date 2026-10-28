"""Production canonical-v2 executor boundary for LFM2 isolation cells.

The technique kernels and the model/evidence backend remain independent.  This
module is the authority that joins them: it authenticates both implementations,
builds phase-specific cache keys, enforces complete owned-weight installation,
requires component-specific actual-forward evidence, and returns the exact
payload accepted by :mod:`neural_forge.hf_lfm2_matrix_cell`.

It deliberately supports ``isolation_only`` first.  A parent-shaped surrogate
can localise useful folding math, but it cannot claim physical compaction.  A
separate blank-child promotion executor is required for that evidence class.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from .dynamic_circuit_ir import canonical_sha256
from .dynamic_fold_executor import FoldOnlyGuardReport, run_fold_only_callback
from .hf_lfm2_component_method_matrix import load_matrix_kernel_provider
from .hf_lfm2_matrix_cell import (
    Lfm2MatrixCellError,
    MatrixCellRequest,
    MatrixTechniqueExecutor,
    build_matrix_phase_key,
)

ISOLATION_EXECUTOR_SCHEMA = "neural-forge-lfm2-matrix-isolation-executor-v1"
_PHASES = (
    "evidence_preparation",
    "target_materialization",
    "technique_solve",
    "component_install",
    "behavior_evaluation",
)
_STAGES = (
    "evidence_opened",
    "checkpoint_reuse_resolved",
    "copy_exact_control_materialized",
    "calibration_completed",
    "component_solve_completed",
    "component_installed",
    "boundary_evaluation_completed",
    "composed_evaluation_completed",
    "free_run_completed",
    "decode_smoke_completed",
)
_METRIC_GROUPS = {
    "weight",
    "component_write",
    "macrocell",
    "terminal",
    "attention_relation",
    "mlp_relation",
    "state_mixer_relation",
    "interface_relation",
    "cache_trajectory",
}
_RELATION_GROUP = {
    "C01": "interface_relation",
    "C02": "interface_relation",
    "C03": "state_mixer_relation",
    "C04": "state_mixer_relation",
    "C05": "state_mixer_relation",
    "C06": "attention_relation",
    "C07": "attention_relation",
    "C08": "attention_relation",
    "C09": "attention_relation",
    "C10": "attention_relation",
    "C11": "mlp_relation",
    "C12": "mlp_relation",
    "C13": "mlp_relation",
    "C14": "weight",
    "C15": "interface_relation",
}
_AUDIT_ALIASES = {
    "pruned_parameters": "pruning_operations",
    "quantized_parameters": "quantization_operations",
    "network_access": "network_requests",
}
_AUDIT_FIELDS = (
    "autograd_grad_calls",
    "backward_calls",
    "optimizer_constructions",
    "optimizer_steps",
    "training_steps",
    "pruning_operations",
    "quantization_operations",
    "parent_trace_recaptures",
    "network_requests",
)


class Lfm2MatrixIsolationExecutorError(Lfm2MatrixCellError):
    """The production executor or its backend failed a scientific contract."""


class MatrixIsolationBackendSession(Protocol):
    """One backend-owned parent-shell session.

    Implementations may own GPU models and cache handles.  Every method is
    called at most once per attempt; reusable work is addressed by the supplied
    phase keys and reported through ``checkpoint_reuse``.
    """

    technique_request: Any
    code_identity: Mapping[str, Any]
    evidence: Mapping[str, Any]
    numeric_execution: Mapping[str, Any]
    resources: Mapping[str, Any]
    operation_audit: Mapping[str, Any]
    checkpoint_reuse: Mapping[str, Any]

    def materialize_copy_exact_control(
        self, phase_key_sha256: str
    ) -> Mapping[str, Any]: ...

    def calibrate(self, phase_key_sha256: str) -> Mapping[str, Any]: ...

    def install(self, result: Any, phase_key_sha256: str) -> Mapping[str, Any]: ...

    def evaluate_boundary(self) -> Mapping[str, Any]: ...

    def evaluate_composed(self) -> Mapping[str, Any]: ...

    def evaluate_free_run(self, phase_key_sha256: str) -> Mapping[str, Any]: ...

    def evaluate_decode_smoke(
        self, phase_key_sha256: str
    ) -> list[Mapping[str, Any]]: ...

    def close(self) -> None: ...


class MatrixIsolationBackendFactory(Protocol):
    def __call__(
        self,
        request: MatrixCellRequest,
        run_directory: Path,
        phase_key_sha256s: Mapping[str, str],
        emit: Callable[..., Mapping[str, Any]],
    ) -> MatrixIsolationBackendSession: ...


def _plain(value: Any, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Lfm2MatrixIsolationExecutorError(f"{label} is not finite.")
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain(item, f"{label}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item, f"{label}[]") for item in value]
    raise Lfm2MatrixIsolationExecutorError(
        f"{label} contains non-JSON {type(value).__name__}."
    )


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Lfm2MatrixIsolationExecutorError(f"{label} must be an object.")
    return _plain(value, label)


def _reference(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.count(":") != 1:
        raise Lfm2MatrixIsolationExecutorError(f"{label} must be module:attribute.")
    module, attribute = value.split(":", 1)
    if not module or not attribute:
        raise Lfm2MatrixIsolationExecutorError(f"{label} must be module:attribute.")
    return value


def _source_sha256(value: Any) -> str:
    source_path = inspect.getsourcefile(value)
    if source_path is None:
        raise Lfm2MatrixIsolationExecutorError(
            "Isolation backend source is not inspectable."
        )
    return hashlib.sha256(Path(source_path).read_bytes()).hexdigest()


def _load_backend(reference: str) -> tuple[MatrixIsolationBackendFactory, str]:
    module_name, attribute = _reference(reference, "isolation_backend").split(":", 1)
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise Lfm2MatrixIsolationExecutorError("Isolation backend is not callable.")
    return value, _source_sha256(value)


def _executor(technique_id: str) -> MatrixTechniqueExecutor:
    if technique_id not in {"T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"}:
        raise Lfm2MatrixIsolationExecutorError("Unknown technique executor id.")
    implementation_sha256 = hashlib.sha256(
        Path(__file__).read_bytes() + b"\0" + technique_id.encode("ascii")
    ).hexdigest()
    return MatrixTechniqueExecutor(
        technique_id=technique_id,
        implementation_id=f"lfm2-isolation-v2-{technique_id.lower()}",
        implementation_sha256=implementation_sha256,
        execute=_execute_isolation,
    )


def _phase_projections(request: MatrixCellRequest) -> dict[str, dict[str, Any]]:
    policy = dict(request.execution_policy)
    evidence_policy = _mapping(policy.get("evidence_policy", {}), "evidence_policy")
    target_policy = _mapping(policy.get("target_policy", {}), "target_policy")
    behavior_policy = _mapping(policy.get("behavior_policy", {}), "behavior_policy")
    return {
        "evidence_preparation": {
            "parent_parameter_count": request.parent_parameter_count,
            "site_id": request.site_id,
            "evidence_binding": dict(request.evidence_binding),
            "evidence_policy": evidence_policy,
        },
        "target_materialization": {
            "parent_parameter_count": request.parent_parameter_count,
            "target_parameter_count": request.target_parameter_count,
            "evidence_class": request.evidence_class,
            "target_policy": target_policy,
        },
        "technique_solve": {
            "component_ids": list(request.component_ids),
            "technique_id": request.technique_id,
            "level_id": request.level_id,
            "site_id": request.site_id,
            "technique_configuration": dict(request.technique_configuration),
        },
        "component_install": {
            "component_ids": list(request.component_ids),
            "technique_id": request.technique_id,
            "level_id": request.level_id,
            "site_id": request.site_id,
            "evidence_class": request.evidence_class,
        },
        "behavior_evaluation": {
            "component_ids": list(request.component_ids),
            "technique_id": request.technique_id,
            "level_id": request.level_id,
            "site_id": request.site_id,
            "behavior_policy": behavior_policy,
        },
    }


def _phase_keys(
    request: MatrixCellRequest,
    executor: MatrixTechniqueExecutor,
    *,
    provider_sha256: str,
    backend_sha256: str,
) -> dict[str, str]:
    numeric = _mapping(
        request.execution_policy.get("numeric_contract"), "numeric_contract"
    )
    if not numeric:
        raise Lfm2MatrixIsolationExecutorError("numeric_contract cannot be empty.")
    projections = _phase_projections(request)
    backend_dependencies = _mapping(
        request.execution_policy.get("backend_dependency_sha256s", {}),
        "backend_dependency_sha256s",
    )
    for name, digest in backend_dependencies.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Lfm2MatrixIsolationExecutorError(
                f"backend_dependency_sha256s.{name} is not a SHA-256 digest."
            )
    result: dict[str, str] = {}
    for phase in _PHASES:
        dependencies = {
            "backend": backend_sha256,
            "provider": provider_sha256,
            **{
                f"backend_{name}": digest
                for name, digest in sorted(backend_dependencies.items())
            },
        }
        if phase == "technique_solve":
            dependencies["evidence_phase"] = result["evidence_preparation"]
        elif phase == "component_install":
            dependencies["target_phase"] = result["target_materialization"]
            dependencies["solve_phase"] = result["technique_solve"]
        elif phase == "behavior_evaluation":
            dependencies["install_phase"] = result["component_install"]
        key = build_matrix_phase_key(
            request,
            executor,
            phase_name=phase,
            numeric_contract=numeric,
            dependency_sha256s=dependencies,
            abi={
                "isolation_executor_schema": ISOLATION_EXECUTOR_SCHEMA,
                "isolation_backend_sha256": backend_sha256,
                "kernel_provider_sha256": provider_sha256,
            },
            request_projection=projections[phase],
        )
        result[phase] = key.phase_key_sha256
    return result


def _timed(
    stage: str,
    call: Callable[[], Any],
    timings: dict[str, dict[str, Any]],
    *,
    cache_status: str,
) -> Any:
    started = time.perf_counter()
    result = call()
    timings[stage] = {
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": 0,
        "peak_vram_bytes": 0,
        "cache_status": cache_status,
    }
    return result


def _cache_statuses(
    session: MatrixIsolationBackendSession,
    phase_keys: Mapping[str, str],
) -> dict[str, Any]:
    rows = _mapping(session.checkpoint_reuse, "checkpoint_reuse")
    if set(rows) != set(_PHASES):
        raise Lfm2MatrixIsolationExecutorError(
            "Backend checkpoint_reuse must resolve all five phases."
        )
    for phase, expected_key in phase_keys.items():
        row = _mapping(rows[phase], f"checkpoint_reuse.{phase}")
        if row.get("status") not in {"hit", "miss", "not_applicable"}:
            raise Lfm2MatrixIsolationExecutorError(
                f"checkpoint_reuse.{phase}.status is invalid."
            )
        if (
            row.get("status") != "not_applicable"
            and row.get("cache_key_sha256") != expected_key
        ):
            raise Lfm2MatrixIsolationExecutorError(
                f"checkpoint_reuse.{phase} is bound to the wrong phase key."
            )
    return rows


def _operation_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(value, "operation_audit")
    for old, new in _AUDIT_ALIASES.items():
        if old in raw:
            if new in raw and raw[new] != raw[old]:
                raise Lfm2MatrixIsolationExecutorError(
                    f"Operation audit aliases disagree for {old}/{new}."
                )
            raw[new] = raw.pop(old)
    missing = sorted(set(_AUDIT_FIELDS) - set(raw))
    if missing:
        raise Lfm2MatrixIsolationExecutorError(
            "Operation audit is missing " + ", ".join(missing) + "."
        )
    for field in _AUDIT_FIELDS:
        if raw[field] not in {0, False}:
            raise Lfm2MatrixIsolationExecutorError(
                f"Fold-only operation audit failed: {field}={raw[field]!r}."
            )
    return raw


def _observed_operation_audit(
    session_audit: Mapping[str, Any],
    technique_solve: Mapping[str, Any],
    guard_reports: list[FoldOnlyGuardReport],
) -> dict[str, Any]:
    merged = _mapping(session_audit, "operation_audit")
    raw_kernel = technique_solve.get("operation_audit", {})
    kernel = _mapping(raw_kernel, "technique solve operation_audit")
    for old, new in _AUDIT_ALIASES.items():
        if old in kernel:
            if new in kernel and kernel[new] != kernel[old]:
                raise Lfm2MatrixIsolationExecutorError(
                    f"Technique operation-audit aliases disagree for {old}/{new}."
                )
            kernel[new] = kernel.pop(old)
    declared_violations = {
        name: kernel[name]
        for name in _AUDIT_FIELDS
        if name in kernel and kernel[name] not in {0, False}
    }
    if declared_violations:
        raise Lfm2MatrixIsolationExecutorError(
            "Technique declared forbidden fold operations: "
            + ", ".join(
                f"{name}={value!r}"
                for name, value in sorted(declared_violations.items())
            )
        )
    observed_fields = (
        "autograd_grad_calls",
        "backward_calls",
        "optimizer_constructions",
        "optimizer_steps",
        "pruning_operations",
        "quantization_operations",
    )
    for name in observed_fields:
        merged[name] = int(merged.get(name, 0)) + sum(
            int(getattr(report, name)) for report in guard_reports
        )
    merged["technique_declared_operation_audit"] = kernel
    merged["runtime_fold_only_guard_reports"] = [
        report.as_dict() for report in guard_reports
    ]
    return _operation_audit(merged)


def _install_proof(
    session: MatrixIsolationBackendSession,
    technique_result: Any,
    phase_key_sha256: str,
) -> dict[str, Any]:
    expected = tuple(session.technique_request.component.weight_keys)
    delivered = tuple(sorted(technique_result.installed_weights))
    if delivered != tuple(sorted(expected)):
        raise Lfm2MatrixIsolationExecutorError(
            "Technique must deliver every and only owned weight key; "
            f"expected {sorted(expected)}, got {list(delivered)}."
        )
    proof = _mapping(
        session.install(technique_result, phase_key_sha256), "installation proof"
    )
    required = {
        "expected_owned_weight_keys": list(expected),
        "delivered_weight_keys": list(delivered),
        "installed_weight_keys": list(delivered),
        "unchanged_nonowned_state_verified": True,
        "alias_closure_verified": True,
    }
    bad = {
        key: proof.get(key)
        for key, expected_value in required.items()
        if proof.get(key) != expected_value
    }
    if bad:
        raise Lfm2MatrixIsolationExecutorError(
            "Installation ownership proof failed: "
            + ", ".join(f"{key}={value!r}" for key, value in bad.items())
        )
    installed = proof.get("installed_tensor_sha256s")
    if not isinstance(installed, Mapping) or set(installed) != set(delivered):
        raise Lfm2MatrixIsolationExecutorError(
            "Installation proof lacks every installed tensor digest."
        )
    return proof


def _actual_metric_group(groups: Mapping[str, Any], name: str) -> None:
    row = groups.get(name)
    if not isinstance(row, Mapping):
        raise Lfm2MatrixIsolationExecutorError(
            f"Required component metric group {name!r} is unavailable."
        )
    if row.get("measurement_status") != "measured":
        raise Lfm2MatrixIsolationExecutorError(
            f"Component metric group {name!r} was not measured."
        )
    if row.get("measurement_source") not in {
        "actual_model_forward",
        "actual_cached_circuit_replay",
        "sealed_heldout_operator",
    }:
        raise Lfm2MatrixIsolationExecutorError(
            f"Component metric group {name!r} has no authoritative source."
        )


def _component_diagnostics(
    request: MatrixCellRequest,
    provider: Any,
    boundary: Mapping[str, Any],
    composed: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = boundary.get("component_diagnostics")
    if candidate is None:
        candidate = composed.get("component_diagnostics")
    row = _mapping(candidate, "component_diagnostics")
    groups = _mapping(row.get("metric_groups"), "component_diagnostics.metric_groups")
    if set(groups) != _METRIC_GROUPS:
        raise Lfm2MatrixIsolationExecutorError(
            "Component evaluator must resolve every metric-group slot."
        )
    _actual_metric_group(groups, "component_write")
    _actual_metric_group(groups, "macrocell")
    _actual_metric_group(groups, "terminal")
    relation = _RELATION_GROUP[request.component_ids[0]]
    _actual_metric_group(groups, relation)
    if relation == "attention_relation":
        _actual_metric_group(groups, "cache_trajectory")
    physical = list(provider.physical_component_ids(request.component_ids))
    expected = {
        "measurement_component_ids": list(request.component_ids),
        "physical_component_ids": physical,
        "metric_groups": groups,
        "numeric_decision_authority": False,
    }
    if row != expected:
        raise Lfm2MatrixIsolationExecutorError(
            "Component diagnostics identities or authority are inconsistent."
        )
    return expected


def _execute_isolation(
    request: MatrixCellRequest,
    run_directory: Path,
    emit: Callable[..., Mapping[str, Any]],
) -> Mapping[str, Any]:
    if request.evidence_class != "isolation_only":
        raise Lfm2MatrixIsolationExecutorError(
            "The isolation executor cannot publish complete-child evidence."
        )
    if len(request.component_ids) != 1:
        raise Lfm2MatrixIsolationExecutorError(
            "Isolation matrix cells require one measurement component."
        )
    provider = load_matrix_kernel_provider(request.technique_id)
    component_id = request.component_ids[0]
    if component_id not in provider.components:
        raise Lfm2MatrixIsolationExecutorError(
            f"{request.technique_id} has no kernel for {component_id}."
        )
    backend_reference = _reference(
        request.execution_policy.get("isolation_backend"), "isolation_backend"
    )
    backend, backend_sha256 = _load_backend(backend_reference)
    executor = _executor(request.technique_id)
    phase_keys = _phase_keys(
        request,
        executor,
        provider_sha256=provider.fingerprint,
        backend_sha256=backend_sha256,
    )
    timings: dict[str, dict[str, Any]] = {}
    guard_reports: list[FoldOnlyGuardReport] = []

    def guarded(
        stage: str,
        callback: Callable[[], Any],
    ) -> Any:
        value, report = run_fold_only_callback(
            callback,
            phase=f"lfm2_matrix_{request.cell_id}_{stage}",
        )
        guard_reports.append(report)
        return value

    session: MatrixIsolationBackendSession | None = None
    try:
        session = _timed(
            "evidence_opened",
            lambda: backend(request, run_directory, phase_keys, emit),
            timings,
            cache_status="mixed",
        )
        emit("evidence_opened", phase_key_sha256=phase_keys["evidence_preparation"])
        checkpoints = _cache_statuses(session, phase_keys)
        timings["checkpoint_reuse_resolved"] = {
            "wall_seconds": 0.0,
            "peak_rss_bytes": 0,
            "peak_vram_bytes": 0,
            "cache_status": "mixed",
        }
        emit("checkpoint_reuse_resolved", checkpoint_reuse=checkpoints)

        control = _timed(
            "copy_exact_control_materialized",
            lambda: guarded(
                "copy_exact_control_materialized",
                lambda: session.materialize_copy_exact_control(
                    phase_keys["target_materialization"]
                ),
            ),
            timings,
            cache_status=checkpoints["target_materialization"]["status"],
        )
        control = _mapping(control, "copy_exact control")
        if (
            control.get("materialized") is not True
            or control.get("authenticated") is not True
        ):
            raise Lfm2MatrixIsolationExecutorError(
                "Copy-exact control was not materialized and authenticated."
            )
        emit(
            "copy_exact_control_materialized", control_sha256=canonical_sha256(control)
        )

        calibration = _timed(
            "calibration_completed",
            lambda: guarded(
                "calibration_completed",
                lambda: session.calibrate(phase_keys["evidence_preparation"]),
            ),
            timings,
            cache_status=checkpoints["evidence_preparation"]["status"],
        )
        calibration = _mapping(calibration, "calibration")
        if calibration.get("passed") is not True:
            raise Lfm2MatrixIsolationExecutorError(
                "Copy-exact calibration did not reproduce the measured operator."
            )
        emit("calibration_completed", calibration_sha256=canonical_sha256(calibration))

        technique_request = session.technique_request
        if (
            technique_request.cell.text != request.cell_id
            or technique_request.component.component_id != component_id
            or technique_request.level != request.level_id
        ):
            raise Lfm2MatrixIsolationExecutorError(
                "Backend technique request differs from the canonical request."
            )
        from .matrix_technique_registry import resolve, validate_result

        technique = resolve(request.technique_id, component_id)
        technique_result = _timed(
            "component_solve_completed",
            lambda: guarded(
                "component_solve_completed",
                lambda: technique(technique_request),
            ),
            timings,
            cache_status=checkpoints["technique_solve"]["status"],
        )
        validate_result(technique_request, technique_result)
        emit(
            "component_solve_completed",
            stored_parameter_count=int(technique_result.stored_parameter_count),
            provider_sha256=provider.fingerprint,
        )

        install = _timed(
            "component_installed",
            lambda: guarded(
                "component_installed",
                lambda: _install_proof(
                    session, technique_result, phase_keys["component_install"]
                ),
            ),
            timings,
            cache_status=checkpoints["component_install"]["status"],
        )
        emit("component_installed", installation_sha256=canonical_sha256(install))

        boundary = _timed(
            "boundary_evaluation_completed",
            lambda: guarded("boundary_evaluation_completed", session.evaluate_boundary),
            timings,
            cache_status="miss",
        )
        boundary = _mapping(boundary, "boundary")
        emit(
            "boundary_evaluation_completed", boundary_sha256=canonical_sha256(boundary)
        )

        composed = _timed(
            "composed_evaluation_completed",
            lambda: guarded("composed_evaluation_completed", session.evaluate_composed),
            timings,
            cache_status="miss",
        )
        composed = _mapping(composed, "composed")
        emit(
            "composed_evaluation_completed", composed_sha256=canonical_sha256(composed)
        )

        free_run = _timed(
            "free_run_completed",
            lambda: guarded(
                "free_run_completed",
                lambda: session.evaluate_free_run(phase_keys["behavior_evaluation"]),
            ),
            timings,
            cache_status=checkpoints["behavior_evaluation"]["status"],
        )
        free_run = _mapping(free_run, "free_run")
        emit("free_run_completed", free_run_sha256=canonical_sha256(free_run))

        smoke = _timed(
            "decode_smoke_completed",
            lambda: guarded(
                "decode_smoke_completed",
                lambda: session.evaluate_decode_smoke(
                    phase_keys["behavior_evaluation"]
                ),
            ),
            timings,
            cache_status=checkpoints["behavior_evaluation"]["status"],
        )
        smoke = _plain(smoke, "decode_smoke")
        if not isinstance(smoke, list):
            raise Lfm2MatrixIsolationExecutorError("decode_smoke must be a list.")
        emit("decode_smoke_completed", decode_smoke_sha256=canonical_sha256(smoke))

        diagnostics = _component_diagnostics(request, provider, boundary, composed)
        boundary.pop("component_diagnostics", None)
        composed.pop("component_diagnostics", None)
        solve = _mapping(technique_result.solve, "solve")
        solve.update(
            {
                "stored_parameter_count": int(technique_result.stored_parameter_count),
                "parent_component_parameter_count": int(
                    technique_request.component.parent_parameter_count
                ),
                "theoretical_parameter_delta": int(
                    technique_result.stored_parameter_count
                    - technique_request.component.parent_parameter_count
                ),
                "kernel_provider_sha256": provider.fingerprint,
                "physical_storage_claim_authority": False,
            }
        )
        artifacts = {
            "execution_surface": "parent_shell_component_substitution",
            "physical_child_materialized": False,
            "stored_parameter_count_claim_authority": False,
            "kernel_provider": provider.as_dict(),
            "kernel_provider_sha256": provider.fingerprint,
            "isolation_backend_reference": backend_reference,
            "isolation_backend_sha256": backend_sha256,
            "phase_key_sha256s": phase_keys,
            "installation": install,
            "theoretical_storage_only": True,
        }
        return {
            "code_identity": {
                **_mapping(session.code_identity, "code_identity"),
                "isolation_executor_schema": ISOLATION_EXECUTOR_SCHEMA,
                "isolation_executor_sha256": executor.implementation_sha256,
                "isolation_backend_sha256": backend_sha256,
                "kernel_provider_sha256": provider.fingerprint,
            },
            "evidence": _mapping(session.evidence, "evidence"),
            "calibration": calibration,
            "solve": solve,
            "boundary": boundary,
            "composed": composed,
            "behavior": {"free_run": free_run, "decode_smoke": smoke},
            "operation_audit": _observed_operation_audit(
                session.operation_audit,
                technique_result.solve,
                guard_reports,
            ),
            "numeric_execution": _mapping(
                session.numeric_execution, "numeric_execution"
            ),
            "resources": {
                **_mapping(session.resources, "resources"),
                "theoretical_stored_parameter_count": int(
                    technique_result.stored_parameter_count
                ),
                "physical_compaction_claimed": False,
            },
            "checkpoint_reuse": checkpoints,
            "phase_timings": {stage: timings[stage] for stage in _STAGES},
            "component_diagnostics": diagnostics,
            "artifacts": artifacts,
        }
    finally:
        if session is not None:
            session.close()


def t4_isolation_executor() -> MatrixTechniqueExecutor:
    return _executor("T4")


def t5_isolation_executor() -> MatrixTechniqueExecutor:
    return _executor("T5")


def t7_isolation_executor() -> MatrixTechniqueExecutor:
    return _executor("T7")


def t8_isolation_executor() -> MatrixTechniqueExecutor:
    return _executor("T8")


__all__ = [
    "ISOLATION_EXECUTOR_SCHEMA",
    "Lfm2MatrixIsolationExecutorError",
    "MatrixIsolationBackendFactory",
    "MatrixIsolationBackendSession",
    "t4_isolation_executor",
    "t5_isolation_executor",
    "t7_isolation_executor",
    "t8_isolation_executor",
]
