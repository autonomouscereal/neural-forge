#!/usr/bin/env python3
"""Deterministically recover authenticated guided checkpoints 40/56/64.

This wrapper does not duplicate the heavy guided execution path.  It binds a
fresh current guided preregistration to the completed historical schemafix-v2
run, adds the durable checkpoint sink to the canonical trainer call, and then
delegates model/cache/transaction execution and exact rollback to the existing
guided runner.  Research packs remain private sink staging until the complete
historical terminal record authenticates.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from torch import nn

import run_qwen_exact_target_native_guided_continuation as guided
import run_qwen_exact_target_native_stage as stage
from neural_forge.hf_guided_checkpoint_recovery import (
    GuidedCheckpointRecoverySink,
    HistoricalCheckpointExpectation,
    canonical_json_bytes,
    canonical_sha256,
    file_sha256,
)


RECOVERY_PREREGISTRATION_SCHEMA = (
    "neural-forge-qwen-guided-checkpoint-recovery-preregistration-v1"
)
RECOVERY_RUN_SCHEMA = "neural-forge-qwen-guided-checkpoint-recovery-run-v1"
RETAINED_STEPS = (40, 56, 64)
EXPECTED_CHECKPOINT_STEPS = (0, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64)
EXPECTED_HISTORICAL_STATUS = "rejected"
EXPECTED_HISTORICAL_REASON = "fit-gate-rejected"
EXPECTED_HISTORICAL_SELECTED_STEP = 0
EXPECTED_ROLLBACK_STATE_SHA256 = (
    "39c57212e804280b67e9706fc2935a038c41c0f8491ab950af6b45f4a21a606e"
)
EXPECTED_RUNTIME = {
    "dtype": "float16",
    "attention_implementation": "eager",
    "explicit_execute_required": True,
    "cuda_visible_devices": "4,6",
    "parent_device_map": "cuda:0",
    "target_device_map": "cuda:1",
    "parent_device_name": "NVIDIA GeForce RTX 3090",
    "parent_compute_capability": [8, 6],
    "target_device_name": "Quadro RTX 8000",
    "target_compute_capability": [7, 5],
}
_SHA = re.compile(r"^[0-9a-f]{64}$")


class GuidedCheckpointRecoveryRunError(RuntimeError):
    """Recovery preregistration, replay, or terminal evidence is invalid."""


def _sha256(value: Any, name: str) -> str:
    result = str(value).strip().lower()
    if _SHA.fullmatch(result) is None:
        raise GuidedCheckpointRecoveryRunError(f"{name} must be a SHA-256.")
    return result


def _strict_keys(value: Any, expected: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise GuidedCheckpointRecoveryRunError(
            f"{name} keys differ; expected={sorted(expected)!r}, actual={actual!r}."
        )
    return value


def _bound_json(path: Path | str, expected_sha256: str, name: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    expected = _sha256(expected_sha256, f"{name}_sha256")
    if source.is_symlink() or not source.is_file() or file_sha256(source) != expected:
        raise GuidedCheckpointRecoveryRunError(f"{name} file binding changed.")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GuidedCheckpointRecoveryRunError(f"{name} is not strict JSON.") from error
    if not isinstance(value, dict):
        raise GuidedCheckpointRecoveryRunError(f"{name} must contain one object.")
    canonical_json_bytes(value)
    return value


def _module_sha256(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise GuidedCheckpointRecoveryRunError(
            f"Required source module {module_name!r} is unavailable."
        )
    return file_sha256(Path(spec.origin).resolve())


def _source_hashes(*, builder_path: Path) -> dict[str, str]:
    runner_path = Path(__file__).resolve()
    guided_path = Path(guided.__file__).resolve()
    sink_spec = importlib.util.find_spec("neural_forge.hf_guided_checkpoint_recovery")
    trainer_spec = importlib.util.find_spec(guided.GUIDED_TRAINER_MODULE)
    if (
        sink_spec is None
        or sink_spec.origin is None
        or trainer_spec is None
        or trainer_spec.origin is None
    ):
        raise GuidedCheckpointRecoveryRunError("Recovery source modules are absent.")
    return {
        "builder_sha256": file_sha256(builder_path.resolve()),
        "runner_sha256": file_sha256(runner_path),
        "current_guided_runner_sha256": file_sha256(guided_path),
        "trainer_sha256": file_sha256(Path(trainer_spec.origin).resolve()),
        "checkpoint_sink_sha256": file_sha256(Path(sink_spec.origin).resolve()),
    }


def _resolve_split(
    row: Mapping[str, Any], *, base: Path, name: str
) -> stage.SplitBinding:
    try:
        return stage._load_split(row, name=name, base=base)
    except Exception as error:
        raise GuidedCheckpointRecoveryRunError(
            f"Historical {name} split failed authentication."
        ) from error


def _split_semantics(binding: stage.SplitBinding) -> dict[str, Any]:
    return {
        "sha256": binding.sha256,
        "expected_count": binding.expected_count,
        "row_ids": list(binding.row_ids),
        "prompt_sha256s": list(binding.prompt_sha256s),
        "payload_sha256": canonical_sha256(binding.payload),
    }


def _checkpoint_expectations(
    evidence: Mapping[str, Any], training: Mapping[str, Any]
) -> tuple[HistoricalCheckpointExpectation, ...]:
    rows = evidence.get("checkpoints")
    if not isinstance(rows, list):
        raise GuidedCheckpointRecoveryRunError(
            "Historical trainer evidence omits checkpoint rows."
        )
    schedule_steps = tuple(int(value) for value in training.get("checkpoint_steps", ()))
    if schedule_steps != EXPECTED_CHECKPOINT_STEPS:
        raise GuidedCheckpointRecoveryRunError(
            "Historical recovery checkpoint schedule is not exact."
        )
    values: list[HistoricalCheckpointExpectation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise GuidedCheckpointRecoveryRunError(
                f"Historical checkpoint row {index} is malformed."
            )
        step_value = row.get("step")
        metrics = row.get("metrics")
        if not isinstance(step_value, int) or not isinstance(metrics, Mapping):
            raise GuidedCheckpointRecoveryRunError(
                f"Historical checkpoint row {index} is incomplete."
            )
        values.append(
            HistoricalCheckpointExpectation(
                step=step_value,
                metrics=dict(metrics),
                metrics_sha256=canonical_sha256(metrics),
                factor_state_sha256=_sha256(
                    row.get("factor_state_sha256"),
                    f"historical checkpoint {step_value} factor state",
                ),
            )
        )
    if tuple(value.step for value in values) != EXPECTED_CHECKPOINT_STEPS:
        raise GuidedCheckpointRecoveryRunError(
            "Historical checkpoint rows do not match the exact ordered schedule."
        )
    if not set(RETAINED_STEPS).issubset(value.step for value in values):
        raise GuidedCheckpointRecoveryRunError("Retained checkpoint rows are absent.")
    return tuple(values)


def build_recovery_preregistration_payload(
    *,
    current_guided_preregistration: Path,
    current_guided_preregistration_sha256: str,
    historical_guided_preregistration: Path,
    historical_guided_preregistration_sha256: str,
    historical_guided_run_report: Path,
    historical_guided_run_report_sha256: str,
    builder_path: Path,
) -> tuple[dict[str, Any], guided.AuthenticatedGuidedContinuation]:
    current_path = current_guided_preregistration.expanduser().resolve()
    current_sha = _sha256(
        current_guided_preregistration_sha256,
        "current_guided_preregistration_sha256",
    )
    historical_prereg_path = historical_guided_preregistration.expanduser().resolve()
    historical_prereg_sha = _sha256(
        historical_guided_preregistration_sha256,
        "historical_guided_preregistration_sha256",
    )
    historical_report_path = historical_guided_run_report.expanduser().resolve()
    historical_report_sha = _sha256(
        historical_guided_run_report_sha256,
        "historical_guided_run_report_sha256",
    )
    current_raw = _bound_json(current_path, current_sha, "current guided preregistration")
    current_build_path = current_path.with_name("build_report.json")
    if current_build_path.is_symlink() or not current_build_path.is_file():
        raise GuidedCheckpointRecoveryRunError(
            "Fresh current guided preregistration has no build report."
        )
    current_build_sha = file_sha256(current_build_path)
    current_build = _bound_json(
        current_build_path, current_build_sha, "current guided build report"
    )
    if (
        current_build.get("preregistration_sha256") != current_sha
        or current_build.get("ready_for_explicit_preflight") is not True
        or current_build.get("training_launched") is not False
        or current_build.get("model_weights_loaded") is not False
        or current_build.get("model_checkpoint_created") is not False
    ):
        raise GuidedCheckpointRecoveryRunError(
            "Current guided build report does not prove a fresh model-free contract."
        )
    historical_raw = _bound_json(
        historical_prereg_path,
        historical_prereg_sha,
        "historical guided preregistration",
    )
    historical_report = _bound_json(
        historical_report_path,
        historical_report_sha,
        "historical guided run report",
    )
    try:
        current = guided.authenticate_guided_preregistration(current_path, current_sha)
    except guided.GuidedContinuationError as error:
        raise GuidedCheckpointRecoveryRunError(
            "Fresh current guided preregistration failed production authentication."
        ) from error

    semantic_fields = ("source", "audit", "training", "seed", "runtime")
    for field_name in semantic_fields:
        if canonical_json_bytes(current_raw.get(field_name)) != canonical_json_bytes(
            historical_raw.get(field_name)
        ):
            raise GuidedCheckpointRecoveryRunError(
                f"Current guided preregistration has historical {field_name} drift."
            )
    training = current_raw.get("training")
    if not isinstance(training, Mapping):
        raise GuidedCheckpointRecoveryRunError("Current guided training is malformed.")
    schedule_type = getattr(
        importlib.import_module(guided.GUIDED_TRAINER_MODULE),
        "BehaviorDistillationSchedule",
    )
    normalized_training = schedule_type(**dict(training)).as_dict()
    if canonical_json_bytes(normalized_training) != canonical_json_bytes(training):
        raise GuidedCheckpointRecoveryRunError(
            "Current guided schedule does not normalize to its exact preregistration."
        )
    if normalized_training.get("sealed_evaluation_policy") is not None:
        raise GuidedCheckpointRecoveryRunError(
            "Recovery must retain the historical default candidate-gated schedule."
        )

    historical_split_rows = historical_raw.get("splits")
    if not isinstance(historical_split_rows, Mapping) or set(historical_split_rows) != {
        "fit",
        "candidate",
        "sealed",
    }:
        raise GuidedCheckpointRecoveryRunError("Historical split bindings are malformed.")
    split_semantics: dict[str, Any] = {}
    for name in ("fit", "candidate", "sealed"):
        row = historical_split_rows[name]
        if not isinstance(row, Mapping):
            raise GuidedCheckpointRecoveryRunError(
                f"Historical {name} split binding is malformed."
            )
        historical_binding = _resolve_split(
            row, base=historical_prereg_path.parent, name=name
        )
        current_binding = current.execution_contract.splits[name]
        historical_semantics = _split_semantics(historical_binding)
        current_semantics = _split_semantics(current_binding)
        if canonical_json_bytes(historical_semantics) != canonical_json_bytes(
            current_semantics
        ):
            raise GuidedCheckpointRecoveryRunError(
                f"Current {name} split has historical semantic drift."
            )
        split_semantics[name] = current_semantics

    evidence = historical_report.get("trainer_evidence")
    evidence_sha = _sha256(
        historical_report.get("trainer_evidence_sha256"),
        "historical trainer evidence",
    )
    if not isinstance(evidence, Mapping) or canonical_sha256(evidence) != evidence_sha:
        raise GuidedCheckpointRecoveryRunError(
            "Historical trainer evidence failed canonical authentication."
        )
    expectations = _checkpoint_expectations(evidence, training)
    if (
        historical_report.get("preregistration_sha256") != historical_prereg_sha
        or historical_report.get("status") != EXPECTED_HISTORICAL_STATUS
        or historical_report.get("reason") != EXPECTED_HISTORICAL_REASON
        or historical_report.get("rollback_exact") is not True
        or historical_report.get("rollback_state_sha256")
        != EXPECTED_ROLLBACK_STATE_SHA256
        or historical_report.get("model_checkpoint_written") is not False
        or evidence.get("selected_step") != EXPECTED_HISTORICAL_SELECTED_STEP
        or evidence.get("sealed_metrics") is not None
        or evidence.get("sealed_evaluation_unlocked") is not False
        or evidence.get("schedule_sha256") != canonical_sha256(training)
        or not isinstance(evidence.get("domain_binding_sha256"), str)
        or not isinstance(evidence.get("training_audit"), Mapping)
    ):
        raise GuidedCheckpointRecoveryRunError(
            "Historical guided terminal boundary changed."
        )
    cache = current.execution_contract.compiler_cache
    historical_cache = historical_report.get("compiled_child_cache")
    if (
        cache is None
        or not isinstance(historical_cache, Mapping)
        or cache.expected_compiler_identity_sha256
        != historical_cache.get("compiler_identity_sha256")
        or cache.expected_receipt_sha256 != historical_cache.get("receipt_sha256")
    ):
        raise GuidedCheckpointRecoveryRunError(
            "Current compiler cache differs from historical replay identity."
        )
    payload = {
        "schema": RECOVERY_PREREGISTRATION_SCHEMA,
        "current_guided": {
            "preregistration": str(current_path),
            "preregistration_sha256": current_sha,
            "build_report": str(current_build_path),
            "build_report_sha256": current_build_sha,
            "evaluation_contract_sha256": current.evaluation_contract_sha256,
        },
        "historical_guided": {
            "preregistration": str(historical_prereg_path),
            "preregistration_sha256": historical_prereg_sha,
            "run_report": str(historical_report_path),
            "run_report_sha256": historical_report_sha,
            "trainer_evidence_sha256": evidence_sha,
            "schedule_sha256": evidence["schedule_sha256"],
            "domain_binding_sha256": evidence["domain_binding_sha256"],
            "selected_step": EXPECTED_HISTORICAL_SELECTED_STEP,
            "status": EXPECTED_HISTORICAL_STATUS,
            "reason": EXPECTED_HISTORICAL_REASON,
            "rollback_state_sha256": EXPECTED_ROLLBACK_STATE_SHA256,
            "training_audit": dict(evidence["training_audit"]),
            "checkpoint_rows": [dict(value) for value in evidence["checkpoints"]],
        },
        "checkpoint_expectations": [value.as_dict() for value in expectations],
        "retained_steps": list(RETAINED_STEPS),
        "semantic_contract": {
            "seed": current_raw["seed"],
            "training": dict(training),
            "runtime": dict(current_raw["runtime"]),
            "splits": split_semantics,
            "source_sha256": canonical_sha256(current_raw["source"]),
            "audit_sha256": canonical_sha256(current_raw["audit"]),
            "compiler_identity_sha256": cache.expected_compiler_identity_sha256,
            "compiler_cache_receipt_sha256": cache.expected_receipt_sha256,
            "historical_semantics_match": True,
        },
        "execution_runtime": dict(EXPECTED_RUNTIME),
        "source_implementation": _source_hashes(builder_path=builder_path),
        "model_free_preregistration": True,
        "model_weights_loaded": False,
        "gpu_work_launched": False,
        "sealed_policy": "historical-default-candidate-gated",
        "sealed_must_remain_unaccessed": True,
        "publication_policy": {
            "publish_only_after_exact_terminal_authentication": True,
            "retained_steps": list(RETAINED_STEPS),
            "maximum_research_packs": 3,
            "research_only_not_standalone": True,
        },
    }
    payload["semantic_contract_sha256"] = canonical_sha256(
        payload["semantic_contract"]
    )
    payload["checkpoint_expectations_sha256"] = canonical_sha256(
        payload["checkpoint_expectations"]
    )
    return payload, current


@dataclass(frozen=True, slots=True)
class AuthenticatedGuidedCheckpointRecovery:
    path: Path
    file_sha256: str
    current_contract: guided.AuthenticatedGuidedContinuation
    expectations: tuple[HistoricalCheckpointExpectation, ...]
    retained_steps: tuple[int, ...]
    raw: Mapping[str, Any]

    def preflight_report(self) -> dict[str, Any]:
        return {
            "schema": RECOVERY_RUN_SCHEMA,
            "status": "preflight-passed",
            "recovery_preregistration": str(self.path),
            "recovery_preregistration_sha256": self.file_sha256,
            "current_guided": dict(self.raw["current_guided"]),
            "historical_guided": {
                name: value
                for name, value in self.raw["historical_guided"].items()
                if name not in {"training_audit", "checkpoint_rows"}
            },
            "expected_checkpoint_steps": [value.step for value in self.expectations],
            "retained_steps": list(self.retained_steps),
            "execution_runtime": dict(self.raw["execution_runtime"]),
            "source_implementation": dict(self.raw["source_implementation"]),
            "semantic_contract_sha256": self.raw["semantic_contract_sha256"],
            "checkpoint_expectations_sha256": self.raw[
                "checkpoint_expectations_sha256"
            ],
            "models_loaded": False,
            "gpu_work_launched": False,
        }


def authenticate_recovery_preregistration(
    path: Path | str, expected_sha256: str
) -> AuthenticatedGuidedCheckpointRecovery:
    source = Path(path).expanduser().resolve()
    expected_sha = _sha256(expected_sha256, "recovery_preregistration_sha256")
    raw = _bound_json(source, expected_sha, "recovery preregistration")
    expected_keys = {
        "schema",
        "current_guided",
        "historical_guided",
        "checkpoint_expectations",
        "retained_steps",
        "semantic_contract",
        "execution_runtime",
        "source_implementation",
        "model_free_preregistration",
        "model_weights_loaded",
        "gpu_work_launched",
        "sealed_policy",
        "sealed_must_remain_unaccessed",
        "publication_policy",
        "semantic_contract_sha256",
        "checkpoint_expectations_sha256",
    }
    _strict_keys(raw, expected_keys, "recovery preregistration")
    current_row = _strict_keys(
        raw["current_guided"],
        {
            "preregistration",
            "preregistration_sha256",
            "build_report",
            "build_report_sha256",
            "evaluation_contract_sha256",
        },
        "current_guided",
    )
    historical_row = _strict_keys(
        raw["historical_guided"],
        {
            "preregistration",
            "preregistration_sha256",
            "run_report",
            "run_report_sha256",
            "trainer_evidence_sha256",
            "schedule_sha256",
            "domain_binding_sha256",
            "selected_step",
            "status",
            "reason",
            "rollback_state_sha256",
            "training_audit",
            "checkpoint_rows",
        },
        "historical_guided",
    )
    builder_path = Path(__file__).resolve().with_name(
        "prepare_qwen_exact_target_native_guided_checkpoint_recovery.py"
    )
    expected_payload, current = build_recovery_preregistration_payload(
        current_guided_preregistration=Path(str(current_row["preregistration"])),
        current_guided_preregistration_sha256=str(
            current_row["preregistration_sha256"]
        ),
        historical_guided_preregistration=Path(
            str(historical_row["preregistration"])
        ),
        historical_guided_preregistration_sha256=str(
            historical_row["preregistration_sha256"]
        ),
        historical_guided_run_report=Path(str(historical_row["run_report"])),
        historical_guided_run_report_sha256=str(historical_row["run_report_sha256"]),
        builder_path=builder_path,
    )
    if canonical_json_bytes(raw) != canonical_json_bytes(expected_payload):
        raise GuidedCheckpointRecoveryRunError(
            "Recovery preregistration differs from freshly authenticated sources."
        )
    expectations = tuple(
        HistoricalCheckpointExpectation(
            step=value["step"],
            metrics=value["metrics"],
            metrics_sha256=value["metrics_sha256"],
            factor_state_sha256=value["factor_state_sha256"],
            expected_checkpoint_sha256=value["expected_checkpoint_sha256"],
        )
        for value in raw["checkpoint_expectations"]
    )
    return AuthenticatedGuidedCheckpointRecovery(
        path=source,
        file_sha256=expected_sha,
        current_contract=current,
        expectations=expectations,
        retained_steps=tuple(int(value) for value in raw["retained_steps"]),
        raw=raw,
    )


class RecoveryProductionGuidedRuntime(guided.ProductionGuidedRuntime):
    """Canonical guided runtime with only the durable callback added."""

    def __init__(self, recovery_sink: GuidedCheckpointRecoverySink, **kwargs: Any):
        super().__init__(**kwargs)
        self.recovery_sink = recovery_sink

    def train_guided(
        self,
        transaction: Any,
        parent: nn.Module,
        contract: guided.AuthenticatedGuidedContinuation,
    ) -> Any:
        module = importlib.import_module(guided.GUIDED_TRAINER_MODULE)
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
        if (
            schedule.sealed_evaluation_policy != "candidate-gated"
            or canonical_json_bytes(schedule.as_dict())
            != canonical_json_bytes(base_contract.training)
        ):
            raise GuidedCheckpointRecoveryRunError(
                "Recovery changed the exact historical guided schedule."
            )
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
            durable_checkpoint_callback=self.recovery_sink,
        )
        result_type = getattr(module, "BehaviorDistillationResult")
        if type(result) is not result_type:
            raise GuidedCheckpointRecoveryRunError(
                "Recovery trainer returned a noncanonical type."
            )
        return result


def _validate_terminal_report(
    report: Mapping[str, Any], contract: AuthenticatedGuidedCheckpointRecovery
) -> tuple[str, list[Mapping[str, Any]]]:
    historical = contract.raw["historical_guided"]
    evidence = report.get("trainer_evidence")
    evidence_sha = report.get("trainer_evidence_sha256")
    if (
        not isinstance(evidence, Mapping)
        or evidence_sha != historical["trainer_evidence_sha256"]
        or canonical_sha256(evidence) != evidence_sha
        or evidence.get("schedule_sha256") != historical["schedule_sha256"]
        or evidence.get("domain_binding_sha256") != historical["domain_binding_sha256"]
        or evidence.get("selected_step") != EXPECTED_HISTORICAL_SELECTED_STEP
        or canonical_json_bytes(evidence.get("training_audit"))
        != canonical_json_bytes(historical["training_audit"])
        or canonical_json_bytes(evidence.get("checkpoints"))
        != canonical_json_bytes(historical["checkpoint_rows"])
        or evidence.get("sealed_metrics") is not None
        or evidence.get("sealed_evaluation_unlocked") is not False
        or report.get("status") != EXPECTED_HISTORICAL_STATUS
        or report.get("reason") != EXPECTED_HISTORICAL_REASON
        or report.get("rollback_exact") is not True
        or report.get("rollback_state_sha256") != EXPECTED_ROLLBACK_STATE_SHA256
        or report.get("model_checkpoint_written") is not False
    ):
        raise GuidedCheckpointRecoveryRunError(
            "Guided replay terminal evidence differs from the historical run."
        )
    decisions = report.get("pre_bake_behavior_decisions")
    if not isinstance(decisions, Mapping) or decisions.get("sealed") is not None:
        raise GuidedCheckpointRecoveryRunError(
            "Guided replay unexpectedly accessed sealed evaluation."
        )
    return str(evidence_sha), list(evidence["checkpoints"])


def execute_authenticated_guided_checkpoint_recovery(
    contract: AuthenticatedGuidedCheckpointRecovery,
    output_root: Path | str,
    *,
    runtime_factory: Callable[..., Any] = RecoveryProductionGuidedRuntime,
    guided_executor: Callable[..., Mapping[str, Any]] = (
        guided.execute_authenticated_guided_continuation
    ),
    dtype: str = "float16",
    parent_device_map: str = "cuda:0",
    target_device_map: str = "cuda:1",
) -> dict[str, Any]:
    refreshed = authenticate_recovery_preregistration(contract.path, contract.file_sha256)
    if canonical_json_bytes(refreshed.raw) != canonical_json_bytes(contract.raw):
        raise GuidedCheckpointRecoveryRunError(
            "Recovery sources changed immediately before execution."
        )
    expected_runtime = refreshed.raw["execution_runtime"]
    if (
        dtype != expected_runtime["dtype"]
        or parent_device_map != expected_runtime["parent_device_map"]
        or target_device_map != expected_runtime["target_device_map"]
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        != expected_runtime["cuda_visible_devices"]
    ):
        raise GuidedCheckpointRecoveryRunError(
            "Recovery runtime/GPU mapping differs from the historical contract."
        )
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise GuidedCheckpointRecoveryRunError("output_root must not already exist.")
    output.mkdir(parents=True)
    guided_output = output / "guided_replay"
    research_output = output / "recovered_research"
    historical = refreshed.raw["historical_guided"]
    sink = GuidedCheckpointRecoverySink(
        research_output,
        expectations=refreshed.expectations,
        retained_steps=refreshed.retained_steps,
        recovery_preregistration_sha256=refreshed.file_sha256,
        recovery_report_sha256=historical["run_report_sha256"],
        source_run_report_sha256=(
            refreshed.current_contract.artifacts.source_run_report_sha256
        ),
    )
    report: dict[str, Any] = {
        **refreshed.preflight_report(),
        "status": "running",
        "output_root": str(output),
        "guided_replay": str(guided_output),
        "recovered_research": str(research_output),
    }
    publication_complete = False
    try:
        runtime = runtime_factory(
            sink,
            dtype=dtype,
            parent_device_map=parent_device_map,
            target_device_map=target_device_map,
        )
        guided_report = dict(
            guided_executor(
                refreshed.current_contract,
                guided_output,
                runtime=runtime,
            )
        )
        evidence_sha, checkpoint_rows = _validate_terminal_report(
            guided_report, refreshed
        )
        publication = sink.finalize(
            trainer_evidence_sha256=evidence_sha,
            terminal_checkpoint_rows=checkpoint_rows,
        )
        publication_complete = True
        report.update(
            {
                "status": "complete",
                "historical_replay_exact": True,
                "sealed_evaluation_accessed": False,
                "guided_replay_report": str(guided_output / "run_report.json"),
                "guided_replay_report_sha256": file_sha256(
                    guided_output / "run_report.json"
                ),
                "research_publication": publication.as_dict(),
                "retained_steps": list(RETAINED_STEPS),
                "research_only_not_standalone": True,
            }
        )
        stage._atomic_json(output / "run_report.json", report)
        return report
    except BaseException as error:
        sink.cleanup()
        if publication_complete and research_output.exists():
            if (
                research_output.parent != output
                or research_output.name != "recovered_research"
                or research_output.is_symlink()
            ):
                raise GuidedCheckpointRecoveryRunError(
                    "Refusing to clean an unexpected published research path."
                ) from error
            shutil.rmtree(research_output)
        if research_output.exists():
            raise GuidedCheckpointRecoveryRunError(
                "Failed recovery unexpectedly published a research tree."
            ) from error
        failure = {
            **report,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "research_tree_published": False,
            "sink_state": sink.state,
        }
        try:
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
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--dtype", choices=("float16",), default="float16")
    parser.add_argument("--parent-device-map", default="cuda:0")
    parser.add_argument("--target-device-map", default="cuda:1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    contract = authenticate_recovery_preregistration(
        args.preregistration, args.preregistration_sha256
    )
    if args.preflight:
        if args.output_root is not None:
            raise SystemExit("--output-root is only valid with --execute.")
        print(json.dumps(contract.preflight_report(), indent=2, sort_keys=True))
        return 0
    if args.output_root is None:
        raise SystemExit("--output-root is required with --execute.")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_RUNTIME["cuda_visible_devices"]:
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 4,6 for recovery.")
    result = execute_authenticated_guided_checkpoint_recovery(
        contract,
        args.output_root,
        dtype=args.dtype,
        parent_device_map=args.parent_device_map,
        target_device_map=args.target_device_map,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
