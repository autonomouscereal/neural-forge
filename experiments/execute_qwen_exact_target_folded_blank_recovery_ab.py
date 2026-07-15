#!/usr/bin/env python3
"""Execute the preregistered exact-target folded-vs-blank recovery experiment.

This is deliberately an experimental materialization and recovery hook, not a
publication path.  It fresh-authenticates the model-free preregistration,
temporarily materializes the retained v4 winner through the existing bridge,
authenticates both zero-step controls, and only then launches the two symmetric
recovery arms.  Recovery reports are detached from model directories before
any standalone-child reload or artifact hash.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import run_qwen_exact_target_folded_blank_recovery_ab as recovery_ab
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import run_qwen_exact_target_native_guided_gud_winner_publication as publication
import run_qwen_exact_target_native_stage as stage
from neural_forge import hf_experimental_child_artifact as child_artifact
from neural_forge import hf_recovery
from neural_forge import hf_teacher_token_replay
from neural_forge import hf_progressive_target_bridge as progressive_bridge
from neural_forge.hf_compiled_child_cache import (
    CACHE_RECEIPT_FILENAME,
    CompiledChildCacheReceipt,
    CompilerIdentity,
    model_state_fingerprint,
)


EXECUTION_SCHEMA = "neural-forge-folded-blank-recovery-ab-execution-v1"
COMPARISON_SCHEMA = "neural-forge-folded-blank-recovery-ab-comparison-v1"
RECOVERY_REPORT_FILENAME = "neural_forge_xfamily_recovery_report.json"
COMPARISON_REPORT_FILENAME = "comparison_report.json"
EVIDENCE_DIRECTORY_NAME = "evidence"
FOLDED_INIT_DIRECTORY_NAME = "folded_init"
FOLDED_FINAL_DIRECTORY_NAME = "folded_final"
BLANK_FINAL_DIRECTORY_NAME = "blank_final"
ARM_IDS = ("folded", "blank")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_AUDIT_FIELDS = frozenset(
    {
        "adapter_or_registry_module_count",
        "optimizer_parameter_count",
        "source_sized_sidecar_parameter_count",
        "parent_runtime_reference_count",
    }
)


class FoldedBlankRecoveryExecutionError(RuntimeError):
    """The production materialization or symmetric recovery failed closed."""


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FoldedBlankRecoveryExecutionError(
            f"{name} must be one lowercase SHA-256."
        )
    return value


def _plain_json_value(value: Any) -> Any:
    """Thaw immutable JSON containers without accepting non-JSON leaves."""

    if isinstance(value, Mapping):
        return {key: _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json_value(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _plain_json_value(value),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as error:
        raise FoldedBlankRecoveryExecutionError(
            "Execution evidence must be canonical JSON."
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_bytes(value).decode("ascii"))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FoldedBlankRecoveryExecutionError(f"{name} must be a mapping.")
    return value


def _self_hashed_receipt(body: Mapping[str, Any]) -> dict[str, Any]:
    result = _json_copy(body)
    if not isinstance(result, dict) or "receipt_sha256" in result:
        raise FoldedBlankRecoveryExecutionError(
            "Receipt bodies must be JSON objects without a preexisting self-hash."
        )
    result["receipt_sha256"] = _canonical_sha256(result)
    return result


@dataclass(frozen=True, slots=True)
class MaterializationProduct:
    """Raw materialization receipts and the only two initialization paths."""

    receipts: Mapping[str, Mapping[str, Any]]
    folded_publication_root: Path
    folded_candidate_path: Path
    blank_candidate_path: Path
    source_model_path: Path
    trace: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ZeroStepEvaluation:
    """One independently run and cleaned step-zero evaluation."""

    event: Mapping[str, Any]
    event_sha256: str
    behavior: Mapping[str, Any]
    report: Mapping[str, Any]
    report_file_sha256: str
    clean_artifact: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class DetachedRecoveryReport:
    """A report moved out of a now-clean model artifact directory."""

    report: Mapping[str, Any]
    path: Path
    file_sha256: str


@dataclass(frozen=True, slots=True)
class ExecutorDependencies:
    """Dependency-injected heavy and evidence boundaries for workstation tests."""

    authenticate_contract: Callable[[Path | str, str], Any]
    authenticate_teacher_replay: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    materialize_initializations: Callable[..., MaterializationProduct]
    authenticate_materializations: Callable[..., Mapping[str, Any]]
    authenticate_source_model: Callable[..., Mapping[str, Any]]
    authenticate_zero_steps: Callable[..., Mapping[str, Any]]
    zero_step_evaluator: Callable[..., ZeroStepEvaluation]
    recover: Callable[..., Mapping[str, Any]]
    reset_rng: Callable[[Mapping[str, Any]], None]
    authenticate_clean_artifact: Callable[..., Mapping[str, Any]]
    load_json: Callable[[Path], Mapping[str, Any]]
    file_sha256: Callable[[Path], str]
    atomic_json: Callable[[Path, Mapping[str, Any]], None]
    release_memory: Callable[[], None]


_BEHAVIOR_ENTRY_FIELDS = frozenset(
    {
        "step",
        "loss",
        "correct",
        "total",
        "pass_rate",
        "generation_health_correct",
        "generation_health_total",
        "generation_health_rate",
        "behavior_category_breakdown",
        "generation_category_breakdown",
        "behavior_category_summary",
        "generation_category_summary",
        "passed_ids",
        "generation_passed_prompts",
    }
)


def _validate_behavior_entry(value: Any, *, expected_step: int) -> None:
    if not isinstance(value, Mapping) or set(value) != _BEHAVIOR_ENTRY_FIELDS:
        raise FoldedBlankRecoveryExecutionError(
            "Behavior checkpoint does not have the exact recovery entry shape."
        )
    loss = value["loss"]
    if (
        value["step"] != expected_step
        or (expected_step == 0 and loss is not None)
        or (
            expected_step > 0
            and (
                isinstance(loss, bool)
                or not isinstance(loss, (int, float))
                or not math.isfinite(float(loss))
            )
        )
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Behavior checkpoint step/loss semantics changed."
        )
    count_fields = (
        ("correct", "total", "pass_rate", True),
        (
            "generation_health_correct",
            "generation_health_total",
            "generation_health_rate",
            False,
        ),
    )
    for correct_name, total_name, rate_name, require_positive_total in count_fields:
        correct = value[correct_name]
        total = value[total_name]
        rate = value[rate_name]
        if (
            isinstance(correct, bool)
            or not isinstance(correct, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or correct < 0
            or total < 0
            or (require_positive_total and total == 0)
            or correct > total
            or isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or abs(float(rate) - correct / max(1, total)) > 1e-12
        ):
            raise FoldedBlankRecoveryExecutionError(
                f"Behavior checkpoint {correct_name}/{total_name}/{rate_name} is inconsistent."
            )
    for prefix in ("behavior", "generation"):
        breakdown = value[f"{prefix}_category_breakdown"]
        summary = value[f"{prefix}_category_summary"]
        if not isinstance(breakdown, Mapping) or not isinstance(summary, Mapping):
            raise FoldedBlankRecoveryExecutionError(
                f"{prefix} category evidence must be mappings."
            )
        if list(breakdown) != sorted(breakdown):
            raise FoldedBlankRecoveryExecutionError(
                f"{prefix} category breakdown is not canonically ordered."
            )
        correct_sum = 0
        total_sum = 0
        for category, row in breakdown.items():
            if (
                not isinstance(category, str)
                or not category
                or category != category.strip().lower().replace("-", "_")
                or not isinstance(row, Mapping)
                or set(row)
                != {"correct", "total", "pass_rate", "failed_ids", "passed_ids"}
            ):
                raise FoldedBlankRecoveryExecutionError(
                    f"{prefix} category breakdown row is malformed."
                )
            correct = row["correct"]
            total = row["total"]
            rate = row["pass_rate"]
            if (
                isinstance(correct, bool)
                or not isinstance(correct, int)
                or isinstance(total, bool)
                or not isinstance(total, int)
                or correct < 0
                or total < 0
                or correct > total
                or isinstance(rate, bool)
                or not isinstance(rate, (int, float))
                or abs(float(rate) - correct / max(1, total)) > 1e-12
                or not isinstance(row["passed_ids"], list)
                or not isinstance(row["failed_ids"], list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in row["passed_ids"]
                )
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in row["failed_ids"]
                )
                or len(row["passed_ids"]) != len(set(row["passed_ids"]))
                or len(row["failed_ids"]) != len(set(row["failed_ids"]))
                or set(row["passed_ids"]).intersection(row["failed_ids"])
                or len(row["passed_ids"]) > min(correct, 20)
                or len(row["failed_ids"]) > min(total - correct, 20)
            ):
                raise FoldedBlankRecoveryExecutionError(
                    f"{prefix} category counts/rate/IDs are inconsistent."
                )
            correct_sum += correct
            total_sum += total
        expected_correct = (
            value["correct"]
            if prefix == "behavior"
            else value["generation_health_correct"]
        )
        expected_total = (
            value["total"] if prefix == "behavior" else value["generation_health_total"]
        )
        if (
            correct_sum != expected_correct
            or total_sum != expected_total
            or _canonical_bytes(summary)
            != _canonical_bytes(
                hf_recovery._category_breakdown_summary(dict(breakdown))
            )
        ):
            raise FoldedBlankRecoveryExecutionError(
                f"{prefix} category totals or derived summary changed."
            )
    passed_ids = value["passed_ids"]
    passed_prompts = value["generation_passed_prompts"]
    if (
        not isinstance(passed_ids, list)
        or not isinstance(passed_prompts, list)
        or any(not isinstance(item, str) or not item.strip() for item in passed_ids)
        or any(not isinstance(item, str) or not item.strip() for item in passed_prompts)
        or len(passed_ids) != len(set(passed_ids))
        or len(passed_ids) != value["correct"]
        or len(passed_prompts) != value["generation_health_correct"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Behavior checkpoint passed IDs/prompts disagree with counts."
        )


class FrozenBehaviorObserver:
    """Capture canonical immutable events and gate training at callback step zero."""

    def __init__(
        self,
        *,
        expected_steps: Sequence[int],
        authenticated_zero_event: Mapping[str, Any] | None = None,
    ) -> None:
        if (
            isinstance(expected_steps, (str, bytes))
            or not isinstance(expected_steps, Sequence)
            or not expected_steps
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Observer expected_steps must be a nonempty ordered sequence."
            )
        steps = tuple(expected_steps)
        if any(isinstance(step, bool) or not isinstance(step, int) for step in steps):
            raise FoldedBlankRecoveryExecutionError(
                "Observer checkpoint steps must be integers."
            )
        if steps != tuple(sorted(set(steps))) or steps[0] != 0:
            raise FoldedBlankRecoveryExecutionError(
                "Observer checkpoint steps must be unique, ordered, and start at zero."
            )
        self._expected_steps = steps
        self._event_bytes: list[bytes] = []
        self._authenticated_zero_bytes = (
            None
            if authenticated_zero_event is None
            else _canonical_bytes(authenticated_zero_event)
        )

    def __call__(self, value: Mapping[str, Any]) -> None:
        event = _json_copy(value)
        if not isinstance(event, dict) or set(event) != {
            "schema",
            "operation",
            "checkpoint_index",
            "step",
            "behavior",
        }:
            raise FoldedBlankRecoveryExecutionError(
                "Recovery emitted a malformed behavior observer event."
            )
        index = len(self._event_bytes)
        behavior = event.get("behavior")
        if (
            index >= len(self._expected_steps)
            or event.get("schema")
            != "neural-forge.recovery-behavior-checkpoint-observer.v1"
            or event.get("operation") != "recover_cross_tokenizer_causal_lm"
            or event.get("checkpoint_index") != index
            or event.get("step") != self._expected_steps[index]
            or not isinstance(behavior, dict)
            or behavior.get("step") != event.get("step")
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Recovery behavior checkpoint order or payload changed."
            )
        _validate_behavior_entry(behavior, expected_step=self._expected_steps[index])
        encoded = _canonical_bytes(event)
        if (
            index == 0
            and self._authenticated_zero_bytes is not None
            and encoded != self._authenticated_zero_bytes
        ):
            # The recovery callback runs before the optimizer loop.  Raising here
            # guarantees a changed zero step cannot be followed by one update.
            raise FoldedBlankRecoveryExecutionError(
                "Training step zero differs from the authenticated zero-step receipt."
            )
        self._event_bytes.append(encoded)

    def finalize(self) -> tuple[Mapping[str, Any], ...]:
        if len(self._event_bytes) != len(self._expected_steps):
            raise FoldedBlankRecoveryExecutionError(
                "Recovery did not emit every preregistered behavior checkpoint."
            )
        return tuple(
            MappingProxyType(json.loads(value.decode("ascii")))
            for value in self._event_bytes
        )

    def event_records(self) -> list[dict[str, Any]]:
        events = self.finalize()
        return [
            {
                "event": _json_copy(event),
                "event_sha256": hashlib.sha256(encoded).hexdigest(),
            }
            for event, encoded in zip(events, self._event_bytes, strict=True)
        ]

    def zero_event(self) -> Mapping[str, Any]:
        if not self._event_bytes or self._expected_steps[0] != 0:
            raise FoldedBlankRecoveryExecutionError(
                "Observer has no completed zero-step event."
            )
        return MappingProxyType(json.loads(self._event_bytes[0].decode("ascii")))


@dataclass(frozen=True, slots=True)
class _WinnerRuntimeBinding:
    tournament: Any
    winner: Mapping[str, Any]


def build_experimental_recovery_contract(
    contract: Any,
    *,
    native_child_state_sha256: str,
    live_factor_state_sha256: str,
    restored_runtime_factor_state_sha256: str,
) -> dict[str, Any]:
    """Create exactly the bridge contract authenticated by the A/B runner."""

    blank = _mapping(contract.raw["compiled_blank"], "compiled blank")
    v4 = _mapping(contract.raw["v4_candidate"], "v4 candidate")
    winner = _mapping(v4["winner"], "v4 winner")
    native_sha = _sha256(native_child_state_sha256, "native child state SHA")
    live_sha = _sha256(live_factor_state_sha256, "live factor state SHA")
    runtime_sha = _sha256(
        restored_runtime_factor_state_sha256,
        "restored runtime factor state SHA",
    )
    if runtime_sha != winner["runtime_factor_state_sha256"]:
        raise FoldedBlankRecoveryExecutionError(
            "Restored v4 runtime factor state differs from the retained winner."
        )
    return {
        "schema": recovery_ab.EXPERIMENTAL_RECOVERY_CONTRACT_SCHEMA,
        "recovery_ab_preregistration_sha256": contract.file_sha256,
        "recovery_ab_semantic_contract_sha256": contract.raw[
            "semantic_contract_sha256"
        ],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "factor_state_sha256": winner["portable_factor_state_sha256"],
        "factor_payload_schema": v4["factor_pack_schema"],
        "native_child_state_sha256": native_sha,
        "blank_control_state_sha256": native_sha,
        "compiled_blank_cache_state_dict_sha256": blank["state_dict_sha256"],
        "restored_runtime_factor_state_sha256": runtime_sha,
        "restored_portable_factor_state_sha256": winner["portable_factor_state_sha256"],
        "bridge_native_child_state_sha256": native_sha,
        "bridge_live_factor_state_sha256": live_sha,
        "exact_unique_parameters": recovery_ab.EXACT_UNIQUE_PARAMETERS,
        "publication_authorized": False,
        "sealed_data_accessed": False,
    }


def _production_runtime_inputs(
    contract: Any,
) -> tuple[Any, Any, Path, Mapping[str, Any]]:
    try:
        v4_contract = contract.v4.contract
        source_guided = v4_contract.source.tournament_contract.source
        base = dataclasses.replace(
            source_guided.execution_contract,
            splits=v4_contract.splits,
            training=v4_contract.schedule.as_dict(),
        )
        source_path = Path(base.parent.path).expanduser().resolve()
        runtime_contract = _mapping(v4_contract.runtime, "v4 runtime contract")
    except (AttributeError, KeyError, TypeError) as error:
        raise FoldedBlankRecoveryExecutionError(
            "Authenticated v4 roots do not expose the canonical runtime chain."
        ) from error
    return base, source_guided, source_path, runtime_contract


def _require_compiled_cache_hit(
    contract: Any,
    cache_report: Mapping[str, Any] | None,
) -> None:
    blank = contract.raw["compiled_blank"]
    if (
        not isinstance(cache_report, Mapping)
        or cache_report.get("disposition") != "hit"
        or Path(str(cache_report.get("path", ""))).expanduser().resolve()
        != Path(blank["cache_path"]).expanduser().resolve()
        or cache_report.get("receipt_sha256") != blank["receipt_sha256"]
        or cache_report.get("required_hit") is not True
        or cache_report.get("verified_before_model_access") is not True
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Production runtime did not use the exact authenticated compiled blank hit."
        )


def _restore_v4_winner(
    *,
    contract: Any,
    runtime: Any,
    transaction: Any,
    source_guided: Any,
) -> dict[str, Any]:
    """Restore v2 -> recovered v3 -> retained v4 with live source rebinding."""

    inherited_install = transaction.installation_report()
    source_restore = runtime.restore_source_v2(transaction, source_guided)
    guided_install = runtime.install_guided_transaction(transaction, source_guided)
    source_v2_state = publication.capture_target_native_factor_state(transaction)
    live_qkvo_binding = publication.authenticate_guided_factor_sources(
        transaction,
        runtime._guided_tranche,
        source_v2_state,
    )
    runtime_v3_payload = publication.rebind_recovered_v3_factor_payload(
        contract.v4.contract.source.factor_payload,
        live_qkvo_binding,
    )
    set_qkvo_stage = getattr(runtime._guided_tranche, "set_training_stage", None)
    if not callable(set_qkvo_stage):
        raise FoldedBlankRecoveryExecutionError(
            "Guided QKVO composite does not expose its authenticated joint stage."
        )
    set_qkvo_stage("joint")
    restored_v3 = publication.restore_target_native_factor_payload(
        transaction,
        runtime_v3_payload,
        guided_tranche=runtime._guided_tranche,
        expected_guided_source_binding=live_qkvo_binding,
    )
    if (
        restored_v3.schema != publication.BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        or restored_v3.factor_state_sha256 != runtime_v3_payload["factor_state_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Authenticated recovered-v3 restore changed its runtime factor state."
        )
    source_v3_state = publication.capture_target_native_factor_state(
        transaction,
        guided_tranche=runtime._guided_tranche,
        guided_source_binding=live_qkvo_binding,
    )
    if source_v3_state.factor_state_sha256 != restored_v3.factor_state_sha256:
        raise FoldedBlankRecoveryExecutionError(
            "Live recovered-v3 factors differ from their restored state."
        )

    winner_binding = _WinnerRuntimeBinding(
        tournament=contract.v4.contract,
        winner=contract.v4.winner,
    )
    guided_gud_install = runtime.install_guided_gud_transaction(
        transaction,
        winner_binding,
    )
    live_gud_binding = publication.authenticate_guided_gud_factor_sources(
        transaction,
        runtime._guided_tranche,
        runtime._guided_gud_tranche,
        source_v3_state,
    )
    runtime_v4_payload = publication.rebind_recovered_v4_factor_payload(
        contract.v4.factor_payload,
        live_qkvo_binding,
        live_gud_binding,
    )
    restored_v4 = publication.restore_target_native_factor_payload(
        transaction,
        runtime_v4_payload,
        guided_tranche=runtime._guided_tranche,
        expected_guided_source_binding=live_qkvo_binding,
        guided_gud_tranche=runtime._guided_gud_tranche,
        expected_guided_gud_source_binding=live_gud_binding,
    )
    portable_sha = publication.portable_v4_factor_state_sha256(
        runtime_v4_payload,
        live_qkvo_binding,
        live_gud_binding,
    )
    winner = contract.raw["v4_candidate"]["winner"]
    if (
        restored_v4.schema != publication.BEHAVIOR_FACTOR_PACK_V4_SCHEMA
        or restored_v4.factor_state_sha256 != runtime_v4_payload["factor_state_sha256"]
        or restored_v4.factor_state_sha256 != winner["runtime_factor_state_sha256"]
        or portable_sha != winner["portable_factor_state_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Authenticated retained-v4 restore changed runtime or portable identity."
        )
    handoff = runtime.handoff_v4_for_publication(transaction, winner_binding)
    if (
        not isinstance(handoff, Mapping)
        or handoff.get("handoff_complete") is not True
        or handoff.get("transaction_qkvo_handle_is_guided_composite") is not True
        or handoff.get("transaction_gud_handle_is_guided_composite") is not True
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Guided QKVO/GUD composite handoff was not exact."
        )
    return {
        "inherited_installation": _json_copy(inherited_install),
        "source_v2_restore": _json_copy(source_restore),
        "guided_qkvo_installation": _json_copy(guided_install),
        "guided_gud_installation": _json_copy(guided_gud_install),
        "runtime_rebound_v3_factor_state_sha256": runtime_v3_payload[
            "factor_state_sha256"
        ],
        "runtime_rebound_v4_factor_state_sha256": runtime_v4_payload[
            "factor_state_sha256"
        ],
        "portable_v4_factor_state_sha256": portable_sha,
        "handoff": _json_copy(handoff),
    }


def _rollback_transaction_exactly(
    runtime: Any,
    transaction: Any,
    model: Any,
    expected_sha256: str,
) -> None:
    errors: list[BaseException] = []
    for action in (
        lambda: transaction.restore_prepublication_native(),
        lambda: (
            transaction.rollback()
            if bool(getattr(transaction, "is_live", False))
            else None
        ),
    ):
        try:
            action()
        except BaseException as error:
            errors.append(error)
    try:
        if runtime.state_sha256(model) != expected_sha256:
            raise FoldedBlankRecoveryExecutionError(
                "Transaction cleanup did not restore exact compiled blank bytes."
            )
    except BaseException as error:
        errors.append(error)
    if errors:
        primary = errors[0]
        for secondary in errors[1:]:
            primary.add_note(f"Additional rollback failure: {secondary!r}")
        raise primary


def _production_materialize_initializations(
    contract: Any,
    output_root: Path,
    *,
    folded_artifact_hooks: recovery_ab.FoldedArtifactVerificationHooks,
    runtime: Any | None = None,
) -> MaterializationProduct:
    """Use the existing v4 runtime and bridge to materialize one folded init."""

    if type(folded_artifact_hooks) is not recovery_ab.FoldedArtifactVerificationHooks:
        raise FoldedBlankRecoveryExecutionError(
            "Production folded materialization requires explicit canonical hooks."
        )
    base, source_guided, source_model_path, runtime_contract = (
        _production_runtime_inputs(contract)
    )
    try:
        cuda_preflight = (
            tournament.v3_tournament.validate_candidate_tournament_cuda_preflight(
                contract.v4.contract.source.tournament_contract,
                dtype=str(runtime_contract["dtype"]),
                parent_device_map=str(runtime_contract["parent_device_map"]),
                target_device_map=str(runtime_contract["target_device_map"]),
            )
        )
    except Exception as error:
        raise FoldedBlankRecoveryExecutionError(
            "V4 production CUDA/runtime preflight failed."
        ) from error
    runner = (
        publication.ProductionGuidedGUDWinnerPublicationRuntime(
            dtype=str(runtime_contract["dtype"]),
            parent_device_map=str(runtime_contract["parent_device_map"]),
            target_device_map=str(runtime_contract["target_device_map"]),
        )
        if runtime is None
        else runtime
    )
    folded_root = (output_root / FOLDED_INIT_DIRECTORY_NAME).resolve()
    parent: Any = None
    target: Any = None
    tokenizer: Any = None
    compiled: Any = None
    transaction: Any = None
    compiled_sha: str | None = None
    published = False
    try:
        prepare_cache = getattr(runner, "prepare_compiler_cache", None)
        if not callable(prepare_cache):
            raise FoldedBlankRecoveryExecutionError(
                "Production runtime cannot authenticate the compiler cache."
            )
        cache_report = prepare_cache(base)
        _require_compiled_cache_hit(contract, cache_report)
        parent, target, tokenizer = runner.load_models(base)
        compiled = runner.compile(parent, target, tokenizer, base)
        compiler_report = stage._compiled_report_dict(compiled)
        compiled_sha = runner.state_sha256(compiled.child_model)
        stage._validate_continuation_compilation(base, compiler_report, compiled_sha)
        finalize = getattr(runner, "finalize_compiled_base", None)
        if callable(finalize):
            compiled = finalize(compiled, base)
        blank = contract.raw["compiled_blank"]
        if (
            runner.state_sha256(compiled.child_model) != compiled_sha
            or compiled_sha != blank["state_dict_sha256"]
            or _canonical_sha256(compiler_report) != blank["compiler_report_sha256"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Loaded compiler-cache model/report differs from the authenticated blank."
            )
        cache_evidence = getattr(compiled, "cache_evidence", None)
        if (
            not isinstance(cache_evidence, Mapping)
            or cache_evidence.get("disposition") != "hit"
            or cache_evidence.get("receipt_sha256") != blank["receipt_sha256"]
            or cache_evidence.get("state_dict_sha256") != blank["state_dict_sha256"]
            or cache_evidence.get("artifact_tree_sha256")
            != blank["artifact_tree_sha256"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Compiled object lost the authenticated blank-cache evidence."
            )

        transaction = runner.install_transaction(compiled)
        restore_trace = _restore_v4_winner(
            contract=contract,
            runtime=runner,
            transaction=transaction,
            source_guided=source_guided,
        )
        native_sha = _sha256(
            transaction.native_child_state_sha256,
            "transaction native child state SHA",
        )
        live_factor_sha = _sha256(
            transaction._live_factor_state_sha256(),
            "transaction live factor state SHA",
        )
        experiment_contract = build_experimental_recovery_contract(
            contract,
            native_child_state_sha256=native_sha,
            live_factor_state_sha256=live_factor_sha,
            restored_runtime_factor_state_sha256=restore_trace[
                "runtime_rebound_v4_factor_state_sha256"
            ],
        )
        bridge_receipt = transaction.experimental_recovery_receipt(
            experiment_contract=experiment_contract
        )
        if (
            bridge_receipt.live_factor_state_sha256 != live_factor_sha
            or bridge_receipt.native_child_state_sha256 != native_sha
            or bridge_receipt.factor_state_sha256
            != contract.raw["v4_candidate"]["winner"]["portable_factor_state_sha256"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Bridge receipt differs from the restored live v4 transaction."
            )
        expectations = child_artifact.ExperimentalChildArtifactExpectations(
            preregistration_sha256=contract.file_sha256,
            shared_execution_contract_sha256=contract.raw[
                "shared_execution_contract_sha256"
            ],
            config_payload_sha256=blank["config_payload_sha256"],
            state_key_sha256=blank["state_key_sha256"],
            parameter_topology_sha256=blank["parameter_topology_sha256"],
            buffer_topology_sha256=blank["buffer_topology_sha256"],
            exact_unique_parameters=recovery_ab.EXACT_UNIQUE_PARAMETERS,
            exact_logical_parameters=recovery_ab.EXACT_UNIQUE_PARAMETERS,
            arm_id="folded",
        )
        artifact_receipt = child_artifact.publish_experimental_child_artifact(
            folded_root,
            model=compiled.child_model,
            tokenizer=tokenizer,
            materialization_context=transaction.experimental_recovery_materialization(
                bridge_receipt,
                experiment_contract=experiment_contract,
            ),
            expectations=expectations,
            model_loader=folded_artifact_hooks.model_loader,
            native_certifier=folded_artifact_hooks.native_certifier,
            runtime_auditor=folded_artifact_hooks.runtime_auditor,
            dense_state_hasher=folded_artifact_hooks.dense_state_hasher,
        )
        published = True
        post_materialization_sha = runner.state_sha256(compiled.child_model)
        if post_materialization_sha != compiled_sha:
            raise FoldedBlankRecoveryExecutionError(
                "Experimental context exit did not restore exact compiled blank bytes."
            )
        bridge_report_path = folded_root / child_artifact.BRIDGE_REPORT_FILENAME
        bridge_report = stage._strict_json(bridge_report_path)
        if not isinstance(bridge_report, Mapping):
            raise FoldedBlankRecoveryExecutionError(
                "Published folded initialization has no bridge report."
            )
        artifact_payload = artifact_receipt.as_dict()
        embedded_bridge_receipt = _mapping(
            bridge_report.get("experimental_recovery_receipt"),
            "embedded bridge receipt",
        )
        folded_initial_sha = _sha256(
            artifact_payload["fresh_reload_state_dict_sha256"],
            "folded initial state SHA",
        )
        common = {
            "schema": recovery_ab.MATERIALIZATION_RECEIPT_SCHEMA,
            "status": "experimental-unaccepted",
            "exact_unique_parameters": recovery_ab.EXACT_UNIQUE_PARAMETERS,
            "config_payload_sha256": blank["config_payload_sha256"],
            "parameter_topology_sha256": blank["parameter_topology_sha256"],
            "buffer_topology_sha256": blank["buffer_topology_sha256"],
            "shared_execution_contract_sha256": contract.raw[
                "shared_execution_contract_sha256"
            ],
            "optimizer_steps": 0,
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
        }
        folded_receipt = _self_hashed_receipt(
            {
                **common,
                "arm_id": "folded",
                "initial_state_sha256": folded_initial_sha,
                "source_portable_factor_state_sha256": contract.raw["v4_candidate"][
                    "winner"
                ]["portable_factor_state_sha256"],
                "source_factor_pack_sha256": contract.raw["v4_candidate"]["winner"][
                    "factor_pack_sha256"
                ],
                "factor_source_binding_sha256": contract.raw["v4_candidate"][
                    "factor_source_binding_sha256"
                ],
                "factor_pack_authenticated_separately": True,
                "materialization_evidence_sha256": embedded_bridge_receipt[
                    "receipt_sha256"
                ],
                "materialization_transaction_sha256": bridge_report[
                    "materialization_transaction_sha256"
                ],
                "artifact_publication_root": str(folded_root),
                "artifact_receipt_sha256": artifact_receipt.receipt_sha256,
                "bridge_native_child_state_sha256": bridge_report[
                    "native_child_state_sha256"
                ],
                "bridge_live_factor_state_sha256": bridge_report[
                    "live_factor_state_sha256"
                ],
                "materialized_bridge_state_sha256": bridge_report[
                    "materialized_native_state_sha256"
                ],
            }
        )
        blank_receipt = _self_hashed_receipt(
            {
                **common,
                "arm_id": "blank",
                "initial_state_sha256": blank["state_dict_sha256"],
                "source_portable_factor_state_sha256": None,
                "source_factor_pack_sha256": None,
                "factor_source_binding_sha256": None,
                "factor_pack_authenticated_separately": False,
                "materialization_evidence_sha256": blank["receipt_sha256"],
                "materialization_transaction_sha256": None,
                "artifact_publication_root": None,
                "artifact_receipt_sha256": None,
                "bridge_native_child_state_sha256": None,
                "bridge_live_factor_state_sha256": None,
                "materialized_bridge_state_sha256": None,
            }
        )
        product = MaterializationProduct(
            receipts=MappingProxyType(
                {"folded": folded_receipt, "blank": blank_receipt}
            ),
            folded_publication_root=folded_root,
            folded_candidate_path=folded_root / child_artifact.ARTIFACT_DIRECTORY_NAME,
            blank_candidate_path=Path(blank["cache_path"]).expanduser().resolve(),
            source_model_path=source_model_path,
            trace=MappingProxyType(
                {
                    "cuda_preflight": _json_copy(cuda_preflight),
                    "compiled_child_cache": _json_copy(cache_report),
                    "compiled_blank_state_sha256_before": compiled_sha,
                    "compiled_blank_state_sha256_after": post_materialization_sha,
                    "compiled_blank_bytes_restored_exactly": True,
                    "compiler_report_sha256": _canonical_sha256(compiler_report),
                    "v4_restore": restore_trace,
                    "experimental_recovery_contract": experiment_contract,
                    "experimental_recovery_contract_sha256": _canonical_sha256(
                        experiment_contract
                    ),
                    "bridge_receipt_sha256": embedded_bridge_receipt["receipt_sha256"],
                    "artifact_receipt_sha256": artifact_receipt.receipt_sha256,
                    "transaction_rollback_exact": True,
                    "sealed_evaluation_accessed": False,
                    "publication_authorized": False,
                }
            ),
        )
        _rollback_transaction_exactly(
            runner,
            transaction,
            compiled.child_model,
            compiled_sha,
        )
        transaction = None
        return product
    except BaseException:
        if (
            transaction is not None
            and compiled is not None
            and compiled_sha is not None
        ):
            _rollback_transaction_exactly(
                runner,
                transaction,
                compiled.child_model,
                compiled_sha,
            )
        if published and folded_root.is_dir() and folded_root.parent == output_root:
            shutil.rmtree(folded_root)
        raise
    finally:
        del transaction, compiled, target, tokenizer, parent
        _release_production_memory()


def _runtime_audit(model: Any, artifact_path: Path) -> dict[str, int]:
    """Explicitly audit forbidden runtime owners after a local-only reload."""

    del artifact_path
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise FoldedBlankRecoveryExecutionError(
            "Torch is required for runtime audit."
        ) from error
    modules = tuple(model.modules())
    adapter_count = int(child_artifact._built_in_adapter_or_registry_count(model))
    optimizer_parameter_ids: set[int] = set()
    parent_references = 0
    for module in modules:
        for name, value in vars(module).items():
            lowered = str(name).lower()
            if isinstance(value, torch.optim.Optimizer):
                for group in value.param_groups:
                    optimizer_parameter_ids.update(
                        id(param) for param in group["params"]
                    )
            if (
                any(
                    marker in lowered
                    for marker in (
                        "parent_model",
                        "source_model",
                        "teacher_model",
                        "donor_model",
                    )
                )
                and value is not None
            ):
                parent_references += 1
    sidecar_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if any(
            marker in name.casefold()
            for marker in ("adapter", "factor", "sidecar", "teacher", "donor")
        )
    }
    return {
        "adapter_or_registry_module_count": adapter_count,
        "optimizer_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if id(parameter) in optimizer_parameter_ids
        ),
        "source_sized_sidecar_parameter_count": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if id(parameter) in sidecar_ids
        ),
        "parent_runtime_reference_count": int(parent_references),
    }


def build_production_folded_artifact_hooks(
    contract: Any,
) -> recovery_ab.FoldedArtifactVerificationHooks:
    """Build explicit CPU, local-only, trust-disabled reload/certification hooks."""

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - production dependency
        raise FoldedBlankRecoveryExecutionError(
            "Torch and Transformers are required for production artifact hooks."
        ) from error
    _base, _source, _source_path, runtime_contract = _production_runtime_inputs(
        contract
    )
    dtype_name = str(runtime_contract["dtype"])
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise FoldedBlankRecoveryExecutionError(
            f"Unsupported production artifact dtype {dtype_name!r}."
        )

    def model_loader(path: Path) -> Any:
        return AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
            device_map="cpu",
            attn_implementation=str(runtime_contract["attention_implementation"]),
        ).eval()

    def native_certifier(model: Any) -> Mapping[str, Any]:
        certificate = stage.certify_native_dense_gqa_target(
            model,
            exact_unique_parameters=recovery_ab.EXACT_UNIQUE_PARAMETERS,
            stage=stage.default_qwen4b_progressive_target_plan().stages[0],
            require_dense_swiglu=True,
        )
        return certificate.as_dict()

    return recovery_ab.FoldedArtifactVerificationHooks(
        model_loader=model_loader,
        native_certifier=native_certifier,
        runtime_auditor=_runtime_audit,
        dense_state_hasher=progressive_bridge._dense_model_state_sha256,
    )


def _authenticate_clean_artifact(
    path: Path,
    *,
    folded_artifact_hooks: recovery_ab.FoldedArtifactVerificationHooks,
    expected_state_sha256: str | None,
) -> dict[str, Any]:
    """Fresh-reload a report-free standalone child and reject every sidecar."""

    root = path.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise FoldedBlankRecoveryExecutionError(
            "Clean child authentication requires one existing real directory."
        )
    if (root / RECOVERY_REPORT_FILENAME).exists():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery report remains inside the candidate model directory."
        )
    tree_before = child_artifact.artifact_tree(root)
    model = folded_artifact_hooks.model_loader(root)
    try:
        fingerprint = model_state_fingerprint(model)
        if (
            fingerprint.get("unique_parameter_count")
            != recovery_ab.EXACT_UNIQUE_PARAMETERS
            or fingerprint.get("logical_parameter_count")
            != recovery_ab.EXACT_UNIQUE_PARAMETERS
            or fingerprint.get("all_state_tensors_finite") is not True
            or (
                expected_state_sha256 is not None
                and fingerprint.get("state_dict_sha256") != expected_state_sha256
            )
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Freshly loaded clean child state/count differs from expectation."
            )
        expected_keys = tuple(sorted(str(key) for key in model.state_dict()))
        config_sha = child_artifact._validate_hf_artifact_files(
            root,
            expected_state_keys=expected_keys,
            expected_state_key_sha256=None,
        )
        certificate_value = folded_artifact_hooks.native_certifier(model)
        certificate = (
            certificate_value.as_dict()
            if callable(getattr(certificate_value, "as_dict", None))
            else certificate_value
        )
        audit = folded_artifact_hooks.runtime_auditor(model, root)
        if (
            not isinstance(certificate, Mapping)
            or not isinstance(audit, Mapping)
            or set(audit) != _AUDIT_FIELDS
            or any(
                isinstance(audit[name], bool)
                or not isinstance(audit[name], int)
                or audit[name] != 0
                for name in _AUDIT_FIELDS
            )
            or child_artifact._built_in_adapter_or_registry_count(model) != 0
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Clean child native/runtime audit found a forbidden owner."
            )
        post_fingerprint = model_state_fingerprint(model)
        if post_fingerprint != fingerprint:
            raise FoldedBlankRecoveryExecutionError(
                "Clean child authenticator mutated model state."
            )
    finally:
        del model
        _release_production_memory()
    tree_after = child_artifact.artifact_tree(root)
    if tree_after != tree_before:
        raise FoldedBlankRecoveryExecutionError(
            "Clean child reload or audit mutated artifact bytes."
        )
    return {
        "path": str(root),
        "state_fingerprint": _json_copy(fingerprint),
        "state_fingerprint_sha256": _canonical_sha256(fingerprint),
        "native_certificate_sha256": _canonical_sha256(certificate),
        "runtime_audit": _json_copy(audit),
        "config_payload_sha256": config_sha,
        "artifact_tree": _json_copy(tree_after),
        "report_detached_before_reload": True,
        "strict_model_tokenizer_allowlist_verified": True,
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _authenticate_source_model(
    contract: Any,
    *,
    source_model_path: Path,
    expected_fingerprint: Mapping[str, Any],
    expected_source_tokenizer_fingerprint: Mapping[str, Any],
    expected_candidate_tokenizer_fingerprint: Mapping[str, Any],
    candidate_model_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Fresh-load donor/tokenizers and bind replay to compiler lineage."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - production dependency
        raise FoldedBlankRecoveryExecutionError(
            "Torch and Transformers are required for source authentication."
        ) from error

    source_input = source_model_path.expanduser()
    if source_input.is_symlink():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery source root cannot itself be a symbolic link."
        )
    try:
        source_root = source_input.resolve(strict=True)
    except FileNotFoundError as error:
        raise FoldedBlankRecoveryExecutionError(
            "Recovery source model is absent."
        ) from error
    if not source_root.is_dir():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery source model must be one local directory."
        )

    blank = _mapping(contract.raw["compiled_blank"], "compiled blank")
    cache_root = Path(str(blank["cache_path"])).expanduser().resolve()
    receipt_path = cache_root / CACHE_RECEIPT_FILENAME
    receipt_raw = stage._strict_json(receipt_path)
    if not isinstance(receipt_raw, Mapping):
        raise FoldedBlankRecoveryExecutionError(
            "Compiled blank cache receipt is malformed."
        )
    try:
        receipt = CompiledChildCacheReceipt.from_dict(receipt_raw)
        identity = CompilerIdentity.from_dict(
            _mapping(receipt.as_dict()["compiler_identity"], "compiler identity")
        )
    except Exception as error:
        raise FoldedBlankRecoveryExecutionError(
            "Compiled blank cache does not expose an authentic compiler identity."
        ) from error
    if (
        receipt.receipt_sha256 != blank["receipt_sha256"]
        or identity.compiler_identity_sha256 != blank["compiler_identity_sha256"]
        or receipt.as_dict().get("compiler_identity_sha256")
        != identity.compiler_identity_sha256
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Compiled blank receipt/identity changed before donor authentication."
        )
    identity_payload = identity.as_dict()
    expected_tree_sha256 = _sha256(
        identity_payload["parent_artifact_tree_sha256"],
        "compiler parent artifact tree SHA",
    )
    tree_before = stage._artifact_tree(source_root)
    if tree_before.get("tree_sha256") != expected_tree_sha256:
        raise FoldedBlankRecoveryExecutionError(
            "Recovery source artifact tree differs from CompilerIdentity parent lineage."
        )

    _base, _source, canonical_source_path, _runtime_contract = (
        _production_runtime_inputs(contract)
    )
    if canonical_source_path.resolve() != source_root:
        raise FoldedBlankRecoveryExecutionError(
            "Materialization and runtime contracts name different source models."
        )
    if set(candidate_model_paths) != set(ARM_IDS):
        raise FoldedBlankRecoveryExecutionError(
            "Source authentication requires exact folded/blank candidate paths."
        )
    candidate_roots: dict[str, Path] = {}
    candidate_trees_before: dict[str, dict[str, Any]] = {}
    for arm_id in ARM_IDS:
        candidate_input = Path(candidate_model_paths[arm_id]).expanduser()
        if candidate_input.is_symlink():
            raise FoldedBlankRecoveryExecutionError(
                f"{arm_id} candidate root cannot itself be a symbolic link."
            )
        try:
            candidate_root = candidate_input.resolve(strict=True)
        except FileNotFoundError as error:
            raise FoldedBlankRecoveryExecutionError(
                f"{arm_id} candidate tokenizer artifact is absent."
            ) from error
        if not candidate_root.is_dir():
            raise FoldedBlankRecoveryExecutionError(
                f"{arm_id} candidate tokenizer artifact is not a directory."
            )
        candidate_roots[arm_id] = candidate_root
        candidate_trees_before[arm_id] = stage._artifact_tree(candidate_root)

    model: Any = None
    source_tokenizer: Any = None
    candidate_tokenizer: Any = None
    candidate_tokenizer_fingerprints: dict[str, dict[str, Any]] = {}
    try:
        source_tokenizer = AutoTokenizer.from_pretrained(
            source_root,
            local_files_only=True,
            trust_remote_code=False,
        )
        actual_source_tokenizer_fingerprint = (
            hf_teacher_token_replay._tokenizer_functional_fingerprint(source_tokenizer)
        )
        if _canonical_bytes(actual_source_tokenizer_fingerprint) != _canonical_bytes(
            expected_source_tokenizer_fingerprint
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Fresh source tokenizer fingerprint differs from teacher replay."
            )
        del source_tokenizer
        source_tokenizer = None

        for arm_id in ARM_IDS:
            candidate_tokenizer = AutoTokenizer.from_pretrained(
                candidate_roots[arm_id],
                local_files_only=True,
                trust_remote_code=False,
            )
            candidate_tokenizer_fingerprint = (
                hf_teacher_token_replay._tokenizer_functional_fingerprint(
                    candidate_tokenizer
                )
            )
            if _canonical_bytes(candidate_tokenizer_fingerprint) != _canonical_bytes(
                expected_candidate_tokenizer_fingerprint
            ):
                raise FoldedBlankRecoveryExecutionError(
                    f"Fresh {arm_id} candidate tokenizer fingerprint differs "
                    "from teacher replay."
                )
            candidate_tokenizer_fingerprints[arm_id] = _json_copy(
                candidate_tokenizer_fingerprint
            )
            del candidate_tokenizer
            candidate_tokenizer = None

        model = AutoModelForCausalLM.from_pretrained(
            source_root,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float16,
            device_map=None,
        ).eval()
        actual_fingerprint = hf_teacher_token_replay._model_functional_fingerprint(
            model
        )
        if _canonical_bytes(actual_fingerprint) != _canonical_bytes(
            expected_fingerprint
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Fresh recovery source fingerprint differs from preregistered teacher replay."
            )
    finally:
        del candidate_tokenizer, source_tokenizer, model
        _release_production_memory()

    tree_after = stage._artifact_tree(source_root)
    if tree_after != tree_before:
        raise FoldedBlankRecoveryExecutionError(
            "Fresh donor load or fingerprinting mutated source artifact bytes."
        )
    candidate_trees_after = {
        arm_id: stage._artifact_tree(candidate_roots[arm_id]) for arm_id in ARM_IDS
    }
    if any(
        candidate_trees_after[arm_id] != candidate_trees_before[arm_id]
        for arm_id in ARM_IDS
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Fresh tokenizer loading mutated a candidate artifact tree."
        )
    return {
        "status": "source-model-authenticated",
        "source_model_path": str(source_root),
        "source_model_fingerprint": _json_copy(actual_fingerprint),
        "source_model_fingerprint_sha256": actual_fingerprint["sha256"],
        "source_tokenizer_fingerprint": _json_copy(actual_source_tokenizer_fingerprint),
        "source_tokenizer_fingerprint_sha256": (
            actual_source_tokenizer_fingerprint["sha256"]
        ),
        "candidate_tokenizer_fingerprints": candidate_tokenizer_fingerprints,
        "candidate_tokenizer_fingerprint_sha256": (
            expected_candidate_tokenizer_fingerprint["sha256"]
        ),
        "source_artifact_tree": _json_copy(tree_after),
        "source_artifact_tree_sha256": tree_after["tree_sha256"],
        "compiler_parent_artifact_tree_sha256": expected_tree_sha256,
        "compiled_blank_compiler_identity_sha256": (identity.compiler_identity_sha256),
        "fresh_model_load_verified": True,
        "source_fingerprint_exactly_matches_teacher_token_replay": True,
        "source_tokenizer_exactly_matches_teacher_token_replay": True,
        "both_candidate_tokenizers_exactly_match_teacher_token_replay": True,
        "source_tree_exactly_matches_compiler_parent_lineage": True,
        "artifact_tree_unchanged_by_authentication": True,
        "candidate_artifact_trees_unchanged_by_authentication": True,
        "local_files_only": True,
        "trust_remote_code": False,
    }


def _release_production_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _reset_production_rng(seeds: Mapping[str, Any]) -> None:
    for name in ("python", "torch", "cuda", "data_order"):
        value = seeds.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FoldedBlankRecoveryExecutionError(
                f"Recovery seed {name!r} is not a nonnegative integer."
            )
    random.seed(int(seeds["python"]))
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production dependency
        raise FoldedBlankRecoveryExecutionError(
            "Torch is required for recovery."
        ) from error
    torch.manual_seed(int(seeds["torch"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seeds["cuda"]))


def _load_json(path: Path) -> Mapping[str, Any]:
    value = stage._strict_json(path)
    if not isinstance(value, Mapping):
        raise FoldedBlankRecoveryExecutionError(f"{path} must contain one JSON object.")
    return value


def production_dependencies() -> ExecutorDependencies:
    return ExecutorDependencies(
        authenticate_contract=recovery_ab.authenticate_recovery_ab_preregistration,
        authenticate_teacher_replay=(
            hf_teacher_token_replay.verify_teacher_token_replay_payload
        ),
        materialize_initializations=_production_materialize_initializations,
        authenticate_materializations=recovery_ab.authenticate_materialization_receipts,
        authenticate_source_model=_authenticate_source_model,
        authenticate_zero_steps=recovery_ab.authenticate_zero_step_receipts,
        zero_step_evaluator=_production_zero_step_evaluator,
        recover=recovery_ab.recover_cross_tokenizer_causal_lm,
        reset_rng=_reset_production_rng,
        authenticate_clean_artifact=_authenticate_clean_artifact,
        load_json=_load_json,
        file_sha256=stage._file_sha256,
        atomic_json=stage._atomic_json,
        release_memory=_release_production_memory,
    )


def _fresh_replay_payload(
    contract: Any, dependencies: ExecutorDependencies
) -> dict[str, Any]:
    binding = contract.raw["replay_bank"]
    path = Path(binding["path"]).expanduser().resolve()
    if dependencies.file_sha256(path) != binding["file_sha256"]:
        raise FoldedBlankRecoveryExecutionError(
            "Frozen replay bank changed after preregistration authentication."
        )
    payload = _json_copy(dependencies.load_json(path))
    if not isinstance(payload, dict):
        raise FoldedBlankRecoveryExecutionError("Frozen replay payload is malformed.")
    replay_body = dict(payload)
    replay_payload_sha256 = replay_body.pop("replay_payload_sha256", None)
    if (
        replay_payload_sha256 != binding["payload_sha256"]
        or _canonical_sha256(replay_body) != binding["payload_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Frozen replay outer payload self-hash changed."
        )
    teacher_raw = _mapping(payload.get("teacher_token_replay"), "teacher token replay")
    try:
        verified_teacher = _json_copy(
            dependencies.authenticate_teacher_replay(teacher_raw)
        )
    except Exception as error:
        raise FoldedBlankRecoveryExecutionError(
            "Teacher-token replay failed fresh strict authentication."
        ) from error
    if not isinstance(verified_teacher, dict):
        raise FoldedBlankRecoveryExecutionError(
            "Teacher-token replay authenticator returned a malformed payload."
        )
    fingerprint = _mapping(
        verified_teacher.get("source_model_fingerprint"),
        "teacher replay source model fingerprint",
    )
    state_fingerprint = _mapping(
        fingerprint.get("model_state_fingerprint"),
        "teacher replay source state fingerprint",
    )
    tokenizer_fingerprints = _mapping(
        verified_teacher.get("tokenizer_fingerprints"),
        "teacher replay tokenizer fingerprints",
    )
    source_tokenizer = _mapping(
        tokenizer_fingerprints.get("source"), "source tokenizer fingerprint"
    )
    candidate_tokenizer = _mapping(
        tokenizer_fingerprints.get("candidate"),
        "candidate tokenizer fingerprint",
    )
    parameters = _mapping(
        verified_teacher.get("parameters"), "teacher replay parameters"
    )
    if (
        verified_teacher.get("schema") != binding["teacher_token_replay_schema"]
        or verified_teacher.get("payload_sha256")
        != binding["teacher_token_replay_payload_sha256"]
        or _canonical_sha256(parameters)
        != binding["teacher_token_replay_parameters_sha256"]
        or source_tokenizer.get("sha256")
        != binding["source_tokenizer_fingerprint_sha256"]
        or candidate_tokenizer.get("sha256")
        != binding["candidate_tokenizer_fingerprint_sha256"]
        or fingerprint.get("sha256") != binding["source_model_fingerprint_sha256"]
        or state_fingerprint.get("state_dict_sha256")
        != binding["source_model_state_dict_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Teacher-token replay differs from its preregistered binding."
        )
    calibration = tuple(verified_teacher.get("calibration_texts", ()))
    teacher = tuple(verified_teacher.get("precomputed_teacher_generated_texts", ()))
    on_policy = verified_teacher.get("on_policy_replay_bank")
    stepwise = verified_teacher.get("stepwise_replay_bank", [])
    behavior_suite = payload.get("behavior_eval_suite")
    if (
        hf_recovery._ordered_recovery_texts_sha256(calibration)
        != binding["calibration_texts_sha256"]
        or hf_recovery._ordered_recovery_texts_sha256(teacher)
        != binding["teacher_generated_texts_sha256"]
        or hf_recovery._on_policy_replay_bank_sha256(
            hf_recovery._freeze_on_policy_replay_bank(on_policy)
        )
        != binding["on_policy_replay_bank_sha256"]
        or hf_recovery._stepwise_replay_bank_sha256(
            hf_recovery._freeze_stepwise_replay_bank(stepwise)
        )
        != binding["stepwise_replay_bank_sha256"]
        or _canonical_sha256(behavior_suite) != binding["behavior_eval_suite_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Frozen replay payload differs from its authenticated binding."
        )
    return {
        "teacher_token_replay": verified_teacher,
        "calibration_texts": list(calibration),
        "precomputed_teacher_generated_texts": list(teacher),
        "on_policy_replay_bank": _json_copy(on_policy),
        "stepwise_replay_bank": _json_copy(stepwise),
        "behavior_eval_suite": _json_copy(behavior_suite),
    }


def _expected_source_model_fingerprint(
    contract: Any,
    replay_payload: Mapping[str, Any],
) -> dict[str, Any]:
    teacher = _mapping(
        replay_payload.get("teacher_token_replay"), "teacher token replay"
    )
    fingerprint = _json_copy(
        _mapping(
            teacher.get("source_model_fingerprint"),
            "teacher replay source model fingerprint",
        )
    )
    if not isinstance(fingerprint, dict):
        raise FoldedBlankRecoveryExecutionError(
            "Teacher replay source fingerprint is malformed."
        )
    claimed = fingerprint.get("sha256")
    body = dict(fingerprint)
    body.pop("sha256", None)
    state = _mapping(
        fingerprint.get("model_state_fingerprint"),
        "teacher replay source state fingerprint",
    )
    state_body = dict(state)
    state_claimed = state_body.pop("sha256", None)
    binding = contract.raw["replay_bank"]
    if (
        claimed != _canonical_sha256(body)
        or state_claimed != _canonical_sha256(state_body)
        or claimed != binding["source_model_fingerprint_sha256"]
        or state.get("state_dict_sha256") != binding["source_model_state_dict_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Preregistered source model fingerprint self-hash does not verify."
        )
    return fingerprint


def _expected_tokenizer_fingerprints(
    contract: Any,
    replay_payload: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    teacher = _mapping(
        replay_payload.get("teacher_token_replay"), "teacher token replay"
    )
    tokenizer_fingerprints = _mapping(
        teacher.get("tokenizer_fingerprints"),
        "teacher replay tokenizer fingerprints",
    )
    binding = contract.raw["replay_bank"]
    expected_hashes = {
        "source": binding["source_tokenizer_fingerprint_sha256"],
        "candidate": binding["candidate_tokenizer_fingerprint_sha256"],
    }
    result: dict[str, dict[str, Any]] = {}
    for name in ("source", "candidate"):
        fingerprint = _json_copy(
            _mapping(
                tokenizer_fingerprints.get(name),
                f"teacher replay {name} tokenizer fingerprint",
            )
        )
        if not isinstance(fingerprint, dict):
            raise FoldedBlankRecoveryExecutionError(
                f"Teacher replay {name} tokenizer fingerprint is malformed."
            )
        claimed = fingerprint.get("sha256")
        body = dict(fingerprint)
        body.pop("sha256", None)
        if claimed != _canonical_sha256(body) or claimed != expected_hashes[name]:
            raise FoldedBlankRecoveryExecutionError(
                f"Preregistered {name} tokenizer fingerprint self-hash does not verify."
            )
        result[name] = fingerprint
    return result


def _recovery_static_inputs(
    contract: Any,
    replay_payload: Mapping[str, Any],
    *,
    source_model_path: Path,
) -> dict[str, Any]:
    plan = contract.raw["recovery_plan"]
    kwargs = _json_copy(plan["recovery_kwargs"])
    if (
        kwargs.get("local_files_only") is not True
        or kwargs.get("trust_remote_code", False) is not False
        or kwargs.get("adamw_foreach") is not False
        or kwargs.get("save_best_loss", False) is not False
        or kwargs.get("save_best_behavior", False) is not False
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Recovery invocation lost local-only/trust-disabled deterministic pins."
        )
    result = {
        **kwargs,
        "source_model": str(source_model_path),
        "calibration_texts": _json_copy(replay_payload["calibration_texts"]),
        "behavior_eval_suite": _json_copy(replay_payload["behavior_eval_suite"]),
        "precomputed_teacher_generated_texts": _json_copy(
            replay_payload["precomputed_teacher_generated_texts"]
        ),
        "on_policy_replay_bank": _json_copy(replay_payload["on_policy_replay_bank"]),
        "stepwise_replay_bank": _json_copy(
            replay_payload.get("stepwise_replay_bank", [])
        ),
    }
    return result


def _detach_recovery_report(
    *,
    recover: Callable[..., Mapping[str, Any]],
    invocation: Mapping[str, Any],
    evidence_path: Path,
    dependencies: ExecutorDependencies,
) -> DetachedRecoveryReport:
    output = Path(str(invocation["output_dir"])).expanduser().resolve()
    if output.exists() or evidence_path.exists():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery output and evidence paths must both start absent."
        )
    returned = recover(**dict(invocation))
    report_path = output / RECOVERY_REPORT_FILENAME
    if not report_path.is_file():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery did not write its canonical report inside output_dir."
        )
    on_disk = dependencies.load_json(report_path)
    if _canonical_bytes(on_disk) != _canonical_bytes(returned):
        raise FoldedBlankRecoveryExecutionError(
            "Returned recovery report differs from the written report."
        )
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(report_path, evidence_path)
    if report_path.exists() or not evidence_path.is_file():
        raise FoldedBlankRecoveryExecutionError(
            "Recovery report was not detached from the model artifact."
        )
    return DetachedRecoveryReport(
        report=MappingProxyType(dict(_json_copy(on_disk))),
        path=evidence_path,
        file_sha256=dependencies.file_sha256(evidence_path),
    )


def _validate_observer_report(
    report: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    expected_steps: Sequence[int],
) -> None:
    behavior_history = report.get("behavior_history")
    if (
        report.get("behavior_eval_checkpoint_steps") != list(expected_steps)
        or report.get("behavior_checkpoint_observer_calls") != len(expected_steps)
        or report.get("behavior_checkpoint_observer_read_only_payload") is not True
        or not isinstance(behavior_history, list)
        or len(behavior_history) != len(events)
        or [event["step"] for event in behavior_history] != list(expected_steps)
        or any(
            _canonical_bytes(event["behavior"]) != _canonical_bytes(history)
            for event, history in zip(events, behavior_history, strict=True)
        )
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Recovery report differs from immutable behavior observer events."
        )


def _validate_stepwise_replay_report(contract: Any, report: Mapping[str, Any]) -> None:
    expected = contract.raw["replay_bank"]
    plan = contract.raw["recovery_plan"]
    required = bool(plan["stepwise_replay_bank_required"])
    if report.get("stepwise_dynamic_candidate_generation") is not False:
        raise FoldedBlankRecoveryExecutionError(
            "Stepwise donor replay attempted dynamic candidate generation."
        )
    if (
        report.get("stepwise_replay_bank_supplied") is not True
        or report.get("stepwise_replay_bank_schema")
        != expected["stepwise_replay_bank_schema"]
        or report.get("stepwise_replay_bank_event_count")
        != expected["stepwise_event_count"]
        or report.get("stepwise_replay_bank_sha256")
        != expected["stepwise_replay_bank_sha256"]
        or report.get("stepwise_scheduled_steps") != plan["stepwise_event_steps"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Stepwise replay report differs from the frozen contract."
        )
    consumption = report.get("stepwise_replay_consumption")
    preflight = report.get("stepwise_replay_preflight")
    if not required:
        if consumption not in ([], None) or preflight not in (None, {}):
            raise FoldedBlankRecoveryExecutionError(
                "Disabled stepwise replay emitted preflight or consumption evidence."
            )
        return
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("status") != "validated"
        or preflight.get("event_count") != expected["stepwise_event_count"]
        or preflight.get("replay_bank_sha256")
        != expected["stepwise_replay_bank_sha256"]
        or not isinstance(preflight.get("events"), list)
        or not isinstance(consumption, list)
        or len(consumption) != expected["stepwise_event_count"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Stepwise replay preflight/consumption is incomplete."
        )
    preflight_body = dict(preflight)
    claimed = preflight_body.pop("sha256", None)
    if hf_recovery._stepwise_replay_preflight_sha256(preflight_body) != claimed:
        raise FoldedBlankRecoveryExecutionError(
            "Stepwise replay preflight self-hash changed."
        )
    for index, expected_event in enumerate(expected["stepwise_events"]):
        consumed = consumption[index]
        checked = preflight["events"][index]
        if (
            consumed.get("event_index") != index
            or consumed.get("step") != expected_event["optimizer_step"]
            or consumed.get("event_sha256") != expected_event["event_sha256"]
            or consumed.get("kd_position_count")
            != expected_event["continuation_token_count"]
            or checked.get("event_index") != index
            or checked.get("optimizer_step") != expected_event["optimizer_step"]
            or checked.get("event_sha256") != expected_event["event_sha256"]
            or checked.get("kd_position_count")
            != expected_event["continuation_token_count"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Stepwise replay event order, hash, or KD consumption changed."
            )


def _validate_on_policy_replay_report(contract: Any, report: Mapping[str, Any]) -> None:
    expected = contract.raw["replay_bank"]
    plan = contract.raw["recovery_plan"]
    if report.get("on_policy_dynamic_candidate_generation") is not False:
        raise FoldedBlankRecoveryExecutionError(
            "On-policy recovery attempted dynamic candidate generation."
        )
    consumption = report.get("on_policy_replay_consumption")
    if (
        report.get("on_policy_replay_bank_supplied") is not True
        or report.get("on_policy_replay_bank_event_count")
        != expected["on_policy_event_count"]
        or report.get("on_policy_replay_bank_sha256")
        != expected["on_policy_replay_bank_sha256"]
        or report.get("on_policy_scheduled_steps") != plan["on_policy_event_steps"]
        or not isinstance(consumption, list)
        or len(consumption) != expected["on_policy_event_count"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "On-policy replay report/consumption differs from the frozen contract."
        )
    for index, expected_event in enumerate(expected["on_policy_events"]):
        consumed = consumption[index]
        if (
            consumed.get("event_index") != index
            or consumed.get("step") != plan["on_policy_event_steps"][index]
            or consumed.get("text_count") != expected_event["text_count"]
            or consumed.get("ordered_text_sha256")
            != expected_event["ordered_text_sha256"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "On-policy replay event order, text count, or hash changed."
            )


def _validate_training_report(
    contract: Any,
    report: Mapping[str, Any],
    *,
    invocation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
) -> None:
    plan = contract.raw["recovery_plan"]
    replay = contract.raw["replay_bank"]
    checkpoints = plan["checkpoint_steps"]
    if (
        report.get("operation") != "recover_cross_tokenizer_causal_lm"
        or report.get("source_model") != invocation["source_model"]
        or report.get("candidate_model") != invocation["candidate_model"]
        or Path(str(report.get("output_dir", ""))).expanduser().resolve()
        != Path(str(invocation["output_dir"])).expanduser().resolve()
        or report.get("steps") != plan["recovery_kwargs"]["steps"]
        or report.get("local_files_only") is not True
        or invocation.get("trust_remote_code", False) is not False
        or report.get("adamw_foreach") is not False
        or report.get("adamw_foreach_explicit") is not True
        or report.get("save_best_loss") is not False
        or report.get("save_best_behavior") is not False
        or report.get("param_count") != recovery_ab.EXACT_UNIQUE_PARAMETERS
        or report.get("calibration_text_count") != replay["calibration_text_count"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Recovery report violates model/output/local-only/shared-plan bindings."
        )
    for name, expected in plan["recovery_kwargs"].items():
        if name in report and report[name] != expected:
            raise FoldedBlankRecoveryExecutionError(
                f"Recovery report changed preregistered kwarg {name!r}."
            )
    if float(plan["recovery_kwargs"].get("teacher_generate_lm_weight", 0.0)) > 0.0:
        if (
            report.get("teacher_generated_text_source") != "precomputed"
            or report.get("teacher_generated_text_count")
            != replay["teacher_generated_text_count"]
            or report.get("teacher_generated_ordered_text_sha256")
            != replay["teacher_generated_texts_sha256"]
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Recovery did not consume the frozen teacher-generated texts."
            )
    _validate_observer_report(report, events, checkpoints)
    _validate_stepwise_replay_report(contract, report)
    _validate_on_policy_replay_report(contract, report)


def _validate_zero_report(
    contract: Any,
    report: Mapping[str, Any],
    *,
    invocation: Mapping[str, Any],
    event: Mapping[str, Any],
) -> None:
    if (
        report.get("operation") != "recover_cross_tokenizer_causal_lm"
        or report.get("steps") != 0
        or report.get("losses") != []
        or report.get("local_files_only") is not True
        or invocation.get("trust_remote_code", False) is not False
        or report.get("param_count") != recovery_ab.EXACT_UNIQUE_PARAMETERS
        or report.get("stepwise_scheduled_steps") != []
        or report.get("stepwise_replay_consumption") != []
        or report.get("on_policy_scheduled_steps") != []
        or report.get("on_policy_replay_consumption") != []
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Zero-step evaluator performed work or lost local-only bindings."
        )
    _validate_observer_report(report, [event], [0])


def _production_zero_step_evaluator(
    contract: Any,
    *,
    arm_id: str,
    candidate_path: Path,
    source_model_path: Path,
    replay_payload: Mapping[str, Any],
    work_root: Path,
    folded_artifact_hooks: recovery_ab.FoldedArtifactVerificationHooks,
    dependencies: ExecutorDependencies,
) -> ZeroStepEvaluation:
    """Run recovery's read-only step-zero path as a separate injected phase."""

    observer = FrozenBehaviorObserver(expected_steps=[0])
    static = _recovery_static_inputs(
        contract,
        replay_payload,
        source_model_path=source_model_path,
    )
    invocation = {
        **static,
        "steps": 0,
        "behavior_eval_checkpoint_steps": [0],
        "on_policy_replay_bank": [],
        "stepwise_replay_bank": [],
        "candidate_model": str(candidate_path),
        "output_dir": str(work_root / f"{arm_id}_zero_child"),
        "behavior_checkpoint_observer": observer,
    }
    detached = _detach_recovery_report(
        recover=dependencies.recover,
        invocation=invocation,
        evidence_path=work_root / f"{arm_id}_zero_report.json",
        dependencies=dependencies,
    )
    events = observer.finalize()
    event = events[0]
    _validate_zero_report(
        contract,
        detached.report,
        invocation=invocation,
        event=event,
    )
    expected_initial = contract.raw["compiled_blank"]["state_dict_sha256"]
    if arm_id == "folded":
        expected_initial = None
    clean = dependencies.authenticate_clean_artifact(
        Path(invocation["output_dir"]),
        folded_artifact_hooks=folded_artifact_hooks,
        expected_state_sha256=expected_initial,
    )
    return ZeroStepEvaluation(
        event=event,
        event_sha256=_canonical_sha256(event),
        behavior=MappingProxyType(dict(_json_copy(event["behavior"]))),
        report=detached.report,
        report_file_sha256=detached.file_sha256,
        clean_artifact=MappingProxyType(dict(_json_copy(clean))),
    )


def _zero_step_receipt(
    contract: Any,
    materialization: Mapping[str, Any],
    arm_id: str,
    evaluation: ZeroStepEvaluation,
) -> dict[str, Any]:
    metrics = _json_copy(evaluation.behavior)
    return _self_hashed_receipt(
        {
            "schema": recovery_ab.ZERO_STEP_RECEIPT_SCHEMA,
            "status": "experimental-unaccepted",
            "arm_id": arm_id,
            "initial_state_sha256": materialization["receipts"][arm_id][
                "initial_state_sha256"
            ],
            "shared_execution_contract_sha256": contract.raw[
                "shared_execution_contract_sha256"
            ],
            "replay_bank_payload_sha256": contract.raw["replay_bank"]["payload_sha256"],
            "checkpoint_step": 0,
            "optimizer_steps": 0,
            "metrics": metrics,
            "metrics_sha256": _canonical_sha256(metrics),
            "evaluator_implementation_sha256": contract.raw["implementation"][
                "behavior_evaluator_implementation_sha256"
            ],
            "sealed_data_accessed": False,
            "publication_authorized": False,
            "accepted_artifact": False,
        }
    )


def _comparison_rows(arms: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    folded_events = arms["folded"]["behavior_events"]
    blank_events = arms["blank"]["behavior_events"]
    if len(folded_events) != len(blank_events):
        raise FoldedBlankRecoveryExecutionError(
            "A/B behavior event counts differ after symmetric recovery."
        )
    rows: list[dict[str, Any]] = []
    for folded, blank in zip(folded_events, blank_events, strict=True):
        folded_behavior = folded["event"]["behavior"]
        blank_behavior = blank["event"]["behavior"]
        if folded["event"]["step"] != blank["event"]["step"]:
            raise FoldedBlankRecoveryExecutionError(
                "A/B checkpoint steps differ after symmetric recovery."
            )
        rows.append(
            {
                "step": folded["event"]["step"],
                "folded_event_sha256": folded["event_sha256"],
                "blank_event_sha256": blank["event_sha256"],
                "folded_pass_rate": folded_behavior.get("pass_rate"),
                "blank_pass_rate": blank_behavior.get("pass_rate"),
                "pass_rate_delta_folded_minus_blank": float(
                    folded_behavior.get("pass_rate", 0.0)
                )
                - float(blank_behavior.get("pass_rate", 0.0)),
                "folded_generation_health_rate": folded_behavior.get(
                    "generation_health_rate"
                ),
                "blank_generation_health_rate": blank_behavior.get(
                    "generation_health_rate"
                ),
                "generation_health_rate_delta_folded_minus_blank": float(
                    folded_behavior.get("generation_health_rate", 0.0)
                )
                - float(blank_behavior.get("generation_health_rate", 0.0)),
            }
        )
    return rows


def _trajectory_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise FoldedBlankRecoveryExecutionError(
            "A/B trajectory summary requires at least one checkpoint."
        )
    steps = [int(row["step"]) for row in rows]
    if steps != sorted(set(steps)) or steps[0] != 0:
        raise FoldedBlankRecoveryExecutionError(
            "A/B trajectory checkpoints must be unique, ordered, and start at zero."
        )

    def normalized_auc(field: str) -> float:
        if len(rows) == 1 or steps[-1] == steps[0]:
            return float(rows[0][field])
        area = 0.0
        for index in range(1, len(rows)):
            width = steps[index] - steps[index - 1]
            area += (
                width
                * (float(rows[index - 1][field]) + float(rows[index][field]))
                / 2.0
            )
        return float(area / (steps[-1] - steps[0]))

    folded_behavior_auc = normalized_auc("folded_pass_rate")
    blank_behavior_auc = normalized_auc("blank_pass_rate")
    folded_generation_auc = normalized_auc("folded_generation_health_rate")
    blank_generation_auc = normalized_auc("blank_generation_health_rate")
    post_zero = [row for row in rows if int(row["step"]) > 0]
    behavior_deltas = [
        float(row["pass_rate_delta_folded_minus_blank"]) for row in post_zero
    ]
    generation_deltas = [
        float(row["generation_health_rate_delta_folded_minus_blank"])
        for row in post_zero
    ]
    final = rows[-1]
    return {
        "checkpoint_count": len(rows),
        "optimizer_step_horizon": steps[-1],
        "folded_behavior_normalized_auc": folded_behavior_auc,
        "blank_behavior_normalized_auc": blank_behavior_auc,
        "behavior_auc_delta_folded_minus_blank": (
            folded_behavior_auc - blank_behavior_auc
        ),
        "folded_generation_normalized_auc": folded_generation_auc,
        "blank_generation_normalized_auc": blank_generation_auc,
        "generation_auc_delta_folded_minus_blank": (
            folded_generation_auc - blank_generation_auc
        ),
        "post_zero_folded_behavior_wins": sum(delta > 0.0 for delta in behavior_deltas),
        "post_zero_blank_behavior_wins": sum(delta < 0.0 for delta in behavior_deltas),
        "post_zero_behavior_ties": sum(delta == 0.0 for delta in behavior_deltas),
        "post_zero_folded_generation_wins": sum(
            delta > 0.0 for delta in generation_deltas
        ),
        "post_zero_blank_generation_wins": sum(
            delta < 0.0 for delta in generation_deltas
        ),
        "post_zero_generation_ties": sum(delta == 0.0 for delta in generation_deltas),
        "best_behavior_delta_folded_minus_blank": max(
            behavior_deltas,
            default=0.0,
        ),
        "best_generation_delta_folded_minus_blank": max(
            generation_deltas,
            default=0.0,
        ),
        "final_behavior_delta_folded_minus_blank": float(
            final["pass_rate_delta_folded_minus_blank"]
        ),
        "final_generation_delta_folded_minus_blank": float(
            final["generation_health_rate_delta_folded_minus_blank"]
        ),
        "candidate_only_folded_learning_advantage_observed": bool(
            folded_behavior_auc > blank_behavior_auc
            or folded_generation_auc > blank_generation_auc
        ),
        "sealed_capability_claim": False,
    }


def _prune_zero_step_artifacts(zero_root: Path, output_root: Path) -> None:
    """Delete only the four authenticated ephemeral zero-step products."""

    root = zero_root.resolve()
    output = output_root.resolve()
    if root.parent != output or not root.name.startswith(".zero-step-"):
        raise FoldedBlankRecoveryExecutionError(
            "Refusing to prune a zero-step directory outside output_root."
        )
    expected_directories = {f"{arm_id}_zero_child" for arm_id in ARM_IDS}
    expected_files = {f"{arm_id}_zero_report.json" for arm_id in ARM_IDS}
    entries = {path.name: path for path in root.iterdir()}
    if set(entries) != expected_directories | expected_files:
        raise FoldedBlankRecoveryExecutionError(
            "Ephemeral zero-step directory contains unexpected products."
        )
    for name in sorted(expected_directories):
        path = entries[name]
        if path.is_symlink() or not path.is_dir() or path.resolve().parent != root:
            raise FoldedBlankRecoveryExecutionError(
                f"Ephemeral zero-step child {name!r} is not an ordinary directory."
            )
        shutil.rmtree(path)
    for name in sorted(expected_files):
        path = entries[name]
        if path.is_symlink() or not path.is_file() or path.resolve().parent != root:
            raise FoldedBlankRecoveryExecutionError(
                f"Ephemeral zero-step report {name!r} is not an ordinary file."
            )
        path.unlink()
    if any(root.iterdir()):
        raise FoldedBlankRecoveryExecutionError(
            "Ephemeral zero-step products were not fully pruned."
        )


def _validate_retained_layout(output: Path) -> None:
    allowed = {
        FOLDED_INIT_DIRECTORY_NAME,
        FOLDED_FINAL_DIRECTORY_NAME,
        BLANK_FINAL_DIRECTORY_NAME,
        EVIDENCE_DIRECTORY_NAME,
        COMPARISON_REPORT_FILENAME,
    }
    actual = {path.name for path in output.iterdir()}
    if actual != allowed:
        raise FoldedBlankRecoveryExecutionError(
            f"Executor retained unexpected output entries: {sorted(actual - allowed)!r}."
        )
    evidence = output / EVIDENCE_DIRECTORY_NAME
    if {path.name for path in evidence.iterdir()} != {
        "folded_recovery_report.json",
        "blank_recovery_report.json",
    }:
        raise FoldedBlankRecoveryExecutionError(
            "Executor retained unexpected recovery evidence files."
        )


def execute_recovery_ab(
    *,
    preregistration: Path | str,
    preregistration_sha256: str,
    output_root: Path | str,
    folded_artifact_hooks: recovery_ab.FoldedArtifactVerificationHooks
    | Any
    | None = None,
    runtime: Any | None = None,
    dependencies: ExecutorDependencies | None = None,
) -> dict[str, Any]:
    """Materialize and run the symmetric experiment without accepting a child."""

    deps = production_dependencies() if dependencies is None else dependencies
    contract = deps.authenticate_contract(preregistration, preregistration_sha256)
    output = Path(output_root).expanduser().resolve()
    if output.exists() or output.name in {"", ".", ".."}:
        raise FoldedBlankRecoveryExecutionError(
            "output_root must be one absent named directory."
        )
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise FoldedBlankRecoveryExecutionError(
            "output_root parent must be one existing real directory."
        )
    # Authenticate again immediately before any model or GPU access.
    refreshed = deps.authenticate_contract(contract.path, contract.file_sha256)
    if _canonical_bytes(refreshed.raw) != _canonical_bytes(contract.raw):
        raise FoldedBlankRecoveryExecutionError(
            "Preregistration roots changed before production execution."
        )
    contract = refreshed
    hooks = (
        build_production_folded_artifact_hooks(contract)
        if folded_artifact_hooks is None
        else folded_artifact_hooks
    )
    output.mkdir()
    evidence_dir = output / EVIDENCE_DIRECTORY_NAME
    evidence_dir.mkdir()
    replay_payload = _fresh_replay_payload(contract, deps)
    product = deps.materialize_initializations(
        contract,
        output,
        folded_artifact_hooks=hooks,
        runtime=runtime,
    )
    materialization = deps.authenticate_materializations(
        contract,
        product.receipts,
        folded_artifact_hooks=hooks,
    )
    if (
        materialization.get("status") != "materialization-authenticated"
        or materialization.get("optimizer_steps") != 0
        or materialization.get("sealed_data_accessed") is not False
        or materialization.get("publication_authorized") is not False
        or materialization.get("accepted_artifact") is not False
        or materialization.get("folded_artifact_independent_fresh_reload_verified")
        is not True
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Materialization receipts did not independently authenticate."
        )
    folded_binding = materialization["authenticated_materialization_reports"]["folded"]
    if (
        Path(folded_binding["publication_root"]).resolve()
        != product.folded_publication_root.resolve()
        or product.folded_candidate_path
        != product.folded_publication_root / child_artifact.ARTIFACT_DIRECTORY_NAME
        or product.blank_candidate_path.resolve()
        != Path(contract.raw["compiled_blank"]["cache_path"]).resolve()
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Authenticated materialization paths differ from execution paths."
        )

    expected_source_fingerprint = _expected_source_model_fingerprint(
        contract,
        replay_payload,
    )
    expected_tokenizer_fingerprints = _expected_tokenizer_fingerprints(
        contract,
        replay_payload,
    )
    source_authentication = deps.authenticate_source_model(
        contract,
        source_model_path=product.source_model_path,
        expected_fingerprint=expected_source_fingerprint,
        expected_source_tokenizer_fingerprint=expected_tokenizer_fingerprints["source"],
        expected_candidate_tokenizer_fingerprint=expected_tokenizer_fingerprints[
            "candidate"
        ],
        candidate_model_paths={
            "folded": product.folded_candidate_path,
            "blank": product.blank_candidate_path,
        },
    )
    authenticated_candidate_tokenizers = source_authentication.get(
        "candidate_tokenizer_fingerprints"
    )
    if (
        source_authentication.get("status") != "source-model-authenticated"
        or Path(str(source_authentication.get("source_model_path", ""))).resolve()
        != product.source_model_path.resolve()
        or _canonical_bytes(source_authentication.get("source_model_fingerprint"))
        != _canonical_bytes(expected_source_fingerprint)
        or source_authentication.get("source_model_fingerprint_sha256")
        != expected_source_fingerprint["sha256"]
        or _canonical_bytes(source_authentication.get("source_tokenizer_fingerprint"))
        != _canonical_bytes(expected_tokenizer_fingerprints["source"])
        or source_authentication.get("source_tokenizer_fingerprint_sha256")
        != expected_tokenizer_fingerprints["source"]["sha256"]
        or not isinstance(authenticated_candidate_tokenizers, Mapping)
        or set(authenticated_candidate_tokenizers) != set(ARM_IDS)
        or any(
            _canonical_bytes(authenticated_candidate_tokenizers[arm_id])
            != _canonical_bytes(expected_tokenizer_fingerprints["candidate"])
            for arm_id in ARM_IDS
        )
        or source_authentication.get("candidate_tokenizer_fingerprint_sha256")
        != expected_tokenizer_fingerprints["candidate"]["sha256"]
        or source_authentication.get("source_artifact_tree_sha256")
        != source_authentication.get("compiler_parent_artifact_tree_sha256")
        or source_authentication.get("compiled_blank_compiler_identity_sha256")
        != contract.raw["compiled_blank"]["compiler_identity_sha256"]
        or source_authentication.get("fresh_model_load_verified") is not True
        or source_authentication.get(
            "source_fingerprint_exactly_matches_teacher_token_replay"
        )
        is not True
        or source_authentication.get(
            "source_tokenizer_exactly_matches_teacher_token_replay"
        )
        is not True
        or source_authentication.get(
            "both_candidate_tokenizers_exactly_match_teacher_token_replay"
        )
        is not True
        or source_authentication.get(
            "source_tree_exactly_matches_compiler_parent_lineage"
        )
        is not True
        or source_authentication.get("artifact_tree_unchanged_by_authentication")
        is not True
        or source_authentication.get(
            "candidate_artifact_trees_unchanged_by_authentication"
        )
        is not True
        or source_authentication.get("local_files_only") is not True
        or source_authentication.get("trust_remote_code") is not False
    ):
        raise FoldedBlankRecoveryExecutionError(
            "Recovery source did not authenticate against replay and compiler lineage."
        )
    deps.release_memory()

    seeds = contract.raw["recovery_plan"]["seeds"]
    zero_evaluations: dict[str, ZeroStepEvaluation] = {}
    zero_receipts: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(
        prefix=".zero-step-",
        dir=output,
    ) as zero_directory:
        zero_root = Path(zero_directory).resolve()
        for arm_id, candidate_path in (
            ("folded", product.folded_candidate_path),
            ("blank", product.blank_candidate_path),
        ):
            deps.reset_rng(seeds)
            evaluation = deps.zero_step_evaluator(
                contract,
                arm_id=arm_id,
                candidate_path=candidate_path,
                source_model_path=product.source_model_path,
                replay_payload=replay_payload,
                work_root=zero_root,
                folded_artifact_hooks=hooks,
                dependencies=deps,
            )
            expected_initial = materialization["receipts"][arm_id][
                "initial_state_sha256"
            ]
            if (
                evaluation.clean_artifact["state_fingerprint"]["state_dict_sha256"]
                != expected_initial
            ):
                raise FoldedBlankRecoveryExecutionError(
                    f"{arm_id} zero-step output differs from materialized initialization."
                )
            zero_evaluations[arm_id] = evaluation
            zero_receipts[arm_id] = _zero_step_receipt(
                contract,
                materialization,
                arm_id,
                evaluation,
            )
            deps.release_memory()
        zero_step = deps.authenticate_zero_steps(
            contract,
            materialization,
            zero_receipts,
            folded_artifact_hooks=hooks,
        )
        if (
            zero_step.get("status") != "zero-step-authenticated"
            or zero_step.get("training_launched") is not False
            or zero_step.get("optimizer_steps") != 0
            or zero_step.get("sealed_data_accessed") is not False
            or zero_step.get("publication_authorized") is not False
            or zero_step.get("accepted_artifact") is not False
        ):
            raise FoldedBlankRecoveryExecutionError(
                "Both zero-step receipts were not authenticated before training."
            )
        _prune_zero_step_artifacts(zero_root, output)

        static = _recovery_static_inputs(
            contract,
            replay_payload,
            source_model_path=product.source_model_path,
        )
        static_hash = _canonical_sha256(static)
        checkpoints = contract.raw["recovery_plan"]["checkpoint_steps"]
        arm_reports: dict[str, dict[str, Any]] = {}
        for arm_id, candidate_path, output_name in (
            ("folded", product.folded_candidate_path, FOLDED_FINAL_DIRECTORY_NAME),
            ("blank", product.blank_candidate_path, BLANK_FINAL_DIRECTORY_NAME),
        ):
            observer = FrozenBehaviorObserver(
                expected_steps=checkpoints,
                authenticated_zero_event=zero_evaluations[arm_id].event,
            )
            deps.reset_rng(seeds)
            invocation = {
                **static,
                "candidate_model": str(candidate_path),
                "output_dir": str(output / output_name),
                "behavior_checkpoint_observer": observer,
            }
            detached = _detach_recovery_report(
                recover=deps.recover,
                invocation=invocation,
                evidence_path=evidence_dir / f"{arm_id}_recovery_report.json",
                dependencies=deps,
            )
            events = observer.finalize()
            _validate_training_report(
                contract,
                detached.report,
                invocation=invocation,
                events=events,
            )
            clean = deps.authenticate_clean_artifact(
                Path(invocation["output_dir"]),
                folded_artifact_hooks=hooks,
                expected_state_sha256=None,
            )
            if clean.get("report_detached_before_reload") is not True:
                raise FoldedBlankRecoveryExecutionError(
                    "Final child was authenticated before report detachment."
                )
            arm_reports[arm_id] = {
                "arm_id": arm_id,
                "initial_state_sha256": materialization["receipts"][arm_id][
                    "initial_state_sha256"
                ],
                "zero_step_receipt": zero_receipts[arm_id],
                "zero_step_event_sha256": zero_evaluations[arm_id].event_sha256,
                "zero_step_authenticated_before_training": True,
                "training_step_zero_matches_authenticated_zero": True,
                "recovery_static_inputs_sha256": static_hash,
                "recovery_report_path": str(detached.path),
                "recovery_report_file_sha256": detached.file_sha256,
                "recovery_report_payload_sha256": _canonical_sha256(detached.report),
                "behavior_events": observer.event_records(),
                "behavior_events_sha256": _canonical_sha256(observer.event_records()),
                "final_artifact": _json_copy(clean),
                "optimizer_steps": contract.raw["recovery_plan"]["recovery_kwargs"][
                    "steps"
                ],
                "sealed_data_accessed": False,
                "publication_authorized": False,
                "accepted_artifact": False,
            }
            deps.release_memory()

    if any(output.glob(".zero-step-*")):
        raise FoldedBlankRecoveryExecutionError(
            "Ephemeral zero-step artifacts were not pruned."
        )
    if (
        arm_reports["folded"]["recovery_static_inputs_sha256"]
        != arm_reports["blank"]["recovery_static_inputs_sha256"]
    ):
        raise FoldedBlankRecoveryExecutionError(
            "A/B arms did not receive identical frozen recovery inputs."
        )
    comparison_rows = _comparison_rows(arm_reports)
    trajectory_summary = _trajectory_summary(comparison_rows)
    body = {
        "schema": COMPARISON_SCHEMA,
        "status": "experimental-unaccepted",
        "preregistration": str(contract.path),
        "preregistration_sha256": contract.file_sha256,
        "semantic_contract_sha256": contract.raw["semantic_contract_sha256"],
        "shared_execution_contract_sha256": contract.raw[
            "shared_execution_contract_sha256"
        ],
        "execution": {
            "schema": EXECUTION_SCHEMA,
            "fresh_preregistration_authentication_before_model_access": True,
            "production_model_hook_implemented": True,
            "compiled_blank_bytes_restored_exactly": product.trace[
                "compiled_blank_bytes_restored_exactly"
            ],
            "materialization": _json_copy(materialization),
            "materialization_trace": _json_copy(product.trace),
            "source_model_authentication": _json_copy(source_authentication),
            "source_model_authenticated_before_zero_step_and_training": True,
            "zero_step_authentication": _json_copy(zero_step),
            "both_zero_steps_authenticated_before_any_optimizer_step": True,
            "ephemeral_zero_step_children_pruned_before_training": True,
            "identical_frozen_replay_kwargs_seeds_checkpoints": True,
            "sequential_arm_order": list(ARM_IDS),
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "arms": arm_reports,
        "checkpoint_comparison": comparison_rows,
        "trajectory_summary": trajectory_summary,
        "retained_model_artifacts": {
            "blank_cache": str(product.blank_candidate_path),
            "folded_init": str(product.folded_publication_root),
            "folded_final": str(output / FOLDED_FINAL_DIRECTORY_NAME),
            "blank_final": str(output / BLANK_FINAL_DIRECTORY_NAME),
        },
        "retained_checkpoint_count": 0,
        "sealed_evaluation_accessed": False,
        "publication_attempted": False,
        "publication_authorized": False,
        "accepted_artifact": False,
        "behavior_capability_proven": False,
        "next_action": "human-review-experimental-comparison-only",
    }
    report = {**body, "report_payload_sha256": _canonical_sha256(body)}
    report_path = output / COMPARISON_REPORT_FILENAME
    deps.atomic_json(report_path, report)
    persisted = deps.load_json(report_path)
    if _canonical_bytes(persisted) != _canonical_bytes(report):
        raise FoldedBlankRecoveryExecutionError(
            "Persisted comparison report differs from its in-memory evidence."
        )
    _validate_retained_layout(output)
    return {
        **report,
        "comparison_report_path": str(report_path),
        "comparison_report_file_sha256": deps.file_sha256(report_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = execute_recovery_ab(
        preregistration=args.preregistration,
        preregistration_sha256=args.preregistration_sha256,
        output_root=args.output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
