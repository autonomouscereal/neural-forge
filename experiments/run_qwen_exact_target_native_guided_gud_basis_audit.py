#!/usr/bin/env python3
"""Produce the fit-only guided-GUD replay basis, mask, and audit receipt."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import run_qwen_exact_target_native_guided_candidate_tournament as v3_tournament
import run_qwen_exact_target_native_guided_gud_candidate_tournament as tournament
import run_qwen_exact_target_native_stage as stage
import neural_forge.hf_gud_output_jacobian_replay as gud_replay
import neural_forge.hf_target_native_behavior_distillation as behavior
from neural_forge.hf_guided_checkpoint_recovery import (
    portable_v3_factor_state_sha256,
    rebind_recovered_v3_factor_payload,
)


RUN_SCHEMA = "neural-forge-guided-gud-production-basis-audit-run-v1"
DEFAULT_OUTPUT = Path(
    "/mnt/ai_data/neural-forge/audits/"
    "qwen_exact_target_native_guided_gud_basis_v4"
)
DEFAULT_SOURCE_PREREGISTRATION = Path(
    "/mnt/ai_data/neural-forge/prereg/"
    "qwen_exact_target_native_guided_candidate_tournament_portable_v3_20260713/"
    "preregistration.json"
)
DEFAULT_SOURCE_RUN_ROOT = Path(
    "/mnt/ai_data/neural-forge/runs/"
    "qwen_exact_target_native_guided_candidate_tournament_portable_v3_20260713"
)


class GuidedGUDBasisAuditError(RuntimeError):
    """The production basis audit or exact rollback failed."""


def _source_row(
    *,
    preregistration: Path,
    preregistration_sha256: str,
    run_report: Path,
    winner_selection: Path,
) -> dict[str, Any]:
    winner = stage._strict_json(winner_selection)
    selected = winner.get("winner") if isinstance(winner, Mapping) else None
    if not isinstance(selected, Mapping):
        raise GuidedGUDBasisAuditError("Source winner selection is malformed.")
    return {
        "tournament_preregistration": {
            "path": str(preregistration),
            "sha256": preregistration_sha256,
        },
        "tournament_run_report": {
            "path": str(run_report),
            "sha256": stage._file_sha256(run_report),
        },
        "winner_selection": {
            "path": str(winner_selection),
            "sha256": stage._file_sha256(winner_selection),
        },
        "expected_winner": {
            "arm_id": selected.get("arm_id"),
            "step": selected.get("step"),
            "runtime_factor_state_sha256": selected.get(
                "runtime_factor_state_sha256"
            ),
            "portable_factor_state_sha256": selected.get(
                "portable_factor_state_sha256"
            ),
            "factor_pack_file_sha256": selected.get("factor_pack_sha256"),
        },
    }


def _reauthenticate_source(
    source: tournament.AuthenticatedV3WinnerSource,
) -> tournament.AuthenticatedV3WinnerSource:
    if type(source) is not tournament.AuthenticatedV3WinnerSource:
        raise GuidedGUDBasisAuditError("Basis audit requires an authenticated v3 source.")
    return tournament.authenticate_v3_winner_source(
        {
            "tournament_preregistration": {
                "path": str(source.tournament_preregistration_path),
                "sha256": source.tournament_preregistration_sha256,
            },
            "tournament_run_report": {
                "path": str(source.run_report_path),
                "sha256": source.run_report_sha256,
            },
            "winner_selection": {
                "path": str(source.winner_selection_path),
                "sha256": source.winner_selection_sha256,
            },
            "expected_winner": {
                "arm_id": source.arm_id,
                "step": source.step,
                "runtime_factor_state_sha256": source.runtime_factor_state_sha256,
                "portable_factor_state_sha256": source.portable_factor_state_sha256,
                "factor_pack_file_sha256": source.factor_pack_file_sha256,
            },
        },
        base=source.tournament_preregistration_path.parent,
    )


def _rollback_exact(runner: Any, transaction: Any) -> str:
    failures: list[tuple[str, BaseException]] = []
    try:
        runner.rollback_guided_installation(transaction)
    except BaseException as error:
        failures.append(("guided-QKVO rollback", error))
    try:
        transaction.rollback()
    except BaseException as error:
        failures.append(("base transaction rollback", error))
    try:
        observed = runner.state_sha256(runner.compiled.child_model)
    except BaseException as error:
        failures.append(("compiled-state authentication", error))
        observed = ""
    if observed != runner.compiled_state_sha256:
        failures.append(
            (
                "compiled-state mismatch",
                GuidedGUDBasisAuditError(
                    "Basis audit did not restore exact compiled child bytes."
                ),
            )
        )
    if failures:
        combined = GuidedGUDBasisAuditError(
            "One or more basis-audit rollback layers failed."
        )
        for label, error in failures:
            combined.add_note(f"{label}: {type(error).__name__}: {error}")
        raise combined
    return observed


def _producer_implementation() -> dict[str, str]:
    return {
        "producer_sha256": stage._file_sha256(Path(__file__).resolve()),
        "tournament_consumer_sha256": stage._file_sha256(
            Path(tournament.__file__).resolve()
        ),
        "production_replay_sha256": stage._file_sha256(
            Path(gud_replay.__file__).resolve()
        ),
        "behavior_trainer_sha256": stage._file_sha256(
            Path(behavior.__file__).resolve()
        ),
    }


def execute_guided_gud_basis_audit(
    source: tournament.AuthenticatedV3WinnerSource,
    output_dir: Path | str,
    *,
    runtime: Any | None = None,
    audit_runner: Callable[..., gud_replay.GUDOutputJacobianReplayResult] | None = None,
) -> dict[str, Any]:
    """Restore v3, run one fit-only replay audit, publish artifacts, and stop."""

    source = _reauthenticate_source(source)
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise GuidedGUDBasisAuditError("output_dir must not already exist.")
    contract = source.tournament_contract
    preflight = None
    if runtime is None:
        preflight = v3_tournament.validate_candidate_tournament_cuda_preflight(
            contract,
            dtype=str(contract.runtime["dtype"]),
            parent_device_map=str(contract.runtime["parent_device_map"]),
            target_device_map=str(contract.runtime["target_device_map"]),
        )
        runner = v3_tournament.ProductionCandidateTournamentRuntime(
            dtype=str(contract.runtime["dtype"]),
            parent_device_map=str(contract.runtime["parent_device_map"]),
            target_device_map=str(contract.runtime["target_device_map"]),
            cuda_preflight_report=preflight,
        )
    else:
        runner = runtime
    replay = (
        gud_replay.run_gud_output_jacobian_replay_audit
        if audit_runner is None
        else audit_runner
    )
    transaction: Any | None = None
    prepare_report: Mapping[str, Any] | None = None
    bundle_payload: Mapping[str, Any] | None = None
    result: gud_replay.GUDOutputJacobianReplayResult | None = None
    mask: tournament.GuidedGUDLayerMask | None = None
    derivation: Mapping[str, Any] | None = None
    runtime_v3_sha = ""
    rollback_sha = ""
    source_evidence: Mapping[str, Any] | None = None
    try:
        prepare_report = runner.prepare(contract)
        if (
            runner.parent is None
            or runner.compiled is None
            or runner.compiled_state_sha256 is None
            or prepare_report.get("compiled_state_sha256")
            != runner.compiled_state_sha256
        ):
            raise GuidedGUDBasisAuditError(
                "Production runtime did not prepare one authenticated compiled child."
            )
        transaction = runner.install_transaction(runner.compiled)
        runner.restore_source_v2(transaction, contract.source)
        runner.install_guided_transaction(transaction, contract.source)
        source_v2 = behavior.capture_target_native_factor_state(transaction)
        live_binding = behavior.authenticate_guided_factor_sources(
            transaction,
            runner._guided_tranche,
            source_v2,
        )
        del source_v2
        runtime_payload = rebind_recovered_v3_factor_payload(
            source.factor_payload,
            live_binding,
        )
        if (
            portable_v3_factor_state_sha256(runtime_payload, live_binding)
            != source.portable_factor_state_sha256
        ):
            raise GuidedGUDBasisAuditError(
                "Runtime v3 rebinding changed portable winner content."
            )
        runtime_v3_sha = str(runtime_payload["factor_state_sha256"])
        runner._guided_tranche.set_training_stage("joint")
        if runner._guided_tranche.training_stage != "joint":
            raise GuidedGUDBasisAuditError(
                "Guided QKVO tranche did not enter joint stage before restore."
            )
        restored = behavior.restore_target_native_factor_payload(
            transaction,
            runtime_payload,
            guided_tranche=runner._guided_tranche,
            expected_guided_source_binding=live_binding,
        )
        if (
            restored.factor_state_sha256 != runtime_v3_sha
            or restored.schema != behavior.BEHAVIOR_FACTOR_PACK_V3_SCHEMA
        ):
            raise GuidedGUDBasisAuditError("Exact v3 restore changed factor bytes.")
        del restored, runtime_payload
        gc.collect()
        source_candidate = contract.splits["candidate"]
        basis_domain = behavior.load_behavior_domain_suite(
            "fit",
            source_candidate.payload,
            artifact_sha256=source_candidate.sha256,
        )
        schedule = tournament.guided_gud_basis_audit_schedule()
        config = tournament.guided_gud_basis_audit_config(
            contract.source.execution_contract.seed
        )
        result = replay(
            transaction=transaction,
            guided_tranche=runner._guided_tranche,
            source_guided_qkvo_binding=live_binding,
            parent_model=runner.parent,
            basis_domain=basis_domain,
            schedule=schedule,
            config=config,
        )
        if type(result) is not gud_replay.GUDOutputJacobianReplayResult:
            raise GuidedGUDBasisAuditError(
                "Production replay returned a noncanonical result."
            )
        bundle_payload = tournament.guided_gud_basis_bundle_from_audit(result)
        mask, derivation = tournament.guided_gud_layer_mask_from_replay(result)
        if result.report.payload.get("source_v3_portable_factor_state_sha256") != (
            source.portable_factor_state_sha256
        ):
            raise GuidedGUDBasisAuditError(
                "Production replay observed another portable v3 source."
            )
        cleanup_transaction = transaction
        transaction = None
        rollback_sha = _rollback_exact(runner, cleanup_transaction)
    except BaseException as error:
        if transaction is not None:
            cleanup_transaction = transaction
            transaction = None
            try:
                _rollback_exact(runner, cleanup_transaction)
            except BaseException as rollback_error:
                rollback_error.add_note(
                    f"Original basis-audit failure: {type(error).__name__}: {error}"
                )
                raise rollback_error from error
        raise
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()

    assert prepare_report is not None
    assert bundle_payload is not None
    assert result is not None
    assert mask is not None
    assert derivation is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".staging",
            dir=output.parent,
        )
    ).resolve()
    try:
        bundle_path = staging / "guided_gud_basis_bundle.pt"
        v3_tournament._atomic_torch_save(bundle_path, bundle_payload)
        bundle_sha = stage._file_sha256(bundle_path)
        authenticated_bundle = tournament.authenticate_guided_gud_basis_bundle(
            bundle_path,
            bundle_sha,
        )
        source_evidence = tournament.authenticate_guided_gud_basis_source(
            authenticated_bundle,
            source,
        )
        audit_report_path = staging / "gud_output_jacobian_replay_report.json"
        v3_tournament._atomic_json(
            audit_report_path,
            result.report.to_dict(),
            exclusive=True,
        )
        mask_path = staging / "guided_gud_layer_mask.json"
        v3_tournament._atomic_json(mask_path, mask.as_dict(), exclusive=True)
        mask_sha = stage._file_sha256(mask_path)
        authenticated_mask, _, _ = tournament.authenticate_guided_gud_layer_mask(
            mask_path,
            mask_sha,
            expected_audit_report_sha256=authenticated_bundle.audit_report_sha256,
        )
        tournament.authenticate_guided_gud_mask_derivation(
            authenticated_mask,
            authenticated_bundle,
        )
        derivation_path = staging / "guided_gud_layer_mask_derivation.json"
        v3_tournament._atomic_json(
            derivation_path,
            derivation,
            exclusive=True,
        )
        run_report: dict[str, Any] = {
            "schema": RUN_SCHEMA,
            "status": "basis-audit-complete-before-training",
            "source_v3_winner": {
                "arm_id": source.arm_id,
                "step": source.step,
                "historical_runtime_factor_state_sha256": (
                    source.runtime_factor_state_sha256
                ),
                "audit_runtime_factor_state_sha256": runtime_v3_sha,
                "portable_factor_state_sha256": source.portable_factor_state_sha256,
            },
            "source_authentication": dict(source_evidence),
            "runtime_preparation": dict(prepare_report),
            "compiled_state_sha256_before": prepare_report[
                "compiled_state_sha256"
            ],
            "compiled_state_sha256_after_rollback": rollback_sha,
            "basis_bundle": {
                "path": str(output / bundle_path.name),
                "sha256": bundle_sha,
                "payload_sha256": authenticated_bundle.payload_sha256,
            },
            "replay_report": {
                "path": str(output / audit_report_path.name),
                "sha256": stage._file_sha256(audit_report_path),
                "report_sha256": authenticated_bundle.audit_report_sha256,
            },
            "guided_gud_layer_mask": {
                "path": str(output / mask_path.name),
                "sha256": mask_sha,
                "contract_sha256": authenticated_mask.mask_contract_sha256,
                "derivation_sha256": authenticated_mask.derivation_sha256,
                "gate_up_selected_layer_count": sum(
                    authenticated_mask.gate_up_layer_mask
                ),
                "down_selected_layer_count": sum(
                    authenticated_mask.down_layer_mask
                ),
            },
            "mask_derivation_report": {
                "path": str(output / derivation_path.name),
                "sha256": stage._file_sha256(derivation_path),
                "derivation_sha256": derivation["derivation_sha256"],
            },
            "producer_implementation": _producer_implementation(),
            "models_loaded_once": True,
            "compiler_invocation_count": 1,
            "optimizer_created": False,
            "training_launched": False,
            "candidate_evaluation_launched": False,
            "sealed_evaluation_launched": False,
            "sealed_prompt_count_replayed": 0,
            "model_checkpoint_written": False,
            "stopped_after_basis_artifacts": True,
        }
        run_report["run_report_sha256"] = tournament.canonical_sha256(run_report)
        v3_tournament._atomic_json(
            staging / "basis_audit_run_report.json",
            run_report,
            exclusive=True,
        )
        if output.exists():
            raise GuidedGUDBasisAuditError("output_dir appeared during publication.")
        os.replace(staging, output)
        final_bundle = tournament.authenticate_guided_gud_basis_bundle(
            output / bundle_path.name,
            bundle_sha,
        )
        tournament.authenticate_guided_gud_basis_source(final_bundle, source)
        return dict(stage._strict_json(output / "basis_audit_run_report.json"))
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-tournament-preregistration",
        type=Path,
        default=DEFAULT_SOURCE_PREREGISTRATION,
    )
    parser.add_argument(
        "--source-tournament-preregistration-sha256",
        default=tournament.EXPECTED_SOURCE_TOURNAMENT_PREREGISTRATION_SHA256,
    )
    parser.add_argument(
        "--source-tournament-run-report",
        type=Path,
        default=DEFAULT_SOURCE_RUN_ROOT / "run_report.json",
    )
    parser.add_argument(
        "--source-winner-selection",
        type=Path,
        default=DEFAULT_SOURCE_RUN_ROOT / "winner_selection.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preregistration = args.source_tournament_preregistration.expanduser().resolve()
    run_report = args.source_tournament_run_report.expanduser().resolve()
    winner_selection = args.source_winner_selection.expanduser().resolve()
    source = tournament.authenticate_v3_winner_source(
        _source_row(
            preregistration=preregistration,
            preregistration_sha256=(
                args.source_tournament_preregistration_sha256
            ),
            run_report=run_report,
            winner_selection=winner_selection,
        ),
        base=preregistration.parent,
    )
    report = execute_guided_gud_basis_audit(
        source,
        args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
