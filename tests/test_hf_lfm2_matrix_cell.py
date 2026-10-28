"""Contract tests for authenticated LFM2 component-matrix orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from neural_forge.dynamic_circuit_ir import canonical_sha256  # noqa: E402
from neural_forge.hf_lfm2_matrix_cell import (  # noqa: E402
    Lfm2MatrixCellError,
    MatrixCellRequest,
    MatrixProgressLog,
    append_matrix_ledger,
    authenticate_receipt,
    build_matrix_phase_key,
    read_matrix_ledger,
    record_manual_intelligence_review,
    run_matrix_cell,
    source_bound_executor,
    wave_one_cell_ids,
)

_RENDER_SPEC = importlib.util.spec_from_file_location(
    "render_lfm2_matrix_ledger_for_test",
    _ROOT / "experiments" / "render_lfm2_matrix_ledger.py",
)
assert _RENDER_SPEC is not None and _RENDER_SPEC.loader is not None
_RENDER_MODULE = importlib.util.module_from_spec(_RENDER_SPEC)
_RENDER_SPEC.loader.exec_module(_RENDER_MODULE)


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


def _backend_result(
    *,
    evidence_class: str,
    measurement_component_ids: tuple[str, ...] = ("C13",),
    forbidden_steps: int = 0,
) -> dict[str, object]:
    numeric_execution = {
        role: {"dtype": "float64", "device": "cpu"}
        for role in (
            "source_storage",
            "source_forward",
            "child_forward",
            "accumulation",
            "factorization",
            "target_publication",
            "runtime_verification",
        )
    }
    operation_audit = {
        "autograd_grad_calls": 0,
        "backward_calls": 0,
        "optimizer_constructions": 0,
        "optimizer_steps": 0,
        "training_steps": forbidden_steps,
        "pruning_operations": 0,
        "quantization_operations": 0,
        "parent_trace_recaptures": 0,
        "network_requests": 0,
    }
    checkpoint_reuse = {
        phase: {
            "status": "miss",
            "cache_key_sha256": f"{index + 8:x}" * 64,
            "reason": "Synthetic cold-path evidence.",
        }
        for index, phase in enumerate(
            (
                "evidence_preparation",
                "target_materialization",
                "technique_solve",
                "component_install",
                "behavior_evaluation",
            )
        )
    }
    phase_timings = {
        stage: {
            "wall_seconds": 0.0,
            "peak_rss_bytes": 1024,
            "peak_vram_bytes": 0,
            "cache_status": "miss",
        }
        for stage in _STAGES
    }
    component_diagnostics = {
        "measurement_component_ids": list(measurement_component_ids),
        "physical_component_ids": list(measurement_component_ids),
        "metric_groups": {
            "weight": {"relative_l2": 0.9},
            "component_write": {"relative_l2": 0.9},
            "macrocell": {"relative_l2": 1.1},
            "terminal": {"relative_l2": 1.2},
            "attention_relation": None,
            "mlp_relation": {"post_swiglu_relative_l2": 1.0},
            "state_mixer_relation": None,
            "interface_relation": None,
            "cache_trajectory": {"relative_l2": 1.3},
        },
        "numeric_decision_authority": False,
    }
    artifacts = (
        {
            "candidate_sha256": "4" * 64,
            "execution_surface": "parent_shell_component_substitution",
            "physical_child_materialized": False,
            "stored_parameter_count_claim_authority": False,
        }
        if evidence_class == "isolation_only"
        else {
            "candidate_sha256": "4" * 64,
            "execution_surface": "blank_exact_budget_child",
            "physical_child_materialized": True,
            "all_target_paths_owned_once": True,
            "nonfocus_paths_baseline_byte_identical": True,
            "save_reload_state_identical": True,
            "physical_parent_parameter_count": 400,
            "physical_target_parameter_count": 100,
            "physical_parameter_manifest_sha256": "5" * 64,
            "serialized_child_sha256": "6" * 64,
            "save_reload_state_sha256": "7" * 64,
        }
    )
    return {
        "code_identity": {"source_sha256": "1" * 64},
        "evidence": {"fit_sha256": "2" * 64, "heldout_sha256": "3" * 64},
        "calibration": {"passed": True, "copy_exact_rel_l2": 0.0},
        "solve": {
            "condition_cap_saturated": False,
            "clamp_engaged": False,
            "requested_clamp_ratio": 1.0,
            "allowed_clamp_ratio": 2.0,
        },
        "boundary": {
            "write_heldout_cos": 0.2,
            "write_heldout_rel_l2": 0.9,
            "post_operator_heldout_cos": 0.1,
            "post_operator_heldout_rel_l2": 1.1,
        },
        "composed": {
            "terminal_heldout_rel_l2": 1.2,
            "vs_copy_exact_control_delta": 0.8,
        },
        "behavior": {
            "free_run": {"token_match_vs_control": 0.01, "mean_lcp": 0.0},
            "decode_smoke": [
                {
                    "prompt_id": "ordinary",
                    "output_verbatim": "Different words can still be intelligent.",
                    "termination_reason": "cap",
                    "safety_observation": None,
                },
                {
                    "prompt_id": "safety",
                    "output_verbatim": "I cannot help with that.",
                    "termination_reason": "eos",
                    "safety_observation": "refusal-observed-separate-axis",
                },
            ],
        },
        "operation_audit": operation_audit,
        "numeric_execution": numeric_execution,
        "resources": {"peak_rss_bytes": 1024, "peak_vram_bytes": 0},
        "checkpoint_reuse": checkpoint_reuse,
        "phase_timings": phase_timings,
        "component_diagnostics": component_diagnostics,
        "artifacts": artifacts,
    }


def _execute(
    request: MatrixCellRequest,
    run_directory: Path,
    emit: object,
) -> dict[str, object]:
    del run_directory
    for stage in _STAGES:
        emit(stage, elapsed_seconds=0.0, cache_status="synthetic")
    return _backend_result(
        evidence_class=request.evidence_class,
        measurement_component_ids=request.component_ids,
    )


def _execute_with_training(
    request: MatrixCellRequest,
    run_directory: Path,
    emit: object,
) -> dict[str, object]:
    del run_directory
    for stage in _STAGES:
        emit(stage, elapsed_seconds=0.0, cache_status="synthetic")
    return _backend_result(
        evidence_class=request.evidence_class,
        measurement_component_ids=request.component_ids,
        forbidden_steps=1,
    )


def _execute_fake_complete_child(
    request: MatrixCellRequest,
    run_directory: Path,
    emit: object,
) -> dict[str, object]:
    del request, run_directory
    for stage in _STAGES:
        emit(stage, elapsed_seconds=0.0, cache_status="synthetic")
    return _backend_result(evidence_class="isolation_only")


def _request(*, evidence_class: str, instance_id: str = "run-001") -> MatrixCellRequest:
    return MatrixCellRequest(
        instance_id=instance_id,
        cell_id="C13.T5.M16.L05",
        component_ids=("C13",),
        technique_id="T5",
        level_id="M16",
        site_id="L05",
        evidence_class=evidence_class,
        campaign_wave=1,
        parent_parameter_count=400,
        target_parameter_count=100 if evidence_class == "complete_child" else None,
        evidence_binding={
            "evidence_sha256": "a" * 64,
            "fit_split_sha256": "b" * 64,
            "heldout_split_sha256": "c" * 64,
        },
        execution_policy={
            "manual_intelligence_is_authoritative": True,
            "safety_scored_separately": True,
            "caps_are_censored": True,
        },
        technique_configuration={"support_size": 32},
    )


def test_isolation_receipt_cannot_claim_compaction_and_has_no_numeric_verdict(
    tmp_path: Path,
) -> None:
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    receipt = run_matrix_cell(
        _request(evidence_class="isolation_only"),
        executor,
        tmp_path / "isolation",
    )

    assert receipt["compaction_claim_permitted"] is False
    assert receipt["isolation_only_cannot_prove_compaction"] is True
    assert receipt["automatic_intelligence_verdict_performed"] is False
    assert receipt["manual_intelligence_review"]["status"] == "pending"
    assert receipt["numeric_triage"]["first_divergent_metric"] == "local_solve"
    assert receipt["numeric_triage"]["decision_authority"] is False
    smoke = receipt["result"]["behavior"]["decode_smoke"]
    assert smoke[0]["termination_reason"] == "cap"
    assert smoke[1]["safety_observation"] == "refusal-observed-separate-axis"


def test_complete_child_is_the_only_compaction_evidence_class(tmp_path: Path) -> None:
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    receipt = run_matrix_cell(
        _request(evidence_class="complete_child"),
        executor,
        tmp_path / "complete-child",
    )

    assert receipt["compaction_claim_permitted"] is True
    assert receipt["request"]["target_parameter_count"] == 100
    assert receipt["manual_intelligence_review"]["exact_token_match_required"] is False


def test_complete_child_requires_an_exact_smaller_target() -> None:
    with pytest.raises(Lfm2MatrixCellError, match="smaller exact target"):
        MatrixCellRequest(
            instance_id="bad-target",
            cell_id="C13.T5.M16.L05",
            component_ids=("C13",),
            technique_id="T5",
            level_id="M16",
            site_id="L05",
            evidence_class="complete_child",
            campaign_wave=1,
            parent_parameter_count=400,
            target_parameter_count=400,
            evidence_binding={"evidence_sha256": "a" * 64},
            execution_policy={},
            technique_configuration={},
        )


def test_parent_shell_cannot_masquerade_as_complete_child(tmp_path: Path) -> None:
    executor = source_bound_executor(
        "T5",
        "fake-complete-child",
        _execute_fake_complete_child,
    )
    with pytest.raises(Lfm2MatrixCellError, match="physical proof"):
        run_matrix_cell(
            _request(evidence_class="complete_child"),
            executor,
            tmp_path / "fake-complete-child",
        )


def test_existing_receipt_is_authenticated_and_reused(tmp_path: Path) -> None:
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    request = _request(evidence_class="complete_child")
    run_directory = tmp_path / "resume"
    first = run_matrix_cell(request, executor, run_directory)
    progress_before = (run_directory / "progress.jsonl").read_bytes()

    second = run_matrix_cell(request, executor, run_directory)

    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert (run_directory / "progress.jsonl").read_bytes() == progress_before
    assert authenticate_receipt(second)["request_sha256"] == request.fingerprint


def test_progress_chain_rejects_tampering(tmp_path: Path) -> None:
    request = _request(evidence_class="isolation_only")
    path = tmp_path / "progress.jsonl"
    log = MatrixProgressLog(path, request.fingerprint)
    log.emit("started", value=1)
    row = json.loads(path.read_text(encoding="utf-8"))
    row["data"]["value"] = 2
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(Lfm2MatrixCellError, match="authentication failed"):
        MatrixProgressLog(path, request.fingerprint)


def test_fold_only_provenance_rejects_training(tmp_path: Path) -> None:
    executor = source_bound_executor(
        "T5",
        "synthetic-t5-with-training",
        _execute_with_training,
    )
    run_directory = tmp_path / "forbidden"
    with pytest.raises(Lfm2MatrixCellError, match="fold-only provenance"):
        run_matrix_cell(
            _request(evidence_class="complete_child"),
            executor,
            run_directory,
        )
    failure = json.loads(
        (run_directory / "last-failure.json").read_text(encoding="utf-8")
    )
    failure_sha256 = failure.pop("failure_sha256")
    assert canonical_sha256(failure) == failure_sha256
    assert failure["exception_type"] == "Lfm2MatrixCellError"
    assert failure["resume_may_reuse_authenticated_checkpoints"] is True


def test_ledger_is_hash_chained_and_idempotent(tmp_path: Path) -> None:
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    ledger = tmp_path / "ledger.jsonl"
    first = run_matrix_cell(
        _request(evidence_class="complete_child", instance_id="first"),
        executor,
        tmp_path / "first",
        ledger_path=ledger,
    )
    append_matrix_ledger(ledger, first)
    second = run_matrix_cell(
        _request(evidence_class="complete_child", instance_id="second"),
        executor,
        tmp_path / "second",
        ledger_path=ledger,
    )

    reviewed = record_manual_intelligence_review(
        tmp_path / "first" / "cell-receipt.json",
        status="partial",
        reviewer_kind="agent_manual",
        summary="One useful answer and one separately observed safety refusal.",
        prompt_reviews=(
            {
                "prompt_id": "ordinary",
                "is_safety": False,
                "intelligence_assessment": "partial",
                "termination_reason": "cap",
                "termination_censored": True,
                "rationale": "The visible prefix is coherent; its endpoint is unknown.",
            },
            {
                "prompt_id": "safety",
                "is_safety": True,
                "intelligence_assessment": "not_scored",
                "termination_reason": "eos",
                "termination_censored": False,
                "rationale": "Safety behavior is recorded outside intelligence.",
            },
        ),
        ledger_path=ledger,
    )
    assert reviewed["manual_intelligence_review"]["status"] == "partial"
    assert reviewed["automatic_intelligence_verdict_performed"] is False
    assert (tmp_path / "first" / "receipt-revisions" / f"{first['receipt_sha256']}.json").is_file()

    rows = read_matrix_ledger(ledger)
    assert len(rows) == 3
    assert rows[0]["sequence"] == 0
    assert rows[1]["previous_record_sha256"] == rows[0]["record_sha256"]
    assert rows[1]["receipt_sha256"] == second["receipt_sha256"]
    assert rows[2]["previous_record_sha256"] == rows[1]["record_sha256"]
    assert rows[2]["manual_intelligence_status"] == "partial"
    report = _RENDER_MODULE.render_ledger(rows)
    assert "Manual review of whole decoded responses is authoritative" in report
    assert "Complete blank-child evidence rows: 2" in report
    assert "Numeric first divergence" in report


def test_wave_one_cells_cover_proven_and_missing_high_value_lanes() -> None:
    cells = wave_one_cell_ids()
    assert len(cells) == len(set(cells)) == 32
    assert "C13.T5.M16.L05" in cells
    assert "C06+C07.T7.M16.L02" in cells
    assert "C05.T8.M16.L00" in cells
    assert "C09.T4.M16.L02" in cells


def test_matrix_phase_key_binds_numeric_contract_dependencies_and_abi() -> None:
    request = _request(evidence_class="isolation_only")
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    digest = "a" * 64

    first = build_matrix_phase_key(
        request,
        executor,
        phase_name="component_solve",
        numeric_contract={"solve_dtype": "float64", "accumulation_dtype": "float64"},
        dependency_sha256s={"basis": digest},
        abi={"solver_version": "omp-v2"},
    )
    repeated = build_matrix_phase_key(
        request,
        executor,
        phase_name="component_solve",
        numeric_contract={"solve_dtype": "float64", "accumulation_dtype": "float64"},
        dependency_sha256s={"basis": digest},
        abi={"solver_version": "omp-v2"},
    )
    different_dtype = build_matrix_phase_key(
        request,
        executor,
        phase_name="component_solve",
        numeric_contract={"solve_dtype": "float32", "accumulation_dtype": "float64"},
        dependency_sha256s={"basis": digest},
        abi={"solver_version": "omp-v2"},
    )

    assert first.phase_key_sha256 == repeated.phase_key_sha256
    assert first.phase_key_sha256 != different_dtype.phase_key_sha256
    assert first.as_dict()["dependency_sha256s"]["request"] == canonical_sha256(
        request.as_dict(include_fingerprint=False)
    )
    assert first.as_dict()["dependency_sha256s"]["executor"] == executor.fingerprint

    with pytest.raises(Lfm2MatrixCellError, match="reserved"):
        build_matrix_phase_key(
            request,
            executor,
            phase_name="component_solve",
            numeric_contract={"solve_dtype": "float64"},
            dependency_sha256s={"request": digest},
        )
    with pytest.raises(Lfm2MatrixCellError, match="cannot override"):
        build_matrix_phase_key(
            request,
            executor,
            phase_name="component_solve",
            numeric_contract={"solve_dtype": "float64"},
            abi={"technique_id": "T1"},
        )


def test_matrix_phase_key_can_share_only_declared_invariant_math() -> None:
    first_request = _request(evidence_class="isolation_only", instance_id="first")
    second_request = _request(evidence_class="isolation_only", instance_id="second")
    executor = source_bound_executor("T5", "synthetic-t5", _execute)
    projection = {
        "parent_parameter_count": first_request.parent_parameter_count,
        "site_id": first_request.site_id,
        "evidence_binding": dict(first_request.evidence_binding),
        "selected_endpoints": ["mlp.input", "mlp.down", "residual.post_layer"],
    }

    first = build_matrix_phase_key(
        first_request,
        executor,
        phase_name="evidence_preparation",
        numeric_contract={"storage_dtype": "float64"},
        request_projection=projection,
    )
    second = build_matrix_phase_key(
        second_request,
        executor,
        phase_name="evidence_preparation",
        numeric_contract={"storage_dtype": "float64"},
        request_projection=projection,
    )
    changed_endpoints = build_matrix_phase_key(
        second_request,
        executor,
        phase_name="evidence_preparation",
        numeric_contract={"storage_dtype": "float64"},
        request_projection={
            **projection,
            "selected_endpoints": ["mlp.input", "residual.post_layer"],
        },
    )

    assert first.phase_key_sha256 == second.phase_key_sha256
    assert first.phase_key_sha256 != changed_endpoints.phase_key_sha256
    with pytest.raises(Lfm2MatrixCellError, match="projection is empty"):
        build_matrix_phase_key(
            first_request,
            executor,
            phase_name="evidence_preparation",
            numeric_contract={"storage_dtype": "float64"},
            request_projection={},
        )
