from __future__ import annotations

import copy
import hashlib
import json

import pytest

from experiments import (  # noqa: E402
    verify_lfm2_cpcf_v5_real_attention_causal_pair as verifier_cli,
)
from neural_forge import (  # noqa: E402
    hf_lfm2_sequence_effective_attention_fold as sequence_fold,
)
from neural_forge.hf_lfm2_cpcf_v5_real_attention_causal_pair import (
    LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA,
    Lfm2CpcfV5RealAttentionCausalPairError,
    verify_lfm2_cpcf_v5_real_attention_causal_pair,
)
from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (
    H776_ATTENTION_CANDIDATE_IDS,
    H776_BLANK_REPORT_SHA256,
    H776_BLANK_STATE_SHA256,
    H776_CACHE_SHA256,
    H776_FIT_QUERY_ROWS,
    H776_FULL_POLICY_SHA256,
    H776_LAYER_COUNT,
    H776_LAYER_TYPES,
    H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
    H776_MAXIMUM_PROCESS_RSS_BYTES,
    H776_OPERATOR_KINDS,
    H776_ORIGIN_COMPILER_SHA256,
    H776_PARAMETER_COMPRESSION_RATIO,
    H776_PARENT_UNIQUE_PARAMETERS,
    H776_PHASE_COUNT,
    H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    H776_SEQUENCE_SOLVER_PLAN_SHA256,
    H776_SELECTIVE_CATALOG_SHA256,
    H776_SELECTIVE_PROVIDER_SHA256,
    H776_SELECTIVE_SOURCE_SHA256,
    H776_SOURCE_INVENTORY_SHA256,
    H776_SOURCE_STATE_SHA256,
    H776_SOURCE_TIED_TABLE_SHA256,
    H776_SPLIT_PHASE_COUNTS,
    H776_SPLIT_TRAJECTORY_COUNTS,
    H776_TARGET_PLAN_FINGERPRINT,
    H776_TARGET_TOPOLOGY_SHA256,
    H776_TARGET_UNIQUE_PARAMETERS,
    H776_TOKEN_BRIDGE_SHA256,
    H776_TOKENIZER_SHA256,
    H776_TRAJECTORY_COUNT,
    H776_VALIDATION_QUERY_ROWS,
    LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA,
    LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA,
    Lfm2SequenceEffectivePcgCollector,
)
from neural_forge.hf_lfm2_sequence_effective_attention_fold import (
    LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
)


_PHASES = (
    "prefill",
    "decode.000",
    "decode.001",
    "decode.002",
    "decode.003",
    "decode.004",
    "decode.005",
    "decode.006",
    "decode.007",
)


def _sha(value) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _seal(body, field: str):
    return {**body, field: _sha(body)}


def _inventory(unique_parameters: int):
    return _seal(
        {
            "method": ("unique_materialized_parameter_storage_offset_numel_identity"),
            "unique_parameters": unique_parameters,
            "raw_named_parameter_occurrences": 1,
            "raw_named_parameter_numel": unique_parameters,
            "parameter_iterator_unique_numel": unique_parameters,
            "independent_methods_agree": True,
            "alias_group_count": 0,
            "alias_groups": [],
            "checkpoint_byte_size_used": False,
            "quantization_adjustment_used": False,
            "pruning_mask_adjustment_used": False,
        },
        "inventory_sha256",
    )


def _pcg_event(role: str):
    report = _seal(
        {
            "role": role,
            "solver": "deterministic_jacobi_pcg_single_tensor",
            "maximum_iterations": 64,
            "iterations_executed": 0,
            "required_relative_normal_residual": 1e-8,
            "initial_relative_normal_residual": 0.0,
            "final_relative_normal_residual": 0.0,
            "relative_normal_residual_history": [],
            "solution_sha256": _sha({"solution": role}),
        },
        "report_sha256",
    )
    return _seal(
        {
            "schema": ("neural-forge-lfm2-sequence-effective-pcg-observation-v1"),
            "solver_kind": "single_tensor",
            "report": report,
            "report_sha256": report["report_sha256"],
            "solver_result_or_arguments_modified": False,
        },
        "observation_sha256",
    )


def _continuous_pcg_evidence():
    history = [2e-8] * 64 + [8e-9, 5e-9]
    checkpoints = [
        {
            "iteration": 64,
            "relative_normal_residual": 2e-8,
            "finite": True,
        },
        {
            "iteration": 65,
            "relative_normal_residual": 1.5e-8,
            "finite": True,
        },
        {
            "iteration": 66,
            "relative_normal_residual": 5e-9,
            "finite": True,
        },
    ]
    lane = {
        "start_iteration": 64,
        "planned_stop_iteration": 4096,
        "iterations_executed": 2,
        "attempted_matvec_calls": 4,
        "exact_diagnostic_matvec_calls": 2,
        "termination": "exact_tolerance_reached",
        "ending_iteration": 66,
        "ending_recurrence_relative_normal_residual": 5e-9,
        "ending_exact_relative_normal_residual": 5e-9,
        "ending_exact_residual_authenticated": True,
    }
    return history, checkpoints, lane


def _continuous_pcg_observation(report_body):
    report = _seal(report_body, "report_sha256")
    return _seal(
        {
            "schema": "neural-forge-lfm2-sequence-effective-pcg-observation-v1",
            "solver_kind": "single_tensor",
            "report": report,
            "report_sha256": report["report_sha256"],
            "solver_result_or_arguments_modified": False,
        },
        "observation_sha256",
    )


def _hidden_partial_trace_pcg_event(role: str):
    history, checkpoints, lane = _continuous_pcg_evidence()
    group = int(role.rsplit("_", 1)[-1])
    group_heads = (2 * group, 2 * group + 1)
    source_terms = [
        {
            "head": head,
            "other_head": other_head,
            "left_sha256": _sha({"left": role, "head": head, "other_head": other_head}),
            "coupling_sha256": _sha(
                {"coupling": role, "head": head, "other_head": other_head}
            ),
            "partial_trace_weight": 0.25 * (head + other_head + 1),
        }
        for head in group_heads
        for other_head in group_heads
    ]
    construction = _seal(
        {
            "schema": (
                "neural-forge-lfm2-sequence-effective-hidden-partial-trace-cholesky-v1"
            ),
            "role": role,
            "construction": "normalized_hidden_side_partial_trace",
            "formula": ("P_H=ridge*I_H+sum_(h,o)(trace(C_o,h)/head_dim)*A_h,o"),
            "ridge": 1e-6,
            "hidden_size": 776,
            "head_dim": 64,
            "partial_trace_shape": [776, 776],
            "partial_trace_dtype": "torch.float64",
            "partial_trace_device": "cpu",
            "partial_trace_sha256": _sha({"partial_trace": role}),
            "partial_trace_tensor_bytes": 776 * 776 * 8,
            "source_term_count": 4,
            "source_terms": source_terms,
            "source_terms_sha256": _sha(source_terms),
            "raw_maximum_absolute_asymmetry": 0.0,
            "symmetry_tolerance": 1e-12,
            "symmetric_within_numerical_tolerance": True,
            "symmetrization": "half_sum_with_transpose",
            "full_operator_materialized": False,
        },
        "report_sha256",
    )
    receipt = _seal(
        {
            "kind": "deterministic_hidden_partial_trace_cholesky",
            "construction_report": construction,
            "construction_report_sha256": construction["report_sha256"],
            "partial_trace_sha256": construction["partial_trace_sha256"],
            "factorization": "lower_cholesky_ex",
            "factorization_info": 0,
            "factor_sha256": _sha({"factor": role}),
            "factor_shape": [776, 776],
            "factor_dtype": "torch.float64",
            "factor_device": "cpu",
            "hidden_size": 776,
            "head_dim": 64,
            "partial_trace_spd": True,
            "positive_finite_factor_diagonal": True,
            "inverse_orientation_probe_relative_residual": 1e-15,
            "inverse_orientation_probe_tolerance": 1e-10,
            "factorization_materialized_only_after_original_lane_exact_miss": True,
            "full_operator_materialized": False,
        },
        "report_sha256",
    )
    preconditioner = {
        "kind": "deterministic_hidden_partial_trace_cholesky",
        "supplied": True,
        "authentication_attempted": True,
        "authenticated": True,
        "applied": True,
        "application_attempt_count": 2,
        "application_count": 2,
        "receipt": receipt,
        "authentication_failure_reason": None,
    }
    return _continuous_pcg_observation(
        {
            "role": role,
            "solver": (
                "deterministic_continuous_hidden_partial_trace_"
                "cholesky_pcg_single_tensor"
            ),
            "maximum_iterations": 4096,
            "hard_cap_includes_original_lane_iterations": True,
            "original_lane_maximum_iterations": 64,
            "original_lane_iterations_executed": 64,
            "single_restart_after_original_lane_exact_authentication_miss": True,
            "periodic_restart_count": 0,
            "iterations_executed": 66,
            "continuous_hidden_partial_trace_lane_iterations_executed": 2,
            "required_relative_normal_residual": 1e-8,
            "initial_relative_normal_residual": 1.0,
            "original_lane_final_recurrence_relative_normal_residual": 2e-8,
            "final_recurrence_relative_normal_residual": 5e-9,
            "final_relative_normal_residual": 5e-9,
            "relative_normal_residual_history": history,
            "recurrence_history_sha256": _sha(history),
            "exact_relative_normal_residual_checkpoints": checkpoints,
            "exact_residual_checkpoints_sha256": _sha(checkpoints),
            "total_matvec_calls": 70,
            "exact_final_residual_authenticated": True,
            "extended_after_original_lane_exact_authentication_miss": True,
            "exact_residual_used_to_initialize_continuous_lane_once": True,
            "exact_diagnostic_matvecs_modify_recurrence_state_or_direction": False,
            "execution_profile": dict(
                sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE
            ),
            "execution_profile_sha256": (
                sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
            ),
            "continuous_hidden_partial_trace_lane": lane,
            "restart_preconditioner": preconditioner,
            "solution_finite": True,
            "solution_sha256": _sha({"solution": role}),
        }
    )


def _archived_row_block_pcg_event(role: str):
    history, checkpoints, lane = _continuous_pcg_evidence()
    construction = _seal(
        {
            "schema": "neural-forge-lfm2-sequence-effective-row-block-jacobi-v1",
            "role": role,
            "construction": "exact_structured_operator_row_blocks",
            "row_block_count": 776,
            "row_block_size": 64,
            "row_blocks_shape": [776, 64, 64],
            "row_blocks_dtype": "torch.float64",
            "row_blocks_sha256": _sha({"row_blocks": role}),
            "scalar_diagonal_sha256": _sha({"diagonal": role}),
            "scalar_diagonal_maximum_absolute_error": 0.0,
            "scalar_diagonal_match_tolerance": 1e-12,
            "row_blocks_maximum_asymmetry": 0.0,
            "row_blocks_symmetry_tolerance": 1e-12,
            "source_term_count": 4,
            "source_terms_sha256": _sha({"source_terms": role}),
            "ridge": 1e-6,
            "full_operator_materialized": False,
        },
        "report_sha256",
    )
    receipt = _seal(
        {
            "kind": "deterministic_exact_row_block_jacobi_cholesky",
            "construction_report": construction,
            "construction_report_sha256": construction["report_sha256"],
            "row_blocks_sha256": construction["row_blocks_sha256"],
            "factorization": "batched_lower_cholesky_ex",
            "factor_sha256": _sha({"factor": role}),
            "row_block_count": 776,
            "row_block_size": 64,
            "all_row_blocks_spd": True,
            "factorization_materialized_only_after_original_lane_exact_miss": True,
            "full_operator_materialized": False,
        },
        "report_sha256",
    )
    preconditioner = {
        "kind": "deterministic_exact_row_block_jacobi_cholesky",
        "supplied": True,
        "authentication_attempted": True,
        "authenticated": True,
        "applied": True,
        "application_attempt_count": 2,
        "application_count": 2,
        "receipt": receipt,
        "authentication_failure_reason": None,
    }
    return _continuous_pcg_observation(
        {
            "role": role,
            "solver": ("deterministic_continuous_row_block_jacobi_pcg_single_tensor"),
            "maximum_iterations": 4096,
            "hard_cap_includes_original_lane_iterations": True,
            "original_lane_maximum_iterations": 64,
            "original_lane_iterations_executed": 64,
            "single_restart_after_original_lane_exact_authentication_miss": True,
            "periodic_restart_count": 0,
            "iterations_executed": 66,
            "continuous_row_block_lane_iterations_executed": 2,
            "required_relative_normal_residual": 1e-8,
            "initial_relative_normal_residual": 1.0,
            "original_lane_final_recurrence_relative_normal_residual": 2e-8,
            "final_recurrence_relative_normal_residual": 5e-9,
            "final_relative_normal_residual": 5e-9,
            "relative_normal_residual_history": history,
            "recurrence_history_sha256": _sha(history),
            "exact_relative_normal_residual_checkpoints": checkpoints,
            "exact_residual_checkpoints_sha256": _sha(checkpoints),
            "total_matvec_calls": 70,
            "exact_final_residual_authenticated": True,
            "extended_after_original_lane_exact_authentication_miss": True,
            "exact_residual_used_to_initialize_continuous_lane_once": True,
            "exact_diagnostic_matvecs_modify_recurrence_state_or_direction": False,
            "continuous_row_block_lane": lane,
            "restart_preconditioner": preconditioner,
            "solution_finite": True,
            "solution_sha256": _sha({"solution": role}),
        }
    )


def _sequence_artifact():
    effective = _sha({"solve": "effective"})
    seed = _sha({"solve": "seed"})
    values = [_sha({"solve": "value", "group": group}) for group in range(5)]
    output = _sha({"solve": "output"})
    anchor = _sha({"solve": "anchor"})
    producer_state = _sha({"producer_state": True})
    frontier = [
        {
            "proposal_index": 0,
            "stage": "scalable_producer_shared_value_sequence_output_seed",
            "accepted": True,
            "solve_report_sha256": seed,
        },
        {
            "proposal_index": 1,
            "stage": "monotone_streamed_effective_space_value_output_als",
            "accepted": False,
            "value_solve_report_sha256": values,
            "output_solve_report_sha256": output,
        },
        {
            "proposal_index": 2,
            "stage": "fixed_producer_value_subspace_anchor",
            "accepted": True,
            "selected_for_materialization": True,
            "output_solve_report_sha256": anchor,
        },
    ]
    return _seal(
        {
            "schema": LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA,
            "candidate_id": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
            "fixed_query_key_state_reused_exactly": True,
            "producer_state_sha256_before": producer_state,
            "producer_state_sha256_after": producer_state,
            "producer_state_unchanged": True,
            "fit_only_compilation": True,
            "compiled_from_splits": ["fit"],
            "selection_case_count_consumed": 0,
            "validation_case_count_consumed": 0,
            "selection_inputs_accessible": False,
            "validation_inputs_accessible": False,
            "dense_target_native_qkvo": True,
            "ordinary_dense_target_value_output": True,
            "blank_target_values_read": False,
            "forbidden_value_coefficient_tensor_materialized": False,
            "forbidden_hidden_head_dim_gram_materialized": False,
            "solver_plan": sequence_fold._sequence_effective_solver_plan(
                fit_rows=H776_FIT_QUERY_ROWS,
                target_query_heads=10,
                target_hidden_size=776,
                target_output_size=776,
                head_dim=64,
            ),
            "effective_operator_solve": {"report_sha256": effective},
            "candidate_frontier": frontier,
            "candidate_frontier_sha256": _sha(frontier),
            "bounded_proposal_count": 3,
            "selected_proposal_index": 2,
            "accepted_als_iterations": 0,
            "executed_als_iteration_limit": 2,
            "solve_report_sha256": [
                effective,
                seed,
                *values,
                output,
                anchor,
            ],
            "execution": {
                "parent_model_forward_calls": 0,
                "target_model_forward_calls": 0,
                "target_layer_forward_calls": 0,
                "generation_calls": 0,
                "training_steps": 0,
                "backward_calls": 0,
                "optimizer_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
                "autograd_graphs": 0,
            },
        },
        "report_sha256",
    )


def _hidden_sequence_artifact():
    artifact = _sequence_artifact()
    body = dict(artifact)
    body.pop("report_sha256")
    body["streamed_pcg_execution_profile"] = dict(
        sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE
    )
    body["streamed_pcg_execution_profile_sha256"] = (
        sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
    )
    return _seal(body, "report_sha256")


def _pcg_audit(artifact=None):
    artifact = _sequence_artifact() if artifact is None else artifact
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _pcg_event(f"sequence_effective_structured_value_iteration_1_group_{group}")
        )
    audit = dict(collector.finalize(sequence_artifact=artifact))
    assert audit["schema"] == LFM2_CPCF_V5_REAL_ATTENTION_PCG_AUDIT_SCHEMA
    return audit


def _continuous_pcg_audit(*, mode: str):
    artifact = (
        _hidden_sequence_artifact()
        if mode in {"hidden", "mixed"}
        else _sequence_artifact()
    )
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        if mode == "row" or (mode == "mixed" and group == 0):
            event = _archived_row_block_pcg_event(role)
        else:
            event = _hidden_partial_trace_pcg_event(role)
        collector(event)
    return artifact, dict(collector.finalize(sequence_artifact=artifact))


def _metric(
    *,
    rows: int,
    relative_rmse: float,
    cosine: float,
    label: str,
    target_label: str | None = None,
):
    return {
        "rows": rows,
        "width": 776,
        "rmse": relative_rmse / 2.0,
        "relative_rmse": relative_rmse,
        "cosine_similarity": cosine,
        "maximum_absolute_error": relative_rmse * 2.0,
        "prediction_sha256": _sha({"prediction": label}),
        "target_sha256": _sha(
            {"target": label if target_label is None else target_label}
        ),
    }


def _validation(*, combined: bool, regress_phase: bool = False):
    phase_reports = {}
    all_seals = []
    total_rows = 0
    for index, phase in enumerate(_PHASES):
        seals = [
            _sha({"validation_phase": phase, "trajectory": trajectory})
            for trajectory in range(50)
        ]
        all_seals.extend(seals)
        rows = H776_VALIDATION_QUERY_ROWS - 8 * 50 if index == 0 else 50
        total_rows += rows
        baseline_rmse = 0.40 + 0.01 * index
        baseline_cosine = 0.78 - 0.01 * index
        relative_rmse = baseline_rmse - (0.04 if combined else 0.0)
        cosine = baseline_cosine + (0.04 if combined else 0.0)
        if regress_phase and phase == "decode.003":
            relative_rmse = baseline_rmse + 0.01
            cosine = baseline_cosine - 0.01
        phase_reports[phase] = _seal(
            {
                "phase": phase,
                "phase_count": 50,
                "phase_sha256": seals,
                "post_operator": _metric(
                    rows=rows,
                    relative_rmse=relative_rmse + 0.05,
                    cosine=cosine - 0.05,
                    label=f"{combined}-{phase}-operator",
                    target_label=f"{phase}-operator",
                ),
                "post_layer": _metric(
                    rows=rows,
                    relative_rmse=relative_rmse,
                    cosine=cosine,
                    label=f"{combined}-{phase}-layer",
                    target_label=f"{phase}-layer",
                ),
            },
            "report_sha256",
        )
    assert total_rows == H776_VALIDATION_QUERY_ROWS
    return _seal(
        {
            "split": "validation",
            "phase_count": H776_SPLIT_PHASE_COUNTS["validation"],
            "phase_sha256": all_seals,
            "post_operator": _metric(
                rows=H776_VALIDATION_QUERY_ROWS,
                relative_rmse=0.50 - (0.04 if combined else 0.0),
                cosine=0.70 + (0.04 if combined else 0.0),
                label=f"{combined}-aggregate-operator",
                target_label="aggregate-operator",
            ),
            "post_layer": _metric(
                rows=H776_VALIDATION_QUERY_ROWS,
                relative_rmse=0.45 - (0.04 if combined else 0.0),
                cosine=0.75 + (0.04 if combined else 0.0),
                label=f"{combined}-aggregate-layer",
                target_label="aggregate-layer",
            ),
            "phase_metrics": phase_reports,
        },
        "report_sha256",
    )


def _selection(*, combined: bool, behavioral_improves: bool = True):
    values = {
        H776_ATTENTION_CANDIDATE_IDS[0]: 0.40,
        H776_ATTENTION_CANDIDATE_IDS[1]: 0.30,
        H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID: 0.20,
    }
    if combined and behavioral_improves:
        values = {
            H776_ATTENTION_CANDIDATE_IDS[0]: 0.35,
            H776_ATTENTION_CANDIDATE_IDS[1]: 0.25,
            H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID: 0.10,
        }
    selection = {
        candidate_id: [value] * 10 + [candidate_id]
        for candidate_id, value in values.items()
    }
    behavioral = {candidate_id: [value] * 10 for candidate_id, value in values.items()}
    return selection, behavioral


def _sequential_audit(*, source_chart: str):
    chart_hashes = {
        candidate_id: _sha({"chart": candidate_id})
        for candidate_id in H776_ATTENTION_CANDIDATE_IDS
    }
    bundle_hashes = {
        candidate_id: _sha({"bundle": candidate_id})
        for candidate_id in H776_ATTENTION_CANDIDATE_IDS
    }
    return _seal(
        {
            "enabled": True,
            "policy_report_sha256": _sha({"policy": "sequential"}),
            "candidate_selection_report_sha256": _sha({"selection": "sequential"}),
            "selected_candidate_id": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
            "selected_candidate_report_sha256": _sha(
                {"candidate": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID}
            ),
            "selected_effective_boundary_charts_sha256": chart_hashes[
                H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            ],
            "selected_operator_transport_mlp_bundle_sha256": bundle_hashes[
                H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
            ],
            "source_boundary_charts_sha256": source_chart,
            "source_boundary_charts_payload_sha256": _sha(
                {"source_chart_payload": True}
            ),
            "candidate_effective_boundary_charts_sha256": chart_hashes,
            "candidate_operator_transport_mlp_bundle_sha256": bundle_hashes,
            "fit_only_candidate_local_transport_authenticated": True,
            "selection_selects_frozen_bundle_without_refit": True,
            "validation_rows_used_by_transport_solver": 0,
            "winning_chart_used_for_final_metrics": True,
        },
        "audit_sha256",
    )


def _layer_result(
    *,
    combined: bool,
    input_bank: str,
    compiler_chart: str,
    regress_phase: bool,
    behavioral_improves: bool,
):
    selection, behavioral = _selection(
        combined=combined,
        behavioral_improves=behavioral_improves,
    )
    sequence_artifact = _sequence_artifact()
    solver_plan = sequence_artifact["solver_plan"]
    body = {
        "layer_result_sha256": _sha({"layer_result": combined}),
        "input_bank_sha256": input_bank,
        "next_bank_sha256": _sha({"next_bank": combined}),
        "candidate_ids": list(H776_ATTENTION_CANDIDATE_IDS),
        "selected_candidate_id": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        "selection_keys": selection,
        "behavioral_selection_keys": behavioral,
        "lexical_tie_break_excluded_from_strict_improvement": True,
        "sequence_effective_won": True,
        "sequence_effective_strictly_better_than_both_controls": True,
        "scientific_outcome": "positive_sequence_effective_selection",
        "full_validation_metrics": _validation(
            combined=combined,
            regress_phase=regress_phase,
        ),
        "sequence_effective_artifact": sequence_artifact,
        "sequence_solver_plan": solver_plan,
        "pcg_convergence": _pcg_audit(sequence_artifact),
        "in_memory_next_bank_created_by_layer_replay": True,
        "k3_checkpoint_published": False,
        "layer_3_compiled": False,
        "automatic_retry_or_retuning_performed": False,
    }
    if combined:
        body["attention_sequential_transport"] = _sequential_audit(
            source_chart=compiler_chart
        )
    return _seal(body, "audit_sha256")


def _allocator_policy():
    return _seal(
        {
            "schema": "test-cuda-allocator-policy",
            "logical_devices": [0, 1],
            "requested_maximum_cuda_bytes_per_device": (
                H776_MAXIMUM_CUDA_BYTES_PER_DEVICE
            ),
            "device_receipts": [
                {"logical_device_index": 0},
                {"logical_device_index": 1},
            ],
            "all_devices_enforced": True,
            "enforced_before_substantive_cuda_allocations": True,
        },
        "receipt_sha256",
    )


def _runtime_guard():
    return _seal(
        {
            "schema": ("neural-forge-lfm2-cpcf-v5-real-attention-runtime-guard-v1"),
            "cuda_allocator_policy": _allocator_policy(),
            "optimizer_constructions": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "autograd_grad_calls": 0,
            "pruning_operations": 0,
            "quantization_operations": 0,
            "parent_module_forward_calls": 0,
            "parent_root_forward_calls": 0,
            "parent_inference_entrypoint_calls": 0,
            "target_module_forward_calls": 100,
            "target_root_forward_calls": 0,
            "target_inference_entrypoint_calls": 0,
            "training_steps": 0,
            "ordinary_bfloat16_and_float32_casts_blocked": False,
            "scope_started": "immediately_after_parent_model_load",
            "parent_model_load_guarded": False,
            "parent_model_observer_started_immediately_after_load": True,
            "blank_target_synthesis_guarded": True,
            "target_model_observer_started_immediately_after_synthesis": True,
            "selective_provider_open_and_close_guarded": True,
            "source_tied_table_mapping_guarded": True,
            "layer_2_probe_guarded": True,
            "all_model_touching_operations_after_parent_load_guarded": True,
            "clean": True,
        },
        "report_sha256",
    )


def _resource(*, combined: bool):
    labels = [
        "enter",
        "periodic",
        "authenticated_inputs_ready",
        "interface_recomputed_final_norm_released",
        "boundary_frontier_hydrated",
        "k2_prefix_installed",
    ]
    if combined:
        labels.append("layer_2_post_operator_chart_transport_frozen")
    labels.extend(
        [
            "layer_2_evidence_and_charts_ready",
            "layer_2_compiled_and_validated",
            "layer_2_evidence_released",
            "exit",
        ]
    )
    samples = [
        {
            "label": label,
            "monotonic_ns": index + 1,
            "process": {"VmRSS": 100 + index, "VmHWM": 200 + index, "VmSwap": 0},
            "cuda_peak_reserved_bytes": {
                "cuda:0": 300 + index,
                "cuda:1": 400 + index,
            },
        }
        for index, label in enumerate(labels)
    ]
    return _seal(
        {
            "schema": LFM2_CPCF_V5_REAL_ATTENTION_RESOURCE_REPORT_SCHEMA,
            "sample_interval_seconds": 0.05,
            "sample_count": len(samples),
            "maximum_process_rss_bytes": H776_MAXIMUM_PROCESS_RSS_BYTES,
            "maximum_cuda_bytes_per_device": (H776_MAXIMUM_CUDA_BYTES_PER_DEVICE),
            "peak_vm_rss_bytes": max(sample["process"]["VmRSS"] for sample in samples),
            "peak_vm_hwm_bytes": max(sample["process"]["VmHWM"] for sample in samples),
            "peak_vm_swap_bytes": 0,
            "peak_cuda_reserved_bytes": {
                device: max(
                    sample["cuda_peak_reserved_bytes"][device] for sample in samples
                )
                for device in ("cuda:0", "cuda:1")
            },
            "samples": samples,
            "rss_cap_passed": True,
            "cuda_cap_passed": True,
            "zero_swap_passed": True,
        },
        "report_sha256",
    )


def _chart_binding(*, combined: bool):
    source_chart = _sha({"source_chart": True})
    if not combined:
        return (
            _seal(
                {
                    "enabled": False,
                    "mode": "off",
                    "source_layer_charts_sha256": source_chart,
                    "compiler_input_layer_charts_sha256": source_chart,
                    "controller_report_sha256": None,
                    "transport_report_sha256": None,
                    "selection_used_by_controller_solver": None,
                    "validation_rows_read_by_controller": None,
                    "default_off_semantics_preserved": True,
                    "controller_report": None,
                },
                "binding_sha256",
            ),
            source_chart,
        )
    compiler_chart = _sha({"transported_chart": True})
    transport_sha = _sha({"transport": True})
    controller = _seal(
        {
            "schema": "neural-forge-lfm2-cpcf-residual-error-controller-v1",
            "component": "fit_only_post_operator_chart_transport_shadow",
            "layer_index": 2,
            "operator_kind": "full_attention",
            "source_width": 1024,
            "target_width": 776,
            "source_layer_charts_sha256": source_chart,
            "transported_layer_charts_sha256": compiler_chart,
            "transport_report_sha256": transport_sha,
            "selection_read_after_transport_freeze": True,
            "selection_used_by_solver": False,
            "validation_rows_read": 0,
            "validation_used_by_solver": False,
            "fit_only_closed_form_orthogonal_procrustes": True,
            "source_bank_unchanged": True,
            "source_charts_unchanged": True,
            "source_evidence_unchanged": True,
            "post_layer_chart_same_object": True,
            "decoded_state_predictions_invariant": True,
            "feature_predictions_invariant": True,
            "numeric_parent_weight_matching_objective": False,
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "target_layer_forward_calls": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "sparsity_introduced": False,
            "parent_sidecar_required": False,
            "automatic_retry_or_retuning_performed": False,
            "forced_shadow_for_controlled_comparison": True,
            "automatic_accept_reject_controller": False,
        },
        "report_sha256",
    )
    return (
        _seal(
            {
                "enabled": True,
                "mode": "fit_only_post_operator_orthogonal_chart_transport",
                "source_layer_charts_sha256": source_chart,
                "compiler_input_layer_charts_sha256": compiler_chart,
                "controller_report_sha256": controller["report_sha256"],
                "transport_report_sha256": transport_sha,
                "selection_used_by_controller_solver": False,
                "validation_rows_read_by_controller": 0,
                "default_off_semantics_preserved": False,
                "controller_report": controller,
            },
            "binding_sha256",
        ),
        compiler_chart,
    )


def _sequential_binding(*, combined: bool):
    return _seal(
        {
            "enabled": combined,
            "mode": (
                "candidate_local_fit_only_post_layer_transport" if combined else "off"
            ),
            "runtime_matches_independent_invocation_pin": True,
            "layer_call_only_diagnostic": True,
            "authenticated_prefix_full_policy_sha256": H776_FULL_POLICY_SHA256,
            "authenticated_prefix_full_policy_unchanged": True,
            "prefix_runtime_policy_mutated": False,
        },
        "binding_sha256",
    )


def _runtime_implementation():
    return _seal(
        {
            "schema": "test-v5-implementation",
            "ordered_source_files": [
                {
                    "module": "neural_forge.test",
                    "filename": "test.py",
                    "sha256": _sha({"source": True}),
                }
            ],
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
        "implementation_sha256",
    )


def _pins(*, combined: bool, implementation_sha: str):
    return {
        "checkpoint_store_directory": "/sealed/checkpoint/store",
        "checkpoint_key": _sha({"checkpoint": "k2"}),
        "checkpoint_manifest_sha256": _sha({"manifest": "k2"}),
        "expected_trace_sha256": _sha({"trace": True}),
        "expected_runtime_compiler_implementation_sha256": implementation_sha,
        "expected_sequence_source_sha256": _sha({"sequence_source": True}),
        "expected_target_plan_fingerprint": H776_TARGET_PLAN_FINGERPRINT,
        "expected_target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        "expected_selective_catalog_sha256": H776_SELECTIVE_CATALOG_SHA256,
        "expected_selective_source_sha256": H776_SELECTIVE_SOURCE_SHA256,
        "expected_selective_provider_sha256": H776_SELECTIVE_PROVIDER_SHA256,
        "expected_cache_sha256": H776_CACHE_SHA256,
        "expected_token_bridge_sha256": H776_TOKEN_BRIDGE_SHA256,
        "expected_tokenizer_sha256": H776_TOKENIZER_SHA256,
        "expected_source_tied_table_sha256": H776_SOURCE_TIED_TABLE_SHA256,
        "expected_source_state_sha256": H776_SOURCE_STATE_SHA256,
        "expected_source_inventory_sha256": H776_SOURCE_INVENTORY_SHA256,
        "expected_blank_report_sha256": H776_BLANK_REPORT_SHA256,
        "expected_blank_state_sha256": H776_BLANK_STATE_SHA256,
        "expected_full_policy_sha256": H776_FULL_POLICY_SHA256,
        "expected_origin_compiler_sha256": H776_ORIGIN_COMPILER_SHA256,
        "expected_sequence_solver_plan_sha256": (H776_SEQUENCE_SOLVER_PLAN_SHA256),
        "layer_index": 2,
        "expected_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
        "maximum_process_rss_bytes": H776_MAXIMUM_PROCESS_RSS_BYTES,
        "maximum_cuda_bytes_per_device": H776_MAXIMUM_CUDA_BYTES_PER_DEVICE,
        "require_zero_swap": True,
        "expected_attention_sequential_transport_enabled": combined,
    }


def _checkpoint(pins):
    input_bank = _sha({"input_bank": "k2"})
    checkpoint_reference = {
        "artifact_kind": "checkpoint",
        "artifact_key": pins["checkpoint_key"],
        "manifest_sha256": pins["checkpoint_manifest_sha256"],
    }
    return (
        _seal(
            {
                "schema": (LFM2_CPCF_V5_REAL_ATTENTION_CHECKPOINT_AUDIT_SCHEMA),
                "checkpoint_reference": checkpoint_reference,
                "boundary_reference": {
                    "artifact_kind": "boundary",
                    "artifact_key": _sha({"boundary": True}),
                    "manifest_sha256": _sha({"boundary_manifest": True}),
                },
                "next_layer_index": 2,
                "completed_layer_indices": [0, 1],
                "completed_operator_kinds": ["short_conv", "short_conv"],
                "incoming_boundary": "layer.001.post_layer",
                "incoming_state_bank_sha256": input_bank,
                "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
                "full_policy_sha256": H776_FULL_POLICY_SHA256,
                "origin_compiler_sha256": H776_ORIGIN_COMPILER_SHA256,
                "k1_rejected_as_insufficient": True,
                "k3_not_opened": True,
            },
            "audit_sha256",
        ),
        input_bank,
        checkpoint_reference,
    )


def _frontier():
    return _seal(
        {
            "frontier_sha256": _sha({"frontier": True}),
            "context_sha256": _sha({"context": True}),
            "next_layer_index": 2,
            "expected_layer_count": H776_LAYER_COUNT,
            "source_width": 1024,
            "target_width": 776,
            "phase_count": H776_PHASE_COUNT,
            "split_row_counts": {
                "fit": H776_FIT_QUERY_ROWS,
                "selection": 5000,
                "validation": H776_VALIDATION_QUERY_ROWS,
            },
            "layer_operator_kinds": list(H776_OPERATOR_KINDS),
            "cross_runtime_probe_only": True,
            "ordinary_resume_gate_weakened": False,
        },
        "audit_sha256",
    )


def _phase_audit():
    return _seal(
        {
            "phase_count": H776_PHASE_COUNT,
            "trajectory_count": H776_TRAJECTORY_COUNT,
            "phases_per_trajectory": 9,
            "ordered_phase_names": list(_PHASES),
            "split_phase_counts": dict(H776_SPLIT_PHASE_COUNTS),
            "split_trajectory_counts": dict(H776_SPLIT_TRAJECTORY_COUNTS),
            "split_query_rows": {
                "fit": H776_FIT_QUERY_ROWS,
                "selection": 5000,
                "validation": H776_VALIDATION_QUERY_ROWS,
            },
            "complete_prefill_plus_eight_decode_phases": True,
        },
        "audit_sha256",
    )


def _prefix(checkpoint_reference):
    return _seal(
        {
            "schema": "test-prefix-install-v1",
            "checkpoint_reference": checkpoint_reference,
            "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
            "blank_state_sha256_before": H776_BLANK_STATE_SHA256,
            "installed_state_sha256_after": _sha({"installed_prefix": True}),
            "tensor_count": 100,
            "next_layer_index": 2,
            "exact_inventory_verified": True,
            "exact_object_aliases_verified": True,
            "exact_storage_aliases_verified": True,
            "exact_shape_dtype_stride_offset_device_verified": True,
            "tensor_identities_preserved": True,
            "exact_prefix_tensor_bytes_installed": True,
            "loaders_used": 0,
            "parent_evidence_tensors_copied": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "promotion_claimed": False,
        },
        "report_sha256",
    )


def _structural(source, target):
    return _seal(
        {
            "schema": "test-parameter-compaction-v1",
            "source_inventory": source,
            "target_inventory": target,
            "source_unique_parameters": H776_PARENT_UNIQUE_PARAMETERS,
            "target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
            "planned_target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
            "parameters_removed_from_topology": (
                H776_PARENT_UNIQUE_PARAMETERS - H776_TARGET_UNIQUE_PARAMETERS
            ),
            "realized_compression_ratio": H776_PARAMETER_COMPRESSION_RATIO,
            "promotion_compression_ratio": 4.0,
            "promotion_ratio_met": True,
            "valid_lower_ratio_diagnostic_fold": False,
            "parameter_count_compaction_only": True,
            "checkpoint_byte_ratio_used": False,
            "quantization_used": False,
            "pruning_used": False,
        },
        "receipt_sha256",
    )


def _diagnostic(
    *,
    combined: bool,
    regress_phase: bool = False,
    behavioral_improves: bool = True,
):
    implementation = _runtime_implementation()
    pins = _pins(
        combined=combined,
        implementation_sha=implementation["implementation_sha256"],
    )
    checkpoint, input_bank, checkpoint_reference = _checkpoint(pins)
    chart_binding, compiler_chart = _chart_binding(combined=combined)
    layer = _layer_result(
        combined=combined,
        input_bank=input_bank,
        compiler_chart=compiler_chart,
        regress_phase=regress_phase,
        behavioral_improves=behavioral_improves,
    )
    source = _inventory(H776_PARENT_UNIQUE_PARAMETERS)
    target = _inventory(H776_TARGET_UNIQUE_PARAMETERS)
    structural = _structural(source, target)
    probe = _seal(
        {
            "schema": LFM2_CPCF_V5_REAL_ATTENTION_PROBE_SCHEMA,
            "status": "completed_positive_sequence_effective_selection",
            "pins": pins,
            "runtime_compiler_implementation": implementation,
            "runtime_sequence_source_sha256": pins["expected_sequence_source_sha256"],
            "cross_runtime_source_inventory": {
                "checkpoint_origin_inventory_sha256": (H776_SOURCE_INVENTORY_SHA256),
                "live_process_inventory_sha256": _sha({"live_inventory": combined}),
                "origin_digest_reproduction_required": False,
                "reason": "process local identities",
                "live_inventory_unchanged_before_after": True,
            },
            "target_geometry": {
                "hidden_size": 776,
                "intermediate_size": 797,
                "query_heads": 10,
                "key_value_heads": 5,
                "head_dim": 64,
                "layer_count": H776_LAYER_COUNT,
                "layer_types": list(H776_LAYER_TYPES),
                "layer_2_attention_shapes": {
                    "q_proj.weight": [640, 776],
                    "k_proj.weight": [320, 776],
                    "v_proj.weight": [320, 776],
                    "out_proj.weight": [776, 640],
                },
            },
            "parameter_compaction": {
                "parent_unique_parameters": H776_PARENT_UNIQUE_PARAMETERS,
                "target_unique_parameters": H776_TARGET_UNIQUE_PARAMETERS,
                "compression_ratio": H776_PARAMETER_COMPRESSION_RATIO,
                "parent_inventory_sha256": source["inventory_sha256"],
                "target_inventory_sha256": target["inventory_sha256"],
                "quantization_used": False,
                "pruning_used": False,
            },
            "checkpoint_audit": checkpoint,
            "frontier_audit": _frontier(),
            "phase_audit": _phase_audit(),
            "prefix_install_report": _prefix(checkpoint_reference),
            "layer_2_result": layer,
            "post_operator_chart_transport_shadow": chart_binding,
            "attention_sequential_transport": _sequential_binding(combined=combined),
            "resource_report": _resource(combined=combined),
            "stop_conditions": {
                "exactly_one_layer_compiled": True,
                "compiled_layer_indices": [2],
                "next_layer_compilation_attempted": False,
                "k3_checkpoint_published": False,
                "child_checkpoint_saved": False,
                "production_publication_attempted": False,
                "ordinary_resume_gate_weakened": False,
                "parent_trace_recapture_used": False,
                "training_used": False,
                "optimizer_used": False,
                "backward_used": False,
                "pruning_used": False,
                "quantization_used": False,
                "automatic_retry_or_retuning_used": False,
            },
            "promotion_authorized": False,
            "accepted_child_artifact": False,
            "whole_model_intelligence_retention_proven": False,
            "non_promotable_probe_only": True,
        },
        "report_sha256",
    )
    return _seal(
        {
            "schema": ("neural-forge-lfm2-cpcf-v5-real-attention-probe-run-v1"),
            "status": "completed_positive_sequence_effective_selection",
            "probe": probe,
            "runtime_guard": _runtime_guard(),
            "structural_parameter_compaction": structural,
            "parent_parameter_inventory": source,
            "target_parameter_inventory": target,
            "local_parent_io_audit": {
                "model_directory": "/local/model",
                "config_loader": "direct_local_json_to_concrete_lfm2_config",
                "model_loader": "direct_local_safetensors_to_concrete_lfm2_model",
                "tokenizer_loader": "direct_local_tokenizer_json",
                "checkpoint_writer": "direct_local_json_and_safetensors",
                "safetensor_files": ["model.safetensors"],
                "auto_loader_used": False,
                "from_pretrained_used": False,
                "hub_download_used": False,
                "network_allowed": False,
                "hf_hub_offline": True,
                "transformers_offline": True,
            },
            "catalog_preparation_receipt_sha256": _sha({"catalog_preparation": True}),
            "catalog_reauthenticated_after_provider_close": True,
            "post_operator_chart_transport_shadow_enabled": combined,
            "attention_sequential_transport_enabled": combined,
            "diagnostic_only": True,
            "child_checkpoint_saved": False,
            "publication_attempted": False,
            "promotion_authorized": False,
        },
        "receipt_sha256",
    )


def _reseal_upwards(diagnostic):
    probe_body = dict(diagnostic["probe"])
    probe_body.pop("report_sha256")
    diagnostic["probe"] = _seal(probe_body, "report_sha256")
    run_body = dict(diagnostic)
    run_body.pop("receipt_sha256")
    diagnostic.update(_seal(run_body, "receipt_sha256"))


def _reseal_layer_and_upwards(diagnostic):
    layer_body = dict(diagnostic["probe"]["layer_2_result"])
    layer_body.pop("audit_sha256")
    diagnostic["probe"]["layer_2_result"] = _seal(
        layer_body,
        "audit_sha256",
    )
    _reseal_upwards(diagnostic)


def _reseal_resource_and_upwards(diagnostic):
    resource_body = dict(diagnostic["probe"]["resource_report"])
    resource_body.pop("report_sha256")
    diagnostic["probe"]["resource_report"] = _seal(
        resource_body,
        "report_sha256",
    )
    _reseal_upwards(diagnostic)


def _install_continuous_pcg_audit(diagnostic, *, mode: str):
    artifact, audit = _continuous_pcg_audit(mode=mode)
    layer = diagnostic["probe"]["layer_2_result"]
    layer["sequence_effective_artifact"] = artifact
    layer["sequence_solver_plan"] = artifact["solver_plan"]
    layer["pcg_convergence"] = audit
    _reseal_layer_and_upwards(diagnostic)
    return audit


def _reseal_pcg_audit_and_upwards(diagnostic):
    audit = diagnostic["probe"]["layer_2_result"]["pcg_convergence"]
    body = dict(audit)
    body.pop("audit_sha256")
    diagnostic["probe"]["layer_2_result"]["pcg_convergence"] = _seal(
        body,
        "audit_sha256",
    )
    _reseal_layer_and_upwards(diagnostic)


def test_positive_pair_is_deterministic_and_fully_sealed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    first = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        baseline,
        combined,
    )
    second = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        baseline,
        combined,
    )
    assert first.passed is True
    assert first.as_dict() == second.as_dict()
    assert (
        first.report["schema"] == LFM2_CPCF_V5_REAL_ATTENTION_CAUSAL_PAIR_VERDICT_SCHEMA
    )
    assert (
        first.report["behavioral_selection_comparison"][
            "candidate_id_tie_break_excluded"
        ]
        is True
    )
    assert all(first.report["gates"].values())
    body = first.as_dict()
    seal = body.pop("report_sha256")
    assert seal == _sha(body)


def test_real_collector_hidden_partial_trace_pair_is_accepted() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    baseline_audit = _install_continuous_pcg_audit(
        baseline,
        mode="hidden",
    )
    combined_audit = _install_continuous_pcg_audit(
        combined,
        mode="hidden",
    )
    assert baseline_audit["continuous_hidden_partial_trace_solver_event_count"] == 5
    assert combined_audit["stale_row_block_event_count"] == 0
    verdict = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        baseline,
        combined,
    )
    assert verdict.passed is True


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "hidden_partial_trace_pcg_execution_profile_sha256",
            _sha({"substituted_execution_profile": True}),
        ),
        ("continuous_hidden_partial_trace_solver_event_count", 4),
        ("all_continuous_hidden_partial_trace_lanes_authenticated", False),
        ("all_hidden_partial_trace_preconditioner_receipts_authenticated", False),
        ("stale_row_block_event_count", 1),
    ),
)
def test_resealed_hidden_partial_trace_audit_tamper_fails_closed(
    field: str,
    value,
) -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    _install_continuous_pcg_audit(baseline, mode="hidden")
    _install_continuous_pcg_audit(combined, mode="hidden")
    combined["probe"]["layer_2_result"]["pcg_convergence"][field] = value
    _reseal_pcg_audit_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="hidden partial-trace lanes",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_real_collector_archived_row_block_pair_remains_accepted() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    baseline_audit = _install_continuous_pcg_audit(
        baseline,
        mode="row",
    )
    combined_audit = _install_continuous_pcg_audit(
        combined,
        mode="row",
    )
    assert baseline_audit["continuous_row_block_solver_event_count"] == 5
    assert "stale_row_block_event_count" not in combined_audit
    verdict = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        baseline,
        combined,
    )
    assert verdict.passed is True


def test_real_collector_hidden_audit_rejects_mixed_stale_row_execution() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    _install_continuous_pcg_audit(baseline, mode="hidden")
    mixed_audit = _install_continuous_pcg_audit(combined, mode="mixed")
    assert mixed_audit["continuous_hidden_partial_trace_solver_event_count"] == 4
    assert mixed_audit["continuous_row_block_solver_event_count"] == 1
    assert mixed_audit["stale_row_block_event_count"] == 1
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="hidden partial-trace lanes",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_valid_phase_regression_returns_sealed_negative_verdict() -> None:
    verdict = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        _diagnostic(combined=False),
        _diagnostic(combined=True, regress_phase=True),
    )
    assert verdict.passed is False
    gates = verdict.report["gates"]
    assert gates["every_phase_relative_rmse_nonregression"] is False
    assert gates["every_phase_cosine_nonregression"] is False
    assert verdict.report["verdict"] == ("authenticated_pair_failed_scientific_gates")


def test_behavioral_key_tie_cannot_pass_via_candidate_id() -> None:
    verdict = verify_lfm2_cpcf_v5_real_attention_causal_pair(
        _diagnostic(combined=False),
        _diagnostic(combined=True, behavioral_improves=False),
    )
    assert verdict.passed is False
    assert (
        verdict.report["gates"]["combined_behavioral_selection_key_strictly_improved"]
        is False
    )


def test_resealed_validation_leakage_tamper_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    controller = combined["probe"]["post_operator_chart_transport_shadow"][
        "controller_report"
    ]
    controller_body = dict(controller)
    controller_body.pop("report_sha256")
    controller_body["validation_rows_read"] = 1
    controller = _seal(controller_body, "report_sha256")
    binding = combined["probe"]["post_operator_chart_transport_shadow"]
    binding_body = dict(binding)
    binding_body.pop("binding_sha256")
    binding_body["controller_report"] = controller
    binding_body["controller_report_sha256"] = controller["report_sha256"]
    combined["probe"]["post_operator_chart_transport_shadow"] = _seal(
        binding_body,
        "binding_sha256",
    )
    _reseal_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="fit-only post-operator",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_stale_nested_phase_seal_is_rejected_despite_outer_reseals() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    layer = combined["probe"]["layer_2_result"]
    validation = layer["full_validation_metrics"]
    validation["phase_metrics"]["decode.004"]["post_layer"]["relative_rmse"] -= 0.01
    validation_body = dict(validation)
    validation_body.pop("report_sha256")
    layer["full_validation_metrics"] = _seal(
        validation_body,
        "report_sha256",
    )
    layer_body = dict(layer)
    layer_body.pop("audit_sha256")
    combined["probe"]["layer_2_result"] = _seal(
        layer_body,
        "audit_sha256",
    )
    _reseal_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="phase_metrics",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_common_evidence_mismatch_is_not_a_causal_pair() -> None:
    baseline = _diagnostic(combined=False)
    combined = copy.deepcopy(_diagnostic(combined=True))
    combined["probe"]["pins"]["expected_trace_sha256"] = _sha({"other_trace": True})
    _reseal_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="not one causal pair",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


@pytest.mark.parametrize(
    "phase",
    (None, "decode.006"),
)
def test_resealed_validation_target_mismatch_fails_closed(phase) -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    validation = combined["probe"]["layer_2_result"]["full_validation_metrics"]
    if phase is None:
        validation["post_layer"]["target_sha256"] = _sha(
            {"tampered_aggregate_target": True}
        )
    else:
        phase_report = validation["phase_metrics"][phase]
        phase_report["post_layer"]["target_sha256"] = _sha(
            {"tampered_phase_target": phase}
        )
        phase_body = dict(phase_report)
        phase_body.pop("report_sha256")
        validation["phase_metrics"][phase] = _seal(
            phase_body,
            "report_sha256",
        )
    validation_body = dict(validation)
    validation_body.pop("report_sha256")
    combined["probe"]["layer_2_result"]["full_validation_metrics"] = _seal(
        validation_body,
        "report_sha256",
    )
    _reseal_layer_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="validation_target_sha256_inventory_equal",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_selected_sequential_map_digest_mismatch_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    transport = combined["probe"]["layer_2_result"]["attention_sequential_transport"]
    transport["candidate_effective_boundary_charts_sha256"][
        H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    ] = _sha({"different_selected_chart": True})
    transport_body = dict(transport)
    transport_body.pop("audit_sha256")
    combined["probe"]["layer_2_result"]["attention_sequential_transport"] = _seal(
        transport_body, "audit_sha256"
    )
    _reseal_layer_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="selected chart or bundle",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_pcg_artifact_frontier_mismatch_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    artifact = combined["probe"]["layer_2_result"]["sequence_effective_artifact"]
    replacement = _sha({"different_value_solve": True})
    artifact["candidate_frontier"][1]["value_solve_report_sha256"][0] = replacement
    artifact["solve_report_sha256"][2] = replacement
    artifact["candidate_frontier_sha256"] = _sha(artifact["candidate_frontier"])
    artifact_body = dict(artifact)
    artifact_body.pop("report_sha256")
    combined["probe"]["layer_2_result"]["sequence_effective_artifact"] = _seal(
        artifact_body, "report_sha256"
    )
    _reseal_layer_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match=r"collector\.finalize",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_sequence_artifact_training_step_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    artifact = combined["probe"]["layer_2_result"]["sequence_effective_artifact"]
    artifact["execution"]["training_steps"] = 1
    artifact_body = dict(artifact)
    artifact_body.pop("report_sha256")
    combined["probe"]["layer_2_result"]["sequence_effective_artifact"] = _seal(
        artifact_body,
        "report_sha256",
    )
    _reseal_layer_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match=r"execution\.training_steps",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_sequence_artifact_producer_state_mutation_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    artifact = combined["probe"]["layer_2_result"]["sequence_effective_artifact"]
    artifact["producer_state_sha256_after"] = _sha({"mutated_producer_state": True})
    artifact_body = dict(artifact)
    artifact_body.pop("report_sha256")
    combined["probe"]["layer_2_result"]["sequence_effective_artifact"] = _seal(
        artifact_body,
        "report_sha256",
    )
    _reseal_layer_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="sealed fit-only sequence-effective artifact",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


@pytest.mark.parametrize(
    "device_attack",
    ("zero_device", "missing_device", "extra_device"),
)
def test_resealed_cuda_device_inventory_attacks_fail_closed(
    device_attack,
) -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    resource = combined["probe"]["resource_report"]
    for index, sample in enumerate(resource["samples"]):
        cuda = sample["cuda_peak_reserved_bytes"]
        if device_attack == "zero_device":
            cuda.clear()
        elif device_attack == "missing_device":
            cuda.pop("cuda:1")
        else:
            cuda["cuda:2"] = 500 + index
    devices = sorted(
        {
            device
            for sample in resource["samples"]
            for device in sample["cuda_peak_reserved_bytes"]
        }
    )
    resource["peak_cuda_reserved_bytes"] = {
        device: max(
            sample["cuda_peak_reserved_bytes"][device] for sample in resource["samples"]
        )
        for device in devices
    }
    _reseal_resource_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="resource monitoring",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_zero_resource_sample_interval_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    combined["probe"]["resource_report"]["sample_interval_seconds"] = 0.0
    _reseal_resource_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="resource monitoring",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_resealed_missing_periodic_resource_sample_fails_closed() -> None:
    baseline = _diagnostic(combined=False)
    combined = _diagnostic(combined=True)
    resource = combined["probe"]["resource_report"]
    resource["samples"] = [
        sample for sample in resource["samples"] if sample["label"] != "periodic"
    ]
    resource["sample_count"] = len(resource["samples"])
    resource["peak_vm_rss_bytes"] = max(
        sample["process"]["VmRSS"] for sample in resource["samples"]
    )
    resource["peak_vm_hwm_bytes"] = max(
        sample["process"]["VmHWM"] for sample in resource["samples"]
    )
    resource["peak_cuda_reserved_bytes"] = {
        device: max(
            sample["cuda_peak_reserved_bytes"][device] for sample in resource["samples"]
        )
        for device in ("cuda:0", "cuda:1")
    }
    _reseal_resource_and_upwards(combined)
    with pytest.raises(
        Lfm2CpcfV5RealAttentionCausalPairError,
        match="periodic resource sample",
    ):
        verify_lfm2_cpcf_v5_real_attention_causal_pair(
            baseline,
            combined,
        )


def test_cli_reads_pair_and_emits_create_new_sealed_verdict(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    combined = tmp_path / "combined.json"
    verdict = tmp_path / "verdict.json"
    baseline.write_text(
        json.dumps(_diagnostic(combined=False)),
        encoding="utf-8",
    )
    combined.write_text(
        json.dumps(_diagnostic(combined=True)),
        encoding="utf-8",
    )
    argv = [
        "--baseline-diagnostic-json",
        str(baseline),
        "--combined-diagnostic-json",
        str(combined),
        "--verdict-json",
        str(verdict),
    ]
    assert verifier_cli.main(argv) == 0
    written = json.loads(verdict.read_text(encoding="utf-8"))
    assert written["passed"] is True
    seal = written.pop("report_sha256")
    assert seal == _sha(written)
    assert verifier_cli.main(argv) == 2
