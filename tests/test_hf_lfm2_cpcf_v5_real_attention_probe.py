from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest


torch = pytest.importorskip("torch")

from experiments import (  # noqa: E402
    probe_lfm2_cpcf_v5_real_attention_layer as runner,
)
from neural_forge import (  # noqa: E402
    hf_lfm2_sequence_effective_attention_fold as sequence_fold,
)
from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (  # noqa: E402
    H776_ATTENTION_CANDIDATE_IDS,
    H776_FULL_POLICY_SHA256,
    H776_ORIGIN_COMPILER_SHA256,
    H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    H776_SEQUENCE_SOLVER_PLAN_SHA256,
    H776_SPLIT_PHASE_COUNTS,
    H776_TARGET_TOPOLOGY_SHA256,
    Lfm2CpcfV5RealAttentionProbeError,
    Lfm2CpcfV5RealAttentionProbePins,
    Lfm2CpcfV5RealAttentionProbeRuntime,
    Lfm2CpcfV5RealAttentionResourceMonitor,
    Lfm2SequenceEffectivePcgCollector,
    _sha256,
    authenticate_h776_attention_result,
    authenticate_h776_boundary_frontier,
    authenticate_h776_k2_checkpoint,
    authenticate_h776_phase_inventory,
)


_DIGEST = "1" * 64
_OTHER_DIGEST = "2" * 64


def _pins(
    root: Path,
    *,
    attention_sequential_transport_enabled: bool = False,
) -> Lfm2CpcfV5RealAttentionProbePins:
    return Lfm2CpcfV5RealAttentionProbePins(
        checkpoint_store_directory=root,
        checkpoint_key=_DIGEST,
        checkpoint_manifest_sha256=_DIGEST,
        expected_trace_sha256=_DIGEST,
        expected_runtime_compiler_implementation_sha256=_OTHER_DIGEST,
        expected_sequence_source_sha256=_DIGEST,
        expected_attention_sequential_transport_enabled=(
            attention_sequential_transport_enabled
        ),
    )


def test_probe_pins_are_exact_and_cross_runtime(tmp_path: Path) -> None:
    pins = _pins(tmp_path)
    assert pins.layer_index == 2
    assert pins.maximum_process_rss_bytes == 64 * 1024**3
    assert pins.maximum_cuda_bytes_per_device == 20 * 1024**3
    assert (
        pins.expected_runtime_compiler_implementation_sha256
        != pins.expected_origin_compiler_sha256
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(pins, layer_index=1)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(
            pins,
            expected_runtime_compiler_implementation_sha256=(
                H776_ORIGIN_COMPILER_SHA256
            ),
        )


def test_probe_runtime_chart_transport_shadow_defaults_off() -> None:
    runtime = Lfm2CpcfV5RealAttentionProbeRuntime(
        source_model=object(),
        target_model=object(),
        trace_adapter=object(),
        evidence_provider=object(),
        source_tied_table=object(),
        blank_target_report={},
        target_plan_fingerprint=_DIGEST,
        runtime_policy=object(),
    )
    assert runtime.post_operator_chart_transport_shadow is False
    assert runtime.attention_sequential_transport_enabled is False


class _Reference:
    def __init__(self, pins: Lfm2CpcfV5RealAttentionProbePins) -> None:
        self._pins = pins

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_kind": "checkpoint",
            "artifact_key": self._pins.checkpoint_key,
            "manifest_sha256": self._pins.checkpoint_manifest_sha256,
        }


class _Store:
    def final_reauthenticate_checkpoint(self, verified):
        return verified


def _checkpoint(
    pins: Lfm2CpcfV5RealAttentionProbePins,
    *,
    next_layer_index: int = 2,
):
    first_bank = "3" * 64
    second_bank = "4" * 64
    execution = {
        "closed_form_compilers": True,
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
    }
    receipts = [
        {
            "layer_index": 0,
            "operator_kind": "short_conv",
            "input_bank_sha256": "5" * 64,
            "next_bank_sha256": first_bank,
            "execution": dict(execution),
        },
        {
            "layer_index": 1,
            "operator_kind": "short_conv",
            "input_bank_sha256": first_bank,
            "next_bank_sha256": second_bank,
            "execution": dict(execution),
        },
    ]
    bindings = {
        "next_layer_index": next_layer_index,
        "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        "full_policy_sha256": H776_FULL_POLICY_SHA256,
        "compiler_fingerprint_sha256": H776_ORIGIN_COMPILER_SHA256,
        "boundary": {
            "artifact_kind": "boundary",
            "artifact_key": "boundary-k2",
            "manifest_sha256": "6" * 64,
        },
    }
    completed = {
        "next_layer_index": next_layer_index,
        "incoming_state_layer_index": next_layer_index,
        "completed_layer_indices": [0, 1],
        "completed_layer_receipts": receipts,
        "incoming_state_bank_metadata": {
            "layer_index": next_layer_index,
            "boundary": "layer.001.post_layer",
            "actual_folded_upstream_states": True,
            "state_bank_sha256": second_bank,
        },
        "target_topology_snapshot": {
            "model_type": "lfm2",
            "layer_count": 14,
            "topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
        },
        "target_topology_sha256": H776_TARGET_TOPOLOGY_SHA256,
    }
    return SimpleNamespace(
        reference=_Reference(pins),
        manifest={"bindings": bindings},
        completed_layers=completed,
    )


def test_k2_checkpoint_accepts_only_contiguous_two_layer_prefix(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    audit = authenticate_h776_k2_checkpoint(
        _Store(),
        _checkpoint(pins),
        pins,
    )
    assert audit["next_layer_index"] == 2
    assert audit["completed_layer_indices"] == [0, 1]
    assert audit["k1_rejected_as_insufficient"] is True
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_k2_checkpoint(
            _Store(),
            _checkpoint(pins, next_layer_index=1),
            pins,
        )


def _frontier(pins: Lfm2CpcfV5RealAttentionProbePins):
    context = {
        "source_model_state_sha256": pins.expected_source_state_sha256,
        "source_model_inventory_sha256": pins.expected_source_inventory_sha256,
        "cache_sha256": pins.expected_cache_sha256,
        "selective_catalog_sha256": pins.expected_selective_catalog_sha256,
        "selective_source_sha256": pins.expected_selective_source_sha256,
        "selective_provider_sha256": pins.expected_selective_provider_sha256,
        "trace_sha256": pins.expected_trace_sha256,
        "tokenizer_sha256": pins.expected_tokenizer_sha256,
        "token_bridge_sha256": pins.expected_token_bridge_sha256,
        "source_tied_table_sha256": pins.expected_source_tied_table_sha256,
        "blank_target_report_sha256": pins.expected_blank_report_sha256,
        "blank_target_state_sha256": pins.expected_blank_state_sha256,
        "target_topology_sha256": pins.expected_target_topology_sha256,
        "planned_target_unique_parameters": 57_405_368,
        "runtime_policy_sha256": pins.expected_full_policy_sha256,
        "compiler_implementation_sha256": pins.expected_origin_compiler_sha256,
        "context_sha256": "9" * 64,
    }
    return SimpleNamespace(
        report={
            "frontier_sha256": "a" * 64,
            "context": context,
            "next_layer_index": 2,
            "expected_layer_count": 14,
            "source_width": 1_024,
            "target_width": 776,
            "ordered_phase_sha256": ["b" * 64] * 2_304,
            "split_row_counts": {
                "fit": 40_656,
                "selection": 1,
                "validation": 13_199,
            },
            "layer_operator_kinds": [
                "short_conv",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
                "full_attention",
                "short_conv",
            ],
            "completed_layer_charts": [
                {"operator_kind": "short_conv"},
                {"operator_kind": "short_conv"},
            ],
        }
    )


def test_boundary_frontier_preserves_origin_policy_and_compiler(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    audit = authenticate_h776_boundary_frontier(_frontier(pins), pins)
    assert audit["next_layer_index"] == 2
    assert audit["ordinary_resume_gate_weakened"] is False
    changed = _frontier(pins)
    changed.report["context"]["compiler_implementation_sha256"] = (
        pins.expected_runtime_compiler_implementation_sha256
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_boundary_frontier(changed, pins)


def _complete_phases():
    names = (
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
    split_trajectory_counts = {"fit": 154, "selection": 52, "validation": 50}
    phases = []
    ordinal = 0
    for split, count in split_trajectory_counts.items():
        for _ in range(count):
            for name in names:
                phases.append(
                    SimpleNamespace(
                        batch_index=ordinal,
                        record_id=f"record-{ordinal}",
                        split=split,
                        phase=name,
                        batch_size=1,
                        sampled_suffix_tokens=1,
                    )
                )
            ordinal += 1
    fit_extra = 40_656 - H776_SPLIT_PHASE_COUNTS["fit"]
    validation_extra = 13_199 - H776_SPLIT_PHASE_COUNTS["validation"]
    phases[0].sampled_suffix_tokens += fit_extra
    validation_start = (
        H776_SPLIT_PHASE_COUNTS["fit"] + H776_SPLIT_PHASE_COUNTS["selection"]
    )
    phases[validation_start].sampled_suffix_tokens += validation_extra
    rows = {
        split: sum(
            phase.batch_size * phase.sampled_suffix_tokens
            for phase in phases
            if phase.split == split
        )
        for split in split_trajectory_counts
    }
    frontier = SimpleNamespace(report={"split_row_counts": rows})
    return phases, frontier


def test_phase_inventory_requires_all_trajectories_and_decode_steps() -> None:
    phases, frontier = _complete_phases()
    audit = authenticate_h776_phase_inventory(phases, frontier=frontier)
    assert audit["phase_count"] == 2_304
    assert audit["trajectory_count"] == 256
    assert audit["complete_prefill_plus_eight_decode_phases"] is True
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_phase_inventory(phases[:-1], frontier=frontier)


def _pcg_event(
    role: str = "sequence_effective_structured_value_iteration_1_group_0",
):
    events = []
    right = torch.tensor([1.0, 2.0], dtype=torch.float64)
    diagonal = torch.tensor([2.0, 3.0], dtype=torch.float64)

    def matvec(value):
        return diagonal * value

    with sequence_fold.observe_lfm2_sequence_effective_pcg(events.append):
        observed_solution, observed_report = sequence_fold._pcg_single_tensor(
            right=right,
            matvec=matvec,
            diagonal=diagonal,
            initial=torch.zeros_like(right),
            role=role,
        )
    plain_solution, plain_report = sequence_fold._pcg_single_tensor(
        right=right,
        matvec=matvec,
        diagonal=diagonal,
        initial=torch.zeros_like(right),
        role=role,
    )
    assert torch.equal(observed_solution, plain_solution)
    assert observed_report == plain_report
    assert len(events) == 1
    return events[0]


def _restarted_pcg_event(
    role: str = "sequence_effective_structured_value_iteration_1_group_0",
    *,
    original_lane_iterations: int = 64,
):
    history = [2e-8] * original_lane_iterations + [5e-9]
    checkpoints = [
        {
            "iteration": original_lane_iterations,
            "relative_normal_residual": 2e-8,
            "finite": True,
        },
        {
            "iteration": original_lane_iterations + 1,
            "relative_normal_residual": 5e-9,
            "finite": True,
        },
    ]
    schedule = [
        {
            "restart_iteration": original_lane_iterations,
            "planned_stop_iteration": min(
                original_lane_iterations + 64,
                512,
            ),
            "iterations_executed": 1,
            "ending_iteration": original_lane_iterations + 1,
            "attempted_matvec_calls": 2,
            "termination": "exact_tolerance_reached",
            "ending_recurrence_relative_normal_residual": 5e-9,
            "ending_exact_relative_normal_residual": 5e-9,
            "ending_exact_residual_authenticated": True,
        }
    ]
    report_body = {
        "role": role,
        "solver": "deterministic_restarted_jacobi_pcg_single_tensor",
        "maximum_iterations": 512,
        "original_lane_maximum_iterations": 64,
        "original_lane_iterations_executed": original_lane_iterations,
        "restart_block_iterations": 64,
        "iterations_executed": original_lane_iterations + 1,
        "required_relative_normal_residual": 1e-8,
        "initial_relative_normal_residual": 1.0,
        "original_lane_final_recurrence_relative_normal_residual": 2e-8,
        "final_recurrence_relative_normal_residual": 5e-9,
        "final_relative_normal_residual": 5e-9,
        "relative_normal_residual_history": history,
        "recurrence_history_sha256": _sha256(history),
        "exact_relative_normal_residual_checkpoints": checkpoints,
        "exact_residual_checkpoints_sha256": _sha256(checkpoints),
        "restart_schedule": schedule,
        "restart_schedule_sha256": _sha256(schedule),
        "total_matvec_calls": original_lane_iterations + 4,
        "exact_final_residual_authenticated": True,
        "extended_after_original_lane_exact_authentication_miss": True,
        "solution_finite": True,
        "solution_sha256": _sha256({"solution": role}),
    }
    report = {
        **report_body,
        "report_sha256": _sha256(report_body),
    }
    observation_body = {
        "schema": ("neural-forge-lfm2-sequence-effective-pcg-observation-v1"),
        "solver_kind": "single_tensor",
        "report": report,
        "report_sha256": report["report_sha256"],
        "solver_result_or_arguments_modified": False,
    }
    return {
        **observation_body,
        "observation_sha256": _sha256(observation_body),
    }


def _continuous_hidden_partial_trace_pcg_event(
    role: str = "sequence_effective_structured_value_iteration_1_group_0",
):
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
    group = int(role.rsplit("_", 1)[-1])
    group_heads = (2 * group, 2 * group + 1)
    source_terms = [
        {
            "head": head,
            "other_head": other_head,
            "left_sha256": _sha256(
                {"left": role, "head": head, "other_head": other_head}
            ),
            "coupling_sha256": _sha256(
                {"coupling": role, "head": head, "other_head": other_head}
            ),
            "partial_trace_weight": 0.25 * (head + other_head + 1),
        }
        for head in group_heads
        for other_head in group_heads
    ]
    construction_body = {
        "schema": (
            "neural-forge-lfm2-sequence-effective-"
            "hidden-partial-trace-cholesky-v1"
        ),
        "role": role,
        "construction": "normalized_hidden_side_partial_trace",
        "formula": (
            "P_H=ridge*I_H+sum_(h,o)"
            "(trace(C_o,h)/head_dim)*A_h,o"
        ),
        "ridge": 1e-6,
        "hidden_size": 776,
        "head_dim": 64,
        "partial_trace_shape": [776, 776],
        "partial_trace_dtype": "torch.float64",
        "partial_trace_device": "cpu",
        "partial_trace_sha256": _sha256({"partial_trace": role}),
        "partial_trace_tensor_bytes": 776 * 776 * 8,
        "source_term_count": 4,
        "source_terms": source_terms,
        "source_terms_sha256": _sha256(source_terms),
        "raw_maximum_absolute_asymmetry": 0.0,
        "symmetry_tolerance": 1e-12,
        "symmetric_within_numerical_tolerance": True,
        "symmetrization": "half_sum_with_transpose",
        "full_operator_materialized": False,
    }
    construction = {
        **construction_body,
        "report_sha256": _sha256(construction_body),
    }
    receipt_body = {
        "kind": "deterministic_hidden_partial_trace_cholesky",
        "construction_report": construction,
        "construction_report_sha256": construction["report_sha256"],
        "partial_trace_sha256": construction["partial_trace_sha256"],
        "factorization": "lower_cholesky_ex",
        "factorization_info": 0,
        "factor_sha256": _sha256({"factor": role}),
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
    }
    receipt = {
        **receipt_body,
        "report_sha256": _sha256(receipt_body),
    }
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
    report_body = {
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
        "recurrence_history_sha256": _sha256(history),
        "exact_relative_normal_residual_checkpoints": checkpoints,
        "exact_residual_checkpoints_sha256": _sha256(checkpoints),
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
        "solution_sha256": _sha256({"solution": role}),
    }
    report = {
        **report_body,
        "report_sha256": _sha256(report_body),
    }
    observation_body = {
        "schema": ("neural-forge-lfm2-sequence-effective-pcg-observation-v1"),
        "solver_kind": "single_tensor",
        "report": report,
        "report_sha256": report["report_sha256"],
        "solver_result_or_arguments_modified": False,
    }
    return {
        **observation_body,
        "observation_sha256": _sha256(observation_body),
    }


def _continuous_row_block_pcg_event(
    role: str = "sequence_effective_structured_value_iteration_1_group_0",
):
    """Archived v1 receipt retained as an exact compatibility fixture."""

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
    construction_body = {
        "schema": "neural-forge-lfm2-sequence-effective-row-block-jacobi-v1",
        "role": role,
        "construction": "exact_structured_operator_row_blocks",
        "row_block_count": 776,
        "row_block_size": 64,
        "row_blocks_shape": [776, 64, 64],
        "row_blocks_dtype": "torch.float64",
        "row_blocks_sha256": _sha256({"row_blocks": role}),
        "scalar_diagonal_sha256": _sha256({"diagonal": role}),
        "scalar_diagonal_maximum_absolute_error": 0.0,
        "scalar_diagonal_match_tolerance": 1e-12,
        "row_blocks_maximum_asymmetry": 0.0,
        "row_blocks_symmetry_tolerance": 1e-12,
        "source_term_count": 4,
        "source_terms_sha256": _sha256({"source_terms": role}),
        "ridge": 1e-6,
        "full_operator_materialized": False,
    }
    construction = {
        **construction_body,
        "report_sha256": _sha256(construction_body),
    }
    receipt_body = {
        "kind": "deterministic_exact_row_block_jacobi_cholesky",
        "construction_report": construction,
        "construction_report_sha256": construction["report_sha256"],
        "row_blocks_sha256": construction["row_blocks_sha256"],
        "factorization": "batched_lower_cholesky_ex",
        "factor_sha256": _sha256({"factor": role}),
        "row_block_count": 776,
        "row_block_size": 64,
        "all_row_blocks_spd": True,
        "factorization_materialized_only_after_original_lane_exact_miss": True,
        "full_operator_materialized": False,
    }
    receipt = {
        **receipt_body,
        "report_sha256": _sha256(receipt_body),
    }
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
    report_body = {
        "role": role,
        "solver": "deterministic_continuous_row_block_jacobi_pcg_single_tensor",
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
        "recurrence_history_sha256": _sha256(history),
        "exact_relative_normal_residual_checkpoints": checkpoints,
        "exact_residual_checkpoints_sha256": _sha256(checkpoints),
        "total_matvec_calls": 70,
        "exact_final_residual_authenticated": True,
        "extended_after_original_lane_exact_authentication_miss": True,
        "exact_residual_used_to_initialize_continuous_lane_once": True,
        "exact_diagnostic_matvecs_modify_recurrence_state_or_direction": False,
        "continuous_row_block_lane": lane,
        "restart_preconditioner": preconditioner,
        "solution_finite": True,
        "solution_sha256": _sha256({"solution": role}),
    }
    report = {
        **report_body,
        "report_sha256": _sha256(report_body),
    }
    observation_body = {
        "schema": "neural-forge-lfm2-sequence-effective-pcg-observation-v1",
        "solver_kind": "single_tensor",
        "report": report,
        "report_sha256": report["report_sha256"],
        "solver_result_or_arguments_modified": False,
    }
    return {
        **observation_body,
        "observation_sha256": _sha256(observation_body),
    }


def _reseal_pcg_event(event):
    report_body = dict(event["report"])
    report_body.pop("report_sha256", None)
    event["report"] = {
        **report_body,
        "report_sha256": _sha256(report_body),
    }
    event["report_sha256"] = event["report"]["report_sha256"]
    observation_body = dict(event)
    observation_body.pop("observation_sha256", None)
    event["observation_sha256"] = _sha256(observation_body)
    return event


def _reseal_continuous_hidden_partial_trace_pcg_event(event):
    report = event["report"]
    preconditioner = report["restart_preconditioner"]
    receipt = preconditioner["receipt"]
    construction_body = dict(receipt["construction_report"])
    construction_body.pop("report_sha256", None)
    construction = {
        **construction_body,
        "report_sha256": _sha256(construction_body),
    }
    receipt["construction_report"] = construction
    receipt["construction_report_sha256"] = construction["report_sha256"]
    receipt["partial_trace_sha256"] = construction["partial_trace_sha256"]
    receipt_body = dict(receipt)
    receipt_body.pop("report_sha256", None)
    preconditioner["receipt"] = {
        **receipt_body,
        "report_sha256": _sha256(receipt_body),
    }
    report["recurrence_history_sha256"] = _sha256(
        report["relative_normal_residual_history"]
    )
    report["exact_residual_checkpoints_sha256"] = _sha256(
        report["exact_relative_normal_residual_checkpoints"]
    )
    return _reseal_pcg_event(event)


def _reseal_continuous_row_block_pcg_event(event):
    report = event["report"]
    preconditioner = report["restart_preconditioner"]
    receipt = preconditioner["receipt"]
    construction_body = dict(receipt["construction_report"])
    construction_body.pop("report_sha256", None)
    construction = {
        **construction_body,
        "report_sha256": _sha256(construction_body),
    }
    receipt["construction_report"] = construction
    receipt["construction_report_sha256"] = construction["report_sha256"]
    receipt["row_blocks_sha256"] = construction["row_blocks_sha256"]
    receipt_body = dict(receipt)
    receipt_body.pop("report_sha256", None)
    preconditioner["receipt"] = {
        **receipt_body,
        "report_sha256": _sha256(receipt_body),
    }
    report["recurrence_history_sha256"] = _sha256(
        report["relative_normal_residual_history"]
    )
    report["exact_residual_checkpoints_sha256"] = _sha256(
        report["exact_relative_normal_residual_checkpoints"]
    )
    return _reseal_pcg_event(event)


def test_scoped_pcg_observation_does_not_change_solver() -> None:
    event = _pcg_event()
    assert event["solver_result_or_arguments_modified"] is False
    right = torch.tensor([1.0], dtype=torch.float64)

    def mutate_detached_event(observed):
        observed["report"]["role"] = "observer-mutated-copy"

    with sequence_fold.observe_lfm2_sequence_effective_pcg(mutate_detached_event):
        _solution, returned = sequence_fold._pcg_single_tensor(
            right=right,
            matvec=lambda value: 2.0 * value,
            diagonal=torch.tensor([2.0], dtype=torch.float64),
            initial=torch.zeros_like(right),
            role="returned-report-stays-authentic",
        )
    assert returned["role"] == "returned-report-stays-authentic"
    with sequence_fold.observe_lfm2_sequence_effective_pcg(lambda _event: None):
        with pytest.raises(sequence_fold.Lfm2SequenceEffectiveAttentionFoldError):
            with sequence_fold.observe_lfm2_sequence_effective_pcg(lambda _event: None):
                pass


def test_pcg_collector_accepts_exact_five_group_sweeps() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _pcg_event(f"sequence_effective_structured_value_iteration_1_group_{group}")
        )
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["single_tensor_solve_count"] == 5
    assert audit["multi_rhs_solve_count"] == 0
    assert audit["expected_roles"] == audit["observed_roles"]
    duplicate = Lfm2SequenceEffectivePcgCollector()
    event = _pcg_event()
    duplicate(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        duplicate(event)
    incomplete = Lfm2SequenceEffectivePcgCollector()
    for group in range(4):
        incomplete(
            _pcg_event(f"sequence_effective_structured_value_iteration_1_group_{group}")
        )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        incomplete.finalize(sequence_artifact=artifact)


def test_pcg_collector_authenticates_restarted_exact_residual_lane() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _restarted_pcg_event(
                f"sequence_effective_structured_value_iteration_1_group_{group}"
            )
        )
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["all_iterations_at_most"] == 512
    assert audit["restarted_solver_event_count"] == 5
    assert audit["legacy_solver_event_count"] == 0
    assert audit["all_extended_final_residuals_exactly_recomputed"] is True
    assert audit["all_restart_schedules_authenticated"] is True
    assert audit["all_matvec_counts_authenticated"] is True

    early = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        early(
            _restarted_pcg_event(
                f"sequence_effective_structured_value_iteration_1_group_{group}",
                original_lane_iterations=(7 if group == 0 else 64),
            )
        )
    early_audit = early.finalize(sequence_artifact=artifact)
    assert early_audit["restarted_solver_event_count"] == 5

    mixed = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        mixed(_pcg_event(role) if group == 0 else _restarted_pcg_event(role))
    mixed_audit = mixed.finalize(sequence_artifact=artifact)
    assert mixed_audit["legacy_solver_event_count"] == 1
    assert mixed_audit["restarted_solver_event_count"] == 4


def test_pcg_collector_authenticates_continuous_hidden_partial_trace_lane() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _continuous_hidden_partial_trace_pcg_event(
                f"sequence_effective_structured_value_iteration_1_group_{group}"
            )
        )
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["all_iterations_at_most"] == 4096
    assert audit["legacy_solver_event_count"] == 0
    assert audit["restarted_solver_event_count"] == 0
    assert audit["continuous_hidden_partial_trace_solver_event_count"] == 5
    assert (
        audit["all_continuous_hidden_partial_trace_lanes_authenticated"]
        is True
    )
    assert (
        audit[
            "all_hidden_partial_trace_preconditioner_receipts_authenticated"
        ]
        is True
    )
    assert (
        audit["continuous_hidden_partial_trace_solver_events_authenticated"]
        is True
    )
    assert (
        audit["all_hidden_partial_trace_execution_profiles_authenticated"]
        is True
    )
    assert audit["hidden_partial_trace_pcg_execution_profile_sha256"] == (
        sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
    )
    assert audit["stale_row_block_event_count"] == 0


def test_pcg_collector_retains_archived_row_block_audit_compatibility() -> None:
    artifact = _sequence_artifact(include_execution_profile=False)
    audit = _continuous_row_block_pcg_audit(artifact)
    assert audit["all_iterations_at_most"] == 4096
    assert audit["legacy_solver_event_count"] == 0
    assert audit["restarted_solver_event_count"] == 0
    assert audit["continuous_row_block_solver_event_count"] == 5
    assert audit["all_continuous_row_block_lanes_authenticated"] is True
    assert audit["all_row_block_preconditioner_receipts_authenticated"] is True
    assert audit["continuous_row_block_solver_events_authenticated"] is True
    assert "stale_row_block_event_count" not in audit
    assert "continuous_hidden_partial_trace_solver_event_count" not in audit
    assert audit["audit_sha256"] == (
        "ee55335584e7b966346e62c4193edb595e540a791a52cf6b9681d2d54470a76d"
    )


def test_pcg_collector_marks_authenticated_row_events_stale_in_hidden_audit() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        collector(
            _continuous_row_block_pcg_event(role)
            if group == 0
            else _continuous_hidden_partial_trace_pcg_event(role)
        )
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["continuous_row_block_solver_event_count"] == 1
    assert audit["continuous_hidden_partial_trace_solver_event_count"] == 4
    assert audit["stale_row_block_event_count"] == 1


def test_pcg_collector_accepts_mixed_three_solver_inventory() -> None:
    artifact = _sequence_artifact()
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        if group == 0:
            event = _pcg_event(role)
        elif group == 1:
            event = _restarted_pcg_event(role)
        else:
            event = _continuous_hidden_partial_trace_pcg_event(role)
        collector(event)
    audit = collector.finalize(sequence_artifact=artifact)
    assert audit["all_iterations_at_most"] == 4096
    assert audit["legacy_solver_event_count"] == 1
    assert audit["restarted_solver_event_count"] == 1
    assert audit["continuous_hidden_partial_trace_solver_event_count"] == 3
    assert audit["restarted_extended_solver_events_authenticated"] is True
    assert (
        audit["continuous_hidden_partial_trace_solver_events_authenticated"]
        is True
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("total_matvec_calls", 67),
        ("final_relative_normal_residual", 2e-8),
        ("restart_schedule_sha256", _DIGEST),
    ),
)
def test_pcg_collector_rejects_resealed_restarted_semantic_tamper(
    field: str,
    value,
) -> None:
    event = _restarted_pcg_event()
    event["report"][field] = value
    _reseal_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("continuous_row_block_lane", "attempted_matvec_calls"), 3),
        (("restart_preconditioner", "application_count"), 1),
        (
            (
                "restart_preconditioner",
                "receipt",
                "construction_report",
                "row_blocks_shape",
            ),
            [775, 64, 64],
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "construction_report",
                "row_blocks_maximum_asymmetry",
            ),
            2e-12,
        ),
        (
            ("restart_preconditioner", "receipt", "all_row_blocks_spd"),
            False,
        ),
    ),
)
def test_pcg_collector_retains_row_block_semantic_authentication(
    path: tuple[str, ...],
    value,
) -> None:
    event = _continuous_row_block_pcg_event()
    target = event["report"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    _reseal_continuous_row_block_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("periodic_restart_count",), 1),
        (("original_lane_iterations_executed",), 63),
        (("total_matvec_calls",), 69),
        (
            (
                "continuous_hidden_partial_trace_lane",
                "attempted_matvec_calls",
            ),
            3,
        ),
        (
            ("restart_preconditioner", "application_count"),
            1,
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "construction_report",
                "partial_trace_shape",
            ),
            [775, 775],
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "construction_report",
                "partial_trace_sha256",
            ),
            "x" * 64,
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "construction_report",
                "raw_maximum_absolute_asymmetry",
            ),
            2e-12,
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "positive_finite_factor_diagonal",
            ),
            False,
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "partial_trace_spd",
            ),
            False,
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "factorization",
            ),
            "upper_cholesky_ex",
        ),
        (
            (
                "restart_preconditioner",
                "receipt",
                "inverse_orientation_probe_relative_residual",
            ),
            2e-10,
        ),
    ),
)
def test_pcg_collector_rejects_resealed_partial_trace_semantic_tamper(
    path: tuple[str, ...],
    value,
) -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    target = event["report"]
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


@pytest.mark.parametrize(
    "path",
    (
        (),
        ("continuous_hidden_partial_trace_lane",),
        ("restart_preconditioner",),
        ("restart_preconditioner", "receipt"),
        (
            "restart_preconditioner",
            "receipt",
            "construction_report",
        ),
    ),
)
def test_pcg_collector_rejects_partial_trace_extra_receipt_fields(
    path: tuple[str, ...],
) -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    target = event["report"]
    for field in path:
        target = target[field]
    target["unauthenticated_extra"] = True
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_pcg_collector_rejects_stale_nested_partial_trace_seal() -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    construction = event["report"]["restart_preconditioner"]["receipt"][
        "construction_report"
    ]
    construction["ridge"] = 2e-6
    _reseal_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_pcg_collector_rejects_partial_trace_exact_residual_gate_tamper() -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    checkpoints = event["report"]["exact_relative_normal_residual_checkpoints"]
    checkpoints[-1]["relative_normal_residual"] = 2e-8
    event["report"]["final_relative_normal_residual"] = 2e-8
    event["report"]["continuous_hidden_partial_trace_lane"][
        "ending_exact_relative_normal_residual"
    ] = 2e-8
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_pcg_collector_rejects_stale_row_block_solver_execution() -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    event["report"]["solver"] = (
        "deterministic_continuous_row_block_jacobi_pcg_single_tensor"
    )
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_pcg_collector_rejects_resealed_execution_profile_substitution() -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    profile = event["report"]["execution_profile"]
    profile["hard_maximum_iterations"] = 2048
    profile_body = dict(profile)
    profile_body.pop("report_sha256", None)
    profile["report_sha256"] = _sha256(profile_body)
    event["report"]["execution_profile_sha256"] = profile["report_sha256"]
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_pcg_collector_rejects_resealed_artifact_execution_profile_drift() -> None:
    artifact = _sequence_artifact()
    profile = artifact["streamed_pcg_execution_profile"]
    profile["production_row_block_preconditioner_selected"] = True
    profile_body = dict(profile)
    profile_body.pop("report_sha256", None)
    profile["report_sha256"] = _sha256(profile_body)
    artifact["streamed_pcg_execution_profile_sha256"] = profile["report_sha256"]
    artifact_body = dict(artifact)
    artifact_body.pop("report_sha256", None)
    artifact["report_sha256"] = _sha256(artifact_body)
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        collector(
            _continuous_hidden_partial_trace_pcg_event(
                f"sequence_effective_structured_value_iteration_1_group_{group}"
            )
        )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        collector.finalize(sequence_artifact=artifact)


def test_pcg_collector_rejects_resealed_wrong_partial_trace_source_group() -> None:
    event = _continuous_hidden_partial_trace_pcg_event()
    construction = event["report"]["restart_preconditioner"]["receipt"][
        "construction_report"
    ]
    construction["source_terms"][0]["head"] = 2
    construction["source_terms_sha256"] = _sha256(
        construction["source_terms"]
    )
    _reseal_continuous_hidden_partial_trace_pcg_event(event)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        Lfm2SequenceEffectivePcgCollector()(event)


def test_resource_monitor_enforces_rss_cuda_and_zero_swap() -> None:
    def process():
        return {"VmRSS": 10, "VmHWM": 20, "VmSwap": 0}

    monitor = Lfm2CpcfV5RealAttentionResourceMonitor(
        process_sampler=process,
        cuda_sampler=lambda: {"cuda:0": 30, "cuda:1": 40},
        sample_interval_seconds=0.01,
    )
    with monitor:
        monitor.checkpoint("test")
    report = monitor.report()
    assert report["zero_swap_passed"] is True
    assert report["peak_cuda_reserved_bytes"] == {"cuda:0": 30, "cuda:1": 40}
    failing = Lfm2CpcfV5RealAttentionResourceMonitor(
        process_sampler=lambda: {"VmRSS": 10, "VmHWM": 20, "VmSwap": 1},
        cuda_sampler=lambda: {},
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        failing.__enter__()


def _sequence_artifact(
    *,
    iterations: int = 1,
    include_execution_profile: bool = True,
):
    effective_hash = _sha256({"solve": "effective"})
    seed_hash = _sha256({"solve": "seed"})
    anchor_hash = _sha256({"solve": "anchor"})
    frontier = [
        {
            "proposal_index": 0,
            "stage": "scalable_producer_shared_value_sequence_output_seed",
            "accepted": True,
            "solve_report_sha256": seed_hash,
        }
    ]
    solve_hashes = [effective_hash, seed_hash]
    accepted = 0
    for iteration in range(1, iterations + 1):
        value_hashes = [
            _sha256(
                {
                    "solve": "structured_value",
                    "iteration": iteration,
                    "group": group,
                }
            )
            for group in range(5)
        ]
        output_hash = _sha256({"solve": "iteration_output", "iteration": iteration})
        proposal_accepted = iteration < iterations
        accepted += int(proposal_accepted)
        frontier.append(
            {
                "proposal_index": iteration,
                "stage": ("monotone_streamed_effective_space_value_output_als"),
                "accepted": proposal_accepted,
                "value_solve_report_sha256": value_hashes,
                "output_solve_report_sha256": output_hash,
            }
        )
        solve_hashes.extend(value_hashes)
        solve_hashes.append(output_hash)
    frontier.append(
        {
            "proposal_index": len(frontier),
            "stage": "fixed_producer_value_subspace_anchor",
            "accepted": True,
            "selected_for_materialization": True,
            "output_solve_report_sha256": anchor_hash,
        }
    )
    solve_hashes.append(anchor_hash)
    body = {
        "schema": (sequence_fold.LFM2_FIXED_QK_SEQUENCE_EFFECTIVE_ATTENTION_SCHEMA),
        "candidate_id": H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        "fixed_query_key_state_reused_exactly": True,
        "fit_only_compilation": True,
        "compiled_from_splits": ["fit"],
        "selection_case_count_consumed": 0,
        "validation_case_count_consumed": 0,
        "dense_target_native_qkvo": True,
        "ordinary_dense_target_value_output": True,
        "forbidden_value_coefficient_tensor_materialized": False,
        "forbidden_hidden_head_dim_gram_materialized": False,
        "solver_plan": {
            "strategy": "streamed_cross_head_statistics_pcg",
            "report_sha256": H776_SEQUENCE_SOLVER_PLAN_SHA256,
            "fit_rows": 40_656,
            "target_query_heads": 10,
            "target_hidden_size": 776,
            "target_output_size": 776,
            "head_dim": 64,
            "forbidden_value_coefficient_materialized": False,
            "forbidden_value_dense_gram_materialized": False,
            "complete_operator_dense_factorization_avoided": True,
        },
        "effective_operator_solve": {"report_sha256": effective_hash},
        "candidate_frontier": frontier,
        "candidate_frontier_sha256": _sha256(frontier),
        "bounded_proposal_count": len(frontier),
        "selected_proposal_index": len(frontier) - 1,
        "accepted_als_iterations": accepted,
        "executed_als_iteration_limit": 2,
        "solve_report_sha256": solve_hashes,
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
    }
    if include_execution_profile:
        body["streamed_pcg_execution_profile"] = dict(
            sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE
        )
        body["streamed_pcg_execution_profile_sha256"] = (
            sequence_fold.LFM2_SEQUENCE_EFFECTIVE_HIDDEN_PARTIAL_TRACE_PCG_EXECUTION_PROFILE_SHA256
        )
    return {**body, "report_sha256": _sha256(body)}


def _pcg_audit(artifact):
    collector = Lfm2SequenceEffectivePcgCollector()
    iterations = len(artifact["candidate_frontier"]) - 2
    for iteration in range(1, iterations + 1):
        for group in range(5):
            collector(
                _pcg_event(
                    "sequence_effective_structured_value_iteration_"
                    f"{iteration}_group_{group}"
                )
            )
    return collector.finalize(sequence_artifact=artifact)


def _continuous_hidden_partial_trace_pcg_audit(artifact):
    collector = Lfm2SequenceEffectivePcgCollector()
    iterations = len(artifact["candidate_frontier"]) - 2
    for iteration in range(1, iterations + 1):
        for group in range(5):
            collector(
                _continuous_hidden_partial_trace_pcg_event(
                    "sequence_effective_structured_value_iteration_"
                    f"{iteration}_group_{group}"
                )
            )
    return collector.finalize(sequence_artifact=artifact)


def _continuous_row_block_pcg_audit(artifact):
    collector = Lfm2SequenceEffectivePcgCollector()
    iterations = len(artifact["candidate_frontier"]) - 2
    for iteration in range(1, iterations + 1):
        for group in range(5):
            collector(
                _continuous_row_block_pcg_event(
                    "sequence_effective_structured_value_iteration_"
                    f"{iteration}_group_{group}"
                )
            )
    return collector.finalize(sequence_artifact=artifact)


def test_pcg_audit_accepts_exact_two_iteration_inventory() -> None:
    artifact = _sequence_artifact(iterations=2)
    audit = _pcg_audit(artifact)
    assert audit["als_iteration_count"] == 2
    assert audit["single_tensor_solve_count"] == 10
    assert audit["observed_roles"] == [
        (f"sequence_effective_structured_value_iteration_{iteration}_group_{group}")
        for iteration in (1, 2)
        for group in range(5)
    ]
    assert len(audit["event_to_artifact_solve_bindings"]) == 10


def _attention_result(*, selected: str, sequence_tied: bool = False):
    evaluations = {}
    for ordinal, candidate_id in enumerate(H776_ATTENTION_CANDIDATE_IDS):
        artifact = {"candidate_id": candidate_id}
        if candidate_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID:
            artifact = _sequence_artifact()
        metric = float(ordinal + 2)
        if candidate_id == H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID:
            metric = 2.0 if sequence_tied else 1.0
        evaluations[candidate_id] = {
            "candidate_id": candidate_id,
            "operator_artifact": artifact,
            "operator_compiled_from_splits": ["fit"],
            "mlp_compiled_from_splits": ["fit"],
            "candidate_evaluated_on_splits": ["selection"],
            "validation_phase_count_consumed": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "selection_key": [metric] * 10 + [candidate_id],
        }
    if sequence_tied:
        evaluations[H776_ATTENTION_CANDIDATE_IDS[0]]["selection_key"] = [2.0] * 10 + [
            H776_ATTENTION_CANDIDATE_IDS[0]
        ]
    elif selected == H776_ATTENTION_CANDIDATE_IDS[0]:
        evaluations[selected]["selection_key"] = [0.5] * 10 + [selected]
    selection = {
        "candidate_ids": list(H776_ATTENTION_CANDIDATE_IDS),
        "selected_candidate_id": selected,
        "candidate_count": 3,
        "selection_split_only": True,
        "winner_frozen_before_validation": True,
        "validation_phase_count_consumed_before_freeze": 0,
    }
    next_sha = "7" * 64
    body = {
        "layer_index": 2,
        "operator_kind": "full_attention",
        "input_bank_sha256": "8" * 64,
        "next_bank_sha256": next_sha,
        "split_phase_counts": dict(H776_SPLIT_PHASE_COUNTS),
        "split_metrics": {"validation": {"relative_rmse": 0.5}},
        "solver_evidence_policy": {
            "attention_candidate_selection": selection,
            "attention_candidate_evaluations": evaluations,
            "attention_each_candidate_has_own_fit_only_mlp": True,
            "attention_winner_frozen_before_validation": True,
            "attention_validation_replays_before_winner_freeze": 0,
        },
    }
    report = {**body, "result_sha256": _sha256(body)}
    return SimpleNamespace(
        report=report,
        next_bank=SimpleNamespace(layer_index=3, state_bank_sha256=next_sha),
    )


def _attention_result_with_sequential_transport(*, selected: str):
    result = _attention_result(selected=selected)
    report = runner._plain(result.report)
    policy = report["solver_evidence_policy"]
    evaluations = policy["attention_candidate_evaluations"]
    transport_policy_body = {
        "enabled": True,
        "fit_split_only": True,
        "selection_refits_or_mutates_transport": False,
        "selection_selects_frozen_bundle": True,
        "validation_argument_supported": False,
    }
    transport_policy = {
        **transport_policy_body,
        "report_sha256": _sha256(transport_policy_body),
    }
    source_chart_sha = _sha256({"chart": "source"})
    source_chart_payload_sha = _sha256({"chart_payload": "source"})
    candidate_charts = {}
    candidate_bundles = {}
    candidate_reports = {}
    for candidate_id in H776_ATTENTION_CANDIDATE_IDS:
        evaluation = evaluations[candidate_id]
        chart_sha = _sha256({"chart": candidate_id})
        chart_payload_sha = _sha256({"chart_payload": candidate_id})
        bundle_sha = _sha256({"bundle": candidate_id})
        transport_body = {
            "candidate_id": candidate_id,
            "fit_only": True,
        }
        transport = {
            **transport_body,
            "report_sha256": _sha256(transport_body),
        }
        binding_body = {
            "candidate_id": candidate_id,
            "effective_boundary_charts_sha256": chart_sha,
            "sequential_transport_report_sha256": (transport["report_sha256"]),
            "policy_report_sha256": transport_policy["report_sha256"],
            "source_boundary_charts_sha256": source_chart_sha,
            "source_boundary_charts_payload_sha256": (source_chart_payload_sha),
            "effective_boundary_charts_payload_sha256": (chart_payload_sha),
            "fit_split_only": True,
            "selection_rows_read_by_transport": 0,
            "selection_rows_read_before_transport_freeze": 0,
            "validation_rows_read": 0,
            "equal_total_mass_per_fit_phase": True,
            "one_closed_form_svd": True,
            "source_post_operator_chart_same_object": True,
            "source_post_layer_chart_unchanged": True,
            "transported_post_layer_chart_candidate_local": True,
            "automatic_retry_or_retuning_performed": False,
        }
        binding = {
            **binding_body,
            "report_sha256": _sha256(binding_body),
        }
        evaluation.update(
            {
                "sequential_boundary_transport": transport,
                "sequential_transport_binding": binding,
                "effective_boundary_charts_sha256": chart_sha,
                "effective_boundary_charts_payload_sha256": (chart_payload_sha),
                "operator_transport_mlp_bundle_sha256": bundle_sha,
                "candidate_local_post_layer_transport_frozen_before_mlp": (True),
                "candidate_local_mlp_basis_preparation_count": 1,
                "cross_candidate_prepared_mlp_basis_reused": False,
                "selection_metrics_use_effective_boundary_charts": True,
                "operator_transport_and_mlp_ranked_as_one_bundle": True,
            }
        )
        evaluation["report_sha256"] = _sha256(evaluation)
        candidate_charts[candidate_id] = chart_sha
        candidate_bundles[candidate_id] = bundle_sha
        candidate_reports[candidate_id] = evaluation["report_sha256"]

    selection = policy["attention_candidate_selection"]
    selection.update(
        {
            "candidate_report_sha256": candidate_reports,
            "candidate_effective_boundary_charts_sha256": (candidate_charts),
            "candidate_operator_transport_mlp_bundle_sha256": (candidate_bundles),
            "selected_effective_boundary_charts_sha256": (candidate_charts[selected]),
            "selected_candidate_report_sha256": (candidate_reports[selected]),
            "selected_operator_transport_mlp_bundle_sha256": (
                candidate_bundles[selected]
            ),
            "candidate_local_post_layer_transport_enabled": True,
            "candidate_local_mlp_preparation_count": 3,
            "cross_candidate_prepared_mlp_basis_reused": False,
            "operator_transport_mlp_bundle_frozen_together": True,
            "selection_metrics_use_candidate_effective_charts": True,
            "selection_refits_or_mutates_transport": False,
            "selection_selects_frozen_bundle": True,
        }
    )
    selection["report_sha256"] = _sha256(selection)
    selected_evaluation = evaluations[selected]
    final_binding_body = {
        "policy": transport_policy,
        "selected_candidate_id": selected,
        "selected_candidate_report_sha256": (selected_evaluation["report_sha256"]),
        "candidate_selection_report_sha256": selection["report_sha256"],
        "operator_transport_mlp_bundle_sha256": candidate_bundles[selected],
        "effective_boundary_charts_sha256": candidate_charts[selected],
        "effective_boundary_charts_payload_sha256": (
            selected_evaluation["effective_boundary_charts_payload_sha256"]
        ),
        "source_boundary_charts_sha256": source_chart_sha,
        "source_boundary_charts_payload_sha256": source_chart_payload_sha,
        "sequential_transport_report_sha256": (
            selected_evaluation["sequential_boundary_transport"]["report_sha256"]
        ),
        "sequential_transport_binding_report_sha256": (
            selected_evaluation["sequential_transport_binding"]["report_sha256"]
        ),
        "final_endpoint_metrics_use_winning_effective_chart": True,
        "validation_first_endpoint_replay_and_metric_after_bundle_freeze": (True),
        "validation_rows_used_by_transport_solver": 0,
        "selection_refits_or_mutates_transport": False,
        "selection_selects_frozen_bundle": True,
        "operator_transport_mlp_bundle_installed_exactly": True,
    }
    policy.update(
        {
            "attention_candidate_local_post_layer_transport": {
                **final_binding_body,
                "report_sha256": _sha256(final_binding_body),
            },
            "attention_sequential_transport_policy": transport_policy,
            "attention_each_candidate_prepares_own_mlp_basis": True,
            "attention_cross_candidate_prepared_mlp_basis_reuse": False,
            "attention_final_endpoint_effective_boundary_charts_sha256": (
                candidate_charts[selected]
            ),
        }
    )
    report["boundary_charts_sha256"] = source_chart_sha
    report["audit"] = {
        "source_boundary_charts_unchanged_after_candidate_transports": True,
        "winning_effective_boundary_chart_used_for_validation": True,
    }
    report["actual_child_state_policy"] = {
        "attention_operator_transport_mlp_frozen_together": True,
    }
    report_body = dict(report)
    report_body.pop("result_sha256", None)
    report = {
        **report_body,
        "result_sha256": _sha256(report_body),
    }
    return SimpleNamespace(report=report, next_bank=result.next_bank)


def test_attention_result_reports_positive_or_honest_negative(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    positive_result = _attention_result(selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID)
    positive_artifact = positive_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    positive = authenticate_h776_attention_result(
        positive_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(positive_artifact),
        pins=pins,
    )
    assert positive["scientific_outcome"] == ("positive_sequence_effective_selection")
    assert (
        positive["sequence_effective_artifact"]["report_sha256"]
        == positive_artifact["report_sha256"]
    )
    negative_result = _attention_result(selected=H776_ATTENTION_CANDIDATE_IDS[0])
    negative_artifact = negative_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    negative = authenticate_h776_attention_result(
        negative_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(negative_artifact),
        pins=pins,
    )
    assert negative["scientific_outcome"] == ("honest_negative_or_control_selected")


def test_attention_result_accepts_authenticated_hidden_partial_trace_pcg_audit(
    tmp_path: Path,
) -> None:
    result = _attention_result(selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID)
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    audit = authenticate_h776_attention_result(
        result,
        input_bank_sha256="8" * 64,
        pcg_audit=_continuous_hidden_partial_trace_pcg_audit(artifact),
        pins=_pins(tmp_path),
    )
    assert audit["sequence_effective_won"] is True


def test_attention_result_rejects_authenticated_archived_row_block_audit(
    tmp_path: Path,
) -> None:
    result = _attention_result(selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID)
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=_continuous_row_block_pcg_audit(artifact),
            pins=_pins(tmp_path),
        )


def test_attention_result_rejects_mixed_hidden_and_stale_row_execution(
    tmp_path: Path,
) -> None:
    result = _attention_result(selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID)
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    collector = Lfm2SequenceEffectivePcgCollector()
    for group in range(5):
        role = f"sequence_effective_structured_value_iteration_1_group_{group}"
        collector(
            _continuous_row_block_pcg_event(role)
            if group == 0
            else _continuous_hidden_partial_trace_pcg_event(role)
        )
    mixed_audit = collector.finalize(sequence_artifact=artifact)
    assert mixed_audit["stale_row_block_event_count"] == 1
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=mixed_audit,
            pins=_pins(tmp_path),
        )


def test_attention_behavioral_tie_is_not_positive(tmp_path: Path) -> None:
    pins = _pins(tmp_path)
    tied_result = _attention_result(
        selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
        sequence_tied=True,
    )
    artifact = tied_result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    audit = authenticate_h776_attention_result(
        tied_result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(artifact),
        pins=pins,
    )
    assert audit["sequence_effective_won"] is True
    assert audit["sequence_effective_strictly_better_than_both_controls"] is False
    assert audit["scientific_outcome"] == ("honest_negative_or_control_selected")


def test_attention_authenticator_cross_binds_frozen_transport_winner(
    tmp_path: Path,
) -> None:
    pins = _pins(
        tmp_path,
        attention_sequential_transport_enabled=True,
    )
    result = _attention_result_with_sequential_transport(
        selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID
    )
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    audit = authenticate_h776_attention_result(
        result,
        input_bank_sha256="8" * 64,
        pcg_audit=_pcg_audit(artifact),
        pins=pins,
        expected_attention_sequential_transport_enabled=True,
    )
    transport = audit["attention_sequential_transport"]
    assert transport["fit_only_candidate_local_transport_authenticated"] is True
    assert transport["selection_selects_frozen_bundle_without_refit"] is True
    assert transport["winning_chart_used_for_final_metrics"] is True

    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=_pcg_audit(artifact),
            pins=replace(
                pins,
                expected_attention_sequential_transport_enabled=False,
            ),
        )

    final_binding = result.report["solver_evidence_policy"][
        "attention_candidate_local_post_layer_transport"
    ]
    final_binding["selection_refits_or_mutates_transport"] = True
    final_binding_body = dict(final_binding)
    final_binding_body.pop("report_sha256")
    final_binding["report_sha256"] = _sha256(final_binding_body)
    layer_body = dict(result.report)
    layer_body.pop("result_sha256")
    result.report["result_sha256"] = _sha256(layer_body)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=_pcg_audit(artifact),
            pins=pins,
            expected_attention_sequential_transport_enabled=True,
        )


def test_attention_rejects_resealed_pcg_cross_binding_tamper(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path)
    result = _attention_result(selected=H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID)
    artifact = result.report["solver_evidence_policy"][
        "attention_candidate_evaluations"
    ][H776_SEQUENCE_EFFECTIVE_CANDIDATE_ID]["operator_artifact"]
    tampered = runner._plain(dict(_pcg_audit(artifact)))
    tampered.pop("audit_sha256")
    tampered["event_to_artifact_solve_bindings"][0][
        "artifact_value_solve_report_sha256"
    ] = "f" * 64
    tampered["audit_sha256"] = _sha256(tampered)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_attention_result(
            result,
            input_bank_sha256="8" * 64,
            pcg_audit=tampered,
            pins=pins,
        )


def _parser_args(tmp_path: Path) -> list[str]:
    paths = {}
    for name in (
        "parent",
        "donor",
        "target",
        "bridge",
        "cache",
        "corpus",
        "output",
        "resume",
        "catalog",
        "catalog_receipt",
    ):
        path = tmp_path / name
        path.mkdir(parents=True)
        paths[name] = path
    corpus = paths["corpus"] / "trajectories.json"
    corpus.write_text("[]", encoding="utf-8")
    receipt = paths["catalog_receipt"] / "receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    return [
        "--parent-model-directory",
        str(paths["parent"]),
        "--donor-tokenizer-directory",
        str(paths["donor"]),
        "--target-tokenizer-directory",
        str(paths["target"]),
        "--bridge-directory",
        str(paths["bridge"]),
        "--trace-cache-directory",
        str(paths["cache"]),
        "--trajectory-corpus",
        str(corpus),
        "--diagnostic-json",
        str(paths["output"] / "probe.json"),
        "--compression-ratio",
        "4",
        "--maximum-process-rss-gib",
        "64",
        "--expected-target-plan-fingerprint",
        "3f27369567214c7f01d3f437e9c4ec50bce05c8345bbee4bf23570c204be6153",
        "--resume-store-directory",
        str(paths["resume"]),
        "--selective-catalog-store-directory",
        str(paths["catalog"]),
        "--selective-catalog-receipt-json",
        str(receipt),
        "--expected-selective-catalog-receipt-sha256",
        _DIGEST,
        "--checkpoint-key",
        _DIGEST,
        "--checkpoint-manifest-sha256",
        _DIGEST,
        "--expected-trace-sha256",
        _DIGEST,
        "--expected-runtime-compiler-implementation-sha256",
        _OTHER_DIGEST,
        "--expected-sequence-source-sha256",
        _DIGEST,
    ]


def test_cli_accepts_only_exact_h776_resource_and_parameter_lane(
    tmp_path: Path,
) -> None:
    parser = runner.build_parser()
    args = parser.parse_args(_parser_args(tmp_path))
    config = runner.config_from_args(args)
    assert config.fold.compression.compression_ratio == "4"
    assert config.fold.compression.target_unique_parameters is None
    assert config.fold.selected_candidate_rank is None
    assert config.fold.maximum_process_rss_bytes == 64 * 1024**3
    assert config.post_operator_chart_transport_shadow is False
    assert config.attention_sequential_transport_enabled is False
    shadow_args = _parser_args(tmp_path / "shadow")
    shadow_args.append("--post-operator-chart-transport-shadow")
    shadow = runner.config_from_args(parser.parse_args(shadow_args))
    assert shadow.post_operator_chart_transport_shadow is True
    transport_args = _parser_args(tmp_path / "transport")
    transport_args.append("--attention-sequential-transport")
    transport = runner.config_from_args(parser.parse_args(transport_args))
    assert transport.attention_sequential_transport_enabled is True
    bad = _parser_args(tmp_path / "bad")
    ratio_index = bad.index("--compression-ratio") + 1
    bad[ratio_index] = "4.01"
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(bad))
    ranked = _parser_args(tmp_path / "ranked")
    ranked.extend(["--selected-candidate-rank", "2"])
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(ranked))


def _tree_bytes(root: Path):
    return {
        path.relative_to(root).as_posix(): (
            "directory" if path.is_dir() else path.read_bytes()
        )
        for path in sorted(root.rglob("*"))
    }


def test_overlapping_failure_diagnostic_never_mutates_checkpoint_store(
    tmp_path: Path,
) -> None:
    argv = _parser_args(tmp_path)
    resume = Path(argv[argv.index("--resume-store-directory") + 1])
    sentinel = resume / "sentinel.bin"
    sentinel.write_bytes(b"immutable-checkpoint-store")
    forbidden_parent = resume / "diagnostic-parent"
    forbidden = forbidden_parent / "failure.json"
    argv[argv.index("--diagnostic-json") + 1] = str(forbidden)
    before = _tree_bytes(resume)
    assert runner.main(argv) == 2
    assert _tree_bytes(resume) == before
    assert not forbidden_parent.exists()


def test_runtime_guard_starts_before_every_post_load_model_operation() -> None:
    guard = SimpleNamespace(
        optimizer_constructions=0,
        optimizer_steps=0,
        backward_calls=0,
        autograd_grad_calls=0,
        pruning_operations=0,
        quantization_operations=0,
    )
    report = runner._runtime_guard_report(
        guard,
        parent_execution={},
        target_execution={},
        allocator_policy={},
    )
    assert report["scope_started"] == "immediately_after_parent_model_load"
    assert report["parent_model_load_guarded"] is False
    assert report["parent_model_observer_started_immediately_after_load"] is True
    assert report["blank_target_synthesis_guarded"] is True
    assert report["target_model_observer_started_immediately_after_synthesis"] is True
    assert report["selective_provider_open_and_close_guarded"] is True
    assert report["source_tied_table_mapping_guarded"] is True
    assert report["layer_2_probe_guarded"] is True
    assert report["all_model_touching_operations_after_parent_load_guarded"] is True

    source = inspect.getsource(runner.run)
    parent_load = source.index("parent = load_local_lfm2_model")
    guard_start = source.index(
        "guard = dynamic_fold_executor._FoldOnlyTorchGuard()",
        parent_load,
    )
    guard_context = source.index("with torch.no_grad(), guard:", guard_start)
    parent_observer = source.index(
        "observe_model_execution(parent)",
        guard_context,
    )
    blank_synthesis = source.index(
        "blank = plan_and_synthesize_lfm2_cpcf_blank",
        parent_observer,
    )
    target_observer = source.index(
        "observe_model_execution(target)",
        blank_synthesis,
    )
    provider_open = source.index(
        "open_lfm2_cpcf_selective_evidence_provider_from_catalog",
        target_observer,
    )
    tied_mapping = source.index(
        "source_tied_table = base._mapped_source_tied_table",
        provider_open,
    )
    probe = source.index(
        "probe = execute_lfm2_cpcf_v5_real_attention_probe",
        tied_mapping,
    )
    assert (
        parent_load
        < guard_start
        < guard_context
        < parent_observer
        < blank_synthesis
        < target_observer
        < provider_open
        < tied_mapping
        < probe
    )


def test_failure_diagnostic_preserves_authenticated_nested_solver_report() -> None:
    failure_body = {
        "status": "failed_closed",
        "solver": "deterministic_restarted_jacobi_pcg_single_tensor",
        "failure_reason": "hard_iteration_cap_reached",
    }
    failure_report = {
        **failure_body,
        "report_sha256": _sha256(failure_body),
    }
    inner = RuntimeError("solver failed")
    inner.failure_report = MappingProxyType(failure_report)
    outer = RuntimeError("layer failed")
    outer.__cause__ = inner

    diagnostic = runner._failure(outer, config=None)

    assert diagnostic["nested_failure_report_count"] == 1
    assert diagnostic["nested_failure_reports_authenticated"] is True
    nested = diagnostic["nested_failure_reports"][0]
    assert nested["error_chain_index"] == 1
    assert nested["report_sha256"] == failure_report["report_sha256"]
    assert nested["failure_report"] == failure_report
    assert nested["self_seal_authenticated"] is True
    body = dict(diagnostic)
    claimed = body.pop("failure_sha256")
    assert claimed == _sha256(body)
