from __future__ import annotations

import inspect
import math
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
from neural_forge.hf_lfm2_cached_endpoint_qk_residual_fold import (  # noqa: E402
    LFM2_ENDPOINT_QK_EXECUTION_SCHEMA,
)
from neural_forge.hf_lfm2_cpcf_v5_real_attention_probe import (  # noqa: E402
    H776_ATTENTION_CANDIDATE_IDS,
    H776_COMPOSED_ATTENTION_CANDIDATE_ID,
    H776_COMPOSED_ATTENTION_CANDIDATE_IDS,
    H776_COMPOSED_CUDA_DEVICE_NAME,
    H776_COMPOSED_LOGICAL_CUDA_UUIDS,
    H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES,
    H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS,
    H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID,
    H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256,
    H776_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS,
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
    _h776_composed_qk_attention_policy,
    _h776_endpoint_residual_attention_policy,
    _authenticate_h776_endpoint_device_execution,
    _sha256,
    authenticate_h776_attention_result,
    authenticate_h776_composed_attention_result,
    authenticate_h776_endpoint_residual_attention_result,
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
    composed_qk_attention_enabled: bool = False,
    endpoint_residual_attention_enabled: bool = False,
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
        expected_composed_qk_attention_enabled=composed_qk_attention_enabled,
        expected_endpoint_residual_attention_enabled=(
            endpoint_residual_attention_enabled
        ),
        maximum_process_rss_bytes=(
            H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES
            if (composed_qk_attention_enabled or endpoint_residual_attention_enabled)
            else 64 * 1024**3
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
    assert runtime.composed_qk_attention_enabled is False
    assert runtime.endpoint_residual_attention_enabled is False


def test_composed_probe_pins_and_two_gpu_policy_are_exact(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path, composed_qk_attention_enabled=True)
    policy = _h776_composed_qk_attention_policy()

    assert pins.maximum_process_rss_bytes == (H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES)
    assert pins.expected_composed_qk_attention_enabled is True
    assert pins.expected_composed_logical_cuda_uuids == (
        H776_COMPOSED_LOGICAL_CUDA_UUIDS
    )
    assert (
        pins.expected_composed_qk_attention_policy_sha256
        == policy.authentication_sha256
    )
    assert policy.score_policy.solver_device == "cuda:1"
    assert policy.score_policy.compute_devices == ("cuda:0", "cuda:1")
    assert policy.global_policy.maximum_total_score_elements == 134_217_728
    assert policy.verify_authentication() is True
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(pins, maximum_process_rss_bytes=64 * 1024**3)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(
            pins,
            expected_composed_qk_attention_policy_sha256=_DIGEST,
        )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(
            pins,
            expected_composed_logical_cuda_uuids=tuple(
                reversed(H776_COMPOSED_LOGICAL_CUDA_UUIDS)
            ),
        )


def test_endpoint_probe_pins_and_two_gpu_policy_are_exact(
    tmp_path: Path,
) -> None:
    pins = _pins(tmp_path, endpoint_residual_attention_enabled=True)
    policy = _h776_endpoint_residual_attention_policy()

    assert pins.maximum_process_rss_bytes == H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES
    assert pins.expected_composed_qk_attention_enabled is False
    assert pins.expected_endpoint_residual_attention_enabled is True
    assert (
        pins.expected_endpoint_residual_attention_policy_sha256
        == H776_ENDPOINT_RESIDUAL_ATTENTION_POLICY_SHA256
        == policy.authentication_sha256
    )
    assert policy.maximum_outer_blocks == 2
    assert policy.endpoint_policy.maximum_cg_iterations == 64
    assert policy.endpoint_policy.cg_restart_iterations == 64
    assert policy.endpoint_policy.cg_relative_tolerance == 1e-2
    assert policy.endpoint_policy.solver_device == "cuda:1"
    assert policy.endpoint_policy.compute_devices == ("cuda:0", "cuda:1")
    assert policy.allow_legacy_recomputed_parent_probabilities is False
    assert policy.verify_authentication() is True
    assert len(H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS) == 5
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(
            pins,
            expected_endpoint_residual_attention_policy_sha256=_DIGEST,
        )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        replace(pins, expected_composed_qk_attention_enabled=True)


def _endpoint_execution_proposal(
    trajectories: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], tuple[str, ...]]:
    trajectories = (
        tuple(f"{1_000 + index:064x}" for index in range(134))
        if trajectories is None
        else trajectories
    )
    assert len(trajectories) == 134
    numerical_sweep_count = 2
    numerical_normal_operator_calls = 8
    exact_replay_calls = 5
    observations = []
    for index, trajectory in enumerate(trajectories):
        device_index = index % 2
        observation_body = {
            "assignment_index": index,
            "intended_logical_device": f"cuda:{device_index}",
            "observed_logical_device": f"cuda:{device_index}",
            "observed_device_type": "cuda",
            "observed_device_index": device_index,
            "cuda_uuid": H776_COMPOSED_LOGICAL_CUDA_UUIDS[device_index],
            "cuda_name": H776_COMPOSED_CUDA_DEVICE_NAME,
            "evidence_sha256": f"{2_000 + index:064x}",
            "trajectory_sha256": trajectory,
            "target_receipt_sha256": f"{3_000 + index:064x}",
            "analytical_operation": "exact_endpoint_jvp_then_vjp_normal_probe",
            "exact_replay_calls": exact_replay_calls,
            "endpoint_jvp_calls": numerical_normal_operator_calls + 1,
            "endpoint_vjp_calls": (
                numerical_normal_operator_calls + numerical_sweep_count + 1
            ),
            "normal_operator_calls": numerical_normal_operator_calls + 1,
            "semantic_probe_calls": 1,
            "semantic_delta_probability_sha256": f"{4_000 + index:064x}",
            "semantic_delta_endpoint_sha256": f"{5_000 + index:064x}",
            "semantic_query_gradient_sha256": f"{6_000 + index:064x}",
            "semantic_key_gradient_sha256": f"{7_000 + index:064x}",
            "prepared_persistent_tensor_bytes": 1_000_000,
            "attached_replay_tensor_bytes": 2_000_000,
            "semantic_probe_tensor_bytes": 10_000_000,
            "analytical_tensor_element_count": 4_096,
            "analytical_scalar": float(index + 1),
            "forced_host_scalar_read": True,
        }
        observations.append(
            {
                **observation_body,
                "case_execution_receipt_sha256": _sha256(observation_body),
            }
        )
    summaries = []
    static_shards = []
    fixed_cache_bytes = 4_470_528
    shard_workspace_bytes = 11_919_360
    solver_workspace_bytes = 47_677_440
    for device_index in range(2):
        indices = list(range(device_index, len(trajectories), 2))
        fixed_cache_body = {
            "schema": "neural-forge-lfm2-endpoint-fixed-device-cache-v1",
            "logical_device": f"cuda:{device_index}",
            "fixed_value_bfloat16_sha256": f"{8_000 + device_index:064x}",
            "fixed_output_float64_sha256": f"{8_010 + device_index:064x}",
            "query_rms_bfloat16_sha256": f"{8_020 + device_index:064x}",
            "key_rms_bfloat16_sha256": f"{8_030 + device_index:064x}",
            "query_rms_float32_sha256": f"{8_040 + device_index:064x}",
            "key_rms_float32_sha256": f"{8_050 + device_index:064x}",
            "shared_fixed_tensor_bytes": fixed_cache_bytes,
            "fixed_cache_builds": 1,
            "fixed_tensor_materialization": ("once_per_requested_compute_device"),
        }
        fixed_cache = {
            **fixed_cache_body,
            "fixed_cache_receipt_sha256": _sha256(fixed_cache_body),
        }
        static_shards.append(
            {
                "logical_device": f"cuda:{device_index}",
                "case_assignment_indices": indices,
                "case_count": len(indices),
                "all_cases_share_fixed_cache_by_identity": True,
                "fixed_cache_receipt": fixed_cache,
            }
        )
        prepared_bytes = len(indices) * 1_000_000
        replay_bytes = len(indices) * 2_000_000
        device_solver_workspace = solver_workspace_bytes if device_index == 1 else 0
        conservative_peak_bytes = (
            prepared_bytes
            + replay_bytes
            + 10_000_000
            + fixed_cache_bytes
            + shard_workspace_bytes
            + device_solver_workspace
        )
        summaries.append(
            {
                "logical_device": f"cuda:{device_index}",
                "case_assignment_indices": indices,
                "case_count": len(indices),
                "semantic_probe_calls": len(indices),
                "cuda_uuid": H776_COMPOSED_LOGICAL_CUDA_UUIDS[device_index],
                "cuda_name": H776_COMPOSED_CUDA_DEVICE_NAME,
                "exact_replay_calls": len(indices) * exact_replay_calls,
                "endpoint_jvp_calls": (
                    len(indices) * (numerical_normal_operator_calls + 1)
                ),
                "endpoint_vjp_calls": (
                    len(indices)
                    * (numerical_normal_operator_calls + numerical_sweep_count + 1)
                ),
                "normal_operator_calls": (
                    len(indices) * (numerical_normal_operator_calls + 1)
                ),
                "prepared_persistent_tensor_bytes": prepared_bytes,
                "attached_replay_tensor_bytes": replay_bytes,
                "shared_fixed_tensor_cache_bytes": fixed_cache_bytes,
                "fixed_cache_receipt": fixed_cache,
                "replay_qk_shard_broadcasts": exact_replay_calls,
                "normal_qk_direction_shard_broadcasts": (
                    numerical_normal_operator_calls
                ),
                "rhs_gradient_shard_reductions": numerical_sweep_count,
                "normal_gradient_shard_reductions": (numerical_normal_operator_calls),
                "semantic_direction_shard_builds": 1,
                "semantic_gradient_shard_reductions": 1,
                "concurrent_rhs_shard_launches": numerical_sweep_count,
                "concurrent_normal_shard_launches": (numerical_normal_operator_calls),
                "qk_transfer_execution": (
                    "one_local_accumulation_per_concurrent_static_cuda_shard_then_"
                    "canonical_requested_device_reduction"
                ),
                "maximum_serial_semantic_probe_tensor_bytes": 10_000_000,
                "conservative_static_shard_workspace_bytes": (shard_workspace_bytes),
                "conservative_solver_workspace_bytes": device_solver_workspace,
                "conservative_peak_accounted_bytes": conservative_peak_bytes,
                "byte_accounting_scope": (
                    "deterministic_declared_tensor_bytes_not_allocator_peak"
                ),
                "maximum_bytes_per_compute_device": 20 * 1024**3,
                "within_compute_device_byte_ceiling": True,
            }
        )
    static_body = {
        "schema": "neural-forge-lfm2-endpoint-static-device-shards-v1",
        "assignment": "authenticated_bank_order_round_robin_static_shards",
        "device_order": ["cuda:0", "cuda:1"],
        "shards": static_shards,
    }
    static_receipt = {
        **static_body,
        "static_device_shards_sha256": _sha256(static_body),
    }
    solver_body = {
        "intended_logical_device": "cuda:1",
        "observed_logical_device": "cuda:1",
        "observed_device_type": "cuda",
        "observed_device_index": 1,
        "cuda_uuid": H776_COMPOSED_LOGICAL_CUDA_UUIDS[1],
        "cuda_name": H776_COMPOSED_CUDA_DEVICE_NAME,
        "analytical_operation": (
            "dense_seed_qk_plus_worker_normal_sum_energy_and_parameter_count"
        ),
        "analytical_scalar": 1.0,
        "forced_host_scalar_read": True,
    }
    solver = {
        **solver_body,
        "solver_execution_receipt_sha256": _sha256(solver_body),
    }
    resource = {
        "fit_case_count": len(trajectories),
        "prepared_persistent_tensor_bytes": 134 * 1_000_000,
        "attached_replay_tensor_bytes": 134 * 2_000_000,
        "shared_fixed_tensor_cache_bytes": 2 * fixed_cache_bytes,
        "fixed_tensor_cache_count": 2,
        "maximum_serial_semantic_probe_tensor_bytes": 10_000_000,
        "conservative_static_shard_workspace_bytes_per_device": (shard_workspace_bytes),
        "byte_accounting_scope": (
            "deterministic_declared_tensor_bytes_not_allocator_peak"
        ),
        "production_allocator_peak_authority": (
            "outer_real_probe_continuous_cuda_peak_reserved_monitor"
        ),
        "within_all_resource_ceilings": True,
        "every_compute_device_within_byte_ceiling": True,
    }
    execution_body = {
        "schema": LFM2_ENDPOINT_QK_EXECUTION_SCHEMA,
        "requested_solver_device": "cuda:1",
        "requested_compute_devices": ["cuda:0", "cuda:1"],
        "requested_compute_device_count": 2,
        "case_assignment": "authenticated_bank_order_round_robin",
        "compute_device_concurrency": ("thread_barrier_concurrent_static_cuda_shards"),
        "qk_transfer_execution": (
            "one_local_accumulation_per_concurrent_static_cuda_shard_then_"
            "canonical_requested_device_reduction"
        ),
        "static_device_shard_receipt": static_receipt,
        "every_requested_compute_device_observed": True,
        "distinct_cuda_uuid_count": 2,
        "numerical_sweep_count": numerical_sweep_count,
        "numerical_normal_operator_calls": numerical_normal_operator_calls,
        "total_exact_replay_calls": 134 * exact_replay_calls,
        "total_endpoint_jvp_calls": (134 * (numerical_normal_operator_calls + 1)),
        "total_endpoint_vjp_calls": (
            134 * (numerical_normal_operator_calls + numerical_sweep_count + 1)
        ),
        "total_normal_operator_calls": (134 * (numerical_normal_operator_calls + 1)),
        "total_semantic_probe_calls": len(trajectories),
        "compute_observations": observations,
        "compute_device_summaries": summaries,
        "solver_observation": solver,
        "resource_observation": resource,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    execution = {
        **execution_body,
        "receipt_sha256": _sha256(execution_body),
    }
    return (
        {
            "solver_device": "cuda:1",
            "compute_devices": ["cuda:0", "cuda:1"],
            "case_assignment": "authenticated_bank_order_round_robin",
            "fit_check_accessible": False,
            "fit_check_cases_consumed": 0,
            "observed_execution_receipt": execution,
        },
        trajectories,
    )


def test_endpoint_device_execution_authenticates_all_134_round_robin_cases(
    tmp_path: Path,
) -> None:
    proposal, trajectories = _endpoint_execution_proposal()
    audit = _authenticate_h776_endpoint_device_execution(
        proposal,
        expected_fit_trajectory_sha256s=trajectories,
        pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
    )

    assert audit["fit_case_count"] == 134
    assert audit["per_device_case_counts"] == {"cuda:0": 67, "cuda:1": 67}
    assert audit["semantic_probe_calls"] == 134
    assert audit["allocator_peak_authority"] == (
        "outer_real_probe_continuous_cuda_peak_reserved_monitor"
    )


def test_endpoint_device_execution_rejects_resealed_non_round_robin_case(
    tmp_path: Path,
) -> None:
    proposal, trajectories = _endpoint_execution_proposal()
    receipt = proposal["observed_execution_receipt"]
    observation = receipt["compute_observations"][2]
    observation["intended_logical_device"] = "cuda:1"
    observation["observed_logical_device"] = "cuda:1"
    observation["observed_device_index"] = 1
    observation["cuda_uuid"] = H776_COMPOSED_LOGICAL_CUDA_UUIDS[1]
    observation.pop("case_execution_receipt_sha256")
    observation["case_execution_receipt_sha256"] = _sha256(observation)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        _authenticate_h776_endpoint_device_execution(
            proposal,
            expected_fit_trajectory_sha256s=trajectories,
            pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
        )


def test_endpoint_device_execution_rejects_resealed_shard_transfer_counter(
    tmp_path: Path,
) -> None:
    proposal, trajectories = _endpoint_execution_proposal()
    receipt = proposal["observed_execution_receipt"]
    summary = receipt["compute_device_summaries"][0]
    summary["normal_qk_direction_shard_broadcasts"] += 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(
        Lfm2CpcfV5RealAttentionProbeError,
        match="cuda:0 summary",
    ):
        _authenticate_h776_endpoint_device_execution(
            proposal,
            expected_fit_trajectory_sha256s=trajectories,
            pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
        )


def test_endpoint_device_execution_rejects_resealed_concurrent_launch_counter(
    tmp_path: Path,
) -> None:
    proposal, trajectories = _endpoint_execution_proposal()
    receipt = proposal["observed_execution_receipt"]
    summary = receipt["compute_device_summaries"][0]
    summary["concurrent_normal_shard_launches"] -= 1
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(
        Lfm2CpcfV5RealAttentionProbeError,
        match="cuda:0 summary",
    ):
        _authenticate_h776_endpoint_device_execution(
            proposal,
            expected_fit_trajectory_sha256s=trajectories,
            pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
        )


def test_endpoint_device_execution_rejects_resealed_fixed_cache_build_count(
    tmp_path: Path,
) -> None:
    proposal, trajectories = _endpoint_execution_proposal()
    receipt = proposal["observed_execution_receipt"]
    static = receipt["static_device_shard_receipt"]
    fixed_cache = static["shards"][0]["fixed_cache_receipt"]
    fixed_cache["fixed_cache_builds"] = 2
    fixed_cache.pop("fixed_cache_receipt_sha256")
    fixed_cache["fixed_cache_receipt_sha256"] = _sha256(fixed_cache)
    static.pop("static_device_shards_sha256")
    static["static_device_shards_sha256"] = _sha256(static)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = _sha256(receipt)

    with pytest.raises(
        Lfm2CpcfV5RealAttentionProbeError,
        match="cuda:0 summary",
    ):
        _authenticate_h776_endpoint_device_execution(
            proposal,
            expected_fit_trajectory_sha256s=trajectories,
            pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
        )


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
            "deterministic_continuous_hidden_partial_trace_cholesky_pcg_single_tensor"
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
    assert audit["all_continuous_hidden_partial_trace_lanes_authenticated"] is True
    assert (
        audit["all_hidden_partial_trace_preconditioner_receipts_authenticated"] is True
    )
    assert audit["continuous_hidden_partial_trace_solver_events_authenticated"] is True
    assert audit["all_hidden_partial_trace_execution_profiles_authenticated"] is True
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
    assert audit["continuous_hidden_partial_trace_solver_events_authenticated"] is True


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
    construction["source_terms_sha256"] = _sha256(construction["source_terms"])
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


class _FakeSelectedDirectArtifact:
    def __init__(self, report: dict[str, object]) -> None:
        self._report = dict(report)
        self._state = {"q_proj.weight": torch.tensor([1.0], dtype=torch.float32)}
        self.compiled_state_sha256 = str(report["compiled_state_sha256"])

    def as_dict(self) -> dict[str, object]:
        return dict(self._report)

    def state_dict(self):
        return {name: value.clone() for name, value in self._state.items()}


def _patch_fake_selected_direct_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neural_forge import hf_lfm2_structural_residual_attention_fold

    monkeypatch.setattr(
        hf_lfm2_structural_residual_attention_fold,
        "Lfm2CompiledStructuralAttentionFold",
        _FakeSelectedDirectArtifact,
    )


def _composed_attention_result():
    from neural_forge.hf_lfm2_structural_residual_attention_fold import (
        _state_sha256 as _structural_attention_state_sha256,
    )

    composed_policy = _h776_composed_qk_attention_policy().as_dict()
    score_policy = composed_policy["score_policy"]
    fit_bank_sha256 = "9" * 64
    trajectory_sha256s = tuple(f"{index:064x}" for index in range(1, 155))
    trajectory_bundle_sha256 = _sha256(list(trajectory_sha256s))
    upstream_case_bundle_sha256 = "6" * 64
    fit_metrics = {
        "case_count": 134,
        "eligible_query_rows": 1_000,
        "centered_score_relative_rmse": 0.05,
        "probability_relative_rmse": 0.10,
        "js_divergence_mean": 0.01,
        "js_divergence_worst": 0.02,
        "top1_route_agreement_fraction": 0.95,
        "maximum_causal_probability_leakage": 0.0,
    }
    fit_check_metrics = {
        **fit_metrics,
        "case_count": 20,
        "eligible_query_rows": 200,
        "centered_score_relative_rmse": 0.10,
    }
    score_operation = {
        "fit_cases_consumed": 134,
        "fit_check_cases_consumed": 20,
        "selection_cases_consumed": 0,
        "true_validation_cases_consumed": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "target_mutations_during_compile": 0,
    }
    compute_observations = []
    distinct_case_receipts = []
    for index, cuda_uuid in enumerate(H776_COMPOSED_LOGICAL_CUDA_UUIDS):
        case_receipt = f"{300 + index:064x}"
        distinct_case_receipts.append(case_receipt)
        observation_body = {
            "assignment_index": index,
            "intended_logical_device": f"cuda:{index}",
            "observed_logical_device": f"cuda:{index}",
            "observed_device_type": "cuda",
            "observed_device_index": index,
            "cuda_uuid": cuda_uuid,
            "cuda_name": H776_COMPOSED_CUDA_DEVICE_NAME,
            "cached_evidence_sha256": case_receipt,
            "cached_content_sha256": f"{310 + index:064x}",
            "cached_transform_receipt_sha256": f"{320 + index:064x}",
            "cached_trajectory_sha256": trajectory_sha256s[index],
            "cached_case_ordinal": index,
            "analytical_operation": (
                "masked_centered_score_energy_plus_valid_cell_count"
            ),
            "analytical_tensor_element_count": 1_024,
            "analytical_scalar_dtype": "torch.float64",
            "analytical_scalar": float(index + 1),
            "forced_host_scalar_read": True,
        }
        compute_observations.append(
            {
                **observation_body,
                "case_execution_receipt_sha256": _sha256(observation_body),
            }
        )
    solver_body = {
        "intended_logical_device": "cuda:1",
        "observed_logical_device": "cuda:1",
        "observed_device_type": "cuda",
        "observed_device_index": 1,
        "cuda_uuid": H776_COMPOSED_LOGICAL_CUDA_UUIDS[1],
        "cuda_name": H776_COMPOSED_CUDA_DEVICE_NAME,
        "source_seed_sha256": f"{340:064x}",
        "source_query_weight_sha256": f"{341:064x}",
        "source_key_weight_sha256": f"{342:064x}",
        "analytical_operation": "dense_seed_qk_energy_plus_parameter_count",
        "analytical_tensor_element_count": 2_048,
        "analytical_scalar_dtype": "torch.float64",
        "analytical_scalar": 3.0,
        "forced_host_scalar_read": True,
    }
    solver_observation = {
        **solver_body,
        "solver_execution_receipt_sha256": _sha256(solver_body),
    }
    observed_execution_body = {
        "schema": "neural-forge-lfm2-qk-score-observed-execution-v1",
        "requested_solver_device": "cuda:1",
        "requested_compute_devices": ["cuda:0", "cuda:1"],
        "requested_compute_device_count": 2,
        "observed_compute_device_count": 2,
        "every_requested_compute_device_observed": True,
        "solver_device_observed": True,
        "cached_evidence_analytical_tensor_operation_count": 2,
        "solver_tensor_operation_count": 1,
        "distinct_case_receipts": distinct_case_receipts,
        "distinct_case_receipt_count": 2,
        "distinct_case_execution_receipt_count": 2,
        "per_device_case_counts": {"cuda:0": 1, "cuda:1": 1},
        "compute_observations": compute_observations,
        "solver_observation": solver_observation,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    observed_execution = {
        **observed_execution_body,
        "receipt_sha256": _sha256(observed_execution_body),
    }
    cache_resource_body = {
        "schema": "neural-forge-lfm2-qk-fixed-linearization-cache-resource-v2",
        "cache_scope": "one_gauss_newton_sweep",
        "resident_storage": "compute_device",
        "case_assignment": "authenticated_bank_order_round_robin",
        "static_device_shards": [
            {
                "logical_device": f"cuda:{device_index}",
                "case_assignment_indices": list(range(device_index, 134, 2)),
                "case_count": 67,
            }
            for device_index in range(2)
        ],
        "source_fit_case_count": 134,
        "resident_prepared_case_count": 134,
        "linearization_replay_call_count": 134,
        "per_device_case_counts": {"cuda:0": 67, "cuda:1": 67},
        "per_device_logical_tensor_bytes": {
            "cuda:0": 1_112_504_448,
            "cuda:1": 1_112_504_448,
        },
        "total_logical_tensor_bytes": 2_225_008_896,
        "maximum_device_logical_tensor_bytes": 1_112_504_448,
        "cache_reused_for_right_hand_side": True,
        "cache_reused_for_normal_operator_calls": True,
        "cache_invalidated_at_sweep_end": True,
        "compute_device_concurrency": ("thread_barrier_concurrent_static_cuda_shards"),
        "actual_multi_cuda_concurrent_path_executed": True,
        "successful_right_hand_side_calls": 1,
        "successful_normal_operator_calls": 3,
        "concurrent_right_hand_side_calls": 1,
        "concurrent_normal_operator_calls": 3,
        "concurrent_per_case_gradients_retained_until_reduction": True,
        "solver_gradient_reduction_order": "original_global_fit_case_order",
        "nondeterministic_timing_fields_included": False,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    cache_resource = {
        **cache_resource_body,
        "resource_receipt_sha256": _sha256(cache_resource_body),
    }
    execution = {
        "solver_device": "cuda:1",
        "compute_devices": ["cuda:0", "cuda:1"],
        "case_assignment": "authenticated_bank_order_round_robin",
        "compute_device_concurrency": ("thread_barrier_concurrent_static_cuda_shards"),
        "streamed_evidence": False,
        "fixed_linearization_cache": True,
        "linearization_cache_scope": "one_gauss_newton_sweep",
        "maximum_logically_resident_prepared_cases": 134,
        "jvp_vjp_dtype": "torch.float32",
        "pcg_vector_dtype": "torch.float32",
        "scalar_reduction_dtype": "torch.float64",
        "observed_execution_receipt": observed_execution,
    }
    score_body = {
        "schema": "neural-forge-lfm2-cached-qk-score-residual-fold-v1",
        "policy": score_policy,
        "seed_sha256": f"{340:064x}",
        "query_weight_sha256": f"{350:064x}",
        "key_weight_sha256": f"{351:064x}",
        "accepted": True,
        "changed": True,
        "status": "fit_check_accepted",
        "fit_check_evaluated": True,
        "fit_check_triggered_refit": False,
        "source_evidence_unchanged": True,
        "fit_metrics": fit_metrics,
        "fit_check_metrics": fit_check_metrics,
        "execution": execution,
        "sweeps": [
            {
                "sweep": 0,
                "pcg": {"normal_operator_calls": 3},
                "linearization_cache_resource_receipt": cache_resource,
            }
        ],
        "operation_audit": score_operation,
    }
    score = {**score_body, "report_sha256": _sha256(score_body)}
    global_body = {
        "schema": "neural-forge-lfm2-qk-global-gauge-sync-v1",
        "policy": composed_policy["global_policy"],
        "query_weight_sha256": f"{341:064x}",
        "key_weight_sha256": f"{342:064x}",
        "resource_usage": {
            "total_rows": 40_656,
            "total_score_elements": 50_000_000,
            "total_factor_elements": 30_000_000,
            "total_output_elements": 100_000_000,
            "total_svd_workspace_elements": 40_000_000,
        },
    }
    global_sync = {
        **global_body,
        "report_sha256": _sha256(global_body),
    }
    producer_body = {
        "schema": "neural-forge-lfm2-composed-qk-attention-producer-v1",
        "candidate_id": H776_COMPOSED_ATTENTION_CANDIDATE_ID,
        "policy": composed_policy,
        "policy_sha256": composed_policy["authentication_sha256"],
        "source_split_sha256": fit_bank_sha256,
        "upstream_fit_case_count": 154,
        "upstream_fit_case_bundle_sha256": upstream_case_bundle_sha256,
        "upstream_trajectory_sha256s": list(trajectory_sha256s),
        "upstream_trajectory_bundle_sha256": trajectory_bundle_sha256,
        "qk_fit_source_ordinals": list(range(134)),
        "qk_fit_check_source_ordinals": list(range(134, 154)),
        "global_sync_report_sha256": global_sync["report_sha256"],
        "score_result_report_sha256": score["report_sha256"],
        "corrected_producer_sha256": f"{352:064x}",
        "corrected_query_weight_sha256": score["query_weight_sha256"],
        "corrected_key_weight_sha256": score["key_weight_sha256"],
        "corrected_value_weight_sha256": f"{353:064x}",
        "ordinary_dense_target_qkv": True,
        "operation_audit": {
            "model_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        },
    }
    producer = {
        **producer_body,
        "artifact_sha256": _sha256(producer_body),
    }
    qk_sha = _sha256({"qk": "composed"})
    outer_body = {
        "schema": ("neural-forge-lfm2-composed-qk-sequence-effective-attention-v1"),
        "candidate_id": H776_COMPOSED_ATTENTION_CANDIDATE_ID,
        "composed_qk_artifact_sha256": producer["artifact_sha256"],
        "fixed_query_key_sha256": qk_sha,
        "upstream_fit_case_bundle_sha256": upstream_case_bundle_sha256,
        "upstream_trajectory_sha256s": list(trajectory_sha256s),
        "upstream_trajectory_bundle_sha256": trajectory_bundle_sha256,
        "operation_audit": {
            "model_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        },
    }
    outer = {**outer_body, "artifact_sha256": _sha256(outer_body)}
    evaluations = {}
    candidate_reports = {}
    for ordinal, candidate_id in enumerate(H776_COMPOSED_ATTENTION_CANDIDATE_IDS):
        if candidate_id == H776_COMPOSED_ATTENTION_CANDIDATE_ID:
            artifact = outer
        elif candidate_id == H776_ATTENTION_CANDIDATE_IDS[0]:
            artifact = {
                "candidate_id": candidate_id,
                "compiled_state_sha256": _structural_attention_state_sha256(
                    {
                        "q_proj.weight": torch.tensor(
                            [1.0],
                            dtype=torch.float32,
                        )
                    }
                ),
            }
        else:
            artifact = {"candidate_id": candidate_id}
        evaluation_body = {
            "candidate_id": candidate_id,
            "operator_artifact": artifact,
            "operator_artifact_sha256": _sha256(artifact),
            "query_key_score_state_sha256": (
                qk_sha
                if candidate_id == H776_COMPOSED_ATTENTION_CANDIDATE_ID
                else _sha256({"qk": candidate_id})
            ),
            "operator_compiled_from_splits": ["fit"],
            "mlp_compiled_from_splits": ["fit"],
            "candidate_evaluated_on_splits": ["selection"],
            "validation_phase_count_consumed": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "selection_key": [float(ordinal + 1)] * 10 + [candidate_id],
        }
        evaluation = {
            **evaluation_body,
            "report_sha256": _sha256(evaluation_body),
        }
        evaluations[candidate_id] = evaluation
        candidate_reports[candidate_id] = evaluation["report_sha256"]
    selected = H776_COMPOSED_ATTENTION_CANDIDATE_IDS[0]
    selection_body = {
        "candidate_ids": list(H776_COMPOSED_ATTENTION_CANDIDATE_IDS),
        "selected_candidate_id": selected,
        "candidate_count": 4,
        "candidate_report_sha256": candidate_reports,
        "selected_candidate_report_sha256": candidate_reports[selected],
        "selection_split_only": True,
        "winner_frozen_before_validation": True,
        "validation_phase_count_consumed_before_freeze": 0,
    }
    selection = {
        **selection_body,
        "report_sha256": _sha256(selection_body),
    }
    next_sha = "7" * 64
    body = {
        "layer_index": 2,
        "operator_kind": "full_attention",
        "input_bank_sha256": "8" * 64,
        "next_bank_sha256": next_sha,
        "operator_artifact": evaluations[selected]["operator_artifact"],
        "split_phase_counts": dict(H776_SPLIT_PHASE_COUNTS),
        "split_metrics": {"validation": {"relative_rmse": 0.5}},
        "solver_evidence_policy": {
            "operator_case_count": 154,
            "attention_candidate_selection": selection,
            "attention_candidate_evaluations": evaluations,
            "attention_each_candidate_has_own_fit_only_mlp": True,
            "attention_winner_frozen_before_validation": True,
            "attention_validation_replays_before_winner_freeze": 0,
            "attention_candidate_inventory_mode": ("opt_in_authenticated_four_arm"),
            "composed_qk_attention_policy": composed_policy,
            "composed_qk_producer_artifact_sha256": (producer["artifact_sha256"]),
            "composed_qk_producer_artifact": producer,
            "composed_qk_global_sync": global_sync,
            "composed_qk_score_result": score,
        },
    }
    report = {**body, "result_sha256": _sha256(body)}
    return SimpleNamespace(
        report=report,
        operator_artifact=_FakeSelectedDirectArtifact(
            evaluations[selected]["operator_artifact"]
        ),
        next_bank=SimpleNamespace(layer_index=3, state_bank_sha256=next_sha),
        expected_fit_bank_sha256=fit_bank_sha256,
        expected_fit_trajectory_sha256s=trajectory_sha256s,
    )


def _reseal_composed_result(result, mutate):
    report = runner._plain(result.report)
    policy = report["solver_evidence_policy"]
    score = policy["composed_qk_score_result"]
    global_sync = policy["composed_qk_global_sync"]
    producer = policy["composed_qk_producer_artifact"]
    evaluations = policy["attention_candidate_evaluations"]
    composed_evaluation = evaluations[H776_COMPOSED_ATTENTION_CANDIDATE_ID]
    outer = composed_evaluation["operator_artifact"]
    mutate(
        policy=policy,
        score=score,
        global_sync=global_sync,
        producer=producer,
        outer=outer,
    )

    composed_policy = policy["composed_qk_attention_policy"]
    score["policy"] = composed_policy["score_policy"]
    global_sync["policy"] = composed_policy["global_policy"]
    producer["policy"] = composed_policy
    producer["policy_sha256"] = composed_policy["authentication_sha256"]

    score.pop("report_sha256", None)
    score["report_sha256"] = _sha256(score)
    global_sync.pop("report_sha256", None)
    global_sync["report_sha256"] = _sha256(global_sync)
    producer["score_result_report_sha256"] = score["report_sha256"]
    producer["global_sync_report_sha256"] = global_sync["report_sha256"]
    producer.pop("artifact_sha256", None)
    producer["artifact_sha256"] = _sha256(producer)
    outer["composed_qk_artifact_sha256"] = producer["artifact_sha256"]
    outer.pop("artifact_sha256", None)
    outer["artifact_sha256"] = _sha256(outer)

    composed_evaluation["operator_artifact"] = outer
    composed_evaluation["operator_artifact_sha256"] = _sha256(outer)
    composed_evaluation.pop("report_sha256", None)
    composed_evaluation["report_sha256"] = _sha256(composed_evaluation)
    selection = policy["attention_candidate_selection"]
    selection["candidate_report_sha256"][H776_COMPOSED_ATTENTION_CANDIDATE_ID] = (
        composed_evaluation["report_sha256"]
    )
    selected_id = selection["selected_candidate_id"]
    selection["selected_candidate_report_sha256"] = selection[
        "candidate_report_sha256"
    ][selected_id]
    selection.pop("report_sha256", None)
    selection["report_sha256"] = _sha256(selection)
    policy["composed_qk_producer_artifact_sha256"] = producer["artifact_sha256"]
    report["operator_artifact"] = evaluations[selected_id]["operator_artifact"]
    report.pop("result_sha256", None)
    report["result_sha256"] = _sha256(report)
    return SimpleNamespace(
        report=report,
        operator_artifact=_FakeSelectedDirectArtifact(
            evaluations[selected_id]["operator_artifact"]
        ),
        next_bank=result.next_bank,
        expected_fit_bank_sha256=result.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=(result.expected_fit_trajectory_sha256s),
    )


def _endpoint_attention_result():
    result = _composed_attention_result()
    report = runner._plain(result.report)
    policy = report["solver_evidence_policy"]
    producer = policy["composed_qk_producer_artifact"]
    endpoint_policy = _h776_endpoint_residual_attention_policy().as_dict()
    fit_trajectories = tuple(result.expected_fit_trajectory_sha256s[:134])
    fit_check_trajectories = tuple(result.expected_fit_trajectory_sha256s[134:])

    source_receipts = {}
    source_shapes = {
        "q_proj.weight": [10, 10],
        "k_proj.weight": [5, 10],
        "v_proj.weight": [5, 10],
        "out_proj.weight": [10, 10],
    }
    for ordinal, (name, shape) in enumerate(source_shapes.items()):
        source_receipts[name] = {
            "shape": shape,
            "dtype": "torch.bfloat16",
            "logical_parameters": math.prod(shape),
            "tensor_sha256": f"{8_000 + ordinal:064x}",
            "structural_tensor_sha256": f"{8_100 + ordinal:064x}",
        }
    source_state_sha = f"{8_200:064x}"
    source_structural_sha = f"{8_201:064x}"
    source_inventory_body = {
        "schema": "neural-forge-lfm2-endpoint-source-qkvo-inventory-v1",
        "tensor_receipts": source_receipts,
        "source_state_sha256": source_state_sha,
        "structural_source_state_sha256": source_structural_sha,
        "logical_parameters": 300,
        "count_derived_from_actual_dense_tensors": True,
    }
    source_inventory = {
        **source_inventory_body,
        "inventory_sha256": _sha256(source_inventory_body),
    }

    proposal, _ = _endpoint_execution_proposal(fit_trajectories)
    proposal_body = {
        "schema": "neural-forge-lfm2-endpoint-qk-fit-proposal-v1",
        "policy": endpoint_policy["endpoint_policy"],
        "policy_sha256": endpoint_policy["endpoint_policy"]["authentication_sha256"],
        "changed": True,
        "solver_device": "cuda:1",
        "compute_devices": ["cuda:0", "cuda:1"],
        "case_assignment": "authenticated_bank_order_round_robin",
        "observed_execution_receipt": proposal["observed_execution_receipt"],
        "fit_check_accessible": False,
        "fit_check_cases_consumed": 0,
        "sweeps": [
            {
                "sweep": 0,
                "linear_solver": "deterministic_matrix_free_ridge_normal_cg",
                "solve_report": {
                    "converged": True,
                    "usable_direction": True,
                    "usable_direction_rule": (
                        "finite_nonzero_solution_and_relative_residual_or_"
                        "maximum_iterations"
                    ),
                    "direction_is_finite": True,
                    "direction_has_nonzero_entry": True,
                    "iterations": 4,
                    "relative_normal_residual": 0.005,
                    "termination": "relative_normal_residual",
                    "normal_operator_calls": 5,
                },
                "endpoint_residual_energy": 1.0,
                "route_residual_energy": 1.0,
                "unscaled_query_relative_delta": 0.1,
                "unscaled_key_relative_delta": 0.1,
                "trust_scale": 0.75,
                "frontier": [],
                "selected_alpha": 0.5,
            }
        ],
        "operation_audit": {
            "fit_cases_consumed": 134,
            "fit_check_cases_consumed": 0,
            "matrix_free_ridge_normal_solves": 1,
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "blank_target_value_reads": 0,
        },
    }
    proposal_report = {
        **proposal_body,
        "report_sha256": _sha256(proposal_body),
    }
    block_body = {
        "schema": "neural-forge-lfm2-composed-endpoint-residual-block-v1",
        "block_index": 0,
        "fit_trajectory_sha256s": list(fit_trajectories),
        "qk_fit_proposal_sha256": proposal_report["report_sha256"],
        "fit_check_accessible": False,
        "fit_check_cases_consumed": 0,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    block = {**block_body, "block_sha256": _sha256(block_body)}
    metric = {
        "case_count": 134,
        "eligible_route_rows": 1_000,
        "endpoint_relative_rmse": 0.5,
        "probability_relative_rmse": 0.1,
        "js_divergence_mean": 0.01,
        "js_divergence_worst": 0.02,
        "top1_route_agreement_fraction": 0.95,
        "maximum_causal_probability_leakage": 0.0,
    }
    final_metric = {**metric, "endpoint_relative_rmse": 0.1}
    check_metric = {**metric, "case_count": 20, "eligible_route_rows": 200}
    endpoint_body = {
        "schema": "neural-forge-lfm2-composed-endpoint-residual-attention-v1",
        "candidate_id": "endpoint_residual_alternating_qkvo",
        "accepted": True,
        "policy": endpoint_policy,
        "policy_sha256": endpoint_policy["authentication_sha256"],
        "composed_qk_artifact_sha256": producer["artifact_sha256"],
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_inventory["inventory_sha256"],
        "source_structural_state_sha256": source_structural_sha,
        "fit_trajectory_sha256s": list(fit_trajectories),
        "fit_check_trajectory_sha256s": list(fit_check_trajectories),
        "block_sha256s": [block["block_sha256"]],
        "outer_block_count": 1,
        "baseline_fit_metrics": metric,
        "final_fit_metrics": final_metric,
        "baseline_fit_check_metrics": check_metric,
        "final_fit_check_metrics": check_metric,
        "fit_check_endpoint_and_route_nonregressing": True,
        "fit_check_first_metric_evaluation_after_all_outer_blocks": True,
        "fit_check_triggered_refit": False,
        "fit_check_role": (
            "reused_development_acceptance_gate_not_unbiased_validation"
        ),
        "unbiased_generalization_claimed": False,
        "fresh_whole_child_intelligence_validation_required": True,
        "proof_scope": (
            "source_bound_local_attention_operator_fold_not_whole_child_intelligence"
        ),
        "direct_parent_probability_barycenter_target": True,
        "direct_parent_attention_endpoint_target": True,
        "parent_probability_evidence_modes": [
            "authenticated_cached_parent_attention_probabilities"
        ],
        "authenticated_cached_parent_probability_case_count": 154,
        "legacy_recomputed_parent_probability_case_count": 0,
        "all_parent_probability_targets_from_authenticated_cache": True,
        "legacy_parent_probability_recomputation_allowed": False,
        "ordinary_dense_target_qkvo": True,
        "source_logical_parameters": 300,
        "target_logical_parameters": 150,
        "parameter_compaction_ratio": 2.0,
        "operation_audit": {
            "model_forward_calls": 0,
            "model_generate_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
        },
    }
    endpoint = {
        **endpoint_body,
        "artifact_sha256": _sha256(endpoint_body),
    }
    endpoint_id = "endpoint_residual_alternating_qkvo"
    endpoint_evaluation_body = {
        "candidate_id": endpoint_id,
        "operator_artifact": endpoint,
        "operator_artifact_sha256": _sha256(endpoint),
        "query_key_score_state_sha256": f"{8_300:064x}",
        "operator_compiled_from_splits": ["fit"],
        "mlp_compiled_from_splits": ["fit"],
        "candidate_evaluated_on_splits": ["selection"],
        "validation_phase_count_consumed": 0,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "selection_key": [5.0] * 10 + [endpoint_id],
    }
    endpoint_evaluation = {
        **endpoint_evaluation_body,
        "report_sha256": _sha256(endpoint_evaluation_body),
    }
    evaluations = policy["attention_candidate_evaluations"]
    evaluations[endpoint_id] = endpoint_evaluation
    selection = policy["attention_candidate_selection"]
    selection.pop("report_sha256")
    selection["candidate_ids"] = list(H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS)
    selection["candidate_count"] = 5
    selection["candidate_report_sha256"][endpoint_id] = endpoint_evaluation[
        "report_sha256"
    ]
    selection["candidate_local_mlp_preparation_count"] = 5
    selection["cross_candidate_prepared_mlp_basis_reused"] = False
    selection["endpoint_residual_candidate_fit_only_mlp_evaluated"] = True
    selection["report_sha256"] = _sha256(selection)

    output_prior = {
        "construction": "zero_from_target_shape_and_dtype_metadata_only",
        "shape": [10, 10],
        "dtype": "torch.bfloat16",
        "tensor_sha256": f"{8_400:064x}",
        "blank_target_values_read": False,
    }
    producer = policy["composed_qk_producer_artifact"]
    score = policy["composed_qk_score_result"]
    stage_one_body = {
        "schema": "neural-forge-lfm2-composed-qk-stage-one-attempt-v1",
        "endpoint_start_artifact_sha256": producer["artifact_sha256"],
        "global_gauge_fallback_start_artifact_sha256": f"{8_401:064x}",
        "endpoint_start_source": "accepted_score_residual_corrected_dense_qkv",
        "score_result_report_sha256": score["report_sha256"],
        "score_result_status": score["status"],
        "score_result_accepted": True,
        "composed_candidate_eligible": True,
        "composed_candidate_artifact_sha256": producer["artifact_sha256"],
        "failed_composed_candidate_omitted": False,
        "endpoint_start_issued_independently": True,
        "operation_audit": {
            "model_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        },
    }
    stage_one = {
        **stage_one_body,
        "attempt_sha256": _sha256(stage_one_body),
    }
    policy.update(
        {
            "attention_candidate_inventory_mode": ("opt_in_authenticated_five_arm"),
            "endpoint_residual_attention_policy": endpoint_policy,
            "endpoint_residual_attention_artifact_sha256": (
                endpoint["artifact_sha256"]
            ),
            "endpoint_residual_attention_artifact": endpoint,
            "endpoint_residual_block_reports": [block],
            "endpoint_residual_qk_fit_proposal_reports": [proposal_report],
            "endpoint_residual_output_prior_receipt": output_prior,
            "endpoint_residual_blank_target_values_read": False,
            "attention_each_candidate_prepares_own_mlp_basis": True,
            "attention_cross_candidate_prepared_mlp_basis_reuse": False,
            "endpoint_residual_fit_only_downstream_mlp_evaluation": True,
            "endpoint_residual_fit_check_final_one_shot_only": True,
            "composed_qk_candidate_eligible": True,
            "composed_qk_stage_one_attempt": stage_one,
            "endpoint_qk_start_artifact": producer,
            "endpoint_qk_start_source": ("accepted_score_residual_corrected_dense_qkv"),
        }
    )
    report.pop("result_sha256")
    report["result_sha256"] = _sha256(report)
    return SimpleNamespace(
        report=report,
        operator_artifact=result.operator_artifact,
        next_bank=result.next_bank,
        expected_fit_bank_sha256=result.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=result.expected_fit_trajectory_sha256s,
    )


def _endpoint_attention_result_with_rejected_composed():
    original = _endpoint_attention_result()
    report = runner._plain(original.report)
    policy = report["solver_evidence_policy"]
    score = policy["composed_qk_score_result"]
    score.pop("report_sha256")
    score.update(
        {
            "accepted": False,
            "changed": False,
            "status": "pcg_nonconverged_seed_fallback",
            "fit_check_evaluated": False,
            "fit_check_metrics": None,
        }
    )
    score["operation_audit"]["fit_check_cases_consumed"] = 0
    score.pop("seed_fit_check_metrics", None)
    score.pop("proposed_fit_check_metrics", None)
    score["report_sha256"] = _sha256(score)

    accepted_producer = policy["composed_qk_producer_artifact"]
    fallback_body = {
        **accepted_producer,
        "schema": "neural-forge-lfm2-endpoint-qk-start-v1",
        "score_result_report_sha256": score["report_sha256"],
        "score_result_accepted": False,
        "score_residual_fourth_arm_eligible": False,
        "starting_producer_sha256": f"{8_402:064x}",
        "starting_query_weight_sha256": policy["composed_qk_global_sync"][
            "query_weight_sha256"
        ],
        "starting_key_weight_sha256": policy["composed_qk_global_sync"][
            "key_weight_sha256"
        ],
        "starting_value_weight_sha256": f"{8_403:064x}",
        "starting_state_source": (
            "fit_structural_groups_plus_global_gauge_qk_plus_fit_only_v"
        ),
    }
    fallback_body.pop("artifact_sha256", None)
    fallback_start = {
        **fallback_body,
        "artifact_sha256": _sha256(fallback_body),
    }
    stage_one_body = {
        "schema": "neural-forge-lfm2-composed-qk-stage-one-attempt-v1",
        "endpoint_start_artifact_sha256": fallback_start["artifact_sha256"],
        "global_gauge_fallback_start_artifact_sha256": (
            fallback_start["artifact_sha256"]
        ),
        "endpoint_start_source": "structural_global_gauge_dense_qkv",
        "score_result_report_sha256": score["report_sha256"],
        "score_result_status": score["status"],
        "score_result_accepted": False,
        "composed_candidate_eligible": False,
        "composed_candidate_artifact_sha256": None,
        "failed_composed_candidate_omitted": True,
        "endpoint_start_issued_independently": True,
        "operation_audit": {
            "model_forward_calls": 0,
            "training_steps": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
        },
    }
    stage_one = {
        **stage_one_body,
        "attempt_sha256": _sha256(stage_one_body),
    }

    endpoint = policy["endpoint_residual_attention_artifact"]
    endpoint.pop("artifact_sha256")
    endpoint.update(
        {
            "composed_qk_artifact_sha256": None,
            "endpoint_qk_start_artifact_sha256": (fallback_start["artifact_sha256"]),
            "endpoint_qk_start_artifact_schema": (
                "neural-forge-lfm2-endpoint-qk-start-v1"
            ),
            "endpoint_qk_start_source": "structural_global_gauge_dense_qkv",
            "composed_score_residual_candidate_eligible": False,
            "composed_score_residual_result_sha256": score["report_sha256"],
        }
    )
    endpoint["artifact_sha256"] = _sha256(endpoint)
    endpoint_id = H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_ID
    evaluations = policy["attention_candidate_evaluations"]
    evaluations.pop(H776_COMPOSED_ATTENTION_CANDIDATE_ID)
    endpoint_evaluation = evaluations[endpoint_id]
    endpoint_evaluation.pop("report_sha256")
    endpoint_evaluation["operator_artifact"] = endpoint
    endpoint_evaluation["operator_artifact_sha256"] = _sha256(endpoint)
    endpoint_evaluation["report_sha256"] = _sha256(endpoint_evaluation)

    selection = policy["attention_candidate_selection"]
    selection.pop("report_sha256")
    selection["candidate_ids"] = list(
        H776_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS
    )
    selection["candidate_count"] = 4
    selection["candidate_local_mlp_preparation_count"] = 4
    selection["candidate_report_sha256"].pop(H776_COMPOSED_ATTENTION_CANDIDATE_ID)
    selection["candidate_report_sha256"][endpoint_id] = endpoint_evaluation[
        "report_sha256"
    ]
    selected_id = selection["selected_candidate_id"]
    selection["selected_candidate_report_sha256"] = selection[
        "candidate_report_sha256"
    ][selected_id]
    selection["report_sha256"] = _sha256(selection)

    policy.update(
        {
            "attention_candidate_inventory_mode": (
                "opt_in_authenticated_endpoint_four_arm_without_rejected_"
                "composed_candidate"
            ),
            "composed_qk_producer_artifact_sha256": None,
            "composed_qk_producer_artifact": None,
            "composed_qk_candidate_eligible": False,
            "composed_qk_stage_one_attempt": stage_one,
            "endpoint_qk_start_artifact": fallback_start,
            "endpoint_qk_start_source": "structural_global_gauge_dense_qkv",
            "composed_qk_score_result": score,
            "endpoint_residual_attention_artifact_sha256": (
                endpoint["artifact_sha256"]
            ),
            "endpoint_residual_attention_artifact": endpoint,
        }
    )
    report.pop("result_sha256")
    report["result_sha256"] = _sha256(report)
    return SimpleNamespace(
        report=report,
        operator_artifact=original.operator_artifact,
        next_bank=original.next_bank,
        expected_fit_bank_sha256=original.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
    )


def test_endpoint_attention_result_authenticates_five_arm_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    result = _endpoint_attention_result()
    audit = authenticate_h776_endpoint_residual_attention_result(
        result,
        input_bank_sha256="8" * 64,
        expected_fit_bank_sha256=result.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=result.expected_fit_trajectory_sha256s,
        pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
    )

    assert audit["candidate_ids"] == list(
        H776_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS
    )
    assert audit["scientific_outcome"] == ("honest_negative_or_prior_arm_selected")
    assert audit["all_parent_probability_targets_from_authenticated_cache"] is True
    assert audit["blank_target_values_read"] is False
    assert len(audit["endpoint_block_device_execution"]) == 1


def test_endpoint_attention_authenticates_rejected_composed_without_fit_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    result = _endpoint_attention_result_with_rejected_composed()
    audit = authenticate_h776_endpoint_residual_attention_result(
        result,
        input_bank_sha256="8" * 64,
        expected_fit_bank_sha256=result.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=result.expected_fit_trajectory_sha256s,
        pins=_pins(tmp_path, endpoint_residual_attention_enabled=True),
    )

    assert audit["candidate_ids"] == list(
        H776_ENDPOINT_WITHOUT_COMPOSED_ATTENTION_CANDIDATE_IDS
    )
    assert audit["composed_score_residual_candidate_eligible"] is False
    assert audit["endpoint_qk_start_source"] == ("structural_global_gauge_dense_qkv")
    assert audit["composed_score_outcome"]["fit_check_evaluated"] is False
    assert audit["composed_score_outcome"]["fit_check_cases_consumed"] == 0
    assert audit["composed_score_outcome"]["status"] == (
        "pcg_nonconverged_seed_fallback"
    )


def test_composed_attention_result_authenticates_four_arm_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    result = _composed_attention_result()
    audit = authenticate_h776_composed_attention_result(
        result,
        input_bank_sha256="8" * 64,
        expected_fit_bank_sha256=result.expected_fit_bank_sha256,
        expected_fit_trajectory_sha256s=(result.expected_fit_trajectory_sha256s),
        pins=_pins(tmp_path, composed_qk_attention_enabled=True),
    )

    assert audit["candidate_ids"] == list(H776_COMPOSED_ATTENTION_CANDIDATE_IDS)
    assert audit["scientific_outcome"] == ("honest_negative_or_legacy_arm_selected")
    assert audit["two_3090_score_execution_authenticated"] is True


def test_composed_attention_result_rejects_resealed_score_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    result = _composed_attention_result()
    report = runner._plain(result.report)
    score = report["solver_evidence_policy"]["composed_qk_score_result"]
    score.pop("report_sha256")
    score["accepted"] = False
    score["report_sha256"] = _sha256(score)
    body = dict(report)
    body.pop("result_sha256")
    tampered = SimpleNamespace(
        report={**body, "result_sha256": _sha256(body)},
        operator_artifact=result.operator_artifact,
        next_bank=result.next_bank,
    )

    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=result.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(result.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_consumed_count_lie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        parts["score"]["operation_audit"]["fit_cases_consumed"] = 0
        parts["score"]["operation_audit"]["fit_check_cases_consumed"] = 0

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_relaxed_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        composed_policy = parts["policy"]["composed_qk_attention_policy"]
        composed_policy["score_policy"]["maximum_fit_centered_score_relative_rmse"] = (
            0.35
        )
        composed_policy.pop("authentication_sha256", None)
        composed_policy["authentication_sha256"] = _sha256(composed_policy)

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_alien_fit_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        alien = ["a" * 64, *parts["producer"]["upstream_trajectory_sha256s"][1:]]
        alien_bundle = _sha256(alien)
        parts["producer"]["source_split_sha256"] = "b" * 64
        parts["producer"]["upstream_trajectory_sha256s"] = alien
        parts["producer"]["upstream_trajectory_bundle_sha256"] = alien_bundle
        parts["outer"]["upstream_trajectory_sha256s"] = alien
        parts["outer"]["upstream_trajectory_bundle_sha256"] = alien_bundle

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_missing_live_legacy_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()
    missing_live = SimpleNamespace(
        report=original.report,
        next_bank=original.next_bank,
    )
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            missing_live,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_mutated_live_direct_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    result = _composed_attention_result()
    result.operator_artifact._state["q_proj.weight"].add_(1.0)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            result,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=result.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(result.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_fake_second_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        receipt = parts["score"]["execution"]["observed_execution_receipt"]
        observation = receipt["compute_observations"][1]
        observation["cuda_uuid"] = H776_COMPOSED_LOGICAL_CUDA_UUIDS[0]
        observation.pop("case_execution_receipt_sha256", None)
        observation["case_execution_receipt_sha256"] = _sha256(observation)
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = _sha256(receipt)

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_cache_resource_lie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        resource = parts["score"]["sweeps"][0]["linearization_cache_resource_receipt"]
        resource["resident_prepared_case_count"] = 1
        resource["linearization_replay_call_count"] = 1
        resource["per_device_case_counts"] = {"cuda:0": 1, "cuda:1": 0}
        resource.pop("resource_receipt_sha256", None)
        resource["resource_receipt_sha256"] = _sha256(resource)

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


def test_composed_attention_result_rejects_resealed_concurrent_score_count_lie(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        resource = parts["score"]["sweeps"][0]["linearization_cache_resource_receipt"]
        resource["concurrent_normal_operator_calls"] -= 1
        resource.pop("resource_receipt_sha256", None)
        resource["resource_receipt_sha256"] = _sha256(resource)

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
        )


@pytest.mark.parametrize("tamper_kind", ["alien_trajectory", "alien_seed"])
def test_composed_attention_result_rejects_resealed_gpu_workload_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_kind: str,
) -> None:
    _patch_fake_selected_direct_artifact(monkeypatch)
    original = _composed_attention_result()

    def mutate(**parts) -> None:
        receipt = parts["score"]["execution"]["observed_execution_receipt"]
        if tamper_kind == "alien_trajectory":
            observation = receipt["compute_observations"][0]
            observation["cached_trajectory_sha256"] = "e" * 64
            observation.pop("case_execution_receipt_sha256", None)
            observation["case_execution_receipt_sha256"] = _sha256(observation)
        else:
            solver = receipt["solver_observation"]
            solver["source_seed_sha256"] = "e" * 64
            solver.pop("solver_execution_receipt_sha256", None)
            solver["solver_execution_receipt_sha256"] = _sha256(solver)
        receipt.pop("receipt_sha256", None)
        receipt["receipt_sha256"] = _sha256(receipt)

    tampered = _reseal_composed_result(original, mutate)
    with pytest.raises(Lfm2CpcfV5RealAttentionProbeError):
        authenticate_h776_composed_attention_result(
            tampered,
            input_bank_sha256="8" * 64,
            expected_fit_bank_sha256=original.expected_fit_bank_sha256,
            expected_fit_trajectory_sha256s=(original.expected_fit_trajectory_sha256s),
            pins=_pins(tmp_path, composed_qk_attention_enabled=True),
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
    composed_args = _parser_args(tmp_path / "composed")
    composed_args.append("--composed-qk-attention")
    composed = runner.config_from_args(parser.parse_args(composed_args))
    assert composed.composed_qk_attention_enabled is True
    assert composed.fold.maximum_process_rss_bytes == 64 * 1024**3
    assert (
        composed.observer_maximum_process_rss_bytes
        == H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES
    )
    endpoint_args = _parser_args(tmp_path / "endpoint")
    endpoint_args.append("--endpoint-residual-attention")
    endpoint = runner.config_from_args(parser.parse_args(endpoint_args))
    assert endpoint.composed_qk_attention_enabled is False
    assert endpoint.endpoint_residual_attention_enabled is True
    assert endpoint.fold.maximum_process_rss_bytes == 64 * 1024**3
    assert (
        endpoint.observer_maximum_process_rss_bytes
        == H776_COMPOSED_MAXIMUM_PROCESS_RSS_BYTES
    )
    endpoint_composed = list(endpoint_args)
    endpoint_composed.append("--composed-qk-attention")
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(endpoint_composed))
    endpoint_legacy_transport = list(endpoint_args)
    endpoint_legacy_transport.append("--attention-sequential-transport")
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(endpoint_legacy_transport))
    composed_legacy_transport = list(composed_args)
    composed_legacy_transport.append("--attention-sequential-transport")
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(composed_legacy_transport))
    composed_bad_rss = _parser_args(tmp_path / "composed_bad_rss")
    composed_bad_rss.append("--composed-qk-attention")
    rss_index = composed_bad_rss.index("--maximum-process-rss-gib") + 1
    composed_bad_rss[rss_index] = "192"
    with pytest.raises(runner.Lfm2CpcfV5RealAttentionProbeRunnerError):
        runner.config_from_args(parser.parse_args(composed_bad_rss))
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
