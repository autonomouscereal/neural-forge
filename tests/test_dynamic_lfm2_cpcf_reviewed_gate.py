from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch
from torch import nn

import neural_forge.dynamic_lfm2_cpcf_reviewed_gate as reviewed_gate_module
import neural_forge.hf_lfm2_cpcf_candidate_replay as candidate_replay_module
import neural_forge.hf_lfm2_cpcf_v5_model_compiler as v5_compiler_module
from neural_forge.dynamic_behavior_execution import (
    BehaviorInferenceExecutionContract,
)
from neural_forge.dynamic_behavior_gate import resolve_behavior_gate_binding
from neural_forge.dynamic_circuit_ir import canonical_json_bytes, canonical_sha256
from neural_forge.dynamic_evidence_residency import (
    EVIDENCE_RESIDENCY_WINDOW_SCHEMA,
)
from neural_forge.dynamic_replay_scheduler import (
    DYNAMIC_REPLAY_BUFFER_SCHEMA,
    DYNAMIC_REPLAY_EXECUTION_SCHEMA,
    DYNAMIC_REPLAY_PARITY_SCHEMA,
    CudaReplayDeviceReceipt,
    InstalledReplayStateFingerprint,
    ReplayReplicaReceipt,
    ReplaySchedulerPolicy,
    ReplayWorkItem,
    build_replay_schedule_plan,
    compare_cpu_cuda_replay,
    fingerprint_replay_value,
)
from neural_forge.dynamic_lfm2_cpcf_reviewed_gate import (
    AUTOMATIC_RUNNER_REPORT_SCHEMA,
    COMPOSITE_PROMOTION_SCHEMA,
    CPCF_CAPTURE_SCHEMA,
    CPCF_RUN_SCHEMA,
    CanonicalJsonArtifact,
    Lfm2CpcfReviewedPromotionArtifacts,
    Lfm2CpcfReviewedPromotionError,
    Lfm2CpcfReviewedPromotionEvidence,
    Lfm2CpcfReviewedPromotionGate,
    authenticate_lfm2_cpcf_reviewed_gate_implementation,
    authenticate_lfm2_cpcf_reviewed_promotion_evidence,
    authenticate_lfm2_cpcf_reviewed_promotion_gate,
    build_cpcf_reviewed_promotion_report,
    fingerprint_cpcf_candidate_state,
    fingerprint_local_tokenizer_package,
    load_lfm2_cpcf_reviewed_promotion_evidence,
)
from neural_forge.hf_lfm2_intelligence_gate import (
    IntelligenceGatePolicy,
    build_lfm2_intelligence_holdout_manifest,
    build_manual_review_packet,
    evaluate_cached_intelligence,
    load_lfm2_intelligence_holdout_manifest,
)
from neural_forge.hf_lfm2_intelligence_holdout import (
    baseline_from_lfm2_intelligence_holdout_envelope,
    load_lfm2_intelligence_holdout_candidate_outputs,
    load_lfm2_intelligence_holdout_parent_envelope,
    seal_lfm2_intelligence_holdout_candidate_outputs,
    seal_lfm2_intelligence_holdout_parent_envelope,
)
from neural_forge.hf_lfm2_cpcf_model_fold import (
    LFM2_CPCF_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
    Lfm2CpcfCompletedLayerReceipt,
    Lfm2CpcfModelFoldPolicy,
)
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
    LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA,
    replay_lfm2_cpcf_candidate_work_payload,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    Lfm2CpcfShortConvV4SolverPolicy,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    Lfm2CpcfV5FirBasisPolicy,
    rescore_sealed_v4_short_conv_selection_frontier,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5_numerical_adapter import (
    Lfm2CpcfShortConvV5NumericalFrontierAdapter,
)
from neural_forge.hf_lfm2_cpcf_v5_model_compiler import (
    LFM2_CPCF_V5_ARTIFACT_SCHEMA,
    LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
    LFM2_CPCF_V5_RESUME_SCHEMA,
    Lfm2CpcfV5RuntimePolicy,
    lfm2_cpcf_v5_compiler_implementation,
)
from neural_forge.hf_lfm2_parent_relative_manual_gate import (
    bind_parent_relative_manual_ratings,
    build_parent_relative_manual_packet,
    evaluate_parent_relative_manual_retention,
)
from neural_forge.hf_tokenizer_bridge import fingerprint_hf_tokenizer_semantics


def _word_count(value: str) -> int:
    return len(value.replace(",", " ").replace(".", " ").split())


_CODE_BODIES = {
    "holdout_v1_python_00": "def holdout_is_even(n):\n    return n % 2 == 0",
    "holdout_v1_python_01": (
        "def holdout_clamp(x, low, high):\n"
        "    return low if x < low else high if x > high else x"
    ),
    "holdout_v1_python_02": "def holdout_last(items):\n    return items[-1]",
    "holdout_v1_python_03": "def holdout_reverse(text):\n    return text[::-1]",
    "holdout_v1_python_04": (
        "def holdout_between(x, low, high):\n    return low <= x and x <= high"
    ),
    "holdout_v1_python_05": (
        "def holdout_choose(flag, yes, no):\n    return yes if flag else no"
    ),
    "holdout_v1_python_06": ("def holdout_difference(a, b):\n    return a - b"),
    "holdout_v1_python_07": ("def holdout_join(left, right):\n    return left + right"),
}


def _passing_output(spec: Any, *, candidate: bool) -> str:
    validator = spec.validator
    if validator is None:
        assert spec.capability is not None
        terms = [
            group[-1] if candidate and len(group) > 1 else group[0]
            for group in spec.capability.required_concept_groups
        ]
        filler = "a coherent useful answer with clear practical context".split()
        output = " ".join(terms)
        index = 0
        while _word_count(output) < spec.capability.minimum_words:
            output = f"{output} {filler[index % len(filler)]}".strip()
            index += 1
        return output
    if validator.kind == "arithmetic_exact":
        return f"The checked result is {validator.expected_numeric}"
    if validator.kind == "python_pure_function":
        source = _CODE_BODIES[spec.case_id]
        return f"```python\n{source}\n```" if candidate else source
    if validator.exact_word is not None:
        return validator.exact_word
    terms = [
        group[-1] if candidate and len(group) > 1 else group[0]
        for group in validator.required_groups
    ]
    filler = "this concise response explains the requested result safely".split()
    output = " ".join(terms)
    index = 0
    while _word_count(output) < validator.minimum_words:
        output = f"{output} {filler[index % len(filler)]}".strip()
        index += 1
    return output


class _Tokenizer:
    def __init__(self, manifest: Any) -> None:
        self._prompt_ids = {
            spec.prompt: index + 1 for index, spec in enumerate(manifest.cases)
        }
        self._outputs = {
            index + 1: _passing_output(spec, candidate=True)
            for index, spec in enumerate(manifest.cases)
        }
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = None
        self.unk_token_id = None
        self.all_special_ids = [0]

    def __len__(self) -> int:
        return 65

    def get_vocab(self) -> dict[str, int]:
        return {f"token_{index}": index for index in range(len(self))}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"token_{token_id}"

    def __call__(
        self,
        prompt: str,
        *,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert add_special_tokens is True
        token_id = self._prompt_ids[prompt]
        return {
            "input_ids": torch.tensor([[token_id]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool | None = None,
    ) -> str:
        del clean_up_tokenization_spaces
        retained = [
            token_id
            for token_id in token_ids
            if not (skip_special_tokens and token_id == self.eos_token_id)
        ]
        if len(retained) == 1 and retained[0] in self._outputs:
            return self._outputs[retained[0]]
        return " ".join(f"token_{value}" for value in retained)


class _ReplayModel(nn.Module):
    def __init__(self, *, output_offset: int = 0, execute_forward: bool = True):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor([1.25]))
        self.output_offset = output_offset
        self.execute_forward = execute_forward

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(dtype=self.anchor.dtype) * self.anchor

    def generate(self, **kwargs: Any) -> torch.Tensor:
        input_ids = kwargs["input_ids"]
        if self.execute_forward:
            self(input_ids)
        token_id = int(input_ids[0, -1]) + self.output_offset
        continuation = torch.tensor(
            [[token_id, 0]],
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return torch.cat((input_ids, continuation), dim=1)


def _artifact(value: dict[str, Any]) -> CanonicalJsonArtifact:
    return CanonicalJsonArtifact(
        source=value,
        document_sha256=canonical_sha256(value),
    )


def _seal(body: dict[str, Any], field: str) -> dict[str, Any]:
    return {**body, field: canonical_sha256(body)}


def _parameter_inventory(unique_parameters: int) -> dict[str, Any]:
    body = {
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
    }
    return _seal(body, "inventory_sha256")


def _parameter_receipt(
    source_inventory: dict[str, Any],
    target_inventory: dict[str, Any],
    *,
    promotion_compression_ratio: float = 100.0,
) -> dict[str, Any]:
    source_parameters = int(source_inventory["unique_parameters"])
    target_parameters = int(target_inventory["unique_parameters"])
    realized_ratio = source_parameters / target_parameters
    body = {
        "schema": LFM2_CPCF_PARAMETER_RECEIPT_SCHEMA,
        "source_inventory": source_inventory,
        "target_inventory": target_inventory,
        "source_unique_parameters": source_parameters,
        "target_unique_parameters": target_parameters,
        "planned_target_unique_parameters": target_parameters,
        "parameters_removed_from_topology": (source_parameters - target_parameters),
        "realized_compression_ratio": realized_ratio,
        "promotion_compression_ratio": promotion_compression_ratio,
        "promotion_ratio_met": (realized_ratio >= promotion_compression_ratio),
        "valid_lower_ratio_diagnostic_fold": (
            realized_ratio < promotion_compression_ratio
        ),
        "parameter_count_compaction_only": True,
        "checkpoint_byte_ratio_used": False,
        "quantization_used": False,
        "pruning_used": False,
    }
    return _seal(body, "receipt_sha256")


def _v5_selection_report(candidate_state_sha256: str) -> dict[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-cpcf-short-conv-v4",
        "kind": "target_dtype_evaluation",
        "policy_sha256": canonical_sha256({"v5": "solver-policy"}),
        "split": "selection",
        "target_dtype": "bfloat16",
        "case_balanced": True,
        "case_count": 1,
        "cases": [
            {
                "case_id": "selection-trajectory-000",
                "rows": 8,
                "width": 2,
                "target_sha256": canonical_sha256({"v5": "target"}),
                "valid_rows_sha256": None,
                "projection_case_authentication_sha256": canonical_sha256(
                    {"v5": "projection-case"}
                ),
                "context_rows": 12,
                "context_width": 2,
                "context_target_sha256": canonical_sha256({"v5": "context-target"}),
                "valid_context_rows_sha256": None,
            }
        ],
        "evidence_sha256": canonical_sha256({"v5": "selection-evidence"}),
        "candidate_state_sha256": candidate_state_sha256,
        "maximum_case_relative_rmse": 0.5,
        "minimum_case_cosine_similarity": 0.5,
        "mean_case_relative_rmse": 0.5,
        "mean_case_cosine_similarity": 0.5,
        "maximum_case_context_relative_rmse": 0.5,
        "minimum_case_context_cosine_similarity": 0.5,
        "mean_case_context_relative_rmse": 0.5,
        "mean_case_context_cosine_similarity": 0.5,
        "context_evidence_present": True,
        "actual_parameter_and_intermediate_target_dtype_replay": True,
        "selection_used_for_parameter_fitting": False,
        "validation_used_for_parameter_fitting": False,
    }
    return _seal(body, "report_sha256")


def _v5_semantic_layer_receipt() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    dict[str, Any],
]:
    adapter = Lfm2CpcfShortConvV5NumericalFrontierAdapter(
        basis_policy=Lfm2CpcfV5FirBasisPolicy(
            maximum_projected_fir_offdiagonal_relative_frobenius=0.0,
            maximum_per_lag_commutator_relative_frobenius=1.0,
        )
    )
    contract_value = v5_compiler_module._real_v5_numerical_adapter_contract(adapter)
    assert contract_value is not None
    contract = dict(contract_value)
    zero_numerical_execution = {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    provenance = {
        "implementation": contract["implementation"],
        "real_v5_numerical_algorithm": True,
        "dynamic_source_target_topology": True,
        "basis_policy_sha256": contract["basis_policy_sha256"],
        "ridge_policy_sha256": contract["ridge_policy_sha256"],
        "basis_actual_commutator_gate": 1.0,
        "basis_actual_commutator_residual": 0.5,
        "basis_actual_commutator_gate_passed": True,
        "basis_full_source_numeric_participation": True,
        "source_channels_dropped": 0,
        "source_channels_pruned": 0,
        "source_channels_quantized": 0,
        "source_fir_resized_before_basis": True,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "selection_performed_by_adapter": False,
        "validation_performed_by_adapter": False,
        "model_forward_calls": 0,
        "training": False,
        "pruning": False,
        "quantization": False,
        "execution": zero_numerical_execution,
    }
    frontier_body = {
        "kind": "authenticated_fit_only_numerical_frontier",
        "adapter_id": adapter.adapter_id,
        "full_source_numeric_participation": True,
        "source_channels_dropped": 0,
        "fit_only": True,
        "selection_cases_consumed": 0,
        "validation_cases_consumed": 0,
        "execution": zero_numerical_execution,
        "adapter_provenance": provenance,
        "adapter_provenance_sha256": canonical_sha256(provenance),
    }
    frontier = _seal(frontier_body, "report_sha256")

    prior_id = "ridge-alpha-00-1e-08--anchor-00-prior"
    alternate_id = "ridge-beta-01-1e-07--candidate"

    def candidate_state(offset: float) -> dict[str, torch.Tensor]:
        return {
            "in_proj_weight": (
                torch.arange(12, dtype=torch.bfloat16).reshape(6, 2) / 16 + offset
            ),
            "in_proj_bias": torch.linspace(
                -0.25 + offset,
                0.25 + offset,
                6,
                dtype=torch.bfloat16,
            ),
            "fir_kernel": torch.tensor(
                [[0.75 + offset, 0.25], [0.50, 0.50 + offset]],
                dtype=torch.bfloat16,
            ),
            "fir_bias": torch.tensor(
                [offset, -offset],
                dtype=torch.bfloat16,
            ),
            "writer_weight": torch.tensor(
                [[1.0 + offset, 0.25], [-0.25, 1.0 + offset]],
                dtype=torch.bfloat16,
            ),
            "writer_bias": torch.tensor(
                [0.125 + offset, -0.125],
                dtype=torch.bfloat16,
            ),
        }

    candidate_states = {
        prior_id: candidate_state(0.0),
        alternate_id: candidate_state(0.03125),
    }
    candidate_state_sha256 = {
        candidate_id: candidate_replay_module._state_sha256(
            state,
            "target_dtype_candidate",
        )
        for candidate_id, state in candidate_states.items()
    }
    selection_reports = {
        candidate_id: _v5_selection_report(candidate_state_sha256[candidate_id])
        for candidate_id in sorted(candidate_states)
    }
    selector = dict(rescore_sealed_v4_short_conv_selection_frontier(selection_reports))

    installed_state = InstalledReplayStateFingerprint.create(
        state_sha256=canonical_sha256({"v5": "installed-state"}),
        tensor_count=12,
        logical_bytes=sum(
            int(tensor.numel()) * int(tensor.element_size())
            for state in candidate_states.values()
            for tensor in state.values()
        ),
    )
    devices = tuple(
        CudaReplayDeviceReceipt.create(
            device_index=index,
            device_uuid=f"GPU-v5-reviewed-{index}",
            device_name="NVIDIA GeForce RTX 3090",
            pci_domain_id=0,
            pci_bus_id=index + 1,
            pci_device_id=index + 10,
            capability=(8, 6),
            total_memory_bytes=25_769_803_776,
            multiprocessor_count=82,
            torch_version="2.13.0+cu128",
            cuda_runtime_version="12.8",
            cudnn_version=91_900,
            deterministic_algorithms_enabled=True,
            deterministic_debug_mode=2,
            cudnn_deterministic=True,
            cudnn_benchmark=False,
            cudnn_allow_tf32=False,
            cuda_matmul_allow_tf32=False,
            cuda_matmul_allow_bf16_reduced_precision_reduction=False,
            cuda_matmul_allow_fp16_reduced_precision_reduction=False,
            float32_matmul_precision="highest",
            cublas_workspace_config=":4096:8",
        )
        for index in (0, 1)
    )
    replicas = tuple(
        ReplayReplicaReceipt.create(
            device=device,
            installed_state=installed_state,
        )
        for device in devices
    )
    scheduler_policy = ReplaySchedulerPolicy(
        maximum_pinned_bytes_per_slot=256 << 20,
        parity_absolute_tolerance=0.125,
        parity_relative_tolerance=0.02,
        parity_maximum_relative_rmse=0.02,
        parity_minimum_cosine_similarity=0.999,
    )
    case_payload = {
        "case_id": "selection-trajectory-000",
        "split": "selection",
        "suffix_start": 1,
        "projection_case_authentication_sha256": canonical_sha256(
            {"v5": "projection-case"}
        ),
        "child_input_history": torch.tensor(
            [
                [
                    [0.25, -0.50],
                    [0.75, 0.125],
                    [-0.25, 0.50],
                    [0.375, -0.625],
                ]
            ],
            dtype=torch.bfloat16,
        ),
        "valid_prefix_mask": torch.tensor(
            [[True, True, True, True]],
            dtype=torch.bool,
        ),
    }
    work_items = []
    candidate_outputs = []
    solver_policy_sha256 = canonical_sha256({"v5": "solver-policy"})
    for ordinal, candidate_id in enumerate(sorted(candidate_states)):
        payload = {
            "schema": LFM2_CPCF_CANDIDATE_REPLAY_WORK_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_state_sha256": candidate_state_sha256[candidate_id],
            "policy_sha256": solver_policy_sha256,
            "target_dtype": "bfloat16",
            "required_split": "selection",
            "state": candidate_states[candidate_id],
            "cases": (case_payload,),
        }
        work_items.append(
            ReplayWorkItem.create(
                ordinal=ordinal,
                work_id=f"lfm2-cpcf-candidate::{candidate_id}",
                shard_key=f"lfm2-cpcf-candidate::{candidate_id}",
                payload=payload,
            )
        )
        candidate_outputs.append(replay_lfm2_cpcf_candidate_work_payload(payload))
    work_items = tuple(work_items)
    candidate_outputs = tuple(candidate_outputs)
    plan = build_replay_schedule_plan(
        work_items,
        replicas=replicas,
        policy=scheduler_policy,
    )
    output_sha256 = [fingerprint_replay_value(output) for output in candidate_outputs]
    scheduled_records = []
    device_record_counts = {0: 0, 1: 0}
    for assignment, candidate_output_sha256 in zip(
        plan.assignments,
        output_sha256,
        strict=True,
    ):
        slot_index = device_record_counts[assignment.device_index] % 2
        device_record_counts[assignment.device_index] += 1
        scheduled_body = {
            "ordinal": assignment.ordinal,
            "work_id": assignment.work_id,
            "shard_key": assignment.shard_key,
            "device_index": assignment.device_index,
            "device_receipt_sha256": assignment.device_receipt_sha256,
            "installed_state_sha256": assignment.installed_state_sha256,
            "input_payload_sha256": assignment.input_payload_sha256,
            "output_sha256": candidate_output_sha256,
            "slot_index": slot_index,
            "folding_math_changed": False,
        }
        scheduled_records.append(_seal(scheduled_body, "record_sha256"))
    representative_output = candidate_outputs[0]
    parity_reports = [
        dict(
            compare_cpu_cuda_replay(
                representative_output,
                representative_output,
                policy=scheduler_policy,
                device_index=device.device_index,
            )
        )
        for device in devices
    ]
    assert parity_reports[0]["tensor_leaf_count"] == 5
    assert (
        parity_reports[0]["cpu_output_sha256"] == parity_reports[1]["cpu_output_sha256"]
    )
    assert (
        parity_reports[0]["structure_sha256"] == parity_reports[1]["structure_sha256"]
    )
    for device in devices:
        assert parity_reports[device.device_index]["schema"] == (
            DYNAMIC_REPLAY_PARITY_SCHEMA
        )
    buffer_reports = []
    for device in devices:
        buffer_body = {
            "schema": DYNAMIC_REPLAY_BUFFER_SCHEMA,
            "device_index": device.device_index,
            "device_receipt_sha256": device.receipt_sha256,
            "slot_count": 2,
            "maximum_bytes_per_slot": 256 << 20,
            "maximum_total_pinned_bytes": 512 << 20,
            "maximum_observed_slot_bytes": 1024,
            "stage_count": device_record_counts[device.device_index] + 1,
            "pinned_host_memory": True,
            "asynchronous_h2d": True,
            "canonical_pinned_payload_hash_verified": True,
        }
        buffer_reports.append(_seal(buffer_body, "report_sha256"))
    execution_body = {
        "schema": DYNAMIC_REPLAY_EXECUTION_SCHEMA,
        "plan_sha256": plan.plan_sha256,
        "plan": plan.as_dict(),
        "parity_work_id": "lfm2-cpcf-candidate::representative-parity",
        "parity_payload_sha256": work_items[0].payload_sha256,
        "parity_reports": parity_reports,
        "parity_passed_before_scheduled_work": True,
        "scheduled_records": scheduled_records,
        "canonical_output_work_ids": [item.work_id for item in work_items],
        "canonical_output_sha256": output_sha256,
        "buffer_reports": buffer_reports,
        "backend_receipts_reauthenticated": True,
        "installed_state_reauthenticated": True,
        "cpu_reference_state_reauthenticated": True,
        "dynamic_work_stealing": False,
        "canonical_result_order_restored": True,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
        "folding_math_changed": False,
    }
    execution = _seal(execution_body, "report_sha256")
    backend_body = {
        "schema": LFM2_CPCF_CANDIDATE_REPLAY_BACKEND_SCHEMA,
        "kind": "authenticated_two_cuda_scheduler",
        "canonical_output_work_ids": [item.work_id for item in work_items],
        "canonical_input_payload_sha256": [item.payload_sha256 for item in work_items],
        "canonical_output_sha256": output_sha256,
        "compute_device_by_work_id": {
            assignment.work_id: f"cuda:{assignment.device_index}"
            for assignment in plan.assignments
        },
        "execution_report": execution,
        "representative_cpu_cuda_parity_passed": True,
        "canonical_candidate_order_restored": True,
        "closed_form_solve_math_changed": False,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    backend = _seal(backend_body, "report_sha256")

    contract_tensor_names = ["candidate_replay_contract"]
    state_contract_body = {
        "schema": ("neural-forge-dynamic-replay-live-state-probes-v1"),
        "logical_device_indices": [0, 1],
        "contract_tensor_names": contract_tensor_names,
        "installed_state": installed_state.as_dict(),
        "cpu_tensors_retained": True,
        "cuda_tensors_retained": True,
    }
    state_contract = _seal(state_contract_body, "contract_sha256")
    runtime_body = {
        "schema": ("neural-forge-dynamic-replay-live-runtime-authentication-v1"),
        "replicas": [replica.as_dict() for replica in replicas],
        "policy": scheduler_policy.as_dict(),
        "state_probe_contract": state_contract,
        "device_receipts_freshly_probed": True,
        "installed_states_freshly_rehashed": True,
        "bounded_pinned_host_factory_authenticated": True,
    }
    runtime = _seal(runtime_body, "report_sha256")
    parity_authority = {
        "path": str(Path("reports/approved-v5-parity.json").resolve()),
        "file_sha256": canonical_sha256({"v5": "parity-file"}),
        "receipt_sha256": canonical_sha256({"v5": "parity-receipt"}),
        "source_sha256": {
            "dynamic_replay_scheduler.py": canonical_sha256({"v5": "scheduler-source"}),
            "hf_lfm2_cpcf_candidate_replay.py": canonical_sha256(
                {"v5": "candidate-source"}
            ),
            "verify_lfm2_cpcf_candidate_replay_gpu_parity.py": (
                canonical_sha256({"v5": "verifier-source"})
            ),
        },
        "scheduler_policy": scheduler_policy.as_dict(),
        "target_dtype": "bfloat16",
        "real_two_gpu_parity_passed": True,
    }
    gpu_candidate_body = {
        "schema": ("neural-forge-lfm2-cpcf-v5-authenticated-two-gpu-replay-v1"),
        "logical_device_indices": [0, 1],
        "devices": [device.as_dict() for device in devices],
        "replicas": [replica.as_dict() for replica in replicas],
        "cpu_installed_state": installed_state.as_dict(),
        "scheduler_policy": scheduler_policy.as_dict(),
        "runtime_authentication": runtime,
        "runtime_authentication_sha256": runtime["report_sha256"],
        "proven_parity_receipt": parity_authority,
        "contract_tensor_names": contract_tensor_names,
        "contract_cpu_tensors_retained_for_runtime_reauthentication": True,
        "contract_gpu_tensors_retained_for_runtime_reauthentication": True,
        "homogeneous_authenticated_receipts": True,
        "candidate_payload_staging": (
            "bounded_double_buffered_pinned_system_memory_to_cuda"
        ),
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    gpu_candidate_replay = _seal(
        gpu_candidate_body,
        "report_sha256",
    )

    artifact_body = {
        "numerical_frontier_adapter_active": True,
        "fit_only_numerical_generation": True,
        "legacy_v4_numerical_generation_claimed": False,
        "numerical_frontier": frontier,
        "candidate_selection_reports": selection_reports,
        "preinstall_candidate_selector_receipt": selector,
        "preinstall_candidate_selector_receipt_sha256": (selector["report_sha256"]),
        "preinstall_candidate_selector_active": True,
        "selected_candidate_id": selector["selected_candidate_id"],
        "default_v4_selected_candidate_id": prior_id,
        "selection_changed_from_v4": False,
        "post_install_rescoring_performed": False,
        "legacy_v4_report_claimed": False,
        "candidate_replay_backend": backend,
        "candidate_replay_backend_report_sha256": backend["report_sha256"],
        "selection_candidate_replay_backend": ("authenticated_injected_backend"),
        "fit_and_validation_winner_replay_backend": "cpu_reference",
    }
    artifact = _seal(artifact_body, "report_sha256")
    layer_body = {
        "schema": "neural-forge-lfm2-cpcf-layer-compiler-v1",
        "layer_index": 0,
        "operator_kind": "short_conv",
        "source_width": 4,
        "target_width": 2,
        "input_bank_sha256": canonical_sha256({"v5": "input-bank"}),
        "next_bank_sha256": canonical_sha256({"v5": "next-bank"}),
        "boundary_charts_sha256": canonical_sha256({"v5": "boundary-charts"}),
        "preinstall_replay_sha256": canonical_sha256({"v5": "preinstall-replay"}),
        "operator_replay_sha256": canonical_sha256({"v5": "operator-replay"}),
        "final_replay_sha256": canonical_sha256({"v5": "final-replay"}),
        "operator_artifact": artifact,
        "mlp_artifact": {"schema": "mlp-v1"},
        "operator_install_report": {"installed": True},
        "mlp_install_report": {"installed": True},
        "split_metrics": {"fit": {}, "validation": {}},
        "split_phase_counts": {"fit": 1, "validation": 1},
        "solver_evidence_policy": {"fit_only": True},
        "actual_child_state_policy": {"actual": True},
        "audit": {"source_unchanged": True},
        "execution": {
            "closed_form_compilers": True,
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "target_layer_forward_calls": 0,
            "generation_calls": 0,
            "cache_generation_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
            "fallback_paths": 0,
        },
    }
    layer = _seal(layer_body, "result_sha256")
    typed = Lfm2CpcfCompletedLayerReceipt(layer)
    proof_value = v5_compiler_module._prove_real_v5_numerical_layer(
        typed,
        adapter_contract=contract,
    )
    assert proof_value is not None
    return (
        layer,
        contract,
        dict(proof_value),
        execution["report_sha256"],
        gpu_candidate_replay,
    )


def _rebuild_cpcf_compiler_chain(
    cpcf_body: dict[str, Any],
    *,
    policy: Lfm2CpcfModelFoldPolicy | None = None,
    phase_sha256: tuple[str, ...] | None = None,
    compiled_schema: str = LFM2_CPCF_MODEL_FOLD_SCHEMA,
) -> dict[str, Any]:
    """Rebuild every nested compiler seal after synthetic provenance edits."""

    resolved_policy = (
        Lfm2CpcfModelFoldPolicy(
            promotion_compression_ratio=100.0,
            require_promotion_ratio=True,
        )
        if policy is None
        else policy
    )
    source_parameters = int(
        cpcf_body["parent_parameter_inventory"]["unique_parameters"]
    )
    target_parameters = int(cpcf_body["blank_target"]["target_unique_parameters"])
    parent_inventory = _parameter_inventory(source_parameters)
    blank_inventory = _parameter_inventory(target_parameters)
    compiled_inventory = _parameter_inventory(target_parameters)
    reloaded_inventory = _parameter_inventory(target_parameters)
    parameter_receipt = _parameter_receipt(
        parent_inventory,
        blank_inventory,
        promotion_compression_ratio=(resolved_policy.promotion_compression_ratio),
    )

    blank_body = copy.deepcopy(cpcf_body["blank_target"])
    blank_body.pop("report_sha256", None)
    blank_body.update(
        {
            "schema": "neural-forge-lfm2-cpcf-blank-target-v1",
            "parent_unique_parameters": source_parameters,
            "target_unique_parameters": target_parameters,
            "realized_compression_ratio": (source_parameters / target_parameters),
            "compiler_created_blank": True,
            "all_parameters_zero": True,
            "dense_materialized_parameters": True,
            "topology_provenance_verified": True,
            "parent_state_preserved": True,
            "quantization_used": False,
            "pruning_used": False,
            "training_used": False,
            "weights_installed": False,
        }
    )
    blank_report = _seal(blank_body, "report_sha256")

    prior_compiled = cpcf_body.get("compiled_model_fold")
    if not isinstance(prior_compiled, dict):
        prior_compiled = {}
    ordered_phases = (
        tuple(prior_compiled.get("phase_sha256", ()))
        if phase_sha256 is None
        else tuple(phase_sha256)
    )
    if not ordered_phases:
        ordered_phases = (
            canonical_sha256({"compiler_phase": 0}),
            canonical_sha256({"compiler_phase": 1}),
        )
    digest_defaults = {
        "boundary_bundle_sha256": canonical_sha256({"compiled": "boundary-bundle"}),
        "interface_install_report_sha256": canonical_sha256(
            {"compiled": "interface-install"}
        ),
        "neutral_norm_install_report_sha256": canonical_sha256(
            {"compiled": "neutral-norm-install"}
        ),
        "final_state_bank_sha256": canonical_sha256({"compiled": "final-state-bank"}),
        "final_norm_report_sha256": canonical_sha256({"compiled": "final-norm"}),
    }
    compiled_policy: dict[str, Any]
    v5_metadata: dict[str, Any] = {}
    v5_layer_results: list[dict[str, Any]] | None = None
    v5_gpu_execution_audit: dict[str, Any] | None = None
    v5_gpu_candidate_replay: dict[str, Any] | None = None
    ordered_layer_result_sha256 = prior_compiled.get(
        "ordered_layer_result_sha256",
        [canonical_sha256({"compiled": "layer-result-0"})],
    )
    if compiled_schema == LFM2_CPCF_V5_MODEL_FOLD_SCHEMA:
        runtime_policy = Lfm2CpcfV5RuntimePolicy(
            model_fold_policy=resolved_policy,
            maximum_resident_evidence_bytes=1024,
            full_phase_tree_logical_bytes=4096,
            minimum_physical_rss_reduction_fraction=0.25,
            rss_sample_interval_seconds=0.1,
            require_short_conv_v5_endpoint_first_selector=True,
            require_short_conv_v5_real_numerical_frontier_adapter=True,
        )
        compiled_policy = runtime_policy.as_dict()
        (
            v5_layer,
            numerical_contract,
            numerical_proof,
            gpu_execution_sha256,
            v5_gpu_candidate_replay,
        ) = _v5_semantic_layer_receipt()
        v5_layer_results = [v5_layer]
        ordered_layer_result_sha256 = [v5_layer["result_sha256"]]
        v5_gpu_execution_audit = {
            "candidate_gpu_replay_invoked": True,
            "execution_count": 1,
            "execution_report_sha256": [gpu_execution_sha256],
            "both_logical_devices_reauthenticated_per_execution": True,
            "no_gpu_work_claimed_when_no_candidate_batch_was_scheduled": True,
        }
        catalog_sha256 = canonical_sha256({"selective": "catalog"})
        source_sha256 = canonical_sha256({"selective": "source"})
        provider_sha256 = canonical_sha256({"selective": "provider"})
        residency = _seal(
            {
                "schema": EVIDENCE_RESIDENCY_WINDOW_SCHEMA,
                "catalog_sha256": catalog_sha256,
                "source_sha256": source_sha256,
                "provider_sha256": provider_sha256,
                "state": "completed",
                "max_resident_layers": 2,
                "maximum_resident_bytes": 1024,
                "resident_bytes": 0,
                "prefetch_distance_layers": 1,
                "peak_occupied_layers": 2,
                "peak_resident_bytes": 512,
            },
            "snapshot_sha256",
        )
        physical = _seal(
            {
                "schema": LFM2_CPCF_V5_PHYSICAL_RSS_SCHEMA,
                "start_rss_bytes": 1000,
                "current_rss_bytes": 1100,
                "peak_rss_bytes": 1100,
                "start_lifetime_peak_bytes": 1000,
                "current_increase_bytes": 100,
                "peak_increase_bytes": 100,
                "full_phase_tree_logical_bytes": 4096,
                "minimum_reduction_fraction": 0.25,
                "maximum_increase_bytes": 3072,
                "sample_interval_seconds": 0.1,
                "checkpoint_count": 0,
                "checkpoints": [],
                "sampler_stopped": True,
                "physical_rss_gate_passed": True,
                "comparison_basis": (
                    "peak_process_rss_increase_from_precompiler_baseline_"
                    "versus_full_phase_tree_logical_bytes"
                ),
            },
            "report_sha256",
        )
        v5_metadata = {
            "artifact_schema": LFM2_CPCF_V5_ARTIFACT_SCHEMA,
            "resume_schema": LFM2_CPCF_V5_RESUME_SCHEMA,
            "resume_supported": True,
            "resume_fail_closed": True,
            "resume_used": False,
            "resume_start_layer_index": 0,
            "resume_completed_prefix_layer_indices": [],
            "resume_executed_layer_indices": [0],
            "resume_completed_layers_replayed": False,
            "resume_actual_upstream_child_state_restored": False,
            "resume_prefix_install_report_sha256": None,
            "resume_frontier_sha256": None,
            "resume_admission_sha256": None,
            "resume_publication_enabled": False,
            "resume_admitted_prefixes": [],
            "resume_parent_or_cache_tensor_sidecars": 0,
            "resume_checkpoint_receipt_chain_authenticated": True,
            "compiler_implementation": dict(lfm2_cpcf_v5_compiler_implementation()),
            "selective_catalog_sha256": catalog_sha256,
            "selective_source_sha256": source_sha256,
            "selective_provider_sha256": provider_sha256,
            "authenticated_full_phase_tree_logical_bytes": 4096,
            "physical_rss_baseline_bound_to_authenticated_catalog": True,
            "residency_receipt": residency,
            "physical_rss_receipt": physical,
            "full_all_layer_phase_tree_materialized": False,
            "tensor_free_phase_shells_used": True,
            "active_layer_evidence_released_before_successor_compile": True,
            "final_norm_evidence_separately_bounded_and_reauthenticated": True,
            "candidate_replay_backend_injected": True,
            "preinstall_candidate_selector_provided": True,
            "preinstall_candidate_selector_active": True,
            "preinstall_candidate_selector_layer_indices": [0],
            "short_conv_v5_endpoint_first_selector_integrated": True,
            "short_conv_numerical_frontier_adapter_provided": True,
            "short_conv_numerical_frontier_adapter_active": True,
            "short_conv_numerical_frontier_adapter_layer_indices": [0],
            "short_conv_layer_indices": [0],
            "short_conv_v5_real_numerical_adapter_required": True,
            "short_conv_v5_real_numerical_adapter_concrete_instance_provided": (True),
            "short_conv_v5_real_numerical_adapter_contract": (numerical_contract),
            "short_conv_v5_real_numerical_provenance_layer_indices": [0],
            "short_conv_v5_real_numerical_layer_proofs": [numerical_proof],
            "short_conv_v5_numerical_candidate_generation_integrated": True,
            "short_conv_v5_numerical_policy_integrated": True,
            "short_conv_v5_post_install_rescoring_performed": False,
            "legacy_v4_default_or_fingerprint_changed": False,
        }
    else:
        compiled_policy = resolved_policy.as_dict()
    compiled_body = {
        "schema": compiled_schema,
        "blank_target_report_sha256": blank_report["report_sha256"],
        "policy": compiled_policy,
        "phase_count": len(ordered_phases),
        "phase_sha256": list(ordered_phases),
        **v5_metadata,
        **{
            name: prior_compiled.get(name, value)
            for name, value in digest_defaults.items()
        },
        "ordered_layer_result_sha256": ordered_layer_result_sha256,
        "parameter_receipt": parameter_receipt,
        "target_parameter_inventory_before": blank_inventory,
        "target_parameter_inventory_after": compiled_inventory,
        "source_state_preserved": True,
        "target_parameter_count_preserved": True,
        "all_layers_compiled_in_order": True,
        "actual_folded_upstream_replay_used": True,
        "operation_audit": {
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
        },
    }
    compiled = _seal(compiled_body, "result_sha256")
    reload_receipt = _parameter_receipt(
        parent_inventory,
        reloaded_inventory,
        promotion_compression_ratio=(resolved_policy.promotion_compression_ratio),
    )
    cpcf_body.update(
        {
            "blank_target": blank_report,
            "parent_parameter_inventory": parent_inventory,
            "blank_parameter_inventory": blank_inventory,
            "compiled_parameter_inventory": compiled_inventory,
            "reloaded_parameter_inventory": reloaded_inventory,
            "compression": reload_receipt,
            "compiled_model_fold": compiled,
        }
    )
    if (
        v5_layer_results is not None
        and v5_gpu_execution_audit is not None
        and v5_gpu_candidate_replay is not None
    ):
        cpcf_body["layer_results"] = v5_layer_results
        cpcf_body["gpu_execution_audit"] = v5_gpu_execution_audit
        cpcf_body["gpu_candidate_replay"] = v5_gpu_candidate_replay
    ordered_phase_binding = canonical_sha256(
        {
            "schema": ("neural-forge-lfm2-cpcf-ordered-phase-binding-v1"),
            "phase_count": len(ordered_phases),
            "ordered_phase_sha256": list(ordered_phases),
        }
    )
    return {
        "compiled_model_fold_schema": compiled_schema,
        "compiled_model_fold_result_sha256": compiled["result_sha256"],
        "compiled_model_fold_policy": compiled_policy,
        "compiled_model_fold_policy_sha256": canonical_sha256(compiled_policy),
        "compiled_model_fold_phase_count": len(ordered_phases),
        "compiled_model_fold_phase_sha256": list(ordered_phases),
        "compiled_model_fold_phase_binding_sha256": ordered_phase_binding,
        "compiled_blank_target_report_sha256": (blank_report["report_sha256"]),
        "compiled_parameter_receipt_sha256": (parameter_receipt["receipt_sha256"]),
    }


def _candidate_file_sha(document: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(document) + b"\n").hexdigest()


def _write_tokenizer_package(path: Path) -> str:
    path.mkdir()
    (path / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [{"id": 0, "content": "<eos>", "special": True}],
                "model": {
                    "type": "BPE",
                    "vocab": {f"token_{index}": index for index in range(65)},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {"eos_token": "<eos>", "pad_token": "<eos>"},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return fingerprint_local_tokenizer_package(path)[0]


def _fixture(
    tmp_path: Path,
    *,
    bridge_artifact_sha256: str | None = None,
    bridge_profile_sha256: str | None = None,
    tokenizer: Any | None = None,
    tokenizer_directory: Path | None = None,
    compiled_schema: str = LFM2_CPCF_MODEL_FOLD_SCHEMA,
) -> dict[str, Any]:
    manifest_document = build_lfm2_intelligence_holdout_manifest()
    manifest = load_lfm2_intelligence_holdout_manifest(manifest_document)
    tokenizer = _Tokenizer(manifest) if tokenizer is None else tokenizer
    if tokenizer_directory is None:
        tokenizer_directory = tmp_path / "tokenizer"
        tokenizer_package = _write_tokenizer_package(tokenizer_directory)
    else:
        tokenizer_package = fingerprint_local_tokenizer_package(tokenizer_directory)[0]
    tokenizer_semantics = fingerprint_hf_tokenizer_semantics(tokenizer)
    eos = [0]

    parent_rows = [
        {
            "case_id": spec.case_id,
            "input_token_count": index + 1,
            "output": _passing_output(spec, candidate=False),
            "output_token_ids": [index + 1, 0],
        }
        for index, spec in enumerate(manifest.cases)
    ]
    parent_state = canonical_sha256({"parent": "state"})
    parent_model = canonical_sha256({"parent": "model"})
    parent = seal_lfm2_intelligence_holdout_parent_envelope(
        manifest,
        parent_rows,
        parent_state_fingerprint=parent_state,
        parent_model_fingerprint=parent_model,
        tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )
    loaded_parent = load_lfm2_intelligence_holdout_parent_envelope(
        parent,
        expected_envelope_sha256=parent["envelope_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_parent_state_fingerprint=parent_state,
        expected_parent_model_fingerprint=parent_model,
        expected_tokenizer_fingerprint=tokenizer_package,
    )
    baseline = baseline_from_lfm2_intelligence_holdout_envelope(loaded_parent)

    model = _ReplayModel()
    candidate_state = fingerprint_cpcf_candidate_state(model)
    candidate_rows = [
        {
            "case_id": spec.case_id,
            "output": _passing_output(spec, candidate=True),
            "output_token_ids": [index + 1, 0],
            "generation_limit": spec.generation_limit,
            "topical_relevance": True,
        }
        for index, spec in enumerate(manifest.cases)
    ]
    candidate = seal_lfm2_intelligence_holdout_candidate_outputs(
        manifest,
        candidate_rows,
        candidate_state_fingerprint=candidate_state,
        tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )
    loaded_candidate = load_lfm2_intelligence_holdout_candidate_outputs(
        candidate,
        expected_candidate_outputs_sha256=candidate["candidate_outputs_sha256"],
        expected_manifest_sha256=manifest.manifest_sha256,
        expected_candidate_state_fingerprint=candidate_state,
        expected_tokenizer_fingerprint=tokenizer_package,
        eos_token_ids=eos,
    )

    bridge_artifact = (
        canonical_sha256({"bridge": "artifact"})
        if bridge_artifact_sha256 is None
        else bridge_artifact_sha256
    )
    bridge_profile = (
        canonical_sha256({"bridge": "profile"})
        if bridge_profile_sha256 is None
        else bridge_profile_sha256
    )
    cache_key = canonical_sha256({"cache": "parent-traces"})
    trace_adapter = canonical_sha256({"trace": "adapter"})
    trajectory_corpus = canonical_sha256({"trajectory": "corpus"})
    parent_graph = canonical_sha256({"parent": "graph"})
    live_parent_state = canonical_sha256({"parent": "live-state"})
    requested_budget = {
        "schema": "neural-forge-dense-parameter-budget-v1",
        "kind": "budget_request",
        "target_family": "lfm2",
        "target_modalities": ["text"],
        "target_unique_parameters": 1,
        "compression_ratio": None,
        "target_vocabulary_size": 65,
        "match_policy": "at_or_below",
        "maximum_candidates": 8,
    }
    frontier = canonical_sha256({"target": "frontier"})
    target_certificate = canonical_sha256({"target": "certificate"})
    selected_candidate = canonical_sha256({"target": "candidate"})
    target_topology = canonical_sha256({"target": "topology"})
    blank_state = canonical_sha256({"target": "zero-blank"})
    parent_inventory = {"unique_parameters": 100}
    inventory = {"unique_parameters": 1}
    cpcf_body = {
        "schema": CPCF_RUN_SCHEMA,
        "status": "folded_child_requires_manual_intelligence_review",
        "inputs": {
            "bridge_artifact_sha256": bridge_artifact,
            "bridge_profile_sha256": bridge_profile,
            "cache_key": cache_key,
            "parent_fingerprint": parent_model,
            "trace_adapter_sha256": trace_adapter,
            "trajectory_corpus_sha256": trajectory_corpus,
        },
        "blank_target": {
            "parent_graph_fingerprint": parent_graph,
            "parent_state_fingerprint": live_parent_state,
            "target_unique_parameters": 1,
            "requested_budget": requested_budget,
            "frontier_fingerprint": frontier,
            "target_certificate_sha256": target_certificate,
            "selected_candidate_fingerprint": selected_candidate,
            "selected_candidate_rank": 0,
            "target_topology_fingerprint": target_topology,
            "blank_state_fingerprint": blank_state,
        },
        "parent_parameter_inventory": parent_inventory,
        "blank_parameter_inventory": inventory,
        "compiled_parameter_inventory": inventory,
        "reloaded_parameter_inventory": inventory,
        "compression": {
            "source_unique_parameters": 100,
            "target_unique_parameters": 1,
            "planned_target_unique_parameters": 1,
            "parameter_count_compaction_only": True,
            "checkpoint_byte_ratio_used": False,
            "promotion_ratio_met": True,
            "quantization_used": False,
            "pruning_used": False,
        },
        "fresh_reload_verified": True,
        "structural_parameter_compaction_proven": True,
        "parent_state_preserved": True,
        "project_promotion_ratio_met": True,
        "diagnostic_control": False,
        "behavioral_intelligence_retention_proven": False,
        "manual_intelligence_review_required": True,
        "operation_audit": {
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
        },
        "claims": {
            "folding_only": True,
            "blank_target_filled": True,
            "training_used": False,
            "quantization_used": False,
            "pruning_used": False,
            "parent_backed_fallback_used": False,
            "exact_token_match_used_as_principal_objective": False,
            "marginal_compaction_reported_as_success": False,
        },
    }
    compiled_provenance = _rebuild_cpcf_compiler_chain(
        cpcf_body,
        compiled_schema=compiled_schema,
    )
    parent_inventory = cpcf_body["parent_parameter_inventory"]
    inventory = cpcf_body["reloaded_parameter_inventory"]
    cpcf = _seal(cpcf_body, "receipt_sha256")

    capture_body = {
        "schema": CPCF_CAPTURE_SCHEMA,
        "status": "captured_frozen_child_holdout",
        "manifest": {
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_document_sha256": manifest.document_sha256,
            "case_count": 64,
        },
        "cpcf_provenance": {
            "receipt_sha256": cpcf["receipt_sha256"],
            "external_receipt_file_sha256": _candidate_file_sha(cpcf),
            "internal_receipt_file_sha256": _candidate_file_sha(cpcf),
            "external_internal_receipts_identical": True,
            "saved_checkpoint_files_match_receipt": True,
            "reloaded_parameter_inventory": inventory,
            "reloaded_parameter_inventory_matches_receipt": True,
            "parameter_inventory_unchanged_after_generation": True,
        },
        "strict_tokenizer_bridge": {
            "artifact_sha256": bridge_artifact,
            "profile_sha256": bridge_profile,
            "files_verified": True,
        },
        "target_tokenizer": {
            "semantics_sha256": tokenizer_semantics,
            "source_package_fingerprint": tokenizer_package,
            "checkpoint_package_fingerprint": tokenizer_package,
            "source_and_checkpoint_packages_identical": True,
            "source_and_checkpoint_semantics_identical": True,
            "eos_token_ids": eos,
        },
        "candidate": {
            "output_file_sha256": _candidate_file_sha(candidate),
            "candidate_outputs_sha256": candidate["candidate_outputs_sha256"],
            "candidate_state_fingerprint": candidate_state,
            "state_before_sha256": candidate_state,
            "state_after_sha256": candidate_state,
            "state_unchanged": True,
            "case_count": 64,
            "all_output_token_ids_are_child_domain_only": True,
            "decoded_text_sealed": True,
        },
        "generation_protocol": {
            "mode": "greedy_whole_response",
            "do_sample": False,
            "use_cache": True,
            "one_generate_invocation_per_case": True,
            "per_case_manifest_generation_limits_used": True,
            "global_64_token_cap_used": False,
            "eos_token_ids": eos,
            "exact_parent_wording_used": False,
            "exact_parent_token_matching_used": False,
        },
        "operation_audit": {
            "candidate_generate_invocations": 64,
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
        },
        "claims": {
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
        },
    }
    capture = _seal(capture_body, "capture_receipt_sha256")

    policy = IntelligenceGatePolicy()
    automatic = evaluate_cached_intelligence(
        baseline,
        loaded_candidate.candidates,
        policy=policy,
        manual_ratings=None,
    ).as_dict()
    manual = build_manual_review_packet(
        baseline,
        loaded_candidate.candidates,
        policy,
    )
    parent_packet = build_parent_relative_manual_packet(manual)
    ratings = bind_parent_relative_manual_ratings(
        parent_packet,
        [
            {
                "case_id": value["case_id"],
                "parent": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "candidate": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "reason": "Both responses are correct, coherent, and useful.",
            }
            for value in parent_packet["cases"]
        ],
        reviewer_id="reviewer-1",
    )
    parent_report = evaluate_parent_relative_manual_retention(
        parent_packet,
        ratings,
    )
    documents = {
        "manifest": manifest_document,
        "parent": parent,
        "cpcf": cpcf,
        "candidate": candidate,
        "capture": capture,
        "automatic": automatic,
        "manual": manual,
        "parent_packet": parent_packet,
        "ratings": ratings,
        "parent_report": parent_report,
    }
    evidence_manifest = {
        "schema": ("neural-forge-lfm2-cpcf-reviewed-promotion-evidence-v1"),
        "case_count": 64,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest_document_sha256": canonical_sha256(manifest_document),
        "parent_envelope_sha256": parent["envelope_sha256"],
        "parent_envelope_document_sha256": canonical_sha256(parent),
        "cpcf_run_receipt_sha256": cpcf["receipt_sha256"],
        "cpcf_run_receipt_document_sha256": canonical_sha256(cpcf),
        "parent_cache_key": cache_key,
        "cache_live_parent_fingerprint": parent_model,
        "live_parent_state_fingerprint": live_parent_state,
        "trace_adapter_sha256": trace_adapter,
        "trajectory_corpus_sha256": trajectory_corpus,
        "blank_target_parent_graph_fingerprint": parent_graph,
        "exact_target_unique_parameters": 1,
        "target_requested_budget": requested_budget,
        "target_frontier_fingerprint": frontier,
        "target_certificate_sha256": target_certificate,
        "selected_target_candidate_fingerprint": selected_candidate,
        "selected_target_candidate_rank": 0,
        "target_topology_fingerprint": target_topology,
        "blank_target_state_fingerprint": blank_state,
        "target_plan_binding_sha256": canonical_sha256(
            {
                "schema": "neural-forge-lfm2-cpcf-target-plan-binding-v1",
                "parent_graph_fingerprint": parent_graph,
                "exact_target_unique_parameters": 1,
                "requested_budget": requested_budget,
                "frontier_fingerprint": frontier,
                "target_certificate_sha256": target_certificate,
                "selected_candidate_fingerprint": selected_candidate,
                "selected_candidate_rank": 0,
                "target_topology_fingerprint": target_topology,
                "blank_state_fingerprint": blank_state,
            }
        ),
        **compiled_provenance,
        "bridge_artifact_sha256": bridge_artifact,
        "bridge_profile_sha256": bridge_profile,
        "candidate_state_fingerprint": candidate_state,
        "candidate_outputs_sha256": candidate["candidate_outputs_sha256"],
        "candidate_outputs_document_sha256": canonical_sha256(candidate),
        "capture_receipt_sha256": capture["capture_receipt_sha256"],
        "capture_receipt_document_sha256": canonical_sha256(capture),
        "automatic_report_sha256": automatic["report_sha256"],
        "automatic_report_document_sha256": canonical_sha256(automatic),
        "automatic_core_report_sha256": automatic["report_sha256"],
        "automatic_gate_ready_for_manual_completion": True,
        "manual_packet_sha256": manual["packet_sha256"],
        "manual_packet_document_sha256": canonical_sha256(manual),
        "parent_relative_manual_packet_sha256": parent_packet["packet_sha256"],
        "parent_relative_manual_packet_document_sha256": canonical_sha256(
            parent_packet
        ),
        "parent_relative_ratings_sha256": ratings["ratings_sha256"],
        "parent_relative_ratings_document_sha256": canonical_sha256(ratings),
        "parent_relative_manual_report_sha256": parent_report["report_sha256"],
        "parent_relative_manual_report_document_sha256": canonical_sha256(
            parent_report
        ),
        "parent_relative_manual_report_passed": True,
        "parent_relative_manual_retention": parent_report["retention"],
        "candidate_tokenizer_fingerprint": tokenizer_package,
        "target_tokenizer_semantics_sha256": tokenizer_semantics,
        "eos_token_ids": eos,
        "parent_model_loaded": False,
        "training_used": False,
        "pruning_used": False,
        "quantization_used": False,
        "network_access": False,
    }
    composite = build_cpcf_reviewed_promotion_report(
        reviewed_evidence=evidence_manifest,
        inference_execution_contract=(
            BehaviorInferenceExecutionContract.top_level_forward()
        ),
    )
    documents["composite"] = composite
    artifacts = Lfm2CpcfReviewedPromotionArtifacts(
        manifest=_artifact(manifest_document),
        parent_envelope=_artifact(parent),
        cpcf_run_receipt=_artifact(cpcf),
        candidate_outputs=_artifact(candidate),
        capture_receipt=_artifact(capture),
        automatic_report=_artifact(automatic),
        manual_packet=_artifact(manual),
        parent_relative_manual_packet=_artifact(parent_packet),
        parent_relative_ratings=_artifact(ratings),
        parent_relative_report=_artifact(parent_report),
        composite_promotion_report=_artifact(composite),
    )
    return {
        "artifacts": artifacts,
        "documents": documents,
        "model": model,
        "tokenizer": tokenizer,
        "tokenizer_directory": tokenizer_directory,
    }


def _replace_artifact(
    artifacts: Lfm2CpcfReviewedPromotionArtifacts,
    name: str,
    replacement: CanonicalJsonArtifact,
) -> Lfm2CpcfReviewedPromotionArtifacts:
    values = {
        field: getattr(artifacts, field) for field in artifacts.__dataclass_fields__
    }
    values[name] = replacement
    return Lfm2CpcfReviewedPromotionArtifacts(**values)


def test_reviewed_gate_accepts_live_and_fresh_reload_and_binds_all_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    live = gate(fixture["model"])
    fresh_reload = gate(_ReplayModel())
    binding = resolve_behavior_gate_binding(gate)

    assert live.accepted is True
    assert fresh_reload.accepted is True
    assert binding.baseline_fingerprint == (evidence.full_reviewed_evidence_fingerprint)
    assert binding.manifest["inference_execution"] == (
        evidence.inference_execution_contract.as_dict()
    )
    assert (
        binding.manifest["components"][0]["full_reviewed_evidence_fingerprint"]
        == evidence.full_reviewed_evidence_fingerprint
    )
    assert (
        evidence.bridge_artifact_sha256
        == fixture["documents"]["cpcf"]["inputs"]["bridge_artifact_sha256"]
    )
    assert (
        evidence.bridge_profile_sha256
        == fixture["documents"]["cpcf"]["inputs"]["bridge_profile_sha256"]
    )
    receipt = fixture["documents"]["cpcf"]
    blank = receipt["blank_target"]
    component = binding.manifest["components"][0]
    assert evidence.cpcf_run_receipt_sha256 == receipt["receipt_sha256"]
    assert evidence.cpcf_run_receipt_document_sha256 == canonical_sha256(receipt)
    assert evidence.parent_cache_key == receipt["inputs"]["cache_key"]
    assert (
        evidence.cache_live_parent_fingerprint
        == receipt["inputs"]["parent_fingerprint"]
    )
    assert evidence.live_parent_state_fingerprint == blank["parent_state_fingerprint"]
    assert evidence.trace_adapter_sha256 == (receipt["inputs"]["trace_adapter_sha256"])
    assert (
        evidence.trajectory_corpus_sha256
        == (receipt["inputs"]["trajectory_corpus_sha256"])
    )
    assert (
        evidence.blank_target_parent_graph_fingerprint
        == blank["parent_graph_fingerprint"]
    )
    assert evidence.exact_target_unique_parameters == 1
    assert dict(evidence.target_requested_budget) == blank["requested_budget"]
    assert evidence.target_frontier_fingerprint == blank["frontier_fingerprint"]
    assert evidence.target_certificate_sha256 == blank["target_certificate_sha256"]
    assert (
        evidence.selected_target_candidate_fingerprint
        == blank["selected_candidate_fingerprint"]
    )
    assert evidence.selected_target_candidate_rank == 0
    assert evidence.target_topology_fingerprint == blank["target_topology_fingerprint"]
    assert evidence.blank_target_state_fingerprint == blank["blank_state_fingerprint"]
    assert (
        component["target_plan_binding_sha256"] == evidence.target_plan_binding_sha256
    )
    assert component["cpcf_run_receipt_sha256"] == receipt["receipt_sha256"]
    assert component["parent_cache_key"] == receipt["inputs"]["cache_key"]
    assert component["exact_target_unique_parameters"] == 1
    compiled = receipt["compiled_model_fold"]
    assert evidence.compiled_model_fold_schema == LFM2_CPCF_MODEL_FOLD_SCHEMA
    assert component["compiled_model_fold_schema"] == LFM2_CPCF_MODEL_FOLD_SCHEMA
    assert evidence.compiled_model_fold_result_sha256 == compiled["result_sha256"]
    assert dict(evidence.compiled_model_fold_policy) == compiled["policy"]
    assert evidence.compiled_model_fold_policy_sha256 == canonical_sha256(
        compiled["policy"]
    )
    assert evidence.compiled_model_fold_phase_count == compiled["phase_count"]
    assert list(evidence.compiled_model_fold_phase_sha256) == compiled["phase_sha256"]
    assert evidence.compiled_blank_target_report_sha256 == blank["report_sha256"]
    assert (
        evidence.compiled_parameter_receipt_sha256
        == compiled["parameter_receipt"]["receipt_sha256"]
    )
    assert component["compiled_model_fold_result_sha256"] == compiled["result_sha256"]
    assert component["compiled_model_fold_policy"] == compiled["policy"]
    assert component["compiled_model_fold_phase_sha256"] == compiled["phase_sha256"]


def _evidence_constructor_fields(
    evidence: Lfm2CpcfReviewedPromotionEvidence,
) -> dict[str, Any]:
    return {
        name: getattr(evidence, name)
        for name in (
            "manifest",
            "cpcf_run_receipt_sha256",
            "cpcf_run_receipt_document_sha256",
            "parent_cache_key",
            "cache_live_parent_fingerprint",
            "live_parent_state_fingerprint",
            "trace_adapter_sha256",
            "trajectory_corpus_sha256",
            "blank_target_parent_graph_fingerprint",
            "exact_target_unique_parameters",
            "target_requested_budget",
            "target_frontier_fingerprint",
            "target_certificate_sha256",
            "selected_target_candidate_fingerprint",
            "selected_target_candidate_rank",
            "target_topology_fingerprint",
            "blank_target_state_fingerprint",
            "target_plan_binding_sha256",
            "compiled_model_fold_schema",
            "compiled_model_fold_result_sha256",
            "compiled_model_fold_policy",
            "compiled_model_fold_policy_sha256",
            "compiled_model_fold_phase_count",
            "compiled_model_fold_phase_sha256",
            "compiled_model_fold_phase_binding_sha256",
            "compiled_blank_target_report_sha256",
            "compiled_parameter_receipt_sha256",
            "candidate_rows",
            "candidate_state_fingerprint",
            "candidate_outputs_sha256",
            "candidate_tokenizer_fingerprint",
            "target_tokenizer_semantics_sha256",
            "eos_token_ids",
            "reviewed_evidence",
            "reviewed_evidence_fingerprint",
            "composite_report_sha256",
            "composite_report_document_sha256",
            "full_reviewed_evidence_fingerprint",
            "inference_execution_contract",
        )
    }


def test_reviewed_evidence_is_loader_only(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])

    with pytest.raises(TypeError, match="loader-only"):
        Lfm2CpcfReviewedPromotionEvidence(**_evidence_constructor_fields(evidence))
    forged = object.__new__(Lfm2CpcfReviewedPromotionEvidence)
    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="lacks loader authentication",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(forged)
    assert authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence) is evidence


def test_reviewed_loader_accepts_and_version_binds_real_v5_schema(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        compiled_schema=LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    )

    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    component = resolve_behavior_gate_binding(gate).manifest["components"][0]

    assert evidence.compiled_model_fold_schema == (LFM2_CPCF_V5_MODEL_FOLD_SCHEMA)
    assert (
        dict(evidence.compiled_model_fold_policy)
        == fixture["documents"]["cpcf"]["compiled_model_fold"]["policy"]
    )
    assert evidence.reviewed_evidence["compiled_model_fold_schema"] == (
        LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
    )
    assert component["compiled_model_fold_schema"] == (LFM2_CPCF_V5_MODEL_FOLD_SCHEMA)
    assert gate(fixture["model"]).accepted is True


def _v5_unsealed_run_body(tmp_path: Path) -> dict[str, Any]:
    fixture = _fixture(
        tmp_path,
        compiled_schema=LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    )
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    return body


def _reseal_v5_compiled_report(run_body: dict[str, Any]) -> None:
    compiled = run_body["compiled_model_fold"]
    compiled.pop("result_sha256", None)
    run_body["compiled_model_fold"] = _seal(
        compiled,
        "result_sha256",
    )


def _reseal_v5_gpu_layer_and_compiled(
    run_body: dict[str, Any],
) -> None:
    layer = run_body["layer_results"][0]
    layer.pop("result_sha256", None)
    artifact = layer["operator_artifact"]
    artifact.pop("report_sha256", None)
    backend = artifact["candidate_replay_backend"]
    backend.pop("report_sha256", None)
    execution = backend["execution_report"]
    execution.pop("report_sha256", None)

    plan = execution["plan"]
    plan.pop("plan_sha256", None)
    execution["plan"] = _seal(plan, "plan_sha256")
    execution["plan_sha256"] = execution["plan"]["plan_sha256"]
    execution["parity_reports"] = [
        _seal(
            {name: value for name, value in report.items() if name != "report_sha256"},
            "report_sha256",
        )
        for report in execution["parity_reports"]
    ]
    execution["buffer_reports"] = [
        _seal(
            {name: value for name, value in report.items() if name != "report_sha256"},
            "report_sha256",
        )
        for report in execution["buffer_reports"]
    ]
    execution["scheduled_records"] = [
        _seal(
            {name: value for name, value in report.items() if name != "record_sha256"},
            "record_sha256",
        )
        for report in execution["scheduled_records"]
    ]
    backend["execution_report"] = _seal(execution, "report_sha256")
    artifact["candidate_replay_backend"] = _seal(
        backend,
        "report_sha256",
    )
    artifact["candidate_replay_backend_report_sha256"] = artifact[
        "candidate_replay_backend"
    ]["report_sha256"]
    layer["operator_artifact"] = _seal(artifact, "report_sha256")
    run_body["layer_results"][0] = _seal(layer, "result_sha256")

    compiled = run_body["compiled_model_fold"]
    compiled["ordered_layer_result_sha256"] = [
        run_body["layer_results"][0]["result_sha256"]
    ]
    contract = compiled["short_conv_v5_real_numerical_adapter_contract"]
    proof = v5_compiler_module._prove_real_v5_numerical_layer(
        Lfm2CpcfCompletedLayerReceipt(run_body["layer_results"][0]),
        adapter_contract=contract,
    )
    assert proof is not None
    compiled["short_conv_v5_real_numerical_layer_proofs"] = [dict(proof)]
    _reseal_v5_compiled_report(run_body)
    execution_sha256 = run_body["layer_results"][0]["operator_artifact"][
        "candidate_replay_backend"
    ]["execution_report"]["report_sha256"]
    run_body["gpu_execution_audit"]["execution_report_sha256"] = [execution_sha256]


def _extract_test_run_provenance(run_body: dict[str, Any]) -> dict[str, Any]:
    return reviewed_gate_module._extract_cpcf_run_provenance(
        run_body,
        expected_parent_model_fingerprint=(run_body["inputs"]["parent_fingerprint"]),
    )


def test_reviewed_v5_rejects_self_resealed_blank_prefix_references(
    tmp_path: Path,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    compiled = body["compiled_model_fold"]
    compiled["resume_publication_enabled"] = True
    compiled["resume_admitted_prefixes"] = [
        {
            "next_layer_index": index,
            "frontier_reference": {},
            "checkpoint_reference": {},
            "frontier_sha256": canonical_sha256({"frontier": index}),
            "admission_sha256": canonical_sha256({"admission": index}),
            "admission_chain_proof": {},
            "admission_chain_proof_sha256": canonical_sha256(
                {"proof": index}
            ),
        }
        for index in range(
            compiled["resume_start_layer_index"],
            len(compiled["ordered_layer_result_sha256"]) + 1,
        )
    ]
    _reseal_v5_compiled_report(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="invalid typed artifact references",
    ):
        _extract_test_run_provenance(body)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("candidate_replay_backend_injected", False),
        ("preinstall_candidate_selector_active", False),
        ("short_conv_v5_endpoint_first_selector_integrated", False),
        ("short_conv_v5_numerical_policy_integrated", False),
        ("short_conv_v5_real_numerical_layer_proofs", []),
        ("short_conv_v5_real_numerical_adapter_contract", None),
    ),
)
def test_reviewed_v5_rejects_missing_or_altered_semantic_fields(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    if replacement is None:
        body["compiled_model_fold"].pop(field)
    else:
        body["compiled_model_fold"][field] = replacement
    _reseal_v5_compiled_report(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="v5",
    ):
        _extract_test_run_provenance(body)


def test_reviewed_v5_recomputes_numerical_contract_and_layer_proof(
    tmp_path: Path,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    contract = body["compiled_model_fold"][
        "short_conv_v5_real_numerical_adapter_contract"
    ]
    contract.pop("contract_sha256")
    contract["adapter_id"] = "forged-but-self-sealed-adapter-id"
    contract["contract_sha256"] = canonical_sha256(contract)
    _reseal_v5_compiled_report(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="numerical",
    ):
        _extract_test_run_provenance(body)


def test_reviewed_v5_requires_current_run_gpu_execution_audit(
    tmp_path: Path,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    body["gpu_execution_audit"]["execution_report_sha256"] = [
        canonical_sha256({"forged": "prior-gpu-receipt"})
    ]

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="GPU execution audit",
    ):
        _extract_test_run_provenance(body)


@pytest.mark.parametrize(
    "attack",
    (
        "plan-policy",
        "parity-pass",
        "buffer-bound",
        "assignment-work-id",
        "scheduled-device",
        "execution-zero-audit",
    ),
)
def test_reviewed_v5_rejects_self_resealed_gpu_execution_attack(
    tmp_path: Path,
    attack: str,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    execution = body["layer_results"][0]["operator_artifact"][
        "candidate_replay_backend"
    ]["execution_report"]
    if attack == "plan-policy":
        execution["plan"]["policy"]["parity_absolute_tolerance"] = 1.0
    elif attack == "parity-pass":
        execution["parity_reports"][0]["passed"] = False
    elif attack == "buffer-bound":
        execution["buffer_reports"][0]["maximum_observed_slot_bytes"] = (256 << 20) + 1
    elif attack == "assignment-work-id":
        assignment = execution["plan"]["assignments"][0]
        assignment["work_id"] = "forged-resealed-work-id"
        assignment_body = {
            name: value
            for name, value in assignment.items()
            if name != "assignment_sha256"
        }
        assignment["assignment_sha256"] = canonical_sha256(assignment_body)
    elif attack == "scheduled-device":
        execution["scheduled_records"][0]["device_index"] = 1
    else:
        execution["training_steps"] = 1
    _reseal_v5_gpu_layer_and_compiled(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="GPU|scheduler|parity|buffer|scheduled|zero",
    ):
        _extract_test_run_provenance(body)


@pytest.mark.parametrize(
    "attack",
    (
        "empty-tensor-metrics",
        "different-cpu-output",
        "structure-drift",
    ),
)
def test_reviewed_v5_rejects_impossible_resealed_representative_parity(
    tmp_path: Path,
    attack: str,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    parity_reports = body["layer_results"][0]["operator_artifact"][
        "candidate_replay_backend"
    ]["execution_report"]["parity_reports"]
    if attack == "empty-tensor-metrics":
        parity_reports[0]["tensor_reports"] = []
        parity_reports[0]["tensor_leaf_count"] = 0
    elif attack == "different-cpu-output":
        parity_reports[1]["cpu_output_sha256"] = canonical_sha256(
            {"forged": "second-cpu-reference"}
        )
    else:
        descriptor = parity_reports[1]["structure_descriptor"]
        target_dtype = next(
            child
            for child in descriptor["children"]
            if child["path"] == "$.target_dtype"
        )
        target_dtype["value"] = "float32"
        parity_reports[1]["structure_sha256"] = canonical_sha256(descriptor)
    _reseal_v5_gpu_layer_and_compiled(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="parity|tensor|structure|CPU reference",
    ):
        _extract_test_run_provenance(body)


def test_reviewed_v5_recomputes_round_robin_after_joint_assignment_reseal(
    tmp_path: Path,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    backend = body["layer_results"][0]["operator_artifact"]["candidate_replay_backend"]
    execution = backend["execution_report"]
    assignments = execution["plan"]["assignments"]
    records = execution["scheduled_records"]
    assert [assignment["device_index"] for assignment in assignments] == [0, 1]
    device_bindings = [
        {
            "device_index": assignment["device_index"],
            "device_receipt_sha256": assignment["device_receipt_sha256"],
            "installed_state_sha256": assignment["installed_state_sha256"],
        }
        for assignment in assignments
    ]
    for index, assignment in enumerate(assignments):
        forged_binding = device_bindings[1 - index]
        assignment.update(forged_binding)
        assignment_body = {
            name: value
            for name, value in assignment.items()
            if name != "assignment_sha256"
        }
        assignment["assignment_sha256"] = canonical_sha256(assignment_body)
        records[index].update(forged_binding)
        backend["compute_device_by_work_id"][assignment["work_id"]] = (
            f"cuda:{assignment['device_index']}"
        )
    _reseal_v5_gpu_layer_and_compiled(body)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="round-robin|scheduler assignment",
    ):
        _extract_test_run_provenance(body)


@pytest.mark.parametrize(
    "attack",
    ("device-identity", "runtime-reauthentication", "top-zero-audit"),
)
def test_reviewed_v5_rejects_self_resealed_gpu_runtime_attack(
    tmp_path: Path,
    attack: str,
) -> None:
    body = _v5_unsealed_run_body(tmp_path)
    gpu = body["gpu_candidate_replay"]
    gpu.pop("report_sha256")
    if attack == "device-identity":
        device = gpu["devices"][0]
        device.pop("receipt_sha256")
        device["device_uuid"] = "GPU-forged-reviewed-identity"
        gpu["devices"][0] = _seal(device, "receipt_sha256")
    elif attack == "runtime-reauthentication":
        runtime = gpu["runtime_authentication"]
        runtime.pop("report_sha256")
        runtime["installed_states_freshly_rehashed"] = False
        gpu["runtime_authentication"] = _seal(runtime, "report_sha256")
        gpu["runtime_authentication_sha256"] = gpu["runtime_authentication"][
            "report_sha256"
        ]
    else:
        gpu["training_steps"] = 1
    body["gpu_candidate_replay"] = _seal(gpu, "report_sha256")

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="GPU|runtime|replica|zero-operation",
    ):
        _extract_test_run_provenance(body)


def test_reviewed_v5_rejects_v4_report_relabelled_as_v5(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["schema"] = LFM2_CPCF_V5_MODEL_FOLD_SCHEMA
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="typed policy",
    ):
        _extract_test_run_provenance(body)


def test_reviewed_loader_rejects_unversioned_compiler_schema(
    tmp_path: Path,
) -> None:
    fixture = _fixture(
        tmp_path,
        compiled_schema="neural-forge-lfm2-cpcf-unreviewed-fold-v99",
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="schema is unsupported",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])


def test_reviewed_evidence_loader_seal_binds_compiler_schema(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])

    object.__setattr__(
        evidence,
        "compiled_model_fold_schema",
        LFM2_CPCF_V5_MODEL_FOLD_SCHEMA,
    )
    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="self-signed, or mutated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "replacement_value"),
    (
        (
            "cpcf_run_receipt_sha256",
            canonical_sha256({"spoof": "run-receipt"}),
        ),
        (
            "cpcf_run_receipt_document_sha256",
            canonical_sha256({"spoof": "run-document"}),
        ),
        (
            "parent_cache_key",
            canonical_sha256({"spoof": "cache"}),
        ),
        (
            "cache_live_parent_fingerprint",
            canonical_sha256({"spoof": "parent"}),
        ),
        (
            "trace_adapter_sha256",
            canonical_sha256({"spoof": "trace-adapter"}),
        ),
        (
            "trajectory_corpus_sha256",
            canonical_sha256({"spoof": "trajectory"}),
        ),
        (
            "blank_target_parent_graph_fingerprint",
            canonical_sha256({"spoof": "parent-graph"}),
        ),
        ("exact_target_unique_parameters", 2),
        (
            "target_requested_budget",
            MappingProxyType({"spoof": "budget"}),
        ),
        (
            "selected_target_candidate_fingerprint",
            canonical_sha256({"spoof": "candidate"}),
        ),
        ("selected_target_candidate_rank", 1),
        (
            "target_plan_binding_sha256",
            canonical_sha256({"spoof": "target-plan"}),
        ),
        (
            "compiled_model_fold_result_sha256",
            canonical_sha256({"spoof": "compiled-result"}),
        ),
        (
            "compiled_model_fold_policy_sha256",
            canonical_sha256({"spoof": "compiled-policy"}),
        ),
        ("compiled_model_fold_phase_count", 1),
        (
            "compiled_model_fold_phase_binding_sha256",
            canonical_sha256({"spoof": "compiled-phases"}),
        ),
        (
            "candidate_state_fingerprint",
            canonical_sha256({"spoof": "state"}),
        ),
        (
            "candidate_outputs_sha256",
            canonical_sha256({"spoof": "outputs"}),
        ),
        (
            "candidate_tokenizer_fingerprint",
            canonical_sha256({"spoof": "tokenizer"}),
        ),
        (
            "target_tokenizer_semantics_sha256",
            canonical_sha256({"spoof": "semantics"}),
        ),
        ("eos_token_ids", (1,)),
    ),
)
def test_loader_authenticated_evidence_rejects_typed_field_mutation(
    tmp_path: Path,
    field: str,
    replacement_value: Any,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    object.__setattr__(evidence, field, replacement_value)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="self-signed, or mutated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


@pytest.mark.parametrize(
    ("field", "replacement_value"),
    (
        ("schema", "spoofed-reviewed-schema-v1"),
        ("bridge_artifact_sha256", "not-a-digest"),
        ("bridge_profile_sha256", "not-a-digest"),
        (
            "parent_cache_key",
            canonical_sha256({"spoof": "reviewed-cache"}),
        ),
        ("exact_target_unique_parameters", 2),
        (
            "target_plan_binding_sha256",
            canonical_sha256({"spoof": "reviewed-plan"}),
        ),
        (
            "compiled_model_fold_result_sha256",
            canonical_sha256({"spoof": "reviewed-compiled-result"}),
        ),
        (
            "compiled_model_fold_phase_binding_sha256",
            canonical_sha256({"spoof": "reviewed-compiled-phases"}),
        ),
        (
            "candidate_state_fingerprint",
            canonical_sha256({"spoof": "reviewed-state"}),
        ),
    ),
)
def test_loader_authenticated_evidence_rejects_reviewed_map_mutation(
    tmp_path: Path,
    field: str,
    replacement_value: Any,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    reviewed = dict(evidence.reviewed_evidence)
    reviewed[field] = replacement_value
    object.__setattr__(
        evidence,
        "reviewed_evidence",
        MappingProxyType(reviewed),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="self-signed, or mutated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_evidence(evidence)


def test_spoofed_trusted_document_digest_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    spoofed = _replace_artifact(
        fixture["artifacts"],
        "candidate_outputs",
        CanonicalJsonArtifact(
            source=fixture["documents"]["candidate"],
            document_sha256="0" * 64,
        ),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="trusted document",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(spoofed)


def test_run_receipt_parent_must_match_reviewed_holdout_parent(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["inputs"]["parent_fingerprint"] = canonical_sha256({"different": "parent"})
    modified = _seal(body, "receipt_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="differs from the reviewed holdout parent",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_run_receipt_exact_target_count_must_cross_bind_every_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["reloaded_parameter_inventory"] = _parameter_inventory(2)
    modified = _seal(body, "receipt_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="one exact target parameter count",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_run_receipt_requires_compiled_model_fold(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body.pop("compiled_model_fold")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="no compiled_model_fold report",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_result_seal_is_recomputed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    body["compiled_model_fold"]["policy"]["ridge"] = 0.125
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="compiled model-fold report schema or canonical seal",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_requires_zero_operation_audit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["operation_audit"]["training_steps"] = 1
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="zero-operation invariants",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_binds_every_ordered_phase(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["phase_count"] += 1
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="phase seals do not cover every phase",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_cross_binds_compiled_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["target_parameter_inventory_after"] = _parameter_inventory(2)
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="do not share exact inventories",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_policy_must_be_complete_and_typed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    body = copy.deepcopy(fixture["documents"]["cpcf"])
    body.pop("receipt_sha256")
    compiled = body["compiled_model_fold"]
    compiled.pop("result_sha256")
    compiled["policy"].pop("short_conv_gate_policy")
    body["compiled_model_fold"] = _seal(compiled, "result_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "cpcf_run_receipt",
        _artifact(_seal(body, "receipt_sha256")),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="no short-conv gate policy",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_compiled_model_fold_v4_policy_round_trips_typed() -> None:
    expected = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
            context_objective_weight=0.075,
        )
    )

    observed = reviewed_gate_module._normalize_model_fold_policy(expected.as_dict())

    assert observed == expected.as_dict()


def test_compiled_model_fold_v4_policy_rejects_authentication_drift() -> None:
    policy = Lfm2CpcfModelFoldPolicy(
        short_conv_v4_solver_policy=Lfm2CpcfShortConvV4SolverPolicy(
            ridge_alphas=(1e-8, 1e-4),
            variable_projection_passes=2,
        )
    ).as_dict()
    policy["short_conv_v4_solver_policy"]["authentication_sha256"] = "0" * 64

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="invalid",
    ):
        reviewed_gate_module._normalize_model_fold_policy(policy)


@pytest.mark.parametrize("status,promoted", [("pending", False), ("rejected", False)])
def test_pending_or_rejected_composite_report_rejects(
    tmp_path: Path,
    status: str,
    promoted: bool,
) -> None:
    fixture = _fixture(tmp_path)
    document = copy.deepcopy(fixture["documents"]["composite"])
    body = dict(document)
    body.pop("promotion_report_sha256")
    body["status"] = status
    body["promoted"] = promoted
    modified = _seal(body, "promotion_report_sha256")
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "composite_promotion_report",
        _artifact(modified),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="pending, rejected, spoofed, or stale",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_rejected_parent_relative_report_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    packet = fixture["documents"]["parent_packet"]
    ratings = bind_parent_relative_manual_ratings(
        packet,
        [
            {
                "case_id": value["case_id"],
                "parent": {
                    "correctness": 4,
                    "coherence": 4,
                    "instruction_following": 4,
                    "capability_retention": 4,
                },
                "candidate": {
                    "correctness": 0,
                    "coherence": 0,
                    "instruction_following": 0,
                    "capability_retention": 0,
                },
                "reason": "The candidate clearly regressed on every dimension.",
            }
            for value in packet["cases"]
        ],
        reviewer_id="reviewer-2",
    )
    rejected = evaluate_parent_relative_manual_retention(packet, ratings)
    artifacts = _replace_artifact(
        fixture["artifacts"],
        "parent_relative_ratings",
        _artifact(ratings),
    )
    artifacts = _replace_artifact(
        artifacts,
        "parent_relative_report",
        _artifact(rejected),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="pending, rejected, or regressed",
    ):
        load_lfm2_cpcf_reviewed_promotion_evidence(artifacts)


def test_exact_candidate_state_mismatch_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    mismatched = _ReplayModel()
    with torch.no_grad():
        mismatched.anchor.add_(1.0)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="model state differs",
    ):
        gate(mismatched)


def test_reviewed_output_replay_mismatch_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="reviewed candidate output digest",
    ):
        gate(_ReplayModel(output_offset=1))


def test_generate_that_skips_model_forward_rejects(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="without executing",
    ):
        gate(_ReplayModel(execute_forward=False))


def test_reviewed_gate_instance_is_immutable_and_cannot_shadow_evaluation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )

    assert not hasattr(gate, "__dict__")
    with pytest.raises(AttributeError, match="immutable"):
        gate.evaluate_model = lambda model: model  # type: ignore[method-assign]
    with pytest.raises(AttributeError):
        object.__setattr__(gate, "evaluate_model", lambda model: model)
    with pytest.raises(AttributeError, match="immutable"):
        gate.evidence = evidence
    with pytest.raises(AttributeError, match="immutable"):
        del gate.tokenizer
    assert authenticate_lfm2_cpcf_reviewed_promotion_gate(gate) is gate


def test_object_new_gate_cannot_impersonate_constructor_built_gate() -> None:
    forged = object.__new__(Lfm2CpcfReviewedPromotionGate)

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="constructor-authenticated",
    ):
        authenticate_lfm2_cpcf_reviewed_promotion_gate(forged)
    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="constructor-authenticated",
    ):
        forged(_ReplayModel())


def test_replaced_replay_callable_rejects_before_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    monkeypatch.setattr(
        reviewed_gate_module,
        "_greedy_replay_rows",
        lambda model, tokenizer, reviewed: (),
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="implementation object was replaced",
    ):
        gate(fixture["model"])


def test_replaced_v4_policy_parser_invalidates_implementation_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reviewed_gate_module,
        "parse_lfm2_cpcf_short_conv_v4_solver_policy",
        lambda value: value,
    )

    with pytest.raises(
        Lfm2CpcfReviewedPromotionError,
        match="implementation object was replaced",
    ):
        authenticate_lfm2_cpcf_reviewed_gate_implementation()


def test_replay_callable_code_and_resources_are_reauthenticated() -> None:
    replay = reviewed_gate_module._greedy_replay_rows
    original_code = replay.__code__
    original_keyword_defaults = replay.__kwdefaults__

    def spoofed_replay(model: Any, tokenizer: Any, evidence: Any) -> tuple:
        del model, tokenizer, evidence
        return ()

    try:
        replay.__code__ = spoofed_replay.__code__
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_gate_implementation()
    finally:
        replay.__code__ = original_code

    try:
        replay.__kwdefaults__ = {"tampered_resource": True}
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_gate_implementation()
    finally:
        replay.__kwdefaults__ = original_keyword_defaults

    authenticate_lfm2_cpcf_reviewed_gate_implementation()


def test_gate_class_replacement_is_blocked_and_bypass_is_detected(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    original = vars(Lfm2CpcfReviewedPromotionGate)["evaluate_model"]

    with pytest.raises(TypeError, match="implementation-sealed"):
        setattr(
            Lfm2CpcfReviewedPromotionGate,
            "evaluate_model",
            lambda self, model: model,
        )

    type.__setattr__(
        Lfm2CpcfReviewedPromotionGate,
        "evaluate_model",
        lambda self, model: model,
    )
    try:
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="implementation object was replaced",
        ):
            gate(fixture["model"])
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="implementation object was replaced",
        ):
            authenticate_lfm2_cpcf_reviewed_promotion_gate(gate)
    finally:
        type.__setattr__(
            Lfm2CpcfReviewedPromotionGate,
            "evaluate_model",
            original,
        )

    assert gate(fixture["model"]).accepted is True


def test_in_place_gate_method_code_mutation_is_detected_before_dispatch(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    evidence = load_lfm2_cpcf_reviewed_promotion_evidence(fixture["artifacts"])
    gate = Lfm2CpcfReviewedPromotionGate(
        evidence,
        tokenizer=fixture["tokenizer"],
        tokenizer_directory=fixture["tokenizer_directory"],
    )
    method = vars(Lfm2CpcfReviewedPromotionGate)["evaluate_model"]
    original_code = method.__code__

    def spoofed_evaluate_model(self: Any, model: Any) -> Any:
        del self
        return model

    try:
        method.__code__ = spoofed_evaluate_model.__code__
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            gate(fixture["model"])
        with pytest.raises(
            Lfm2CpcfReviewedPromotionError,
            match="callable code/resources changed",
        ):
            authenticate_lfm2_cpcf_reviewed_promotion_gate(gate)
    finally:
        method.__code__ = original_code

    assert gate(fixture["model"]).accepted is True


def test_constants_are_canonical_production_schemas() -> None:
    assert AUTOMATIC_RUNNER_REPORT_SCHEMA.endswith("-v1")
    assert COMPOSITE_PROMOTION_SCHEMA.endswith("-v1")
