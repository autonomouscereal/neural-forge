"""Sequential target-native CPCF compilation for one materialized LFM2 layer.

This module is the bridge between sealed parent circuit evidence and the
already-created dense blank child.  It deliberately compiles one layer in two
causal stages:

1. replay the current child full prefixes and compile/install the attention or
   short-convolution operator against those *actual* child coordinates;
2. replay the installed operator, then compile/install SwiGLU against the
   resulting actual post-operator residual and normalized MLP input.

Only ``fit`` phases enter attention and MLP closed-form solvers.  The
suffix-aware short-convolution compiler receives disjoint ``selection`` cases
for bounded solver choice and ``validation`` cases solely for its final
post-materialization evaluator; its own sealed report proves that neither
split enters any fit.

No model forward method, generation, cache update, backward pass, optimizer,
training, pruning, quantization, or fallback path is used here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from neural_forge.dynamic_replay_scheduler import (
    DeterministicReplayScheduler,
    ReplayReplicaReceipt,
)
from neural_forge.hf_cpcf_sequential_boundary_transport import (
    CpcfSequentialBoundaryTransport,
    fit_cpcf_sequential_downstream_chart_transport,
)
from neural_forge.hf_lfm2_composed_qk_attention_fold import (
    ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
    SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
    Lfm2CompiledComposedEndpointResidualAttentionFold,
    Lfm2CompiledComposedQkAttentionProducerFold,
    Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
    Lfm2ComposedEndpointResidualAttentionFoldPolicy,
    Lfm2ComposedQkAttentionFoldPolicy,
    compile_lfm2_composed_endpoint_residual_attention_fold,
    compile_lfm2_composed_qk_attention_producer_fold,
    compile_lfm2_composed_qk_sequence_effective_attention_fold,
)
from neural_forge.hf_lfm2_cpcf_boundary_bundle import (
    LFM2_CPCF_LAYER_CHARTS_SCHEMA,
    Lfm2CpcfLayerBoundaryCharts,
)
from neural_forge.hf_lfm2_cpcf_candidate_replay import (
    AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator,
    Lfm2CpcfCandidateReplayEvaluator,
)
from neural_forge.hf_lfm2_cpcf_model_compiler import (
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfPhaseStateBank,
    Lfm2CpcfTargetPhase,
    replay_lfm2_cpcf_phase_bank,
    select_lfm2_cpcf_phase_state_bank,
)
from neural_forge.hf_lfm2_cpcf_short_conv_fold import (
    LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA,
    Lfm2CompiledCpcfShortConvFold,
    Lfm2CpcfShortConvFoldCase,
    Lfm2CpcfShortConvGatePolicy,
    Lfm2CpcfShortConvValidationError,
    compile_lfm2_cpcf_short_conv_fold,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4 import (
    LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
    Lfm2CpcfShortConvV4SolverPolicy,
    fixed_projection_trust_path_fractions,
    rank_short_conv_v4_selection_candidates,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_fold import (
    LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_RECEIPT_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA,
    LFM2_CPCF_SHORT_CONV_V5_SELECTOR_REQUEST_SCHEMA,
    Lfm2CompiledCpcfShortConvV4Fold,
    Lfm2CpcfShortConvNumericalFrontierAdapter,
    Lfm2CpcfShortConvV4FoldCase,
    Lfm2CpcfShortConvV5CandidateFrontier,
    Lfm2CpcfShortConvV5CandidateSelector,
    Lfm2CpcfShortConvV5MacrocellCandidate,
    compile_lfm2_cpcf_short_conv_v4_fold,
    select_lfm2_cpcf_short_conv_v5_macrocell_candidate_ids,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v5 import (
    LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
)
from neural_forge.hf_lfm2_cpcf_short_conv_v4_policy import (
    parse_lfm2_cpcf_short_conv_v4_solver_policy,
)
from neural_forge.hf_lfm2_sequence_effective_attention_fold import (
    FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID,
    Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    compile_lfm2_fixed_qk_sequence_effective_attention_fold,
    install_lfm2_fixed_qk_sequence_effective_attention_fold,
)
from neural_forge.hf_lfm2_structural_residual_attention_fold import (
    Lfm2AttentionScorePreservationReport,
    Lfm2CachedAttentionFoldCase,
    Lfm2CachedMlpFoldCase,
    Lfm2CompiledStructuralAttentionFold,
    Lfm2CompiledStructuralAttentionProducerFold,
    Lfm2CompiledStructuralAttentionValueOutputFold,
    Lfm2CompiledStructuralMlpFold,
    Lfm2PreparedStructuralMlpFoldBasis,
    compile_lfm2_structural_attention_producer_fold,
    compile_lfm2_structural_attention_value_output_fold,
    compile_lfm2_structural_residual_attention_fold,
    compile_lfm2_structural_residual_mlp_fold,
    evaluate_lfm2_compiled_attention_producer_score_preservation,
    install_lfm2_structural_attention_fold,
    install_lfm2_structural_attention_producer_fold,
    install_lfm2_structural_attention_score_producer_fold,
    install_lfm2_structural_attention_value_output_fold,
    install_lfm2_structural_mlp_fold,
    prepare_lfm2_structural_residual_mlp_fold_basis,
)


LFM2_CPCF_LAYER_COMPILER_SCHEMA = "neural-forge-lfm2-cpcf-layer-compiler-v1"
LFM2_CPCF_LAYER_FAILURE_SCHEMA = "neural-forge-lfm2-cpcf-layer-failure-v3"
LFM2_CPCF_RESIDUAL_ATTENTION_SPLIT_SCHEMA = (
    "neural-forge-lfm2-cpcf-residual-attention-split-v1"
)
LFM2_CPCF_ATTENTION_CANDIDATE_SELECTION_SCHEMA = (
    "neural-forge-lfm2-cpcf-attention-candidate-selection-v2"
)
LFM2_CPCF_ATTENTION_SEQUENTIAL_TRANSPORT_POLICY_SCHEMA = (
    "neural-forge-lfm2-cpcf-attention-sequential-transport-policy-v1"
)
LFM2_CPCF_SHORT_CONV_TWO_LANE_EXECUTION_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-two-lane-execution-v1"
)
LFM2_CPCF_SHORT_CONV_TWO_LANE_SEMANTIC_PARITY_SCHEMA = (
    "neural-forge-lfm2-cpcf-short-conv-two-lane-semantic-parity-v1"
)
DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID = "direct_monolithic"
RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID = "residual_attention_split"
LFM2_CPCF_ATTENTION_CANDIDATE_IDS = (
    DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID,
    RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID,
    FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID,
)
LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS = (
    *LFM2_CPCF_ATTENTION_CANDIDATE_IDS,
    SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID,
)
LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS = (
    *LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS,
    ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID,
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVALUATION_SPLITS = ("fit", "validation")
_SHORT_CONV_CASE_SPLITS = ("selection", "fit", "validation")
_LAYER_TRANSACTION_STAGES = (
    "preinstall_replay",
    "operator_compile",
    "operator_install",
    "operator_replay",
    "mlp_compile",
    "mlp_install",
    "final_replay",
    "endpoint_evaluation",
    "immutability_audit",
    "result_seal",
)


def _canonical_attention_candidate_ids(
    candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    values = tuple(candidate_ids)
    if values not in {
        LFM2_CPCF_ATTENTION_CANDIDATE_IDS,
        LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS,
        LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS,
    }:
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate inventory is not one authenticated canonical lane."
        )
    return values


@dataclass(frozen=True, slots=True)
class Lfm2CpcfAttentionSequentialTransportPolicy:
    """Explicit opt-in for candidate-local post-layer gauge transport."""

    enabled: bool = False
    alignment_tolerance: float = 1e-12
    orthogonality_tolerance: float = 1e-10
    preservation_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean.")
        for name in (
            "alignment_tolerance",
            "orthogonality_tolerance",
            "preservation_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"{name} must be nonnegative and finite."
                )

    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": LFM2_CPCF_ATTENTION_SEQUENTIAL_TRANSPORT_POLICY_SCHEMA,
            "enabled": self.enabled,
            "alignment_tolerance": float(self.alignment_tolerance),
            "orthogonality_tolerance": float(self.orthogonality_tolerance),
            "preservation_tolerance": float(self.preservation_tolerance),
            "fit_split_only": True,
            "selection_refits_or_mutates_transport": False,
            "selection_selects_frozen_bundle": True,
            "validation_argument_supported": False,
            "automatic_retry_or_retuning_performed": False,
            "training_steps": 0,
            "optimizer_steps": 0,
            "backward_calls": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        return {**body, "report_sha256": _sha(body)}


def parse_lfm2_cpcf_attention_sequential_transport_policy(
    value: Mapping[str, Any],
) -> Lfm2CpcfAttentionSequentialTransportPolicy:
    """Strictly reconstruct one sealed candidate-local transport policy."""

    if not isinstance(value, Mapping):
        raise TypeError("attention sequential transport policy must be a mapping.")
    payload = dict(value)
    try:
        policy = Lfm2CpcfAttentionSequentialTransportPolicy(
            enabled=payload["enabled"],
            alignment_tolerance=payload["alignment_tolerance"],
            orthogonality_tolerance=payload["orthogonality_tolerance"],
            preservation_tolerance=payload["preservation_tolerance"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            "Attention sequential transport policy is incomplete or invalid."
        ) from exc
    expected = policy.as_dict()
    if set(payload) != set(expected) or _sha(payload) != _sha(expected):
        raise Lfm2CpcfLayerCompilerError(
            "Attention sequential transport policy does not round-trip "
            "through its typed authenticated form."
        )
    return policy


_LAYER_ROLLBACK_PROOF_FIELDS = frozenset(
    {
        "target_state_sha256_before",
        "target_state_sha256_after",
        "target_full_inventory_sha256_before",
        "target_full_inventory_sha256_after",
        "target_tensor_bytes_restored",
        "target_tensor_identities_preserved",
        "target_tensor_aliases_preserved",
        "target_tensor_shapes_preserved",
        "target_tensor_dtypes_preserved",
        "target_all_named_parameters_and_buffers_restored",
        "target_parameter_and_buffer_registries_restored",
        "target_nonpersistent_buffer_membership_restored",
        "target_parameter_gradient_identity_and_bytes_restored",
        "target_parameter_gradient_storage_aliases_restored",
        "target_module_modes_restored",
        "source_state_sha256_before",
        "source_state_sha256_after",
        "source_full_inventory_sha256_before",
        "source_full_inventory_sha256_after",
        "source_state_preserved",
        "source_full_inventory_preserved",
        "evidence_sha256_before",
        "evidence_sha256_after",
        "evidence_preserved",
        "input_tensor_sha256_before",
        "input_tensor_sha256_after",
        "input_tensors_preserved",
        "boundary_charts_sha256_before",
        "boundary_charts_sha256_after",
        "boundary_charts_payload_sha256_before",
        "boundary_charts_payload_sha256_after",
        "boundary_charts_preserved",
        "exact_layer_rollback_proven",
        "proof_sha256",
    }
)
_LAYER_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "stage",
        "layer_index",
        "operator_kind",
        "input_bank_sha256",
        "input_tensor_sha256",
        "boundary_charts_sha256",
        "boundary_charts_payload_sha256",
        "cause",
        "nested_short_conv_validation_report",
        "nested_short_conv_validation_report_sha256",
        "nested_short_conv_validation_report_canonical_bytes_sha256",
        "nested_short_conv_validation_report_preserved_byte_for_byte",
        "rollback_proof",
        "layer_result_issued",
        "promotion_claimed",
        "report_sha256",
    }
)
_SHORT_CONV_VALIDATION_REPORT_FIELDS = frozenset(
    {
        "schema",
        "source_hidden_size",
        "target_hidden_size",
        "hidden_width_compression_ratio",
        "source_fir_length",
        "target_fir_length",
        "target_logical_parameters",
        "fit_case_count",
        "validation_case_count",
        "fit_metrics",
        "validation_metrics",
        "fit_case_reports",
        "validation_case_reports",
        "output_writer_scope",
        "producer_and_fir_scope",
        "fir_state_policy",
        "writer_fit_rows",
        "fit_prefix_rows",
        "fit_fir_warmup_rows",
        "provenance",
        "behavior_gate",
        "audit",
        "execution",
        "compiled_state_sha256",
        "report_sha256",
    }
)
_SHORT_CONV_GATE_POLICY_FIELDS = frozenset(
    {
        "maximum_fit_endpoint_relative_rmse",
        "maximum_validation_endpoint_relative_rmse",
        "minimum_fit_endpoint_cosine_similarity",
        "minimum_validation_endpoint_cosine_similarity",
        "enforce",
    }
)
_SHORT_CONV_GATE_CHECK_FIELDS = frozenset(
    {
        "fit_endpoint_relative_rmse",
        "validation_endpoint_relative_rmse",
        "fit_endpoint_cosine_similarity",
        "validation_endpoint_cosine_similarity",
    }
)
_SHORT_CONV_AUDIT_FIELDS = frozenset(
    {
        "source_inputs_sha256_before",
        "source_inputs_sha256_after",
        "source_inputs_unchanged",
        "target_templates_sha256_before",
        "target_templates_sha256_after",
        "target_templates_unchanged",
        "evidence_sha256_before",
        "evidence_sha256_after",
        "evidence_unchanged",
        "case_authentication_verified",
        "blank_target_values_used_by_compiler",
        "target_templates_used_for_shape_and_dtype_only",
        "returned_tensors_detached",
    }
)
_SHORT_CONV_EXECUTION_FIELDS = frozenset(
    {
        "closed_form_linear_solve",
        "deterministic_barycentric_channel_clustering",
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "backward_calls",
        "optimizer_steps",
        "training_steps",
        "parameters_pruned",
        "parameters_quantized",
    }
)
_SHORT_CONV_V4_FAILURE_REPORT_FIELDS = frozenset(
    {
        "schema",
        "solver_schema",
        "solver_policy",
        "solver_policy_sha256",
        "gate_policy",
        "source_hidden_size",
        "target_hidden_size",
        "source_fir_length",
        "target_fir_length",
        "hidden_width_compression_ratio",
        "case_authentication_sha256",
        "basis",
        "input_lift",
        "transported_priors",
        "ridge_fit_reports",
        "candidate_fit_reports",
        "candidate_selection_reports",
        "candidate_pool_report",
        "selection_ranking",
        "selected_candidate_id",
        "fit_evaluation",
        "selection_evaluation",
        "validation_evaluation",
        "behavior_gate",
        "compiled_state_sha256",
        "source_channels_selected",
        "source_channels_dropped",
        "source_channels_pruned",
        "source_channels_quantized",
        "fit_only_parameter_solves",
        "selection_used_only_for_candidate_ranking",
        "validation_used_only_after_candidate_freeze",
        "execution",
        "compiled_short_conv_fold_issued",
        "promotion_claimed",
        "promotable_candidate",
        "failure_evidence_only",
        "report_sha256",
    }
)
_SHORT_CONV_V4_FAILURE_V1_REPORT_FIELDS = _SHORT_CONV_V4_FAILURE_REPORT_FIELDS - {
    "ridge_fit_reports",
    "candidate_pool_report",
}
_SHORT_CONV_V4_GATE_CHECK_FIELDS = frozenset(
    {
        "fit_endpoint_relative_rmse",
        "validation_endpoint_relative_rmse",
        "fit_endpoint_cosine_similarity",
        "validation_endpoint_cosine_similarity",
        "fit_context_relative_rmse",
        "validation_context_relative_rmse",
        "fit_context_cosine_similarity",
        "validation_context_cosine_similarity",
        "full_source_numeric_participation",
        "actual_fit_target_dtype_replay",
        "actual_validation_target_dtype_replay",
    }
)
_SHORT_CONV_V4_EXECUTION_FIELDS = frozenset(
    {
        "parent_model_forward_calls",
        "target_model_forward_calls",
        "generation_calls",
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
        "fallback_paths",
    }
)
_SHORT_CONV_V5_NUMERICAL_EXECUTION_FIELDS = _SHORT_CONV_V4_EXECUTION_FIELDS - {
    "fallback_paths"
}
_SHORT_CONV_V5_SELECTOR_EXECUTION_FIELDS = frozenset(
    {
        "training_steps",
        "backward_calls",
        "optimizer_steps",
        "parameters_pruned",
        "parameters_quantized",
    }
)
_SHORT_CONV_V5_NUMERICAL_REQUEST_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "policy_sha256",
        "target_dtype",
        "source_tensor_sha256",
        "source_tensor_inventory_sha256",
        "fit_case_authentication_sha256",
        "parent_context_sha256",
        "parent_context_inventory_sha256",
        "target_template_sha256",
        "target_template_inventory_sha256",
        "source_hidden_size",
        "target_hidden_size",
        "source_fir_length",
        "target_fir_length",
        "selection_cases_exposed",
        "validation_cases_exposed",
        "fit_only",
        "execution",
        "report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "request_sha256",
        "candidate_id",
        "candidate_kind",
        "state_sha256",
        "method_provenance",
        "method_provenance_sha256",
        "fit_only",
        "selection_cases_consumed",
        "validation_cases_consumed",
        "execution",
        "report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_FRONTIER_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "adapter_id",
        "request_sha256",
        "context_encoder_sha256",
        "fit_case_authentication_sha256",
        "raw_candidate_count",
        "raw_candidate_ids",
        "raw_candidate_state_sha256",
        "raw_candidate_report_sha256",
        "raw_candidate_reports",
        "source_channel_count",
        "source_numeric_participant_indices",
        "source_channels_selected",
        "source_channels_dropped",
        "full_source_numeric_participation",
        "adapter_provenance",
        "adapter_provenance_sha256",
        "fit_only",
        "selection_cases_consumed",
        "validation_cases_consumed",
        "execution",
        "report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_PROVENANCE_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "candidate_id",
        "policy_sha256",
        "target_dtype",
        "request_sha256",
        "numerical_frontier_report_sha256",
        "target_dtype_candidate_state_sha256",
        "source_alias_count",
        "source_alias_candidate_ids",
        "sources",
        "fit_case_authentication_sha256",
        "fit_only",
        "selection_cases_consumed",
        "validation_cases_consumed",
        "actual_target_dtype_materialization_before_deduplication",
        "legacy_v4_candidate_provenance_claimed",
        "execution",
        "report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_SOURCE_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_kind",
        "numerical_raw_state_sha256",
        "numerical_raw_candidate_report",
        "numerical_raw_candidate_report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_POOL_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "policy_sha256",
        "target_dtype",
        "request_sha256",
        "numerical_frontier_report_sha256",
        "raw_candidate_count",
        "unique_target_dtype_candidate_count",
        "target_dtype_duplicate_count",
        "selection_feasibility_policy",
        "selection_feasibility_policy_sha256",
        "fold_case_authentication_sha256",
        "projection_case_authentication_sha256",
        "candidate_ids",
        "candidate_state_sha256",
        "source_channels_selected",
        "source_channels_dropped",
        "full_source_numeric_participation",
        "fit_only_candidate_generation",
        "selection_cases_consumed_during_generation",
        "validation_cases_consumed_during_generation",
        "legacy_v4_candidate_pool_claimed",
        "report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_POOL_REPLAY_FIELDS = frozenset(
    {
        "candidate_replay_backend_report_sha256",
        "candidate_replay_work_items",
        "candidate_replay_work_item_count",
        "canonical_candidate_replay_order",
        "scheduled_replay_changes_closed_form_solve_math",
    }
)
_SHORT_CONV_V4_EVALUATION_FIELDS = frozenset(
    {
        "schema",
        "kind",
        "policy_sha256",
        "split",
        "target_dtype",
        "case_balanced",
        "case_count",
        "cases",
        "evidence_sha256",
        "candidate_state_sha256",
        "mean_case_relative_rmse",
        "maximum_case_relative_rmse",
        "mean_case_cosine_similarity",
        "minimum_case_cosine_similarity",
        "maximum_case_absolute_error",
        "mean_case_target_dtype_drift_relative_rmse",
        "context_evidence_present",
        "mean_case_context_relative_rmse",
        "maximum_case_context_relative_rmse",
        "mean_case_context_cosine_similarity",
        "minimum_case_context_cosine_similarity",
        "maximum_case_context_absolute_error",
        "mean_case_context_target_dtype_drift_relative_rmse",
        "actual_parameter_and_intermediate_target_dtype_replay",
        "final_output_materialization_only",
        "selection_used_for_parameter_fitting",
        "validation_used_for_parameter_fitting",
        "report_sha256",
    }
)
_SHORT_CONV_V4_EVALUATION_CASE_FIELDS = frozenset(
    {
        "case_id",
        "rows",
        "width",
        "relative_rmse",
        "cosine_similarity",
        "maximum_absolute_error",
        "mean_absolute_error",
        "target_dtype_drift_relative_rmse",
        "fp64_prediction_sha256",
        "materialized_prediction_sha256",
        "target_sha256",
        "valid_rows_sha256",
        "projection_case_authentication_sha256",
        "context_rows",
        "context_width",
        "context_relative_rmse",
        "context_cosine_similarity",
        "context_maximum_absolute_error",
        "context_target_dtype_drift_relative_rmse",
        "context_target_sha256",
        "valid_context_rows_sha256",
        "case_authentication_sha256",
        "candidate_state_sha256",
        "actual_parameter_and_intermediate_target_dtype_replay",
        "arithmetic_report_sha256",
    }
)
_SHORT_CONV_V5_NUMERICAL_SOURCE_TENSOR_ROLES = frozenset(
    {
        "source_in_proj_weight",
        "source_in_proj_bias",
        "source_fir_kernel",
        "source_fir_bias",
        "source_out_proj_weight",
        "source_out_proj_bias",
        "output_encoder",
    }
)
_SHORT_CONV_V5_NUMERICAL_TARGET_TEMPLATE_ROLES = frozenset(
    {
        "in_proj_weight",
        "in_proj_bias",
        "fir_kernel",
        "fir_bias",
        "writer_weight",
        "writer_bias",
    }
)
_NO_SUCCESS_FIELDS = frozenset(
    {
        "accepted_artifact",
        "behavior_success_proven",
        "compaction_success_proven",
        "compiled_model_fold_issued",
        "layer_result_issued",
        "parameter_compaction_success_receipt_issued",
        "promotable_candidate",
        "promotion_authorized",
        "promotion_claimed",
        "success_claimed",
    }
)


class Lfm2CpcfLayerCompilerError(ValueError):
    """One target layer cannot be compiled from the supplied sealed evidence."""


class Lfm2CpcfLayerInstallError(RuntimeError):
    """A target layer installation failed and was rolled back."""


class Lfm2CpcfLayerCompilationFailure(Lfm2CpcfLayerInstallError):
    """A typed layer failure emitted only after exact rollback is proven."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        layer_index: int,
        operator_kind: str,
        input_bank_sha256: str,
        input_tensor_sha256: str,
        boundary_charts_sha256: str,
        boundary_charts_payload_sha256: str,
        rollback_proof: Mapping[str, Any],
        report: Mapping[str, Any],
        short_conv_validation_report: Mapping[str, Any] | None = None,
        short_conv_validation_report_bytes: bytes | None = None,
    ) -> None:
        super().__init__(message)
        if stage not in _LAYER_TRANSACTION_STAGES:
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure stage is not a strict transaction stage."
            )
        index = _nonnegative_int(layer_index, "layer_index")
        if operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure operator_kind is not canonical LFM2."
            )
        input_sha = _digest(input_bank_sha256, "input_bank_sha256")
        input_tensor_sha = _digest(
            input_tensor_sha256,
            "input_tensor_sha256",
        )
        charts_sha = _digest(
            boundary_charts_sha256,
            "boundary_charts_sha256",
        )
        charts_payload_sha = _digest(
            boundary_charts_payload_sha256,
            "boundary_charts_payload_sha256",
        )
        proof = dict(rollback_proof)
        proof_sha = _digest(
            proof.pop("proof_sha256", None),
            "rollback_proof.proof_sha256",
        )
        proof_with_seal = {**proof, "proof_sha256": proof_sha}
        for name in (
            "target_state_sha256_before",
            "target_state_sha256_after",
            "target_full_inventory_sha256_before",
            "target_full_inventory_sha256_after",
            "source_state_sha256_before",
            "source_state_sha256_after",
            "source_full_inventory_sha256_before",
            "source_full_inventory_sha256_after",
            "evidence_sha256_before",
            "evidence_sha256_after",
            "input_tensor_sha256_before",
            "input_tensor_sha256_after",
            "boundary_charts_sha256_before",
            "boundary_charts_sha256_after",
            "boundary_charts_payload_sha256_before",
            "boundary_charts_payload_sha256_after",
        ):
            _digest(proof.get(name), f"rollback_proof.{name}")
        true_proof_fields = (
            "target_tensor_bytes_restored",
            "target_tensor_identities_preserved",
            "target_tensor_aliases_preserved",
            "target_tensor_shapes_preserved",
            "target_tensor_dtypes_preserved",
            "target_all_named_parameters_and_buffers_restored",
            "target_parameter_and_buffer_registries_restored",
            "target_nonpersistent_buffer_membership_restored",
            "target_parameter_gradient_identity_and_bytes_restored",
            "target_parameter_gradient_storage_aliases_restored",
            "target_module_modes_restored",
            "source_state_preserved",
            "source_full_inventory_preserved",
            "evidence_preserved",
            "input_tensors_preserved",
            "boundary_charts_preserved",
            "exact_layer_rollback_proven",
        )
        if (
            set(proof_with_seal) != _LAYER_ROLLBACK_PROOF_FIELDS
            or _sha(proof) != proof_sha
            or any(proof.get(name) is not True for name in true_proof_fields)
            or proof.get("target_state_sha256_before")
            != proof.get("target_state_sha256_after")
            or proof.get("target_full_inventory_sha256_before")
            != proof.get("target_full_inventory_sha256_after")
            or proof.get("source_state_sha256_before")
            != proof.get("source_state_sha256_after")
            or proof.get("source_full_inventory_sha256_before")
            != proof.get("source_full_inventory_sha256_after")
            or proof.get("evidence_sha256_before") != proof.get("evidence_sha256_after")
            or proof.get("input_tensor_sha256_before")
            != proof.get("input_tensor_sha256_after")
            or proof.get("boundary_charts_sha256_before")
            != proof.get("boundary_charts_sha256_after")
            or proof.get("boundary_charts_payload_sha256_before")
            != proof.get("boundary_charts_payload_sha256_after")
            or proof.get("input_tensor_sha256_before") != input_tensor_sha
            or proof.get("boundary_charts_sha256_before") != charts_sha
            or proof.get("boundary_charts_payload_sha256_before") != charts_payload_sha
        ):
            raise Lfm2CpcfLayerInstallError(
                "Layer failure lacks an exact rollback proof."
            )
        failure_report = dict(report)
        report_sha = _digest(
            failure_report.pop("report_sha256", None),
            "failure_report.report_sha256",
        )
        report_with_seal = {
            **failure_report,
            "report_sha256": report_sha,
        }
        if (
            set(report_with_seal) != _LAYER_FAILURE_REPORT_FIELDS
            or _sha(failure_report) != report_sha
        ):
            raise Lfm2CpcfLayerCompilerError("Layer failure report seal is invalid.")
        if (
            failure_report.get("schema") != LFM2_CPCF_LAYER_FAILURE_SCHEMA
            or failure_report.get("stage") != stage
            or failure_report.get("layer_index") != index
            or failure_report.get("operator_kind") != operator_kind
            or failure_report.get("input_bank_sha256") != input_sha
            or failure_report.get("input_tensor_sha256") != input_tensor_sha
            or failure_report.get("boundary_charts_sha256") != charts_sha
            or failure_report.get("boundary_charts_payload_sha256")
            != charts_payload_sha
            or failure_report.get("rollback_proof") != proof_with_seal
            or failure_report.get("layer_result_issued") is not False
            or failure_report.get("promotion_claimed") is not False
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Layer failure report differs from its typed context."
            )
        cause = failure_report.get("cause")
        if (
            not isinstance(cause, Mapping)
            or set(cause) != {"module", "type", "message"}
            or any(
                not isinstance(cause.get(name), str)
                for name in ("module", "type", "message")
            )
        ):
            raise Lfm2CpcfLayerCompilerError("Layer failure cause is not canonical.")
        nested = short_conv_validation_report
        nested_bytes = short_conv_validation_report_bytes
        if (nested is None) != (nested_bytes is None):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv validation report and bytes must travel together."
            )
        if nested is not None:
            observed_bytes = _canonical_json(dict(nested))
            if observed_bytes != nested_bytes:
                raise Lfm2CpcfLayerCompilerError(
                    "Nested short-conv validation report bytes changed."
                )
            nested_body = dict(nested)
            nested_sha = _digest(
                nested_body.pop("report_sha256", None),
                "short_conv_validation_report.report_sha256",
            )
            if (
                operator_kind != "short_conv"
                or stage != "operator_compile"
                or _sha(nested_body) != nested_sha
                or not _valid_short_conv_validation_report(nested)
                or failure_report.get("nested_short_conv_validation_report")
                != dict(nested)
                or failure_report.get("nested_short_conv_validation_report_sha256")
                != nested_sha
                or failure_report.get(
                    "nested_short_conv_validation_report_canonical_bytes_sha256"
                )
                != hashlib.sha256(nested_bytes).hexdigest()
                or failure_report.get(
                    "nested_short_conv_validation_report_preserved_byte_for_byte"
                )
                is not True
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Nested short-conv validation evidence is not preserved "
                    "and self-sealed."
                )
        elif (
            failure_report.get("nested_short_conv_validation_report") is not None
            or failure_report.get("nested_short_conv_validation_report_sha256")
            is not None
            or failure_report.get(
                "nested_short_conv_validation_report_canonical_bytes_sha256"
            )
            is not None
            or failure_report.get(
                "nested_short_conv_validation_report_preserved_byte_for_byte"
            )
            is not False
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Absent short-conv validation evidence is inconsistent."
            )
        self.stage = stage
        self.layer_index = index
        self.operator_kind = operator_kind
        self.input_bank_sha256 = input_sha
        self.input_tensor_sha256 = input_tensor_sha
        self.boundary_charts_sha256 = charts_sha
        self.boundary_charts_payload_sha256 = charts_payload_sha
        self._rollback_proof_bytes = _canonical_json(proof_with_seal)
        self._short_conv_validation_report_bytes = nested_bytes
        self._report_bytes = _canonical_json(report_with_seal)

    @property
    def rollback_proof(self) -> Mapping[str, Any]:
        """Return an independent canonical copy of the rollback proof."""

        return json.loads(self._rollback_proof_bytes.decode("ascii"))

    @property
    def short_conv_validation_report(self) -> Mapping[str, Any] | None:
        """Return an independent canonical copy of nested gate evidence."""

        if self._short_conv_validation_report_bytes is None:
            return None
        return json.loads(self._short_conv_validation_report_bytes.decode("ascii"))

    @property
    def short_conv_validation_report_bytes(self) -> bytes | None:
        return self._short_conv_validation_report_bytes

    @property
    def report(self) -> Mapping[str, Any]:
        """Return an independent canonical copy of the failure report."""

        return json.loads(self._report_bytes.decode("ascii"))


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional HF dependency
        raise RuntimeError("PyTorch is required for LFM2 CPCF compilation.") from exc
    return torch


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            "Layer-compiler metadata must be finite canonical JSON."
        ) from exc


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be a lowercase SHA-256 digest.")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be nonnegative.")
    return value


def _nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be finite and nonnegative."
        ) from exc
    if not math.isfinite(result) or result < 0.0:
        raise Lfm2CpcfLayerCompilerError(f"{name} must be finite and nonnegative.")
    return result


def _tensor_sha256(value: Any) -> str:
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Only tensors can be content authenticated.")
    tensor = value.detach().to(device="cpu").contiguous()
    descriptor = {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": [int(size) for size in tensor.shape],
    }
    digest = hashlib.sha256(_canonical_json(descriptor))
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_bundle_sha256(values: Mapping[str, Any]) -> str:
    return _sha({name: _tensor_sha256(value) for name, value in sorted(values.items())})


def _short_conv_state_sha256(values: Mapping[str, Any]) -> str:
    """Match the suffix-aware short-conv compiler's sealed state digest."""

    torch = _torch()
    inventory = {}
    for name, value in sorted(values.items()):
        tensor = value.detach().to(device="cpu").contiguous()
        descriptor = {
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": [int(size) for size in tensor.shape],
            "layout": str(torch.strided),
        }
        digest = hashlib.sha256(_canonical_json(descriptor))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
        inventory[name] = digest.hexdigest()
    return _sha(inventory)


def _model_core(model: Any, name: str) -> Any:
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if core is None or layers is None:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must expose an LFM2-compatible model.layers sequence."
        )
    model_type = getattr(getattr(model, "config", None), "model_type", "lfm2")
    if model_type != "lfm2":
        raise Lfm2CpcfLayerCompilerError(f"{name} must have model_type='lfm2'.")
    return core


def _layer(model: Any, index: int, name: str) -> Any:
    layers = tuple(_model_core(model, name).layers)
    if index >= len(layers):
        raise Lfm2CpcfLayerCompilerError(f"layer_index is outside {name}.model.layers.")
    return layers[index]


def _named_values_all(module: Any, method_name: str) -> tuple[tuple[str, Any], ...]:
    method = getattr(module, method_name, None)
    if not callable(method):
        raise Lfm2CpcfLayerCompilerError(f"Layer must expose {method_name}().")
    try:
        values = tuple(method(remove_duplicate=False))
    except TypeError:  # pragma: no cover - older supported Torch boundary
        values = tuple(method())
    names = [str(name) for name, _value in values]
    if len(set(names)) != len(names):
        raise Lfm2CpcfLayerCompilerError(
            f"Layer {method_name}() contains duplicate names."
        )
    return tuple((str(name), value) for name, value in values)


def _named_modules_all(module: Any) -> tuple[tuple[str, Any], ...]:
    return _named_values_all(module, "named_modules")


def _module_owner(
    module: Any,
    qualified_name: str,
) -> tuple[Any, str]:
    owner_name, _separator, local_name = qualified_name.rpartition(".")
    modules = dict(_named_modules_all(module))
    try:
        return modules[owner_name], local_name
    except KeyError as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"Tensor owner for {qualified_name!r} is absent."
        ) from exc


def _module_tensors(module: Any) -> dict[str, Any]:
    torch = _torch()
    values = _named_values_all(module, "named_parameters") + _named_values_all(
        module, "named_buffers"
    )
    if not values:
        raise Lfm2CpcfLayerCompilerError("Layer has no named parameters or buffers.")
    if len({name for name, _value in values}) != len(values):
        raise Lfm2CpcfLayerCompilerError("Layer parameter and buffer names collide.")
    if any(not isinstance(value, torch.Tensor) for _name, value in values):
        raise Lfm2CpcfLayerCompilerError(
            "Layer parameters and buffers must be tensors."
        )
    return dict(values)


def _tensor_identity(value: Any) -> tuple[Any, ...]:
    storage_pointer = (
        0 if value.numel() == 0 else int(value.untyped_storage().data_ptr())
    )
    return (
        id(value),
        int(value.data_ptr()),
        storage_pointer,
        int(value.storage_offset()),
        tuple(int(item) for item in value.stride()),
        tuple(int(item) for item in value.shape),
        value.dtype,
        value.device.type,
        value.device.index,
    )


def _tensor_identity_record(value: Any) -> dict[str, Any]:
    identity = _tensor_identity(value)
    return {
        "object_id": identity[0],
        "data_pointer": identity[1],
        "storage_pointer": identity[2],
        "storage_offset": identity[3],
        "stride": list(identity[4]),
        "shape": list(identity[5]),
        "dtype": str(identity[6]).removeprefix("torch."),
        "device_type": identity[7],
        "device_index": identity[8],
    }


def _alias_groups(
    values: Mapping[str, Any],
    *,
    storage: bool,
) -> list[list[str]]:
    groups: dict[Any, list[str]] = {}
    for name, value in values.items():
        if storage and value.numel() != 0:
            key = (
                value.device.type,
                value.device.index,
                int(value.untyped_storage().data_ptr()),
            )
        elif storage:
            key = ("empty", id(value))
        else:
            key = id(value)
        groups.setdefault(key, []).append(name)
    return sorted(
        (sorted(names) for names in groups.values()),
        key=lambda names: names,
    )


def _module_registry_inventory(module: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "object_id": id(value),
            "parameters": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._parameters.items()
            ],
            "buffers": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._buffers.items()
            ],
            "children": [
                [local_name, None if child is None else id(child)]
                for local_name, child in value._modules.items()
            ],
            "nonpersistent_buffers": sorted(
                str(item) for item in value._non_persistent_buffers_set
            ),
        }
        for name, value in _named_modules_all(module)
    ]


def _layer_full_inventory(layer: Any) -> dict[str, Any]:
    parameters = dict(_named_values_all(layer, "named_parameters"))
    buffers = dict(_named_values_all(layer, "named_buffers"))
    tensors = {**parameters, **buffers}
    if len(tensors) != len(parameters) + len(buffers):
        raise Lfm2CpcfLayerCompilerError("Layer parameter and buffer names collide.")
    tensor_inventory = {}
    for name, value in sorted(tensors.items()):
        owner, local_name = _module_owner(layer, name)
        tensor_inventory[name] = {
            **_tensor_identity_record(value),
            "role": "parameter" if name in parameters else "buffer",
            "persistent": (
                True
                if name in parameters
                else local_name not in owner._non_persistent_buffers_set
            ),
            "tensor_sha256": _tensor_sha256(value),
        }
    gradients = {}
    gradient_values = {}
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        if gradient is not None:
            gradient_values[name] = gradient
        gradients[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "present": gradient is not None,
            "identity": (
                None if gradient is None else _tensor_identity_record(gradient)
            ),
            "tensor_sha256": (None if gradient is None else _tensor_sha256(gradient)),
        }
    return {
        "tensor_inventory": tensor_inventory,
        "tensor_object_alias_groups": _alias_groups(
            tensors,
            storage=False,
        ),
        "tensor_storage_alias_groups": _alias_groups(
            tensors,
            storage=True,
        ),
        "parameter_gradients": gradients,
        "gradient_object_alias_groups": _alias_groups(
            gradient_values,
            storage=False,
        ),
        "gradient_storage_alias_groups": _alias_groups(
            gradient_values,
            storage=True,
        ),
        "module_registries": _module_registry_inventory(layer),
        "module_modes": [
            [name, id(value), bool(value.training)]
            for name, value in _named_modules_all(layer)
        ],
    }


@dataclass(frozen=True, slots=True)
class _GradientSnapshot:
    requires_grad: bool
    gradient: Any | None
    storage_anchor: Any | None
    value: Any | None
    identity: tuple[Any, ...] | None


@dataclass(frozen=True, slots=True)
class _ModuleRegistrySnapshot:
    module: Any
    parameters: tuple[tuple[str, Any], ...]
    buffers: tuple[tuple[str, Any], ...]
    children: tuple[tuple[str, Any], ...]
    nonpersistent_buffers: frozenset[str]


def _capture_module_registries(
    module: Any,
) -> tuple[_ModuleRegistrySnapshot, ...]:
    seen = set()
    result = []
    for _name, value in _named_modules_all(module):
        if id(value) in seen:
            continue
        seen.add(id(value))
        result.append(
            _ModuleRegistrySnapshot(
                module=value,
                parameters=tuple(value._parameters.items()),
                buffers=tuple(value._buffers.items()),
                children=tuple(value._modules.items()),
                nonpersistent_buffers=frozenset(value._non_persistent_buffers_set),
            )
        )
    return tuple(result)


def _restore_module_registries(
    snapshots: Sequence[_ModuleRegistrySnapshot],
) -> None:
    for snapshot in snapshots:
        module = snapshot.module
        module._parameters.clear()
        module._parameters.update(snapshot.parameters)
        module._buffers.clear()
        module._buffers.update(snapshot.buffers)
        module._modules.clear()
        module._modules.update(snapshot.children)
        module._non_persistent_buffers_set.clear()
        module._non_persistent_buffers_set.update(snapshot.nonpersistent_buffers)


@dataclass(frozen=True, slots=True)
class _LayerSnapshot:
    tensors: Mapping[str, Any]
    tensor_storage_anchors: Mapping[str, Any]
    tensor_identity: Mapping[str, tuple[Any, ...]]
    tensor_bytes_sha256: str
    full_inventory_sha256: str
    registry_inventory_sha256: str
    parameter_gradients: Mapping[str, _GradientSnapshot]
    modes: tuple[tuple[str, int, bool], ...]
    module_registries: tuple[_ModuleRegistrySnapshot, ...]


def _layer_snapshot(layer: Any) -> _LayerSnapshot:
    tensors = _module_tensors(layer)
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    gradients = {}
    for name, parameter in named_parameters.items():
        gradient = parameter.grad
        gradients[name] = _GradientSnapshot(
            requires_grad=bool(parameter.requires_grad),
            gradient=gradient,
            storage_anchor=(None if gradient is None else gradient.detach()),
            value=(
                None
                if gradient is None
                else gradient.detach().to(device="cpu").contiguous().clone()
            ),
            identity=(None if gradient is None else _tensor_identity(gradient)),
        )
    modes = tuple(
        (name, id(module), bool(module.training))
        for name, module in _named_modules_all(layer)
    )
    return _LayerSnapshot(
        tensors=MappingProxyType(
            {
                name: value.detach().to(device="cpu").contiguous().clone()
                for name, value in tensors.items()
            }
        ),
        tensor_storage_anchors=MappingProxyType(
            {name: value.detach() for name, value in tensors.items()}
        ),
        tensor_identity=MappingProxyType(
            {name: _tensor_identity(value) for name, value in tensors.items()}
        ),
        tensor_bytes_sha256=_tensor_bundle_sha256(tensors),
        full_inventory_sha256=_sha(_layer_full_inventory(layer)),
        registry_inventory_sha256=_sha(_module_registry_inventory(layer)),
        parameter_gradients=MappingProxyType(gradients),
        modes=modes,
        module_registries=_capture_module_registries(layer),
    )


def _restore_layer(layer: Any, snapshot: _LayerSnapshot) -> None:
    torch = _torch()
    _restore_module_registries(snapshot.module_registries)
    current = _module_tensors(layer)
    if set(current) != set(snapshot.tensors):
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose state keys changed."
        )
    with torch.no_grad():
        for name, tensor in current.items():
            if _tensor_identity(tensor) != snapshot.tensor_identity[name]:
                tensor.data = snapshot.tensor_storage_anchors[name]
            if _tensor_identity(tensor) != snapshot.tensor_identity[name]:
                raise Lfm2CpcfLayerInstallError(
                    "Cannot restore target tensor storage identity."
                )
            tensor.copy_(
                snapshot.tensors[name].to(
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
            )
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose Parameter set changed."
        )
    for name, parameter in named_parameters.items():
        gradient_snapshot = snapshot.parameter_gradients[name]
        parameter.requires_grad_(gradient_snapshot.requires_grad)
        parameter.grad = gradient_snapshot.gradient
        if gradient_snapshot.gradient is not None:
            gradient = gradient_snapshot.gradient
            with torch.no_grad():
                if _tensor_identity(gradient) != gradient_snapshot.identity:
                    gradient.data = gradient_snapshot.storage_anchor
                gradient.copy_(
                    gradient_snapshot.value.to(
                        device=gradient.device,
                        dtype=gradient.dtype,
                    )
                )
    modules = dict(_named_modules_all(layer))
    if set(modules) != {name for name, _id, _mode in snapshot.modes}:
        raise Lfm2CpcfLayerInstallError(
            "Cannot roll back a target layer whose module tree changed."
        )
    for name, identity, mode in snapshot.modes:
        if id(modules[name]) != identity:
            raise Lfm2CpcfLayerInstallError(
                "Cannot roll back a target layer whose module aliases changed."
            )
        modules[name].training = mode


def _assert_snapshot_identity(layer: Any, snapshot: _LayerSnapshot) -> None:
    current = _module_tensors(layer)
    if set(current) != set(snapshot.tensor_identity) or any(
        _tensor_identity(current[name]) != snapshot.tensor_identity[name]
        for name in current
    ):
        raise Lfm2CpcfLayerInstallError(
            "Target layer tensor identity, shape, or dtype changed during install."
        )
    named_parameters = dict(_named_values_all(layer, "named_parameters"))
    if set(named_parameters) != set(snapshot.parameter_gradients):
        raise Lfm2CpcfLayerInstallError(
            "Target layer Parameter set changed during install."
        )
    for name, parameter in _named_values_all(
        layer,
        "named_parameters",
    ):
        gradient = parameter.grad
        expected = snapshot.parameter_gradients[name]
        if (
            bool(parameter.requires_grad) != expected.requires_grad
            or gradient is not expected.gradient
            or (
                gradient is not None
                and (
                    _tensor_identity(gradient) != expected.identity
                    or _tensor_sha256(gradient) != _tensor_sha256(expected.value)
                )
            )
        ):
            raise Lfm2CpcfLayerInstallError(
                "Target layer gradient state changed during install."
            )
    modes = tuple(
        (name, id(module), bool(module.training))
        for name, module in _named_modules_all(layer)
    )
    if modes != snapshot.modes:
        raise Lfm2CpcfLayerInstallError(
            "Target layer training modes changed during install."
        )
    if _sha(_module_registry_inventory(layer)) != snapshot.registry_inventory_sha256:
        raise Lfm2CpcfLayerInstallError(
            "Target layer registries or buffer persistence changed."
        )


def _boundary_charts_payload_sha256(
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
) -> str:
    values = {}
    for boundary_name in ("post_operator", "post_layer"):
        chart = getattr(boundary_charts, boundary_name)
        mapping = chart.mapping
        values[f"{boundary_name}.encoder"] = mapping.encoder
        values[f"{boundary_name}.decoder"] = mapping.decoder
        for name, tensor in sorted(mapping.feature_predictors.items()):
            values[f"{boundary_name}.feature_predictors.{name}"] = tensor
    return _sha(
        {
            "layer_charts": boundary_charts.as_dict(),
            "post_operator_report": dict(boundary_charts.post_operator.report),
            "post_layer_report": dict(boundary_charts.post_layer.report),
            "tensor_payload": {
                name: {
                    **_tensor_identity_record(value),
                    "tensor_sha256": _tensor_sha256(value),
                }
                for name, value in sorted(values.items())
            },
        }
    )


def _reseal_lfm2_cpcf_layer_boundary_charts(
    source: Lfm2CpcfLayerBoundaryCharts,
    *,
    post_layer: Any,
) -> Lfm2CpcfLayerBoundaryCharts:
    """Create a candidate-local chart pair without mutating its source."""

    body = {
        "schema": LFM2_CPCF_LAYER_CHARTS_SCHEMA,
        "layer_index": source.layer_index,
        "operator_kind": source.operator_kind,
        "source_width": source.source_width,
        "target_width": source.target_width,
        "post_operator_report_sha256": (source.post_operator.report["report_sha256"]),
        "post_layer_report_sha256": post_layer.report["report_sha256"],
        "independent_boundary_coordinates": True,
    }
    return Lfm2CpcfLayerBoundaryCharts(
        layer_index=source.layer_index,
        operator_kind=source.operator_kind,
        post_operator=source.post_operator,
        post_layer=post_layer,
        layer_charts_sha256=_sha(body),
    )


def _live_layer_evidence_sha256(
    phases: Sequence[Lfm2CpcfTargetPhase],
    layer_index: int,
) -> str:
    inventory = []
    for phase in phases:
        evidence = phase.parent_evidence.layers[layer_index]
        inventory.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "phase_sha256": phase.phase_sha256,
                "parent_evidence_sha256": (phase.parent_evidence.evidence_sha256),
                "layer_evidence": evidence.as_dict(),
                "layer_tensor_identity": {
                    name: {
                        **_tensor_identity_record(value),
                        "tensor_sha256": _tensor_sha256(value),
                    }
                    for name, value in sorted(evidence.tensors.items())
                },
            }
        )
    return _sha(inventory)


def _live_input_bank_sha256(
    input_bank: Lfm2CpcfPhaseStateBank,
) -> str:
    phases = []
    for phase in input_bank.phases:
        phases.append(
            {
                "phase": phase.as_dict(),
                "input_ids": {
                    **_tensor_identity_record(phase.input_ids),
                    "tensor_sha256": _tensor_sha256(phase.input_ids),
                },
                "attention_mask": {
                    **_tensor_identity_record(phase.attention_mask),
                    "tensor_sha256": _tensor_sha256(phase.attention_mask),
                },
                "position_ids": {
                    **_tensor_identity_record(phase.position_ids),
                    "tensor_sha256": _tensor_sha256(phase.position_ids),
                },
                "parent_evidence_binding": {
                    "cache_key": phase.parent_evidence.cache_key,
                    "batch_index": phase.parent_evidence.batch_index,
                    "input_sha256": phase.parent_evidence.input_sha256,
                    "phase": phase.parent_evidence.phase,
                    "evidence_sha256": (phase.parent_evidence.evidence_sha256),
                },
                "state": {
                    **_tensor_identity_record(input_bank.states[phase.identity]),
                    "tensor_sha256": _tensor_sha256(input_bank.states[phase.identity]),
                },
            }
        )
    return _sha(
        {
            "boundary": input_bank.boundary,
            "layer_index": input_bank.layer_index,
            "state_bank_sha256": input_bank.state_bank_sha256,
            "phases": phases,
        }
    )


def _prove_exact_layer_rollback(
    layer: Any,
    snapshot: _LayerSnapshot,
    *,
    source_layer: Any,
    source_snapshot: _LayerSnapshot,
    input_bank: Lfm2CpcfPhaseStateBank,
    layer_index: int,
    evidence_sha256_before: str,
    input_sha256_before: str,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    boundary_charts_sha256_before: str,
    boundary_charts_payload_sha256_before: str,
) -> Mapping[str, Any]:
    """Verify target rollback and all immutable transaction inputs."""

    _assert_snapshot_identity(layer, snapshot)
    state_after = _tensor_bundle_sha256(_module_tensors(layer))
    target_full_inventory_after = _sha(_layer_full_inventory(layer))
    if (
        state_after != snapshot.tensor_bytes_sha256
        or target_full_inventory_after != snapshot.full_inventory_sha256
    ):
        raise Lfm2CpcfLayerInstallError(
            "Target layer tensor bytes failed exact rollback."
        )
    _assert_snapshot_identity(source_layer, source_snapshot)
    source_state_after = _tensor_bundle_sha256(_module_tensors(source_layer))
    source_full_inventory_after = _sha(_layer_full_inventory(source_layer))
    evidence_after = _live_layer_evidence_sha256(
        input_bank.phases,
        layer_index,
    )
    input_after = _live_input_bank_sha256(input_bank)
    boundary_payload_after = _boundary_charts_payload_sha256(boundary_charts)
    if (
        source_state_after != source_snapshot.tensor_bytes_sha256
        or source_full_inventory_after != source_snapshot.full_inventory_sha256
        or evidence_after != evidence_sha256_before
        or input_after != input_sha256_before
        or boundary_charts.layer_charts_sha256 != boundary_charts_sha256_before
        or boundary_payload_after != boundary_charts_payload_sha256_before
    ):
        raise Lfm2CpcfLayerInstallError(
            "Layer failure changed source, evidence, incoming states, or "
            "boundary charts."
        )
    body = {
        "target_state_sha256_before": snapshot.tensor_bytes_sha256,
        "target_state_sha256_after": state_after,
        "target_full_inventory_sha256_before": (snapshot.full_inventory_sha256),
        "target_full_inventory_sha256_after": (target_full_inventory_after),
        "target_tensor_bytes_restored": True,
        "target_tensor_identities_preserved": True,
        "target_tensor_aliases_preserved": True,
        "target_tensor_shapes_preserved": True,
        "target_tensor_dtypes_preserved": True,
        "target_all_named_parameters_and_buffers_restored": True,
        "target_parameter_and_buffer_registries_restored": True,
        "target_nonpersistent_buffer_membership_restored": True,
        "target_parameter_gradient_identity_and_bytes_restored": True,
        "target_parameter_gradient_storage_aliases_restored": True,
        "target_module_modes_restored": True,
        "source_state_sha256_before": source_snapshot.tensor_bytes_sha256,
        "source_state_sha256_after": source_state_after,
        "source_full_inventory_sha256_before": (source_snapshot.full_inventory_sha256),
        "source_full_inventory_sha256_after": (source_full_inventory_after),
        "source_state_preserved": True,
        "source_full_inventory_preserved": True,
        "evidence_sha256_before": evidence_sha256_before,
        "evidence_sha256_after": evidence_after,
        "evidence_preserved": True,
        "input_tensor_sha256_before": input_sha256_before,
        "input_tensor_sha256_after": input_after,
        "input_tensors_preserved": True,
        "boundary_charts_sha256_before": (boundary_charts_sha256_before),
        "boundary_charts_sha256_after": (boundary_charts.layer_charts_sha256),
        "boundary_charts_payload_sha256_before": (
            boundary_charts_payload_sha256_before
        ),
        "boundary_charts_payload_sha256_after": (boundary_payload_after),
        "boundary_charts_preserved": True,
        "exact_layer_rollback_proven": True,
    }
    return MappingProxyType({**body, "proof_sha256": _sha(body)})


def _evidence_tensors(
    phases: Sequence[Lfm2CpcfTargetPhase],
    layer_index: int,
) -> dict[str, Any]:
    result = {}
    for phase in phases:
        evidence = phase.parent_evidence.layers[layer_index]
        for name, value in evidence.tensors.items():
            result[f"{phase.batch_index}:{phase.phase}:layer.{layer_index}:{name}"] = (
                value
            )
    return result


def _phase_evidence(
    phase: Lfm2CpcfTargetPhase,
    layer_index: int,
    operator_kind: str,
) -> Mapping[str, Any]:
    try:
        evidence = phase.parent_evidence.layers[layer_index]
    except IndexError as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} lacks layer {layer_index} evidence."
        ) from exc
    if evidence.layer_index != layer_index or evidence.operator_kind != operator_kind:
        raise Lfm2CpcfLayerCompilerError(
            f"Phase {phase.identity!r} has the wrong layer evidence."
        )
    return evidence.tensors


def _finite_tensor(
    value: Any,
    name: str,
    *,
    dimensions: int,
) -> Any:
    torch = _torch()
    if (
        not isinstance(value, torch.Tensor)
        or value.layout != torch.strided
        or value.device.type == "meta"
        or value.ndim != dimensions
        or value.numel() == 0
        or not torch.is_floating_point(value)
        or torch.is_complex(value)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be a nonempty dense floating rank-{dimensions} tensor."
        )
    result = value.detach().to(device="cpu").contiguous()
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError(f"{name} contains NaN or infinity.")
    return result


def _weight(module: Any, name: str, *, dimensions: int) -> Any:
    return _finite_tensor(
        getattr(module, "weight", None),
        f"{name}.weight",
        dimensions=dimensions,
    )


def _variance_epsilon(module: Any, name: str) -> float:
    value = getattr(module, "variance_epsilon", None)
    epsilon = _nonnegative_float(value, f"{name}.variance_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError(f"{name}.variance_epsilon must be positive.")
    return epsilon


def _rotate_half(value: Any) -> Any:
    half = int(value.shape[-1]) // 2
    return _torch().cat((-value[..., half:], value[..., :half]), dim=-1)


def _rope_terms(
    model: Any,
    position_ids: Any,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    core = _model_core(model, "model")
    rotary = getattr(core, "rotary_emb", None)
    inv_freq = getattr(rotary, "inv_freq", None)
    scaling = getattr(rotary, "attention_scaling", None)
    if (
        not isinstance(inv_freq, torch.Tensor)
        or inv_freq.ndim != 1
        or inv_freq.numel() == 0
        or isinstance(scaling, bool)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 rotary embedding lacks static inv_freq/attention_scaling."
        )
    try:
        scale = float(scaling)
    except (TypeError, ValueError, OverflowError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling is invalid."
        ) from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE attention scaling must be positive and finite."
        )
    positions = position_ids.detach().to(device="cpu", dtype=torch.float32)
    frequencies = positions.unsqueeze(-1) * inv_freq.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).reshape(1, 1, -1)
    angles = torch.cat((frequencies, frequencies), dim=-1)
    return (
        (angles.cos() * scale).to(dtype=dtype).contiguous(),
        (angles.sin() * scale).to(dtype=dtype).contiguous(),
    )


def invert_lfm2_rope(
    rotated: Any,
    cosine: Any,
    sine: Any,
) -> Any:
    """Invert duplicated-half LFM2 RoPE, including a finite non-unit radius."""

    torch = _torch()
    value = _finite_tensor(rotated, "rotated", dimensions=4)
    cos = _finite_tensor(cosine, "cosine", dimensions=3)
    sin = _finite_tensor(sine, "sine", dimensions=3)
    if (
        int(value.shape[-1]) % 2
        or tuple(cos.shape)
        != (int(value.shape[0]), int(value.shape[2]), int(value.shape[3]))
        or tuple(sin.shape) != tuple(cos.shape)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Rotated rows and RoPE terms have incompatible shapes."
        )
    half = int(cos.shape[-1]) // 2
    if not torch.equal(cos[..., :half], cos[..., half:]) or not torch.equal(
        sin[..., :half],
        sin[..., half:],
    ):
        raise Lfm2CpcfLayerCompilerError(
            "LFM2 RoPE terms do not have duplicated-half pair structure."
        )
    radius = cos.square() + sin.square()
    floor = torch.finfo(torch.float32).tiny
    if bool((radius.to(torch.float32) <= floor).any().item()):
        raise Lfm2CpcfLayerCompilerError("Cannot invert a zero-radius RoPE row.")
    numerator = value * cos.unsqueeze(1) - _rotate_half(value) * sin.unsqueeze(1)
    result = numerator / radius.unsqueeze(1)
    if not bool(torch.isfinite(result).all().item()):
        raise Lfm2CpcfLayerCompilerError("Inverse RoPE produced a non-finite row.")
    return result.to(dtype=value.dtype).contiguous()


def _runtime_rms_norm(value: Any, weight: Any, epsilon: float) -> Any:
    torch = _torch()
    input_dtype = value.dtype
    direction = value.to(torch.float32)
    variance = direction.square().mean(dim=-1, keepdim=True)
    direction = direction * torch.rsqrt(variance + epsilon)
    return weight.to(dtype=input_dtype) * direction.to(dtype=input_dtype)


def _rmsnorm_preimage(
    normalized: Any,
    weight: Any,
    *,
    epsilon: float,
) -> Any:
    """Choose a deterministic finite-precision preimage of an RMSNorm row."""

    torch = _torch()
    values = _finite_tensor(normalized, "normalized", dimensions=4)
    norm_weight = _finite_tensor(weight, "weight", dimensions=1)
    width = int(values.shape[-1])
    if tuple(norm_weight.shape) != (width,):
        raise Lfm2CpcfLayerCompilerError(
            "RMSNorm weight width differs from normalized rows."
        )
    weight64 = norm_weight.to(torch.float64)
    if bool((weight64.abs() <= torch.finfo(torch.float64).tiny).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Cannot recover an RMSNorm preimage through a zero learned weight."
        )
    rows = values.reshape(-1, width)
    direction = rows.to(torch.float64) / weight64.reshape(1, width)
    mean_square = direction.square().mean(dim=-1, keepdim=True)
    denominator = 1.0 - mean_square
    ideal = torch.sqrt(
        torch.as_tensor(epsilon, dtype=torch.float64)
        / denominator.clamp_min(torch.finfo(torch.float64).eps)
    )
    ideal = torch.where(
        denominator > torch.finfo(torch.float32).eps,
        ideal,
        torch.full_like(ideal, 128.0),
    ).clamp(min=2.0**-8, max=2.0**16)
    scales = torch.stack(
        (
            ideal,
            torch.ones_like(ideal),
            torch.full_like(ideal, 16.0),
            torch.full_like(ideal, 128.0),
            torch.full_like(ideal, 1024.0),
        ),
        dim=0,
    )
    candidates = (direction.unsqueeze(0) * scales).to(dtype=values.dtype)
    predictions = _runtime_rms_norm(
        candidates,
        norm_weight.reshape(1, 1, width),
        epsilon,
    )
    targets = rows.unsqueeze(0)
    errors = (
        (predictions.to(torch.float64) - targets.to(torch.float64))
        .square()
        .mean(dim=-1)
    )
    choice = errors.argmin(dim=0)
    row_indices = torch.arange(int(rows.shape[0]))
    selected = candidates[choice, row_indices, :]
    return selected.reshape_as(values).contiguous()


def recover_lfm2_full_key_linear_history(
    *,
    parent_key_history: Any,
    parent_key_norm_weight: Any,
    full_cosine: Any,
    full_sine: Any,
    norm_epsilon: float,
    cached_current_key_linear: Any | None = None,
    cached_current_normalized_pre_rope: Any | None = None,
) -> Any:
    """Recover full-prefix pre-norm K rows from a sealed post-RoPE history.

    Decode traces contain exact post-RoPE K history but only current-step
    projection rows.  RoPE is inverted analytically and RMSNorm is assigned a
    deterministic finite-precision preimage.  Exact cached current rows then
    replace the reconstructed suffix.
    """

    torch = _torch()
    epsilon = _nonnegative_float(norm_epsilon, "norm_epsilon")
    if epsilon == 0.0:
        raise Lfm2CpcfLayerCompilerError("norm_epsilon must be positive.")
    history = _finite_tensor(
        parent_key_history,
        "parent_key_history",
        dimensions=4,
    )
    normalized = invert_lfm2_rope(history, full_cosine, full_sine)
    if cached_current_normalized_pre_rope is not None:
        cached_normalized = _finite_tensor(
            cached_current_normalized_pre_rope,
            "cached_current_normalized_pre_rope",
            dimensions=4,
        )
        current = int(cached_normalized.shape[2])
        if (
            tuple(cached_normalized.shape[:2]) != tuple(normalized.shape[:2])
            or int(cached_normalized.shape[3]) != int(normalized.shape[3])
            or current > int(normalized.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current normalized K rows do not align to K history."
            )
        tolerance = (
            4.0 * torch.finfo(history.dtype).eps
            if history.dtype in {torch.float16, torch.bfloat16}
            else 2e-5
        )
        if not bool(
            torch.allclose(
                normalized[:, :, -current:, :].to(torch.float32),
                cached_normalized.to(torch.float32),
                rtol=tolerance,
                atol=tolerance,
            )
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Inverse RoPE disagrees with cached current pre-RoPE K rows."
            )
    linear = _rmsnorm_preimage(
        normalized,
        parent_key_norm_weight,
        epsilon=epsilon,
    )
    if cached_current_key_linear is not None:
        cached_linear = _finite_tensor(
            cached_current_key_linear,
            "cached_current_key_linear",
            dimensions=4,
        )
        current = int(cached_linear.shape[2])
        if (
            tuple(cached_linear.shape[:2]) != tuple(linear.shape[:2])
            or int(cached_linear.shape[3]) != int(linear.shape[3])
            or current > int(linear.shape[2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Cached current linear K rows do not align to K history."
            )
        linear = linear.clone()
        linear[:, :, -current:, :] = cached_linear.to(dtype=linear.dtype)
    reconstructed_normalized = _runtime_rms_norm(
        linear,
        _finite_tensor(
            parent_key_norm_weight,
            "parent_key_norm_weight",
            dimensions=1,
        ).reshape(1, 1, 1, -1),
        epsilon,
    )
    reconstructed = reconstructed_normalized * full_cosine.unsqueeze(1) + _rotate_half(
        reconstructed_normalized
    ) * full_sine.unsqueeze(1)
    tolerance = (
        8.0 * torch.finfo(history.dtype).eps
        if history.dtype in {torch.float16, torch.bfloat16}
        else 5e-5
    )
    if not bool(
        torch.allclose(
            reconstructed.to(torch.float32),
            history.to(torch.float32),
            rtol=tolerance,
            atol=tolerance,
        )
    ):
        maximum = float(
            (reconstructed.to(torch.float32) - history.to(torch.float32))
            .abs()
            .max()
            .item()
        )
        raise Lfm2CpcfLayerCompilerError(
            "Recovered full K-linear history does not reproduce the sealed "
            f"post-RoPE history (maximum absolute error {maximum:.6g})."
        )
    return linear


def _same_rope(
    source_model: Any,
    target_model: Any,
    phase: Lfm2CpcfTargetPhase,
    *,
    dtype: Any,
) -> tuple[Any, Any]:
    torch = _torch()
    source_cos, source_sin = _rope_terms(
        source_model,
        phase.position_ids,
        dtype=dtype,
    )
    target_cos, target_sin = _rope_terms(
        target_model,
        phase.position_ids,
        dtype=dtype,
    )
    tolerance = (
        2.0 * torch.finfo(dtype).eps
        if dtype in {torch.float16, torch.bfloat16}
        else 1e-6
    )
    if not (
        torch.allclose(source_cos, target_cos, rtol=tolerance, atol=tolerance)
        and torch.allclose(source_sin, target_sin, rtol=tolerance, atol=tolerance)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source and target LFM2 RoPE differ; this preserved-head-dimension "
            "compiler requires an explicit rotary bridge."
        )
    return source_cos, source_sin


def _copy_attention_norms(
    source_attention: Any, target_attention: Any
) -> Mapping[str, Any]:
    torch = _torch()
    source = {
        "q_layernorm.weight": _weight(
            source_attention.q_layernorm,
            "source_attention.q_layernorm",
            dimensions=1,
        ),
        "k_layernorm.weight": _weight(
            source_attention.k_layernorm,
            "source_attention.k_layernorm",
            dimensions=1,
        ),
    }
    target = {
        "q_layernorm.weight": getattr(
            getattr(target_attention, "q_layernorm", None),
            "weight",
            None,
        ),
        "k_layernorm.weight": getattr(
            getattr(target_attention, "k_layernorm", None),
            "weight",
            None,
        ),
    }
    if any(not isinstance(value, torch.Tensor) for value in target.values()):
        raise Lfm2CpcfLayerCompilerError("Target attention lacks Q/K RMSNorm weights.")
    if any(tuple(target[name].shape) != tuple(source[name].shape) for name in source):
        raise Lfm2CpcfLayerCompilerError(
            "Source/target attention norm widths differ; route this layer "
            "through the attention-dimension compiler."
        )
    identities = {name: id(value) for name, value in target.items()}
    gradients = {
        name: (
            bool(value.requires_grad),
            None if value.grad is None else id(value.grad),
            None if value.grad is None else _tensor_sha256(value.grad),
        )
        for name, value in target.items()
    }
    with torch.no_grad():
        for name, value in target.items():
            value.copy_(source[name].to(device=value.device, dtype=value.dtype))
    body = {
        "scope": "attention_qk_norm_transport",
        "weights": {
            name: _tensor_sha256(value) for name, value in sorted(target.items())
        },
        "parameter_identities_preserved": all(
            id(target[name]) == identities[name] for name in target
        ),
        "gradient_state_preserved": all(
            (
                bool(target[name].requires_grad),
                None if target[name].grad is None else id(target[name].grad),
                None
                if target[name].grad is None
                else _tensor_sha256(target[name].grad),
            )
            == gradients[name]
            for name in target
        ),
        "source_derived": True,
        "training_steps": 0,
    }
    if (
        not body["parameter_identities_preserved"]
        or not body["gradient_state_preserved"]
    ):
        raise Lfm2CpcfLayerInstallError(
            "Attention Q/K norm transport changed parameter or gradient identity."
        )
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _attention_key_mask(additive_mask: Any) -> Any:
    torch = _torch()
    value = _finite_tensor(additive_mask, "attention.additive_mask", dimensions=4)
    if bool((value > 0).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            "Parent additive attention mask contains a positive score."
        )
    allowed = value == 0
    if int(allowed.shape[1]) > 1:
        first = allowed[:, :1, :, :]
        if not bool(torch.equal(allowed, first.expand_as(allowed))):
            raise Lfm2CpcfLayerCompilerError(
                "Head-specific parent masks cannot be reduced to one GQA mask."
            )
    return allowed[:, 0, :, :].contiguous()


def _validated_cached_attention_probabilities(
    *,
    probabilities: Any,
    context_from_probabilities: Any,
    parent_context: Any,
    value_history: Any,
    key_mask: Any,
    absolute_query_start: int,
    name: str,
) -> Any:
    """Cross-check direct cached softmax evidence and its executed context."""

    torch = _torch()
    observed = _finite_tensor(
        probabilities,
        f"{name}.probabilities",
        dimensions=4,
    )
    cached_context = _finite_tensor(
        context_from_probabilities,
        f"{name}.context_from_probabilities",
        dimensions=4,
    )
    context = _finite_tensor(
        parent_context,
        f"{name}.context",
        dimensions=4,
    )
    values = _finite_tensor(
        value_history,
        f"{name}.value_history",
        dimensions=4,
    )
    if not isinstance(key_mask, torch.Tensor) or key_mask.dtype is not torch.bool:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} key mask must be a dense boolean tensor."
        )
    batch, query_heads, query_rows, key_rows = map(int, observed.shape)
    key_value_heads = int(values.shape[1])
    head_dim = int(values.shape[3])
    if (
        absolute_query_start < 0
        or query_heads <= 0
        or key_value_heads <= 0
        or query_heads % key_value_heads
        or tuple(key_mask.shape) != (batch, query_rows, key_rows)
        or tuple(values.shape) != (batch, key_value_heads, key_rows, head_dim)
        or tuple(context.shape)
        != (batch, query_heads, query_rows, head_dim)
        or tuple(cached_context.shape) != tuple(context.shape)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached probabilities, V history, mask, and contexts "
            "are not one aligned attention execution."
        )
    absolute_queries = torch.arange(
        absolute_query_start,
        absolute_query_start + query_rows,
        dtype=torch.long,
    ).reshape(1, query_rows, 1)
    absolute_keys = torch.arange(key_rows, dtype=torch.long).reshape(
        1,
        1,
        key_rows,
    )
    if bool((key_mask & (absolute_keys > absolute_queries)).any().item()):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached mask permits a future causal key."
        )
    expanded_mask = key_mask[:, None, :, :].expand_as(observed)
    if (
        bool((observed < 0).any().item())
        or not bool(key_mask.any(dim=-1).all().item())
        or not bool(torch.equal(
            observed.masked_fill(expanded_mask, 0),
            torch.zeros_like(observed),
        ))
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached probabilities are negative, mask-leaking, or "
            "contain a query row with no legal key."
        )
    probability_tolerance = max(
        2e-3,
        4.0 * float(torch.finfo(observed.dtype).eps),
    )
    row_sums = observed.to(torch.float32).sum(dim=-1)
    if not bool(
        torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            rtol=probability_tolerance,
            atol=probability_tolerance,
        )
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached probability rows do not sum to one."
        )
    repeated_values = values.repeat_interleave(
        query_heads // key_value_heads,
        dim=1,
    )
    reconstructed = torch.matmul(
        observed.to(torch.float32),
        repeated_values.to(torch.float32),
    )
    if not (
        torch.allclose(
            reconstructed,
            cached_context.to(torch.float32),
            rtol=2e-2,
            atol=2e-3,
        )
        and torch.allclose(
            cached_context.to(torch.float32),
            context.to(torch.float32),
            rtol=2e-2,
            atol=2e-3,
        )
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached probability replay differs from its authenticated "
            "context_from_probabilities or executed context."
        )
    return observed


def _equal_causal_phase_sqrt_weights(
    phases: Sequence[Lfm2CpcfTargetPhase],
    *,
    required_split: str = "fit",
) -> Mapping[tuple[int, str], Any]:
    """Give every phase in one required split equal least-squares mass.

    The weights retain a global mean square of one, so changing from uniform
    rows does not silently rescale the ridge term.
    """

    torch = _torch()
    if not isinstance(required_split, str) or not required_split:
        raise Lfm2CpcfLayerCompilerError(
            "Attention phase weighting requires one nonempty split name."
        )
    selected_phases = tuple(phase for phase in phases if phase.split == required_split)
    if not selected_phases:
        raise Lfm2CpcfLayerCompilerError(
            f"Attention phase weighting requires at least one {required_split!r} phase."
        )
    total_rows = sum(
        phase.batch_size * phase.sampled_suffix_tokens for phase in selected_phases
    )
    phase_count = len(selected_phases)
    result = {}
    for phase in selected_phases:
        rows = phase.batch_size * phase.sampled_suffix_tokens
        row_weight = total_rows / (phase_count * rows)
        result[phase.identity] = torch.full(
            (phase.batch_size, phase.sampled_suffix_tokens),
            math.sqrt(row_weight),
            dtype=torch.float64,
        )
    observed_mass = sum(float(value.square().sum().item()) for value in result.values())
    if not math.isclose(
        observed_mass,
        float(total_rows),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Equal causal phase weights do not preserve global row mass."
        )
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class _PreparedAttentionOutputTransport:
    """One content-authenticated parent-O/boundary-chart composition."""

    source_width: int
    query_heads: int
    head_dim: int
    target_width: int
    source_output_weight_dtype: Any
    source_output_blocks: Any
    target_encoder: Any
    target_output_blocks: Any
    seal_sha256: str


def _attention_output_transport_seal_payload(
    *,
    source_width: int,
    query_heads: int,
    head_dim: int,
    target_width: int,
    source_output_weight_dtype: Any,
    source_output_blocks: Any,
    target_encoder: Any,
    target_output_blocks: Any,
) -> dict[str, Any]:
    """Return the complete canonical identity of one prepared transport."""

    return {
        "head_dim": head_dim,
        "query_heads": query_heads,
        "source_output_blocks_sha256": _tensor_sha256(source_output_blocks),
        "source_output_weight_dtype": str(source_output_weight_dtype).removeprefix(
            "torch."
        ),
        "source_width": source_width,
        "target_encoder_sha256": _tensor_sha256(target_encoder),
        "target_output_blocks_sha256": _tensor_sha256(target_output_blocks),
        "target_width": target_width,
    }


def _verify_attention_output_transport(
    transport: Any,
    *,
    name: str,
) -> _PreparedAttentionOutputTransport:
    """Fail closed unless every prepared tensor still matches its seal."""

    torch = _torch()
    if type(transport) is not _PreparedAttentionOutputTransport:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} must be an exact prepared O transport."
        )
    source_width = _nonnegative_int(
        transport.source_width,
        f"{name}.source_width",
    )
    query_heads = _nonnegative_int(
        transport.query_heads,
        f"{name}.query_heads",
    )
    head_dim = _nonnegative_int(
        transport.head_dim,
        f"{name}.head_dim",
    )
    target_width = _nonnegative_int(
        transport.target_width,
        f"{name}.target_width",
    )
    if min(source_width, query_heads, head_dim, target_width) == 0:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} topology dimensions must be positive."
        )
    try:
        dtype_probe = torch.empty(
            (),
            dtype=transport.source_output_weight_dtype,
        )
    except (TypeError, RuntimeError) as exc:
        raise Lfm2CpcfLayerCompilerError(
            f"{name}.source_output_weight_dtype is invalid."
        ) from exc
    if not torch.is_floating_point(dtype_probe) or torch.is_complex(dtype_probe):
        raise Lfm2CpcfLayerCompilerError(
            f"{name}.source_output_weight_dtype must be real floating."
        )

    tensor_contract = (
        (
            "source_output_blocks",
            transport.source_output_blocks,
            (source_width, query_heads, head_dim),
        ),
        (
            "target_encoder",
            transport.target_encoder,
            (source_width, target_width),
        ),
        (
            "target_output_blocks",
            transport.target_output_blocks,
            (query_heads, head_dim, target_width),
        ),
    )
    storage_pointers = []
    for field_name, value, expected_shape in tensor_contract:
        if (
            not isinstance(value, torch.Tensor)
            or value.layout != torch.strided
            or value.device.type != "cpu"
            or value.dtype != torch.float64
            or not value.is_contiguous()
            or tuple(value.shape) != expected_shape
            or not bool(torch.isfinite(value).all().item())
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"{name}.{field_name} violates its sealed dense FP64 tensor contract."
            )
        storage_pointers.append(value.untyped_storage().data_ptr())
    if len(set(storage_pointers)) != len(storage_pointers):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} prepared tensors must own disjoint storage."
        )

    expected_seal = _sha(
        _attention_output_transport_seal_payload(
            source_width=source_width,
            query_heads=query_heads,
            head_dim=head_dim,
            target_width=target_width,
            source_output_weight_dtype=(transport.source_output_weight_dtype),
            source_output_blocks=transport.source_output_blocks,
            target_encoder=transport.target_encoder,
            target_output_blocks=transport.target_output_blocks,
        )
    )
    if _digest(transport.seal_sha256, f"{name}.seal_sha256") != expected_seal:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} content differs from its prepared O transport seal."
        )
    return transport


def _prepare_attention_output_transport_from_tensors(
    *,
    source_output_weight: Any,
    target_encoder: Any,
    target_output_blocks: Any | None = None,
    query_heads: int | None = None,
    head_dim: int | None = None,
    name: str,
) -> _PreparedAttentionOutputTransport:
    """Copy, compose, validate, and seal invariant O/chart tensors once."""

    torch = _torch()
    output_weight_native = _finite_tensor(
        source_output_weight,
        f"{name}.source_output_weight",
        dimensions=2,
    )
    encoder_native = _finite_tensor(
        target_encoder,
        f"{name}.target_encoder",
        dimensions=2,
    )
    supplied_blocks = (
        None
        if target_output_blocks is None
        else _finite_tensor(
            target_output_blocks,
            f"{name}.target_output_blocks",
            dimensions=3,
        )
    )
    source_width, context_width = map(int, output_weight_native.shape)
    if int(encoder_native.shape[0]) != source_width:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} source O weight and target encoder are misaligned."
        )
    target_width = int(encoder_native.shape[1])

    if supplied_blocks is not None:
        supplied_query_heads, supplied_head_dim, supplied_target_width = map(
            int, supplied_blocks.shape
        )
        if query_heads is None:
            query_heads = supplied_query_heads
        if head_dim is None:
            head_dim = supplied_head_dim
        if (
            supplied_query_heads != query_heads
            or supplied_head_dim != head_dim
            or supplied_target_width != target_width
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"{name} target O blocks do not match the declared topology."
            )
    if query_heads is None or head_dim is None:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} requires query-head and head-dimension topology."
        )
    query_heads = _nonnegative_int(query_heads, f"{name}.query_heads")
    head_dim = _nonnegative_int(head_dim, f"{name}.head_dim")
    if query_heads == 0 or head_dim == 0 or context_width != query_heads * head_dim:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} source O weight does not match its attention topology."
        )

    # Clone the two static inputs before sealing so subsequent mutation of the
    # caller's module/chart cannot change the prepared transport.
    output_weight = output_weight_native.to(torch.float64).clone().contiguous()
    encoder = encoder_native.to(torch.float64).clone().contiguous()
    source_blocks = output_weight.reshape(
        source_width,
        query_heads,
        head_dim,
    ).contiguous()
    expected_blocks = torch.einsum(
        "ohd,ot->hdt",
        source_blocks,
        encoder,
    ).contiguous()
    if not bool(torch.isfinite(expected_blocks).all().item()):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} source O/chart composition produced NaN or infinity."
        )
    if supplied_blocks is None:
        blocks = expected_blocks
    else:
        blocks = supplied_blocks.to(torch.float64).clone().contiguous()
        block_scale = max(
            float(expected_blocks.abs().max().item()),
            torch.finfo(torch.float64).tiny,
        )
        block_error = float((blocks - expected_blocks).abs().max().item())
        if (
            not math.isfinite(block_scale)
            or not math.isfinite(block_error)
            or block_error > (32.0 * torch.finfo(torch.float64).eps * block_scale)
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"{name} target O blocks do not match the source O/chart composition."
            )

    seal_sha256 = _sha(
        _attention_output_transport_seal_payload(
            source_width=source_width,
            query_heads=query_heads,
            head_dim=head_dim,
            target_width=target_width,
            source_output_weight_dtype=output_weight_native.dtype,
            source_output_blocks=source_blocks,
            target_encoder=encoder,
            target_output_blocks=blocks,
        )
    )
    return _verify_attention_output_transport(
        _PreparedAttentionOutputTransport(
            source_width=source_width,
            query_heads=query_heads,
            head_dim=head_dim,
            target_width=target_width,
            source_output_weight_dtype=output_weight_native.dtype,
            source_output_blocks=source_blocks,
            target_encoder=encoder,
            target_output_blocks=blocks,
            seal_sha256=seal_sha256,
        ),
        name=name,
    )


def _prepare_attention_output_transport(
    *,
    source_attention: Any,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> _PreparedAttentionOutputTransport:
    """Prepare the production parent-O/chart transport outside phase loops."""

    head_dim = _nonnegative_int(
        int(source_attention.head_dim),
        "source_attention.head_dim",
    )
    if head_dim == 0:
        raise Lfm2CpcfLayerCompilerError("source_attention.head_dim must be positive.")
    output_weight = _weight(
        source_attention.out_proj,
        "source_attention.out_proj",
        dimensions=2,
    )
    context_width = int(output_weight.shape[1])
    if context_width % head_dim:
        raise Lfm2CpcfLayerCompilerError(
            "Parent O topology does not divide into complete attention heads."
        )
    prepared = _prepare_attention_output_transport_from_tensors(
        source_output_weight=output_weight,
        target_encoder=chart.post_operator.mapping.encoder,
        query_heads=context_width // head_dim,
        head_dim=head_dim,
        name="parent attention O transport",
    )
    if prepared.source_width != chart.source_width:
        raise Lfm2CpcfLayerCompilerError(
            "Parent O topology does not compose with the post-operator chart."
        )
    return prepared


def _attention_output_transport_complexity_receipt(
    *,
    transport: _PreparedAttentionOutputTransport,
    phase_count: int,
) -> Mapping[str, Any]:
    """Report exact static-composition work removed by prepared transport."""

    phase_count = _nonnegative_int(phase_count, "phase_count")
    transport = _verify_attention_output_transport(
        transport,
        name="attention output transport complexity receipt",
    )
    composition_terms = (
        transport.source_width
        * transport.query_heads
        * transport.head_dim
        * transport.target_width
    )
    body = {
        "schema": "neural-forge-lfm2-attention-o-transport-runtime-v1",
        "phase_count": phase_count,
        "source_width": transport.source_width,
        "query_heads": transport.query_heads,
        "head_dim": transport.head_dim,
        "target_width": transport.target_width,
        "source_output_weight_dtype": str(
            transport.source_output_weight_dtype
        ).removeprefix("torch."),
        "composition_scalar_terms": composition_terms,
        "composition_flops": 2 * composition_terms,
        "legacy_composition_count": phase_count + 1,
        "prepared_composition_count": 1,
        "avoided_composition_count": phase_count,
        "avoided_composition_flops": (2 * composition_terms * phase_count),
        "avoided_phase_fp64_weight_materialization_bytes": (
            phase_count
            * transport.source_width
            * transport.query_heads
            * transport.head_dim
            * 8
        ),
        "transport_seal_sha256": transport.seal_sha256,
    }
    return MappingProxyType(
        {
            **body,
            "report_sha256": _sha(body),
        }
    )


def _attention_runtime_complexity_receipt(
    *,
    output_transport_receipt: Mapping[str, Any],
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    operator_artifact: Lfm2CompiledStructuralAttentionFold,
) -> Mapping[str, Any]:
    """Bind both structural-attention runtime shortcuts to actual inputs."""

    torch = _torch()
    transport_receipt = dict(output_transport_receipt)
    transport_digest = transport_receipt.pop("report_sha256", None)
    if _digest(
        transport_digest,
        "attention output transport receipt report_sha256",
    ) != _sha(transport_receipt):
        raise Lfm2CpcfLayerCompilerError(
            "Attention output transport receipt authentication failed."
        )
    topology = operator_artifact.topology
    cluster_report = operator_artifact.cluster_plan.report
    if (
        transport_receipt.get("source_width") != topology.source_hidden_size
        or transport_receipt.get("query_heads") != topology.source_query_heads
        or transport_receipt.get("head_dim") != topology.head_dim
        or transport_receipt.get("target_width") != topology.target_output_size
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Prepared O transport receipt differs from compiled attention topology."
        )

    fit_observation_rows = 0
    contribution_width = None
    for index, case in enumerate(cases):
        query = case.parent_query
        contribution = case.parent_contribution
        if (
            not isinstance(query, torch.Tensor)
            or query.ndim != 4
            or int(query.shape[1]) != topology.source_query_heads
            or int(query.shape[3]) != topology.head_dim
            or not isinstance(contribution, torch.Tensor)
            or contribution.ndim != 4
            or tuple(contribution.shape[:3]) != tuple(query.shape[:3])
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Attention case {index} does not match its runtime receipt topology."
            )
        current_width = int(contribution.shape[3])
        if contribution_width is None:
            contribution_width = current_width
        elif current_width != contribution_width:
            raise Lfm2CpcfLayerCompilerError(
                "Attention contribution width changed across fit cases."
            )
        fit_observation_rows += int(query.shape[0]) * int(query.shape[2])
    if not cases or contribution_width != topology.target_output_size:
        raise Lfm2CpcfLayerCompilerError(
            "Attention runtime receipt requires nonempty aligned fit cases."
        )

    source_offsets = topology.source_query_heads_per_group
    target_offsets = topology.target_query_heads_per_group
    equal_offsets = source_offsets == target_offsets
    identity_strategy = "deterministic-identity-real-medoid"
    short_circuit_executed = (
        equal_offsets
        and cluster_report.query_offset_cluster_strategy == identity_strategy
        and cluster_report.query_offset_medoids == tuple(range(source_offsets))
        and cluster_report.query_offset_assignment == tuple(range(source_offsets))
    )
    if equal_offsets != short_circuit_executed:
        raise Lfm2CpcfLayerCompilerError(
            "Attention offset runtime receipt differs from the authenticated "
            "activation-cluster result."
        )

    query_signature_elements = (
        topology.source_query_heads * fit_observation_rows * topology.head_dim
    )
    contribution_signature_elements = (
        topology.source_query_heads * fit_observation_rows * contribution_width
    )
    offset_feature_elements = (
        2 * query_signature_elements + contribution_signature_elements
    )
    avoided_feature_elements = offset_feature_elements if short_circuit_executed else 0
    offset_body = {
        "source_query_heads": topology.source_query_heads,
        "source_key_value_heads": topology.source_key_value_heads,
        "target_query_heads": topology.target_query_heads,
        "target_key_value_heads": topology.target_key_value_heads,
        "source_query_offsets_per_kv": source_offsets,
        "target_query_offsets_per_kv": target_offsets,
        "head_dim": topology.head_dim,
        "target_output_width": contribution_width,
        "fit_case_count": len(cases),
        "fit_observation_rows": fit_observation_rows,
        "query_signature_elements": query_signature_elements,
        "contribution_signature_elements": (contribution_signature_elements),
        "legacy_offset_feature_elements": offset_feature_elements,
        "identity_short_circuit_executed": short_circuit_executed,
        "query_offset_cluster_strategy": (cluster_report.query_offset_cluster_strategy),
        "avoided_offset_feature_elements": avoided_feature_elements,
        "avoided_offset_feature_bytes": avoided_feature_elements * 8,
        "avoided_minimum_live_bytes_at_legacy_cat": (avoided_feature_elements * 8 * 3),
        "identity_result_bytes": (
            3 * source_offsets * 8 if short_circuit_executed else 0
        ),
    }
    body = {
        "schema": "neural-forge-lfm2-attention-runtime-complexity-v1",
        "output_transport": dict(output_transport_receipt),
        "equal_offset_identity_short_circuit": {
            **offset_body,
            "report_sha256": _sha(offset_body),
        },
    }
    return MappingProxyType(
        {
            **body,
            "report_sha256": _sha(body),
        }
    )


def _attention_output_blocks_in_target_space(
    *,
    source_attention: Any,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> Any:
    """Compose every parent O head block with the predictive boundary chart."""

    return _prepare_attention_output_transport(
        source_attention=source_attention,
        chart=chart,
    ).target_output_blocks


def _encoded_parent_attention_contribution(
    *,
    parent_context: Any,
    cached_source_output: Any,
    cached_target_output: Any,
    name: str,
    source_output_weight: Any | None = None,
    target_encoder: Any | None = None,
    target_output_blocks: Any | None = None,
    prepared_transport: _PreparedAttentionOutputTransport | None = None,
) -> Any:
    """Return the directional target-space residual write of every parent head."""

    torch = _torch()
    context_native = _finite_tensor(
        parent_context,
        f"{name}.context",
        dimensions=4,
    )
    source_output_native = _finite_tensor(
        cached_source_output,
        f"{name}.source_output",
        dimensions=3,
    )
    target_output = _finite_tensor(
        cached_target_output,
        f"{name}.target_output",
        dimensions=3,
    )
    raw_transport_values = (
        source_output_weight,
        target_encoder,
        target_output_blocks,
    )
    if prepared_transport is None:
        if any(value is None for value in raw_transport_values):
            raise Lfm2CpcfLayerCompilerError(
                f"{name} requires either one prepared O transport or all raw "
                "O/chart tensors."
            )
        prepared_transport = _prepare_attention_output_transport_from_tensors(
            source_output_weight=source_output_weight,
            target_encoder=target_encoder,
            target_output_blocks=target_output_blocks,
            name=name,
        )
    elif type(prepared_transport) is not _PreparedAttentionOutputTransport or any(
        value is not None for value in raw_transport_values
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} prepared O transport must be supplied alone."
        )
    prepared_transport = _verify_attention_output_transport(
        prepared_transport,
        name=f"{name}.prepared_transport",
    )
    encoder = prepared_transport.target_encoder
    blocks = prepared_transport.target_output_blocks
    batch, query_heads, query_rows, head_dim = map(
        int,
        context_native.shape,
    )
    source_width = prepared_transport.source_width
    context_width = prepared_transport.query_heads * prepared_transport.head_dim
    target_width = prepared_transport.target_width
    if (
        context_width != query_heads * head_dim
        or query_heads != prepared_transport.query_heads
        or head_dim != prepared_transport.head_dim
        or tuple(source_output_native.shape) != (batch, query_rows, source_width)
        or tuple(target_output.shape) != (batch, query_rows, target_width)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} parent context, O output, and target O blocks are misaligned."
        )
    context = context_native.to(torch.float64)
    contribution = torch.einsum(
        "bhqd,hdt->bhqt",
        context,
        blocks,
    ).contiguous()
    source_prediction = torch.einsum(
        "bhqd,ohd->bqo",
        context,
        prepared_transport.source_output_blocks,
    ).contiguous()
    source_reference = source_output_native.to(torch.float64)
    encoded_source_reference = torch.matmul(
        source_reference,
        encoder,
    ).contiguous()
    target_reference = target_output.to(torch.float64)
    if not (
        bool(torch.isfinite(contribution).all().item())
        and bool(torch.isfinite(source_prediction).all().item())
        and bool(torch.isfinite(encoded_source_reference).all().item())
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} O authentication produced a non-finite prediction."
        )
    target_reference_scale = max(
        float(encoded_source_reference.abs().max().item()),
        torch.finfo(torch.float64).tiny,
    )
    target_reference_error = float(
        (target_reference - encoded_source_reference).abs().max().item()
    )
    if (
        not math.isfinite(target_reference_scale)
        or not math.isfinite(target_reference_error)
        or target_reference_error
        > (32.0 * torch.finfo(torch.float64).eps * target_reference_scale)
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached target O output is not the FP64 chart encoding "
            "of the cached source O output."
        )

    # Cached low-precision O output is rounded after an FP32-accumulating
    # projection.  Authenticate it against that arithmetic contract instead
    # of allowing an arbitrary multiple of the BF16/FP16 epsilon.  The bound
    # accounts for dot-product accumulation, optional TF32 FP32 products, and
    # final output materialization, then transports the source-space bound
    # through the absolute chart map.
    dot_terms = context_width
    accumulator_dtype = (
        torch.float64
        if (
            context_native.dtype == torch.float64
            and prepared_transport.source_output_weight_dtype == torch.float64
        )
        else torch.float32
    )
    accumulator_unit = torch.finfo(accumulator_dtype).eps / 2.0
    operation_count = 2 * dot_terms
    accumulated = operation_count * accumulator_unit
    accumulation_gamma = (
        accumulated / (1.0 - accumulated) if accumulated < 1.0 else float("inf")
    )
    if not math.isfinite(accumulation_gamma):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} O projection is too wide for a finite accumulation bound."
        )
    tf32_product_unit = (
        2.0**-10
        if (
            context_native.dtype == torch.float32
            or prepared_transport.source_output_weight_dtype == torch.float32
        )
        else 0.0
    )
    output_unit = torch.finfo(source_output_native.dtype).eps / 2.0
    absolute_source_output_blocks = prepared_transport.source_output_blocks.abs()
    absolute_target_encoder = prepared_transport.target_encoder.abs()
    absolute_term_sum = torch.einsum(
        "bhqd,ohd->bqo",
        context.abs(),
        absolute_source_output_blocks,
    )
    if not bool(torch.isfinite(absolute_term_sum).all().item()):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} O projection produced a non-finite roundoff bound."
        )
    source_roundoff_bound = 2.0 * (
        (accumulation_gamma + tf32_product_unit) * absolute_term_sum
        + output_unit * source_prediction.abs()
    )
    source_floor = (
        torch.finfo(source_output_native.dtype).tiny
        * torch.finfo(source_output_native.dtype).eps
    )
    source_roundoff_bound = source_roundoff_bound + source_floor
    if not bool(torch.isfinite(source_roundoff_bound).all().item()):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} O projection produced a non-finite roundoff bound."
        )
    source_error_rmse = float(
        (source_prediction - source_reference).square().mean().sqrt().item()
    )
    source_bound_rmse = float(source_roundoff_bound.square().mean().sqrt().item())
    if not math.isfinite(source_error_rmse) or not math.isfinite(source_bound_rmse):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} O projection produced a non-finite error comparison."
        )
    if source_error_rmse > source_bound_rmse:
        raise Lfm2CpcfLayerCompilerError(
            f"{name} cached source O output exceeds its accumulation-aware "
            f"rounding bound ({source_error_rmse} > {source_bound_rmse})."
        )

    # The chart and O map are linear, so head-local writes must sum to the
    # same target-space write as the cached complete parent O output.
    reconstructed = contribution.sum(dim=1)
    target_roundoff_bound = torch.matmul(
        source_roundoff_bound,
        absolute_target_encoder,
    )
    target_association_scale = torch.matmul(
        absolute_term_sum,
        absolute_target_encoder,
    )
    target_roundoff_bound = target_roundoff_bound + (
        64.0 * torch.finfo(torch.float64).eps * target_association_scale
    )
    if not (
        bool(torch.isfinite(reconstructed).all().item())
        and bool(torch.isfinite(target_roundoff_bound).all().item())
        and bool(torch.isfinite(target_association_scale).all().item())
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} target-space O write produced a non-finite error bound."
        )
    target_error_rmse = float(
        (reconstructed - target_reference).square().mean().sqrt().item()
    )
    target_bound_rmse = float(target_roundoff_bound.square().mean().sqrt().item())
    if (
        int(contribution.shape[-1]) <= 0
        or not math.isfinite(target_error_rmse)
        or not math.isfinite(target_bound_rmse)
        or target_error_rmse > target_bound_rmse
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"{name} target-space O contribution does not reconstruct the "
            "FP64-encoded cached complete O write within its accumulation-aware "
            f"bound ({target_error_rmse} > {target_bound_rmse})."
        )
    return contribution


def _attention_trajectory_sha256s(
    input_bank: Lfm2CpcfPhaseStateBank,
    *,
    required_split: str,
) -> tuple[str, ...]:
    """Seal one receipt per packed causal trajectory in attention-case order."""

    if required_split not in {"fit", "selection", "validation"}:
        raise Lfm2CpcfLayerCompilerError(
            "Attention trajectory receipts require one canonical split."
        )
    grouped: dict[tuple[int, str], list[Lfm2CpcfTargetPhase]] = {}
    for phase in input_bank.phases:
        if phase.split == required_split:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)
    receipts = tuple(
        _sha(
            {
                "schema": ("neural-forge-lfm2-cpcf-packed-attention-trajectory-v1"),
                "source_state_bank_sha256": input_bank.state_bank_sha256,
                "split": required_split,
                "batch_index": batch_index,
                "record_id": record_id,
                "ordered_phase_sha256": [phase.phase_sha256 for phase in trajectory],
            }
        )
        for (batch_index, record_id), trajectory in grouped.items()
    )
    if not receipts or len(set(receipts)) != len(receipts):
        raise Lfm2CpcfLayerCompilerError(
            "Attention trajectories must produce distinct authenticated receipts."
        )
    return receipts


def _attention_cases(
    *,
    source_model: Any,
    target_model: Any,
    source_layer: Any,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
    required_split: str = "fit",
) -> tuple[
    tuple[Lfm2CachedAttentionFoldCase, ...],
    Mapping[str, Any],
]:
    """Assemble one complete causal attention trajectory per fit record.

    The sealed prefill contains every raw K-linear row.  Each decode phase
    contains the one newly executed K-linear suffix plus the complete
    post-RoPE K/V history.  Concatenating those exact raw rows is both more
    faithful and more stable than assigning an artificial RMSNorm preimage to
    every historical key.  It also avoids fitting the same prompt prefix once
    per decode phase.
    """

    torch = _torch()
    source_attention = getattr(source_layer, "self_attn", None)
    if source_attention is None:
        raise Lfm2CpcfLayerCompilerError("Source attention layer lacks self_attn.")
    q_epsilon = _variance_epsilon(
        source_attention.q_layernorm,
        "source_attention.q_layernorm",
    )
    k_epsilon = _variance_epsilon(
        source_attention.k_layernorm,
        "source_attention.k_layernorm",
    )
    if not math.isclose(q_epsilon, k_epsilon, rel_tol=0.0, abs_tol=0.0):
        raise Lfm2CpcfLayerCompilerError("Source Q/K normalization epsilon differs.")
    prepared_output_transport = _prepare_attention_output_transport(
        source_attention=source_attention,
        chart=chart,
    )
    phase_sqrt_weights = _equal_causal_phase_sqrt_weights(
        input_bank.phases,
        required_split=required_split,
    )
    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    for phase in input_bank.phases:
        if phase.split == required_split:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    prepared_transport_use_count = 0
    for (batch_index, record_id), trajectory in grouped.items():
        query_linear_parts = []
        query_parts = []
        context_parts = []
        contribution_parts = []
        endpoint_parts = []
        row_sqrt_weight_parts = []
        key_linear_history = None
        key_history = None
        value_history = None
        key_mask_history = None
        probability_history = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        final_cos = None
        final_sin = None

        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill" if phase_index == 0 else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} skips "
                    f"or reorders causal phase {expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (previous_phase.full_prefix_tokens + phase.sampled_suffix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention trajectory {batch_index}:{record_id} changed "
                    "split or target prefix geometry."
                )

            tensors = _phase_evidence(
                phase,
                layer_index,
                "full_attention",
            )
            replay = preinstall_replay.replays[phase.identity]
            parent_q_linear = _finite_tensor(
                tensors["attention.q_linear"],
                f"phase {phase.identity!r} attention.q_linear",
                dimensions=4,
            )
            parent_k_linear = _finite_tensor(
                tensors["attention.k_linear"],
                f"phase {phase.identity!r} attention.k_linear",
                dimensions=4,
            )
            parent_query = _finite_tensor(
                tensors["attention.q"],
                f"phase {phase.identity!r} attention.q",
                dimensions=4,
            )
            parent_context = _finite_tensor(
                tensors["attention.context"],
                f"phase {phase.identity!r} attention.context",
                dimensions=4,
            )
            current_key = _finite_tensor(
                tensors["attention.k"],
                f"phase {phase.identity!r} attention.k",
                dimensions=4,
            )
            current_value = _finite_tensor(
                tensors["attention.v"],
                f"phase {phase.identity!r} attention.v",
                dimensions=4,
            )
            observed_key_history = _finite_tensor(
                tensors["attention.key_history"],
                f"phase {phase.identity!r} attention.key_history",
                dimensions=4,
            )
            observed_value_history = _finite_tensor(
                tensors["attention.value_history"],
                f"phase {phase.identity!r} attention.value_history",
                dimensions=4,
            )
            current = phase.sampled_suffix_tokens
            if (
                int(parent_q_linear.shape[2]) != current
                or int(parent_k_linear.shape[2]) != current
                or int(parent_query.shape[2]) != current
                or int(parent_context.shape[2]) != current
                or int(current_key.shape[2]) != current
                or int(current_value.shape[2]) != current
                or int(observed_key_history.shape[2]) != phase.full_prefix_tokens
                or tuple(observed_value_history.shape)
                != tuple(observed_key_history.shape)
                or tuple(parent_q_linear.shape) != tuple(parent_query.shape)
                or tuple(parent_k_linear.shape) != tuple(current_key.shape)
                or tuple(current_value.shape) != tuple(current_key.shape)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention rows do not describe "
                    "one complete history plus one aligned terminal suffix."
                )

            full_cos, full_sin = _same_rope(
                source_model,
                target_model,
                phase,
                dtype=parent_q_linear.dtype,
            )
            cached_cos = _finite_tensor(
                tensors["attention.rope_cos"],
                f"phase {phase.identity!r} attention.rope_cos",
                dimensions=3,
            )
            cached_sin = _finite_tensor(
                tensors["attention.rope_sin"],
                f"phase {phase.identity!r} attention.rope_sin",
                dimensions=3,
            )
            if tuple(cached_cos.shape) != (
                phase.batch_size,
                current,
                int(parent_q_linear.shape[3]),
            ) or tuple(cached_sin.shape) != tuple(cached_cos.shape):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} cached RoPE is not its exact "
                    "terminal query/key suffix."
                )
            tolerance = (
                4.0 * torch.finfo(parent_q_linear.dtype).eps
                if parent_q_linear.dtype in {torch.float16, torch.bfloat16}
                else 2e-5
            )
            if not (
                torch.allclose(
                    full_cos[:, -current:, :].to(torch.float32),
                    cached_cos.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
                and torch.allclose(
                    full_sin[:, -current:, :].to(torch.float32),
                    cached_sin.to(torch.float32),
                    rtol=tolerance,
                    atol=tolerance,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Static source RoPE differs from sealed parent cache "
                    f"evidence for phase {phase.identity!r}."
                )
            if not (
                torch.equal(
                    observed_key_history[:, :, -current:, :],
                    current_key,
                )
                and torch.equal(
                    observed_value_history[:, :, -current:, :],
                    current_value,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} current K/V rows are not the "
                    "exact terminal suffix of their sealed histories."
                )

            observed_mask = _attention_key_mask(tensors["attention.additive_mask"])
            if tuple(observed_mask.shape) != (
                phase.batch_size,
                current,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} attention mask does not cover "
                    "its terminal queries over the complete key history."
                )
            current_probabilities = _validated_cached_attention_probabilities(
                probabilities=tensors["attention.probabilities"],
                context_from_probabilities=(
                    tensors["attention.context_from_probabilities"]
                ),
                parent_context=parent_context,
                value_history=observed_value_history,
                key_mask=observed_mask,
                absolute_query_start=phase.full_prefix_tokens - current,
                name=f"phase {phase.identity!r} attention",
            )
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if tuple(child_input.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ) or tuple(child_state.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )

            if previous_phase is None:
                if current != phase.full_prefix_tokens:
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} must seal the complete "
                        "prefill Q/K/V trajectory, not a sampled tail."
                    )
                key_linear_history = parent_k_linear
                key_history = observed_key_history
                value_history = observed_value_history
                key_mask_history = observed_mask
                probability_history = current_probabilities
            else:
                previous_tokens = previous_phase.full_prefix_tokens
                if not (
                    _floating_causal_prefix_matches(
                        previous_child_input,
                        child_input,
                    )
                    and _floating_causal_prefix_matches(
                        previous_child_state,
                        child_state,
                    )
                    and _integer_causal_prefix_matches(
                        previous_input_ids,
                        phase.input_ids,
                    )
                    and _integer_causal_prefix_matches(
                        previous_attention_mask,
                        phase.attention_mask,
                    )
                    and _integer_causal_prefix_matches(
                        previous_position_ids,
                        phase.position_ids,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed an authenticated "
                        "causal child prefix."
                    )
                if (
                    key_linear_history is None
                    or key_history is None
                    or value_history is None
                    or key_mask_history is None
                    or probability_history is None
                    or int(key_linear_history.shape[2]) != previous_tokens
                    or tuple(probability_history.shape[2:])
                    != (previous_tokens, previous_tokens)
                    or not torch.equal(
                        observed_key_history[:, :, :previous_tokens, :],
                        key_history,
                    )
                    or not torch.equal(
                        observed_value_history[:, :, :previous_tokens, :],
                        value_history,
                    )
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} changed its exact sealed "
                        "parent K/V causal prefix."
                    )
                key_linear_history = torch.cat(
                    (key_linear_history, parent_k_linear),
                    dim=2,
                ).contiguous()
                key_history = observed_key_history
                value_history = observed_value_history
                padded_previous_mask = torch.nn.functional.pad(
                    key_mask_history,
                    (0, current),
                    value=False,
                )
                key_mask_history = torch.cat(
                    (padded_previous_mask, observed_mask),
                    dim=1,
                ).contiguous()
                padded_previous_probabilities = torch.nn.functional.pad(
                    probability_history,
                    (0, current),
                    value=0.0,
                )
                probability_history = torch.cat(
                    (
                        padded_previous_probabilities,
                        current_probabilities,
                    ),
                    dim=2,
                ).contiguous()

            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            encoded_output = chart.post_operator.encode_shaped(tensors["attention.o"])
            parent_contribution = _encoded_parent_attention_contribution(
                parent_context=parent_context,
                cached_source_output=tensors["attention.o"],
                cached_target_output=encoded_output,
                name=f"phase {phase.identity!r} attention",
                prepared_transport=prepared_output_transport,
            )
            prepared_transport_use_count += 1
            row_sqrt_weights = phase_sqrt_weights[phase.identity]
            if (
                int(endpoint.shape[1]) != current
                or tuple(parent_contribution.shape[:3])
                != (
                    phase.batch_size,
                    int(parent_query.shape[1]),
                    current,
                )
                or tuple(row_sqrt_weights.shape) != (phase.batch_size, current)
                or int(key_linear_history.shape[2]) != phase.full_prefix_tokens
                or tuple(key_mask_history.shape)
                != (
                    phase.batch_size,
                    phase.full_prefix_tokens,
                    phase.full_prefix_tokens,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} did not extend one contiguous "
                    "attention trajectory."
                )
            query_linear_parts.append(parent_q_linear)
            query_parts.append(parent_query)
            context_parts.append(parent_context)
            contribution_parts.append(parent_contribution)
            endpoint_parts.append(endpoint)
            row_sqrt_weight_parts.append(row_sqrt_weights)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
            final_cos = full_cos
            final_sin = full_sin

        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or final_cos is None
            or final_sin is None
            or key_linear_history is None
            or key_history is None
                or value_history is None
                or key_mask_history is None
                or probability_history is None
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Attention trajectory did not produce one complete causal case."
            )
        parent_query_linear = torch.cat(
            query_linear_parts,
            dim=2,
        ).contiguous()
        parent_query = torch.cat(query_parts, dim=2).contiguous()
        parent_context = torch.cat(context_parts, dim=2).contiguous()
        parent_contribution = torch.cat(
            contribution_parts,
            dim=2,
        ).contiguous()
        parent_endpoint = torch.cat(endpoint_parts, dim=1).contiguous()
        row_sqrt_weights = torch.cat(
            row_sqrt_weight_parts,
            dim=1,
        ).contiguous()
        if not (
            int(parent_query_linear.shape[2])
            == int(parent_query.shape[2])
            == int(parent_context.shape[2])
            == int(parent_contribution.shape[2])
            == int(parent_endpoint.shape[1])
            == int(row_sqrt_weights.shape[1])
            == int(probability_history.shape[2])
            == int(probability_history.shape[3])
            == final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Attention trajectory {batch_index}:{record_id} did not "
                "cover every causal query row exactly once."
            )
        result.append(
            Lfm2CachedAttentionFoldCase(
                target_query_input=final_child_input,
                target_key_value_input=final_child_input,
                parent_query_linear=parent_query_linear,
                parent_key_linear=key_linear_history,
                parent_value=value_history,
                parent_query=parent_query,
                parent_key=key_history,
                parent_context=parent_context,
                parent_post_attention_residual_target_space=parent_endpoint,
                target_pre_attention_residual=final_child_state,
                query_cosine=final_cos,
                query_sine=final_sin,
                key_cosine=final_cos,
                key_sine=final_sin,
                key_mask=key_mask_history,
                parent_contribution=parent_contribution,
                query_row_sqrt_weights=row_sqrt_weights,
                key_value_row_sqrt_weights=row_sqrt_weights.clone(),
                parent_probabilities=probability_history,
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "Attention case assembly requires at least one "
            f"{required_split!r} trajectory."
        )
    return (
        tuple(result),
        _attention_output_transport_complexity_receipt(
            transport=prepared_output_transport,
            phase_count=prepared_transport_use_count,
        ),
    )


@dataclass(frozen=True, slots=True)
class Lfm2CompiledCpcfResidualAttentionSplitFold:
    """Authenticated Q/K producer plus coupled V/O residual-attention state."""

    producer_artifact: Lfm2CompiledStructuralAttentionProducerFold
    value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold
    score_selection_report: Lfm2AttentionScorePreservationReport
    fit_context_replay_sha256: str
    fit_context_bundle_sha256: str
    temporary_producer_install_report: Mapping[str, Any]
    temporary_rollback_proof: Mapping[str, Any]
    compiled_state_sha256: str
    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(
            self.producer_artifact,
            Lfm2CompiledStructuralAttentionProducerFold,
        ):
            raise TypeError(
                "producer_artifact must be an LFM2 structural attention producer fold."
            )
        if not isinstance(
            self.value_output_artifact,
            Lfm2CompiledStructuralAttentionValueOutputFold,
        ):
            raise TypeError("value_output_artifact must be an LFM2 coupled V/O fold.")
        if not isinstance(
            self.score_selection_report,
            Lfm2AttentionScorePreservationReport,
        ):
            raise TypeError(
                "score_selection_report must be an authenticated attention "
                "score-preservation report."
            )
        if (
            self.value_output_artifact.topology != self.producer_artifact.topology
            or self.value_output_artifact.fixed_query_key_sha256
            != self.producer_artifact.score_state_sha256
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Coupled V/O state does not belong to the compiled Q/K producer."
            )
        fit_replay = _digest(
            self.fit_context_replay_sha256,
            "fit_context_replay_sha256",
        )
        fit_context = _digest(
            self.fit_context_bundle_sha256,
            "fit_context_bundle_sha256",
        )
        compiled_sha = _digest(
            self.compiled_state_sha256,
            "compiled_state_sha256",
        )
        if _tensor_bundle_sha256(self.state_dict()) != compiled_sha:
            raise Lfm2CpcfLayerCompilerError(
                "Residual-attention split state authentication failed."
            )
        report = dict(self.report)
        claimed_report_sha = report.pop("report_sha256", None)
        if (
            _digest(
                claimed_report_sha,
                "residual attention split report_sha256",
            )
            != _sha(report)
            or report.get("schema") != LFM2_CPCF_RESIDUAL_ATTENTION_SPLIT_SCHEMA
            or report.get("compiled_state_sha256") != compiled_sha
            or report.get("fit_context_replay_sha256") != fit_replay
            or report.get("fit_context_bundle_sha256") != fit_context
            or report.get("score_selection_report_sha256")
            != self.score_selection_report.report_sha256
            or report.get("producer_artifact") != self.producer_artifact.as_dict()
            or report.get("producer_artifact_sha256")
            != _sha(self.producer_artifact.as_dict())
            or report.get("value_output_artifact")
            != self.value_output_artifact.as_dict()
            or report.get("value_output_artifact_sha256")
            != _sha(self.value_output_artifact.as_dict())
            or report.get("score_selection_report")
            != self.score_selection_report.as_dict()
            or report.get("temporary_producer_install_report")
            != dict(self.temporary_producer_install_report)
            or report.get("temporary_rollback_proof")
            != dict(self.temporary_rollback_proof)
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Residual-attention split artifact seal is invalid."
            )
        rollback = dict(self.temporary_rollback_proof)
        rollback_sha = rollback.pop("proof_sha256", None)
        if (
            _digest(
                rollback_sha,
                "temporary_rollback_proof.proof_sha256",
            )
            != _sha(rollback)
            or rollback.get("exact_layer_rollback_proven") is not True
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Residual-attention temporary rollback proof is invalid."
            )
        object.__setattr__(
            self,
            "temporary_producer_install_report",
            MappingProxyType(dict(self.temporary_producer_install_report)),
        )
        object.__setattr__(
            self,
            "temporary_rollback_proof",
            MappingProxyType(dict(self.temporary_rollback_proof)),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "report_sha256": claimed_report_sha}),
        )

    @property
    def topology(self) -> Any:
        return self.producer_artifact.topology

    @property
    def cluster_plan(self) -> Any:
        return self.producer_artifact.cluster_plan

    def state_dict(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "q_proj.weight": (
                    self.producer_artifact.query_weight.detach().cpu().clone()
                ),
                "k_proj.weight": (
                    self.producer_artifact.key_weight.detach().cpu().clone()
                ),
                "v_proj.weight": (
                    self.value_output_artifact.value_weight.detach().cpu().clone()
                ),
                "out_proj.weight": (
                    self.value_output_artifact.output_weight.detach().cpu().clone()
                ),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _compose_residual_attention_split_artifact(
    *,
    producer_artifact: Lfm2CompiledStructuralAttentionProducerFold,
    value_output_artifact: Lfm2CompiledStructuralAttentionValueOutputFold,
    score_selection_report: Lfm2AttentionScorePreservationReport,
    fit_context_replay: Lfm2CpcfLayerReplayBank,
    actual_contexts: Sequence[Any],
    temporary_producer_install_report: Mapping[str, Any],
    temporary_rollback_proof: Mapping[str, Any],
) -> Lfm2CompiledCpcfResidualAttentionSplitFold:
    if not actual_contexts:
        raise Lfm2CpcfLayerCompilerError(
            "Residual-attention split requires actual fit contexts."
        )
    state = {
        "q_proj.weight": producer_artifact.query_weight,
        "k_proj.weight": producer_artifact.key_weight,
        "v_proj.weight": value_output_artifact.value_weight,
        "out_proj.weight": value_output_artifact.output_weight,
    }
    compiled_sha = _tensor_bundle_sha256(state)
    context_sha = _tensor_bundle_sha256(
        {f"context.{index:04d}": value for index, value in enumerate(actual_contexts)}
    )
    body = {
        "schema": LFM2_CPCF_RESIDUAL_ATTENTION_SPLIT_SCHEMA,
        "producer_artifact": producer_artifact.as_dict(),
        "producer_artifact_sha256": _sha(producer_artifact.as_dict()),
        "value_output_artifact": value_output_artifact.as_dict(),
        "value_output_artifact_sha256": _sha(value_output_artifact.as_dict()),
        "score_selection_report": score_selection_report.as_dict(),
        "score_selection_report_sha256": (score_selection_report.report_sha256),
        "fit_context_replay_sha256": (fit_context_replay.report["report_sha256"]),
        "fit_context_bundle_sha256": context_sha,
        "temporary_producer_install_report": dict(temporary_producer_install_report),
        "temporary_rollback_proof": dict(temporary_rollback_proof),
        "compiled_state_sha256": compiled_sha,
        "fit_only_producer_compile": True,
        "fit_only_value_output_compile": True,
        "selection_only_query_key_score_evaluation": True,
        "validation_used_for_compilation_or_selection": False,
        "actual_context_source": (
            "temporary_installed_qkv_fit_bank_operator_output_input"
        ),
        "live_value_source": "compiled_producer_v",
        "value_prior_source": "compiled_producer_v",
        "output_prior_source": "original_blank_target_o",
        "fixed_query_key_then_coupled_value_output_install": True,
        "dense_target_native_state": True,
        "execution": {
            "parent_model_forward_calls": 0,
            "target_model_forward_calls": 0,
            "target_layer_forward_calls": 0,
            "generation_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        },
    }
    return Lfm2CompiledCpcfResidualAttentionSplitFold(
        producer_artifact=producer_artifact,
        value_output_artifact=value_output_artifact,
        score_selection_report=score_selection_report,
        fit_context_replay_sha256=(fit_context_replay.report["report_sha256"]),
        fit_context_bundle_sha256=context_sha,
        temporary_producer_install_report=temporary_producer_install_report,
        temporary_rollback_proof=temporary_rollback_proof,
        compiled_state_sha256=compiled_sha,
        report={**body, "report_sha256": _sha(body)},
    )


def _attention_actual_contexts(
    *,
    input_bank: Lfm2CpcfPhaseStateBank,
    replay_bank: Lfm2CpcfLayerReplayBank,
    required_split: str,
) -> tuple[Any, ...]:
    """Return final full-prefix live contexts in attention-case order."""

    grouped: dict[
        tuple[int, str],
        list[Lfm2CpcfTargetPhase],
    ] = {}
    for phase in input_bank.phases:
        if phase.split == required_split:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)
    if not grouped:
        raise Lfm2CpcfLayerCompilerError(
            "Actual attention context extraction requires at least one "
            f"{required_split!r} trajectory."
        )
    contexts = []
    for trajectory_id, trajectory in grouped.items():
        final_phase = trajectory[-1]
        try:
            replay = replay_bank.replays[final_phase.identity]
        except KeyError as exc:
            raise Lfm2CpcfLayerCompilerError(
                f"Actual attention context replay lacks trajectory {trajectory_id!r}."
            ) from exc
        context = _finite_tensor(
            replay.operator_output_input,
            f"trajectory {trajectory_id!r} actual attention context",
            dimensions=3,
        )
        if tuple(context.shape[:2]) != (
            final_phase.batch_size,
            final_phase.full_prefix_tokens,
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Actual attention context for trajectory {trajectory_id!r} "
                "does not cover its complete prefix."
            )
        contexts.append(context)
    return tuple(contexts)


_SHORT_CONV_HISTORY_FIELDS = (
    "short_conv.b_history",
    "short_conv.c_history",
    "short_conv.x_history",
)
_SHORT_CONV_V4_HISTORY_FIELDS = (
    "operator.input",
    *_SHORT_CONV_HISTORY_FIELDS,
)


def _extend_short_conv_parent_histories(
    phase: Lfm2CpcfTargetPhase,
    tensors: Mapping[str, Any],
    *,
    layer_index: int,
    histories: dict[tuple[int, str], Any],
    include_operator_input: bool = False,
) -> dict[str, Any]:
    """Rebuild full causal parent histories from cached decode suffixes.

    The strict tokenizer trace adapter maps one target row to one donor row
    without changing batch or sequence geometry.  A prefill therefore seals
    the complete prompt history, while each cached decode phase contributes
    only its newly executed terminal suffix.  Both short-convolution solvers
    consume authenticated B/C/x histories.  V4 additionally consumes the
    complete parent operator-input history for its child-to-parent affine
    lift; it is never reconstructed from projected B/C/x values.
    """

    torch = _torch()
    history_fields = (
        _SHORT_CONV_V4_HISTORY_FIELDS
        if include_operator_input
        else _SHORT_CONV_HISTORY_FIELDS
    )
    result = {}
    previous_fields = {
        field: histories.get((phase.batch_index, field)) for field in history_fields
    }
    for field in history_fields:
        current = _finite_tensor(
            tensors[field],
            f"phase {phase.identity!r} layer {layer_index} {field}",
            dimensions=3,
        )
        if int(current.shape[0]) != phase.batch_size:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} changed its batch boundary."
            )
        key = (phase.batch_index, field)
        if phase.phase == "prefill":
            if key in histories or int(current.shape[1]) != phase.full_prefix_tokens:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not one complete "
                    "causal prefill history."
                )
            history = current
        else:
            previous = histories.get(key)
            previous_tokens = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            if (
                previous is None
                or tuple(previous.shape[:1]) != tuple(current.shape[:1])
                or int(previous.shape[1]) != previous_tokens
                or int(previous.shape[2]) != int(current.shape[2])
                or previous.dtype != current.dtype
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} is not a causal "
                    "continuation of its authenticated parent history."
                )
            current_tokens = int(current.shape[1])
            if current_tokens == phase.sampled_suffix_tokens:
                history = torch.cat((previous, current), dim=1).contiguous()
            elif current_tokens == phase.full_prefix_tokens:
                if not torch.equal(current[:, :previous_tokens, :], previous):
                    raise Lfm2CpcfLayerCompilerError(
                        f"Phase {phase.identity!r} full {field} changed its "
                        "authenticated causal prefix."
                    )
                history = current
            else:
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} {field} contains neither one "
                    "terminal suffix nor one complete causal history."
                )
        if int(history.shape[1]) != phase.full_prefix_tokens:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} {field} did not reconstruct its "
                "complete target-aligned prefix."
            )
        histories[key] = history
        result[field] = history
    if phase.phase != "prefill":
        previous_b = previous_fields["short_conv.b_history"]
        previous_x = previous_fields["short_conv.x_history"]
        if previous_b is None or previous_x is None:
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} has no authenticated short-conv "
                "warm-state prefix."
            )
        pre_bx_state = _finite_tensor(
            tensors["short_conv.pre_bx_state"],
            (f"phase {phase.identity!r} layer {layer_index} short_conv.pre_bx_state"),
            dimensions=3,
        )
        fir_length = int(pre_bx_state.shape[-1])
        previous_bx = (previous_b * previous_x).transpose(1, 2).contiguous()
        if int(previous_bx.shape[-1]) < fir_length:
            previous_bx = torch.nn.functional.pad(
                previous_bx,
                (fir_length - int(previous_bx.shape[-1]), 0),
            )
        expected_pre_state = previous_bx[..., -fir_length:].contiguous()
        if (
            tuple(pre_bx_state.shape) != tuple(expected_pre_state.shape)
            or pre_bx_state.dtype != expected_pre_state.dtype
            or not torch.equal(pre_bx_state, expected_pre_state)
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} short-conv warm state is not the "
                "exact authenticated tail of its reconstructed causal prefix."
            )
    return result


def _floating_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    if (
        not isinstance(previous, torch.Tensor)
        or not isinstance(current, torch.Tensor)
        or previous.ndim != current.ndim
        or previous.ndim < 2
        or previous.dtype != current.dtype
        or tuple(previous.shape[:1]) != tuple(current.shape[:1])
        or tuple(previous.shape[2:]) != tuple(current.shape[2:])
        or int(previous.shape[1]) > int(current.shape[1])
    ):
        return False
    tolerance = 8.0 * torch.finfo(previous.dtype).eps
    return bool(
        torch.allclose(
            current[:, : int(previous.shape[1]), ...].to(torch.float64),
            previous.to(torch.float64),
            rtol=tolerance,
            atol=tolerance,
        )
    )


def _integer_causal_prefix_matches(previous: Any, current: Any) -> bool:
    torch = _torch()
    return bool(
        isinstance(previous, torch.Tensor)
        and isinstance(current, torch.Tensor)
        and previous.ndim == current.ndim
        and previous.ndim >= 2
        and previous.dtype == current.dtype
        and tuple(previous.shape[:1]) == tuple(current.shape[:1])
        and tuple(previous.shape[2:]) == tuple(current.shape[2:])
        and int(previous.shape[1]) <= int(current.shape[1])
        and torch.equal(
            current[:, : int(previous.shape[1]), ...],
            previous,
        )
    )


def _short_conv_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    preinstall_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
    use_v4: bool = False,
) -> tuple[
    Lfm2CpcfShortConvFoldCase | Lfm2CpcfShortConvV4FoldCase,
    ...,
]:
    grouped: dict[tuple[int, str], list[Lfm2CpcfTargetPhase]] = {}
    for phase in input_bank.phases:
        if phase.split in _SHORT_CONV_CASE_SPLITS:
            grouped.setdefault(
                (phase.batch_index, phase.record_id),
                [],
            ).append(phase)

    result = []
    for (_batch_index, _record_id), trajectory in grouped.items():
        histories: dict[tuple[int, str], Any] = {}
        parent_histories: dict[str, Any] | None = None
        endpoint_parts = []
        endpoint_start: int | None = None
        endpoint_end: int | None = None
        previous_phase: Lfm2CpcfTargetPhase | None = None
        previous_child_input = None
        previous_child_state = None
        previous_input_ids = None
        previous_attention_mask = None
        previous_position_ids = None
        final_phase: Lfm2CpcfTargetPhase | None = None
        final_child_input = None
        final_child_state = None
        for phase_index, phase in enumerate(trajectory):
            expected_phase = (
                "prefill" if phase_index == 0 else f"decode.{phase_index - 1:03d}"
            )
            if phase.phase != expected_phase:
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} skips or reorders causal phase "
                    f"{expected_phase!r}."
                )
            if previous_phase is not None and (
                phase.split != previous_phase.split
                or phase.full_prefix_tokens
                != (previous_phase.full_prefix_tokens + phase.sampled_suffix_tokens)
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Short-conv trajectory {phase.batch_index}:"
                    f"{phase.record_id} changed split or target prefix "
                    "geometry."
                )
            tensors = _phase_evidence(phase, layer_index, "short_conv")
            parent_histories = _extend_short_conv_parent_histories(
                phase,
                tensors,
                layer_index=layer_index,
                histories=histories,
                include_operator_input=use_v4,
            )
            replay = preinstall_replay.replays[phase.identity]
            child_input = _finite_tensor(
                replay.operator_input,
                f"phase {phase.identity!r} actual child operator input",
                dimensions=3,
            )
            child_state = _finite_tensor(
                input_bank.state(phase),
                f"phase {phase.identity!r} actual child pre-operator state",
                dimensions=3,
            )
            if tuple(child_input.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ) or tuple(child_state.shape[:2]) != (
                phase.batch_size,
                phase.full_prefix_tokens,
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} actual child states do not "
                    "cover their complete target prefix."
                )
            if previous_phase is not None and not (
                _floating_causal_prefix_matches(
                    previous_child_input,
                    child_input,
                )
                and _floating_causal_prefix_matches(
                    previous_child_state,
                    child_state,
                )
                and _integer_causal_prefix_matches(
                    previous_input_ids,
                    phase.input_ids,
                )
                and _integer_causal_prefix_matches(
                    previous_attention_mask,
                    phase.attention_mask,
                )
                and _integer_causal_prefix_matches(
                    previous_position_ids,
                    phase.position_ids,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} changed an authenticated causal "
                    "child prefix."
                )
            endpoint = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            )
            current_start = phase.full_prefix_tokens - phase.sampled_suffix_tokens
            if int(endpoint.shape[1]) != phase.sampled_suffix_tokens or (
                endpoint_end is not None and current_start != endpoint_end
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Phase {phase.identity!r} residual endpoint is not the "
                    "next contiguous causal suffix."
                )
            if endpoint_start is None:
                endpoint_start = current_start
            endpoint_end = current_start + int(endpoint.shape[1])
            endpoint_parts.append(endpoint)
            previous_phase = phase
            previous_child_input = child_input
            previous_child_state = child_state
            previous_input_ids = phase.input_ids
            previous_attention_mask = phase.attention_mask
            previous_position_ids = phase.position_ids
            final_phase = phase
            final_child_input = child_input
            final_child_state = child_state
        if (
            final_phase is None
            or final_child_input is None
            or final_child_state is None
            or parent_histories is None
            or endpoint_start is None
            or endpoint_end != final_phase.full_prefix_tokens
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv trajectory did not produce one complete causal case."
            )
        endpoint = _torch().cat(endpoint_parts, dim=1).contiguous()
        actual_pre = final_child_state[:, endpoint_start:, :].contiguous()
        case_arguments = {
            "case_id": (
                f"{final_phase.batch_index}:{final_phase.record_id}:trajectory:"
                f"layer.{layer_index}"
            ),
            "split": final_phase.split,
            "suffix_start": endpoint_start,
            "actual_child_operator_input_history": final_child_input,
            "parent_b_history": parent_histories["short_conv.b_history"],
            "parent_c_history": parent_histories["short_conv.c_history"],
            "parent_x_history": parent_histories["short_conv.x_history"],
            "parent_post_operator_residual_target_space_suffix": endpoint,
            "actual_child_pre_operator_residual_suffix": actual_pre,
            "valid_prefix_mask": final_phase.attention_mask,
        }
        if use_v4:
            result.append(
                Lfm2CpcfShortConvV4FoldCase(
                    **case_arguments,
                    parent_operator_input_history=parent_histories["operator.input"],
                )
            )
        else:
            result.append(Lfm2CpcfShortConvFoldCase(**case_arguments))
    if not any(value.split == "fit" for value in result) or not any(
        value.split == "validation" for value in result
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv compilation requires fit and validation phases."
        )
    if use_v4 and not any(value.split == "selection" for value in result):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv v4 compilation requires selection phases."
        )
    return tuple(result)


def _mlp_cases(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> tuple[Lfm2CachedMlpFoldCase, ...]:
    result = []
    for phase in input_bank.phases:
        if phase.split != "fit":
            continue
        tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
        replay = operator_replay.replays[phase.identity]
        target_input = phase.suffix(replay.mlp_input)
        target_pre = phase.suffix(replay.post_operator_residual)
        parent_endpoint = chart.post_layer.encode_shaped(tensors["residual.post_layer"])
        result.append(
            Lfm2CachedMlpFoldCase(
                target_input=target_input.reshape(-1, int(target_input.shape[-1])),
                parent_gate=tensors["mlp.gate"].reshape(
                    -1,
                    int(tensors["mlp.gate"].shape[-1]),
                ),
                parent_up=tensors["mlp.up"].reshape(
                    -1,
                    int(tensors["mlp.up"].shape[-1]),
                ),
                parent_post_swiglu=tensors["mlp.post_swiglu"].reshape(
                    -1,
                    int(tensors["mlp.post_swiglu"].shape[-1]),
                ),
                parent_post_mlp_residual_target_space=parent_endpoint.reshape(
                    -1,
                    int(parent_endpoint.shape[-1]),
                ),
                target_pre_mlp_residual=target_pre.reshape(
                    -1,
                    int(target_pre.shape[-1]),
                ),
            )
        )
    if not result:
        raise Lfm2CpcfLayerCompilerError(
            "SwiGLU compilation requires at least one fit phase."
        )
    return tuple(result)


def _install_short_conv(
    target_short_conv: Any,
    compiled: Lfm2CompiledCpcfShortConvFold | Lfm2CompiledCpcfShortConvV4Fold,
) -> Mapping[str, Any]:
    torch = _torch()
    if not isinstance(
        compiled,
        (Lfm2CompiledCpcfShortConvFold, Lfm2CompiledCpcfShortConvV4Fold),
    ):
        raise TypeError("compiled must be an authenticated LFM2 CPCF short-conv fold.")
    if (
        isinstance(
            compiled,
            Lfm2CompiledCpcfShortConvV4Fold,
        )
        and not compiled.verify_authentication()
    ):
        raise Lfm2CpcfLayerInstallError(
            "Compiled short-conv v4 authentication failed before install."
        )
    available = {
        "in_proj.weight": getattr(
            getattr(target_short_conv, "in_proj", None),
            "weight",
            None,
        ),
        "in_proj.bias": getattr(
            getattr(target_short_conv, "in_proj", None),
            "bias",
            None,
        ),
        "conv.weight": getattr(
            getattr(target_short_conv, "conv", None),
            "weight",
            None,
        ),
        "conv.bias": getattr(
            getattr(target_short_conv, "conv", None),
            "bias",
            None,
        ),
        "out_proj.weight": getattr(
            getattr(target_short_conv, "out_proj", None),
            "weight",
            None,
        ),
        "out_proj.bias": getattr(
            getattr(target_short_conv, "out_proj", None),
            "bias",
            None,
        ),
    }
    expected = compiled.state_dict()
    if set(expected) != {
        name for name, value in available.items() if isinstance(value, torch.Tensor)
    }:
        raise Lfm2CpcfLayerInstallError(
            "Target short-conv bias topology differs from compiled state."
        )
    for name, expected_value in expected.items():
        observed = available[name]
        if (
            not isinstance(observed, torch.Tensor)
            or tuple(observed.shape) != tuple(expected_value.shape)
            or observed.dtype != expected_value.dtype
        ):
            raise Lfm2CpcfLayerInstallError(
                f"Target short-conv {name} differs in shape or dtype."
            )
    identities = {name: id(available[name]) for name in expected}
    gradients = {
        name: (
            bool(available[name].requires_grad),
            None if available[name].grad is None else id(available[name].grad),
            None
            if available[name].grad is None
            else _tensor_sha256(available[name].grad),
        )
        for name in expected
    }
    backup = {
        name: available[name].detach().to(device="cpu").clone() for name in expected
    }
    try:
        with torch.no_grad():
            for name, expected_value in expected.items():
                available[name].copy_(
                    expected_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        installed_sha = _short_conv_state_sha256(
            {name: available[name] for name in expected}
        )
        body = {
            "scope": (
                "authenticated_residual_endpoint_cpcf_short_conv_v4_install"
                if isinstance(compiled, Lfm2CompiledCpcfShortConvV4Fold)
                else "suffix_aware_cpcf_short_conv_install"
            ),
            "parameter_identities_preserved": all(
                id(available[name]) == identities[name] for name in expected
            ),
            "parameter_shapes_preserved": True,
            "gradient_state_preserved": all(
                (
                    bool(available[name].requires_grad),
                    None if available[name].grad is None else id(available[name].grad),
                    None
                    if available[name].grad is None
                    else _tensor_sha256(available[name].grad),
                )
                == gradients[name]
                for name in expected
            ),
            "compiled_state_sha256": compiled.report["compiled_state_sha256"],
            "installed_state_sha256": installed_sha,
            "target_logical_parameters": sum(
                int(available[name].numel()) for name in expected
            ),
            "training_steps": 0,
        }
        if (
            not body["parameter_identities_preserved"]
            or not body["gradient_state_preserved"]
            or body["compiled_state_sha256"] != installed_sha
        ):
            raise Lfm2CpcfLayerInstallError(
                "Installed short-conv state failed identity or content checks."
            )
        return MappingProxyType({**body, "report_sha256": _sha(body)})
    except Exception:
        with torch.no_grad():
            for name, backup_value in backup.items():
                available[name].copy_(
                    backup_value.to(
                        device=available[name].device,
                        dtype=available[name].dtype,
                    )
                )
        raise


def _metrics(prediction: Any, target: Any) -> Mapping[str, Any]:
    torch = _torch()
    left = prediction.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    right = target.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if tuple(left.shape) != tuple(right.shape) or left.numel() == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint metric tensors are empty or misaligned."
        )
    difference = left - right
    rmse = torch.sqrt(torch.mean(difference.square()))
    target_scale = torch.sqrt(torch.mean(right.square())).clamp_min(
        torch.finfo(torch.float64).eps
    )
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    cosine = (
        1.0
        if float(left_norm.item()) == 0.0 and float(right_norm.item()) == 0.0
        else (
            0.0
            if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0
            else float(torch.dot(left, right).div(left_norm * right_norm).item())
        )
    )
    return MappingProxyType(
        {
            "rows": int(prediction.numel() // prediction.shape[-1]),
            "width": int(prediction.shape[-1]),
            "rmse": float(rmse.item()),
            "relative_rmse": float((rmse / target_scale).item()),
            "cosine_similarity": cosine,
            "maximum_absolute_error": float(difference.abs().max().item()),
            "prediction_sha256": _tensor_sha256(prediction),
            "target_sha256": _tensor_sha256(target),
        }
    )


def _attention_context_preservation_metrics(
    *,
    operator_artifact: (
        Lfm2CompiledStructuralAttentionFold
        | Lfm2CompiledCpcfResidualAttentionSplitFold
        | Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
        | Lfm2CompiledComposedQkSequenceEffectiveAttentionFold
        | Lfm2CompiledComposedEndpointResidualAttentionFold
    ),
    cases: Sequence[Lfm2CachedAttentionFoldCase],
    actual_child_contexts: Sequence[Any],
) -> Mapping[str, Any]:
    """Compare live child contexts to cached parent target-head barycenters."""

    torch = _torch()
    if isinstance(
        operator_artifact,
        (
            Lfm2CompiledCpcfResidualAttentionSplitFold,
            Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
            Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
            Lfm2CompiledComposedEndpointResidualAttentionFold,
        ),
    ):
        producer = operator_artifact.producer_artifact
        query_groups = producer.query_groups
        barycentric_weights = producer.query_barycentric_weights
        topology = producer.topology
    elif isinstance(
        operator_artifact,
        Lfm2CompiledStructuralAttentionFold,
    ):
        query_groups = operator_artifact.query_groups
        barycentric_weights = operator_artifact.query_barycentric_weights
        topology = operator_artifact.topology
    else:
        raise TypeError("Unknown attention artifact for context preservation.")
    selected_cases = tuple(cases)
    selected_contexts = tuple(actual_child_contexts)
    if (
        not selected_cases
        or len(selected_cases) != len(selected_contexts)
        or len(query_groups) != topology.target_query_heads
        or len(barycentric_weights) != topology.target_query_heads
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Selection contexts, cases, and target-head groups are misaligned."
        )

    predictions = []
    targets = []
    case_reports = []
    for case_index, (case, raw_context) in enumerate(
        zip(selected_cases, selected_contexts, strict=True)
    ):
        parent_context = _finite_tensor(
            case.parent_context,
            f"selection cases[{case_index}].parent_context",
            dimensions=4,
        ).to(dtype=torch.float64)
        actual_context = _finite_tensor(
            raw_context,
            f"selection cases[{case_index}].actual_child_context",
            dimensions=3,
        ).to(dtype=torch.float64)
        batch = int(parent_context.shape[0])
        query_length = int(parent_context.shape[2])
        if tuple(parent_context.shape) != (
            batch,
            topology.source_query_heads,
            query_length,
            topology.head_dim,
        ) or tuple(actual_context.shape) != (
            batch,
            query_length,
            topology.target_query_heads * topology.head_dim,
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Selection parent/child attention contexts differ from the "
                "compiled topology."
            )
        target_heads = []
        for target_head, (members, weights) in enumerate(
            zip(query_groups, barycentric_weights, strict=True)
        ):
            if (
                not members
                or len(members) != len(weights)
                or any(
                    isinstance(member, bool)
                    or not isinstance(member, int)
                    or member < 0
                    or member >= topology.source_query_heads
                    for member in members
                )
                or any(
                    not math.isfinite(float(weight)) or float(weight) < 0.0
                    for weight in weights
                )
                or not math.isclose(
                    math.fsum(float(weight) for weight in weights),
                    1.0,
                    rel_tol=1e-8,
                    abs_tol=1e-10,
                )
            ):
                raise Lfm2CpcfLayerCompilerError(
                    f"Target attention head {target_head} has an invalid "
                    "parent-context barycenter."
                )
            selected = parent_context.index_select(
                1,
                torch.tensor(members, dtype=torch.long),
            )
            coefficients = torch.tensor(
                weights,
                dtype=torch.float64,
            ).reshape(1, len(members), 1, 1)
            target_heads.append((selected * coefficients).sum(dim=1))
        target_context = torch.stack(
            target_heads,
            dim=2,
        ).reshape(
            batch,
            query_length,
            topology.target_query_heads * topology.head_dim,
        )
        case_metric = _metrics(actual_context, target_context)
        predictions.append(actual_context.reshape(-1, int(actual_context.shape[-1])))
        targets.append(target_context.reshape(-1, int(target_context.shape[-1])))
        case_body = {
            "case_index": case_index,
            "metrics": dict(case_metric),
        }
        case_reports.append({**case_body, "report_sha256": _sha(case_body)})
    aggregate = _metrics(
        torch.cat(predictions, dim=0),
        torch.cat(targets, dim=0),
    )
    body = {
        "schema": ("neural-forge-lfm2-cpcf-attention-context-preservation-v1"),
        "case_count": len(selected_cases),
        "aggregate": dict(aggregate),
        "cases": case_reports,
        "reference": ("cached_parent_attention_context_target_head_barycenter"),
        "candidate": "live_selection_replay_operator_output_input",
        "query_groups_sha256": _sha(
            {
                "groups": [list(value) for value in query_groups],
                "weights": [list(value) for value in barycentric_weights],
            }
        ),
        "selection_split_only": True,
        "validation_cases_consumed": 0,
        "model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "training_steps": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _split_endpoint_metrics(
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    operator_replay: Lfm2CpcfLayerReplayBank,
    final_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
    splits: Sequence[str] = _EVALUATION_SPLITS,
) -> Mapping[str, Any]:
    torch = _torch()
    requested_splits = tuple(splits)
    if (
        not requested_splits
        or any(not isinstance(split, str) or not split for split in requested_splits)
        or len(set(requested_splits)) != len(requested_splits)
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint metrics require unique nonempty split names."
        )
    result = {}
    for split in requested_splits:
        operator_predictions = []
        operator_targets = []
        layer_predictions = []
        layer_targets = []
        phase_seals = []
        phase_groups: dict[str, dict[str, list[Any]]] = {}
        for phase in input_bank.phases:
            if phase.split != split:
                continue
            tensors = _phase_evidence(phase, layer_index, chart.operator_kind)
            operator_prediction = phase.suffix(
                operator_replay.replays[phase.identity].post_operator_residual
            ).to(device="cpu")
            operator_target = chart.post_operator.encode_shaped(
                tensors["residual.post_operator"]
            ).to(device="cpu")
            layer_prediction = phase.suffix(
                final_replay.replays[phase.identity].post_layer_residual
            ).to(device="cpu")
            layer_target = chart.post_layer.encode_shaped(
                tensors["residual.post_layer"]
            ).to(device="cpu")
            operator_predictions.append(operator_prediction)
            operator_targets.append(operator_target)
            layer_predictions.append(layer_prediction)
            layer_targets.append(layer_target)
            phase_group = phase_groups.setdefault(
                phase.phase,
                {
                    "operator_predictions": [],
                    "operator_targets": [],
                    "layer_predictions": [],
                    "layer_targets": [],
                    "phase_sha256": [],
                },
            )
            phase_group["operator_predictions"].append(operator_prediction)
            phase_group["operator_targets"].append(operator_target)
            phase_group["layer_predictions"].append(layer_prediction)
            phase_group["layer_targets"].append(layer_target)
            phase_group["phase_sha256"].append(phase.phase_sha256)
            phase_seals.append(phase.phase_sha256)
        if not operator_predictions:
            raise Lfm2CpcfLayerCompilerError(
                f"Layer evaluation requires at least one {split} phase."
            )
        phase_metrics = {}
        for phase_name, values in phase_groups.items():
            phase_operator_prediction = torch.cat(
                [
                    value.reshape(-1, int(value.shape[-1]))
                    for value in values["operator_predictions"]
                ],
                dim=0,
            )
            phase_operator_target = torch.cat(
                [
                    value.reshape(-1, int(value.shape[-1]))
                    for value in values["operator_targets"]
                ],
                dim=0,
            )
            phase_layer_prediction = torch.cat(
                [
                    value.reshape(-1, int(value.shape[-1]))
                    for value in values["layer_predictions"]
                ],
                dim=0,
            )
            phase_layer_target = torch.cat(
                [
                    value.reshape(-1, int(value.shape[-1]))
                    for value in values["layer_targets"]
                ],
                dim=0,
            )
            phase_body = {
                "phase": phase_name,
                "phase_count": len(values["phase_sha256"]),
                "phase_sha256": list(values["phase_sha256"]),
                "post_operator": dict(
                    _metrics(
                        phase_operator_prediction,
                        phase_operator_target,
                    )
                ),
                "post_layer": dict(
                    _metrics(
                        phase_layer_prediction,
                        phase_layer_target,
                    )
                ),
            }
            phase_metrics[phase_name] = {
                **phase_body,
                "report_sha256": _sha(phase_body),
            }
        operator_prediction = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in operator_predictions],
            dim=0,
        )
        operator_target = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in operator_targets],
            dim=0,
        )
        layer_prediction = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_predictions],
            dim=0,
        )
        layer_target = torch.cat(
            [value.reshape(-1, int(value.shape[-1])) for value in layer_targets],
            dim=0,
        )
        body = {
            "split": split,
            "phase_count": len(phase_seals),
            "phase_sha256": phase_seals,
            "post_operator": dict(_metrics(operator_prediction, operator_target)),
            "post_layer": dict(_metrics(layer_prediction, layer_target)),
            "phase_metrics": phase_metrics,
        }
        result[split] = {**body, "report_sha256": _sha(body)}
    return MappingProxyType(result)


def _install_cpcf_attention_artifact(
    *,
    source_attention: Any,
    target_attention: Any,
    artifact: (
        Lfm2CompiledStructuralAttentionFold
        | Lfm2CompiledCpcfResidualAttentionSplitFold
        | Lfm2CompiledFixedQkSequenceEffectiveAttentionFold
        | Lfm2CompiledComposedQkSequenceEffectiveAttentionFold
        | Lfm2CompiledComposedEndpointResidualAttentionFold
    ),
) -> Mapping[str, Any]:
    """Install one complete attention candidate into the existing blank shell."""

    norm_report = _copy_attention_norms(
        source_attention,
        target_attention,
    )
    if isinstance(artifact, Lfm2CompiledStructuralAttentionFold):
        raw_operator_install = install_lfm2_structural_attention_fold(
            target_attention,
            artifact,
        ).as_dict()
    elif isinstance(
        artifact,
        Lfm2CompiledComposedEndpointResidualAttentionFold,
    ):
        raw_operator_install = install_lfm2_fixed_qk_sequence_effective_attention_fold(
            target_attention,
            artifact.blocks[-1].vo_artifact,
        ).as_dict()
    elif isinstance(
        artifact,
        Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
    ):
        raw_operator_install = install_lfm2_fixed_qk_sequence_effective_attention_fold(
            target_attention,
            artifact.inner_artifact,
        ).as_dict()
    elif isinstance(
        artifact,
        Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
    ):
        raw_operator_install = install_lfm2_fixed_qk_sequence_effective_attention_fold(
            target_attention,
            artifact,
        ).as_dict()
    elif isinstance(
        artifact,
        Lfm2CompiledCpcfResidualAttentionSplitFold,
    ):
        transaction = install_lfm2_structural_attention_score_producer_fold(
            target_attention,
            artifact.producer_artifact,
        )
        query_key_report = transaction.producer_report
        if query_key_report is None:
            raise Lfm2CpcfLayerInstallError(
                "Residual-attention Q/K transaction issued no install report."
            )
        value_output_report = install_lfm2_structural_attention_value_output_fold(
            target_attention,
            artifact.value_output_artifact,
            transaction,
        )
        installed_state = {
            "q_proj.weight": target_attention.q_proj.weight,
            "k_proj.weight": target_attention.k_proj.weight,
            "v_proj.weight": target_attention.v_proj.weight,
            "out_proj.weight": target_attention.out_proj.weight,
        }
        installed_sha = _tensor_bundle_sha256(installed_state)
        split_body = {
            "scope": (
                "fit_compiled_selection_chosen_coupled_residual_attention_install"
            ),
            "query_key_install": query_key_report.as_dict(),
            "value_output_install": value_output_report.as_dict(),
            "compiled_state_sha256": artifact.compiled_state_sha256,
            "installed_state_sha256": installed_sha,
            "parameter_identities_preserved": (
                query_key_report.parameter_identities_preserved
                and value_output_report.parameter_identities_preserved
            ),
            "parameter_shapes_preserved": (
                query_key_report.parameter_shapes_preserved
                and value_output_report.parameter_shapes_preserved
            ),
            "parameter_dtypes_preserved": (
                query_key_report.parameter_dtypes_preserved
                and value_output_report.parameter_dtypes_preserved
            ),
            "gradient_state_preserved": (
                query_key_report.gradient_objects_preserved
                and value_output_report.gradient_objects_preserved
            ),
            "fixed_query_key_verified_during_value_output_commit": True,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        if (
            not split_body["parameter_identities_preserved"]
            or not split_body["parameter_shapes_preserved"]
            or not split_body["parameter_dtypes_preserved"]
            or not split_body["gradient_state_preserved"]
            or installed_sha != artifact.compiled_state_sha256
        ):
            raise Lfm2CpcfLayerInstallError(
                "Residual-attention split installation failed its exact "
                "identity or content audit."
            )
        raw_operator_install = {
            **split_body,
            "report_sha256": _sha(split_body),
        }
    else:  # pragma: no cover - guarded by all callers
        raise TypeError("Unknown CPCF attention artifact type.")
    return MappingProxyType(
        {
            "operator": raw_operator_install,
            "attention_norms": dict(norm_report),
        }
    )


def _prepare_cpcf_mlp_basis_from_fit_replay(
    *,
    source_layer: Any,
    target_layer: Any,
    layer_index: int,
    fit_bank: Lfm2CpcfPhaseStateBank,
    operator_fit_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
) -> Lfm2PreparedStructuralMlpFoldBasis:
    mlp_cases = _mlp_cases(
        layer_index=layer_index,
        input_bank=fit_bank,
        operator_replay=operator_fit_replay,
        chart=chart,
    )
    return prepare_lfm2_structural_residual_mlp_fold_basis(
        source_gate_weight=source_layer.feed_forward.w1.weight,
        source_up_weight=source_layer.feed_forward.w3.weight,
        source_down_weight=source_layer.feed_forward.w2.weight,
        target_gate_weight_template=target_layer.feed_forward.w1.weight,
        target_up_weight_template=target_layer.feed_forward.w3.weight,
        target_down_weight_template=target_layer.feed_forward.w2.weight,
        cases=mlp_cases,
        residual_assignment_sha256=(chart.post_layer.report["report_sha256"]),
    )


def _compile_cpcf_mlp_from_fit_replay(
    *,
    source_layer: Any,
    target_layer: Any,
    layer_index: int,
    fit_bank: Lfm2CpcfPhaseStateBank,
    operator_fit_replay: Lfm2CpcfLayerReplayBank,
    chart: Lfm2CpcfLayerBoundaryCharts,
    ridge: float,
    prepared_basis: Lfm2PreparedStructuralMlpFoldBasis | None = None,
) -> Lfm2CompiledStructuralMlpFold:
    mlp_cases = _mlp_cases(
        layer_index=layer_index,
        input_bank=fit_bank,
        operator_replay=operator_fit_replay,
        chart=chart,
    )
    return compile_lfm2_structural_residual_mlp_fold(
        source_gate_weight=source_layer.feed_forward.w1.weight,
        source_up_weight=source_layer.feed_forward.w3.weight,
        source_down_weight=source_layer.feed_forward.w2.weight,
        target_gate_weight_template=target_layer.feed_forward.w1.weight,
        target_up_weight_template=target_layer.feed_forward.w3.weight,
        target_down_weight_template=target_layer.feed_forward.w2.weight,
        cases=mlp_cases,
        residual_assignment_sha256=(chart.post_layer.report["report_sha256"]),
        ridge=ridge,
        max_prior_delta_ratio=None,
        prepared_basis=prepared_basis,
    )


def _fit_attention_candidate_post_layer_transport(
    *,
    candidate_id: str,
    layer_index: int,
    fit_bank: Lfm2CpcfPhaseStateBank,
    operator_fit_replay: Lfm2CpcfLayerReplayBank,
    source_charts: Lfm2CpcfLayerBoundaryCharts,
    policy: Lfm2CpcfAttentionSequentialTransportPolicy,
) -> tuple[
    CpcfSequentialBoundaryTransport,
    Lfm2CpcfLayerBoundaryCharts,
    Mapping[str, Any],
]:
    """Freeze one candidate's fit-only post-operator -> post-layer gauge."""

    torch = _torch()
    if not policy.enabled:
        raise Lfm2CpcfLayerCompilerError(
            "Candidate-local post-layer transport requires an enabled policy."
        )
    if source_charts.operator_kind != "full_attention":
        raise Lfm2CpcfLayerCompilerError(
            "Candidate-local sequential transport is attention-only."
        )
    if (
        fit_bank.layer_index != layer_index
        or operator_fit_replay.layer_index != layer_index
        or fit_bank.state_bank_sha256 != operator_fit_replay.input_bank_sha256
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate transport fit bank and operator replay are misbound."
        )
    phases = tuple(fit_bank.phases)
    if not phases or any(phase.split != "fit" for phase in phases):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate post-layer transport accepts fit phases only."
        )
    phase_sqrt_weights = _equal_causal_phase_sqrt_weights(
        phases,
        required_split="fit",
    )
    parent_rows = []
    actual_rows = []
    row_weights = []
    phase_reports = []
    for phase in phases:
        tensors = _phase_evidence(
            phase,
            layer_index,
            source_charts.operator_kind,
        )
        parent = _finite_tensor(
            tensors["residual.post_layer"],
            (
                f"candidate {candidate_id} phase {phase.identity!r} "
                "parent residual.post_layer"
            ),
            dimensions=3,
        )
        if int(parent.shape[0]) != phase.batch_size:
            raise Lfm2CpcfLayerCompilerError(
                "Candidate transport parent batch differs from its phase."
            )
        if int(parent.shape[1]) == phase.full_prefix_tokens:
            parent = phase.suffix(parent)
        elif int(parent.shape[1]) != phase.sampled_suffix_tokens:
            raise Lfm2CpcfLayerCompilerError(
                "Candidate transport parent endpoint is neither the full "
                "prefix nor the authenticated sampled suffix."
            )
        actual = phase.suffix(
            _finite_tensor(
                operator_fit_replay.replays[phase.identity].post_operator_residual,
                (
                    f"candidate {candidate_id} phase {phase.identity!r} "
                    "actual post-operator residual"
                ),
                dimensions=3,
            )
        )
        if (
            int(parent.shape[-1]) != source_charts.source_width
            or int(actual.shape[-1]) != source_charts.target_width
            or tuple(parent.shape[:2]) != tuple(actual.shape[:2])
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Candidate transport parent and actual replay rows are "
                "not boundary aligned."
            )
        weight = phase_sqrt_weights[phase.identity].square().reshape(-1)
        flattened_parent = parent.reshape(-1, int(parent.shape[-1]))
        flattened_actual = actual.reshape(-1, int(actual.shape[-1]))
        if int(weight.numel()) != int(flattened_parent.shape[0]):
            raise Lfm2CpcfLayerCompilerError(
                "Candidate transport phase weights do not cover its suffix rows."
            )
        parent_rows.append(flattened_parent)
        actual_rows.append(flattened_actual)
        row_weights.append(weight)
        phase_reports.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "phase_sha256": phase.phase_sha256,
                "rows": int(flattened_parent.shape[0]),
                "total_row_weight": float(weight.sum().item()),
                "parent_post_layer_sha256": _tensor_sha256(flattened_parent),
                "actual_post_operator_sha256": _tensor_sha256(flattened_actual),
                "row_weights_sha256": _tensor_sha256(weight),
            }
        )
    parent_matrix = torch.cat(parent_rows, dim=0).contiguous()
    actual_matrix = torch.cat(actual_rows, dim=0).contiguous()
    weights = torch.cat(row_weights, dim=0).contiguous()
    phase_masses = tuple(float(value["total_row_weight"]) for value in phase_reports)
    if not all(
        math.isclose(
            value,
            phase_masses[0],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        for value in phase_masses[1:]
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate transport does not assign equal total mass per phase."
        )

    source_charts_payload_before = _boundary_charts_payload_sha256(source_charts)
    provenance_body = {
        "schema": (
            "neural-forge-lfm2-cpcf-attention-candidate-"
            "post-layer-transport-provenance-v1"
        ),
        "candidate_id": candidate_id,
        "layer_index": layer_index,
        "fit_bank_sha256": fit_bank.state_bank_sha256,
        "fit_operator_replay_sha256": (operator_fit_replay.report["report_sha256"]),
        "source_boundary_charts_sha256": source_charts.layer_charts_sha256,
        "source_boundary_charts_payload_sha256": (source_charts_payload_before),
        "fit_parent_post_layer_sha256": _tensor_sha256(parent_matrix),
        "fit_actual_post_operator_sha256": _tensor_sha256(actual_matrix),
        "fit_weights_sha256": _tensor_sha256(weights),
        "fit_phases": phase_reports,
        "equal_total_mass_per_fit_phase": True,
        "policy_report_sha256": policy.as_dict()["report_sha256"],
    }
    provenance_sha = _sha(provenance_body)
    fit_provenance = (
        f"lfm2-cpcf/layer.{layer_index:03d}/attention/"
        f"{candidate_id}/post-layer-fit/{provenance_sha}"
    )
    transport = fit_cpcf_sequential_downstream_chart_transport(
        source_charts.post_operator,
        source_charts.post_layer,
        fit_parent_downstream_states=parent_matrix,
        fit_actual_child_upstream_rows=actual_matrix,
        fit_provenance=fit_provenance,
        fit_weights=weights,
        alignment_tolerance=float(policy.alignment_tolerance),
        orthogonality_tolerance=float(policy.orthogonality_tolerance),
        preservation_tolerance=float(policy.preservation_tolerance),
    )
    effective_charts = _reseal_lfm2_cpcf_layer_boundary_charts(
        source_charts,
        post_layer=transport.downstream_chart,
    )
    if (
        source_charts.post_operator is not effective_charts.post_operator
        or source_charts.post_layer is effective_charts.post_layer
        or _boundary_charts_payload_sha256(source_charts)
        != source_charts_payload_before
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate-local chart transport mutated or reused a source chart."
        )
    body = {
        **provenance_body,
        "provenance_sha256": provenance_sha,
        "fit_provenance": fit_provenance,
        "sequential_transport": transport.as_dict(),
        "sequential_transport_report_sha256": (transport.report["report_sha256"]),
        "effective_boundary_charts_sha256": (effective_charts.layer_charts_sha256),
        "effective_boundary_charts_payload_sha256": (
            _boundary_charts_payload_sha256(effective_charts)
        ),
        "source_post_operator_chart_same_object": True,
        "source_post_layer_chart_unchanged": True,
        "transported_post_layer_chart_candidate_local": True,
        "fit_split_only": True,
        "selection_rows_read_by_transport": 0,
        "selection_rows_read_before_transport_freeze": 0,
        "validation_rows_read": 0,
        "one_closed_form_svd": True,
        "automatic_retry_or_retuning_performed": False,
        "training_steps": 0,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return (
        transport,
        effective_charts,
        MappingProxyType({**body, "report_sha256": _sha(body)}),
    )


def _attention_candidate_selection_key(
    candidate_id: str,
    selection_metrics: Mapping[str, Any],
    selection_context_metrics: Mapping[str, Any],
    query_key_score_report: Lfm2AttentionScorePreservationReport,
) -> tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    str,
]:
    """Rank closed-loop sequence endpoint/drift before context safeguards."""

    if set(selection_metrics) != {"selection"}:
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate ranking requires selection metrics only."
        )
    context = selection_context_metrics.get("aggregate")
    if not isinstance(context, Mapping):
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate lacks selection-context preservation."
        )
    context_relative_error = context.get("relative_rmse")
    context_cosine = context.get("cosine_similarity")
    if (
        isinstance(context_relative_error, bool)
        or not isinstance(context_relative_error, (int, float))
        or not math.isfinite(float(context_relative_error))
        or float(context_relative_error) < 0.0
        or isinstance(context_cosine, bool)
        or not isinstance(context_cosine, (int, float))
        or not math.isfinite(float(context_cosine))
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Attention selection-context metrics are non-finite."
        )
    if not isinstance(
        query_key_score_report,
        Lfm2AttentionScorePreservationReport,
    ):
        raise TypeError("query_key_score_report must be an attention score report.")
    split = selection_metrics["selection"]
    phase_metrics = split.get("phase_metrics")
    if not isinstance(phase_metrics, Mapping) or not phase_metrics:
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate lacks causal-phase selection metrics."
        )
    ordered_phases: list[tuple[int, str]] = []
    for phase_name in phase_metrics:
        if phase_name == "prefill":
            causal_index = 0
        else:
            match = re.fullmatch(r"decode\.(\d{3,})", phase_name)
            if match is None:
                raise Lfm2CpcfLayerCompilerError(
                    f"Attention candidate phase {phase_name!r} is not causal."
                )
            causal_index = int(match.group(1)) + 1
        ordered_phases.append((causal_index, phase_name))
    ordered_phases.sort()
    if ordered_phases[0][1] != "prefill" or [
        value[0] for value in ordered_phases
    ] != list(range(len(ordered_phases))):
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate selection phases are not one complete "
            "prefill-plus-decode trajectory."
        )
    relative_errors = []
    cosines = []
    for _causal_index, phase_name in ordered_phases:
        post_layer = phase_metrics[phase_name].get("post_layer")
        if not isinstance(post_layer, Mapping):
            raise Lfm2CpcfLayerCompilerError(
                f"Attention candidate phase {phase_name!r} lacks post-layer metrics."
            )
        relative_error = post_layer.get("relative_rmse")
        cosine = post_layer.get("cosine_similarity")
        if (
            isinstance(relative_error, bool)
            or not isinstance(relative_error, (int, float))
            or not math.isfinite(float(relative_error))
            or float(relative_error) < 0.0
            or isinstance(cosine, bool)
            or not isinstance(cosine, (int, float))
            or not math.isfinite(float(cosine))
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Attention candidate selection metrics are non-finite."
            )
        relative_errors.append(float(relative_error))
        cosines.append(float(cosine))
    aggregate_metrics = split.get("post_layer")
    if not isinstance(aggregate_metrics, Mapping):
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidate lacks aggregate post-layer metrics."
        )
    aggregate = aggregate_metrics.get("relative_rmse")
    if (
        isinstance(aggregate, bool)
        or not isinstance(aggregate, (int, float))
        or not math.isfinite(float(aggregate))
        or float(aggregate) < 0.0
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Attention aggregate selection metric is invalid."
        )
    maximum_causal_drift = max(
        (
            abs(current - previous)
            for previous, current in zip(
                relative_errors,
                relative_errors[1:],
            )
        ),
        default=0.0,
    )
    return (
        relative_errors[-1],
        maximum_causal_drift,
        max(relative_errors),
        float(aggregate),
        -cosines[-1],
        -min(cosines),
        float(context_relative_error),
        -float(context_cosine),
        float(query_key_score_report.js_divergence_worst),
        -float(query_key_score_report.top1_key_agreement_fraction),
        candidate_id,
    )


@dataclass(frozen=True, slots=True)
class _Lfm2CpcfAttentionCandidateEvaluation:
    candidate_id: str
    operator_artifact: Any
    mlp_artifact: Lfm2CompiledStructuralMlpFold
    effective_boundary_charts: Lfm2CpcfLayerBoundaryCharts
    sequential_boundary_transport: CpcfSequentialBoundaryTransport | None
    sequential_transport_binding: Mapping[str, Any] | None
    operator_transport_mlp_bundle_sha256: str | None
    operator_install_report: Mapping[str, Any]
    mlp_install_report: Mapping[str, Any]
    selection_metrics: Mapping[str, Any]
    selection_context_metrics: Mapping[str, Any]
    query_key_score_report: Lfm2AttentionScorePreservationReport
    installed_operator_state_sha256: str
    installed_mlp_state_sha256: str
    rollback_proof: Mapping[str, Any]
    selection_key: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
    ]
    report: Mapping[str, Any]


def _evaluate_installed_attention_candidate(
    *,
    candidate_id: str,
    operator_artifact: Any,
    operator_install_report: Mapping[str, Any],
    source_layer: Any,
    target_model: Any,
    target_layer: Any,
    layer_index: int,
    fit_bank: Lfm2CpcfPhaseStateBank,
    selection_bank: Lfm2CpcfPhaseStateBank,
    selection_cases: Sequence[Lfm2CachedAttentionFoldCase],
    chart: Lfm2CpcfLayerBoundaryCharts,
    ridge: float,
    prepared_mlp_basis: Lfm2PreparedStructuralMlpFoldBasis | None,
    sequential_transport_policy: (
        Lfm2CpcfAttentionSequentialTransportPolicy | None
    ) = None,
) -> tuple[
    Lfm2CompiledStructuralMlpFold,
    Mapping[str, Any],
    Lfm2CpcfLayerReplayBank,
    Lfm2CpcfLayerReplayBank,
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    str,
    Lfm2PreparedStructuralMlpFoldBasis,
    Lfm2CpcfLayerBoundaryCharts,
    CpcfSequentialBoundaryTransport | None,
    Mapping[str, Any] | None,
]:
    """Fit one candidate's MLP on fit and evaluate its macrocell on selection."""

    target_attention = target_layer.self_attn
    expected_operator_state_sha = _tensor_bundle_sha256(operator_artifact.state_dict())
    installed_operator_state_sha = _tensor_bundle_sha256(
        {
            "q_proj.weight": target_attention.q_proj.weight,
            "k_proj.weight": target_attention.k_proj.weight,
            "v_proj.weight": target_attention.v_proj.weight,
            "out_proj.weight": target_attention.out_proj.weight,
        }
    )
    if expected_operator_state_sha != installed_operator_state_sha:
        raise Lfm2CpcfLayerInstallError(
            f"Attention candidate {candidate_id!r} live Q/K/V/O state does "
            "not match its artifact before fit replay."
        )
    _sha(dict(operator_install_report))
    fit_operator_replay = replay_lfm2_cpcf_phase_bank(
        target_model,
        layer_index=layer_index,
        input_bank=fit_bank,
    )
    sequential_transport = None
    sequential_transport_binding = None
    effective_chart = chart
    if sequential_transport_policy is not None and sequential_transport_policy.enabled:
        if prepared_mlp_basis is not None:
            raise Lfm2CpcfLayerCompilerError(
                f"Attention candidate {candidate_id!r} received a "
                "cross-candidate prepared MLP basis while candidate-local "
                "sequential transport is enabled."
            )
        (
            sequential_transport,
            effective_chart,
            sequential_transport_binding,
        ) = _fit_attention_candidate_post_layer_transport(
            candidate_id=candidate_id,
            layer_index=layer_index,
            fit_bank=fit_bank,
            operator_fit_replay=fit_operator_replay,
            source_charts=chart,
            policy=sequential_transport_policy,
        )
    if prepared_mlp_basis is None:
        prepared_mlp_basis = _prepare_cpcf_mlp_basis_from_fit_replay(
            source_layer=source_layer,
            target_layer=target_layer,
            layer_index=layer_index,
            fit_bank=fit_bank,
            operator_fit_replay=fit_operator_replay,
            chart=effective_chart,
        )
    mlp_artifact = _compile_cpcf_mlp_from_fit_replay(
        source_layer=source_layer,
        target_layer=target_layer,
        layer_index=layer_index,
        fit_bank=fit_bank,
        operator_fit_replay=fit_operator_replay,
        chart=effective_chart,
        ridge=ridge,
        prepared_basis=prepared_mlp_basis,
    )
    if (
        mlp_artifact.prepared_basis_reused is not True
        or mlp_artifact.prepared_basis_sha256 != prepared_mlp_basis.preparation_sha256
    ):
        raise Lfm2CpcfLayerCompilerError(
            f"Attention candidate {candidate_id!r} did not consume the "
            "authenticated invocation-local MLP basis."
        )
    mlp_install_report = install_lfm2_structural_mlp_fold(
        target_layer.feed_forward,
        mlp_artifact,
    ).as_dict()
    expected_mlp_state_sha = _tensor_bundle_sha256(mlp_artifact.state_dict())
    installed_mlp_state_sha = _tensor_bundle_sha256(
        {
            "w1.weight": target_layer.feed_forward.w1.weight,
            "w3.weight": target_layer.feed_forward.w3.weight,
            "w2.weight": target_layer.feed_forward.w2.weight,
        }
    )
    if expected_mlp_state_sha != installed_mlp_state_sha:
        raise Lfm2CpcfLayerInstallError(
            f"Attention candidate {candidate_id!r} live MLP state does not "
            "match its artifact before selection replay."
        )
    _sha(dict(mlp_install_report))
    selection_replay = replay_lfm2_cpcf_phase_bank(
        target_model,
        layer_index=layer_index,
        input_bank=selection_bank,
    )
    selection_metrics = _split_endpoint_metrics(
        layer_index=layer_index,
        input_bank=selection_bank,
        operator_replay=selection_replay,
        final_replay=selection_replay,
        chart=effective_chart,
        splits=("selection",),
    )
    selection_contexts = _attention_actual_contexts(
        input_bank=selection_bank,
        replay_bank=selection_replay,
        required_split="selection",
    )
    selection_context_metrics = _attention_context_preservation_metrics(
        operator_artifact=operator_artifact,
        cases=selection_cases,
        actual_child_contexts=selection_contexts,
    )
    return (
        mlp_artifact,
        mlp_install_report,
        fit_operator_replay,
        selection_replay,
        selection_metrics,
        selection_context_metrics,
        installed_operator_state_sha,
        installed_mlp_state_sha,
        prepared_mlp_basis,
        effective_chart,
        sequential_transport,
        sequential_transport_binding,
    )


def _seal_attention_candidate_evaluation(
    *,
    candidate_id: str,
    operator_artifact: Any,
    mlp_artifact: Lfm2CompiledStructuralMlpFold,
    operator_install_report: Mapping[str, Any],
    mlp_install_report: Mapping[str, Any],
    fit_operator_replay: Lfm2CpcfLayerReplayBank,
    selection_replay: Lfm2CpcfLayerReplayBank,
    selection_metrics: Mapping[str, Any],
    selection_context_metrics: Mapping[str, Any],
    query_key_score_report: Lfm2AttentionScorePreservationReport,
    query_key_score_state_sha256: str,
    installed_operator_state_sha256: str,
    installed_mlp_state_sha256: str,
    rollback_proof: Mapping[str, Any],
    effective_boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    sequential_boundary_transport: CpcfSequentialBoundaryTransport | None,
    sequential_transport_binding: Mapping[str, Any] | None,
    candidate_ids: Sequence[str] = LFM2_CPCF_ATTENTION_CANDIDATE_IDS,
) -> _Lfm2CpcfAttentionCandidateEvaluation:
    active_candidate_ids = _canonical_attention_candidate_ids(candidate_ids)
    if candidate_id not in active_candidate_ids:
        raise Lfm2CpcfLayerCompilerError(
            f"Unknown attention candidate {candidate_id!r}."
        )
    selection_key = _attention_candidate_selection_key(
        candidate_id,
        selection_metrics,
        selection_context_metrics,
        query_key_score_report,
    )
    operator_state_sha = _digest(
        installed_operator_state_sha256,
        "installed_operator_state_sha256",
    )
    mlp_state_sha = _digest(
        installed_mlp_state_sha256,
        "installed_mlp_state_sha256",
    )
    score_state_sha = _digest(
        query_key_score_state_sha256,
        "query_key_score_state_sha256",
    )
    artifact_state = operator_artifact.state_dict()
    artifact_query_key_sha = _tensor_bundle_sha256(
        {
            "q_proj.weight": artifact_state["q_proj.weight"],
            "k_proj.weight": artifact_state["k_proj.weight"],
        }
    )
    if artifact_query_key_sha != score_state_sha:
        raise Lfm2CpcfLayerCompilerError(
            "Attention score report producer view differs from the "
            "candidate's exact Q/K state."
        )
    if not isinstance(
        effective_boundary_charts,
        Lfm2CpcfLayerBoundaryCharts,
    ):
        raise TypeError(
            "effective_boundary_charts must be Lfm2CpcfLayerBoundaryCharts."
        )
    transport_enabled = sequential_boundary_transport is not None
    if transport_enabled != (sequential_transport_binding is not None):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate sequential transport and binding must be supplied together."
        )
    operator_transport_mlp_bundle_sha256 = None
    if transport_enabled:
        if not isinstance(
            sequential_boundary_transport,
            CpcfSequentialBoundaryTransport,
        ):
            raise TypeError("sequential_boundary_transport has the wrong type.")
        binding = dict(sequential_transport_binding)
        binding_body = dict(binding)
        binding_seal = binding_body.pop("report_sha256", None)
        if (
            binding_seal != _sha(binding_body)
            or binding.get("candidate_id") != candidate_id
            or binding.get("effective_boundary_charts_sha256")
            != effective_boundary_charts.layer_charts_sha256
            or binding.get("sequential_transport_report_sha256")
            != sequential_boundary_transport.report["report_sha256"]
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Candidate sequential transport binding is invalid."
            )
        operator_transport_mlp_bundle_sha256 = _sha(
            {
                "candidate_id": candidate_id,
                "operator_artifact_sha256": _sha(operator_artifact.as_dict()),
                "installed_operator_state_sha256": operator_state_sha,
                "sequential_transport_report_sha256": (
                    sequential_boundary_transport.report["report_sha256"]
                ),
                "effective_boundary_charts_sha256": (
                    effective_boundary_charts.layer_charts_sha256
                ),
                "effective_boundary_charts_payload_sha256": (
                    _boundary_charts_payload_sha256(effective_boundary_charts)
                ),
                "mlp_artifact_sha256": _sha(mlp_artifact.as_dict()),
                "installed_mlp_state_sha256": mlp_state_sha,
            }
        )
    body = {
        "schema": "neural-forge-lfm2-cpcf-attention-candidate-evaluation-v1",
        "candidate_id": candidate_id,
        "operator_artifact_schema": operator_artifact.as_dict().get("schema"),
        "operator_artifact": operator_artifact.as_dict(),
        "operator_artifact_sha256": _sha(operator_artifact.as_dict()),
        "mlp_artifact_sha256": _sha(mlp_artifact.as_dict()),
        "operator_install_report": dict(operator_install_report),
        "operator_install_report_sha256": _sha(dict(operator_install_report)),
        "mlp_install_report": dict(mlp_install_report),
        "mlp_install_report_sha256": _sha(dict(mlp_install_report)),
        "installed_operator_state_sha256": operator_state_sha,
        "installed_mlp_state_sha256": mlp_state_sha,
        "fit_operator_replay_sha256": (fit_operator_replay.report["report_sha256"]),
        "selection_replay_sha256": (selection_replay.report["report_sha256"]),
        "selection_metrics": {"selection": dict(selection_metrics["selection"])},
        "selection_context_metrics": dict(selection_context_metrics),
        "query_key_score_report": query_key_score_report.as_dict(),
        "query_key_score_report_sha256": (query_key_score_report.report_sha256),
        "query_key_score_state_sha256": score_state_sha,
        "query_key_score_producer_view_exact_state_proven": True,
        "selection_key": list(selection_key),
        "rollback_proof": dict(rollback_proof),
        "operator_compiled_from_splits": ["fit"],
        "mlp_compiled_from_splits": ["fit"],
        "candidate_evaluated_on_splits": ["selection"],
        "validation_phase_count_consumed": 0,
        "live_operator_state_authenticated_before_fit_replay": True,
        "live_mlp_state_authenticated_before_selection_replay": True,
        "sequence_endpoint_ranked_before_context_reconstruction": True,
        "maximum_consecutive_causal_phase_drift_ranked": True,
        "selection_context_used_as_finite_safeguard": True,
        "whole_macrocell_post_layer_causal_phase_ranking": True,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    if transport_enabled:
        body = {
            **body,
            "sequential_boundary_transport": dict(sequential_boundary_transport.report),
            "sequential_transport_binding": dict(sequential_transport_binding),
            "effective_boundary_charts_sha256": (
                effective_boundary_charts.layer_charts_sha256
            ),
            "effective_boundary_charts_payload_sha256": (
                _boundary_charts_payload_sha256(effective_boundary_charts)
            ),
            "operator_transport_mlp_bundle_sha256": (
                operator_transport_mlp_bundle_sha256
            ),
            "candidate_local_post_layer_transport_frozen_before_mlp": True,
            "candidate_local_mlp_basis_preparation_count": 1,
            "cross_candidate_prepared_mlp_basis_reused": False,
            "selection_metrics_use_effective_boundary_charts": True,
            "operator_transport_and_mlp_ranked_as_one_bundle": True,
        }
    report = MappingProxyType({**body, "report_sha256": _sha(body)})
    return _Lfm2CpcfAttentionCandidateEvaluation(
        candidate_id=candidate_id,
        operator_artifact=operator_artifact,
        mlp_artifact=mlp_artifact,
        effective_boundary_charts=effective_boundary_charts,
        sequential_boundary_transport=sequential_boundary_transport,
        sequential_transport_binding=(
            None
            if sequential_transport_binding is None
            else MappingProxyType(dict(sequential_transport_binding))
        ),
        operator_transport_mlp_bundle_sha256=(operator_transport_mlp_bundle_sha256),
        operator_install_report=MappingProxyType(dict(operator_install_report)),
        mlp_install_report=MappingProxyType(dict(mlp_install_report)),
        selection_metrics=MappingProxyType(dict(selection_metrics)),
        selection_context_metrics=MappingProxyType(dict(selection_context_metrics)),
        query_key_score_report=query_key_score_report,
        installed_operator_state_sha256=operator_state_sha,
        installed_mlp_state_sha256=mlp_state_sha,
        rollback_proof=MappingProxyType(dict(rollback_proof)),
        selection_key=selection_key,
        report=report,
    )


def _freeze_attention_candidate_winner(
    candidates: Sequence[_Lfm2CpcfAttentionCandidateEvaluation],
    *,
    candidate_ids: Sequence[str] = LFM2_CPCF_ATTENTION_CANDIDATE_IDS,
) -> tuple[
    _Lfm2CpcfAttentionCandidateEvaluation,
    Mapping[str, Any],
]:
    values = tuple(candidates)
    active_candidate_ids = _canonical_attention_candidate_ids(candidate_ids)
    if {value.candidate_id for value in values} != set(active_candidate_ids) or len(
        values
    ) != len(active_candidate_ids):
        inventory_label = {
            3: "three-arm",
            4: "four-arm",
            5: "five-arm",
        }.get(len(active_candidate_ids), "noncanonical")
        raise Lfm2CpcfLayerCompilerError(
            "Attention selection requires exactly the authenticated "
            f"{inventory_label} macrocell inventory."
        )
    transport_flags = tuple(
        value.sequential_boundary_transport is not None for value in values
    )
    if any(transport_flags) and not all(transport_flags):
        raise Lfm2CpcfLayerCompilerError(
            "Attention candidates cannot mix transported and untransported "
            "post-layer chart policies."
        )
    transport_enabled = all(transport_flags)
    if transport_enabled:
        if any(
            value.operator_transport_mlp_bundle_sha256 is None
            or value.sequential_transport_binding is None
            for value in values
        ) or len({id(value.effective_boundary_charts) for value in values}) != len(
            values
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Candidate-local transported macrocell bundles are incomplete "
                "or not candidate distinct."
            )
    winner = min(values, key=lambda value: value.selection_key)
    by_id = {value.candidate_id: value for value in values}
    ordered = [by_id[candidate_id] for candidate_id in active_candidate_ids]
    body = {
        "schema": LFM2_CPCF_ATTENTION_CANDIDATE_SELECTION_SCHEMA,
        "ranking_rule": [
            "minimum_final_causal_phase_post_layer_relative_rmse",
            "minimum_maximum_consecutive_causal_phase_relative_rmse_drift",
            "minimum_worst_causal_phase_post_layer_relative_rmse",
            "minimum_aggregate_post_layer_relative_rmse",
            "maximum_final_causal_phase_post_layer_cosine_similarity",
            "maximum_worst_causal_phase_post_layer_cosine_similarity",
            "minimum_selection_context_relative_rmse_safeguard",
            "maximum_selection_context_cosine_similarity_safeguard",
            "minimum_selection_query_key_js_divergence_safeguard",
            "maximum_selection_query_key_top1_agreement_safeguard",
            "lexical_candidate_id_tie_break",
        ],
        "candidate_count": len(ordered),
        "candidate_ids": [value.candidate_id for value in ordered],
        "candidate_report_sha256": {
            value.candidate_id: value.report["report_sha256"] for value in ordered
        },
        "selected_candidate_id": winner.candidate_id,
        "selected_candidate_report_sha256": (winner.report["report_sha256"]),
        "selection_split_only": True,
        "winner_frozen_before_validation": True,
        "validation_phase_count_consumed_before_freeze": 0,
        "operator_and_own_fit_mlp_ranked_together": True,
        "selection_objective": (
            "closed_loop_sequence_endpoint_then_causal_drift_with_context_"
            "and_query_key_finite_safeguards"
        ),
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    if transport_enabled:
        body = {
            **body,
            "candidate_effective_boundary_charts_sha256": {
                value.candidate_id: (
                    value.effective_boundary_charts.layer_charts_sha256
                )
                for value in ordered
            },
            "candidate_operator_transport_mlp_bundle_sha256": {
                value.candidate_id: (value.operator_transport_mlp_bundle_sha256)
                for value in ordered
            },
            "selected_effective_boundary_charts_sha256": (
                winner.effective_boundary_charts.layer_charts_sha256
            ),
            "selected_operator_transport_mlp_bundle_sha256": (
                winner.operator_transport_mlp_bundle_sha256
            ),
            "candidate_local_post_layer_transport_enabled": True,
            "candidate_local_mlp_preparation_count": len(ordered),
            "cross_candidate_prepared_mlp_basis_reused": False,
            "operator_transport_mlp_bundle_frozen_together": True,
            "selection_metrics_use_candidate_effective_charts": True,
            "selection_refits_or_mutates_transport": False,
            "selection_selects_frozen_bundle": True,
        }
    return (
        winner,
        MappingProxyType({**body, "report_sha256": _sha(body)}),
    )


def _macrocell_zero_execution() -> dict[str, int]:
    return {
        "parent_model_forward_calls": 0,
        "target_model_forward_calls": 0,
        "generation_calls": 0,
        "training_steps": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }


def _plain_json_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _plain_json_metadata(child) for name, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_plain_json_metadata(child) for child in value]
    return value


def _select_short_conv_phase_state_bank(
    bank: Lfm2CpcfPhaseStateBank,
    *,
    splits: Sequence[str],
) -> Lfm2CpcfPhaseStateBank:
    """Reseal one short-conv split whose prefill evidence may be suffix sampled."""

    requested = tuple(str(value) for value in splits)
    if not requested or len(set(requested)) != len(requested):
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv split selection must be nonempty and unique."
        )
    phases = tuple(phase for phase in bank.phases if phase.split in set(requested))
    if not phases:
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv phase bank has no requested phases."
        )
    states = {phase.identity: bank.state(phase) for phase in phases}
    body = {
        "schema": "neural-forge-lfm2-cpcf-target-runtime-v1",
        "scope": "phase_state_bank",
        "boundary": bank.boundary,
        "layer_index": bank.layer_index,
        "hidden_size": bank.hidden_size,
        "ordered_states": [
            {
                "identity": [phase.batch_index, phase.phase],
                "phase_sha256": phase.phase_sha256,
                "state_sha256": _tensor_sha256(states[phase.identity]),
            }
            for phase in phases
        ],
        "actual_folded_upstream_states": True,
    }
    return Lfm2CpcfPhaseStateBank(
        boundary=bank.boundary,
        layer_index=bank.layer_index,
        phases=phases,
        states=states,
        state_bank_sha256=_sha(body),
    )


def _seal_macrocell_actual(
    kind: str,
    candidate_id: str,
    *,
    source_report_sha256: str,
    **bindings: Any,
) -> dict[str, Any]:
    body = {
        "schema": "neural-forge-lfm2-cpcf-layer-macrocell-actual-v1",
        "kind": kind,
        "candidate_id": candidate_id,
        "source_report_sha256": _digest(
            source_report_sha256,
            f"{kind}.source_report_sha256",
        ),
        **bindings,
        "execution": _macrocell_zero_execution(),
    }
    return {**body, "report_sha256": _sha(body)}


def _seal_macrocell_evidence(
    kind: str,
    candidate_id: str,
    actual_report: Mapping[str, Any],
    **bindings: Any,
) -> dict[str, Any]:
    actual = dict(actual_report)
    actual_body = dict(actual)
    actual_sha = _digest(
        actual_body.pop("report_sha256", None),
        f"{kind}.actual_report.report_sha256",
    )
    if _sha(actual_body) != actual_sha:
        raise Lfm2CpcfLayerCompilerError(
            f"{kind} actual macrocell evidence is not sealed."
        )
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA,
        "kind": kind,
        "candidate_id": candidate_id,
        **bindings,
        "actual_report": actual,
        "actual_report_sha256": actual_sha,
        "execution": _macrocell_zero_execution(),
    }
    return {**body, "report_sha256": _sha(body)}


def _install_short_conv_macrocell_candidate(
    target_short_conv: Any,
    candidate: Lfm2CpcfShortConvV5MacrocellCandidate,
) -> Mapping[str, Any]:
    """Install one detached native candidate without replacing Parameters."""

    torch = _torch()
    if (
        not isinstance(candidate, Lfm2CpcfShortConvV5MacrocellCandidate)
        or not candidate.verify_authentication()
    ):
        raise Lfm2CpcfLayerInstallError(
            "Short-conv macrocell candidate authentication failed."
        )
    available = {
        "in_proj.weight": getattr(target_short_conv.in_proj, "weight", None),
        "in_proj.bias": getattr(target_short_conv.in_proj, "bias", None),
        "conv.weight": getattr(target_short_conv.conv, "weight", None),
        "conv.bias": getattr(target_short_conv.conv, "bias", None),
        "out_proj.weight": getattr(target_short_conv.out_proj, "weight", None),
        "out_proj.bias": getattr(target_short_conv.out_proj, "bias", None),
    }
    expected = candidate.install_state_dict()
    if set(expected) != {
        name for name, value in available.items() if isinstance(value, torch.Tensor)
    }:
        raise Lfm2CpcfLayerInstallError(
            "Macrocell candidate and target short-conv bias topology differ."
        )
    for name, expected_value in expected.items():
        observed = available[name]
        if (
            not isinstance(observed, torch.Tensor)
            or tuple(observed.shape) != tuple(expected_value.shape)
            or observed.dtype != expected_value.dtype
        ):
            raise Lfm2CpcfLayerInstallError(
                f"Macrocell candidate {name} differs in shape or dtype."
            )
    identities = {name: id(available[name]) for name in expected}
    gradients = {
        name: (
            bool(available[name].requires_grad),
            None if available[name].grad is None else id(available[name].grad),
            None
            if available[name].grad is None
            else _tensor_sha256(available[name].grad),
        )
        for name in expected
    }
    with torch.no_grad():
        for name, expected_value in expected.items():
            available[name].copy_(
                expected_value.to(
                    device=available[name].device,
                    dtype=available[name].dtype,
                )
            )
    installed_sha = _short_conv_state_sha256(
        {name: available[name] for name in expected}
    )
    body = {
        "scope": "exact_native_short_conv_macrocell_candidate_install",
        "candidate_id": candidate.candidate_id,
        "candidate_state_sha256": candidate.state_sha256,
        "compiled_state_sha256": candidate.report["candidate_install_state_sha256"],
        "installed_state_sha256": installed_sha,
        "parameter_identities_preserved": all(
            id(available[name]) == identities[name] for name in expected
        ),
        "parameter_shapes_preserved": True,
        "gradient_state_preserved": all(
            (
                bool(available[name].requires_grad),
                None if available[name].grad is None else id(available[name].grad),
                None
                if available[name].grad is None
                else _tensor_sha256(available[name].grad),
            )
            == gradients[name]
            for name in expected
        ),
        "training_steps": 0,
    }
    if (
        body["compiled_state_sha256"] != installed_sha
        or not body["parameter_identities_preserved"]
        or not body["gradient_state_preserved"]
    ):
        raise Lfm2CpcfLayerInstallError(
            "Live macrocell candidate state failed exact authentication."
        )
    return MappingProxyType({**body, "report_sha256": _sha(body)})


def _normalized_macrocell_selection_metrics(
    candidate_id: str,
    selection_bank_sha256: str,
    split_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if set(split_metrics) != {"selection"}:
        raise Lfm2CpcfLayerCompilerError(
            "Short-conv macrocell ranking requires selection metrics only."
        )
    source = split_metrics["selection"]

    def endpoint(value: Mapping[str, Any]) -> dict[str, float]:
        return {
            "relative_rmse": float(value["relative_rmse"]),
            "cosine_similarity": float(value["cosine_similarity"]),
        }

    phase_metrics = {}
    for phase_name, value in source["phase_metrics"].items():
        phase_body = {
            "phase": phase_name,
            "phase_count": int(value["phase_count"]),
            "phase_sha256": list(value["phase_sha256"]),
            "post_operator": endpoint(value["post_operator"]),
            "post_layer": endpoint(value["post_layer"]),
        }
        phase_metrics[phase_name] = {
            **phase_body,
            "report_sha256": _sha(phase_body),
        }
    actual_body = {
        "split": "selection",
        "phase_count": int(source["phase_count"]),
        "phase_sha256": list(source["phase_sha256"]),
        "post_operator": endpoint(source["post_operator"]),
        "post_layer": endpoint(source["post_layer"]),
        "phase_metrics": phase_metrics,
        "source_report_sha256": source["report_sha256"],
        "execution": _macrocell_zero_execution(),
    }
    actual = {**actual_body, "report_sha256": _sha(actual_body)}
    body = {
        "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA,
        "kind": "actual_selection_metrics",
        "candidate_id": candidate_id,
        "selection_bank_sha256": selection_bank_sha256,
        "split": "selection",
        "phase_count": actual["phase_count"],
        "phase_sha256": actual["phase_sha256"],
        "post_operator": actual["post_operator"],
        "post_layer": actual["post_layer"],
        "phase_metrics": actual["phase_metrics"],
        "actual_report": actual,
        "actual_report_sha256": actual["report_sha256"],
        "execution": _macrocell_zero_execution(),
    }
    return {**body, "report_sha256": _sha(body)}


def _short_conv_macrocell_score(
    candidate_id: str,
    metrics: Mapping[str, Any],
) -> tuple[float | str, ...]:
    phases = tuple(metrics["phase_metrics"].values())
    return (
        max(float(value["post_layer"]["relative_rmse"]) for value in phases),
        -min(float(value["post_layer"]["cosine_similarity"]) for value in phases),
        float(metrics["post_layer"]["relative_rmse"]),
        -float(metrics["post_layer"]["cosine_similarity"]),
        candidate_id,
    )


@dataclass(frozen=True, slots=True)
class _Lfm2CpcfShortConvExecutionLane:
    lane_index: int
    device: Any
    device_index: int
    device_uuid: str
    device_receipt_sha256: str
    replica_receipt_sha256: str
    installed_state_fingerprint_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "lane_index": self.lane_index,
            "device": str(self.device),
            "device_index": self.device_index,
            "device_uuid": self.device_uuid,
            "device_receipt_sha256": self.device_receipt_sha256,
            "replica_receipt_sha256": self.replica_receipt_sha256,
            "installed_state_fingerprint_sha256": (
                self.installed_state_fingerprint_sha256
            ),
        }


@dataclass(frozen=True, slots=True)
class _Lfm2CpcfShortConvTwoLaneRuntime:
    lanes: tuple[
        _Lfm2CpcfShortConvExecutionLane,
        _Lfm2CpcfShortConvExecutionLane,
    ]
    runtime_authentication_report: Mapping[str, Any]
    runtime_authentication_report_sha256: str
    scheduler_policy_sha256: str


def _authenticated_short_conv_two_lane_runtime(
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None,
) -> _Lfm2CpcfShortConvTwoLaneRuntime | None:
    """Derive two explicit devices solely from a freshly sealed scheduler."""

    scheduler = getattr(candidate_replay_evaluator, "scheduler", None)
    if scheduler is None:
        return None
    if (
        type(candidate_replay_evaluator)
        is not AuthenticatedTwoGpuLfm2CpcfCandidateReplayEvaluator
        or type(scheduler) is not DeterministicReplayScheduler
    ):
        raise Lfm2CpcfLayerCompilerError(
            "A requested macrocell scheduler must be the exact authenticated "
            "two-CUDA replay runtime."
        )
    runtime_value = scheduler.authenticate_runtime()
    runtime_report = _plain_json_metadata(runtime_value)
    runtime_body = dict(runtime_report)
    runtime_sha = _digest(
        runtime_body.pop("report_sha256", None),
        "two_lane_runtime.report_sha256",
    )
    replicas = scheduler.replicas
    policy = scheduler.policy
    if (
        _sha(runtime_body) != runtime_sha
        or runtime_report.get("schema")
        != "neural-forge-dynamic-replay-live-runtime-authentication-v1"
        or runtime_report.get("device_receipts_freshly_probed") is not True
        or runtime_report.get("installed_states_freshly_rehashed") is not True
        or not isinstance(replicas, tuple)
        or len(replicas) != 2
        or policy.expected_device_count != 2
        or runtime_report.get("replicas")
        != [_plain_json_metadata(replica.as_dict()) for replica in replicas]
        or runtime_report.get("policy") != _plain_json_metadata(policy.as_dict())
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Fresh two-lane scheduler authentication is incomplete or changed."
        )
    torch = _torch()
    lanes = []
    seen_indices: set[int] = set()
    seen_uuids: set[str] = set()
    seen_pci: set[tuple[int, int, int]] = set()
    for lane_index, replica in enumerate(replicas):
        if type(replica) is not ReplayReplicaReceipt:
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane execution requires exact sealed replay replicas."
            )
        device_receipt = replica.device
        installed_state = replica.installed_state
        device_index = device_receipt.device_index
        pci_identity = (
            device_receipt.pci_domain_id,
            device_receipt.pci_bus_id,
            device_receipt.pci_device_id,
        )
        if (
            replica.receipt_sha256 != _sha(replica._body())
            or device_receipt.receipt_sha256 != _sha(device_receipt._body())
            or installed_state.fingerprint_sha256 != _sha(installed_state._body())
            or isinstance(device_index, bool)
            or not isinstance(device_index, int)
            or device_index < 0
            or device_index in seen_indices
            or device_receipt.device_uuid in seen_uuids
            or pci_identity in seen_pci
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane CUDA replica identity or receipt authentication failed."
            )
        explicit_device = torch.device(f"cuda:{device_index}")
        if explicit_device.type != "cuda" or explicit_device.index != device_index:
            raise Lfm2CpcfLayerCompilerError(
                "Authenticated scheduler did not resolve an explicit CUDA device."
            )
        seen_indices.add(device_index)
        seen_uuids.add(device_receipt.device_uuid)
        seen_pci.add(pci_identity)
        lanes.append(
            _Lfm2CpcfShortConvExecutionLane(
                lane_index=lane_index,
                device=explicit_device,
                device_index=device_index,
                device_uuid=device_receipt.device_uuid,
                device_receipt_sha256=device_receipt.receipt_sha256,
                replica_receipt_sha256=replica.receipt_sha256,
                installed_state_fingerprint_sha256=(installed_state.fingerprint_sha256),
            )
        )
    if len(lanes) != 2:
        raise Lfm2CpcfLayerCompilerError(
            "Two-lane execution requires exactly two authenticated devices."
        )
    return _Lfm2CpcfShortConvTwoLaneRuntime(
        lanes=(lanes[0], lanes[1]),
        runtime_authentication_report=MappingProxyType(runtime_report),
        runtime_authentication_report_sha256=runtime_sha,
        scheduler_policy_sha256=_sha(runtime_report["policy"]),
    )


def _replay_archive_logical_bytes(replay_bank: Lfm2CpcfLayerReplayBank) -> int:
    """Count unique archived tensor storages retained by one replay tree."""

    torch = _torch()
    tensors = []
    for replay in replay_bank.replays.values():
        tensors.extend(
            (
                replay.operator_input,
                replay.operator_delta,
                replay.post_operator_residual,
                replay.mlp_input,
                replay.mlp_delta,
                replay.post_layer_residual,
            )
        )
    tensors.extend(replay_bank.post_layer_bank.states.values())
    seen: set[tuple[str, int | None, int]] = set()
    total = 0
    for value in tensors:
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        key = (
            value.device.type,
            value.device.index,
            int(value.untyped_storage().data_ptr()),
        )
        if key in seen:
            continue
        seen.add(key)
        total += int(value.untyped_storage().nbytes())
    return total


def _macrocell_replay_semantic_signature(
    replay_bank: Lfm2CpcfLayerReplayBank,
) -> Mapping[str, Any]:
    """Seal lane-independent replay endpoints and the causal output bank."""

    expected_endpoints = {
        "operator_input",
        "operator_output_input",
        "operator_delta",
        "post_operator_residual",
        "mlp_input",
        "mlp_delta",
        "post_layer_residual",
    }
    ordered = []
    for phase in replay_bank.post_layer_bank.phases:
        replay = replay_bank.replays[phase.identity]
        endpoint_sha256 = replay.report.get("endpoint_sha256")
        if (
            not isinstance(endpoint_sha256, Mapping)
            or set(endpoint_sha256) != expected_endpoints
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane semantic parity lacks one complete replay endpoint "
                "hash bundle."
            )
        normalized_endpoints = {
            str(name): _digest(
                endpoint_sha256[name],
                f"replay.endpoint_sha256.{name}",
            )
            for name in sorted(expected_endpoints)
        }
        ordered.append(
            {
                "identity": [phase.batch_index, phase.phase],
                "endpoint_sha256": normalized_endpoints,
                "endpoint_bundle_sha256": _sha(normalized_endpoints),
            }
        )
    body = {
        "schema": ("neural-forge-lfm2-cpcf-short-conv-replay-semantic-signature-v1"),
        "layer_index": replay_bank.layer_index,
        "input_bank_sha256": replay_bank.input_bank_sha256,
        "ordered_phase_endpoint_sha256": ordered,
        "post_layer_bank_sha256": (replay_bank.post_layer_bank.state_bank_sha256),
    }
    return MappingProxyType({**body, "signature_sha256": _sha(body)})


def _private_two_lane_replicas(
    target_layer: Any,
    runtime: _Lfm2CpcfShortConvTwoLaneRuntime,
) -> tuple[Any, Any]:
    """Create exactly two layer-only copies and prove storage isolation."""

    replicas = tuple(
        copy.deepcopy(target_layer).to(device=lane.device) for lane in runtime.lanes
    )
    if len(replicas) != 2 or replicas[0] is replicas[1]:
        raise Lfm2CpcfLayerCompilerError(
            "Two-lane macrocell execution did not create exactly two replicas."
        )
    canonical_tensors = _module_tensors(target_layer)
    replica_tensors = tuple(_module_tensors(replica) for replica in replicas)
    expected = {
        name: (tuple(value.shape), value.dtype)
        for name, value in canonical_tensors.items()
    }
    storage_sets = []
    for lane, values in zip(runtime.lanes, replica_tensors, strict=True):
        observed = {
            name: (tuple(value.shape), value.dtype) for name, value in values.items()
        }
        devices = {value.device for value in values.values()}
        if observed != expected or devices != {lane.device}:
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane target replica topology or device differs from its "
                "canonical layer contract."
            )
        storage_sets.append(
            {
                (
                    value.device.type,
                    value.device.index,
                    int(value.untyped_storage().data_ptr()),
                )
                for value in values.values()
                if value.numel() != 0
            }
        )
    canonical_storage = {
        (
            value.device.type,
            value.device.index,
            int(value.untyped_storage().data_ptr()),
        )
        for value in canonical_tensors.values()
        if value.numel() != 0
    }
    if (
        canonical_storage & storage_sets[0]
        or canonical_storage & storage_sets[1]
        or storage_sets[0] & storage_sets[1]
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Two-lane target replicas share tensor storage with another lane "
            "or the canonical target layer."
        )
    return replicas[0], replicas[1]


@dataclass(frozen=True, slots=True)
class _Lfm2CpcfShortConvMacrocellEvaluation:
    candidate: Lfm2CpcfShortConvV5MacrocellCandidate
    mlp_artifact: Lfm2CompiledStructuralMlpFold
    selection_key: tuple[float | str, ...]
    report: Mapping[str, Any]


def _prove_exact_cross_lane_macrocell_parity(
    left: _Lfm2CpcfShortConvMacrocellEvaluation,
    right: _Lfm2CpcfShortConvMacrocellEvaluation,
    *,
    left_signature: Mapping[str, Any],
    right_signature: Mapping[str, Any],
    runtime: _Lfm2CpcfShortConvTwoLaneRuntime,
) -> Mapping[str, Any]:
    """Fail closed unless both real lanes produce byte-identical semantics."""

    signatures = []
    for lane, evaluation, signature_value in zip(
        runtime.lanes,
        (left, right),
        (left_signature, right_signature),
        strict=True,
    ):
        if not isinstance(signature_value, Mapping):
            raise Lfm2CpcfLayerCompilerError(
                "Cross-lane macrocell parity lacks a semantic signature."
            )
        signature = _plain_json_metadata(signature_value)
        signature_body = dict(signature)
        signature_sha256 = _digest(
            signature_body.pop("signature_sha256", None),
            "semantic_parity_signature.signature_sha256",
        )
        if (
            signature.get("schema")
            != LFM2_CPCF_SHORT_CONV_TWO_LANE_SEMANTIC_PARITY_SCHEMA
            or _sha(signature_body) != signature_sha256
            or signature.get("candidate_id") != evaluation.candidate.candidate_id
            or signature.get("candidate_state_sha256")
            != evaluation.candidate.state_sha256
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Lane {lane.lane_index} semantic parity signature is unauthenticated."
            )
        signatures.append(signature)
    if _canonical_json(signatures[0]) != _canonical_json(signatures[1]):
        raise Lfm2CpcfLayerCompilerError(
            "Cross-lane macrocell semantic parity failed exact comparison."
        )
    representative_id = left.candidate.candidate_id
    if representative_id != right.candidate.candidate_id:
        raise Lfm2CpcfLayerCompilerError(
            "Cross-lane macrocell parity evaluated different candidates."
        )
    body = {
        "schema": ("neural-forge-lfm2-cpcf-short-conv-cross-lane-parity-proof-v1"),
        "representative_candidate_id": representative_id,
        "representative_candidate_state_sha256": left.candidate.state_sha256,
        "semantic_signature_sha256": signatures[0]["signature_sha256"],
        "lane_evaluation_report_sha256": [
            left.report["report_sha256"],
            right.report["report_sha256"],
        ],
        "comparison_policy": "exact_canonical_json_byte_equality",
        "exact_semantic_parity_proven": True,
        "numeric_tolerances_used": False,
        "relative_tolerance": 0.0,
        "absolute_tolerance": 0.0,
        "same_two_authenticated_execution_lanes_used": True,
        "cpu_parity_replica_used": False,
        "third_replica_used": False,
        "representative_evaluation_count": 2,
        "lane_zero_representative_result_reused": True,
        "lane_one_representative_result_discarded_after_parity": True,
        "training_steps": 0,
        "parameters_pruned": 0,
        "parameters_quantized": 0,
    }
    return MappingProxyType({**body, "report_sha256": _sha(body)})


class _Lfm2CpcfShortConvMacrocellSelector:
    """Invocation-local real fit/selection whole-macrocell selector."""

    def __init__(
        self,
        *,
        source_layer: Any,
        target_model: Any,
        target_layer: Any,
        layer_index: int,
        audit_bank: Lfm2CpcfPhaseStateBank,
        fit_bank: Lfm2CpcfPhaseStateBank,
        selection_bank: Lfm2CpcfPhaseStateBank,
        boundary_charts: Lfm2CpcfLayerBoundaryCharts,
        ridge: float,
        source_snapshot: _LayerSnapshot,
        target_snapshot: _LayerSnapshot,
        audit_evidence_sha256_before: str,
        audit_input_sha256_before: str,
        boundary_charts_sha256_before: str,
        boundary_charts_payload_sha256_before: str,
        candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None,
        endpoint_selector: Lfm2CpcfShortConvV5CandidateSelector | None,
    ) -> None:
        self.source_layer = source_layer
        self.target_model = target_model
        self.target_layer = target_layer
        self.layer_index = layer_index
        self.audit_bank = audit_bank
        self.fit_bank = fit_bank
        self.selection_bank = selection_bank
        self.boundary_charts = boundary_charts
        self.ridge = ridge
        self.source_snapshot = source_snapshot
        self.target_snapshot = target_snapshot
        self.audit_evidence_sha256_before = audit_evidence_sha256_before
        self.audit_input_sha256_before = audit_input_sha256_before
        self.boundary_charts_sha256_before = boundary_charts_sha256_before
        self.boundary_charts_payload_sha256_before = (
            boundary_charts_payload_sha256_before
        )
        self.candidate_replay_evaluator = candidate_replay_evaluator
        self.endpoint_selector = endpoint_selector
        self.endpoint_control_id: str | None = None
        self.endpoint_selector_report: Mapping[str, Any] | None = None
        self.endpoint_selector_diagnostic: Mapping[str, Any] | None = None
        self.evaluations: dict[
            str,
            _Lfm2CpcfShortConvMacrocellEvaluation,
        ] = {}
        self.selected: _Lfm2CpcfShortConvMacrocellEvaluation | None = None
        self.decision: Mapping[str, Any] | None = None
        self.two_lane_execution_receipt: Mapping[str, Any] | None = None
        self.mlp_prepared_basis: Lfm2PreparedStructuralMlpFoldBasis | None = None
        self.mlp_preparation_count = 0
        self.mlp_candidate_compile_count = 0

    def _prepare_two_lane_mlp_basis(
        self,
        *,
        candidate: Lfm2CpcfShortConvV5MacrocellCandidate,
        execution_layer: Any,
        execution_snapshot: _LayerSnapshot,
    ) -> tuple[int, Mapping[str, Any]]:
        """Prepare the immutable parent-side MLP basis exactly once."""

        fit_replay: Lfm2CpcfLayerReplayBank | None = None
        peak_replay_bytes = 0
        _restore_layer(execution_layer, execution_snapshot)
        try:
            _install_short_conv_macrocell_candidate(
                execution_layer.conv,
                candidate,
            )
            fit_replay = replay_lfm2_cpcf_phase_bank(
                self.target_model,
                layer_index=self.layer_index,
                input_bank=self.fit_bank,
                execution_layer=execution_layer,
            )
            peak_replay_bytes = _replay_archive_logical_bytes(fit_replay)
            self.mlp_prepared_basis = _prepare_cpcf_mlp_basis_from_fit_replay(
                source_layer=self.source_layer,
                target_layer=execution_layer,
                layer_index=self.layer_index,
                fit_bank=self.fit_bank,
                operator_fit_replay=fit_replay,
                chart=self.boundary_charts,
            )
            self.mlp_preparation_count += 1
            del fit_replay
            fit_replay = None
        finally:
            _restore_layer(execution_layer, execution_snapshot)
        rollback = _prove_exact_layer_rollback(
            execution_layer,
            execution_snapshot,
            source_layer=self.source_layer,
            source_snapshot=self.source_snapshot,
            input_bank=self.audit_bank,
            layer_index=self.layer_index,
            evidence_sha256_before=self.audit_evidence_sha256_before,
            input_sha256_before=self.audit_input_sha256_before,
            boundary_charts=self.boundary_charts,
            boundary_charts_sha256_before=self.boundary_charts_sha256_before,
            boundary_charts_payload_sha256_before=(
                self.boundary_charts_payload_sha256_before
            ),
        )
        if (
            fit_replay is not None
            or self.mlp_preparation_count != 1
            or self.mlp_prepared_basis is None
            or not self.mlp_prepared_basis.verify_authentication()
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane MLP basis preparation retained replay state or "
                "failed authentication."
            )
        return peak_replay_bytes, rollback

    def _evaluate_two_lane_candidate(
        self,
        *,
        candidate: Lfm2CpcfShortConvV5MacrocellCandidate,
        execution_layer: Any,
        execution_snapshot: _LayerSnapshot,
        lane: _Lfm2CpcfShortConvExecutionLane,
        runtime: _Lfm2CpcfShortConvTwoLaneRuntime,
        semantic_parity_sink: list[Mapping[str, Any]] | None = None,
    ) -> _Lfm2CpcfShortConvMacrocellEvaluation:
        """Evaluate one candidate on one lane-local target-layer replica."""

        if semantic_parity_sink is not None and (
            not isinstance(semantic_parity_sink, list) or semantic_parity_sink
        ):
            raise Lfm2CpcfLayerCompilerError(
                "A semantic parity sink must be one empty invocation-local list."
            )
        prepared_basis = self.mlp_prepared_basis
        if (
            prepared_basis is None
            or self.mlp_preparation_count != 1
            or not prepared_basis.verify_authentication()
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane candidate evaluation lacks its shared MLP basis."
            )
        candidate_id = candidate.candidate_id
        fit_replay: Lfm2CpcfLayerReplayBank | None = None
        selection_replay: Lfm2CpcfLayerReplayBank | None = None
        fit_semantic_signature: Mapping[str, Any] | None = None
        selection_semantic_signature: Mapping[str, Any] | None = None
        peak_replay_bytes = 0
        _restore_layer(execution_layer, execution_snapshot)
        try:
            install = _install_short_conv_macrocell_candidate(
                execution_layer.conv,
                candidate,
            )
            fit_replay = replay_lfm2_cpcf_phase_bank(
                self.target_model,
                layer_index=self.layer_index,
                input_bank=self.fit_bank,
                execution_layer=execution_layer,
            )
            peak_replay_bytes = max(
                peak_replay_bytes,
                _replay_archive_logical_bytes(fit_replay),
            )
            if semantic_parity_sink is not None:
                fit_semantic_signature = _macrocell_replay_semantic_signature(
                    fit_replay
                )
            fit_replay_sha256 = fit_replay.report["report_sha256"]
            mlp_artifact = _compile_cpcf_mlp_from_fit_replay(
                source_layer=self.source_layer,
                target_layer=execution_layer,
                layer_index=self.layer_index,
                fit_bank=self.fit_bank,
                operator_fit_replay=fit_replay,
                chart=self.boundary_charts,
                ridge=self.ridge,
                prepared_basis=prepared_basis,
            )
            if (
                mlp_artifact.prepared_basis_reused is not True
                or mlp_artifact.prepared_basis_sha256
                != prepared_basis.preparation_sha256
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Two-lane candidate MLP does not bind the shared prepared "
                    "parent basis."
                )
            mlp_install = install_lfm2_structural_mlp_fold(
                execution_layer.feed_forward,
                mlp_artifact,
            ).as_dict()
            # The fit replay is no longer needed after the candidate MLP is
            # materialized.  Releasing it before selection bounds the process
            # to one live replay tree per lane, two globally.
            del fit_replay
            fit_replay = None
            selection_replay = replay_lfm2_cpcf_phase_bank(
                self.target_model,
                layer_index=self.layer_index,
                input_bank=self.selection_bank,
                execution_layer=execution_layer,
            )
            peak_replay_bytes = max(
                peak_replay_bytes,
                _replay_archive_logical_bytes(selection_replay),
            )
            selection_replay_sha256 = selection_replay.report["report_sha256"]
            raw_metrics = _split_endpoint_metrics(
                layer_index=self.layer_index,
                input_bank=self.selection_bank,
                operator_replay=selection_replay,
                final_replay=selection_replay,
                chart=self.boundary_charts,
                splits=("selection",),
            )
            metrics = _normalized_macrocell_selection_metrics(
                candidate_id,
                self.selection_bank.state_bank_sha256,
                raw_metrics,
            )
            if semantic_parity_sink is not None:
                selection_semantic_signature = _macrocell_replay_semantic_signature(
                    selection_replay
                )
            del selection_replay
            selection_replay = None
        finally:
            _restore_layer(execution_layer, execution_snapshot)
        rollback = _prove_exact_layer_rollback(
            execution_layer,
            execution_snapshot,
            source_layer=self.source_layer,
            source_snapshot=self.source_snapshot,
            input_bank=self.audit_bank,
            layer_index=self.layer_index,
            evidence_sha256_before=self.audit_evidence_sha256_before,
            input_sha256_before=self.audit_input_sha256_before,
            boundary_charts=self.boundary_charts,
            boundary_charts_sha256_before=self.boundary_charts_sha256_before,
            boundary_charts_payload_sha256_before=(
                self.boundary_charts_payload_sha256_before
            ),
        )
        if fit_replay is not None or selection_replay is not None:
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane candidate retained a replay tree after evaluation."
            )
        install_actual = _seal_macrocell_actual(
            "operator_install",
            candidate_id,
            source_report_sha256=install["report_sha256"],
            compiled_state_sha256=install["compiled_state_sha256"],
            installed_state_sha256=install["installed_state_sha256"],
        )
        install_evidence = _seal_macrocell_evidence(
            "actual_operator_install",
            candidate_id,
            install_actual,
            candidate_state_sha256=candidate.state_sha256,
            installed_state_sha256=install["installed_state_sha256"],
        )
        fit_actual = _seal_macrocell_actual(
            "fit_operator_replay",
            candidate_id,
            source_report_sha256=fit_replay_sha256,
            input_bank_sha256=self.fit_bank.state_bank_sha256,
            macrocell_execution_mode="authenticated_two_static_lanes",
            macrocell_execution_lane_index=lane.lane_index,
            macrocell_execution_device=str(lane.device),
            macrocell_execution_device_receipt_sha256=(lane.device_receipt_sha256),
            macrocell_execution_replica_receipt_sha256=(lane.replica_receipt_sha256),
            macrocell_execution_runtime_authentication_report_sha256=(
                runtime.runtime_authentication_report_sha256
            ),
            maximum_live_replay_trees_for_lane=1,
            peak_replay_archive_logical_bytes=peak_replay_bytes,
            fit_replay_released_before_selection_replay=True,
            selection_replay_released_before_candidate_return=True,
        )
        fit_evidence = _seal_macrocell_evidence(
            "actual_fit_operator_replay",
            candidate_id,
            fit_actual,
            candidate_state_sha256=candidate.state_sha256,
            input_bank_sha256=self.fit_bank.state_bank_sha256,
        )
        mlp_actual = _seal_macrocell_actual(
            "fit_mlp_artifact",
            candidate_id,
            source_report_sha256=_sha(mlp_artifact.as_dict()),
            compiled_state_sha256=mlp_artifact.compiled_state_sha256,
        )
        mlp_evidence = _seal_macrocell_evidence(
            "actual_fit_mlp_artifact",
            candidate_id,
            mlp_actual,
            fit_bank_sha256=self.fit_bank.state_bank_sha256,
            compiled_state_sha256=mlp_artifact.compiled_state_sha256,
        )
        mlp_install_actual = _seal_macrocell_actual(
            "fit_mlp_install",
            candidate_id,
            source_report_sha256=_sha(mlp_install),
            compiled_state_sha256=mlp_install["compiled_state_sha256"],
            installed_state_sha256=mlp_install["installed_state_sha256"],
        )
        mlp_install_evidence = _seal_macrocell_evidence(
            "actual_fit_mlp_install",
            candidate_id,
            mlp_install_actual,
            compiled_state_sha256=mlp_artifact.compiled_state_sha256,
            installed_state_sha256=mlp_install["installed_state_sha256"],
        )
        selection_actual = _seal_macrocell_actual(
            "selection_final_replay",
            candidate_id,
            source_report_sha256=selection_replay_sha256,
            input_bank_sha256=self.selection_bank.state_bank_sha256,
        )
        selection_evidence = _seal_macrocell_evidence(
            "actual_selection_final_replay",
            candidate_id,
            selection_actual,
            candidate_state_sha256=candidate.state_sha256,
            input_bank_sha256=self.selection_bank.state_bank_sha256,
            mlp_compiled_state_sha256=mlp_artifact.compiled_state_sha256,
        )
        rollback_actual = _seal_macrocell_actual(
            "candidate_rollback",
            candidate_id,
            source_report_sha256=rollback["proof_sha256"],
            target_state_sha256_before=rollback["target_state_sha256_before"],
            target_state_sha256_after=rollback["target_state_sha256_after"],
            target_tensor_bytes_restored=True,
            target_tensor_identities_preserved=True,
            target_gradient_state_preserved=True,
            target_module_modes_restored=True,
            exact_layer_snapshot_restored=True,
        )
        rollback_body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA,
            "kind": "actual_candidate_rollback",
            "candidate_id": candidate_id,
            "target_state_sha256_before": rollback["target_state_sha256_before"],
            "target_state_sha256_after": rollback["target_state_sha256_after"],
            "target_tensor_bytes_restored": True,
            "target_tensor_identities_preserved": True,
            "target_gradient_state_preserved": True,
            "target_module_modes_restored": True,
            "exact_layer_snapshot_restored": True,
            "actual_report": rollback_actual,
            "actual_report_sha256": rollback_actual["report_sha256"],
            "execution": _macrocell_zero_execution(),
        }
        rollback_evidence = {
            **rollback_body,
            "report_sha256": _sha(rollback_body),
        }
        evaluation_body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA,
            "candidate_id": candidate_id,
            "candidate_state_sha256": candidate.state_sha256,
            "operator_selection_report_sha256": (
                candidate.operator_selection_report["report_sha256"]
            ),
            "fit_provenance_report_sha256": (candidate.fit_provenance["report_sha256"]),
            "fit_bank_sha256": self.fit_bank.state_bank_sha256,
            "selection_bank_sha256": self.selection_bank.state_bank_sha256,
            "operator_install_report": install_evidence,
            "operator_install_report_sha256": install_evidence["report_sha256"],
            "fit_operator_replay_report": fit_evidence,
            "fit_operator_replay_sha256": fit_evidence["report_sha256"],
            "fit_mlp_artifact_report": mlp_evidence,
            "fit_mlp_artifact_report_sha256": mlp_evidence["report_sha256"],
            "fit_mlp_compiled_state_sha256": mlp_artifact.compiled_state_sha256,
            "fit_mlp_install_report": mlp_install_evidence,
            "fit_mlp_install_report_sha256": (mlp_install_evidence["report_sha256"]),
            "selection_final_replay_report": selection_evidence,
            "selection_final_replay_sha256": selection_evidence["report_sha256"],
            "selection_metrics": metrics,
            "selection_metrics_report_sha256": metrics["report_sha256"],
            "fit_only_mlp_fold": True,
            "selection_only_candidate_evaluation": True,
            "validation_accessible": False,
            "validation_replays": 0,
            "rollback_proof": rollback_evidence,
            "rollback_proof_sha256": rollback_evidence["report_sha256"],
            "execution": _macrocell_zero_execution(),
        }
        evaluation_report = {
            **evaluation_body,
            "report_sha256": _sha(evaluation_body),
        }
        key = _short_conv_macrocell_score(candidate_id, metrics)
        if semantic_parity_sink is not None:
            if fit_semantic_signature is None or selection_semantic_signature is None:
                raise Lfm2CpcfLayerCompilerError(
                    "Two-lane candidate lacks its lane-independent semantic "
                    "replay signatures."
                )
            semantic_body = {
                "schema": LFM2_CPCF_SHORT_CONV_TWO_LANE_SEMANTIC_PARITY_SCHEMA,
                "candidate_id": candidate_id,
                "candidate_state_sha256": candidate.state_sha256,
                "operator_installed_state_sha256": (install["installed_state_sha256"]),
                "fit_replay_semantic_signature": _plain_json_metadata(
                    fit_semantic_signature
                ),
                "fit_replay_semantic_signature_sha256": (
                    fit_semantic_signature["signature_sha256"]
                ),
                "shared_mlp_basis_sha256": prepared_basis.preparation_sha256,
                "mlp_compiled_state_sha256": (mlp_artifact.compiled_state_sha256),
                "mlp_installed_state_sha256": (mlp_install["installed_state_sha256"]),
                "selection_replay_semantic_signature": _plain_json_metadata(
                    selection_semantic_signature
                ),
                "selection_replay_semantic_signature_sha256": (
                    selection_semantic_signature["signature_sha256"]
                ),
                "selection_metrics_report_sha256": metrics["report_sha256"],
                "selection_key": list(key),
            }
            semantic_parity_sink.append(
                MappingProxyType(
                    {
                        **semantic_body,
                        "signature_sha256": _sha(semantic_body),
                    }
                )
            )
        return _Lfm2CpcfShortConvMacrocellEvaluation(
            candidate=candidate,
            mlp_artifact=mlp_artifact,
            selection_key=key,
            report=MappingProxyType(evaluation_report),
        )

    def _evaluate_two_lane(
        self,
        *,
        frontier: Lfm2CpcfShortConvV5CandidateFrontier,
        evaluated_ids: Sequence[str],
        runtime: _Lfm2CpcfShortConvTwoLaneRuntime,
    ) -> tuple[
        tuple[_Lfm2CpcfShortConvMacrocellEvaluation, ...],
        Mapping[str, Any],
    ]:
        """Run exact cross-lane parity, then canonical static assignments."""

        canonical_ids = tuple(evaluated_ids)
        if (
            not canonical_ids
            or len(set(canonical_ids)) != len(canonical_ids)
            or len(runtime.lanes) != 2
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane macrocell work must be nonempty and canonical."
            )
        replicas = _private_two_lane_replicas(self.target_layer, runtime)
        replica_snapshots = tuple(_layer_snapshot(replica) for replica in replicas)
        representative_id = canonical_ids[0]
        # Candidate zero is first run once on each existing lane.  After exact
        # semantic parity, lane zero's real result is reused; every other
        # candidate keeps its original canonical-index parity assignment.
        assignments = (
            canonical_ids[2::2],
            canonical_ids[1::2],
        )
        preparation_peak = 0
        preparation_rollback: Mapping[str, Any] | None = None
        lane_final_rollbacks: list[Mapping[str, Any] | None] = [None, None]
        canonical_rollback: Mapping[str, Any] | None = None
        parity_results: list[_Lfm2CpcfShortConvMacrocellEvaluation | None] = [
            None,
            None,
        ]
        parity_signatures: list[Mapping[str, Any] | None] = [None, None]
        lane_results: list[tuple[_Lfm2CpcfShortConvMacrocellEvaluation, ...] | None] = [
            None,
            None,
        ]
        parity_proof: Mapping[str, Any] | None = None
        active_lock = threading.Lock()
        stop_event = threading.Event()
        active_lanes = 0
        maximum_active_lanes = 0
        failure: BaseException | None = None
        futures = ()
        try:
            preparation_peak, preparation_rollback = self._prepare_two_lane_mlp_basis(
                candidate=frontier.candidate(representative_id),
                execution_layer=replicas[0],
                execution_snapshot=replica_snapshots[0],
            )

            parity_barrier = threading.Barrier(2)

            def run_parity_lane(
                lane_index: int,
            ) -> tuple[
                _Lfm2CpcfShortConvMacrocellEvaluation,
                Mapping[str, Any],
            ]:
                nonlocal active_lanes, maximum_active_lanes
                parity_barrier.wait()
                with active_lock:
                    active_lanes += 1
                    maximum_active_lanes = max(maximum_active_lanes, active_lanes)
                try:
                    semantic_parity_sink: list[Mapping[str, Any]] = []
                    evaluation = self._evaluate_two_lane_candidate(
                        candidate=frontier.candidate(representative_id),
                        execution_layer=replicas[lane_index],
                        execution_snapshot=replica_snapshots[lane_index],
                        lane=runtime.lanes[lane_index],
                        runtime=runtime,
                        semantic_parity_sink=semantic_parity_sink,
                    )
                    if len(semantic_parity_sink) != 1:
                        raise Lfm2CpcfLayerCompilerError(
                            "Cross-lane representative did not emit exactly "
                            "one semantic signature."
                        )
                    return evaluation, semantic_parity_sink[0]
                except BaseException:
                    stop_event.set()
                    raise
                finally:
                    with active_lock:
                        active_lanes -= 1

            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="lfm2-shortconv-parity",
            ) as executor:
                futures = (
                    executor.submit(run_parity_lane, 0),
                    executor.submit(run_parity_lane, 1),
                )
                lane_by_future = {
                    future: lane_index for lane_index, future in enumerate(futures)
                }
                for future in as_completed(futures):
                    lane_index = lane_by_future[future]
                    evaluation, semantic_signature = future.result()
                    parity_results[lane_index] = evaluation
                    parity_signatures[lane_index] = semantic_signature
            if any(value is None for value in parity_results) or any(
                value is None for value in parity_signatures
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Cross-lane representative evaluation did not complete."
                )
            parity_proof = _prove_exact_cross_lane_macrocell_parity(
                parity_results[0],
                parity_results[1],
                left_signature=parity_signatures[0],
                right_signature=parity_signatures[1],
                runtime=runtime,
            )
            stop_event.clear()
            work_barrier = threading.Barrier(2)

            def run_lane(
                lane_index: int,
            ) -> tuple[
                _Lfm2CpcfShortConvMacrocellEvaluation,
                ...,
            ]:
                nonlocal active_lanes, maximum_active_lanes
                work_barrier.wait()
                values = []
                with active_lock:
                    active_lanes += 1
                    maximum_active_lanes = max(maximum_active_lanes, active_lanes)
                try:
                    for candidate_id in assignments[lane_index]:
                        if stop_event.is_set():
                            break
                        values.append(
                            self._evaluate_two_lane_candidate(
                                candidate=frontier.candidate(candidate_id),
                                execution_layer=replicas[lane_index],
                                execution_snapshot=replica_snapshots[lane_index],
                                lane=runtime.lanes[lane_index],
                                runtime=runtime,
                            )
                        )
                except BaseException:
                    stop_event.set()
                    raise
                finally:
                    with active_lock:
                        active_lanes -= 1
                return tuple(values)

            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix="lfm2-shortconv-macrocell",
            ) as executor:
                futures = (
                    executor.submit(run_lane, 0),
                    executor.submit(run_lane, 1),
                )
                lane_by_future = {
                    future: lane_index for lane_index, future in enumerate(futures)
                }
                for future in as_completed(futures):
                    lane_index = lane_by_future[future]
                    lane_results[lane_index] = future.result()
        except BaseException as error:
            failure = error
        finally:
            for lane_index, (replica, snapshot) in enumerate(
                zip(replicas, replica_snapshots, strict=True)
            ):
                _restore_layer(replica, snapshot)
                lane_final_rollbacks[lane_index] = _prove_exact_layer_rollback(
                    replica,
                    snapshot,
                    source_layer=self.source_layer,
                    source_snapshot=self.source_snapshot,
                    input_bank=self.audit_bank,
                    layer_index=self.layer_index,
                    evidence_sha256_before=self.audit_evidence_sha256_before,
                    input_sha256_before=self.audit_input_sha256_before,
                    boundary_charts=self.boundary_charts,
                    boundary_charts_sha256_before=(self.boundary_charts_sha256_before),
                    boundary_charts_payload_sha256_before=(
                        self.boundary_charts_payload_sha256_before
                    ),
                )
            _restore_layer(self.target_layer, self.target_snapshot)
            canonical_rollback = _prove_exact_layer_rollback(
                self.target_layer,
                self.target_snapshot,
                source_layer=self.source_layer,
                source_snapshot=self.source_snapshot,
                input_bank=self.audit_bank,
                layer_index=self.layer_index,
                evidence_sha256_before=self.audit_evidence_sha256_before,
                input_sha256_before=self.audit_input_sha256_before,
                boundary_charts=self.boundary_charts,
                boundary_charts_sha256_before=self.boundary_charts_sha256_before,
                boundary_charts_payload_sha256_before=(
                    self.boundary_charts_payload_sha256_before
                ),
            )
        if failure is not None:
            # Futures otherwise retain worker traceback frames and their last
            # replay locals.  Drop those references before failing closed.
            futures = ()
            parity_results = [None, None]
            parity_signatures = [None, None]
            lane_results = [None, None]
            raise failure.with_traceback(None)
        if (
            any(value is None for value in parity_results)
            or any(value is None for value in parity_signatures)
            or any(value is None for value in lane_results)
            or parity_proof is None
            or preparation_rollback is None
            or canonical_rollback is None
            or any(value is None for value in lane_final_rollbacks)
            or active_lanes != 0
            or maximum_active_lanes > 2
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane macrocell execution did not close every bounded lane."
            )
        representative = parity_results[0]
        if representative is None:  # pragma: no cover - guarded above
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane representative result disappeared."
            )
        by_id = {representative.candidate.candidate_id: representative}
        for lane_values in lane_results:
            if lane_values is None:  # pragma: no cover - guarded above
                continue
            for value in lane_values:
                candidate_id = value.candidate.candidate_id
                if candidate_id in by_id:
                    raise Lfm2CpcfLayerCompilerError(
                        "Two-lane candidate result was evaluated more than "
                        "its authenticated parity budget."
                    )
                by_id[candidate_id] = value
        if set(by_id) != set(canonical_ids) or len(by_id) != len(canonical_ids):
            raise Lfm2CpcfLayerCompilerError(
                "Two-lane candidate results are missing, duplicated, or changed."
            )
        ordered = tuple(by_id[candidate_id] for candidate_id in canonical_ids)
        per_lane_sources = (
            (parity_results[0], *(lane_results[0] or ())),
            (parity_results[1], *(lane_results[1] or ())),
        )
        per_lane_peak = [
            max(
                int(
                    value.report["fit_operator_replay_report"]["actual_report"][
                        "peak_replay_archive_logical_bytes"
                    ]
                )
                for value in lane_values
                if value is not None
            )
            for lane_values in per_lane_sources
        ]
        estimated_peak = max(preparation_peak, sum(per_lane_peak))
        lane_assignment = {
            candidate_id: runtime.lanes[index % 2].lane_index
            for index, candidate_id in enumerate(canonical_ids)
        }
        compile_executions = len(canonical_ids) + 1
        receipt_body = {
            "schema": LFM2_CPCF_SHORT_CONV_TWO_LANE_EXECUTION_SCHEMA,
            "runtime_authentication_report_sha256": (
                runtime.runtime_authentication_report_sha256
            ),
            "scheduler_policy_sha256": runtime.scheduler_policy_sha256,
            "lanes": [lane.as_dict() for lane in runtime.lanes],
            "lane_assignment": lane_assignment,
            "assignment_policy": "canonical_candidate_index_parity",
            "canonical_candidate_result_order": list(canonical_ids),
            "canonical_result_order_restored_after_parallel_completion": True,
            "cross_lane_semantic_parity_proof": _plain_json_metadata(parity_proof),
            "cross_lane_semantic_parity_proof_sha256": (parity_proof["report_sha256"]),
            "cross_lane_semantic_parity_completed_before_remaining_candidates": (True),
            "representative_candidate_id": representative_id,
            "representative_candidate_evaluated_once_per_lane": True,
            "representative_candidate_compile_execution_count": 2,
            "representative_lane_zero_result_reused": True,
            "representative_lane_one_result_discarded": True,
            "all_nonrepresentative_candidates_evaluated_exactly_once": True,
            "candidate_result_count": len(canonical_ids),
            "candidate_compile_result_count": len(canonical_ids),
            "candidate_compile_execution_count": compile_executions,
            "duplicate_parity_compile_execution_count": 1,
            "per_lane_candidate_compile_execution_count": [
                1 + len(assignments[0]),
                1 + len(assignments[1]),
            ],
            "static_lane_count": 2,
            "worker_phase_count": 2,
            "worker_future_count": 4,
            "maximum_simultaneous_worker_future_count": 2,
            "maximum_active_lanes_observed": maximum_active_lanes,
            "maximum_live_replay_tree_slots": 2,
            "maximum_live_replay_trees_per_lane": 1,
            "per_lane_peak_replay_archive_logical_bytes": per_lane_peak,
            "shared_basis_preparation_peak_replay_archive_logical_bytes": (
                preparation_peak
            ),
            "estimated_peak_replay_archive_logical_bytes": estimated_peak,
            "fit_replay_released_before_selection_replay": True,
            "candidate_replays_released_before_next_candidate": True,
            "completed_replay_accelerator_storage_retained": False,
            "two_private_deep_copied_target_layers": True,
            "whole_model_clones": 0,
            "replica_parameter_storage_isolated": True,
            "replica_parameter_and_buffer_storage_isolated": True,
            "cached_banks_charts_and_source_shared_read_only": True,
            "shared_mlp_basis_preparation_count": self.mlp_preparation_count,
            "shared_mlp_basis_sha256": (
                self.mlp_prepared_basis.preparation_sha256
                if self.mlp_prepared_basis is not None
                else None
            ),
            "preparation_replica_rollback_proof_sha256": (
                preparation_rollback["proof_sha256"]
            ),
            "lane_final_rollback_proof_sha256": [
                value["proof_sha256"]
                for value in lane_final_rollbacks
                if value is not None
            ],
            "canonical_target_rollback_proof_sha256": (
                canonical_rollback["proof_sha256"]
            ),
            "canonical_target_untouched_until_winner_install": True,
            "parent_forward_count": 0,
            "training_steps": 0,
            "parameters_pruned": 0,
            "parameters_quantized": 0,
        }
        receipt = MappingProxyType(
            {
                **receipt_body,
                "report_sha256": _sha(receipt_body),
            }
        )
        return ordered, receipt

    def select(
        self,
        frontier: Lfm2CpcfShortConvV5CandidateFrontier,
    ) -> Mapping[str, Any]:
        if (
            not isinstance(frontier, Lfm2CpcfShortConvV5CandidateFrontier)
            or not frontier.verify_authentication()
            or frontier.fit_bank_sha256 != self.fit_bank.state_bank_sha256
            or frontier.selection_bank_sha256 != self.selection_bank.state_bank_sha256
            or {phase.split for phase in self.audit_bank.phases} != {"fit", "selection"}
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv macrocell frontier is not bound to real phase banks."
            )
        budget = min(8, len(frontier.candidates))
        evaluated_ids = list(
            select_lfm2_cpcf_short_conv_v5_macrocell_candidate_ids(
                frontier,
                budget,
            )
        )
        self.endpoint_control_id = str(
            frontier.default_ranking["selected_candidate_id"]
        )
        if self.endpoint_selector is not None:
            candidate_ids = [
                candidate.candidate_id for candidate in frontier.candidates
            ]
            state_sha = {
                candidate.candidate_id: candidate.state_sha256
                for candidate in frontier.candidates
            }
            selection_reports = {
                candidate.candidate_id: _plain_json_metadata(
                    candidate.operator_selection_report
                )
                for candidate in frontier.candidates
            }
            request_body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_SELECTOR_REQUEST_SCHEMA,
                "solver_policy_sha256": frontier.solver_policy_sha256,
                "target_dtype": frontier.target_dtype,
                "candidate_ids": candidate_ids,
                "candidate_state_sha256": state_sha,
                "candidate_state_inventory_sha256": _sha(state_sha),
                "candidate_selection_report_sha256": {
                    candidate_id: selection_reports[candidate_id]["report_sha256"]
                    for candidate_id in candidate_ids
                },
                "candidate_selection_reports": selection_reports,
                "default_v4_ranking_report_sha256": (
                    frontier.default_ranking["report_sha256"]
                ),
                "default_v4_selected_candidate_id": (
                    frontier.default_ranking["selected_candidate_id"]
                ),
                "candidate_state_tensors_exposed_to_selector": False,
                "fit_reports_exposed_to_selector": False,
                "validation_reports_exposed_to_selector": False,
                "post_install_rescoring": False,
                "execution": {
                    "training_steps": 0,
                    "backward_calls": 0,
                    "optimizer_steps": 0,
                    "parameters_pruned": 0,
                    "parameters_quantized": 0,
                },
            }
            request = {**request_body, "report_sha256": _sha(request_body)}
            request_before = _canonical_json(request)
            diagnostic_status = "accepted"
            diagnostic_reason = None
            endpoint_report = None
            try:
                endpoint_value = self.endpoint_selector.select(request)
                endpoint_report = _plain_json_metadata(endpoint_value)
                endpoint_body = dict(endpoint_report)
                endpoint_sha = _digest(
                    endpoint_body.pop("report_sha256", None),
                    "endpoint_selector_report.report_sha256",
                )
                selected_control = endpoint_report.get("selected_candidate_id")
                endpoint_execution = endpoint_report.get("execution")
                if (
                    _sha(endpoint_body) != endpoint_sha
                    or endpoint_report.get("schema") != LFM2_CPCF_SHORT_CONV_V5_SCHEMA
                    or selected_control not in state_sha
                    or endpoint_report.get("selected_candidate_state_sha256")
                    != state_sha[selected_control]
                    or endpoint_report.get("candidate_count") != len(candidate_ids)
                    or endpoint_report.get("selection_reports_consumed")
                    != len(candidate_ids)
                    or endpoint_report.get("fit_reports_consumed") != 0
                    or endpoint_report.get("validation_reports_consumed") != 0
                    or endpoint_report.get("validation_accessible") is not False
                    or not isinstance(endpoint_execution, Mapping)
                    or set(endpoint_execution)
                    != {
                        "training_steps",
                        "backward_calls",
                        "optimizer_steps",
                        "parameters_pruned",
                        "parameters_quantized",
                    }
                    or any(
                        type(value) is not int or value != 0
                        for value in endpoint_execution.values()
                    )
                    or selected_control not in evaluated_ids
                ):
                    diagnostic_status = "rejected_fallback_to_default"
                    diagnostic_reason = "invalid_or_outside_bounded_frontier"
                else:
                    self.endpoint_control_id = str(selected_control)
                    self.endpoint_selector_report = MappingProxyType(endpoint_report)
            except Exception as error:
                diagnostic_status = "raised_fallback_to_default"
                diagnostic_reason = type(error).__name__
            if _canonical_json(request) != request_before:
                diagnostic_status = "mutated_request_fallback_to_default"
                diagnostic_reason = "sealed_request_changed"
                self.endpoint_control_id = str(
                    frontier.default_ranking["selected_candidate_id"]
                )
                self.endpoint_selector_report = None
            diagnostic_body = {
                "schema": ("neural-forge-lfm2-cpcf-endpoint-control-diagnostic-v1"),
                "status": diagnostic_status,
                "reason": diagnostic_reason,
                "fallback_candidate_id": frontier.default_ranking[
                    "selected_candidate_id"
                ],
                "effective_control_candidate_id": self.endpoint_control_id,
                "bounded_candidate_budget": budget,
                "non_authoritative_for_final_macrocell_selection": True,
                "validation_accessible": False,
                "training_steps": 0,
            }
            self.endpoint_selector_diagnostic = MappingProxyType(
                {
                    **diagnostic_body,
                    "report_sha256": _sha(diagnostic_body),
                }
            )
        evaluations: dict[str, dict[str, Any]] = {}
        scored = []
        two_lane_receipt: Mapping[str, Any] | None = None
        two_lane_runtime = _authenticated_short_conv_two_lane_runtime(
            self.candidate_replay_evaluator
        )
        serial_evaluated_ids: Sequence[str] = evaluated_ids
        if two_lane_runtime is not None:
            parallel_evaluations, two_lane_receipt = self._evaluate_two_lane(
                frontier=frontier,
                evaluated_ids=evaluated_ids,
                runtime=two_lane_runtime,
            )
            serial_evaluated_ids = ()
            self.mlp_candidate_compile_count = len(parallel_evaluations)
            for evaluation in parallel_evaluations:
                candidate_id = evaluation.candidate.candidate_id
                self.evaluations[candidate_id] = evaluation
                evaluation_report = dict(evaluation.report)
                evaluations[candidate_id] = evaluation_report
                scored.append(
                    {
                        "candidate_id": candidate_id,
                        "score": list(evaluation.selection_key),
                        "evaluation_report_sha256": (
                            evaluation_report["report_sha256"]
                        ),
                    }
                )
        for candidate_id in serial_evaluated_ids:
            candidate = frontier.candidate(candidate_id)
            _restore_layer(self.target_layer, self.target_snapshot)
            try:
                install = _install_short_conv_macrocell_candidate(
                    self.target_layer.conv,
                    candidate,
                )
                fit_replay = replay_lfm2_cpcf_phase_bank(
                    self.target_model,
                    layer_index=self.layer_index,
                    input_bank=self.fit_bank,
                )
                if self.mlp_prepared_basis is None:
                    self.mlp_prepared_basis = _prepare_cpcf_mlp_basis_from_fit_replay(
                        source_layer=self.source_layer,
                        target_layer=self.target_layer,
                        layer_index=self.layer_index,
                        fit_bank=self.fit_bank,
                        operator_fit_replay=fit_replay,
                        chart=self.boundary_charts,
                    )
                    self.mlp_preparation_count += 1
                if self.mlp_preparation_count != 1:
                    raise Lfm2CpcfLayerCompilerError(
                        "Short-conv candidate MLP preparation is not one "
                        "authenticated layer-local basis."
                    )
                mlp_artifact = _compile_cpcf_mlp_from_fit_replay(
                    source_layer=self.source_layer,
                    target_layer=self.target_layer,
                    layer_index=self.layer_index,
                    fit_bank=self.fit_bank,
                    operator_fit_replay=fit_replay,
                    chart=self.boundary_charts,
                    ridge=self.ridge,
                    prepared_basis=self.mlp_prepared_basis,
                )
                self.mlp_candidate_compile_count += 1
                if (
                    mlp_artifact.prepared_basis_reused is not True
                    or mlp_artifact.prepared_basis_sha256
                    != self.mlp_prepared_basis.preparation_sha256
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        "Short-conv candidate MLP does not bind the shared "
                        "prepared parent basis."
                    )
                mlp_install = install_lfm2_structural_mlp_fold(
                    self.target_layer.feed_forward,
                    mlp_artifact,
                ).as_dict()
                selection_replay = replay_lfm2_cpcf_phase_bank(
                    self.target_model,
                    layer_index=self.layer_index,
                    input_bank=self.selection_bank,
                )
                raw_metrics = _split_endpoint_metrics(
                    layer_index=self.layer_index,
                    input_bank=self.selection_bank,
                    operator_replay=selection_replay,
                    final_replay=selection_replay,
                    chart=self.boundary_charts,
                    splits=("selection",),
                )
                metrics = _normalized_macrocell_selection_metrics(
                    candidate_id,
                    self.selection_bank.state_bank_sha256,
                    raw_metrics,
                )
            finally:
                _restore_layer(self.target_layer, self.target_snapshot)
            rollback = _prove_exact_layer_rollback(
                self.target_layer,
                self.target_snapshot,
                source_layer=self.source_layer,
                source_snapshot=self.source_snapshot,
                input_bank=self.audit_bank,
                layer_index=self.layer_index,
                evidence_sha256_before=self.audit_evidence_sha256_before,
                input_sha256_before=self.audit_input_sha256_before,
                boundary_charts=self.boundary_charts,
                boundary_charts_sha256_before=(self.boundary_charts_sha256_before),
                boundary_charts_payload_sha256_before=(
                    self.boundary_charts_payload_sha256_before
                ),
            )
            install_actual = _seal_macrocell_actual(
                "operator_install",
                candidate_id,
                source_report_sha256=install["report_sha256"],
                compiled_state_sha256=install["compiled_state_sha256"],
                installed_state_sha256=install["installed_state_sha256"],
            )
            install_evidence = _seal_macrocell_evidence(
                "actual_operator_install",
                candidate_id,
                install_actual,
                candidate_state_sha256=candidate.state_sha256,
                installed_state_sha256=install["installed_state_sha256"],
            )
            fit_actual = _seal_macrocell_actual(
                "fit_operator_replay",
                candidate_id,
                source_report_sha256=fit_replay.report["report_sha256"],
                input_bank_sha256=self.fit_bank.state_bank_sha256,
            )
            fit_evidence = _seal_macrocell_evidence(
                "actual_fit_operator_replay",
                candidate_id,
                fit_actual,
                candidate_state_sha256=candidate.state_sha256,
                input_bank_sha256=self.fit_bank.state_bank_sha256,
            )
            mlp_actual = _seal_macrocell_actual(
                "fit_mlp_artifact",
                candidate_id,
                source_report_sha256=_sha(mlp_artifact.as_dict()),
                compiled_state_sha256=mlp_artifact.compiled_state_sha256,
            )
            mlp_evidence = _seal_macrocell_evidence(
                "actual_fit_mlp_artifact",
                candidate_id,
                mlp_actual,
                fit_bank_sha256=self.fit_bank.state_bank_sha256,
                compiled_state_sha256=mlp_artifact.compiled_state_sha256,
            )
            mlp_install_actual = _seal_macrocell_actual(
                "fit_mlp_install",
                candidate_id,
                source_report_sha256=_sha(mlp_install),
                compiled_state_sha256=mlp_install["compiled_state_sha256"],
                installed_state_sha256=mlp_install["installed_state_sha256"],
            )
            mlp_install_evidence = _seal_macrocell_evidence(
                "actual_fit_mlp_install",
                candidate_id,
                mlp_install_actual,
                compiled_state_sha256=mlp_artifact.compiled_state_sha256,
                installed_state_sha256=mlp_install["installed_state_sha256"],
            )
            selection_actual = _seal_macrocell_actual(
                "selection_final_replay",
                candidate_id,
                source_report_sha256=selection_replay.report["report_sha256"],
                input_bank_sha256=self.selection_bank.state_bank_sha256,
            )
            selection_evidence = _seal_macrocell_evidence(
                "actual_selection_final_replay",
                candidate_id,
                selection_actual,
                candidate_state_sha256=candidate.state_sha256,
                input_bank_sha256=self.selection_bank.state_bank_sha256,
                mlp_compiled_state_sha256=mlp_artifact.compiled_state_sha256,
            )
            rollback_actual = _seal_macrocell_actual(
                "candidate_rollback",
                candidate_id,
                source_report_sha256=rollback["proof_sha256"],
                target_state_sha256_before=rollback["target_state_sha256_before"],
                target_state_sha256_after=rollback["target_state_sha256_after"],
                target_tensor_bytes_restored=True,
                target_tensor_identities_preserved=True,
                target_gradient_state_preserved=True,
                target_module_modes_restored=True,
                exact_layer_snapshot_restored=True,
            )
            rollback_body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVIDENCE_SCHEMA,
                "kind": "actual_candidate_rollback",
                "candidate_id": candidate_id,
                "target_state_sha256_before": rollback["target_state_sha256_before"],
                "target_state_sha256_after": rollback["target_state_sha256_after"],
                "target_tensor_bytes_restored": True,
                "target_tensor_identities_preserved": True,
                "target_gradient_state_preserved": True,
                "target_module_modes_restored": True,
                "exact_layer_snapshot_restored": True,
                "actual_report": rollback_actual,
                "actual_report_sha256": rollback_actual["report_sha256"],
                "execution": _macrocell_zero_execution(),
            }
            rollback_evidence = {
                **rollback_body,
                "report_sha256": _sha(rollback_body),
            }
            evaluation_body = {
                "schema": LFM2_CPCF_SHORT_CONV_V5_MACROCELL_EVALUATION_SCHEMA,
                "candidate_id": candidate_id,
                "candidate_state_sha256": candidate.state_sha256,
                "operator_selection_report_sha256": (
                    candidate.operator_selection_report["report_sha256"]
                ),
                "fit_provenance_report_sha256": (
                    candidate.fit_provenance["report_sha256"]
                ),
                "fit_bank_sha256": self.fit_bank.state_bank_sha256,
                "selection_bank_sha256": self.selection_bank.state_bank_sha256,
                "operator_install_report": install_evidence,
                "operator_install_report_sha256": install_evidence["report_sha256"],
                "fit_operator_replay_report": fit_evidence,
                "fit_operator_replay_sha256": fit_evidence["report_sha256"],
                "fit_mlp_artifact_report": mlp_evidence,
                "fit_mlp_artifact_report_sha256": mlp_evidence["report_sha256"],
                "fit_mlp_compiled_state_sha256": (mlp_artifact.compiled_state_sha256),
                "fit_mlp_install_report": mlp_install_evidence,
                "fit_mlp_install_report_sha256": (
                    mlp_install_evidence["report_sha256"]
                ),
                "selection_final_replay_report": selection_evidence,
                "selection_final_replay_sha256": (selection_evidence["report_sha256"]),
                "selection_metrics": metrics,
                "selection_metrics_report_sha256": metrics["report_sha256"],
                "fit_only_mlp_fold": True,
                "selection_only_candidate_evaluation": True,
                "validation_accessible": False,
                "validation_replays": 0,
                "rollback_proof": rollback_evidence,
                "rollback_proof_sha256": rollback_evidence["report_sha256"],
                "execution": _macrocell_zero_execution(),
            }
            evaluation_report = {
                **evaluation_body,
                "report_sha256": _sha(evaluation_body),
            }
            del fit_replay, selection_replay
            key = _short_conv_macrocell_score(candidate_id, metrics)
            evaluation = _Lfm2CpcfShortConvMacrocellEvaluation(
                candidate=candidate,
                mlp_artifact=mlp_artifact,
                selection_key=key,
                report=MappingProxyType(evaluation_report),
            )
            self.evaluations[candidate_id] = evaluation
            evaluations[candidate_id] = evaluation_report
            scored.append(
                {
                    "candidate_id": candidate_id,
                    "score": list(key),
                    "evaluation_report_sha256": evaluation_report["report_sha256"],
                }
            )
        if (
            self.mlp_prepared_basis is None
            or self.mlp_preparation_count != 1
            or self.mlp_candidate_compile_count != len(evaluated_ids)
            or not self.mlp_prepared_basis.verify_authentication()
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Short-conv macrocell MLP preparation/compile counts are incomplete."
            )
        scored.sort(key=lambda value: tuple(value["score"]))
        ranking = [{"rank": rank + 1, **value} for rank, value in enumerate(scored)]
        selected_id = ranking[0]["candidate_id"]
        self.selected = self.evaluations[selected_id]
        body = {
            "schema": LFM2_CPCF_SHORT_CONV_V5_SCHEMA,
            "kind": "authenticated_fit_selection_whole_macrocell_ranking",
            "frontier_sha256": frontier.frontier_sha256,
            "candidate_count": len(frontier.candidates),
            "candidate_budget": budget,
            "evaluated_candidate_ids": evaluated_ids,
            "candidate_evaluations": evaluations,
            "fit_bank_sha256": self.fit_bank.state_bank_sha256,
            "selection_bank_sha256": self.selection_bank.state_bank_sha256,
            "macrocell_ranking": ranking,
            "ranking_policy": (
                "minimum_worst_causal_phase_post_layer_relative_rmse_then_"
                "maximum_minimum_causal_phase_post_layer_cosine_then_minimum_"
                "aggregate_post_layer_relative_rmse_then_maximum_aggregate_"
                "post_layer_cosine_then_candidate_id"
            ),
            "operator_endpoint_first_control_included": True,
            "selected_candidate_id": selected_id,
            "selected_candidate_state_sha256": (self.selected.candidate.state_sha256),
            "selection_split_only_for_ranking": True,
            "validation_accessible": False,
            "validation_reports_consumed": 0,
            "validation_replays": 0,
            "candidate_specific_fit_mlp_folds": len(evaluated_ids),
            "candidate_specific_selection_replays": len(evaluated_ids),
            "exact_candidate_rollbacks": len(evaluated_ids),
            "execution": _macrocell_zero_execution(),
        }
        if two_lane_receipt is not None:
            self.two_lane_execution_receipt = two_lane_receipt
        self.decision = MappingProxyType({**body, "report_sha256": _sha(body)})
        return self.decision


@dataclass(frozen=True, slots=True)
class Lfm2CpcfCompiledLayerResult:
    """One installed layer and its actual-child next-boundary state bank."""

    layer_index: int
    operator_kind: str
    input_bank_sha256: str
    operator_artifact: Any
    mlp_artifact: Lfm2CompiledStructuralMlpFold
    operator_install_report: Mapping[str, Any]
    mlp_install_report: Mapping[str, Any]
    preinstall_replay: Lfm2CpcfLayerReplayBank
    operator_replay: Lfm2CpcfLayerReplayBank
    final_replay: Lfm2CpcfLayerReplayBank
    split_metrics: Mapping[str, Any]
    report: Mapping[str, Any]
    result_sha256: str

    def __post_init__(self) -> None:
        index = _nonnegative_int(self.layer_index, "layer_index")
        if self.operator_kind not in {"full_attention", "short_conv"}:
            raise Lfm2CpcfLayerCompilerError("operator_kind is not canonical LFM2.")
        _digest(self.input_bank_sha256, "input_bank_sha256")
        if self.operator_kind == "full_attention":
            if not isinstance(
                self.operator_artifact,
                (
                    Lfm2CompiledStructuralAttentionFold,
                    Lfm2CompiledCpcfResidualAttentionSplitFold,
                    Lfm2CompiledFixedQkSequenceEffectiveAttentionFold,
                    Lfm2CompiledComposedQkSequenceEffectiveAttentionFold,
                    Lfm2CompiledComposedEndpointResidualAttentionFold,
                ),
            ):
                raise TypeError("Attention result has the wrong operator artifact.")
        elif not isinstance(
            self.operator_artifact,
            (Lfm2CompiledCpcfShortConvFold, Lfm2CompiledCpcfShortConvV4Fold),
        ):
            raise TypeError("Short-conv result has the wrong operator artifact.")
        if not isinstance(
            self.mlp_artifact,
            Lfm2CompiledStructuralMlpFold,
        ):
            raise TypeError("mlp_artifact has the wrong type.")
        if (
            self.preinstall_replay.layer_index != index
            or self.operator_replay.layer_index != index
            or self.final_replay.layer_index != index
            or self.final_replay.post_layer_bank.layer_index != index + 1
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Layer replay sequence is incomplete or out of order."
            )
        if set(self.split_metrics) != set(_EVALUATION_SPLITS):
            raise Lfm2CpcfLayerCompilerError(
                "split_metrics must contain fit and validation."
            )
        result_sha = _digest(self.result_sha256, "result_sha256")
        report = dict(self.report)
        if (
            report.pop("result_sha256", None) != result_sha
            or _sha(report) != result_sha
        ):
            raise Lfm2CpcfLayerCompilerError("Layer compiler result seal is invalid.")
        object.__setattr__(
            self,
            "operator_install_report",
            MappingProxyType(dict(self.operator_install_report)),
        )
        object.__setattr__(
            self,
            "mlp_install_report",
            MappingProxyType(dict(self.mlp_install_report)),
        )
        object.__setattr__(
            self,
            "split_metrics",
            MappingProxyType(dict(self.split_metrics)),
        )
        object.__setattr__(
            self,
            "report",
            MappingProxyType({**report, "result_sha256": result_sha}),
        )

    @property
    def next_bank(self) -> Lfm2CpcfPhaseStateBank:
        return self.final_replay.post_layer_bank

    def as_dict(self) -> dict[str, Any]:
        return dict(self.report)


def _no_recursive_success_claims(value: Any) -> bool:
    if isinstance(value, Mapping):
        for name, child in value.items():
            if name in _NO_SUCCESS_FIELDS and child is not False:
                return False
            if not _no_recursive_success_claims(child):
                return False
        return True
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_no_recursive_success_claims(child) for child in value)
    return True


def _valid_short_conv_v1_validation_report(
    report: Mapping[str, Any],
) -> bool:
    behavior_gate = report.get("behavior_gate")
    policy = behavior_gate.get("policy") if isinstance(behavior_gate, Mapping) else None
    checks = behavior_gate.get("checks") if isinstance(behavior_gate, Mapping) else None
    audit = report.get("audit")
    execution = report.get("execution")
    audit_hashes_match = bool(
        isinstance(audit, Mapping)
        and all(
            isinstance(audit.get(name), str)
            and _SHA256.fullmatch(audit[name]) is not None
            for name in (
                "source_inputs_sha256_before",
                "source_inputs_sha256_after",
                "target_templates_sha256_before",
                "target_templates_sha256_after",
                "evidence_sha256_before",
                "evidence_sha256_after",
            )
        )
        and audit.get("source_inputs_sha256_before")
        == audit.get("source_inputs_sha256_after")
        and audit.get("target_templates_sha256_before")
        == audit.get("target_templates_sha256_after")
        and audit.get("evidence_sha256_before") == audit.get("evidence_sha256_after")
    )
    return bool(
        set(report) == _SHORT_CONV_VALIDATION_REPORT_FIELDS
        and report.get("schema") == LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA
        and isinstance(report.get("compiled_state_sha256"), str)
        and _SHA256.fullmatch(report["compiled_state_sha256"]) is not None
        and isinstance(behavior_gate, Mapping)
        and set(behavior_gate)
        == {
            "policy",
            "checks",
            "passed",
            "heldout_validation_used_for_fitting",
        }
        and behavior_gate.get("passed") is False
        and behavior_gate.get("heldout_validation_used_for_fitting") is False
        and isinstance(policy, Mapping)
        and set(policy) == _SHORT_CONV_GATE_POLICY_FIELDS
        and policy.get("enforce") is True
        and all(
            not isinstance(policy.get(name), bool)
            and isinstance(policy.get(name), (int, float))
            and math.isfinite(float(policy[name]))
            for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
        )
        and isinstance(checks, Mapping)
        and set(checks) == _SHORT_CONV_GATE_CHECK_FIELDS
        and all(isinstance(value, bool) for value in checks.values())
        and not all(checks.values())
        and isinstance(audit, Mapping)
        and set(audit) == _SHORT_CONV_AUDIT_FIELDS
        and audit_hashes_match
        and audit.get("source_inputs_unchanged") is True
        and audit.get("target_templates_unchanged") is True
        and audit.get("evidence_unchanged") is True
        and audit.get("case_authentication_verified") is True
        and audit.get("blank_target_values_used_by_compiler") is False
        and audit.get("target_templates_used_for_shape_and_dtype_only") is True
        and audit.get("returned_tensors_detached") is True
        and isinstance(execution, Mapping)
        and set(execution) == _SHORT_CONV_EXECUTION_FIELDS
        and execution.get("closed_form_linear_solve") is True
        and execution.get("deterministic_barycentric_channel_clustering") is True
        and all(
            type(execution.get(name)) is int and execution[name] == 0
            for name in (
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "backward_calls",
                "optimizer_steps",
                "training_steps",
                "parameters_pruned",
                "parameters_quantized",
            )
        )
        and _no_recursive_success_claims(report)
    )


def _recursive_report_seals_valid(value: Any) -> bool:
    if isinstance(value, Mapping):
        document = dict(value)
        if "report_sha256" in document:
            seal = document.pop("report_sha256")
            if (
                not isinstance(seal, str)
                or _SHA256.fullmatch(seal) is None
                or _sha(document) != seal
            ):
                return False
        return all(_recursive_report_seals_valid(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return all(_recursive_report_seals_valid(child) for child in value)
    return True


def _valid_short_conv_v4_candidate_provenance(
    *,
    candidate_id: str,
    provenance: Any,
    selection: Any,
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    try:
        if not isinstance(provenance, Mapping) or not isinstance(
            selection,
            Mapping,
        ):
            return False
        expected_fields = {
            "schema",
            "kind",
            "candidate_id",
            "policy_sha256",
            "target_dtype",
            "target_dtype_candidate_state_sha256",
            "source_alias_count",
            "source_alias_candidate_ids",
            "sources",
            "fit_only",
            "selection_cases_consumed",
            "validation_cases_consumed",
            "actual_target_dtype_materialization_before_deduplication",
            "execution",
            "report_sha256",
        }
        if set(provenance) != expected_fields:
            return False
        sources = provenance.get("sources")
        aliases = provenance.get("source_alias_candidate_ids")
        alias_count = provenance.get("source_alias_count")
        target_state_sha = provenance.get("target_dtype_candidate_state_sha256")
        if not (
            provenance.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
            and provenance.get("kind") == "authenticated_fit_only_candidate_provenance"
            and provenance.get("candidate_id") == candidate_id
            and provenance.get("policy_sha256") == solver_policy.authentication_sha256
            and provenance.get("target_dtype") == solver_policy.target_dtype
            and isinstance(target_state_sha, str)
            and _SHA256.fullmatch(target_state_sha) is not None
            and selection.get("candidate_state_sha256") == target_state_sha
            and type(alias_count) is int
            and alias_count > 0
            and isinstance(aliases, list)
            and len(aliases) == alias_count
            and len(set(aliases)) == alias_count
            and aliases[0] == candidate_id
            and isinstance(sources, list)
            and len(sources) == alias_count
            and aliases
            == [
                source.get("candidate_id")
                for source in sources
                if isinstance(source, Mapping)
            ]
            and provenance.get("fit_only") is True
            and provenance.get("selection_cases_consumed") == 0
            and provenance.get("validation_cases_consumed") == 0
            and provenance.get(
                "actual_target_dtype_materialization_before_deduplication"
            )
            is True
        ):
            return False
        execution = provenance.get("execution")
        if not (
            isinstance(execution, Mapping)
            and set(execution)
            == {
                "parent_model_forward_calls",
                "target_model_forward_calls",
                "generation_calls",
                "training_steps",
                "backward_calls",
                "optimizer_steps",
                "parameters_pruned",
                "parameters_quantized",
            }
            and all(type(value) is int and value == 0 for value in execution.values())
        ):
            return False
        for source in sources:
            if not isinstance(source, Mapping) or set(source) != {
                "candidate_id",
                "ridge_id",
                "candidate_kind",
                "stage_id",
                "fit_path_state_sha256",
                "candidate_report",
            }:
                return False
            ridge_id = source.get("ridge_id")
            candidate_report = source.get("candidate_report")
            if not (
                isinstance(ridge_id, str)
                and ridge_id in ridge_fit
                and isinstance(ridge_fit[ridge_id], Mapping)
                and isinstance(candidate_report, Mapping)
                and source.get("stage_id") == candidate_report.get("stage_id")
                and source.get("fit_path_state_sha256")
                == candidate_report.get("state_sha256")
                and candidate_report.get("policy_sha256")
                == solver_policy.authentication_sha256
                and candidate_report.get("alpha") == ridge_fit[ridge_id].get("alpha")
                and candidate_report.get("fit_case_authentication_sha256")
                == ridge_fit[ridge_id].get("fit_case_authentication_sha256")
                and candidate_report.get("fit_only") is True
                and candidate_report.get("selection_cases_consumed") == 0
                and candidate_report.get("validation_cases_consumed") == 0
            ):
                return False
            inventory = ridge_fit[ridge_id].get("path_state_inventory")
            if not isinstance(inventory, list) or not inventory:
                return False
            candidate_kind = source.get("candidate_kind")
            if candidate_kind == "accepted_fit_path_anchor":
                inventory_record = {
                    (
                        value.get("stage_id"),
                        value.get("state_sha256"),
                        value.get("candidate_report_sha256"),
                    )
                    for value in inventory
                    if isinstance(value, Mapping)
                }
                if not (
                    candidate_report.get("kind")
                    == "authenticated_fit_only_projection_path_state"
                    and (
                        candidate_report.get("stage_id"),
                        candidate_report.get("state_sha256"),
                        candidate_report.get("report_sha256"),
                    )
                    in inventory_record
                ):
                    return False
            elif candidate_kind == "prior_to_final_trust_path":
                if not (
                    candidate_report.get("kind")
                    == "authenticated_fit_only_projection_trust_path_state"
                    and candidate_report.get("left_stage_id")
                    == inventory[0].get("stage_id")
                    and candidate_report.get("left_state_sha256")
                    == inventory[0].get("state_sha256")
                    and candidate_report.get("left_candidate_report_sha256")
                    == inventory[0].get("candidate_report_sha256")
                    and candidate_report.get("right_stage_id")
                    == inventory[-1].get("stage_id")
                    and candidate_report.get("right_state_sha256")
                    == inventory[-1].get("state_sha256")
                    and candidate_report.get("right_candidate_report_sha256")
                    == inventory[-1].get("candidate_report_sha256")
                    and candidate_report.get("fraction")
                    in fixed_projection_trust_path_fractions()
                ):
                    return False
            elif candidate_kind == "prior_to_first_writer_trust_path":
                first_writer_record = next(
                    (
                        value
                        for value in inventory
                        if value.get("stage_id", "").endswith("-writer")
                    ),
                    None,
                )
                if not (
                    isinstance(first_writer_record, Mapping)
                    and candidate_report.get("kind")
                    == "authenticated_fit_only_projection_trust_path_state"
                    and candidate_report.get("left_stage_id")
                    == inventory[0].get("stage_id")
                    and candidate_report.get("left_state_sha256")
                    == inventory[0].get("state_sha256")
                    and candidate_report.get("left_candidate_report_sha256")
                    == inventory[0].get("candidate_report_sha256")
                    and candidate_report.get("right_stage_id")
                    == first_writer_record.get("stage_id")
                    and candidate_report.get("right_state_sha256")
                    == first_writer_record.get("state_sha256")
                    and candidate_report.get("right_candidate_report_sha256")
                    == first_writer_record.get("candidate_report_sha256")
                    and candidate_report.get("fraction")
                    in fixed_projection_trust_path_fractions()
                ):
                    return False
            else:
                return False
        return True
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_raw_frontier_coverage(
    *,
    candidate_fit: Mapping[str, Any],
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    """Prove every configured pre-dedup path source appears exactly once."""

    try:
        expected_sources: dict[str, tuple[str, str, str]] = {}
        fractions = fixed_projection_trust_path_fractions()
        for ridge_index, alpha in enumerate(solver_policy.ridge_alphas):
            ridge_id = f"ridge-alpha-{ridge_index:02d}-{alpha:.17g}"
            ridge = ridge_fit.get(ridge_id)
            inventory = (
                ridge.get("path_state_inventory")
                if isinstance(ridge, Mapping)
                else None
            )
            if not (
                isinstance(inventory, list)
                and inventory
                and all(isinstance(record, Mapping) for record in inventory)
            ):
                return False
            for path_index, record in enumerate(inventory):
                stage_id = record.get("stage_id")
                if not isinstance(stage_id, str) or not stage_id:
                    return False
                candidate_id = f"{ridge_id}--anchor-{path_index:02d}-{stage_id}"
                expected_sources[candidate_id] = (
                    ridge_id,
                    "accepted_fit_path_anchor",
                    stage_id,
                )

            for fraction_index, fraction in enumerate(fractions):
                stage_id = f"prior-to-final-{fraction_index:02d}-{fraction:.17g}"
                candidate_id = f"{ridge_id}--trust-{fraction_index:02d}-{fraction:.17g}"
                expected_sources[candidate_id] = (
                    ridge_id,
                    "prior_to_final_trust_path",
                    stage_id,
                )

            first_writer_record = next(
                (
                    record
                    for record in inventory
                    if record.get("stage_id", "").endswith("-writer")
                ),
                None,
            )
            if first_writer_record is not None:
                for fraction_index, fraction in enumerate(fractions):
                    stage_id = (
                        f"prior-to-first-writer-{fraction_index:02d}-{fraction:.17g}"
                    )
                    candidate_id = (
                        f"{ridge_id}--writer-trust-{fraction_index:02d}-{fraction:.17g}"
                    )
                    expected_sources[candidate_id] = (
                        ridge_id,
                        "prior_to_first_writer_trust_path",
                        stage_id,
                    )

        actual_sources: dict[str, tuple[Any, Any, Any]] = {}
        for candidate_id, provenance in candidate_fit.items():
            if not isinstance(provenance, Mapping):
                return False
            sources = provenance.get("sources")
            aliases = provenance.get("source_alias_candidate_ids")
            if not (
                isinstance(sources, list)
                and sources
                and isinstance(aliases, list)
                and aliases
                == [
                    source.get("candidate_id") if isinstance(source, Mapping) else None
                    for source in sources
                ]
                and aliases[0] == candidate_id
            ):
                return False
            for source in sources:
                if not isinstance(source, Mapping):
                    return False
                source_id = source.get("candidate_id")
                if not isinstance(source_id, str) or source_id in actual_sources:
                    return False
                actual_sources[source_id] = (
                    source.get("ridge_id"),
                    source.get("candidate_kind"),
                    source.get("stage_id"),
                )
        return actual_sources == expected_sources
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_candidate_pool(
    *,
    pool: Any,
    candidate_fit: Mapping[str, Any],
    candidate_selection: Mapping[str, Any],
    ridge_fit: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    gate_policy: Mapping[str, Any],
    fold_case_seals: Mapping[str, Any],
    split_evaluations: Mapping[str, Any],
) -> bool:
    try:
        if not isinstance(pool, Mapping):
            return False
        expected_fields = {
            "schema",
            "kind",
            "policy_sha256",
            "target_dtype",
            "ridge_candidate_count",
            "raw_candidate_count",
            "unique_target_dtype_candidate_count",
            "target_dtype_duplicate_count",
            "accepted_phase_anchors_included",
            "rejected_phase_states_included",
            "prior_to_final_trust_path_fractions",
            "prior_to_first_writer_trust_path_fractions",
            "selection_feasibility_policy",
            "selection_feasibility_policy_sha256",
            "fold_case_authentication_sha256",
            "projection_case_authentication_sha256",
            "candidate_ids",
            "candidate_state_sha256",
            "fit_only_candidate_generation",
            "selection_cases_consumed_during_generation",
            "validation_cases_consumed_during_generation",
            "report_sha256",
        }
        if set(pool) != expected_fields:
            return False
        counts = {
            name: pool.get(name)
            for name in (
                "ridge_candidate_count",
                "raw_candidate_count",
                "unique_target_dtype_candidate_count",
                "target_dtype_duplicate_count",
            )
        }
        if not all(type(value) is int and value >= 0 for value in counts.values()):
            return False
        candidate_ids = pool.get("candidate_ids")
        state_map = pool.get("candidate_state_sha256")
        projection_case_seals = pool.get("projection_case_authentication_sha256")
        expected_feasibility = {
            "maximum_endpoint_relative_rmse": min(
                float(gate_policy["maximum_fit_endpoint_relative_rmse"]),
                float(gate_policy["maximum_validation_endpoint_relative_rmse"]),
            ),
            "maximum_context_relative_rmse": min(
                float(gate_policy["maximum_fit_endpoint_relative_rmse"]),
                float(gate_policy["maximum_validation_endpoint_relative_rmse"]),
            ),
            "minimum_endpoint_cosine_similarity": max(
                float(gate_policy["minimum_fit_endpoint_cosine_similarity"]),
                float(gate_policy["minimum_validation_endpoint_cosine_similarity"]),
            ),
            "minimum_context_cosine_similarity": max(
                float(gate_policy["minimum_fit_endpoint_cosine_similarity"]),
                float(gate_policy["minimum_validation_endpoint_cosine_similarity"]),
            ),
        }
        expected_feasibility_sha = _sha(
            {
                "schema": LFM2_CPCF_SHORT_CONV_V4_SCHEMA,
                "kind": "selection_feasibility_policy",
                **expected_feasibility,
            }
        )
        raw_alias_count = sum(
            int(value.get("source_alias_count", -1))
            for value in candidate_fit.values()
            if isinstance(value, Mapping)
        )
        split_names = {"fit", "selection", "validation"}
        if not (
            isinstance(fold_case_seals, Mapping)
            and set(fold_case_seals) == split_names
            and pool.get("fold_case_authentication_sha256") == dict(fold_case_seals)
            and isinstance(projection_case_seals, Mapping)
            and set(projection_case_seals) == split_names
            and all(
                isinstance(projection_case_seals[split], list)
                and projection_case_seals[split]
                and all(
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                    for value in projection_case_seals[split]
                )
                for split in split_names
            )
            and isinstance(split_evaluations, Mapping)
            and set(split_evaluations) == split_names
        ):
            return False

        def evaluation_projection_seals(value: Any) -> list[Any] | None:
            if not isinstance(value, Mapping) or not isinstance(
                value.get("cases"),
                list,
            ):
                return None
            return [
                case.get("projection_case_authentication_sha256")
                if isinstance(case, Mapping)
                else None
                for case in value["cases"]
            ]

        return bool(
            pool.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA
            and pool.get("kind") == "authenticated_target_dtype_candidate_pool"
            and pool.get("policy_sha256") == solver_policy.authentication_sha256
            and pool.get("target_dtype") == solver_policy.target_dtype
            and counts["ridge_candidate_count"] == len(ridge_fit)
            and counts["unique_target_dtype_candidate_count"] == len(candidate_fit)
            and counts["raw_candidate_count"] == raw_alias_count
            and counts["raw_candidate_count"]
            == counts["unique_target_dtype_candidate_count"]
            + counts["target_dtype_duplicate_count"]
            and _valid_short_conv_v4_raw_frontier_coverage(
                candidate_fit=candidate_fit,
                ridge_fit=ridge_fit,
                solver_policy=solver_policy,
            )
            and isinstance(candidate_ids, list)
            and candidate_ids == sorted(candidate_fit)
            and len(candidate_ids) == len(set(candidate_ids))
            and isinstance(state_map, Mapping)
            and set(state_map) == set(candidate_fit)
            and all(
                isinstance(state_map[candidate_id], str)
                and _SHA256.fullmatch(state_map[candidate_id]) is not None
                and isinstance(candidate_selection[candidate_id], Mapping)
                and state_map[candidate_id]
                == candidate_selection[candidate_id].get("candidate_state_sha256")
                for candidate_id in candidate_fit
            )
            and pool.get("accepted_phase_anchors_included") is True
            and pool.get("rejected_phase_states_included") is False
            and pool.get("prior_to_final_trust_path_fractions")
            == list(fixed_projection_trust_path_fractions())
            and pool.get("prior_to_first_writer_trust_path_fractions")
            == list(fixed_projection_trust_path_fractions())
            and pool.get("selection_feasibility_policy") == expected_feasibility
            and pool.get("selection_feasibility_policy_sha256")
            == expected_feasibility_sha
            and all(
                ridge.get("fit_case_authentication_sha256")
                == projection_case_seals["fit"]
                for ridge in ridge_fit.values()
                if isinstance(ridge, Mapping)
            )
            and all(
                evaluation_projection_seals(selection)
                == projection_case_seals["selection"]
                for selection in candidate_selection.values()
            )
            and all(
                evaluation_projection_seals(split_evaluations[split])
                == projection_case_seals[split]
                for split in split_names
            )
            and pool.get("fit_only_candidate_generation") is True
            and pool.get("selection_cases_consumed_during_generation") == 0
            and pool.get("validation_cases_consumed_during_generation") == 0
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_ridge_fit_reports(
    ridge_fit: Any,
    *,
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
) -> bool:
    try:
        if not isinstance(ridge_fit, Mapping):
            return False
        expected_ids = [
            f"ridge-alpha-{index:02d}-{alpha:.17g}"
            for index, alpha in enumerate(solver_policy.ridge_alphas)
        ]
        if set(ridge_fit) != set(expected_ids):
            return False
        for ridge_id, alpha in zip(expected_ids, solver_policy.ridge_alphas):
            value = ridge_fit[ridge_id]
            if not isinstance(value, Mapping):
                return False
            inventory = value.get("path_state_inventory")
            fit_case_seals = value.get("fit_case_authentication_sha256")
            if not (
                value.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
                and value.get("kind") == "fixed_bounded_sampled_cp_variable_projection"
                and value.get("policy_sha256") == solver_policy.authentication_sha256
                and value.get("alpha") == alpha
                and value.get("fit_only") is True
                and value.get("selection_cases_consumed") == 0
                and value.get("validation_cases_consumed") == 0
                and isinstance(inventory, list)
                and inventory
                and all(
                    isinstance(item, Mapping)
                    and set(item)
                    == {
                        "stage_id",
                        "state_sha256",
                        "candidate_report_sha256",
                    }
                    and isinstance(item.get("stage_id"), str)
                    and isinstance(item.get("state_sha256"), str)
                    and _SHA256.fullmatch(item["state_sha256"]) is not None
                    and isinstance(item.get("candidate_report_sha256"), str)
                    and _SHA256.fullmatch(item["candidate_report_sha256"]) is not None
                    for item in inventory
                )
                and value.get("path_state_count") == len(inventory)
                and value.get("path_state_inventory_sha256") == _sha(inventory)
                and value.get("rejected_phases_excluded_from_path") is True
                and value.get("final_state_is_last_path_state") is True
                and inventory[-1].get("state_sha256") == value.get("state_sha256")
                and isinstance(fit_case_seals, list)
                and fit_case_seals
                and all(
                    isinstance(item, str) and _SHA256.fullmatch(item) is not None
                    for item in fit_case_seals
                )
            ):
                return False
        return True
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_selection_ranking(
    ranking: Any,
    *,
    candidate_selection: Mapping[str, Any],
    solver_policy: Lfm2CpcfShortConvV4SolverPolicy,
    feasibility_policy: Mapping[str, Any],
) -> bool:
    try:
        if not isinstance(ranking, Mapping):
            return False
        recomputed = rank_short_conv_v4_selection_candidates(
            candidate_selection,
            policy_sha256=solver_policy.authentication_sha256,
            feasibility_policy=feasibility_policy,
        )
        return dict(ranking) == dict(recomputed)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_behavior_gate(
    gate: Any,
    *,
    gate_policy: Mapping[str, Any],
    fit_evaluation: Mapping[str, Any],
    validation_evaluation: Mapping[str, Any],
    full_source_coverage: bool,
) -> bool:
    try:
        expected_checks = {
            "fit_endpoint_relative_rmse": (
                float(fit_evaluation["maximum_case_relative_rmse"])
                <= float(gate_policy["maximum_fit_endpoint_relative_rmse"])
            ),
            "validation_endpoint_relative_rmse": (
                float(validation_evaluation["maximum_case_relative_rmse"])
                <= float(gate_policy["maximum_validation_endpoint_relative_rmse"])
            ),
            "fit_endpoint_cosine_similarity": (
                float(fit_evaluation["minimum_case_cosine_similarity"])
                >= float(gate_policy["minimum_fit_endpoint_cosine_similarity"])
            ),
            "validation_endpoint_cosine_similarity": (
                float(validation_evaluation["minimum_case_cosine_similarity"])
                >= float(gate_policy["minimum_validation_endpoint_cosine_similarity"])
            ),
            "fit_context_relative_rmse": (
                fit_evaluation.get("context_evidence_present") is True
                and float(fit_evaluation["maximum_case_context_relative_rmse"])
                <= float(gate_policy["maximum_fit_endpoint_relative_rmse"])
            ),
            "validation_context_relative_rmse": (
                validation_evaluation.get("context_evidence_present") is True
                and float(validation_evaluation["maximum_case_context_relative_rmse"])
                <= float(gate_policy["maximum_validation_endpoint_relative_rmse"])
            ),
            "fit_context_cosine_similarity": (
                fit_evaluation.get("context_evidence_present") is True
                and float(fit_evaluation["minimum_case_context_cosine_similarity"])
                >= float(gate_policy["minimum_fit_endpoint_cosine_similarity"])
            ),
            "validation_context_cosine_similarity": (
                validation_evaluation.get("context_evidence_present") is True
                and float(
                    validation_evaluation["minimum_case_context_cosine_similarity"]
                )
                >= float(gate_policy["minimum_validation_endpoint_cosine_similarity"])
            ),
            "full_source_numeric_participation": full_source_coverage,
            "actual_fit_target_dtype_replay": (
                fit_evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
            ),
            "actual_validation_target_dtype_replay": (
                validation_evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
            ),
        }
        expected_failed = [
            name for name, passed in expected_checks.items() if not passed
        ]
        return bool(
            isinstance(gate, Mapping)
            and set(gate) == {"policy", "checks", "failed_checks", "passed", "enforced"}
            and gate.get("policy") == dict(gate_policy)
            and gate.get("checks") == expected_checks
            and gate.get("failed_checks") == expected_failed
            and gate.get("passed") is all(expected_checks.values())
            and gate.get("enforced") is gate_policy.get("enforce")
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_legacy_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate immutable v4 failure-v1 receipts emitted before path selection."""

    try:
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )
        gate_policy = report.get("gate_policy")
        gate = report.get("behavior_gate")
        checks = gate.get("checks") if isinstance(gate, Mapping) else None
        failed_checks = gate.get("failed_checks") if isinstance(gate, Mapping) else None
        case_seals = report.get("case_authentication_sha256")
        execution = report.get("execution")
        candidate_fit = report.get("candidate_fit_reports")
        candidate_selection = report.get("candidate_selection_reports")
        ranking = report.get("selection_ranking")
        selected = report.get("selected_candidate_id")
        fit_evaluation = report.get("fit_evaluation")
        selection_evaluation = report.get("selection_evaluation")
        validation_evaluation = report.get("validation_evaluation")
        selected_state_sha = (
            selection_evaluation.get("candidate_state_sha256")
            if isinstance(selection_evaluation, Mapping)
            else None
        )
        dimensions = (
            report.get("source_hidden_size"),
            report.get("target_hidden_size"),
            report.get("source_fir_length"),
            report.get("target_fir_length"),
        )
        evaluations = (
            ("fit", fit_evaluation),
            ("selection", selection_evaluation),
            ("validation", validation_evaluation),
        )
        return bool(
            set(report) == _SHORT_CONV_V4_FAILURE_V1_REPORT_FIELDS
            and report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA
            and report.get("solver_schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and report.get("solver_policy_sha256")
            == solver_policy.authentication_sha256
            and solver_policy.verify_authentication()
            and isinstance(gate_policy, Mapping)
            and set(gate_policy) == _SHORT_CONV_GATE_POLICY_FIELDS
            and gate_policy.get("enforce") is True
            and all(
                not isinstance(gate_policy.get(name), bool)
                and isinstance(gate_policy.get(name), (int, float))
                and math.isfinite(float(gate_policy[name]))
                for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
            )
            and isinstance(gate, Mapping)
            and set(gate) == {"policy", "checks", "failed_checks", "passed", "enforced"}
            and gate.get("policy") == dict(gate_policy)
            and gate.get("passed") is False
            and gate.get("enforced") is True
            and isinstance(checks, Mapping)
            and set(checks) == _SHORT_CONV_V4_GATE_CHECK_FIELDS
            and all(isinstance(value, bool) for value in checks.values())
            and isinstance(failed_checks, list)
            and failed_checks
            and len(failed_checks) == len(set(failed_checks))
            and set(failed_checks)
            == {name for name, passed in checks.items() if not passed}
            and isinstance(case_seals, Mapping)
            and set(case_seals) == {"fit", "selection", "validation"}
            and all(
                isinstance(values, list)
                and values
                and all(
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                    for value in values
                )
                for values in case_seals.values()
            )
            and all(type(value) is int and value > 0 for value in dimensions)
            and report["source_hidden_size"] >= report["target_hidden_size"]
            and isinstance(
                report.get("hidden_width_compression_ratio"),
                (int, float),
            )
            and not isinstance(
                report.get("hidden_width_compression_ratio"),
                bool,
            )
            and math.isfinite(float(report["hidden_width_compression_ratio"]))
            and float(report["hidden_width_compression_ratio"]) >= 1.0
            and isinstance(report.get("basis"), Mapping)
            and isinstance(report.get("input_lift"), Mapping)
            and isinstance(report.get("transported_priors"), Mapping)
            and isinstance(candidate_fit, Mapping)
            and isinstance(candidate_selection, Mapping)
            and candidate_fit
            and set(candidate_fit) == set(candidate_selection)
            and isinstance(selected, str)
            and selected in candidate_fit
            and isinstance(selected_state_sha, str)
            and _SHA256.fullmatch(selected_state_sha) is not None
            and isinstance(candidate_selection[selected], Mapping)
            and candidate_selection[selected].get("split") == "selection"
            and candidate_selection[selected].get("candidate_state_sha256")
            == selected_state_sha
            and isinstance(ranking, Mapping)
            and ranking.get("kind") == "selection_only_candidate_ranking"
            and ranking.get("selected_candidate_id") == selected
            and ranking.get("validation_isolation") is True
            and isinstance(ranking.get("ranking"), list)
            and ranking["ranking"]
            and ranking["ranking"][0].get("candidate_id") == selected
            and ranking["ranking"][0].get("candidate_state_sha256")
            == selected_state_sha
            and all(
                isinstance(evaluation, Mapping)
                and evaluation.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
                and evaluation.get("kind") == "target_dtype_evaluation"
                and evaluation.get("split") == split
                and evaluation.get("policy_sha256")
                == solver_policy.authentication_sha256
                and evaluation.get("candidate_state_sha256") == selected_state_sha
                and evaluation.get(
                    "actual_parameter_and_intermediate_target_dtype_replay"
                )
                is True
                for split, evaluation in evaluations
            )
            and isinstance(report.get("compiled_state_sha256"), str)
            and _SHA256.fullmatch(report["compiled_state_sha256"]) is not None
            and all(
                type(report.get(name)) is int and report[name] >= 0
                for name in (
                    "source_channels_selected",
                    "source_channels_dropped",
                    "source_channels_pruned",
                    "source_channels_quantized",
                )
            )
            and report.get("source_channels_pruned") == 0
            and report.get("source_channels_quantized") == 0
            and report.get("fit_only_parameter_solves") is True
            and report.get("selection_used_only_for_candidate_ranking") is True
            and report.get("validation_used_only_after_candidate_freeze") is True
            and isinstance(execution, Mapping)
            and set(execution) == _SHORT_CONV_V4_EXECUTION_FIELDS
            and all(type(value) is int and value == 0 for value in execution.values())
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError):
        return False


def _valid_short_conv_v4_validation_report_unsafe(
    report: Mapping[str, Any],
) -> bool:
    if report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA:
        return _valid_short_conv_v4_legacy_validation_report(report)
    try:
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )
    except Exception:
        return False
    gate_policy = report.get("gate_policy")
    gate = report.get("behavior_gate")
    checks = gate.get("checks") if isinstance(gate, Mapping) else None
    failed_checks = gate.get("failed_checks") if isinstance(gate, Mapping) else None
    case_seals = report.get("case_authentication_sha256")
    execution = report.get("execution")
    fit_evaluation = report.get("fit_evaluation")
    selection_evaluation = report.get("selection_evaluation")
    validation_evaluation = report.get("validation_evaluation")
    ranking = report.get("selection_ranking")
    ridge_fit = report.get("ridge_fit_reports")
    candidate_fit = report.get("candidate_fit_reports")
    candidate_selection = report.get("candidate_selection_reports")
    candidate_pool = report.get("candidate_pool_report")
    selected_candidate = report.get("selected_candidate_id")
    compiled_state_sha = report.get("compiled_state_sha256")
    basis = report.get("basis")
    transported_priors = report.get("transported_priors")
    full_source_coverage = bool(
        isinstance(basis, Mapping)
        and basis.get("full_source_numeric_participation") is True
        and isinstance(transported_priors, Mapping)
        and transported_priors.get("source_channels_dropped") == 0
    )
    selected_candidate_state_sha = (
        selection_evaluation.get("candidate_state_sha256")
        if isinstance(selection_evaluation, Mapping)
        else None
    )
    dimensions = (
        report.get("source_hidden_size"),
        report.get("target_hidden_size"),
        report.get("source_fir_length"),
        report.get("target_fir_length"),
    )
    evaluations = (
        ("fit", fit_evaluation),
        ("selection", selection_evaluation),
        ("validation", validation_evaluation),
    )
    return bool(
        set(report) == _SHORT_CONV_V4_FAILURE_REPORT_FIELDS
        and report.get("schema") == LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA
        and report.get("solver_schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
        and report.get("solver_policy_sha256") == solver_policy.authentication_sha256
        and solver_policy.verify_authentication()
        and isinstance(gate_policy, Mapping)
        and set(gate_policy) == _SHORT_CONV_GATE_POLICY_FIELDS
        and gate_policy.get("enforce") is True
        and all(
            not isinstance(gate_policy.get(name), bool)
            and isinstance(gate_policy.get(name), (int, float))
            and math.isfinite(float(gate_policy[name]))
            for name in _SHORT_CONV_GATE_POLICY_FIELDS - {"enforce"}
        )
        and isinstance(gate, Mapping)
        and set(gate) == {"policy", "checks", "failed_checks", "passed", "enforced"}
        and gate.get("policy") == dict(gate_policy)
        and gate.get("passed") is False
        and gate.get("enforced") is True
        and isinstance(checks, Mapping)
        and set(checks) == _SHORT_CONV_V4_GATE_CHECK_FIELDS
        and all(isinstance(value, bool) for value in checks.values())
        and isinstance(failed_checks, list)
        and failed_checks
        and len(failed_checks) == len(set(failed_checks))
        and set(failed_checks)
        == {name for name, passed in checks.items() if not passed}
        and _valid_short_conv_v4_behavior_gate(
            gate,
            gate_policy=gate_policy,
            fit_evaluation=fit_evaluation,
            validation_evaluation=validation_evaluation,
            full_source_coverage=full_source_coverage,
        )
        and isinstance(case_seals, Mapping)
        and set(case_seals) == {"fit", "selection", "validation"}
        and all(
            isinstance(values, list)
            and values
            and all(
                isinstance(value, str) and _SHA256.fullmatch(value) is not None
                for value in values
            )
            for values in case_seals.values()
        )
        and all(type(value) is int and value > 0 for value in dimensions)
        and report["source_hidden_size"] >= report["target_hidden_size"]
        and isinstance(report.get("hidden_width_compression_ratio"), (int, float))
        and not isinstance(report.get("hidden_width_compression_ratio"), bool)
        and math.isfinite(float(report["hidden_width_compression_ratio"]))
        and float(report["hidden_width_compression_ratio"]) >= 1.0
        and isinstance(basis, Mapping)
        and isinstance(report.get("input_lift"), Mapping)
        and isinstance(transported_priors, Mapping)
        and _valid_short_conv_v4_ridge_fit_reports(
            ridge_fit,
            solver_policy=solver_policy,
        )
        and isinstance(candidate_fit, Mapping)
        and isinstance(candidate_selection, Mapping)
        and candidate_fit
        and set(candidate_fit) == set(candidate_selection)
        and all(
            _valid_short_conv_v4_candidate_provenance(
                candidate_id=candidate_id,
                provenance=candidate_fit[candidate_id],
                selection=candidate_selection[candidate_id],
                ridge_fit=ridge_fit,
                solver_policy=solver_policy,
            )
            for candidate_id in candidate_fit
        )
        and _valid_short_conv_v4_candidate_pool(
            pool=candidate_pool,
            candidate_fit=candidate_fit,
            candidate_selection=candidate_selection,
            ridge_fit=ridge_fit,
            solver_policy=solver_policy,
            gate_policy=gate_policy,
            fold_case_seals=case_seals,
            split_evaluations={
                "fit": fit_evaluation,
                "selection": selection_evaluation,
                "validation": validation_evaluation,
            },
        )
        and isinstance(selected_candidate, str)
        and bool(selected_candidate)
        and selected_candidate in candidate_fit
        and isinstance(selected_candidate_state_sha, str)
        and _SHA256.fullmatch(selected_candidate_state_sha) is not None
        and isinstance(candidate_selection[selected_candidate], Mapping)
        and candidate_selection[selected_candidate].get("split") == "selection"
        and candidate_selection[selected_candidate].get("candidate_state_sha256")
        == selected_candidate_state_sha
        and _valid_short_conv_v4_selection_ranking(
            ranking,
            candidate_selection=candidate_selection,
            solver_policy=solver_policy,
            feasibility_policy=candidate_pool["selection_feasibility_policy"],
        )
        and ranking.get("selected_candidate_id") == selected_candidate
        and ranking["ranking"][0].get("candidate_state_sha256")
        == selected_candidate_state_sha
        and all(
            isinstance(evaluation, Mapping)
            and evaluation.get("schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and evaluation.get("kind") == "target_dtype_evaluation"
            and evaluation.get("split") == split
            and evaluation.get("policy_sha256") == solver_policy.authentication_sha256
            and evaluation.get("candidate_state_sha256") == selected_candidate_state_sha
            and evaluation.get("actual_parameter_and_intermediate_target_dtype_replay")
            is True
            for split, evaluation in evaluations
        )
        and isinstance(compiled_state_sha, str)
        and _SHA256.fullmatch(compiled_state_sha) is not None
        and all(
            type(report.get(name)) is int and report[name] >= 0
            for name in (
                "source_channels_selected",
                "source_channels_dropped",
                "source_channels_pruned",
                "source_channels_quantized",
            )
        )
        and report.get("source_channels_pruned") == 0
        and report.get("source_channels_quantized") == 0
        and report.get("fit_only_parameter_solves") is True
        and report.get("selection_used_only_for_candidate_ranking") is True
        and report.get("validation_used_only_after_candidate_freeze") is True
        and isinstance(execution, Mapping)
        and set(execution) == _SHORT_CONV_V4_EXECUTION_FIELDS
        and all(type(value) is int and value == 0 for value in execution.values())
        and report.get("compiled_short_conv_fold_issued") is False
        and report.get("promotion_claimed") is False
        and report.get("promotable_candidate") is False
        and report.get("failure_evidence_only") is True
        and _recursive_report_seals_valid(report)
        and _no_recursive_success_claims(report)
    )


def _valid_short_conv_v4_validation_report(
    report: Mapping[str, Any],
) -> bool:
    try:
        return _valid_short_conv_v4_validation_report_unsafe(report)
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_v5_selector_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate a selector-only failure without pretending it is v4."""

    try:
        receipt = report.get("preinstall_candidate_selector_receipt")
        selector_report = (
            receipt.get("selector_report") if isinstance(receipt, Mapping) else None
        )
        selection = report.get("selection_evaluation")
        candidate_reports = report.get("candidate_selection_reports")
        gate = report.get("behavior_gate")
        execution = report.get("execution")
        selected_id = report.get("selected_candidate_id")
        selected_sha = (
            selection.get("candidate_state_sha256")
            if isinstance(selection, Mapping)
            else None
        )
        receipt_body = dict(receipt) if isinstance(receipt, Mapping) else {}
        receipt_seal = receipt_body.pop("report_sha256", None)
        selector_body = (
            dict(selector_report) if isinstance(selector_report, Mapping) else {}
        )
        selector_seal = selector_body.pop("report_sha256", None)
        return bool(
            report.get("schema") == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA
            and isinstance(receipt, Mapping)
            and receipt.get("schema") == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA
            and isinstance(receipt_seal, str)
            and _SHA256.fullmatch(receipt_seal) is not None
            and _sha(receipt_body) == receipt_seal
            and report.get("preinstall_candidate_selector_receipt_sha256")
            == receipt_seal
            and isinstance(selector_report, Mapping)
            and isinstance(selector_seal, str)
            and _SHA256.fullmatch(selector_seal) is not None
            and _sha(selector_body) == selector_seal
            and receipt.get("selector_report_sha256") == selector_seal
            and receipt.get("all_live_candidate_states_verified_before_callback")
            is True
            and receipt.get("all_live_candidate_states_verified_after_callback") is True
            and receipt.get("live_candidate_state_bytes_unchanged") is True
            and receipt.get("candidate_state_tensors_exposed_to_selector") is False
            and receipt.get("fit_reports_exposed_to_selector") is False
            and receipt.get("validation_reports_exposed_to_selector") is False
            and receipt.get("post_install_rescoring") is False
            and report.get("post_install_rescoring_performed") is False
            and report.get("preinstall_candidate_selector_active") is True
            and report.get("legacy_v4_report_claimed") is False
            and isinstance(selected_id, str)
            and bool(selected_id)
            and receipt.get("selected_candidate_id") == selected_id
            and selector_report.get("selected_candidate_id") == selected_id
            and isinstance(selected_sha, str)
            and _SHA256.fullmatch(selected_sha) is not None
            and receipt.get("selected_candidate_state_sha256") == selected_sha
            and selector_report.get("selected_candidate_state_sha256") == selected_sha
            and isinstance(candidate_reports, Mapping)
            and isinstance(candidate_reports.get(selected_id), Mapping)
            and candidate_reports[selected_id].get("candidate_state_sha256")
            == selected_sha
            and isinstance(gate, Mapping)
            and gate.get("passed") is False
            and gate.get("enforced") is True
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and isinstance(execution, Mapping)
            and all(type(value) is int and value == 0 for value in execution.values())
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_v5_macrocell_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate a whole-macrocell gate failure and its embedded evidence."""

    try:
        receipt = report.get("macrocell_candidate_selector_receipt")
        decision = (
            receipt.get("selector_decision") if isinstance(receipt, Mapping) else None
        )
        frontier = receipt.get("frontier") if isinstance(receipt, Mapping) else None
        selection = report.get("selection_evaluation")
        candidate_reports = report.get("candidate_selection_reports")
        selected_id = report.get("selected_candidate_id")
        selected_sha = (
            selection.get("candidate_state_sha256")
            if isinstance(selection, Mapping)
            else None
        )
        receipt_body = dict(receipt) if isinstance(receipt, Mapping) else {}
        receipt_sha = receipt_body.pop("report_sha256", None)
        decision_body = dict(decision) if isinstance(decision, Mapping) else {}
        decision_sha = decision_body.pop("report_sha256", None)
        frontier_body = dict(frontier) if isinstance(frontier, Mapping) else {}
        frontier_sha = frontier_body.pop("report_sha256", None)
        gate = report.get("behavior_gate")
        execution = report.get("execution")
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )
        candidate_fit = report.get("candidate_fit_reports")
        candidate_selection = report.get("candidate_selection_reports")
        ridge_fit = report.get("ridge_fit_reports")
        candidate_pool = report.get("candidate_pool_report")
        ranking = report.get("selection_ranking")
        gate_policy = report.get("gate_policy")
        case_seals = report.get("case_authentication_sha256")
        split_evaluations = {
            "fit": report.get("fit_evaluation"),
            "selection": report.get("selection_evaluation"),
            "validation": report.get("validation_evaluation"),
        }
        legacy_candidate_pool = (
            {
                **dict(candidate_pool),
                "schema": LFM2_CPCF_SHORT_CONV_V4_FOLD_SCHEMA,
            }
            if isinstance(candidate_pool, Mapping)
            else candidate_pool
        )
        if isinstance(legacy_candidate_pool, dict):
            legacy_candidate_pool.pop("report_sha256", None)
            legacy_candidate_pool["report_sha256"] = _sha(legacy_candidate_pool)
        legacy_candidate_evidence_valid = bool(
            "numerical_frontier" in report
            or (
                isinstance(candidate_fit, Mapping)
                and candidate_fit
                and isinstance(candidate_selection, Mapping)
                and set(candidate_fit) == set(candidate_selection)
                and _valid_short_conv_v4_ridge_fit_reports(
                    ridge_fit,
                    solver_policy=solver_policy,
                )
                and all(
                    _valid_short_conv_v4_candidate_provenance(
                        candidate_id=candidate_id,
                        provenance=candidate_fit[candidate_id],
                        selection=candidate_selection[candidate_id],
                        ridge_fit=ridge_fit,
                        solver_policy=solver_policy,
                    )
                    for candidate_id in candidate_fit
                )
                and isinstance(gate_policy, Mapping)
                and isinstance(case_seals, Mapping)
                and _valid_short_conv_v4_candidate_pool(
                    pool=legacy_candidate_pool,
                    candidate_fit=candidate_fit,
                    candidate_selection=candidate_selection,
                    ridge_fit=ridge_fit,
                    solver_policy=solver_policy,
                    gate_policy=gate_policy,
                    fold_case_seals=case_seals,
                    split_evaluations=split_evaluations,
                )
                and _valid_short_conv_v4_selection_ranking(
                    ranking,
                    candidate_selection=candidate_selection,
                    solver_policy=solver_policy,
                    feasibility_policy=candidate_pool["selection_feasibility_policy"],
                )
            )
        )
        return bool(
            report.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA
            and isinstance(receipt, Mapping)
            and receipt.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_RECEIPT_SCHEMA
            and _SHA256.fullmatch(str(receipt_sha)) is not None
            and _sha(receipt_body) == receipt_sha
            and report.get("macrocell_candidate_selector_receipt_sha256") == receipt_sha
            and isinstance(decision, Mapping)
            and _SHA256.fullmatch(str(decision_sha)) is not None
            and _sha(decision_body) == decision_sha
            and receipt.get("selector_decision_sha256") == decision_sha
            and isinstance(frontier, Mapping)
            and _SHA256.fullmatch(str(frontier_sha)) is not None
            and _sha(frontier_body) == frontier_sha
            and receipt.get("frontier_sha256") == frontier_sha
            and decision.get("frontier_sha256") == frontier_sha
            and decision.get("selected_candidate_id") == selected_id
            and decision.get("selected_candidate_state_sha256") == selected_sha
            and receipt.get("selected_candidate_id") == selected_id
            and receipt.get("selected_candidate_state_sha256") == selected_sha
            and isinstance(candidate_reports, Mapping)
            and isinstance(candidate_reports.get(selected_id), Mapping)
            and candidate_reports[selected_id].get("candidate_state_sha256")
            == selected_sha
            and solver_policy.verify_authentication()
            and report.get("solver_policy_sha256")
            == solver_policy.authentication_sha256
            and legacy_candidate_evidence_valid
            and receipt.get("candidate_state_tensors_exposed_to_macrocell_selector")
            is True
            and receipt.get("fit_provenance_exposed_to_macrocell_selector") is True
            and receipt.get("operator_selection_reports_exposed_to_macrocell_selector")
            is True
            and receipt.get("validation_evidence_exposed_to_macrocell_selector")
            is False
            and receipt.get("validation_reports_consumed") == 0
            and decision.get("validation_accessible") is False
            and decision.get("validation_reports_consumed") == 0
            and decision.get("validation_replays") == 0
            and report.get("validation_used_for_macrocell_selection") is False
            and report.get("validation_frozen_until_after_macrocell_winner") is True
            and report.get("post_install_rescoring_performed") is False
            and report.get("macrocell_candidate_selector_active") is True
            and isinstance(gate, Mapping)
            and gate.get("passed") is False
            and gate.get("enforced") is True
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and isinstance(execution, Mapping)
            and all(type(value) is int and value == 0 for value in execution.values())
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_exact_zero_execution(
    value: Any,
    *,
    fields: frozenset[str],
) -> bool:
    return bool(
        isinstance(value, Mapping)
        and set(value) == fields
        and all(type(value[field]) is int and value[field] == 0 for field in fields)
    )


def _valid_short_conv_v5_selector_receipt_only(
    report: Mapping[str, Any],
) -> bool:
    receipt = report.get("preinstall_candidate_selector_receipt")
    if not isinstance(receipt, Mapping):
        return False
    selector_report = receipt.get("selector_report")
    candidate_reports = report.get("candidate_selection_reports")
    selected_id = report.get("selected_candidate_id")
    selection = report.get("selection_evaluation")
    selected_sha = (
        selection.get("candidate_state_sha256")
        if isinstance(selection, Mapping)
        else None
    )
    receipt_body = dict(receipt)
    receipt_seal = receipt_body.pop("report_sha256", None)
    selector_body = (
        dict(selector_report) if isinstance(selector_report, Mapping) else {}
    )
    selector_seal = selector_body.pop("report_sha256", None)
    receipt_execution = receipt.get("execution")
    selector_execution = (
        selector_report.get("execution")
        if isinstance(selector_report, Mapping)
        else None
    )
    return bool(
        receipt.get("schema") == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_RECEIPT_SCHEMA
        and isinstance(receipt_seal, str)
        and _SHA256.fullmatch(receipt_seal) is not None
        and _sha(receipt_body) == receipt_seal
        and report.get("preinstall_candidate_selector_receipt_sha256") == receipt_seal
        and isinstance(selector_report, Mapping)
        and isinstance(selector_seal, str)
        and _SHA256.fullmatch(selector_seal) is not None
        and _sha(selector_body) == selector_seal
        and receipt.get("selector_report_sha256") == selector_seal
        and receipt.get("all_live_candidate_states_verified_before_callback") is True
        and receipt.get("all_live_candidate_states_verified_after_callback") is True
        and receipt.get("live_candidate_state_bytes_unchanged") is True
        and receipt.get("candidate_state_tensors_exposed_to_selector") is False
        and receipt.get("fit_reports_exposed_to_selector") is False
        and receipt.get("validation_reports_exposed_to_selector") is False
        and receipt.get("post_install_rescoring") is False
        and _valid_exact_zero_execution(
            receipt_execution,
            fields=_SHORT_CONV_V5_SELECTOR_EXECUTION_FIELDS,
        )
        and _valid_exact_zero_execution(
            selector_execution,
            fields=_SHORT_CONV_V5_SELECTOR_EXECUTION_FIELDS,
        )
        and report.get("post_install_rescoring_performed") is False
        and report.get("preinstall_candidate_selector_active") is True
        and receipt.get("selected_candidate_id") == selected_id
        and selector_report.get("selected_candidate_id") == selected_id
        and receipt.get("selected_candidate_state_sha256") == selected_sha
        and selector_report.get("selected_candidate_state_sha256") == selected_sha
        and isinstance(candidate_reports, Mapping)
        and isinstance(candidate_reports.get(selected_id), Mapping)
        and candidate_reports[selected_id].get("candidate_state_sha256") == selected_sha
    )


def _valid_short_conv_v5_numerical_frontier_validation_report(
    report: Mapping[str, Any],
) -> bool:
    """Validate explicit v5 numerical evidence without weakening v4."""

    try:
        request = report.get("numerical_frontier_request")
        frontier = report.get("numerical_frontier")
        candidate_fit = report.get("candidate_fit_reports")
        candidate_selection = report.get("candidate_selection_reports")
        pool = report.get("candidate_pool_report")
        ranking = report.get("selection_ranking")
        fit_evaluation = report.get("fit_evaluation")
        selection_evaluation = report.get("selection_evaluation")
        validation_evaluation = report.get("validation_evaluation")
        gate = report.get("behavior_gate")
        gate_policy = report.get("gate_policy")
        case_seals = report.get("case_authentication_sha256")
        execution = report.get("execution")
        solver_policy = parse_lfm2_cpcf_short_conv_v4_solver_policy(
            report.get("solver_policy")
        )

        def sealed(
            value: Any,
            schema: str,
        ) -> bool:
            if not isinstance(value, Mapping) or value.get("schema") != schema:
                return False
            body = dict(value)
            seal = body.pop("report_sha256", None)
            return bool(
                isinstance(seal, str)
                and _SHA256.fullmatch(seal) is not None
                and _sha(body) == seal
            )

        def valid_evaluation(
            value: Any,
            *,
            split: str,
            candidate_state_sha256: str,
        ) -> bool:
            if (
                not sealed(value, LFM2_CPCF_SHORT_CONV_V4_SCHEMA)
                or set(value) != _SHORT_CONV_V4_EVALUATION_FIELDS
            ):
                return False
            cases = value.get("cases")
            if not (
                value.get("kind") == "target_dtype_evaluation"
                and value.get("policy_sha256") == solver_policy.authentication_sha256
                and value.get("target_dtype") == solver_policy.target_dtype
                and value.get("split") == split
                and value.get("case_balanced") is True
                and isinstance(cases, list)
                and cases
                and type(value.get("case_count")) is int
                and value.get("case_count") == len(cases)
                and isinstance(value.get("evidence_sha256"), str)
                and _SHA256.fullmatch(value["evidence_sha256"]) is not None
                and value.get("candidate_state_sha256") == candidate_state_sha256
                and value.get("context_evidence_present") is True
                and value.get("actual_parameter_and_intermediate_target_dtype_replay")
                is True
                and value.get("final_output_materialization_only") is False
                and value.get("selection_used_for_parameter_fitting") is False
                and value.get("validation_used_for_parameter_fitting") is False
            ):
                return False
            case_ids = []
            for case in cases:
                if not (
                    isinstance(case, Mapping)
                    and set(case) == _SHORT_CONV_V4_EVALUATION_CASE_FIELDS
                    and isinstance(case.get("case_id"), str)
                    and bool(case["case_id"].strip())
                    and type(case.get("rows")) is int
                    and case["rows"] > 0
                    and type(case.get("width")) is int
                    and case["width"] > 0
                    and type(case.get("context_rows")) is int
                    and case["context_rows"] > 0
                    and type(case.get("context_width")) is int
                    and case["context_width"] > 0
                    and isinstance(
                        case.get("projection_case_authentication_sha256"),
                        str,
                    )
                    and _SHA256.fullmatch(case["projection_case_authentication_sha256"])
                    is not None
                    and isinstance(
                        case.get("case_authentication_sha256"),
                        str,
                    )
                    and _SHA256.fullmatch(case["case_authentication_sha256"])
                    is not None
                    and case.get("candidate_state_sha256") == candidate_state_sha256
                    and case.get(
                        "actual_parameter_and_intermediate_target_dtype_replay"
                    )
                    is True
                ):
                    return False
                case_ids.append(case["case_id"])
            return len(case_ids) == len(set(case_ids))

        if not (
            report.get("schema")
            == LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA
            and solver_policy.verify_authentication()
            and report.get("solver_policy_sha256")
            == solver_policy.authentication_sha256
            and report.get("solver_schema") == LFM2_CPCF_SHORT_CONV_V4_SCHEMA
            and sealed(
                request,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_REQUEST_SCHEMA,
            )
            and sealed(
                frontier,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_SCHEMA,
            )
            and sealed(
                pool,
                LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_POOL_SCHEMA,
            )
            and set(request) == _SHORT_CONV_V5_NUMERICAL_REQUEST_FIELDS
            and set(frontier) == _SHORT_CONV_V5_NUMERICAL_FRONTIER_FIELDS
            and (
                set(pool) == _SHORT_CONV_V5_NUMERICAL_POOL_FIELDS
                or set(pool)
                == (
                    _SHORT_CONV_V5_NUMERICAL_POOL_FIELDS
                    | _SHORT_CONV_V5_NUMERICAL_POOL_REPLAY_FIELDS
                )
            )
        ):
            return False
        request_sha = request["report_sha256"]
        frontier_sha = frontier["report_sha256"]
        source_inventory = request.get("source_tensor_sha256")
        context_inventory = request.get("parent_context_sha256")
        template_inventory = request.get("target_template_sha256")
        adapter_provenance = frontier.get("adapter_provenance")
        dimensions = (
            report.get("source_hidden_size"),
            report.get("target_hidden_size"),
            report.get("source_fir_length"),
            report.get("target_fir_length"),
        )
        if not (
            report.get("numerical_frontier_request_sha256") == request_sha
            and report.get("numerical_frontier_report_sha256") == frontier_sha
            and frontier.get("request_sha256") == request_sha
            and pool.get("request_sha256") == request_sha
            and pool.get("numerical_frontier_report_sha256") == frontier_sha
            and pool.get("kind")
            == "authenticated_v5_numerical_target_dtype_candidate_pool"
            and pool.get("policy_sha256") == solver_policy.authentication_sha256
            and pool.get("target_dtype") == solver_policy.target_dtype
            and request.get("kind") == "detached_fit_only_numerical_frontier_request"
            and request.get("fit_only") is True
            and request.get("selection_cases_exposed") is False
            and request.get("validation_cases_exposed") is False
            and request.get("policy_sha256") == solver_policy.authentication_sha256
            and request.get("target_dtype") == solver_policy.target_dtype
            and _valid_exact_zero_execution(
                request.get("execution"),
                fields=_SHORT_CONV_V5_NUMERICAL_EXECUTION_FIELDS,
            )
            and isinstance(source_inventory, Mapping)
            and set(source_inventory) == _SHORT_CONV_V5_NUMERICAL_SOURCE_TENSOR_ROLES
            and isinstance(context_inventory, Mapping)
            and context_inventory
            and isinstance(template_inventory, Mapping)
            and set(template_inventory)
            == _SHORT_CONV_V5_NUMERICAL_TARGET_TEMPLATE_ROLES
            and request.get("source_tensor_inventory_sha256") == _sha(source_inventory)
            and request.get("parent_context_inventory_sha256")
            == _sha(context_inventory)
            and request.get("target_template_inventory_sha256")
            == _sha(template_inventory)
            and all(type(value) is int and value > 0 for value in dimensions)
            and dimensions[0] >= dimensions[1]
            and request.get("source_hidden_size") == dimensions[0]
            and request.get("target_hidden_size") == dimensions[1]
            and request.get("source_fir_length") == dimensions[2]
            and request.get("target_fir_length") == dimensions[3]
            and isinstance(
                report.get("hidden_width_compression_ratio"),
                (int, float),
            )
            and not isinstance(
                report.get("hidden_width_compression_ratio"),
                bool,
            )
            and math.isfinite(float(report["hidden_width_compression_ratio"]))
            and float(report["hidden_width_compression_ratio"])
            == dimensions[0] / dimensions[1]
            and frontier.get("kind") == "authenticated_fit_only_numerical_frontier"
            and isinstance(frontier.get("adapter_id"), str)
            and bool(frontier["adapter_id"].strip())
            and isinstance(frontier.get("context_encoder_sha256"), str)
            and _SHA256.fullmatch(frontier["context_encoder_sha256"]) is not None
            and isinstance(adapter_provenance, Mapping)
            and isinstance(adapter_provenance.get("implementation"), str)
            and bool(adapter_provenance["implementation"].strip())
            and frontier.get("adapter_provenance_sha256") == _sha(adapter_provenance)
            and frontier.get("fit_only") is True
            and frontier.get("selection_cases_consumed") == 0
            and frontier.get("validation_cases_consumed") == 0
            and _valid_exact_zero_execution(
                frontier.get("execution"),
                fields=_SHORT_CONV_V5_NUMERICAL_EXECUTION_FIELDS,
            )
            and frontier.get("fit_case_authentication_sha256")
            == request.get("fit_case_authentication_sha256")
            and isinstance(case_seals, Mapping)
            and set(case_seals) == {"fit", "selection", "validation"}
            and all(
                isinstance(values, list)
                and values
                and all(
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                    for value in values
                )
                for values in case_seals.values()
            )
            and case_seals.get("fit") == request.get("fit_case_authentication_sha256")
            and len(context_inventory) == len(case_seals["fit"])
            and pool.get("fold_case_authentication_sha256") == dict(case_seals)
        ):
            return False
        raw_reports = frontier.get("raw_candidate_reports")
        raw_ids = frontier.get("raw_candidate_ids")
        raw_state_sha = frontier.get("raw_candidate_state_sha256")
        raw_report_sha = frontier.get("raw_candidate_report_sha256")
        if not (
            isinstance(raw_reports, Mapping)
            and isinstance(raw_ids, list)
            and raw_ids
            and len(raw_ids) == len(set(raw_ids))
            and set(raw_ids) == set(raw_reports)
            and isinstance(raw_state_sha, Mapping)
            and set(raw_state_sha) == set(raw_ids)
            and isinstance(raw_report_sha, Mapping)
            and set(raw_report_sha) == set(raw_ids)
            and frontier.get("raw_candidate_count") == len(raw_ids)
            and all(
                sealed(
                    raw_reports[candidate_id],
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_SCHEMA,
                )
                and set(raw_reports[candidate_id])
                == _SHORT_CONV_V5_NUMERICAL_RAW_CANDIDATE_FIELDS
                and raw_reports[candidate_id].get("kind")
                == "authenticated_fit_only_numerical_raw_candidate"
                and raw_reports[candidate_id].get("candidate_id") == candidate_id
                and isinstance(
                    raw_reports[candidate_id].get("candidate_kind"),
                    str,
                )
                and bool(raw_reports[candidate_id]["candidate_kind"].strip())
                and raw_reports[candidate_id].get("request_sha256") == request_sha
                and raw_reports[candidate_id].get("state_sha256")
                == raw_state_sha[candidate_id]
                and isinstance(raw_state_sha[candidate_id], str)
                and _SHA256.fullmatch(raw_state_sha[candidate_id]) is not None
                and raw_reports[candidate_id].get("report_sha256")
                == raw_report_sha[candidate_id]
                and isinstance(
                    raw_reports[candidate_id].get("method_provenance"),
                    Mapping,
                )
                and isinstance(
                    raw_reports[candidate_id]["method_provenance"].get("method"),
                    str,
                )
                and bool(
                    raw_reports[candidate_id]["method_provenance"]["method"].strip()
                )
                and raw_reports[candidate_id].get("method_provenance_sha256")
                == _sha(raw_reports[candidate_id]["method_provenance"])
                and raw_reports[candidate_id].get("fit_only") is True
                and raw_reports[candidate_id].get("selection_cases_consumed") == 0
                and raw_reports[candidate_id].get("validation_cases_consumed") == 0
                and _valid_exact_zero_execution(
                    raw_reports[candidate_id].get("execution"),
                    fields=_SHORT_CONV_V5_NUMERICAL_EXECUTION_FIELDS,
                )
                for candidate_id in raw_ids
            )
        ):
            return False
        participants = frontier.get("source_numeric_participant_indices")
        source_count = frontier.get("source_channel_count")
        selected_count = frontier.get("source_channels_selected")
        dropped_count = frontier.get("source_channels_dropped")
        full_coverage = frontier.get("full_source_numeric_participation")
        if not (
            type(source_count) is int
            and source_count > 0
            and source_count == dimensions[0]
            and isinstance(participants, list)
            and participants == sorted(set(participants))
            and all(
                type(index) is int and 0 <= index < source_count
                for index in participants
            )
            and selected_count == len(participants)
            and dropped_count == source_count - len(participants)
            and full_coverage is (len(participants) == source_count)
            and pool.get("source_channels_selected") == selected_count
            and pool.get("source_channels_dropped") == dropped_count
            and pool.get("full_source_numeric_participation") is full_coverage
            and report.get("source_channels_selected") == selected_count
            and report.get("source_channels_dropped") == dropped_count
        ):
            return False
        projection_case_seals = pool.get("projection_case_authentication_sha256")
        pool_state_sha = pool.get("candidate_state_sha256")
        if not (
            isinstance(candidate_fit, Mapping)
            and candidate_fit
            and isinstance(candidate_selection, Mapping)
            and set(candidate_fit) == set(candidate_selection)
            and isinstance(pool_state_sha, Mapping)
            and set(pool_state_sha) == set(candidate_fit)
            and pool.get("candidate_ids") == sorted(candidate_fit)
            and len(pool["candidate_ids"]) == len(set(pool["candidate_ids"]))
            and type(pool.get("unique_target_dtype_candidate_count")) is int
            and pool.get("unique_target_dtype_candidate_count") == len(candidate_fit)
            and type(pool.get("raw_candidate_count")) is int
            and pool.get("raw_candidate_count") == len(raw_ids)
            and type(pool.get("target_dtype_duplicate_count")) is int
            and pool.get("target_dtype_duplicate_count")
            == len(raw_ids) - len(candidate_fit)
            and pool["target_dtype_duplicate_count"] >= 0
            and isinstance(projection_case_seals, Mapping)
            and set(projection_case_seals) == {"fit", "selection", "validation"}
            and all(
                isinstance(projection_case_seals[split], list)
                and projection_case_seals[split]
                and all(
                    isinstance(value, str) and _SHA256.fullmatch(value) is not None
                    for value in projection_case_seals[split]
                )
                for split in ("fit", "selection", "validation")
            )
            and pool.get("selection_feasibility_policy_sha256")
            == ranking.get("selection_feasibility_policy_sha256")
            and pool.get("fit_only_candidate_generation") is True
            and pool.get("selection_cases_consumed_during_generation") == 0
            and pool.get("validation_cases_consumed_during_generation") == 0
            and pool.get("legacy_v4_candidate_pool_claimed") is False
        ):
            return False
        for candidate_id, provenance in candidate_fit.items():
            selection = candidate_selection[candidate_id]
            candidate_state_sha = provenance.get("target_dtype_candidate_state_sha256")
            sources = provenance.get("sources")
            aliases = provenance.get("source_alias_candidate_ids")
            if not (
                sealed(
                    provenance,
                    LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_PROVENANCE_SCHEMA,
                )
                and set(provenance) == _SHORT_CONV_V5_NUMERICAL_PROVENANCE_FIELDS
                and provenance.get("kind")
                == ("authenticated_fit_only_numerical_frontier_candidate_provenance")
                and provenance.get("candidate_id") == candidate_id
                and provenance.get("policy_sha256")
                == solver_policy.authentication_sha256
                and provenance.get("target_dtype") == solver_policy.target_dtype
                and isinstance(candidate_state_sha, str)
                and _SHA256.fullmatch(candidate_state_sha) is not None
                and provenance.get("request_sha256") == request_sha
                and provenance.get("numerical_frontier_report_sha256") == frontier_sha
                and provenance.get("fit_case_authentication_sha256")
                == case_seals["fit"]
                and provenance.get("fit_only") is True
                and provenance.get("selection_cases_consumed") == 0
                and provenance.get("validation_cases_consumed") == 0
                and provenance.get("legacy_v4_candidate_provenance_claimed") is False
                and isinstance(sources, list)
                and sources
                and type(provenance.get("source_alias_count")) is int
                and provenance.get("source_alias_count") == len(sources)
                and isinstance(aliases, list)
                and aliases
                and aliases
                == [
                    source.get("candidate_id") if isinstance(source, Mapping) else None
                    for source in sources
                ]
                and len(aliases) == len(set(aliases))
                and candidate_id == aliases[0]
                and all(
                    isinstance(source, Mapping)
                    and set(source) == _SHORT_CONV_V5_NUMERICAL_SOURCE_FIELDS
                    and source.get("candidate_id") in raw_reports
                    and source.get("candidate_kind")
                    == raw_reports[source["candidate_id"]].get("candidate_kind")
                    and source.get("numerical_raw_state_sha256")
                    == raw_state_sha[source["candidate_id"]]
                    and isinstance(
                        source.get("numerical_raw_candidate_report"),
                        Mapping,
                    )
                    and dict(source["numerical_raw_candidate_report"])
                    == dict(raw_reports[source["candidate_id"]])
                    and source.get("numerical_raw_candidate_report_sha256")
                    == raw_report_sha[source["candidate_id"]]
                    for source in sources
                )
                and provenance.get(
                    "actual_target_dtype_materialization_before_deduplication"
                )
                is True
                and _valid_exact_zero_execution(
                    provenance.get("execution"),
                    fields=_SHORT_CONV_V5_NUMERICAL_EXECUTION_FIELDS,
                )
                and isinstance(selection, Mapping)
                and valid_evaluation(
                    selection,
                    split="selection",
                    candidate_state_sha256=candidate_state_sha,
                )
                and pool_state_sha.get(candidate_id) == candidate_state_sha
            ):
                return False
        selected_id = report.get("selected_candidate_id")
        selected_sha = selection_evaluation.get("candidate_state_sha256")
        evaluations = (
            ("fit", fit_evaluation),
            ("selection", selection_evaluation),
            ("validation", validation_evaluation),
        )

        def projection_seals(value: Mapping[str, Any]) -> list[Any]:
            return [
                case.get("projection_case_authentication_sha256")
                if isinstance(case, Mapping)
                else None
                for case in value.get("cases", [])
            ]

        if not (
            isinstance(selected_id, str)
            and selected_id in candidate_fit
            and candidate_selection[selected_id].get("candidate_state_sha256")
            == selected_sha
            and all(
                valid_evaluation(
                    evaluation,
                    split=split,
                    candidate_state_sha256=selected_sha,
                )
                for split, evaluation in evaluations
            )
            and dict(selection_evaluation) == dict(candidate_selection[selected_id])
            and all(
                projection_seals(evaluation) == projection_case_seals[split]
                for split, evaluation in evaluations
            )
            and all(
                projection_seals(selection) == projection_case_seals["selection"]
                for selection in candidate_selection.values()
            )
            and _valid_short_conv_v4_selection_ranking(
                ranking,
                candidate_selection=candidate_selection,
                solver_policy=solver_policy,
                feasibility_policy=pool["selection_feasibility_policy"],
            )
            and (
                (
                    "preinstall_candidate_selector_receipt" not in report
                    and ranking.get("selected_candidate_id") == selected_id
                )
                or (
                    "preinstall_candidate_selector_receipt" in report
                    and report.get("default_v4_selected_candidate_id")
                    == ranking.get("selected_candidate_id")
                )
            )
            and isinstance(gate_policy, Mapping)
            and set(gate_policy) == _SHORT_CONV_GATE_POLICY_FIELDS
            and gate_policy.get("enforce") is True
            and isinstance(gate, Mapping)
            and _valid_short_conv_v4_behavior_gate(
                gate,
                gate_policy=gate_policy,
                fit_evaluation=fit_evaluation,
                validation_evaluation=validation_evaluation,
                full_source_coverage=full_coverage,
            )
            and gate.get("passed") is False
            and gate.get("enforced") is True
        ):
            return False
        selector_present = "preinstall_candidate_selector_receipt" in report
        return bool(
            (not selector_present) or _valid_short_conv_v5_selector_receipt_only(report)
        ) and bool(
            report.get("numerical_frontier_adapter_active") is True
            and report.get("legacy_v4_numerical_generation_claimed") is False
            and report.get("legacy_v4_report_claimed") is False
            and report.get("fit_only_numerical_generation") is True
            and report.get("selection_used_only_for_candidate_ranking") is True
            and report.get("validation_used_only_after_candidate_freeze") is True
            and report.get("source_channels_pruned") == 0
            and report.get("source_channels_quantized") == 0
            and isinstance(report.get("compiled_state_sha256"), str)
            and _SHA256.fullmatch(report["compiled_state_sha256"]) is not None
            and _valid_exact_zero_execution(
                execution,
                fields=_SHORT_CONV_V4_EXECUTION_FIELDS,
            )
            and report.get("compiled_short_conv_fold_issued") is False
            and report.get("promotion_claimed") is False
            and report.get("promotable_candidate") is False
            and report.get("failure_evidence_only") is True
            and _recursive_report_seals_valid(report)
            and _no_recursive_success_claims(report)
        )
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _valid_short_conv_validation_report(
    report: Mapping[str, Any],
) -> bool:
    try:
        schema = report.get("schema")
        if schema == LFM2_CPCF_SHORT_CONV_FOLD_SCHEMA:
            return _valid_short_conv_v1_validation_report(report)
        if schema in {
            LFM2_CPCF_SHORT_CONV_V4_FAILURE_V1_SCHEMA,
            LFM2_CPCF_SHORT_CONV_V4_FAILURE_SCHEMA,
        }:
            return _valid_short_conv_v4_validation_report(report)
        if schema == LFM2_CPCF_SHORT_CONV_V5_SELECTOR_FAILURE_SCHEMA:
            return _valid_short_conv_v5_selector_validation_report(report)
        if schema == LFM2_CPCF_SHORT_CONV_V5_MACROCELL_SELECTOR_FAILURE_SCHEMA:
            return _valid_short_conv_v5_macrocell_validation_report(report)
        if schema == LFM2_CPCF_SHORT_CONV_V5_NUMERICAL_FRONTIER_FAILURE_SCHEMA:
            return _valid_short_conv_v5_numerical_frontier_validation_report(report)
        return False
    except (
        AttributeError,
        KeyError,
        OverflowError,
        TypeError,
        ValueError,
    ):
        return False


def _verified_short_conv_validation_failure(
    error: Exception,
) -> tuple[Mapping[str, Any] | None, bytes | None, str | None]:
    if not isinstance(error, Lfm2CpcfShortConvValidationError):
        return None, None, None
    report = error.report
    if not isinstance(report, Mapping):
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure lacks a structured report."
        )
    report_body = dict(report)
    report_sha = _digest(
        report_body.pop("report_sha256", None),
        "short_conv_validation_report.report_sha256",
    )
    if _sha(report_body) != report_sha:
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure report is not self-sealed."
        )
    if not _valid_short_conv_validation_report(report):
        raise Lfm2CpcfLayerInstallError(
            "Short-conv validation failure report has invalid schema, "
            "gate semantics, or a nested success claim."
        )
    report_bytes = _canonical_json(dict(report))
    return (
        report,
        report_bytes,
        hashlib.sha256(report_bytes).hexdigest(),
    )


def _wrap_layer_failure(
    error: Exception,
    *,
    stage: str,
    layer_index: int,
    operator_kind: str,
    input_bank_sha256: str,
    input_tensor_sha256: str,
    boundary_charts_sha256: str,
    boundary_charts_payload_sha256: str,
    rollback_proof: Mapping[str, Any],
) -> Lfm2CpcfLayerCompilationFailure:
    nested_report, nested_bytes, nested_bytes_sha = (
        _verified_short_conv_validation_failure(error)
    )
    body = {
        "schema": LFM2_CPCF_LAYER_FAILURE_SCHEMA,
        "stage": stage,
        "layer_index": layer_index,
        "operator_kind": operator_kind,
        "input_bank_sha256": input_bank_sha256,
        "input_tensor_sha256": input_tensor_sha256,
        "boundary_charts_sha256": boundary_charts_sha256,
        "boundary_charts_payload_sha256": (boundary_charts_payload_sha256),
        "cause": {
            "module": type(error).__module__,
            "type": type(error).__qualname__,
            "message": str(error),
        },
        "nested_short_conv_validation_report": (
            None if nested_report is None else dict(nested_report)
        ),
        "nested_short_conv_validation_report_sha256": (
            None if nested_report is None else nested_report["report_sha256"]
        ),
        "nested_short_conv_validation_report_canonical_bytes_sha256": (
            nested_bytes_sha
        ),
        "nested_short_conv_validation_report_preserved_byte_for_byte": (
            nested_report is not None
        ),
        "rollback_proof": dict(rollback_proof),
        "layer_result_issued": False,
        "promotion_claimed": False,
    }
    report = {**body, "report_sha256": _sha(body)}
    message = (
        f"LFM2 CPCF layer {layer_index} ({operator_kind}) failed at "
        f"{stage} after exact rollback: {error}"
    )
    return Lfm2CpcfLayerCompilationFailure(
        message,
        stage=stage,
        layer_index=layer_index,
        operator_kind=operator_kind,
        input_bank_sha256=input_bank_sha256,
        input_tensor_sha256=input_tensor_sha256,
        boundary_charts_sha256=boundary_charts_sha256,
        boundary_charts_payload_sha256=(boundary_charts_payload_sha256),
        rollback_proof=rollback_proof,
        report=report,
        short_conv_validation_report=nested_report,
        short_conv_validation_report_bytes=nested_bytes,
    )


def compile_and_install_lfm2_cpcf_layer(
    source_model: Any,
    target_model: Any,
    *,
    layer_index: int,
    input_bank: Lfm2CpcfPhaseStateBank,
    boundary_charts: Lfm2CpcfLayerBoundaryCharts,
    ridge: float = 1e-8,
    short_conv_gate_policy: Lfm2CpcfShortConvGatePolicy | None = None,
    short_conv_v4_solver_policy: Lfm2CpcfShortConvV4SolverPolicy | None = None,
    candidate_replay_evaluator: Lfm2CpcfCandidateReplayEvaluator | None = None,
    candidate_selector: Lfm2CpcfShortConvV5CandidateSelector | None = None,
    numerical_frontier_adapter: (
        Lfm2CpcfShortConvNumericalFrontierAdapter | None
    ) = None,
    attention_sequential_transport_policy: (
        Lfm2CpcfAttentionSequentialTransportPolicy | None
    ) = None,
    composed_qk_attention_policy: Lfm2ComposedQkAttentionFoldPolicy | None = None,
    endpoint_residual_attention_policy: (
        Lfm2ComposedEndpointResidualAttentionFoldPolicy | None
    ) = None,
) -> Lfm2CpcfCompiledLayerResult:
    """Compile, install, and replay one next sequential LFM2 target layer."""

    index = _nonnegative_int(layer_index, "layer_index")
    ridge_value = _nonnegative_float(ridge, "ridge")
    if not isinstance(input_bank, Lfm2CpcfPhaseStateBank):
        raise TypeError("input_bank must be Lfm2CpcfPhaseStateBank.")
    if not isinstance(boundary_charts, Lfm2CpcfLayerBoundaryCharts):
        raise TypeError("boundary_charts must be Lfm2CpcfLayerBoundaryCharts.")
    if attention_sequential_transport_policy is not None and not isinstance(
        attention_sequential_transport_policy,
        Lfm2CpcfAttentionSequentialTransportPolicy,
    ):
        raise TypeError(
            "attention_sequential_transport_policy must be "
            "Lfm2CpcfAttentionSequentialTransportPolicy."
        )
    if composed_qk_attention_policy is not None and (
        not isinstance(
            composed_qk_attention_policy,
            Lfm2ComposedQkAttentionFoldPolicy,
        )
        or not composed_qk_attention_policy.verify_authentication()
    ):
        raise TypeError(
            "composed_qk_attention_policy must be an authenticated "
            "Lfm2ComposedQkAttentionFoldPolicy."
        )
    if endpoint_residual_attention_policy is not None and (
        type(endpoint_residual_attention_policy)
        is not Lfm2ComposedEndpointResidualAttentionFoldPolicy
        or not endpoint_residual_attention_policy.verify_authentication()
    ):
        raise TypeError(
            "endpoint_residual_attention_policy must be an exact authenticated "
            "Lfm2ComposedEndpointResidualAttentionFoldPolicy."
        )
    if (
        endpoint_residual_attention_policy is not None
        and composed_qk_attention_policy is None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint-residual attention requires the authenticated composed "
            "Q/K producer policy."
        )
    if (
        composed_qk_attention_policy is not None
        and attention_sequential_transport_policy is not None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Composed Q/K attention and candidate-local sequential transport "
            "are separate authenticated compiler lanes."
        )
    if (
        endpoint_residual_attention_policy is not None
        and attention_sequential_transport_policy is not None
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint-residual attention and candidate-local sequential "
            "transport are separate authenticated compiler lanes."
        )
    if short_conv_v4_solver_policy is not None and not isinstance(
        short_conv_v4_solver_policy,
        Lfm2CpcfShortConvV4SolverPolicy,
    ):
        raise TypeError(
            "short_conv_v4_solver_policy must be Lfm2CpcfShortConvV4SolverPolicy."
        )
    if (
        short_conv_v4_solver_policy is not None
        and not short_conv_v4_solver_policy.verify_authentication()
    ):
        raise Lfm2CpcfLayerCompilerError(
            "short_conv_v4_solver_policy authentication failed."
        )
    if candidate_replay_evaluator is not None and not callable(
        getattr(candidate_replay_evaluator, "evaluate", None)
    ):
        raise TypeError(
            "candidate_replay_evaluator must implement the authenticated "
            "candidate replay evaluator protocol."
        )
    if candidate_replay_evaluator is not None and short_conv_v4_solver_policy is None:
        raise Lfm2CpcfLayerCompilerError(
            "Injected candidate replay is defined only for the authenticated "
            "short-conv v4 solver lane."
        )
    if candidate_selector is not None and not callable(
        getattr(candidate_selector, "select", None)
    ):
        raise TypeError("candidate_selector must implement select(request).")
    if candidate_selector is not None and short_conv_v4_solver_policy is None:
        raise Lfm2CpcfLayerCompilerError(
            "Pre-install candidate selection is defined only for the "
            "authenticated short-conv v4 solver lane."
        )
    if numerical_frontier_adapter is not None and not callable(
        getattr(numerical_frontier_adapter, "generate", None)
    ):
        raise TypeError("numerical_frontier_adapter must implement generate(request).")
    if numerical_frontier_adapter is not None and short_conv_v4_solver_policy is None:
        raise Lfm2CpcfLayerCompilerError(
            "Numerical-frontier generation is defined only for the "
            "authenticated short-conv v4 solver lane."
        )
    if input_bank.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Input state bank is not the next sequential target layer."
        )
    if boundary_charts.layer_index != index:
        raise Lfm2CpcfLayerCompilerError(
            "Boundary charts do not describe the requested layer."
        )
    if input_bank.hidden_size != boundary_charts.target_width:
        raise Lfm2CpcfLayerCompilerError(
            "Actual child state width differs from boundary-chart width."
        )
    split_counts = {
        split: sum(phase.split == split for phase in input_bank.phases)
        for split in _SHORT_CONV_CASE_SPLITS
    }
    if split_counts["fit"] == 0 or split_counts["validation"] == 0:
        raise Lfm2CpcfLayerCompilerError(
            "Layer compilation requires nonempty fit and validation phases."
        )
    source_layer = _layer(source_model, index, "source_model")
    target_layer = _layer(target_model, index, "target_model")
    source_is_attention = bool(getattr(source_layer, "is_attention_layer", False))
    target_is_attention = bool(getattr(target_layer, "is_attention_layer", False))
    operator_kind = "full_attention" if source_is_attention else "short_conv"
    if (
        source_is_attention != target_is_attention
        or operator_kind != boundary_charts.operator_kind
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Source, target, and boundary chart operator kinds differ."
        )
    if (
        attention_sequential_transport_policy is not None
        and attention_sequential_transport_policy.enabled
        and operator_kind != "full_attention"
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Candidate-local sequential boundary transport is attention-only."
        )
    if composed_qk_attention_policy is not None and operator_kind != "full_attention":
        raise Lfm2CpcfLayerCompilerError(
            "Composed Q/K residual attention folding is attention-only."
        )
    if (
        endpoint_residual_attention_policy is not None
        and operator_kind != "full_attention"
    ):
        raise Lfm2CpcfLayerCompilerError(
            "Endpoint-residual attention folding is attention-only."
        )
    for phase in input_bank.phases:
        evidence = phase.parent_evidence.layers[index]
        if (
            evidence.source_width != boundary_charts.source_width
            or evidence.operator_kind != operator_kind
        ):
            raise Lfm2CpcfLayerCompilerError(
                f"Phase {phase.identity!r} evidence differs from layer charts."
            )

    source_before = _layer_snapshot(source_layer)
    target_before = _layer_snapshot(target_layer)
    evidence_before = _live_layer_evidence_sha256(
        input_bank.phases,
        index,
    )
    input_before = _live_input_bank_sha256(input_bank)
    chart_seal_before = boundary_charts.layer_charts_sha256
    chart_payload_before = _boundary_charts_payload_sha256(boundary_charts)
    stage = "preinstall_replay"

    try:
        attention_runtime_receipt: Mapping[str, Any] | None = None
        attention_candidate_selection_report: Mapping[str, Any] | None = None
        attention_candidate_reports: Mapping[str, Any] | None = None
        attention_final_transport_binding: Mapping[str, Any] | None = None
        if endpoint_residual_attention_policy is not None:
            attention_candidate_ids = (
                LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS
            )
        elif composed_qk_attention_policy is not None:
            attention_candidate_ids = LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS
        else:
            attention_candidate_ids = LFM2_CPCF_ATTENTION_CANDIDATE_IDS
        endpoint_evaluation_charts = boundary_charts
        short_conv_macrocell_selector: _Lfm2CpcfShortConvMacrocellSelector | None = None
        if operator_kind == "full_attention":
            if split_counts["selection"] == 0:
                raise Lfm2CpcfLayerCompilerError(
                    "Residual-attention macrocell selection requires nonempty "
                    "selection phases."
                )
            source_attention = source_layer.self_attn
            target_attention = target_layer.self_attn
            source_q_norm = _weight(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
                dimensions=1,
            )
            source_k_norm = _weight(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
                dimensions=1,
            )
            source_q_epsilon = _variance_epsilon(
                source_attention.q_layernorm,
                "source_attention.q_layernorm",
            )
            source_k_epsilon = _variance_epsilon(
                source_attention.k_layernorm,
                "source_attention.k_layernorm",
            )
            target_q_epsilon = _variance_epsilon(
                target_attention.q_layernorm,
                "target_attention.q_layernorm",
            )
            target_k_epsilon = _variance_epsilon(
                target_attention.k_layernorm,
                "target_attention.k_layernorm",
            )
            if not (
                source_q_epsilon
                == source_k_epsilon
                == target_q_epsilon
                == target_k_epsilon
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Source/target Q/K RMSNorm epsilon differs."
                )

            # Fit and selection archives are reconstructed before any
            # candidate execution.  Validation is intentionally absent until
            # the macrocell winner has been frozen.
            fit_bank = select_lfm2_cpcf_phase_state_bank(
                input_bank,
                splits=("fit",),
            )
            selection_bank = select_lfm2_cpcf_phase_state_bank(
                input_bank,
                splits=("selection",),
            )
            fit_preinstall_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=fit_bank,
            )
            selection_preinstall_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=selection_bank,
            )
            stage = "operator_compile"
            fit_cases, output_transport_receipt = _attention_cases(
                source_model=source_model,
                target_model=target_model,
                source_layer=source_layer,
                layer_index=index,
                input_bank=fit_bank,
                preinstall_replay=fit_preinstall_replay,
                chart=boundary_charts,
                required_split="fit",
            )
            selection_cases, _selection_transport_receipt = _attention_cases(
                source_model=source_model,
                target_model=target_model,
                source_layer=source_layer,
                layer_index=index,
                input_bank=selection_bank,
                preinstall_replay=selection_preinstall_replay,
                chart=boundary_charts,
                required_split="selection",
            )
            common_attention_inputs = {
                "source_query_weight": source_attention.q_proj.weight,
                "source_key_weight": source_attention.k_proj.weight,
                "source_value_weight": source_attention.v_proj.weight,
                "source_output_weight": source_attention.out_proj.weight,
                "target_query_weight_template": target_attention.q_proj.weight,
                "target_key_weight_template": target_attention.k_proj.weight,
                "target_value_weight_template": target_attention.v_proj.weight,
                "target_output_weight_template": target_attention.out_proj.weight,
                "cases": fit_cases,
                "head_dim": int(source_attention.head_dim),
                "query_norm_weight": source_q_norm,
                "key_norm_weight": source_k_norm,
                "residual_assignment_sha256": (
                    boundary_charts.post_operator.report["report_sha256"]
                ),
                "norm_epsilon": source_q_epsilon,
                "ridge": ridge_value,
                "max_prior_delta_ratio": None,
            }
            direct_operator_artifact = compile_lfm2_structural_residual_attention_fold(
                **common_attention_inputs
            )
            producer_artifact = compile_lfm2_structural_attention_producer_fold(
                **common_attention_inputs
            )
            producer_query_key_sha = _tensor_bundle_sha256(
                producer_artifact.score_state_dict()
            )
            direct_query_key_sha = _tensor_bundle_sha256(
                {
                    "q_proj.weight": direct_operator_artifact.query_weight,
                    "k_proj.weight": direct_operator_artifact.key_weight,
                }
            )
            if direct_query_key_sha != producer_query_key_sha:
                raise Lfm2CpcfLayerCompilerError(
                    "Direct and residual-attention candidates do not share "
                    "the exact fit-compiled Q/K producer state."
                )
            composed_qk_artifact: Lfm2CompiledComposedQkAttentionProducerFold | None = (
                None
            )
            if composed_qk_attention_policy is not None:
                trajectory_sha256s = _attention_trajectory_sha256s(
                    fit_bank,
                    required_split="fit",
                )
                if len(trajectory_sha256s) != len(fit_cases):
                    raise Lfm2CpcfLayerCompilerError(
                        "Packed attention cases and trajectory receipts differ."
                    )
                composed_qk_artifact = compile_lfm2_composed_qk_attention_producer_fold(
                    source_query_weight=source_attention.q_proj.weight,
                    source_key_weight=source_attention.k_proj.weight,
                    source_value_weight=source_attention.v_proj.weight,
                    source_output_weight=source_attention.out_proj.weight,
                    target_query_weight_template=(target_attention.q_proj.weight),
                    target_key_weight_template=(target_attention.k_proj.weight),
                    target_value_weight_template=(target_attention.v_proj.weight),
                    target_output_weight_template=(target_attention.out_proj.weight),
                    upstream_fit_cases=fit_cases,
                    trajectory_sha256s=trajectory_sha256s,
                    source_split_sha256=fit_bank.state_bank_sha256,
                    source_query_norm_weight=source_q_norm,
                    source_key_norm_weight=source_k_norm,
                    target_query_norm_weight=source_q_norm,
                    target_key_norm_weight=source_k_norm,
                    source_head_dim=int(source_attention.head_dim),
                    target_head_dim=int(target_attention.head_dim),
                    norm_epsilon=source_q_epsilon,
                    residual_assignment_sha256=(
                        boundary_charts.post_operator.report["report_sha256"]
                    ),
                    policy=composed_qk_attention_policy,
                )
            sequence_effective_operator_artifact = (
                compile_lfm2_fixed_qk_sequence_effective_attention_fold(
                    producer=producer_artifact,
                    target_value_weight_template=(target_attention.v_proj.weight),
                    target_output_weight_template=(target_attention.out_proj.weight),
                    fit_cases=fit_cases,
                    query_norm_weight=source_q_norm,
                    key_norm_weight=source_k_norm,
                    norm_epsilon=source_q_epsilon,
                )
            )
            sequence_effective_query_key_sha = _tensor_bundle_sha256(
                {
                    "q_proj.weight": (
                        sequence_effective_operator_artifact.producer_artifact.query_weight
                    ),
                    "k_proj.weight": (
                        sequence_effective_operator_artifact.producer_artifact.key_weight
                    ),
                }
            )
            if (
                not sequence_effective_operator_artifact.verify_authentication()
                or sequence_effective_query_key_sha != producer_query_key_sha
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Sequence-effective attention did not preserve the exact "
                    "fit-compiled Q/K producer state."
                )
            query_key_score_producers = {
                candidate_id: producer_artifact
                for candidate_id in LFM2_CPCF_ATTENTION_CANDIDATE_IDS
            }
            if composed_qk_artifact is not None:
                query_key_score_producers[
                    SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
                ] = composed_qk_artifact.producer_artifact
            score_candidate_ids = (
                LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS
                if composed_qk_artifact is not None
                else LFM2_CPCF_ATTENTION_CANDIDATE_IDS
            )
            query_key_score_reports = {
                candidate_id: (
                    evaluate_lfm2_compiled_attention_producer_score_preservation(
                        query_key_score_producers[candidate_id],
                        selection_cases,
                        query_norm_weight=source_q_norm,
                        key_norm_weight=source_k_norm,
                        norm_epsilon=source_q_epsilon,
                    )
                )
                for candidate_id in score_candidate_ids
            }
            query_key_state_sha256s = {
                candidate_id: _tensor_bundle_sha256(
                    query_key_score_producers[candidate_id].score_state_dict()
                )
                for candidate_id in score_candidate_ids
            }
            score_selection_report = query_key_score_reports[
                RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID
            ]
            attention_runtime_receipt = _attention_runtime_complexity_receipt(
                output_transport_receipt=output_transport_receipt,
                cases=fit_cases,
                operator_artifact=producer_artifact,
            )

            # The first split pass installs Q/K/V only.  The replay-exposed
            # concatenated context is actual target-native fit evidence for
            # coupled V/O.  Both the split transaction and the complete layer
            # are then restored to the exact original blank snapshot.
            blank_output_prior = (
                target_attention.out_proj.weight.detach()
                .to(device="cpu")
                .contiguous()
                .clone()
            )
            stage = "operator_install"
            temporary_norm_report = _copy_attention_norms(
                source_attention,
                target_attention,
            )
            temporary_transaction = install_lfm2_structural_attention_producer_fold(
                target_attention,
                producer_artifact,
            )
            if temporary_transaction.producer_report is None:
                raise Lfm2CpcfLayerInstallError(
                    "Temporary Q/K/V installation issued no audit report."
                )
            temporary_qkv_install = temporary_transaction.producer_report.as_dict()
            temporary_qkv_rollback = None
            try:
                stage = "operator_replay"
                fit_context_replay = replay_lfm2_cpcf_phase_bank(
                    target_model,
                    layer_index=index,
                    input_bank=fit_bank,
                )
                actual_fit_contexts = _attention_actual_contexts(
                    input_bank=fit_bank,
                    replay_bank=fit_context_replay,
                    required_split="fit",
                )
                if len(actual_fit_contexts) != len(fit_cases):
                    raise Lfm2CpcfLayerCompilerError(
                        "Actual fit contexts do not align one-to-one with "
                        "compiled attention trajectories."
                    )
            finally:
                if temporary_transaction.active:
                    temporary_qkv_rollback = temporary_transaction.rollback().as_dict()
            _restore_layer(target_layer, target_before)
            temporary_rollback_proof = _prove_exact_layer_rollback(
                target_layer,
                target_before,
                source_layer=source_layer,
                source_snapshot=source_before,
                input_bank=input_bank,
                layer_index=index,
                evidence_sha256_before=evidence_before,
                input_sha256_before=input_before,
                boundary_charts=boundary_charts,
                boundary_charts_sha256_before=chart_seal_before,
                boundary_charts_payload_sha256_before=chart_payload_before,
            )
            temporary_install_report = {
                "scope": "temporary_qkv_actual_context_capture",
                "attention_norms": dict(temporary_norm_report),
                "qkv_install": temporary_qkv_install,
                "qkvo_transaction_rollback": temporary_qkv_rollback,
                "complete_layer_rollback_proof_sha256": (
                    temporary_rollback_proof["proof_sha256"]
                ),
            }

            stage = "operator_compile"
            value_output_artifact = compile_lfm2_structural_attention_value_output_fold(
                producer=producer_artifact,
                target_value_weight_template=(target_attention.v_proj.weight),
                target_output_weight_template=(target_attention.out_proj.weight),
                actual_child_contexts=actual_fit_contexts,
                cases=fit_cases,
                live_value_weight=producer_artifact.value_weight,
                value_weight_prior=producer_artifact.value_weight,
                output_weight_prior=blank_output_prior,
                query_norm_weight=source_q_norm,
                key_norm_weight=source_k_norm,
                norm_epsilon=source_q_epsilon,
                ridge=ridge_value,
                max_prior_delta_ratio=None,
            )
            split_operator_artifact = _compose_residual_attention_split_artifact(
                producer_artifact=producer_artifact,
                value_output_artifact=value_output_artifact,
                score_selection_report=score_selection_report,
                fit_context_replay=fit_context_replay,
                actual_contexts=actual_fit_contexts,
                temporary_producer_install_report=(temporary_install_report),
                temporary_rollback_proof=temporary_rollback_proof,
            )

            composed_operator_artifact: (
                Lfm2CompiledComposedQkSequenceEffectiveAttentionFold | None
            ) = None
            endpoint_operator_artifact: (
                Lfm2CompiledComposedEndpointResidualAttentionFold | None
            ) = None
            endpoint_output_prior_receipt: Mapping[str, Any] | None = None
            if composed_qk_artifact is not None:
                corrected_producer = composed_qk_artifact.producer_artifact
                stage = "operator_install"
                _copy_attention_norms(
                    source_attention,
                    target_attention,
                )
                corrected_transaction = install_lfm2_structural_attention_producer_fold(
                    target_attention,
                    corrected_producer,
                )
                if corrected_transaction.producer_report is None:
                    raise Lfm2CpcfLayerInstallError(
                        "Temporary corrected Q/K/V installation issued no audit report."
                    )
                try:
                    stage = "operator_replay"
                    corrected_fit_context_replay = replay_lfm2_cpcf_phase_bank(
                        target_model,
                        layer_index=index,
                        input_bank=fit_bank,
                    )
                    corrected_actual_fit_contexts = _attention_actual_contexts(
                        input_bank=fit_bank,
                        replay_bank=corrected_fit_context_replay,
                        required_split="fit",
                    )
                    if len(corrected_actual_fit_contexts) != len(fit_cases):
                        raise Lfm2CpcfLayerCompilerError(
                            "Corrected actual fit contexts do not align "
                            "one-to-one with attention trajectories."
                        )
                finally:
                    if corrected_transaction.active:
                        corrected_transaction.rollback()
                _restore_layer(target_layer, target_before)
                _prove_exact_layer_rollback(
                    target_layer,
                    target_before,
                    source_layer=source_layer,
                    source_snapshot=source_before,
                    input_bank=input_bank,
                    layer_index=index,
                    evidence_sha256_before=evidence_before,
                    input_sha256_before=input_before,
                    boundary_charts=boundary_charts,
                    boundary_charts_sha256_before=chart_seal_before,
                    boundary_charts_payload_sha256_before=(chart_payload_before),
                )
                stage = "operator_compile"
                corrected_value_output_artifact = (
                    compile_lfm2_structural_attention_value_output_fold(
                        producer=corrected_producer,
                        target_value_weight_template=(target_attention.v_proj.weight),
                        target_output_weight_template=(
                            target_attention.out_proj.weight
                        ),
                        actual_child_contexts=(corrected_actual_fit_contexts),
                        cases=fit_cases,
                        live_value_weight=corrected_producer.value_weight,
                        value_weight_prior=corrected_producer.value_weight,
                        output_weight_prior=blank_output_prior,
                        query_norm_weight=source_q_norm,
                        key_norm_weight=source_k_norm,
                        norm_epsilon=source_q_epsilon,
                        ridge=ridge_value,
                        max_prior_delta_ratio=None,
                    )
                )
                composed_operator_artifact = (
                    compile_lfm2_composed_qk_sequence_effective_attention_fold(
                        composed_qk=composed_qk_artifact,
                        fit_value_output_artifact=(corrected_value_output_artifact),
                        upstream_fit_cases=fit_cases,
                        actual_child_contexts=corrected_actual_fit_contexts,
                        target_value_weight_template=(target_attention.v_proj.weight),
                        target_output_weight_template=(
                            target_attention.out_proj.weight
                        ),
                        query_norm_weight=source_q_norm,
                        key_norm_weight=source_k_norm,
                        norm_epsilon=source_q_epsilon,
                    )
                )
                if (
                    composed_operator_artifact.verify_authentication()
                    != composed_operator_artifact.artifact_sha256
                    or _tensor_bundle_sha256(
                        {
                            "q_proj.weight": (
                                composed_operator_artifact.producer_artifact.query_weight
                            ),
                            "k_proj.weight": (
                                composed_operator_artifact.producer_artifact.key_weight
                            ),
                        }
                    )
                    != query_key_state_sha256s[
                        SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
                    ]
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        "Composed sequence-effective attention changed its "
                        "authenticated corrected Q/K state."
                    )
                if endpoint_residual_attention_policy is not None:
                    qk_fit_ordinals = tuple(
                        composed_qk_artifact.qk_fit_bank.source_case_ordinals
                    )
                    if (
                        not qk_fit_ordinals
                        or len(qk_fit_ordinals)
                        != len(composed_qk_artifact.qk_fit_bank.cases)
                        or any(
                            fit_cases[ordinal] is not bank_case
                            for ordinal, bank_case in zip(
                                qk_fit_ordinals,
                                composed_qk_artifact.qk_fit_bank.cases,
                                strict=True,
                            )
                        )
                    ):
                        raise Lfm2CpcfLayerCompilerError(
                            "Endpoint-residual V/O fitting does not match the "
                            "authenticated Q/K-FIT partition."
                        )
                    endpoint_fit_cases = tuple(
                        fit_cases[ordinal] for ordinal in qk_fit_ordinals
                    )
                    endpoint_actual_fit_contexts = tuple(
                        corrected_actual_fit_contexts[ordinal]
                        for ordinal in qk_fit_ordinals
                    )
                    torch = _torch()
                    endpoint_output_prior = torch.zeros(
                        tuple(target_attention.out_proj.weight.shape),
                        dtype=target_attention.out_proj.weight.dtype,
                        device="cpu",
                    )
                    endpoint_output_prior_receipt = MappingProxyType(
                        {
                            "construction": (
                                "zero_from_target_shape_and_dtype_metadata_only"
                            ),
                            "shape": list(endpoint_output_prior.shape),
                            "dtype": str(endpoint_output_prior.dtype),
                            "tensor_sha256": _tensor_sha256(
                                endpoint_output_prior
                            ),
                            "blank_target_values_read": False,
                        }
                    )
                    endpoint_fit_value_output_artifact = (
                        compile_lfm2_structural_attention_value_output_fold(
                            producer=corrected_producer,
                            target_value_weight_template=(
                                target_attention.v_proj.weight
                            ),
                            target_output_weight_template=(
                                target_attention.out_proj.weight
                            ),
                            actual_child_contexts=endpoint_actual_fit_contexts,
                            cases=endpoint_fit_cases,
                            live_value_weight=corrected_producer.value_weight,
                            value_weight_prior=corrected_producer.value_weight,
                            output_weight_prior=endpoint_output_prior,
                            query_norm_weight=source_q_norm,
                            key_norm_weight=source_k_norm,
                            norm_epsilon=source_q_epsilon,
                            ridge=ridge_value,
                            max_prior_delta_ratio=None,
                        )
                    )
                    endpoint_operator_artifact = (
                        compile_lfm2_composed_endpoint_residual_attention_fold(
                            composed_qk=composed_qk_artifact,
                            fit_value_output_artifact=(
                                endpoint_fit_value_output_artifact
                            ),
                            source_query_weight=source_attention.q_proj.weight,
                            source_key_weight=source_attention.k_proj.weight,
                            source_value_weight=source_attention.v_proj.weight,
                            source_output_weight=source_attention.out_proj.weight,
                            target_query_weight_template=(
                                target_attention.q_proj.weight
                            ),
                            target_key_weight_template=(
                                target_attention.k_proj.weight
                            ),
                            target_value_weight_template=(
                                target_attention.v_proj.weight
                            ),
                            target_output_weight_template=(
                                target_attention.out_proj.weight
                            ),
                            query_norm_weight=source_q_norm,
                            key_norm_weight=source_k_norm,
                            norm_epsilon=source_q_epsilon,
                            policy=endpoint_residual_attention_policy,
                        )
                    )
                    if (
                        endpoint_operator_artifact.accepted is not True
                        or endpoint_operator_artifact.candidate_id
                        != ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
                        or endpoint_operator_artifact.verify_authentication()
                        != endpoint_operator_artifact.artifact_sha256
                    ):
                        raise Lfm2CpcfLayerCompilerError(
                            "Endpoint-residual fifth arm failed authentication."
                        )
                    endpoint_score_producer = (
                        endpoint_operator_artifact.producer_artifact
                    )
                    query_key_score_producers[
                        ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
                    ] = endpoint_score_producer
                    query_key_score_reports[
                        ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
                    ] = evaluate_lfm2_compiled_attention_producer_score_preservation(
                        endpoint_score_producer,
                        selection_cases,
                        query_norm_weight=source_q_norm,
                        key_norm_weight=source_k_norm,
                        norm_epsilon=source_q_epsilon,
                    )
                    query_key_state_sha256s[
                        ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
                    ] = _tensor_bundle_sha256(
                        endpoint_score_producer.score_state_dict()
                    )
                del corrected_fit_context_replay

            candidate_evaluations = []
            attention_mlp_prepared_basis = None
            candidate_artifacts = {
                DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID: (direct_operator_artifact),
                RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID: split_operator_artifact,
                FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID: (
                    sequence_effective_operator_artifact
                ),
            }
            if composed_operator_artifact is not None:
                candidate_artifacts[
                    SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID
                ] = composed_operator_artifact
            if endpoint_operator_artifact is not None:
                candidate_artifacts[
                    ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID
                ] = endpoint_operator_artifact
            for candidate_id in attention_candidate_ids:
                candidate_artifact = candidate_artifacts[candidate_id]
                candidate_prepared_mlp_basis = (
                    None
                    if (
                        attention_sequential_transport_policy is not None
                        and attention_sequential_transport_policy.enabled
                    )
                    or endpoint_residual_attention_policy is not None
                    else attention_mlp_prepared_basis
                )
                stage = "operator_install"
                candidate_operator_install = _install_cpcf_attention_artifact(
                    source_attention=source_attention,
                    target_attention=target_attention,
                    artifact=candidate_artifact,
                )
                # The helper's first operation is an operator replay followed
                # by the candidate-owned closed-form MLP compile.  Keep the
                # transaction stage on the latter so a solver failure is
                # reported consistently with the canonical layer contract.
                stage = "mlp_compile"
                (
                    candidate_mlp_artifact,
                    candidate_mlp_install,
                    candidate_fit_replay,
                    candidate_selection_replay,
                    candidate_selection_metrics,
                    candidate_selection_context_metrics,
                    candidate_installed_operator_sha,
                    candidate_installed_mlp_sha,
                    candidate_prepared_mlp_basis,
                    candidate_effective_charts,
                    candidate_sequential_transport,
                    candidate_sequential_transport_binding,
                ) = _evaluate_installed_attention_candidate(
                    candidate_id=candidate_id,
                    operator_artifact=candidate_artifact,
                    operator_install_report=candidate_operator_install,
                    source_layer=source_layer,
                    target_model=target_model,
                    target_layer=target_layer,
                    layer_index=index,
                    fit_bank=fit_bank,
                    selection_bank=selection_bank,
                    selection_cases=selection_cases,
                    chart=boundary_charts,
                    ridge=ridge_value,
                    prepared_mlp_basis=candidate_prepared_mlp_basis,
                    sequential_transport_policy=(attention_sequential_transport_policy),
                )
                if (
                    endpoint_residual_attention_policy is None
                    and not (
                        attention_sequential_transport_policy is not None
                        and attention_sequential_transport_policy.enabled
                    )
                ):
                    attention_mlp_prepared_basis = candidate_prepared_mlp_basis
                _restore_layer(target_layer, target_before)
                candidate_rollback_proof = _prove_exact_layer_rollback(
                    target_layer,
                    target_before,
                    source_layer=source_layer,
                    source_snapshot=source_before,
                    input_bank=input_bank,
                    layer_index=index,
                    evidence_sha256_before=evidence_before,
                    input_sha256_before=input_before,
                    boundary_charts=boundary_charts,
                    boundary_charts_sha256_before=chart_seal_before,
                    boundary_charts_payload_sha256_before=(chart_payload_before),
                )
                candidate_evaluations.append(
                    _seal_attention_candidate_evaluation(
                        candidate_id=candidate_id,
                        operator_artifact=candidate_artifact,
                        mlp_artifact=candidate_mlp_artifact,
                        operator_install_report=(candidate_operator_install),
                        mlp_install_report=candidate_mlp_install,
                        fit_operator_replay=candidate_fit_replay,
                        selection_replay=candidate_selection_replay,
                        selection_metrics=candidate_selection_metrics,
                        selection_context_metrics=(candidate_selection_context_metrics),
                        query_key_score_report=(query_key_score_reports[candidate_id]),
                        query_key_score_state_sha256=(
                            query_key_state_sha256s[candidate_id]
                        ),
                        installed_operator_state_sha256=(
                            candidate_installed_operator_sha
                        ),
                        installed_mlp_state_sha256=(candidate_installed_mlp_sha),
                        rollback_proof=candidate_rollback_proof,
                        effective_boundary_charts=(candidate_effective_charts),
                        sequential_boundary_transport=(candidate_sequential_transport),
                        sequential_transport_binding=(
                            candidate_sequential_transport_binding
                        ),
                        candidate_ids=attention_candidate_ids,
                    )
                )
                del candidate_fit_replay, candidate_selection_replay

            if composed_qk_attention_policy is None:
                (
                    selected_candidate,
                    attention_candidate_selection_report,
                ) = _freeze_attention_candidate_winner(candidate_evaluations)
            else:
                (
                    selected_candidate,
                    attention_candidate_selection_report,
                ) = _freeze_attention_candidate_winner(
                    candidate_evaluations,
                    candidate_ids=attention_candidate_ids,
                )
            if endpoint_residual_attention_policy is not None:
                five_arm_selection = dict(attention_candidate_selection_report)
                five_arm_selection.pop("report_sha256", None)
                five_arm_selection.update(
                    {
                        "candidate_local_mlp_preparation_count": len(
                            attention_candidate_ids
                        ),
                        "cross_candidate_prepared_mlp_basis_reused": False,
                        "endpoint_residual_candidate_fit_only_mlp_evaluated": True,
                    }
                )
                attention_candidate_selection_report = MappingProxyType(
                    {
                        **five_arm_selection,
                        "report_sha256": _sha(five_arm_selection),
                    }
                )
            attention_candidate_reports = MappingProxyType(
                {
                    value.candidate_id: dict(value.report)
                    for value in candidate_evaluations
                }
            )
            operator_artifact = selected_candidate.operator_artifact
            mlp_artifact = selected_candidate.mlp_artifact
            endpoint_evaluation_charts = selected_candidate.effective_boundary_charts

            # The winner is immutable now.  This is the first full-bank replay
            # and therefore the first validation access in the attention
            # transaction.
            stage = "preinstall_replay"
            preinstall_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=input_bank,
            )
            stage = "operator_install"
            operator_install_report = _install_cpcf_attention_artifact(
                source_attention=source_attention,
                target_attention=target_attention,
                artifact=operator_artifact,
            )
            stage = "operator_replay"
            operator_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=input_bank,
            )
            stage = "mlp_install"
            mlp_install_report = install_lfm2_structural_mlp_fold(
                target_layer.feed_forward,
                mlp_artifact,
            ).as_dict()
            cases = fit_cases
        else:
            preinstall_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=input_bank,
            )
            stage = "operator_compile"
            source_conv = source_layer.conv
            target_conv = target_layer.conv
            cases = _short_conv_cases(
                layer_index=index,
                input_bank=input_bank,
                preinstall_replay=preinstall_replay,
                chart=boundary_charts,
                use_v4=short_conv_v4_solver_policy is not None,
            )
            policy = (
                Lfm2CpcfShortConvGatePolicy(enforce=False)
                if short_conv_gate_policy is None
                else short_conv_gate_policy
            )
            if short_conv_v4_solver_policy is None:
                operator_artifact = compile_lfm2_cpcf_short_conv_fold(
                    source_fir_weight=source_conv.conv.weight,
                    source_fir_bias=source_conv.conv.bias,
                    target_in_proj_weight_template=target_conv.in_proj.weight,
                    target_in_proj_bias_template=target_conv.in_proj.bias,
                    target_conv_weight_template=target_conv.conv.weight,
                    target_conv_bias_template=target_conv.conv.bias,
                    target_out_proj_weight_template=target_conv.out_proj.weight,
                    target_out_proj_bias_template=target_conv.out_proj.bias,
                    cases=cases,
                    ridge=ridge_value,
                    gate_policy=policy,
                )
            else:
                if split_counts["selection"] == 0:
                    raise Lfm2CpcfLayerCompilerError(
                        "Short-conv whole-macrocell selection requires "
                        "nonempty selection phases."
                    )
                fit_bank = _select_short_conv_phase_state_bank(
                    input_bank,
                    splits=("fit",),
                )
                selection_bank = _select_short_conv_phase_state_bank(
                    input_bank,
                    splits=("selection",),
                )
                selector_audit_bank = _select_short_conv_phase_state_bank(
                    input_bank,
                    splits=("fit", "selection"),
                )
                short_conv_macrocell_selector = _Lfm2CpcfShortConvMacrocellSelector(
                    source_layer=source_layer,
                    target_model=target_model,
                    target_layer=target_layer,
                    layer_index=index,
                    audit_bank=selector_audit_bank,
                    fit_bank=fit_bank,
                    selection_bank=selection_bank,
                    boundary_charts=boundary_charts,
                    ridge=ridge_value,
                    source_snapshot=source_before,
                    target_snapshot=target_before,
                    audit_evidence_sha256_before=(
                        _live_layer_evidence_sha256(
                            selector_audit_bank.phases,
                            index,
                        )
                    ),
                    audit_input_sha256_before=(
                        _live_input_bank_sha256(selector_audit_bank)
                    ),
                    boundary_charts_sha256_before=chart_seal_before,
                    boundary_charts_payload_sha256_before=(chart_payload_before),
                    candidate_replay_evaluator=candidate_replay_evaluator,
                    endpoint_selector=candidate_selector,
                )
                operator_artifact = compile_lfm2_cpcf_short_conv_v4_fold(
                    source_in_proj_weight=source_conv.in_proj.weight,
                    source_in_proj_bias=source_conv.in_proj.bias,
                    source_fir_weight=source_conv.conv.weight,
                    source_fir_bias=source_conv.conv.bias,
                    source_out_proj_weight=source_conv.out_proj.weight,
                    source_out_proj_bias=source_conv.out_proj.bias,
                    target_in_proj_weight_template=target_conv.in_proj.weight,
                    target_in_proj_bias_template=target_conv.in_proj.bias,
                    target_conv_weight_template=target_conv.conv.weight,
                    target_conv_bias_template=target_conv.conv.bias,
                    target_out_proj_weight_template=target_conv.out_proj.weight,
                    target_out_proj_bias_template=target_conv.out_proj.bias,
                    output_encoder=(boundary_charts.post_operator.mapping.encoder),
                    cases=cases,
                    solver_policy=short_conv_v4_solver_policy,
                    gate_policy=policy,
                    candidate_replay_evaluator=candidate_replay_evaluator,
                    candidate_selector=None,
                    macrocell_candidate_selector=(short_conv_macrocell_selector),
                    macrocell_fit_bank_sha256=fit_bank.state_bank_sha256,
                    macrocell_selection_bank_sha256=(selection_bank.state_bank_sha256),
                    numerical_frontier_adapter=(numerical_frontier_adapter),
                )
                selected_macrocell = short_conv_macrocell_selector.selected
                decision = short_conv_macrocell_selector.decision
                if selected_macrocell is None or decision is None:
                    raise Lfm2CpcfLayerCompilerError(
                        "Short-conv compiler returned without freezing its "
                        "whole-macrocell winner."
                    )
                selected_install_sha = selected_macrocell.candidate.report[
                    "candidate_install_state_sha256"
                ]
                if (
                    operator_artifact.report.get("selected_candidate_id")
                    != selected_macrocell.candidate.candidate_id
                    or operator_artifact.report.get("compiled_state_sha256")
                    != selected_install_sha
                    or _short_conv_state_sha256(operator_artifact.state_dict())
                    != selected_install_sha
                    or operator_artifact.report["macrocell_candidate_selector_receipt"][
                        "selector_decision_sha256"
                    ]
                    != decision["report_sha256"]
                ):
                    raise Lfm2CpcfLayerCompilerError(
                        "Compiled short-conv artifact differs from the frozen "
                        "whole-macrocell candidate."
                    )
            stage = "operator_install"
            operator_install_report = _install_short_conv(
                target_conv,
                operator_artifact,
            )
            stage = "operator_replay"
            operator_replay = replay_lfm2_cpcf_phase_bank(
                target_model,
                layer_index=index,
                input_bank=input_bank,
            )
            if short_conv_macrocell_selector is None:
                stage = "mlp_compile"
                mlp_artifact = _compile_cpcf_mlp_from_fit_replay(
                    source_layer=source_layer,
                    target_layer=target_layer,
                    layer_index=index,
                    fit_bank=input_bank,
                    operator_fit_replay=operator_replay,
                    chart=boundary_charts,
                    ridge=ridge_value,
                )
            else:
                selected_macrocell = short_conv_macrocell_selector.selected
                if selected_macrocell is None:
                    raise Lfm2CpcfLayerCompilerError(
                        "Frozen short-conv macrocell MLP is unavailable."
                    )
                mlp_artifact = selected_macrocell.mlp_artifact
            stage = "mlp_install"
            mlp_install_report = install_lfm2_structural_mlp_fold(
                target_layer.feed_forward,
                mlp_artifact,
            ).as_dict()

        stage = "final_replay"
        final_replay = replay_lfm2_cpcf_phase_bank(
            target_model,
            layer_index=index,
            input_bank=input_bank,
        )
        stage = "endpoint_evaluation"
        split_metrics = _split_endpoint_metrics(
            layer_index=index,
            input_bank=input_bank,
            operator_replay=operator_replay,
            final_replay=final_replay,
            chart=endpoint_evaluation_charts,
        )
        if (
            operator_kind == "full_attention"
            and attention_sequential_transport_policy is not None
            and attention_sequential_transport_policy.enabled
        ):
            if (
                selected_candidate.sequential_boundary_transport is None
                or selected_candidate.sequential_transport_binding is None
                or selected_candidate.operator_transport_mlp_bundle_sha256 is None
                or attention_candidate_selection_report is None
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Frozen attention winner lacks its sequential transport bundle."
                )
            final_operator_state_sha = _tensor_bundle_sha256(
                {
                    "q_proj.weight": target_layer.self_attn.q_proj.weight,
                    "k_proj.weight": target_layer.self_attn.k_proj.weight,
                    "v_proj.weight": target_layer.self_attn.v_proj.weight,
                    "out_proj.weight": target_layer.self_attn.out_proj.weight,
                }
            )
            final_mlp_state_sha = _tensor_bundle_sha256(
                {
                    "w1.weight": target_layer.feed_forward.w1.weight,
                    "w3.weight": target_layer.feed_forward.w3.weight,
                    "w2.weight": target_layer.feed_forward.w2.weight,
                }
            )
            if (
                final_operator_state_sha
                != selected_candidate.installed_operator_state_sha256
                or final_mlp_state_sha != selected_candidate.installed_mlp_state_sha256
            ):
                raise Lfm2CpcfLayerInstallError(
                    "Final attention live state differs from the frozen "
                    "operator/transport/MLP winner."
                )
            final_binding_body = {
                "schema": (
                    "neural-forge-lfm2-cpcf-attention-sequential-"
                    "transport-final-binding-v1"
                ),
                "policy": attention_sequential_transport_policy.as_dict(),
                "selected_candidate_id": selected_candidate.candidate_id,
                "selected_candidate_report_sha256": (
                    selected_candidate.report["report_sha256"]
                ),
                "candidate_selection_report_sha256": (
                    attention_candidate_selection_report["report_sha256"]
                ),
                "operator_transport_mlp_bundle_sha256": (
                    selected_candidate.operator_transport_mlp_bundle_sha256
                ),
                "source_boundary_charts_sha256": (boundary_charts.layer_charts_sha256),
                "source_boundary_charts_payload_sha256": chart_payload_before,
                "effective_boundary_charts_sha256": (
                    endpoint_evaluation_charts.layer_charts_sha256
                ),
                "effective_boundary_charts_payload_sha256": (
                    _boundary_charts_payload_sha256(endpoint_evaluation_charts)
                ),
                "sequential_transport_report_sha256": (
                    selected_candidate.sequential_boundary_transport.report[
                        "report_sha256"
                    ]
                ),
                "sequential_transport_binding_report_sha256": (
                    selected_candidate.sequential_transport_binding["report_sha256"]
                ),
                "final_operator_state_sha256": final_operator_state_sha,
                "final_mlp_state_sha256": final_mlp_state_sha,
                "final_replay_sha256": final_replay.report["report_sha256"],
                "fit_endpoint_metrics_report_sha256": (
                    split_metrics["fit"]["report_sha256"]
                ),
                "validation_endpoint_metrics_report_sha256": (
                    split_metrics["validation"]["report_sha256"]
                ),
                "final_endpoint_metrics_use_winning_effective_chart": True,
                "validation_first_endpoint_replay_and_metric_after_bundle_freeze": (
                    True
                ),
                "validation_rows_used_by_transport_solver": 0,
                "selection_refits_or_mutates_transport": False,
                "selection_selects_frozen_bundle": True,
                "operator_transport_mlp_bundle_installed_exactly": True,
                "training_steps": 0,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            }
            attention_final_transport_binding = MappingProxyType(
                {
                    **final_binding_body,
                    "report_sha256": _sha(final_binding_body),
                }
            )
        short_conv_final_binding: Mapping[str, Any] | None = None
        if short_conv_macrocell_selector is not None:
            selected_macrocell = short_conv_macrocell_selector.selected
            decision = short_conv_macrocell_selector.decision
            if selected_macrocell is None or decision is None:
                raise Lfm2CpcfLayerCompilerError(
                    "Short-conv final state lacks its frozen macrocell evidence."
                )
            live_mlp_sha = _tensor_bundle_sha256(
                {
                    "w1.weight": target_layer.feed_forward.w1.weight,
                    "w3.weight": target_layer.feed_forward.w3.weight,
                    "w2.weight": target_layer.feed_forward.w2.weight,
                }
            )
            expected_live_mlp_sha = _tensor_bundle_sha256(mlp_artifact.state_dict())
            if live_mlp_sha != expected_live_mlp_sha:
                raise Lfm2CpcfLayerInstallError(
                    "Final live MLP differs from the selected candidate-owned "
                    "fit-only MLP."
                )
            binding_body = {
                "schema": (
                    "neural-forge-lfm2-cpcf-short-conv-final-macrocell-binding-v1"
                ),
                "selector_decision_sha256": decision["report_sha256"],
                "endpoint_control_candidate_id": (
                    short_conv_macrocell_selector.endpoint_control_id
                ),
                "endpoint_selector_report_sha256": (
                    None
                    if short_conv_macrocell_selector.endpoint_selector_report is None
                    else short_conv_macrocell_selector.endpoint_selector_report[
                        "report_sha256"
                    ]
                ),
                "endpoint_selector_diagnostic": (
                    None
                    if short_conv_macrocell_selector.endpoint_selector_diagnostic
                    is None
                    else dict(
                        short_conv_macrocell_selector.endpoint_selector_diagnostic
                    )
                ),
                "selected_candidate_id": (selected_macrocell.candidate.candidate_id),
                "selected_candidate_state_sha256": (
                    selected_macrocell.candidate.state_sha256
                ),
                "selected_candidate_install_state_sha256": (
                    selected_macrocell.candidate.report[
                        "candidate_install_state_sha256"
                    ]
                ),
                "output_artifact_report_sha256": operator_artifact.report[
                    "report_sha256"
                ],
                "operator_install_report_sha256": operator_install_report[
                    "report_sha256"
                ],
                "selected_fit_mlp_state_sha256": (
                    selected_macrocell.mlp_artifact.compiled_state_sha256
                ),
                "selected_fit_mlp_live_bundle_sha256": expected_live_mlp_sha,
                "final_live_mlp_bundle_sha256": live_mlp_sha,
                "mlp_install_report_sha256": _sha(dict(mlp_install_report)),
                "final_replay_sha256": final_replay.report["report_sha256"],
                "output_artifact_equals_selected_candidate": True,
                "exact_selected_fit_only_mlp_reused": True,
                "whole_macrocell_overturned_endpoint_control": (
                    selected_macrocell.candidate.candidate_id
                    != short_conv_macrocell_selector.endpoint_control_id
                ),
                "validation_used_for_macrocell_selection": False,
                "training_steps": 0,
                "parameters_pruned": 0,
                "parameters_quantized": 0,
            }
            short_conv_final_binding = MappingProxyType(
                {**binding_body, "report_sha256": _sha(binding_body)}
            )

        stage = "immutability_audit"
        source_after = _layer_snapshot(source_layer)
        evidence_after = _live_layer_evidence_sha256(
            input_bank.phases,
            index,
        )
        input_after = _live_input_bank_sha256(input_bank)
        _assert_snapshot_identity(source_layer, source_before)
        _assert_snapshot_identity(target_layer, target_before)
        if (
            source_after.tensor_bytes_sha256 != source_before.tensor_bytes_sha256
            or source_after.full_inventory_sha256 != source_before.full_inventory_sha256
            or source_after.tensor_identity != source_before.tensor_identity
            or source_after.modes != source_before.modes
            or evidence_after != evidence_before
            or input_after != input_before
            or boundary_charts.layer_charts_sha256 != chart_seal_before
            or _boundary_charts_payload_sha256(boundary_charts) != chart_payload_before
        ):
            raise Lfm2CpcfLayerCompilerError(
                "Source, evidence, charts, or actual input bank changed during "
                "closed-form layer compilation."
            )
        stage = "result_seal"
        body = {
            "schema": LFM2_CPCF_LAYER_COMPILER_SCHEMA,
            "layer_index": index,
            "operator_kind": operator_kind,
            "source_width": boundary_charts.source_width,
            "target_width": boundary_charts.target_width,
            "input_bank_sha256": input_bank.state_bank_sha256,
            "next_bank_sha256": final_replay.post_layer_bank.state_bank_sha256,
            "boundary_charts_sha256": boundary_charts.layer_charts_sha256,
            "preinstall_replay_sha256": preinstall_replay.report["report_sha256"],
            "operator_replay_sha256": operator_replay.report["report_sha256"],
            "final_replay_sha256": final_replay.report["report_sha256"],
            "operator_artifact": operator_artifact.as_dict(),
            "mlp_artifact": mlp_artifact.as_dict(),
            "operator_install_report": dict(operator_install_report),
            "mlp_install_report": dict(mlp_install_report),
            "split_metrics": {
                split: dict(split_metrics[split]) for split in _EVALUATION_SPLITS
            },
            "split_phase_counts": split_counts,
            "solver_evidence_policy": {
                "operator_case_count": len(cases),
                "selection_used_for_operator_fit": False,
                "selection_used_for_mlp_fit": False,
                "validation_used_for_attention_fit": False,
                "validation_used_for_mlp_fit": False,
                "validation_used_for_short_conv_fit": False,
                "attention_fit_case_granularity": (
                    "one_complete_causal_trajectory_per_fit_record"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_key_linear_history": (
                    "exact_prefill_plus_ordered_decode_suffixes"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_inverse_rope_preimage_used": (
                    False if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_repeated_prefix_fit_rows": (
                    False if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_fit_phase_weighting": (
                    "equal_total_mass_per_causal_phase_with_global_mean_square_one"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_parent_residual_contribution": (
                    "head_local_O_write_composed_through_post_operator_chart"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_residual_contribution_direction_preserved": (
                    True if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_runtime_complexity_receipt": (
                    None
                    if attention_runtime_receipt is None
                    else dict(attention_runtime_receipt)
                ),
                "attention_candidate_selection": (
                    "not_applicable"
                    if attention_candidate_selection_report is None
                    else dict(attention_candidate_selection_report)
                ),
                "attention_candidate_evaluations": (
                    "not_applicable"
                    if attention_candidate_reports is None
                    else {
                        name: dict(value)
                        for name, value in attention_candidate_reports.items()
                    }
                ),
                "attention_fit_state_bank_isolated": (
                    True if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_selection_state_bank_isolated": (
                    True if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_query_key_score_evaluation_split": (
                    "selection"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_macrocell_selection_endpoint": (
                    "post_layer_by_causal_phase"
                    if operator_kind == "full_attention"
                    else "not_applicable"
                ),
                "attention_each_candidate_has_own_fit_only_mlp": (
                    True if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_winner_frozen_before_validation": (
                    True if operator_kind == "full_attention" else "not_applicable"
                ),
                "attention_validation_replays_before_winner_freeze": (
                    0 if operator_kind == "full_attention" else "not_applicable"
                ),
                "short_conv_macrocell_selection": (
                    "not_applicable"
                    if short_conv_macrocell_selector is None
                    else dict(short_conv_macrocell_selector.decision)
                ),
                "short_conv_final_macrocell_binding": (
                    "not_applicable"
                    if short_conv_final_binding is None
                    else dict(short_conv_final_binding)
                ),
                "short_conv_each_candidate_has_own_fit_only_mlp": (
                    True
                    if short_conv_macrocell_selector is not None
                    else "not_applicable"
                ),
                "short_conv_macrocell_candidate_budget": (
                    "not_applicable"
                    if short_conv_macrocell_selector is None
                    else short_conv_macrocell_selector.decision["candidate_budget"]
                ),
                "short_conv_mlp_prepared_basis_sha256": (
                    "not_applicable"
                    if short_conv_macrocell_selector is None
                    or short_conv_macrocell_selector.mlp_prepared_basis is None
                    else short_conv_macrocell_selector.mlp_prepared_basis.preparation_sha256
                ),
                "short_conv_mlp_preparation_count": (
                    "not_applicable"
                    if short_conv_macrocell_selector is None
                    else short_conv_macrocell_selector.mlp_preparation_count
                ),
                "short_conv_mlp_candidate_compiles_from_prepared_basis": (
                    "not_applicable"
                    if short_conv_macrocell_selector is None
                    else short_conv_macrocell_selector.mlp_candidate_compile_count
                ),
                "short_conv_mlp_prepared_basis_validation_evidence_consumed": (
                    "not_applicable" if short_conv_macrocell_selector is None else 0
                ),
                "endpoint_metrics_by_causal_phase": True,
                "short_conv_validation_role": (
                    "post_materialization_heldout_evaluator"
                    if operator_kind == "short_conv"
                    else "not_applicable"
                ),
            },
            "actual_child_state_policy": {
                "operator_fit_input": (
                    "causally_stitched_full_prefix_preinstall_target_native_replay"
                ),
                "mlp_fit_input": ("suffix_of_full_prefix_installed_operator_replay"),
                "next_layer_input": (
                    "complete_prefill_plus_ordered_decode_suffix_layer_replay"
                ),
                "recomputed_historical_prefix_propagated": False,
                "independent_parent_projection_substitution": False,
            },
            "audit": {
                "source_state_sha256_before": (source_before.tensor_bytes_sha256),
                "source_state_sha256_after": (source_after.tensor_bytes_sha256),
                "source_unchanged": True,
                "evidence_sha256_before": evidence_before,
                "evidence_sha256_after": evidence_after,
                "evidence_unchanged": True,
                "input_bank_sha256_before": input_before,
                "input_bank_sha256_after": input_after,
                "input_bank_unchanged": True,
                "chart_seal_unchanged": True,
                "target_tensor_identities_preserved": True,
                "target_gradient_state_preserved": True,
                "target_training_modes_preserved": True,
                "layer_transaction_rollback_available": True,
            },
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
        if composed_qk_attention_policy is not None:
            if composed_qk_artifact is None:
                raise Lfm2CpcfLayerCompilerError(
                    "Composed Q/K policy produced no authenticated artifact."
                )
            body = {
                **body,
                "solver_evidence_policy": {
                    **body["solver_evidence_policy"],
                    "composed_qk_attention_policy": (
                        composed_qk_attention_policy.as_dict()
                    ),
                    "composed_qk_producer_artifact_sha256": (
                        composed_qk_artifact.artifact_sha256
                    ),
                    "composed_qk_producer_artifact": (composed_qk_artifact.as_dict()),
                    "composed_qk_global_sync": (
                        composed_qk_artifact.global_sync.as_dict()
                    ),
                    "composed_qk_score_result": (
                        dict(composed_qk_artifact.score_result.report)
                    ),
                    "attention_candidate_inventory_mode": (
                        "opt_in_authenticated_five_arm"
                        if endpoint_residual_attention_policy is not None
                        else "opt_in_authenticated_four_arm"
                    ),
                },
            }
        if endpoint_residual_attention_policy is not None:
            if (
                endpoint_operator_artifact is None
                or endpoint_output_prior_receipt is None
            ):
                raise Lfm2CpcfLayerCompilerError(
                    "Endpoint-residual policy produced no authenticated "
                    "fifth-arm artifact."
                )
            body = {
                **body,
                "solver_evidence_policy": {
                    **body["solver_evidence_policy"],
                    "endpoint_residual_attention_policy": (
                        endpoint_residual_attention_policy.as_dict()
                    ),
                    "endpoint_residual_attention_artifact_sha256": (
                        endpoint_operator_artifact.artifact_sha256
                    ),
                    "endpoint_residual_attention_artifact": (
                        endpoint_operator_artifact.as_dict()
                    ),
                    "endpoint_residual_block_reports": [
                        dict(block.report)
                        for block in endpoint_operator_artifact.blocks
                    ],
                    "endpoint_residual_qk_fit_proposal_reports": [
                        dict(block.qk_proposal.report)
                        for block in endpoint_operator_artifact.blocks
                    ],
                    "endpoint_residual_output_prior_receipt": dict(
                        endpoint_output_prior_receipt
                    ),
                    "endpoint_residual_blank_target_values_read": False,
                    "attention_each_candidate_prepares_own_mlp_basis": True,
                    "attention_cross_candidate_prepared_mlp_basis_reuse": False,
                    "endpoint_residual_fit_only_downstream_mlp_evaluation": True,
                    "endpoint_residual_fit_check_final_one_shot_only": True,
                },
            }
        if attention_final_transport_binding is not None:
            body = {
                **body,
                "solver_evidence_policy": {
                    **body["solver_evidence_policy"],
                    "attention_candidate_local_post_layer_transport": dict(
                        attention_final_transport_binding
                    ),
                    "attention_sequential_transport_policy": (
                        attention_sequential_transport_policy.as_dict()
                    ),
                    "attention_each_candidate_prepares_own_mlp_basis": True,
                    "attention_cross_candidate_prepared_mlp_basis_reuse": False,
                    "attention_final_endpoint_effective_boundary_charts_sha256": (
                        endpoint_evaluation_charts.layer_charts_sha256
                    ),
                },
                "actual_child_state_policy": {
                    **body["actual_child_state_policy"],
                    "attention_post_layer_coordinate": (
                        "candidate_local_fit_only_transport_from_actual_"
                        "post_operator_replay"
                    ),
                    "attention_operator_transport_mlp_frozen_together": True,
                },
                "audit": {
                    **body["audit"],
                    "source_boundary_charts_unchanged_after_candidate_transports": (
                        True
                    ),
                    "winning_effective_boundary_chart_used_for_validation": True,
                },
            }
        if (
            short_conv_macrocell_selector is not None
            and short_conv_macrocell_selector.two_lane_execution_receipt is not None
        ):
            solver_policy_report = {
                **body["solver_evidence_policy"],
                "short_conv_two_lane_macrocell_execution": dict(
                    short_conv_macrocell_selector.two_lane_execution_receipt
                ),
            }
            body = {
                **body,
                "solver_evidence_policy": solver_policy_report,
            }
        result_sha = _sha(body)
        return Lfm2CpcfCompiledLayerResult(
            layer_index=index,
            operator_kind=operator_kind,
            input_bank_sha256=input_bank.state_bank_sha256,
            operator_artifact=operator_artifact,
            mlp_artifact=mlp_artifact,
            operator_install_report=operator_install_report,
            mlp_install_report=mlp_install_report,
            preinstall_replay=preinstall_replay,
            operator_replay=operator_replay,
            final_replay=final_replay,
            split_metrics=split_metrics,
            report={**body, "result_sha256": result_sha},
            result_sha256=result_sha,
        )
    except BaseException as error:
        try:
            _restore_layer(target_layer, target_before)
            rollback_proof = _prove_exact_layer_rollback(
                target_layer,
                target_before,
                source_layer=source_layer,
                source_snapshot=source_before,
                input_bank=input_bank,
                layer_index=index,
                evidence_sha256_before=evidence_before,
                input_sha256_before=input_before,
                boundary_charts=boundary_charts,
                boundary_charts_sha256_before=chart_seal_before,
                boundary_charts_payload_sha256_before=(chart_payload_before),
            )
        except BaseException as rollback_error:
            raise Lfm2CpcfLayerInstallError(
                f"LFM2 CPCF layer {index} ({operator_kind}) failed at "
                f"{stage}, and exact layer rollback could not be proven."
            ) from rollback_error
        if not isinstance(error, Exception):
            try:
                error.add_note(
                    "LFM2 CPCF layer target was exactly restored before "
                    "cancellation propagated."
                )
            except AttributeError:  # pragma: no cover - Python boundary
                pass
            raise
        failure = _wrap_layer_failure(
            error,
            stage=stage,
            layer_index=index,
            operator_kind=operator_kind,
            input_bank_sha256=input_bank.state_bank_sha256,
            input_tensor_sha256=input_before,
            boundary_charts_sha256=chart_seal_before,
            boundary_charts_payload_sha256=chart_payload_before,
            rollback_proof=rollback_proof,
        )
        raise failure from error


__all__ = [
    "DIRECT_MONOLITHIC_ATTENTION_CANDIDATE_ID",
    "ENDPOINT_RESIDUAL_ALTERNATING_QKVO_CANDIDATE_ID",
    "FIXED_QK_SEQUENCE_EFFECTIVE_VO_ALS_CANDIDATE_ID",
    "LFM2_CPCF_ATTENTION_CANDIDATE_IDS",
    "LFM2_CPCF_COMPOSED_ATTENTION_CANDIDATE_IDS",
    "LFM2_CPCF_ENDPOINT_RESIDUAL_ATTENTION_CANDIDATE_IDS",
    "LFM2_CPCF_ATTENTION_CANDIDATE_SELECTION_SCHEMA",
    "LFM2_CPCF_ATTENTION_SEQUENTIAL_TRANSPORT_POLICY_SCHEMA",
    "LFM2_CPCF_LAYER_COMPILER_SCHEMA",
    "LFM2_CPCF_LAYER_FAILURE_SCHEMA",
    "LFM2_CPCF_RESIDUAL_ATTENTION_SPLIT_SCHEMA",
    "LFM2_CPCF_SHORT_CONV_TWO_LANE_EXECUTION_SCHEMA",
    "Lfm2CompiledCpcfResidualAttentionSplitFold",
    "Lfm2CompiledComposedEndpointResidualAttentionFold",
    "Lfm2CompiledComposedQkSequenceEffectiveAttentionFold",
    "Lfm2CompiledFixedQkSequenceEffectiveAttentionFold",
    "Lfm2ComposedQkAttentionFoldPolicy",
    "Lfm2ComposedEndpointResidualAttentionFoldPolicy",
    "Lfm2CpcfAttentionSequentialTransportPolicy",
    "Lfm2CpcfCompiledLayerResult",
    "Lfm2CpcfLayerCompilationFailure",
    "Lfm2CpcfLayerCompilerError",
    "Lfm2CpcfLayerInstallError",
    "RESIDUAL_ATTENTION_SPLIT_CANDIDATE_ID",
    "SCORE_ROUTED_RESIDUAL_SEEDED_SEQUENCE_EFFECTIVE_CANDIDATE_ID",
    "compile_and_install_lfm2_cpcf_layer",
    "invert_lfm2_rope",
    "parse_lfm2_cpcf_attention_sequential_transport_policy",
    "recover_lfm2_full_key_linear_history",
]
